"""
Two GED variants computed per-image, anchored on the gt ∩ pred intersection graph:

  ged_hal = GED( gt ∩ pred, pred )       — work to "shrink" pred down to the
                                           shared/grounded subgraph (pred-only
                                           content, i.e. hallucination mass)
  ged_cov = GED( gt ∩ pred, gt ∪ pred )  — total symmetric-difference mass
                                           (gt-only content + pred-only content)

Inputs are the same per-record JSON files that `graph_distance.py` produces
(<base>_sg_ged.json or _sg_delta_con.json) — they already carry the parsed
`unique_sg` strings. Output is a list of dicts with image_id and the two
distances per image.

Reuses scene_graph_to_nx / build_graph_from_string / compare_scene_graphs /
similar_to_any from graph_distance.py.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import networkx as nx

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from graph_distance import (
    scene_graph_to_nx,
    build_graph_from_string,
    compare_scene_graphs,
    similar_to_any,
)


def _node_match(name1, name2):
    """Two node-name strings match iff they share a WordNet synset, OR are
    identical after lowercase/strip (covers parsed-name strings WordNet
    doesn't know)."""
    if name1 is None or name2 is None:
        return False
    if str(name1).strip().lower() == str(name2).strip().lower():
        return True
    return similar_to_any(name1, name2) == 0


def _node_attrs(g, n):
    a = g.nodes[n]
    return list(a.get("attributes", []) or [])


def _node_name(g, n):
    return g.nodes[n].get("name", n)


def build_intersection_and_union(gt_g: nx.DiGraph, pred_g: nx.DiGraph):
    """Build canonical intersection and union DiGraphs.

    Node alignment: greedy match — each pred node maps to a gt node whose
    name shares a WordNet synset (or a literal match). One-to-one (each gt
    node may receive at most one pred match).

    Intersection nodes: matched pairs.
        - canonical key      = gt node id
        - canonical name     = gt name
        - canonical attrs    = set(attrs_gt) ∩ set(attrs_pred)
    Union nodes: all gt nodes + unmatched pred nodes.
        - matched pred attrs are merged into the gt node (set union).

    Edges: directed (subject -> object).
    Intersection edge exists iff there is a directed edge between the same
    matched node pair in both graphs AND their `predicates` lists share at
    least one entry; canonical predicates = predicate intersection.
    Union edges: all gt edges + all pred edges (collapsed onto canonical ids);
    predicate lists are unioned when both graphs supply an edge.

    Returns (inter_g, union_g, n_pred_only_nodes, n_gt_only_nodes).
    """
    # Build adjacency-friendly views
    gt_nodes = list(gt_g.nodes())
    pred_nodes = list(pred_g.nodes())

    # Greedy 1-1 match: pred -> gt
    used_gt = set()
    pred_to_gt = {}
    for pn in pred_nodes:
        pn_name = _node_name(pred_g, pn)
        for gn in gt_nodes:
            if gn in used_gt:
                continue
            if _node_match(pn_name, _node_name(gt_g, gn)):
                pred_to_gt[pn] = gn
                used_gt.add(gn)
                break

    matched_pred = set(pred_to_gt.keys())
    matched_gt = set(pred_to_gt.values())

    # ── Intersection ──────────────────────────────────────────────────
    inter_g = nx.DiGraph()
    for pn, gn in pred_to_gt.items():
        attrs = sorted(set(_node_attrs(gt_g, gn)) & set(_node_attrs(pred_g, pn)))
        inter_g.add_node(gn, name=_node_name(gt_g, gn), attributes=attrs)

    for s_pred, o_pred, edata in pred_g.edges(data=True):
        if s_pred not in pred_to_gt or o_pred not in pred_to_gt:
            continue
        s_gt = pred_to_gt[s_pred]
        o_gt = pred_to_gt[o_pred]
        if not gt_g.has_edge(s_gt, o_gt):
            continue
        gt_pred_set = set(gt_g[s_gt][o_gt].get("predicates", []))
        pr_pred_set = set(edata.get("predicates", []))
        shared = gt_pred_set & pr_pred_set
        if not shared:
            continue
        inter_g.add_edge(s_gt, o_gt, predicates=sorted(shared))

    # ── Union ─────────────────────────────────────────────────────────
    union_g = nx.DiGraph()
    # All gt nodes (matched ones already in canonical id-space)
    for gn in gt_nodes:
        attrs = list(_node_attrs(gt_g, gn))
        union_g.add_node(gn, name=_node_name(gt_g, gn), attributes=list(attrs))
    # Merge attrs from matched pred nodes
    for pn, gn in pred_to_gt.items():
        merged = sorted(set(_node_attrs(gt_g, gn)) | set(_node_attrs(pred_g, pn)))
        union_g.nodes[gn]["attributes"] = merged
    # Unmatched pred nodes — keep pred id (won't collide with gt ids)
    for pn in pred_nodes:
        if pn in matched_pred:
            continue
        union_g.add_node(pn, name=_node_name(pred_g, pn),
                         attributes=list(_node_attrs(pred_g, pn)))

    # gt edges (collapsed)
    for s, o, edata in gt_g.edges(data=True):
        preds = list(edata.get("predicates", []))
        if union_g.has_edge(s, o):
            cur = union_g[s][o].get("predicates", [])
            union_g[s][o]["predicates"] = sorted(set(cur) | set(preds))
        else:
            union_g.add_edge(s, o, predicates=preds)
    # pred edges (collapsed via pred_to_gt)
    for s_pred, o_pred, edata in pred_g.edges(data=True):
        s = pred_to_gt.get(s_pred, s_pred)
        o = pred_to_gt.get(o_pred, o_pred)
        preds = list(edata.get("predicates", []))
        if union_g.has_edge(s, o):
            cur = union_g[s][o].get("predicates", [])
            union_g[s][o]["predicates"] = sorted(set(cur) | set(preds))
        else:
            union_g.add_edge(s, o, predicates=preds)

    n_pred_only = len(pred_nodes) - len(matched_pred)
    n_gt_only = len(gt_nodes) - len(matched_gt)
    return inter_g, union_g, n_pred_only, n_gt_only


def _build_pair(sample):
    """Return (gt_graph, pred_graph) or (None, None) if pred unavailable."""
    if "sg" in sample:
        rels = sample["sg"]["relationships"]
        objs = sample["sg"]["objects"]
        attrs = sample["sg"]["objects"]
    else:
        rels = sample.get("relationships", [])
        objs = sample.get("objects", [])
        attrs = sample.get("attributes", [])
    gt_g = scene_graph_to_nx(rels, attrs, objs)

    if not sample.get("unique_sg"):
        return gt_g, None
    pred_g = build_graph_from_string(' , '.join(sample["unique_sg"]))
    return gt_g, pred_g


def compute_per_image(samples, timeout=10, max_workers=10):
    tasks = []
    for idx, s in enumerate(samples):
        gt_g, pred_g = _build_pair(s)
        if pred_g is None:
            continue
        tasks.append((idx, gt_g, pred_g))

    print(f"Computing ged_hal & ged_cov for {len(tasks)} samples (timeout={timeout}s/each)...")

    def _one(t):
        idx, gt_g, pred_g = t
        inter, union, n_pred_only, n_gt_only = build_intersection_and_union(gt_g, pred_g)
        ged_full = compare_scene_graphs(gt_g, pred_g, timeout, method="ged")
        ged_hal = compare_scene_graphs(inter, pred_g, timeout, method="ged")
        ged_cov = compare_scene_graphs(inter, union, timeout, method="ged")
        return idx, {
            "ged_full": ged_full,
            "ged_hal": ged_hal,
            "ged_cov": ged_cov,
            "n_inter_nodes": inter.number_of_nodes(),
            "n_inter_edges": inter.number_of_edges(),
            "n_union_nodes": union.number_of_nodes(),
            "n_union_edges": union.number_of_edges(),
            "n_pred_only_nodes": n_pred_only,
            "n_gt_only_nodes": n_gt_only,
            "n_pred_nodes": pred_g.number_of_nodes(),
            "n_gt_nodes": gt_g.number_of_nodes(),
            "n_pred_edges": pred_g.number_of_edges(),
            "n_gt_edges": gt_g.number_of_edges(),
        }

    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, t) for t in tasks]
        for f in tqdm.tqdm(as_completed(futs), total=len(futs),
                           desc="Computing ged_hal+ged_cov"):
            idx, vals = f.result()
            out[idx] = vals
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="JSON list with parsed unique_sg per record (e.g. <m>_sg_ged.json).")
    p.add_argument("--output", required=True,
                   help="Output JSON: per-image ged_hal & ged_cov.")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="GED timeout per pair (seconds)")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--merge-first", action="store_true",
                   help="Pre-merge records by image_id (mirrors graph_distance.py).")
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if args.merge_first:
        from graph_distance import merge_conversations_by_image_id
        data = merge_conversations_by_image_id(data)
        print(f"After merge_first: {len(data)} unique images")

    results = compute_per_image(data, timeout=args.timeout, max_workers=args.max_workers)

    out_records = []
    for idx, sample in enumerate(data):
        rec = {"image_id": sample.get("image_id", idx), "record_index": idx}
        if idx in results:
            rec.update(results[idx])
        else:
            rec["ged_full"] = None
            rec["ged_hal"] = None
            rec["ged_cov"] = None
        out_records.append(rec)

    def _mean(key):
        vals = [r[key] for r in out_records if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n_total":       len(data),
        "n_with_pred":   sum(1 for r in out_records if r.get("ged_hal") is not None),
        "ged_full_mean": _mean("ged_full"),
        "ged_hal_mean":  _mean("ged_hal"),
        "ged_cov_mean":  _mean("ged_cov"),
    }

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, "w") as fout:
        json.dump({"summary": summary, "per_image": out_records}, fout, indent=2)

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
