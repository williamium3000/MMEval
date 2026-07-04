"""Step B: deterministically match each decomposed claim to the scene graph.

For each claim, walk ``sg["objects"]`` looking for an object whose ``names``
contain (or are contained in) one of the claim's noun phrases, with light
plural/lemma tolerance. Produce a verdict + the relevant bboxes/attrs.

The judge (Step C) is told the sg is *partial*; it may override a verdict when
the image clearly says otherwise.
"""

import re

_PLURAL_STRIP = re.compile(r"s$|es$|ies$")


def _normalize(name):
    n = (name or "").strip().lower()
    if not n:
        return n
    # crude singularization to ease matching ("bikes" vs "bike", "trees" vs "tree")
    if n.endswith("ies"):
        return n[:-3] + "y"
    if n.endswith("es") and len(n) > 3:
        return n[:-2]
    if n.endswith("s") and len(n) > 2 and not n.endswith("ss"):
        return n[:-1]
    return n


# Strip determiners and adjective-style words from a noun phrase to get its
# head noun(s) (rough — good enough for VG-style names).
_DETS = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his",
         "her", "its", "our", "their", "some", "any", "one", "two", "three"}


def _head_noun_tokens(phrase):
    toks = [t for t in re.split(r"[\s\-,]+", (phrase or "").lower()) if t]
    toks = [t for t in toks if t not in _DETS]
    return toks


def _match_object(entity_phrase, sg_objects):
    """Return list of matching object records (may be empty)."""
    if not entity_phrase:
        return []
    e_norm = _normalize(entity_phrase)
    e_tokens = set(_head_noun_tokens(entity_phrase))
    e_tokens_norm = {_normalize(t) for t in e_tokens}
    matches = []
    for oid, obj in sg_objects.items():
        names = [n.lower() for n in (obj.get("names") or [])]
        names_norm = [_normalize(n) for n in names]
        # exact phrase, exact-singular, name-as-token, or token-overlap of head
        hit = False
        for n, nn in zip(names, names_norm):
            if n == entity_phrase.lower() or nn == e_norm:
                hit = True; break
            n_tokens_norm = {_normalize(t) for t in n.split()}
            if e_tokens_norm and (e_tokens_norm <= n_tokens_norm or n_tokens_norm <= e_tokens_norm):
                hit = True; break
            if nn and (nn in e_tokens_norm or any(nn == _normalize(t) for t in e_tokens)):
                hit = True; break
        if not hit:
            continue
        bbox = [int(obj.get("x", 0)), int(obj.get("y", 0)),
                int(obj.get("x", 0) + obj.get("w", 0)),
                int(obj.get("y", 0) + obj.get("h", 0))]
        matches.append({
            "vg_id": oid,
            "names": obj.get("names") or [],
            "attributes": obj.get("attributes") or [],
            "bbox_xyxy": bbox,
        })
    return matches


def _attr_match(claim_attr, obj_attrs):
    if not claim_attr or not obj_attrs:
        return None
    claim_tokens = {_normalize(t) for t in re.split(r"[\s,/&-]+|\band\b|\bor\b", claim_attr.lower()) if t and t.strip()}
    obj_tokens = {_normalize(t) for t in (a.lower() for a in obj_attrs)}
    if claim_tokens & obj_tokens:
        return True
    return False


def evaluate_claim(claim, sg):
    sg_objects = sg.get("objects") or {}
    entity_results = []
    for ent in claim.get("entities", []):
        ms = _match_object(ent, sg_objects)
        entity_results.append({"entity": ent, "matches": ms,
                               "in_sg": bool(ms)})

    # Per-entity + primary status. The primary entity is the FIRST entity (the
    # decomposer puts the subject first). For existence claims, this is the
    # object whose presence/absence is being asserted.
    if not entity_results:
        primary_status = "unknown"
        status = "unknown"
        notes = "no entities extracted from claim"
    else:
        primary_status = "present_in_sg" if entity_results[0]["in_sg"] else "absent_from_sg"
        any_match = any(er["in_sg"] for er in entity_results)
        if not any_match:
            status = "all_entities_absent_from_sg"
            notes = "no entity in this claim matched any sg object name"
        elif not entity_results[0]["in_sg"]:
            status = "primary_entity_absent_from_sg"
            notes = f"primary entity '{entity_results[0]['entity']}' not in sg (other entities may be)"
        else:
            status = "primary_entity_present_in_sg"
            notes = "primary entity matched in sg"

    # attribute check: if the claim has an attribute and the primary entity is matched,
    # compare attributes.
    attr_verdict = None
    if claim.get("kind") == "attribute" and claim.get("attribute"):
        primary = entity_results[0]["matches"] if entity_results and entity_results[0]["matches"] else []
        all_obj_attrs = []
        for m in primary:
            all_obj_attrs.extend(m["attributes"])
        if primary:
            hit = _attr_match(claim["attribute"], all_obj_attrs)
            if hit is True:
                attr_verdict = "attr_match"
            elif hit is False:
                attr_verdict = "attr_mismatch_or_unknown"
            if attr_verdict == "attr_mismatch_or_unknown" and all_obj_attrs:
                attr_verdict = "attr_mismatch"  # sg has attrs but they don't overlap
        else:
            attr_verdict = "entity_absent"

    # collect bboxes for the judge
    bboxes = []
    for er in entity_results:
        for m in er["matches"][:3]:  # cap per entity
            bboxes.append({"entity": er["entity"], "names": m["names"],
                           "attributes": m["attributes"], "xyxy": m["bbox_xyxy"]})

    # relation check (best-effort, optional)
    rel_present = None
    if claim.get("kind") == "relation" and len(entity_results) >= 2:
        e1 = entity_results[0]["matches"]
        e2 = entity_results[1]["matches"]
        if e1 and e2:
            ids1 = {m["vg_id"] for m in e1}
            ids2 = {m["vg_id"] for m in e2}
            rel_present = False
            for r in sg.get("relationships", []):
                sid = str(r.get("subject", {}).get("object_id"))
                oid = str(r.get("object", {}).get("object_id"))
                # relationships in baked sg sometimes use the local object_id;
                # we matched on vg_id (the dict key), so try both
                if ((sid in ids1 and oid in ids2) or (sid in ids2 and oid in ids1)):
                    rel_present = True; break

    return {
        "claim_text": claim.get("claim_text", ""),
        "kind": claim.get("kind"),
        "polarity": claim.get("polarity"),
        "status": status,
        "primary_status": primary_status,
        "per_entity": [{"entity": er["entity"], "in_sg": er["in_sg"]}
                       for er in entity_results],
        "attr_verdict": attr_verdict,
        "rel_present": rel_present,
        "bboxes": bboxes,
        "notes": notes,
    }


def evaluate_claims(claims, sg):
    return [evaluate_claim(c, sg) for c in claims]
