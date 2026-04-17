from datasets import load_dataset
import tqdm
import fsspec
import aiohttp

# Load the dataset with the specified download configuration
objects = load_dataset("visual_genome","objects_v1.2.0", trust_remote_code=True, split="train", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=7200)}})
attributes = load_dataset("visual_genome","attributes_v1.2.0", split="train", trust_remote_code=True)
relationships = load_dataset("visual_genome","relationships_v1.2.0", split="train", trust_remote_code=True)
regions = load_dataset("visual_genome","region_descriptions_v1.2.0", split="train", trust_remote_code=True)

# ['region_descriptions_v1.0.0', 
# 'region_descriptions_v1.2.0', 
# 'question_answers_v1.0.0', 
# 'question_answers_v1.2.0', 
# 'objects_v1.0.0', 'objects_v1.2.0', 'attributes_v1.0.0', 'attributes_v1.2.0', 
# 'relationships_v1.0.0', 'relationships_v1.2.0']

assert len(objects) == len(attributes) == len(relationships) == len(regions)

def load_sample_vg(idx):
    sample = objects[idx] 
    attrs = attributes[idx]["attributes"]
    rels = relationships[idx]["relationships"]
    regs = regions[idx]["regions"]
    assert len(sample["objects"]) == len(attrs)
    cur_objects = {}
    for i, obj in enumerate(attrs):
        object_id = obj["object_id"]
        obj["object_id"] = i
        cur_objects[object_id] = obj
    
    new_rels = []
    for rel in rels:
        del rel["relationship_id"]
        sub = rel["subject"]
        obj = rel["object"]
        
        new_sub_id = cur_objects[sub["object_id"]]
        new_obj_id = cur_objects[obj["object_id"]]
        rel["subject"] = new_sub_id
        rel["object"] = new_obj_id
        new_rels.append(rel)        

    scene_graph = {
        "objects": cur_objects,
        "relationships": new_rels,
        "regions": regs,
    }
    del sample["objects"]
    sample["sg"] = scene_graph
    sample["metadata"] =  {
            "objects": objects[idx]["objects"],
            "relationships": relationships[idx]["relationships"],
            "regions": regions[idx]["regions"],
            "attributes": attributes[idx]["attributes"],
        }
    return sample


def load_vg(num_samples=None):
    all_img_ids = range(len(objects))
    if num_samples is not None:
        all_img_ids = all_img_ids[:num_samples]

    samples = []
    print(f"loading vg: total {len(all_img_ids)}")
    for img_id in tqdm.tqdm(all_img_ids):
        case = load_sample_vg(img_id)
        samples.append(case)
    return samples

def format_case_vg(case, use_region=False):
    formatted = "Instances:\n"
    H = case["height"]
    W = case["width"]
    
    sg = case["sg"]

    for ori_id, ins in sg["objects"].items():
        object_id = ins["object_id"]
        x, y, w, h = ins['x'], ins['y'], ins['w'], ins['h']
        x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
        if ins.get("attributes", []) is None or len(ins.get("attributes", [])) == 0:
            cur_attr = "none"
        else:
            attrs = ins.get("attributes", [])
            cur_attr = ", ".join(attrs)
        formatted += f"instance {object_id}, {ins['names'][0]}, bbox: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}), attributes: {cur_attr}\n"
    
    formatted += "\nRelation between the above instances:\n"
    for rel in sg["relationships"]:
        formatted += f"{rel['subject']['names'][0]} (instance {rel['subject']['object_id']}) {rel['predicate'].lower()} {rel['object']['names'][0]} (instance {rel['object']['object_id']})\n"
    
    if use_region:
        formatted += "\nRegion descriptions:\n"
        for reg in case["metadata"]["regions"]:
            x, y, w, h = reg['x'], reg['y'], reg['width'], reg['height']
            x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
            formatted += f"description: {reg['phrase']}, bbox: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})\n"
    
    return formatted
# Noisy VG object names that are attribute fragments, not real objects
NOISY_OBJECT_NAMES = frozenset({
    "color", "colour", "colors", "colours",
})


def format_case_vg_compact(case, use_region=False):
    """Compact VG formatter for v19 — same info, ~50% fewer tokens.

    Changes vs format_case_vg:
    - Drop 'instance' prefix: '0: clock [.53,.15,.62,.72] green, tall'
    - 2-decimal bbox in bracket notation, no 'bbox:' label
    - Omit attributes field entirely when empty (instead of 'attributes: none')
    - Filter out noisy pseudo-objects (color/colour)
    - Deduplicate relations
    - Relations use IDs only: '0 on 1' (names already in Objects section)
    """
    H = case["height"]
    W = case["width"]
    sg = case["sg"]

    formatted = "Objects:\n"
    valid_ids = set()
    for ori_id, ins in sg["objects"].items():
        object_id = ins["object_id"]
        name = ins["names"][0]
        if name.lower() in NOISY_OBJECT_NAMES:
            continue
        valid_ids.add(object_id)
        x, y, w, h = ins['x'], ins['y'], ins['w'], ins['h']
        x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
        attrs = ins.get("attributes", [])
        if attrs:
            formatted += f"{object_id}: {name} [{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}] {', '.join(attrs)}\n"
        else:
            formatted += f"{object_id}: {name} [{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}]\n"

    formatted += "\nRelations:\n"
    seen_rels = set()
    for rel in sg["relationships"]:
        sub_id = rel['subject']['object_id']
        obj_id = rel['object']['object_id']
        pred = rel['predicate'].lower()
        if sub_id not in valid_ids or obj_id not in valid_ids:
            continue
        key = (sub_id, pred, obj_id)
        if key in seen_rels:
            continue
        seen_rels.add(key)
        formatted += f"{sub_id} {pred} {obj_id}\n"

    if use_region:
        formatted += "\nRegions:\n"
        for reg in case["metadata"]["regions"]:
            x, y, w, h = reg['x'], reg['y'], reg['width'], reg['height']
            x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
            formatted += f"{reg['phrase']} [{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}]\n"

    return formatted


if __name__ == "__main__":
    case = load_vg(num_samples=5)[0]
    print("=== ORIGINAL ===")
    orig = format_case_vg(case)
    print(orig)
    print(f"Chars: {len(orig)}, ~Tokens: {len(orig)//4}")
    print()
    print("=== COMPACT ===")
    comp = format_case_vg_compact(case)
    print(comp)
    print(f"Chars: {len(comp)}, ~Tokens: {len(comp)//4}")
    print(f"Reduction: {(1 - len(comp)/len(orig))*100:.1f}%")
