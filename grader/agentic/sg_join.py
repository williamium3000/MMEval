"""Get the Visual Genome scene graph for an image.

- If the input ``image`` dict already has an ``sg`` field (as in the dyna-v18
  prediction files), return that directly — no VG load needed.
- Otherwise, lazily load ``utils.vg`` (which parses the cached VG JSONs on first
  import) and build the sg by joining on ``image_id``. This is the path for the
  opera / llava human files (no sg baked in).

Returned shape (matches the format used in ``InternVL3-8B-Instruct.json``)::

    {"objects": {orig_id: {x,y,w,h,names,synsets,attributes,object_id}, ...},
     "relationships": [{predicate, subject:{...}, object:{...}}, ...],
     "regions": [{phrase, x, y, width, height}, ...]}
"""

import os
import threading

_vg_lock = threading.Lock()
_id_to_idx = None  # built lazily from utils.vg


def _ensure_vg_index():
    global _id_to_idx
    if _id_to_idx is not None:
        return
    with _vg_lock:
        if _id_to_idx is not None:
            return
        # heavy import — parses ~2GB of VG json on first call
        from utils import vg as _vg  # noqa: F401
        idx = {}
        for i, rec in enumerate(_vg._objects_raw):
            idx[rec["image_id"]] = i
        _id_to_idx = idx


def get_sg(image):
    sg = image.get("sg")
    if isinstance(sg, dict) and sg.get("objects"):
        return sg
    image_id = image["image_id"]
    _ensure_vg_index()
    from utils.vg import load_sample_vg
    idx = _id_to_idx.get(image_id)
    if idx is None:
        return {"objects": {}, "relationships": [], "regions": []}
    sample = load_sample_vg(idx)
    return sample["sg"]
