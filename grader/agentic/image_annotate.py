"""Draw labeled bboxes on a copy of the image and return base64."""

import base64
import io
import os

from PIL import Image, ImageDraw, ImageFont

_PALETTE = [
    (255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 165, 0),
    (200, 0, 200), (0, 200, 200), (255, 255, 0), (180, 0, 0),
]


def _font(size):
    for cand in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(cand):
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                pass
    return ImageFont.load_default()


def annotate(image_path, bboxes, max_boxes=12):
    """bboxes: list of {entity, names, attributes, xyxy}. Returns base64 jpeg."""
    img = Image.open(image_path).convert("RGB")
    if not bboxes:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    draw = ImageDraw.Draw(img)
    size = max(12, int(min(img.size) / 30))
    font = _font(size)
    seen_xyxy = set()
    drawn = 0
    for i, b in enumerate(bboxes):
        if drawn >= max_boxes:
            break
        key = tuple(b["xyxy"])
        if key in seen_xyxy:
            continue
        seen_xyxy.add(key)
        x1, y1, x2, y2 = b["xyxy"]
        x1 = max(0, min(int(x1), img.width - 1))
        y1 = max(0, min(int(y1), img.height - 1))
        x2 = max(0, min(int(x2), img.width - 1))
        y2 = max(0, min(int(y2), img.height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        color = _PALETTE[drawn % len(_PALETTE)]
        for w in range(3):
            draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=color)
        names = "/".join((b.get("names") or [b.get("entity", "")])[:2])
        label = names if not b.get("attributes") else f"{names}: {','.join(b['attributes'][:2])}"
        # label background
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        ly = max(0, y1 - th - 2)
        draw.rectangle([x1, ly, x1 + tw + 6, ly + th + 4], fill=color)
        draw.text((x1 + 3, ly + 1), label, fill=(255, 255, 255), font=font)
        drawn += 1

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
