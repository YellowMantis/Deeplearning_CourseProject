import json
import os

COCO_JSON = "annotations/instances_default.json"
OUTPUT_DIR = "train"  # label .txt files go next to the train images

with open(COCO_JSON) as f:
    data = json.load(f)

# Build image_id -> (filename, width, height) map
images = {}
for img in data["images"]:
    images[img["id"]] = (img["file_name"], img["width"], img["height"])

# Group annotations by image_id
from collections import defaultdict
annotations_by_image = defaultdict(list)
for ann in data["annotations"]:
    annotations_by_image[ann["image_id"]].append(ann)

written = 0
skipped = 0

for image_id, (filename, img_w, img_h) in images.items():
    anns = annotations_by_image[image_id]
    base_name = os.path.splitext(os.path.basename(filename))[0]
    out_path = os.path.join(OUTPUT_DIR, base_name + ".txt")

    lines = []
    for ann in anns:
        # COCO bbox: [x_min, y_min, width, height]
        x_min, y_min, w, h = ann["bbox"]
        # Convert to YOLO: normalized center x, center y, width, height
        x_center = (x_min + w / 2) / img_w
        y_center = (y_min + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        # class index 0 = cassette
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    if lines:
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        written += 1
    else:
        skipped += 1

print(f"Done. Written: {written} label files | Skipped (no annotations): {skipped}")
