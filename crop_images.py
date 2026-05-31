from pathlib import Path
from PIL import Image

LABELS_DIR = Path("runs/detect/runs/evaluate/test_predictions/labels")
IMAGES_DIR = Path("test")
OUTPUT_DIR = Path("project2/crops")
PADDING = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

saved = 0
skipped = 0

for label_file in LABELS_DIR.glob("*.txt"):
    img_path = IMAGES_DIR / (label_file.stem + ".jpg")

    if not img_path.exists():
        print(f"  Image not found: {img_path.name}")
        skipped += 1
        continue

    lines = label_file.read_text().strip().splitlines()
    if not lines:
        print(f"  Empty label: {label_file.name}")
        skipped += 1
        continue

    img = Image.open(img_path)
    w, h = img.size

    # pick the largest box (highest area) as the main cassette
    best_box = None
    best_area = 0
    for line in lines:
        parts = line.split()
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        area = bw * bh
        if area > best_area:
            best_area = area
            best_box = (cx, cy, bw, bh)

    cx, cy, bw, bh = best_box
    x1 = int((cx - bw / 2) * w) - PADDING
    y1 = int((cy - bh / 2) * h) - PADDING
    x2 = int((cx + bw / 2) * w) + PADDING
    y2 = int((cy + bh / 2) * h) + PADDING

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = img.crop((x1, y1, x2, y2))
    crop.save(OUTPUT_DIR / img_path.name)
    saved += 1

print(f"\nDone. Saved {saved} crops to '{OUTPUT_DIR}/', skipped {skipped}.")
