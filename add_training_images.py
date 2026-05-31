"""
Workflow:
  1. Drop internet images into project2/incoming/positive, /negative, or /invalid
  2. Run:  python add_training_images.py
  3. Run:  python train_classifier.py

The script YOLO-crops each image, saves it to project2/train/<class>/,
and reports how many were added or skipped (no cassette detected).
"""
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

YOLO_MODEL   = "runs/detect/runs/covid_lft-3/weights/best.pt"
INCOMING_DIR = Path("project2/incoming")
TRAIN_DIR    = Path("project2/train")
PADDING      = 20
CLASSES      = ["positive", "negative", "invalid"]

detector = YOLO(YOLO_MODEL)

total_added = 0
total_skipped = 0

for cls in CLASSES:
    src_dir = INCOMING_DIR / cls
    dst_dir = TRAIN_DIR / cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in src_dir.glob("*")
              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    if not images:
        continue

    print(f"\n[{cls}] {len(images)} image(s) found")

    for img_path in images:
        results = detector(str(img_path), verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            print(f"  SKIP (no cassette detected): {img_path.name}")
            total_skipped += 1
            continue

        best = boxes[boxes.conf.argmax()]
        conf = float(best.conf[0])
        x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        cx1, cy1 = max(0, x1 - PADDING), max(0, y1 - PADDING)
        cx2, cy2 = min(w, x2 + PADDING), min(h, y2 + PADDING)

        if cx2 <= cx1 or cy2 <= cy1:
            print(f"  SKIP (degenerate box): {img_path.name}")
            total_skipped += 1
            continue

        crop = img.crop((cx1, cy1, cx2, cy2))

        out_path = dst_dir / (img_path.stem + ".jpg")
        # avoid overwriting existing files
        counter = 1
        while out_path.exists():
            out_path = dst_dir / f"{img_path.stem}_{counter}.jpg"
            counter += 1

        crop.save(out_path)
        img_path.unlink()  # remove from incoming once processed
        print(f"  OK  conf={conf*100:.0f}%  crop={crop.size}  -> {out_path.name}")
        total_added += 1

print(f"\nDone. Added: {total_added}  Skipped: {total_skipped}")
if total_added > 0:
    print("Now run:  python train_classifier.py")
