"""
Run any raw/screenshot positives through YOLO so they match the training format
(tight cassette crop), then replace the originals in-place.
"""
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

YOLO_MODEL   = "runs/detect/runs/covid_lft-3/weights/best.pt"
POSITIVE_DIR = Path("project2/train/positive")
PADDING      = 20

detector = YOLO(YOLO_MODEL)

targets = [p for p in POSITIVE_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
           and not p.stem.startswith("aug_")
           and not p.stem.endswith(".rf." + p.stem.split(".rf.")[-1])  # skip already-rf files
           ]

# simpler: just target files that aren't the original rf-named crops
targets = [p for p in POSITIVE_DIR.glob("*")
           if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
           and ".rf." not in p.stem]

print(f"Found {len(targets)} non-rf files to process.\n")

for img_path in targets:
    results = detector(str(img_path), verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print(f"  [SKIP] No cassette detected: {img_path.name}")
        continue

    best = boxes[boxes.conf.argmax()]
    conf = float(best.conf[0])
    x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    crop = img.crop((max(0, x1 - PADDING), max(0, y1 - PADDING),
                     min(w, x2 + PADDING), min(h, y2 + PADDING)))

    # save as jpg with same stem
    out = img_path.with_suffix(".jpg")
    crop.save(out)

    # remove original if it was a png (replaced by jpg above)
    if img_path.suffix.lower() == ".png" and out != img_path:
        img_path.unlink()

    print(f"  [OK] {img_path.name}  conf={conf*100:.1f}%  crop={crop.size}")

print("\nDone. Re-run train_classifier.py to retrain.")
