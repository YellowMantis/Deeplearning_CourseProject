import shutil
from pathlib import Path

for label in ["positive", "negative", "invalid"]:
    src = Path(f"project2/test/{label}")
    dst = Path(f"project2/train/{label}")
    dst.mkdir(parents=True, exist_ok=True)
    for img in src.glob("*.jpg"):
        shutil.move(str(img), dst / img.name)
        print(f"Moved {img.name} -> train/{label}/")

print("\nDone. All images are now in train/.")
