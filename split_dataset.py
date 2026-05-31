import shutil, random
from pathlib import Path

TEST_RATIO = 0.2
random.seed(42)

for label in ["positive", "negative", "invalid"]:
    src = Path(f"project2/train/{label}")
    dst = Path(f"project2/test/{label}")
    dst.mkdir(parents=True, exist_ok=True)

    originals = [p for p in src.glob("*.jpg") if "aug_" not in p.name]
    n_test = max(1, int(len(originals) * TEST_RATIO))
    to_move = random.sample(originals, n_test)

    for img in to_move:
        shutil.move(str(img), dst / img.name)

    print(f"{label}: {len(originals)} originals → {n_test} to test, {len(originals)-n_test} stay in train")
