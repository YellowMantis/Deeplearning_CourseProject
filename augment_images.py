from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import random

TARGET = 200

def augment(img):
    img = img.rotate(random.uniform(-30, 30), expand=True)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    return img

for label in ["positive", "negative", "invalid"]:
    folder = Path(f"project2/train/{label}")
    originals = [p for p in folder.glob("*.jpg") if "aug_" not in p.name]
    current   = len(list(folder.glob("*.jpg")))

    if current >= TARGET:
        print(f"{label}: {current} images — already at target, skipping.")
        continue

    needed = TARGET - current
    generated = 0

    for i, img_path in enumerate(originals * ((needed // len(originals)) + 2)):
        if generated >= needed:
            break
        img = Image.open(img_path).convert("RGB")
        new_img = augment(img)
        new_img.save(folder / f"aug_{generated}_{img_path.name}")
        generated += 1

    print(f"{label}: {current} originals → +{generated} augmented = {current + generated} total")
