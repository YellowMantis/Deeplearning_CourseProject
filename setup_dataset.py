import os, shutil, random

BASE = "c:/Users/yahia/OneDrive/Desktop/school/DeepLearning"
TRAIN_SRC = os.path.join(BASE, "train")

# Output structure
DATASET = os.path.join(BASE, "dataset")
DIRS = [
    "dataset/images/train",
    "dataset/images/val",
    "dataset/labels/train",
    "dataset/labels/val",
]
for d in DIRS:
    os.makedirs(os.path.join(BASE, d), exist_ok=True)

# Collect all labeled image basenames (must have both .jpg and .txt)
exts = {".jpg", ".jpeg", ".png"}
all_bases = []
for f in os.listdir(TRAIN_SRC):
    name, ext = os.path.splitext(f)
    if ext.lower() in exts:
        label_path = os.path.join(TRAIN_SRC, name + ".txt")
        if os.path.exists(label_path):
            all_bases.append(name)

random.seed(42)
random.shuffle(all_bases)

split = int(len(all_bases) * 0.85)
train_bases = all_bases[:split]
val_bases = all_bases[split:]

def copy_files(bases, img_dest, lbl_dest):
    for name in bases:
        # find the image (could be .jpg .jpeg .png)
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            img_src = os.path.join(TRAIN_SRC, name + ext)
            if os.path.exists(img_src):
                shutil.copy2(img_src, os.path.join(img_dest, name + ext))
                break
        shutil.copy2(
            os.path.join(TRAIN_SRC, name + ".txt"),
            os.path.join(lbl_dest, name + ".txt")
        )

copy_files(train_bases, os.path.join(BASE, "dataset/images/train"), os.path.join(BASE, "dataset/labels/train"))
copy_files(val_bases,   os.path.join(BASE, "dataset/images/val"),   os.path.join(BASE, "dataset/labels/val"))

print(f"Train: {len(train_bases)} images")
print(f"Val:   {len(val_bases)} images")
print("Dataset structure ready.")

# Write dataset.yaml
yaml_content = f"""path: {DATASET}
train: images/train
val: images/val

nc: 1
names:
  0: cassette
"""

with open(os.path.join(BASE, "dataset.yaml"), "w") as f:
    f.write(yaml_content)

print("dataset.yaml written.")
