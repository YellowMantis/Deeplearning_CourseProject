# Deep Learning Project: Cassette Detection with YOLOv8

## What This Project Does

This project trains a computer vision model to **detect cassettes** (COVID lateral flow test cassettes) in images. The model learns to draw a bounding box around the cassette in any photo. It uses **YOLOv8**, a state-of-the-art real-time object detection algorithm.

---

## Step 1 — Collecting and Labeling the Data

**Files involved:** `annotations/instances_default.json`, `train/` folder (images)

Before any coding, images of cassettes were collected and manually **annotated** (labeled). Each image had a bounding box drawn around the cassette using a labeling tool. The labeling tool exported those annotations in a standard format called **COCO JSON**.

- `train/` — contains all the raw training images (`.jpg` files)
- `annotations/instances_default.json` — the COCO JSON file. It stores, for every image: the image size (width & height), the filename, and the bounding box coordinates of every cassette in that image. The only class/category defined is `"cassette"` (category id = 1).

The COCO format stores bounding boxes as `[x_min, y_min, width, height]` in **pixel coordinates**.

---

## Step 2 — Converting Labels from COCO Format to YOLO Format

**File:** `convert_to_yolo.py`

YOLO requires labels in a **completely different format** from COCO. This script does the conversion.

**What YOLO needs:** For every image, a `.txt` file with the same base name as the image. Each line in the `.txt` represents one object:
```
class_index  x_center  y_center  width  height
```
All values are **normalized** (divided by the image width/height), so they are between 0 and 1. This makes the coordinates resolution-independent.

**What the script does:**
1. Opens `annotations/instances_default.json` and reads all images and their annotations.
2. For each image, calculates the normalized center coordinates and size of each bounding box.
3. Writes a `.txt` label file into the `train/` folder alongside the image.
4. The class index is always `0` (because there is only one class: cassette).

**Result:** For every `.jpg` in `train/`, there is now a matching `.txt` label file.

---

## Step 3 — Organizing the Dataset into Train and Validation Splits

**File:** `setup_dataset.py`  
**Output files:** `dataset/` folder, `dataset.yaml`

YOLOv8 needs the data organized into a specific folder structure. This script does that automatically.

**What it does:**
1. Looks at all images in `train/` that have a matching `.txt` label file.
2. Randomly shuffles them (with a fixed seed of 42 so results are reproducible).
3. Splits them **85% for training, 15% for validation**.
4. Copies the images and their label files into the proper YOLO folder structure:
   ```
   dataset/
   ├── images/
   │   ├── train/   ← 85% of labeled images
   │   └── val/     ← 15% of labeled images
   └── labels/
       ├── train/   ← matching .txt label files
       └── val/     ← matching .txt label files
   ```
5. Writes a `dataset.yaml` configuration file.

**`dataset.yaml`** tells YOLO where to find the data and what the classes are:
```yaml
path: .../DeepLearning/dataset
train: images/train
val: images/val
nc: 1          # number of classes
names:
  0: cassette
```

### Why 85% Train and 15% Validation?

These are **two separate roles** for the data:

- **Training set (85%)** — the images the model actually *learns from*. During each epoch, the model looks at every training image, makes a prediction, compares it to the correct label, and adjusts its internal weights to do better next time. Think of it as the homework problems the model practices on.

- **Validation set (15%)** — these images are *never used for learning*. After every single epoch, the model is tested on the validation set with its weights frozen — no learning happens. This tells you: "how well is the model actually generalizing to new data, or is it just memorizing the training images?" Think of it as a quiz after each lesson. The best model (`best.pt`) is saved based on whichever epoch scored best on the **validation set**, not the training set.

- **Why not use 100% for training?** — If you trained on everything, you would have no way to know if the model is truly learning or just memorizing. A model can score 99% on training images while being terrible on new images. The validation set catches that.

- **Why 85/15 and not 80/20?** — With smaller datasets, giving more data to training (85%) helps the model learn better. 80/20 is more common with large datasets where you have plenty of examples to spare for validation.

---

## Step 4 — Training the Model

**File:** `train_model.py`  
**Output:** `runs/detect/runs/covid_lft-3/weights/best.pt`

This script actually trains the YOLOv8 neural network on the cassette dataset.

**What it does:**
1. Loads **YOLOv8s** (the "small" variant) — pre-trained on the massive COCO dataset. This is called **transfer learning**: rather than starting from scratch, we start with a model that already understands general visual features (edges, shapes, textures) and fine-tune it specifically for cassettes.
2. Runs training with these settings:
   - `epochs=50` — trains for up to 50 full passes through the dataset
   - `imgsz=640` — resizes all images to 640×640 pixels
   - `batch=16` — processes 16 images at a time
   - `device=0` — uses the GPU (RTX 2080) for fast training
   - `patience=10` — stops early if there is no improvement for 10 consecutive epochs (early stopping)
3. After every epoch, evaluates on the validation set to track performance.
4. Saves the best-performing model weights.

**Key output files:**
- `runs/detect/runs/covid_lft-3/weights/best.pt` — the best model checkpoint (best validation score)
- `runs/detect/runs/covid_lft-3/weights/last.pt` — the last epoch's model checkpoint

---

## Step 5 — Evaluating / Testing the Model

**File:** `evaluate_model.py`  
**Input:** `test/` folder (images the model has never seen)  
**Output:** `runs/evaluate/test_predictions/`

After training, the model is tested on completely new images it was never trained or validated on (the `test/` folder).

**What it does:**
1. Loads the trained model from `runs/detect/runs/covid_lft-3/weights/best.pt`.
2. Runs the model on every image in the `test/` folder.
3. Uses a confidence threshold of `0.25` — only draws boxes if the model is at least 25% confident.
4. Saves two things for each test image:
   - A **visual image** with the bounding box drawn on it (saved to `runs/evaluate/test_predictions/`)
   - A **`.txt` file** with the raw predicted box coordinates

This is how you visually verify the model actually works correctly on real-world, totally unseen images.

---

## The Three Data Groups

| Group | Folder | Size | Purpose |
|---|---|---|---|
| Train | `dataset/images/train/` | 85% of labeled data | Model learns from these images every epoch |
| Validation | `dataset/images/val/` | 15% of labeled data | Checks model performance after each epoch, never used for learning |
| Test | `test/` | Separate set | Final real-world check on images the model has never seen at all |

---

## Summary of All Files

| File / Folder | Purpose |
|---|---|
| `annotations/instances_default.json` | COCO-format annotations: bounding boxes + image info for all training images |
| `train/` | Raw training images (.jpg) + converted YOLO label files (.txt) |
| `test/` | Unseen test images used only for final evaluation |
| `convert_to_yolo.py` | Converts COCO JSON annotations → YOLO .txt label files |
| `setup_dataset.py` | Splits data 85/15 train/val, builds `dataset/` folder structure, writes `dataset.yaml` |
| `dataset.yaml` | Tells YOLO where the data is and what the classes are (1 class: cassette) |
| `dataset/images/train/` | 85% of images used for training |
| `dataset/images/val/` | 15% of images used for validation during training |
| `dataset/labels/train/` | YOLO .txt labels for training images |
| `dataset/labels/val/` | YOLO .txt labels for validation images |
| `train_model.py` | Trains YOLOv8s on the dataset for up to 50 epochs on GPU |
| `runs/detect/runs/covid_lft-3/weights/best.pt` | The saved trained model (best checkpoint) |
| `evaluate_model.py` | Runs the trained model on test images, saves visual results |
| `runs/evaluate/test_predictions/` | Output: test images with predicted bounding boxes drawn |

---

## The Big Picture (in one paragraph)

You had images of COVID lateral flow test cassettes. You labeled those images with bounding boxes using an annotation tool, which saved the labels in COCO JSON format. Since YOLO uses a different label format, you first converted those labels. Then you split the labeled data into a training set (85%) and a validation set (15%) and organized everything into the folder structure YOLO expects. You then trained a YOLOv8 small model on this data, using transfer learning (starting from a model pre-trained on general images) to fine-tune it specifically for detecting cassettes. The model trained on GPU for up to 50 epochs — after each epoch it was checked against the validation set to track real progress and avoid memorization, and the best checkpoint was saved. Finally, you ran the trained model on a completely separate test set it had never seen, and it drew bounding boxes around the cassettes — proving the model can generalize to new, unseen data.
