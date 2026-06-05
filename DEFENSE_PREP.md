# Defense Preparation — COVID Test Analyzer

---

## Section 1: Architecture & Big Picture

**Q: Describe the full system in one minute.**

The project is a two-stage pipeline. Stage 1 uses a YOLOv8 object detection model to find the COVID cassette in any photo and draw a bounding box around it. Stage 2 takes the cropped cassette and feeds it to a ResNet18 classifier that reads the test result: Positive, Negative, or Invalid. The two models run back-to-back in `covid_pipeline.py`, and the result is displayed in a Tkinter UI (`covid_ui.py`). Everything runs on GPU if available, CPU otherwise.

---

**Q: Why use two separate models instead of one model that does everything?**

Each model is specialized for a different task. Detection requires the model to search the entire image for an object — it needs to be translation-invariant and handle varying scales, backgrounds, and angles. Classification requires the model to analyze fine texture details (are there one or two lines?) on an already-isolated, centered image. Forcing one model to do both would require it to simultaneously learn coarse spatial localization and fine-grained line detection, which is much harder. Splitting the problem means each model can excel at what it's best at.

---

**Q: What is YOLOv8 and why did you use the "small" variant (yolov8s)?**

YOLO stands for "You Only Look Once." It's a real-time object detection architecture that processes the entire image in a single forward pass — no region proposals, no two-stage detection. YOLOv8 is the latest generation from Ultralytics. We used `yolov8s` (small) because:
1. The dataset is small (~a few hundred images of one simple class).
2. The hardware is an RTX 2080 — mid-range GPU.
3. A small model with transfer learning is enough for a single-class detection problem. Larger variants (m, l, x) would take longer to train and likely overfit.

---

**Q: What is transfer learning and where does it appear in this project?**

Transfer learning means starting from a model already trained on a large dataset rather than training from scratch. It appears in both models:
- **Phase 1:** YOLOv8s is initialized with weights pre-trained on the COCO dataset (80 classes, millions of images). We fine-tune it to detect cassettes.
- **Phase 2:** ResNet18 is initialized with ImageNet weights (`IMAGENET1K_V1`). We freeze all layers except the final fully connected layer, which we replace with a new one with 3 outputs.

This matters because both models start already knowing how to recognize edges, shapes, and textures — they just need to learn the task-specific patterns.

---

**Q: What are the 3 classes and what physically distinguishes them?**

A COVID lateral flow test cassette has two possible lines:
- **C-line (Control line):** Always appears if the test is valid.
- **T-line (Test line):** Appears only if COVID antigens are present.

| Class    | Lines visible | Meaning              |
|----------|---------------|----------------------|
| Negative | C only        | No COVID detected    |
| Positive | T + C         | COVID detected       |
| Invalid  | Neither       | Test failed, unusable|

---

## Section 2: Data Pipeline

**Q: Walk me through the full data pipeline from raw images to trained model.**

1. Collect images of cassettes → annotate with bounding boxes using a labeling tool → exported in COCO JSON format (`annotations/instances_default.json`).
2. `convert_to_yolo.py` — converts COCO format (absolute pixel coords `[x_min, y_min, w, h]`) to YOLO format (normalized center coords `[class x_center y_center w h]`, all values 0–1).
3. `setup_dataset.py` — shuffles with seed 42, splits 85% train / 15% val, copies images + label files into `dataset/images/` and `dataset/labels/`, writes `dataset.yaml`.
4. `train_model.py` — trains YOLOv8s on this structure.
5. For Phase 2: `crop_images.py` runs the Phase 1 model to auto-crop cassettes from images. Crops are manually sorted into `project2/train/positive/`, `negative/`, `invalid/`.
6. `augment_images.py` — pads each class up to 200 images using synthetic augmentations.
7. `train_classifier.py` — trains ResNet18 on the cropped dataset.

---

**Q: Why did you normalize YOLO labels to 0–1?**

Normalized coordinates make the labels resolution-independent. If you have a bounding box at pixel [320, 240] in a 640×480 image, that same box is at [0.5, 0.5] normalized. This means the label files work correctly regardless of image resolution, and YOLO can resize images internally without having to update labels.

---

**Q: Why 85% train and 15% validation, not 80/20?**

With a smaller dataset, every image matters more for training. 80/20 is more common with large datasets where you have thousands of images to spare. With a few hundred images, 85/15 gives the model more examples to learn from while still having a meaningful validation set. The validation set is not used for learning — only for checking performance after each epoch.

---

**Q: What is the difference between validation and test sets?**

| Set        | Used during training? | Purpose                                           |
|------------|-----------------------|---------------------------------------------------|
| Validation | Yes (after each epoch)| Tracks overfitting, picks the best model checkpoint|
| Test       | Never                 | Final one-time check on truly unseen images       |

The validation set shapes which model gets saved (`best.pt`). The test set simulates real-world deployment — the model has never seen these images at all.

---

**Q: What does `augment_images.py` do and why was it needed?**

It generates synthetic variations of real images to reach a target of 200 images per class. The augmentations applied are:
- Random rotation ±30 degrees
- Random horizontal flip (50% chance)
- Gaussian blur (radius 0.5–1.5, 50% chance)
- Brightness variation ±30%
- Contrast variation ±20%

This was needed because some classes (especially "invalid" and "positive") had very few real images. Without augmentation, the model would see too few examples and likely overfit or fail to generalize for underrepresented classes.

---

**Q: Why does `setup_dataset.py` use `random.seed(42)`?**

To make the train/val split **reproducible**. Without a fixed seed, every time you run the script you get a different split — you can't compare results, and someone else can't reproduce your experiment. `42` is a convention; the actual value doesn't matter, only that it's fixed.

---

## Section 3: Training & Model Details

**Q: What hyperparameters did you use for Phase 1 (YOLOv8)?**

From `train_model.py`:
- Model: `yolov8s.pt` (small)
- Epochs: 50
- Image size: 640×640
- Batch size: 16
- Device: GPU (`device=0`)
- Patience: 10 (early stopping)

---

**Q: What is early stopping and why did you use it?**

Early stopping (`patience=10`) means training automatically halts if the validation metric does not improve for 10 consecutive epochs. Without it, the model might train all 50 epochs even though it stopped improving at epoch 20 — wasting time and potentially starting to overfit. It's a safeguard that saves the best model and stops when there's nothing more to gain.

---

**Q: What hyperparameters did you use for Phase 2 (ResNet18)?**

From `train_classifier.py`:
- Backbone: ResNet18, pre-trained on ImageNet, **frozen** (all `requires_grad = False`)
- Final layer: replaced with `nn.Linear(512, 3)` — only this layer trains
- Epochs: 32
- Batch size: 16
- Learning rate: 0.001
- Optimizer: Adam (only `model.fc.parameters()`)
- Scheduler: StepLR — LR multiplied by 0.1 every 7 epochs
- Loss: CrossEntropyLoss

---

**Q: Why did you freeze the ResNet18 backbone?**

Because the dataset of cropped cassettes is small (~200 images per class after augmentation). If you unfreeze the entire network and fine-tune all layers, the model will overfit — it will memorize the training images perfectly but perform poorly on new images. By freezing the backbone (which already knows how to extract visual features from ImageNet), we only train the final 3-neuron output layer. This drastically reduces the number of trainable parameters and prevents overfitting on a small dataset.

---

**Q: What does the StepLR scheduler do?**

It reduces the learning rate over time. Every 7 epochs, it multiplies the LR by 0.1 (gamma). So: 0.001 → 0.0001 (epoch 7) → 0.00001 (epoch 14), etc. The idea is to take big steps early when far from a solution, then smaller steps later to fine-tune and avoid overshooting the minimum.

---

**Q: What is CrossEntropyLoss and why is it used for classification?**

CrossEntropyLoss is the standard loss function for multi-class classification. Internally it combines softmax (converts raw scores to probabilities) and negative log-likelihood (penalizes the model more heavily when it's confidently wrong). For 3 classes, the model outputs 3 numbers; the loss measures how far those predicted probabilities are from the true one-hot label.

---

**Q: What is the ImageNet normalization and why is it applied?**

```python
Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```
These are the mean and standard deviation of the ImageNet dataset per color channel (R, G, B). Since ResNet18 was pre-trained on ImageNet with this normalization, we must apply the same transformation to our input images. If we didn't, the pixel values would be in a completely different range from what the model was trained on, and the pre-trained features would be meaningless.

---

**Q: How are class indices assigned in Phase 2?**

`ImageFolder` assigns indices **alphabetically**. The three folders are `invalid`, `negative`, `positive` — so:
- `invalid` → 0
- `negative` → 1
- `positive` → 2

This is why in `predict.py` and `covid_pipeline.py`, `CLASSES = ["invalid", "negative", "positive"]` — the order must match exactly.

---

## Section 4: Inference Pipeline & UI

**Q: What happens step by step when you drop an image into the UI?**

1. `_choose_file()` or `_on_drop()` in `covid_ui.py` receives the path.
2. A **background thread** is spawned (`threading.Thread`) so the UI doesn't freeze during inference.
3. `run_pipeline()` is called:
   - YOLOv8 runs on the full image, returns bounding boxes.
   - The highest-confidence box is selected.
   - A 20-pixel padding is added and the cassette is cropped.
   - The crop is saved to `project2/last_crop.jpg`.
   - The crop is rotated to 0°, 90°, 180°, 270° and classified independently.
   - The 4 probability vectors are averaged (soft ensemble).
   - The class with the highest mean probability is the final result.
4. `self.after(0, ...)` sends the result back to the main thread to update the UI safely.

---

**Q: Why do you run inference on 4 rotations of the crop?**

Cassette photos are taken in many orientations — the device could be upside-down or sideways. By averaging predictions at 0°, 90°, 180°, and 270°, we make the final classification rotation-invariant. This is called **test-time augmentation (TTA)**. It costs 4× the inference time but significantly improves robustness on real-world photos where orientation is unpredictable.

---

**Q: Why use `mean` of the 4 probability vectors instead of majority vote?**

Averaging the softmax probability vectors (soft voting) is more nuanced than majority vote (hard voting). If three rotations give 80% confidence for "negative" and one gives 60% confidence for "positive", averaging correctly weights the strength of each prediction. Majority vote treats all four predictions equally regardless of confidence level.

---

**Q: Why does the crop have 20 pixels of padding added?**

```python
pad = 20
crop = img.crop((max(0, x1-pad), max(0, y1-pad), min(w, x2+pad), min(h, y2+pad)))
```
Bounding box predictions are never pixel-perfect — there's always slight regression error. The 20px padding ensures the crop includes the full cassette including its edges and lines, not a tightly cropped version that might cut off the very features the classifier needs to read.

---

**Q: Why is the UI inference run in a separate thread?**

Tkinter's main loop runs on the main thread and is not thread-safe. Running YOLOv8 + ResNet18 inference synchronously on the main thread would freeze the UI for the entire duration. By spawning a `daemon=True` background thread for `run_pipeline()`, the UI stays responsive (shows "Analyzing…"). The result is sent back to the main thread via `self.after(0, callback)`, which is the correct Tkinter-safe way to update the UI from a background thread.

---

**Q: What happens if no cassette is detected in the image?**

In `run_pipeline()`:
```python
if boxes is None or len(boxes) == 0:
    return None, None, None
```
The UI then shows "No cassette detected" in red. The pipeline does not proceed to classification. This is an explicit safety check — the classifier must never receive uncropped or irrelevant input.

---

## Section 5: "Why Not" Challenges

**Q: Why not use one end-to-end model (e.g., a classifier on the whole image)?**

An end-to-end classifier would need to simultaneously localize the cassette and read the lines — two very different tasks. It would also require much more labeled data (you'd need the cassette to be consistently positioned or the model would struggle). The two-stage approach is cleaner: YOLO is a proven detector, ResNet18 is a proven classifier. Each is used exactly as designed.

---

**Q: Why not use a larger model like ResNet50 or EfficientNet-B4?**

With ~200 images per class after augmentation, a larger model has far too many parameters and will overfit. ResNet18 with a frozen backbone has only 512×3 = 1,536 trainable parameters (the final layer). That's appropriate for the dataset size. A bigger model would memorize training images rather than learn to generalize.

---

**Q: Why not unfreeze the full ResNet18 and fine-tune everything?**

Same answer: dataset too small. Full fine-tuning on 200 images per class will overfit. The feature extractor from ImageNet is already useful — it can detect edges, textures, and shapes. We only need to teach the final layer what those features mean for our specific classes.

---

**Q: Why not just use one model (skip YOLO) and crop manually?**

Manual cropping doesn't scale and is not reproducible. YOLO automatically handles any photo regardless of background, camera angle, or cassette position. Manual cropping would require human intervention for every new image, making deployment impossible.

---

**Q: Why not use data from the internet instead of collecting your own?**

COVID cassette images are highly specific — brand, lighting, cassette size, background, photo angle vary enormously. A model trained on internet images would likely not match the distribution of real photos taken in the field. Collecting and annotating your own data ensures the model learns the specific visual characteristics of the cassettes you care about.

---

**Q: Why use Adam instead of SGD?**

Adam is adaptive — it adjusts the learning rate per parameter based on gradient history. For fine-tuning a single linear layer with few parameters and a small dataset, Adam converges faster and more reliably than SGD without careful LR tuning. SGD requires more hyperparameter tuning (momentum, LR schedule) to get right.

---

## Section 6: Code Navigation Cheatsheet

| What the teacher asks                        | File to show                  | Key line(s)         |
|----------------------------------------------|-------------------------------|---------------------|
| Where is the YOLO model trained?             | `train_model.py`              | Lines 4–15          |
| Where is the dataset split defined?          | `setup_dataset.py`            | Lines 30–31         |
| Where is the ResNet model defined?           | `train_classifier.py`         | Lines 44–47         |
| Where are the backbone weights frozen?       | `train_classifier.py`         | Lines 45–46         |
| Where is the training loop?                  | `train_classifier.py`         | Lines 56–90         |
| Where is the best model saved?               | `train_classifier.py`         | Lines 87–90         |
| Where is the full two-model pipeline?        | `covid_pipeline.py`           | Lines 32–73         |
| Where is the 4-rotation ensemble?            | `covid_pipeline.py`           | Lines 55–58         |
| Where is the crop + padding logic?           | `covid_pipeline.py`           | Lines 46–48         |
| Where is the UI defined?                     | `covid_ui.py`                 | Lines 59–255        |
| Where is the background thread for UI?       | `covid_ui.py`                 | Lines 196–204       |
| Where are augmentations applied?             | `augment_images.py`           | Lines 7–15          |
| Where is the YOLO label conversion?          | `convert_to_yolo.py`          | (full file)         |
| Where are class names defined?               | `covid_pipeline.py`, line 10  | `CLASSES = [...]`   |
| Where is "no cassette" handled?              | `covid_pipeline.py`           | Lines 36–38         |

---

## Section 7: Quick Concept Glossary (if you blank)

- **Epoch:** One full pass through the entire training dataset.
- **Batch size:** Number of images processed before the model weights are updated.
- **Overfitting:** Model memorizes training data but fails on new data.
- **Transfer learning:** Start from a pre-trained model instead of random weights.
- **Bounding box:** Rectangle (x1, y1, x2, y2) that marks an object's location.
- **Confidence score:** YOLO's certainty that a detected region contains the object (0–1).
- **Softmax:** Converts raw model scores into probabilities that sum to 1.
- **Loss function:** Measures how wrong the model is; training minimizes this.
- **Backbone:** The feature-extraction part of a neural network (frozen in Phase 2).
- **Fine-tuning:** Training a pre-trained model on a new specific task.
- **Data augmentation:** Artificially creating new training images by transforming existing ones.
- **Test-time augmentation (TTA):** Running inference multiple times with different transforms and averaging results.
