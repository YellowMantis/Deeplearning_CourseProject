import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image

YOLO_MODEL  = "runs/detect/runs/covid_lft-3/weights/best.pt"
CLASS_MODEL = "project2/best_model.pth"
CLASSES     = ["invalid", "negative", "positive"]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# load YOLO
detector = YOLO(YOLO_MODEL)

# load classifier
classifier = models.resnet18()
classifier.fc = nn.Linear(classifier.fc.in_features, 3)
classifier.load_state_dict(torch.load(CLASS_MODEL, map_location=DEVICE))
classifier.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

img_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter image path: ")

# step 1 — detect and crop
print("Step 1: Detecting COVID cassette...")
results = detector(img_path, verbose=False)
boxes = results[0].boxes

if boxes is None or len(boxes) == 0:
    print("No COVID test device detected in the image.")
    sys.exit(1)

best = boxes[boxes.conf.argmax()]
x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
conf = float(best.conf[0])
print(f"  Cassette found (confidence: {conf*100:.1f}%)")

img = Image.open(img_path).convert("RGB")
w, h = img.size
pad = 20
crop = img.crop((max(0, x1-pad), max(0, y1-pad), min(w, x2+pad), min(h, y2+pad)))
crop.save("project2/last_crop.jpg")
print("  Crop saved to project2/last_crop.jpg")

# step 2 — classify (average over 4 rotations to handle orientation)
print("Step 2: Classifying test result...")
import torchvision.transforms.functional as TF
angles = [0, 90, 180, 270]
with torch.no_grad():
    tensors = torch.stack([transform(TF.rotate(crop, a)) for a in angles]).to(DEVICE)
    probs = torch.softmax(classifier(tensors), dim=1).mean(0)

print("\n--- Probabilities ---")
for cls, p in zip(CLASSES, probs):
    bar = "█" * int(p * 30)
    print(f"  {cls:10s}: {p*100:5.1f}%  {bar}")

result = CLASSES[probs.argmax()]
print(f"\n=== RESULT: {result.upper()} ===")

if result == "positive":
    print("Two lines detected (T and C) — COVID POSITIVE")
elif result == "negative":
    print("One line detected (C only) — COVID NEGATIVE")
else:
    print("No lines detected — TEST INVALID")
