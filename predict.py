import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "project2/best_model.pth"
CLASSES    = ["invalid", "negative", "positive"]  # alphabetical — matches ImageFolder order
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

img_path = sys.argv[1] if len(sys.argv) > 1 else input("Image path: ")
img = Image.open(img_path).convert("RGB")
tensor = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    probs = torch.softmax(model(tensor), dim=1)[0]

for cls, p in zip(CLASSES, probs):
    print(f"  {cls:10s}: {p*100:.1f}%")

print(f"\nResult: {CLASSES[probs.argmax()]}")
