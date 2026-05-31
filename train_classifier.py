import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from pathlib import Path

TRAIN_DIR = "project2/train"
TEST_DIR  = "project2/test"
EPOCHS    = 32
BATCH     = 16
LR        = 0.001
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
test_data  = datasets.ImageFolder(TEST_DIR,  transform=test_transforms)

train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH, shuffle=False)

print(f"Classes: {train_data.classes}")
print(f"Train: {len(train_data)} images | Test: {len(test_data)} images")

# pretrained ResNet18 — freeze backbone, only train the final layer
# (with ~100 images/class, full fine-tuning overfits to training distribution)
model = models.resnet18(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))  # new layer is trainable by default
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

best_acc = 0.0

for epoch in range(EPOCHS):
    # --- train ---
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    scheduler.step()

    train_acc = 100 * correct / total

    # --- evaluate ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    test_acc = 100 * correct / total
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {running_loss/len(train_loader):.3f} | Train: {train_acc:.1f}% | Test: {test_acc:.1f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "project2/best_model.pth")
        print(f"  -> Best model saved ({best_acc:.1f}%)")

print(f"\nTraining complete. Best test accuracy: {best_acc:.1f}%")
print(f"Model saved to project2/best_model.pth")
