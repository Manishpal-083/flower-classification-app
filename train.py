from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import Adam

print("THIS IS FINAL TRAIN FILE ✅")

train_dir = "data/train"
test_dir = "data/test"

device = "cpu"
print("device =", device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

print("classes =", train_data.classes)
print("train =", len(train_data), "test =", len(test_data))

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

model = models.mobilenet_v2(weights="IMAGENET1K_V1")
for p in model.features.parameters():
    p.requires_grad = False
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_data.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

def accuracy(loader):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total

for epoch in range(2):
    print("epoch started")
    model.train()
    running=0
    for x,y in train_loader:
        optimizer.zero_grad()
        out=model(x.to(device))
        loss=criterion(out,y.to(device))
        loss.backward()
        optimizer.step()
        running+=loss.item()

    tr=accuracy(train_loader)
    ts=accuracy(test_loader)
    print(f"epoch {epoch+1}/2 loss={running/len(train_loader):.4f} train_acc={tr:.3f} test_acc={ts:.3f}")

torch.save(model.state_dict(),"model.pth")
print("model saved ✅")
