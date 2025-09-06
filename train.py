import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from models.emotion_cnn import EmotionCNN
from utils.preprocess import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets & Loaders
transform = get_transforms()
train_dataset = ImageFolder(root='dataset/train', transform=transform)
test_dataset = ImageFolder(root='dataset/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model, Loss, Optimizer
model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

transform = get_transforms()

# Training Loop
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save Model
torch.save(model.state_dict(), "emotion_model.pth")
print("Training complete. Model saved as emotion_model.pth")
