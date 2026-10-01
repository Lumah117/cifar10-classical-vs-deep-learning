# Machine Vision
# Assignment 2
# Author: Christopher Mitchell
# Date: 10/03/25

# ========== Task 5.4 - CNN Training on Augmented Data (ResNet-18) ========== #

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Timer start
total_start_time = time.time()

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Using device: {device}")

# ========== Define Augmented Dataset Transform (Convert Grayscale to RGB) ==========
transform_augmented = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
    transforms.RandomHorizontalFlip(),            # Apply horizontal flip
    transforms.RandomRotation(15),                # Apply random rotation
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Apply random translation (10%)
    transforms.RandomResizedCrop(32, scale=(0.8, 1.2)), # Apply random scaling
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for CIFAR-10
])

# ========== Load Augmented Dataset ==========
print(" Loading augmented CIFAR-10 dataset...")
trainset_augmented = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_augmented)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization for test set
]))

trainloader = DataLoader(trainset_augmented, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
print(" Augmented dataset loaded!")

# ========== Load and Modify ResNet-18 Model ==========
print("\n Loading ResNet-18 model...")
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modify final layer for CIFAR-10 classification (10 classes)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 10)  

# Move model to device (GPU/CPU)
resnet18 = resnet18.to(device)
print(" Model created and moved to device!")

# ========== Define Loss and Optimizer ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=0.001)

# ========== Training Loop ==========
epochs = 10
print("\n Starting training with augmented data...\n")
training_start_time = time.time()

for epoch in range(epochs):
    print(f" Starting epoch {epoch + 1}/{epochs}...")
    epoch_start_time = time.time()
    running_loss = 0.0
    resnet18.train()
    
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"    Batch {i + 1}/{len(trainloader)} - Loss: {running_loss / (i + 1):.4f}")

    epoch_time = time.time() - epoch_start_time
    print(f" Epoch {epoch + 1} complete. Average Loss: {running_loss / len(trainloader):.4f} - Time: {epoch_time:.2f} seconds\n")

training_time = time.time() - training_start_time
print(f" Training complete in {training_time / 60:.2f} minutes!\n")

# ========== Evaluation ==========
print(" Evaluating on test set...")
resnet18.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet18(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

# ========== Print Evaluation Metrics ==========
print("\n Evaluation Metrics:")
print(f" Accuracy: {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall: {recall:.4f}")
print(f" F1 Score: {f1:.4f}")

total_time = time.time() - total_start_time
print(f"\n ResNet-18 evaluation complete in {total_time / 60:.2f} minutes")
