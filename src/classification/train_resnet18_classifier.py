# Machine Vision
# Assignment 2
# Author: Christopher Mitchell
# Date: 04/03/25

# ========== Task 4.3 - CNN Classification with ResNet-18 ========== #

import time
import pickle
import numpy as np
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

# Load CIFAR-10 dataset
print(" Loading CIFAR-10 dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
print(" Dataset loaded!")

# Load pre-trained ResNet-18
print("\n Loading ResNet-18 model...")
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes
resnet18 = resnet18.to(device)
print(" Model created and moved to device!")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=0.001)

# Training loop
epochs = 10
print("\n Starting training...\n")
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
            print(f"   Batch {i + 1}/{len(trainloader)} - Loss: {running_loss / (i + 1):.4f}")

    epoch_time = time.time() - epoch_start_time
    print(f" Epoch {epoch + 1} complete. Average Loss: {running_loss / len(trainloader):.4f} - Time: {epoch_time:.2f} seconds\n")

training_time = time.time() - training_start_time
print(f" Training complete in {training_time / 60:.2f} minutes!\n")

# Evaluation
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

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print("\n Evaluation Metrics:")
print(f" Accuracy: {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall: {recall:.4f}")
print(f" F1 Score: {f1:.4f}")

total_time = time.time() - total_start_time
print(f"\n ResNet-18 evaluation complete in {total_time / 60:.2f} minutes!")
