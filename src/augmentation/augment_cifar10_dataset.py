# Machine Vision
# Assignment 2
# Author: Christopher Mitchell
# Date: 04/03/25

# ========== Assignment 1 - Incorporating Data Augmentation (Task 5.1) ========== #

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import cv2

print("\n Applying data augmentation to the CIFAR-10 training set...")

# Define augmentation pipeline
transform_augmented = transforms.Compose([
    transforms.RandomHorizontalFlip(),            # Random horizontal flip
    transforms.RandomRotation(15),                # Random rotation (degrees)
    transforms.RandomAffine(0, translate=(0.1, 0.1)), # Random translation (10%)
    transforms.RandomResizedCrop(32, scale=(0.8, 1.2)), # Random scaling
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load augmented dataset
trainset_augmented = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_augmented
)
trainloader_augmented = DataLoader(trainset_augmented, batch_size=1, shuffle=True)

# Convert images to NumPy format
x_train_augmented = []
y_train_augmented = []

print(" Converting dataset to NumPy format...")
for img, label in trainloader_augmented:
    img = img.numpy().squeeze(0)  # Remove batch dimension
    img = np.transpose(img, (1, 2, 0))  # Convert from (C, H, W) → (H, W, C)
    img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)  # Undo normalization

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    x_train_augmented.append(img_gray)
    y_train_augmented.append(label.item())

# Convert lists to NumPy arrays
x_train_augmented = np.array(x_train_augmented)
y_train_augmented = np.array(y_train_augmented)

print(f" Augmented dataset shape: {x_train_augmented.shape}, Labels shape: {y_train_augmented.shape}")

# Save augmented dataset
with open("x_train_gray_augmented.pkl", "wb") as file:
    pickle.dump(x_train_augmented, file)
with open("y_train_augmented.pkl", "wb") as file:
    pickle.dump(y_train_augmented, file)

print(" Augmented dataset saved successfully!")

# Visualise a few augmented images
def show_augmented_images():
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    plt.figure(figsize=(12, 6))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(x_train_augmented[i], cmap='gray')
        plt.title(class_names[y_train_augmented[i]])
        plt.axis('off')

    plt.suptitle(" Augmented CIFAR-10 Samples (Grayscale)", fontsize=16)
    plt.show()

# Show samples
show_augmented_images()
