# Machine Vision
# Assignment 2
# Author: Christopher Mitchell
# Date: 25/02/25

# ========== Assignment 1 - Data Exploration & Preprocessing (Task 1.2) ========== #

# Import required libraries
import pickle
import numpy as np
import cv2  # OpenCV for grayscale conversion
import matplotlib.pyplot as plt

# CIFAR-10 class labels
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Function to load a CIFAR-10 batch
def load_cifar10_batch(filepath):
    with open(filepath, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
    return batch[b'data'], batch[b'labels']

# Load all training batches
data_batches = []
labels_batches = []

for i in range(1, 6):  # CIFAR-10 has 5 training batches
    data, labels = load_cifar10_batch(f'cifar-10-batches-py/data_batch_{i}')
    data_batches.append(data)
    labels_batches.append(labels)

# Combine all batches into a single training dataset
x_train = np.vstack(data_batches)  # Stack image data
y_train = np.hstack(labels_batches)  # Stack labels

# Load test batch
x_test, y_test = load_cifar10_batch('cifar-10-batches-py/test_batch')

# Correct Reshaping from (50000, 3072) → (50000, 32, 32, 3)
x_train = x_train.reshape(-1, 3, 32, 32)  # CIFAR-10 stores images as (3, 32, 32)
x_train = x_train.transpose(0, 2, 3, 1)   # Convert to (32, 32, 3)

x_test = x_test.reshape(-1, 3, 32, 32)  
x_test = x_test.transpose(0, 2, 3, 1)  

# Function to convert images to grayscale using the correct RGB formula
def convert_to_grayscale(images):
    return np.dot(images[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)  # Proper grayscale conversion

# Apply grayscale conversion
x_train_gray = convert_to_grayscale(x_train)
x_test_gray = convert_to_grayscale(x_test)

# Print shapes to confirm transformation
print("Training set shape (grayscale):", x_train_gray.shape)  # Expected: (50000, 32, 32)
print("Testing set shape (grayscale):", x_test_gray.shape)    # Expected: (10000, 32, 32)

# Visualize 10 grayscale images (one from each class)
plt.figure(figsize=(12, 6))
for i in range(10):
    idx = np.where(y_train == i)[0][0]  # Get first occurrence of each class
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train_gray[idx], cmap='gray')  # Show in grayscale
    plt.title(class_names[i])
    plt.axis('off')

plt.suptitle("CIFAR-10 Images in Grayscale", fontsize=14, fontweight='bold')
plt.show()


