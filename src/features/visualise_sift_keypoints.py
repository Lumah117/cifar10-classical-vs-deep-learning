# Machine Vision
# Assignment 2
# Author: Christopher Mitchell
# Date: 25/02/25

# ========== Assignment 1 - Feature Extraction (Task 2.1) ========== #

import pickle
import numpy as np
import cv2
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
x_train = np.vstack(data_batches)
y_train = np.hstack(labels_batches)

# Correct Reshaping from (50000, 3072) → (50000, 32, 32, 3)
x_train = x_train.reshape(-1, 3, 32, 32)
x_train = x_train.transpose(0, 2, 3, 1)

# Grayscale conversion function
def convert_to_grayscale(images):
    return np.dot(images[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

# Apply grayscale conversion
x_train_gray = convert_to_grayscale(x_train)

# Check shape
print("Training set shape (grayscale):", x_train_gray.shape)  # (50000, 32, 32)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Loop through each class and process one sample image per class
plt.figure(figsize=(15, 8))

for i, class_name in enumerate(class_names):
    sample_index = np.where(y_train == i)[0][0]  # Get first occurrence of each class
    sample_image = x_train_gray[sample_index]

    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(sample_image, None)

    if descriptors is None or len(keypoints) == 0:
        print(f"{class_name}: No keypoints detected.")
        image_with_keypoints = sample_image  # Show the plain image if no keypoints
    else:
        print(f"{class_name}: {len(keypoints)} keypoints detected.")
        image_with_keypoints = cv2.drawKeypoints(
            sample_image, keypoints, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

    # Plotting each image with keypoints
    plt.subplot(2, 5, i + 1)
    plt.imshow(image_with_keypoints, cmap='gray')
    plt.title(f"{class_name}\n{len(keypoints)} Keypoints")
    plt.axis('off')

plt.suptitle("SIFT Keypoints on One Image from Each Class", fontsize=16)
plt.tight_layout()
plt.show()


