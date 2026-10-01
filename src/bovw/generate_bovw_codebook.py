# Machine Vision
# Assignment 2
# Author: Christopher Mitchell
# Date: 25/02/25

# ========== Assignment 1 - Codebook Generation (Task 3.1) ========== #

import pickle
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans

# CIFAR-10 class labels
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Function to load CIFAR-10 batch
def load_cifar10_batch(filepath):
    with open(filepath, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
    return batch[b'data'], batch[b'labels']

# Load CIFAR-10 dataset
data_batches = []
labels_batches = []
for i in range(1, 6):  
    data, labels = load_cifar10_batch(f'cifar-10-batches-py/data_batch_{i}')
    data_batches.append(data)
    labels_batches.append(labels)

x_train = np.vstack(data_batches)
y_train = np.hstack(labels_batches)

# Reshape and convert to grayscale
x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (32, 32, 3)
x_train_gray = np.dot(x_train[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

# Save grayscale images for use in other scripts
with open("x_train_gray.pkl", "wb") as file:
    pickle.dump(x_train_gray, file)

# Extract SIFT descriptors
sift = cv2.SIFT_create()
all_descriptors = []

# Track if we've already printed descriptor info for each class
printed_classes = set()

for idx, (img, label) in enumerate(zip(x_train_gray[:5000], y_train[:5000])):
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    if descriptors is not None:
        all_descriptors.append(descriptors)
        
        # Check if we've already printed info for this class
        class_name = class_names[label]
        if class_name not in printed_classes:
            print(f"{class_name}: {len(descriptors)} descriptors extracted from image index {idx}.")
            printed_classes.add(class_name)
            
        # Stop if we've printed one example per class (10 classes)
        if len(printed_classes) == 10:
            break

# Stack all descriptors into a single array
all_descriptors = np.vstack(all_descriptors)

# Save extracted descriptors
with open("sift_descriptors.pkl", "wb") as file:
    pickle.dump(all_descriptors, file)

print(f"\nTotal descriptors extracted: {all_descriptors.shape}")

# Apply K-Means clustering
K = 25  # Default visual words
kmeans = MiniBatchKMeans(n_clusters=K, batch_size=5000, random_state=42)
kmeans.fit(all_descriptors)

# Save the trained K-Means model
with open("kmeans_model.pkl", "wb") as file:
    pickle.dump(kmeans, file)

print(f"\nK-Means clustering completed with {K} visual words.")
