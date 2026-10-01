# Machine Vision
# Assignment 2
# Author: Christopher Mitchell
# Date: 25/02/25

# ========== Assignment 1 - Data Exploration & Preprocessing (Task 1.1) ========== #

# Code to load imports
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Code to load CIFAR-10 batch
with open('cifar-10-batches-py/data_batch_1', 'rb') as file:
    batch = pickle.load(file, encoding='bytes')

# Code to declare CIFAR-10 class labels
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Code to extract image data and labels
images = batch[b'data']
labels = batch[b'labels']

# Convert to NumPy array
images = np.array(images, dtype=np.uint8)  # Ensure correct dtype
labels = np.array(labels)

# Correctly reshape images from (10000, 3072) → (10000, 32, 32, 3)
images = images.reshape(-1, 3, 32, 32)  # CIFAR-10 stores as (channel, height, width)
images = images.transpose(0, 2, 3, 1)  # Reorder to (batch, height, width, channels)

# Normalize pixel values to 0-1 for proper display
images = images / 255.0

# Code to plot 10 sample images (one per class)
plt.figure(figsize=(12, 6))
for i in range(10):
    idx = np.where(labels == i)[0][0]  # Get first occurrence of class
    plt.subplot(2, 5, i+1)
    plt.imshow(images[idx])  # Now properly displayed in RGB
    plt.title(class_names[i])
    plt.axis('off')

plt.show()
