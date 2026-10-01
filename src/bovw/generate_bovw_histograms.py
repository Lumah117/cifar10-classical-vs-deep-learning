# Machine Vision
# Assignment 2
# Author: Christopher Mitchell
# Date: 25/02/25

# ========== Assignment 1 - Codebook Generation (Task 3.3) ========== #

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Codebook sizes to process
codebook_sizes = [50, 100, 200]

# Load grayscale images
with open("x_train_gray.pkl", "rb") as file:
    x_train_gray = pickle.load(file)

# Initialize SIFT with even stronger feature detection
sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.01, edgeThreshold=5, sigma=1.2)

# Function to create BoVW histogram representation
def create_bovw_histogram(descriptors, kmeans, K, img_idx):
    """Convert descriptors into a histogram of visual words using KMeans clusters."""
    histogram = np.ones(K) * 1e-2  # Initialize with small nonzero values

    if descriptors is not None and descriptors.shape[0] > 0:
        cluster_assignments = kmeans.predict(descriptors)

        for cluster in cluster_assignments:
            histogram[cluster] += 1

        # Print cluster distribution for first 5 images
        unique_clusters = np.unique(cluster_assignments)
        if img_idx < 5:
            print(f"[Image {img_idx}] Using {len(unique_clusters)}/{K} clusters. First 10: {unique_clusters[:10]}")

    # Normalize histogram
    histogram /= np.sum(histogram)

    return histogram

# Dictionary to store histograms for different codebook sizes
image_histograms = {}

# Compute BoVW histograms for each codebook size
for K in codebook_sizes:
    print(f"\nProcessing images for K={K} visual words...")

    # Load the trained K-Means model for this codebook size
    with open(f"kmeans_model_{K}.pkl", "rb") as file:
        kmeans = pickle.load(file)

    histograms = []
    for img_idx, img in enumerate(x_train_gray[:30000]):  # Use 30,000 images
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        # Debug: Print keypoint counts for the first 10 images
        if img_idx < 10:
            if descriptors is None:
                print(f"Image {img_idx}: No keypoints detected.")
            else:
                print(f"Image {img_idx}: {len(keypoints)} keypoints detected (Goal: 100+), Descriptor shape: {descriptors.shape}")

        if descriptors is not None:
            histograms.append(create_bovw_histogram(descriptors, kmeans, K, img_idx))
        else:
            histograms.append(np.zeros(K))  # If no keypoints, return empty histogram

    histograms = np.array(histograms)
    image_histograms[K] = histograms  # Store histograms for analysis

    # Save BoVW histograms for this K
    with open(f"bovw_histograms_{K}.pkl", "wb") as file:
        pickle.dump(histograms, file)

    print(f"Generated BoVW histograms for {len(histograms)} images with {K} visual words.")

# ========== Display a Sample Histogram for Each Codebook Size ========== #

# Select a random image index that has keypoints
sample_idx = np.random.randint(0, 30000)
while np.count_nonzero(image_histograms[100][sample_idx]) == 0:  # Ensure we pick an image with keypoints
    sample_idx = np.random.randint(0, 30000)

plt.figure(figsize=(12, 4))
for i, K in enumerate(codebook_sizes):
    plt.subplot(1, 3, i + 1)
    
    # Print histogram values to debug
    print(f"\nHistogram values for K={K} (first 10 bins):", image_histograms[K][sample_idx][:10])

    plt.bar(range(K), image_histograms[K][sample_idx], color='blue')
    plt.title(f"BoVW Histogram (K={K})")
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")

plt.suptitle(f"Histogram Representation for Sample Image {sample_idx}")
plt.show()


