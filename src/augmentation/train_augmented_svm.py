# Machine Vision
# Assignment 2
# Author: Christopher Mitchell
# Date: 04/03/25

# ========== Assignment 1 - Incorporating Data Augmentation (Task 5.2) ========== #

import pickle
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Load the augmented grayscale dataset
with open("x_train_gray_augmented.pkl", "rb") as file:
    x_train_augmented = pickle.load(file)

with open("y_train_augmented.pkl", "rb") as file:
    y_train_augmented = pickle.load(file)

print(f" Loaded augmented dataset with shape: {x_train_augmented.shape}")

# SIFT feature extractor
sift = cv2.SIFT_create()

# Extract descriptors from augmented dataset
all_descriptors = []
for img in x_train_augmented:
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        all_descriptors.append(descriptors)

# Combine all descriptors
all_descriptors = np.vstack(all_descriptors)
print(f" Extracted total descriptors: {all_descriptors.shape}")

# Codebook sizes to evaluate
codebook_sizes = [50, 100, 200]

for K in codebook_sizes:
    print(f"\n Training codebook with K={K} visual words...")

    # Train K-Means
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=5000, random_state=42)
    kmeans.fit(all_descriptors)

    # Save the updated K-Means model
    with open(f"kmeans_model_augmented_{K}.pkl", "wb") as file:
        pickle.dump(kmeans, file)

    print(f" Codebook updated with {K} visual words!")

    # Generate BoVW histograms
    histograms = []
    for img in x_train_augmented:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        histogram = np.zeros(K)

        if descriptors is not None:
            cluster_assignments = kmeans.predict(descriptors)
            for cluster in cluster_assignments:
                histogram[cluster] += 1

        histogram /= (np.sum(histogram) + 1e-7)
        histograms.append(histogram)

    histograms = np.array(histograms)
    print(f" BoVW histograms generated with shape: {histograms.shape}")

    # Split the dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        histograms, y_train_augmented, test_size=0.1, random_state=42, stratify=y_train_augmented
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    print(f" Dataset split - Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    # Train the SVM
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)

    # Evaluate on validation set
    val_preds = svm.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_prec = precision_score(y_val, val_preds, average='weighted')
    val_rec = recall_score(y_val, val_preds, average='weighted')
    val_f1 = f1_score(y_val, val_preds, average='weighted')

    # Evaluate on test set
    test_preds = svm.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, average='weighted')
    test_rec = recall_score(y_test, test_preds, average='weighted')
    test_f1 = f1_score(y_test, test_preds, average='weighted')

    # Display results
    print(f"\n Validation Metrics (K={K}):")
    print(f"   Accuracy:  {val_acc:.4f}")
    print(f"   Precision: {val_prec:.4f}")
    print(f"   Recall:    {val_rec:.4f}")
    print(f"   F1 Score:  {val_f1:.4f}")

    print(f"\n Test Metrics (K={K}):")
    print(f"   Accuracy:  {test_acc:.4f}")
    print(f"   Precision: {test_prec:.4f}")
    print(f"   Recall:    {test_rec:.4f}")
    print(f"   F1 Score:  {test_f1:.4f}")

print("\n Task 5.2 complete - Features extracted, codebooks updated, and SVM retrained on augmented data!")
