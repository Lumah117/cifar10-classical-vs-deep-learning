# Machine Vision
# Assignment 2
# Author: Christopher Mitchell
# Date: 04/03/25

# ========== Assignment 1 - Dataset Splitting (Task 4.1) ========== #

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Codebook sizes we're processing
codebook_sizes = [50, 100, 200]

#  Function to save CIFAR-10 labels if missing
def ensure_labels_exist():
    if not os.path.exists("y_train.pkl"):
        print(" 'y_train.pkl' not found. Generating labels now...")

        def load_cifar10_batch(filepath):
            with open(filepath, 'rb') as file:
                batch = pickle.load(file, encoding='bytes')
            return batch[b'data'], batch[b'labels']

        labels_batches = []
        for i in range(1, 6):
            _, labels = load_cifar10_batch(f'cifar-10-batches-py/data_batch_{i}')
            labels_batches.append(labels)

        y_train = np.hstack(labels_batches)

        with open("y_train.pkl", "wb") as file:
            pickle.dump(y_train, file)

        print(f" Labels generated and saved! Shape: {y_train.shape}")
    else:
        print(" 'y_train.pkl' found and ready to use!")

# Ensure labels are ready
ensure_labels_exist()

# Load the labels
with open("y_train.pkl", "rb") as file:
    y_train = pickle.load(file)

#  Match labels to BoVW features (first 30,000 only)
y_train = y_train[:30000]

#  Process for each codebook size
for K in codebook_sizes:
    print(f"\n Processing BoVW histograms for K={K} visual words...")

    bovw_file = f"bovw_histograms_{K}.pkl"
    if not os.path.exists(bovw_file):
        print(f" ERROR: '{bovw_file}' not found. Please generate BoVW histograms for K={K} first!")
        continue

    with open(bovw_file, "rb") as file:
        bovw_features = pickle.load(file)

    print(f"BoVW feature shape: {bovw_features.shape}")
    print(f"Labels shape: {y_train.shape}")

    # Split dataset (70% train, 15% validation, 15% test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        bovw_features, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    # (0.1765 × 85% = ~15% total)

    print(f"\nK={K} splits:")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Save the splits
    with open(f"dataset_split_K{K}.pkl", "wb") as file:
        pickle.dump({
            'X_train': X_train, 'y_train': y_train_split,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }, file)

    print(f" Dataset split for K={K} saved as 'dataset_split_K{K}.pkl'")

print("\n All dataset splits completed successfully!")
