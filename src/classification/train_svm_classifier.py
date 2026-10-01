# Machine Vision
# Assignment 2
# Author: Christopher Mitchell
# Date: 04/03/25

# ========== Assignment 1 - Classification with SVM (Task 4.2) ========== #

import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Codebook sizes we'll process
codebook_sizes = [50, 100, 200]

for K in codebook_sizes:
    print(f"\n Loading dataset split for K={K}...")

    try:
        with open(f"dataset_split_K{K}.pkl", "rb") as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print(f" ERROR: Dataset split for K={K} not found. Did you run Task 4.1?")
        continue

    # Extract the splits
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    print(f" Training set: {X_train.shape}")
    print(f" Validation set: {X_val.shape}")
    print(f" Test set: {X_test.shape}")

    # Initialize the SVM classifier
    svm = SVC(kernel='linear', random_state=42)

    print(f" Training SVM on K={K} visual words...")
    svm.fit(X_train, y_train)

    # Evaluate on validation set
    val_predictions = svm.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(y_val, val_predictions, average='weighted')
    val_recall = recall_score(y_val, val_predictions, average='weighted')
    val_f1 = f1_score(y_val, val_predictions, average='weighted')

    print(f"\n Validation Metrics for K={K}:")
    print(f"    Accuracy:  {val_accuracy:.4f}")
    print(f"    Precision: {val_precision:.4f}")
    print(f"    Recall:    {val_recall:.4f}")
    print(f"    F1 Score:  {val_f1:.4f}")

    # Evaluate on test set
    test_predictions = svm.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions, average='weighted')
    test_recall = recall_score(y_test, test_predictions, average='weighted')
    test_f1 = f1_score(y_test, test_predictions, average='weighted')

    print(f"\n Test Metrics for K={K}:")
    print(f"    Accuracy:  {test_accuracy:.4f}")
    print(f"    Precision: {test_precision:.4f}")
    print(f"    Recall:    {test_recall:.4f}")
    print(f"    F1 Score:  {test_f1:.4f}")

print("\n SVM training and evaluation complete for all codebook sizes!")


