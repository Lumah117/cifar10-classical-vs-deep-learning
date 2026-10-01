# Machine Vision
# Assignment 2
# Author: Christopher Mitchell
# Date: 04/03/25

# ========== Assignment 1 - Incorporating Data Augmentation (Task 5.3) ========== #

import matplotlib.pyplot as plt
import numpy as np

# Metrics before augmentation
metrics_pre_aug = {
    "K=50": [0.2209, 0.2184, 0.2209, 0.2167],
    "K=100": [0.2478, 0.2450, 0.2478, 0.2424],
    "K=200": [0.2400, 0.2396, 0.2400, 0.2364],
}

# Metrics after augmentation
metrics_post_aug = {
    "K=50": [0.2212, 0.2178, 0.2212, 0.2156],
    "K=100": [0.2238, 0.2218, 0.2238, 0.2202],
    "K=200": [0.2318, 0.2324, 0.2318, 0.2292],
}

# Labels for metrics
metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
codebook_sizes = ["K=50", "K=100", "K=200"]

# Set bar width
bar_width = 0.35
x = np.arange(len(metric_labels))

# Create subplots for each K value
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, K in enumerate(codebook_sizes):
    pre_aug = metrics_pre_aug[K]
    post_aug = metrics_post_aug[K]

    ax = axes[i]
    ax.bar(x - bar_width/2, pre_aug, bar_width, label="Pre-Augmentation", color="blue", alpha=0.7)
    ax.bar(x + bar_width/2, post_aug, bar_width, label="Post-Augmentation", color="orange", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.2, 0.26)  # Adjust y-axis for better visualization
    ax.set_title(f"{K} Codebook Size")
    ax.legend()
    ax.set_ylabel("Score")

plt.suptitle("Comparison of Pre-Augmentation vs Post-Augmentation Performance")
plt.tight_layout()
plt.show()
