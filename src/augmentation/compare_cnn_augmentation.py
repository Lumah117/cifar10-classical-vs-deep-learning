# Machine Vision
# Assignment 2
# Author: Christopher Mitchell
# Date: 10/03/25

# Bar Graph of CNN results before and after data augmentation

import matplotlib.pyplot as plt
import numpy as np

original_cnn_results = {
    "Accuracy": 0.8085,
    "Precision": 0.8090,
    "Recall": 0.8085,
    "F1 Score": 0.8082
}

augmented_cnn_results = {
    "Accuracy": 0.7335,
    "Precision": 0.7401,
    "Recall": 0.7335,
    "F1 Score": 0.7248
}

# Extracting values
metric_labels = list(original_cnn_results.keys())
original_values = list(original_cnn_results.values())
augmented_values = list(augmented_cnn_results.values())

# Bar width and positions
x = np.arange(len(metric_labels))
width = 0.35  

# Plotting the bar graph
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, original_values, width, label="Original CNN", color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, augmented_values, width, label="Augmented CNN", color='green', alpha=0.7)

# Labels and title
ax.set_xlabel("Evaluation Metrics", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Comparison of CNN Performance Before and After Data Augmentation", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=12)
ax.legend(fontsize=12)

# Show values on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height), 
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# Display the graph
plt.show()
