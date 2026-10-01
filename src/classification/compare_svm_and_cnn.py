# Machine Vision
# Assignment 2
# Author: Christopher Mitchell
# Date: 04/03/25

# ========== Assignment 1 - Comparison of SVM Versus CNN (Task 4.4) ========== #

import matplotlib.pyplot as plt
import numpy as np

# Metrics
models = ['SVM (K=50)', 'SVM (K=100)', 'SVM (K=200)', 'CNN (ResNet-18)']
accuracy = [0.2209, 0.2478, 0.2400, 0.8085]
precision = [0.2184, 0.2450, 0.2396, 0.8090]
recall = [0.2209, 0.2478, 0.2400, 0.8085]
f1_score = [0.2167, 0.2424, 0.2364, 0.8082]

# Position of bars on X axis
x = np.arange(len(models))
width = 0.2  # Width of each bar

# Plotting
plt.figure(figsize=(14, 8))

plt.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#1f77b4')
plt.bar(x - 0.5*width, precision, width, label='Precision', color='#ff7f0e')
plt.bar(x + 0.5*width, recall, width, label='Recall', color='#2ca02c')
plt.bar(x + 1.5*width, f1_score, width, label='F1 Score', color='#d62728')

plt.title('Performance Comparison: SVM vs CNN (ResNet-18)', fontsize=18, weight='bold')
plt.xlabel('Model', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.ylim(0, 1)
plt.xticks(x, models, fontsize=12)
plt.yticks(np.arange(0, 1.05, 0.05), fontsize=12)  # 🔥 Smaller increments of 0.05
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
