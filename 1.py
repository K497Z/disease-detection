import matplotlib.pyplot as plt

# Method names and various metrics
methods = [
    "Base", "RandomRotation", "RandomVerticalFlip", "RandomHorizontalFlip",
    "ColorJitter", "RandomGrayscale", "RandomResizedCrop", "RandomErasing"
]

accuracy = [68.64, 70.34, 69.49, 69.49, 69.07, 66.10, 71.19, 68.64]
precision = [75.81, 72.55, 73.09, 73.67, 73.80, 70.99, 73.26, 73.10]
f1_score = [69.49, 66.98, 67.58, 69.01, 67.78, 65.59, 70.15, 67.58]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(methods, accuracy, marker='o', label='Accuracy', color='gold')
plt.plot(methods, precision, marker='s', label='Precision', color='orangered')
plt.plot(methods, f1_score, marker='^', label='F1-score', color='crimson')

plt.xlabel('Method')
plt.ylabel('Score (%)')
plt.title('Performance Comparison of Image Augmentation Methods')
plt.xticks(rotation=45)
plt.ylim(60, 80)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()show()
