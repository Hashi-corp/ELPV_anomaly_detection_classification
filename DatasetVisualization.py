import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from elpv_reader import load_dataset
import seaborn as sns

paths, images, probs, types = load_dataset()

N = images.shape[0]
pixel_range = np.amin(images), np.amax(images)

print(N)
print(pixel_range)
prob_values, probs_counts = np.unique(probs, return_counts=True)
prob_names = ["0", "⅓", "⅔", "1"]
assert len(prob_values) == len(prob_names)
type_values, type_counts = np.unique(types, return_counts=True)
plot_fmt = 'svg'
rng = np.random.default_rng()

if True:
    print("Dataset Characteristics")
    print("-----------------------")
    print("Input:")
    print(f"  {N} images of size {'x'.join(str(d) for d in images.shape[1:])}")
    print(
        f"  Pixels are in range {pixel_range[0]}-{pixel_range[1]} "
        f"of type {images.dtype}"
    )
    print("Labels:")
    print(
        f"- defect probabilities ({', '.join('{:.2f}'.format(p) for p in prob_values)})"
    )
    print(f"- PV cell types ({', '.join(type_values)})")
    print()

    n_samples = 3
    plt.figure(figsize=(6.4, n_samples * 3.2))
    k = 0
    for i, type_ in enumerate(["mono"] * n_samples + ["poly"] * n_samples):
        for j in range(4):
            prob = j / 3
            sample = images[(probs == prob) & (types == type_)][i % n_samples]
            k += 1
            plt.subplot(2 * n_samples, 4, k)
            plt.imshow(sample)
            plt.axis("image")
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                plt.title(f"p = {round(j/3, 3)}")
            if j == 0 and i % n_samples == 0:
                plt.ylabel(f"← {type_}")
    plt.tight_layout()
    plt.savefig(f"solarcell_samples.{plot_fmt}")
    plt.show()

    plt.figure(figsize=(6.4, 3.6))
    plt.suptitle("Distribution of Labels")

    plt.subplot(1, 2, 1)
    plt.hist(types, bins=2, rwidth=0.8)
    plt.xlabel("Cell types")

    plt.subplot(1, 2, 2)
    plt.hist(probs, bins=4, rwidth=0.8)
    plt.xticks(prob_values, prob_names)
    plt.xlabel("Defect Probabilities")
    plt.ylabel("Counts")
    plt.tight_layout()

    plt.savefig(f"dists.{plot_fmt}", dpi=200)
    plt.show()




'''outlier_label = "outlier"  # Adjust if needed

# Use probabilities as labels for outlier detection (can be modified)
normal_indices = np.where(probs < 0.5)[0]
outlier_indices = np.where(probs >= 0.5)[0]

# Distance-based outlier detection (without features)
distances = np.zeros((N, N))  # Pre-allocate for efficiency

# Calculate pairwise image distances (consider using a more efficient method for large datasets)
for i in range(N):
    for j in range(i + 1, N):  # Avoid redundant calculations
        distances[i, j] = np.linalg.norm(images[i] - images[j])  # Euclidean distance
        distances[j, i] = distances[i, j]  # Fill the other half for symmetry

# Define a threshold for outlier distance (adjust as needed)
distance_threshold = np.percentile(distances.flatten(), 95)

# Identify outliers based on distance
outlier_indices_distance = np.where(distances.max(axis=1) > distance_threshold)[0]

# Combine outlier detection methods (optional)
all_outliers = np.unique(np.concatenate((outlier_indices, outlier_indices_distance)))

# Split data into training and testing sets (assuming "types" is the target variable)
X_train, X_test, y_train, y_test = train_test_split(images[normal_indices], types[normal_indices], test_size=0.2)

# Train the KNN model on normal data labels ("types")
knn = KNeighborsClassifier(n_neighbors=5)  # Adjust k as needed
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

# Print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Weighted precision for imbalanced classes (optional)
recall = recall_score(y_test, y_pred, average='weighted')  # Weighted recall for imbalanced classes (optional)
f1 = f1_score(y_test, y_pred, average='weighted')  # Weighted F1-score for imbalanced classes (optional)

print("KNN Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Print identified outliers (optional)
print("\nOutliers (probability-based):", outlier_indices.shape[0])
print("Outliers (distance-based):", outlier_indices_distance.shape[0])
print("Total Identified Outliers:", all_outliers.shape[0])
'''
