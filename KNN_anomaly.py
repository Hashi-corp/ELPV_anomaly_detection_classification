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
