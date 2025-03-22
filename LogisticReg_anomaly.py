import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from elpv_reader import load_dataset
import seaborn as sns

images, proba, types = load_dataset()

# Split data into training and testing sets for robust evaluation
X_train, X_test, y_train, y_test = train_test_split(proba, types, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
# Accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix for visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(types))
disp.plot()
plt.show()

# (Optional) Further analysis
# Explore other evaluation metrics (precision, recall, F1-score) using:
# from sklearn.metrics import precision_score, recall_score, f1_score
