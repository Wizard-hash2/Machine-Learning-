# Import necessary libraries for numerical operations, data generation, plotting, and modeling.
import numpy as np  
from sklearn.datasets import make_classification  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import classification_report  

# Generate a synthetic dataset for classification.
# X contains the feature values; y contains the true labels.
X, y = make_classification(n_samples=200,     # Create 200 samples.
                           n_features=2,        # Two features per sample.
                           n_redundant=0,       # No redundant features.
                           n_informative=2,     # Both features are informative.
                           n_clusters_per_class=1,
                           random_state=42)     # Ensure reproducibility.

# Visualize the generated dataset.
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title("Synthetic Classification Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Initialize the Logistic Regression classifier.
clf = LogisticRegression()

# Train (fit) the classifier on our dataset (features X and labels y).
clf.fit(X, y)

# Predict the labels on the training data.
y_pred = clf.predict(X)

# Evaluate the classifier's performance by printing a classification report.
print(classification_report(y, y_pred))
