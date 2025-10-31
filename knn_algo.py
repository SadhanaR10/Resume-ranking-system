from collections import Counter
import numpy as np
from scipy.linalg import norm  # Importing norm from scipy.linalg
from scipy.sparse import issparse, csr_matrix

def euclidean_distance(x1, x2):
    # Convert sparse matrices to dense matrices
    if issparse(x1):
        x1 = x1.toarray()
    if issparse(x2):
        x2 = x2.toarray()

    # Compute Euclidean distance between dense matrices
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes_ = np.unique(y_train)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        class_votes = Counter(k_nearest_labels)
        return class_votes.most_common(1)[0][0]

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return predictions

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        return accuracy

    def predict_proba(self, X):
        proba = []
        for x in X:
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            # Compute weights based on inverse distances with a small epsilon added to avoid division by zero
            weights = [1 / (d + 1e-6) for d in distances]

            # Normalize weights to sum to 1
            weights /= np.sum(weights)

            # Count weighted votes for each class
            class_votes = Counter({cls: 0 for cls in self.classes_})
            for idx, label in zip(k_indices, k_nearest_labels):
                class_votes[label] += weights[idx]

            # Normalize votes to probabilities
            class_prob = {cls: class_votes[cls] / self.k for cls in self.classes_}
            proba.append([class_prob[cls] for cls in self.classes_])

        return np.array(proba)
