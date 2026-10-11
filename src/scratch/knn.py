"""K-Nearest Neighbours (KNN) from scratch with NumPy.

Idea: "you are like your neighbours". To classify a new transaction, find the
k most similar transactions in the training data (smallest Euclidean
distance) and look at their labels. If 3 of the 5 nearest were fraud,
the fraud probability is 3/5 = 0.6.

There is no real "training" — fit() just memorises the data. The work happens
at prediction time, which is why KNN gets slow on big datasets.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class KNNScratch(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5, batch_size=500):
        self.k = k
        self.batch_size = batch_size  # predict in chunks so memory stays small

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1])
        return self

    def _distances(self, X_batch):
        # Euclidean distance between every row in X_batch and every training row,
        # using (a - b)^2 = a^2 + b^2 - 2ab so it is fully vectorised (no Python loops)
        a2 = (X_batch**2).sum(axis=1)[:, None]
        b2 = (self.X_train_**2).sum(axis=1)[None, :]
        d2 = a2 + b2 - 2 * X_batch @ self.X_train_.T
        return np.sqrt(np.maximum(d2, 0))

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        fraud_share = []
        for start in range(0, len(X), self.batch_size):
            distances = self._distances(X[start : start + self.batch_size])
            nearest = np.argpartition(distances, self.k - 1, axis=1)[:, : self.k]  # indices of k closest
            fraud_share.append(self.y_train_[nearest].mean(axis=1))
        p = np.concatenate(fraud_share)
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
