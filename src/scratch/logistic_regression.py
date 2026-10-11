"""Logistic Regression from scratch with NumPy (gradient descent).

Same idea as linear regression, but the output is squeezed through the
sigmoid function so it becomes a probability between 0 and 1
(e.g. 0.92 = "92% likely to be fraud"). The error measure is log loss
(binary cross-entropy), which punishes confident wrong answers heavily.

class_weight="balanced": fraud is only ~3% of the data, so without help the
model could just say "not fraud" every time and be 97% "accurate". Balancing
makes each fraud example count more in the loss, so the model pays attention
to the rare class.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid(z):
    z = np.clip(z, -500, 500)  # avoid overflow in exp for very large values
    return 1 / (1 + np.exp(-z))


class LogisticRegressionScratch(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.1, n_iterations=1000, class_weight=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.class_weight = class_weight

    def _sample_weights(self, y):
        if self.class_weight != "balanced":
            return np.ones_like(y, dtype=float)
        n = len(y)
        n_pos = max(y.sum(), 1)
        n_neg = max(n - y.sum(), 1)
        # Same formula sklearn uses: n_samples / (n_classes * count_of_class)
        return np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        sample_weight = self._sample_weights(y)

        self.classes_ = np.array([0, 1])
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            p = sigmoid(X @ self.weights_ + self.bias_)
            error = (p - y) * sample_weight

            # Gradients of log loss
            dw = (X.T @ error) / n_samples
            db = error.sum() / n_samples

            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            eps = 1e-12
            loss = -np.mean(sample_weight * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
            self.loss_history_.append(loss)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = sigmoid(X @ self.weights_ + self.bias_)
        return np.column_stack([1 - p, p])  # [P(not fraud), P(fraud)] — same shape as sklearn

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
