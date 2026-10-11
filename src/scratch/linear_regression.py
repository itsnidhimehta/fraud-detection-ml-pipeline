"""Linear Regression from scratch with NumPy (gradient descent).

Idea: find a straight line (y = X·w + b) that is as close as possible to the
real values. "Close" is measured by Mean Squared Error (MSE). Gradient descent
nudges w and b a little at a time in the direction that lowers the error —
like walking downhill in small steps until you reach the bottom.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class LinearRegressionScratch(BaseEstimator, RegressorMixin):
    # BaseEstimator/RegressorMixin only add sklearn "plumbing" (get_params, score)
    # so this class plugs into sklearn pipelines. All the maths below is my own.

    def __init__(self, learning_rate=0.01, n_iterations=2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            y_pred = X @ self.weights_ + self.bias_
            error = y_pred - y

            # Gradients of MSE with respect to weights and bias
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * error.sum()

            # Take a small step downhill
            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            self.loss_history_.append(np.mean(error**2))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights_ + self.bias_
