"""Check the from-scratch algorithms give the same answers as scikit-learn."""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.scratch.knn import KNNScratch
from src.scratch.linear_regression import LinearRegressionScratch
from src.scratch.logistic_regression import LogisticRegressionScratch, sigmoid


def test_sigmoid_basic_values():
    assert sigmoid(0) == 0.5
    assert sigmoid(100) > 0.99
    assert sigmoid(-100) < 0.01


def test_linear_regression_matches_sklearn():
    X, y = make_regression(n_samples=300, n_features=4, noise=5, random_state=0)
    mine = LinearRegressionScratch(learning_rate=0.05, n_iterations=3000).fit(X, y)
    sk = LinearRegression().fit(X, y)
    np.testing.assert_allclose(mine.weights_, sk.coef_, rtol=1e-2)


def test_linear_regression_loss_goes_down():
    X, y = make_regression(n_samples=100, n_features=2, random_state=0)
    model = LinearRegressionScratch(n_iterations=200).fit(X, y)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_logistic_regression_close_to_sklearn():
    X, y = make_classification(n_samples=500, n_features=5, random_state=0)
    mine = LogisticRegressionScratch(learning_rate=0.5, n_iterations=3000).fit(X, y)
    sk = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)  # C large = no regularisation, like mine
    agreement = (mine.predict(X) == sk.predict(X)).mean()
    assert agreement > 0.97


def test_logistic_regression_probabilities_are_valid():
    X, y = make_classification(n_samples=200, n_features=4, random_state=1)
    proba = LogisticRegressionScratch().fit(X, y).predict_proba(X)
    assert proba.shape == (200, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_knn_matches_sklearn():
    X, y = make_classification(n_samples=300, n_features=4, random_state=2)
    X_train, X_test, y_train = X[:250], X[250:], y[:250]
    mine = KNNScratch(k=5).fit(X_train, y_train)
    sk = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    np.testing.assert_allclose(mine.predict_proba(X_test), sk.predict_proba(X_test))
