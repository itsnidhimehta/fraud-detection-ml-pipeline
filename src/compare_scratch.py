"""Sanity check: are my from-scratch algorithms correct?

Run:  python -m src.compare_scratch

Trains each from-scratch model and the matching scikit-learn model on the same
data and prints the scores side by side. If my maths is right, the numbers
should be almost identical.

- Linear Regression -> scikit-learn's built-in diabetes dataset (a regression task)
- Logistic Regression and KNN -> the fraud transactions dataset
"""

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src import config
from src.data import clean_data, load_data
from src.features import add_features, build_preprocessor, split_features_target
from src.scratch.knn import KNNScratch
from src.scratch.linear_regression import LinearRegressionScratch
from src.scratch.logistic_regression import LogisticRegressionScratch


def compare_linear_regression():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    mine = LinearRegressionScratch(learning_rate=0.05, n_iterations=5000).fit(X_train, y_train)
    sk = LinearRegression().fit(X_train, y_train)
    return r2_score(y_test, mine.predict(X_test)), r2_score(y_test, sk.predict(X_test))


def compare_classifiers():
    df = add_features(clean_data(load_data()))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    prep = build_preprocessor().fit(X_train)
    X_train, X_test = prep.transform(X_train), prep.transform(X_test)

    pairs = {
        "Logistic Regression": (
            LogisticRegressionScratch(class_weight="balanced"),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        ),
        "KNN (k=5)": (KNNScratch(k=5), KNeighborsClassifier(n_neighbors=5)),
    }
    rows = []
    for name, (mine, sk) in pairs.items():
        mine.fit(X_train, y_train)
        sk.fit(X_train, y_train)
        rows.append(
            (name,
             roc_auc_score(y_test, mine.predict_proba(X_test)[:, 1]),
             roc_auc_score(y_test, sk.predict_proba(X_test)[:, 1]))
        )
    return rows


if __name__ == "__main__":
    print(f"{'Algorithm':<22}{'Metric':<10}{'From scratch':>14}{'scikit-learn':>14}")
    print("-" * 60)
    mine_r2, sk_r2 = compare_linear_regression()
    print(f"{'Linear Regression':<22}{'R2':<10}{mine_r2:>14.4f}{sk_r2:>14.4f}")
    for name, mine_auc, sk_auc in compare_classifiers():
        print(f"{name:<22}{'ROC-AUC':<10}{mine_auc:>14.4f}{sk_auc:>14.4f}")
