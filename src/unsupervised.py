"""Unsupervised learning — catching fraud WITHOUT labels.

Supervised learning (train.py) needs historical transactions already labelled
"fraud" / "not fraud". In real life new fraud patterns appear before anyone has
labelled them. Unsupervised anomaly detection answers a different question:
"which transactions look unusual compared to everything else?"

Isolation Forest idea: keep splitting the data with random cuts. A normal
transaction sits in a crowd, so it takes many cuts to isolate it. A weird
transaction sits alone, so it gets isolated in very few cuts. Few cuts = anomaly.

The model is trained with NO labels. We only use the labels afterwards, to
measure how many of the anomalies it flagged really were fraud.
"""

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src import config
from src.features import build_preprocessor


def run_isolation_forest(X_train, X_test, y_test, contamination=0.03):
    """contamination = the share of data we expect to be anomalies (~3% fraud)."""
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", IsolationForest(n_estimators=200, contamination=contamination,
                                      random_state=config.RANDOM_STATE)),
        ]
    )
    pipeline.fit(X_train)  # note: no y — the model never sees the fraud labels

    # predict() returns -1 for anomaly, 1 for normal; score_samples(): lower = more anomalous
    flagged = (pipeline.predict(X_test) == -1).astype(int)
    anomaly_score = -pipeline.score_samples(X_test)

    return {
        "model": "isolation_forest (unsupervised, no labels used in training)",
        "precision": round(precision_score(y_test, flagged, zero_division=0), 4),
        "recall": round(recall_score(y_test, flagged, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, anomaly_score), 4),
        "flagged_count": int(flagged.sum()),
    }
