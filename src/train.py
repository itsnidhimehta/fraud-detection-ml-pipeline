"""Step 3 — Train, evaluate and save the best model.

Run:  python -m src.train

What happens:
  1. Load + clean data, add engineered features
  2. Split into train (80%) and test (20%), keeping the same fraud % in both (stratify)
  3. Train 4 models, each inside the same pipeline (preprocess -> model):
        - Logistic Regression  (my from-scratch NumPy version)
        - KNN                  (my from-scratch NumPy version)
        - Logistic Regression  (scikit-learn)
        - Random Forest        (scikit-learn)
  4. 5-fold cross-validation on the training set (checks the score is stable, not luck)
  5. Score every model on the unseen test set
     (Random Forest's hyperparameters are first tuned with GridSearchCV — see tuning.py)
  6. Pick the best model by ROC-AUC and save it + its metrics
     (ROC-AUC = how well the model ranks fraud above normal transactions,
      independent of the 0.5 cut-off. F1, precision and recall are reported too.)
  7. Extra analysis on the best model:
       - threshold analysis: how precision/recall change if the 0.5 cut-off moves
       - feature importance: which columns drive the predictions
  8. Unsupervised comparison: Isolation Forest trained WITHOUT labels (unsupervised.py)
"""

import json
import logging
from datetime import datetime, timezone

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.data import clean_data, load_data
from src.features import add_features, build_preprocessor, split_features_target
from src.scratch.knn import KNNScratch
from src.scratch.logistic_regression import LogisticRegressionScratch
from src.tuning import tune_random_forest
from src.unsupervised import run_isolation_forest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_models(rf_params=None):
    rf_params = rf_params or {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5}
    return {
        "logistic_regression_scratch": LogisticRegressionScratch(
            learning_rate=0.1, n_iterations=1000, class_weight="balanced"
        ),
        "knn_scratch": KNNScratch(k=5),
        "logistic_regression_sklearn": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE
        ),
        "random_forest_sklearn": RandomForestClassifier(
            **rf_params,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate(model, X_test, y_test, threshold=config.DECISION_THRESHOLD):
    """Accuracy alone is misleading for fraud (predicting 'never fraud' scores ~97%),
    so we report precision, recall, F1 and ROC-AUC as well."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp),
                             "false_negative": int(fn), "true_positive": int(tp)},
    }


def threshold_analysis(model, X_test, y_test, thresholds=(0.3, 0.5, 0.7)):
    """Lower threshold = catch more fraud (higher recall) but more false alarms
    (lower precision). The business decides where to sit on this trade-off."""
    proba = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "flagged": int(pred.sum()),
        })
    return rows


def feature_importance(pipeline, top_n=8):
    """Which input columns matter most? Works for tree models (feature_importances_)
    and linear models (size of the coefficients)."""
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]  # "numeric__amount" -> "amount"
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_[0])
    elif hasattr(model, "weights_"):
        values = abs(model.weights_)
    else:
        return {}  # e.g. KNN has no per-feature weights
    ranked = sorted(zip(names, values), key=lambda pair: pair[1], reverse=True)[:top_n]
    return {name: round(float(v), 4) for name, v in ranked}


def run_training(n_rows: int = config.N_ROWS, cv_folds: int = 5, save: bool = True):
    # 1. Data
    df = clean_data(load_data(n_rows=n_rows))
    df = add_features(df)
    X, y = split_features_target(df)
    logger.info("Dataset: %s rows, fraud rate %.2f%%", len(df), 100 * y.mean())

    # 2. Train/test split (stratify keeps the fraud % equal in both parts)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    # 3. Tune Random Forest hyperparameters with GridSearchCV
    rf_params, tuning_summary = tune_random_forest(X_train, y_train)

    # 4-5. Train, cross-validate and test each model
    results, pipelines = {}, {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    for name, model in get_models(rf_params).items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", model)])

        cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        metrics["cv_f1_mean"] = round(cv_f1.mean(), 4)
        metrics["cv_f1_std"] = round(cv_f1.std(), 4)

        results[name], pipelines[name] = metrics, pipeline
        logger.info(
            "%-30s F1=%.3f  precision=%.3f  recall=%.3f  ROC-AUC=%.3f  (CV F1 %.3f ± %.3f)",
            name, metrics["f1"], metrics["precision"], metrics["recall"],
            metrics["roc_auc"], metrics["cv_f1_mean"], metrics["cv_f1_std"],
        )

    # 6. Pick and save the best model (highest ROC-AUC on the test set)
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    logger.info("Best model: %s", best_name)

    # 7. Extra analysis on the best model
    thresholds = threshold_analysis(pipelines[best_name], X_test, y_test)
    importance = feature_importance(pipelines[best_name])
    logger.info("Top features: %s", list(importance)[:5])

    # 8. Unsupervised comparison (no labels used in training)
    unsupervised = run_isolation_forest(X_train, X_test, y_test)
    logger.info("Isolation Forest (no labels): precision=%.3f recall=%.3f ROC-AUC=%.3f",
                unsupervised["precision"], unsupervised["recall"], unsupervised["roc_auc"])

    summary = {
        "best_model": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": len(df),
        "fraud_rate": round(float(y.mean()), 4),
        "decision_threshold": config.DECISION_THRESHOLD,
        "results": results,
        "hyperparameter_tuning": tuning_summary,
        "threshold_analysis": thresholds,
        "feature_importance": importance,
        "unsupervised_isolation_forest": unsupervised,
    }
    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipelines[best_name], config.MODEL_PATH)
        config.METRICS_PATH.write_text(json.dumps(summary, indent=2))
        logger.info("Saved model to %s and metrics to %s", config.MODEL_PATH, config.METRICS_PATH)
    return summary


if __name__ == "__main__":
    run_training()
