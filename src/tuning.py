"""Hyperparameter tuning with GridSearchCV.

Parameters  = what the model learns by itself from data (e.g. the splits inside a tree).
Hyperparameters = settings WE choose before training (e.g. how many trees, how deep).

GridSearchCV tries every combination in the grid, scores each one with
cross-validation, and keeps the best. Like trying every oven temperature and
baking time combination and keeping the one that gives the best cake.
"""

import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src import config
from src.features import build_preprocessor

logger = logging.getLogger(__name__)

PARAM_GRID = {
    "model__n_estimators": [100, 200],   # number of trees
    "model__max_depth": [6, 10, None],   # how deep each tree can grow (None = no limit)
    "model__min_samples_leaf": [1, 5],   # minimum rows in a leaf (bigger = simpler trees)
}


def tune_random_forest(X_train, y_train, cv_folds=3):
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", RandomForestClassifier(class_weight="balanced",
                                             random_state=config.RANDOM_STATE, n_jobs=-1)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    logger.info("Grid search tried %s combinations. Best: %s (CV ROC-AUC %.4f)",
                len(search.cv_results_["params"]), best_params, search.best_score_)
    return best_params, {
        "combinations_tried": len(search.cv_results_["params"]),
        "best_params": best_params,
        "best_cv_roc_auc": round(search.best_score_, 4),
    }
