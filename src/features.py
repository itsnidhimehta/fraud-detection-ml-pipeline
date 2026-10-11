"""Step 2 — Feature engineering and preprocessing.

Feature engineering = creating new, more useful columns from the raw ones.
Preprocessing      = filling missing values, scaling numbers, and turning text
                     categories into numbers so a model can use them.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns. Works on a full table or a single transaction."""
    df = df.copy()
    # How unusual is this amount for this customer? 5.0 = five times their normal spend
    df["amount_to_avg_ratio"] = df["amount"] / (df["avg_amount_30d"] + 1)
    # Fraud happens more at night (midnight to 5 am)
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    # Amounts are very skewed (a few huge ones); log makes them easier to learn from
    df["log_amount"] = np.log1p(df["amount"])
    return df


def build_preprocessor() -> ColumnTransformer:
    """Numbers: fill gaps with the median, then scale.
    Categories: fill gaps with the most common value, then one-hot encode."""
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_steps, config.NUMERIC_FEATURES),
            ("categorical", categorical_steps, config.CATEGORICAL_FEATURES),
        ]
    )


def split_features_target(df: pd.DataFrame):
    """Separate the input columns (X) from the answer column (y)."""
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET].astype(int)
    return X, y
