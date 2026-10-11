"""Check data generation, cleaning and feature engineering."""

import pandas as pd

from src import config
from src.data import clean_data, generate_transactions
from src.features import add_features, build_preprocessor, split_features_target


def test_generated_data_has_expected_columns_and_fraud_rate():
    df = generate_transactions(n_rows=5000)
    assert config.TARGET in df.columns
    assert 0.01 < df[config.TARGET].mean() < 0.06  # fraud is rare


def test_clean_data_removes_duplicates():
    df = generate_transactions(n_rows=2000)
    cleaned = clean_data(df)
    assert cleaned.duplicated().sum() == 0
    assert len(cleaned) < len(df)


def test_engineered_features():
    df = pd.DataFrame([{"amount": 9000.0, "avg_amount_30d": 2999.0, "hour": 3}])
    out = add_features(df)
    assert out.loc[0, "amount_to_avg_ratio"] == 3.0
    assert out.loc[0, "is_night"] == 1
    assert out.loc[0, "log_amount"] > 0


def test_preprocessor_handles_missing_values():
    df = add_features(clean_data(generate_transactions(n_rows=1000)))
    X, _ = split_features_target(df)
    assert X.isna().sum().sum() > 0  # the raw data has gaps on purpose
    X_ready = build_preprocessor().fit_transform(X)
    assert not pd.isna(X_ready).any()  # ...and the preprocessor fills them
