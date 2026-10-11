"""Step 1 — Data: create and load the transactions dataset.

Real bank transaction data is confidential, so this project generates a
realistic *synthetic* dataset. Fraud is rare (about 3% of rows), just like in
real life, and it follows simple patterns: unusually large amounts, late-night
transactions, foreign transactions, card-not-present purchases, brand-new
accounts, and many transactions in a short time.

A few messy rows (missing values and duplicates) are added on purpose so the
pipeline has real cleaning work to do.
"""

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def generate_transactions(n_rows: int = config.N_ROWS, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Create a synthetic table of card transactions with an is_fraud label."""
    rng = np.random.default_rng(seed)

    # First decide which rows are fraud (~3%), then generate each column so that
    # fraud rows *tend* to look suspicious. "Tend to" matters: some normal rows also
    # look odd and some fraud rows look normal, so the model can't be 100% right.
    is_fraud = rng.binomial(1, 0.03, size=n_rows)
    # About 15% of fraud is "disguised" and looks exactly like a normal transaction —
    # real fraudsters try hard to blend in. Only the other 85% shows suspicious patterns.
    fraud = (is_fraud == 1) & (rng.random(n_rows) < 0.85)

    def pick(normal_values, fraud_values):
        """Use fraud_values for fraud rows and normal_values for the rest."""
        return np.where(fraud, fraud_values, normal_values)

    merchant_category = pick(
        rng.choice(config.MERCHANT_CATEGORIES, size=n_rows, p=[0.30, 0.20, 0.15, 0.20, 0.08, 0.07]),
        rng.choice(config.MERCHANT_CATEGORIES, size=n_rows, p=[0.08, 0.05, 0.07, 0.35, 0.30, 0.15]),
    )
    avg_amount_30d = rng.gamma(shape=2.0, scale=1500, size=n_rows).round(2)  # customer's usual spend (₹)
    spend_multiplier = pick(
        rng.lognormal(mean=0.0, sigma=0.5, size=n_rows),   # normal: around usual spend
        rng.lognormal(mean=1.2, sigma=0.8, size=n_rows),   # fraud: often 3-4x usual spend
    )
    amount = (avg_amount_30d * spend_multiplier).round(2)
    night_hours = rng.integers(0, 6, size=n_rows)
    day_hours = rng.integers(0, 24, size=n_rows)
    hour = pick(day_hours, np.where(rng.random(n_rows) < 0.55, night_hours, day_hours))
    is_international = pick(rng.binomial(1, 0.04, n_rows), rng.binomial(1, 0.40, n_rows))
    card_present = pick(rng.binomial(1, 0.70, n_rows), rng.binomial(1, 0.25, n_rows))
    distance_from_home_km = pick(
        rng.exponential(scale=10, size=n_rows), rng.exponential(scale=90, size=n_rows)
    ).round(1)
    txn_count_last_24h = pick(rng.poisson(lam=2, size=n_rows), rng.poisson(lam=5, size=n_rows))
    account_age_days = pick(
        rng.integers(30, 3650, size=n_rows),
        np.where(rng.random(n_rows) < 0.5, rng.integers(1, 60, n_rows), rng.integers(30, 3650, n_rows)),
    )

    df = pd.DataFrame(
        {
            "amount": amount,
            "hour": hour,
            "merchant_category": merchant_category,
            "is_international": is_international,
            "card_present": card_present,
            "distance_from_home_km": distance_from_home_km,
            "txn_count_last_24h": txn_count_last_24h,
            "account_age_days": account_age_days,
            "avg_amount_30d": avg_amount_30d,
            config.TARGET: is_fraud,
        }
    )

    # Add some real-world mess: ~1% missing values and ~0.5% duplicate rows
    for col in ["distance_from_home_km", "merchant_category"]:
        missing_idx = rng.choice(n_rows, size=n_rows // 100, replace=False)
        df.loc[missing_idx, col] = np.nan
    duplicates = df.sample(n=n_rows // 200, random_state=seed)
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def load_data(path=config.DATA_PATH, n_rows: int = config.N_ROWS) -> pd.DataFrame:
    """Load the CSV. If it doesn't exist yet, generate it and save it first."""
    if not path.exists():
        logger.info("No dataset found — generating %s synthetic transactions", n_rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_transactions(n_rows).to_csv(path, index=False)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop exact duplicates and rows without a label.

    Missing *feature* values are not dropped here — they are filled in later by
    the imputer inside the preprocessing pipeline, so the same rule is applied
    to new transactions at prediction time.
    """
    before = len(df)
    df = df.drop_duplicates().dropna(subset=[config.TARGET]).reset_index(drop=True)
    logger.info("Cleaning: %s rows -> %s rows (removed duplicates)", before, len(df))
    return df
