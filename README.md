# Fraud Detection ML Pipeline

An end-to-end machine learning project that detects fraudulent card transactions — from raw data to a tested, deployed REST API.

It brings together the core ML skills in one place:

1. **ML pipeline** — data cleaning, feature engineering, preprocessing, training, cross-validation, evaluation and model saving
2. **ML algorithms from scratch** — Linear Regression, Logistic Regression and KNN written with only NumPy, and checked against scikit-learn
3. **Supervised vs unsupervised** — 4 supervised classifiers compared with an unsupervised Isolation Forest that never sees the fraud labels
4. **Model tuning & analysis** — GridSearchCV hyperparameter tuning, decision-threshold trade-off, feature importance
5. **Production serving** — the best model is served through a tested **FastAPI** service with CI

**Live API:** https://fraud-detection-api-ji5z.onrender.com/docs — try `/predict` in the browser (free hosting: the first request after inactivity can take ~50 seconds while the server wakes up)

---

## How it works

```
 transactions.csv
       │
       ▼
 1. Clean data            remove duplicate rows
       │
       ▼
 2. Feature engineering   amount_to_avg_ratio, is_night, log_amount
       │
       ▼
 3. Preprocess            fill missing values → scale numbers → one-hot encode categories
       │
       ▼
 4. Tune                  GridSearchCV over Random Forest hyperparameters (12 combinations)
       │
       ▼
 5. Train 4 models        Logistic Regression (scratch) · KNN (scratch)
                          Logistic Regression (sklearn) · Random Forest (sklearn)
       │                  + 5-fold stratified cross-validation
       ▼
 6. Evaluate & pick best  precision, recall, F1, ROC-AUC → best by ROC-AUC
       │                  + threshold analysis + feature importance
       │                  + unsupervised Isolation Forest for comparison
       ▼
 7. Save pipeline         models/fraud_model.joblib (preprocessing + model together)
       │
       ▼
 8. FastAPI service       /predict  /predict/batch  /model-info  /health
```

## Results (test set, 4,000 transactions, ~3% fraud)

### Supervised models (trained with fraud labels)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (from scratch) | 0.45 | 0.76 | 0.56 | 0.879 |
| KNN, k=5 (from scratch) | 0.99 | 0.59 | 0.74 | 0.841 |
| Logistic Regression (scikit-learn) | 0.46 | 0.76 | 0.57 | 0.878 |
| **Random Forest, tuned (scikit-learn) — deployed** | **0.94** | **0.67** | **0.78** | **0.882** |

### Unsupervised model (trained WITHOUT fraud labels)

| Model | Precision | Recall | ROC-AUC |
|---|---|---|---|
| Isolation Forest | 0.67 | 0.55 | 0.874 |

Without ever seeing a fraud label, Isolation Forest ranks transactions almost as well as the supervised models — useful for new fraud patterns that nobody has labelled yet.

### Threshold trade-off (deployed Random Forest)

| Threshold | Precision | Recall | Transactions flagged |
|---|---|---|---|
| 0.3 | 0.80 | 0.72 | 108 |
| **0.5 (default)** | **0.94** | **0.67** | **85** |
| 0.7 | 1.00 | 0.49 | 58 |

Lower threshold → catch more fraud but more false alarms. The right setting is a business decision (cost of missed fraud vs cost of blocking a genuine customer).

### Top features (deployed Random Forest)

`distance_from_home_km` · `amount_to_avg_ratio` · `account_age_days` · `txn_count_last_24h` · `amount`

`amount_to_avg_ratio` is an engineered feature (not in the raw data) and is the 2nd most important — feature engineering adds real value.

### Notes

- **Why not accuracy?** Only ~3% of transactions are fraud, so a model that always says "not fraud" would be 97% accurate and completely useless. Precision, recall, F1 and ROC-AUC show how well fraud is actually caught.
- **Why Random Forest?** Highest ROC-AUC on the test set; at the default threshold 94% of its alerts are real fraud and it catches 67% of fraud.
- **Tuning:** GridSearchCV picked 200 trees, no depth limit, min 1 row per leaf (CV ROC-AUC 0.944). The test-set score (0.882) is lower than the CV score — a reminder to always confirm on unseen data.
- **From-scratch check:** my Logistic Regression scores 0.879 vs scikit-learn's 0.879, and my KNN gives identical probabilities to scikit-learn's — so the maths is correct.

## From scratch vs scikit-learn

```
python -m src.compare_scratch

Algorithm             Metric      From scratch  scikit-learn
------------------------------------------------------------
Linear Regression     R2                0.4527        0.4526
Logistic Regression   ROC-AUC           0.8791        0.8785
KNN (k=5)             ROC-AUC           0.8407        0.8407
```

## Project structure

```
fraud-detection-ml-pipeline/
├── src/
│   ├── config.py              all settings (paths, column names, threshold)
│   ├── data.py                generate / load / clean the dataset
│   ├── features.py            feature engineering + preprocessing pipeline
│   ├── train.py               train, cross-validate, compare, save best model
│   ├── tuning.py              GridSearchCV hyperparameter tuning
│   ├── unsupervised.py        Isolation Forest anomaly detection (no labels)
│   ├── predict.py             load model, score one transaction
│   ├── compare_scratch.py     from-scratch vs scikit-learn comparison
│   └── scratch/
│       ├── linear_regression.py
│       ├── logistic_regression.py
│       └── knn.py
├── api/
│   ├── main.py                FastAPI app (endpoints, request logging)
│   └── schemas.py             Pydantic input validation
├── tests/                     17 pytest tests (algorithms, pipeline, API)
├── .github/workflows/tests.yml   CI: runs training + tests on every push
├── render.yaml                deployment config for Render
└── requirements.txt
```

## Run it locally

```bash
git clone https://github.com/itsnidhimehta/fraud-detection-ml-pipeline.git
cd fraud-detection-ml-pipeline
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

python -m src.train            # creates the data, trains 4 models, saves the best
python -m src.compare_scratch  # from-scratch vs scikit-learn
pytest -v                      # runs all tests
uvicorn api.main:app --reload  # starts the API → open http://127.0.0.1:8000/docs
```

## API

| Method | Endpoint | What it does |
|---|---|---|
| GET | `/health` | Is the service up and the model loaded? |
| GET | `/model-info` | Serving model, test scores, best hyperparameters, top features, threshold trade-off |
| POST | `/predict` | Score one transaction |
| POST | `/predict/batch` | Score up to 1,000 transactions |

Example request:

```json
POST /predict
{
  "amount": 18500, "hour": 2, "merchant_category": "electronics",
  "is_international": 1, "card_present": 0, "distance_from_home_km": 250,
  "txn_count_last_24h": 7, "account_age_days": 20, "avg_amount_30d": 2500
}
```

Response:

```json
{ "fraud_probability": 1.0, "is_fraud": true, "risk_level": "high" }
```

Invalid input (e.g. `"hour": 30`) is rejected with a `422` error before it reaches the model.

## About the data

Real bank transactions are confidential, so `src/data.py` generates a realistic synthetic dataset of 20,000 transactions with ~3% fraud. Fraud rows tend to have larger-than-usual amounts, night-time hours, foreign or card-not-present purchases, many recent transactions and new accounts — and ~15% of fraud is deliberately "disguised" to look normal, so no model can be perfect. A few missing values and duplicate rows are added so the cleaning steps have real work to do.

## Tech stack

Python · NumPy · Pandas · scikit-learn · FastAPI · Pydantic · Uvicorn · pytest · GitHub Actions · Render
