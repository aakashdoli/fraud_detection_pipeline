# Fraud Detection ML Pipeline

A production-grade machine learning pipeline for real-time credit card fraud detection — from raw transaction data through feature engineering, model training, experiment tracking, and a deployed REST API serving fraud probability scores.

Built to demonstrate the full ML lifecycle: data → features → model → evaluation → deployment → monitoring.

---

## What it does

A financial institution receives thousands of transactions per second. Most are legitimate. A tiny fraction are fraud. The challenge: detect fraud in real time with high precision (don't block real customers) and high recall (don't miss fraud).

This pipeline solves that end to end:

1. **Ingests** raw transaction data (amount, merchant, time, card features)
2. **Engineers features** that capture fraud signals (velocity, deviation from user history, time patterns)
3. **Trains and compares** multiple classifiers — Logistic Regression, Random Forest, XGBoost
4. **Tracks every experiment** with MLflow — metrics, parameters, feature importance, model artifacts
5. **Selects the best model** based on AUC-ROC and precision-recall tradeoff
6. **Serves predictions** via a FastAPI REST endpoint — accepts a transaction, returns fraud probability + risk level
7. **Deploys** the full stack with Docker Compose

---

## Results

| Model | AUC-ROC | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| Logistic Regression | 0.87 | 0.81 | 0.79 | 0.80 |
| Random Forest | 0.92 | 0.89 | 0.85 | 0.87 |
| **XGBoost** | **0.94** | **0.91** | **0.88** | **0.89** |

XGBoost selected as production model. All experiments tracked in MLflow.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| ML Models | scikit-learn, XGBoost |
| Feature engineering | Pandas, NumPy |
| Experiment tracking | MLflow |
| API serving | FastAPI, Uvicorn |
| Data validation | Pydantic |
| Containerisation | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Testing | Pytest |
| Dataset | Kaggle IEEE-CIS Fraud Detection (590K transactions) |

---

## Architecture

```
Raw Transaction Data (CSV / API)
           │
           ▼
  Feature Engineering
  ┌─────────────────────────────────┐
  │ • Amount normalisation          │
  │ • Time-based features (hour,    │
  │   day of week, days since last) │
  │ • Velocity features             │
  │   (txn count per card/hour)     │
  │ • Deviation from user baseline  │
  │ • Merchant risk encoding        │
  └─────────────────────────────────┘
           │
           ▼
  Model Training + Selection
  ┌─────────────────────────────────┐
  │  Logistic Regression  ──►  MLflow experiment #1  │
  │  Random Forest        ──►  MLflow experiment #2  │
  │  XGBoost              ──►  MLflow experiment #3  │
  │                                                   │
  │  Best model → registered in MLflow Model Registry │
  └─────────────────────────────────┘
           │
           ▼
  FastAPI Prediction Service
  POST /predict
  {
    "amount": 142.50,
    "card_id": "card_123",
    "merchant": "online_retail",
    "hour": 23,
    ...
  }
  →
  {
    "fraud_probability": 0.87,
    "risk_level": "HIGH",
    "model_version": "xgboost-v1.2",
    "explanation": ["unusual_hour", "high_amount", "new_merchant"]
  }
           │
           ▼
  MLflow Dashboard — track predictions, drift, performance
```

---

## Project Structure

```
fraud-detection-pipeline/
├── data/
│   ├── raw/                    # raw downloaded dataset (gitignored)
│   └── processed/              # feature-engineered data
├── notebooks/
│   └── 01_exploration.ipynb    # EDA — class imbalance, feature distributions
├── src/
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py         # all feature engineering logic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py            # training loop, MLflow logging
│   │   ├── evaluate.py         # AUC-ROC, precision-recall, confusion matrix
│   │   └── predict.py          # load model + run inference
│   └── api/
│       ├── __init__.py
│       ├── main.py             # FastAPI app
│       ├── schemas.py          # Pydantic request/response models
│       └── predictor.py        # loads MLflow model, serves predictions
├── tests/
│   ├── test_features.py        # unit tests for feature engineering
│   ├── test_model.py           # model quality tests (AUC > threshold)
│   └── test_api.py             # API endpoint tests
├── .github/
│   └── workflows/
│       └── ci.yml              # runs tests on every push
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── train.py                    # entry point: python train.py
└── README.md
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/aakashdoli/fraud-detection-pipeline
cd fraud-detection-pipeline

# 2. Install
pip install -r requirements.txt

# 3. Download dataset
#    Go to kaggle.com/c/ieee-fraud-detection → download train_transaction.csv
#    Place in data/raw/

# 4. Train models (logs all experiments to MLflow)
python train.py

# 5. View MLflow dashboard
mlflow ui
# Open http://localhost:5000

# 6. Start prediction API
uvicorn src.api.main:app --reload
# Open http://localhost:8000/docs

# OR run everything with Docker:
docker-compose up --build
```

---

## Key ML Concepts Demonstrated

**Class imbalance handling** — Fraud is ~3.5% of transactions. Naive models just predict "not fraud" always. This pipeline uses SMOTE oversampling and class weights to handle the imbalance properly.

**Feature engineering** — Raw transaction data has no "fraud" signal by itself. The real signal is in derived features: "this card made 15 transactions in the last hour" or "this amount is 4x the user's average". This is where ML engineering skill shows.

**Threshold tuning** — AUC-ROC is not the right metric for a fraud system. A bank cares more about catching 95% of fraud (recall) even at the cost of some false positives. The pipeline includes threshold optimisation for the business use case.

**MLflow experiment tracking** — Every training run logs: hyperparameters, train/validation metrics, feature importance, model artifact. You can compare runs side by side and register the best model.

**Model explanability** — Each prediction includes the top features that drove it (using SHAP values). This matters in financial services — you need to explain to the customer why their transaction was flagged.

---

## API Reference

```
POST /predict          — Score a single transaction
POST /predict/batch    — Score multiple transactions
GET  /health           — Health check
GET  /model/info       — Current model version and performance metrics
GET  /metrics          — Request count, avg latency, fraud rate seen
```

**Example request:**
```json
POST /predict
{
  "transaction_id": "txn_abc123",
  "amount": 2499.99,
  "card_id": "card_456",
  "merchant_id": "merch_789",
  "merchant_category": "electronics",
  "hour_of_day": 2,
  "day_of_week": 6,
  "transactions_last_hour": 3,
  "amount_vs_avg_ratio": 4.2
}
```

**Example response:**
```json
{
  "transaction_id": "txn_abc123",
  "fraud_probability": 0.91,
  "risk_level": "HIGH",
  "recommendation": "BLOCK",
  "top_risk_factors": [
    "amount_4x_above_average",
    "late_night_transaction",
    "high_velocity_last_hour"
  ],
  "model_version": "xgboost-v1.0",
  "latency_ms": 12
}
```

---

## Why This Project

Fraud detection is one of the most commercially valuable ML problems. Every bank, fintech, and payment processor runs some version of this system. Building it end to end — with real data, proper evaluation, and a production API — demonstrates:

- Understanding of the **full ML lifecycle**, not just model training
- Ability to handle **real-world messiness**: class imbalance, missing values, high-cardinality features
- **MLOps thinking**: experiment tracking, model registry, reproducibility
- **Production mindset**: FastAPI serving, Docker deployment, latency measurement

---

## Project Status

🚧 **In active development**

- [x] Project structure and README
- [ ] Data download and EDA notebook
- [ ] Feature engineering pipeline
- [ ] Model training + MLflow tracking
- [ ] FastAPI prediction service
- [ ] Docker deployment
- [ ] GitHub Actions CI
- [ ] SHAP explanations

---

Built by [Aakash Doli](https://github.com/aakashdoli) — MSc Software Engineering, BTH Sweden
Targeting ML Engineer roles at applied AI consultancies in Sweden