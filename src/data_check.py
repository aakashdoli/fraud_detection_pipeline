"""
data_check.py
-------------
Quick sanity-check script for the raw training data.

Usage:
    python src/data_check.py
"""

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train_transaction.csv"
TARGET_COL = "isFraud"
TOP_N_MISSING = 5


def load_data(path: Path) -> pd.DataFrame:
    """Load CSV from *path*, raising a clear error if the file is absent."""
    if not path.exists():
        print(f"[ERROR] Data file not found: {path}")
        print("  → Make sure 'data/raw/train_transaction.csv' exists in the project root.")
        sys.exit(1)
    print(f"[INFO]  Loading data from: {path}")
    return pd.read_csv(path)


def report_shape(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    print(f"\n{'='*50}")
    print(f"  Dataset shape : {rows:,} rows × {cols} columns")
    print(f"{'='*50}")


def report_fraud_rate(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns:
        print(f"[WARN]  Column '{TARGET_COL}' not found – skipping fraud-rate check.")
        return

    total = len(df)
    fraud_count = df[TARGET_COL].sum()
    fraud_pct = fraud_count / total * 100

    print(f"\n  Fraud cases   : {int(fraud_count):,} / {total:,}")
    print(f"  Fraud rate    : {fraud_pct:.4f}%")
    print(f"  Class balance : 1:{(total - fraud_count) / max(fraud_count, 1):.1f}  (legit:fraud)")


def report_top_missing(df: pd.DataFrame, n: int = TOP_N_MISSING) -> None:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    top_missing = (
        pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
        .query("missing_count > 0")
        .sort_values("missing_count", ascending=False)
        .head(n)
    )

    print(f"\n  Top {n} columns with the most missing values:")
    if top_missing.empty:
        print("  → No missing values found. 🎉")
    else:
        print(f"  {'Column':<35} {'Missing #':>10} {'Missing %':>10}")
        print(f"  {'-'*35} {'-'*10} {'-'*10}")
        for col, row in top_missing.iterrows():
            print(f"  {col:<35} {int(row['missing_count']):>10,} {row['missing_pct']:>9.2f}%")
    print()


def main() -> None:
    df = load_data(DATA_PATH)
    report_shape(df)
    report_fraud_rate(df)
    report_top_missing(df)


if __name__ == "__main__":
    main()
