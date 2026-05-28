import uuid
from pathlib import Path

import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery", "gas", "restaurant", "retail",
    "travel", "entertainment", "healthcare",
]
US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]


def generate_transactions(
    n_customers: int = 500,
    n_merchants: int = 100,
    n_transactions: int = 50_000,
    start_date: str = "2025-01-01",
    end_date: str = "2026-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_ids = [f"cust_{i:04d}" for i in range(n_customers)]
    merchant_ids = [f"merch_{i:03d}" for i in range(n_merchants)]
    merchant_category_map = {
        mid: MERCHANT_CATEGORIES[i % len(MERCHANT_CATEGORIES)]
        for i, mid in enumerate(merchant_ids)
    }

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    span_seconds = int((end - start).total_seconds())
    offsets = rng.integers(0, span_seconds, size=n_transactions)
    timestamps = start + pd.to_timedelta(offsets, unit="s")

    customer_idx = rng.integers(0, n_customers, size=n_transactions)
    merchant_idx = rng.integers(0, n_merchants, size=n_transactions)

    # per-customer log-mean gives each customer a stable spending baseline
    customer_log_mean = rng.uniform(2.5, 4.5, size=n_customers)
    amounts = np.exp(
        customer_log_mean[customer_idx] + rng.normal(0, 0.4, size=n_transactions)
    ).round(2)
    amounts = np.maximum(amounts, 0.50)

    df = pd.DataFrame({
        "transaction_id": [str(uuid.uuid4()) for _ in range(n_transactions)],
        "customer_id": [customer_ids[i] for i in customer_idx],
        "merchant_id": [merchant_ids[i] for i in merchant_idx],
        "merchant_category": [merchant_category_map[merchant_ids[i]] for i in merchant_idx],
        "amount": amounts,
        "timestamp": timestamps,
        "state": rng.choice(US_STATES, size=n_transactions),
    })

    return df.sort_values("timestamp").reset_index(drop=True)


def save_raw(df: pd.DataFrame, output_dir: str = "data/raw") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    date_str = pd.Timestamp.now().strftime("%Y%m%d")
    path = out / f"transactions_{date_str}.parquet"
    df.to_parquet(path, index=False)
    return path
