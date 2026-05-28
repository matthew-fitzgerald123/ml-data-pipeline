# ml-data-pipeline

End-to-end feature engineering pipeline for financial transaction data, with schema validation, windowed aggregations, data contracts, and point-in-time correct serving.

## Overview

Synthetic financial transactions are generated, ingested, and validated against a raw schema before passing through a feature engineering pipeline. Derived features use time-windowed aggregations with `closed="left"` windows to enforce point-in-time correctness, preventing target leakage into historical feature sets.

## Components

| Component | Description |
|-----------|-------------|
| Ingestion | Synthetic transaction generator and raw Parquet ingestion |
| Validation | Pandera schemas for raw data and engineered features |
| Features | Windowed customer aggregations and behavioral ratios |
| Store | Versioned Parquet offline store (PostgreSQL + Redis serving planned) |
| Serving | Point-in-time FastAPI endpoint (in progress) |

## Getting Started

```bash
pip install -r requirements.txt

# generate and save raw transactions
python -c "from src.ingestion.generator import generate_transactions, save_raw; save_raw(generate_transactions())"

# run feature pipeline
python -c "from src.features.pipeline import run_feature_pipeline; run_feature_pipeline()"

# run tests
pytest tests/
```

## Feature Engineering

Rolling window features are computed per customer over 7, 30, and 90 day lookback windows:

- `customer_tx_count_{N}d` - transaction count in prior N days
- `customer_spend_{N}d` - total spend in prior N days
- `customer_avg_amount_{N}d` - mean transaction amount in prior N days
- `amount_vs_30d_avg` - current amount relative to customer 30-day baseline

Time features: `hour_of_day`, `day_of_week`, `is_weekend`

## Point-in-Time Correctness

All windows use `closed="left"` to exclude the current row's timestamp. Features reflect only information available at transaction time, making them safe for model training without leakage.

## Project Structure

```
src/
  ingestion/    transaction generator and ingestion pipeline
  features/     feature transforms and engineering pipeline
  validation/   Pandera schemas and data contracts
  store/        offline and online feature store
  serving/      FastAPI serving layer
flows/          Prefect pipeline flows
data/
  raw/          ingested raw Parquet files
  processed/    validated, transformed feature sets
```
