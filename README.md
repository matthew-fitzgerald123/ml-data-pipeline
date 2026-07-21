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
| Store | Versioned Parquet offline store with per-version manifests (schema hash, row count, git SHA) |
| Serving | FastAPI point-in-time feature lookups, single customer or batch training-set joins |

## Getting Started

```bash
pip install -r requirements.txt

# generate and save raw transactions
python -c "from src.ingestion.generator import generate_transactions, save_raw; save_raw(generate_transactions())"

# run feature pipeline
python -c "from src.features.pipeline import run_feature_pipeline; run_feature_pipeline()"

# or run the full Prefect flow (generate, build features, write to versioned store)
python -m flows.pipeline_flow

# run tests
pytest tests/

# serve point-in-time feature lookups
uvicorn src.serving.app:app --reload
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

## Serving

The serving layer exposes point-in-time feature retrieval:

- `GET /features/{customer_id}?as_of=<ISO8601>` returns the latest features for one customer as of a single timestamp.
- `POST /features/batch` resolves many `(customer_id, as_of)` pairs in one call, each against its own timestamp. This is the training-set generation join: every row is read as of its own event time, so labels observed later cannot leak backward into the features. Customers or timestamps with no eligible snapshot are reported per entry with `found: false` rather than failing the whole request.

```bash
curl -X POST localhost:8000/features/batch \
  -H "Content-Type: application/json" \
  -d '{"entries": [
        {"customer_id": "cust_0001", "as_of": "2024-01-15T12:00:00"},
        {"customer_id": "cust_0003", "as_of": "2024-02-01T09:30:00"}
      ]}'
```

## Project Structure

```
src/
  ingestion/    transaction generator and ingestion pipeline
  features/     feature transforms and engineering pipeline
  validation/   Pandera schemas and data contracts
  store/        versioned Parquet offline feature store
  serving/      FastAPI point-in-time serving layer
flows/          Prefect pipeline flows
config/         pipeline configuration
data/
  raw/          ingested raw Parquet files
  processed/    validated, transformed feature sets
tests/          unit tests for ingestion, features, store, and serving
```
