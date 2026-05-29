from prefect import flow, task, get_run_logger

from src.ingestion.generator import generate_transactions, save_raw
from src.store.feature_store import FeatureStore


@task(name="generate-transactions")
def generate_and_save(
    n_customers: int,
    n_merchants: int,
    n_transactions: int,
    start_date: str,
    end_date: str,
    seed: int,
) -> str:
    logger = get_run_logger()
    df = generate_transactions(
        n_customers=n_customers,
        n_merchants=n_merchants,
        n_transactions=n_transactions,
        start_date=start_date,
        end_date=end_date,
        seed=seed,
    )
    path = save_raw(df)
    logger.info(f"Generated {len(df):,} transactions saved to {path}")
    return str(path)


@task(name="build-features")
def build_features(raw_dir: str = "data/raw", windows: list[int] = [7, 30, 90]):
    import pandas as pd
    from src.features.transforms import build_feature_set
    from src.ingestion.ingestor import ingest_directory
    from src.validation.feature_contracts import feature_schema

    logger = get_run_logger()
    raw = ingest_directory(raw_dir)
    features = build_feature_set(raw, windows=windows)
    validated = feature_schema.validate(features)
    logger.info(f"Built {len(validated):,} feature rows for {validated['customer_id'].nunique()} customers")
    return validated


@task(name="write-to-store")
def write_to_store(features, store_dir: str = "data/store") -> dict:
    logger = get_run_logger()
    store = FeatureStore(store_dir=store_dir)
    manifest = store.write(features)
    logger.info(
        f"Wrote {manifest['row_count']:,} rows to store version {manifest['version']} "
        f"(git {manifest['git_sha']})"
    )
    return manifest


@flow(name="ml-data-pipeline")
def ml_pipeline_flow(
    n_customers: int = 500,
    n_merchants: int = 100,
    n_transactions: int = 50_000,
    start_date: str = "2025-01-01",
    end_date: str = "2026-01-01",
    seed: int = 42,
    raw_dir: str = "data/raw",
    store_dir: str = "data/store",
    windows: list[int] = [7, 30, 90],
) -> dict:
    generate_and_save(
        n_customers=n_customers,
        n_merchants=n_merchants,
        n_transactions=n_transactions,
        start_date=start_date,
        end_date=end_date,
        seed=seed,
    )
    features = build_features(raw_dir=raw_dir, windows=windows)
    manifest = write_to_store(features, store_dir=store_dir)
    return manifest


if __name__ == "__main__":
    ml_pipeline_flow()
