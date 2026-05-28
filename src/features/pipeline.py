from pathlib import Path

import pandas as pd

from src.features.transforms import build_feature_set
from src.ingestion.ingestor import ingest_directory
from src.validation.feature_contracts import feature_schema


def run_feature_pipeline(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    windows: list[int] = [7, 30, 90],
) -> pd.DataFrame:
    raw = ingest_directory(raw_dir)
    features = build_feature_set(raw, windows=windows)
    validated = feature_schema.validate(features)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    date_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"features_{date_str}.parquet"
    validated.to_parquet(path, index=False)
    print(f"Feature set written to {path} ({len(validated):,} rows)")
    return validated
