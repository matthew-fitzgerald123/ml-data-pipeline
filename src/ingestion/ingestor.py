from pathlib import Path

import pandas as pd

from src.validation.raw_schema import raw_transaction_schema


def load_and_validate(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return raw_transaction_schema.validate(df)


def ingest_directory(raw_dir: str = "data/raw") -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")
    frames = [load_and_validate(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("timestamp").reset_index(drop=True)
