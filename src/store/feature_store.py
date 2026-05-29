import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


def _schema_hash(df: pd.DataFrame) -> str:
    schema_str = ",".join(f"{c}:{dt}" for c, dt in zip(df.columns, df.dtypes))
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


class FeatureStore:
    def __init__(self, store_dir: str = "data/store"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        (self.store_dir / "manifests").mkdir(exist_ok=True)

    def write(
        self,
        features: pd.DataFrame,
        version: str | None = None,
        timestamp: datetime | None = None,
    ) -> dict:
        if version is None:
            version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if timestamp is None:
            timestamp = datetime.utcnow()

        parquet_path = self.store_dir / f"features_{version}.parquet"
        features.to_parquet(parquet_path, index=False)

        manifest = {
            "version": version,
            "timestamp": timestamp.isoformat(),
            "schema_hash": _schema_hash(features),
            "row_count": len(features),
            "git_sha": _git_sha(),
            "path": str(parquet_path),
        }

        manifest_path = self.store_dir / "manifests" / f"manifest_{version}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return manifest

    def list_versions(self) -> list[dict]:
        manifests_dir = self.store_dir / "manifests"
        return [
            json.loads(p.read_text())
            for p in sorted(manifests_dir.glob("manifest_*.json"))
        ]

    def read_as_of(self, as_of: datetime) -> pd.DataFrame | None:
        eligible = [
            v for v in self.list_versions()
            if datetime.fromisoformat(v["timestamp"]) <= as_of
        ]
        if not eligible:
            return None
        latest = max(eligible, key=lambda v: v["timestamp"])
        df = pd.read_parquet(latest["path"])
        return df[df["timestamp"] <= pd.Timestamp(as_of)].copy()
