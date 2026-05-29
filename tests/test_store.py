from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.features.transforms import build_feature_set
from src.ingestion.generator import generate_transactions
from src.store.feature_store import FeatureStore


@pytest.fixture
def features():
    raw = generate_transactions(n_transactions=500, seed=7)
    return build_feature_set(raw)


def test_write_creates_parquet_and_manifest(tmp_path, features):
    store = FeatureStore(store_dir=str(tmp_path / "store"))
    store.write(features, version="test_v1")
    assert (tmp_path / "store" / "features_test_v1.parquet").exists()
    assert (tmp_path / "store" / "manifests" / "manifest_test_v1.json").exists()


def test_manifest_has_required_fields(tmp_path, features):
    store = FeatureStore(store_dir=str(tmp_path / "store"))
    manifest = store.write(features, version="test_v2")
    for field in ("version", "timestamp", "schema_hash", "row_count", "git_sha", "path"):
        assert field in manifest


def test_manifest_row_count_matches(tmp_path, features):
    store = FeatureStore(store_dir=str(tmp_path / "store"))
    manifest = store.write(features, version="test_v3")
    assert manifest["row_count"] == len(features)


def test_list_versions(tmp_path, features):
    store = FeatureStore(store_dir=str(tmp_path / "store"))
    store.write(features, version="v1")
    store.write(features, version="v2")
    versions = store.list_versions()
    assert len(versions) == 2


def test_read_as_of_returns_snapshot(tmp_path, features):
    store = FeatureStore(store_dir=str(tmp_path / "store"))
    past = datetime.utcnow() - timedelta(hours=1)
    store.write(features, version="v_past", timestamp=past)
    result = store.read_as_of(datetime.utcnow())
    assert result is not None
    assert len(result) > 0


def test_read_as_of_excludes_future_snapshots(tmp_path, features):
    store = FeatureStore(store_dir=str(tmp_path / "store"))
    future = datetime.utcnow() + timedelta(hours=2)
    store.write(features, version="v_future", timestamp=future)
    result = store.read_as_of(datetime.utcnow())
    assert result is None


def test_read_as_of_filters_transaction_timestamps(tmp_path):
    raw = generate_transactions(n_transactions=500, seed=9, start_date="2025-01-01", end_date="2026-01-01")
    features = build_feature_set(raw)
    store = FeatureStore(store_dir=str(tmp_path / "store"))
    past = datetime(2025, 1, 1)
    store.write(features, version="v_early", timestamp=past)
    as_of = datetime(2025, 7, 1)
    result = store.read_as_of(as_of)
    assert result is not None
    assert (result["timestamp"] <= pd.Timestamp(as_of)).all()
