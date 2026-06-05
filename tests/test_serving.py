from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from src.features.transforms import build_feature_set
from src.ingestion.generator import generate_transactions
from src.serving.app import app, get_store
from src.store.feature_store import FeatureStore


@pytest.fixture
def seeded_store(tmp_path):
    raw = generate_transactions(n_transactions=500, seed=8)
    features = build_feature_set(raw)
    store = FeatureStore(store_dir=str(tmp_path / "store"))
    past = datetime.utcnow() - timedelta(hours=1)
    store.write(features, version="test_snapshot", timestamp=past)
    return store


@pytest.fixture
def client(seeded_store):
    app.dependency_overrides[get_store] = lambda: seeded_store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_get_features_valid_customer(client):
    as_of = datetime.utcnow().isoformat()
    resp = client.get("/features/cust_0001", params={"as_of": as_of})
    assert resp.status_code == 200
    data = resp.json()
    assert data["customer_id"] == "cust_0001"


def test_get_features_response_has_window_columns(client):
    as_of = datetime.utcnow().isoformat()
    resp = client.get("/features/cust_0001", params={"as_of": as_of})
    assert resp.status_code == 200
    data = resp.json()
    for col in ("customer_tx_count_7d", "customer_spend_30d", "customer_avg_amount_90d"):
        assert col in data


def test_get_features_unknown_customer(client):
    as_of = datetime.utcnow().isoformat()
    resp = client.get("/features/cust_XXXX", params={"as_of": as_of})
    assert resp.status_code == 404


def test_get_features_no_snapshot_before_as_of(client):
    as_of = "2020-01-01T00:00:00"
    resp = client.get("/features/cust_0001", params={"as_of": as_of})
    assert resp.status_code == 404


def test_list_versions_returns_snapshot(client):
    resp = client.get("/versions")
    assert resp.status_code == 200
    versions = resp.json()
    assert isinstance(versions, list)
    assert len(versions) == 1
    v = versions[0]
    assert v["version"] == "test_snapshot"
    assert "timestamp" in v
    assert "row_count" in v
    assert "schema_hash" in v


def test_list_versions_empty_store(tmp_path):
    empty_store = FeatureStore(store_dir=str(tmp_path / "empty"))
    app.dependency_overrides[get_store] = lambda: empty_store
    client = TestClient(app)
    resp = client.get("/versions")
    app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json() == []
