import math
from datetime import datetime
from typing import Any, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.store.feature_store import FeatureStore

app = FastAPI(title="Feature Serving API")


def get_store() -> FeatureStore:
    return FeatureStore()


class FeatureRequest(BaseModel):
    customer_id: str
    as_of: datetime = Field(..., description="Point-in-time timestamp in ISO 8601 format")


class BatchFeatureRequest(BaseModel):
    entries: List[FeatureRequest] = Field(..., min_length=1)


class BatchFeatureResult(BaseModel):
    customer_id: str
    as_of: datetime
    found: bool
    features: Optional[dict] = None


class BatchFeatureResponse(BaseModel):
    results: List[BatchFeatureResult]


def _serialize_row(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, pd.Timestamp):
            out[k] = v.isoformat()
        elif isinstance(v, float) and math.isnan(v):
            out[k] = None
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/versions")
def list_versions(store: FeatureStore = Depends(get_store)) -> list[dict[str, Any]]:
    return store.list_versions()


def _lookup_as_of(
    store: FeatureStore, customer_id: str, as_of: datetime
) -> Optional[dict[str, Any]]:
    """Return the point-in-time-correct feature row for a customer, or None.

    Resolves the newest snapshot written at or before as_of, then the customer's
    most recent row within that snapshot whose timestamp is not after as_of.
    Every entry is resolved against its own as_of so a batch built from many
    event timestamps stays free of target leakage."""
    snapshot = store.read_as_of(as_of)
    if snapshot is None:
        return None
    customer_rows = snapshot[snapshot["customer_id"] == customer_id]
    if customer_rows.empty:
        return None
    row = customer_rows.sort_values("timestamp").iloc[-1]
    return _serialize_row(row)


@app.get("/features/{customer_id}")
def get_features(
    customer_id: str,
    as_of: datetime = Query(..., description="Point-in-time timestamp in ISO 8601 format"),
    store: FeatureStore = Depends(get_store),
) -> dict[str, Any]:
    features = _lookup_as_of(store, customer_id, as_of)
    if features is None:
        raise HTTPException(
            status_code=404,
            detail=f"No features found for customer {customer_id} as of {as_of.isoformat()}",
        )
    return features


@app.post("/features/batch", response_model=BatchFeatureResponse)
def get_features_batch(
    req: BatchFeatureRequest,
    store: FeatureStore = Depends(get_store),
) -> BatchFeatureResponse:
    """Point-in-time feature retrieval for many (customer_id, as_of) pairs at once.

    This is the training-set generation join: each requested row carries its own
    event timestamp, and features are read as of that timestamp so labels drawn
    later cannot leak backward into the feature values. Missing customers or
    timestamps before the first snapshot are reported per entry with found=False
    rather than failing the whole request."""
    results: List[BatchFeatureResult] = []
    for entry in req.entries:
        features = _lookup_as_of(store, entry.customer_id, entry.as_of)
        results.append(
            BatchFeatureResult(
                customer_id=entry.customer_id,
                as_of=entry.as_of,
                found=features is not None,
                features=features,
            )
        )
    return BatchFeatureResponse(results=results)
