import math
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query

from src.store.feature_store import FeatureStore

app = FastAPI(title="Feature Serving API")


def get_store() -> FeatureStore:
    return FeatureStore()


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


@app.get("/features/{customer_id}")
def get_features(
    customer_id: str,
    as_of: datetime = Query(..., description="Point-in-time timestamp in ISO 8601 format"),
    store: FeatureStore = Depends(get_store),
) -> dict[str, Any]:
    snapshot = store.read_as_of(as_of)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No feature snapshot available before as_of")

    customer_rows = snapshot[snapshot["customer_id"] == customer_id]
    if customer_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No features found for customer {customer_id} as of {as_of.isoformat()}",
        )

    row = customer_rows.sort_values("timestamp").iloc[-1]
    return _serialize_row(row)
