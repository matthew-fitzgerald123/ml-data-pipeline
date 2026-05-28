import pytest

from src.features.transforms import (
    add_amount_ratio,
    add_customer_window_features,
    add_time_features,
    build_feature_set,
)
from src.ingestion.generator import generate_transactions
from src.validation.feature_contracts import feature_schema


@pytest.fixture
def sample_df():
    return generate_transactions(n_transactions=2000, seed=42)


def test_time_features(sample_df):
    out = add_time_features(sample_df)
    assert "hour_of_day" in out.columns
    assert "is_weekend" in out.columns
    assert out["hour_of_day"].between(0, 23).all()
    assert out["is_weekend"].isin([0, 1]).all()


def test_window_feature_columns_present(sample_df):
    df = add_time_features(sample_df)
    out = add_customer_window_features(df, windows=[7, 30])
    for w in [7, 30]:
        assert f"customer_tx_count_{w}d" in out.columns
        assert f"customer_spend_{w}d" in out.columns
        assert f"customer_avg_amount_{w}d" in out.columns


def test_point_in_time_no_leakage(sample_df):
    # closed="left" excludes the current row; the first tx per customer has no prior history
    df = add_time_features(sample_df)
    out = add_customer_window_features(df, windows=[7])
    # nth(0) gets the actual first row per group, unlike first() which skips NaN
    first_rows = out.groupby("customer_id").nth(0)
    assert (first_rows["customer_tx_count_7d"] == 0).all()


def test_amount_ratio_nan_for_no_history(sample_df):
    df = add_time_features(sample_df)
    df = add_customer_window_features(df, windows=[30])
    out = add_amount_ratio(df)
    first_rows = out.groupby("customer_id").nth(0)
    assert first_rows["amount_vs_30d_avg"].isna().all()


def test_row_count_preserved(sample_df):
    out = build_feature_set(sample_df)
    assert len(out) == len(sample_df)


def test_feature_contract_passes(sample_df):
    features = build_feature_set(sample_df)
    validated = feature_schema.validate(features)
    assert len(validated) == len(features)
