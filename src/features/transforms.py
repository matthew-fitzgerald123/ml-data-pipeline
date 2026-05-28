import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_customer_window_features(
    df: pd.DataFrame, windows: list[int] = [7, 30, 90]
) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()

    feature_frames = []
    for cid, group in df.groupby("customer_id", sort=False):
        g = group.set_index("timestamp").sort_index()
        for w in windows:
            rule = f"{w}D"
            rolling = g["amount"].rolling(rule, closed="left")
            # empty window (first tx per customer) returns NaN from pandas; fill to 0 for count/sum
            g[f"customer_tx_count_{w}d"] = rolling.count().fillna(0)
            g[f"customer_spend_{w}d"] = rolling.sum().fillna(0).round(2)
            g[f"customer_avg_amount_{w}d"] = rolling.mean().round(4)
        feature_frames.append(g.reset_index())

    return (
        pd.concat(feature_frames, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def add_amount_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    baseline = df["customer_avg_amount_30d"].replace(0, np.nan)
    df["amount_vs_30d_avg"] = (df["amount"] / baseline).round(4)
    return df


def build_feature_set(
    df: pd.DataFrame, windows: list[int] = [7, 30, 90]
) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_customer_window_features(df, windows=windows)
    df = add_amount_ratio(df)
    return df
