import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

feature_schema = DataFrameSchema(
    {
        "transaction_id": Column(str, nullable=False),
        "customer_id": Column(str, nullable=False),
        "merchant_id": Column(str, nullable=False),
        "amount": Column(float, checks=Check.greater_than(0)),
        "timestamp": Column("datetime64[ns]", nullable=False),
        "hour_of_day": Column(
            int,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(23),
            ],
        ),
        "day_of_week": Column(
            int,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(6),
            ],
        ),
        "is_weekend": Column(int, checks=Check.isin([0, 1])),
        "customer_tx_count_7d": Column(float, nullable=True),
        "customer_spend_7d": Column(float, nullable=True),
        "customer_avg_amount_7d": Column(float, nullable=True),
        "customer_tx_count_30d": Column(float, nullable=True),
        "customer_spend_30d": Column(float, nullable=True),
        "customer_avg_amount_30d": Column(float, nullable=True),
        "customer_tx_count_90d": Column(float, nullable=True),
        "customer_spend_90d": Column(float, nullable=True),
        "customer_avg_amount_90d": Column(float, nullable=True),
        "amount_vs_30d_avg": Column(float, nullable=True),
    },
    coerce=True,
)
