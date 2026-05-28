import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

raw_transaction_schema = DataFrameSchema(
    {
        "transaction_id": Column(str, nullable=False, unique=True),
        "customer_id": Column(str, nullable=False),
        "merchant_id": Column(str, nullable=False),
        "merchant_category": Column(
            str,
            checks=Check.isin([
                "grocery", "gas", "restaurant", "retail",
                "travel", "entertainment", "healthcare",
            ]),
        ),
        "amount": Column(
            float,
            checks=[Check.greater_than(0), Check.less_than(100_000)],
        ),
        "timestamp": Column("datetime64[ns]", nullable=False),
        "state": Column(str, nullable=False),
    },
    coerce=True,
)
