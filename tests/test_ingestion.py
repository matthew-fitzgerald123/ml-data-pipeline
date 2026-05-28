import pytest

from src.ingestion.generator import generate_transactions, save_raw
from src.ingestion.ingestor import load_and_validate
from src.validation.raw_schema import raw_transaction_schema


def test_generate_transaction_count():
    df = generate_transactions(n_transactions=1000, seed=0)
    assert len(df) == 1000


def test_generate_sorted_timestamps():
    df = generate_transactions(n_transactions=500, seed=1)
    assert df["timestamp"].is_monotonic_increasing


def test_generate_no_nulls():
    df = generate_transactions(n_transactions=500, seed=2)
    assert df.isnull().sum().sum() == 0


def test_schema_validates_generated_data():
    df = generate_transactions(n_transactions=200, seed=3)
    validated = raw_transaction_schema.validate(df)
    assert len(validated) == 200


def test_save_and_reload(tmp_path):
    df = generate_transactions(n_transactions=100, seed=4)
    path = save_raw(df, output_dir=str(tmp_path))
    reloaded = load_and_validate(path)
    assert len(reloaded) == 100


def test_amounts_positive():
    df = generate_transactions(n_transactions=500, seed=5)
    assert (df["amount"] > 0).all()
