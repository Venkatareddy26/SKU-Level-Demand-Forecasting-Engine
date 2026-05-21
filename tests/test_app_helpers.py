"""Tests for dashboard helper functions."""
import pandas as pd

import app


def _valid_history(rows=365):
    return pd.DataFrame({
        'id': ['SKU_001'] * rows,
        'date': pd.date_range('2024-01-01', periods=rows, freq='D'),
        'sales': range(rows),
    })


def test_validate_csv_requires_default_lag_history():
    df = _valid_history(rows=30)

    is_valid, message = app.validate_csv(df)

    assert not is_valid
    assert str(app.MIN_HISTORY_DAYS) in message


def test_validate_csv_accepts_one_year_history():
    df = _valid_history(rows=app.MIN_HISTORY_DAYS)

    is_valid, message = app.validate_csv(df)

    assert is_valid
    assert message == ""


def test_validate_csv_warns_for_short_skus():
    df = pd.concat([
        _valid_history(rows=app.MIN_HISTORY_DAYS),
        pd.DataFrame({
            'id': ['SKU_002'] * 30,
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'sales': range(30),
        }),
    ], ignore_index=True)

    is_valid, message = app.validate_csv(df)

    assert is_valid
    assert "fewer than 365" in message
