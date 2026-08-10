"""Formatting helpers. Spec: docs/ARCHITECTURE.md -> utils/helpers.py."""

from datetime import date, datetime
from typing import Optional

import pandas as pd


def format_currency(amount: float, currency: str = "INR") -> str:
    """Format amount with the currency's symbol. Symbols read from a map, never hardcoded literals."""
    raise NotImplementedError


def format_date(dt: date, locale: str = "en_IN") -> str:
    """Format a date per locale/style (e.g. DD-MM-YYYY vs YYYY-MM-DD)."""
    raise NotImplementedError


def drop_holiday_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where Indian market holidays / non-trading days leave NaN."""
    raise NotImplementedError