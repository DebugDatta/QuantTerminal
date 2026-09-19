"""Formatting helpers. Spec: docs/ARCHITECTURE.md -> utils/helpers.py."""

from datetime import date, datetime
from typing import Optional

import pandas as pd

CURRENCY_SYMBOLS = {
    "INR": "\u20b9",
    "USD": "$",
    "EUR": "\u20ac",
    "GBP": "\u00a3",
    "JPY": "\u00a5",
    "SGD": "S$",
}


def format_currency(amount: float, currency: str = "INR") -> str:
    """Format amount with the currency symbol from the CURRENCY_SYMBOLS map."""
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")
    if abs(amount) >= 1e7:
        return f"{symbol}{amount / 1e7:,.2f}Cr"
    if abs(amount) >= 1e5:
        return f"{symbol}{amount / 1e5:,.2f}L"
    return f"{symbol}{amount:,.2f}"


def format_date(dt: date, locale: str = "en_IN") -> str:
    """Format a date per locale. en_IN -> DD-MM-YYYY, else YYYY-MM-DD."""
    if locale == "en_IN":
        return dt.strftime("%d-%m-%Y")
    return dt.strftime("%Y-%m-%d")


def drop_holiday_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where non-trading days (Indian market holidays) leave NaN in Close."""
    return df.dropna(subset=["Close"])
