"""Data validation and safe formatting helpers for QuantTerminal."""

from __future__ import annotations

import math
from typing import Any, Optional
import numpy as np
import pandas as pd


def safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert any value to float, returning default if invalid or NaN."""
    if val is None:
        return default
    if isinstance(val, (int, float, np.integer, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return default
        return float(val)
    if isinstance(val, str):
        cleaned = val.strip().replace(",", "").replace("%", "").replace("x", "").replace("X", "").replace("₹", "").replace("$", "")
        if cleaned == "" or cleaned.lower() in ("nan", "none", "n/a", "null", "-"):
            return default
        try:
            res = float(cleaned)
            return default if (math.isnan(res) or math.isinf(res)) else res
        except (ValueError, TypeError):
            return default
    return default


def safe_division(numerator: Any, denominator: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely divide numerator by denominator, guarding against zero division and NaNs."""
    num = safe_float(numerator)
    den = safe_float(denominator)
    if num is None or den is None or den == 0.0:
        return default
    res = num / den
    return default if (math.isnan(res) or math.isinf(res)) else res


def format_large_number(val: Any, currency: str = "₹", is_indian: bool = True) -> str:
    """Format large numbers in Indian units (Crores/Lakhs) or International units (Billion/Million)."""
    f = safe_float(val)
    if f is None:
        return "N/A"

    sign = "-" if f < 0 else ""
    abs_f = abs(f)

    if is_indian or currency == "₹":
        if abs_f >= 1e12:  # Lakh Crore
            return f"{sign}{currency}{abs_f / 1e12:.2f}L Cr"
        elif abs_f >= 1e7:  # Crore
            return f"{sign}{currency}{abs_f / 1e7:.2f} Cr"
        elif abs_f >= 1e5:  # Lakh
            return f"{sign}{currency}{abs_f / 1e5:.2f} L"
        elif abs_f >= 1e3:  # Thousand
            return f"{sign}{currency}{abs_f / 1e3:.2f} K"
        else:
            return f"{sign}{currency}{abs_f:,.2f}"
    else:
        if abs_f >= 1e12:
            return f"{sign}{currency}{abs_f / 1e12:.2f}T"
        elif abs_f >= 1e9:
            return f"{sign}{currency}{abs_f / 1e9:.2f}B"
        elif abs_f >= 1e6:
            return f"{sign}{currency}{abs_f / 1e6:.2f}M"
        elif abs_f >= 1e3:
            return f"{sign}{currency}{abs_f / 1e3:.2f}K"
        else:
            return f"{sign}{currency}{abs_f:,.2f}"


def format_currency(val: Any, currency: str = "₹", decimals: int = 2) -> str:
    """Format standard currency values with appropriate commas."""
    f = safe_float(val)
    if f is None:
        return "N/A"
    return f"{currency}{f:,.{decimals}f}"


def format_percentage(val: Any, decimals: int = 2, show_sign: bool = True) -> str:
    """Format decimal or percentage values consistently (e.g. 0.124 -> +12.40%)."""
    f = safe_float(val)
    if f is None:
        return "N/A"
    # If the number is already > 1 or < -1, assume it's already in percentage form
    # If -1.0 <= f <= 1.0 and user specifies, could be decimal. We treat as percentage directly if typical ratio
    sign = "+" if (show_sign and f > 0) else ""
    return f"{sign}{f:.{decimals}f}%"


def format_ratio(val: Any, decimals: int = 2, suffix: str = "x") -> str:
    """Format multiple or ratio values (e.g. 24.16x)."""
    f = safe_float(val)
    if f is None:
        return "N/A"
    return f"{f:.{decimals}f}{suffix}"
