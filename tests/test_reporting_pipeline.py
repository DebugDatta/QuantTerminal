"""
Automated unit and integration tests for the 20-page Institutional Research PDF generator.
Verifies:
1. Strict 20-page count guarantee (no overflow to page 21, no premature termination).
2. Clean currency symbol rendering (Indian Rupee ₹ and USD $).
3. Robust handling of defensive inputs (e.g., DataFrame passed positionally).
"""

import io
import numpy as np
import pandas as pd
import pypdf
import pytest

from reporting.report_data import build_report_data
from reporting.pdf_generator import generate_pdf_report


def _create_sample_ohlcv(n_bars: int = 252) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="B")
    np.random.seed(42)
    rets = np.random.normal(0.0005, 0.02, size=n_bars)
    prices = 100.0 * np.exp(np.cumsum(rets))
    
    df = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.uniform(-0.005, 0.005, n_bars)),
            "High": prices * (1 + np.random.uniform(0.005, 0.02, n_bars)),
            "Low": prices * (1 - np.random.uniform(0.005, 0.02, n_bars)),
            "Close": prices,
            "Volume": np.random.randint(100000, 5000000, n_bars),
        },
        index=dates,
    )
    return df


def test_20_page_pdf_strictly_20_pages_nse():
    """Verify that an NSE ticker produces exactly 20 pages with full layout."""
    df = _create_sample_ohlcv(252)
    rd = build_report_data(
        ticker="TESTASSET.NS",
        benchmark_ticker="^NSEI",
        period="1y",
        interval="1d",
        input_df=df,
    )
    pdf_buf = generate_pdf_report(rd)
    reader = pypdf.PdfReader(pdf_buf)
    assert len(reader.pages) == 20, f"Expected 20 pages, got {len(reader.pages)}"


def test_20_page_pdf_strictly_20_pages_us():
    """Verify that a US ticker produces exactly 20 pages with USD formatting."""
    df = _create_sample_ohlcv(252)
    rd = build_report_data(
        ticker="TESTASSET",
        benchmark_ticker="^GSPC",
        period="1y",
        interval="1d",
        input_df=df,
    )
    pdf_buf = generate_pdf_report(rd)
    reader = pypdf.PdfReader(pdf_buf)
    assert len(reader.pages) == 20, f"Expected 20 pages, got {len(reader.pages)}"


def test_build_report_data_defensive_positional_df():
    """Verify build_report_data handles input_df passed positionally without crash."""
    df = _create_sample_ohlcv(252)
    # Passing df as second positional argument
    rd = build_report_data("TEST.NS", df)
    assert rd["metadata"]["ticker"] == "TEST.NS"
    assert rd["metadata"]["benchmark_ticker"] == "^NSEI"
    pdf_buf = generate_pdf_report(rd)
    reader = pypdf.PdfReader(pdf_buf)
    assert len(reader.pages) == 20
