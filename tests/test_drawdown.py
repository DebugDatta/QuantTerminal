"""Tests for core/drawdown.py"""

import numpy as np
import pandas as pd
import pytest

from core.drawdown import drawdown_series, max_drawdown


class TestDrawdownSeries:
    def test_known_sequence(self):
        equity = pd.Series([100.0, 90.0, 80.0, 100.0, 95.0, 120.0])
        result = drawdown_series(equity)
        expected = pd.Series([0.0, -0.10, -0.20, 0.0, -0.05, 0.0])
        pd.testing.assert_series_equal(result, expected)

    def test_underwater_never_positive(self):
        equity = pd.Series([100.0, 90.0, 120.0, 110.0, 130.0])
        result = drawdown_series(equity)
        assert (result <= 0.0).all()

    def test_constant_equity_zero_drawdown(self):
        equity = pd.Series([100.0, 100.0, 100.0])
        result = drawdown_series(equity)
        assert (result == 0.0).all()

    @staticmethod
    def test_running_max_uses_historical_peak():
        equity = pd.Series([100.0, 150.0, 120.0, 140.0])
        result = drawdown_series(equity)
        # After new peak at 150, 120 is still referenced to 150
        expected = pd.Series([0.0, 0.0, 120.0 / 150.0 - 1.0, 140.0 / 150.0 - 1.0])
        pd.testing.assert_series_equal(result, expected)

    def test_dataframe_column_wise(self):
        equity = pd.DataFrame(
            {
                "A": [100.0, 90.0, 80.0],
                "B": [50.0, 60.0, 55.0],
            }
        )
        result = drawdown_series(equity)
        expected = equity / equity.expanding(min_periods=1).max() - 1.0
        pd.testing.assert_frame_equal(result, expected)

    def test_nan_propagation(self):
        equity = pd.Series([100.0, np.nan, 80.0])
        result = drawdown_series(equity)
        assert pd.isna(result.iloc[1])


class TestMaxDrawdown:
    def test_known_value_is_min_of_series(self):
        equity = pd.Series([100.0, 90.0, 80.0, 100.0, 95.0, 120.0])
        dd = drawdown_series(equity)
        assert max_drawdown(equity) == pytest.approx(dd.min())

    def test_no_drawdown(self):
        equity = pd.Series([100.0, 110.0, 120.0])
        assert max_drawdown(equity) == pytest.approx(0.0)

    def test_dataframe_returns_per_column(self):
        equity = pd.DataFrame(
            {
                "A": [100.0, 90.0, 80.0],
                "B": [50.0, 60.0, 55.0],
            }
        )
        result = max_drawdown(equity)
        assert isinstance(result, pd.Series)
        assert result["A"] == pytest.approx(-0.20)
        assert result["B"] == pytest.approx(55.0 / 60.0 - 1.0)