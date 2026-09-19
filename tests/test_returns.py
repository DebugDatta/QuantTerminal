"""Tests for core/returns.py"""

import numpy as np
import pandas as pd
import pytest

from core.returns import compute_returns, cagr


class TestComputeReturns:
    def test_simple_known_sequence(self):
        prices = pd.Series([100.0, 110.0, 121.0, 108.9])
        result = compute_returns(prices)
        expected = pd.Series([np.nan, 0.10, 0.10, -0.10])
        pd.testing.assert_series_equal(result, expected)

    def test_first_observation_is_nan(self):
        prices = pd.Series([100.0, 110.0])
        result = compute_returns(prices)
        assert np.isnan(result.iloc[0])

    def test_output_shape_matches_input(self):
        prices = pd.Series([100.0, 110.0, 121.0])
        result = compute_returns(prices)
        assert len(result) == len(prices)

    def test_dataframe_supported_column_wise(self):
        prices = pd.DataFrame(
            {
                "A": [100.0, 110.0, 121.0],
                "B": [10.0, 9.0, 9.9],
            }
        )
        result = compute_returns(prices)
        assert isinstance(result, pd.DataFrame)
        expected = prices.pct_change()
        pd.testing.assert_frame_equal(result, expected)

    def test_nan_propagates(self):
        prices = pd.Series([100.0, np.nan, 110.0])
        result = compute_returns(prices)
        assert pd.isna(result).all()

    def test_zero_denominator_behavior(self):
        prices = pd.Series([0.0, 10.0])
        result = compute_returns(prices)
        # First observation is NaN; second is (10-0)/0 -> inf (numpy)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == float("inf")


class TestCagr:
    def test_known_value(self):
        equity = pd.Series([100.0, 110.0, 121.0])
        result = cagr(equity, periods_per_year=1)
        expected = (121.0 / 100.0) ** (1.0 / 3.0) - 1.0
        assert result == pytest.approx(expected)

    def test_daily_data_full_year(self):
        equity = pd.Series([100.0, 110.0, 121.0])
        result = cagr(equity, periods_per_year=252)
        years = 3 / 252
        expected = (121.0 / 100.0) ** (1.0 / years) - 1.0
        assert result == pytest.approx(expected)

    def test_initial_zero_returns_nan(self):
        equity = pd.Series([0.0, 100.0])
        assert np.isnan(cagr(equity))

    def test_no_change_is_zero(self):
        equity = pd.Series([100.0, 100.0, 100.0])
        assert cagr(equity, periods_per_year=1) == pytest.approx(0.0)

    def test_no_risk_free_rate_param(self):
        # CAGR formula (final/initial)^(1/years)-1 has no risk-free term
        equity = pd.Series([100.0, 110.0])
        with pytest.raises(TypeError):
            cagr(equity, risk_free_rate=0.0)


class TestEngineShapedInterface:
    def test_datetime_index_preserved(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        prices = pd.Series([100.0, 110.0, 121.0, 108.9, 119.79], index=idx)
        result = compute_returns(prices)
        assert result.index.equals(idx)

        frame = pd.DataFrame({"A": prices.values, "B": prices.values * 2}, index=idx)
        result_df = compute_returns(frame)
        assert result_df.index.equals(idx)

    def test_multiindex_column_pass_through(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        cols = pd.MultiIndex.from_tuples([("RELIANCE.NS", "Close"), ("TCS.NS", "Close")])
        frame = pd.DataFrame(
            np.column_stack([[100.0, 110.0, 121.0, 108.9], [500.0, 520.0, 510.0, 530.0]]),
            index=idx,
            columns=cols,
        )
        result = compute_returns(frame)
        assert isinstance(result, pd.DataFrame)
        assert result.index.equals(idx)
        assert result.columns.equals(cols)
        assert np.isnan(result.iloc[0]).all()

    def test_cagr_rejects_equity_dataframe(self):
        equity = pd.DataFrame({"A": [100.0, 90.0, 80.0], "B": [50.0, 60.0, 55.0]})
        with pytest.raises(ValueError):
            cagr(equity)