"""Tests for core/metrics.py

Each test verifies the documented formula directly rather than just
checking a return value exists. Annualization, risk-free-rate, and
benchmark rules are tested explicitly per metric.
"""

import math

import numpy as np
import pandas as pd
import pytest

from core.metrics import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    information_ratio,
    treynor_ratio,
    beta,
    alpha,
)

RETURNS = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01, -0.01])
BENCHMARK = pd.Series([0.005, 0.01, 0.0, 0.01, 0.005, -0.005])

MEAN_R = RETURNS.mean()
STD_R = RETURNS.std()
MEAN_B = BENCHMARK.mean()


class TestSharpeRatio:
    def test_documented_formula_no_rf(self):
        expected = (MEAN_R - 0.0) / STD_R * math.sqrt(252)
        assert sharpe_ratio(RETURNS) == pytest.approx(expected)

    def test_annualization_applied(self):
        unannualized = (MEAN_R - 0.0) / STD_R
        annualized = sharpe_ratio(RETURNS)
        assert annualized == pytest.approx(unannualized * math.sqrt(252))

    def test_risk_free_rate_used_directly(self):
        rf = 0.02
        expected = (MEAN_R - rf) / STD_R * math.sqrt(252)
        assert sharpe_ratio(RETURNS, risk_free_rate=rf) == pytest.approx(expected)

    def test_zero_std_constant_returns(self):
        const = pd.Series([0.01, 0.01, 0.01])
        assert sharpe_ratio(const) == float("inf")

    def test_zero_rf_default(self):
        assert sharpe_ratio(RETURNS, risk_free_rate=0.0) == pytest.approx(sharpe_ratio(RETURNS))

    def test_hand_calculated_constant(self):
        # Hand-computed for returns [0.01, 0.03]:
        #   mean = 0.02, std(ddof=1) = sqrt(0.0002)
        #   sharpe = (0.02 - 0)/sqrt(0.0002) * sqrt(252) = 22.4499443206
        const = pd.Series([0.01, 0.03])
        assert sharpe_ratio(const) == pytest.approx(22.4499443206, rel=1e-9)


class TestSortinoRatio:
    def test_documented_formula(self):
        negative = RETURNS[RETURNS < 0]
        downside_std = negative.std()
        expected = (MEAN_R - 0.0) / downside_std
        assert sortino_ratio(RETURNS) == pytest.approx(expected)

    def test_uses_only_negative_returns(self):
        negative = RETURNS[RETURNS < 0]
        downside_std = negative.std()
        # Confirm we did NOT use full-series std
        assert sortino_ratio(RETURNS) == pytest.approx((MEAN_R - 0.0) / downside_std)
        assert sortino_ratio(RETURNS) != pytest.approx((MEAN_R - 0.0) / STD_R)

    def test_no_undocumented_annualization(self):
        negative = RETURNS[RETURNS < 0]
        expected = (MEAN_R - 0.0) / negative.std()
        assert sortino_ratio(RETURNS) == pytest.approx(expected)
        assert not math.isclose(
            sortino_ratio(RETURNS),
            expected * math.sqrt(252),
            rel_tol=1e-9,
        )

    def test_risk_free_rate_used_directly(self):
        rf = 0.02
        negative = RETURNS[RETURNS < 0]
        expected = (MEAN_R - rf) / negative.std()
        assert sortino_ratio(RETURNS, risk_free_rate=rf) == pytest.approx(expected)

    def test_no_negative_returns_is_nan(self):
        # pandas .std() on an empty series is NaN; no documented behavior
        all_positive = pd.Series([0.01, 0.02, 0.03])
        assert np.isnan(sortino_ratio(all_positive))


class TestCalmarRatio:
    def test_documented_formula(self):
        equity = pd.Series([100.0, 90.0, 80.0, 100.0, 120.0, 110.0])
        cagr_value = (110.0 / 100.0) ** (1.0 / (6 / 252.0)) - 1.0
        mdd = min(equity / equity.expanding(min_periods=1).max() - 1.0)
        expected = cagr_value / abs(mdd)
        assert calmar_ratio(equity) == pytest.approx(expected)

    def test_uses_core_drawdown(self):
        from core.drawdown import max_drawdown

        equity = pd.Series([100.0, 90.0, 80.0, 100.0, 120.0, 110.0])
        mdd = abs(max_drawdown(equity))
        assert calmar_ratio(equity) == pytest.approx(1.0 / mdd * ((110.0 / 100.0) ** (1.0 / (6 / 252.0)) - 1.0))

    def test_no_risk_free_rate_param(self):
        # CAGR has no risk-free term per STRATEGIES_BACKTESTING.md
        equity = pd.Series([100.0, 90.0, 80.0, 100.0, 120.0, 110.0])
        with pytest.raises(TypeError):
            calmar_ratio(equity, risk_free_rate=0.0)

    def test_no_drawdown_is_nan_when_cagr_zero(self):
        equity = pd.Series([100.0, 100.0, 100.0])
        assert np.isnan(calmar_ratio(equity))


class TestInformationRatio:
    def test_documented_formula(self):
        active = RETURNS - BENCHMARK
        expected = active.mean() / active.std()
        assert information_ratio(RETURNS, BENCHMARK) == pytest.approx(expected)

    def test_no_undocumented_annualization(self):
        active = RETURNS - BENCHMARK
        expected = active.mean() / active.std()
        assert information_ratio(RETURNS, BENCHMARK) == pytest.approx(expected)
        assert not math.isclose(
            information_ratio(RETURNS, BENCHMARK),
            expected * math.sqrt(252),
            rel_tol=1e-9,
        )

    def test_benchmark_required(self):
        with pytest.raises(ValueError):
            information_ratio(RETURNS)
        with pytest.raises(ValueError):
            information_ratio(RETURNS, benchmark=None)

    def test_identical_series_zero_tracking_error(self):
        # TE == 0 and active mean == 0 -> 0/0 -> NaN via numpy semantics
        assert np.isnan(information_ratio(RETURNS, RETURNS.copy()))


class TestBeta:
    def test_documented_cov_var_formula(self):
        expected = RETURNS.cov(BENCHMARK) / BENCHMARK.var()
        assert beta(RETURNS, BENCHMARK) == pytest.approx(expected)

    def test_benchmark_required(self):
        with pytest.raises(ValueError):
            beta(RETURNS)

    def test_constant_benchmark_is_nan(self):
        # Var(R_b) == 0 -> 0/0 -> NaN via numpy semantics
        const = pd.Series([0.005] * len(RETURNS))
        assert np.isnan(beta(RETURNS, const))

    def test_no_annualization_ratio_is_scale_free(self):
        # beta = Cov(R_p, R_b) / Var(R_b). Cov is linear in the first
        # argument, so scaling the PORTFOLIO returns by k scales beta by k.
        scaled = RETURNS * 3.0
        assert beta(scaled, BENCHMARK) == pytest.approx(beta(RETURNS, BENCHMARK) * 3.0)
        # Scaling the BENCHMARK does NOT leave the denominator linear the
        # same way; verify against the direct formula:
        expected = (RETURNS * 3.0).cov(BENCHMARK) / BENCHMARK.var()
        assert beta(scaled, BENCHMARK) == pytest.approx(expected)


class TestTreynorRatio:
    def test_documented_formula(self):
        b = RETURNS.cov(BENCHMARK) / BENCHMARK.var()
        expected = (MEAN_R - 0.0) / b
        assert treynor_ratio(RETURNS, BENCHMARK) == pytest.approx(expected)

    def test_no_undocumented_annualization(self):
        b = RETURNS.cov(BENCHMARK) / BENCHMARK.var()
        expected = (MEAN_R - 0.0) / b
        assert treynor_ratio(RETURNS, BENCHMARK) == pytest.approx(expected)
        assert not math.isclose(
            treynor_ratio(RETURNS, BENCHMARK),
            expected * math.sqrt(252),
            rel_tol=1e-9,
        )

    def test_benchmark_required(self):
        with pytest.raises(ValueError):
            treynor_ratio(RETURNS)


class TestAlpha:
    def test_documented_capm_expression(self):
        b = RETURNS.cov(BENCHMARK) / BENCHMARK.var()
        rf = 0.0
        expected = MEAN_R - (rf + b * (MEAN_B - rf))
        assert alpha(RETURNS, BENCHMARK) == pytest.approx(expected)

    def test_risk_free_rate_and_no_annualization(self):
        rf = 0.03
        b = RETURNS.cov(BENCHMARK) / BENCHMARK.var()
        expected = MEAN_R - (rf + b * (MEAN_B - rf))
        assert alpha(RETURNS, BENCHMARK, risk_free_rate=rf) == pytest.approx(expected)

    def test_benchmark_required(self):
        with pytest.raises(ValueError):
            alpha(RETURNS)

    def test_no_undocumented_annualization(self):
        rf = 0.03
        b = RETURNS.cov(BENCHMARK) / BENCHMARK.var()
        expected = MEAN_R - (rf + b * (MEAN_B - rf))
        assert alpha(RETURNS, BENCHMARK, risk_free_rate=rf) == pytest.approx(expected)
        assert not math.isclose(
            alpha(RETURNS, BENCHMARK, risk_free_rate=rf),
            expected * math.sqrt(252),
            rel_tol=1e-9,
        )

    def test_matches_market_implies_zero_alpha(self):
        # portfolio = benchmark returns => alpha = 0
        assert alpha(BENCHMARK, BENCHMARK.copy()) == pytest.approx(0.0)