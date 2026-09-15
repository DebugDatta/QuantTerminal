"""Tests for risk/metrics.py (Phase 2 tail-risk metrics)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from risk.metrics import conditional_var, tail_risk, value_at_risk

VAR_KEYS = {"historical", "parametric", "confidence_level", "n"}
CVAR_KEYS = {"cvar", "var", "confidence_level", "n"}
TAIL_KEYS = {"tail_ratio", "left_mean", "right_mean", "n"}


def _returns(n: int = 500, seed: int = 0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(0.0, 0.01, n))


def _manual_historical_var(values: np.ndarray, c: float) -> float:
    return float(np.percentile(values, 100.0 * (1.0 - c)))


def _manual_parametric_var(values: np.ndarray, c: float) -> float:
    return float(np.mean(values) - np.std(values, ddof=1) * norm.ppf(c))


# ---------------------------------------------------------------- structure


def test_var_exact_keys():
    assert set(value_at_risk(_returns()).keys()) == VAR_KEYS


def test_cvar_exact_keys():
    assert set(conditional_var(_returns()).keys()) == CVAR_KEYS


def test_tail_exact_keys():
    assert set(tail_risk(_returns()).keys()) == TAIL_KEYS


# ---------------------------------------------------------------- formulas


def test_historical_var_matches_manual():
    s = _returns()
    out = value_at_risk(s)
    assert out["historical"] == pytest.approx(
        _manual_historical_var(s.dropna().to_numpy(), 0.95), abs=1e-12
    )


def test_parametric_var_matches_manual():
    s = _returns()
    out = value_at_risk(s)
    assert out["parametric"] == pytest.approx(
        _manual_parametric_var(s.dropna().to_numpy(), 0.95), abs=1e-12
    )


def test_parametric_var_normal_closed_form():
    data = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    out = value_at_risk(pd.Series(data), confidence_level=0.95)
    expected = np.mean(data) - np.std(data, ddof=1) * 1.6448536269514729
    assert out["parametric"] == pytest.approx(expected, abs=1e-12)


def test_cvar_matches_manual_conditional_mean():
    s = _returns()
    vals = s.dropna().to_numpy()
    var = _manual_historical_var(vals, 0.95)
    expected = float(np.mean(vals[vals < var]))
    assert conditional_var(s)["cvar"] == pytest.approx(expected, abs=1e-12)


def test_cvar_more_negative_than_var():
    out = conditional_var(_returns())
    assert out["cvar"] <= out["var"]


def test_cvar_var_matches_value_at_risk_historical():
    s = _returns()
    assert conditional_var(s)["var"] == pytest.approx(
        value_at_risk(s)["historical"], abs=1e-12
    )


def test_tail_ratio_matches_manual():
    s = _returns()
    vals = s.dropna().to_numpy()
    p5, p95 = np.percentile(vals, 5), np.percentile(vals, 95)
    expected = -np.mean(vals[vals < p5]) / np.mean(vals[vals > p95])
    out = tail_risk(s)
    assert out["tail_ratio"] == pytest.approx(expected, abs=1e-12)
    assert out["left_mean"] == pytest.approx(np.mean(vals[vals < p5]), abs=1e-12)
    assert out["right_mean"] == pytest.approx(np.mean(vals[vals > p95]), abs=1e-12)


def test_tail_ratio_nonnegative():
    for seed in range(5):
        assert tail_risk(_returns(seed=seed))["tail_ratio"] >= 0


def test_left_tail_fatter_when_skewed_down():
    rng = np.random.default_rng(0)
    normal = pd.Series(rng.normal(0, 0.01, 2000))
    skewed = pd.Series(rng.normal(0, 0.01, 2000))
    skewed.iloc[:50] = -0.15
    assert tail_risk(skewed)["tail_ratio"] > tail_risk(normal)["tail_ratio"]


# ---------------------------------------------------------------- confidence


def test_confidence_default_095():
    assert value_at_risk(_returns())["confidence_level"] == 0.95
    assert conditional_var(_returns())["confidence_level"] == 0.95


def test_confidence_changes_historical_var():
    s = _returns()
    v90 = value_at_risk(s, confidence_level=0.90)["historical"]
    v99 = value_at_risk(s, confidence_level=0.99)["historical"]
    assert v99 <= v90


@pytest.mark.parametrize("c", [0.90, 0.95, 0.99])
def test_confidence_range_accepted(c):
    assert value_at_risk(_returns(), confidence_level=c)["confidence_level"] == c
    assert conditional_var(_returns(), confidence_level=c)["confidence_level"] == c


def test_confidence_out_of_range():
    with pytest.raises(ValueError, match="confidence_level"):
        value_at_risk(_returns(), confidence_level=0.5)
    with pytest.raises(ValueError, match="confidence_level"):
        conditional_var(_returns(), confidence_level=0.995)


def test_confidence_non_number():
    with pytest.raises(TypeError, match="confidence_level"):
        value_at_risk(_returns(), confidence_level="95%")
    with pytest.raises(TypeError, match="confidence_level"):
        value_at_risk(_returns(), confidence_level=True)


# ---------------------------------------------------------------- nan/guards


def test_drops_nan():
    s = pd.Series([-0.01, np.nan, 0.02, -0.03, 0.01, 0.0])
    assert value_at_risk(s)["n"] == 5


def test_nan_propagates_consistent_n():
    s = pd.Series([-0.01, np.nan, 0.02, -0.03, 0.01, 0.0])
    assert conditional_var(s)["n"] == 5
    assert tail_risk(s)["n"] == 5


def test_rejects_non_series():
    with pytest.raises(TypeError, match="Series"):
        value_at_risk(np.array([0.1, 0.2]))


def test_insufficient_observations():
    with pytest.raises(ValueError, match="observations"):
        value_at_risk(pd.Series([0.01]))
    with pytest.raises(ValueError, match="observations"):
        conditional_var(pd.Series([0.01]))
    with pytest.raises(ValueError, match="observations"):
        tail_risk(pd.Series([0.01]))


def test_no_risk_free_rate_parameter():
    import inspect

    for fn in (value_at_risk, conditional_var, tail_risk):
        assert "risk_free_rate" not in inspect.signature(fn).parameters


def test_values_float_types():
    out = value_at_risk(_returns())
    assert isinstance(out["historical"], float)
    assert isinstance(out["parametric"], float)
    assert isinstance(conditional_var(_returns())["cvar"], float)
    assert isinstance(tail_risk(_returns())["tail_ratio"], float)