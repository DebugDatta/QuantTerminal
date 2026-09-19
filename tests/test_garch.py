"""Tests for volatility/garch.py (Phase 2 GARCH models)."""

import numpy as np
import pandas as pd
import pytest

from volatility.garch import (
    DISTRIBUTIONS,
    fit_egarch,
    fit_garch,
    fit_gjr_garch,
)

GARCH_KEYS = {
    "model",
    "p",
    "q",
    "distribution",
    "coefficients",
    "conditional_volatility",
    "residuals",
    "aic",
    "bic",
    "ljung_box",
    "forecast",
    "n",
    "converged",
}
COEF_KEYS = {"omega", "alpha", "beta", "gamma"}


def _series(n: int = 300, seed: int = 42, vol: float = 1.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    sigma = np.zeros(n)
    z = rng.normal(0.0, 1.0, n)
    for t in range(1, n):
        sigma[t] = np.sqrt(0.05 + 0.1 * sigma[t - 1] ** 2 + 0.2 * (sigma[t - 1] * z[t - 1]) ** 2)
    return pd.Series(sigma * z * vol)


@pytest.fixture(scope="module")
def fitted():
    """Cached fits for structural assertions (avoids refitting each test)."""
    import warnings

    s = _series()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "garch": fit_garch(s),
            "egarch": fit_egarch(s),
            "gjr": fit_gjr_garch(s),
        }


# ---------------------------------------------------------------- structure


def test_garch_exact_keys(fitted):
    assert set(fitted["garch"].keys()) == GARCH_KEYS


def test_egarch_exact_keys(fitted):
    assert set(fitted["egarch"].keys()) == GARCH_KEYS


def test_gjr_exact_keys(fitted):
    assert set(fitted["gjr"].keys()) == GARCH_KEYS


def test_model_names(fitted):
    assert fitted["garch"]["model"] == "GARCH"
    assert fitted["egarch"]["model"] == "EGARCH"
    assert fitted["gjr"]["model"] == "GJR-GARCH"


def test_coefficients_structure(fitted):
    for key in ("garch", "egarch", "gjr"):
        coefs = fitted[key]["coefficients"]
        assert set(["omega", "alpha", "beta"]).issubset(set(coefs.keys()))
        assert isinstance(coefs["alpha"], list)
        assert isinstance(coefs["beta"], list)
        assert isinstance(coefs["gamma"], list)
        assert len(coefs["alpha"]) == 1
        assert len(coefs["beta"]) == 1


def test_mu_is_present(fitted):
    for key in ("garch", "egarch", "gjr"):
        assert "mu" in fitted[key]["coefficients"]


def test_garch_no_gamma_egarch_and_gjr_have_gamma(fitted):
    assert fitted["garch"]["coefficients"]["gamma"] == []
    assert len(fitted["egarch"]["coefficients"]["gamma"]) == 1
    assert len(fitted["gjr"]["coefficients"]["gamma"]) == 1


# ---------------------------------------------------------------- outputs


def test_conditional_volatility_series(fitted):
    for key in ("garch", "egarch", "gjr"):
        sig = fitted[key]["conditional_volatility"]
        assert len(sig) == 300
        assert np.all(np.isfinite(sig))
        assert np.all(sig > 0)


def test_standardized_residuals(fitted):
    for key in ("garch", "egarch", "gjr"):
        r = fitted[key]["residuals"]
        assert len(r) == 300
        assert np.all(np.isfinite(r))


def test_badge_inputs_present(fitted):
    for key in ("garch", "egarch", "gjr"):
        out = fitted[key]
        assert isinstance(out["n"], int) and out["n"] == 300
        assert isinstance(out["converged"], bool)
        assert isinstance(out["aic"], float)
        assert isinstance(out["bic"], float)
        lb = out["ljung_box"]
        assert set(["statistic", "p_value", "lags"]).issubset(set(lb.keys()))
        assert 0.0 <= lb["p_value"] <= 1.0


def test_forecast_horizon(fitted):
    for key in ("garch", "egarch", "gjr"):
        fc = fitted[key]["forecast"]
        assert fc["horizon"] == 5
        assert len(fc["volatility"]) == 5
        assert len(fc["variance"]) == 5
        assert np.all(fc["volatility"] > 0)


def test_forecast_custom_horizon():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fit_garch(_series(), horizon=3)
    assert out["forecast"]["horizon"] == 3
    assert len(out["forecast"]["volatility"]) == 3


# ---------------------------------------------------------------- parameters


def test_custom_p_q():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fit_garch(_series(), p=2, q=2)
    assert out["p"] == 2
    assert out["q"] == 2
    assert len(out["coefficients"]["alpha"]) == 2
    assert len(out["coefficients"]["beta"]) == 2


def test_distribution_option_accepted():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fit_garch(_series(), distribution="studentt")
    assert out["distribution"] == "studentt"
    assert all(dist in DISTRIBUTIONS for dist in ("normal", "studentt", "skewedstudentt"))


def test_model_type_pure_garch_recovers_vol():
    # Higher moments and magnitude sanity on a generated path.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fit_garch(_series(vol=2.0), p=1, q=1)
    assert out["conditional_volatility"].mean() > 0.4


# ---------------------------------------------------------------- guards


def test_rejects_non_series():
    with pytest.raises(TypeError, match="Series"):
        fit_garch(pd.DataFrame({"A": [1.0, 2.0]}))


def test_order_out_of_range():
    s = _series()
    with pytest.raises(ValueError, match="p"):
        fit_garch(s, p=0)
    with pytest.raises(ValueError, match="p"):
        fit_garch(s, p=6)
    with pytest.raises(ValueError, match="q"):
        fit_garch(s, q=0)
    with pytest.raises(ValueError, match="q"):
        fit_garch(s, q=99)


def test_order_non_int():
    s = _series()
    with pytest.raises(TypeError, match="p"):
        fit_garch(s, p=1.5)
    with pytest.raises(TypeError, match="p"):
        fit_garch(s, p=True)


def test_invalid_distribution():
    s = _series()
    with pytest.raises(ValueError, match="distribution"):
        fit_garch(s, distribution="t-copula")


def test_horizon_invalid():
    s = _series()
    with pytest.raises(ValueError, match="horizon"):
        fit_garch(s, horizon=0)
    with pytest.raises(TypeError, match="horizon"):
        fit_garch(s, horizon=2.5)


def test_insufficient_observations():
    s = pd.Series([1.0, -0.5, 0.3, 1.2])
    with pytest.raises(ValueError, match="observations"):
        fit_garch(s)


def test_all_models_accept_same_interface():
    import warnings

    s = _series(n=120)
    for fn in (fit_garch, fit_egarch, fit_gjr_garch):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fn(s, p=1, q=1, distribution="normal", horizon=2)
        assert out["n"] == 120
        assert out["forecast"]["horizon"] == 2


def test_deterministic_garch_fit():
    import warnings

    s = _series(seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = fit_garch(s)
        b = fit_garch(s)
    assert a["aic"] == b["aic"]
    assert a["coefficients"]["omega"] == b["coefficients"]["omega"]
    np.testing.assert_allclose(a["conditional_volatility"], b["conditional_volatility"])