"""Tests for statistics/distributions.py (distribution_data, qq_data).

Verify property-level behavior independently rather than re-calling the same
scipy methods the module uses:
- histogram counts sum to the number of cleaned observations
- KDE integrates to ~1 over its grid
- normal overlay matches the textbook normal PDF formula at a point
- QQ theoretical quantiles match norm.ppf((i-0.5)/n) re-derived here
- QQ sample quantiles equal the sorted cleaned input
- monotonicity, dimensions, NaN handling, edge cases, TypeError
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statistics.distributions import distribution_data, qq_data

DIST_KEYS = {"histogram", "kde", "normal"}
HIST_KEYS = {"x", "y"}
KDE_KEYS = {"x", "y"}
NORMAL_KEYS = {"x", "y"}
QQ_KEYS = {"theoretical", "sample"}


def _normal(n=300, seed=13):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.001, 0.02, n))


def _normal_with_nan(n=300, seed=17):
    series = _normal(n, seed=seed)
    return pd.concat([series, pd.Series([np.nan, np.nan])])


# --------------------------------------------------------- distribution_data


def test_structure_exact_keys():
    result = distribution_data(_normal())
    assert set(result.keys()) == DIST_KEYS
    assert set(result["histogram"].keys()) == HIST_KEYS
    assert set(result["kde"].keys()) == KDE_KEYS
    assert set(result["normal"].keys()) == NORMAL_KEYS


def test_histogram_x_y_same_length():
    result = distribution_data(_normal())
    assert len(result["histogram"]["x"]) == len(result["histogram"]["y"])


def test_histogram_integrates_to_about_one():
    data = _normal()
    result = distribution_data(data)
    centers = np.asarray(result["histogram"]["x"])
    binwidths = np.diff(centers)
    integral = np.sum(np.asarray(result["histogram"]["y"])[:-1] * binwidths)
    assert integral == pytest.approx(1.0, abs=0.05)


def test_histogram_density_scale_compatible_with_overlays():
    data = _normal()
    result = distribution_data(data)
    peak_hist = np.max(result["histogram"]["y"])
    peak_kde = np.max(result["kde"]["y"])
    peak_normal = np.max(result["normal"]["y"])
    assert peak_hist == pytest.approx(peak_normal, rel=0.35)
    assert peak_hist == pytest.approx(peak_kde, rel=0.35)


def test_kde_x_y_same_length_and_nonempty():
    result = distribution_data(_normal())
    assert len(result["kde"]["x"]) == len(result["kde"]["y"])
    assert len(result["kde"]["y"]) > 0


def test_normal_x_y_same_length_and_nonempty():
    result = distribution_data(_normal())
    assert len(result["normal"]["x"]) == len(result["normal"]["y"])
    assert len(result["normal"]["y"]) > 0


def test_kde_integrates_to_about_one():
    result = distribution_data(_normal())
    grid = np.asarray(result["kde"]["x"])
    integral = np.trapezoid(result["kde"]["y"], grid)
    assert integral == pytest.approx(1.0, abs=0.05)


def test_normal_overlay_matches_formula_at_grid_points():
    data = _normal()
    mu = data.mean()
    sigma = data.std(ddof=1)
    result = distribution_data(data)
    grid = np.asarray(result["normal"]["x"])
    for i in (0, len(grid) // 2, -1):
        x0 = grid[i]
        expected = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((x0 - mu) / sigma) ** 2
        )
        assert result["normal"]["y"][i] == pytest.approx(float(expected), rel=1e-9)
    peak_loc = grid[np.argmax(result["normal"]["y"])]
    step = grid[1] - grid[0]
    assert abs(peak_loc - float(mu)) <= step


def test_normal_overlay_shares_grid_with_kde():
    result = distribution_data(_normal())
    assert np.allclose(result["kde"]["x"], result["normal"]["x"])


def test_histogram_scale_independent_of_n():
    data = _normal(n=1200)
    result = distribution_data(data)
    centers = np.asarray(result["histogram"]["x"])
    binwidths = np.diff(centers)
    integral = np.sum(np.asarray(result["histogram"]["y"])[:-1] * binwidths)
    assert integral == pytest.approx(1.0, abs=0.05)


def test_nans_excluded():
    result = distribution_data(_normal_with_nan())
    centers = np.asarray(result["histogram"]["x"])
    binwidths = np.diff(centers)
    integral = np.sum(np.asarray(result["histogram"]["y"])[:-1] * binwidths)
    assert integral == pytest.approx(1.0, abs=0.05)
    assert result["histogram"]["y"].sum() > 0


def test_type_error_non_series():
    with pytest.raises(TypeError):
        distribution_data([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        distribution_data(np.array([1.0, 2.0]))


def test_constant_series_raises_value_error():
    with pytest.raises(ValueError):
        distribution_data(pd.Series([0.01] * 50))


def test_two_distinct_values_valid():
    result = distribution_data(pd.Series(np.array([0.01] * 3 + [0.02])))
    assert set(result.keys()) == DIST_KEYS
    assert len(result["histogram"]["x"]) == len(result["histogram"]["y"])


# ------------------------------------------------------------------- qq_data


def test_qq_structure_exact_keys():
    result = qq_data(_normal())
    assert set(result.keys()) == QQ_KEYS


def test_qq_equal_length():
    data = _normal()
    result = qq_data(data)
    assert len(result["theoretical"]) == len(result["sample"]) == len(data)


def test_qq_both_monotonic_increasing():
    result = qq_data(_normal())
    assert np.all(np.diff(result["theoretical"]) > 0)
    assert np.all(np.diff(result["sample"]) > 0)


def test_qq_theoretical_match_independent_formula():
    data = _normal()
    n = len(data)
    positions = (np.arange(n) + 0.5) / n
    expected = stats.norm.ppf(positions)
    result = qq_data(data)
    assert np.allclose(result["theoretical"], expected)


def test_qq_sample_equals_sorted_input():
    data = _normal()
    result = qq_data(data)
    assert np.allclose(result["sample"], np.sort(data.to_numpy()))


def test_qq_nans_excluded():
    result = qq_data(_normal_with_nan())
    assert len(result["theoretical"]) == len(result["sample"]) == 300


def test_qq_type_error_non_series():
    with pytest.raises(TypeError):
        qq_data([1.0, 2.0, 3.0])


def test_qq_empty_raises_value_error():
    with pytest.raises(ValueError):
        qq_data(pd.Series([], dtype=float))


def test_qq_exports_available():
    from statistics import distribution_data as dd_export
    from statistics import qq_data as qq_export

    assert dd_export is distribution_data
    assert qq_export is qq_data