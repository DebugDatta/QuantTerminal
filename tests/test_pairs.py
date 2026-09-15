"""Tests for statarb/pairs.py (Phase 2 pair selection metrics).

Verifies the STATISTICAL_MODELS.md §9 distances with independently computed
oracles (numpy correlation / explicit math), never the implementation
itself, plus ranking, alignment, NaN, and guard behavior.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from statarb.pairs import find_pairs, pair_distance

OUT_COLS = ["asset_1", "asset_2", "distance", "correlation", "n", "rank"]


def _index(n: int, start: str = "2023-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _price(rets, start: str = "2023-01-02") -> pd.Series:
    arr = np.asarray(rets, dtype=float)
    return pd.Series((1.0 + arr).cumprod() * 100.0, index=_index(len(arr), start))


def _manual_pearson_dist(rx, ry) -> float:
    r = np.corrcoef(rx, ry)[0, 1]
    return float(1.0 - abs(r))


def _manual_spearman_dist(rx, ry) -> float:
    r = np.corrcoef(rankdata(rx), rankdata(ry))[0, 1]
    return float(1.0 - abs(r))


def _manual_euclidean(px, py) -> float:
    nx = np.asarray(px, dtype=float) / float(px[0])
    ny = np.asarray(py, dtype=float) / float(py[0])
    return float(math.sqrt(float(np.sum((nx - ny) ** 2))))


# ------------------------------------------------------------------ structure


def test_required_functions_exist():
    assert callable(find_pairs)
    assert callable(pair_distance)


def test_default_method_pearson():
    a = _price([0.01, 0.02, -0.01, 0.005, 0.015])
    b = _price([0.01, 0.02, -0.01, 0.005, 0.015])
    assert pair_distance(a, b) == pytest.approx(0.0)
    assert pair_distance(a, b, "pearson") == pytest.approx(0.0)


# ---------------------------------------------------------------- pearson


def test_pearson_distance_matches_manual():
    a = _price([0.01, -0.02, 0.03, 0.0, 0.015, -0.005, 0.02])
    b = _price([0.008, -0.015, 0.025, -0.002, 0.012, -0.01, 0.018])
    ra = a.pct_change().dropna().to_numpy()
    rb = b.pct_change().dropna().to_numpy()
    assert pair_distance(a, b, "pearson") == pytest.approx(
        _manual_pearson_dist(ra, rb)
    )


def test_pearson_distance_symmetric():
    a = _price([0.01, -0.02, 0.03, 0.0, 0.015])
    b = _price([0.008, -0.015, 0.025, -0.002, 0.012])
    assert pair_distance(a, b, "pearson") == pytest.approx(
        pair_distance(b, a, "pearson")
    )


def test_pearson_distance_range():
    a = _price([0.01, 0.02, 0.015, 0.03, 0.01, 0.025, 0.02, 0.018])
    b = _price([0.5, 0.51, 0.505, 0.52, 0.5, 0.515, 0.51, 0.508])
    c = _price([-0.01, -0.02, -0.015, -0.03, -0.01, -0.025, -0.02, -0.018])
    assert 0.0 <= pair_distance(a, b, "pearson") <= 2.0
    assert pair_distance(a, b, "pearson") == pytest.approx(0.0)
    assert pair_distance(b, c, "pearson") == pytest.approx(0.0)


def test_correlation_column_is_signed_rho():
    a = _price([0.01, 0.02, 0.015, 0.03])
    b = _price([0.5, 0.51, 0.505, 0.52])
    out = find_pairs(pd.DataFrame({"A": a, "B": b}), method="pearson").iloc[0]
    assert out["distance"] == pytest.approx(0.0)
    assert out["correlation"] == pytest.approx(1.0)


def test_constant_leg_pearson_is_nan():
    a = _price([0.01, 0.02, 0.01, 0.02, 0.01])
    const = pd.Series([100.0] * 5, index=a.index)
    assert np.isnan(pair_distance(a, const, "pearson"))


# ---------------------------------------------------------------- spearman


def test_spearman_distance_matches_manual():
    a = _price([0.01, -0.02, 0.03, 0.0, 0.015, -0.005, 0.02, 0.008])
    b = _price([0.002, -0.03, 0.04, -0.01, 0.02, -0.02, 0.025, 0.001])
    ra = a.pct_change().dropna().to_numpy()
    rb = b.pct_change().dropna().to_numpy()
    assert pair_distance(a, b, "spearman") == pytest.approx(
        _manual_spearman_dist(ra, rb)
    )


def test_spearman_distance_perfect_monotone_is_zero():
    a = _price([0.01, 0.02, -0.03, 0.04, 0.015, -0.01])
    b = _price([0.005, 0.015, -0.035, 0.045, 0.01, -0.02])
    assert pair_distance(a, b, "spearman") == pytest.approx(0.0)
    assert pair_distance(a, b, "spearman") == pytest.approx(
        _manual_spearman_dist(
            a.pct_change().dropna().to_numpy(),
            b.pct_change().dropna().to_numpy(),
        )
    )


def test_spearman_differs_from_pearson_on_exponential_relation():
    raw = np.linspace(0.0, 1.0, 20)
    a = pd.Series((1.0 + raw).cumprod() * 100.0, index=_index(20))
    b = pd.Series((1.0 + raw**1.5).cumprod() * 100.0, index=_index(20))
    assert pair_distance(a, b, "spearman") == pytest.approx(0.0)
    assert pair_distance(a, b, "pearson") > 1e-3
    assert pair_distance(a, b, "spearman") != pytest.approx(
        pair_distance(a, b, "pearson")
    )


# ---------------------------------------------------------------- euclidean


def test_euclidean_distance_matches_manual():
    a = _price([0.01, 0.05, -0.02, 0.03, -0.01])
    b = _price([0.03, 0.04, 0.02, -0.02, 0.015])
    assert pair_distance(a, b, "euclidean") == pytest.approx(
        _manual_euclidean(a.to_numpy(), b.to_numpy())
    )


def test_euclidean_hand_computed():
    a = pd.Series([1.0, 2.0, 3.0], index=_index(3))
    b = pd.Series([4.0, 5.0, 6.0], index=_index(3))
    expected = math.sqrt(0.0 + (2.0 / 1.0 - 5.0 / 4.0) ** 2 + (3.0 / 1.0 - 6.0 / 4.0) ** 2)
    assert pair_distance(a, b, "euclidean") == pytest.approx(expected)


def test_euclidean_scale_invariant():
    a = _price([0.01, 0.02, -0.01, 0.005])
    b = _price([0.015, 0.03, -0.005, 0.01])
    scaled = a * 1000.0
    assert pair_distance(a, b, "euclidean") == pytest.approx(
        pair_distance(scaled, b, "euclidean")
    )


def test_euclidean_identical_series_zero():
    a = _price([0.01, 0.02, -0.01, 0.005, 0.02])
    assert pair_distance(a, a.copy(), "euclidean") == pytest.approx(0.0)


def test_euclidean_correlation_column_is_nan():
    a = _price([0.01, 0.02, -0.01, 0.005])
    b = _price([0.015, 0.03, -0.005, 0.01])
    out = find_pairs(pd.DataFrame({"A": a, "B": b}), method="euclidean").iloc[0]
    assert np.isnan(out["correlation"])


# --------------------------------------------------------------- find_pairs


def test_find_pairs_returns_all_unordered_pairs():
    names = ["A", "B", "C", "D"]
    frame = pd.DataFrame(
        {n: _price([0.01, -0.02, 0.03, 0.0, 0.015, 0.02, -0.01]) for n in names}
    )
    out = find_pairs(frame)
    assert len(out) == 6
    assert list(out.columns) == OUT_COLS
    assert list(out["rank"]) == list(range(1, 7))
    pairs = set(zip(out["asset_1"], out["asset_2"]))
    assert pairs == {("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")}


def test_find_pairs_sorted_ascending_distance():
    frame = pd.DataFrame(
        {
            "A": _price([0.01, -0.02, 0.03, 0.0, 0.015]),
            "B": _price([0.011, -0.019, 0.031, 0.001, 0.016]),
            "C": _price([-0.02, 0.01, -0.03, 0.02, -0.01]),
        }
    )
    out = find_pairs(frame)
    assert list(out["distance"]) == sorted(out["distance"])


def test_find_pairs_pearson_ranking_matches_oracle():
    r1 = [0.01, 0.02, -0.01, 0.005, 0.015, -0.02, 0.02]
    r2 = [0.01, 0.02, -0.01, 0.005, 0.015, -0.02, 0.02]
    r3 = [0.001, 0.002, 0.0, 0.0005, 0.0015, -0.002, 0.002]
    r4 = [0.005, -0.01, 0.0, 0.008, -0.006, 0.01, -0.005]
    frame = pd.DataFrame(
        {"A": _price(r1), "B": _price(r2), "C": _price(r3), "D": _price(r4)}
    )
    out = find_pairs(frame, method="pearson")
    assert out.iloc[0]["asset_1"] == "A"
    assert out.iloc[0]["asset_2"] == "B"
    assert out.iloc[0]["distance"] == pytest.approx(0.0, abs=1e-9)
    manual_ab = _manual_pearson_dist(np.asarray(r1), np.asarray(r2))
    assert out.iloc[0]["distance"] == pytest.approx(manual_ab, abs=1e-9)
    assert (out.iloc[1:]["distance"] > 1e-3).all()


def test_ties_keep_input_order_ranked_stably():
    # (B, D) normalize to identical series -> distance exactly 0.0;
    # (A, B) and (A, D) tie at a bit-exact equal Euclidean distance and must
    # keep input order (A, B) before (A, D) under the stable sort.
    frame = pd.DataFrame(
        {
            "A": pd.Series([1.0, 2.0, 4.0, 8.0], index=_index(4)),
            "B": pd.Series([2.0, 3.0, 5.0, 9.0], index=_index(4)),
            "D": pd.Series([4.0, 6.0, 10.0, 18.0], index=_index(4)),
        }
    )
    out = find_pairs(frame, method="euclidean")
    row0 = out.iloc[0]
    assert (row0["asset_1"], row0["asset_2"]) == ("B", "D")
    assert row0["distance"] == pytest.approx(0.0)
    ab = out[(out["asset_1"] == "A") & (out["asset_2"] == "B")].iloc[0]
    ad = out[(out["asset_1"] == "A") & (out["asset_2"] == "D")].iloc[0]
    assert ab["distance"] == ad["distance"]
    assert ab["rank"] == ad["rank"] - 1


def test_top_n_filters_sorted():
    frame = pd.DataFrame(
        {
            "A": _price([0.01, -0.02, 0.03, 0.0, 0.015, 0.02]),
            "B": _price([0.011, -0.019, 0.031, 0.001, 0.016, 0.021]),
            "C": _price([-0.02, 0.01, -0.03, 0.02, -0.01, 0.0]),
        }
    )
    out = find_pairs(frame, top_n=1)
    assert len(out) == 1
    full = find_pairs(frame)
    assert out.iloc[0]["rank"] == 1
    assert out.iloc[0]["distance"] == pytest.approx(full.iloc[0]["distance"])


def test_asset_labels_preserve_input_columns():
    cols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    frame = pd.DataFrame({c: _price([0.01, -0.02, 0.03, 0.0]) for c in cols})
    pairs = set(zip(find_pairs(frame)["asset_1"], find_pairs(frame)["asset_2"]))
    assert all(set(p) <= set(cols) for p in pairs)
    assert len(pairs) == 3


def test_deterministic_output():
    frame = pd.DataFrame(
        {
            "A": _price([0.01, -0.02, 0.03, 0.0, 0.015, 0.02]),
            "B": _price([0.011, -0.019, 0.031, 0.001, 0.016, -0.01]),
            "C": _price([-0.02, 0.01, -0.03, 0.02, -0.01, 0.005]),
        }
    )
    first = find_pairs(frame)
    second = find_pairs(frame)
    assert first.equals(second)


def test_method_aliases():
    a = _price([0.01, -0.02, 0.03, 0.0, 0.015])
    b = _price([0.008, -0.015, 0.025, -0.002, 0.012])
    frame = pd.DataFrame({"A": a, "B": b})
    assert pair_distance(a, b, "correlation") == pytest.approx(
        pair_distance(a, b, "pearson")
    )
    assert pair_distance(a, b, "distance") == pytest.approx(
        pair_distance(a, b, "euclidean")
    )
    assert find_pairs(frame, method="correlation").equals(
        find_pairs(frame, method="pearson")
    )
    assert find_pairs(frame, method="distance").equals(
        find_pairs(frame, method="euclidean")
    )


def test_n_column_reports_aligned_observations():
    a = _price([0.01, -0.02, 0.03, 0.0, 0.015])
    b = _price([0.008, -0.015, 0.025, -0.002, 0.012])
    out = find_pairs(pd.DataFrame({"A": a, "B": b}))
    assert out.iloc[0]["n"] == 4


# -------------------------------------------------------------- nan/alignment


def test_nan_rows_dropped_listwise():
    a = _price([0.01, -0.02, 0.03, 0.0, 0.015, np.nan])
    b = _price([0.008, -0.015, 0.025, 0.002, 0.012, 0.01])
    frame = pd.DataFrame({"A": a, "B": b})
    out = find_pairs(frame)
    assert out.iloc[0]["n"] == 4
    clean = frame.dropna()
    assert len(clean) == 5
    assert out.iloc[0]["distance"] == pytest.approx(
        pair_distance(clean["A"], clean["B"], "pearson")
    )


def test_pair_distance_aligns_on_common_dates():
    idx1 = pd.bdate_range("2024-01-01", periods=12)
    idx2 = pd.bdate_range("2024-01-02", periods=12)
    ra = np.random.default_rng(5).normal(0.0, 0.01, 12)
    rb = np.random.default_rng(6).normal(0.0, 0.008, 12)
    a = pd.Series((1.0 + ra).cumprod() * 100.0, index=idx1)
    b = pd.Series((1.0 + rb).cumprod() * 100.0, index=idx2)
    joined = pd.concat([a, b], axis=1, join="inner")
    oracle = _manual_pearson_dist(
        joined.iloc[:, 0].pct_change().dropna().to_numpy(),
        joined.iloc[:, 1].pct_change().dropna().to_numpy(),
    )
    assert pair_distance(a, b, "pearson") == pytest.approx(oracle)
    assert len(joined) == 11


# ------------------------------------------------------------------- guards


def test_insufficient_rows():
    a = _price([0.01, 0.02])
    b = _price([0.008, 0.015])
    with pytest.raises(ValueError, match="complete observations"):
        find_pairs(pd.DataFrame({"A": a, "B": b}))
    with pytest.raises(ValueError, match="complete.*observations"):
        pair_distance(a, b, "pearson")


def test_single_asset_column():
    a = _price([0.01, 0.02, 0.03, 0.0])
    with pytest.raises(ValueError, match="asset columns"):
        find_pairs(pd.DataFrame({"A": a}))


def test_non_dataframe_prices():
    with pytest.raises(TypeError, match="DataFrame"):
        find_pairs(np.zeros((5, 2)))


def test_non_series_leg():
    a = _price([0.01, 0.02, 0.03, 0.0])
    with pytest.raises(TypeError, match="Series"):
        pair_distance(a, [0.008, 0.015, 0.02, 0.0], "pearson")


def test_invalid_method():
    a = _price([0.01, 0.02, 0.03, 0.0])
    b = _price([0.008, 0.015, 0.02, 0.0])
    with pytest.raises(ValueError, match="method"):
        pair_distance(a, b, "euclid")
    with pytest.raises(ValueError, match="method"):
        find_pairs(pd.DataFrame({"A": a, "B": b}), method="foo")


def test_cointegration_method_rejected_ownership():
    a = _price([0.01, 0.02, 0.03, 0.0])
    b = _price([0.008, 0.015, 0.02, 0.0])
    with pytest.raises(ValueError, match="method"):
        find_pairs(pd.DataFrame({"A": a, "B": b}), method="cointegration")
    with pytest.raises(ValueError, match="method"):
        pair_distance(a, b, "cointegration")


@pytest.mark.parametrize("top_n", [0, -1, True, "3", 2.5])
def test_invalid_top_n(top_n):
    frame = pd.DataFrame(
        {
            "A": _price([0.01, 0.02, 0.03, 0.0]),
            "B": _price([0.008, 0.015, 0.02, 0.0]),
            "C": _price([0.005, 0.01, 0.01, 0.005]),
        }
    )
    with pytest.raises((TypeError, ValueError)):
        find_pairs(frame, top_n=top_n)