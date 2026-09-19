"""Return-series distribution analysis - plot-ready data builders.

Functions:
    distribution_data - histogram, KDE density, and normal-overlay data
    qq_data          - normal-theory Q-Q plot data

Contract sources:
    docs/STATISTICAL_MODELS.md §3         - Distribution-analysis visuals (histogram
                                            with normal overlay, Density plot (KDE),
                                            Q-Q plot (quantiles vs normal))
    docs/ARCHITECTURE.md                   - Function names pinned in directory tree;
                                            plotting lives in plots/distributions.py
                                            (plot_histogram, plot_density, plot_qq)
    docs/STREAMLIT_PAGES.md Page 3         - Return distribution histogram with normal
                                            overlay, density plot, Q-Q plot

Boundary (ARCHITECTURE.md):
    This module returns plot-ready numeric data only - figures are built by
    plots/distributions.py. No plotting/imports here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from statistics._common import clean_series


def _grid(data: pd.Series, points: int = 200) -> np.ndarray:
    mu = float(data.mean())
    lo, hi = float(data.min()), float(data.max())
    half = (hi - lo) / 2.0 if hi != lo else abs(mu) * 0.05
    if half == 0.0:
        half = 1.0
    return np.linspace(lo - half, hi + half, points)


def distribution_data(returns: pd.Series) -> dict:
    """Compute plot-ready histogram, KDE density, and normal-overlay data.

    Parameters
    ----------
    returns : pd.Series
        Return series (e.g. from core.returns.compute_returns).

    Returns
    -------
    dict
        Keys:
            histogram : {"x": bin centers, "y": bin densities}
            kde       : {"x": grid, "y": KDE density}
            normal    : {"x": grid, "y": normal PDF from sample mean/std}

    Notes
    -----
    - Only plot-ready numeric data is returned; figures are built by
      plots/distributions.py (ARCHITECTURE.md split - B-class boundary for
      the data/figures contract).
    - Everything below is B-class inference because the docs specify the
      visuals (histogram + normal overlay, KDE) but not the details:
      * output container/keys (dict)
      * the histogram is a DENSITY (numpy density=True), not raw counts, so
        it shares a y-scale with the KDE and normal-overlay PDFs - required
        for the documented "histogram with overlay of normal distribution"
        (a counts histogram would not be overlappable with a PDF);
        numpy's default bin rule is used
      * KDE via scipy.stats.gaussian_kde with its default bandwidth
      * a shared 200-point x-grid spanning the data plus half-range padding
      * normal overlay is the normal PDF evaluated at sample mean/std on the
        same raw-return scale (returns are NOT standardized)
      * NaNs dropped before computation (module convention)
      * edge case: a constant/zero-variance sample with a single unique
        value cannot support a density - a ValueError is raised rather than
        returning a degenerate curve.
    """
    data = clean_series(returns)
    if data.nunique() < 2:
        raise ValueError(
            "distribution_data requires at least 2 distinct observations "
            f"for a histogram/KDE; got {data.nunique()} distinct value(s)"
        )

    density_vals, edges = np.histogram(data, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    histogram = {"x": centers, "y": density_vals}

    grid = _grid(data)
    kde = stats.gaussian_kde(data.to_numpy())
    kde_y = kde(grid)
    density = {"x": grid, "y": kde_y}

    mu = float(data.mean())
    sigma = float(data.std(ddof=1))
    pdf = stats.norm.pdf(grid, loc=mu, scale=sigma)
    normal = {"x": grid, "y": pdf}

    return {"histogram": histogram, "kde": density, "normal": normal}


def qq_data(returns: pd.Series) -> dict:
    """Compute normal-theory Q-Q plot data.

    Parameters
    ----------
    returns : pd.Series
        Return series.

    Returns
    -------
    dict
        Keys:
            theoretical : standard-normal quantiles (norm.ppf((i-0.5)/n))
            sample      : ordered cleaned return values

    Notes
    -----
    - Q-Q against the normal distribution is documented; plotting positions
      and output keys are B-class inference:
      * plotting positions p_i = (i - 0.5) / n with theoretical quantiles
        norm.ppf(p_i) from a standard normal
      * sample values are the ascending sorted cleaned returns
      * both arrays share length n and are monotonic increasing
      * no reference/fitted line is returned (docs do not require one)
      * NaNs dropped; a single observation still yields a valid single point.
    """
    data = clean_series(returns)
    if len(data) == 0:
        raise ValueError("qq_data requires at least one observation")

    n = len(data)
    positions = (np.arange(n) + 0.5) / n
    theoretical = stats.norm.ppf(positions)
    sample = np.sort(data.to_numpy())
    return {"theoretical": theoretical, "sample": sample}