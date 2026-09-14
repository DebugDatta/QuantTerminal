"""Statistics & risk modules - descriptive statistics and diagnostic tests.

Exports:
    summary_statistics - descriptive statistics of a return series
    adf_test           - Augmented Dickey-Fuller unit-root test
    kpss_test          - KPSS stationarity test
    pp_test            - Phillips-Perron unit-root test
    zivot_andrews      - Zivot-Andrews unit-root test with one break
    ljung_box          - Ljung-Box autocorrelation test on residuals
    jarque_bera        - Jarque-Bera normality test
    shapiro_wilk       - Shapiro-Wilk normality test
    distribution_data  - histogram / KDE / normal-overlay plot data
    qq_data            - normal-theory Q-Q plot data
"""

from statistics.diagnostics import jarque_bera, ljung_box, shapiro_wilk
from statistics.distributions import distribution_data, qq_data
from statistics.stationarity import adf_test, kpss_test, pp_test, zivot_andrews
from statistics.summary import summary_statistics

__all__ = [
    "summary_statistics",
    "adf_test",
    "kpss_test",
    "pp_test",
    "zivot_andrews",
    "ljung_box",
    "jarque_bera",
    "shapiro_wilk",
    "distribution_data",
    "qq_data",
]