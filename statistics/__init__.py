"""Statistics & risk modules - descriptive statistics and stationarity tests.

Exports:
    summary_statistics - descriptive statistics of a return series
    adf_test           - Augmented Dickey-Fuller unit-root test
    kpss_test          - KPSS stationarity test
    pp_test            - Phillips-Perron unit-root test
    zivot_andrews      - Zivot-Andrews unit-root test with one break
"""

from statistics.stationarity import adf_test, kpss_test, pp_test, zivot_andrews
from statistics.summary import summary_statistics

__all__ = [
    "summary_statistics",
    "adf_test",
    "kpss_test",
    "pp_test",
    "zivot_andrews",
]