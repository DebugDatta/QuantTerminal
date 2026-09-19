"""Signal generation helpers. Spec: docs/TECHNICAL_INDICATORS.md -> Signal Generation."""

import numpy as np
import pandas as pd


def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """1 when series1 crosses above series2, -1 when crosses below, else 0.

    A crossing occurs when series1 transitions from below/equal series2 to above
    series1 (signal = 1), or from above/equal series2 to below series2 (signal = -1).
    NaN values in either series prevent a crossing signal at that index.
    """
    s1 = series1.values
    s2 = series2.values
    result = np.zeros(len(s1), dtype=int)

    prev_below_or_eq = s1[0] <= s2[0]
    prev_above_or_eq = s1[0] >= s2[0]

    for i in range(1, len(s1)):
        curr_above = s1[i] > s2[i]
        curr_below = s1[i] < s2[i]

        if curr_above and prev_below_or_eq:
            result[i] = 1
        elif curr_below and prev_above_or_eq:
            result[i] = -1

        prev_below_or_eq = s1[i] <= s2[i]
        prev_above_or_eq = s1[i] >= s2[i]

        if np.isnan(s1[i]) or np.isnan(s2[i]):
            prev_below_or_eq = False
            prev_above_or_eq = False

    return pd.Series(result, index=series1.index)


def threshold(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """1 when series crosses above upper, -1 when crosses below lower, else 0.

    A crossing above upper occurs when series transitions from <= upper to > upper
    (signal = 1). A crossing below lower occurs when series transitions from >= lower
    to < lower (signal = -1). NaN values prevent a crossing signal at that index.
    """
    s = series.values
    result = np.zeros(len(s), dtype=int)

    prev_le_upper = s[0] <= upper
    prev_ge_lower = s[0] >= lower

    for i in range(1, len(s)):
        if np.isnan(s[i]) or np.isnan(s[i - 1]):
            prev_le_upper = False
            prev_ge_lower = False
            continue

        if s[i] > upper and prev_le_upper:
            result[i] = 1
        if s[i] < lower and prev_ge_lower:
            result[i] = -1

        prev_le_upper = s[i] <= upper
        prev_ge_lower = s[i] >= lower

    return pd.Series(result, index=series.index)


def signal_table(
    price: pd.Series,
    indicator: pd.Series,
    signals: pd.Series,
) -> pd.DataFrame:
    """Build a signal table aligned to the given index.

    Returns a DataFrame with columns:
      - Date: the original index values
      - Indicator Value: the indicator reading at each point
      - Signal: the signal value (+1, -1, or 0)
      - Direction: 'Bullish' for +1, 'Bearish' for -1, 'Neutral' for 0
    """
    direction_map = {1: "Bullish", -1: "Bearish", 0: "Neutral"}
    directions = signals.map(direction_map)

    df = pd.DataFrame(
        {
            "Date": price.index,
            "Indicator Value": indicator.values,
            "Signal": signals.values,
            "Direction": directions.values,
        }
    )

    return df
