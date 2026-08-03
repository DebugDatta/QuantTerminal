# Technical Indicators

All indicators derived exclusively from OHLCV data. Grouped into 7 categories.

> **Market-agnostic**: All indicators work identically on Indian (NSE/BSE) and Global markets. No modifications needed.

---

## 1. Trend Indicators

Indicators that identify the direction and strength of a trend.

| Indicator | Parameters | Formula / Description |
|---|---|---|
| **SMA** | `window` | Simple Moving Average: `sum(Close, n) / n` |
| **EMA** | `window` | Exponential Moving Average: `EMA = α * Close + (1-α) * EMA_prev`, where `α = 2/(n+1)` |
| **WMA** | `window` | Weighted Moving Average: more weight to recent prices, linearly decreasing |
| **HMA** | `window` | Hull Moving Average: reduces lag using weighted moving averages of WMA |
| **VWMA** | `window` | Volume Weighted Moving Average: `sum(Close * Volume, n) / sum(Volume, n)` |

### SMA Parameters
| Name | Default | Range |
|---|---|---|
| window | 20 | 2–200 |

### EMA Parameters
| Name | Default | Range |
|---|---|---|
| window | 20 | 2–200 |

### WMA Parameters
| Name | Default | Range |
|---|---|---|
| window | 20 | 2–200 |

### HMA Parameters
| Name | Default | Range |
|---|---|---|
| window | 20 | 2–200 |

### VWMA Parameters
| Name | Default | Range |
|---|---|---|
| window | 20 | 2–200 |

---

## 2. Momentum Indicators

Indicators that measure the rate of price change.

| Indicator | Parameters | Formula / Description |
|---|---|---|
| **RSI** | `window` | Relative Strength Index: `100 - 100 / (1 + RS)`, where `RS = avg_gain / avg_loss` (Wilder smoothing) |
| **MACD** | `fast, slow, signal` | MACD Line: `EMA(fast) - EMA(slow)`. Signal: `EMA(MACD, signal)`. Histogram: `MACD - Signal` |
| **ROC** | `window` | Rate of Change: `(Close / Close_n_ago - 1) * 100` |
| **Stochastic** | `k_window, d_window` | `%K = (Close - Low_n) / (High_n - Low_n) * 100`. `%D = SMA(%K, d_window)` |
| **Williams %R** | `window` | `-100 * (High_n - Close) / (High_n - Low_n)` |

### RSI Parameters
| Name | Default | Range |
|---|---|---|
| window | 14 | 2–50 |
| oversold | 30 | 0–100 (full range; 0–50 is sensible for strategy use) |
| overbought | 70 | 0–100 (full range; 50–100 is sensible for strategy use) |

### MACD Parameters
| Name | Default | Range |
|---|---|---|
| fast | 12 | 1–50 |
| slow | 26 | 1–50 |
| signal | 9 | 1–20 |

### ROC Parameters
| Name | Default | Range |
|---|---|---|
| window | 12 | 1–50 |

### Stochastic Parameters
| Name | Default | Range |
|---|---|---|
| k_window | 14 | 2–50 |
| d_window | 3 | 2–20 |

### Williams %R Parameters
| Name | Default | Range |
|---|---|---|
| window | 14 | 2–50 |

---

## 3. Volatility Indicators

Indicators that measure price volatility and price channels.

| Indicator | Parameters | Formula / Description |
|---|---|---|
| **ATR** | `window` | Average True Range. `TR = max(H-L, |H-C_prev|, |L-C_prev|)`. `ATR = EMA(TR, window)` |
> **Note:** ATR feeds into Keltner Channels, ADX, and Vortex — the `abs()` correction propagates through all three.
| **Bollinger Bands** | `window, num_std` | Middle: `SMA(close, window)`. Upper: `Middle + num_std * σ`. Lower: `Middle - num_std * σ` |
| **Keltner Channels** | `window, multiplier` | Middle: `EMA(close, window)`. Upper: `Middle + multiplier * ATR`. Lower: `Middle - multiplier * ATR` |
| **Donchian Channels** | `window` | Upper: `Highest(High, window)`. Middle: `(Upper + Lower) / 2`. Lower: `Lowest(Low, window)` |

### ATR Parameters
| Name | Default | Range |
|---|---|---|
| window | 14 | 2–50 |

### Bollinger Bands Parameters
| Name | Default | Range |
|---|---|---|
| window | 20 | 2–50 |
| num_std | 2 | 1–4 |

### Keltner Channels Parameters
| Name | Default | Range |
|---|---|---|
| window | 20 | 2–50 |
| multiplier | 2 | 1–4 |

### Donchian Channels Parameters
| Name | Default | Range |
|---|---|---|
| window | 20 | 2–200 |

---

## 4. Volume Indicators

Indicators that incorporate volume data.

| Indicator | Parameters | Formula / Description |
|---|---|---|
| **OBV** | — | On-Balance Volume: cumulative sum of `±Volume` based on close direction |
| **CMF** | `window` | Chaikin Money Flow: `sum(MFV, n) / sum(Volume, n)`, where `MFV = Volume * (2*Close - High - Low) / (High - Low)` |
| **ADL** | — | Accumulation/Distribution Line: cumulative `MFV` |

### OBV Parameters
| Name | Default | Range |
|---|---|---|
| — | — | No parameters |

### CMF Parameters
| Name | Default | Range |
|---|---|---|
| window | 20 | 2–50 |

### ADL Parameters
| Name | Default | Range |
|---|---|---|
| — | — | No parameters |

---

## 5. Trend Strength Indicators

Indicators that measure the strength of a trend (not direction).

| Indicator | Parameters | Formula / Description |
|---|---|---|
| **ADX** | `window` | Average Directional Index. Computed from `+DI` and `-DI` using smoothed directional movement |
| **Aroon** | `window` | Aroon Up: `100 * (window - days_since_high) / window`. Aroon Down: `100 * (window - days_since_low) / window` |
| **Vortex** | `window` | True Range normalized. `VI+ = sum(VM+, n) / sum(TR, n)`. `VI- = sum(VM-, n) / sum(TR, n)` |

### ADX Parameters
| Name | Default | Range |
|---|---|---|
| window | 14 | 2–50 |

### Aroon Parameters
| Name | Default | Range |
|---|---|---|
| window | 25 | 2–50 |

### Vortex Parameters
| Name | Default | Range |
|---|---|---|
| window | 14 | 2–50 |

---

## 6. Money-Flow Indicators

Indicators that combine price and volume to estimate money-flow direction and accumulation/distribution.

| Indicator | Parameters | Formula / Description |
|---|---|---|
| **MFI** | `window` | Money Flow Index. `MF = TypicalPrice * Volume`, `TP=(H+L+C)/3`. `MFI = 100 - 100/(1 + pos_MF/neg_MF)` over `window`; overbought>80, oversold<20 |
| **VWAP** | — | Volume-Weighted Average Price: `cumsum(TP * Volume) / cumsum(Volume)` |
| **VWMA** | `window` | Volume-Weighted Moving Average: `sum(Close * Volume, n) / sum(Volume, n)` |
| **Chaikin A/D Oscillator** | `fast, slow` | `EMA(fast) - EMA(slow)` of the Chaikin A/D Line; A/D Line = cumulative Money Flow Multiplier × Volume |
| **Relative Volume (RVOL)** | `window` | `Volume / SMA(Volume, window)`; liquidity/spike measure |
| **Price Volume Trend (PVT)** | — | Cumulative `Close_pct_change * Volume` |
| **Volume Oscillator** | `fast, slow` | `100 * (EMA(fast) - EMA(slow)) / EMA(slow)` |
| **Vol-axis ROC** | `window` | Volume rate of change `pct_change(Volume, window) * 100` |
| **Elder Force Index** | `window` | `EMA(window, Close.diff() * Volume)` |
| **Ease-of-Movement** | `window`, smoothing | `(mid.diff()) / box_ratio` smoothed by EMA; `box_ratio = Volume / (High - Low)` |
| **Net Volume Index (NVI)** | — | Values accumulate only on days with falling volume |
| **Positive Volume Index (PVI)** | — | Values accumulate only on days with rising volume |

### MFI Parameters
| Name | Default | Range |
|---|---|---|
| window | 14 | 2–50 |

### Chaikin A/D Oscillator Parameters
| Name | Default | Range |
|---|---|---|
| fast | 3 | 1–20 |
| slow | 10 | 1–50 |

### RVOL Parameters
| Name | Default | Range |
|---|---|---|
| window | 20 | 2–50 |

### VWMA Parameters
| Name | Default | Range |
|---|---|---|
| window | 20 | 2–200 |

### Volume Oscillator / Volume ROC / Ease-of-Movement / Force Index Parameters
| Name | Default | Range |
|---|---|---|
| window | 12–13 | 2–50 |

---

## 7. Trend-Path Indicators

Indicators that trace trend continuation, swings, and trailing reversal levels.

| Indicator | Parameters | Formula / Description |
|---|---|---|
| **SuperTrend** | `multiplier, atr_window` | Upper/Lower bands based on ATR; flips trend when Close crosses band. Band = `HL2 ± multiplier * ATR` |
| **Ichimoku** | `tenkan, kijun, senkou_b` | Tenkan=(HL-Pivot9), Kijun=(HL-Pivot26), Senkou A/B shifted +26, Chikou shifted −26; cloud = 0.5×(A+B) |
| **Parabolic SAR** | `step, max_step` | Reversal after `step`, accelerating up to `max_step` |
| **ZigZag** | `threshold` | Pivot series replacing moves below `threshold` with straight lines |
| **Fractals** | `bars` | 5-bar local extrema (Williams fractal) marking potential reversal points |
| **CMO** | `window` | Chande Momentum Oscillator: `100 * (gain − loss) / (gain + loss)` |
| **TRIX** | `window` | `pct_change` of triple exponential moving average of Close |

### SuperTrend Parameters
| Name | Default | Range |
|---|---|---|
| multiplier | 3 | 1–8 |
| atr_window | 10 | 2–30 |

### Ichimoku Parameters
| Name | Default | Range |
|---|---|---|
| tenkan | 9 | 2–50 |
| kijun | 26 | 2–100 |
| senkou_b | 52 | 2–200 |

### Parabolic SAR Parameters
| Name | Default | Range |
|---|---|---|
| step | 0.02 | 0.001–0.2 |
| max_step | 0.2 | 0.01–1.0 |

### ZigZag Parameters
| Name | Default | Range |
|---|---|---|
| threshold | 0.05 | 0.01–0.5 |

### Fractals Parameters
| Name | Default | Range |
|---|---|---|
| bars | 5 | 2–20 |

### CMO / TRIX Parameters
| Name | Default | Range |
|---|---|---|
| window | 14 / 15 | 2–50 |

---

## Signal Generation (`technical/signals.py`)

All indicators can generate signals via:

### Crossover Signal
`crossover(series1, series2)` — 1 when series1 crosses above series2, -1 when crosses below, 0 otherwise.

### Threshold Signal
`threshold(series, lower, upper)` — 1 when series crosses above upper, -1 when crosses below lower, 0 otherwise.

### Signal Table Columns
| Column | Description |
|---|---|
| Date | Timestamp |
| Indicator Value | Current value |
| Signal | Buy (+1), Sell (-1), Neutral (0) |
| Direction | Bullish, Bearish, Neutral |

---

## Indicator Count

**Curated presets:** Window values of [5, 9, 10, 12, 14, 20, 21, 26, 50, 100, 200] applied across the full indicator set = 300+ meaningful combinations.
