"""Technical indicators derived exclusively from OHLCV data.

Module layout (see docs/ARCHITECTURE.md):
- trend.py      sma, ema, wma, hma, vwma + trend-path (supertrend, ichimoku, psar, zigzag, fractals)
- momentum.py   rsi, macd, roc, stochastic, williams_r + cmo, trix
- volatility.py atr, bollinger_bands, keltner, donchian
- volume.py     obv, cmf, adl + money-flow (mfi, vwap, chaikin_osc, rvol, pvt, nvi, pvi,
                volume_osc, force_index, emv)
- strength.py   adx, aroon, vortex
- signals.py    crossover, threshold + signal table

All indicators return pd.Series aligned to the input index or a pd.DataFrame
for multi-line outputs (e.g. Bollinger bands). Input is a DataFrame with
columns Open/High/Low/Close/Volume.
"""

from . import trend, momentum, volatility, volume, strength, signals

__all__ = ["trend", "momentum", "volatility", "volume", "strength", "signals"]