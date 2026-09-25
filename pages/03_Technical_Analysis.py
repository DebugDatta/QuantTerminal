"""Page 12: Technical Analysis Terminal — Interactive Multi-Indicator Quantitative Dashboard."""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from technical import trend, momentum, volatility, volume, strength, signals
from utils.helper import (
    inject_custom_theme,
    load_data,
    _fmt_num,
    _fmt_money,
    CURRENCY_SYMBOLS,
)
from utils.sidebar import render_sidebar

PRESETS = {
    "Standard": {
        "overlays": ["SMA", "Bollinger_Bands"],
        "subpanels": ["RSI", "MACD"],
        "title": "Standard Balanced Technical Analysis",
        "description": "Balanced baseline configuration combining moving averages, volatility bands, momentum oscillators, and trend divergence.",
        "regime": "All-Weather / Dual-Regime",
        "timeframe": "Daily / 4-Hour (Holding: 5-20 Days)",
        "win_rate": "55% - 65% (Typical R:R 1:1.8)",
        "philosophy": "Identifies macro direction using 20/50 period averages while tracking overbought/oversold extremities and momentum shifts via RSI and MACD. Avoids cognitive overload by pairing one primary trend anchor with volatility boundaries and dual-momentum validation.",
        "entry_rules": [
            "Trend Alignment: Price must be trading above the 20-period SMA for Longs (or below for Shorts).",
            "Momentum Confirmation: RSI(14) is above 50 and rising out of neutral zone without being overbought (>70).",
            "Convergence Trigger: MACD line crosses above the 9-period Signal line with an expanding positive histogram."
        ],
        "exit_rules": [
            "Target 1: Upper Bollinger Band touch or test of key Resistance 1 pivot level (take 50% profits).",
            "Trailing Stop: Trailing stop triggered when price closes below the 20-period SMA midline.",
            "Signal Invalidation: Bearish MACD signal line crossover or RSI dropping sharply below 45."
        ],
        "entry_exit": "Long entries favored when price bounces from lower Bollinger Band or breaks above 20 SMA with RSI > 50 and positive MACD histogram expansion. Exits or shorts on upper band rejection or bearish MACD cross.",
        "risk_management": "Set hard initial stop 1.5x ATR below the entry swing low. Never risk more than 1.0% of portfolio capital on a single position.",
        "pitfalls": "Whipsaws occur in low-volatility sideways compression. Always verify Bollinger Bandwidth is expanding (> 10%) before taking directional breakouts."
    },
    "Swing Trading": {
        "overlays": ["EMA", "SuperTrend"],
        "subpanels": ["RSI", "Stochastic"],
        "title": "Multi-Day Swing Momentum Strategy",
        "description": "Optimized for capturing 3-to-15 day price waves around established intermediate trends.",
        "regime": "Trending Regimes with Pullbacks",
        "timeframe": "Daily / 4-Hour (Holding: 3-15 Days)",
        "win_rate": "52% - 62% (Typical R:R 1:2.5)",
        "philosophy": "Captures institutional 'buying the dip' behavior within established directional trends. By anchoring bias to the adaptive ATR SuperTrend line and using fast EMA pullbacks, traders identify high-probability entry pockets timed with Stochastic cycle hooks.",
        "entry_rules": [
            "Regime Filter: SuperTrend must be Bullish (Green trailing stop active below price).",
            "Pullback Trigger: Price retraces towards the 20-period EMA without violating the SuperTrend stop.",
            "Oscillator Hook: Stochastic %K crosses above %D after dipping below 30 (or exiting oversold < 20), confirmed by RSI > 45."
        ],
        "exit_rules": [
            "Take Profit: 2.0x to 3.0x ATR profit target or when Stochastic %K exceeds 80 and crosses below %D.",
            "Trailing Stop: Dynamically trail stops directly behind the active SuperTrend line.",
            "Emergency Stop: Exit immediately if price produces a daily close below the SuperTrend line."
        ],
        "entry_exit": "Enter Long when SuperTrend is bullish (green), price tests EMA from above, and Stochastic exits oversold (< 20). Take profit near prior swing highs or when RSI reaches > 70.",
        "risk_management": "Hard stop anchored directly to the SuperTrend trailing stop line. Position size = (Account Equity * 1.0%) / (Entry Price - SuperTrend Stop).",
        "pitfalls": "Avoid entering when price is extended > 3x ATR away from the EMA. Wait patiently for the pullback retest."
    },
    "Trend Following": {
        "overlays": ["SMA", "EMA", "SuperTrend", "Parabolic_SAR"],
        "subpanels": ["MACD", "ADX", "Aroon"],
        "title": "Institutional Trend-Following System",
        "description": "Captures extended multi-week or multi-month trends while filtering out sideways choppy regimes.",
        "regime": "Strong Structural Macro Trends",
        "timeframe": "Daily / Weekly (Holding: Weeks to Months)",
        "win_rate": "42% - 50% (High R:R 1:3.5 to 1:6.0)",
        "philosophy": "Based on the premia of trend persistence (momentum factor). Institutional capital cannot enter quickly without moving price; thus, strong trends persist. Uses ADX and Aroon to filter out rangebound noise and capture large macro moves.",
        "entry_rules": [
            "Trend Strength Gate: ADX(14) must be strictly > 25 and sloping upward, confirming an active directional market.",
            "Directional Dominance: Aroon Up > 70 while Aroon Down < 30; +DI strictly above -DI.",
            "Trend Alignment: EMA(20) > SMA(50), SuperTrend is green, and Parabolic SAR dots are plotted below price."
        ],
        "exit_rules": [
            "Trailing Exit: Trail stop along Parabolic SAR dots for fast moves, or SuperTrend line for macro waves.",
            "Signal Flip: Exit full position when Aroon Down crosses above Aroon Up or ADX rolls over below 20.",
            "Trend Breakdown: Daily close below the 50-day SMA midline."
        ],
        "entry_exit": "Enter Long when fast EMA is above slow SMA, SuperTrend is positive, MACD line is above zero, and ADX confirms strength (> 25). Exit on MACD signal break or SuperTrend flip.",
        "risk_management": "Initial stop set at 2x ATR or the Parabolic SAR level. Scale into winners (pyramiding) only after 1R profit has been banked.",
        "pitfalls": "High drawdown in choppy markets. Strictly enforce the ADX > 25 filter to avoid being chopped up in sideways consolidation."
    },
    "Breakout": {
        "overlays": ["Bollinger_Bands", "Donchian_Channels", "VWAP"],
        "subpanels": ["RVOL", "ADX", "Bollinger_Bandwidth"],
        "title": "Volatility Squeeze & Volume Breakout",
        "description": "Capitalizes on sharp explosive moves emerging from periods of compressed volatility.",
        "regime": "Pre-Breakout Volatility Squeeze",
        "timeframe": "Intraday / Daily (Holding: 1-5 Days)",
        "win_rate": "45% - 55% (Typical R:R 1:3.0)",
        "philosophy": "Volatility is mean-reverting and cyclical: periods of extreme low volatility (compression) are inevitably followed by explosive expansion. High RVOL confirms institutional accumulation or distribution at the breakout point.",
        "entry_rules": [
            "Squeeze Condition: Bollinger Bandwidth reaches multi-week lows (< 10% or lowest 15% quantile).",
            "Price Trigger: Daily candle pierces and closes outside the 20-day Donchian Channel or Upper Bollinger Band.",
            "Volume Confirmation: Relative Volume (RVOL) must be >= 1.5x (validating heavy institutional commitment)."
        ],
        "exit_rules": [
            "Initial Target: 2x the width of the pre-breakout consolidation range.",
            "Profit Protection: Move stop to breakeven once price moves 1x ATR in your favor.",
            "Failure Invalidation: Exit immediately if price falls back inside the pre-breakout Bollinger Band midline (false breakout)."
        ],
        "entry_exit": "Enter Long when price pierces the Upper Bollinger Band accompanied by RVOL > 1.5x and rising ADX. Avoid entries if RVOL is below 1.0 (false breakout risk).",
        "risk_management": "Stop-loss placed immediately below the breakout candle's low or the 20-period VWAP support line.",
        "pitfalls": "Low-volume breakouts (RVOL < 1.0) fail over 70% of the time. Never chase without volume confirmation."
    },
    "Mean Reversion": {
        "overlays": ["Bollinger_Bands", "Keltner_Channels"],
        "subpanels": ["RSI", "Stochastic", "Williams_%R"],
        "title": "Statistical Mean Reversion System",
        "description": "Exploits temporary price overextensions in range-bound or neutral market regimes.",
        "regime": "Sideways / Range-Bound / Low ADX (< 20)",
        "timeframe": "Daily / 1-Hour (Holding: 1-4 Days)",
        "win_rate": "65% - 75% (Typical R:R 1:1.2 to 1:1.5)",
        "philosophy": "Prices in non-trending regimes exhibit Gaussian distribution characteristics, returning to their statistical rolling mean after piercing 2 standard deviations. Overextensions create liquidity vacuums that pull price back to equilibrium.",
        "entry_rules": [
            "Regime Filter: ADX must be < 20 (confirming market is non-trending and rangebound).",
            "Statistical Boundary: Price pierces the Lower Bollinger Band (2-std) or Lower Keltner Channel.",
            "Confluent Extremity: RSI < 30, Stochastic < 20, and Williams %R < -80 with bullish reversal candle."
        ],
        "exit_rules": [
            "Primary Target: The 20-period Moving Average (Bollinger Band midline / mean).",
            "Extended Target: Opposite Bollinger Band (Upper Band) if momentum accelerates.",
            "Time Stop: If price does not revert within 5 bars, close position to prevent trend transition risk."
        ],
        "entry_exit": "Buy when price touches/pierces Lower Band with RSI < 30, Stochastic < 20, and Williams %R < -80 curling upward. Target is the 20-period Middle Band (mean).",
        "risk_management": "Hard stop set at 1.0x ATR below the extreme swing low. If a trend develops, exit without hesitation.",
        "pitfalls": "Attempting mean reversion during strong macroeconomic trends or earnings announcements causes catastrophic 'falling knife' losses."
    },
    "Momentum": {
        "overlays": ["EMA", "VWMA", "HMA"],
        "subpanels": ["RSI", "ROC", "MACD", "TRIX"],
        "title": "Aggressive Price Momentum",
        "description": "Focuses on high-velocity acceleration for fast-moving breakout runners.",
        "regime": "High-Velocity Acceleration & Markup",
        "timeframe": "Daily / Intraday (Holding: 2-7 Days)",
        "win_rate": "50% - 60% (Typical R:R 1:2.0)",
        "philosophy": "Assets demonstrating positive relative velocity and acceleration continue to attract algorithmic flow and momentum liquidity. By tracking Rate of Change (ROC) and triple-smoothed TRIX, traders ride the steepest phase of markup.",
        "entry_rules": [
            "Velocity Threshold: Rate of Change (ROC 12) is positive (> 0) and expanding aggressively.",
            "Momentum Cross: MACD histogram is expanding upward while TRIX oscillator is positive and rising.",
            "Moving Average Alignment: Price is accelerating above Hull Moving Average (HMA 20) and EMA(20)."
        ],
        "exit_rules": [
            "Momentum Stall: ROC turns negative or MACD histogram prints a lower high.",
            "Fast Trailing Stop: Close below the 9-period EMA or Hull Moving Average (HMA).",
            "Exhaustion Target: RSI touching 80+ with bearish candlestick divergence."
        ],
        "entry_exit": "Buy when RSI breaks above 60 with MACD histogram surging positively, ROC > 0, and price accelerating above HMA/EMA. Exit when momentum stalls.",
        "risk_management": "Tight trailing stop along short-term EMA/HMA. Risk capped at 1.0% of portfolio.",
        "pitfalls": "Late entries into parabolic moves. If RSI is already > 75 at entry, the risk/reward is unfavorable."
    },
    "Volatility": {
        "overlays": ["Bollinger_Bands", "Keltner_Channels"],
        "subpanels": ["ATR", "Bollinger_Bandwidth", "Vortex"],
        "title": "Volatility Dispersion & Channel Regime",
        "description": "Monitors ATR expansion, channel envelopes, and volatility clustering.",
        "regime": "Volatility Regime Shifts (Compression to Expansion)",
        "timeframe": "Daily (Strategic allocation & sizing)",
        "win_rate": "55% - 65% (Typical R:R 1:2.2)",
        "philosophy": "Volatility clusters in financial markets: high volatility days follow high volatility days, and low volatility days cluster together. Dynamic position sizing pegged inversely to ATR protects capital during turbulent market regimes.",
        "entry_rules": [
            "Channel Envelope: Price expands beyond Keltner Channels into outer Bollinger Bands.",
            "Volatility Ignition: ATR(14) begins rising from historical percentile troughs (< 20th percentile).",
            "Directional Vortex: Vortex VI+ cleanly crosses above VI- by at least 0.20 margin."
        ],
        "exit_rules": [
            "Peak Expansion: ATR reaches 90th percentile or Bollinger Bandwidth prints parabolic spike.",
            "Vortex Convergence: VI+ and VI- lines converge back towards 1.0 parity.",
            "Channel Re-entry: Price pulls back inside the Keltner Channel envelope."
        ],
        "entry_exit": "Scale into positions when ATR is low and expanding. Reduce leverage or exit when ATR hits extreme percentiles.",
        "risk_management": "Inverse-volatility sizing: Dollar Risk / (ATR * Multiplier). Position size is automatically cut in half if ATR doubles.",
        "pitfalls": "Failing to adjust position sizing when ATR expands leads to unexpected account variance."
    },
    "Volume Analysis": {
        "overlays": ["VWAP", "VWMA"],
        "subpanels": ["RVOL", "OBV", "CMF", "MFI", "Volume"],
        "title": "Institutional Order Flow & Volume Profile",
        "description": "Decodes smart money accumulation and distribution via VWAP, OBV, CMF, and RVOL.",
        "regime": "Institutional Accumulation / Distribution",
        "timeframe": "Daily / 1-Hour (Order Flow Confirmation)",
        "win_rate": "58% - 68% (Typical R:R 1:2.0)",
        "philosophy": "Volume precedes price. Large institutional algorithms (TWAP/VWAP engines) leave footprints in On-Balance Volume (OBV), Chaikin Money Flow (CMF), and Money Flow Index (MFI) before significant markups occur.",
        "entry_rules": [
            "Benchmark Support: Price holds firmly above the Volume-Weighted Average Price (VWAP).",
            "Smart Money Inflow: Chaikin Money Flow (CMF 20) is positive (> +0.05), confirming net institutional accumulation.",
            "Volume Lead: OBV prints new multi-week highs ahead of price, and RVOL > 1.2x on up candles."
        ],
        "exit_rules": [
            "Flow Reversal: CMF drops below 0 with heavy distribution volume.",
            "VWAP Failure: Decisive breakdown and candle close below VWAP support.",
            "Climax Target: Extreme high volume (RVOL > 3x) at key resistance without price follow-through (exhaustion)."
        ],
        "entry_exit": "Buy when price holds above VWAP with OBV making new relative highs, CMF > 0, and RVOL confirming institutional buying. Sell if price breaks below VWAP with heavy volume.",
        "risk_management": "Stop-loss anchored 0.5% below intraday/multi-day VWAP benchmark.",
        "pitfalls": "False accumulation readings on low volume holiday sessions. Always ensure absolute volume exceeds 20-day SMA."
    },
    "Custom": {
        "overlays": ["SMA", "Bollinger_Bands"],
        "subpanels": ["RSI", "MACD"],
        "title": "Custom Quantitative Workspace",
        "description": "Fully customizable multi-indicator configuration with up to 46 quantitative indicators.",
        "regime": "User Configurable",
        "timeframe": "Configurable (Any Interval)",
        "win_rate": "Depends on Custom Configuration",
        "philosophy": "Gives quantitative researchers and professional traders complete freedom to compose bespoke indicator combinations across 18 Price Axis Overlays and 28 Analytical Subpanels.",
        "entry_rules": [
            "Establish 4-Pillar Confluence: Check Trend + Momentum + Volatility + Volume.",
            "Verify at least 3 indicators agree in direction (Directional consensus >= 75%).",
            "Confirm trade direction aligns with primary market regime badges."
        ],
        "exit_rules": [
            "Pre-calculated Support & Resistance pivot targets.",
            "Trailing stop keyed to ATR or SuperTrend line.",
            "Key oscillator threshold crossing (e.g. RSI > 70 or < 30)."
        ],
        "entry_exit": "Custom indicator confluence.",
        "risk_management": "Fixed fractional risk (1-2% max capital per idea). Always check Pivot Support/Resistance levels before sizing.",
        "pitfalls": "Indicator multicollinearity: avoid using 3 indicators that measure the exact same thing without a volume or trend filter."
    },
}

ALL_OVERLAYS = [
    "SMA", "EMA", "WMA", "HMA", "VWMA", "DEMA", "TEMA",
    "Bollinger_Bands", "Keltner_Channels", "Donchian_Channels", "Price_Envelopes",
    "SuperTrend", "Parabolic_SAR", "Ichimoku_Cloud", "VWAP",
    "ZigZag", "Fractals", "Pivot_Points"
]

ALL_SUBPANELS = [
    "RSI", "MACD", "Stochastic", "Williams_%R", "ROC", "CMO", "TRIX", "CCI",
    "ADX", "Aroon", "Vortex", "ATR", "Bollinger_Bandwidth", "Bollinger_%B", "Historical_Volatility",
    "RVOL", "OBV", "CMF", "MFI", "ADL", "Chaikin_Oscillator", "PVT",
    "Ease_of_Movement", "Force_Index", "Volume_Oscillator", "PVI", "NVI", "Volume"
]


def _calculate_supertrend_line(
    high: pd.Series, low: pd.Series, close: pd.Series, multiplier: float = 3.0, atr_window: int = 10
) -> tuple[pd.Series, pd.Series]:
    """Calculate adaptive ATR SuperTrend trailing stop line and trend direction (+1/-1)."""
    hl2 = (high + low) / 2.0
    tr0 = high - low
    tr1 = (high - close.shift(1)).abs()
    tr2 = (low - close.shift(1)).abs()
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_window, adjust=False).mean()

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    st_line = pd.Series(index=close.index, dtype=float)
    st_dir = pd.Series(index=close.index, dtype=float)

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    curr_dir = 1
    for i in range(1, len(close)):
        if upper_band.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if lower_band.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if curr_dir == 1:
            if close.iloc[i] < final_lower.iloc[i]:
                curr_dir = -1
                st_line.iloc[i] = final_upper.iloc[i]
            else:
                st_line.iloc[i] = final_lower.iloc[i]
        else:
            if close.iloc[i] > final_upper.iloc[i]:
                curr_dir = 1
                st_line.iloc[i] = final_lower.iloc[i]
            else:
                st_line.iloc[i] = final_upper.iloc[i]
        st_dir.iloc[i] = curr_dir

    return st_line, st_dir


def _calculate_pivot_points(df: pd.DataFrame) -> dict:
    """Calculate Classic Pivot Points from period high, low, close."""
    high = float(df["High"].max())
    low = float(df["Low"].min())
    close = float(df["Close"].iloc[-1])

    pivot = (high + low + close) / 3.0
    r1 = 2.0 * pivot - low
    s1 = 2.0 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)

    return {
        "Pivot": pivot,
        "Resistance 2": r2,
        "Resistance 1": r1,
        "Current Close": close,
        "Support 1": s1,
        "Support 2": s2,
    }


def _compute_regime_summary(
    df: pd.DataFrame,
    sma20: pd.Series = None,
    bb: pd.DataFrame = None,
    rsi_s: pd.Series = None,
    macd_df: pd.DataFrame = None,
    adx_df: pd.DataFrame = None,
    rvol_s: pd.Series = None,
    df_chart: pd.DataFrame = None,
) -> dict:
    """Derive institutional technical regime status badges using multi-factor consensus."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"] if "Volume" in df.columns else pd.Series(1.0, index=df.index)

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else last_close
    day_open = float(df["Open"].iloc[-1]) if "Open" in df.columns else prev_close
    pct_change = ((last_close - prev_close) / prev_close) * 100 if prev_close != 0 else 0.0

    # Indicators on buffered regime dataset
    ema20 = trend.ema(close, 20)
    sma50 = trend.sma(close, 50)
    sma200 = trend.sma(close, 200)

    if bb is None or bb.empty:
        bb = volatility.bollinger_bands(close, window=20, num_std=2.0)
    if rsi_s is None or len(rsi_s.dropna()) == 0:
        rsi_s = momentum.rsi(close, window=14)
    if macd_df is None or macd_df.empty:
        macd_df = momentum.macd(close, fast=12, slow=26, signal=9)
    if adx_df is None or adx_df.empty:
        adx_df = strength.adx(high, low, close, window=14)
    if rvol_s is None or len(rvol_s.dropna()) == 0:
        rvol_s = volume.rvol(vol, window=20)

    atr_s = volatility.atr(high, low, close, window=14)

    c_ema20 = float(ema20.dropna().iloc[-1]) if len(ema20.dropna()) > 0 else last_close
    c_sma50 = float(sma50.dropna().iloc[-1]) if len(sma50.dropna()) > 0 else c_ema20
    c_sma200 = float(sma200.dropna().iloc[-1]) if len(sma200.dropna()) > 0 else c_sma50

    # EMA20 slope over the last 3-4 bars
    ema_slope = (
        (c_ema20 - float(ema20.dropna().iloc[-4])) / float(ema20.dropna().iloc[-4]) * 100
        if len(ema20.dropna()) >= 4 and float(ema20.dropna().iloc[-4]) != 0
        else 0.0
    )

    # -------------------------------------------------------------------------
    # 1. Multi-Factor Trend Evaluation
    # -------------------------------------------------------------------------
    trend_score = 0.0
    if last_close >= c_ema20:
        trend_score += 1.0
    else:
        trend_score -= 1.0

    if last_close >= c_sma50:
        trend_score += 0.5
    else:
        trend_score -= 0.5

    if ema_slope > 0.15:
        trend_score += 0.5
    elif ema_slope < -0.15:
        trend_score -= 0.5

    if len(close.dropna()) >= 200:
        if last_close >= c_sma200:
            trend_score += 0.5
        else:
            trend_score -= 0.5

    if trend_score >= 1.5:
        trend_state = "STRONG BULLISH"
        trend_color, trend_bg, trend_border = "#10B981", "rgba(16, 185, 129, 0.15)", "rgba(16, 185, 129, 0.35)"
    elif trend_score >= 0.5:
        trend_state = "BULLISH"
        trend_color, trend_bg, trend_border = "#10B981", "rgba(16, 185, 129, 0.15)", "rgba(16, 185, 129, 0.35)"
    elif trend_score == 0.0:
        if last_close >= c_sma50 and ema_slope >= 0:
            trend_state = "BULLISH PULLBACK"
            trend_color, trend_bg, trend_border = "#38BDF8", "rgba(56, 189, 248, 0.15)", "rgba(56, 189, 248, 0.35)"
        elif last_close < c_sma50 and ema_slope <= 0:
            trend_state = "BEARISH PULLBACK"
            trend_color, trend_bg, trend_border = "#FB923C", "rgba(251, 146, 60, 0.15)", "rgba(251, 146, 60, 0.35)"
        else:
            trend_state = "CONSOLIDATING"
            trend_color, trend_bg, trend_border = "#94A3B8", "rgba(148, 163, 184, 0.15)", "rgba(148, 163, 184, 0.35)"
    elif trend_score > -1.5:
        trend_state = "BEARISH"
        trend_color, trend_bg, trend_border = "#EF4444", "rgba(239, 68, 68, 0.15)", "rgba(239, 68, 68, 0.35)"
    else:
        trend_state = "STRONG BEARISH"
        trend_color, trend_bg, trend_border = "#F87171", "rgba(239, 68, 68, 0.22)", "rgba(248, 113, 113, 0.45)"

    # -------------------------------------------------------------------------
    # 2. Multi-Factor Momentum Evaluation
    # -------------------------------------------------------------------------
    last_rsi = float(rsi_s.dropna().iloc[-1]) if len(rsi_s.dropna()) > 0 else 50.0
    rsi_delta = last_rsi - float(rsi_s.dropna().iloc[-3]) if len(rsi_s.dropna()) >= 3 else 0.0

    last_hist = float(macd_df["histogram"].dropna().iloc[-1]) if len(macd_df["histogram"].dropna()) > 0 else 0.0
    prev_hist = float(macd_df["histogram"].dropna().iloc[-2]) if len(macd_df["histogram"].dropna()) > 1 else last_hist
    hist_expansion = last_hist - prev_hist

    mom_score = 0.0
    if last_rsi >= 60:
        mom_score += 1.5
    elif last_rsi >= 52:
        mom_score += 0.8
    elif last_rsi <= 40:
        mom_score -= 1.5
    elif last_rsi <= 48:
        mom_score -= 0.8

    if rsi_delta >= 2.0:
        mom_score += 0.5
    elif rsi_delta <= -2.0:
        mom_score -= 0.5

    if last_hist > 0:
        mom_score += 0.5
        if hist_expansion > 0:
            mom_score += 0.5
    else:
        mom_score -= 0.5
        if hist_expansion < 0:
            mom_score -= 0.5

    if last_rsi < 32 and (hist_expansion > 0 or rsi_delta > 0):
        momentum_state = "OVERSOLD BOUNCE"
        mom_color, mom_bg, mom_border = "#38BDF8", "rgba(56, 189, 248, 0.15)", "rgba(56, 189, 248, 0.35)"
    elif last_rsi < 32:
        momentum_state = "OVERSOLD"
        mom_color, mom_bg, mom_border = "#F59E0B", "rgba(245, 158, 11, 0.15)", "rgba(245, 158, 11, 0.35)"
    elif mom_score >= 2.0:
        momentum_state = "STRONG POSITIVE"
        mom_color, mom_bg, mom_border = "#10B981", "rgba(16, 185, 129, 0.15)", "rgba(16, 185, 129, 0.35)"
    elif mom_score >= 0.6:
        momentum_state = "POSITIVE"
        mom_color, mom_bg, mom_border = "#10B981", "rgba(16, 185, 129, 0.15)", "rgba(16, 185, 129, 0.35)"
    elif mom_score <= -2.0:
        momentum_state = "STRONG NEGATIVE"
        mom_color, mom_bg, mom_border = "#F87171", "rgba(239, 68, 68, 0.22)", "rgba(248, 113, 113, 0.45)"
    elif mom_score <= -0.6:
        momentum_state = "NEGATIVE"
        mom_color, mom_bg, mom_border = "#EF4444", "rgba(239, 68, 68, 0.15)", "rgba(239, 68, 68, 0.35)"
    else:
        momentum_state = "NEUTRAL / STALLED"
        mom_color, mom_bg, mom_border = "#94A3B8", "rgba(148, 163, 184, 0.15)", "rgba(148, 163, 184, 0.35)"

    # -------------------------------------------------------------------------
    # 3. Directional Strength Evaluation (ADX + Directional Quality)
    # -------------------------------------------------------------------------
    last_adx = float(adx_df["adx"].dropna().iloc[-1]) if len(adx_df["adx"].dropna()) > 0 else 20.0
    last_pdi = float(adx_df["plus_di"].dropna().iloc[-1]) if len(adx_df["plus_di"].dropna()) > 0 else 25.0
    last_mdi = float(adx_df["minus_di"].dropna().iloc[-1]) if len(adx_df["minus_di"].dropna()) > 0 else 25.0

    if last_adx >= 25:
        if last_pdi > last_mdi + 5:
            strength_state = "STRONG TREND (+)"
            str_color, str_bg, str_border = "#10B981", "rgba(16, 185, 129, 0.15)", "rgba(16, 185, 129, 0.35)"
        elif last_mdi > last_pdi + 5:
            strength_state = "STRONG TREND (-)"
            str_color, str_bg, str_border = "#EF4444", "rgba(239, 68, 68, 0.15)", "rgba(239, 68, 68, 0.35)"
        else:
            strength_state = "STRONG TREND"
            str_color, str_bg, str_border = "#38BDF8", "rgba(56, 189, 248, 0.15)", "rgba(56, 189, 248, 0.35)"
    elif last_adx >= 20:
        strength_state = "MODERATE TREND"
        str_color, str_bg, str_border = "#38BDF8", "rgba(56, 189, 248, 0.15)", "rgba(56, 189, 248, 0.35)"
    else:
        strength_state = "RANGE-BOUND"
        str_color, str_bg, str_border = "#94A3B8", "rgba(148, 163, 184, 0.15)", "rgba(148, 163, 184, 0.35)"

    # -------------------------------------------------------------------------
    # 4. Volatility Evaluation (ATR% + Bandwidth Quantile)
    # -------------------------------------------------------------------------
    last_atr = float(atr_s.dropna().iloc[-1]) if len(atr_s.dropna()) > 0 else (last_close * 0.015)
    atr_pct = (last_atr / last_close) * 100 if last_close > 0 else 1.5

    if not bb.empty and "upper" in bb.columns and "lower" in bb.columns and "middle" in bb.columns:
        bw = (bb["upper"] - bb["lower"]) / bb["middle"].replace(0, np.nan)
        bw_clean = bw.dropna()
        last_bw = float(bw_clean.iloc[-1]) if len(bw_clean) > 0 else 0.08
        bw_quantile = float((bw_clean <= last_bw).mean()) if len(bw_clean) >= 20 else 0.5
    else:
        bw_quantile = 0.5

    if bw_quantile <= 0.15 or atr_pct < 1.0:
        vol_state = "COMPRESSED / SQUEEZE"
        vol_color, vol_bg, vol_border = "#38BDF8", "rgba(56, 189, 248, 0.15)", "rgba(56, 189, 248, 0.35)"
    elif bw_quantile >= 0.85 or atr_pct > 3.5:
        vol_state = "HIGH / EXPANDING"
        vol_color, vol_bg, vol_border = "#EF4444", "rgba(239, 68, 68, 0.15)", "rgba(239, 68, 68, 0.35)"
    else:
        vol_state = "NORMAL"
        vol_color, vol_bg, vol_border = "#10B981", "rgba(16, 185, 129, 0.15)", "rgba(16, 185, 129, 0.35)"

    # -------------------------------------------------------------------------
    # 5. Volume Evaluation (Relative Volume + Accumulation / Distribution)
    # -------------------------------------------------------------------------
    last_rvol = float(rvol_s.dropna().iloc[-1]) if len(rvol_s.dropna()) > 0 else 1.0
    vol_is_up = (last_close >= day_open) or (pct_change >= 0)

    if last_rvol >= 1.25:
        if vol_is_up:
            volm_state = "ACCUMULATION"
            volm_color, volm_bg, volm_border = "#10B981", "rgba(16, 185, 129, 0.15)", "rgba(16, 185, 129, 0.35)"
            vol_score = 1.0
        else:
            volm_state = "DISTRIBUTION"
            volm_color, volm_bg, volm_border = "#EF4444", "rgba(239, 68, 68, 0.15)", "rgba(239, 68, 68, 0.35)"
            vol_score = -1.0
    elif last_rvol < 0.75:
        volm_state = "LOW / DRYING UP"
        volm_color, volm_bg, volm_border = "#F59E0B", "rgba(245, 158, 11, 0.15)", "rgba(245, 158, 11, 0.35)"
        vol_score = 0.0
    else:
        volm_state = "AVERAGE VOL"
        volm_color, volm_bg, volm_border = "#94A3B8", "rgba(148, 163, 184, 0.15)", "rgba(148, 163, 184, 0.35)"
        vol_score = 0.0

    # -------------------------------------------------------------------------
    # 6. Composite Regime / Bias
    # -------------------------------------------------------------------------
    composite_score = trend_score + mom_score + vol_score
    if composite_score >= 2.5:
        regime_bias = "STRONG BULLISH"
        reg_color, reg_bg, reg_border = "#10B981", "rgba(16, 185, 129, 0.15)", "rgba(16, 185, 129, 0.35)"
    elif composite_score >= 0.8:
        regime_bias = "BULLISH"
        reg_color, reg_bg, reg_border = "#10B981", "rgba(16, 185, 129, 0.15)", "rgba(16, 185, 129, 0.35)"
    elif trend_score >= 0.5 and (momentum_state in ["OVERSOLD BOUNCE", "OVERSOLD"] or mom_score < -0.5):
        regime_bias = "BULLISH PULLBACK"
        reg_color, reg_bg, reg_border = "#38BDF8", "rgba(56, 189, 248, 0.15)", "rgba(56, 189, 248, 0.35)"
    elif composite_score <= -2.5:
        regime_bias = "STRONG BEARISH"
        reg_color, reg_bg, reg_border = "#F87171", "rgba(239, 68, 68, 0.22)", "rgba(248, 113, 113, 0.45)"
    elif composite_score <= -0.8:
        regime_bias = "BEARISH"
        reg_color, reg_bg, reg_border = "#EF4444", "rgba(239, 68, 68, 0.15)", "rgba(239, 68, 68, 0.35)"
    else:
        regime_bias = "RANGE / NEUTRAL"
        reg_color, reg_bg, reg_border = "#94A3B8", "rgba(148, 163, 184, 0.15)", "rgba(148, 163, 184, 0.35)"

    return {
        "Trend": {"label": trend_state, "color": trend_color, "bg": trend_bg, "border": trend_border},
        "Momentum": {"label": momentum_state, "color": mom_color, "bg": mom_bg, "border": mom_border},
        "Strength": {"label": strength_state, "color": str_color, "bg": str_bg, "border": str_border},
        "Volatility": {"label": vol_state, "color": vol_color, "bg": vol_bg, "border": vol_border},
        "Volume": {"label": volm_state, "color": volm_color, "bg": volm_bg, "border": volm_border},
        "Regime / Bias": {"label": regime_bias, "color": reg_color, "bg": reg_bg, "border": reg_border},
    }


def render_page():
    """Render the Modern Technical Analysis Terminal."""
    st.set_page_config(
        page_title="Technical Terminal - QuantTerminal",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_theme()

    # 1. Sidebar & Parameter Extraction
    ticker, company, exchange, period, interval, region = render_sidebar()
    currency_sym = CURRENCY_SYMBOLS.get("INR" if region == "India" else "USD", "₹")

    # 2. Fetch Data
    df = load_data(ticker, period=period, interval=interval)
    if df.empty or len(df) < 5:
        st.error(f"No price data available for **{ticker}** ({period}, {interval}). Please select a different stock or period from the sidebar.")
        return

    # Buffer 1-year daily data for true 52W range metrics and unstarved regime calculations
    df_1y = load_data(ticker, period="1y", interval="1d")

    # 3. Base Metrics Calculation
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
    price_delta = last_close - prev_close
    pct_delta = (price_delta / prev_close) * 100 if prev_close != 0 else 0.0
    day_high = float(df["High"].iloc[-1])
    day_low = float(df["Low"].iloc[-1])

    # 52-Week Range: Use 1-year daily dataset to guarantee accurate 52-week metrics even when 1mo or intraday chart period is selected
    if not df_1y.empty and len(df_1y) >= 20:
        w52_high = float(df_1y["High"].max())
        w52_low = float(df_1y["Low"].min())
    else:
        w52_high = float(df["High"].tail(252).max()) if len(df) >= 252 else float(df["High"].max())
        w52_low = float(df["Low"].tail(252).min()) if len(df) >= 252 else float(df["Low"].min())

    latest_vol = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0.0
    if not df_1y.empty and "Volume" in df_1y.columns and len(df_1y) >= 20:
        avg20_vol = float(df_1y["Volume"].tail(20).mean())
    elif "Volume" in df.columns and len(df) >= 20:
        avg20_vol = float(df["Volume"].tail(20).mean())
    else:
        avg20_vol = latest_vol

    delta_color = "#00E676" if price_delta >= 0 else "#EF4444"
    delta_sign = "+" if price_delta >= 0 else ""

    # -------------------------------------------------------------------------
    # Hero Section: Company Header + Key Market Data Badges
    # -------------------------------------------------------------------------
    col_hero_left, col_hero_right = st.columns([1.1, 1.4])

    with col_hero_left:
        hero_left_html = (
            f'<div style="padding-top: 4px;">'
            f'<h1 style="margin: 0; font-size: 2.15rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.02em;">{company}</h1>'
            f'<div style="color: #38BDF8; font-size: 0.92rem; font-weight: 500; margin: 4px 0 10px 0;">{ticker} · {exchange} ({region})</div>'
            f'<div style="display: flex; align-items: baseline; gap: 14px;">'
            f'<span style="font-family: \'JetBrains Mono\', monospace; font-size: 2.3rem; font-weight: 700; color: #FFFFFF;">{currency_sym}{last_close:,.2f}</span>'
            f'<span style="font-size: 1.15rem; font-weight: 600; color: {delta_color};">{delta_sign}{price_delta:,.2f} ({delta_sign}{pct_delta:.2f}%)</span>'
            f'</div></div>'
        )
        st.markdown(hero_left_html, unsafe_allow_html=True)

    with col_hero_right:
        hero_right_html = (
            f'<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 10px;">'
            f'<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px; text-align: center;">'
            f'<div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">Day Range</div>'
            f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.88rem; font-weight: 600; color: #F1F5F9; margin-top: 4px;">{currency_sym}{day_low:,.2f} – {currency_sym}{day_high:,.2f}</div>'
            f'</div>'
            f'<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px; text-align: center;">'
            f'<div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">52W Range</div>'
            f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.88rem; font-weight: 600; color: #F1F5F9; margin-top: 4px;">{currency_sym}{w52_low:,.2f} – {currency_sym}{w52_high:,.2f}</div>'
            f'</div>'
            f'<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px; text-align: center;">'
            f'<div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">Latest Vol</div>'
            f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.95rem; font-weight: 600; color: #F1F5F9; margin-top: 4px;">{_fmt_num(latest_vol)}</div>'
            f'</div>'
            f'<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px; text-align: center;">'
            f'<div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">20D Avg Vol</div>'
            f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.95rem; font-weight: 600; color: #F1F5F9; margin-top: 4px;">{_fmt_num(avg20_vol)}</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(hero_right_html, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Compute Baseline Indicators & Regime Summary
    # -------------------------------------------------------------------------
    sma20_base = trend.sma(df["Close"], window=20)

    # Use 1Y daily dataset for regime calculations when chart interval is daily/macro or period is short (<50 bars) to guarantee unstarved indicators
    if interval in ["1d", "1wk", "1mo"] or len(df) < 50:
        df_regime = df_1y if (not df_1y.empty and len(df_1y) >= 30) else df
    else:
        df_regime = df

    regime_data = _compute_regime_summary(df_regime, df_chart=df)

    # -------------------------------------------------------------------------
    # Technical Regime Summary Card
    # -------------------------------------------------------------------------
    badges_html = "".join([
        f'<div style="text-align: center; flex: 1; min-width: 110px;">'
        f'<div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em; margin-bottom: 6px; text-transform: uppercase;">{cat}</div>'
        f'<div style="background: {info["bg"]}; color: {info["color"]}; border: 1px solid {info["border"]}; border-radius: 14px; padding: 4px 12px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; display: inline-block;">{info["label"]}</div>'
        f'</div>'
        for cat, info in regime_data.items()
    ])

    regime_card_html = (
        f'<div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 14px 20px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35); margin-bottom: 20px;">'
        f'<div style="font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em; color: #38BDF8; margin-bottom: 12px; text-transform: uppercase;">TECHNICAL REGIME SUMMARY</div>'
        f'<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 12px;">{badges_html}</div>'
        f'</div>'
    )
    st.markdown(regime_card_html, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Terminal Controls Row (Mode, Preset, Event Markers, S&R)
    # -------------------------------------------------------------------------
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.1, 1.4, 1.1, 1.2])

    with ctrl1:
        mode = st.radio("Terminal Mode", ["Standard", "Advanced"], horizontal=True)

    with ctrl2:
        preset_names = list(PRESETS.keys())
        active_preset = st.selectbox("Analysis Preset", preset_names, index=0)

    with ctrl3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        show_events = st.checkbox("Show Event Markers", value=True)

    with ctrl4:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        show_sr = st.checkbox("Show Support & Resistance", value=True)

    # Advanced Parameters Expander
    adv_params = {
        "sma_win": 20, "ema_win": 20, "bb_win": 20, "bb_std": 2.0,
        "rsi_win": 14, "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
        "adx_win": 14, "atr_win": 14, "stoch_k": 14, "stoch_d": 3,
        "supertrend_mult": 3.0, "supertrend_atr": 10, "roc_win": 12,
        "cmo_win": 14, "williams_win": 14, "aroon_win": 25, "vortex_win": 14
    }

    if mode == "Advanced":
        with st.expander("⚙️ Advanced Indicator Hyperparameters", expanded=True):
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                adv_params["sma_win"] = st.slider("SMA Period", 5, 200, 20)
                adv_params["ema_win"] = st.slider("EMA Period", 5, 200, 20)
                adv_params["supertrend_mult"] = st.slider("SuperTrend Multiplier", 1.0, 6.0, 3.0, 0.5)
            with p2:
                adv_params["bb_win"] = st.slider("Bollinger Band Period", 5, 100, 20)
                adv_params["bb_std"] = st.slider("Bollinger Std Dev", 1.0, 4.0, 2.0, 0.1)
                adv_params["atr_win"] = st.slider("ATR Period", 2, 50, 14)
            with p3:
                adv_params["rsi_win"] = st.slider("RSI Period", 2, 50, 14)
                adv_params["adx_win"] = st.slider("ADX Period", 5, 50, 14)
                adv_params["aroon_win"] = st.slider("Aroon Period", 10, 50, 25)
            with p4:
                adv_params["roc_win"] = st.slider("ROC Period", 2, 30, 12)
                adv_params["williams_win"] = st.slider("Williams %R Period", 5, 30, 14)
                adv_params["vortex_win"] = st.slider("Vortex Period", 5, 30, 14)

    # -------------------------------------------------------------------------
    # Terminal Indicator Composition (14 Overlays + 20 Subpanels = 34 Total)
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.subheader("🛠️ Terminal Indicator Composition")

    preset_overlays = PRESETS[active_preset]["overlays"]
    preset_subpanels = PRESETS[active_preset]["subpanels"]

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        selected_overlays = st.multiselect(
            f"Price Axis Overlays ({len(ALL_OVERLAYS)} available)",
            options=ALL_OVERLAYS,
            default=preset_overlays,
            key=f"overlays_{active_preset}",
        )

    with comp_col2:
        selected_subpanels = st.multiselect(
            f"Analytical Subpanels ({len(ALL_SUBPANELS)} available)",
            options=ALL_SUBPANELS,
            default=preset_subpanels,
            key=f"subpanels_{active_preset}",
        )

    # -------------------------------------------------------------------------
    # Indicator Computations (Computed based on user selection)
    # -------------------------------------------------------------------------
    indicators = {}

    # Overlays (18 total)
    if "SMA" in selected_overlays or True:
        indicators["SMA"] = trend.sma(df["Close"], window=adv_params["sma_win"])
    if "EMA" in selected_overlays:
        indicators["EMA"] = trend.ema(df["Close"], window=adv_params["ema_win"])
    if "WMA" in selected_overlays:
        indicators["WMA"] = trend.wma(df["Close"], window=20)
    if "HMA" in selected_overlays:
        indicators["HMA"] = trend.hma(df["Close"], window=20)
    if "VWMA" in selected_overlays and "Volume" in df.columns:
        indicators["VWMA"] = trend.vwma(df["Close"], df["Volume"], window=20)
    if "DEMA" in selected_overlays:
        e1 = df["Close"].ewm(span=adv_params["ema_win"], adjust=False).mean()
        e2 = e1.ewm(span=adv_params["ema_win"], adjust=False).mean()
        indicators["DEMA"] = 2 * e1 - e2
    if "TEMA" in selected_overlays:
        e1 = df["Close"].ewm(span=adv_params["ema_win"], adjust=False).mean()
        e2 = e1.ewm(span=adv_params["ema_win"], adjust=False).mean()
        e3 = e2.ewm(span=adv_params["ema_win"], adjust=False).mean()
        indicators["TEMA"] = 3 * e1 - 3 * e2 + e3
    if "Bollinger_Bands" in selected_overlays or "Bollinger_Bandwidth" in selected_subpanels or "Bollinger_%B" in selected_subpanels or True:
        indicators["Bollinger_Bands"] = volatility.bollinger_bands(df["Close"], window=adv_params["bb_win"], num_std=adv_params["bb_std"])
    if "Keltner_Channels" in selected_overlays:
        indicators["Keltner_Channels"] = volatility.keltner(df["High"], df["Low"], df["Close"])
    if "Donchian_Channels" in selected_overlays:
        indicators["Donchian_Channels"] = volatility.donchian(df["High"], df["Low"], window=20)
    if "Price_Envelopes" in selected_overlays:
        env_mid = df["Close"].rolling(20).mean()
        indicators["Price_Envelopes"] = {
            "upper": env_mid * 1.025,
            "middle": env_mid,
            "lower": env_mid * 0.975,
        }
    if "SuperTrend" in selected_overlays:
        st_line, st_dir = _calculate_supertrend_line(df["High"], df["Low"], df["Close"], multiplier=adv_params["supertrend_mult"], atr_window=adv_params["supertrend_atr"])
        indicators["SuperTrend"] = st_line
        indicators["SuperTrend_Dir"] = st_dir
    if "Parabolic_SAR" in selected_overlays:
        indicators["Parabolic_SAR"] = trend.sar(df["High"], df["Low"])
    if "Ichimoku_Cloud" in selected_overlays:
        indicators["Ichimoku_Cloud"] = trend.ichimoku(df["High"], df["Low"], df["Close"])
    if "VWAP" in selected_overlays and "Volume" in df.columns:
        indicators["VWAP"] = volume.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
    if "ZigZag" in selected_overlays:
        indicators["ZigZag"] = trend.zigzag(df["Close"])
    if "Fractals" in selected_overlays:
        indicators["Fractals"] = trend.fractals(df["High"], df["Low"])
    if "Pivot_Points" in selected_overlays:
        indicators["Pivot_Points"] = _calculate_pivot_points(df)

    # Subpanels (28 total)
    if "RSI" in selected_subpanels or True:
        indicators["RSI"] = momentum.rsi(df["Close"], window=adv_params["rsi_win"])
    if "MACD" in selected_subpanels or True:
        indicators["MACD"] = momentum.macd(df["Close"], fast=adv_params["macd_fast"], slow=adv_params["macd_slow"], signal=adv_params["macd_sig"])
    if "Stochastic" in selected_subpanels:
        indicators["Stochastic"] = momentum.stochastic(df["High"], df["Low"], df["Close"], k_window=adv_params["stoch_k"], d_window=adv_params["stoch_d"])
    if "Williams_%R" in selected_subpanels:
        indicators["Williams_%R"] = momentum.williams_r(df["High"], df["Low"], df["Close"], window=adv_params["williams_win"])
    if "ROC" in selected_subpanels:
        indicators["ROC"] = momentum.roc(df["Close"], window=adv_params["roc_win"])
    if "CMO" in selected_subpanels:
        indicators["CMO"] = momentum.cmo(df["Close"], window=adv_params["cmo_win"])
    if "TRIX" in selected_subpanels:
        indicators["TRIX"] = momentum.trix(df["Close"], window=15)
    if "CCI" in selected_subpanels:
        tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        indicators["CCI"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    if "ADX" in selected_subpanels or True:
        indicators["ADX"] = strength.adx(df["High"], df["Low"], df["Close"], window=adv_params["adx_win"])
    if "Aroon" in selected_subpanels:
        indicators["Aroon"] = strength.aroon(df["High"], df["Low"], window=adv_params["aroon_win"])
    if "Vortex" in selected_subpanels:
        indicators["Vortex"] = strength.vortex(df["High"], df["Low"], df["Close"], window=adv_params["vortex_win"])
    if "ATR" in selected_subpanels:
        indicators["ATR"] = volatility.atr(df["High"], df["Low"], df["Close"], window=adv_params["atr_win"])
    if "Bollinger_Bandwidth" in selected_subpanels:
        bb = indicators["Bollinger_Bands"]
        indicators["Bollinger_Bandwidth"] = (bb["upper"] - bb["lower"]) / bb["middle"].replace(0, np.nan) * 100.0
    if "Bollinger_%B" in selected_subpanels:
        bb = indicators["Bollinger_Bands"]
        indicators["Bollinger_%B"] = (df["Close"] - bb["lower"]) / (bb["upper"] - bb["lower"]).replace(0, np.nan)
    if "Historical_Volatility" in selected_subpanels:
        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        indicators["Historical_Volatility"] = log_ret.rolling(30).std() * np.sqrt(252) * 100.0
    if "RVOL" in selected_subpanels or True:
        indicators["RVOL"] = volume.rvol(df["Volume"], window=20) if "Volume" in df.columns else pd.Series(1.0, index=df.index)
    if "OBV" in selected_subpanels:
        indicators["OBV"] = volume.obv(df["Close"], df["Volume"]) if "Volume" in df.columns else df["Close"]
    if "CMF" in selected_subpanels:
        indicators["CMF"] = volume.cmf(df["High"], df["Low"], df["Close"], df["Volume"], window=20) if "Volume" in df.columns else pd.Series(0.0, index=df.index)
    if "MFI" in selected_subpanels:
        indicators["MFI"] = volume.mfi(df["High"], df["Low"], df["Close"], df["Volume"], window=14) if "Volume" in df.columns else pd.Series(50.0, index=df.index)
    if "ADL" in selected_subpanels:
        indicators["ADL"] = volume.adl(df["High"], df["Low"], df["Close"], df["Volume"]) if "Volume" in df.columns else df["Close"]
    if "Chaikin_Oscillator" in selected_subpanels:
        indicators["Chaikin_Oscillator"] = volume.chaikin_oscillator(df["High"], df["Low"], df["Close"], df["Volume"]) if "Volume" in df.columns else pd.Series(0.0, index=df.index)
    if "PVT" in selected_subpanels:
        indicators["PVT"] = volume.pvt(df["Close"], df["Volume"]) if "Volume" in df.columns else df["Close"]
    if "Ease_of_Movement" in selected_subpanels and "Volume" in df.columns:
        indicators["Ease_of_Movement"] = volume.ease_of_movement(df["High"], df["Low"], df["Volume"])
    if "Force_Index" in selected_subpanels and "Volume" in df.columns:
        indicators["Force_Index"] = volume.force_index(df["Close"], df["Volume"])
    if "Volume_Oscillator" in selected_subpanels and "Volume" in df.columns:
        indicators["Volume_Oscillator"] = volume.volume_oscillator(df["Volume"])
    if "PVI" in selected_subpanels and "Volume" in df.columns:
        indicators["PVI"] = volume.pvi(df["Close"], df["Volume"])
    if "NVI" in selected_subpanels and "Volume" in df.columns:
        indicators["NVI"] = volume.nvi(df["Close"], df["Volume"])

    # Pivot levels
    pivots = _calculate_pivot_points(df)

    # -------------------------------------------------------------------------
    # Plotly Interactive Charting
    # -------------------------------------------------------------------------
    num_subpanels = len(selected_subpanels)
    total_rows = 1 + num_subpanels

    row_heights = [0.55] + [0.45 / max(num_subpanels, 1)] * num_subpanels if num_subpanels > 0 else [1.0]

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=row_heights,
    )

    # 1. Main Candlestick Chart (Row 1)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#00E676",
            decreasing_line_color="#FF3B30",
            increasing_fillcolor="#00E676",
            decreasing_fillcolor="#FF3B30",
        ),
        row=1,
        col=1,
    )

    # Overlays on Row 1
    if "SMA" in selected_overlays:
        fig.add_trace(go.Scatter(x=df.index, y=indicators["SMA"], name=f"SMA ({adv_params['sma_win']})", line=dict(color="#F59E0B", width=1.5)), row=1, col=1)
    if "EMA" in selected_overlays:
        fig.add_trace(go.Scatter(x=df.index, y=indicators["EMA"], name=f"EMA ({adv_params['ema_win']})", line=dict(color="#38BDF8", width=1.5)), row=1, col=1)
    if "WMA" in selected_overlays:
        fig.add_trace(go.Scatter(x=df.index, y=indicators["WMA"], name="WMA (20)", line=dict(color="#EC4899", width=1.5)), row=1, col=1)
    if "HMA" in selected_overlays:
        fig.add_trace(go.Scatter(x=df.index, y=indicators["HMA"], name="HMA (20)", line=dict(color="#8B5CF6", width=1.5)), row=1, col=1)
    if "VWMA" in selected_overlays and "VWMA" in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=indicators["VWMA"], name="VWMA (20)", line=dict(color="#10B981", width=1.5, dash="dash")), row=1, col=1)
    if "DEMA" in selected_overlays and "DEMA" in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=indicators["DEMA"], name=f"DEMA ({adv_params['ema_win']})", line=dict(color="#A7F3D0", width=1.5)), row=1, col=1)
    if "TEMA" in selected_overlays and "TEMA" in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=indicators["TEMA"], name=f"TEMA ({adv_params['ema_win']})", line=dict(color="#6EE7B7", width=1.5, dash="dot")), row=1, col=1)

    if "Bollinger_Bands" in selected_overlays:
        bb = indicators["Bollinger_Bands"]
        fig.add_trace(go.Scatter(x=df.index, y=bb["upper"], name="Bollinger Upper", line=dict(color="rgba(168, 85, 247, 0.8)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb["middle"], name="Bollinger Middle", line=dict(color="rgba(192, 132, 252, 0.8)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb["lower"], name="Bollinger Lower", line=dict(color="rgba(168, 85, 247, 0.8)", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(168, 85, 247, 0.08)"), row=1, col=1)

    if "Keltner_Channels" in selected_overlays:
        kc = indicators["Keltner_Channels"]
        fig.add_trace(go.Scatter(x=df.index, y=kc["upper"], name="Keltner Upper", line=dict(color="#60A5FA", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=kc["middle"], name="Keltner Middle", line=dict(color="#93C5FD", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=kc["lower"], name="Keltner Lower", line=dict(color="#60A5FA", width=1, dash="dot")), row=1, col=1)

    if "Donchian_Channels" in selected_overlays:
        dc = indicators["Donchian_Channels"]
        fig.add_trace(go.Scatter(x=df.index, y=dc["upper"], name="Donchian Upper", line=dict(color="#06B6D4", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=dc["middle"], name="Donchian Middle", line=dict(color="#22D3EE", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=dc["lower"], name="Donchian Lower", line=dict(color="#06B6D4", width=1, dash="dot")), row=1, col=1)

    if "Price_Envelopes" in selected_overlays and "Price_Envelopes" in indicators:
        pe = indicators["Price_Envelopes"]
        fig.add_trace(go.Scatter(x=df.index, y=pe["upper"], name="Envelope Upper (2.5%)", line=dict(color="rgba(244, 114, 182, 0.7)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=pe["middle"], name="Envelope Middle", line=dict(color="rgba(244, 114, 182, 0.9)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=pe["lower"], name="Envelope Lower (2.5%)", line=dict(color="rgba(244, 114, 182, 0.7)", width=1, dash="dot")), row=1, col=1)

    if "SuperTrend" in selected_overlays and "SuperTrend" in indicators:
        st_series = indicators["SuperTrend"]
        st_dir_last = indicators.get("SuperTrend_Dir", pd.Series([1])).iloc[-1]
        st_color = "#10B981" if st_dir_last == 1 else "#EF4444"
        fig.add_trace(go.Scatter(x=df.index, y=st_series, name="SuperTrend Stop", line=dict(color=st_color, width=1.8, dash="dash")), row=1, col=1)

    if "Parabolic_SAR" in selected_overlays:
        fig.add_trace(go.Scatter(x=df.index, y=indicators["Parabolic_SAR"], name="SAR", mode="markers", marker=dict(color="#06B6D4", size=3)), row=1, col=1)

    if "Ichimoku_Cloud" in selected_overlays:
        ich = indicators["Ichimoku_Cloud"]
        fig.add_trace(go.Scatter(x=df.index, y=ich["tenkan_sen"], name="Tenkan-sen", line=dict(color="#FFFFFF", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ich["kijun_sen"], name="Kijun-sen", line=dict(color="#38BDF8", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ich["senkou_a"], name="Senkou A", line=dict(color="#10B981", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ich["senkou_b"], name="Senkou B", line=dict(color="#EF4444", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(16, 185, 129, 0.06)"), row=1, col=1)

    if "VWAP" in selected_overlays and "VWAP" in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=indicators["VWAP"], name="VWAP", line=dict(color="#F43F5E", width=1.5)), row=1, col=1)

    if "ZigZag" in selected_overlays:
        fig.add_trace(go.Scatter(x=df.index, y=indicators["ZigZag"], name="ZigZag", line=dict(color="#FBBF24", width=1.5), connectgaps=True), row=1, col=1)

    if "Fractals" in selected_overlays:
        fr = indicators["Fractals"]
        high_mask = fr["fractal_high"] == 1
        low_mask = fr["fractal_low"] == 1
        if high_mask.any():
            fig.add_trace(go.Scatter(x=df.index[high_mask], y=df.loc[high_mask, "High"] * 1.005, mode="markers", marker=dict(symbol="triangle-down", color="#EF4444", size=7), name="Fractal High"), row=1, col=1)
        if low_mask.any():
            fig.add_trace(go.Scatter(x=df.index[low_mask], y=df.loc[low_mask, "Low"] * 0.995, mode="markers", marker=dict(symbol="triangle-up", color="#10B981", size=7), name="Fractal Low"), row=1, col=1)

    if "Pivot_Points" in selected_overlays and "Pivot_Points" in indicators:
        pv = indicators["Pivot_Points"]
        for p_name, p_val, p_col in [("Pivot", pv["Pivot"], "#F59E0B"), ("R1", pv["Resistance 1"], "#F87171"), ("R2", pv["Resistance 2"], "#EF4444"), ("S1", pv["Support 1"], "#34D399"), ("S2", pv["Support 2"], "#10B981")]:
            fig.add_hline(y=p_val, line_dash="dash", line_color=p_col, line_width=1, annotation_text=f"{p_name}: {currency_sym}{p_val:,.2f}", annotation_position="top right", annotation_font_size=9, annotation_font_color=p_col, row=1, col=1)

    # Support & Resistance Lines on Row 1
    if show_sr:
        sr_levels = [
            ("Resistance 2", pivots["Resistance 2"], "#EF4444"),
            ("Resistance 1", pivots["Resistance 1"], "#F87171"),
            ("Support 1", pivots["Support 1"], "#34D399"),
            ("Support 2", pivots["Support 2"], "#10B981"),
        ]
        for name, val, color in sr_levels:
            fig.add_hline(
                y=val,
                line_dash="dot",
                line_color=color,
                line_width=1.2,
                annotation_text=f"{name}: {currency_sym}{val:,.2f}",
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color=color,
                row=1,
                col=1,
            )

    # Event Markers on Row 1
    if show_events:
        buy_cond = (df["Close"] > sma20_base) & (df["Close"].shift(1) <= sma20_base.shift(1))
        sell_cond = (df["Close"] < sma20_base) & (df["Close"].shift(1) >= sma20_base.shift(1))

        buy_idx = df.index[buy_cond]
        sell_idx = df.index[sell_cond]

        if len(buy_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_idx,
                    y=df.loc[buy_idx, "Low"] * 0.995,
                    mode="markers",
                    marker=dict(symbol="triangle-up", color="#FBBF24", size=10),
                    name="Bullish Event",
                ),
                row=1,
                col=1,
            )
        if len(sell_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_idx,
                    y=df.loc[sell_idx, "High"] * 1.005,
                    mode="markers",
                    marker=dict(symbol="triangle-down", color="#EF4444", size=10),
                    name="Bearish Event",
                ),
                row=1,
                col=1,
            )

    # Analytical Subpanels (Rows 2..N)
    current_row = 2
    for sub in selected_subpanels:
        if sub == "RSI":
            fig.add_trace(go.Scatter(x=df.index, y=indicators["RSI"], name="RSI", line=dict(color="#A855F7", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)", line_width=1, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(16, 185, 129, 0.6)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
            current_row += 1

        elif sub == "MACD":
            m_df = indicators["MACD"]
            fig.add_trace(go.Scatter(x=df.index, y=m_df["macd"], name="MACD", line=dict(color="#38BDF8", width=1.5)), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=m_df["signal"], name="Signal", line=dict(color="#F43F5E", width=1.2, dash="dash")), row=current_row, col=1)
            hist_colors = np.where(m_df["histogram"].values >= 0, "#10B981", "#EF4444")
            fig.add_trace(go.Bar(x=df.index, y=m_df["histogram"], name="Histogram", marker_color=hist_colors), row=current_row, col=1)
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
            current_row += 1

        elif sub == "Stochastic":
            st_df = indicators["Stochastic"]
            fig.add_trace(go.Scatter(x=df.index, y=st_df["K"], name="%K", line=dict(color="#38BDF8", width=1.5)), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=st_df["D"], name="%D", line=dict(color="#F43F5E", width=1.2, dash="dash")), row=current_row, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)", line_width=1, row=current_row, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="rgba(16, 185, 129, 0.6)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="Stoch", range=[0, 100], row=current_row, col=1)
            current_row += 1

        elif sub == "Williams_%R":
            fig.add_trace(go.Scatter(x=df.index, y=indicators["Williams_%R"], name="Williams %R", line=dict(color="#38BDF8", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=-20, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)", line_width=1, row=current_row, col=1)
            fig.add_hline(y=-80, line_dash="dash", line_color="rgba(16, 185, 129, 0.6)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="W%R", range=[-100, 0], row=current_row, col=1)
            current_row += 1

        elif sub == "ROC":
            fig.add_trace(go.Scatter(x=df.index, y=indicators["ROC"], name="ROC", line=dict(color="#10B981", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="ROC %", row=current_row, col=1)
            current_row += 1

        elif sub == "CMO":
            fig.add_trace(go.Scatter(x=df.index, y=indicators["CMO"], name="CMO", line=dict(color="#F59E0B", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)", line_width=1, row=current_row, col=1)
            fig.add_hline(y=-50, line_dash="dash", line_color="rgba(16, 185, 129, 0.6)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="CMO", range=[-100, 100], row=current_row, col=1)
            current_row += 1

        elif sub == "TRIX":
            fig.add_trace(go.Scatter(x=df.index, y=indicators["TRIX"], name="TRIX", line=dict(color="#A855F7", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="TRIX", row=current_row, col=1)
            current_row += 1

        elif sub == "CCI":
            fig.add_trace(go.Scatter(x=df.index, y=indicators["CCI"], name="CCI", line=dict(color="#38BDF8", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=100, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)", line_width=1, row=current_row, col=1)
            fig.add_hline(y=-100, line_dash="dash", line_color="rgba(16, 185, 129, 0.6)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="CCI", row=current_row, col=1)
            current_row += 1

        elif sub == "ADX":
            a_df = indicators["ADX"]
            fig.add_trace(go.Scatter(x=df.index, y=a_df["adx"], name="ADX", line=dict(color="#F59E0B", width=1.5)), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=a_df["plus_di"], name="+DI", line=dict(color="#10B981", width=1, dash="dot")), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=a_df["minus_di"], name="-DI", line=dict(color="#EF4444", width=1, dash="dot")), row=current_row, col=1)
            fig.add_hline(y=25, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="ADX", row=current_row, col=1)
            current_row += 1

        elif sub == "Aroon":
            ar_df = indicators["Aroon"]
            fig.add_trace(go.Scatter(x=df.index, y=ar_df["aroon_up"], name="Aroon Up", line=dict(color="#10B981", width=1.5)), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=ar_df["aroon_down"], name="Aroon Down", line=dict(color="#EF4444", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", line_width=1, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="Aroon", range=[0, 100], row=current_row, col=1)
            current_row += 1

        elif sub == "Vortex":
            vx_df = indicators["Vortex"]
            fig.add_trace(go.Scatter(x=df.index, y=vx_df["vi_plus"], name="VI+", line=dict(color="#10B981", width=1.5)), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=vx_df["vi_minus"], name="VI-", line=dict(color="#EF4444", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="Vortex", row=current_row, col=1)
            current_row += 1

        elif sub == "ATR":
            fig.add_trace(go.Scatter(x=df.index, y=indicators["ATR"], name="ATR", line=dict(color="#38BDF8", width=1.5)), row=current_row, col=1)
            fig.update_yaxes(title_text="ATR", row=current_row, col=1)
            current_row += 1

        elif sub == "Bollinger_Bandwidth":
            fig.add_trace(go.Scatter(x=df.index, y=indicators["Bollinger_Bandwidth"], name="BB Bandwidth %", line=dict(color="#C084FC", width=1.5)), row=current_row, col=1)
            fig.update_yaxes(title_text="BBW %", row=current_row, col=1)
            current_row += 1

        elif sub == "Bollinger_%B":
            fig.add_trace(go.Scatter(x=df.index, y=indicators["Bollinger_%B"], name="%B", line=dict(color="#C084FC", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)", line_width=1, row=current_row, col=1)
            fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(16, 185, 129, 0.6)", line_width=1, row=current_row, col=1)
            fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255, 255, 255, 0.3)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="%B", row=current_row, col=1)
            current_row += 1

        elif sub == "Historical_Volatility":
            fig.add_trace(go.Scatter(x=df.index, y=indicators["Historical_Volatility"], name="Hist Vol %", line=dict(color="#F43F5E", width=1.5)), row=current_row, col=1)
            fig.update_yaxes(title_text="HV %", row=current_row, col=1)
            current_row += 1

        elif sub == "RVOL":
            rvol_s = indicators["RVOL"]
            rvol_colors = np.where(rvol_s.values >= 1.0, "#38BDF8", "#64748B")
            fig.add_trace(go.Bar(x=df.index, y=rvol_s, name="RVOL", marker_color=rvol_colors), row=current_row, col=1)
            fig.add_hline(y=1.0, line_dash="dash", line_color="#EF4444", line_width=1.2, row=current_row, col=1)
            fig.update_yaxes(title_text="RVOL", row=current_row, col=1)
            current_row += 1

        elif sub == "OBV":
            fig.add_trace(go.Scatter(x=df.index, y=indicators["OBV"], name="OBV", line=dict(color="#10B981", width=1.5)), row=current_row, col=1)
            fig.update_yaxes(title_text="OBV", row=current_row, col=1)
            current_row += 1

        elif sub == "CMF":
            cmf_s = indicators["CMF"]
            fig.add_trace(go.Scatter(x=df.index, y=cmf_s, name="CMF", line=dict(color="#10B981", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="CMF", row=current_row, col=1)
            current_row += 1

        elif sub == "MFI":
            fig.add_trace(go.Scatter(x=df.index, y=indicators["MFI"], name="MFI", line=dict(color="#F59E0B", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)", line_width=1, row=current_row, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="rgba(16, 185, 129, 0.6)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="MFI", range=[0, 100], row=current_row, col=1)
            current_row += 1

        elif sub == "ADL":
            fig.add_trace(go.Scatter(x=df.index, y=indicators["ADL"], name="ADL", line=dict(color="#38BDF8", width=1.5)), row=current_row, col=1)
            fig.update_yaxes(title_text="ADL", row=current_row, col=1)
            current_row += 1

        elif sub == "Chaikin_Oscillator":
            co_s = indicators["Chaikin_Oscillator"]
            co_colors = np.where(co_s.values >= 0, "#10B981", "#EF4444")
            fig.add_trace(go.Bar(x=df.index, y=co_s, name="Chaikin Osc", marker_color=co_colors), row=current_row, col=1)
            fig.update_yaxes(title_text="Chaikin", row=current_row, col=1)
            current_row += 1

        elif sub == "PVT":
            fig.add_trace(go.Scatter(x=df.index, y=indicators["PVT"], name="PVT", line=dict(color="#8B5CF6", width=1.5)), row=current_row, col=1)
            fig.update_yaxes(title_text="PVT", row=current_row, col=1)
            current_row += 1

        elif sub == "Ease_of_Movement" and "Ease_of_Movement" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=indicators["Ease_of_Movement"], name="EMV", line=dict(color="#2DD4BF", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="EMV", row=current_row, col=1)
            current_row += 1

        elif sub == "Force_Index" and "Force_Index" in indicators:
            fi_s = indicators["Force_Index"]
            fi_colors = np.where(fi_s.values >= 0, "#10B981", "#EF4444")
            fig.add_trace(go.Bar(x=df.index, y=fi_s, name="Force Index", marker_color=fi_colors), row=current_row, col=1)
            fig.update_yaxes(title_text="Force Idx", row=current_row, col=1)
            current_row += 1

        elif sub == "Volume_Oscillator" and "Volume_Oscillator" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=indicators["Volume_Oscillator"], name="Vol Osc %", line=dict(color="#FB923C", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="Vol Osc %", row=current_row, col=1)
            current_row += 1

        elif sub == "PVI" and "PVI" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=indicators["PVI"], name="PVI", line=dict(color="#10B981", width=1.5)), row=current_row, col=1)
            fig.update_yaxes(title_text="PVI", row=current_row, col=1)
            current_row += 1

        elif sub == "NVI" and "NVI" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=indicators["NVI"], name="NVI", line=dict(color="#38BDF8", width=1.5)), row=current_row, col=1)
            fig.update_yaxes(title_text="NVI", row=current_row, col=1)
            current_row += 1

        elif sub == "Volume":
            vol_colors = np.where(df["Close"].values >= df["Open"].values, "#00E676", "#FF3B30")
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors), row=current_row, col=1)
            vol_sma = df["Volume"].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=df.index, y=vol_sma, name="Vol SMA (20)", line=dict(color="#F59E0B", width=1.2)), row=current_row, col=1)
            fig.update_yaxes(title_text="Volume", row=current_row, col=1)
            current_row += 1

    chart_height = 480 + (num_subpanels * 160)
    fig.update_layout(
        template="plotly_dark",
        height=chart_height,
        margin=dict(l=40, r=60, t=30, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

    # -------------------------------------------------------------------------
    # Bottom Dashboard: Current Technical State & Technical Confluence
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    bot_col1, bot_col2 = st.columns([1.2, 1.0])

    with bot_col1:
        st.markdown("### 📋 Current Technical State")

        # Build table of indicators
        state_rows = []

        # RSI
        if "RSI" in indicators:
            rsi_v = float(indicators["RSI"].dropna().iloc[-1]) if len(indicators["RSI"].dropna()) > 0 else 50.0
            if rsi_v > 70:
                rsi_st, rsi_ev, rsi_dir = "Overbought", "Upper Threshold", "Bearish"
            elif rsi_v < 30:
                rsi_st, rsi_ev, rsi_dir = "Oversold", "Lower Threshold", "Bullish"
            elif rsi_v > 50:
                rsi_st, rsi_ev, rsi_dir = "Positive Momentum", "—", "Bullish"
            else:
                rsi_st, rsi_ev, rsi_dir = "Negative Momentum", "—", "Bearish"
            state_rows.append({"Indicator": f"RSI ({adv_params['rsi_win']})", "Value": f"{rsi_v:.1f}", "State": rsi_st, "Event": rsi_ev, "Direction": rsi_dir})

        # MACD
        if "MACD" in indicators:
            m_df = indicators["MACD"]
            last_m = float(m_df["macd"].dropna().iloc[-1]) if len(m_df["macd"].dropna()) > 0 else 0.0
            last_s = float(m_df["signal"].dropna().iloc[-1]) if len(m_df["signal"].dropna()) > 0 else 0.0
            macd_dir = "Bullish" if last_m >= last_s else "Bearish"
            state_rows.append({"Indicator": f"MACD ({adv_params['macd_fast']},{adv_params['macd_slow']},{adv_params['macd_sig']})", "Value": f"{last_m:.2f} / {last_s:.2f}", "State": macd_dir, "Event": "—", "Direction": macd_dir})

        # ADX
        if "ADX" in indicators:
            a_df = indicators["ADX"]
            last_adx_v = float(a_df["adx"].dropna().iloc[-1]) if len(a_df["adx"].dropna()) > 0 else 20.0
            adx_st = "Strong Trend" if last_adx_v >= 25 else ("Moderate Trend" if last_adx_v >= 20 else "Rangebound")
            last_sma20_v = float(sma20_base.dropna().iloc[-1]) if len(sma20_base.dropna()) > 0 else last_close
            adx_dir = "Bullish" if last_close >= last_sma20_v else "Bearish"
            state_rows.append({"Indicator": f"ADX ({adv_params['adx_win']})", "Value": f"{last_adx_v:.1f}", "State": adx_st, "Event": "—", "Direction": adx_dir})

        # RVOL
        if "RVOL" in indicators:
            last_rvol_v = float(indicators["RVOL"].dropna().iloc[-1]) if len(indicators["RVOL"].dropna()) > 0 else 1.0
            rvol_st = "High Volume" if last_rvol_v >= 1.2 else ("Low Volume" if last_rvol_v < 0.8 else "Normal Volume")
            rvol_dir = "Bullish" if price_delta >= 0 and last_rvol_v >= 1.0 else ("Bearish" if price_delta < 0 and last_rvol_v >= 1.0 else "Neutral")
            state_rows.append({"Indicator": "RVOL (20)", "Value": f"{last_rvol_v:.2f}x", "State": rvol_st, "Event": "—", "Direction": rvol_dir})

        # Bollinger Bands
        if "Bollinger_Bands" in indicators:
            bb = indicators["Bollinger_Bands"]
            last_up = float(bb["upper"].iloc[-1])
            last_mid = float(bb["middle"].iloc[-1])
            last_low = float(bb["lower"].iloc[-1])
            if last_close > last_up:
                bb_st, bb_dir = "Above Upper Band", "Bearish"
            elif last_close < last_low:
                bb_st, bb_dir = "Below Lower Band", "Bullish"
            else:
                bb_st = "Inside Bands"
                bb_dir = "Bullish" if last_close >= last_mid else "Bearish"
            state_rows.append({"Indicator": "Bollinger Bands", "Value": f"{last_low:,.1f} / {last_up:,.1f}", "State": bb_st, "Event": "—", "Direction": bb_dir})

        # SMA
        if "SMA" in indicators and "SMA" in selected_overlays:
            sma_val = float(indicators["SMA"].dropna().iloc[-1])
            s_dir = "Bullish" if last_close >= sma_val else "Bearish"
            state_rows.append({"Indicator": f"SMA ({adv_params['sma_win']})", "Value": f"{sma_val:,.2f}", "State": f"Price {'Above' if s_dir == 'Bullish' else 'Below'} SMA", "Event": "—", "Direction": s_dir})

        # EMA
        if "EMA" in indicators and "EMA" in selected_overlays:
            ema_val = float(indicators["EMA"].dropna().iloc[-1])
            e_dir = "Bullish" if last_close >= ema_val else "Bearish"
            state_rows.append({"Indicator": f"EMA ({adv_params['ema_win']})", "Value": f"{ema_val:,.2f}", "State": f"Price {'Above' if e_dir == 'Bullish' else 'Below'} EMA", "Event": "—", "Direction": e_dir})

        # DEMA
        if "DEMA" in indicators and "DEMA" in selected_overlays:
            dema_val = float(indicators["DEMA"].dropna().iloc[-1])
            d_dir = "Bullish" if last_close >= dema_val else "Bearish"
            state_rows.append({"Indicator": f"DEMA ({adv_params['ema_win']})", "Value": f"{dema_val:,.2f}", "State": f"Price {'Above' if d_dir == 'Bullish' else 'Below'} DEMA", "Event": "—", "Direction": d_dir})

        # TEMA
        if "TEMA" in indicators and "TEMA" in selected_overlays:
            tema_val = float(indicators["TEMA"].dropna().iloc[-1])
            t_dir = "Bullish" if last_close >= tema_val else "Bearish"
            state_rows.append({"Indicator": f"TEMA ({adv_params['ema_win']})", "Value": f"{tema_val:,.2f}", "State": f"Price {'Above' if t_dir == 'Bullish' else 'Below'} TEMA", "Event": "—", "Direction": t_dir})

        # Price Envelopes
        if "Price_Envelopes" in indicators and "Price_Envelopes" in selected_overlays:
            pe = indicators["Price_Envelopes"]
            pe_up = float(pe["upper"].dropna().iloc[-1])
            pe_low = float(pe["lower"].dropna().iloc[-1])
            if last_close > pe_up:
                pe_st, pe_dir = "Above Upper Envelope", "Bullish Extension"
            elif last_close < pe_low:
                pe_st, pe_dir = "Below Lower Envelope", "Oversold Discount"
            else:
                pe_st, pe_dir = "Inside Envelope Channel", "Neutral"
            state_rows.append({"Indicator": "Price Envelopes (2.5%)", "Value": f"{pe_low:,.1f} / {pe_up:,.1f}", "State": pe_st, "Event": "—", "Direction": pe_dir})

        # SuperTrend
        if "SuperTrend" in indicators and "SuperTrend" in selected_overlays:
            st_val = float(indicators["SuperTrend"].dropna().iloc[-1])
            st_dir = "Bullish" if last_close >= st_val else "Bearish"
            state_rows.append({"Indicator": "SuperTrend", "Value": f"{st_val:,.2f}", "State": f"{'Bullish' if st_dir == 'Bullish' else 'Bearish'} Trailing Stop", "Event": "—", "Direction": st_dir})

        # Stochastic
        if "Stochastic" in indicators:
            st_k = float(indicators["Stochastic"]["K"].dropna().iloc[-1])
            st_d = float(indicators["Stochastic"]["D"].dropna().iloc[-1])
            st_dir = "Bullish" if st_k >= st_d else "Bearish"
            state_rows.append({"Indicator": "Stochastic (%K/%D)", "Value": f"{st_k:.1f} / {st_d:.1f}", "State": "Overbought" if st_k > 80 else ("Oversold" if st_k < 20 else "Neutral Range"), "Event": "—", "Direction": st_dir})

        # Williams %R
        if "Williams_%R" in indicators:
            wr_val = float(indicators["Williams_%R"].dropna().iloc[-1])
            wr_dir = "Bearish" if wr_val > -20 else ("Bullish" if wr_val < -80 else "Neutral")
            state_rows.append({"Indicator": "Williams %R", "Value": f"{wr_val:.1f}", "State": "Overbought" if wr_val > -20 else ("Oversold" if wr_val < -80 else "Normal Range"), "Event": "—", "Direction": wr_dir})

        # ROC
        if "ROC" in indicators:
            roc_val = float(indicators["ROC"].dropna().iloc[-1])
            roc_dir = "Bullish" if roc_val > 0 else "Bearish"
            state_rows.append({"Indicator": f"ROC ({adv_params['roc_win']})", "Value": f"{roc_val:+.2f}%", "State": "Positive Velocity" if roc_val > 0 else "Negative Velocity", "Event": "—", "Direction": roc_dir})

        # CCI
        if "CCI" in indicators:
            cci_val = float(indicators["CCI"].dropna().iloc[-1])
            cci_dir = "Bullish" if cci_val > 100 else ("Bearish" if cci_val < -100 else ("Bullish" if cci_val > 0 else "Bearish"))
            cci_st = "Overbought (>100)" if cci_val > 100 else ("Oversold (<-100)" if cci_val < -100 else "Neutral Band")
            state_rows.append({"Indicator": "CCI (20)", "Value": f"{cci_val:.1f}", "State": cci_st, "Event": "—", "Direction": cci_dir})

        # Aroon
        if "Aroon" in indicators:
            ar_up = float(indicators["Aroon"]["aroon_up"].dropna().iloc[-1])
            ar_dn = float(indicators["Aroon"]["aroon_down"].dropna().iloc[-1])
            ar_dir = "Bullish" if ar_up >= ar_dn else "Bearish"
            state_rows.append({"Indicator": "Aroon (Up/Down)", "Value": f"{ar_up:.0f} / {ar_dn:.0f}", "State": "Bullish Trend" if ar_up > 70 else ("Bearish Trend" if ar_dn > 70 else "Consolidating"), "Event": "—", "Direction": ar_dir})

        # Bollinger %B
        if "Bollinger_%B" in indicators:
            pct_b_val = float(indicators["Bollinger_%B"].dropna().iloc[-1])
            b_dir = "Bullish" if pct_b_val >= 0.5 else "Bearish"
            b_st = "Above Upper Band" if pct_b_val > 1.0 else ("Below Lower Band" if pct_b_val < 0.0 else "Inside Band")
            state_rows.append({"Indicator": "Bollinger %B", "Value": f"{pct_b_val:.2f}", "State": b_st, "Event": "—", "Direction": b_dir})

        # Historical Volatility
        if "Historical_Volatility" in indicators:
            hv_val = float(indicators["Historical_Volatility"].dropna().iloc[-1])
            hv_st = "High Volatility" if hv_val > 30 else ("Low Volatility" if hv_val < 15 else "Moderate Volatility")
            state_rows.append({"Indicator": "Hist Vol (30D Ann)", "Value": f"{hv_val:.1f}%", "State": hv_st, "Event": "—", "Direction": "Neutral"})

        # CMF
        if "CMF" in indicators:
            cmf_val = float(indicators["CMF"].dropna().iloc[-1])
            cmf_dir = "Bullish" if cmf_val > 0.05 else ("Bearish" if cmf_val < -0.05 else "Neutral")
            state_rows.append({"Indicator": "CMF (20)", "Value": f"{cmf_val:+.3f}", "State": "Net Inflow" if cmf_val > 0 else "Net Outflow", "Event": "—", "Direction": cmf_dir})

        # MFI
        if "MFI" in indicators:
            mfi_val = float(indicators["MFI"].dropna().iloc[-1])
            mfi_dir = "Bullish" if mfi_val > 50 else "Bearish"
            state_rows.append({"Indicator": "MFI (14)", "Value": f"{mfi_val:.1f}", "State": "Overbought" if mfi_val > 80 else ("Oversold" if mfi_val < 20 else "Neutral Flow"), "Event": "—", "Direction": mfi_dir})

        # Ease of Movement
        if "Ease_of_Movement" in indicators:
            emv_val = float(indicators["Ease_of_Movement"].dropna().iloc[-1])
            emv_dir = "Bullish" if emv_val >= 0 else "Bearish"
            state_rows.append({"Indicator": "Ease of Movement", "Value": f"{emv_val:+.2e}", "State": "Light Resistance Up" if emv_val > 0 else "Downward Flow", "Event": "—", "Direction": emv_dir})

        # Force Index
        if "Force_Index" in indicators:
            fi_val = float(indicators["Force_Index"].dropna().iloc[-1])
            fi_dir = "Bullish" if fi_val >= 0 else "Bearish"
            state_rows.append({"Indicator": "Elder Force Index", "Value": f"{fi_val:,.0f}", "State": "Positive Force" if fi_val > 0 else "Negative Force", "Event": "—", "Direction": fi_dir})

        # Volume Oscillator
        if "Volume_Oscillator" in indicators:
            vo_val = float(indicators["Volume_Oscillator"].dropna().iloc[-1])
            vo_dir = "Bullish" if vo_val >= 0 else "Neutral"
            state_rows.append({"Indicator": "Volume Osc (3/10)", "Value": f"{vo_val:+.1f}%", "State": "Expanding Volume" if vo_val > 0 else "Contracting Volume", "Event": "—", "Direction": vo_dir})

        state_df = pd.DataFrame(state_rows)

        # Style and render dataframe
        st.dataframe(
            state_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Indicator": st.column_config.TextColumn("Indicator", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="small"),
                "State": st.column_config.TextColumn("State", width="medium"),
                "Event": st.column_config.TextColumn("Event", width="small"),
                "Direction": st.column_config.TextColumn("Direction", width="small"),
            },
        )

    with bot_col2:
        st.markdown("### 🎯 Technical Confluence & Levels")

        # Key Pivot Levels Display
        pivot_html = (
            f'<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;">'
            f'<div style="font-size: 0.76rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 10px;">Key Pivot Levels</div>'
            f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.92rem; line-height: 1.8;">'
            f'<div style="color: #EF4444;">• Resistance 2: <strong>{currency_sym}{pivots["Resistance 2"]:,.2f}</strong></div>'
            f'<div style="color: #F87171;">• Resistance 1: <strong>{currency_sym}{pivots["Resistance 1"]:,.2f}</strong></div>'
            f'<div style="color: #38BDF8;">• Current Close: <strong>{currency_sym}{pivots["Current Close"]:,.2f}</strong></div>'
            f'<div style="color: #34D399;">• Support 1: <strong>{currency_sym}{pivots["Support 1"]:,.2f}</strong></div>'
            f'<div style="color: #10B981;">• Support 2: <strong>{currency_sym}{pivots["Support 2"]:,.2f}</strong></div>'
            f'</div></div>'
        )
        st.markdown(pivot_html, unsafe_allow_html=True)

        # Confluence Counts
        bull_count = sum(1 for r in state_rows if r["Direction"] == "Bullish")
        bear_count = sum(1 for r in state_rows if r["Direction"] == "Bearish")
        neut_count = sum(1 for r in state_rows if r["Direction"] == "Neutral")

        confluence_html = (
            f'<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px 20px;">'
            f'<div style="font-size: 0.76rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 10px;">Transparent Confluence Summary</div>'
            f'<div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.9rem;">'
            f'<div style="display: flex; align-items: center; gap: 8px;">'
            f'<span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #10B981;"></span>'
            f'<span style="color: #F1F5F9;">Bullish Signals:</span>'
            f'<strong style="color: #10B981; font-family: \'JetBrains Mono\', monospace;">{bull_count}</strong>'
            f'</div>'
            f'<div style="display: flex; align-items: center; gap: 8px;">'
            f'<span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #EF4444;"></span>'
            f'<span style="color: #F1F5F9;">Bearish Signals:</span>'
            f'<strong style="color: #EF4444; font-family: \'JetBrains Mono\', monospace;">{bear_count}</strong>'
            f'</div>'
            f'<div style="display: flex; align-items: center; gap: 8px;">'
            f'<span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #F59E0B;"></span>'
            f'<span style="color: #F1F5F9;">Neutral Signals:</span>'
            f'<strong style="color: #F59E0B; font-family: \'JetBrains Mono\', monospace;">{neut_count}</strong>'
            f'</div>'
            f'</div></div>'
        )
        st.markdown(confluence_html, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Quantitative Strategy Explainer & Guide
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📖 Quantitative Strategy Explainer & Guide")
    st.caption("Institutional technical playbook detailing quantitative philosophy, execution triggers, risk protocols, and indicator mechanics.")

    STRATEGY_ICONS = {
        "Standard": "⚖️",
        "Swing Trading": "🌊",
        "Trend Following": "🚀",
        "Breakout": "💥",
        "Mean Reversion": "🔄",
        "Momentum": "⚡",
        "Volatility": "🌪️",
        "Volume Analysis": "📦",
        "Custom": "🛠️",
    }

    def _render_strategy_blueprint(p_key: str, p_data: dict, show_title: bool = True):
        """Render a structured institutional quantitative strategy blueprint card."""
        icon = STRATEGY_ICONS.get(p_key, "📊")
        if show_title:
            st.markdown(f"#### **{icon} {p_data['title']}**")
            st.markdown(f"*{p_data['description']}*")

        b_regime = p_data.get("regime", "General Market")
        b_tf = p_data.get("timeframe", "Daily / 4H")
        b_wr = p_data.get("win_rate", "50% - 60%")
        badges_html = (
            f'<div style="display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 16px 0;">'
            f'<span style="background: rgba(56, 189, 248, 0.12); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; font-weight: 600;">Regime: {b_regime}</span>'
            f'<span style="background: rgba(168, 85, 247, 0.12); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; font-weight: 600;">Timeframe: {b_tf}</span>'
            f'<span style="background: rgba(16, 185, 129, 0.12); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; font-weight: 600;">Win Rate & Payoff: {b_wr}</span>'
            f'</div>'
        )
        st.markdown(badges_html, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 💡 Quantitative Philosophy & Alpha Edge")
            st.info(p_data["philosophy"])

            st.markdown("##### 🚦 Mathematical Entry Execution Triggers")
            entry_items = p_data.get("entry_rules", [p_data.get("entry_exit", "Confluence confirmation.")])
            for rule in entry_items:
                st.markdown(f"- **{rule}**")

            st.markdown("##### 🛠️ Core Indicator Stack")
            st.markdown(f"- **Overlays:** `{', '.join(p_data['overlays'])}`\n- **Subpanels:** `{', '.join(p_data['subpanels'])}`")

        with c2:
            st.markdown("##### 🎯 Profit Targets & Invalidation Exit Rules")
            exit_items = p_data.get("exit_rules", ["Exit on signal reversal or trailing stop."])
            for rule in exit_items:
                st.markdown(f"- **{rule}**")

            st.markdown("##### 🛡️ Risk Management & Dynamic Position Sizing")
            st.warning(p_data.get("risk_management", "Fixed fractional risk."))

            st.markdown("##### ⚠️ Key Pitfalls & Regime Vulnerabilities")
            st.error(p_data.get("pitfalls", "Whipsaws during regime shifts."))

    guide_tab1, guide_tab2, guide_tab3, guide_tab4 = st.tabs([
        f"🎯 Active Blueprint: {active_preset}",
        "📚 Strategy Playbook (All 9 Presets)",
        "⚡ 4-Pillar Confluence Framework",
        "🔬 Indicator Encyclopedia (46 Indicators)",
    ])

    with guide_tab1:
        active_info = PRESETS.get(active_preset, PRESETS["Standard"])
        _render_strategy_blueprint(active_preset, active_info)

    with guide_tab2:
        st.markdown("#### 📚 Comprehensive Institutional Strategy Playbook")
        st.markdown(
            "Quantitative funds construct multi-strategy portfolios by matching distinct analytical models to specific market regimes. "
            "No single strategy succeeds in all market conditions: **Trend Following** thrives in persistent directional expansions but suffers drag in chop, "
            "while **Mean Reversion** extracts alpha from rangebound oscillations but risks severe drawdown during breakout regime shifts."
        )

        # 1. Comparative Strategy Matrix Table
        st.markdown("##### 📊 Institutional Strategy Master Comparison Matrix")
        matrix_rows = []
        for p_k, p_v in PRESETS.items():
            icon = STRATEGY_ICONS.get(p_k, "📊")
            matrix_rows.append({
                "Strategy Model": f"{icon} {p_v['title']}",
                "Target Market Regime": p_v.get("regime", "General"),
                "Horizon": p_v.get("timeframe", "Daily"),
                "Win Rate & Payoff": p_v.get("win_rate", "50-60%"),
                "Primary Indicators": f"{', '.join(p_v['overlays'])} | {', '.join(p_v['subpanels'])}",
            })
        st.dataframe(
            pd.DataFrame(matrix_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Strategy Model": st.column_config.TextColumn("Strategy Model", width="medium"),
                "Target Market Regime": st.column_config.TextColumn("Target Market Regime", width="medium"),
                "Horizon": st.column_config.TextColumn("Time Horizon", width="small"),
                "Win Rate & Payoff": st.column_config.TextColumn("Historical Profile", width="small"),
                "Primary Indicators": st.column_config.TextColumn("Key Indicator Stack", width="large"),
            },
        )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # 2. Interactive Single Strategy Deep Dive Inspector
        st.markdown("##### 🔍 Strategy Deep-Dive Inspector")
        ins_col1, ins_col2 = st.columns([1.2, 2.0])
        with ins_col1:
            inspect_preset = st.selectbox(
                "Select Strategy Playbook to Deep-Dive:",
                options=list(PRESETS.keys()),
                index=list(PRESETS.keys()).index(active_preset),
                key="guide_playbook_inspect_select",
            )
        with ins_col2:
            st.caption(f"Currently inspecting full quantitative mandate, execution checklist, and risk parameters for **{inspect_preset}**.")

        inspect_info = PRESETS[inspect_preset]
        _render_strategy_blueprint(inspect_preset, inspect_info, show_title=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # 3. Complete In-Depth Strategy Dossiers (All 9 Strategies)
        st.markdown("##### 📂 Browse All 9 Strategy Dossiers in Detail")
        st.caption("Expand any quantitative strategy below to study its full mathematical execution rules, risk management protocols, and market microstructure foundations.")

        for p_k, p_v in PRESETS.items():
            icon = STRATEGY_ICONS.get(p_k, "📊")
            with st.expander(f"{icon} **{p_v['title']}** — *Regime: {p_v.get('regime', 'General')}*", expanded=False):
                _render_strategy_blueprint(p_k, p_v, show_title=False)

    with guide_tab3:
        st.markdown("#### ⚡ The 4-Pillar Quantitative Confluence Framework")
        st.markdown(
            "High-frequency quantitative hedge funds and proprietary desks do not rely on single indicator signals. "
            "Instead, signals are gated across **four independent analytical domains** to minimize false positives and maximize statistical expectancy."
        )

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.markdown(
                f'<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 14px; min-height: 220px;">'
                f'<div style="color: #38BDF8; font-weight: 700; font-size: 0.95rem; margin-bottom: 6px;">1. Trend Anchor</div>'
                f'<div style="color: #94A3B8; font-size: 0.8rem; margin-bottom: 8px;">Directional filter</div>'
                f'<div style="font-size: 0.82rem; color: #E2E8F0; line-height: 1.4;">Ensures trades align with the dominant institutional macro vector.<br><br><b>Key Tools:</b> 20/50 SMA, SuperTrend, EMA Ribbon, Ichimoku Cloud.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with p2:
            st.markdown(
                f'<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 10px; padding: 14px; min-height: 220px;">'
                f'<div style="color: #C084FC; font-weight: 700; font-size: 0.95rem; margin-bottom: 6px;">2. Momentum Trigger</div>'
                f'<div style="color: #94A3B8; font-size: 0.8rem; margin-bottom: 8px;">Velocity & timing</div>'
                f'<div style="font-size: 0.82rem; color: #E2E8F0; line-height: 1.4;">Trend tells you <i>where</i>; momentum tells you <i>when</i>. Filters entries before momentum exhausts.<br><br><b>Key Tools:</b> RSI, MACD, Stochastic %K/%D, ROC, TRIX.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with p3:
            st.markdown(
                f'<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 14px; min-height: 220px;">'
                f'<div style="color: #F59E0B; font-weight: 700; font-size: 0.95rem; margin-bottom: 6px;">3. Volatility Gate</div>'
                f'<div style="color: #94A3B8; font-size: 0.8rem; margin-bottom: 8px;">Dispersion & bounds</div>'
                f'<div style="font-size: 0.82rem; color: #E2E8F0; line-height: 1.4;">Evaluates statistical overextension. Avoids buying into outer 2-std bands.<br><br><b>Key Tools:</b> Bollinger Bands, Keltner, ATR, Donchian Channels.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with p4:
            st.markdown(
                f'<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 14px; min-height: 220px;">'
                f'<div style="color: #10B981; font-weight: 700; font-size: 0.95rem; margin-bottom: 6px;">4. Volume Flow</div>'
                f'<div style="color: #94A3B8; font-size: 0.8rem; margin-bottom: 8px;">Institutional footprint</div>'
                f'<div style="font-size: 0.82rem; color: #E2E8F0; line-height: 1.4;">Validates whether real institutional smart money is participating.<br><br><b>Key Tools:</b> VWAP, RVOL (> 1.2x), CMF, OBV, MFI.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("##### 📊 Confluence Scorecard & Sizing Rubric")

        rubric_df = pd.DataFrame([
            {"Confluence Score": "4 / 4 (Prime Alignment)", "Setup Grade": "A+ Institutional Grade", "Position Sizing": "100% of Max Risk Allocation", "Execution Protocol": "Immediate market entry; trailing stops enabled."},
            {"Confluence Score": "3 / 4 (Solid Alignment)", "Setup Grade": "Standard Tradeable Setup", "Position Sizing": "50% - 75% Risk Allocation", "Execution Protocol": "Enter on limit pullback to EMA or VWAP."},
            {"Confluence Score": "2 / 4 (Mixed Signals)", "Setup Grade": "Low Conviction / Choppy", "Position Sizing": "25% Tactical Pilot Size", "Execution Protocol": "Wait for breakout resolution or stand aside."},
            {"Confluence Score": "0-1 / 4 (Conflicting)", "Setup Grade": "Unfavorable / Chop Zone", "Position Sizing": "0% (Strictly Cash / No Trade)", "Execution Protocol": "Preserve capital; avoid whipsaw noise."},
        ])
        st.dataframe(rubric_df, hide_index=True, width="stretch")

        st.markdown("##### 📐 Mathematical Position Sizing Formula (Volatility Adjusted)")
        st.latex(r"\text{Shares} = \frac{\text{Account Equity} \times \text{Risk Tolerance \%}}{\text{Entry Price} - \text{Stop Loss}} = \frac{\text{Account Equity} \times 0.010}{1.5 \times \text{ATR}(14)}")

    with guide_tab4:
        st.markdown("#### 🔬 Master Technical Indicator Encyclopedia (46 Indicators)")
        st.caption("Complete quantitative reference of all 18 Price Axis Overlays and 28 Analytical Subpanels available in this terminal.")

        enc_cat = st.radio(
            "Filter Encyclopedia by Indicator Family:",
            ["All Indicators (46)", "Moving Averages & Trend (18)", "Momentum & Velocity (8)", "Volatility & Channels (8)", "Volume & Order Flow (12)"],
            horizontal=True,
        )

        encyclopedia_data = [
            # Moving Averages & Trend
            {"Indicator": "SMA", "Family": "Trend", "Type": "Overlay", "Formula / Math": "Arithmetic mean of closing prices over window N", "Bullish Signal": "Price crosses above SMA; Golden Cross (50>200)", "Bearish Signal": "Price drops below SMA; Death Cross (50<200)"},
            {"Indicator": "EMA", "Family": "Trend", "Type": "Overlay", "Formula / Math": "Exponentially weighted moving average with multiplier 2/(N+1)", "Bullish Signal": "Price closes above rising EMA", "Bearish Signal": "Price breaks below declining EMA"},
            {"Indicator": "WMA", "Family": "Trend", "Type": "Overlay", "Formula / Math": "Linear weighted moving average giving higher priority to recent bars", "Bullish Signal": "Price pulls back to rising WMA support", "Bearish Signal": "Price rejection at descending WMA"},
            {"Indicator": "HMA", "Family": "Trend", "Type": "Overlay", "Formula / Math": "WMA(2*WMA(N/2) - WMA(N), sqrt(N)) zero-lag smoothing", "Bullish Signal": "HMA turns upward with green color shift", "Bearish Signal": "HMA turns downward with red color shift"},
            {"Indicator": "VWMA", "Family": "Trend", "Type": "Overlay", "Formula / Math": "Sum(Close * Volume) / Sum(Volume) over window N", "Bullish Signal": "Price above VWMA confirms volume-backed trend", "Bearish Signal": "Price below VWMA confirms institutional selling"},
            {"Indicator": "DEMA", "Family": "Trend", "Type": "Overlay", "Formula / Math": "2*EMA - EMA(EMA); eliminates lag in high volatility", "Bullish Signal": "Price crosses above DEMA line", "Bearish Signal": "Price breaks below DEMA line"},
            {"Indicator": "TEMA", "Family": "Trend", "Type": "Overlay", "Formula / Math": "3*EMA - 3*EMA(EMA) + EMA(EMA(EMA)) triple smoothed", "Bullish Signal": "Fastest trend continuation trigger", "Bearish Signal": "Decisive breakdown below TEMA"},
            {"Indicator": "SuperTrend", "Family": "Trend", "Type": "Overlay", "Formula / Math": "(High+Low)/2 +/- Multiplier * ATR trailing band", "Bullish Signal": "Green line below price acts as trailing stop", "Bearish Signal": "Red line above price acts as trailing resistance"},
            {"Indicator": "Parabolic SAR", "Family": "Trend", "Type": "Overlay", "Formula / Math": "Acceleration Factor stop and reverse mechanism", "Bullish Signal": "Dots flip below candlestick low", "Bearish Signal": "Dots flip above candlestick high"},
            {"Indicator": "Ichimoku Cloud", "Family": "Trend", "Type": "Overlay", "Formula / Math": "Tenkan, Kijun, Senkou Span A/B equilibrium cloud", "Bullish Signal": "Price clears Kumo Cloud; Tenkan > Kijun", "Bearish Signal": "Price drops below Kumo Cloud; Tenkan < Kijun"},
            {"Indicator": "ZigZag", "Family": "Trend", "Type": "Overlay", "Formula / Math": "Filters price changes smaller than threshold % to link pivots", "Bullish Signal": "Higher High & Higher Low structural sequence", "Bearish Signal": "Lower High & Lower Low structural sequence"},
            {"Indicator": "Fractals", "Family": "Trend", "Type": "Overlay", "Formula / Math": "Williams 5-bar high/low geometric extrema patterns", "Bullish Signal": "Green triangle-up marker under swing low", "Bearish Signal": "Red triangle-down marker over swing high"},
            {"Indicator": "Pivot Points", "Family": "Trend", "Type": "Overlay", "Formula / Math": "Classic floor trader pivot: (H+L+C)/3 with R1/R2 and S1/S2", "Bullish Signal": "Bounce from S1/S2 support; break above Pivot", "Bearish Signal": "Rejection from R1/R2 resistance; drop below Pivot"},

            # Momentum
            {"Indicator": "RSI", "Family": "Momentum", "Type": "Subpanel", "Formula / Math": "100 - (100 / (1 + Avg Gain / Avg Loss)) over 14 bars", "Bullish Signal": "Exiting oversold (< 30) or breaking above 50", "Bearish Signal": "Exiting overbought (> 70) or breaking below 50"},
            {"Indicator": "MACD", "Family": "Momentum", "Type": "Subpanel", "Formula / Math": "EMA(12) - EMA(26); Signal = EMA(9) of MACD", "Bullish Signal": "MACD line crosses above Signal; expanding histogram", "Bearish Signal": "MACD line crosses below Signal; contracting histogram"},
            {"Indicator": "Stochastic", "Family": "Momentum", "Type": "Subpanel", "Formula / Math": "%K = 100*(C-L14)/(H14-L14); %D = 3-period SMA of %K", "Bullish Signal": "%K crosses above %D from below 20 (oversold)", "Bearish Signal": "%K crosses below %D from above 80 (overbought)"},
            {"Indicator": "Williams %R", "Family": "Momentum", "Type": "Subpanel", "Formula / Math": "-100 * (High_N - Close) / (High_N - Low_N)", "Bullish Signal": "Curling upward from extreme oversold (< -80)", "Bearish Signal": "Curling downward from extreme overbought (> -20)"},
            {"Indicator": "ROC", "Family": "Momentum", "Type": "Subpanel", "Formula / Math": "100 * (Close - Close[N]) / Close[N] price velocity", "Bullish Signal": "Crosses above zero baseline with upward slope", "Bearish Signal": "Crosses below zero baseline with downward slope"},
            {"Indicator": "CMO", "Family": "Momentum", "Type": "Subpanel", "Formula / Math": "Chande Momentum: 100 * (Sum(Up) - Sum(Down)) / Total", "Bullish Signal": "Crosses above 0 or bounces out of oversold (<-50)", "Bearish Signal": "Crosses below 0 or rejects from overbought (>+50)"},
            {"Indicator": "TRIX", "Family": "Momentum", "Type": "Subpanel", "Formula / Math": "1-period % change of triple-smoothed EMA(Close)", "Bullish Signal": "Crosses above zero baseline (bullish acceleration)", "Bearish Signal": "Crosses below zero baseline (bearish deceleration)"},
            {"Indicator": "CCI", "Family": "Momentum", "Type": "Subpanel", "Formula / Math": "(Typical Price - SMA(TP)) / (0.015 * Mean Absolute Deviation)", "Bullish Signal": "Crosses above +100 breakout or rebounds from -100", "Bearish Signal": "Rejection from +100 or breakdown below -100"},

            # Volatility
            {"Indicator": "Bollinger Bands", "Family": "Volatility", "Type": "Overlay", "Formula / Math": "SMA(20) +/- 2.0 * Rolling Std Dev of Close", "Bullish Signal": "Band squeeze breakout; tag of lower band in range", "Bearish Signal": "Band squeeze breakdown; tag of upper band in range"},
            {"Indicator": "Keltner Channels", "Family": "Volatility", "Type": "Overlay", "Formula / Math": "EMA(20) +/- 2.0 * ATR(10) envelopes", "Bullish Signal": "Price piercing upper channel confirms strong trend", "Bearish Signal": "Price piercing lower channel confirms downtrend"},
            {"Indicator": "Donchian Channels", "Family": "Volatility", "Type": "Overlay", "Formula / Math": "20-day Highest High and 20-day Lowest Low channels", "Bullish Signal": "Breakout above 20-day Upper Donchian level", "Bearish Signal": "Breakdown below 20-day Lower Donchian level"},
            {"Indicator": "Price Envelopes", "Family": "Volatility", "Type": "Overlay", "Formula / Math": "SMA(20) +/- 2.5% fixed percentage boundary bands", "Bullish Signal": "Bounce from lower 2.5% band", "Bearish Signal": "Rejection from upper 2.5% band"},
            {"Indicator": "ATR", "Family": "Volatility", "Type": "Subpanel", "Formula / Math": "Exponential average of True Range: Max(H-L, |H-C1|, |L-C1|)", "Bullish Signal": "ATR rising during breakout verifies volume push", "Bearish Signal": "ATR expanding during market crash highlights risk"},
            {"Indicator": "Bollinger Bandwidth", "Family": "Volatility", "Type": "Subpanel", "Formula / Math": "((Upper Band - Lower Band) / Middle Band) * 100", "Bullish Signal": "Bandwidth expanding out of multi-week lows (<10%)", "Bearish Signal": "Parabolic bandwidth spike indicates climax"},
            {"Indicator": "Bollinger %B", "Family": "Volatility", "Type": "Subpanel", "Formula / Math": "(Close - Lower Band) / (Upper Band - Lower Band)", "Bullish Signal": "%B > 1.0 indicates breakout; %B > 0.5 bullish bias", "Bearish Signal": "%B < 0.0 indicates breakdown; %B < 0.5 bearish bias"},
            {"Indicator": "Historical Volatility", "Family": "Volatility", "Type": "Subpanel", "Formula / Math": "Annualized standard deviation of daily log returns: Std*sqrt(252)", "Bullish Signal": "HV compression precedes explosive directional moves", "Bearish Signal": "HV > 40% signals dangerous market volatility"},

            # Volume & Order Flow
            {"Indicator": "VWAP", "Family": "Volume", "Type": "Overlay", "Formula / Math": "Cumsum(Typical Price * Volume) / Cumsum(Volume)", "Bullish Signal": "Price holding above VWAP confirms institutional buyers", "Bearish Signal": "Price below VWAP indicates distribution"},
            {"Indicator": "RVOL", "Family": "Volume", "Type": "Subpanel", "Formula / Math": "Current Volume / 20-day Average Volume", "Bullish Signal": "RVOL >= 1.5x validates genuine institutional breakouts", "Bearish Signal": "RVOL < 0.8x warns of false breakout trap"},
            {"Indicator": "OBV", "Family": "Volume", "Type": "Subpanel", "Formula / Math": "Cumsum(Volume * Sign(Close - Close_prev))", "Bullish Signal": "OBV making new highs before price (accumulation)", "Bearish Signal": "OBV divergence making lower lows (distribution)"},
            {"Indicator": "CMF", "Family": "Volume", "Type": "Subpanel", "Formula / Math": "Sum(Money Flow Volume, 20) / Sum(Volume, 20)", "Bullish Signal": "CMF > +0.05 indicates strong institutional net inflow", "Bearish Signal": "CMF < -0.05 indicates persistent institutional outflow"},
            {"Indicator": "MFI", "Family": "Volume", "Type": "Subpanel", "Formula / Math": "Volume-weighted RSI: 100 - (100 / (1 + Pos MF / Neg MF))", "Bullish Signal": "Exiting oversold (< 20) with high volume surge", "Bearish Signal": "Exiting overbought (> 80) with heavy volume exhaustion"},
            {"Indicator": "ADL", "Family": "Volume", "Type": "Subpanel", "Formula / Math": "Cumulative Accumulation / Distribution Line based on close position", "Bullish Signal": "Rising ADL confirms price trend is backed by volume", "Bearish Signal": "Falling ADL warns of institutional unloading"},
            {"Indicator": "Chaikin Oscillator", "Family": "Volume", "Type": "Subpanel", "Formula / Math": "EMA(3) of ADL - EMA(10) of ADL", "Bullish Signal": "Crosses above zero with positive green histogram", "Bearish Signal": "Crosses below zero with negative red histogram"},
            {"Indicator": "PVT", "Family": "Volume", "Type": "Subpanel", "Formula / Math": "Cumulative sum of (% Price Change * Volume)", "Bullish Signal": "PVT slope turning positive and breaking resistance", "Bearish Signal": "PVT slope breaking downward ahead of price"},
            {"Indicator": "Ease of Movement", "Family": "Volume", "Type": "Subpanel", "Formula / Math": "Arms EMV: (Midpoint Change) / (Volume / High-Low Range)", "Bullish Signal": "EMV > 0 shows price rising easily on light resistance", "Bearish Signal": "EMV < 0 shows price falling on heavy volume pressure"},
            {"Indicator": "Force Index", "Family": "Volume", "Type": "Subpanel", "Formula / Math": "Elder Force Index: EMA(13) of (Close.diff() * Volume)", "Bullish Signal": "Force Index surges positively (strong buying impulse)", "Bearish Signal": "Force Index plunges negatively (strong selling impulse)"},
            {"Indicator": "Volume Oscillator", "Family": "Volume", "Type": "Subpanel", "Formula / Math": "100 * (Fast EMA(3) - Slow EMA(10)) / Slow EMA(10)", "Bullish Signal": "Oscillator > 0 indicates volume expanding on rallies", "Bearish Signal": "Oscillator < 0 indicates volume contracting"},
            {"Indicator": "PVI", "Family": "Volume", "Type": "Subpanel", "Formula / Math": "Positive Volume Index: accumulates only on days volume rises", "Bullish Signal": "PVI above its 255-day EMA confirms bull market", "Bearish Signal": "PVI below its 255-day EMA warns of bear market"},
            {"Indicator": "NVI", "Family": "Volume", "Type": "Subpanel", "Formula / Math": "Negative Volume Index: accumulates only on days volume falls (Smart Money)", "Bullish Signal": "NVI above its 255-day EMA reveals smart money accumulation", "Bearish Signal": "NVI below its 255-day EMA indicates smart money exit"},

            # Strength
            {"Indicator": "ADX", "Family": "Strength", "Type": "Subpanel", "Formula / Math": "Welles Wilder directional movement index: 100 * DX smoothed", "Bullish Signal": "ADX > 25 with +DI > -DI confirms strong uptrend", "Bearish Signal": "ADX > 25 with -DI > +DI confirms strong downtrend"},
            {"Indicator": "Aroon", "Family": "Strength", "Type": "Subpanel", "Formula / Math": "Aroon Up: 100*(25-BarsSinceHigh)/25; Aroon Down: Lows", "Bullish Signal": "Aroon Up > 70 while Aroon Down < 30", "Bearish Signal": "Aroon Down > 70 while Aroon Up < 30"},
            {"Indicator": "Vortex", "Family": "Strength", "Type": "Subpanel", "Formula / Math": "VI+ = Sum(|H - L_prev|) / ATR; VI- = Sum(|L - H_prev|) / ATR", "Bullish Signal": "VI+ crosses above VI- by at least 0.2 margin", "Bearish Signal": "VI- crosses above VI+ by at least 0.2 margin"},
        ]

        enc_df = pd.DataFrame(encyclopedia_data)
        if "Moving Averages" in enc_cat:
            enc_df = enc_df[enc_df["Family"] == "Trend"]
        elif "Momentum" in enc_cat:
            enc_df = enc_df[enc_df["Family"] == "Momentum"]
        elif "Volatility" in enc_cat:
            enc_df = enc_df[enc_df["Family"] == "Volatility"]
        elif "Volume" in enc_cat:
            enc_df = enc_df[enc_df["Family"] == "Volume"]

        st.dataframe(
            enc_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Indicator": st.column_config.TextColumn("Indicator", width="small"),
                "Family": st.column_config.TextColumn("Family", width="small"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Formula / Math": st.column_config.TextColumn("Formula / Core Math", width="medium"),
                "Bullish Signal": st.column_config.TextColumn("Bullish Trigger", width="medium"),
                "Bearish Signal": st.column_config.TextColumn("Bearish Trigger", width="medium"),
            },
        )


if __name__ == "__main__":
    render_page()
