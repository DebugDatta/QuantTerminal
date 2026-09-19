"""All 11 built-in trading strategies."""

import pandas as pd
import numpy as np
from strategies.base import Strategy
from technical.trend import sma, ema
from technical.momentum import rsi, macd
from technical.volatility import bollinger_bands


class BuyAndHold(Strategy):
    name = "Buy & Hold"
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals.iloc[0] = 1
        signals.iloc[-1] = -1
        return signals


class SMACrossover(Strategy):
    name = "SMA Crossover"
    _default_params = {"fast_window": 20, "slow_window": 50}
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = sma(df["Close"], self.params["fast_window"])
        slow = sma(df["Close"], self.params["slow_window"])
        signals = pd.Series(0, index=df.index)
        signals[fast > slow] = 1
        signals[fast < slow] = -1
        return signals.diff().clip(-1, 1).fillna(0)


class EMACrossover(Strategy):
    name = "EMA Crossover"
    _default_params = {"fast_window": 12, "slow_window": 26}
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = ema(df["Close"], self.params["fast_window"])
        slow = ema(df["Close"], self.params["slow_window"])
        signals = pd.Series(0, index=df.index)
        signals[fast > slow] = 1
        signals[fast < slow] = -1
        return signals.diff().clip(-1, 1).fillna(0)


class RSIStrategy(Strategy):
    name = "RSI"
    _default_params = {"rsi_window": 14, "oversold": 30, "overbought": 70}
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        r = rsi(df["Close"], self.params["rsi_window"])
        signals = pd.Series(0, index=df.index)
        in_oversold = False
        for i in range(1, len(r)):
            if r.iloc[i - 1] < self.params["oversold"] and r.iloc[i] >= self.params["oversold"]:
                in_oversold = True
            if in_oversold and r.iloc[i] > self.params["oversold"]:
                signals.iloc[i] = 1
                in_oversold = False
        in_overbought = False
        for i in range(1, len(r)):
            if r.iloc[i - 1] > self.params["overbought"] and r.iloc[i] <= self.params["overbought"]:
                in_overbought = True
            if in_overbought and r.iloc[i] < self.params["overbought"]:
                signals.iloc[i] = -1
                in_overbought = False
        return signals


class MACDStrategy(Strategy):
    name = "MACD"
    _default_params = {"fast": 12, "slow": 26, "signal": 9}
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        m = macd(df["Close"], self.params["fast"], self.params["slow"], self.params["signal"])
        cross = m["macd"] - m["signal"]
        signals = pd.Series(0, index=df.index)
        signals[cross > 0] = 1
        signals[cross < 0] = -1
        return signals.diff().clip(-1, 1).fillna(0)


class BollingerBandsStrategy(Strategy):
    name = "Bollinger Bands"
    _default_params = {"window": 20, "num_std": 2}
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        bb = bollinger_bands(df["Close"], self.params["window"], self.params["num_std"])
        signals = pd.Series(0, index=df.index)
        below_lower = False
        for i in range(1, len(df)):
            if df["Close"].iloc[i] < bb["lower"].iloc[i]:
                below_lower = True
            if below_lower and df["Close"].iloc[i] > bb["lower"].iloc[i]:
                signals.iloc[i] = 1
                below_lower = False
        above_upper = False
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > bb["upper"].iloc[i]:
                above_upper = True
            if above_upper and df["Close"].iloc[i] < bb["upper"].iloc[i]:
                signals.iloc[i] = -1
                above_upper = False
        return signals


class DonchianBreakout(Strategy):
    name = "Donchian Breakout"
    _default_params = {"window": 20}
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        upper = df["High"].rolling(self.params["window"]).max()
        lower = df["Low"].rolling(self.params["window"]).min()
        signals = pd.Series(0, index=df.index)
        signals[df["Close"] > upper.shift(1)] = 1
        signals[df["Close"] < lower.shift(1)] = -1
        return signals


class MomentumStrategy(Strategy):
    name = "Momentum"
    _default_params = {"window": 20}
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ret = df["Close"].pct_change(self.params["window"])
        signals = pd.Series(0, index=df.index)
        signals[ret > 0] = 1
        signals[ret < 0] = -1
        return signals


class MeanReversionZScore(Strategy):
    name = "Mean Reversion Z-Score"
    _default_params = {"window": 20, "entry_z": 2.0, "exit_z": 0.5}
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        mean = df["Close"].rolling(self.params["window"]).mean()
        std = df["Close"].rolling(self.params["window"]).std()
        z = (df["Close"] - mean) / (std + 1e-10)
        signals = pd.Series(0, index=df.index)
        in_position = 0
        for i in range(len(z)):
            if in_position == 0:
                if z.iloc[i] < -self.params["entry_z"]:
                    signals.iloc[i] = 1
                    in_position = 1
                elif z.iloc[i] > self.params["entry_z"]:
                    signals.iloc[i] = -1
                    in_position = -1
            elif in_position == 1:
                if z.iloc[i] > -self.params["exit_z"]:
                    signals.iloc[i] = -1
                    in_position = 0
                else:
                    signals.iloc[i] = 1
            elif in_position == -1:
                if z.iloc[i] < self.params["exit_z"]:
                    signals.iloc[i] = 1
                    in_position = 0
                else:
                    signals.iloc[i] = -1
        return signals


class PairTrading(Strategy):
    name = "Spread Mean Reversion"
    _default_params = {"window": 20, "entry_z": 2.0, "exit_z": 0.5}
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        spread = df["Close"] - df["Close"].rolling(self.params["window"]).mean()
        std = spread.rolling(self.params["window"]).std()
        z = spread / (std + 1e-10)
        signals = pd.Series(0, index=df.index)
        in_position = 0
        for i in range(len(z)):
            if in_position == 0:
                if z.iloc[i] < -self.params["entry_z"]:
                    signals.iloc[i] = 1
                    in_position = 1
                elif z.iloc[i] > self.params["entry_z"]:
                    signals.iloc[i] = -1
                    in_position = -1
            elif in_position == 1:
                if z.iloc[i] > -self.params["exit_z"]:
                    signals.iloc[i] = -1
                    in_position = 0
                else:
                    signals.iloc[i] = 1
            elif in_position == -1:
                if z.iloc[i] < self.params["exit_z"]:
                    signals.iloc[i] = 1
                    in_position = 0
                else:
                    signals.iloc[i] = -1
        return signals


class BreakoutStrategy(Strategy):
    name = "Breakout"
    params = {"window": 20}
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        high = df["High"].rolling(self.params["window"]).max()
        low = df["Low"].rolling(self.params["window"]).min()
        signals = pd.Series(0, index=df.index)
        signals[df["Close"] > high.shift(1)] = 1
        signals[df["Close"] < low.shift(1)] = -1
        return signals


STRATEGY_MAP = {
    "Buy & Hold": BuyAndHold,
    "SMA Crossover": SMACrossover,
    "EMA Crossover": EMACrossover,
    "RSI": RSIStrategy,
    "MACD": MACDStrategy,
    "Bollinger Bands": BollingerBandsStrategy,
    "Donchian Breakout": DonchianBreakout,
    "Momentum": MomentumStrategy,
    "Mean Reversion Z-Score": MeanReversionZScore,
    "Spread Mean Reversion": PairTrading,
    "Breakout": BreakoutStrategy,
}

STRATEGY_NAMES = list(STRATEGY_MAP.keys())


def get_strategy(name: str, params: dict = None) -> Strategy:
    """Get a strategy instance by name."""
    cls = STRATEGY_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}")
    strategy = cls()
    if params:
        strategy.params.update(params)
    return strategy
