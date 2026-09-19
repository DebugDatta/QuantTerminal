"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """Base class for trading strategies."""

    name: str = "Base"
    _default_params: dict = {}

    def __init__(self):
        self.params = self._default_params.copy()

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV data.

        Returns a Series of: 1 (buy), -1 (sell), 0 (hold).
        """
        pass

    def get_param_defaults(self) -> dict:
        return self._default_params.copy()
