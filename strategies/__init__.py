"""Strategy module: base class, built-in strategies, signal generation."""

from strategies.base import Strategy
from strategies.builtin import get_strategy, STRATEGY_NAMES
from strategies.signals import generate_signals
