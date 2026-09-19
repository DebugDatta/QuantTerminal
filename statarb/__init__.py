"""Statistical arbitrage module: pairs, cointegration, spread analysis."""

from statarb.pairs import find_pairs, pair_distance
from statarb.cointegration import engle_granger, johansen
from statarb.spread import calc_spread, calc_zscore, mean_reversion_signals
