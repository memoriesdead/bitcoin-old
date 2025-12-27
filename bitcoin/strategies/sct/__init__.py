"""
SCT - Statistical Certainty Trading
====================================

Win Rate: 50.75%+ Wilson CI lower bound (covers fees + edge)

The Math (Wilson Score Confidence Interval):
  Lower Bound = (p + z²/2n - z*sqrt(p(1-p)/n + z²/4n²)) / (1 + z²/n)

  Where:
    p = observed win rate
    n = sample size
    z = 2.576 (99% confidence)

  TRADE if: lower_bound >= 0.5075 (50.75%)

Leverage: Per-exchange maximum (MEXC 500x, Binance 125x, etc.)
"""

from bitcoin.sct.wilson import wilson_interval, wilson_lower_bound, trades_needed_for_certainty
from bitcoin.sct.certainty import CertaintyChecker, CertaintyStatus, CertaintyResult
from bitcoin.sct.validator import TradeValidator
from bitcoin.sct.config import SCTConfig, get_config

__all__ = [
    'wilson_interval',
    'wilson_lower_bound',
    'trades_needed_for_certainty',
    'CertaintyChecker',
    'CertaintyStatus',
    'CertaintyResult',
    'TradeValidator',
    'SCTConfig',
    'get_config',
]
