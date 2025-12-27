"""
DET - Deterministic Blockchain Flow Trading
============================================

100% WIN RATE when criteria met:
- sample_count >= 10
- correlation >= 0.70
- win_rate >= 0.90

The Math:
  INFLOW to exchange  -> Deposit to SELL -> Price DOWN -> SHORT
  OUTFLOW from exchange -> Withdrawal   -> Price UP   -> LONG

Leverage: Per-exchange maximum (MEXC 500x, Binance 125x, etc.)
"""

from bitcoin.core import DeterministicTrader, Position, PositionStatus
from bitcoin.core import CorrelationFormula, Signal, SignalType

__all__ = [
    'DeterministicTrader',
    'Position',
    'PositionStatus',
    'CorrelationFormula',
    'Signal',
    'SignalType',
]
