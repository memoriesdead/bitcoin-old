"""
Freqtrade Bridge - Direct Blockchain Connection
===============================================

YOUR blockchain feed -> 423 formulas -> trading signals
NO MOCK DATA. NO SLOW EXCHANGE FEEDS.

Architecture:
  BlockchainFeed (10+ WebSocket, millisecond) -> FormulaToIndicator -> RenaissanceLiveStrategy

NOT:
  Freqtrade -> Exchange API -> 1 minute candles (TOO SLOW)
"""

from .formula_adapter import FormulaToIndicator
from .strategy_wrapper import RenaissanceLiveStrategy

# Freqtrade strategy only for backtesting (slow exchange data)
try:
    from .strategy_wrapper import RenaissanceFreqtradeStrategy
except:
    RenaissanceFreqtradeStrategy = None

__all__ = [
    'FormulaToIndicator',
    'RenaissanceLiveStrategy',
    'RenaissanceFreqtradeStrategy',
]
