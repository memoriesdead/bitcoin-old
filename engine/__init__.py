"""
CORE ENGINE - 5 Modules Only
=============================
config.py              - Configuration
multi_price_feed.py    - 12 Exchange Prices  
correlation_formula.py - Signal Generation
deterministic_trader.py - Position Management
cpp_master_pipeline.py - C++ Bridge
"""

from .config import TradingConfig, get_config, CONFIG
from .multi_price_feed import MultiExchangePriceFeed, ExchangePrice
from .correlation_formula import Signal, SignalType, CorrelationFormula
from .deterministic_trader import Position, DeterministicTrader

__all__ = [
    'TradingConfig', 'get_config', 'CONFIG',
    'MultiExchangePriceFeed', 'ExchangePrice',
    'Signal', 'SignalType', 'CorrelationFormula',
    'Position', 'DeterministicTrader',
]
