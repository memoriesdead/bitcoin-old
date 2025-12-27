"""
Signal Validation Framework
===========================

RenTech approach: Prove the edge before trading.

Components:
- signal_logger: Captures all blockchain signals
- price_logger: Captures BTC price every second
- data_collector: Orchestrates collection
- analysis/: Scripts to calculate edge metrics
"""

from .signal_logger import SignalLogger
from .price_logger import PriceLogger
from .data_collector import DataCollector

__all__ = ['SignalLogger', 'PriceLogger', 'DataCollector']
