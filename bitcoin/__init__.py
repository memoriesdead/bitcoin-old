#!/usr/bin/env python3
"""
BITCOIN HFT Trading Engine
==========================

Core modules:
  - config: Trading configuration
  - signals: Signal generation (CorrelationFormula)
  - trader: Position management (DeterministicTrader)
  - price_feed: Multi-exchange price feeds
  - run: C++ bridge and entry point
"""

from .config import TradingConfig, get_config
from .core import Signal, SignalType, CorrelationFormula
from .core import Position, DeterministicTrader
