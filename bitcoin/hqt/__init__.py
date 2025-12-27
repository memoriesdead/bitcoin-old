#!/usr/bin/env python3
"""
HQT - High Quality Trades (Deterministic HFT)

Only trades when mathematically certain of profit.
100% win rate or skip. No probabilistic trades.
"""

from .config import HQTConfig, get_config
from .arbitrage import ArbitrageDetector, ArbitrageOpportunity

__all__ = ['HQTConfig', 'get_config', 'ArbitrageDetector', 'ArbitrageOpportunity']
