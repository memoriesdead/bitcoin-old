"""
DET Strategy Configuration
==========================

NOTE: All config now lives in bitcoin/config.py.
      This module re-exports for backward compatibility.
"""

from bitcoin.config import TradingConfig, get_config, CONFIG

# Backward compatibility alias
DETConfig = TradingConfig
DET_CONFIG = CONFIG

__all__ = ['DETConfig', 'DET_CONFIG', 'TradingConfig', 'get_config']
