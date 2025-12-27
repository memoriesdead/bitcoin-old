"""
HQT Strategy Configuration
==========================

NOTE: All config now lives in bitcoin/config.py.
      This module re-exports for backward compatibility.
"""

from bitcoin.config import TradingConfig, get_config, CONFIG

# Backward compatibility alias
HQTConfig = TradingConfig

__all__ = ['HQTConfig', 'TradingConfig', 'get_config']
