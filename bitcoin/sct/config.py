"""
SCT Strategy Configuration
==========================

NOTE: All config now lives in bitcoin/config.py.
      This module re-exports for backward compatibility.
"""

from bitcoin.config import TradingConfig, get_config, set_config, CONFIG

# Backward compatibility alias
SCTConfig = TradingConfig

__all__ = ['SCTConfig', 'TradingConfig', 'get_config', 'set_config']
