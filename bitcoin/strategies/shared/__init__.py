"""
Shared Utilities for All Strategies
====================================

Common components used by DET, HQT, and SCT strategies.

NOTE: All shared code now lives in bitcoin/core/.
      This module re-exports for backward compatibility.
"""

# Re-export from core/ (canonical source)
from bitcoin.core import (
    EXCHANGE_LEVERAGE,
    get_leverage,
    get_max_leverage_exchange,
    TAKER_FEES,
    MAKER_FEES,
    get_taker_fee,
    get_maker_fee,
    get_total_cost,
)

__all__ = [
    'EXCHANGE_LEVERAGE',
    'get_leverage',
    'get_max_leverage_exchange',
    'TAKER_FEES',
    'MAKER_FEES',
    'get_taker_fee',
    'get_maker_fee',
    'get_total_cost',
]
