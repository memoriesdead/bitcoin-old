"""
HQT+SCT C++ Trading Bridges

Provides nanosecond-speed trading calculations via C++ FFI.
Falls back to pure Python implementations if C++ library unavailable.
"""

from .hqt_bridge import HQTBridge, ArbitrageOpportunity
from .sct_bridge import (
    SCTBridge,
    CertaintyStatus,
    CertaintyResult,
    WilsonInterval,
    PositionSize
)

__all__ = [
    'HQTBridge',
    'ArbitrageOpportunity',
    'SCTBridge',
    'CertaintyStatus',
    'CertaintyResult',
    'WilsonInterval',
    'PositionSize',
]
