"""
================================================================================
ENGINE IMPLEMENTATIONS - PURE BLOCKCHAIN MATH (NO APIs)
================================================================================

CRITICAL: ALL engines derive data from blockchain math. ZERO external APIs.

At 300,000+ trades/second, APIs are impossible:
- API latency: 50ms (need 100ns)
- API rate limits: 1000/min (need 1M/sec)
- API data: Lagging (need predictive)

ENGINES:
    HFTEngine:         High-frequency tick trading (300K+ ticks/sec)
    RenaissanceEngine: Compounding growth engine

BLOCKCHAIN MATH REFERENCE:
    See: blockchain/ folder for all math implementations
    - blockchain/price_generator.py      -> Price from Power Law
    - blockchain/mempool_math.py         -> Order flow signals
    - blockchain/blockchain_trading_signal.py -> Trading signals
    - blockchain/pure_blockchain_price.py    -> Fair value calculation

FORMULA PIPELINE (All blockchain-derived):
    ID 701: OFI        -> Block timing + fee pressure
    ID 901: Power Law  -> Days since genesis (R2=94%)
    ID 902: S2F        -> Block rewards + scarcity (R2=95%)
    ID 903: Halving    -> Block height % 210,000

NEVER USE: Exchange APIs, WebSockets, third-party feeds
ALWAYS USE: blockchain/ folder implementations

================================================================================
"""
from .base import BaseEngine
from .hft import HFTEngine
from .renaissance import RenaissanceEngine

__all__ = ['BaseEngine', 'HFTEngine', 'RenaissanceEngine']
