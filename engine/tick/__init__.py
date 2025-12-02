"""
================================================================================
TICK PROCESSING - PURE BLOCKCHAIN MATH (NO APIs)
================================================================================

JIT-compiled tick processing at nanosecond level.
ALL signals derived from blockchain math, ZERO external APIs.

At 300,000+ ticks/second, APIs are impossible:
- API latency: 50ms (need 100ns)
- API rate limits: 1000/min (need 1M/sec)

BLOCKCHAIN MATH REFERENCE:
    See: blockchain/ folder for implementations
    - blockchain/price_generator.py -> Price calculation
    - blockchain/mempool_math.py    -> Order flow signals

FORMULA PIPELINE (All blockchain-derived):
    ID 701: OFI        -> Block timing + fee pressure
    ID 901: Power Law  -> Days since genesis (R2=94%)
    ID 902: S2F        -> Block rewards + scarcity
    ID 903: Halving    -> Block height % 210,000

Contains:
    process_tick_hft: Main tick processing (Numba JIT, 500ns/tick)

================================================================================
"""
from .processor import process_tick_hft

__all__ = ['process_tick_hft']
