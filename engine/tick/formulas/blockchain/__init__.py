"""
BLOCKCHAIN FORMULA MODULES
==========================
Pure blockchain-derived price formulas - NO external API dependencies.

All calculations are based on deterministic blockchain mechanics:
- Block timing (600 second average)
- Halving cycles (210,000 blocks)
- Difficulty adjustments (2,016 blocks)
- Supply schedule (100% deterministic)

Module Contents:
- ID 801: Block Time Volatility - derives volatility from block timing variance
- ID 802: Mempool Flow Simulator - simulates transaction flow patterns
- ID 803: Chaos Price Generator - Lorenz attractor + blockchain dynamics
- ID 804: Whale Detector - infers large UTXO movements

Academic Citations:
- Lorenz (1963) - Deterministic Nonperiodic Flow
- Kyle (1985) - Econometrica (price impact)
- Bitcoin whitepaper (2008) - blockchain fundamentals
"""

from .block_volatility import calc_block_volatility
from .mempool_flow import calc_mempool_flow
from .chaos_price import (
    calc_chaos_price,
    calc_blockchain_signals,
    generate_independent_price,
    lorenz_step_inline,
)
from .whale_detector import calc_whale_detection

__all__ = [
    'calc_block_volatility',
    'calc_mempool_flow',
    'calc_chaos_price',
    'calc_blockchain_signals',
    'generate_independent_price',
    'lorenz_step_inline',
    'calc_whale_detection',
]
