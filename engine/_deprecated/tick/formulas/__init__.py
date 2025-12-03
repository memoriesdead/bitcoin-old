"""
TICK FORMULA MODULES
====================
Modular JIT-compiled formula calculations for HFT engine.

Each formula is in its own module for maintainability while
preserving Numba JIT compilation and nanosecond performance.

Module Structure:
- constants.py: Blockchain constants and parameters
- zscore.py: ID 141 - Z-Score Mean Reversion
- ofi.py: ID 701/702/706 - Order Flow Imbalance + Kyle Lambda + Flow Momentum
- cusum.py: ID 218 - CUSUM Filter
- regime.py: ID 335 - Regime Filter
- confluence.py: ID 333 - Signal Confluence (Condorcet Voting)
- blockchain/: ID 801-804 - Pure blockchain price formulas
- leading/: ID 901-903 - Leading blockchain signals

Academic Citations:
- Cont, Kukanov & Stoikov (2014) - J. Financial Econometrics [OFI]
- Lopez de Prado (2018) - Advances in Financial ML [CUSUM]
- Kyle (1985) - Econometrica [Kyle Lambda]
- Moskowitz, Ooi & Pedersen (2012) - JFE [Regime]
- Giovannetti (2019) - Bitcoin Power Law [Power Law]
- PlanB (2019) - Bitcoin S2F Model [S2F]
- Lorenz (1963) - Deterministic Nonperiodic Flow [Chaos]
"""

# Import all formula functions for easy access
from .constants import (
    BLOCKCHAIN_GENESIS_TIMESTAMP,
    BLOCKCHAIN_BLOCK_TIME,
    BLOCKCHAIN_BLOCKS_PER_HALVING,
    BLOCKCHAIN_INITIAL_REWARD,
    BLOCKCHAIN_TOTAL_SUPPLY,
    BLOCKCHAIN_POWER_LAW_A,
    BLOCKCHAIN_POWER_LAW_B,
    BLOCKCHAIN_S2F_A,
    BLOCKCHAIN_S2F_B,
)

from .zscore import calc_zscore
from .ofi import calc_ofi
from .cusum import calc_cusum
from .regime import calc_regime
from .confluence import calc_confluence

# Blockchain formulas
from .blockchain import (
    calc_block_volatility,
    calc_mempool_flow,
    calc_chaos_price,
    calc_blockchain_signals,
    calc_whale_detection,
    generate_independent_price,
    lorenz_step_inline,
)

# Leading indicators
from .leading import (
    calc_power_law_signal,
    calc_s2f_signal,
    calc_halving_cycle_signal,
)

__all__ = [
    # Constants
    'BLOCKCHAIN_GENESIS_TIMESTAMP',
    'BLOCKCHAIN_BLOCK_TIME',
    'BLOCKCHAIN_BLOCKS_PER_HALVING',
    'BLOCKCHAIN_INITIAL_REWARD',
    'BLOCKCHAIN_TOTAL_SUPPLY',
    'BLOCKCHAIN_POWER_LAW_A',
    'BLOCKCHAIN_POWER_LAW_B',
    'BLOCKCHAIN_S2F_A',
    'BLOCKCHAIN_S2F_B',
    # Core formulas
    'calc_zscore',
    'calc_ofi',
    'calc_cusum',
    'calc_regime',
    'calc_confluence',
    # Blockchain formulas
    'calc_block_volatility',
    'calc_mempool_flow',
    'calc_chaos_price',
    'calc_blockchain_signals',
    'calc_whale_detection',
    'generate_independent_price',
    'lorenz_step_inline',
    # Leading indicators
    'calc_power_law_signal',
    'calc_s2f_signal',
    'calc_halving_cycle_signal',
]
