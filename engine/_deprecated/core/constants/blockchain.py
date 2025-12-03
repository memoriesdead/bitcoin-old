"""
================================================================================
BLOCKCHAIN CONSTANTS (Pure Math - No APIs)
================================================================================

Source: Giovanni Santostasi's Power Law model (93%+ correlation over 14 years)
Formula: Price = 10^(A + B * log10(days_since_genesis))

ARCHITECTURE REFERENCE: docs/BLOCKCHAIN_PIPELINE_ARCHITECTURE.md

These constants are IMMUTABLE and derived from Bitcoin protocol rules.
The entire trading system is built on these mathematical foundations.

KEY FORMULAS USING THESE CONSTANTS:
    ID 901: Power Law    -> Uses GENESIS_TS, POWER_LAW_A, POWER_LAW_B
    ID 902: Stock-to-Flow -> Uses GENESIS_TS, BLOCKS_PER_HALVING, INITIAL_REWARD
    ID 903: Halving Cycle -> Uses GENESIS_TS, BLOCKS_PER_HALVING
    ID 803: Chaos Price   -> Uses GENESIS_TS, all Lorenz constants (if defined)

================================================================================
"""
import numpy as np

# =============================================================================
# BITCOIN GENESIS & PROTOCOL CONSTANTS
# =============================================================================
GENESIS_TS: float = 1230768000.0      # Jan 1, 2009 (Power Law epoch)
MAX_SUPPLY: float = 21000000.0         # Total BTC ever
BLOCKS_PER_HALVING: int = 210000       # Blocks between halvings
INITIAL_REWARD: float = 50.0           # First block reward

# =============================================================================
# POWER LAW MODEL CONSTANTS (from bitbo.io - 7 decimal precision)
# =============================================================================
POWER_LAW_A: float = -17.0161223       # Intercept
POWER_LAW_B: float = 5.8451542         # Slope

# Support/Resistance multipliers (from Power Law research)
SUPPORT_MULT: float = 0.42             # Fair price floor (strong support)
RESIST_MULT: float = 2.38              # Fair price ceiling (strong resistance)

# =============================================================================
# HALVING CYCLE CONSTANTS
# =============================================================================
CURRENT_HALVING: int = 4               # Current halving era (as of 2024)
HALVING_START_BLOCK: int = 840000      # Block height at 4th halving
BLOCK_REWARD_BTC: float = 3.125        # After 4th halving (April 2024)
BLOCKS_PER_DAY: float = 144.0          # 24*60/10 = 144 blocks/day
MINER_DAILY_BTC: float = 450.0         # 3.125 * 144 = 450 BTC/day to miners

# =============================================================================
# BLOCKCHAIN VOLUME BENCHMARKS
# =============================================================================
ONCHAIN_VOLUME_MULT: float = 1000.0    # On-chain volume ≈ 500-1000x miner output
DAILY_BTC_VOLUME: float = 450000.0     # ~450k BTC/day on-chain (conservative)

# BTC volume per time unit
BTC_PER_HOUR: float = 18750.0          # 450000 / 24
BTC_PER_SECOND: float = 5.208333       # 450000 / 86400
BTC_PER_MILLISECOND: float = 0.005208  # BTC_PER_SECOND / 1000
BTC_PER_MICROSECOND: float = 0.000005208
BTC_PER_NANOSECOND: float = 5.208e-12

# =============================================================================
# MATH CONSTANTS (Pre-computed for JIT performance)
# =============================================================================
TWO_PI: float = 6.283185307179586
INV_86400: float = 1.1574074074074073e-05
LOG10_E: float = 0.4342944819032518     # For converting ln to log10
LN_10: float = 2.302585093              # ln(10) for Power Law calc

# =============================================================================
# TIME CONVERSIONS
# =============================================================================
SECONDS_PER_DAY: float = 86400.0
SECONDS_PER_HOUR: float = 3600.0
MS_PER_SECOND: float = 1000.0
US_PER_SECOND: float = 1000000.0
NS_PER_SECOND: float = 1000000000.0

# =============================================================================
# LOOKUP TABLES (for fast sin/cos in JIT functions)
# =============================================================================
SIN_TABLE_SIZE: int = 65536
SIN_TABLE = np.sin(np.linspace(0, 2 * np.pi, SIN_TABLE_SIZE, dtype=np.float64))
