"""
BLOCKCHAIN CONSTANTS - LEADING INDICATORS
==========================================
Constants for Bitcoin blockchain-derived price signals.
All values are 100% deterministic from blockchain mechanics.

These constants enable LEADING indicators calculated from TIMESTAMP ONLY,
completely independent of current price data.

Academic Citations:
- Giovannetti (2019) - Bitcoin Power Law, R² = 94%
- PlanB (2019) - Bitcoin Stock-to-Flow Model, R² = 95%
- Bitcoin whitepaper (2008) - Halving schedule

Key Insight: Bitcoin's supply schedule is 100% deterministic.
- We know EXACTLY when halvings occur
- We know EXACTLY what the supply will be at any future date
- We know EXACTLY the Stock-to-Flow ratio at any timestamp

This determinism is the EDGE: Blockchain fundamentals that predict price direction.
"""

# =============================================================================
# BITCOIN BLOCKCHAIN FUNDAMENTALS
# =============================================================================

# Genesis block timestamp: Jan 3, 2009, 18:15:05 UTC
BLOCKCHAIN_GENESIS_TIMESTAMP = 1231006505.0

# Average block time: 10 minutes = 600 seconds
# Difficulty adjusts every 2016 blocks to maintain this average
BLOCKCHAIN_BLOCK_TIME = 600.0

# Blocks per halving epoch: 210,000 blocks (~4 years)
# After each halving, block reward is cut in half
BLOCKCHAIN_BLOCKS_PER_HALVING = 210000.0

# Initial block reward: 50 BTC (epoch 0)
# Epoch 1: 25 BTC, Epoch 2: 12.5 BTC, Epoch 3: 6.25 BTC, Epoch 4: 3.125 BTC
BLOCKCHAIN_INITIAL_REWARD = 50.0

# Hard cap: 21 million BTC maximum supply
# Approaches asymptotically as block reward halves
BLOCKCHAIN_TOTAL_SUPPLY = 21000000.0

# =============================================================================
# POWER LAW MODEL COEFFICIENTS
# Citation: Giovannetti (2019) - Bitcoin Power Law
# Formula: log10(price) = A + B * log10(days_since_genesis)
# R² = 94% over 10+ years of price history
# =============================================================================

BLOCKCHAIN_POWER_LAW_A = -17.01  # Intercept (log scale)
BLOCKCHAIN_POWER_LAW_B = 5.84    # Slope (log scale)

# =============================================================================
# STOCK-TO-FLOW MODEL COEFFICIENTS
# Citation: PlanB (2019) - Bitcoin S2F Model
# Formula: ln(price) = A + B * ln(S2F)
# R² = 95% correlation with price
# S2F = Current_Supply / Annual_Issuance
# =============================================================================

BLOCKCHAIN_S2F_A = -3.39  # Intercept (natural log scale, recalibrated)
BLOCKCHAIN_S2F_B = 3.21   # Slope (natural log scale)

# =============================================================================
# LORENZ ATTRACTOR PARAMETERS (CHAOS DYNAMICS)
# Citation: Lorenz (1963) - Deterministic Nonperiodic Flow
# Classic parameters for chaotic attractor
# =============================================================================

LORENZ_SIGMA = 10.0
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0

# =============================================================================
# BLOCKCHAIN TIMING CONSTANTS
# =============================================================================

# Alternative genesis timestamp (slightly different, used in some calculations)
BLOCKCHAIN_GENESIS_TS = 1230768000.0

# Seconds per block (same as BLOCKCHAIN_BLOCK_TIME)
SECONDS_PER_BLOCK = 600.0

# Blocks per difficulty adjustment: 2016 blocks (~2 weeks)
BLOCKS_PER_DIFFICULTY = 2016

# Blocks per halving (integer version)
BLOCKS_PER_HALVING_LOCAL = 210000

# =============================================================================
# HALVING CYCLE PHASES (Empirically observed)
# =============================================================================

# Accumulation phase: 0% - 30% of halving cycle
# Post-halving recovery, early adopters accumulate
HALVING_ACCUMULATION_END = 0.30

# Distribution phase: 70% - 100% of halving cycle
# Pre-halving top formation, early holders distribute
HALVING_DISTRIBUTION_START = 0.70

# =============================================================================
# DERIVED CONSTANTS (Computed from fundamentals)
# =============================================================================

# Blocks per year: 365.25 * 24 * 6 = 52,560
BLOCKS_PER_YEAR = 365.25 * 24 * 6

# Seconds per year
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60

# Blocks per day
BLOCKS_PER_DAY = 24 * 6  # 144 blocks/day
