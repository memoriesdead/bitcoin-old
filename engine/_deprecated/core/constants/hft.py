"""
HFT TIMESCALE CONSTANTS
=======================
Tick-based timescales for high-frequency trading.
At 100,000 TPS: 1 tick ≈ 10 microseconds
"""
import numpy as np

# =============================================================================
# HFT TIMESCALES - TICK BASED (NOT SECOND BASED)
# =============================================================================
NUM_BUCKETS: int = 8
TICK_TIMESCALES = np.array([1, 10, 100, 1000, 10000, 50000, 100000, 500000], dtype=np.int64)

# Kelly fraction per timescale (aggressive at fast, conservative at slow)
MAX_KELLY_PER_TS = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12], dtype=np.float64)

# Take Profit in basis points per timescale
# 1 tick: 0.1 bps profit target (how HFT works - tiny profits, massive volume)
TP_BPS_PER_TS = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01], dtype=np.float64)

# Stop Loss in basis points per timescale
SL_BPS_PER_TS = np.array([0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.005], dtype=np.float64)

# Max hold time in ticks
MAX_HOLD_TICKS = np.array([5, 50, 500, 5000, 25000, 100000, 250000, 1000000], dtype=np.int64)

# Minimum confidence to enter trade per timescale
MIN_CONFIDENCE_PER_TS = np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25], dtype=np.float64)

# Capital allocation per timescale (more at faster = more opportunities)
CAPITAL_ALLOC_PER_TS = np.array([0.20, 0.20, 0.15, 0.15, 0.10, 0.08, 0.07, 0.05], dtype=np.float64)
