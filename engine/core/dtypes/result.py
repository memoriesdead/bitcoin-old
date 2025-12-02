"""
RESULT DTYPE - Per-Tick Results
===============================
Logged per tick for analysis and debugging.
"""
import numpy as np

RESULT_DTYPE = np.dtype([
    ('tick_ns', np.int64),            # Timestamp in nanoseconds
    ('true_price', np.float64),       # Power Law true price
    ('market_price', np.float64),     # Actual market price (Lorenz chaos)
    ('edge_pct', np.float64),         # Edge percentage
    ('micro_movement', np.float64),   # Tick-to-tick movement in bps
    ('trades', np.int64),             # Trades this tick
    ('wins', np.int64),               # Wins this tick
    ('pnl', np.float64),              # P&L this tick
    ('capital', np.float64),          # Current capital

    # Formula States (for analysis)
    ('z_score', np.float64),          # ID 141: Z-Score
    ('cusum_event', np.int64),        # ID 218: CUSUM event
    ('regime', np.int64),             # ID 335: Market regime
    ('confluence_signal', np.int64),  # ID 333: Confluence signal
    ('confluence_prob', np.float64),  # ID 333: Confluence probability
    ('ofi_value', np.float64),        # ID 701: OFI value
    ('ofi_signal', np.int64),         # ID 701: OFI signal
    ('kyle_lambda', np.float64),      # ID 702: Kyle lambda
    ('flow_momentum', np.float64),    # ID 706: Flow momentum

    # PURE BLOCKCHAIN SIGNALS (IDs 801-804)
    ('block_volatility', np.float64), # ID 801: Block time volatility
    ('mempool_ofi', np.float64),      # ID 802: Mempool flow OFI
    ('whale_prob', np.float64),       # ID 804: Whale detection probability
    ('chaos_x', np.float64),          # ID 803: Lorenz attractor x-coordinate
])
