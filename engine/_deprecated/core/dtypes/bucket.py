"""
BUCKET DTYPE - Per-Timescale Trading State
==========================================
Each timescale bucket tracks its own positions and P&L.
"""
import numpy as np

BUCKET_DTYPE = np.dtype([
    ('capital', np.float64),          # Capital allocated to this bucket
    ('position', np.int64),           # Current position: 1=LONG, -1=SHORT, 0=FLAT
    ('entry_price', np.float64),      # Price at position entry
    ('entry_tick', np.int64),         # Tick count at entry
    ('position_size', np.float64),    # Size of position in USD
    ('trades', np.int64),             # Total trades executed
    ('wins', np.int64),               # Winning trades
    ('total_pnl', np.float64),        # Cumulative P&L
])
