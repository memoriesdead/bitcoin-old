"""
STATE DTYPE - Global Engine State
=================================
Tracks all formula states and engine-wide metrics.

FORMULA ID MAPPING:
- z_score, price_mean, price_std: ID 141 (Z-Score)
- cusum_*: ID 218 (CUSUM Filter)
- regime, ema_*: ID 335 (Regime Filter)
- confluence_*: ID 333 (Signal Confluence)
- ofi_*, kyle_lambda, flow_momentum: ID 701-706 (Order Flow)
"""
import numpy as np

STATE_DTYPE = np.dtype([
    # Global State
    ('total_capital', np.float64),
    ('total_trades', np.int64),
    ('total_wins', np.int64),
    ('total_pnl', np.float64),
    ('tick_count', np.int64),
    ('last_price', np.float64),
    ('price_direction', np.int64),    # 1=up, -1=down, 0=flat
    ('consecutive_up', np.int64),
    ('consecutive_down', np.int64),
    ('halving_cycle', np.float64),    # Position in halving cycle 0.0-1.0

    # =========================================================================
    # FORMULA ID 141: Z-Score Mean Reversion (LEGACY - ZERO EDGE)
    # =========================================================================
    ('z_score', np.float64),          # Current z-score
    ('price_mean', np.float64),       # Rolling mean
    ('price_std', np.float64),        # Rolling std

    # =========================================================================
    # FORMULA ID 218: CUSUM Filter (+8-12pp WR)
    # Citation: Lopez de Prado (2018)
    # =========================================================================
    ('cusum_pos', np.float64),        # Positive CUSUM accumulator
    ('cusum_neg', np.float64),        # Negative CUSUM accumulator
    ('cusum_event', np.int64),        # Last event: 1=bullish, -1=bearish, 0=none
    ('cusum_volatility', np.float64), # Current volatility for threshold

    # =========================================================================
    # FORMULA ID 335: Regime Filter (+3-5pp WR)
    # Citation: Moskowitz, Ooi & Pedersen (2012)
    # =========================================================================
    ('ema_fast', np.float64),         # Fast EMA (20 period)
    ('ema_slow', np.float64),         # Slow EMA (50 period)
    ('regime', np.int64),             # 2=strong_up, 1=weak_up, 0=ranging, -1=weak_down, -2=strong_down
    ('regime_confidence', np.float64), # Confidence in regime classification

    # =========================================================================
    # FORMULA ID 333: Signal Confluence (Condorcet Voting)
    # =========================================================================
    ('confluence_signal', np.int64),  # Combined signal direction
    ('confluence_prob', np.float64),  # Combined probability
    ('agreeing_signals', np.int64),   # Number of signals agreeing

    # =========================================================================
    # FORMULA IDs 701-706: Order Flow (THE REAL EDGE - R² = 70%)
    # Citation: Cont, Kukanov & Stoikov (2014)
    # =========================================================================
    ('ofi_value', np.float64),        # ID 701: OFI value (buy - sell pressure)
    ('ofi_signal', np.int64),         # ID 701: OFI direction 1=BUY, -1=SELL, 0=NEUTRAL
    ('ofi_strength', np.float64),     # ID 701: Signal strength 0.0-1.0
    ('kyle_lambda', np.float64),      # ID 702: Price impact coefficient
    ('flow_momentum', np.float64),    # ID 706: OFI acceleration

    # =========================================================================
    # LORENZ CHAOS STATE (ID 803) - INDEPENDENT PRICE GENERATION
    # Citation: Lorenz (1963) - Deterministic Nonperiodic Flow
    # =========================================================================
    ('lorenz_x', np.float64),         # Lorenz attractor x coordinate
    ('lorenz_y', np.float64),         # Lorenz attractor y coordinate
    ('lorenz_z', np.float64),         # Lorenz attractor z coordinate
    ('chaos_vol', np.float64),        # Chaos-derived volatility
    ('chaos_vol_ema', np.float64),    # Volatility EMA (GARCH-like)
    ('chaos_last_return', np.float64), # Last return for GARCH

    # =========================================================================
    # ADAPTIVE KELLY - $10 TO $1B POSITION SIZING
    # Citation: Kelly (1956), Thorp (2007)
    # =========================================================================
    ('peak_capital', np.float64),     # Peak capital for drawdown calculation
    ('current_drawdown', np.float64), # Current drawdown from peak (0.0 to 1.0)
    ('win_rate_ema', np.float64),     # EMA of win rate for Kelly
    ('win_loss_ratio', np.float64),   # Average win / average loss
])
