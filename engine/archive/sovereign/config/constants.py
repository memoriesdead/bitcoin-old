"""
Sovereign Engine - Global Constants
All magic numbers and thresholds in one place.
Max 100 lines.
"""

# =============================================================================
# TRADING THRESHOLDS - ULTRA AGGRESSIVE RENTECH
# =============================================================================
# "We're right 50.75% of the time, but we're 100% right 50.75% of the time."
# Target: 150,000 - 300,000 trades
RENTECH_EDGE = 0.5075          # Minimum win probability (50.75%)
CONFIDENCE_BASE = 0.50         # Base confidence - LOWERED (any edge = trade)
CONFIDENCE_BOOST = 0.05        # Boost per agreeing signal

# Stop loss / Take profit - ULTRA TIGHT for rapid turnover
STOP_LOSS_PCT = 0.001          # 0.1% stop loss (10 bps)
TAKE_PROFIT_PCT = 0.002        # 0.2% take profit (20 bps)
MAX_HOLD_SECONDS = 3           # 3 sec max hold - RAPID

# Cooldowns - MINIMAL for maximum volume
TRADE_COOLDOWN = 0.01          # 10ms between trades (was 100ms)
SIGNAL_CACHE_TTL = 0.01        # 10ms cache TTL

# =============================================================================
# BITCOIN CONSTANTS
# =============================================================================
GENESIS_TIMESTAMP = 1231006505  # Jan 3, 2009
BLOCK_TIME = 600               # 10 minutes average
BLOCKS_PER_HALVING = 210_000
INITIAL_REWARD = 50.0
MAX_SUPPLY = 21_000_000

# =============================================================================
# POWER LAW MODEL (R² = 94%) - Giovanni Santostasi (2019)
# =============================================================================
POWER_LAW_A = -17.01
POWER_LAW_B = 5.84

# =============================================================================
# STOCK-TO-FLOW MODEL (R² = 95%) - PlanB (2019)
# =============================================================================
S2F_A = -3.39
S2F_B = 3.21

# =============================================================================
# HALVING CYCLE PHASES
# =============================================================================
HALVING_ACCUMULATION = (0.0, 0.30)   # Buy zone
HALVING_EXPANSION = (0.30, 0.70)     # Hold zone
HALVING_DISTRIBUTION = (0.70, 1.0)   # Sell zone

# =============================================================================
# FORMULA IDS
# =============================================================================
FORMULA_IDS = {
    # Leading signals (blockchain-based)
    "power_law": 901,
    "stock_to_flow": 902,
    "halving_cycle": 903,
    # Lagging signals (price-based)
    "ofi": 701,
    "regime": 5309,
    "momentum": 5347,
    "zscore": 5536,
    "kyle_lambda": 5599,
    # Protection
    "breakeven_gate": 950,
    "kelly": 6003,
}

# =============================================================================
# ZMQ CONFIGURATION
# =============================================================================
ZMQ_HOST = "127.0.0.1"
ZMQ_PORT = 28332
ZMQ_ENDPOINT = f"tcp://{ZMQ_HOST}:{ZMQ_PORT}"

# =============================================================================
# EXCHANGE FLOW THRESHOLDS
# =============================================================================
FLOW_RATE_EXPECTED = 0.6       # Expected BTC/sec baseline
FLOW_RATE_THRESHOLD = 1.5      # Multiplier for significant flow
FLOW_WINDOW_SECONDS = 3        # Time window for flow calculation

# =============================================================================
# OFI PARAMETERS
# =============================================================================
OFI_LOOKBACK = 100             # Number of trades to track
OFI_R_SQUARED = 0.70           # 70% explanatory power
QUARTER_KELLY_FACTOR = 0.25    # Use 1/4 Kelly for safety

# =============================================================================
# TRADING GATES
# =============================================================================
DEFAULT_GATE_THRESHOLD = 0.50  # Default threshold for gates
MIN_CONSECUTIVE_SIGNALS = 2    # Minimum consecutive signals
MIN_TRADE_INTERVAL = 0.1       # Minimum seconds between trades
