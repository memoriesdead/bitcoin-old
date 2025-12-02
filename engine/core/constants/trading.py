"""
TRADING CONSTANTS - Formula Parameters by ID
=============================================
All formula-specific constants organized by Formula ID.

FORMULA ID RANGES:
- 100-199: Entry Signals (Z-Score, etc.)
- 200-299: Filters (CUSUM)
- 300-399: Confluence & Regime
- 600-699: Volume Capture
- 700-799: Order Flow (OFI, Kyle, Momentum)
- 800-899: Renaissance Compounding
"""

# =============================================================================
# FORMULA ID 141: Z-SCORE MEAN REVERSION (LEGACY - ZERO EDGE)
# =============================================================================
# ACADEMIC FINDING: Z-score trades AGAINST flow = ZERO EDGE
# Keep for confluence voting only, OFI is primary signal
ENTRY_Z: float = 2.0              # Trade threshold
EXIT_Z: float = 0.5               # Exit threshold
ZSCORE_LOOKBACK: int = 100        # Price history window

# =============================================================================
# FORMULA ID 218: CUSUM FILTER (+8-12pp Win Rate)
# =============================================================================
# Citation: Lopez de Prado (2018) - Advances in Financial ML
CUSUM_THRESHOLD_STD: float = 1.0  # Threshold in standard deviations
CUSUM_DRIFT_MULT: float = 0.5     # Drift correction (h = threshold * 0.5)
CUSUM_LOOKBACK: int = 20          # Volatility calculation window

# =============================================================================
# FORMULA ID 333: SIGNAL CONFLUENCE (Condorcet Voting)
# =============================================================================
# Condorcet's Jury Theorem: Independent signals > 50% → higher accuracy
MIN_AGREEING_SIGNALS: int = 2     # Minimum signals agreeing
MIN_CONFLUENCE_PROB: float = 0.55 # Minimum combined probability

# =============================================================================
# FORMULA ID 335: REGIME FILTER (+3-5pp Win Rate)
# =============================================================================
# Citation: Moskowitz, Ooi & Pedersen (2012) - Time Series Momentum
REGIME_EMA_FAST: int = 20         # Fast EMA period
REGIME_EMA_SLOW: int = 50         # Slow EMA period
STRONG_TREND_THRESH: float = 0.02 # 2% EMA divergence = strong trend
WEAK_TREND_THRESH: float = 0.005  # 0.5% EMA divergence = weak trend

# =============================================================================
# FORMULA ID 701: OFI (Order Flow Imbalance) - THE REAL EDGE (R² = 70%)
# =============================================================================
# Citation: Cont, Kukanov & Stoikov (2014) - J. Financial Econometrics
OFI_LOOKBACK: int = 50            # OFI history for momentum
OFI_THRESHOLD: float = 0.05       # Signal threshold (OFI ranges -0.05 to +0.08)
FLOW_FOLLOWING: bool = True       # Trade WITH flow, not against

# =============================================================================
# FORMULA IDs 801-810: RENAISSANCE COMPOUNDING
# =============================================================================
# Master Equation: Capital(t) = Capital(0) × (1 + f × edge)^n
# Sources: Kelly (1956), Thorp (2007), Cont-Stoikov (2014)

RENAISSANCE_INITIAL_CAPITAL: float = 100.0   # Starting capital
RENAISSANCE_TARGET_CAPITAL: float = 10000.0  # 100x goal

# Kelly Criterion - Quarter-Kelly for safety
QUARTER_KELLY: float = 0.25       # 75% of optimal growth, 6.25% variance
FULL_KELLY_CAP: float = 0.5       # Never exceed 50% Kelly

# Win Rate Requirements
MIN_WIN_RATE: float = 0.52        # Minimum 52% to trade
TARGET_WIN_RATE: float = 0.55     # Target 55% for optimal edge
WIN_LOSS_RATIO: float = 1.15      # Target W/L ratio

# Sharpe Requirements (Thorp 2007: g = r + S²/2)
MIN_SHARPE: float = 2.0           # Minimum Sharpe to trade
TARGET_SHARPE: float = 3.0        # Target Sharpe for optimal growth

# Edge Requirements
MIN_EDGE_PCT: float = 0.001       # 0.1% minimum edge per trade
TARGET_EDGE_PCT: float = 0.004    # 0.4% target edge per trade
TRADING_COST_PCT: float = 0.001   # 0.1% trading costs

# Trade Frequency
TRADES_PER_DAY: int = 100         # Optimal frequency
MIN_EDGE_COST_RATIO: float = 3.0  # Edge must be 3× costs

# Drawdown Constraints
MAX_DRAWDOWN_PCT: float = 0.20    # Max 20% drawdown from peak
DRAWDOWN_KELLY_SCALE: float = 0.5 # Reduce Kelly by 50% near max DD

# Time-to-Target
TRADES_FOR_100X: int = 4607       # At 0.1% edge per trade
DAYS_FOR_100X: int = 46           # At 100 trades/day, 0.4% edge

# Compounding
COMPOUND_INTERVAL: int = 10       # Compound every N trades
REINVEST_THRESHOLD: float = 0.01  # Reinvest when profits exceed 1%

# =============================================================================
# TRADING FEES
# =============================================================================
FEE: float = 0.0                  # DISABLED for testing
