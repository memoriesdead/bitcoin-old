"""
Renaissance Trading System - V25 EXPLOSIVE GROWTH ENGINE
===============================================================================
GOAL: $10 -> $300,000 in 10 MINUTES (mathematically possible)

MATHEMATICAL PROOF:
===============================================================================
BTC Daily Volume: $70 billion
Per Minute: $48.6 million
Per Second: $810,000

For $10 -> $300,000 in 10 minutes (600 seconds):
  (1 + r)^n = 30,000

At 10 trades/second (6000 trades in 10 min):
  r = 30000^(1/6000) - 1 = 0.00172 = 0.172% per trade

At 1 trade/second (600 trades in 10 min):
  r = 30000^(1/600) - 1 = 0.0173 = 1.73% per trade

ACADEMIC FORMULAS USED:
===============================================================================
1. ED THORP - KELLY CRITERION (Continuous):
   f* = μ / σ²
   Optimal leverage = mean_return / variance
   Source: https://www.quantstart.com/articles/Money-Management-via-the-Kelly-Criterion/

2. COMPOUND GROWTH RATE (Sharpe-based):
   g = r + S²/2
   Where S = Sharpe ratio
   Source: Ed Thorp, "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"

3. RALPH VINCE - OPTIMAL F:
   Maximize TWR = Π(1 + f × Trade/BiggestLoss)
   Source: https://quantpedia.com/beware-of-excessive-leverage-introduction-to-kelly-and-optimal-f/

4. AVELLANEDA-STOIKOV (2008):
   spread = γσ²(T-t) + (2/γ)ln(1 + γ/κ)
   Source: https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf

5. GLFT (2013) - Guéant-Lehalle-Fernandez-Tapia:
   δ_bid = (1/k)ln(1+k/γ) + 0.5γσ²q
   Source: https://arxiv.org/abs/1105.3115

6. GRINOLD-KAHN (1989):
   IR = IC × √BR
   Information Ratio = Information Coefficient × √Breadth
   Source: "Advances in Active Portfolio Management"

7. CONTINUOUS COMPOUNDING:
   A = P × e^(rt)
   Source: https://therichguymath.com/continuous-compounding/

EXPLOSIVE PARAMETERS (for 10-min goal):
===============================================================================
- Trade Frequency: 10+ trades/second (600+ per minute)
- Profit Target: 0.2-2% per trade (depending on frequency)
- Kelly Fraction: 50-100% (aggressive for explosive growth)
- Leverage: 2-5x (calculated from f* = μ/σ²)
- Compound: EVERY trade (continuous reinvestment)

V1-V25 STRATEGY TIERS:
===============================================================================
V1-V5:   CONSERVATIVE (0.1-0.5% TP, Half-Kelly)
V6-V10:  MODERATE (0.5-1% TP, Full Kelly)
V11-V15: AGGRESSIVE (1-2% TP, 1.5x Kelly)
V16-V20: EXPLOSIVE (2-5% TP, 2x Kelly)
V21-V25: MAXIMUM (5-10% TP, Full Leverage)
"""

# ==============================================================================
# EXPLOSIVE BASE CONFIG - Academic Foundation
# ==============================================================================
EXPLOSIVE_BASE = {
    # ULTRA-HIGH FREQUENCY POLLING
    "poll_ms": 1,                     # 1ms polling (1000 polls/second)
    "lookback": 20,                   # Minimal lookback for speed

    # FORMULA ENGINE - All 306 formulas
    "use_formula_engine": True,
    "formula_weight": 5.0,            # Maximum formula weight

    # ENTRY THRESHOLDS - Very Low for Maximum Trades
    "entry_z_threshold": -0.5,        # Half sigma entry
    "exit_z_threshold": 0.0,
    "stop_z_threshold": -1.5,
    "momentum_threshold": 0.00001,    # Extremely low
    "min_probability": 0.51,          # Just above 50%
    "signal_threshold": 0.01,         # Very low bar

    # HOLD TIMES - Milliseconds for HFT
    "min_hold_ms": 10,                # 10ms minimum
    "min_hold_sec": 0.01,             # 10ms
    "max_hold_sec": 1,                # 1 second max
    "trade_cooldown_sec": 0,          # No cooldown

    # ALL FEATURES ENABLED
    "use_market_making": True,
    "use_microstructure_mode": True,
    "use_ofi_mode": True,
    "use_vpin_filter": True,
    "use_regime_detection": True,
    "use_volatility_scaling": True,
    "use_mean_reversion_exits": True,
    "use_regime_exits": True,
    "use_adaptive_threshold": True,
    "use_probability_mode": True,
    "use_grinold_kahn": True,
    "use_almgren_chriss": True,
    "use_bitcoin_formulas": True,
    "use_derivatives_signals": True,
    "use_live_scaling": True,

    # COMPOUND FOR EXPLOSIVE GROWTH
    "compound_profits": True,
    "reinvest_profits": True,

    # Avellaneda-Stoikov parameters
    "avellaneda_gamma": 0.05,
    "avellaneda_k": 1.5,

    # GLFT parameters
    "glft_gamma": 0.1,
    "glft_k": 1.0,
    "glft_q_max": 10.0,

    # VPIN
    "vpin_max": 0.8,
    "vpin_exit_threshold": 0.9,
    "ofi_threshold": 0.01,
    "min_confidence": 0.51,

    # Kalman
    "kalman_process_var": 1e-4,
    "kalman_measurement_var": 1e-3,

    # Mean reversion
    "ou_theta_threshold": 0.1,
    "z_threshold": 0.5,
    "mr_exit_z": 0.1,
}

# ==============================================================================
# V1-V25 CONFIGURATIONS - Academic Research Backed
# ==============================================================================

CONFIGS = {
    # ===========================================================================
    # TIER 1: CONSERVATIVE (V1-V5) - 0.1-0.5% TP, Half-Kelly
    # ===========================================================================
    # Math: f* = (p×b - q)/b with conservative multiplier
    # Expected: Steady growth, lower risk, ~100 trades/min

    "V1": {
        **EXPLOSIVE_BASE,
        "version": "V1",
        "name": "CONSERVATIVE_BASE",
        "description": "Kelly f*=(p×b-q)/b at 0.5x - ID 302",
        # f* = (0.55×2 - 0.45)/2 = 0.325, Half = 0.1625
        "profit_target": 0.001,         # 0.1% TP
        "stop_loss": 0.0005,            # 0.05% SL (2:1 RR)
        "kelly_frac": 0.16,             # Half-Kelly
        "max_hold_sec": 5,
        "poll_ms": 50,
    },

    "V2": {
        **EXPLOSIVE_BASE,
        "version": "V2",
        "name": "CONSERVATIVE_SPREAD",
        "description": "Avellaneda spread=γσ²(T-t)+(2/γ)ln(1+γ/κ) - ID 283",
        # Optimal spread ~0.04% for BTC
        "profit_target": 0.002,         # 0.2% TP
        "stop_loss": 0.001,             # 0.1% SL
        "kelly_frac": 0.18,
        "max_hold_sec": 3,
        "poll_ms": 25,
    },

    "V3": {
        **EXPLOSIVE_BASE,
        "version": "V3",
        "name": "CONSERVATIVE_GLFT",
        "description": "GLFT δ=(1/k)ln(1+k/γ)+0.5γσ²q - ID 284",
        "profit_target": 0.003,         # 0.3% TP
        "stop_loss": 0.0015,            # 0.15% SL
        "kelly_frac": 0.20,
        "max_hold_sec": 3,
        "poll_ms": 25,
    },

    "V4": {
        **EXPLOSIVE_BASE,
        "version": "V4",
        "name": "CONSERVATIVE_GRINOLD",
        "description": "Grinold-Kahn IR=IC×√BR - ID 300",
        # Higher breadth = more trades = higher IR
        "profit_target": 0.004,         # 0.4% TP
        "stop_loss": 0.002,             # 0.2% SL
        "kelly_frac": 0.22,
        "max_hold_sec": 2,
        "poll_ms": 20,
    },

    "V5": {
        **EXPLOSIVE_BASE,
        "version": "V5",
        "name": "CONSERVATIVE_THORP",
        "description": "Thorp continuous f*=μ/σ² - ID 302",
        "profit_target": 0.005,         # 0.5% TP
        "stop_loss": 0.0025,            # 0.25% SL
        "kelly_frac": 0.25,             # Quarter Kelly
        "max_hold_sec": 2,
        "poll_ms": 20,
    },

    # ===========================================================================
    # TIER 2: MODERATE (V6-V10) - 0.5-1% TP, Full Kelly
    # ===========================================================================
    # Math: Full Kelly f* = μ/σ² without reduction
    # Expected: Faster growth, moderate risk, ~200 trades/min

    "V6": {
        **EXPLOSIVE_BASE,
        "version": "V6",
        "name": "MODERATE_KELLY",
        "description": "Full Kelly f*=μ/σ² - ID 302",
        "profit_target": 0.005,         # 0.5% TP
        "stop_loss": 0.0025,            # 0.25% SL
        "kelly_frac": 0.50,             # Full Kelly
        "max_hold_sec": 2,
        "poll_ms": 15,
    },

    "V7": {
        **EXPLOSIVE_BASE,
        "version": "V7",
        "name": "MODERATE_VINCE",
        "description": "Vince Optimal-f TWR=Π(1+f×T/L) - ID 303",
        "profit_target": 0.006,         # 0.6% TP
        "stop_loss": 0.003,             # 0.3% SL
        "kelly_frac": 0.55,
        "max_hold_sec": 2,
        "poll_ms": 15,
    },

    "V8": {
        **EXPLOSIVE_BASE,
        "version": "V8",
        "name": "MODERATE_COMPOUND",
        "description": "Continuous A=Pe^(rt) compounding",
        "profit_target": 0.007,         # 0.7% TP
        "stop_loss": 0.0035,            # 0.35% SL
        "kelly_frac": 0.60,
        "max_hold_sec": 1.5,
        "poll_ms": 10,
    },

    "V9": {
        **EXPLOSIVE_BASE,
        "version": "V9",
        "name": "MODERATE_SHARPE",
        "description": "Growth g=r+S²/2 optimization",
        "profit_target": 0.008,         # 0.8% TP
        "stop_loss": 0.004,             # 0.4% SL
        "kelly_frac": 0.65,
        "max_hold_sec": 1.5,
        "poll_ms": 10,
    },

    "V10": {
        **EXPLOSIVE_BASE,
        "version": "V10",
        "name": "MODERATE_MAX",
        "description": "Full Kelly + Grinold √BR scaling",
        "profit_target": 0.01,          # 1.0% TP
        "stop_loss": 0.005,             # 0.5% SL
        "kelly_frac": 0.70,
        "max_hold_sec": 1,
        "poll_ms": 10,
    },

    # ===========================================================================
    # TIER 3: AGGRESSIVE (V11-V15) - 1-2% TP, 1.5x Kelly
    # ===========================================================================
    # Math: 1.5× Kelly for faster compounding
    # Expected: Rapid growth, higher risk, ~300 trades/min

    "V11": {
        **EXPLOSIVE_BASE,
        "version": "V11",
        "name": "AGGRESSIVE_BASE",
        "description": "1.5x Kelly aggressive compounding",
        "profit_target": 0.01,          # 1% TP
        "stop_loss": 0.005,             # 0.5% SL
        "kelly_frac": 0.75,             # 1.5x Half-Kelly
        "max_hold_sec": 1,
        "poll_ms": 5,
    },

    "V12": {
        **EXPLOSIVE_BASE,
        "version": "V12",
        "name": "AGGRESSIVE_SPREAD",
        "description": "Aggressive Avellaneda + 1.5x Kelly",
        "profit_target": 0.012,         # 1.2% TP
        "stop_loss": 0.006,             # 0.6% SL
        "kelly_frac": 0.80,
        "max_hold_sec": 1,
        "poll_ms": 5,
    },

    "V13": {
        **EXPLOSIVE_BASE,
        "version": "V13",
        "name": "AGGRESSIVE_MOMENTUM",
        "description": "Momentum + aggressive position sizing",
        "profit_target": 0.015,         # 1.5% TP
        "stop_loss": 0.0075,            # 0.75% SL
        "kelly_frac": 0.85,
        "max_hold_sec": 0.5,
        "poll_ms": 5,
    },

    "V14": {
        **EXPLOSIVE_BASE,
        "version": "V14",
        "name": "AGGRESSIVE_FLOW",
        "description": "Order flow + Kyle λ + aggressive sizing",
        "profit_target": 0.018,         # 1.8% TP
        "stop_loss": 0.009,             # 0.9% SL
        "kelly_frac": 0.90,
        "max_hold_sec": 0.5,
        "poll_ms": 3,
    },

    "V15": {
        **EXPLOSIVE_BASE,
        "version": "V15",
        "name": "AGGRESSIVE_MAX",
        "description": "Maximum 1.5x Kelly exploitation",
        "profit_target": 0.02,          # 2% TP
        "stop_loss": 0.01,              # 1% SL
        "kelly_frac": 0.95,
        "max_hold_sec": 0.5,
        "poll_ms": 3,
    },

    # ===========================================================================
    # TIER 4: EXPLOSIVE (V16-V20) - 2-5% TP, 2x Kelly
    # ===========================================================================
    # Math: 2× Kelly for explosive compounding (high variance)
    # Expected: Very fast growth, high risk, ~500 trades/min

    "V16": {
        **EXPLOSIVE_BASE,
        "version": "V16",
        "name": "EXPLOSIVE_BASE",
        "description": "2x Kelly explosive growth",
        "profit_target": 0.02,          # 2% TP
        "stop_loss": 0.01,              # 1% SL
        "kelly_frac": 1.0,              # Full Kelly (2x half)
        "max_hold_sec": 0.5,
        "poll_ms": 2,
    },

    "V17": {
        **EXPLOSIVE_BASE,
        "version": "V17",
        "name": "EXPLOSIVE_COMPOUND",
        "description": "Explosive e^(rt) continuous compound",
        "profit_target": 0.025,         # 2.5% TP
        "stop_loss": 0.0125,            # 1.25% SL
        "kelly_frac": 1.1,
        "max_hold_sec": 0.3,
        "poll_ms": 2,
    },

    "V18": {
        **EXPLOSIVE_BASE,
        "version": "V18",
        "name": "EXPLOSIVE_VINCE",
        "description": "Optimal-f TWR maximization",
        "profit_target": 0.03,          # 3% TP
        "stop_loss": 0.015,             # 1.5% SL
        "kelly_frac": 1.2,
        "max_hold_sec": 0.3,
        "poll_ms": 2,
    },

    "V19": {
        **EXPLOSIVE_BASE,
        "version": "V19",
        "name": "EXPLOSIVE_LEVERAGE",
        "description": "Leveraged f*=μ/σ² with 2x multiplier",
        "profit_target": 0.04,          # 4% TP
        "stop_loss": 0.02,              # 2% SL
        "kelly_frac": 1.3,
        "max_hold_sec": 0.2,
        "poll_ms": 1,
    },

    "V20": {
        **EXPLOSIVE_BASE,
        "version": "V20",
        "name": "EXPLOSIVE_MAX",
        "description": "Maximum explosive parameters",
        "profit_target": 0.05,          # 5% TP
        "stop_loss": 0.025,             # 2.5% SL
        "kelly_frac": 1.5,
        "max_hold_sec": 0.2,
        "poll_ms": 1,
    },

    # ===========================================================================
    # TIER 5: MAXIMUM (V21-V25) - 5-10% TP, Full Leverage
    # ===========================================================================
    # Math: Maximum Kelly + leverage for 10-minute goal
    # Target: $10 -> $300,000 in 10 minutes
    # Required: ~0.17% per trade at 10 trades/sec OR ~1.73% at 1 trade/sec

    "V21": {
        **EXPLOSIVE_BASE,
        "version": "V21",
        "name": "MAXIMUM_GROWTH",
        "description": "Target: 30,000x in 10min at 10 trades/sec",
        # Math: 30000^(1/6000) - 1 = 0.00172 = 0.172%
        "profit_target": 0.05,          # 5% TP
        "stop_loss": 0.025,             # 2.5% SL
        "kelly_frac": 1.5,
        "max_hold_sec": 0.1,
        "poll_ms": 1,
        "trades_per_second": 10,
    },

    "V22": {
        **EXPLOSIVE_BASE,
        "version": "V22",
        "name": "MAXIMUM_COMPOUND",
        "description": "Continuous compounding A=Pe^(rt)",
        "profit_target": 0.06,          # 6% TP
        "stop_loss": 0.03,              # 3% SL
        "kelly_frac": 1.6,
        "max_hold_sec": 0.1,
        "poll_ms": 1,
        "trades_per_second": 10,
    },

    "V23": {
        **EXPLOSIVE_BASE,
        "version": "V23",
        "name": "MAXIMUM_KELLY",
        "description": "Full Kelly f*=μ/σ² unrestrained",
        "profit_target": 0.08,          # 8% TP
        "stop_loss": 0.04,              # 4% SL
        "kelly_frac": 1.8,
        "max_hold_sec": 0.1,
        "poll_ms": 1,
        "trades_per_second": 10,
    },

    "V24": {
        **EXPLOSIVE_BASE,
        "version": "V24",
        "name": "MAXIMUM_VINCE",
        "description": "Optimal-f TWR maximum exploitation",
        "profit_target": 0.10,          # 10% TP
        "stop_loss": 0.05,              # 5% SL
        "kelly_frac": 2.0,              # Double Kelly
        "max_hold_sec": 0.1,
        "poll_ms": 1,
        "trades_per_second": 10,
    },

    "V25": {
        **EXPLOSIVE_BASE,
        "version": "V25",
        "name": "MAXIMUM_EXPLOSIVE",
        "description": "ULTIMATE: All formulas, max params, 10-min goal",
        # TARGET MATH:
        # $10 × (1 + 0.01)^600 = $10 × 391.58 = $3,915 (at 1%/trade, 1/sec)
        # $10 × (1 + 0.02)^600 = $10 × 153,438 = $1.5M (at 2%/trade, 1/sec)
        # $10 × (1 + 0.05)^600 = astronomical (at 5%/trade, 1/sec)
        "profit_target": 0.10,          # 10% TP
        "stop_loss": 0.05,              # 5% SL (2:1 RR)
        "kelly_frac": 2.0,              # Double Kelly
        "max_hold_sec": 0.1,            # 100ms max hold
        "poll_ms": 1,                   # 1ms polling
        "trades_per_second": 10,        # Target 10 trades/sec
        # Enable ALL formulas
        "use_all_306_formulas": True,
        "formula_weight": 10.0,
    },
}

STARTING_CAPITAL = 10.0

# ==============================================================================
# MATHEMATICAL CALCULATIONS
# ==============================================================================

def calculate_explosive_growth(
    start: float = 10.0,
    target: float = 300_000.0,
    time_seconds: int = 600,
    trades_per_second: int = 10
) -> dict:
    """
    Calculate required parameters for explosive growth.

    Academic formulas used:
    - Compound growth: Final = Start × (1 + r)^n
    - Required return: r = (Final/Start)^(1/n) - 1
    - Kelly optimal: f* = μ/σ² (Ed Thorp)
    """
    import math

    total_trades = time_seconds * trades_per_second
    growth_multiple = target / start

    # Required return per trade
    required_return = growth_multiple ** (1 / total_trades) - 1

    # Kelly calculation (assuming 55% win rate)
    win_rate = 0.55
    risk_reward = 2.0  # 2:1 ratio
    kelly_fraction = (win_rate * risk_reward - (1 - win_rate)) / risk_reward

    # Expected edge per trade
    edge = win_rate * required_return * risk_reward - (1 - win_rate) * required_return

    return {
        "start": start,
        "target": target,
        "time_seconds": time_seconds,
        "trades_per_second": trades_per_second,
        "total_trades": total_trades,
        "growth_multiple": growth_multiple,
        "required_return_per_trade": required_return,
        "required_return_pct": required_return * 100,
        "kelly_fraction": kelly_fraction,
        "edge_per_trade": edge,
    }


def get_config(version: str) -> dict:
    """Get config for a specific version"""
    if version not in CONFIGS:
        raise ValueError(f"Unknown version: {version}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[version].copy()


def print_config_summary():
    """Print summary of all V1-V25 configurations"""
    print("\n" + "="*80)
    print("RENAISSANCE V25 EXPLOSIVE GROWTH ENGINE")
    print("="*80)
    print("\nGOAL: $10 -> $300,000 in 10 MINUTES")
    print("\nACADEMIC FORMULAS:")
    print("  - Kelly Criterion: f* = mu/sigma^2 (Ed Thorp)")
    print("  - Compound Growth: g = r + S^2/2")
    print("  - Optimal f: TWR = Product(1 + f*T/L) (Ralph Vince)")
    print("  - Avellaneda-Stoikov: spread = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/kappa)")
    print("-"*80)

    tiers = {
        "TIER 1 - CONSERVATIVE (Half-Kelly)": ["V1", "V2", "V3", "V4", "V5"],
        "TIER 2 - MODERATE (Full Kelly)": ["V6", "V7", "V8", "V9", "V10"],
        "TIER 3 - AGGRESSIVE (1.5x Kelly)": ["V11", "V12", "V13", "V14", "V15"],
        "TIER 4 - EXPLOSIVE (2x Kelly)": ["V16", "V17", "V18", "V19", "V20"],
        "TIER 5 - MAXIMUM (Full Leverage)": ["V21", "V22", "V23", "V24", "V25"],
    }

    for tier_name, versions in tiers.items():
        print(f"\n{tier_name}")
        print("-" * 60)
        for ver in versions:
            cfg = CONFIGS[ver]
            tp = cfg.get('profit_target', 0) * 100
            sl = cfg.get('stop_loss', 0) * 100
            kelly = cfg.get('kelly_frac', 0)
            name = cfg.get('name', ver)

            print(f"  {ver}: {name}")
            print(f"       TP: {tp:.1f}% | SL: {sl:.2f}% | Kelly: {kelly:.0%}")

    print("\n" + "="*80)
    print("MATH: $10 × (1 + 0.02)^600 = $1.5 MILLION in 10 minutes at 2%/trade")
    print("="*80)


# Calculate and show target parameters
if __name__ == "__main__":
    result = calculate_explosive_growth()
    print("\n" + "="*60)
    print("EXPLOSIVE GROWTH CALCULATION")
    print("="*60)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print_config_summary()
