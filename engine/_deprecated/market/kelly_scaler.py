"""
ADAPTIVE KELLY SCALER - $10 TO $1 BILLION POSITION SIZING
==========================================================
RENAISSANCE TECHNOLOGIES APPROACH: Aggressive at small capital,
conservative at large capital to manage liquidity constraints.

Citation: Kelly (1956) - A New Interpretation of Information Rate
Citation: Thorp (2007) - The Mathematics of Gambling

Master Equation: Capital(t) = Capital(0) × (1 + f × edge)^n

Where:
- f = Kelly fraction (varies with capital tier)
- edge = expected return per trade
- n = number of trades
"""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def calculate_kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """
    Classic Kelly Criterion calculation.

    f* = (p × b - q) / b

    Where:
    - p = win probability
    - q = loss probability (1 - p)
    - b = win/loss ratio

    Args:
        win_rate: Probability of winning (0.5 to 1.0)
        win_loss_ratio: Ratio of average win to average loss

    Returns:
        kelly_fraction: Optimal fraction to bet (0.0 to 1.0)
    """
    if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
        return 0.0

    p = win_rate
    q = 1.0 - p
    b = win_loss_ratio

    kelly = (p * b - q) / b

    # Clamp to [0, 1]
    return max(0.0, min(1.0, kelly))


@njit(cache=True, fastmath=True)
def get_capital_tier_kelly_mult(capital: float) -> float:
    """
    Get Kelly multiplier based on capital tier.

    RENAISSANCE APPROACH:
    - Small capital ($10 - $1K): Aggressive (50% Kelly)
    - Medium capital ($1K - $100K): Standard (25% Kelly)
    - Large capital ($100K - $10M): Conservative (10% Kelly)
    - Very large ($10M - $1B): Very conservative (5% Kelly)
    - Whale ($1B+): Minimal (2% Kelly)

    This accounts for:
    1. Liquidity constraints at larger sizes
    2. Drawdown tolerance at different stages
    3. Compounding efficiency
    """
    if capital < 100:
        # Under $100: Very aggressive to build capital
        return 0.50
    elif capital < 1_000:
        # $100 - $1K: Aggressive
        return 0.40
    elif capital < 10_000:
        # $1K - $10K: Moderately aggressive
        return 0.30
    elif capital < 100_000:
        # $10K - $100K: Standard quarter-Kelly
        return 0.25
    elif capital < 1_000_000:
        # $100K - $1M: Conservative
        return 0.15
    elif capital < 10_000_000:
        # $1M - $10M: More conservative
        return 0.10
    elif capital < 100_000_000:
        # $10M - $100M: Very conservative
        return 0.07
    elif capital < 1_000_000_000:
        # $100M - $1B: Institutional level
        return 0.05
    else:
        # $1B+: Whale level
        return 0.02


@njit(cache=True, fastmath=True)
def calculate_position_size(capital: float,
                            win_rate: float,
                            win_loss_ratio: float,
                            confidence: float,
                            volatility: float,
                            drawdown_pct: float) -> float:
    """
    Calculate position size using adaptive Kelly.

    Args:
        capital: Current capital in USD
        win_rate: Historical win rate (0.5 to 1.0)
        win_loss_ratio: Avg win / avg loss ratio
        confidence: Signal confidence (0.0 to 1.0)
        volatility: Current market volatility
        drawdown_pct: Current drawdown from peak (0.0 to 1.0)

    Returns:
        position_size: Position size in USD
    """
    # Step 1: Calculate base Kelly
    base_kelly = calculate_kelly_fraction(win_rate, win_loss_ratio)

    if base_kelly <= 0:
        return 0.0

    # Step 2: Apply capital tier multiplier
    tier_mult = get_capital_tier_kelly_mult(capital)
    scaled_kelly = base_kelly * tier_mult

    # Step 3: Confidence adjustment
    # Low confidence = reduce position
    conf_mult = 0.5 + 0.5 * confidence  # 0.5x to 1.0x
    scaled_kelly *= conf_mult

    # Step 4: Volatility adjustment
    # High vol = reduce position
    # Normal vol (0.0003) = 1.0x
    # High vol (0.001) = 0.6x
    vol_mult = 0.0003 / max(volatility, 0.0001)
    vol_mult = max(0.3, min(1.0, vol_mult))
    scaled_kelly *= vol_mult

    # Step 5: Drawdown protection
    # Reduce position as drawdown increases
    # 0% DD = 1.0x, 10% DD = 0.8x, 20% DD = 0.5x, 30%+ DD = stop trading
    if drawdown_pct >= 0.30:
        return 0.0  # Stop trading at 30% drawdown
    elif drawdown_pct >= 0.20:
        dd_mult = 0.5
    elif drawdown_pct >= 0.10:
        dd_mult = 0.8 - (drawdown_pct - 0.10) * 3.0  # 0.8 to 0.5
    else:
        dd_mult = 1.0 - drawdown_pct * 2.0  # 1.0 to 0.8

    scaled_kelly *= dd_mult

    # Step 6: Calculate position size
    position_size = capital * scaled_kelly

    # Step 7: Apply absolute limits by capital tier
    max_position = get_max_position_by_tier(capital)
    position_size = min(position_size, max_position)

    return max(0.0, position_size)


@njit(cache=True, fastmath=True)
def get_max_position_by_tier(capital: float) -> float:
    """
    Get maximum single position size by capital tier.

    Even with strong signals, limit position sizes to:
    - Manage single-trade risk
    - Account for liquidity
    - Prevent over-concentration
    """
    if capital < 100:
        return capital * 0.90  # Max 90%
    elif capital < 1_000:
        return capital * 0.50  # Max 50%
    elif capital < 10_000:
        return capital * 0.30  # Max 30%
    elif capital < 100_000:
        return capital * 0.20  # Max 20%
    elif capital < 1_000_000:
        return capital * 0.15  # Max 15%
    elif capital < 10_000_000:
        return capital * 0.10  # Max 10%
    elif capital < 100_000_000:
        return min(capital * 0.05, 5_000_000)  # Max 5% or $5M
    elif capital < 1_000_000_000:
        return min(capital * 0.02, 10_000_000)  # Max 2% or $10M
    else:
        return min(capital * 0.01, 10_000_000)  # Max 1% or $10M


@njit(cache=True, fastmath=True)
def calculate_trades_to_target(current_capital: float,
                               target_capital: float,
                               avg_edge_per_trade: float,
                               kelly_fraction: float) -> int:
    """
    Calculate approximate number of trades needed to reach target.

    From: Capital(t) = Capital(0) × (1 + f × edge)^n

    Solving for n:
    n = ln(Target/Current) / ln(1 + f × edge)

    Args:
        current_capital: Current capital
        target_capital: Target capital
        avg_edge_per_trade: Average edge per trade (e.g., 0.001 = 0.1%)
        kelly_fraction: Effective Kelly fraction being used

    Returns:
        trades_needed: Number of trades to reach target
    """
    if current_capital <= 0 or target_capital <= current_capital:
        return 0

    if avg_edge_per_trade <= 0 or kelly_fraction <= 0:
        return 1_000_000_000  # Essentially infinite

    growth_per_trade = 1.0 + kelly_fraction * avg_edge_per_trade

    if growth_per_trade <= 1.0:
        return 1_000_000_000

    ratio = target_capital / current_capital
    n = np.log(ratio) / np.log(growth_per_trade)

    return int(np.ceil(n))


@njit(cache=True, fastmath=True)
def estimate_time_to_target(current_capital: float,
                            target_capital: float,
                            avg_edge_per_trade: float,
                            kelly_fraction: float,
                            trades_per_day: int) -> float:
    """
    Estimate days to reach target capital.

    Args:
        current_capital: Current capital
        target_capital: Target capital
        avg_edge_per_trade: Average edge per trade
        kelly_fraction: Effective Kelly fraction
        trades_per_day: Average trades per day

    Returns:
        days: Estimated days to reach target
    """
    trades_needed = calculate_trades_to_target(
        current_capital, target_capital, avg_edge_per_trade, kelly_fraction
    )

    if trades_per_day <= 0:
        return float('inf')

    return trades_needed / trades_per_day
