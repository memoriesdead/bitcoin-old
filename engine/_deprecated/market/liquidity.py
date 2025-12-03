"""
LIQUIDITY MODEL - REALISTIC MARKET IMPACT & SLIPPAGE
=====================================================
RENAISSANCE TECHNOLOGIES APPROACH: Scale positions realistically
as capital grows from $10 to $1 billion.

Key insight: $10K moves easily, $100M moves markets, $1B is impossible
to trade without massive slippage.

Citation: Kyle (1985) - Continuous Auctions and Insider Trading
"""
import numpy as np
from numba import njit

# Market depth constants (BTC/USD market)
# Based on typical order book depth
AVG_DAILY_VOLUME_BTC = 450000.0  # ~450k BTC daily on-chain
AVG_DAILY_VOLUME_USD = 45_000_000_000.0  # $45B at $100k/BTC

# Kyle Lambda (price impact coefficient)
# Empirical: ~0.1% per $1M for BTC
KYLE_LAMBDA_BASE = 0.001 / 1_000_000  # 0.1% per $1M


@njit(cache=True, fastmath=True)
def calculate_slippage(order_size_usd: float, volatility: float) -> float:
    """
    Calculate execution slippage based on order size and volatility.

    Kyle (1985): Price Impact = lambda * sqrt(order_size)

    Args:
        order_size_usd: Order size in USD
        volatility: Current market volatility (0.0001 to 0.01)

    Returns:
        slippage_pct: Slippage as percentage (0.0 to 0.1)
    """
    if order_size_usd <= 0:
        return 0.0

    # Base slippage from order size
    # sqrt relationship: $1M = 0.1%, $100M = 1%, $1B = 3%
    base_slippage = KYLE_LAMBDA_BASE * np.sqrt(order_size_usd)

    # Volatility multiplier (higher vol = higher slippage)
    # Normal vol ~0.0003, high vol ~0.001
    vol_mult = 1.0 + (volatility / 0.0003 - 1.0) * 0.5
    vol_mult = max(0.5, min(3.0, vol_mult))

    slippage = base_slippage * vol_mult

    # Cap at 10% (beyond this, order is unfillable)
    return min(slippage, 0.10)


@njit(cache=True, fastmath=True)
def calculate_market_impact(order_size_usd: float,
                            daily_volume_usd: float = AVG_DAILY_VOLUME_USD) -> float:
    """
    Calculate permanent market impact from order.

    Market Impact = Order_Size / Daily_Volume * Impact_Factor

    Args:
        order_size_usd: Order size in USD
        daily_volume_usd: Daily trading volume

    Returns:
        impact_pct: Permanent price impact as percentage
    """
    if order_size_usd <= 0 or daily_volume_usd <= 0:
        return 0.0

    # Order as fraction of daily volume
    volume_fraction = order_size_usd / daily_volume_usd

    # Impact scales with volume fraction
    # 0.1% of daily volume = minimal impact
    # 1% of daily volume = ~0.5% impact
    # 10% of daily volume = ~5% impact
    impact = 0.5 * np.sqrt(volume_fraction)

    return min(impact, 0.20)  # Cap at 20%


@njit(cache=True, fastmath=True)
def get_max_position_size(capital: float, volatility: float) -> float:
    """
    Calculate maximum position size based on capital and volatility.

    Limits:
    - $10 - $1K: 90% of capital (aggressive)
    - $1K - $10K: 50% of capital
    - $10K - $100K: 30% of capital
    - $100K - $1M: 20% of capital
    - $1M - $10M: 10% of capital
    - $10M - $100M: 5% of capital
    - $100M - $1B: 2% of capital
    - $1B+: 1% of capital

    Further reduced by volatility (high vol = smaller positions)
    """
    # Base allocation by capital tier
    if capital < 1_000:
        base_alloc = 0.90
    elif capital < 10_000:
        base_alloc = 0.50
    elif capital < 100_000:
        base_alloc = 0.30
    elif capital < 1_000_000:
        base_alloc = 0.20
    elif capital < 10_000_000:
        base_alloc = 0.10
    elif capital < 100_000_000:
        base_alloc = 0.05
    elif capital < 1_000_000_000:
        base_alloc = 0.02
    else:
        base_alloc = 0.01

    # Volatility reduction
    # Normal vol (0.0003) = 1.0x
    # High vol (0.001) = 0.5x
    # Very high vol (0.003) = 0.25x
    vol_factor = 0.0003 / max(volatility, 0.0001)
    vol_factor = min(1.0, vol_factor)

    max_position = capital * base_alloc * vol_factor

    return max_position


@njit(cache=True, fastmath=True)
def calculate_execution_price(mid_price: float,
                              order_size_usd: float,
                              direction: int,
                              volatility: float) -> float:
    """
    Calculate actual execution price including slippage.

    Args:
        mid_price: Current mid price
        order_size_usd: Order size in USD
        direction: 1 for buy, -1 for sell
        volatility: Current volatility

    Returns:
        execution_price: Actual fill price after slippage
    """
    slippage = calculate_slippage(order_size_usd, volatility)

    if direction > 0:
        # Buying: pay more
        execution_price = mid_price * (1.0 + slippage)
    else:
        # Selling: receive less
        execution_price = mid_price * (1.0 - slippage)

    return execution_price


@njit(cache=True, fastmath=True)
def calculate_realistic_pnl(entry_price: float,
                            exit_price: float,
                            position_size_usd: float,
                            direction: int,
                            volatility: float) -> float:
    """
    Calculate realistic P&L including entry and exit slippage.

    Args:
        entry_price: Entry mid price
        exit_price: Exit mid price
        position_size_usd: Position size in USD
        direction: 1 for long, -1 for short
        volatility: Current volatility

    Returns:
        pnl: Realistic P&L after slippage
    """
    # Entry slippage
    entry_slippage = calculate_slippage(position_size_usd, volatility)
    if direction > 0:
        actual_entry = entry_price * (1.0 + entry_slippage)
    else:
        actual_entry = entry_price * (1.0 - entry_slippage)

    # Exit slippage
    exit_slippage = calculate_slippage(position_size_usd, volatility)
    if direction > 0:
        actual_exit = exit_price * (1.0 - exit_slippage)  # Selling long
    else:
        actual_exit = exit_price * (1.0 + exit_slippage)  # Covering short

    # Calculate P&L
    if direction > 0:
        price_change = (actual_exit - actual_entry) / actual_entry
    else:
        price_change = (actual_entry - actual_exit) / actual_entry

    pnl = position_size_usd * price_change

    return pnl
