"""
FORMULA ID 141: Z-SCORE MEAN REVERSION
======================================
Z-Score calculation for statistical deviation from mean.

WARNING: Z-Score alone has ZERO trading edge. It trades AGAINST order flow.
Used only for confluence confirmation, NOT as a primary signal.

Mathematical Formula:
    z = (current_price - mean) / std

Where:
    mean = rolling average over lookback period
    std = rolling standard deviation over lookback period

Citation:
    Standard statistical measure used universally in quantitative finance.
    See Taleb (2007) - "The Black Swan" for limitations of normal assumptions.

Performance: O(n) where n = lookback period
Numba JIT: ~50-100 nanoseconds per tick
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, error_model='numpy', boundscheck=False)
def calc_zscore(prices: np.ndarray, tick: int, lookback: int) -> tuple:
    """
    Z-SCORE CALCULATION (Formula ID 141)

    Calculates how many standard deviations the current price is from the mean.
    Positive z-score = price above mean (potentially overbought)
    Negative z-score = price below mean (potentially oversold)

    CRITICAL INSIGHT:
    Z-score mean reversion has ZERO edge when used alone because it trades
    AGAINST order flow. The market can stay irrational longer than you can
    stay solvent. Only use for confirmation alongside flow-based signals.

    Args:
        prices: Circular price buffer with 1M capacity
                Indexed by (tick % 1000000)
        tick: Current tick index (total ticks processed)
        lookback: Number of historical prices to use for mean/std calculation

    Returns:
        Tuple of (z_score, mean, std):
        - z_score: Number of standard deviations from mean [-inf, +inf]
        - mean: Rolling mean price over lookback period
        - std: Rolling standard deviation (floored at 1.0 to prevent division by zero)

    Example:
        >>> z, mean, std = calc_zscore(prices, 1000, 100)
        >>> if abs(z) > 2.0:  # Price is 2 std devs from mean
        ...     # Use as confirmation only, NOT primary signal
    """
    # Handle warmup period - use available data
    if tick < lookback:
        n = tick if tick > 0 else 1
    else:
        n = lookback

    # =========================================================================
    # STEP 1: Calculate rolling mean
    # =========================================================================
    total = 0.0
    count = 0
    start_idx = max(0, tick - n)

    for i in range(start_idx, tick):
        idx = i % 1000000  # Circular buffer index
        if prices[idx] > 0:  # Skip invalid prices
            total += prices[idx]
            count += 1

    # Need at least 2 data points for meaningful std
    if count < 2:
        return 0.0, 0.0, 1.0

    mean = total / count

    # =========================================================================
    # STEP 2: Calculate rolling standard deviation
    # =========================================================================
    sum_sq = 0.0
    for i in range(start_idx, tick):
        idx = i % 1000000
        if prices[idx] > 0:
            diff = prices[idx] - mean
            sum_sq += diff * diff

    std = np.sqrt(sum_sq / count)

    # Floor std to prevent division by zero
    if std < 1e-10:
        std = 1.0

    # =========================================================================
    # STEP 3: Calculate z-score for current price
    # =========================================================================
    current_idx = (tick - 1) % 1000000 if tick > 0 else 0
    current_price = prices[current_idx]

    if current_price <= 0:
        return 0.0, mean, std

    z_score = (current_price - mean) / std

    return z_score, mean, std
