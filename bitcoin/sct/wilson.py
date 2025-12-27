#!/usr/bin/env python3
"""
Wilson Score Confidence Interval

More accurate than normal approximation for binomial proportions.
Used to determine the MINIMUM win rate we can be certain about.

Reference: Wilson, E.B. (1927) "Probable Inference"
"""

import math
from typing import Tuple

from .config import get_config


def wilson_interval(wins: int, total: int, confidence: float = 0.99) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval.

    Returns (lower_bound, upper_bound) of the true proportion.

    The Wilson interval is more accurate than the normal approximation,
    especially for small samples or extreme proportions.

    Args:
        wins: Number of successful outcomes
        total: Total number of trials
        confidence: Confidence level (0.99 = 99%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0

    if wins < 0 or wins > total:
        raise ValueError(f"Invalid wins={wins} for total={total}")

    config = get_config()
    z = config.get_z_score(confidence)
    z2 = z * z

    p = wins / total

    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return lower, upper


def wilson_lower_bound(wins: int, total: int, confidence: float = 0.99) -> float:
    """
    Calculate lower bound of Wilson score confidence interval.

    This is the KEY function:
    - Returns the MINIMUM win rate we can be confident about
    - Must be >= 50.75% to trade

    Args:
        wins: Number of winning trades
        total: Total trades
        confidence: Confidence level (default 99%)

    Returns:
        Lower bound of win rate (0.0 to 1.0)
    """
    lower, _ = wilson_interval(wins, total, confidence)
    return lower


def wilson_upper_bound(wins: int, total: int, confidence: float = 0.99) -> float:
    """
    Calculate upper bound of Wilson score confidence interval.

    Args:
        wins: Number of winning trades
        total: Total trades
        confidence: Confidence level (default 99%)

    Returns:
        Upper bound of win rate (0.0 to 1.0)
    """
    _, upper = wilson_interval(wins, total, confidence)
    return upper


def trades_needed_for_certainty(observed_wr: float, target_wr: float = 0.5075,
                                 confidence: float = 0.99) -> int:
    """
    Calculate minimum trades needed to prove win rate >= target.

    Args:
        observed_wr: Observed win rate (e.g., 0.58 for 58%)
        target_wr: Target win rate to prove (default 50.75%)
        confidence: Confidence level (default 99%)

    Returns:
        Minimum number of trades needed, or -1 if impossible
    """
    if observed_wr <= target_wr:
        return -1  # Cannot prove target if observed is lower

    # Binary search for minimum n
    lo, hi = 10, 100000

    while lo < hi:
        mid = (lo + hi) // 2
        wins = int(mid * observed_wr)
        lower = wilson_lower_bound(wins, mid, confidence)

        if lower >= target_wr:
            hi = mid
        else:
            lo = mid + 1

    # Verify final answer
    wins = int(lo * observed_wr)
    if wilson_lower_bound(wins, lo, confidence) >= target_wr:
        return lo

    return -1


if __name__ == "__main__":
    # Demo: Show CI for various sample sizes
    print("Wilson Score Interval Demo (99% confidence)")
    print("=" * 60)

    test_cases = [
        (55, 100),   # 55%
        (580, 1000), # 58%
        (70, 100),   # 70%
        (90, 100),   # 90%
    ]

    for wins, total in test_cases:
        lower, upper = wilson_interval(wins, total, 0.99)
        wr = wins / total
        print(f"\nWins: {wins}/{total} (Observed: {wr*100:.1f}%)")
        print(f"  99% CI: [{lower*100:.2f}%, {upper*100:.2f}%]")
        print(f"  Lower bound: {lower*100:.2f}%")
        print(f"  Certain of 50.75%+? {'YES' if lower >= 0.5075 else 'NO'}")
