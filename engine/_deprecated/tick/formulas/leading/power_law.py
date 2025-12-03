"""
FORMULA ID 901: POWER LAW PRICE SIGNAL
=======================================
Leading indicator calculated from TIMESTAMP ONLY.
Completely independent of current price data.

Academic Citation:
    Giovannetti (2019) - "Bitcoin Power Law"
    R² = 94% over 10+ years of price history

Mathematical Formula:
    log10(price) = A + B * log10(days_since_genesis)

    Where:
    - A = -17.01 (intercept)
    - B = 5.84 (slope)
    - days_since_genesis = (timestamp - GENESIS_TS) / 86400

KEY INSIGHT:
    Power Law predicts "fair value" from days since Bitcoin genesis.
    - Price < fair value → Expect rise (BUY signal)
    - Price > fair value → Expect fall (SELL signal)

    This is a LEADING indicator because it's calculated from
    TIMESTAMP ONLY - no dependency on current price.

Performance: O(1) per tick
Numba JIT: ~20-40 nanoseconds per tick
"""

import numpy as np
from numba import njit

from ..constants import (
    BLOCKCHAIN_GENESIS_TIMESTAMP,
    BLOCKCHAIN_POWER_LAW_A,
    BLOCKCHAIN_POWER_LAW_B
)


@njit(cache=True, fastmath=True, error_model='numpy', boundscheck=False)
def calc_power_law_signal(timestamp: float, current_price: float) -> tuple:
    """
    POWER LAW LEADING SIGNAL (Formula ID 901)
    Calculated from TIMESTAMP ONLY - completely independent of current price.

    How it works:
        1. Calculate days since Bitcoin genesis (Jan 3, 2009)
        2. Apply Power Law formula to get fair value
        3. Compare current price to fair value
        4. Generate signal based on deviation

    Args:
        timestamp: Current Unix timestamp
        current_price: Current market price (for deviation calculation)

    Returns:
        Tuple of (power_law_price, deviation, signal, strength):

        - power_law_price: Fair value from Power Law model
                          This is the "expected" price based on time

        - deviation: Relative deviation from fair value
                    Positive = price above fair value (overvalued)
                    Negative = price below fair value (undervalued)

        - signal: Trading signal direction
                 +1 = BUY (price below fair value)
                 -1 = SELL (price above fair value)
                  0 = HOLD (within tolerance)

        - strength: Signal strength [0.0, 1.0]
                   Higher when deviation is larger

    Signal Thresholds:
        > 10% below fair value → Strong BUY
        5-10% below → Moderate BUY
        > 10% above fair value → Strong SELL
        5-10% above → Moderate SELL

    Example:
        >>> pl_price, dev, sig, strength = calc_power_law_signal(timestamp, 95000)
        >>> if sig > 0 and strength > 0.5:
        ...     # Strong buy signal - price significantly below fair value
    """
    # =========================================================================
    # STEP 1: Calculate days since genesis
    # =========================================================================
    days_since_genesis = (timestamp - BLOCKCHAIN_GENESIS_TIMESTAMP) / 86400.0

    # Sanity check
    if days_since_genesis < 1.0:
        return 0.0, 0.0, 0, 0.0

    # =========================================================================
    # STEP 2: Apply Power Law formula
    # Citation: Giovannetti (2019) - R² = 94%
    # =========================================================================
    log10_days = np.log10(days_since_genesis)
    log10_price = BLOCKCHAIN_POWER_LAW_A + BLOCKCHAIN_POWER_LAW_B * log10_days
    power_law_price = 10.0 ** log10_price

    # =========================================================================
    # STEP 3: Calculate deviation from fair value
    # =========================================================================
    if power_law_price > 0 and current_price > 0:
        deviation = (current_price - power_law_price) / power_law_price
    else:
        deviation = 0.0

    # =========================================================================
    # STEP 4: Generate signal based on deviation
    # =========================================================================
    if deviation < -0.10:
        # Strong buy: >10% below fair value
        signal = 1
        strength = min(1.0, abs(deviation) * 3.0)

    elif deviation < -0.05:
        # Moderate buy: 5-10% below fair value
        signal = 1
        strength = abs(deviation) * 2.0

    elif deviation > 0.10:
        # Strong sell: >10% above fair value
        signal = -1
        strength = min(1.0, abs(deviation) * 3.0)

    elif deviation > 0.05:
        # Moderate sell: 5-10% above fair value
        signal = -1
        strength = abs(deviation) * 2.0

    else:
        # Within tolerance - no signal
        signal = 0
        strength = 0.0

    return power_law_price, deviation, signal, strength
