"""
Power Law Position Sizer
ID 901: Bitcoin Power Law (R²=93%)
Max lines: 80
"""
import math
import time
from typing import Tuple

# Power Law Constants (R²=93%)
POWER_LAW_INTERCEPT = -17.0161223
POWER_LAW_SLOPE = 5.8451542
GENESIS_TIMESTAMP = 1230768000  # Jan 1, 2009


class PowerLawSizer:
    """
    ID 901: Bitcoin Power Law Model (R²=93%)

    CRITICAL: Use for SIZING, not direction!

    The Power Law has R²=93% over YEARS - use it to scale position sizes:
    - When undervalued -> bigger longs, smaller shorts
    - When overvalued -> smaller longs, bigger shorts

    Formula: log10(Price) = a + b * log10(days_since_genesis)
    Where: a = -17.0161223, b = 5.8451542
    """

    def __init__(self):
        self.a = POWER_LAW_INTERCEPT
        self.b = POWER_LAW_SLOPE
        self.epoch = GENESIS_TIMESTAMP

    def fair_value(self, timestamp: float = None) -> float:
        """Calculate fair value from Power Law."""
        if timestamp is None:
            timestamp = time.time()

        days = (timestamp - self.epoch) / 86400
        if days <= 0:
            return 0.0

        log_price = self.a + self.b * math.log10(days)
        return 10 ** log_price

    def get_size_multiplier(self, market_price: float, direction: int) -> Tuple[float, float]:
        """
        Get position size multiplier based on deviation from fair value.

        Args:
            market_price: Current market price
            direction: +1 for LONG, -1 for SHORT

        Returns:
            (size_multiplier, deviation_pct)

        Logic:
            - Undervalued + LONG -> 1.5x (conviction)
            - Undervalued + SHORT -> 0.5x (against value)
            - Overvalued + SHORT -> 1.5x (conviction)
            - Overvalued + LONG -> 0.5x (against value)
            - Near fair value -> 1.0x (neutral)
        """
        fair = self.fair_value()
        if fair <= 0:
            return 1.0, 0.0

        deviation_pct = (market_price - fair) / fair * 100

        # Determine multiplier based on alignment
        if deviation_pct < -10:  # More than 10% undervalued
            multiplier = 1.5 if direction == 1 else 0.5
        elif deviation_pct > 10:  # More than 10% overvalued
            multiplier = 0.5 if direction == 1 else 1.5
        else:  # Near fair value
            multiplier = 1.0

        return multiplier, deviation_pct
