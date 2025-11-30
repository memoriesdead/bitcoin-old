#!/usr/bin/env python3
"""
PURE BLOCKCHAIN PRICE - POWER LAW MODEL
========================================
The most accurate blockchain-derived price using the Power Law model.

Based on: Giovanni Santostasi's Bitcoin Power Law research
Source: https://bitcoinfairprice.com/

Formula: Price = 10^(a + b * log10(days_since_genesis))
Where:
  - a = -17.01 (intercept)
  - b = 5.82 (slope)
  - days_since_genesis = days since Jan 3, 2009

This model has shown 93%+ correlation with actual Bitcoin price over 14+ years.

The Power Law derives price from:
1. TIME (pure blockchain data - block timestamps)
2. NETWORK EFFECT (implicit in the model)
3. ADOPTION CURVE (S-curve built into power law)

NO EXCHANGE DATA NEEDED - Pure mathematical derivation from blockchain time.
"""

import time
import math
from dataclasses import dataclass
from typing import Tuple


# Bitcoin Genesis Block Timestamp
GENESIS_TIMESTAMP = 1231006505  # Jan 3, 2009 18:15:05 UTC

# Power Law Reference Date (Jan 1, 2009 00:00:00 UTC)
# Some models use this instead of genesis block for cleaner math
POWER_LAW_EPOCH = 1230768000  # Jan 1, 2009 00:00:00 UTC


@dataclass
class PowerLawPrice:
    """Price derived from Power Law model."""
    timestamp: float
    days_since_genesis: float

    # Derived prices
    fair_value: float           # Central Power Law price
    support_price: float        # -2 std deviations (strong support)
    resistance_price: float     # +2 std deviations (strong resistance)

    # Valuation
    current_price: float = 0    # For comparison (optional)
    deviation_pct: float = 0    # % from fair value
    valuation: str = "FAIR"     # UNDERVALUED / FAIR / OVERVALUED


class PureBlockchainPrice:
    """
    PURE BLOCKCHAIN PRICE ENGINE

    Derives Bitcoin's fair value using ONLY blockchain data (time).
    No exchange APIs, no external data sources.

    The Power Law model has shown remarkable accuracy over 14+ years:
    - R² > 0.93 correlation with actual price
    - Predicted price within 1 standard deviation most of the time
    - Works across 8+ orders of magnitude ($0.01 to $100k+)

    Usage:
        engine = PureBlockchainPrice()
        price = engine.get_price()  # Current fair value
        analysis = engine.analyze(current_api_price)  # Valuation analysis
    """

    def __init__(self):
        # Power Law constants from multiple sources:
        # - bitcoinfairprice.com: Price = 1.0117e-17 * days^5.82
        # - bitbo.io: Price = 10^(-17.0161223 + 5.8451542 * log10(days))
        #
        # Using bitbo.io's more precise parameters (7 decimal places)

        # HIGH PRECISION log-form parameters (from bitbo.io)
        self.a = -17.0161223  # Intercept
        self.b = 5.8451542    # Slope

        # Reference date: Jan 1, 2009 (not genesis block)
        self.epoch = POWER_LAW_EPOCH

        # Support/Resistance multipliers (from bitcoinfairprice.com)
        # Bottom = Fair * 0.42 (BTC rarely goes below this)
        # Top = Fair * 2.38 (inverse of 0.42)
        self.support_multiplier = 0.42
        self.resistance_multiplier = 2.38

    def days_since_genesis(self, timestamp: float = None) -> float:
        """Calculate days since Power Law epoch (Jan 1, 2009)."""
        if timestamp is None:
            timestamp = time.time()
        return (timestamp - self.epoch) / 86400

    def calculate_fair_value(self, timestamp: float = None) -> float:
        """
        Calculate Power Law fair value.

        This is the mathematically-derived "true price" of Bitcoin
        based purely on blockchain time data.

        Formula: Price = 10^(a + b * log10(days))
        Where: a = -17.0161223, b = 5.8451542
        Source: https://bitbo.io/tools/power-law-calculator/
        """
        days = self.days_since_genesis(timestamp)
        if days <= 0:
            return 0.0

        # Use log-form formula: 10^(a + b * log10(days))
        log_price = self.a + self.b * math.log10(days)
        return 10 ** log_price

    def calculate_support(self, timestamp: float = None) -> float:
        """
        Calculate support price (bottom floor).

        This is the "floor" price - historically BTC rarely goes below this.
        Bottom = Fair Price * 0.42 (from bitcoinfairprice.com)
        """
        fair = self.calculate_fair_value(timestamp)
        return fair * self.support_multiplier

    def calculate_resistance(self, timestamp: float = None) -> float:
        """
        Calculate resistance price (top ceiling).

        This is the "ceiling" price - historically BTC rarely exceeds this.
        Top = Fair Price * 2.38 (inverse of 0.42)
        """
        fair = self.calculate_fair_value(timestamp)
        return fair * self.resistance_multiplier

    def get_price(self, timestamp: float = None) -> PowerLawPrice:
        """
        Get complete Power Law price analysis.

        Returns fair value, support, and resistance levels.
        """
        if timestamp is None:
            timestamp = time.time()

        days = self.days_since_genesis(timestamp)
        fair = self.calculate_fair_value(timestamp)
        support = self.calculate_support(timestamp)
        resistance = self.calculate_resistance(timestamp)

        return PowerLawPrice(
            timestamp=timestamp,
            days_since_genesis=days,
            fair_value=fair,
            support_price=support,
            resistance_price=resistance,
        )

    def analyze(self, current_price: float, timestamp: float = None) -> PowerLawPrice:
        """
        Analyze current price against Power Law fair value.

        Returns:
            PowerLawPrice with valuation analysis
        """
        price_data = self.get_price(timestamp)
        price_data.current_price = current_price

        # Calculate deviation from fair value
        if price_data.fair_value > 0:
            deviation = (current_price - price_data.fair_value) / price_data.fair_value * 100
            price_data.deviation_pct = deviation

            # Determine valuation
            if current_price < price_data.support_price:
                price_data.valuation = "EXTREMELY UNDERVALUED"
            elif current_price < price_data.fair_value * 0.8:
                price_data.valuation = "UNDERVALUED"
            elif current_price > price_data.resistance_price:
                price_data.valuation = "EXTREMELY OVERVALUED"
            elif current_price > price_data.fair_value * 1.2:
                price_data.valuation = "OVERVALUED"
            else:
                price_data.valuation = "FAIR"

        return price_data

    def get_trading_signal(self, current_price: float) -> Tuple[float, str]:
        """
        Get trading signal based on deviation from fair value.

        Returns:
            (signal, description)
            signal: -1 (strong sell) to +1 (strong buy)
        """
        analysis = self.analyze(current_price)
        deviation = analysis.deviation_pct

        # Convert deviation to signal
        # -50% deviation = +1 (strong buy)
        # +50% deviation = -1 (strong sell)
        signal = -deviation / 50
        signal = max(-1, min(1, signal))  # Clip to [-1, 1]

        if signal > 0.5:
            desc = "STRONG BUY - Significantly below fair value"
        elif signal > 0.2:
            desc = "BUY - Below fair value"
        elif signal < -0.5:
            desc = "STRONG SELL - Significantly above fair value"
        elif signal < -0.2:
            desc = "SELL - Above fair value"
        else:
            desc = "NEUTRAL - Near fair value"

        return signal, desc

    def print_analysis(self, current_price: float = None):
        """Print comprehensive Power Law analysis."""
        price_data = self.get_price()

        print()
        print("=" * 60)
        print("PURE BLOCKCHAIN PRICE - POWER LAW MODEL")
        print("=" * 60)
        print()
        print("Formula: Price = 10^(-17.0161223 + 5.8451542 * log10(days))")
        print("Source: https://bitbo.io/tools/power-law-calculator/")
        print(f"Days since genesis: {price_data.days_since_genesis:,.0f}")
        print()
        print("DERIVED PRICES (pure blockchain):")
        print(f"  Support (42%):     ${price_data.support_price:>12,.2f}")
        print(f"  FAIR VALUE:        ${price_data.fair_value:>12,.2f}")
        print(f"  Resistance (238%): ${price_data.resistance_price:>12,.2f}")

        if current_price:
            analysis = self.analyze(current_price)
            print()
            print("VALUATION ANALYSIS:")
            print(f"  Current Price:    ${current_price:>12,.2f}")
            print(f"  Deviation:        {analysis.deviation_pct:>+11.1f}%")
            print(f"  Status:           {analysis.valuation}")

            signal, desc = self.get_trading_signal(current_price)
            print()
            print("TRADING SIGNAL:")
            print(f"  Signal: {signal:+.2f}")
            print(f"  {desc}")

        print()
        print("=" * 60)


# Convenience function
def get_blockchain_price() -> float:
    """Get current fair value from blockchain Power Law."""
    return PureBlockchainPrice().calculate_fair_value()


def get_blockchain_analysis(current_price: float) -> dict:
    """Get complete analysis comparing current price to blockchain fair value."""
    engine = PureBlockchainPrice()
    analysis = engine.analyze(current_price)
    signal, desc = engine.get_trading_signal(current_price)

    return {
        'fair_value': analysis.fair_value,
        'support': analysis.support_price,
        'resistance': analysis.resistance_price,
        'deviation_pct': analysis.deviation_pct,
        'valuation': analysis.valuation,
        'signal': signal,
        'signal_description': desc,
    }


if __name__ == "__main__":
    import sys

    # Get current price if provided
    current_price = float(sys.argv[1]) if len(sys.argv) > 1 else 90750

    engine = PureBlockchainPrice()
    engine.print_analysis(current_price)

    # Show formula breakdown
    print()
    print("FORMULA BREAKDOWN:")
    print("-" * 60)
    days = engine.days_since_genesis()
    log_days = math.log10(days)
    log_price = engine.a + engine.b * log_days
    price = 10 ** log_price

    print(f"  Days since Jan 1, 2009: {days:,.2f}")
    print(f"  log10(days):            {log_days:.6f}")
    print(f"  a:                      {engine.a}")
    print(f"  b:                      {engine.b}")
    print(f"  a + b * log10(days):    {engine.a} + {engine.b} * {log_days:.6f}")
    print(f"                        = {log_price:.6f}")
    print(f"  10^{log_price:.6f}:       ${price:,.2f}")
    print()
    print("This is PURE BLOCKCHAIN data - derived only from block time!")
