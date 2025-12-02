#!/usr/bin/env python3
"""
================================================================================
BLOCKCHAIN TRADING SIGNAL GENERATOR (LAYER 2)
================================================================================

ARCHITECTURE REFERENCE: docs/BLOCKCHAIN_PIPELINE_ARCHITECTURE.md

POSITION IN PIPELINE:
    This is a LAYER 2 component - generates trading signals from LAYER 3 data.
    Uses: PureBlockchainPrice (L3) for Power Law valuation
    Used by: BlockchainUnifiedFeed (L1), Trading engines

SIGNAL GENERATION:
    Combines pure blockchain data sources for trading signals.
    This is your EDGE - derived entirely from blockchain, NO APIs.

COMPONENTS:
    1. Power Law Fair Value - Long-term price target (R² > 95%)
    2. Support/Resistance bands - 42%/238% of fair value
    3. Position sizing based on deviation from fair value

SIGNAL OUTPUT:
    - signal:       -1 (sell) to +1 (buy)
    - strength:     STRONG/MODERATE/WEAK
    - action:       BUY/SELL/HOLD
    - position_size: 0.5x to 2.0x based on deviation

DEVIATION-BASED SIGNALS:
    - Price < Fair - 30%: STRONG BUY
    - Price < Fair - 15%: MODERATE BUY
    - Price > Fair + 15%: MODERATE SELL
    - Price > Fair + 30%: STRONG SELL
    - Within ±15%: HOLD (neutral)

SOURCES:
    - Power Law: https://bitbo.io/tools/power-law-calculator/
    - Theory: https://giovannisantostasi.medium.com/the-bitcoin-power-law-theory-962dfaf99ee9
================================================================================
"""

import time
import math
from dataclasses import dataclass
from typing import Tuple, Optional

try:
    from .pure_blockchain_price import PureBlockchainPrice, POWER_LAW_EPOCH
except ImportError:
    from pure_blockchain_price import PureBlockchainPrice, POWER_LAW_EPOCH


@dataclass
class BlockchainTradingSignal:
    """Complete trading signal from blockchain data."""
    timestamp: float

    # Power Law Valuation (long-term)
    fair_value: float           # Power Law fair price
    support_price: float        # 42% of fair (floor)
    resistance_price: float     # 238% of fair (ceiling)

    # Current position in range
    current_price: float
    deviation_pct: float        # % from fair value
    position_in_range: float    # 0.0 = at support, 1.0 = at resistance

    # Trading signals
    signal: float               # -1 (sell) to +1 (buy)
    signal_strength: str        # STRONG/MODERATE/WEAK
    recommended_action: str     # BUY/SELL/HOLD

    # Position sizing
    position_size_multiplier: float  # 0.5x to 2.0x based on deviation


class BlockchainTradingEngine:
    """
    PURE BLOCKCHAIN TRADING ENGINE

    Uses Power Law valuation to generate trading signals:
    - When price < fair value: BUY (stronger signal as price approaches support)
    - When price > fair value: SELL (stronger signal as price approaches resistance)
    - Position size scales with deviation from fair value

    This is your competitive edge:
    - Based on 95%+ correlated model with 14+ years of data
    - Derived ONLY from blockchain time (block timestamps)
    - No exchange API dependencies
    """

    def __init__(self):
        self.power_law = PureBlockchainPrice()
        self.last_signal: Optional[BlockchainTradingSignal] = None

    def get_signal(self, current_price: float) -> BlockchainTradingSignal:
        """
        Generate trading signal based on current price vs Power Law fair value.

        Args:
            current_price: Current market price (from any source)

        Returns:
            BlockchainTradingSignal with complete analysis
        """
        timestamp = time.time()

        # Get Power Law prices
        fair = self.power_law.calculate_fair_value()
        support = self.power_law.calculate_support()
        resistance = self.power_law.calculate_resistance()

        # Calculate deviation from fair value
        deviation_pct = ((current_price - fair) / fair) * 100

        # Calculate position in support-resistance range
        # 0.0 = at support, 0.5 = at fair value, 1.0 = at resistance
        if resistance > support:
            position_in_range = (current_price - support) / (resistance - support)
            position_in_range = max(0.0, min(1.0, position_in_range))
        else:
            position_in_range = 0.5

        # Generate trading signal
        # Signal is based on deviation from fair value
        # Clip to [-50%, +50%] for signal calculation
        capped_deviation = max(-50, min(50, deviation_pct))
        signal = -capped_deviation / 50  # Negative deviation = buy signal

        # Determine signal strength
        abs_deviation = abs(deviation_pct)
        if abs_deviation > 30:
            strength = "STRONG"
        elif abs_deviation > 15:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        # Determine recommended action
        if signal > 0.3:
            action = "BUY"
        elif signal < -0.3:
            action = "SELL"
        else:
            action = "HOLD"

        # Position size multiplier
        # Scale from 0.5x (at fair value) to 2.0x (at extremes)
        position_size = 0.5 + abs(signal) * 1.5

        result = BlockchainTradingSignal(
            timestamp=timestamp,
            fair_value=fair,
            support_price=support,
            resistance_price=resistance,
            current_price=current_price,
            deviation_pct=deviation_pct,
            position_in_range=position_in_range,
            signal=signal,
            signal_strength=strength,
            recommended_action=action,
            position_size_multiplier=position_size,
        )

        self.last_signal = result
        return result

    def get_position_bias(self, current_price: float) -> float:
        """
        Get position bias for trading system.

        Returns:
            Float from -1 (strong sell bias) to +1 (strong buy bias)
        """
        signal = self.get_signal(current_price)
        return signal.signal

    def should_enter_long(self, current_price: float, threshold: float = 0.2) -> bool:
        """Check if conditions favor entering a long position."""
        signal = self.get_signal(current_price)
        return signal.signal > threshold

    def should_enter_short(self, current_price: float, threshold: float = 0.2) -> bool:
        """Check if conditions favor entering a short position."""
        signal = self.get_signal(current_price)
        return signal.signal < -threshold

    def get_target_price(self) -> float:
        """Get fair value as target price."""
        return self.power_law.calculate_fair_value()

    def get_stop_loss_long(self) -> float:
        """Get support as stop loss for long positions."""
        return self.power_law.calculate_support()

    def get_stop_loss_short(self) -> float:
        """Get resistance as stop loss for short positions."""
        return self.power_law.calculate_resistance()

    def print_signal(self, current_price: float):
        """Print human-readable signal analysis."""
        signal = self.get_signal(current_price)

        print()
        print("=" * 60)
        print("BLOCKCHAIN TRADING SIGNAL")
        print("=" * 60)
        print()
        print("POWER LAW ANALYSIS:")
        print(f"  Fair Value:       ${signal.fair_value:>12,.2f}")
        print(f"  Support (floor):  ${signal.support_price:>12,.2f}")
        print(f"  Resistance (cap): ${signal.resistance_price:>12,.2f}")
        print()
        print("CURRENT POSITION:")
        print(f"  Market Price:     ${signal.current_price:>12,.2f}")
        print(f"  Deviation:        {signal.deviation_pct:>+11.1f}%")
        print(f"  Range Position:   {signal.position_in_range:>11.1%}")
        print()
        print("TRADING SIGNAL:")
        print(f"  Signal:           {signal.signal:>+11.2f}")
        print(f"  Strength:         {signal.signal_strength:>12}")
        print(f"  Action:           {signal.recommended_action:>12}")
        print(f"  Position Size:    {signal.position_size_multiplier:>11.1f}x")
        print()
        print("=" * 60)


# Convenience function for quick signal
def get_trading_signal(current_price: float) -> BlockchainTradingSignal:
    """Get trading signal for current price."""
    return BlockchainTradingEngine().get_signal(current_price)


if __name__ == "__main__":
    import sys

    # Get current price from command line or use default
    current_price = float(sys.argv[1]) if len(sys.argv) > 1 else 97000

    engine = BlockchainTradingEngine()
    engine.print_signal(current_price)

    # Show days calculation
    print()
    print("PURE BLOCKCHAIN DATA:")
    print("-" * 60)
    days = engine.power_law.days_since_genesis()
    print(f"  Days since Jan 1, 2009: {days:,.2f}")
    print(f"  This is the ONLY input - pure blockchain time!")
    print()
