#!/usr/bin/env python3
"""
OPTIMIZED BLOCKCHAIN SIGNAL GENERATOR
=====================================

Based on 8-period backtest analysis of 890 INFLOW signals:
- 6/8 periods profitable
- $8,593 total profit in 3.3 days
- 48.2% win rate but winners ($93.64) > losers ($68.50)

OPTIMAL STRATEGY:
- SHORT on large INFLOWS (100+ BTC, preferably 500+ BTC)
- Focus on coinbase/bitfinex for best edge
- Accept ~50% win rate, profit from asymmetric payoffs
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum


class SignalStrength(Enum):
    """Signal strength based on flow size."""
    WEAK = "WEAK"         # 100-200 BTC: ~46% win rate
    MEDIUM = "MEDIUM"     # 200-500 BTC: ~52% win rate
    STRONG = "STRONG"     # 500-1000 BTC: ~59% win rate
    WHALE = "WHALE"       # 1000+ BTC: ~55% win rate


class Exchange(Enum):
    """Exchanges ranked by edge consistency."""
    COINBASE = "coinbase"   # Best: 7 periods, $4,055 profit
    BITFINEX = "bitfinex"   # Good: 3 periods, $3,266 profit
    BINANCE = "binance"     # OK: 59% win rate on larger flows
    KRAKEN = "kraken"       # OK: 55% win rate
    OTHER = "other"         # Skip or reduce size


@dataclass
class OptimizedSignal:
    """Trading signal with confidence metrics."""
    timestamp: datetime
    exchange: str
    direction: str  # "SHORT" only - LONG not profitable
    flow_btc: float
    strength: SignalStrength
    expected_profit_usd: float
    win_probability: float
    position_size_pct: float  # Recommended position size

    @property
    def edge(self) -> float:
        """Expected value per $1 risked."""
        # Winners avg $93.64, losers avg $68.50 per BTC
        return self.win_probability * 93.64 - (1 - self.win_probability) * 68.50


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    min_flow_btc: float = 100.0
    preferred_min_btc: float = 500.0

    # Position sizing by signal strength
    position_pct = {
        SignalStrength.WEAK: 0.05,     # 5% of capital
        SignalStrength.MEDIUM: 0.10,   # 10% of capital
        SignalStrength.STRONG: 0.15,   # 15% of capital
        SignalStrength.WHALE: 0.20,    # 20% of capital
    }

    # Win rates by flow size (from backtest)
    win_rates = {
        SignalStrength.WEAK: 0.455,    # 100-200 BTC
        SignalStrength.MEDIUM: 0.520,  # 200-500 BTC
        SignalStrength.STRONG: 0.593,  # 500-1000 BTC
        SignalStrength.WHALE: 0.550,   # 1000+ BTC
    }

    # Exchange multipliers (profitability ranking)
    exchange_multipliers = {
        "coinbase": 1.0,   # Best consistency
        "bitfinex": 1.0,   # High profit per trade
        "binance": 0.8,    # Lower sample but good
        "kraken": 0.7,     # Decent
    }


class OptimizedSignalGenerator:
    """
    Generates trading signals based on blockchain flow analysis.

    Strategy: SHORT on large exchange inflows.
    - Not trying to predict with 100% accuracy
    - Capturing asymmetric payoff (winners > losers)
    - Position sizing based on signal strength
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.signals_generated = 0
        self.signals_by_strength: Dict[SignalStrength, int] = {s: 0 for s in SignalStrength}

    def get_signal_strength(self, flow_btc: float) -> SignalStrength:
        """Determine signal strength from flow size."""
        if flow_btc >= 1000:
            return SignalStrength.WHALE
        elif flow_btc >= 500:
            return SignalStrength.STRONG
        elif flow_btc >= 200:
            return SignalStrength.MEDIUM
        else:
            return SignalStrength.WEAK

    def should_trade(self, exchange: str, flow_btc: float) -> bool:
        """Filter signals worth trading."""
        # Skip small flows
        if flow_btc < self.config.min_flow_btc:
            return False

        # Skip unknown exchanges
        if exchange.lower() not in self.config.exchange_multipliers:
            return False

        return True

    def generate_signal(
        self,
        timestamp: datetime,
        exchange: str,
        direction: str,  # "INFLOW" or "OUTFLOW"
        flow_btc: float,
        current_price: float
    ) -> Optional[OptimizedSignal]:
        """
        Generate trading signal from flow observation.

        Only generates SHORT signals on INFLOW (proven edge).
        OUTFLOW → LONG has no edge in backtest.
        """
        # Only trade inflows (short signals)
        if direction.upper() != "INFLOW":
            return None

        # Check if worth trading
        if not self.should_trade(exchange, flow_btc):
            return None

        # Calculate signal metrics
        strength = self.get_signal_strength(flow_btc)
        win_prob = self.config.win_rates[strength]

        # Adjust for exchange
        exchange_lower = exchange.lower()
        multiplier = self.config.exchange_multipliers.get(exchange_lower, 0.5)

        # Expected profit calculation
        # Based on backtest: avg profit per 1 BTC trade
        avg_profit_per_btc = {
            "coinbase": 5.52,
            "bitfinex": 25.92,
            "binance": 41.33,
            "kraken": 31.39,
        }
        base_profit = avg_profit_per_btc.get(exchange_lower, 5.0)

        # Position sizing
        base_position = self.config.position_pct[strength]
        position_pct = base_position * multiplier

        signal = OptimizedSignal(
            timestamp=timestamp,
            exchange=exchange,
            direction="SHORT",
            flow_btc=flow_btc,
            strength=strength,
            expected_profit_usd=base_profit,
            win_probability=win_prob,
            position_size_pct=position_pct
        )

        self.signals_generated += 1
        self.signals_by_strength[strength] += 1

        return signal

    def get_stats(self) -> Dict:
        """Get signal generation statistics."""
        return {
            "total_signals": self.signals_generated,
            "by_strength": {s.value: c for s, c in self.signals_by_strength.items()}
        }


def main():
    """Test the signal generator."""
    print("=" * 70)
    print("OPTIMIZED BLOCKCHAIN SIGNAL GENERATOR")
    print("=" * 70)
    print()
    print("Strategy: SHORT on large INFLOWS")
    print()
    print("BACKTEST RESULTS (3.3 days, 890 trades):")
    print("  Win Rate:     48.2%")
    print("  Avg Winner:   $93.64")
    print("  Avg Loser:    $68.50")
    print("  Total Profit: $8,593")
    print("  Profitable:   6/8 periods")
    print()
    print("SIGNAL STRENGTH THRESHOLDS:")
    print("  WEAK:   100-200 BTC  → 45.5% win rate → 5% position")
    print("  MEDIUM: 200-500 BTC  → 52.0% win rate → 10% position")
    print("  STRONG: 500-1000 BTC → 59.3% win rate → 15% position")
    print("  WHALE:  1000+ BTC    → 55.0% win rate → 20% position")
    print()

    # Test signal generation
    generator = OptimizedSignalGenerator()

    test_cases = [
        ("coinbase", "INFLOW", 150.0),
        ("coinbase", "INFLOW", 350.0),
        ("coinbase", "INFLOW", 750.0),
        ("bitfinex", "INFLOW", 1200.0),
        ("binance", "OUTFLOW", 500.0),  # Should return None
    ]

    print("TEST SIGNALS:")
    print("-" * 70)

    for exchange, direction, amount in test_cases:
        signal = generator.generate_signal(
            timestamp=datetime.now(),
            exchange=exchange,
            direction=direction,
            flow_btc=amount,
            current_price=95000.0
        )

        if signal:
            print(f"  {exchange:12} {direction:8} {amount:8.1f} BTC → "
                  f"{signal.direction} {signal.strength.value:8} "
                  f"(win: {signal.win_probability:.1%}, "
                  f"pos: {signal.position_size_pct:.1%})")
        else:
            print(f"  {exchange:12} {direction:8} {amount:8.1f} BTC → NO SIGNAL")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
