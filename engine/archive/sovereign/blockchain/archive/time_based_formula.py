#!/usr/bin/env python3
"""
TIME-BASED TRADING FORMULA
==========================

Derived from 8-hour analysis of 7,141 flow observations.

KEY DISCOVERY:
  Individual flows don't predict price direction (50% accuracy).
  But TIMING matters - specific UTC hours have strong edge.

FORMULA:
  On INFLOW >= 100 BTC:
    - SHORT hours (02, 04, 14, 20, 22): INFLOW → SHORT (68% win rate)
    - LONG hours (09, 12, 18, 19, 21, 23): INFLOW → LONG (68% win rate)
    - SKIP hours (all others): No trade

BACKTEST RESULTS (890 signals, 100+ BTC flows):
  SHORT signals: 138 trades, 68.1% win rate, +8.05% total
  LONG signals:  243 trades, 68.3% win rate, +6.10% total
  COMBINED:      381 trades, 68.2% win rate, +14.15% total

vs Original formula: ~50% win rate, -57% loss
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple
from enum import Enum


class SignalType(Enum):
    """Signal types based on time analysis."""
    SHORT = "SHORT"
    LONG = "LONG"
    SKIP = "SKIP"


@dataclass
class TimeBasedConfig:
    """Configuration for time-based formula."""
    # Minimum flow size in BTC (from backtest: 100+ BTC has edge)
    min_flow_btc: float = 100.0

    # UTC hours where INFLOW → SHORT (price goes DOWN)
    # 20:00 = 90.9%, 22:00 = 71.4%, 14:00 = 60.0%, 02:00 = 66.7%, 04:00 = 62.5%
    short_hours: set = None

    # UTC hours where INFLOW → LONG (contrarian: price goes UP)
    # 23:00 = 88.9%, 21:00 = 72.7%, 12:00 = 69.0%, 09:00 = 64.4%
    long_hours: set = None

    # Expected profit per trade (for position sizing)
    expected_profit_pct: float = 0.04  # 0.04% average profit per trade

    def __post_init__(self):
        if self.short_hours is None:
            # Hours where INFLOW predicts price DOWN (>=60% correct)
            self.short_hours = {2, 4, 14, 20, 22}
        if self.long_hours is None:
            # Hours where INFLOW predicts price UP (<=40% down = >=60% up)
            self.long_hours = {9, 12, 18, 19, 21, 23}


@dataclass
class TimeBasedSignal:
    """Trading signal from time-based formula."""
    timestamp: datetime
    exchange: str
    direction: SignalType
    flow_btc: float
    hour_utc: int
    confidence: float  # Based on historical accuracy
    expected_pnl_pct: float  # Expected profit %


class TimeBasedFormula:
    """
    Time-based trading formula.

    Uses historical analysis to determine when INFLOW signals
    predict price UP vs DOWN based on UTC hour.
    """

    # Historical accuracy by hour (from correlation.db analysis)
    HOUR_ACCURACY = {
        # SHORT hours (INFLOW → price DOWN)
        2: 0.667,   # 66.7% correct
        4: 0.625,   # 62.5% correct
        14: 0.600,  # 60.0% correct
        20: 0.909,  # 90.9% correct - BEST
        22: 0.714,  # 71.4% correct
        # LONG hours (INFLOW → price UP, contrarian)
        9: 0.644,   # 64.4% UP
        12: 0.690,  # 69.0% UP
        18: 0.649,  # 64.9% UP
        19: 0.611,  # 61.1% UP
        21: 0.727,  # 72.7% UP
        23: 0.889,  # 88.9% UP - BEST contrarian
    }

    # Expected profit per trade by hour
    HOUR_EXPECTED_PROFIT = {
        # SHORT hours (profit from SHORT position)
        2: 0.0449,
        4: 0.0250,
        14: 0.0462,
        20: 0.1082,  # Best profit hour
        22: 0.0700,
        # LONG hours (profit from LONG position)
        9: 0.0165,
        12: 0.0203,
        18: 0.0210,
        19: 0.0299,
        21: 0.0436,
        23: 0.0384,
    }

    def __init__(self, config: Optional[TimeBasedConfig] = None):
        self.config = config or TimeBasedConfig()
        self.signals_generated = 0
        self.signals_skipped = 0

    def get_signal_type(self, hour_utc: int) -> SignalType:
        """Determine signal type based on UTC hour."""
        if hour_utc in self.config.short_hours:
            return SignalType.SHORT
        elif hour_utc in self.config.long_hours:
            return SignalType.LONG
        else:
            return SignalType.SKIP

    def process_flow(
        self,
        timestamp: datetime,
        exchange: str,
        direction: str,  # "INFLOW" or "OUTFLOW"
        flow_btc: float,
        current_price: float = None
    ) -> Optional[TimeBasedSignal]:
        """
        Process a flow observation and generate signal if appropriate.

        Only generates signals for:
          - INFLOW direction (outflows not predictive)
          - Flow >= min_flow_btc (100 BTC by default)
          - During tradeable hours (SHORT or LONG hours)
        """
        # Only trade INFLOW (OUTFLOW not predictive in our data)
        if direction.upper() != "INFLOW":
            return None

        # Check minimum flow size
        if flow_btc < self.config.min_flow_btc:
            return None

        # Get UTC hour
        if timestamp.tzinfo is None:
            # Assume UTC if no timezone
            hour_utc = timestamp.hour
        else:
            hour_utc = timestamp.astimezone(timezone.utc).hour

        # Determine signal type
        signal_type = self.get_signal_type(hour_utc)

        if signal_type == SignalType.SKIP:
            self.signals_skipped += 1
            return None

        # Get confidence and expected profit
        confidence = self.HOUR_ACCURACY.get(hour_utc, 0.5)
        expected_pnl = self.HOUR_EXPECTED_PROFIT.get(hour_utc, 0.0)

        signal = TimeBasedSignal(
            timestamp=timestamp,
            exchange=exchange,
            direction=signal_type,
            flow_btc=flow_btc,
            hour_utc=hour_utc,
            confidence=confidence,
            expected_pnl_pct=expected_pnl
        )

        self.signals_generated += 1
        return signal

    def get_stats(self) -> dict:
        """Get formula statistics."""
        return {
            "signals_generated": self.signals_generated,
            "signals_skipped": self.signals_skipped,
            "config": {
                "min_flow_btc": self.config.min_flow_btc,
                "short_hours": list(self.config.short_hours),
                "long_hours": list(self.config.long_hours)
            }
        }


def format_signal(signal: TimeBasedSignal) -> str:
    """Format signal for logging."""
    return (
        f"[{signal.timestamp.strftime('%H:%M:%S')} UTC] "
        f"{signal.direction.value} {signal.exchange.upper()} "
        f"| {signal.flow_btc:.1f} BTC | Hour {signal.hour_utc:02d} "
        f"| Conf: {signal.confidence:.1%} | Exp: +{signal.expected_pnl_pct:.3%}"
    )


def main():
    """Test the time-based formula."""
    print("=" * 70)
    print("TIME-BASED TRADING FORMULA")
    print("=" * 70)
    print()
    print("DISCOVERY FROM 8-HOUR DATA ANALYSIS:")
    print("  - Individual flows: ~50% accuracy (random)")
    print("  - Time-based formula: 68% accuracy (+14% profit)")
    print()
    print("TRADING HOURS (UTC):")
    print()
    print("  SHORT HOURS (INFLOW → price DOWN):")
    for hour in sorted([2, 4, 14, 20, 22]):
        acc = TimeBasedFormula.HOUR_ACCURACY.get(hour, 0)
        pnl = TimeBasedFormula.HOUR_EXPECTED_PROFIT.get(hour, 0)
        print(f"    {hour:02d}:00 UTC - {acc:.1%} win rate, +{pnl:.3%} profit")
    print()
    print("  LONG HOURS (INFLOW → price UP, contrarian):")
    for hour in sorted([9, 12, 18, 19, 21, 23]):
        acc = TimeBasedFormula.HOUR_ACCURACY.get(hour, 0)
        pnl = TimeBasedFormula.HOUR_EXPECTED_PROFIT.get(hour, 0)
        print(f"    {hour:02d}:00 UTC - {acc:.1%} win rate, +{pnl:.3%} profit")
    print()
    print("  SKIP HOURS: 00, 01, 03, 05, 06, 07, 08, 10, 11, 13, 15, 16, 17")
    print()

    # Test signals
    formula = TimeBasedFormula()

    test_cases = [
        # (hour, exchange, direction, btc)
        (20, "coinbase", "INFLOW", 150.0),  # SHORT hour, should trade
        (23, "coinbase", "INFLOW", 200.0),  # LONG hour, should trade (contrarian)
        (15, "binance", "INFLOW", 300.0),   # SKIP hour, no trade
        (14, "kraken", "INFLOW", 50.0),     # Too small, no trade
        (22, "gemini", "OUTFLOW", 500.0),   # Outflow, no trade
    ]

    print("TEST SIGNALS:")
    print("-" * 70)

    for hour, exchange, direction, btc in test_cases:
        ts = datetime(2025, 12, 25, hour, 30, 0, tzinfo=timezone.utc)
        signal = formula.process_flow(ts, exchange, direction, btc)

        if signal:
            print(f"  {format_signal(signal)}")
        else:
            print(f"  [{hour:02d}:30:00 UTC] {exchange} {direction} {btc:.1f} BTC → NO SIGNAL")

    print()
    print(f"Stats: {formula.signals_generated} generated, {formula.signals_skipped} skipped")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
