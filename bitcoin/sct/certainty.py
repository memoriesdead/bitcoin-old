#!/usr/bin/env python3
"""
Certainty Checker - Core SCT Logic

Determines if we are 100% certain (99% CI) of 50.75%+ win rate.
This is the gatekeeper - no trading without mathematical certainty.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .config import SCTConfig, get_config
from .wilson import wilson_lower_bound, wilson_interval, trades_needed_for_certainty


class CertaintyStatus(Enum):
    """Status of certainty check."""
    CERTAIN = "certain"           # Lower bound >= 50.75%
    NEED_MORE_DATA = "need_data"  # Not enough trades yet
    NO_EDGE = "no_edge"           # Win rate too low


@dataclass
class CertaintyResult:
    """Result of certainty check."""
    status: CertaintyStatus
    wins: int
    total: int
    observed_wr: float
    lower_bound: float
    upper_bound: float
    target_wr: float
    confidence: float
    trades_needed: int  # Additional trades needed, 0 if certain
    message: str


class CertaintyChecker:
    """
    Check if we are 100% certain of 50.75%+ win rate.

    Uses Wilson score confidence interval.
    Only returns CERTAIN when lower bound of CI >= threshold.
    """

    def __init__(self, config: Optional[SCTConfig] = None):
        """
        Initialize certainty checker.

        Args:
            config: SCT configuration (uses default if None)
        """
        self.config = config or get_config()

    @property
    def min_win_rate(self) -> float:
        """Target win rate threshold."""
        return self.config.min_win_rate

    @property
    def confidence(self) -> float:
        """Confidence level."""
        return self.config.confidence_level

    def is_certain(self, wins: int, total: int) -> bool:
        """
        Check if we are certain of min_win_rate.

        Args:
            wins: Number of winning trades
            total: Total trades

        Returns:
            True only if lower bound of CI >= min_win_rate
        """
        if total < self.config.min_trades:
            return False

        lower = wilson_lower_bound(wins, total, self.confidence)
        return lower >= self.min_win_rate

    def check(self, wins: int, total: int) -> CertaintyResult:
        """
        Perform full certainty check.

        Args:
            wins: Number of winning trades
            total: Total trades

        Returns:
            CertaintyResult with full details
        """
        if total == 0:
            return CertaintyResult(
                status=CertaintyStatus.NEED_MORE_DATA,
                wins=0,
                total=0,
                observed_wr=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                target_wr=self.min_win_rate,
                confidence=self.confidence,
                trades_needed=self.config.min_trades,
                message="No trades recorded yet"
            )

        observed_wr = wins / total
        lower, upper = wilson_interval(wins, total, self.confidence)

        # Determine status
        if lower >= self.min_win_rate:
            status = CertaintyStatus.CERTAIN
            trades_needed = 0
            message = f"CERTAIN: Lower bound {lower*100:.2f}% >= {self.min_win_rate*100:.2f}%"
        elif observed_wr >= self.min_win_rate:
            status = CertaintyStatus.NEED_MORE_DATA
            trades_needed = trades_needed_for_certainty(observed_wr, self.min_win_rate, self.confidence)
            if trades_needed > 0:
                additional = max(0, trades_needed - total)
                message = f"Need ~{additional} more trades at {observed_wr*100:.1f}% WR"
            else:
                message = "Edge too small to prove with reasonable sample"
        else:
            status = CertaintyStatus.NO_EDGE
            trades_needed = -1
            message = f"Observed WR {observed_wr*100:.1f}% < threshold {self.min_win_rate*100:.2f}%"

        return CertaintyResult(
            status=status,
            wins=wins,
            total=total,
            observed_wr=observed_wr,
            lower_bound=lower,
            upper_bound=upper,
            target_wr=self.min_win_rate,
            confidence=self.confidence,
            trades_needed=trades_needed,
            message=message
        )

    def print_result(self, result: CertaintyResult):
        """Print formatted certainty result."""
        print(f"\n{'='*60}")
        print("CERTAINTY CHECK")
        print(f"{'='*60}")
        print(f"Trades: {result.wins}/{result.total}")
        print(f"Observed Win Rate: {result.observed_wr*100:.2f}%")
        print(f"\nWilson {result.confidence*100:.0f}% CI: "
              f"[{result.lower_bound*100:.2f}%, {result.upper_bound*100:.2f}%]")
        print(f"Lower Bound: {result.lower_bound*100:.2f}%")
        print(f"Target: {result.target_wr*100:.2f}%")
        print(f"\nStatus: {result.status.value.upper()}")
        print(f"  {result.message}")
        print(f"{'='*60}")


if __name__ == "__main__":
    checker = CertaintyChecker()

    # Test cases
    test_cases = [
        (0, 0, "No data"),
        (55, 100, "55% with 100 trades"),
        (580, 1000, "58% with 1000 trades"),
        (70, 100, "70% with 100 trades"),
        (45, 100, "45% - no edge"),
    ]

    for wins, total, desc in test_cases:
        print(f"\n>>> Test: {desc}")
        result = checker.check(wins, total)
        checker.print_result(result)
