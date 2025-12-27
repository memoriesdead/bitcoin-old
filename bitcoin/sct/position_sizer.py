#!/usr/bin/env python3
"""
Kelly Position Sizer - Optimal Position Sizing Based on Edge

Uses Kelly Criterion with safety adjustments (quarter-Kelly).
Only sizes positions when edge is mathematically proven.
"""

from dataclasses import dataclass
from typing import Optional

from .config import SCTConfig, get_config
from .wilson import wilson_lower_bound


@dataclass
class PositionSize:
    """Position sizing result."""
    win_rate_used: float      # Win rate used (lower bound for safety)
    full_kelly: float         # Full Kelly fraction
    quarter_kelly: float      # Quarter Kelly (recommended)
    recommended: float        # Final recommendation with bounds
    capital_pct: float        # As percentage
    message: str


class KellyPositionSizer:
    """
    Calculate optimal position sizes using Kelly Criterion.

    Key features:
    - Uses CI lower bound (not observed WR) for safety
    - Quarter-Kelly by default
    - Hard limits on max/min position size
    """

    def __init__(self, config: Optional[SCTConfig] = None):
        """
        Initialize position sizer.

        Args:
            config: SCT configuration
        """
        self.config = config or get_config()

    def kelly_fraction(self, win_rate: float, risk_reward: float = 1.0) -> float:
        """
        Calculate Kelly fraction.

        f* = (b*p - q) / b

        Where:
            p = probability of win
            q = probability of loss (1-p)
            b = risk/reward ratio

        Args:
            win_rate: Win probability (0-1)
            risk_reward: Risk/reward ratio (default 1:1)

        Returns:
            Optimal fraction of capital to bet
        """
        if win_rate <= 0.5:
            return 0.0

        p = win_rate
        q = 1 - p
        b = risk_reward

        return (b * p - q) / b

    def size_from_stats(self, wins: int, total: int,
                        risk_reward: float = 1.0) -> PositionSize:
        """
        Calculate position size from trade stats.

        Uses Wilson CI lower bound for safety.

        Args:
            wins: Winning trades
            total: Total trades
            risk_reward: Risk/reward ratio

        Returns:
            PositionSize with recommendations
        """
        if total == 0:
            return PositionSize(
                win_rate_used=0.0,
                full_kelly=0.0,
                quarter_kelly=0.0,
                recommended=0.0,
                capital_pct=0.0,
                message="No trades - cannot size"
            )

        # Use lower bound for safety
        observed_wr = wins / total
        lower_bound = wilson_lower_bound(wins, total, self.config.confidence_level)

        # Check if tradeable
        if lower_bound < self.config.min_win_rate:
            return PositionSize(
                win_rate_used=lower_bound,
                full_kelly=0.0,
                quarter_kelly=0.0,
                recommended=0.0,
                capital_pct=0.0,
                message=f"Lower bound {lower_bound*100:.2f}% < threshold {self.config.min_win_rate*100:.2f}%"
            )

        # Calculate Kelly fractions
        full_kelly = self.kelly_fraction(lower_bound, risk_reward)
        quarter_kelly = full_kelly * self.config.kelly_fraction

        # Apply bounds
        recommended = max(self.config.min_position_pct,
                         min(quarter_kelly, self.config.max_position_pct))

        return PositionSize(
            win_rate_used=lower_bound,
            full_kelly=full_kelly,
            quarter_kelly=quarter_kelly,
            recommended=recommended,
            capital_pct=recommended * 100,
            message=f"Using lower bound {lower_bound*100:.2f}% (observed: {observed_wr*100:.2f}%)"
        )

    def size_from_win_rate(self, win_rate: float,
                           risk_reward: float = 1.0) -> PositionSize:
        """
        Calculate position size from known win rate.

        Use this only when win rate is already validated.

        Args:
            win_rate: Validated win rate
            risk_reward: Risk/reward ratio

        Returns:
            PositionSize
        """
        if win_rate < self.config.min_win_rate:
            return PositionSize(
                win_rate_used=win_rate,
                full_kelly=0.0,
                quarter_kelly=0.0,
                recommended=0.0,
                capital_pct=0.0,
                message=f"Win rate {win_rate*100:.2f}% < threshold"
            )

        full_kelly = self.kelly_fraction(win_rate, risk_reward)
        quarter_kelly = full_kelly * self.config.kelly_fraction

        recommended = max(self.config.min_position_pct,
                         min(quarter_kelly, self.config.max_position_pct))

        return PositionSize(
            win_rate_used=win_rate,
            full_kelly=full_kelly,
            quarter_kelly=quarter_kelly,
            recommended=recommended,
            capital_pct=recommended * 100,
            message=f"Sized for {win_rate*100:.2f}% win rate"
        )

    def print_size(self, size: PositionSize):
        """Print position sizing result."""
        print(f"\n{'='*50}")
        print("POSITION SIZING")
        print(f"{'='*50}")
        print(f"Win Rate Used: {size.win_rate_used*100:.2f}%")
        print(f"Full Kelly: {size.full_kelly*100:.2f}%")
        print(f"Quarter Kelly: {size.quarter_kelly*100:.2f}%")
        print(f"Recommended: {size.recommended*100:.2f}%")
        print(f"\n{size.message}")
        print(f"{'='*50}")


if __name__ == "__main__":
    sizer = KellyPositionSizer()

    # Demo sizing
    test_cases = [
        (700, 1000, "70% WR"),
        (580, 1000, "58% WR"),
        (510, 1000, "51% WR - borderline"),
        (490, 1000, "49% WR - no edge"),
    ]

    for wins, total, desc in test_cases:
        print(f"\n>>> {desc}")
        size = sizer.size_from_stats(wins, total)
        sizer.print_size(size)
