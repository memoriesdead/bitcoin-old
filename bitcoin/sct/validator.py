#!/usr/bin/env python3
"""
Strategy Validator - Gate Trading on Mathematical Certainty

Validates strategies meet the 50.75% threshold before allowing live trading.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from .config import SCTConfig, get_config
from .certainty import CertaintyChecker, CertaintyStatus, CertaintyResult
from .strategy_tracker import StrategyStats


class ValidationDecision(Enum):
    """Trading decision based on validation."""
    TRADE = "trade"           # Certain - safe to trade
    WAIT = "wait"             # Need more data
    REJECT = "reject"         # No edge proven


@dataclass
class ValidationResult:
    """Result of strategy validation."""
    strategy: str
    decision: ValidationDecision
    certainty: CertaintyResult
    recommended_size: float  # Position size recommendation
    message: str


class StrategyValidator:
    """
    Validate strategies meet certainty threshold.

    Acts as gatekeeper - no trading without mathematical proof.
    """

    def __init__(self, config: Optional[SCTConfig] = None):
        """
        Initialize validator.

        Args:
            config: SCT configuration
        """
        self.config = config or get_config()
        self.checker = CertaintyChecker(self.config)

    def validate(self, strategy: str, wins: int, total: int) -> ValidationResult:
        """
        Validate if strategy is safe to trade.

        Args:
            strategy: Strategy name
            wins: Winning trades
            total: Total trades

        Returns:
            ValidationResult with decision
        """
        certainty = self.checker.check(wins, total)

        if certainty.status == CertaintyStatus.CERTAIN:
            decision = ValidationDecision.TRADE
            # Use lower bound for position sizing (conservative)
            size = self._calculate_position_size(certainty.lower_bound)
            message = f"APPROVED for trading with {size*100:.2f}% position size"
        elif certainty.status == CertaintyStatus.NEED_MORE_DATA:
            decision = ValidationDecision.WAIT
            size = 0.0
            message = f"Wait for {certainty.trades_needed - total} more trades"
        else:
            decision = ValidationDecision.REJECT
            size = 0.0
            message = "No proven edge - do not trade"

        return ValidationResult(
            strategy=strategy,
            decision=decision,
            certainty=certainty,
            recommended_size=size,
            message=message
        )

    def validate_stats(self, stats: StrategyStats) -> ValidationResult:
        """
        Validate from StrategyStats object.

        Args:
            stats: StrategyStats from tracker

        Returns:
            ValidationResult
        """
        return self.validate(stats.name, stats.wins, stats.total)

    def validate_batch(self, strategies: List[StrategyStats]) -> List[ValidationResult]:
        """
        Validate multiple strategies.

        Args:
            strategies: List of StrategyStats

        Returns:
            List of ValidationResults
        """
        return [self.validate_stats(s) for s in strategies]

    def _calculate_position_size(self, win_rate: float) -> float:
        """
        Calculate position size based on win rate.

        Uses quarter-Kelly for safety.
        """
        if win_rate <= 0.5:
            return 0.0

        # Kelly: f* = (b*p - q) / b
        p = win_rate
        q = 1 - p
        b = self.config.risk_reward_ratio

        kelly = (b * p - q) / b
        quarter_kelly = kelly * self.config.kelly_fraction

        # Apply bounds
        size = max(self.config.min_position_pct,
                   min(quarter_kelly, self.config.max_position_pct))

        return size

    def print_result(self, result: ValidationResult):
        """Print formatted validation result."""
        print(f"\n{'='*60}")
        print(f"VALIDATION: {result.strategy}")
        print(f"{'='*60}")
        print(f"Decision: {result.decision.value.upper()}")
        print(f"Win Rate: {result.certainty.observed_wr*100:.2f}%")
        print(f"Lower Bound: {result.certainty.lower_bound*100:.2f}%")
        print(f"Threshold: {result.certainty.target_wr*100:.2f}%")

        if result.decision == ValidationDecision.TRADE:
            print(f"\nPosition Size: {result.recommended_size*100:.2f}% of capital")

        print(f"\n{result.message}")
        print(f"{'='*60}")


if __name__ == "__main__":
    validator = StrategyValidator()

    # Test validations
    test_cases = [
        ("CERTAIN_STRAT", 700, 1000),   # 70% - certain
        ("PENDING_STRAT", 560, 1000),   # 56% - need more data
        ("NO_EDGE_STRAT", 480, 1000),   # 48% - no edge
    ]

    for name, wins, total in test_cases:
        result = validator.validate(name, wins, total)
        validator.print_result(result)
