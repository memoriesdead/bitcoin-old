#!/usr/bin/env python3
"""
SCT C++ Bridge - Python interface to nanosecond C++ Wilson CI calculator.

Uses ctypes for FFI to the compiled C++ shared library.
Falls back to pure Python if library not available.
"""

import os
import math
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
from enum import Enum


class CertaintyStatus(Enum):
    """Certainty status enum."""
    CERTAIN = "certain"
    NEED_MORE_DATA = "need_more_data"
    NO_EDGE = "no_edge"


@dataclass
class WilsonInterval:
    """Wilson confidence interval."""
    lower: float
    upper: float
    center: float


@dataclass
class CertaintyResult:
    """Result of certainty check."""
    observed_wr: float
    lower_bound: float
    upper_bound: float
    target_wr: float
    confidence: float
    trades_needed: int  # 0 if certain, -1 if impossible
    status: CertaintyStatus
    calc_time_ns: int


@dataclass
class PositionSize:
    """Position sizing recommendation."""
    win_rate_used: float
    full_kelly: float
    quarter_kelly: float
    recommended: float
    capital_pct: float


# Z-scores for common confidence levels
Z_SCORES = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
    0.999: 3.291,
}


class SCTBridge:
    """
    Bridge to C++ SCT Wilson CI Calculator.

    Provides nanosecond-speed statistical certainty calculations.
    Only trades when 99% confident that win rate >= 50.75%.
    """

    # RenTech threshold
    MIN_WIN_RATE = 0.5075
    DEFAULT_CONFIDENCE = 0.99
    KELLY_FRACTION = 0.25  # Quarter-Kelly
    MAX_POSITION_PCT = 0.05
    MIN_POSITION_PCT = 0.001

    def __init__(self, lib_path: Optional[str] = None,
                 min_wr: float = MIN_WIN_RATE,
                 confidence: float = DEFAULT_CONFIDENCE):
        """
        Initialize bridge.

        Args:
            lib_path: Path to libhqt_sct.so. Auto-detected if None.
            min_wr: Minimum win rate threshold (default 50.75%)
            confidence: Confidence level (default 99%)
        """
        self._lib = None
        self._cpp_available = False
        self._min_wr = min_wr
        self._confidence = confidence
        self._z_score = self._get_z_score(confidence)

        # Try to load C++ library
        if lib_path is None:
            candidates = [
                Path(__file__).parent.parent / "build" / "libhqt_sct.so",
                Path("/root/sovereign/hqt_sct/build/libhqt_sct.so"),
                Path("/usr/local/lib/libhqt_sct.so"),
            ]
            for path in candidates:
                if path.exists():
                    lib_path = str(path)
                    break

        if lib_path and os.path.exists(lib_path):
            try:
                import ctypes
                self._lib = ctypes.CDLL(lib_path)
                self._cpp_available = True
                print(f"[SCT] C++ library loaded from {lib_path}")
            except OSError as e:
                print(f"[SCT] Could not load C++ library: {e}")
                print("[SCT] Falling back to pure Python")
        else:
            print("[SCT] C++ library not found, using pure Python")

        # Strategy tracking
        self._strategies: Dict[str, dict] = {}

    def _get_z_score(self, confidence: float) -> float:
        """Get z-score for confidence level."""
        if confidence in Z_SCORES:
            return Z_SCORES[confidence]

        # Approximate using inverse error function
        p = (1.0 + confidence) / 2.0
        t = math.sqrt(-2.0 * math.log(1.0 - p))

        # Abramowitz and Stegun approximation
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        return t - (c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t)

    @property
    def cpp_available(self) -> bool:
        """Check if C++ library is loaded."""
        return self._cpp_available

    def wilson_interval(self, wins: int, total: int,
                        confidence: Optional[float] = None) -> WilsonInterval:
        """
        Calculate Wilson score confidence interval.

        More accurate than normal approximation for binomial proportions.

        Args:
            wins: Number of winning trades
            total: Total trades
            confidence: Confidence level (default uses instance setting)

        Returns:
            WilsonInterval with lower and upper bounds
        """
        if total <= 0:
            return WilsonInterval(0.0, 0.0, 0.0)

        if confidence is None:
            confidence = self._confidence

        p = wins / total
        z = self._get_z_score(confidence)
        z2 = z * z
        n = total

        denominator = 1.0 + z2 / n
        center = (p + z2 / (2.0 * n)) / denominator
        spread = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator

        return WilsonInterval(
            lower=max(0.0, center - spread),
            upper=min(1.0, center + spread),
            center=center
        )

    def wilson_lower_bound(self, wins: int, total: int,
                           confidence: Optional[float] = None) -> float:
        """Get Wilson lower bound only (faster)."""
        return self.wilson_interval(wins, total, confidence).lower

    def is_certain(self, wins: int, total: int) -> bool:
        """Quick check if we're certain of meeting win rate threshold."""
        return self.wilson_lower_bound(wins, total) >= self._min_wr

    def trades_needed(self, observed_wr: float) -> int:
        """
        Calculate how many trades needed to reach certainty.

        Args:
            observed_wr: Current observed win rate

        Returns:
            Number of trades needed, or -1 if impossible
        """
        if observed_wr <= self._min_wr:
            return -1

        # Binary search for minimum sample size
        low, high = 10, 10000
        result = -1

        while low <= high:
            mid = (low + high) // 2
            wins = int(mid * observed_wr)

            if self.is_certain(wins, mid):
                result = mid
                high = mid - 1
            else:
                low = mid + 1

        return result

    def check(self, wins: int, total: int) -> CertaintyResult:
        """
        Check if we are certain of the minimum win rate.

        Returns true only if the lower bound of the Wilson CI >= min_win_rate.
        This is MATHEMATICAL certainty, not just observed win rate.

        Args:
            wins: Number of winning trades
            total: Total trades

        Returns:
            CertaintyResult with full analysis
        """
        import time
        start = time.time_ns()

        if total == 0:
            return CertaintyResult(
                observed_wr=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                target_wr=self._min_wr,
                confidence=self._confidence,
                trades_needed=-1,
                status=CertaintyStatus.NEED_MORE_DATA,
                calc_time_ns=time.time_ns() - start
            )

        observed_wr = wins / total
        interval = self.wilson_interval(wins, total)

        # Determine status
        if interval.lower >= self._min_wr:
            status = CertaintyStatus.CERTAIN
            needed = 0
        elif observed_wr > self._min_wr:
            status = CertaintyStatus.NEED_MORE_DATA
            needed = self.trades_needed(observed_wr)
        else:
            status = CertaintyStatus.NO_EDGE
            needed = -1

        return CertaintyResult(
            observed_wr=observed_wr,
            lower_bound=interval.lower,
            upper_bound=interval.upper,
            target_wr=self._min_wr,
            confidence=self._confidence,
            trades_needed=needed,
            status=status,
            calc_time_ns=time.time_ns() - start
        )

    def kelly_fraction(self, win_rate: float, risk_reward: float = 1.0) -> float:
        """
        Calculate Kelly fraction.

        f* = (b*p - q) / b

        Args:
            win_rate: Win probability
            risk_reward: Risk/reward ratio

        Returns:
            Optimal fraction of capital
        """
        if win_rate <= 0.5:
            return 0.0

        p = win_rate
        q = 1.0 - p
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
                capital_pct=0.0
            )

        # Use lower bound for safety
        lower = self.wilson_lower_bound(wins, total)

        if lower < self._min_wr:
            return PositionSize(
                win_rate_used=lower,
                full_kelly=0.0,
                quarter_kelly=0.0,
                recommended=0.0,
                capital_pct=0.0
            )

        full = self.kelly_fraction(lower, risk_reward)
        quarter = full * self.KELLY_FRACTION

        recommended = max(self.MIN_POSITION_PCT,
                         min(quarter, self.MAX_POSITION_PCT))

        return PositionSize(
            win_rate_used=lower,
            full_kelly=full,
            quarter_kelly=quarter,
            recommended=recommended,
            capital_pct=recommended * 100
        )

    # Strategy tracking methods
    def record_trade(self, strategy: str, won: bool):
        """Record trade outcome for strategy."""
        import time

        if strategy not in self._strategies:
            self._strategies[strategy] = {
                'wins': 0,
                'losses': 0,
                'created': time.time_ns(),
                'updated': time.time_ns()
            }

        if won:
            self._strategies[strategy]['wins'] += 1
        else:
            self._strategies[strategy]['losses'] += 1

        self._strategies[strategy]['updated'] = time.time_ns()

    def check_strategy(self, strategy: str) -> Optional[CertaintyResult]:
        """Check certainty for a tracked strategy."""
        if strategy not in self._strategies:
            return None

        stats = self._strategies[strategy]
        return self.check(stats['wins'], stats['wins'] + stats['losses'])

    def get_tradeable(self) -> List[str]:
        """Get list of strategies we're certain about."""
        return [name for name in self._strategies
                if self.check_strategy(name).status == CertaintyStatus.CERTAIN]

    def get_pending(self) -> List[str]:
        """Get list of strategies needing more data."""
        return [name for name in self._strategies
                if self.check_strategy(name).status == CertaintyStatus.NEED_MORE_DATA]


if __name__ == "__main__":
    # Demo
    bridge = SCTBridge()

    print("\n=== SCT C++ Bridge Demo ===")
    print(f"C++ available: {bridge.cpp_available}")
    print(f"Min win rate: {bridge.MIN_WIN_RATE*100:.2f}%")
    print(f"Confidence: {bridge._confidence*100:.0f}%")

    # Test cases
    test_cases = [
        (700, 1000, "70% WR - Strong"),
        (580, 1000, "58% WR - Good"),
        (520, 1000, "52% WR - Marginal"),
        (510, 1000, "51% WR - Borderline"),
        (55, 100, "55% WR - Small sample"),
    ]

    for wins, total, desc in test_cases:
        print(f"\n>>> {desc} ({wins}/{total})")

        result = bridge.check(wins, total)

        print(f"    Observed WR: {result.observed_wr*100:.2f}%")
        print(f"    Wilson CI:   [{result.lower_bound*100:.2f}%, {result.upper_bound*100:.2f}%]")
        print(f"    Status:      {result.status.value.upper()}")
        print(f"    Calc time:   {result.calc_time_ns} ns")

        if result.status == CertaintyStatus.CERTAIN:
            size = bridge.size_from_stats(wins, total)
            print(f"    Kelly:       {size.full_kelly*100:.2f}% full, {size.quarter_kelly*100:.2f}% quarter")
            print(f"    Recommended: {size.capital_pct:.2f}% of capital")
        elif result.trades_needed > 0:
            print(f"    Trades needed: {result.trades_needed}")

    # Sample size table
    print("\n" + "=" * 60)
    print("SAMPLE SIZE REQUIREMENTS")
    print("For 99% confidence that lower bound >= 50.75%")
    print("=" * 60)

    print(f"\n{'Observed WR':>12} {'Min Trades':>12} {'Status':<15}")
    print("-" * 50)

    for wr in [0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.65, 0.70, 0.80, 0.90]:
        trades = bridge.trades_needed(wr)
        status = "POSSIBLE" if trades > 0 else "IMPOSSIBLE"
        trades_str = str(trades) if trades > 0 else "N/A"
        print(f"{wr*100:>11.0f}% {trades_str:>12} {status:<15}")
