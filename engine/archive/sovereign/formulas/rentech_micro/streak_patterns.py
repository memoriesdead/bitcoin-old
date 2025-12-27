"""
Streak Pattern Analysis
=======================

Formula IDs: 72051-72060

Analyzes consecutive day patterns (streaks) for predictive power.
E.g., what happens after 3 consecutive down days?

RenTech insight: Simple patterns like "3 down days in a row"
can have statistically significant predictive power.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


@dataclass
class SequencePattern:
    """Detected sequence pattern."""
    pattern: str  # e.g., "DDD" for 3 down days
    count: int
    avg_next_return: float
    win_rate: float
    significance: float  # t-stat


@dataclass
class ConditionalProbability:
    """Conditional probability given pattern."""
    pattern: str
    prob_up: float
    prob_down: float
    expected_return: float


@dataclass
class StreakSignal:
    """Signal from streak analysis."""
    direction: int
    confidence: float
    current_streak: int
    streak_type: str  # 'up', 'down', or 'mixed'
    pattern_stats: Optional[SequencePattern]


class StreakAnalyzer:
    """
    Analyzes streak patterns in returns.
    """

    def __init__(self, max_streak: int = 5):
        self.max_streak = max_streak
        self.pattern_stats: Dict[str, SequencePattern] = {}

    def _returns_to_pattern(self, returns: np.ndarray) -> str:
        """Convert returns to pattern string."""
        return ''.join('U' if r > 0 else 'D' for r in returns)

    def build_statistics(self, returns: np.ndarray, min_samples: int = 30):
        """Build pattern statistics from historical data."""
        n = len(returns)
        pattern_returns: Dict[str, List[float]] = defaultdict(list)

        for length in range(1, self.max_streak + 1):
            for i in range(n - length - 1):
                pattern = self._returns_to_pattern(returns[i:i + length])
                next_return = returns[i + length]
                pattern_returns[pattern].append(next_return)

        # Compute statistics
        for pattern, next_returns in pattern_returns.items():
            if len(next_returns) >= min_samples:
                avg_ret = np.mean(next_returns)
                std_ret = np.std(next_returns)
                win_rate = sum(1 for r in next_returns if r > 0) / len(next_returns)
                t_stat = avg_ret / (std_ret / np.sqrt(len(next_returns)) + 1e-10)

                self.pattern_stats[pattern] = SequencePattern(
                    pattern=pattern,
                    count=len(next_returns),
                    avg_next_return=avg_ret,
                    win_rate=win_rate,
                    significance=abs(t_stat),
                )

    def get_current_streak(self, returns: np.ndarray) -> Tuple[int, str]:
        """Get current streak length and type."""
        if len(returns) == 0:
            return 0, 'none'

        streak = 1
        streak_type = 'up' if returns[-1] > 0 else 'down'

        for i in range(len(returns) - 2, -1, -1):
            current_type = 'up' if returns[i] > 0 else 'down'
            if current_type == streak_type:
                streak += 1
            else:
                break

        return streak, streak_type

    def get_pattern_prediction(self, pattern: str) -> Optional[SequencePattern]:
        """Get prediction for a pattern."""
        return self.pattern_stats.get(pattern)

    def generate_signal(self, returns: np.ndarray) -> StreakSignal:
        """Generate trading signal from streak analysis."""
        streak_len, streak_type = self.get_current_streak(returns)

        # Get pattern for current streak
        pattern = self._returns_to_pattern(returns[-min(streak_len, self.max_streak):])
        stats = self.get_pattern_prediction(pattern)

        if stats is None or stats.significance < 2.0:
            return StreakSignal(
                direction=0,
                confidence=0.0,
                current_streak=streak_len,
                streak_type=streak_type,
                pattern_stats=stats,
            )

        # Direction from expected return
        if stats.avg_next_return > 0.001:
            direction = 1
        elif stats.avg_next_return < -0.001:
            direction = -1
        else:
            direction = 0

        # Confidence from significance and win rate
        confidence = min(1.0, (stats.significance - 2.0) / 2.0) * abs(stats.win_rate - 0.5) * 2

        return StreakSignal(
            direction=direction,
            confidence=confidence,
            current_streak=streak_len,
            streak_type=streak_type,
            pattern_stats=stats,
        )


# =============================================================================
# FORMULA IMPLEMENTATIONS (72051-72060)
# =============================================================================

class Streak2DownSignal:
    """
    Formula 72051: 2 Down Days Signal

    Trades after exactly 2 consecutive down days.
    Historical analysis shows mean reversion tendency.
    """

    FORMULA_ID = 72051

    def __init__(self):
        self.analyzer = StreakAnalyzer(max_streak=5)
        self.is_fitted = False

    def fit(self, returns: np.ndarray):
        """Build statistics."""
        self.analyzer.build_statistics(returns)
        self.is_fitted = True

    def generate_signal(self, returns: np.ndarray) -> StreakSignal:
        streak_len, streak_type = self.analyzer.get_current_streak(returns)

        if streak_len == 2 and streak_type == 'down':
            # 2 down days - check pattern stats
            stats = self.analyzer.get_pattern_prediction('DD')

            if stats and stats.significance >= 2.0:
                direction = 1 if stats.avg_next_return > 0 else -1
                confidence = min(1.0, stats.significance / 4.0)
            else:
                # Default: mean reversion
                direction = 1
                confidence = 0.5
        else:
            direction = 0
            confidence = 0.0

        return StreakSignal(
            direction=direction,
            confidence=confidence,
            current_streak=streak_len,
            streak_type=streak_type,
            pattern_stats=self.analyzer.get_pattern_prediction('DD'),
        )


class Streak3DownSignal:
    """
    Formula 72052: 3 Down Days Signal

    Trades after 3 consecutive down days.
    Stronger mean reversion signal than 2 days.
    """

    FORMULA_ID = 72052

    def __init__(self):
        self.analyzer = StreakAnalyzer(max_streak=5)
        self.is_fitted = False

    def fit(self, returns: np.ndarray):
        self.analyzer.build_statistics(returns)
        self.is_fitted = True

    def generate_signal(self, returns: np.ndarray) -> StreakSignal:
        streak_len, streak_type = self.analyzer.get_current_streak(returns)

        if streak_len >= 3 and streak_type == 'down':
            stats = self.analyzer.get_pattern_prediction('DDD')

            if stats and stats.significance >= 2.0:
                direction = 1 if stats.avg_next_return > 0 else -1
                confidence = min(1.0, stats.significance / 4.0)
            else:
                direction = 1  # Default mean reversion
                confidence = 0.6

            # Boost confidence for longer streaks
            if streak_len > 3:
                confidence = min(1.0, confidence * 1.2)
        else:
            direction = 0
            confidence = 0.0

        return StreakSignal(
            direction=direction,
            confidence=confidence,
            current_streak=streak_len,
            streak_type=streak_type,
            pattern_stats=self.analyzer.get_pattern_prediction('DDD'),
        )


class Streak2UpSignal:
    """
    Formula 72053: 2 Up Days Signal

    Trades after 2 consecutive up days.
    Can be momentum or mean reversion depending on data.
    """

    FORMULA_ID = 72053

    def __init__(self):
        self.analyzer = StreakAnalyzer()

    def fit(self, returns: np.ndarray):
        self.analyzer.build_statistics(returns)

    def generate_signal(self, returns: np.ndarray) -> StreakSignal:
        streak_len, streak_type = self.analyzer.get_current_streak(returns)

        if streak_len == 2 and streak_type == 'up':
            stats = self.analyzer.get_pattern_prediction('UU')

            if stats and stats.significance >= 2.0:
                direction = 1 if stats.avg_next_return > 0 else -1
                confidence = min(1.0, stats.significance / 4.0)
            else:
                direction = 1  # Default: momentum continuation
                confidence = 0.4
        else:
            direction = 0
            confidence = 0.0

        return StreakSignal(
            direction=direction,
            confidence=confidence,
            current_streak=streak_len,
            streak_type=streak_type,
            pattern_stats=self.analyzer.get_pattern_prediction('UU'),
        )


class Streak3UpSignal:
    """
    Formula 72054: 3 Up Days Signal

    Trades after 3 consecutive up days.
    """

    FORMULA_ID = 72054

    def __init__(self):
        self.analyzer = StreakAnalyzer()

    def fit(self, returns: np.ndarray):
        self.analyzer.build_statistics(returns)

    def generate_signal(self, returns: np.ndarray) -> StreakSignal:
        streak_len, streak_type = self.analyzer.get_current_streak(returns)

        if streak_len >= 3 and streak_type == 'up':
            stats = self.analyzer.get_pattern_prediction('UUU')

            if stats and stats.significance >= 2.0:
                direction = 1 if stats.avg_next_return > 0 else -1
                confidence = min(1.0, stats.significance / 4.0)
            else:
                direction = 1  # Default: momentum
                confidence = 0.5
        else:
            direction = 0
            confidence = 0.0

        return StreakSignal(
            direction=direction,
            confidence=confidence,
            current_streak=streak_len,
            streak_type=streak_type,
            pattern_stats=self.analyzer.get_pattern_prediction('UUU'),
        )


class MixedStreakSignal:
    """
    Formula 72055: Mixed Streak Signal

    Analyzes alternating patterns like UDUD.
    """

    FORMULA_ID = 72055

    def __init__(self):
        self.analyzer = StreakAnalyzer()

    def fit(self, returns: np.ndarray):
        self.analyzer.build_statistics(returns)

    def generate_signal(self, returns: np.ndarray) -> StreakSignal:
        if len(returns) < 4:
            return StreakSignal(0, 0.0, 0, 'insufficient', None)

        # Check for alternating pattern
        pattern = self.analyzer._returns_to_pattern(returns[-4:])

        if pattern in ['UDUD', 'DUDU']:
            stats = self.analyzer.get_pattern_prediction(pattern)

            if stats and stats.significance >= 2.0:
                direction = 1 if stats.avg_next_return > 0 else -1
                confidence = min(1.0, stats.significance / 4.0)
            else:
                # Alternating tends to continue
                direction = 1 if returns[-1] < 0 else -1
                confidence = 0.4
        else:
            direction = 0
            confidence = 0.0

        return StreakSignal(
            direction=direction,
            confidence=confidence,
            current_streak=0,
            streak_type='mixed',
            pattern_stats=self.analyzer.get_pattern_prediction(pattern) if len(pattern) <= 5 else None,
        )


class StreakBreakSignal:
    """
    Formula 72056: Streak Break Signal

    Trades when a streak is broken.
    """

    FORMULA_ID = 72056

    def __init__(self, min_streak: int = 3):
        self.min_streak = min_streak
        self.prev_streak: int = 0
        self.prev_type: str = 'none'

    def generate_signal(self, returns: np.ndarray) -> StreakSignal:
        if len(returns) < 2:
            return StreakSignal(0, 0.0, 0, 'insufficient', None)

        # Current direction
        current_up = returns[-1] > 0

        # Check previous streak
        streak = 1
        for i in range(len(returns) - 2, -1, -1):
            was_up = returns[i] > 0
            if was_up == (not current_up):  # Opposite direction
                streak += 1
            else:
                break

        # Detect streak break
        if streak >= self.min_streak:
            # Previous streak just broke
            if current_up:
                # Broke down streak - now up
                direction = 1
            else:
                # Broke up streak - now down
                direction = -1
            confidence = min(1.0, streak / 5.0)
        else:
            direction = 0
            confidence = 0.0

        return StreakSignal(
            direction=direction,
            confidence=confidence,
            current_streak=streak,
            streak_type='break',
            pattern_stats=None,
        )


class StreakContinueSignal:
    """
    Formula 72057: Streak Continuation Signal

    Trades with the streak (momentum).
    """

    FORMULA_ID = 72057

    def __init__(self, min_streak: int = 2):
        self.min_streak = min_streak
        self.analyzer = StreakAnalyzer()

    def fit(self, returns: np.ndarray):
        self.analyzer.build_statistics(returns)

    def generate_signal(self, returns: np.ndarray) -> StreakSignal:
        streak_len, streak_type = self.analyzer.get_current_streak(returns)

        if streak_len >= self.min_streak:
            # Get continuation stats
            pattern = 'U' * streak_len if streak_type == 'up' else 'D' * streak_len
            pattern = pattern[-self.analyzer.max_streak:]
            stats = self.analyzer.get_pattern_prediction(pattern)

            if stats and stats.win_rate > 0.5:
                direction = 1 if streak_type == 'up' else -1
                confidence = (stats.win_rate - 0.5) * 2
            else:
                direction = 0
                confidence = 0.0
        else:
            direction = 0
            confidence = 0.0

        return StreakSignal(
            direction=direction,
            confidence=confidence,
            current_streak=streak_len,
            streak_type=streak_type,
            pattern_stats=None,
        )


class ConditionalStreakSignal:
    """
    Formula 72058: Conditional Streak Signal

    Streak signal conditional on volatility.
    """

    FORMULA_ID = 72058

    def __init__(self, vol_lookback: int = 20):
        self.vol_lookback = vol_lookback
        self.analyzer = StreakAnalyzer()

    def fit(self, returns: np.ndarray):
        self.analyzer.build_statistics(returns)

    def generate_signal(self, returns: np.ndarray) -> StreakSignal:
        if len(returns) < self.vol_lookback:
            return StreakSignal(0, 0.0, 0, 'insufficient', None)

        streak_len, streak_type = self.analyzer.get_current_streak(returns)

        # Volatility context
        vol = np.std(returns[-self.vol_lookback:])
        vol_percentile = np.sum(np.std(returns[i:i + self.vol_lookback]) < vol
                                for i in range(len(returns) - self.vol_lookback)) / (len(returns) - self.vol_lookback)

        if streak_len >= 2:
            if vol_percentile < 0.3:
                # Low vol - momentum works
                direction = 1 if streak_type == 'up' else -1
                confidence = 0.5
            else:
                # High vol - mean reversion works
                direction = -1 if streak_type == 'up' else 1
                confidence = 0.4
        else:
            direction = 0
            confidence = 0.0

        return StreakSignal(
            direction=direction,
            confidence=confidence,
            current_streak=streak_len,
            streak_type=streak_type,
            pattern_stats=None,
        )


class VolatilityStreakSignal:
    """
    Formula 72059: Volatility Streak Signal

    Analyzes streaks in volatility, not returns.
    """

    FORMULA_ID = 72059

    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def generate_signal(self, returns: np.ndarray) -> StreakSignal:
        if len(returns) < self.lookback * 3:
            return StreakSignal(0, 0.0, 0, 'insufficient', None)

        # Rolling volatility
        vols = [np.std(returns[i:i + self.lookback])
               for i in range(len(returns) - self.lookback + 1)]

        if len(vols) < 3:
            return StreakSignal(0, 0.0, 0, 'insufficient', None)

        # Vol streak
        vol_changes = np.diff(vols)
        streak = 1
        streak_type = 'up' if vol_changes[-1] > 0 else 'down'

        for i in range(len(vol_changes) - 2, -1, -1):
            change_type = 'up' if vol_changes[i] > 0 else 'down'
            if change_type == streak_type:
                streak += 1
            else:
                break

        if streak >= 3:
            if streak_type == 'up':
                # Rising vol - reduce exposure
                direction = 0
                confidence = 0.0
            else:
                # Falling vol - good for trend
                direction = 1 if returns[-1] > 0 else -1
                confidence = min(1.0, streak / 5.0)
        else:
            direction = 0
            confidence = 0.0

        return StreakSignal(
            direction=direction,
            confidence=confidence,
            current_streak=streak,
            streak_type=f'vol_{streak_type}',
            pattern_stats=None,
        )


class StreakEnsembleSignal:
    """
    Formula 72060: Streak Ensemble Signal

    Combines all streak-based signals.
    """

    FORMULA_ID = 72060

    def __init__(self):
        self.signals = [
            Streak2DownSignal(),
            Streak3DownSignal(),
            Streak2UpSignal(),
            Streak3UpSignal(),
            StreakBreakSignal(),
            StreakContinueSignal(),
        ]

    def fit(self, returns: np.ndarray):
        for s in self.signals:
            if hasattr(s, 'fit'):
                s.fit(returns)

    def generate_signal(self, returns: np.ndarray) -> StreakSignal:
        results = [s.generate_signal(returns) for s in self.signals]

        # Filter active signals
        active = [r for r in results if r.direction != 0]

        if not active:
            streak_len, streak_type = StreakAnalyzer().get_current_streak(returns)
            return StreakSignal(0, 0.0, streak_len, streak_type, None)

        # Weighted vote
        total_dir = sum(r.direction * r.confidence for r in active)
        total_conf = sum(r.confidence for r in active)

        if total_conf > 0:
            avg_dir = total_dir / total_conf
            direction = 1 if avg_dir > 0.3 else (-1 if avg_dir < -0.3 else 0)
            confidence = total_conf / len(self.signals)
        else:
            direction = 0
            confidence = 0.0

        streak_len, streak_type = StreakAnalyzer().get_current_streak(returns)

        return StreakSignal(
            direction=direction,
            confidence=confidence,
            current_streak=streak_len,
            streak_type=streak_type,
            pattern_stats=None,
        )
