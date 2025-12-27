"""
Dynamic Time Warping Pattern Matching
=====================================

Formula IDs: 72011-72015

DTW finds similar patterns regardless of speed - a key technique from
speech recognition applied to market patterns.

RenTech Insight: A pattern that took 10 days last time might take 15 days
this time. DTW matches them anyway.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import deque


@dataclass
class SimilarityScore:
    """DTW similarity result."""
    pattern_id: str
    distance: float
    similarity: float  # 0-1, higher = more similar
    time_warp_factor: float  # How much time stretching occurred
    matched_length: int


@dataclass
class PatternMatch:
    """A matched historical pattern."""
    pattern_id: str
    start_idx: int
    end_idx: int
    similarity: float
    subsequent_return: float  # What happened after this pattern
    subsequent_volatility: float


@dataclass
class DTWSignal:
    """Signal from DTW pattern matching."""
    direction: int
    confidence: float
    best_match: Optional[PatternMatch]
    top_matches: List[PatternMatch]
    avg_subsequent_return: float


class DTWMatcher:
    """
    Dynamic Time Warping for pattern matching.

    Matches current price action to historical patterns,
    allowing for time stretching/compression.
    """

    def __init__(self, window_size: int = 20, n_patterns: int = 100):
        self.window_size = window_size
        self.n_patterns = n_patterns
        self.pattern_library: List[Tuple[np.ndarray, float, float]] = []  # (pattern, return, vol)

    def dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray,
                     window: int = None) -> Tuple[float, np.ndarray]:
        """
        Compute DTW distance between two sequences.

        Uses Sakoe-Chiba band for efficiency.

        Args:
            seq1, seq2: Input sequences
            window: Warping window constraint

        Returns:
            (distance, path)
        """
        n, m = len(seq1), len(seq2)

        if window is None:
            window = max(n, m)

        # Initialize cost matrix
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0

        # Fill cost matrix
        for i in range(1, n + 1):
            for j in range(max(1, i - window), min(m + 1, i + window + 1)):
                cost = abs(seq1[i - 1] - seq2[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j],      # insertion
                                       dtw[i, j - 1],      # deletion
                                       dtw[i - 1, j - 1])  # match

        distance = dtw[n, m]

        # Backtrack to find path
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            candidates = [
                (dtw[i - 1, j], i - 1, j),
                (dtw[i, j - 1], i, j - 1),
                (dtw[i - 1, j - 1], i - 1, j - 1),
            ]
            _, i, j = min(candidates)

        return distance, np.array(path[::-1])

    def normalize_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Normalize sequence for comparison."""
        seq = np.array(seq)
        if len(seq) == 0:
            return seq

        # Z-score normalization
        mean = np.mean(seq)
        std = np.std(seq)
        if std > 0:
            return (seq - mean) / std
        return seq - mean

    def build_library(self, prices: np.ndarray, returns: np.ndarray = None):
        """
        Build library of historical patterns.

        Args:
            prices: Historical price series
            returns: Forward returns (optional, computed if not provided)
        """
        if returns is None:
            returns = np.diff(prices) / prices[:-1]
            returns = np.append(returns, 0)

        self.pattern_library = []

        # Extract patterns with sliding window
        for i in range(len(prices) - self.window_size - 5):
            pattern = self.normalize_sequence(prices[i:i + self.window_size])

            # Subsequent return (5-day forward)
            future_return = (prices[i + self.window_size + 5] -
                           prices[i + self.window_size]) / prices[i + self.window_size]
            future_vol = np.std(returns[i + self.window_size:i + self.window_size + 5])

            self.pattern_library.append((pattern, future_return, future_vol))

        # Keep most diverse patterns
        if len(self.pattern_library) > self.n_patterns:
            # Sample uniformly across time
            indices = np.linspace(0, len(self.pattern_library) - 1,
                                 self.n_patterns, dtype=int)
            self.pattern_library = [self.pattern_library[i] for i in indices]

    def find_matches(self, current_pattern: np.ndarray,
                    top_k: int = 5) -> List[PatternMatch]:
        """
        Find most similar patterns in library.

        Args:
            current_pattern: Current price pattern
            top_k: Number of matches to return

        Returns:
            List of PatternMatch sorted by similarity
        """
        if len(self.pattern_library) == 0:
            return []

        current_norm = self.normalize_sequence(current_pattern)
        matches = []

        for i, (pattern, ret, vol) in enumerate(self.pattern_library):
            distance, path = self.dtw_distance(current_norm, pattern)

            # Convert distance to similarity (0-1)
            similarity = 1.0 / (1.0 + distance / self.window_size)

            matches.append(PatternMatch(
                pattern_id=f"pattern_{i}",
                start_idx=i,
                end_idx=i + self.window_size,
                similarity=similarity,
                subsequent_return=ret,
                subsequent_volatility=vol,
            ))

        # Sort by similarity
        matches.sort(key=lambda x: x.similarity, reverse=True)
        return matches[:top_k]

    def generate_signal(self, current_pattern: np.ndarray,
                       min_similarity: float = 0.5) -> DTWSignal:
        """
        Generate trading signal from pattern matching.

        Args:
            current_pattern: Current price pattern
            min_similarity: Minimum similarity threshold

        Returns:
            DTWSignal with direction and confidence
        """
        matches = self.find_matches(current_pattern, top_k=10)

        if not matches:
            return DTWSignal(
                direction=0,
                confidence=0.0,
                best_match=None,
                top_matches=[],
                avg_subsequent_return=0.0,
            )

        # Filter by similarity threshold
        good_matches = [m for m in matches if m.similarity >= min_similarity]

        if not good_matches:
            return DTWSignal(
                direction=0,
                confidence=0.0,
                best_match=matches[0],
                top_matches=matches[:5],
                avg_subsequent_return=0.0,
            )

        # Weighted average of subsequent returns
        total_weight = sum(m.similarity for m in good_matches)
        avg_return = sum(m.subsequent_return * m.similarity
                        for m in good_matches) / total_weight

        # Direction from average return
        if avg_return > 0.01:
            direction = 1
        elif avg_return < -0.01:
            direction = -1
        else:
            direction = 0

        # Confidence from similarity and consistency
        avg_similarity = np.mean([m.similarity for m in good_matches])
        return_consistency = 1.0 - np.std([m.subsequent_return for m in good_matches])
        confidence = avg_similarity * max(0, return_consistency)

        return DTWSignal(
            direction=direction,
            confidence=min(1.0, confidence),
            best_match=good_matches[0],
            top_matches=good_matches[:5],
            avg_subsequent_return=avg_return,
        )


class PatternLibrary:
    """
    Library of canonical patterns (bull flag, head & shoulders, etc.)

    Instead of mining historical patterns, matches against known formations.
    """

    def __init__(self):
        self.patterns: Dict[str, np.ndarray] = {}
        self.pattern_directions: Dict[str, int] = {}
        self._init_canonical_patterns()

    def _init_canonical_patterns(self):
        """Initialize canonical chart patterns."""
        n = 20  # Pattern length

        # Bull flag: up, consolidate, up
        self.patterns['bull_flag'] = np.concatenate([
            np.linspace(0, 2, n // 3),
            np.linspace(2, 1.8, n // 3) + 0.1 * np.sin(np.linspace(0, 4 * np.pi, n // 3)),
            np.linspace(1.8, 3, n - 2 * (n // 3)),
        ])
        self.pattern_directions['bull_flag'] = 1

        # Bear flag: down, consolidate, down
        self.patterns['bear_flag'] = np.concatenate([
            np.linspace(0, -2, n // 3),
            np.linspace(-2, -1.8, n // 3) + 0.1 * np.sin(np.linspace(0, 4 * np.pi, n // 3)),
            np.linspace(-1.8, -3, n - 2 * (n // 3)),
        ])
        self.pattern_directions['bear_flag'] = -1

        # V-bottom reversal
        self.patterns['v_bottom'] = np.concatenate([
            np.linspace(0, -2, n // 2),
            np.linspace(-2, 0.5, n - n // 2),
        ])
        self.pattern_directions['v_bottom'] = 1

        # V-top reversal
        self.patterns['v_top'] = np.concatenate([
            np.linspace(0, 2, n // 2),
            np.linspace(2, -0.5, n - n // 2),
        ])
        self.pattern_directions['v_top'] = -1

        # Ascending triangle (bullish)
        x = np.linspace(0, 4 * np.pi, n)
        self.patterns['ascending_triangle'] = 2 * np.abs(np.sin(x)) * np.linspace(0.5, 1, n)
        self.pattern_directions['ascending_triangle'] = 1

        # Descending triangle (bearish)
        self.patterns['descending_triangle'] = -2 * np.abs(np.sin(x)) * np.linspace(0.5, 1, n)
        self.pattern_directions['descending_triangle'] = -1

        # Double bottom
        self.patterns['double_bottom'] = np.concatenate([
            np.linspace(0, -2, n // 4),
            np.linspace(-2, -1, n // 4),
            np.linspace(-1, -2, n // 4),
            np.linspace(-2, 0.5, n - 3 * (n // 4)),
        ])
        self.pattern_directions['double_bottom'] = 1

        # Double top
        self.patterns['double_top'] = np.concatenate([
            np.linspace(0, 2, n // 4),
            np.linspace(2, 1, n // 4),
            np.linspace(1, 2, n // 4),
            np.linspace(2, -0.5, n - 3 * (n // 4)),
        ])
        self.pattern_directions['double_top'] = -1


# =============================================================================
# FORMULA IMPLEMENTATIONS (72011-72015)
# =============================================================================

class DTWPatternSignal:
    """
    Formula 72011: DTW Pattern Matching Signal

    Basic DTW pattern matching against historical patterns.
    Trades in direction of most similar historical outcomes.
    """

    FORMULA_ID = 72011

    def __init__(self, window_size: int = 20, n_patterns: int = 200):
        self.matcher = DTWMatcher(window_size=window_size, n_patterns=n_patterns)
        self.is_trained = False

    def train(self, prices: np.ndarray):
        """Build pattern library from historical prices."""
        self.matcher.build_library(prices)
        self.is_trained = True

    def generate_signal(self, prices: np.ndarray) -> DTWSignal:
        """Generate signal from current price pattern."""
        if not self.is_trained:
            raise RuntimeError("Must train first")

        current_pattern = prices[-self.matcher.window_size:]
        return self.matcher.generate_signal(current_pattern)


class DTWBreakoutSignal:
    """
    Formula 72012: DTW Breakout Signal

    Identifies breakout patterns and trades in breakout direction.
    Uses canonical breakout patterns (bull flag, ascending triangle, etc.)
    """

    FORMULA_ID = 72012

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.library = PatternLibrary()
        self.matcher = DTWMatcher(window_size=window_size)

    def generate_signal(self, prices: np.ndarray) -> DTWSignal:
        """Match current pattern against breakout patterns."""
        current = self.matcher.normalize_sequence(prices[-self.window_size:])

        best_match = None
        best_similarity = 0.0
        best_direction = 0

        for name, pattern in self.library.patterns.items():
            pattern_norm = self.matcher.normalize_sequence(pattern)
            distance, _ = self.matcher.dtw_distance(current, pattern_norm)
            similarity = 1.0 / (1.0 + distance / self.window_size)

            if similarity > best_similarity:
                best_similarity = similarity
                best_direction = self.library.pattern_directions[name]
                best_match = PatternMatch(
                    pattern_id=name,
                    start_idx=0,
                    end_idx=self.window_size,
                    similarity=similarity,
                    subsequent_return=0.0,
                    subsequent_volatility=0.0,
                )

        # Only signal if match is strong enough
        if best_similarity < 0.6:
            direction = 0
            confidence = 0.0
        else:
            direction = best_direction
            confidence = (best_similarity - 0.6) * 2.5  # Scale to 0-1

        return DTWSignal(
            direction=direction,
            confidence=min(1.0, confidence),
            best_match=best_match,
            top_matches=[best_match] if best_match else [],
            avg_subsequent_return=0.0,
        )


class DTWReversalSignal:
    """
    Formula 72013: DTW Reversal Signal

    Identifies reversal patterns (V-bottom, double bottom, etc.)
    and signals accordingly.
    """

    FORMULA_ID = 72013

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.library = PatternLibrary()
        self.matcher = DTWMatcher(window_size=window_size)
        self.reversal_patterns = ['v_bottom', 'v_top', 'double_bottom', 'double_top']

    def generate_signal(self, prices: np.ndarray) -> DTWSignal:
        """Match against reversal patterns only."""
        current = self.matcher.normalize_sequence(prices[-self.window_size:])

        best_match = None
        best_similarity = 0.0
        best_direction = 0

        for name in self.reversal_patterns:
            pattern = self.library.patterns[name]
            pattern_norm = self.matcher.normalize_sequence(pattern)
            distance, _ = self.matcher.dtw_distance(current, pattern_norm)
            similarity = 1.0 / (1.0 + distance / self.window_size)

            if similarity > best_similarity:
                best_similarity = similarity
                best_direction = self.library.pattern_directions[name]
                best_match = PatternMatch(
                    pattern_id=name,
                    start_idx=0,
                    end_idx=self.window_size,
                    similarity=similarity,
                    subsequent_return=0.0,
                    subsequent_volatility=0.0,
                )

        if best_similarity < 0.65:
            direction = 0
            confidence = 0.0
        else:
            direction = best_direction
            confidence = (best_similarity - 0.65) * 3.0

        return DTWSignal(
            direction=direction,
            confidence=min(1.0, confidence),
            best_match=best_match,
            top_matches=[best_match] if best_match else [],
            avg_subsequent_return=0.0,
        )


class DTWMomentumSignal:
    """
    Formula 72014: DTW Momentum Signal

    Uses DTW to find historical momentum patterns and
    trades in direction of similar historical outcomes.
    """

    FORMULA_ID = 72014

    def __init__(self, window_size: int = 20, lookback: int = 5):
        self.window_size = window_size
        self.lookback = lookback
        self.matcher = DTWMatcher(window_size=window_size, n_patterns=300)

    def train(self, prices: np.ndarray):
        """Build library with momentum context."""
        returns = np.diff(prices) / prices[:-1]
        self.matcher.build_library(prices, returns)

    def generate_signal(self, prices: np.ndarray) -> DTWSignal:
        """Signal based on momentum characteristics of matches."""
        current_pattern = prices[-self.window_size:]
        signal = self.matcher.generate_signal(current_pattern, min_similarity=0.5)

        # Amplify direction based on recent momentum
        recent_return = (prices[-1] - prices[-self.lookback]) / prices[-self.lookback]

        if signal.direction == 1 and recent_return > 0:
            # Bullish signal + bullish momentum = stronger
            confidence = min(1.0, signal.confidence * 1.3)
        elif signal.direction == -1 and recent_return < 0:
            # Bearish signal + bearish momentum = stronger
            confidence = min(1.0, signal.confidence * 1.3)
        else:
            # Conflicting signals = weaker
            confidence = signal.confidence * 0.7

        return DTWSignal(
            direction=signal.direction,
            confidence=confidence,
            best_match=signal.best_match,
            top_matches=signal.top_matches,
            avg_subsequent_return=signal.avg_subsequent_return,
        )


class DTWEnsembleSignal:
    """
    Formula 72015: DTW Ensemble Signal

    Combines multiple DTW configurations:
    - Different window sizes (10, 20, 40 days)
    - Historical patterns + canonical patterns
    - Weighted voting for final signal
    """

    FORMULA_ID = 72015

    def __init__(self):
        self.matchers = [
            DTWMatcher(window_size=10, n_patterns=150),
            DTWMatcher(window_size=20, n_patterns=150),
            DTWMatcher(window_size=40, n_patterns=100),
        ]
        self.library = PatternLibrary()
        self.weights = [0.25, 0.4, 0.35]  # Medium window gets highest weight

    def train(self, prices: np.ndarray):
        """Train all matchers."""
        for matcher in self.matchers:
            matcher.build_library(prices)

    def generate_signal(self, prices: np.ndarray) -> DTWSignal:
        """Ensemble signal from multiple configurations."""
        signals = []

        # Historical pattern matching
        for matcher, weight in zip(self.matchers, self.weights):
            if len(prices) >= matcher.window_size:
                pattern = prices[-matcher.window_size:]
                sig = matcher.generate_signal(pattern)
                signals.append((sig.direction, sig.confidence * weight))

        # Canonical pattern matching (use 20-day window)
        canonical_matcher = self.matchers[1]
        pattern = prices[-20:] if len(prices) >= 20 else prices
        pattern_norm = canonical_matcher.normalize_sequence(pattern)

        for name, canonical in self.library.patterns.items():
            canonical_norm = canonical_matcher.normalize_sequence(canonical)
            distance, _ = canonical_matcher.dtw_distance(pattern_norm, canonical_norm)
            similarity = 1.0 / (1.0 + distance / 20)

            if similarity > 0.6:
                direction = self.library.pattern_directions[name]
                signals.append((direction, similarity * 0.2))  # Lower weight for canonical

        if not signals:
            return DTWSignal(
                direction=0,
                confidence=0.0,
                best_match=None,
                top_matches=[],
                avg_subsequent_return=0.0,
            )

        # Weighted vote
        total_weight = sum(s[1] for s in signals)
        weighted_direction = sum(s[0] * s[1] for s in signals) / total_weight

        if weighted_direction > 0.3:
            direction = 1
        elif weighted_direction < -0.3:
            direction = -1
        else:
            direction = 0

        confidence = abs(weighted_direction)

        return DTWSignal(
            direction=direction,
            confidence=min(1.0, confidence),
            best_match=None,
            top_matches=[],
            avg_subsequent_return=0.0,
        )
