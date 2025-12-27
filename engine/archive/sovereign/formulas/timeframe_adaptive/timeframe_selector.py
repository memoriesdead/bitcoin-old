"""
Timeframe Selector - Timeframe-Adaptive Mathematical Engine
============================================================

Selects optimal timeframe τ* based on:
1. Entropy measurement (signal quality)
2. Mutual information (predictive power)
3. Regime-aware selection (HMM state)

This module answers: "Which timeframe should we trade right now?"
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

from .math_primitives import (
    tae_001_timeframe_validity,
    tae_001_batch_validity,
    tae_002_mutual_information,
    tae_002_shannon_entropy,
    get_optimal_timeframe_for_regime,
    get_decay_rate_for_regime,
    compute_timeframe_score,
)


@dataclass
class TimeframeCandidate:
    """A candidate timeframe with its scores."""
    tau: float                    # Timeframe in seconds
    validity_score: float = 0.0  # TAE-001 score
    mutual_info: float = 0.0     # TAE-002 score
    entropy: float = 0.0         # Signal entropy
    combined_score: float = 0.0  # Final score


@dataclass
class TimeframeSelection:
    """Result of timeframe selection."""
    optimal_tau: float           # Selected timeframe
    confidence: float            # Confidence in selection
    regime: str                  # Current regime
    all_scores: Dict[float, float] = field(default_factory=dict)
    selection_time: float = 0.0


class EntropyMeasurer:
    """
    Measures signal quality using Shannon entropy.

    Low entropy = predictable signal = good
    High entropy = random signal = bad
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.cache: Dict[float, float] = {}

    def measure(self, signal_values: np.ndarray) -> float:
        """
        Calculate Shannon entropy of signal.

        Returns normalized entropy (0 = perfectly predictable, 1 = random).
        """
        if len(signal_values) < 10:
            return 1.0  # Assume maximum entropy with insufficient data

        entropy = tae_002_shannon_entropy(signal_values, self.n_bins)

        # Normalize by maximum possible entropy (log(n_bins))
        max_entropy = np.log(self.n_bins)
        normalized = entropy / max_entropy if max_entropy > 0 else 1.0

        return min(1.0, max(0.0, normalized))

    def quality_score(self, signal_values: np.ndarray) -> float:
        """
        Convert entropy to quality score (1 - normalized_entropy).

        High quality = low entropy = predictable.
        """
        return 1.0 - self.measure(signal_values)


class MutualInfoCalculator:
    """
    Calculates mutual information between signal and returns.

    High MI = signal predicts returns well = optimal timeframe.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.history: Dict[float, List[float]] = {}  # tau -> MI history

    def calculate(
        self,
        signal_values: np.ndarray,
        returns: np.ndarray
    ) -> float:
        """
        Calculate mutual information I(Signal; Return).

        Returns MI in nats. Higher = more predictive power.
        """
        return tae_002_mutual_information(signal_values, returns, self.n_bins)

    def track_mi(self, tau: float, mi: float) -> None:
        """Track MI history for a timeframe."""
        if tau not in self.history:
            self.history[tau] = []
        self.history[tau].append(mi)

        # Keep last 100 values
        if len(self.history[tau]) > 100:
            self.history[tau] = self.history[tau][-100:]

    def get_rolling_mi(self, tau: float, window: int = 20) -> float:
        """Get rolling average MI for a timeframe."""
        if tau not in self.history or len(self.history[tau]) < 5:
            return 0.0
        return np.mean(self.history[tau][-window:])


class RegimeAwareSelector:
    """
    Selects optimal timeframe based on HMM regime.

    Each regime has a natural optimal timeframe range.
    """

    def __init__(self):
        # Candidate timeframes to evaluate
        self.candidate_taus = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0]

        # Regime-specific timeframe ranges [min, max]
        self.regime_ranges = {
            'accumulation': (30.0, 60.0),
            'distribution': (30.0, 60.0),
            'trending_up': (10.0, 20.0),
            'trending_down': (10.0, 20.0),
            'consolidation': (5.0, 15.0),
            'capitulation': (1.0, 5.0),
            'euphoria': (1.0, 5.0),
            'unknown': (5.0, 30.0),
        }

    def get_regime_range(self, regime: str) -> Tuple[float, float]:
        """Get valid timeframe range for regime."""
        return self.regime_ranges.get(regime.lower(), (5.0, 30.0))

    def filter_candidates_by_regime(
        self,
        regime: str
    ) -> List[float]:
        """Filter candidates to regime-appropriate timeframes."""
        min_tau, max_tau = self.get_regime_range(regime)
        return [tau for tau in self.candidate_taus if min_tau <= tau <= max_tau]

    def get_optimal_for_regime(self, regime: str) -> float:
        """Get optimal timeframe τ* for regime."""
        return get_optimal_timeframe_for_regime(regime)


class TimeframeSelector:
    """
    Main timeframe selection engine.

    Combines entropy, mutual information, and regime awareness
    to select the optimal timeframe for current market conditions.
    """

    def __init__(self, candidate_taus: Optional[List[float]] = None):
        """
        Initialize timeframe selector.

        Args:
            candidate_taus: List of candidate timeframes to evaluate
        """
        self.entropy_measurer = EntropyMeasurer()
        self.mi_calculator = MutualInfoCalculator()
        self.regime_selector = RegimeAwareSelector()

        self.candidate_taus = candidate_taus or [
            1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0
        ]

        # History of selections
        self.selection_history: List[TimeframeSelection] = []
        self.current_selection: Optional[TimeframeSelection] = None

        # Smoothing: don't switch too rapidly
        self.min_switch_interval = 30.0  # seconds
        self.last_switch_time = 0.0
        self.switch_threshold = 0.2  # Need 20% better score to switch

    def select(
        self,
        signals_by_tau: Dict[float, np.ndarray],
        returns_by_tau: Dict[float, np.ndarray],
        regime: str = 'unknown',
        force_regime: bool = False
    ) -> TimeframeSelection:
        """
        Select optimal timeframe.

        Args:
            signals_by_tau: Dict of timeframe -> signal values
            returns_by_tau: Dict of timeframe -> corresponding returns
            regime: Current market regime from HMM
            force_regime: If True, only consider regime-appropriate timeframes

        Returns:
            TimeframeSelection with optimal τ and confidence
        """
        candidates = []
        scores: Dict[float, float] = {}

        # Get regime-specific optimal and decay
        optimal_tau = self.regime_selector.get_optimal_for_regime(regime)
        decay_lambda = get_decay_rate_for_regime(regime)

        # Filter by regime if forced
        if force_regime:
            valid_taus = self.regime_selector.filter_candidates_by_regime(regime)
        else:
            valid_taus = self.candidate_taus

        # Score each timeframe
        for tau in valid_taus:
            signals = signals_by_tau.get(tau, np.array([]))
            returns = returns_by_tau.get(tau, np.array([]))

            if len(signals) < 10 or len(returns) < 10:
                continue

            # TAE-001: Timeframe validity (proximity to regime optimal)
            validity = tae_001_timeframe_validity(tau, optimal_tau, decay_lambda)

            # TAE-002: Mutual information (predictive power)
            mi = self.mi_calculator.calculate(signals, returns)
            self.mi_calculator.track_mi(tau, mi)

            # Entropy quality (1 - normalized entropy)
            quality = self.entropy_measurer.quality_score(signals)

            # Combined score
            # Weight: validity (regime fit) * MI (predictive) * quality (low noise)
            combined = validity * (mi + 0.01) * quality

            candidate = TimeframeCandidate(
                tau=tau,
                validity_score=validity,
                mutual_info=mi,
                entropy=1.0 - quality,
                combined_score=combined
            )
            candidates.append(candidate)
            scores[tau] = combined

        if not candidates:
            # Default to regime optimal if no data
            return TimeframeSelection(
                optimal_tau=optimal_tau,
                confidence=0.0,
                regime=regime,
                all_scores={},
                selection_time=time.time()
            )

        # Select best
        best = max(candidates, key=lambda c: c.combined_score)

        # Calculate confidence based on margin over second-best
        sorted_candidates = sorted(candidates, key=lambda c: c.combined_score, reverse=True)
        if len(sorted_candidates) > 1:
            best_score = sorted_candidates[0].combined_score
            second_score = sorted_candidates[1].combined_score
            margin = (best_score - second_score) / (best_score + 1e-10)
            confidence = min(1.0, margin * 2)  # Scale margin to confidence
        else:
            confidence = 0.5

        # Apply switching hysteresis
        selected_tau = self._apply_hysteresis(best.tau, best.combined_score)

        selection = TimeframeSelection(
            optimal_tau=selected_tau,
            confidence=confidence,
            regime=regime,
            all_scores=scores,
            selection_time=time.time()
        )

        self.current_selection = selection
        self.selection_history.append(selection)

        # Trim history
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-500:]

        return selection

    def _apply_hysteresis(self, new_tau: float, new_score: float) -> float:
        """
        Apply hysteresis to prevent rapid switching.

        Only switch if:
        1. Enough time has passed since last switch
        2. New timeframe is significantly better
        """
        now = time.time()

        if self.current_selection is None:
            self.last_switch_time = now
            return new_tau

        # Check time since last switch
        if now - self.last_switch_time < self.min_switch_interval:
            return self.current_selection.optimal_tau

        # Check if improvement is significant
        current_score = self.current_selection.all_scores.get(
            self.current_selection.optimal_tau, 0.0
        )

        if new_score > current_score * (1 + self.switch_threshold):
            self.last_switch_time = now
            return new_tau

        return self.current_selection.optimal_tau

    def get_consistency_score(self, window: int = 50) -> float:
        """
        Measure how consistently we're selecting the same timeframe.

        Returns value between 0 and 1:
        - 1.0 = always selecting same timeframe
        - 0.0 = completely random selections
        """
        if len(self.selection_history) < 10:
            return 0.5

        recent = self.selection_history[-window:]
        taus = [s.optimal_tau for s in recent]

        # Mode frequency
        from collections import Counter
        counts = Counter(taus)
        mode_count = counts.most_common(1)[0][1]

        return mode_count / len(taus)

    def quick_select(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        regime: str = 'unknown'
    ) -> float:
        """
        Quick selection using only current data and regime.

        For cases where we don't have signals at multiple timeframes.
        Returns optimal τ based primarily on regime.
        """
        optimal = get_optimal_timeframe_for_regime(regime)

        # Adjust based on signal quality
        if len(signals) > 10:
            quality = self.entropy_measurer.quality_score(signals)
            mi = self.mi_calculator.calculate(signals, returns)

            # If high MI, prefer current regime's optimal
            # If low MI, be more conservative (longer timeframes)
            if mi < 0.1:
                # Low predictive power - use longer timeframe
                optimal = min(60.0, optimal * 1.5)

        return optimal
