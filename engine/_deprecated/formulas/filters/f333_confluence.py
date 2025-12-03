"""
FORMULA ID 333: SIGNAL CONFLUENCE
=================================
Condorcet Jury Theorem for Signal Combination

THEOREM:
    If independent signals each have >50% accuracy,
    the majority vote has HIGHER accuracy.

    With 3 independent 55% signals: P(majority correct) ≈ 59%

FORMULA:
    Combined_Signal = sign(sum(individual_signals))
    Combined_Prob = Condorcet_Formula(individual_probs)

PURPOSE:
    Combines multiple signals into a single, more reliable signal.

EDGE CONTRIBUTION: Signal quality improvement
"""
from typing import Tuple, List
import numpy as np

from engine.core.interfaces import IFormula
from engine.core.constants.trading import MIN_AGREEING_SIGNALS, MIN_CONFLUENCE_PROB
from engine.formulas.registry import register_formula


@register_formula
class ConfluenceFormula(IFormula):
    """
    Signal Confluence using Condorcet Voting.

    Combines multiple signals for higher accuracy.
    """
    FORMULA_ID = 333
    FORMULA_NAME = "Signal Confluence"
    EDGE_CONTRIBUTION = "Signal quality improvement"
    CATEGORY = "filters"
    CITATION = "Condorcet Jury Theorem (1785)"

    def __init__(self, min_agreeing: int = MIN_AGREEING_SIGNALS,
                 min_prob: float = MIN_CONFLUENCE_PROB):
        self.min_agreeing = min_agreeing
        self.min_prob = min_prob
        self._last_agreeing = 0

    def compute(self, prices: np.ndarray, tick: int, **kwargs) -> Tuple[float, float]:
        """
        Compute confluence signal from multiple inputs.

        Expects kwargs:
            signals: List of (signal, confidence) tuples

        Returns:
            Tuple of (combined_signal, combined_confidence)
        """
        signals = kwargs.get('signals', [])
        if not signals:
            return 0.0, 0.0

        # Count agreeing signals
        buy_votes = 0
        sell_votes = 0
        total_confidence = 0.0

        for signal, confidence in signals:
            if signal > 0:
                buy_votes += 1
                total_confidence += confidence
            elif signal < 0:
                sell_votes += 1
                total_confidence += confidence

        # Determine direction
        agreeing = max(buy_votes, sell_votes)
        self._last_agreeing = agreeing

        if agreeing < self.min_agreeing:
            return 0.0, 0.0

        if buy_votes > sell_votes:
            direction = 1.0
        elif sell_votes > buy_votes:
            direction = -1.0
        else:
            return 0.0, 0.0

        # Combine confidences (Condorcet formula approximation)
        avg_conf = total_confidence / len(signals)
        combined_conf = self._condorcet_prob(avg_conf, agreeing, len(signals))

        if combined_conf < self.min_prob:
            return 0.0, 0.0

        return direction, combined_conf

    def _condorcet_prob(self, p: float, k: int, n: int) -> float:
        """
        Approximate Condorcet probability.

        Args:
            p: Individual signal accuracy
            k: Number of agreeing signals
            n: Total signals

        Returns:
            Probability that majority is correct
        """
        if p <= 0.5 or k <= n / 2:
            return 0.5

        # Simplified: higher agreement = higher confidence
        agreement_ratio = k / n
        boost = (p - 0.5) * agreement_ratio
        return min(p + boost, 0.95)

    def get_agreeing_count(self) -> int:
        """Get number of agreeing signals from last computation."""
        return self._last_agreeing

    @staticmethod
    def requires_warmup() -> int:
        return 0
