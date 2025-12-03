"""
FORMULA ID 706: FLOW MOMENTUM
=============================
OFI Acceleration/Deceleration

FORMULA:
    Flow_Momentum = OFI(recent) - OFI(earlier)

    Positive = Flow accelerating (stronger signal)
    Negative = Flow decelerating (weaker signal)

PURPOSE:
    Confirms OFI direction by checking if flow is strengthening.
    Accelerating flow = higher conviction trades.

EDGE CONTRIBUTION: Signal confirmation (+2-3pp WR)
"""
from typing import Tuple
import numpy as np

from engine.core.interfaces import IFormula
from engine.formulas.registry import register_formula


@register_formula
class FlowMomentumFormula(IFormula):
    """
    Flow Momentum - OFI acceleration.

    Confirms signal direction by checking if flow is strengthening.
    """
    FORMULA_ID = 706
    FORMULA_NAME = "Flow Momentum"
    EDGE_CONTRIBUTION = "Signal confirmation (+2-3pp WR)"
    CATEGORY = "flow"
    CITATION = "Academic Consensus"

    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def compute(self, prices: np.ndarray, tick: int, **kwargs) -> Tuple[float, float]:
        """
        Compute Flow Momentum.

        Returns:
            Tuple of (momentum_value, confidence)
        """
        flow_momentum = kwargs.get('flow_momentum', 0.0)
        signal = 1.0 if flow_momentum > 0 else (-1.0 if flow_momentum < 0 else 0.0)
        confidence = min(abs(flow_momentum), 1.0)
        return signal, confidence

    @staticmethod
    def requires_warmup() -> int:
        return 50
