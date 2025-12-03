"""
FORMULA ID 702: KYLE LAMBDA
===========================
Price Impact Coefficient

Citation: Kyle (1985)
"Continuous Auctions and Insider Trading"
Econometrica, Vol 53, No 6 (10,000+ citations)

FORMULA:
    Lambda = Cov(Delta_P, V) / Var(V)

    Simplified: Lambda ≈ |OFI| (higher OFI = higher price impact)

PURPOSE:
    Measures how much price moves per unit of order flow.
    High lambda = illiquid market, price moves easily
    Low lambda = liquid market, price resistant to moves

EDGE CONTRIBUTION: Position sizing adjustment based on liquidity
"""
from typing import Tuple
import numpy as np
from numba import njit

from engine.core.interfaces import IFormula
from engine.formulas.registry import register_formula


@register_formula
class KyleLambdaFormula(IFormula):
    """
    Kyle Lambda - Price Impact Coefficient.

    Used for position sizing based on market liquidity.
    """
    FORMULA_ID = 702
    FORMULA_NAME = "Kyle Lambda"
    EDGE_CONTRIBUTION = "Position sizing (liquidity)"
    CATEGORY = "flow"
    CITATION = "Kyle (1985) - Econometrica (10,000+ citations)"

    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def compute(self, prices: np.ndarray, tick: int, **kwargs) -> Tuple[float, float]:
        """
        Compute Kyle Lambda.

        Returns:
            Tuple of (lambda_value, confidence)
        """
        ofi_strength = kwargs.get('ofi_strength', 0.0)
        lambda_value = abs(ofi_strength)
        confidence = min(lambda_value, 1.0)
        return lambda_value, confidence

    @staticmethod
    def requires_warmup() -> int:
        return 50
