"""
FORMULA INTERFACE - Base class for all trading formulas
=======================================================
All formulas must implement this interface for registry auto-discovery.

FORMULA ID CONVENTION:
- 100-199: Entry Signals (Z-Score=141)
- 200-299: Filters (CUSUM=218)
- 300-399: Confluence & Regime (Confluence=333, Regime=335)
- 600-699: Volume Capture
- 700-799: Order Flow (OFI=701, Kyle=702, Momentum=706)
- 800-899: Renaissance Compounding
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class IFormula(ABC):
    """
    Base interface for all trading formulas.

    Each formula must define:
    - FORMULA_ID: Unique identifier (int)
    - FORMULA_NAME: Human readable name (str)
    - EDGE_CONTRIBUTION: Expected edge contribution (str)

    And implement:
    - compute(): Core computation returning (signal, confidence)
    """

    # Required class attributes (override in subclass)
    FORMULA_ID: int = 0
    FORMULA_NAME: str = "BaseFormula"
    EDGE_CONTRIBUTION: str = "Unknown"
    CATEGORY: str = "Uncategorized"
    CITATION: str = ""

    @abstractmethod
    def compute(self, prices: np.ndarray, tick: int, **kwargs) -> Tuple[float, float]:
        """
        Core computation method.

        Args:
            prices: Price history array
            tick: Current tick index
            **kwargs: Additional formula-specific parameters

        Returns:
            Tuple of (signal, confidence):
            - signal: -1.0 to 1.0 (negative=SELL, positive=BUY)
            - confidence: 0.0 to 1.0 (signal strength)
        """
        pass

    @classmethod
    def get_info(cls) -> dict:
        """Return formula metadata."""
        return {
            'id': cls.FORMULA_ID,
            'name': cls.FORMULA_NAME,
            'edge': cls.EDGE_CONTRIBUTION,
            'category': cls.CATEGORY,
            'citation': cls.CITATION,
        }

    @staticmethod
    def requires_warmup() -> int:
        """
        Number of ticks required before formula produces valid signals.
        Override in subclass if formula needs historical data.
        """
        return 0

    def __repr__(self) -> str:
        return f"<Formula ID={self.FORMULA_ID} Name={self.FORMULA_NAME}>"
