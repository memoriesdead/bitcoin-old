"""
Base Formula Class and Registry
===============================
Foundation for all 217 Renaissance formulas.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
from collections import deque


# Global formula registry
FORMULA_REGISTRY: Dict[int, type] = {}


class FormulaRegistry:
    """Registry for tracking all formulas"""

    @staticmethod
    def register(formula_id: int, name: str = None, category: str = None):
        """Decorator to register a formula class

        Args:
            formula_id: Unique ID for the formula
            name: Optional display name
            category: Optional category (e.g., 'order_flow', 'statistical')
        """
        def decorator(cls):
            FORMULA_REGISTRY[formula_id] = cls
            cls.FORMULA_ID = formula_id
            if name:
                cls.NAME = name
            if category:
                cls.CATEGORY = category
            return cls
        return decorator

    @staticmethod
    def get(formula_id: int):
        """Get formula class by ID"""
        return FORMULA_REGISTRY.get(formula_id)

    @staticmethod
    def list_all() -> Dict[int, str]:
        """List all registered formulas"""
        return {k: v.__name__ for k, v in sorted(FORMULA_REGISTRY.items())}


class BaseFormula(ABC):
    """
    Base class for all Renaissance formulas.

    Each formula must implement:
    - update(price, volume, timestamp): Process new data
    - get_signal(): Return trading signal (-1, 0, +1)
    - get_confidence(): Return signal confidence (0-1)
    """

    FORMULA_ID: int = 0
    CATEGORY: str = "base"
    NAME: str = "BaseFormula"
    DESCRIPTION: str = "Base formula class"

    def __init__(self, lookback: int = 100, **kwargs):
        self.lookback = lookback
        self.prices = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        self.timestamps = deque(maxlen=lookback)
        self.returns = deque(maxlen=lookback)
        self.signal = 0
        self.confidence = 0.0
        self.last_update = 0
        self.is_ready = False
        self.min_samples = kwargs.get('min_samples', 20)

    def update(self, price: float, volume: float = 0.0, timestamp: float = 0.0) -> None:
        """Update formula with new data"""
        if len(self.prices) > 0:
            ret = (price - self.prices[-1]) / self.prices[-1] if self.prices[-1] != 0 else 0
            self.returns.append(ret)

        self.prices.append(price)
        self.volumes.append(volume)
        self.timestamps.append(timestamp)
        self.last_update = timestamp

        if len(self.prices) >= self.min_samples:
            self.is_ready = True
            self._compute()

    @abstractmethod
    def _compute(self) -> None:
        """Compute the formula signal (implemented by subclasses)"""
        pass

    def get_signal(self) -> int:
        """Get trading signal: -1 (sell), 0 (neutral), +1 (buy)"""
        return self.signal

    def get_confidence(self) -> float:
        """Get signal confidence: 0.0 to 1.0"""
        return self.confidence

    def get_state(self) -> Dict[str, Any]:
        """Get current state for debugging"""
        return {
            'formula_id': self.FORMULA_ID,
            'name': self.NAME,
            'signal': self.signal,
            'confidence': self.confidence,
            'is_ready': self.is_ready,
            'samples': len(self.prices),
        }

    def reset(self) -> None:
        """Reset formula state"""
        self.prices.clear()
        self.volumes.clear()
        self.timestamps.clear()
        self.returns.clear()
        self.signal = 0
        self.confidence = 0.0
        self.is_ready = False

    # Utility methods for subclasses
    def _prices_array(self) -> np.ndarray:
        """Get prices as numpy array"""
        return np.array(self.prices)

    def _returns_array(self) -> np.ndarray:
        """Get returns as numpy array"""
        return np.array(self.returns)

    def _volumes_array(self) -> np.ndarray:
        """Get volumes as numpy array"""
        return np.array(self.volumes)

    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean"""
        if len(data) < window:
            return np.array([np.mean(data)])
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation"""
        if len(data) < window:
            return np.array([np.std(data)])
        result = []
        for i in range(len(data) - window + 1):
            result.append(np.std(data[i:i+window]))
        return np.array(result)

    def _ema(self, data: np.ndarray, span: int) -> np.ndarray:
        """Calculate exponential moving average"""
        alpha = 2 / (span + 1)
        result = np.zeros(len(data))
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

    def _zscore(self, value: float, mean: float, std: float) -> float:
        """Calculate z-score"""
        if std == 0:
            return 0.0
        return (value - mean) / std

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for probability scaling"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _clip_signal(self, value: float, threshold: float = 0.5) -> int:
        """Convert continuous value to discrete signal"""
        if value > threshold:
            return 1
        elif value < -threshold:
            return -1
        return 0
