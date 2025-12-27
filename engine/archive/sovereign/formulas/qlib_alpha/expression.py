"""
Alpha Expression Framework
===========================

Ported from Microsoft QLib alpha expression concepts.

QLib uses a declarative expression language for alpha factors:
    $close / Ref($close, 5) - 1  # 5-day return
    Rank(Mean($volume, 10))      # Volume rank

We adapt this for blockchain flow data with these expressions:
    FlowMomentum(10)             # 10-sample flow momentum
    FlowZScore(20)               # Z-score of flow
    FlowAcceleration(5)          # Rate of change of momentum

Formula IDs: 70001-70005
"""

import numpy as np
from typing import Optional, List, Dict, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AlphaResult:
    """Result from an alpha expression."""
    value: float
    confidence: float
    timestamp: float
    expression_id: int
    raw_values: Optional[np.ndarray] = None


class AlphaExpression(ABC):
    """
    Base class for alpha expressions.

    All expressions follow QLib pattern:
    1. Take a window of historical data
    2. Compute a single alpha value
    3. Return value with confidence
    """

    formula_id: int = 70000
    name: str = "BaseAlpha"

    def __init__(self, window: int = 20):
        self.window = window
        self._cache: Dict[str, Any] = {}

    @abstractmethod
    def compute(self, data: np.ndarray) -> float:
        """Compute alpha value from data window."""
        pass

    def __call__(self, data: np.ndarray, timestamp: float = 0.0) -> AlphaResult:
        """
        Apply expression to data.

        Args:
            data: Array of values (most recent last)
            timestamp: Current timestamp

        Returns:
            AlphaResult with value and confidence
        """
        if len(data) < self.window:
            return AlphaResult(
                value=0.0,
                confidence=len(data) / self.window,
                timestamp=timestamp,
                expression_id=self.formula_id,
                raw_values=data
            )

        # Use last window values
        window_data = data[-self.window:]
        value = self.compute(window_data)

        # Confidence based on data quality
        confidence = self._compute_confidence(window_data)

        return AlphaResult(
            value=value,
            confidence=confidence,
            timestamp=timestamp,
            expression_id=self.formula_id,
            raw_values=window_data
        )

    def _compute_confidence(self, data: np.ndarray) -> float:
        """Compute confidence based on data quality."""
        # Reduce confidence if data has NaN or extreme outliers
        if np.any(np.isnan(data)):
            return 0.5

        # Check for stale data (all same value)
        if np.std(data) < 1e-10:
            return 0.3

        return 1.0


class FlowMomentum(AlphaExpression):
    """
    Flow Momentum Alpha (Formula 70001)

    Measures the trend in blockchain flow over a window.
    Positive = increasing outflow (bullish)
    Negative = increasing inflow (bearish)

    Equivalent to QLib: Mean($flow, N) - Mean($flow, 2*N)
    """

    formula_id = 70001
    name = "FlowMomentum"

    def __init__(self, window: int = 10, slow_window: Optional[int] = None):
        super().__init__(window)
        self.slow_window = slow_window or window * 2

    def compute(self, data: np.ndarray) -> float:
        """
        Compute flow momentum.

        Returns difference between fast and slow moving averages.
        """
        if len(data) < self.slow_window:
            fast_ma = np.mean(data[-self.window:])
            slow_ma = np.mean(data)
        else:
            fast_ma = np.mean(data[-self.window:])
            slow_ma = np.mean(data[-self.slow_window:])

        # Normalize by standard deviation to get comparable values
        std = np.std(data)
        if std < 1e-10:
            return 0.0

        return (fast_ma - slow_ma) / std


class FlowAcceleration(AlphaExpression):
    """
    Flow Acceleration Alpha (Formula 70002)

    Second derivative of flow - rate of change of momentum.
    Positive = momentum increasing (trend strengthening)
    Negative = momentum decreasing (trend weakening)

    Equivalent to QLib: Delta(Mean($flow, N), 1)
    """

    formula_id = 70002
    name = "FlowAcceleration"

    def __init__(self, window: int = 5):
        super().__init__(window)

    def compute(self, data: np.ndarray) -> float:
        """
        Compute flow acceleration.

        Returns rate of change of the moving average.
        """
        if len(data) < self.window * 2:
            return 0.0

        # Current momentum
        current_ma = np.mean(data[-self.window:])

        # Previous momentum
        prev_ma = np.mean(data[-self.window*2:-self.window])

        # Acceleration
        std = np.std(data)
        if std < 1e-10:
            return 0.0

        return (current_ma - prev_ma) / std


class FlowZScore(AlphaExpression):
    """
    Flow Z-Score Alpha (Formula 70003)

    How many standard deviations current flow is from mean.
    High positive = unusually high outflow (bullish)
    High negative = unusually high inflow (bearish)

    Equivalent to QLib: ($flow - Mean($flow, N)) / Std($flow, N)
    """

    formula_id = 70003
    name = "FlowZScore"

    def __init__(self, window: int = 20):
        super().__init__(window)

    def compute(self, data: np.ndarray) -> float:
        """
        Compute z-score of latest value.
        """
        mean = np.mean(data)
        std = np.std(data)

        if std < 1e-10:
            return 0.0

        return (data[-1] - mean) / std


class FlowSkew(AlphaExpression):
    """
    Flow Skewness Alpha (Formula 70004)

    Measures asymmetry in flow distribution.
    Positive skew = occasional large outflows (bullish tail risk)
    Negative skew = occasional large inflows (bearish tail risk)

    Equivalent to QLib: Skew($flow, N)
    """

    formula_id = 70004
    name = "FlowSkew"

    def __init__(self, window: int = 30):
        super().__init__(window)

    def compute(self, data: np.ndarray) -> float:
        """
        Compute skewness of flow distribution.
        """
        mean = np.mean(data)
        std = np.std(data)

        if std < 1e-10:
            return 0.0

        # Fisher-Pearson skewness
        n = len(data)
        skew = np.sum(((data - mean) / std) ** 3) / n

        return skew


class FlowAutoCorr(AlphaExpression):
    """
    Flow Autocorrelation Alpha (Formula 70005)

    Measures persistence in flow patterns.
    High positive = flows are trending (momentum regime)
    Near zero = flows are random (no momentum)
    Negative = flows are mean-reverting (reversal regime)

    Equivalent to QLib: Corr($flow, Ref($flow, lag), N)
    """

    formula_id = 70005
    name = "FlowAutoCorr"

    def __init__(self, window: int = 20, lag: int = 1):
        super().__init__(window)
        self.lag = lag

    def compute(self, data: np.ndarray) -> float:
        """
        Compute autocorrelation at specified lag.
        """
        if len(data) <= self.lag:
            return 0.0

        # Current values
        current = data[self.lag:]
        # Lagged values
        lagged = data[:-self.lag]

        # Correlation
        if len(current) < 2:
            return 0.0

        corr = np.corrcoef(current, lagged)[0, 1]

        if np.isnan(corr):
            return 0.0

        return corr


class FlowRegimeDetector(AlphaExpression):
    """
    Combines multiple alpha expressions to detect flow regime.

    Returns regime classification:
    - TRENDING_UP: Positive momentum, positive acceleration
    - TRENDING_DOWN: Negative momentum, negative acceleration
    - MEAN_REVERTING: Low autocorr, high z-score
    - NEUTRAL: No clear pattern
    """

    formula_id = 70009
    name = "FlowRegimeDetector"

    def __init__(self, window: int = 20):
        super().__init__(window)
        self.momentum = FlowMomentum(window // 2)
        self.acceleration = FlowAcceleration(window // 4)
        self.zscore = FlowZScore(window)
        self.autocorr = FlowAutoCorr(window)

    def compute(self, data: np.ndarray) -> float:
        """
        Compute regime score.

        Returns:
            +1.0 = Strong trending up
            -1.0 = Strong trending down
            0.0 = Neutral/mean-reverting
        """
        mom = self.momentum.compute(data)
        acc = self.acceleration.compute(data)
        z = self.zscore.compute(data)
        ac = self.autocorr.compute(data)

        # Trending regime: high autocorr, momentum and acceleration aligned
        if ac > 0.3:  # Momentum regime
            if mom > 0 and acc > 0:
                return min(1.0, (mom + acc) / 2)
            elif mom < 0 and acc < 0:
                return max(-1.0, (mom + acc) / 2)

        # Mean-reverting regime: low autocorr, use z-score for reversal
        if ac < 0.1 and abs(z) > 2.0:
            return -np.sign(z) * min(1.0, abs(z) / 3)  # Fade the extreme

        return 0.0  # Neutral


def create_alpha_features(flow_data: np.ndarray,
                          price_data: Optional[np.ndarray] = None,
                          timestamp: float = 0.0) -> Dict[str, AlphaResult]:
    """
    Create all alpha features from flow data.

    Args:
        flow_data: Array of flow values
        price_data: Optional array of price values
        timestamp: Current timestamp

    Returns:
        Dictionary of alpha name -> AlphaResult
    """
    alphas = {
        'momentum': FlowMomentum(10),
        'acceleration': FlowAcceleration(5),
        'zscore': FlowZScore(20),
        'skew': FlowSkew(30),
        'autocorr': FlowAutoCorr(20),
        'regime': FlowRegimeDetector(20),
    }

    results = {}
    for name, alpha in alphas.items():
        results[name] = alpha(flow_data, timestamp)

    return results


def alpha_to_signal(alpha_results: Dict[str, AlphaResult],
                    threshold: float = 0.5) -> Dict[str, Any]:
    """
    Convert alpha results to trading signal.

    Args:
        alpha_results: Output from create_alpha_features
        threshold: Minimum regime score to generate signal

    Returns:
        Signal dict with direction, confidence, and alpha values
    """
    regime = alpha_results.get('regime')
    if regime is None:
        return {'direction': 0, 'confidence': 0, 'reason': 'no_regime'}

    if abs(regime.value) < threshold:
        return {
            'direction': 0,
            'confidence': regime.confidence,
            'reason': 'below_threshold',
            'regime_score': regime.value,
        }

    # Generate signal based on regime
    direction = 1 if regime.value > 0 else -1

    # Confidence based on regime strength and data quality
    confidence = min(1.0, abs(regime.value)) * regime.confidence

    # Gather supporting alpha values
    alpha_values = {
        name: result.value
        for name, result in alpha_results.items()
    }

    return {
        'direction': direction,
        'confidence': confidence,
        'reason': 'regime_signal',
        'regime_score': regime.value,
        'alphas': alpha_values,
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Generate sample flow data
    np.random.seed(42)

    # Trending data
    trend = np.cumsum(np.random.randn(100) * 0.5 + 0.1)

    print("Alpha Expression Demo")
    print("=" * 50)

    # Test each alpha
    alphas = [
        FlowMomentum(10),
        FlowAcceleration(5),
        FlowZScore(20),
        FlowSkew(30),
        FlowAutoCorr(20),
    ]

    for alpha in alphas:
        result = alpha(trend)
        print(f"{alpha.name} (ID {alpha.formula_id}): "
              f"value={result.value:.3f}, conf={result.confidence:.2f}")

    # Full feature set
    print("\nFull Alpha Features:")
    features = create_alpha_features(trend)
    for name, result in features.items():
        print(f"  {name}: {result.value:.3f}")

    # Convert to signal
    signal = alpha_to_signal(features)
    print(f"\nSignal: direction={signal['direction']}, "
          f"confidence={signal['confidence']:.2f}")
