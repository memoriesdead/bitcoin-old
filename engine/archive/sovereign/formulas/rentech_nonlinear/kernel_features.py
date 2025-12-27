"""
Kernel Methods for Non-Linear Feature Extraction
================================================

Formula IDs: 72031-72040

Linear methods miss complex relationships. Kernel methods project
data into higher dimensions where non-linear patterns become linear.

RenTech insight: Markets are highly non-linear. The same pattern
can have opposite meanings in different contexts.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


@dataclass
class KernelSignal:
    """Signal from kernel-based analysis."""
    direction: int
    confidence: float
    kernel_score: float
    feature_importance: Dict[str, float]
    regime_indicator: float


class KernelPCA:
    """
    Kernel Principal Component Analysis.

    Projects data to non-linear feature space for pattern detection.
    """

    def __init__(self, n_components: int = 5, kernel: str = 'rbf', gamma: float = 0.1):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.X_train: Optional[np.ndarray] = None
        self.alphas: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None

    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix."""
        if self.kernel == 'rbf':
            # RBF kernel: exp(-gamma * ||x - y||^2)
            sq_dists = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
                       np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
            return np.exp(-self.gamma * sq_dists)
        elif self.kernel == 'poly':
            # Polynomial kernel: (1 + x.y)^3
            return (1 + X1 @ X2.T) ** 3
        elif self.kernel == 'linear':
            return X1 @ X2.T
        else:
            return X1 @ X2.T

    def fit(self, X: np.ndarray):
        """Fit KPCA on training data."""
        self.X_train = X.copy()
        n = len(X)

        # Compute kernel matrix
        K = self._kernel_function(X, X)

        # Center kernel matrix
        one_n = np.ones((n, n)) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep top n_components
        self.eigenvalues = eigenvalues[:self.n_components]
        self.alphas = eigenvectors[:, :self.n_components]

        # Normalize
        for i in range(self.n_components):
            if self.eigenvalues[i] > 0:
                self.alphas[:, i] /= np.sqrt(self.eigenvalues[i])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data into kernel PCA space."""
        if self.X_train is None:
            raise RuntimeError("Must fit first")

        K = self._kernel_function(X, self.X_train)

        # Center
        n_train = len(self.X_train)
        K_train = self._kernel_function(self.X_train, self.X_train)
        K_centered = K - np.mean(K, axis=1, keepdims=True) - \
                     np.mean(K_train, axis=0) + np.mean(K_train)

        return K_centered @ self.alphas


class KernelFeatureExtractor:
    """
    Extract non-linear features using kernel methods.
    """

    def __init__(self):
        self.kpca = KernelPCA(n_components=5, kernel='rbf')
        self.is_fitted = False

    def fit(self, features: np.ndarray):
        """Fit on historical features."""
        self.kpca.fit(features)
        self.is_fitted = True

    def extract(self, features: np.ndarray) -> np.ndarray:
        """Extract kernel features."""
        if not self.is_fitted:
            raise RuntimeError("Must fit first")
        return self.kpca.transform(features)


class PolynomialInteractions:
    """
    Generate polynomial interaction features.

    Creates products and powers of input features.
    """

    def __init__(self, degree: int = 2, interaction_only: bool = False):
        self.degree = degree
        self.interaction_only = interaction_only

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Generate polynomial features."""
        n_samples, n_features = X.shape
        features = [X]

        if self.degree >= 2:
            # Quadratic terms
            for i in range(n_features):
                for j in range(i, n_features):
                    if i == j and self.interaction_only:
                        continue
                    features.append((X[:, i] * X[:, j]).reshape(-1, 1))

        if self.degree >= 3:
            # Cubic interactions
            for i in range(n_features):
                for j in range(i, n_features):
                    for k in range(j, n_features):
                        if self.interaction_only and (i == j or j == k):
                            continue
                        features.append((X[:, i] * X[:, j] * X[:, k]).reshape(-1, 1))

        return np.hstack(features)


# =============================================================================
# FORMULA IMPLEMENTATIONS (72031-72040)
# =============================================================================

class KernelPCASignal:
    """
    Formula 72031: Kernel PCA Signal

    Uses KPCA to find non-linear structure in features.
    Trades based on position in transformed space.
    """

    FORMULA_ID = 72031

    def __init__(self, n_components: int = 3):
        self.kpca = KernelPCA(n_components=n_components, kernel='rbf')
        self.is_fitted = False
        self.pc_means: Optional[np.ndarray] = None
        self.pc_stds: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, returns: np.ndarray):
        """Fit KPCA and learn relationship with returns."""
        self.kpca.fit(features)
        self.is_fitted = True

        # Transform training data
        transformed = self.kpca.transform(features)

        # Learn mean/std for z-scoring
        self.pc_means = np.mean(transformed, axis=0)
        self.pc_stds = np.std(transformed, axis=0) + 1e-10

    def generate_signal(self, features: np.ndarray) -> KernelSignal:
        if not self.is_fitted:
            return KernelSignal(0, 0.0, 0.0, {}, 0.0)

        # Transform current features
        transformed = self.kpca.transform(features[-1:])
        z_scores = (transformed - self.pc_means) / self.pc_stds

        # First PC often captures trend
        pc1_zscore = z_scores[0, 0]

        if pc1_zscore > 1.5:
            direction = 1
            confidence = min(1.0, pc1_zscore / 3.0)
        elif pc1_zscore < -1.5:
            direction = -1
            confidence = min(1.0, abs(pc1_zscore) / 3.0)
        else:
            direction = 0
            confidence = 0.0

        return KernelSignal(
            direction=direction,
            confidence=confidence,
            kernel_score=pc1_zscore,
            feature_importance={'pc1': abs(pc1_zscore)},
            regime_indicator=z_scores[0, 1] if len(z_scores[0]) > 1 else 0.0,
        )


class RBFKernelSignal:
    """
    Formula 72032: RBF Kernel Signal

    Uses RBF kernel similarity to historical patterns.
    Similar to profitable past = same trade.
    """

    FORMULA_ID = 72032

    def __init__(self, gamma: float = 0.1):
        self.gamma = gamma
        self.history: List[Tuple[np.ndarray, float]] = []  # (features, return)

    def add_sample(self, features: np.ndarray, forward_return: float):
        """Add historical sample."""
        self.history.append((features.flatten(), forward_return))
        if len(self.history) > 500:
            self.history = self.history[-500:]

    def _rbf_similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute RBF similarity."""
        sq_dist = np.sum((x1 - x2) ** 2)
        return np.exp(-self.gamma * sq_dist)

    def generate_signal(self, features: np.ndarray) -> KernelSignal:
        if len(self.history) < 20:
            return KernelSignal(0, 0.0, 0.0, {}, 0.0)

        current = features.flatten()

        # Find similar historical patterns
        similarities = []
        returns = []

        for hist_features, hist_return in self.history:
            sim = self._rbf_similarity(current, hist_features)
            similarities.append(sim)
            returns.append(hist_return)

        similarities = np.array(similarities)
        returns = np.array(returns)

        # Weighted prediction
        weights = similarities / (similarities.sum() + 1e-10)
        predicted_return = np.sum(weights * returns)

        if predicted_return > 0.005:
            direction = 1
            confidence = min(1.0, predicted_return * 20)
        elif predicted_return < -0.005:
            direction = -1
            confidence = min(1.0, abs(predicted_return) * 20)
        else:
            direction = 0
            confidence = 0.0

        return KernelSignal(
            direction=direction,
            confidence=confidence,
            kernel_score=predicted_return,
            feature_importance={},
            regime_indicator=np.max(similarities),
        )


class PolynomialKernelSignal:
    """
    Formula 72033: Polynomial Kernel Signal

    Uses polynomial interactions between features.
    Captures non-linear combinations like vol*momentum.
    """

    FORMULA_ID = 72033

    def __init__(self, degree: int = 2):
        self.poly = PolynomialInteractions(degree=degree)
        self.feature_weights: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, returns: np.ndarray):
        """Learn feature weights via simple regression."""
        poly_features = self.poly.transform(features)

        # Ridge regression
        lambda_reg = 1.0
        XtX = poly_features.T @ poly_features
        XtY = poly_features.T @ returns
        self.feature_weights = np.linalg.solve(
            XtX + lambda_reg * np.eye(XtX.shape[0]), XtY
        )

    def generate_signal(self, features: np.ndarray) -> KernelSignal:
        if self.feature_weights is None:
            return KernelSignal(0, 0.0, 0.0, {}, 0.0)

        poly_features = self.poly.transform(features[-1:])
        predicted = poly_features @ self.feature_weights

        pred_value = predicted[0]

        if pred_value > 0.01:
            direction = 1
            confidence = min(1.0, pred_value * 10)
        elif pred_value < -0.01:
            direction = -1
            confidence = min(1.0, abs(pred_value) * 10)
        else:
            direction = 0
            confidence = 0.0

        return KernelSignal(
            direction=direction,
            confidence=confidence,
            kernel_score=pred_value,
            feature_importance={},
            regime_indicator=0.0,
        )


class InteractionSignal:
    """
    Formula 72034: Feature Interaction Signal

    Explicit interaction terms: vol*momentum, flow*trend, etc.
    """

    FORMULA_ID = 72034

    def __init__(self):
        self.interaction_weights: Dict[str, float] = {}

    def compute_interactions(self, features: Dict[str, float]) -> Dict[str, float]:
        """Compute named interaction features."""
        interactions = {}

        # Key interactions
        if 'volatility' in features and 'momentum' in features:
            interactions['vol_x_mom'] = features['volatility'] * features['momentum']

        if 'flow' in features and 'trend' in features:
            interactions['flow_x_trend'] = features['flow'] * features['trend']

        if 'volume' in features and 'price_change' in features:
            interactions['vol_x_price'] = features['volume'] * features['price_change']

        return interactions

    def generate_signal(self, features: Dict[str, float]) -> KernelSignal:
        interactions = self.compute_interactions(features)

        # Simple heuristic: sum of interactions
        total_signal = sum(interactions.values())

        if total_signal > 0.1:
            direction = 1
            confidence = min(1.0, total_signal)
        elif total_signal < -0.1:
            direction = -1
            confidence = min(1.0, abs(total_signal))
        else:
            direction = 0
            confidence = 0.0

        return KernelSignal(
            direction=direction,
            confidence=confidence,
            kernel_score=total_signal,
            feature_importance=interactions,
            regime_indicator=0.0,
        )


class NonlinearMomentumSignal:
    """
    Formula 72035: Non-Linear Momentum Signal

    Momentum with non-linear scaling.
    Small moves ignored, large moves amplified.
    """

    FORMULA_ID = 72035

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold

    def _nonlinear_transform(self, x: float) -> float:
        """Apply non-linear transformation."""
        if abs(x) < self.threshold:
            return 0.0
        sign = 1 if x > 0 else -1
        # Quadratic above threshold
        return sign * ((abs(x) - self.threshold) ** 2 + self.threshold)

    def generate_signal(self, returns: np.ndarray) -> KernelSignal:
        # Recent momentum
        momentum_3d = np.sum(returns[-3:]) if len(returns) >= 3 else 0
        momentum_10d = np.sum(returns[-10:]) if len(returns) >= 10 else 0

        # Non-linear transformation
        nl_3d = self._nonlinear_transform(momentum_3d)
        nl_10d = self._nonlinear_transform(momentum_10d)

        combined = 0.6 * nl_3d + 0.4 * nl_10d

        if combined > 0.01:
            direction = 1
            confidence = min(1.0, combined * 10)
        elif combined < -0.01:
            direction = -1
            confidence = min(1.0, abs(combined) * 10)
        else:
            direction = 0
            confidence = 0.0

        return KernelSignal(
            direction=direction,
            confidence=confidence,
            kernel_score=combined,
            feature_importance={'nl_3d': nl_3d, 'nl_10d': nl_10d},
            regime_indicator=0.0,
        )


class NonlinearMeanRevSignal:
    """
    Formula 72036: Non-Linear Mean Reversion Signal

    Mean reversion with non-linear response.
    Extreme deviations = stronger reversal signal.
    """

    FORMULA_ID = 72036

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def generate_signal(self, prices: np.ndarray) -> KernelSignal:
        if len(prices) < self.lookback:
            return KernelSignal(0, 0.0, 0.0, {}, 0.0)

        # Z-score from moving average
        ma = np.mean(prices[-self.lookback:])
        std = np.std(prices[-self.lookback:])
        zscore = (prices[-1] - ma) / (std + 1e-10)

        # Non-linear: stronger signal at extremes
        if abs(zscore) < 1.5:
            direction = 0
            confidence = 0.0
        elif abs(zscore) < 2.0:
            direction = -1 if zscore > 0 else 1  # Mean revert
            confidence = 0.4
        elif abs(zscore) < 3.0:
            direction = -1 if zscore > 0 else 1
            confidence = 0.7
        else:
            direction = -1 if zscore > 0 else 1
            confidence = 0.9  # Very confident at extremes

        return KernelSignal(
            direction=direction,
            confidence=confidence,
            kernel_score=zscore,
            feature_importance={'zscore': zscore},
            regime_indicator=abs(zscore) / 3.0,
        )


class KernelRegimeSignal:
    """
    Formula 72037: Kernel Regime Signal

    Uses kernel density to identify regime.
    """

    FORMULA_ID = 72037

    def __init__(self, bandwidth: float = 0.5):
        self.bandwidth = bandwidth
        self.regime_centers: List[np.ndarray] = []

    def fit(self, features: np.ndarray):
        """Find regime centers using kernel density."""
        # Simple k-means-like clustering
        n_regimes = 3
        self.regime_centers = []

        # Initialize with percentiles
        for p in np.linspace(20, 80, n_regimes):
            center = np.percentile(features, p, axis=0)
            self.regime_centers.append(center)

    def _identify_regime(self, features: np.ndarray) -> Tuple[int, float]:
        """Identify current regime."""
        if not self.regime_centers:
            return 0, 0.0

        current = features.flatten()
        min_dist = float('inf')
        regime = 0

        for i, center in enumerate(self.regime_centers):
            dist = np.sqrt(np.sum((current - center) ** 2))
            if dist < min_dist:
                min_dist = dist
                regime = i

        confidence = np.exp(-min_dist / self.bandwidth)
        return regime, confidence

    def generate_signal(self, features: np.ndarray) -> KernelSignal:
        regime, conf = self._identify_regime(features[-1:])

        # Map regime to direction
        # Regime 0 = bearish, 1 = neutral, 2 = bullish
        direction = regime - 1  # -1, 0, or 1

        return KernelSignal(
            direction=direction,
            confidence=conf * 0.5,  # Scale down
            kernel_score=float(regime),
            feature_importance={},
            regime_indicator=float(regime),
        )


class NonlinearTrendSignal:
    """
    Formula 72038: Non-Linear Trend Signal

    Trend detection with non-linear filtering.
    """

    FORMULA_ID = 72038

    def __init__(self, fast: int = 10, slow: int = 30):
        self.fast = fast
        self.slow = slow

    def _exp_ma(self, data: np.ndarray, span: int) -> np.ndarray:
        """Exponential moving average."""
        alpha = 2 / (span + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def generate_signal(self, prices: np.ndarray) -> KernelSignal:
        if len(prices) < self.slow:
            return KernelSignal(0, 0.0, 0.0, {}, 0.0)

        fast_ema = self._exp_ma(prices, self.fast)
        slow_ema = self._exp_ma(prices, self.slow)

        # Trend = ratio
        trend_ratio = fast_ema[-1] / slow_ema[-1] - 1

        # Non-linear: amplify strong trends
        nl_trend = np.sign(trend_ratio) * (trend_ratio ** 2) * 100

        if nl_trend > 0.5:
            direction = 1
            confidence = min(1.0, nl_trend)
        elif nl_trend < -0.5:
            direction = -1
            confidence = min(1.0, abs(nl_trend))
        else:
            direction = 0
            confidence = 0.0

        return KernelSignal(
            direction=direction,
            confidence=confidence,
            kernel_score=nl_trend,
            feature_importance={'trend_ratio': trend_ratio},
            regime_indicator=abs(trend_ratio),
        )


class KernelVolatilitySignal:
    """
    Formula 72039: Kernel Volatility Signal

    Non-linear volatility regime detection.
    """

    FORMULA_ID = 72039

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.vol_history: List[float] = []

    def generate_signal(self, returns: np.ndarray) -> KernelSignal:
        if len(returns) < self.lookback:
            return KernelSignal(0, 0.0, 0.0, {}, 0.0)

        current_vol = np.std(returns[-self.lookback:])
        self.vol_history.append(current_vol)

        if len(self.vol_history) > 100:
            self.vol_history = self.vol_history[-100:]

        if len(self.vol_history) < 20:
            return KernelSignal(0, 0.0, 0.0, {}, 0.0)

        # Vol percentile
        percentile = np.sum(np.array(self.vol_history) < current_vol) / len(self.vol_history)

        # Non-linear: extreme vol = caution
        if percentile > 0.8:
            # High vol - reduce exposure
            direction = 0
            confidence = 0.0
        elif percentile < 0.2:
            # Low vol - good for trend following
            direction = 1 if returns[-1] > 0 else -1
            confidence = 0.6
        else:
            direction = 1 if returns[-1] > 0 else -1
            confidence = 0.3

        return KernelSignal(
            direction=direction,
            confidence=confidence,
            kernel_score=percentile,
            feature_importance={'vol_pct': percentile},
            regime_indicator=percentile,
        )


class KernelEnsembleSignal:
    """
    Formula 72040: Kernel Ensemble Signal

    Combines kernel-based signals.
    """

    FORMULA_ID = 72040

    def __init__(self):
        self.signals = [
            NonlinearMomentumSignal(),
            NonlinearMeanRevSignal(),
            NonlinearTrendSignal(),
        ]

    def generate_signal(self, prices: np.ndarray, returns: np.ndarray = None) -> KernelSignal:
        if returns is None:
            returns = np.diff(prices) / prices[:-1]

        results = []

        # Momentum
        results.append(self.signals[0].generate_signal(returns))

        # Mean rev
        results.append(self.signals[1].generate_signal(prices))

        # Trend
        results.append(self.signals[2].generate_signal(prices))

        # Combine
        total_dir = sum(r.direction * r.confidence for r in results)
        total_conf = sum(r.confidence for r in results)

        if total_conf > 0:
            avg_dir = total_dir / total_conf
            direction = 1 if avg_dir > 0.3 else (-1 if avg_dir < -0.3 else 0)
            confidence = total_conf / len(results)
        else:
            direction = 0
            confidence = 0.0

        return KernelSignal(
            direction=direction,
            confidence=confidence,
            kernel_score=total_dir,
            feature_importance={},
            regime_indicator=np.mean([r.regime_indicator for r in results]),
        )
