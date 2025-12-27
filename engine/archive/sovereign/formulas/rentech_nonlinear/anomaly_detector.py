"""
Anomaly Detection for Unusual Market States
==========================================

Formula IDs: 72041-72050

Detects unusual market conditions that often precede large moves.
Anomalies can be opportunities or warnings.

RenTech insight: The most profitable trades often come from
recognizing when the market is behaving "unusually".
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


@dataclass
class AnomalySignal:
    """Signal from anomaly detection."""
    direction: int
    confidence: float
    anomaly_score: float
    anomaly_type: str
    is_anomaly: bool


class IsolationForestDetector:
    """
    Isolation Forest for anomaly detection.

    Anomalies are isolated with fewer splits.
    """

    def __init__(self, n_trees: int = 100, sample_size: int = 256):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees: List = []
        self.is_fitted = False

    def _build_tree(self, X: np.ndarray, height_limit: int) -> dict:
        """Build isolation tree."""
        n_samples, n_features = X.shape

        if n_samples <= 1 or height_limit <= 0:
            return {'type': 'leaf', 'size': n_samples}

        # Random feature and split point
        feature_idx = np.random.randint(n_features)
        feature_values = X[:, feature_idx]
        min_val, max_val = feature_values.min(), feature_values.max()

        if min_val == max_val:
            return {'type': 'leaf', 'size': n_samples}

        split_val = np.random.uniform(min_val, max_val)

        left_mask = X[:, feature_idx] < split_val
        right_mask = ~left_mask

        return {
            'type': 'split',
            'feature': feature_idx,
            'split': split_val,
            'left': self._build_tree(X[left_mask], height_limit - 1),
            'right': self._build_tree(X[right_mask], height_limit - 1),
        }

    def fit(self, X: np.ndarray):
        """Fit isolation forest."""
        n_samples = len(X)
        height_limit = int(np.ceil(np.log2(max(self.sample_size, 2))))

        self.trees = []
        for _ in range(self.n_trees):
            # Sample subset
            indices = np.random.choice(n_samples, min(self.sample_size, n_samples), replace=False)
            sample = X[indices]
            tree = self._build_tree(sample, height_limit)
            self.trees.append(tree)

        self.is_fitted = True
        return self

    def _path_length(self, x: np.ndarray, tree: dict, depth: int = 0) -> float:
        """Get path length for sample in tree."""
        if tree['type'] == 'leaf':
            # Add expected path length for remaining samples
            n = tree['size']
            if n <= 1:
                return depth
            else:
                # Average path length approximation
                return depth + 2 * (np.log(n - 1) + 0.5772) - 2 * (n - 1) / n

        if x[tree['feature']] < tree['split']:
            return self._path_length(x, tree['left'], depth + 1)
        else:
            return self._path_length(x, tree['right'], depth + 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (higher = more anomalous)."""
        if not self.is_fitted:
            raise RuntimeError("Must fit first")

        n = len(X)
        scores = np.zeros(n)

        for i, x in enumerate(X):
            path_lengths = [self._path_length(x, tree) for tree in self.trees]
            avg_path = np.mean(path_lengths)

            # Expected path length
            c_n = 2 * (np.log(self.sample_size - 1) + 0.5772) - 2 * (self.sample_size - 1) / self.sample_size

            # Anomaly score: 2^(-avg_path / c_n)
            scores[i] = 2 ** (-avg_path / c_n)

        return scores


class LocalOutlierDetector:
    """
    Local Outlier Factor (LOF) for density-based anomaly detection.
    """

    def __init__(self, k: int = 20):
        self.k = k
        self.X_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        """Store training data."""
        self.X_train = X.copy()
        return self

    def _knn_distances(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get k nearest neighbor distances and indices."""
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        indices = np.argsort(distances)[:self.k]
        return distances[indices], indices

    def _local_reachability_density(self, x: np.ndarray) -> float:
        """Compute LRD for a point."""
        k_dists, k_indices = self._knn_distances(x)
        k_dist = k_dists[-1]  # k-distance

        # Reachability distances
        reach_dists = []
        for i, idx in enumerate(k_indices):
            neighbor = self.X_train[idx]
            _, neighbor_k_dists = self._knn_distances(neighbor)
            neighbor_k_dist = neighbor_k_dists[-1]
            reach_dist = max(neighbor_k_dist, k_dists[i])
            reach_dists.append(reach_dist)

        return len(reach_dists) / (sum(reach_dists) + 1e-10)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get LOF scores (higher = more anomalous)."""
        if self.X_train is None:
            raise RuntimeError("Must fit first")

        scores = []
        for x in X:
            lrd_x = self._local_reachability_density(x)
            _, k_indices = self._knn_distances(x)

            lrd_neighbors = [self._local_reachability_density(self.X_train[i])
                            for i in k_indices]

            lof = np.mean(lrd_neighbors) / (lrd_x + 1e-10)
            scores.append(lof)

        return np.array(scores)


class AnomalyScorer:
    """
    Combine multiple anomaly detection methods.
    """

    def __init__(self):
        self.isolation_forest = IsolationForestDetector()
        self.lof = LocalOutlierDetector(k=10)
        self.is_fitted = False

    def fit(self, features: np.ndarray):
        """Fit all detectors."""
        self.isolation_forest.fit(features)
        self.lof.fit(features)
        self.is_fitted = True

    def score(self, features: np.ndarray) -> Tuple[float, str]:
        """Get combined anomaly score."""
        if not self.is_fitted:
            return 0.0, 'not_fitted'

        iso_score = self.isolation_forest.score_samples(features[-1:])[0]
        # LOF can be slow, use simpler check
        lof_score = min(2.0, self.lof.score_samples(features[-1:])[0])

        # Combine scores
        combined = (iso_score * 0.6 + lof_score / 2 * 0.4)

        if combined > 0.7:
            return combined, 'strong_anomaly'
        elif combined > 0.5:
            return combined, 'moderate_anomaly'
        else:
            return combined, 'normal'


# =============================================================================
# FORMULA IMPLEMENTATIONS (72041-72050)
# =============================================================================

class IsolationAnomalySignal:
    """
    Formula 72041: Isolation Forest Anomaly Signal

    Trades when market is in unusual state.
    Anomalies often precede large moves.
    """

    FORMULA_ID = 72041

    def __init__(self):
        self.detector = IsolationForestDetector(n_trees=50)
        self.is_fitted = False

    def fit(self, features: np.ndarray):
        """Fit detector on historical data."""
        self.detector.fit(features)
        self.is_fitted = True

    def generate_signal(self, features: np.ndarray, returns: np.ndarray = None) -> AnomalySignal:
        if not self.is_fitted:
            return AnomalySignal(0, 0.0, 0.0, 'not_fitted', False)

        score = self.detector.score_samples(features[-1:])[0]
        is_anomaly = score > 0.6

        if is_anomaly:
            # Anomaly detected - direction from recent momentum
            if returns is not None and len(returns) >= 3:
                recent_momentum = np.sum(returns[-3:])
                direction = 1 if recent_momentum > 0 else -1
            else:
                direction = 0
            confidence = score
        else:
            direction = 0
            confidence = 0.0

        return AnomalySignal(
            direction=direction,
            confidence=confidence,
            anomaly_score=score,
            anomaly_type='isolation_forest',
            is_anomaly=is_anomaly,
        )


class LOFAnomalySignal:
    """
    Formula 72042: Local Outlier Factor Signal

    Uses density-based anomaly detection.
    """

    FORMULA_ID = 72042

    def __init__(self, k: int = 15):
        self.detector = LocalOutlierDetector(k=k)
        self.is_fitted = False

    def fit(self, features: np.ndarray):
        self.detector.fit(features)
        self.is_fitted = True

    def generate_signal(self, features: np.ndarray, returns: np.ndarray = None) -> AnomalySignal:
        if not self.is_fitted:
            return AnomalySignal(0, 0.0, 0.0, 'not_fitted', False)

        score = self.detector.score_samples(features[-1:])[0]
        is_anomaly = score > 1.5

        if is_anomaly and returns is not None:
            direction = 1 if np.mean(returns[-5:]) > 0 else -1
            confidence = min(1.0, score / 3.0)
        else:
            direction = 0
            confidence = 0.0

        return AnomalySignal(
            direction=direction,
            confidence=confidence,
            anomaly_score=score,
            anomaly_type='lof',
            is_anomaly=is_anomaly,
        )


class ExtremeMoveSignal:
    """
    Formula 72043: Extreme Move Signal

    Detects statistical outliers in price moves.
    """

    FORMULA_ID = 72043

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.returns_history: List[float] = []

    def generate_signal(self, returns: np.ndarray) -> AnomalySignal:
        current = returns[-1] if len(returns) > 0 else 0
        self.returns_history.extend(returns.tolist())

        if len(self.returns_history) > 500:
            self.returns_history = self.returns_history[-500:]

        if len(self.returns_history) < 50:
            return AnomalySignal(0, 0.0, 0.0, 'insufficient_data', False)

        mean = np.mean(self.returns_history)
        std = np.std(self.returns_history)
        zscore = (current - mean) / (std + 1e-10)

        is_extreme = abs(zscore) > self.threshold

        if is_extreme:
            # Fade extreme moves
            direction = -1 if zscore > 0 else 1
            confidence = min(1.0, (abs(zscore) - self.threshold) / 2.0)
        else:
            direction = 0
            confidence = 0.0

        return AnomalySignal(
            direction=direction,
            confidence=confidence,
            anomaly_score=abs(zscore),
            anomaly_type='extreme_move',
            is_anomaly=is_extreme,
        )


class StructuralBreakSignal:
    """
    Formula 72044: Structural Break Signal

    Detects regime changes using CUSUM.
    """

    FORMULA_ID = 72044

    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

    def generate_signal(self, returns: np.ndarray) -> AnomalySignal:
        if len(returns) < 20:
            return AnomalySignal(0, 0.0, 0.0, 'insufficient_data', False)

        # CUSUM statistic
        mean = np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)
        std = np.std(returns[-50:]) if len(returns) >= 50 else np.std(returns)

        z = (returns[-1] - mean) / (std + 1e-10)

        self.cusum_pos = max(0, self.cusum_pos + z - 0.5)
        self.cusum_neg = max(0, self.cusum_neg - z - 0.5)

        # Break detection
        is_break = self.cusum_pos > self.threshold or self.cusum_neg > self.threshold

        if is_break:
            # Trade in direction of break
            direction = 1 if self.cusum_pos > self.cusum_neg else -1
            confidence = min(1.0, max(self.cusum_pos, self.cusum_neg) / (self.threshold * 2))

            # Reset after break
            self.cusum_pos = 0.0
            self.cusum_neg = 0.0
        else:
            direction = 0
            confidence = 0.0

        return AnomalySignal(
            direction=direction,
            confidence=confidence,
            anomaly_score=max(self.cusum_pos, self.cusum_neg),
            anomaly_type='structural_break',
            is_anomaly=is_break,
        )


class AnomalyRegimeSignal:
    """
    Formula 72045: Anomaly Regime Signal

    Identifies when we're in an anomalous regime.
    """

    FORMULA_ID = 72045

    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.scorer = AnomalyScorer()
        self.is_fitted = False

    def fit(self, features: np.ndarray):
        self.scorer.fit(features)
        self.is_fitted = True

    def generate_signal(self, features: np.ndarray, returns: np.ndarray = None) -> AnomalySignal:
        if not self.is_fitted:
            return AnomalySignal(0, 0.0, 0.0, 'not_fitted', False)

        score, regime = self.scorer.score(features)

        if regime == 'strong_anomaly':
            direction = 0  # Stay flat in strong anomaly
            confidence = 0.0
        elif regime == 'moderate_anomaly' and returns is not None:
            # Cautious trade
            direction = 1 if np.mean(returns[-3:]) > 0 else -1
            confidence = 0.3
        else:
            direction = 0
            confidence = 0.0

        return AnomalySignal(
            direction=direction,
            confidence=confidence,
            anomaly_score=score,
            anomaly_type=regime,
            is_anomaly=regime != 'normal',
        )


class ClusterAnomalySignal:
    """
    Formula 72046: Cluster Anomaly Signal

    Detects when current state is far from any cluster.
    """

    FORMULA_ID = 72046

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.cluster_centers: List[np.ndarray] = []

    def fit(self, features: np.ndarray):
        """Simple k-means clustering."""
        n = len(features)
        indices = np.random.choice(n, min(self.n_clusters, n), replace=False)
        self.cluster_centers = [features[i] for i in indices]

        # Few iterations of k-means
        for _ in range(10):
            assignments = []
            for x in features:
                dists = [np.sqrt(np.sum((x - c) ** 2)) for c in self.cluster_centers]
                assignments.append(np.argmin(dists))

            # Update centers
            for k in range(self.n_clusters):
                cluster_points = features[np.array(assignments) == k]
                if len(cluster_points) > 0:
                    self.cluster_centers[k] = np.mean(cluster_points, axis=0)

    def generate_signal(self, features: np.ndarray, returns: np.ndarray = None) -> AnomalySignal:
        if not self.cluster_centers:
            return AnomalySignal(0, 0.0, 0.0, 'not_fitted', False)

        current = features[-1]
        dists = [np.sqrt(np.sum((current - c) ** 2)) for c in self.cluster_centers]
        min_dist = min(dists)

        # Anomaly if far from all clusters
        avg_dist = np.mean(dists)
        is_anomaly = min_dist > avg_dist * 1.5

        if is_anomaly and returns is not None:
            direction = 1 if np.mean(returns[-3:]) > 0 else -1
            confidence = min(1.0, min_dist / (avg_dist * 2))
        else:
            direction = 0
            confidence = 0.0

        return AnomalySignal(
            direction=direction,
            confidence=confidence,
            anomaly_score=min_dist / (avg_dist + 1e-10),
            anomaly_type='cluster_distance',
            is_anomaly=is_anomaly,
        )


class DistributionShiftSignal:
    """
    Formula 72047: Distribution Shift Signal

    Detects when recent returns differ from historical.
    """

    FORMULA_ID = 72047

    def __init__(self, recent: int = 20, historical: int = 100):
        self.recent = recent
        self.historical = historical

    def generate_signal(self, returns: np.ndarray) -> AnomalySignal:
        if len(returns) < self.historical:
            return AnomalySignal(0, 0.0, 0.0, 'insufficient_data', False)

        recent_returns = returns[-self.recent:]
        historical_returns = returns[-self.historical:-self.recent]

        # KS-like statistic (simplified)
        recent_mean = np.mean(recent_returns)
        hist_mean = np.mean(historical_returns)
        recent_std = np.std(recent_returns)
        hist_std = np.std(historical_returns)

        mean_shift = abs(recent_mean - hist_mean) / (hist_std + 1e-10)
        vol_shift = abs(recent_std - hist_std) / (hist_std + 1e-10)

        total_shift = mean_shift + vol_shift

        is_shift = total_shift > 2.0

        if is_shift:
            # Trade with the new regime
            direction = 1 if recent_mean > hist_mean else -1
            confidence = min(1.0, total_shift / 4.0)
        else:
            direction = 0
            confidence = 0.0

        return AnomalySignal(
            direction=direction,
            confidence=confidence,
            anomaly_score=total_shift,
            anomaly_type='distribution_shift',
            is_anomaly=is_shift,
        )


class TailRiskSignal:
    """
    Formula 72048: Tail Risk Signal

    Monitors for fat tail events.
    """

    FORMULA_ID = 72048

    def __init__(self, percentile: float = 5.0):
        self.percentile = percentile
        self.returns_history: List[float] = []

    def generate_signal(self, returns: np.ndarray) -> AnomalySignal:
        self.returns_history.extend(returns.tolist())
        if len(self.returns_history) > 1000:
            self.returns_history = self.returns_history[-1000:]

        if len(self.returns_history) < 100:
            return AnomalySignal(0, 0.0, 0.0, 'insufficient_data', False)

        current = returns[-1]
        low_threshold = np.percentile(self.returns_history, self.percentile)
        high_threshold = np.percentile(self.returns_history, 100 - self.percentile)

        if current < low_threshold:
            # Left tail - potential capitulation
            direction = 1  # Buy the dip
            confidence = min(1.0, (low_threshold - current) / abs(low_threshold + 1e-10))
            anomaly_type = 'left_tail'
            is_anomaly = True
        elif current > high_threshold:
            # Right tail - potential blow-off top
            direction = -1  # Fade the spike
            confidence = min(1.0, (current - high_threshold) / (high_threshold + 1e-10))
            anomaly_type = 'right_tail'
            is_anomaly = True
        else:
            direction = 0
            confidence = 0.0
            anomaly_type = 'normal'
            is_anomaly = False

        return AnomalySignal(
            direction=direction,
            confidence=confidence,
            anomaly_score=abs(current) / (np.std(self.returns_history) + 1e-10),
            anomaly_type=anomaly_type,
            is_anomaly=is_anomaly,
        )


class BlackSwanSignal:
    """
    Formula 72049: Black Swan Signal

    Detects rare, extreme events.
    """

    FORMULA_ID = 72049

    def __init__(self, zscore_threshold: float = 4.0):
        self.zscore_threshold = zscore_threshold
        self.history: List[float] = []

    def generate_signal(self, returns: np.ndarray) -> AnomalySignal:
        current = returns[-1] if len(returns) > 0 else 0
        self.history.extend(returns.tolist())

        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        if len(self.history) < 100:
            return AnomalySignal(0, 0.0, 0.0, 'insufficient_data', False)

        mean = np.mean(self.history)
        std = np.std(self.history)
        zscore = (current - mean) / (std + 1e-10)

        is_black_swan = abs(zscore) > self.zscore_threshold

        if is_black_swan:
            # Black swan = fade the extreme
            direction = -1 if zscore > 0 else 1
            confidence = min(1.0, (abs(zscore) - self.zscore_threshold) / 2.0)
        else:
            direction = 0
            confidence = 0.0

        return AnomalySignal(
            direction=direction,
            confidence=confidence,
            anomaly_score=abs(zscore),
            anomaly_type='black_swan' if is_black_swan else 'normal',
            is_anomaly=is_black_swan,
        )


class AnomalyEnsembleSignal:
    """
    Formula 72050: Anomaly Ensemble Signal

    Combines all anomaly detection methods.
    """

    FORMULA_ID = 72050

    def __init__(self):
        self.signals = [
            ExtremeMoveSignal(),
            StructuralBreakSignal(),
            DistributionShiftSignal(),
            TailRiskSignal(),
            BlackSwanSignal(),
        ]

    def generate_signal(self, features: np.ndarray, returns: np.ndarray) -> AnomalySignal:
        results = [s.generate_signal(returns) for s in self.signals]

        # Count anomalies
        n_anomalies = sum(1 for r in results if r.is_anomaly)

        if n_anomalies >= 3:
            # Multiple anomaly detectors agree
            directions = [r.direction for r in results if r.is_anomaly and r.direction != 0]
            if directions:
                avg_dir = np.mean(directions)
                direction = 1 if avg_dir > 0 else (-1 if avg_dir < 0 else 0)
            else:
                direction = 0
            confidence = n_anomalies / len(self.signals)
            is_anomaly = True
        else:
            direction = 0
            confidence = 0.0
            is_anomaly = False

        avg_score = np.mean([r.anomaly_score for r in results])

        return AnomalySignal(
            direction=direction,
            confidence=confidence,
            anomaly_score=avg_score,
            anomaly_type='ensemble',
            is_anomaly=is_anomaly,
        )
