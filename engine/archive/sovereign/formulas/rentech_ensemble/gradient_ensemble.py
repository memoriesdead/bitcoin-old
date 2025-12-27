"""
Gradient Boosted Ensemble
=========================

Formula IDs: 72081-72085

Combines all formula outputs using gradient boosting.
This is the core of RenTech's approach - many weak signals → one strong signal.

RenTech insight: No single pattern wins consistently. The edge comes from
combining thousands of weak patterns, each with 50.5% edge, into one
system with 55%+ edge.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque


@dataclass
class SignalInput:
    """Input from a single formula."""
    formula_id: int
    direction: int  # -1, 0, 1
    confidence: float  # 0-1
    signal_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleOutput:
    """Output from ensemble."""
    direction: int
    confidence: float
    contributing_signals: int
    agreement_ratio: float
    feature_importance: Dict[int, float]  # formula_id -> importance


@dataclass
class TreeNode:
    """Simple decision tree node for gradient boosting."""
    feature_idx: int = -1
    threshold: float = 0.0
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    value: float = 0.0
    is_leaf: bool = True


class SimpleDecisionTree:
    """
    Simplified decision tree for gradient boosting.
    Keeps it dependency-free while maintaining core functionality.
    """

    def __init__(self, max_depth: int = 3, min_samples_split: int = 10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[TreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """Fit tree to data."""
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self.root = self._build_tree(X, y, sample_weight, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray,
                    weights: np.ndarray, depth: int) -> TreeNode:
        """Recursively build tree."""
        n_samples, n_features = X.shape

        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return TreeNode(value=self._weighted_mean(y, weights), is_leaf=True)

        # Find best split
        best_gain = 0
        best_feature = 0
        best_threshold = 0

        current_value = self._weighted_mean(y, weights)
        current_loss = np.sum(weights * (y - current_value) ** 2)

        for feature_idx in range(n_features):
            thresholds = np.percentile(X[:, feature_idx], [25, 50, 75])

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue

                left_value = self._weighted_mean(y[left_mask], weights[left_mask])
                right_value = self._weighted_mean(y[right_mask], weights[right_mask])

                left_loss = np.sum(weights[left_mask] * (y[left_mask] - left_value) ** 2)
                right_loss = np.sum(weights[right_mask] * (y[right_mask] - right_value) ** 2)

                gain = current_loss - (left_loss + right_loss)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        # No good split found
        if best_gain <= 0:
            return TreeNode(value=current_value, is_leaf=True)

        # Create split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        node = TreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            is_leaf=False
        )
        node.left = self._build_tree(X[left_mask], y[left_mask], weights[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], weights[right_mask], depth + 1)

        return node

    def _weighted_mean(self, y: np.ndarray, weights: np.ndarray) -> float:
        """Compute weighted mean."""
        if len(y) == 0:
            return 0.0
        return np.sum(weights * y) / (np.sum(weights) + 1e-10)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for all samples."""
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x: np.ndarray) -> float:
        """Predict for single sample."""
        node = self.root
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


class GradientBoostingEnsemble:
    """
    Gradient boosting ensemble for combining formula signals.

    Each formula output becomes a feature. The target is future return direction.
    Boosting learns which formulas are predictive and how to combine them.
    """

    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.trees: List[SimpleDecisionTree] = []
        self.initial_prediction: float = 0.0
        self.feature_importance_: Dict[int, float] = {}
        self.formula_ids: List[int] = []
        self.is_fitted = False

    def fit(self, signals_history: List[List[SignalInput]],
            returns: np.ndarray, formula_ids: List[int] = None):
        """
        Fit gradient boosting on historical signals and returns.

        Args:
            signals_history: List of signal lists (one per time period)
            returns: Future returns for each period
            formula_ids: List of formula IDs (for feature mapping)
        """
        # Build feature matrix
        X, y = self._build_features(signals_history, returns)

        if len(X) < 50:
            return self  # Not enough data

        self.formula_ids = formula_ids or list(range(X.shape[1]))

        # Initialize predictions
        self.initial_prediction = np.mean(y)
        predictions = np.full(len(y), self.initial_prediction)

        # Feature importance tracking
        feature_splits = np.zeros(X.shape[1])

        # Fit trees sequentially
        self.trees = []
        for i in range(self.n_estimators):
            # Compute residuals (negative gradient for MSE)
            residuals = y - predictions

            # Fit tree to residuals
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions
            tree_pred = tree.predict(X)
            predictions += self.learning_rate * tree_pred

            # Track feature importance
            self._update_feature_importance(tree.root, feature_splits)

        # Normalize feature importance
        total_splits = np.sum(feature_splits) + 1e-10
        for idx, formula_id in enumerate(self.formula_ids):
            self.feature_importance_[formula_id] = feature_splits[idx] / total_splits

        self.is_fitted = True
        return self

    def _build_features(self, signals_history: List[List[SignalInput]],
                       returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert signal history to feature matrix."""
        if not signals_history:
            return np.array([]), np.array([])

        # Get all unique formula IDs
        all_ids = set()
        for signals in signals_history:
            for s in signals:
                all_ids.add(s.formula_id)

        formula_ids = sorted(all_ids)
        id_to_idx = {fid: idx for idx, fid in enumerate(formula_ids)}

        # Build feature matrix
        n_samples = len(signals_history)
        n_features = len(formula_ids) * 2  # direction + confidence per formula

        X = np.zeros((n_samples, n_features))

        for i, signals in enumerate(signals_history):
            for s in signals:
                idx = id_to_idx[s.formula_id]
                X[i, idx * 2] = s.direction * s.confidence  # Weighted direction
                X[i, idx * 2 + 1] = s.confidence  # Raw confidence

        return X, returns[:n_samples]

    def _update_feature_importance(self, node: TreeNode, importance: np.ndarray):
        """Recursively update feature importance from tree splits."""
        if node is None or node.is_leaf:
            return

        # Map back to formula index (features are direction, confidence pairs)
        formula_idx = node.feature_idx // 2
        if formula_idx < len(importance):
            importance[formula_idx] += 1

        self._update_feature_importance(node.left, importance)
        self._update_feature_importance(node.right, importance)

    def predict(self, signals: List[SignalInput]) -> EnsembleOutput:
        """Generate ensemble prediction from current signals."""
        if not self.is_fitted or not self.trees:
            # Fallback: simple voting
            return self._simple_vote(signals)

        # Build feature vector
        x = self._signals_to_features(signals)

        # Ensemble prediction
        pred = self.initial_prediction
        for tree in self.trees:
            pred += self.learning_rate * tree._predict_one(x)

        # Convert to direction and confidence
        direction = 1 if pred > 0.001 else (-1 if pred < -0.001 else 0)
        confidence = min(1.0, abs(pred) * 10)  # Scale prediction to confidence

        # Calculate agreement
        directions = [s.direction for s in signals if s.direction != 0]
        if directions:
            agreement = abs(sum(directions)) / len(directions)
        else:
            agreement = 0.0

        return EnsembleOutput(
            direction=direction,
            confidence=confidence,
            contributing_signals=len([s for s in signals if s.direction != 0]),
            agreement_ratio=agreement,
            feature_importance=self.feature_importance_
        )

    def _signals_to_features(self, signals: List[SignalInput]) -> np.ndarray:
        """Convert signals to feature vector."""
        n_features = len(self.formula_ids) * 2
        x = np.zeros(n_features)

        id_to_idx = {fid: idx for idx, fid in enumerate(self.formula_ids)}

        for s in signals:
            if s.formula_id in id_to_idx:
                idx = id_to_idx[s.formula_id]
                x[idx * 2] = s.direction * s.confidence
                x[idx * 2 + 1] = s.confidence

        return x

    def _simple_vote(self, signals: List[SignalInput]) -> EnsembleOutput:
        """Simple weighted voting fallback."""
        if not signals:
            return EnsembleOutput(0, 0.0, 0, 0.0, {})

        weighted_sum = sum(s.direction * s.confidence for s in signals)
        total_conf = sum(s.confidence for s in signals)

        if total_conf > 0:
            avg_dir = weighted_sum / total_conf
            direction = 1 if avg_dir > 0.2 else (-1 if avg_dir < -0.2 else 0)
            confidence = total_conf / len(signals)
        else:
            direction = 0
            confidence = 0.0

        return EnsembleOutput(
            direction=direction,
            confidence=confidence,
            contributing_signals=len([s for s in signals if s.direction != 0]),
            agreement_ratio=0.0,
            feature_importance={}
        )


# =============================================================================
# FORMULA IMPLEMENTATIONS (72081-72085)
# =============================================================================

class GradientEnsembleSignal:
    """
    Formula 72081: Basic Gradient Ensemble

    Combines all available signals using gradient boosting.
    """

    FORMULA_ID = 72081

    def __init__(self, n_estimators: int = 30):
        self.ensemble = GradientBoostingEnsemble(n_estimators=n_estimators)
        self.signal_buffer: deque = deque(maxlen=500)
        self.return_buffer: deque = deque(maxlen=500)

    def fit(self, signals_history: List[List[SignalInput]], returns: np.ndarray):
        """Fit on historical data."""
        self.ensemble.fit(signals_history, returns)

    def update(self, signals: List[SignalInput], realized_return: float = None):
        """Update with new signals and optional realized return."""
        self.signal_buffer.append(signals)
        if realized_return is not None:
            self.return_buffer.append(realized_return)

        # Periodic refit
        if len(self.return_buffer) >= 100 and len(self.return_buffer) % 50 == 0:
            self.ensemble.fit(
                list(self.signal_buffer)[:-1],  # Exclude current
                np.array(list(self.return_buffer))
            )

    def generate_signal(self, signals: List[SignalInput]) -> EnsembleOutput:
        """Generate ensemble signal."""
        return self.ensemble.predict(signals)


class AdaptiveGradientEnsemble:
    """
    Formula 72082: Adaptive Gradient Ensemble

    Adapts learning rate and tree depth based on recent performance.
    """

    FORMULA_ID = 72082

    def __init__(self):
        self.ensemble = GradientBoostingEnsemble()
        self.performance_window: deque = deque(maxlen=50)
        self.current_lr = 0.1
        self.current_depth = 3

    def fit(self, signals_history: List[List[SignalInput]], returns: np.ndarray):
        """Fit with current hyperparameters."""
        self.ensemble.learning_rate = self.current_lr
        self.ensemble.max_depth = self.current_depth
        self.ensemble.fit(signals_history, returns)

    def update_performance(self, predicted_dir: int, actual_return: float):
        """Track prediction performance."""
        correct = (predicted_dir > 0 and actual_return > 0) or \
                  (predicted_dir < 0 and actual_return < 0) or \
                  (predicted_dir == 0)
        self.performance_window.append(1 if correct else 0)

        # Adapt hyperparameters
        if len(self.performance_window) >= 30:
            recent_acc = np.mean(list(self.performance_window)[-30:])

            if recent_acc < 0.45:  # Underperforming
                self.current_lr *= 0.9  # Reduce learning rate
                self.current_depth = max(2, self.current_depth - 1)
            elif recent_acc > 0.55:  # Overperforming
                self.current_lr = min(0.3, self.current_lr * 1.1)
                self.current_depth = min(5, self.current_depth + 1)

    def generate_signal(self, signals: List[SignalInput]) -> EnsembleOutput:
        """Generate signal with adaptive ensemble."""
        return self.ensemble.predict(signals)


class RegimeAwareEnsemble:
    """
    Formula 72083: Regime-Aware Gradient Ensemble

    Maintains separate ensembles for different market regimes.
    """

    FORMULA_ID = 72083

    def __init__(self):
        self.regime_ensembles: Dict[str, GradientBoostingEnsemble] = {
            'low_vol': GradientBoostingEnsemble(n_estimators=30),
            'high_vol': GradientBoostingEnsemble(n_estimators=50),
            'trending': GradientBoostingEnsemble(n_estimators=40),
            'mean_reverting': GradientBoostingEnsemble(n_estimators=40),
        }
        self.current_regime = 'low_vol'
        self.vol_history: List[float] = []

    def detect_regime(self, returns: np.ndarray) -> str:
        """Detect current market regime."""
        if len(returns) < 20:
            return 'low_vol'

        vol = np.std(returns[-20:])
        self.vol_history.append(vol)

        if len(self.vol_history) > 100:
            self.vol_history = self.vol_history[-100:]

        vol_percentile = np.sum(np.array(self.vol_history) < vol) / len(self.vol_history)

        # Check trend
        momentum = np.sum(returns[-10:])

        if vol_percentile > 0.7:
            return 'high_vol'
        elif vol_percentile < 0.3:
            if abs(momentum) > 0.05:
                return 'trending'
            else:
                return 'mean_reverting'
        else:
            return 'low_vol'

    def fit(self, signals_history: List[List[SignalInput]],
            returns: np.ndarray, regimes: List[str] = None):
        """Fit regime-specific ensembles."""
        if regimes is None:
            # Auto-detect regimes
            regimes = []
            for i in range(len(returns)):
                if i < 20:
                    regimes.append('low_vol')
                else:
                    regimes.append(self.detect_regime(returns[:i]))

        # Split data by regime
        regime_data: Dict[str, Tuple[List, List]] = {r: ([], []) for r in self.regime_ensembles}

        for i, (signals, ret, regime) in enumerate(zip(signals_history, returns, regimes)):
            if regime in regime_data:
                regime_data[regime][0].append(signals)
                regime_data[regime][1].append(ret)

        # Fit each regime ensemble
        for regime, (signals, rets) in regime_data.items():
            if len(rets) >= 30:
                self.regime_ensembles[regime].fit(signals, np.array(rets))

    def generate_signal(self, signals: List[SignalInput],
                       recent_returns: np.ndarray = None) -> EnsembleOutput:
        """Generate regime-aware signal."""
        if recent_returns is not None:
            self.current_regime = self.detect_regime(recent_returns)

        ensemble = self.regime_ensembles.get(self.current_regime,
                                              self.regime_ensembles['low_vol'])

        output = ensemble.predict(signals)
        output.metadata = {'regime': self.current_regime}
        return output


class FeatureSelectedEnsemble:
    """
    Formula 72084: Feature-Selected Gradient Ensemble

    Automatically selects most predictive formulas.
    """

    FORMULA_ID = 72084

    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.full_ensemble = GradientBoostingEnsemble(n_estimators=50)
        self.selected_ensemble = GradientBoostingEnsemble(n_estimators=30)
        self.selected_formulas: List[int] = []

    def fit(self, signals_history: List[List[SignalInput]], returns: np.ndarray):
        """Fit and select top features."""
        # First fit full ensemble
        self.full_ensemble.fit(signals_history, returns)

        # Select top-k formulas by importance
        importance = self.full_ensemble.feature_importance_
        sorted_formulas = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        self.selected_formulas = [f[0] for f in sorted_formulas[:self.top_k]]

        # Filter signals to selected formulas only
        filtered_history = []
        for signals in signals_history:
            filtered = [s for s in signals if s.formula_id in self.selected_formulas]
            filtered_history.append(filtered)

        # Fit selected ensemble
        if any(filtered_history):
            self.selected_ensemble.fit(filtered_history, returns, self.selected_formulas)

    def generate_signal(self, signals: List[SignalInput]) -> EnsembleOutput:
        """Generate signal using selected formulas only."""
        filtered = [s for s in signals if s.formula_id in self.selected_formulas]

        if not filtered:
            return self.full_ensemble.predict(signals)

        output = self.selected_ensemble.predict(filtered)
        output.metadata = {'selected_formulas': self.selected_formulas}
        return output


class GradientEnsembleWithDecay:
    """
    Formula 72085: Gradient Ensemble with Alpha Decay

    Weights recent data more heavily, models signal decay.
    """

    FORMULA_ID = 72085

    def __init__(self, decay_rate: float = 0.99):
        self.decay_rate = decay_rate
        self.ensemble = GradientBoostingEnsemble(n_estimators=40)
        self.signal_ages: Dict[int, int] = {}  # formula_id -> days since last good signal

    def fit(self, signals_history: List[List[SignalInput]],
            returns: np.ndarray, timestamps: List[int] = None):
        """Fit with time-decay weights."""
        n = len(returns)

        # Create decay weights (more recent = higher weight)
        weights = np.array([self.decay_rate ** (n - 1 - i) for i in range(n)])
        weights /= np.sum(weights)  # Normalize

        # Weight the returns
        weighted_returns = returns * weights * n  # Scale back up

        self.ensemble.fit(signals_history, weighted_returns)

        # Track signal effectiveness over time
        for i, signals in enumerate(signals_history):
            for s in signals:
                if s.formula_id not in self.signal_ages:
                    self.signal_ages[s.formula_id] = 0

                # Check if signal was correct
                if (s.direction > 0 and returns[i] > 0) or (s.direction < 0 and returns[i] < 0):
                    self.signal_ages[s.formula_id] = 0  # Reset age
                else:
                    self.signal_ages[s.formula_id] += 1

    def generate_signal(self, signals: List[SignalInput]) -> EnsembleOutput:
        """Generate signal with decay-adjusted confidence."""
        output = self.ensemble.predict(signals)

        # Adjust confidence based on signal freshness
        if signals:
            avg_age = np.mean([self.signal_ages.get(s.formula_id, 0) for s in signals])
            decay_factor = self.decay_rate ** avg_age
            output.confidence *= decay_factor

        return output
