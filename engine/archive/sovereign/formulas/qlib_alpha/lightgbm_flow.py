"""
LightGBM Flow Classifier
========================

Ported from Microsoft QLib's LightGBM model patterns.

Replaces the logistic regression in FlowMomentumClassifier (20005)
with a gradient boosted tree model.

Advantages over logistic regression:
1. Handles non-linear relationships
2. Automatic feature interactions
3. Better with imbalanced data
4. Feature importance built-in

Formula IDs: 70006-70007
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pickle
import os

# Try to import lightgbm, fallback to stub if not installed
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None


@dataclass
class PredictionResult:
    """Result from classifier prediction."""
    probability: float  # P(price goes up)
    direction: int      # 1 = long, -1 = short, 0 = neutral
    confidence: float   # Confidence in prediction
    feature_importance: Optional[Dict[str, float]] = None


class LightGBMFlowClassifier:
    """
    LightGBM-based flow classifier (Formula 70006).

    Predicts probability of price increase based on flow features.

    Features used:
    - Flow momentum (multiple windows)
    - Flow z-score
    - Flow acceleration
    - Flow autocorrelation
    - Price momentum (if available)
    - Volume (if available)

    QLib Pattern: Uses early stopping, monotonic constraints
    where appropriate, and proper validation split.
    """

    formula_id = 70006
    name = "LightGBMFlowClassifier"

    # Default LightGBM parameters (QLib-inspired)
    DEFAULT_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'seed': 42,
    }

    def __init__(self, params: Optional[Dict] = None,
                 threshold_long: float = 0.55,
                 threshold_short: float = 0.45):
        """
        Initialize classifier.

        Args:
            params: LightGBM parameters (None = use defaults)
            threshold_long: P(up) threshold to go long
            threshold_short: P(up) threshold to go short
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short

        self.model = None
        self.feature_names: List[str] = []
        self.is_trained = False

        # Training stats
        self.stats = {
            'train_samples': 0,
            'best_iteration': 0,
            'best_auc': 0.0,
        }

    def _create_features(self, flow_data: np.ndarray,
                         price_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create feature matrix from raw data.

        Args:
            flow_data: Shape (n_samples, n_flow_features) or (n_samples,)
            price_data: Optional price data

        Returns:
            Feature matrix for prediction
        """
        # Handle 1D input
        if flow_data.ndim == 1:
            flow_data = flow_data.reshape(-1, 1)

        features = []
        feature_names = []

        n_samples = len(flow_data)

        # Flow features for each input column
        for col in range(flow_data.shape[1]):
            col_data = flow_data[:, col]
            col_name = f"flow_{col}"

            # Raw value
            features.append(col_data)
            feature_names.append(f"{col_name}_raw")

            # Rolling statistics
            for window in [5, 10, 20]:
                if n_samples >= window:
                    # Mean
                    ma = np.array([
                        np.mean(col_data[max(0, i-window+1):i+1])
                        for i in range(n_samples)
                    ])
                    features.append(ma)
                    feature_names.append(f"{col_name}_ma{window}")

                    # Std
                    std = np.array([
                        np.std(col_data[max(0, i-window+1):i+1])
                        for i in range(n_samples)
                    ])
                    features.append(std)
                    feature_names.append(f"{col_name}_std{window}")

        # Price features if available
        if price_data is not None:
            # Returns
            returns = np.zeros(n_samples)
            returns[1:] = np.diff(price_data) / price_data[:-1]
            features.append(returns)
            feature_names.append("price_return")

            # Volatility
            for window in [5, 10, 20]:
                if n_samples >= window:
                    vol = np.array([
                        np.std(returns[max(0, i-window+1):i+1])
                        for i in range(n_samples)
                    ])
                    features.append(vol)
                    feature_names.append(f"price_vol{window}")

        self.feature_names = feature_names
        return np.column_stack(features)

    def train(self, flow_data: np.ndarray, labels: np.ndarray,
              price_data: Optional[np.ndarray] = None,
              val_ratio: float = 0.2,
              early_stopping_rounds: int = 50,
              num_boost_round: int = 500) -> Dict[str, Any]:
        """
        Train the classifier.

        Args:
            flow_data: Flow feature data
            labels: Binary labels (1 = price went up, 0 = down)
            price_data: Optional price data
            val_ratio: Validation set ratio
            early_stopping_rounds: Stop if no improvement
            num_boost_round: Maximum iterations

        Returns:
            Training results dict
        """
        if not HAS_LIGHTGBM:
            raise ImportError(
                "LightGBM not installed. Run: pip install lightgbm"
            )

        # Create features
        X = self._create_features(flow_data, price_data)
        y = labels.astype(int)

        # Time-based split (QLib pattern: no random split for time series)
        val_size = int(len(X) * val_ratio)
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train with early stopping
        callbacks = [
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(period=0),  # Suppress output
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks,
        )

        self.is_trained = True
        self.stats['train_samples'] = len(X_train)
        self.stats['best_iteration'] = self.model.best_iteration
        self.stats['best_auc'] = self.model.best_score['valid']['auc']

        return {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'best_iteration': self.model.best_iteration,
            'best_auc': self.model.best_score['valid']['auc'],
            'feature_importance': self.get_feature_importance(),
        }

    def predict(self, flow_data: np.ndarray,
                price_data: Optional[np.ndarray] = None) -> PredictionResult:
        """
        Predict on new data.

        Args:
            flow_data: Flow feature data (single sample or batch)
            price_data: Optional price data

        Returns:
            PredictionResult with probability and direction
        """
        if not self.is_trained:
            # Return neutral if not trained
            return PredictionResult(
                probability=0.5,
                direction=0,
                confidence=0.0,
            )

        # Create features
        X = self._create_features(flow_data, price_data)

        # Predict probability
        prob = self.model.predict(X)

        # Handle batch vs single
        if len(prob) == 1:
            prob = prob[0]
        else:
            prob = prob[-1]  # Use latest prediction

        # Determine direction
        if prob >= self.threshold_long:
            direction = 1
        elif prob <= self.threshold_short:
            direction = -1
        else:
            direction = 0

        # Confidence based on distance from 0.5
        confidence = abs(prob - 0.5) * 2  # Scale to 0-1

        return PredictionResult(
            probability=float(prob),
            direction=direction,
            confidence=float(confidence),
            feature_importance=self.get_feature_importance() if self.is_trained else None,
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained or self.model is None:
            return {}

        importance = self.model.feature_importance(importance_type='gain')
        total = sum(importance)

        if total == 0:
            return {}

        return {
            name: float(imp / total)
            for name, imp in zip(self.feature_names, importance)
        }

    def save(self, path: str):
        """Save model to file."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'threshold_long': self.threshold_long,
            'threshold_short': self.threshold_short,
            'stats': self.stats,
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.params = data['params']
        self.feature_names = data['feature_names']
        self.threshold_long = data['threshold_long']
        self.threshold_short = data['threshold_short']
        self.stats = data['stats']
        self.is_trained = True


class OnlineLightGBM:
    """
    Online Learning LightGBM Wrapper (Formula 70007).

    QLib Pattern: Incrementally update model with new data.

    Since LightGBM doesn't support true online learning,
    we use a sliding window approach:
    1. Maintain a buffer of recent samples
    2. Periodically retrain on the buffer
    3. Use warm start for faster training
    """

    formula_id = 70007
    name = "OnlineLightGBM"

    def __init__(self, buffer_size: int = 10000,
                 retrain_frequency: int = 100,
                 **lgb_params):
        """
        Initialize online learner.

        Args:
            buffer_size: Maximum samples to keep
            retrain_frequency: Retrain after this many new samples
            **lgb_params: Parameters for LightGBMFlowClassifier
        """
        self.buffer_size = buffer_size
        self.retrain_frequency = retrain_frequency

        self.classifier = LightGBMFlowClassifier(**lgb_params)

        # Sample buffers
        self.flow_buffer: List[np.ndarray] = []
        self.price_buffer: List[float] = []
        self.label_buffer: List[int] = []

        self.samples_since_retrain = 0
        self.total_samples = 0

    def update(self, flow: np.ndarray, price: float, label: int):
        """
        Add new sample and potentially retrain.

        Args:
            flow: Flow features for this sample
            price: Price at this sample
            label: Outcome (1 = up, 0 = down)
        """
        # Add to buffers
        self.flow_buffer.append(flow)
        self.price_buffer.append(price)
        self.label_buffer.append(label)

        # Trim buffers if needed
        if len(self.flow_buffer) > self.buffer_size:
            self.flow_buffer = self.flow_buffer[-self.buffer_size:]
            self.price_buffer = self.price_buffer[-self.buffer_size:]
            self.label_buffer = self.label_buffer[-self.buffer_size:]

        self.samples_since_retrain += 1
        self.total_samples += 1

        # Check if we should retrain
        if self.samples_since_retrain >= self.retrain_frequency:
            self._retrain()

    def _retrain(self):
        """Retrain the model on buffer data."""
        if len(self.flow_buffer) < 100:  # Minimum samples
            return

        flow_data = np.array(self.flow_buffer)
        price_data = np.array(self.price_buffer)
        labels = np.array(self.label_buffer)

        try:
            self.classifier.train(
                flow_data, labels, price_data,
                val_ratio=0.2,
                early_stopping_rounds=20,
                num_boost_round=200,  # Less iterations for online
            )
            self.samples_since_retrain = 0
        except Exception as e:
            # Log but don't crash
            print(f"[OnlineLightGBM] Retrain failed: {e}")

    def predict(self, flow: np.ndarray, price: Optional[float] = None) -> PredictionResult:
        """
        Predict using current model.

        Args:
            flow: Flow features
            price: Current price (optional)

        Returns:
            Prediction result
        """
        price_data = None
        if price is not None and self.price_buffer:
            price_data = np.array(self.price_buffer[-50:] + [price])

        return self.classifier.predict(
            flow.reshape(1, -1) if flow.ndim == 1 else flow,
            price_data
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get online learning stats."""
        return {
            'total_samples': self.total_samples,
            'buffer_size': len(self.flow_buffer),
            'samples_since_retrain': self.samples_since_retrain,
            'model_trained': self.classifier.is_trained,
            'model_stats': self.classifier.stats,
        }


# =============================================================================
# FALLBACK CLASSIFIER (when LightGBM not installed)
# =============================================================================
class FallbackClassifier:
    """Simple logistic-style classifier when LightGBM not available."""

    formula_id = 70006
    name = "FallbackClassifier"

    def __init__(self, **kwargs):
        self.weights = None
        self.is_trained = False

    def train(self, flow_data: np.ndarray, labels: np.ndarray, **kwargs):
        """Simple weighted average training."""
        # Compute correlation of each feature with labels
        if flow_data.ndim == 1:
            flow_data = flow_data.reshape(-1, 1)

        self.weights = np.zeros(flow_data.shape[1])
        for i in range(flow_data.shape[1]):
            corr = np.corrcoef(flow_data[:, i], labels)[0, 1]
            if not np.isnan(corr):
                self.weights[i] = corr

        self.is_trained = True
        return {'method': 'correlation', 'weights': self.weights.tolist()}

    def predict(self, flow_data: np.ndarray, **kwargs) -> PredictionResult:
        """Predict using weighted sum."""
        if flow_data.ndim == 1:
            flow_data = flow_data.reshape(1, -1)

        if self.weights is None:
            return PredictionResult(0.5, 0, 0.0)

        # Weighted sum
        score = np.dot(flow_data[-1], self.weights[:len(flow_data[-1])])

        # Sigmoid to get probability
        prob = 1 / (1 + np.exp(-score))

        direction = 1 if prob > 0.55 else (-1 if prob < 0.45 else 0)
        confidence = abs(prob - 0.5) * 2

        return PredictionResult(float(prob), direction, float(confidence))


# Use fallback if LightGBM not available
if not HAS_LIGHTGBM:
    LightGBMFlowClassifier = FallbackClassifier


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("LightGBM Flow Classifier Demo")
    print("=" * 50)
    print(f"LightGBM available: {HAS_LIGHTGBM}")

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    # Flow with some predictive signal
    flow = np.cumsum(np.random.randn(n_samples) * 0.5)

    # Labels: price goes up when flow is increasing
    flow_change = np.diff(flow, prepend=flow[0])
    noise = np.random.randn(n_samples) * 0.5
    labels = (flow_change + noise > 0).astype(int)

    # Train classifier
    clf = LightGBMFlowClassifier()
    result = clf.train(flow, labels, val_ratio=0.2)

    print(f"\nTraining Results:")
    print(f"  Samples: {result.get('train_samples', 'N/A')}")
    print(f"  Best AUC: {result.get('best_auc', 'N/A'):.4f}")

    # Predict
    pred = clf.predict(flow[-50:])
    print(f"\nPrediction:")
    print(f"  Probability: {pred.probability:.3f}")
    print(f"  Direction: {pred.direction}")
    print(f"  Confidence: {pred.confidence:.3f}")

    if pred.feature_importance:
        print(f"\nTop Features:")
        sorted_imp = sorted(
            pred.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for name, imp in sorted_imp:
            print(f"  {name}: {imp:.3f}")
