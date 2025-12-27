"""
Online Learner Module
=====================

Ported from QLib's incremental learning patterns.

Provides mechanisms for:
1. Incremental parameter updates
2. Concept drift detection
3. Adaptive learning rate
4. Performance tracking

Formula IDs: 70010
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class LearningStats:
    """Statistics from online learning."""
    total_updates: int = 0
    correct_predictions: int = 0
    total_predictions: int = 0
    cumulative_loss: float = 0.0
    recent_accuracy: float = 0.5
    drift_detected: bool = False
    last_drift_time: Optional[float] = None


class OnlineLearner:
    """
    Generic online learning wrapper (Formula 70010).

    Wraps any predictor and adds:
    - Performance tracking
    - Concept drift detection
    - Adaptive learning rate
    - Automatic retraining triggers

    QLib Pattern: Track prediction performance and adapt.
    """

    formula_id = 70010
    name = "OnlineLearner"

    def __init__(self,
                 predictor: Any,
                 learning_rate: float = 0.01,
                 drift_window: int = 100,
                 drift_threshold: float = 0.1,
                 retrain_trigger: float = 0.45):
        """
        Initialize online learner.

        Args:
            predictor: Any object with predict() method
            learning_rate: Base learning rate
            drift_window: Window for drift detection
            drift_threshold: Threshold for drift detection
            retrain_trigger: Accuracy threshold to trigger retrain
        """
        self.predictor = predictor
        self.learning_rate = learning_rate
        self.drift_window = drift_window
        self.drift_threshold = drift_threshold
        self.retrain_trigger = retrain_trigger

        # Performance tracking
        self.recent_outcomes: deque = deque(maxlen=drift_window)
        self.recent_predictions: deque = deque(maxlen=drift_window)

        # Stats
        self.stats = LearningStats()

        # Callbacks
        self.on_drift_detected: Optional[Callable] = None
        self.on_retrain_triggered: Optional[Callable] = None

    def predict(self, *args, **kwargs) -> Any:
        """
        Make prediction using wrapped predictor.

        Passes through to predictor.predict() and tracks result.
        """
        result = self.predictor.predict(*args, **kwargs)

        # Store prediction for later validation
        self.recent_predictions.append({
            'time': time.time(),
            'result': result,
            'args': args,
            'kwargs': kwargs,
        })

        self.stats.total_predictions += 1
        return result

    def update(self, outcome: int, prediction_idx: int = -1):
        """
        Update with actual outcome.

        Args:
            outcome: Actual outcome (1 = up, 0 = down)
            prediction_idx: Which prediction to validate (-1 = most recent)
        """
        if not self.recent_predictions:
            return

        # Get the prediction we're validating
        pred_record = self.recent_predictions[prediction_idx]
        pred_result = pred_record['result']

        # Determine if prediction was correct
        if hasattr(pred_result, 'direction'):
            predicted_dir = pred_result.direction
        elif hasattr(pred_result, 'probability'):
            predicted_dir = 1 if pred_result.probability > 0.5 else -1
        else:
            predicted_dir = 1 if pred_result > 0.5 else -1

        actual_dir = 1 if outcome > 0 else -1
        correct = (predicted_dir == actual_dir) or (predicted_dir == 0)

        # Update stats
        self.stats.total_updates += 1
        if correct:
            self.stats.correct_predictions += 1

        # Track recent outcomes
        self.recent_outcomes.append({
            'time': time.time(),
            'correct': correct,
            'predicted': predicted_dir,
            'actual': actual_dir,
        })

        # Update recent accuracy
        if len(self.recent_outcomes) > 0:
            recent_correct = sum(1 for o in self.recent_outcomes if o['correct'])
            self.stats.recent_accuracy = recent_correct / len(self.recent_outcomes)

        # Check for concept drift
        self._check_drift()

        # Check if retrain needed
        self._check_retrain()

    def _check_drift(self):
        """
        Detect concept drift using ADWIN-style approach.

        Compare recent accuracy to historical accuracy.
        If significantly different, drift has occurred.
        """
        if len(self.recent_outcomes) < self.drift_window // 2:
            return

        # Split outcomes into two halves
        outcomes = list(self.recent_outcomes)
        mid = len(outcomes) // 2

        first_half = outcomes[:mid]
        second_half = outcomes[mid:]

        acc_first = sum(1 for o in first_half if o['correct']) / len(first_half)
        acc_second = sum(1 for o in second_half if o['correct']) / len(second_half)

        # Check for significant change
        if abs(acc_first - acc_second) > self.drift_threshold:
            self.stats.drift_detected = True
            self.stats.last_drift_time = time.time()

            if self.on_drift_detected:
                self.on_drift_detected({
                    'first_half_acc': acc_first,
                    'second_half_acc': acc_second,
                    'change': acc_second - acc_first,
                })

    def _check_retrain(self):
        """Check if model needs retraining based on performance."""
        if self.stats.recent_accuracy < self.retrain_trigger:
            if self.on_retrain_triggered:
                self.on_retrain_triggered({
                    'recent_accuracy': self.stats.recent_accuracy,
                    'threshold': self.retrain_trigger,
                })

    def get_adaptive_learning_rate(self) -> float:
        """
        Get learning rate adjusted by recent performance.

        Higher learning rate when accuracy is poor (need to adapt faster).
        Lower learning rate when accuracy is good (don't overfit).
        """
        # Scale learning rate inversely with accuracy
        acc = self.stats.recent_accuracy

        if acc > 0.6:
            # Good accuracy, slow down
            return self.learning_rate * 0.5
        elif acc < 0.45:
            # Poor accuracy, speed up
            return self.learning_rate * 2.0
        else:
            return self.learning_rate

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'total_updates': self.stats.total_updates,
            'total_predictions': self.stats.total_predictions,
            'correct_predictions': self.stats.correct_predictions,
            'overall_accuracy': (
                self.stats.correct_predictions / self.stats.total_predictions
                if self.stats.total_predictions > 0 else 0.5
            ),
            'recent_accuracy': self.stats.recent_accuracy,
            'drift_detected': self.stats.drift_detected,
            'last_drift_time': self.stats.last_drift_time,
            'adaptive_lr': self.get_adaptive_learning_rate(),
        }

    def reset_drift_flag(self):
        """Reset drift detection flag after handling."""
        self.stats.drift_detected = False


class IncrementalUpdater:
    """
    Incremental parameter updater.

    Updates model parameters using online gradient descent.
    Useful for simple linear models.
    """

    def __init__(self, n_features: int, learning_rate: float = 0.01):
        """
        Initialize updater.

        Args:
            n_features: Number of input features
            learning_rate: Base learning rate
        """
        self.n_features = n_features
        self.learning_rate = learning_rate

        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Momentum for faster convergence
        self.velocity_w = np.zeros(n_features)
        self.velocity_b = 0.0
        self.momentum = 0.9

        # Stats
        self.n_updates = 0

    def predict(self, x: np.ndarray) -> float:
        """
        Make prediction.

        Args:
            x: Feature vector

        Returns:
            Probability (sigmoid of linear combination)
        """
        linear = np.dot(x, self.weights) + self.bias
        return 1 / (1 + np.exp(-np.clip(linear, -500, 500)))

    def update(self, x: np.ndarray, y: int, learning_rate: Optional[float] = None):
        """
        Update weights with single sample.

        Args:
            x: Feature vector
            y: True label (0 or 1)
            learning_rate: Override learning rate (optional)
        """
        lr = learning_rate or self.learning_rate

        # Prediction
        pred = self.predict(x)

        # Gradient (binary cross-entropy)
        error = pred - y
        grad_w = error * x
        grad_b = error

        # Momentum update
        self.velocity_w = self.momentum * self.velocity_w - lr * grad_w
        self.velocity_b = self.momentum * self.velocity_b - lr * grad_b

        self.weights += self.velocity_w
        self.bias += self.velocity_b

        self.n_updates += 1

    def batch_update(self, X: np.ndarray, y: np.ndarray,
                     learning_rate: Optional[float] = None):
        """
        Update weights with batch of samples.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            learning_rate: Override learning rate (optional)
        """
        for i in range(len(X)):
            self.update(X[i], y[i], learning_rate)

    def get_weights(self) -> Dict[str, Any]:
        """Get current weights."""
        return {
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'n_updates': self.n_updates,
        }

    def set_weights(self, weights: np.ndarray, bias: float):
        """Set weights directly."""
        self.weights = np.array(weights)
        self.bias = bias


class EnsembleOnlineLearner:
    """
    Ensemble of online learners with dynamic weighting.

    Maintains multiple learners and weights their predictions
    based on recent performance.
    """

    def __init__(self, learners: List[OnlineLearner]):
        """
        Initialize ensemble.

        Args:
            learners: List of OnlineLearner instances
        """
        self.learners = learners
        self.n_learners = len(learners)

        # Performance-based weights
        self.weights = np.ones(self.n_learners) / self.n_learners

        # Track individual performance
        self.performance_history: List[deque] = [
            deque(maxlen=100) for _ in range(self.n_learners)
        ]

    def predict(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Ensemble prediction.

        Returns:
            Weighted combination of learner predictions
        """
        predictions = []
        for learner in self.learners:
            pred = learner.predict(*args, **kwargs)
            predictions.append(pred)

        # Combine predictions based on weights
        if hasattr(predictions[0], 'probability'):
            # Weighted average of probabilities
            weighted_prob = sum(
                self.weights[i] * pred.probability
                for i, pred in enumerate(predictions)
            )

            # Create combined result
            combined = type(predictions[0])(
                probability=weighted_prob,
                direction=1 if weighted_prob > 0.55 else (-1 if weighted_prob < 0.45 else 0),
                confidence=abs(weighted_prob - 0.5) * 2,
            )
        else:
            # Simple weighted average
            combined = sum(
                self.weights[i] * float(pred)
                for i, pred in enumerate(predictions)
            )

        meta = {
            'individual_predictions': predictions,
            'weights': self.weights.tolist(),
        }

        return combined, meta

    def update(self, outcome: int):
        """
        Update all learners and reweight.

        Args:
            outcome: Actual outcome
        """
        # Update each learner
        for i, learner in enumerate(self.learners):
            learner.update(outcome)

            # Track performance
            stats = learner.get_stats()
            self.performance_history[i].append(stats['recent_accuracy'])

        # Reweight based on recent performance
        self._update_weights()

    def _update_weights(self):
        """Update ensemble weights based on performance."""
        # Get recent accuracy for each learner
        accuracies = []
        for history in self.performance_history:
            if len(history) > 0:
                accuracies.append(np.mean(list(history)))
            else:
                accuracies.append(0.5)

        accuracies = np.array(accuracies)

        # Softmax weighting (better performers get more weight)
        exp_acc = np.exp((accuracies - 0.5) * 5)  # Scale for sensitivity
        self.weights = exp_acc / np.sum(exp_acc)

    def get_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        return {
            'n_learners': self.n_learners,
            'weights': self.weights.tolist(),
            'individual_stats': [l.get_stats() for l in self.learners],
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("Online Learner Demo")
    print("=" * 50)

    # Create a simple predictor
    class SimplePredictor:
        def predict(self, x):
            return 0.5 + x * 0.1  # Simple linear

    predictor = SimplePredictor()
    learner = OnlineLearner(
        predictor,
        drift_window=50,
        drift_threshold=0.15,
    )

    # Set up callbacks
    def on_drift(info):
        print(f"[DRIFT] Detected: {info}")

    learner.on_drift_detected = on_drift

    # Simulate predictions and updates
    np.random.seed(42)

    # First phase: predictor works well
    for i in range(50):
        x = np.random.randn()
        pred = learner.predict(x)
        outcome = 1 if x > 0 else 0  # True relationship
        learner.update(outcome)

    print(f"Phase 1 stats: {learner.get_stats()}")

    # Second phase: concept drift (relationship changes)
    for i in range(50):
        x = np.random.randn()
        pred = learner.predict(x)
        outcome = 1 if x < 0 else 0  # Reversed relationship!
        learner.update(outcome)

    print(f"Phase 2 stats: {learner.get_stats()}")

    # Test incremental updater
    print("\n" + "=" * 50)
    print("Incremental Updater Demo")

    updater = IncrementalUpdater(n_features=5)

    # Train on some data
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple rule

    updater.batch_update(X, y, learning_rate=0.1)

    print(f"Weights after training: {updater.get_weights()}")

    # Test prediction
    test_x = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    print(f"Prediction for [1,1,0,0,0]: {updater.predict(test_x):.3f}")
