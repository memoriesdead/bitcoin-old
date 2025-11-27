"""
Machine Learning Formulas (IDs 61-100)
======================================
Neural networks, ensemble methods, clustering, and RL-based signals.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from .base import BaseFormula, FormulaRegistry


# =============================================================================
# NEURAL NETWORK FORMULAS (61-70)
# =============================================================================

@FormulaRegistry.register(61)
class SimplePerceptron(BaseFormula):
    """ID 61: Single Layer Perceptron - Basic neural unit"""

    CATEGORY = "machine_learning"
    NAME = "SimplePerceptron"
    DESCRIPTION = "Single layer perceptron for linear classification"

    def __init__(self, lookback: int = 100, n_features: int = 5, learning_rate: float = 0.01, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.weights = np.random.randn(n_features) * 0.1
        self.bias = 0.0
        self.features = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < self.n_features + 1:
            return
        returns = self._returns_array()
        features = returns[-self.n_features:]
        output = np.dot(features, self.weights) + self.bias
        activation = self._sigmoid(output)
        if len(returns) > self.n_features:
            target = 1.0 if returns[-1] > 0 else 0.0
            error = target - activation
            self.weights += self.learning_rate * error * features
            self.bias += self.learning_rate * error
        self.signal = 1 if activation > 0.55 else (-1 if activation < 0.45 else 0)
        self.confidence = abs(activation - 0.5) * 2


@FormulaRegistry.register(62)
class MultiLayerPerceptron(BaseFormula):
    """ID 62: Multi-Layer Perceptron (2 hidden layers)"""

    CATEGORY = "machine_learning"
    NAME = "MultiLayerPerceptron"
    DESCRIPTION = "Two hidden layer neural network"

    def __init__(self, lookback: int = 100, n_features: int = 10,
                 hidden1: int = 8, hidden2: int = 4, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_features = n_features
        self.w1 = np.random.randn(n_features, hidden1) * 0.1
        self.b1 = np.zeros(hidden1)
        self.w2 = np.random.randn(hidden1, hidden2) * 0.1
        self.b2 = np.zeros(hidden2)
        self.w3 = np.random.randn(hidden2, 1) * 0.1
        self.b3 = 0.0

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _compute(self) -> None:
        if len(self.returns) < self.n_features:
            return
        returns = self._returns_array()
        x = returns[-self.n_features:]
        h1 = self._relu(np.dot(x, self.w1) + self.b1)
        h2 = self._relu(np.dot(h1, self.w2) + self.b2)
        output = self._sigmoid(np.dot(h2, self.w3) + self.b3)
        self.signal = 1 if output > 0.55 else (-1 if output < 0.45 else 0)
        self.confidence = min(abs(output - 0.5) * 2, 1.0)


@FormulaRegistry.register(63)
class RecurrentUnit(BaseFormula):
    """ID 63: Simple Recurrent Neural Network Unit"""

    CATEGORY = "machine_learning"
    NAME = "RecurrentUnit"
    DESCRIPTION = "Basic RNN with hidden state"

    def __init__(self, lookback: int = 100, hidden_size: int = 10, **kwargs):
        super().__init__(lookback, **kwargs)
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(1, hidden_size) * 0.1
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Why = np.random.randn(hidden_size, 1) * 0.1
        self.bh = np.zeros(hidden_size)
        self.by = 0.0
        self.h = np.zeros(hidden_size)

    def _compute(self) -> None:
        if len(self.returns) < 1:
            return
        x = np.array([[self.returns[-1]]])
        self.h = np.tanh(np.dot(x, self.Wxh) + np.dot(self.h, self.Whh) + self.bh).flatten()
        y = self._sigmoid(np.dot(self.h, self.Why) + self.by)
        self.signal = 1 if y > 0.55 else (-1 if y < 0.45 else 0)
        self.confidence = min(abs(y - 0.5) * 2, 1.0)


@FormulaRegistry.register(64)
class LSTMUnit(BaseFormula):
    """ID 64: Long Short-Term Memory Unit"""

    CATEGORY = "machine_learning"
    NAME = "LSTMUnit"
    DESCRIPTION = "LSTM cell with forget, input, output gates"

    def __init__(self, lookback: int = 100, hidden_size: int = 10, **kwargs):
        super().__init__(lookback, **kwargs)
        self.hidden_size = hidden_size
        input_size = 1
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        self.Wy = np.random.randn(hidden_size, 1) * 0.1
        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.by = 0.0
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)

    def _compute(self) -> None:
        if len(self.returns) < 1:
            return
        x = np.array([self.returns[-1]])
        combined = np.concatenate([x, self.h])
        f = self._sigmoid(np.dot(combined, self.Wf) + self.bf)
        i = self._sigmoid(np.dot(combined, self.Wi) + self.bi)
        c_tilde = np.tanh(np.dot(combined, self.Wc) + self.bc)
        self.c = f * self.c + i * c_tilde
        o = self._sigmoid(np.dot(combined, self.Wo) + self.bo)
        self.h = o * np.tanh(self.c)
        y = self._sigmoid(np.dot(self.h, self.Wy) + self.by)
        self.signal = 1 if y > 0.55 else (-1 if y < 0.45 else 0)
        self.confidence = min(abs(y - 0.5) * 2, 1.0)


@FormulaRegistry.register(65)
class GRUUnit(BaseFormula):
    """ID 65: Gated Recurrent Unit"""

    CATEGORY = "machine_learning"
    NAME = "GRUUnit"
    DESCRIPTION = "GRU with reset and update gates"

    def __init__(self, lookback: int = 100, hidden_size: int = 10, **kwargs):
        super().__init__(lookback, **kwargs)
        self.hidden_size = hidden_size
        input_size = 1
        self.Wz = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        self.Wr = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        self.Wh = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        self.Wy = np.random.randn(hidden_size, 1) * 0.1
        self.bz = np.zeros(hidden_size)
        self.br = np.zeros(hidden_size)
        self.bh = np.zeros(hidden_size)
        self.by = 0.0
        self.h = np.zeros(hidden_size)

    def _compute(self) -> None:
        if len(self.returns) < 1:
            return
        x = np.array([self.returns[-1]])
        combined = np.concatenate([x, self.h])
        z = self._sigmoid(np.dot(combined, self.Wz) + self.bz)
        r = self._sigmoid(np.dot(combined, self.Wr) + self.br)
        combined_r = np.concatenate([x, r * self.h])
        h_tilde = np.tanh(np.dot(combined_r, self.Wh) + self.bh)
        self.h = (1 - z) * self.h + z * h_tilde
        y = self._sigmoid(np.dot(self.h, self.Wy) + self.by)
        self.signal = 1 if y > 0.55 else (-1 if y < 0.45 else 0)
        self.confidence = min(abs(y - 0.5) * 2, 1.0)


@FormulaRegistry.register(66)
class AttentionMechanism(BaseFormula):
    """ID 66: Self-Attention for sequence weighting"""

    CATEGORY = "machine_learning"
    NAME = "AttentionMechanism"
    DESCRIPTION = "Self-attention mechanism for temporal data"

    def __init__(self, lookback: int = 100, seq_len: int = 20, d_model: int = 8, **kwargs):
        super().__init__(lookback, **kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.Wq = np.random.randn(1, d_model) * 0.1
        self.Wk = np.random.randn(1, d_model) * 0.1
        self.Wv = np.random.randn(1, d_model) * 0.1
        self.Wo = np.random.randn(d_model, 1) * 0.1

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)

    def _compute(self) -> None:
        if len(self.returns) < self.seq_len:
            return
        returns = self._returns_array()[-self.seq_len:]
        seq = returns.reshape(-1, 1)
        Q = np.dot(seq, self.Wq)
        K = np.dot(seq, self.Wk)
        V = np.dot(seq, self.Wv)
        scores = np.dot(Q, K.T) / np.sqrt(self.d_model)
        attention_weights = np.array([self._softmax(s) for s in scores])
        context = np.dot(attention_weights, V)
        output = self._sigmoid(np.dot(context[-1], self.Wo))
        self.signal = 1 if output > 0.55 else (-1 if output < 0.45 else 0)
        self.confidence = min(abs(output - 0.5) * 2, 1.0)


@FormulaRegistry.register(67)
class ConvolutionalFeature(BaseFormula):
    """ID 67: 1D Convolution for pattern extraction"""

    CATEGORY = "machine_learning"
    NAME = "ConvolutionalFeature"
    DESCRIPTION = "1D CNN for temporal pattern detection"

    def __init__(self, lookback: int = 100, kernel_size: int = 5, n_filters: int = 4, **kwargs):
        super().__init__(lookback, **kwargs)
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.kernels = np.random.randn(n_filters, kernel_size) * 0.1
        self.fc_weights = np.random.randn(n_filters, 1) * 0.1

    def _conv1d(self, x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        k_size = len(kernel)
        out_size = len(x) - k_size + 1
        result = np.zeros(out_size)
        for i in range(out_size):
            result[i] = np.sum(x[i:i+k_size] * kernel)
        return result

    def _compute(self) -> None:
        if len(self.returns) < self.kernel_size + 5:
            return
        returns = self._returns_array()
        conv_outputs = []
        for kernel in self.kernels:
            conv = self._conv1d(returns, kernel)
            pooled = np.max(conv) if len(conv) > 0 else 0
            conv_outputs.append(pooled)
        features = np.array(conv_outputs)
        output = self._sigmoid(np.dot(features, self.fc_weights))
        self.signal = 1 if output > 0.55 else (-1 if output < 0.45 else 0)
        self.confidence = min(abs(output - 0.5) * 2, 1.0)


@FormulaRegistry.register(68)
class AutoencoderAnomaly(BaseFormula):
    """ID 68: Autoencoder for anomaly detection"""

    CATEGORY = "machine_learning"
    NAME = "AutoencoderAnomaly"
    DESCRIPTION = "Detect anomalies via reconstruction error"

    def __init__(self, lookback: int = 100, input_size: int = 10,
                 latent_size: int = 3, **kwargs):
        super().__init__(lookback, **kwargs)
        self.input_size = input_size
        self.latent_size = latent_size
        self.W_enc = np.random.randn(input_size, latent_size) * 0.1
        self.b_enc = np.zeros(latent_size)
        self.W_dec = np.random.randn(latent_size, input_size) * 0.1
        self.b_dec = np.zeros(input_size)
        self.reconstruction_errors = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < self.input_size:
            return
        returns = self._returns_array()
        x = returns[-self.input_size:]
        latent = np.tanh(np.dot(x, self.W_enc) + self.b_enc)
        reconstructed = np.dot(latent, self.W_dec) + self.b_dec
        recon_error = np.mean((x - reconstructed) ** 2)
        self.reconstruction_errors.append(recon_error)
        if len(self.reconstruction_errors) < 10:
            return
        errors = np.array(self.reconstruction_errors)
        mean_err = np.mean(errors)
        std_err = np.std(errors) + 1e-10
        z_score = (recon_error - mean_err) / std_err
        if z_score > 2.0:
            self.signal = -1
            self.confidence = min(z_score / 4.0, 1.0)
        elif z_score < -1.0:
            self.signal = 1
            self.confidence = min(abs(z_score) / 3.0, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(69)
class VariationalAutoencoder(BaseFormula):
    """ID 69: VAE for probabilistic latent space"""

    CATEGORY = "machine_learning"
    NAME = "VariationalAutoencoder"
    DESCRIPTION = "VAE with reparameterization trick"

    def __init__(self, lookback: int = 100, input_size: int = 10,
                 latent_size: int = 3, **kwargs):
        super().__init__(lookback, **kwargs)
        self.input_size = input_size
        self.latent_size = latent_size
        self.W_mu = np.random.randn(input_size, latent_size) * 0.1
        self.W_logvar = np.random.randn(input_size, latent_size) * 0.1
        self.W_dec = np.random.randn(latent_size, input_size) * 0.1
        self.latent_means = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < self.input_size:
            return
        returns = self._returns_array()
        x = returns[-self.input_size:]
        mu = np.dot(x, self.W_mu)
        logvar = np.dot(x, self.W_logvar)
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        z = mu + eps * std
        self.latent_means.append(np.mean(mu))
        if len(self.latent_means) < 10:
            return
        recent_latent = np.array(self.latent_means)
        trend = recent_latent[-1] - np.mean(recent_latent[-10:])
        if trend > 0.1:
            self.signal = 1
            self.confidence = min(abs(trend), 1.0)
        elif trend < -0.1:
            self.signal = -1
            self.confidence = min(abs(trend), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(70)
class ResidualBlock(BaseFormula):
    """ID 70: Residual connection for deep networks"""

    CATEGORY = "machine_learning"
    NAME = "ResidualBlock"
    DESCRIPTION = "Skip connection architecture"

    def __init__(self, lookback: int = 100, n_features: int = 10, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_features = n_features
        self.W1 = np.random.randn(n_features, n_features) * 0.1
        self.W2 = np.random.randn(n_features, n_features) * 0.1
        self.Wo = np.random.randn(n_features, 1) * 0.1

    def _compute(self) -> None:
        if len(self.returns) < self.n_features:
            return
        returns = self._returns_array()
        x = returns[-self.n_features:]
        h1 = np.maximum(0, np.dot(x, self.W1))
        h2 = np.maximum(0, np.dot(h1, self.W2))
        residual = h2 + x
        output = self._sigmoid(np.dot(residual, self.Wo))
        self.signal = 1 if output > 0.55 else (-1 if output < 0.45 else 0)
        self.confidence = min(abs(output - 0.5) * 2, 1.0)


# =============================================================================
# ENSEMBLE METHODS (71-80)
# =============================================================================

@FormulaRegistry.register(71)
class RandomForestSignal(BaseFormula):
    """ID 71: Random Forest ensemble signal"""

    CATEGORY = "machine_learning"
    NAME = "RandomForestSignal"
    DESCRIPTION = "Ensemble of decision stumps"

    def __init__(self, lookback: int = 100, n_trees: int = 10, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_trees = n_trees
        self.tree_thresholds = np.random.randn(n_trees) * 0.01
        self.tree_features = np.random.randint(0, 5, n_trees)

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        features = [
            returns[-1],
            np.mean(returns[-5:]) if len(returns) >= 5 else 0,
            np.std(returns[-5:]) if len(returns) >= 5 else 0,
            returns[-1] - returns[-2] if len(returns) >= 2 else 0,
            np.sum(returns[-5:] > 0) / 5 if len(returns) >= 5 else 0.5,
        ]
        votes = 0
        for i in range(self.n_trees):
            feat_idx = self.tree_features[i] % len(features)
            if features[feat_idx] > self.tree_thresholds[i]:
                votes += 1
            else:
                votes -= 1
        vote_ratio = votes / self.n_trees
        self.signal = 1 if vote_ratio > 0.2 else (-1 if vote_ratio < -0.2 else 0)
        self.confidence = abs(vote_ratio)


@FormulaRegistry.register(72)
class GradientBoostingSignal(BaseFormula):
    """ID 72: Gradient Boosting ensemble"""

    CATEGORY = "machine_learning"
    NAME = "GradientBoostingSignal"
    DESCRIPTION = "Sequential weak learner boosting"

    def __init__(self, lookback: int = 100, n_estimators: int = 5,
                 learning_rate: float = 0.1, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.weights = [np.random.randn(5) * 0.1 for _ in range(n_estimators)]
        self.biases = [0.0 for _ in range(n_estimators)]

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        features = np.array([
            returns[-1],
            np.mean(returns[-5:]),
            np.std(returns[-5:]),
            np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns),
            returns[-1] - returns[-5] if len(returns) >= 5 else 0,
        ])
        prediction = 0.0
        for i in range(self.n_estimators):
            weak_pred = np.tanh(np.dot(features, self.weights[i]) + self.biases[i])
            prediction += self.lr * weak_pred
        output = self._sigmoid(prediction)
        self.signal = 1 if output > 0.55 else (-1 if output < 0.45 else 0)
        self.confidence = min(abs(output - 0.5) * 2, 1.0)


@FormulaRegistry.register(73)
class AdaBoostSignal(BaseFormula):
    """ID 73: AdaBoost adaptive boosting"""

    CATEGORY = "machine_learning"
    NAME = "AdaBoostSignal"
    DESCRIPTION = "Adaptive boosting with sample weights"

    def __init__(self, lookback: int = 100, n_estimators: int = 5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_estimators = n_estimators
        self.alphas = np.ones(n_estimators) / n_estimators
        self.thresholds = np.random.randn(n_estimators) * 0.005

    def _compute(self) -> None:
        if len(self.returns) < 5:
            return
        returns = self._returns_array()
        recent = returns[-5:]
        weighted_vote = 0.0
        for i in range(self.n_estimators):
            feature = recent[i % len(recent)]
            vote = 1 if feature > self.thresholds[i] else -1
            weighted_vote += self.alphas[i] * vote
        self.signal = 1 if weighted_vote > 0.1 else (-1 if weighted_vote < -0.1 else 0)
        self.confidence = min(abs(weighted_vote), 1.0)


@FormulaRegistry.register(74)
class BaggingSignal(BaseFormula):
    """ID 74: Bootstrap Aggregating"""

    CATEGORY = "machine_learning"
    NAME = "BaggingSignal"
    DESCRIPTION = "Bootstrap sampling ensemble"

    def __init__(self, lookback: int = 100, n_bags: int = 10, bag_size: int = 20, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_bags = n_bags
        self.bag_size = bag_size
        self.bag_weights = [np.random.randn(3) * 0.1 for _ in range(n_bags)]

    def _compute(self) -> None:
        if len(self.returns) < self.bag_size:
            return
        returns = self._returns_array()
        predictions = []
        for i in range(self.n_bags):
            indices = np.random.choice(len(returns), min(self.bag_size, len(returns)), replace=True)
            sample = returns[indices]
            features = np.array([np.mean(sample), np.std(sample), sample[-1]])
            pred = np.tanh(np.dot(features, self.bag_weights[i]))
            predictions.append(pred)
        avg_pred = np.mean(predictions)
        self.signal = 1 if avg_pred > 0.1 else (-1 if avg_pred < -0.1 else 0)
        self.confidence = min(abs(avg_pred), 1.0)


@FormulaRegistry.register(75)
class StackingEnsemble(BaseFormula):
    """ID 75: Stacked generalization"""

    CATEGORY = "machine_learning"
    NAME = "StackingEnsemble"
    DESCRIPTION = "Meta-learner on base learner outputs"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.base_weights = [np.random.randn(5) * 0.1 for _ in range(3)]
        self.meta_weights = np.random.randn(3) * 0.1

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        features = np.array([
            returns[-1],
            np.mean(returns[-5:]),
            np.std(returns[-5:]),
            np.mean(returns[-10:]),
            returns[-1] - returns[-2] if len(returns) >= 2 else 0,
        ])
        base_outputs = []
        for w in self.base_weights:
            out = np.tanh(np.dot(features, w))
            base_outputs.append(out)
        meta_input = np.array(base_outputs)
        meta_output = self._sigmoid(np.dot(meta_input, self.meta_weights))
        self.signal = 1 if meta_output > 0.55 else (-1 if meta_output < 0.45 else 0)
        self.confidence = min(abs(meta_output - 0.5) * 2, 1.0)


@FormulaRegistry.register(76)
class VotingClassifier(BaseFormula):
    """ID 76: Majority voting ensemble"""

    CATEGORY = "machine_learning"
    NAME = "VotingClassifier"
    DESCRIPTION = "Hard/soft voting combination"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.classifier_thresholds = [0.0, 0.001, -0.001, 0.002, -0.002]

    def _compute(self) -> None:
        if len(self.returns) < 5:
            return
        returns = self._returns_array()
        recent = returns[-1]
        votes = []
        for thresh in self.classifier_thresholds:
            vote = 1 if recent > thresh else -1
            votes.append(vote)
        vote_sum = sum(votes)
        self.signal = 1 if vote_sum >= 3 else (-1 if vote_sum <= -3 else 0)
        self.confidence = abs(vote_sum) / len(votes)


@FormulaRegistry.register(77)
class XGBoostLike(BaseFormula):
    """ID 77: XGBoost-style gradient boosting"""

    CATEGORY = "machine_learning"
    NAME = "XGBoostLike"
    DESCRIPTION = "Regularized gradient boosting"

    def __init__(self, lookback: int = 100, n_rounds: int = 5,
                 reg_lambda: float = 1.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_rounds = n_rounds
        self.reg_lambda = reg_lambda
        self.tree_weights = [np.random.randn(5) * 0.1 for _ in range(n_rounds)]

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        features = np.array([
            returns[-1],
            np.mean(returns[-3:]),
            np.mean(returns[-5:]),
            np.std(returns[-5:]),
            np.mean(returns[-10:]),
        ])
        prediction = 0.0
        for i, w in enumerate(self.tree_weights):
            reg_w = w / (1 + self.reg_lambda)
            pred = np.tanh(np.dot(features, reg_w))
            prediction += pred * (0.3 ** i)
        output = self._sigmoid(prediction)
        self.signal = 1 if output > 0.55 else (-1 if output < 0.45 else 0)
        self.confidence = min(abs(output - 0.5) * 2, 1.0)


@FormulaRegistry.register(78)
class LightGBMLike(BaseFormula):
    """ID 78: LightGBM-style leaf-wise growth"""

    CATEGORY = "machine_learning"
    NAME = "LightGBMLike"
    DESCRIPTION = "Histogram-based gradient boosting"

    def __init__(self, lookback: int = 100, n_leaves: int = 8, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_leaves = n_leaves
        self.leaf_values = np.random.randn(n_leaves) * 0.1
        self.leaf_thresholds = np.sort(np.random.randn(n_leaves - 1) * 0.01)

    def _compute(self) -> None:
        if len(self.returns) < 5:
            return
        returns = self._returns_array()
        feature = returns[-1]
        leaf_idx = np.searchsorted(self.leaf_thresholds, feature)
        leaf_idx = min(leaf_idx, len(self.leaf_values) - 1)
        output = self._sigmoid(self.leaf_values[leaf_idx])
        self.signal = 1 if output > 0.55 else (-1 if output < 0.45 else 0)
        self.confidence = min(abs(output - 0.5) * 2, 1.0)


@FormulaRegistry.register(79)
class CatBoostLike(BaseFormula):
    """ID 79: CatBoost-style ordered boosting"""

    CATEGORY = "machine_learning"
    NAME = "CatBoostLike"
    DESCRIPTION = "Ordered boosting with categorical handling"

    def __init__(self, lookback: int = 100, n_trees: int = 5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_trees = n_trees
        self.tree_weights = [np.random.randn(4) * 0.1 for _ in range(n_trees)]

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        cat_feature = 0 if returns[-1] < 0 else 1
        num_features = np.array([
            returns[-1],
            np.mean(returns[-5:]),
            np.std(returns[-5:]),
        ])
        features = np.concatenate([[cat_feature], num_features])
        prediction = 0.0
        for w in self.tree_weights:
            pred = np.tanh(np.dot(features, w))
            prediction += pred
        prediction /= self.n_trees
        output = self._sigmoid(prediction)
        self.signal = 1 if output > 0.55 else (-1 if output < 0.45 else 0)
        self.confidence = min(abs(output - 0.5) * 2, 1.0)


@FormulaRegistry.register(80)
class NaiveBayesClassifier(BaseFormula):
    """ID 80: Naive Bayes probabilistic classifier"""

    CATEGORY = "machine_learning"
    NAME = "NaiveBayesClassifier"
    DESCRIPTION = "Bayes theorem with feature independence"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.up_mean = 0.001
        self.up_std = 0.01
        self.down_mean = -0.001
        self.down_std = 0.01
        self.prior_up = 0.5
        self.prior_down = 0.5

    def _gaussian_pdf(self, x: float, mean: float, std: float) -> float:
        if std <= 0:
            return 1.0
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        self.up_mean = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0.001
        self.up_std = np.std(returns[returns > 0]) if np.any(returns > 0) else 0.01
        self.down_mean = np.mean(returns[returns < 0]) if np.any(returns < 0) else -0.001
        self.down_std = np.std(returns[returns < 0]) if np.any(returns < 0) else 0.01
        feature = returns[-1]
        p_up = self._gaussian_pdf(feature, self.up_mean, self.up_std) * self.prior_up
        p_down = self._gaussian_pdf(feature, self.down_mean, self.down_std) * self.prior_down
        total = p_up + p_down + 1e-10
        prob_up = p_up / total
        self.signal = 1 if prob_up > 0.55 else (-1 if prob_up < 0.45 else 0)
        self.confidence = abs(prob_up - 0.5) * 2


# =============================================================================
# CLUSTERING AND DIMENSIONALITY (81-90)
# =============================================================================

@FormulaRegistry.register(81)
class KMeansRegime(BaseFormula):
    """ID 81: K-Means clustering for regime detection"""

    CATEGORY = "machine_learning"
    NAME = "KMeansRegime"
    DESCRIPTION = "Cluster market states with K-Means"

    def __init__(self, lookback: int = 100, n_clusters: int = 3, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_clusters = n_clusters
        self.centroids = np.random.randn(n_clusters, 2) * 0.01

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        features = np.array([returns[-1], np.std(returns[-5:])])
        distances = [np.linalg.norm(features - c) for c in self.centroids]
        cluster = np.argmin(distances)
        if cluster == 0:
            self.signal = 1
            self.confidence = 0.6
        elif cluster == 1:
            self.signal = -1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(82)
class DBSCANAnomaly(BaseFormula):
    """ID 82: DBSCAN density-based clustering"""

    CATEGORY = "machine_learning"
    NAME = "DBSCANAnomaly"
    DESCRIPTION = "Identify noise points as anomalies"

    def __init__(self, lookback: int = 100, eps: float = 0.01, min_pts: int = 3, **kwargs):
        super().__init__(lookback, **kwargs)
        self.eps = eps
        self.min_pts = min_pts

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        current = returns[-1]
        neighbors = np.sum(np.abs(returns[:-1] - current) < self.eps)
        is_core = neighbors >= self.min_pts
        if not is_core:
            if current > 0:
                self.signal = -1
            else:
                self.signal = 1
            self.confidence = 0.7
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(83)
class HierarchicalCluster(BaseFormula):
    """ID 83: Agglomerative hierarchical clustering"""

    CATEGORY = "machine_learning"
    NAME = "HierarchicalCluster"
    DESCRIPTION = "Bottom-up clustering hierarchy"

    def __init__(self, lookback: int = 100, n_clusters: int = 3, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_clusters = n_clusters
        self.cluster_means = np.array([-0.01, 0.0, 0.01])

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        current = returns[-1]
        distances = np.abs(self.cluster_means - current)
        cluster = np.argmin(distances)
        if cluster == 2:
            self.signal = 1
            self.confidence = 0.6
        elif cluster == 0:
            self.signal = -1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(84)
class GaussianMixture(BaseFormula):
    """ID 84: Gaussian Mixture Model"""

    CATEGORY = "machine_learning"
    NAME = "GaussianMixture"
    DESCRIPTION = "Soft clustering with GMM"

    def __init__(self, lookback: int = 100, n_components: int = 3, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_components = n_components
        self.means = np.array([-0.005, 0.0, 0.005])
        self.stds = np.array([0.01, 0.005, 0.01])
        self.weights = np.array([0.3, 0.4, 0.3])

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        x = returns[-1]
        responsibilities = []
        for i in range(self.n_components):
            pdf = np.exp(-0.5 * ((x - self.means[i]) / self.stds[i]) ** 2)
            pdf /= self.stds[i] * np.sqrt(2 * np.pi)
            responsibilities.append(self.weights[i] * pdf)
        total = sum(responsibilities) + 1e-10
        probs = [r / total for r in responsibilities]
        component = np.argmax(probs)
        if component == 2:
            self.signal = 1
            self.confidence = probs[2]
        elif component == 0:
            self.signal = -1
            self.confidence = probs[0]
        else:
            self.signal = 0
            self.confidence = probs[1]


@FormulaRegistry.register(85)
class PCAFeatures(BaseFormula):
    """ID 85: Principal Component Analysis"""

    CATEGORY = "machine_learning"
    NAME = "PCAFeatures"
    DESCRIPTION = "Dimensionality reduction via PCA"

    def __init__(self, lookback: int = 100, n_components: int = 2, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_components = n_components
        self.components = None

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        X = np.column_stack([
            returns,
            np.roll(returns, 1),
            np.roll(returns, 2),
        ])[2:]
        if len(X) < 10:
            return
        X_centered = X - X.mean(axis=0)
        cov = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idx[:self.n_components]]
        projected = np.dot(X_centered[-1], self.components)
        score = projected[0] if len(projected) > 0 else 0
        self.signal = 1 if score > 0.005 else (-1 if score < -0.005 else 0)
        self.confidence = min(abs(score) * 50, 1.0)


@FormulaRegistry.register(86)
class TSNEVisualization(BaseFormula):
    """ID 86: t-SNE manifold learning (simplified)"""

    CATEGORY = "machine_learning"
    NAME = "TSNEVisualization"
    DESCRIPTION = "Neighbor embedding for pattern detection"

    def __init__(self, lookback: int = 100, perplexity: float = 5.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.perplexity = perplexity
        self.embedding = None

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        local_pattern = returns[-5:]
        historical_patterns = []
        for i in range(5, len(returns) - 5):
            pattern = returns[i-5:i]
            subsequent = returns[i:i+3]
            historical_patterns.append((pattern, np.mean(subsequent)))
        if len(historical_patterns) < 5:
            return
        distances = [np.linalg.norm(local_pattern - p[0]) for p in historical_patterns]
        nearest_idx = np.argsort(distances)[:3]
        expected_return = np.mean([historical_patterns[i][1] for i in nearest_idx])
        self.signal = 1 if expected_return > 0.001 else (-1 if expected_return < -0.001 else 0)
        self.confidence = min(abs(expected_return) * 100, 1.0)


@FormulaRegistry.register(87)
class UMAPEmbedding(BaseFormula):
    """ID 87: UMAP dimensionality reduction (simplified)"""

    CATEGORY = "machine_learning"
    NAME = "UMAPEmbedding"
    DESCRIPTION = "Uniform manifold approximation"

    def __init__(self, lookback: int = 100, n_neighbors: int = 5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_neighbors = n_neighbors

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        current_features = np.array([returns[-1], np.std(returns[-5:])])
        neighbor_features = []
        for i in range(10, len(returns) - 1):
            feat = np.array([returns[i], np.std(returns[i-5:i])])
            dist = np.linalg.norm(feat - current_features)
            neighbor_features.append((dist, returns[i+1] if i+1 < len(returns) else 0))
        neighbor_features.sort(key=lambda x: x[0])
        nearest = neighbor_features[:self.n_neighbors]
        predicted = np.mean([n[1] for n in nearest])
        self.signal = 1 if predicted > 0.001 else (-1 if predicted < -0.001 else 0)
        self.confidence = min(abs(predicted) * 100, 1.0)


@FormulaRegistry.register(88)
class IsolationForest(BaseFormula):
    """ID 88: Isolation Forest anomaly detection"""

    CATEGORY = "machine_learning"
    NAME = "IsolationForest"
    DESCRIPTION = "Tree-based outlier detection"

    def __init__(self, lookback: int = 100, n_trees: int = 10, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_trees = n_trees
        self.split_points = [np.random.randn(5) * 0.01 for _ in range(n_trees)]

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        current = returns[-1]
        path_lengths = []
        for splits in self.split_points:
            path = 0
            for split in splits:
                if current < split:
                    path += 1
                    current = current * 0.9
                else:
                    current = current * 1.1
                path += 1
            path_lengths.append(path)
        avg_path = np.mean(path_lengths)
        expected_path = np.log2(len(returns)) + 0.5772
        anomaly_score = 2 ** (-avg_path / expected_path)
        if anomaly_score > 0.6:
            self.signal = -1 if returns[-1] > 0 else 1
            self.confidence = anomaly_score
        else:
            self.signal = 0
            self.confidence = 1 - anomaly_score


@FormulaRegistry.register(89)
class LOFAnomaly(BaseFormula):
    """ID 89: Local Outlier Factor"""

    CATEGORY = "machine_learning"
    NAME = "LOFAnomaly"
    DESCRIPTION = "Density-based local outlier detection"

    def __init__(self, lookback: int = 100, k: int = 5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.k = k

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        current = returns[-1]
        distances = np.sort(np.abs(returns[:-1] - current))
        k_distance = distances[min(self.k, len(distances)-1)] if len(distances) > 0 else 0.01
        local_density = 1.0 / (k_distance + 1e-10)
        neighbor_densities = []
        for r in returns[:-1]:
            r_distances = np.sort(np.abs(returns - r))
            r_k_dist = r_distances[min(self.k, len(r_distances)-1)] if len(r_distances) > 0 else 0.01
            neighbor_densities.append(1.0 / (r_k_dist + 1e-10))
        avg_neighbor_density = np.mean(neighbor_densities) if neighbor_densities else local_density
        lof = avg_neighbor_density / (local_density + 1e-10)
        if lof > 1.5:
            self.signal = -1 if current > 0 else 1
            self.confidence = min(lof / 3.0, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(90)
class OneClassSVM(BaseFormula):
    """ID 90: One-Class SVM for novelty detection"""

    CATEGORY = "machine_learning"
    NAME = "OneClassSVM"
    DESCRIPTION = "SVM boundary for normal data"

    def __init__(self, lookback: int = 100, nu: float = 0.1, **kwargs):
        super().__init__(lookback, **kwargs)
        self.nu = nu
        self.support_vectors = None
        self.threshold = 0.0

    def _rbf_kernel(self, x1: float, x2: float, gamma: float = 10.0) -> float:
        return np.exp(-gamma * (x1 - x2) ** 2)

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        if self.support_vectors is None:
            sorted_returns = np.sort(returns)
            n_sv = max(3, int(len(sorted_returns) * self.nu))
            self.support_vectors = sorted_returns[-n_sv:]
            self.threshold = np.mean(self.support_vectors)
        current = returns[-1]
        score = sum(self._rbf_kernel(current, sv) for sv in self.support_vectors)
        score /= len(self.support_vectors)
        if score < 0.3:
            self.signal = -1 if current > 0 else 1
            self.confidence = 1 - score
        else:
            self.signal = 0
            self.confidence = score


# =============================================================================
# REINFORCEMENT LEARNING (91-100)
# =============================================================================

@FormulaRegistry.register(91)
class QLearningAgent(BaseFormula):
    """ID 91: Q-Learning for trading decisions"""

    CATEGORY = "machine_learning"
    NAME = "QLearningAgent"
    DESCRIPTION = "Tabular Q-learning agent"

    def __init__(self, lookback: int = 100, n_states: int = 10,
                 learning_rate: float = 0.1, gamma: float = 0.95, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_states = n_states
        self.lr = learning_rate
        self.gamma = gamma
        self.Q = np.zeros((n_states, 3))
        self.last_state = 0
        self.last_action = 1

    def _get_state(self, returns: np.ndarray) -> int:
        recent = returns[-1] if len(returns) > 0 else 0
        state = int((recent + 0.05) / 0.01)
        return max(0, min(self.n_states - 1, state))

    def _compute(self) -> None:
        if len(self.returns) < 5:
            return
        returns = self._returns_array()
        current_state = self._get_state(returns)
        reward = returns[-1] * 100
        best_next = np.max(self.Q[current_state])
        self.Q[self.last_state, self.last_action] += self.lr * (
            reward + self.gamma * best_next - self.Q[self.last_state, self.last_action]
        )
        action = np.argmax(self.Q[current_state])
        self.last_state = current_state
        self.last_action = action
        self.signal = action - 1
        q_values = self.Q[current_state]
        self.confidence = abs(q_values[action] - np.mean(q_values)) / (np.std(q_values) + 1e-10)
        self.confidence = min(max(self.confidence, 0), 1)


@FormulaRegistry.register(92)
class SARSAAgent(BaseFormula):
    """ID 92: SARSA on-policy learning"""

    CATEGORY = "machine_learning"
    NAME = "SARSAAgent"
    DESCRIPTION = "State-Action-Reward-State-Action"

    def __init__(self, lookback: int = 100, n_states: int = 10,
                 learning_rate: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 0.1, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_states = n_states
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, 3))
        self.last_state = 0
        self.last_action = 1

    def _get_state(self, returns: np.ndarray) -> int:
        recent = returns[-1] if len(returns) > 0 else 0
        state = int((recent + 0.05) / 0.01)
        return max(0, min(self.n_states - 1, state))

    def _compute(self) -> None:
        if len(self.returns) < 5:
            return
        returns = self._returns_array()
        current_state = self._get_state(returns)
        if np.random.random() < self.epsilon:
            action = np.random.randint(3)
        else:
            action = np.argmax(self.Q[current_state])
        reward = returns[-1] * 100
        self.Q[self.last_state, self.last_action] += self.lr * (
            reward + self.gamma * self.Q[current_state, action] -
            self.Q[self.last_state, self.last_action]
        )
        self.last_state = current_state
        self.last_action = action
        self.signal = action - 1
        self.confidence = min(abs(self.Q[current_state, action]) / 10, 1.0)


@FormulaRegistry.register(93)
class DeepQNetwork(BaseFormula):
    """ID 93: Deep Q-Network (simplified)"""

    CATEGORY = "machine_learning"
    NAME = "DeepQNetwork"
    DESCRIPTION = "Neural network Q-function approximation"

    def __init__(self, lookback: int = 100, state_dim: int = 5,
                 hidden_dim: int = 10, **kwargs):
        super().__init__(lookback, **kwargs)
        self.state_dim = state_dim
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.W2 = np.random.randn(hidden_dim, 3) * 0.1

    def _compute(self) -> None:
        if len(self.returns) < self.state_dim:
            return
        returns = self._returns_array()
        state = returns[-self.state_dim:]
        h = np.maximum(0, np.dot(state, self.W1))
        q_values = np.dot(h, self.W2)
        action = np.argmax(q_values)
        self.signal = action - 1
        self.confidence = min(abs(q_values[action] - np.mean(q_values)) /
                             (np.std(q_values) + 1e-10), 1.0)


@FormulaRegistry.register(94)
class PolicyGradient(BaseFormula):
    """ID 94: REINFORCE policy gradient"""

    CATEGORY = "machine_learning"
    NAME = "PolicyGradient"
    DESCRIPTION = "Direct policy optimization"

    def __init__(self, lookback: int = 100, state_dim: int = 5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.state_dim = state_dim
        self.policy_weights = np.random.randn(state_dim, 3) * 0.1

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)

    def _compute(self) -> None:
        if len(self.returns) < self.state_dim:
            return
        returns = self._returns_array()
        state = returns[-self.state_dim:]
        logits = np.dot(state, self.policy_weights)
        probs = self._softmax(logits)
        action = np.argmax(probs)
        self.signal = action - 1
        self.confidence = probs[action]


@FormulaRegistry.register(95)
class ActorCritic(BaseFormula):
    """ID 95: Actor-Critic architecture"""

    CATEGORY = "machine_learning"
    NAME = "ActorCritic"
    DESCRIPTION = "Combined policy and value network"

    def __init__(self, lookback: int = 100, state_dim: int = 5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.state_dim = state_dim
        self.actor_weights = np.random.randn(state_dim, 3) * 0.1
        self.critic_weights = np.random.randn(state_dim, 1) * 0.1

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)

    def _compute(self) -> None:
        if len(self.returns) < self.state_dim:
            return
        returns = self._returns_array()
        state = returns[-self.state_dim:]
        logits = np.dot(state, self.actor_weights)
        probs = self._softmax(logits)
        value = np.dot(state, self.critic_weights)[0]
        action = np.argmax(probs)
        self.signal = action - 1
        self.confidence = probs[action] * (1 / (1 + np.exp(-value)))


@FormulaRegistry.register(96)
class PPOAgent(BaseFormula):
    """ID 96: Proximal Policy Optimization (simplified)"""

    CATEGORY = "machine_learning"
    NAME = "PPOAgent"
    DESCRIPTION = "Clipped policy gradient"

    def __init__(self, lookback: int = 100, state_dim: int = 5,
                 clip_ratio: float = 0.2, **kwargs):
        super().__init__(lookback, **kwargs)
        self.state_dim = state_dim
        self.clip_ratio = clip_ratio
        self.policy_weights = np.random.randn(state_dim, 3) * 0.1
        self.old_probs = np.array([0.33, 0.34, 0.33])

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)

    def _compute(self) -> None:
        if len(self.returns) < self.state_dim:
            return
        returns = self._returns_array()
        state = returns[-self.state_dim:]
        logits = np.dot(state, self.policy_weights)
        probs = self._softmax(logits)
        ratio = probs / (self.old_probs + 1e-10)
        clipped_ratio = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        action = np.argmax(probs)
        self.old_probs = probs.copy()
        self.signal = action - 1
        self.confidence = min(probs[action], clipped_ratio[action])


@FormulaRegistry.register(97)
class A2CAgent(BaseFormula):
    """ID 97: Advantage Actor-Critic"""

    CATEGORY = "machine_learning"
    NAME = "A2CAgent"
    DESCRIPTION = "Synchronous advantage estimation"

    def __init__(self, lookback: int = 100, state_dim: int = 5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.state_dim = state_dim
        self.actor_weights = np.random.randn(state_dim, 3) * 0.1
        self.critic_weights = np.random.randn(state_dim, 1) * 0.1
        self.last_value = 0.0

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)

    def _compute(self) -> None:
        if len(self.returns) < self.state_dim:
            return
        returns = self._returns_array()
        state = returns[-self.state_dim:]
        value = np.dot(state, self.critic_weights)[0]
        advantage = returns[-1] * 100 + 0.95 * value - self.last_value
        self.last_value = value
        logits = np.dot(state, self.actor_weights)
        probs = self._softmax(logits)
        action = np.argmax(probs)
        self.signal = action - 1
        self.confidence = probs[action] * self._sigmoid(advantage)


@FormulaRegistry.register(98)
class TDLambda(BaseFormula):
    """ID 98: TD(λ) eligibility traces"""

    CATEGORY = "machine_learning"
    NAME = "TDLambda"
    DESCRIPTION = "Temporal difference with traces"

    def __init__(self, lookback: int = 100, n_states: int = 10,
                 lambda_: float = 0.9, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_states = n_states
        self.lambda_ = lambda_
        self.V = np.zeros(n_states)
        self.eligibility = np.zeros(n_states)
        self.last_state = 0

    def _get_state(self, returns: np.ndarray) -> int:
        recent = returns[-1] if len(returns) > 0 else 0
        state = int((recent + 0.05) / 0.01)
        return max(0, min(self.n_states - 1, state))

    def _compute(self) -> None:
        if len(self.returns) < 5:
            return
        returns = self._returns_array()
        current_state = self._get_state(returns)
        reward = returns[-1] * 100
        td_error = reward + 0.95 * self.V[current_state] - self.V[self.last_state]
        self.eligibility *= self.lambda_ * 0.95
        self.eligibility[self.last_state] += 1
        self.V += 0.1 * td_error * self.eligibility
        self.last_state = current_state
        if self.V[current_state] > 0.5:
            self.signal = 1
        elif self.V[current_state] < -0.5:
            self.signal = -1
        else:
            self.signal = 0
        self.confidence = min(abs(self.V[current_state]) / 5, 1.0)


@FormulaRegistry.register(99)
class GeneticAlgorithm(BaseFormula):
    """ID 99: Genetic Algorithm optimization"""

    CATEGORY = "machine_learning"
    NAME = "GeneticAlgorithm"
    DESCRIPTION = "Evolutionary strategy selection"

    def __init__(self, lookback: int = 100, population_size: int = 10, **kwargs):
        super().__init__(lookback, **kwargs)
        self.population_size = population_size
        self.population = [np.random.randn(3) * 0.1 for _ in range(population_size)]
        self.fitness = np.zeros(population_size)
        self.generation = 0

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        features = np.array([returns[-1], np.mean(returns[-5:]), np.std(returns[-5:])])
        for i, individual in enumerate(self.population):
            prediction = np.dot(features, individual)
            self.fitness[i] = prediction * returns[-1] if len(returns) > 1 else 0
        best_idx = np.argmax(self.fitness)
        best = self.population[best_idx]
        prediction = np.dot(features, best)
        self.signal = 1 if prediction > 0.001 else (-1 if prediction < -0.001 else 0)
        self.confidence = min(abs(prediction) * 100, 1.0)
        if self.generation % 10 == 0:
            worst_idx = np.argmin(self.fitness)
            parent1 = self.population[best_idx]
            parent2 = self.population[(best_idx + 1) % self.population_size]
            child = (parent1 + parent2) / 2 + np.random.randn(3) * 0.01
            self.population[worst_idx] = child
        self.generation += 1


@FormulaRegistry.register(100)
class ParticleSwarm(BaseFormula):
    """ID 100: Particle Swarm Optimization"""

    CATEGORY = "machine_learning"
    NAME = "ParticleSwarm"
    DESCRIPTION = "Swarm intelligence optimization"

    def __init__(self, lookback: int = 100, n_particles: int = 10, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_particles = n_particles
        self.positions = [np.random.randn(3) * 0.1 for _ in range(n_particles)]
        self.velocities = [np.random.randn(3) * 0.01 for _ in range(n_particles)]
        self.personal_best = [p.copy() for p in self.positions]
        self.personal_best_fitness = np.full(n_particles, -np.inf)
        self.global_best = np.zeros(3)
        self.global_best_fitness = -np.inf

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        features = np.array([returns[-1], np.mean(returns[-5:]), np.std(returns[-5:])])
        for i in range(self.n_particles):
            prediction = np.dot(features, self.positions[i])
            fitness = prediction * returns[-1] if len(returns) > 1 else 0
            if fitness > self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best[i] = self.positions[i].copy()
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best = self.positions[i].copy()
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        for i in range(self.n_particles):
            r1, r2 = np.random.random(2)
            self.velocities[i] = (w * self.velocities[i] +
                                  c1 * r1 * (self.personal_best[i] - self.positions[i]) +
                                  c2 * r2 * (self.global_best - self.positions[i]))
            self.positions[i] += self.velocities[i]
        prediction = np.dot(features, self.global_best)
        self.signal = 1 if prediction > 0.001 else (-1 if prediction < -0.001 else 0)
        self.confidence = min(abs(prediction) * 100, 1.0)


__all__ = [
    'SimplePerceptron', 'MultiLayerPerceptron', 'RecurrentUnit', 'LSTMUnit',
    'GRUUnit', 'AttentionMechanism', 'ConvolutionalFeature', 'AutoencoderAnomaly',
    'VariationalAutoencoder', 'ResidualBlock',
    'RandomForestSignal', 'GradientBoostingSignal', 'AdaBoostSignal',
    'BaggingSignal', 'StackingEnsemble', 'VotingClassifier', 'XGBoostLike',
    'LightGBMLike', 'CatBoostLike', 'NaiveBayesClassifier',
    'KMeansRegime', 'DBSCANAnomaly', 'HierarchicalCluster', 'GaussianMixture',
    'PCAFeatures', 'TSNEVisualization', 'UMAPEmbedding', 'IsolationForest',
    'LOFAnomaly', 'OneClassSVM',
    'QLearningAgent', 'SARSAAgent', 'DeepQNetwork', 'PolicyGradient',
    'ActorCritic', 'PPOAgent', 'A2CAgent', 'TDLambda', 'GeneticAlgorithm',
    'ParticleSwarm',
]
