"""
Gaussian Hidden Markov Model with Baum-Welch Training
=====================================================

Formula IDs: 72001-72005

True HMM implementation using the Baum-Welch (EM) algorithm for training.
This replaces rule-based regime detection with statistically learned states.

RenTech Background:
- Jim Simons hired Leonard Baum (co-inventor of Baum-Welch) in 1979
- Speech recognition HMMs became the foundation of Medallion Fund
- The same math that recognizes "hello" can recognize market regimes
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import warnings

# Try to use hmmlearn if available, otherwise use our implementation
try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False


class HMMState(Enum):
    """Market regime states discovered by HMM."""
    ACCUMULATION = 0    # Smart money buying
    DISTRIBUTION = 1    # Smart money selling
    TRENDING_UP = 2     # Strong uptrend
    TRENDING_DOWN = 3   # Strong downtrend
    CONSOLIDATION = 4   # Range-bound
    CAPITULATION = 5    # Panic selling
    EUPHORIA = 6        # Extreme greed


@dataclass
class HMMConfig:
    """HMM configuration."""
    n_states: int = 5
    n_features: int = 5
    n_iter: int = 100
    tol: float = 1e-4
    covariance_type: str = 'full'  # 'full', 'diag', 'spherical'
    random_state: int = 42
    min_train_samples: int = 252  # 1 year minimum


@dataclass
class TrainedHMMSignal:
    """Signal from trained HMM."""
    direction: int          # -1, 0, 1
    confidence: float       # 0-1
    current_state: int      # HMM state index
    state_probability: float
    transition_probs: Dict[int, float]
    expected_duration: float
    features_used: List[str]


class GaussianHMM:
    """
    Gaussian Hidden Markov Model for regime detection.

    Uses multivariate Gaussian emissions with full covariance.
    Trained via Baum-Welch (EM algorithm).
    """

    def __init__(self, config: HMMConfig = None):
        self.config = config or HMMConfig()
        self.model = None
        self.is_trained = False
        self.state_labels: Dict[int, str] = {}
        self.training_log_likelihood: float = 0.0
        self.feature_names: List[str] = []

        # State statistics from training
        self.state_means: Optional[np.ndarray] = None
        self.state_covars: Optional[np.ndarray] = None
        self.transition_matrix: Optional[np.ndarray] = None
        self.initial_probs: Optional[np.ndarray] = None

    def _init_model(self):
        """Initialize HMM model."""
        if HAS_HMMLEARN:
            self.model = hmm.GaussianHMM(
                n_components=self.config.n_states,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                tol=self.config.tol,
                random_state=self.config.random_state,
            )
        else:
            # Use our own implementation
            self.model = _SimpleGaussianHMM(
                n_states=self.config.n_states,
                n_iter=self.config.n_iter,
                tol=self.config.tol,
            )

    def train(self, observations: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Train HMM on historical observations.

        Args:
            observations: (T, n_features) array of observations
            feature_names: Names of features for interpretability

        Returns:
            Training statistics
        """
        if len(observations) < self.config.min_train_samples:
            raise ValueError(f"Need at least {self.config.min_train_samples} samples")

        self._init_model()
        self.feature_names = feature_names or [f"f{i}" for i in range(observations.shape[1])]

        # Handle NaN/Inf
        observations = np.nan_to_num(observations, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(observations)

        # Extract learned parameters
        if HAS_HMMLEARN:
            self.state_means = self.model.means_
            self.state_covars = self.model.covars_
            self.transition_matrix = self.model.transmat_
            self.initial_probs = self.model.startprob_
            self.training_log_likelihood = self.model.score(observations)
        else:
            self.state_means = self.model.means
            self.state_covars = self.model.covars
            self.transition_matrix = self.model.trans_mat
            self.initial_probs = self.model.start_prob
            self.training_log_likelihood = self.model.score(observations)

        # Label states based on characteristics
        self._label_states(observations)

        self.is_trained = True

        return {
            'log_likelihood': self.training_log_likelihood,
            'n_states': self.config.n_states,
            'n_samples': len(observations),
            'state_labels': self.state_labels,
            'converged': True,
        }

    def _label_states(self, observations: np.ndarray):
        """Automatically label states based on their characteristics."""
        # Get state sequence
        states = self.decode(observations)

        # Compute statistics per state
        state_returns = {}
        state_volatility = {}

        # Assume first feature is returns
        returns = observations[:, 0] if observations.shape[1] > 0 else np.zeros(len(observations))

        for s in range(self.config.n_states):
            mask = states == s
            if mask.sum() > 0:
                state_returns[s] = returns[mask].mean()
                state_volatility[s] = returns[mask].std()
            else:
                state_returns[s] = 0
                state_volatility[s] = 0

        # Sort states by returns
        sorted_states = sorted(state_returns.keys(), key=lambda x: state_returns[x])

        # Assign labels based on position
        if self.config.n_states == 3:
            labels = ['bearish', 'neutral', 'bullish']
        elif self.config.n_states == 5:
            labels = ['capitulation', 'bearish', 'neutral', 'bullish', 'euphoria']
        elif self.config.n_states == 7:
            labels = ['capitulation', 'strong_bearish', 'bearish', 'neutral',
                     'bullish', 'strong_bullish', 'euphoria']
        else:
            labels = [f'state_{i}' for i in range(self.config.n_states)]

        for i, s in enumerate(sorted_states):
            if i < len(labels):
                self.state_labels[s] = labels[i]
            else:
                self.state_labels[s] = f'state_{s}'

    def decode(self, observations: np.ndarray) -> np.ndarray:
        """
        Decode most likely state sequence using Viterbi algorithm.

        Args:
            observations: (T, n_features) array

        Returns:
            Array of state indices
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        observations = np.nan_to_num(observations, nan=0.0)

        if HAS_HMMLEARN:
            return self.model.predict(observations)
        else:
            return self.model.predict(observations)

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """
        Get state probabilities for each observation.

        Args:
            observations: (T, n_features) array

        Returns:
            (T, n_states) probability array
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        observations = np.nan_to_num(observations, nan=0.0)

        if HAS_HMMLEARN:
            return self.model.predict_proba(observations)
        else:
            return self.model.predict_proba(observations)

    def get_current_state(self, observations: np.ndarray) -> Tuple[int, float]:
        """
        Get current state and its probability.

        Args:
            observations: Historical observations up to now

        Returns:
            (state_index, probability)
        """
        probs = self.predict_proba(observations)
        current_probs = probs[-1]
        state = np.argmax(current_probs)
        return state, current_probs[state]

    def get_transition_probs(self, current_state: int) -> Dict[int, float]:
        """Get transition probabilities from current state."""
        if self.transition_matrix is None:
            return {}
        return {i: self.transition_matrix[current_state, i]
                for i in range(self.config.n_states)}

    def get_expected_duration(self, state: int) -> float:
        """
        Get expected duration in a state (geometric distribution).

        E[duration] = 1 / (1 - p_ii) where p_ii is self-transition probability
        """
        if self.transition_matrix is None:
            return 1.0
        self_trans = self.transition_matrix[state, state]
        if self_trans >= 1.0:
            return float('inf')
        return 1.0 / (1.0 - self_trans)

    def generate_signal(self, observations: np.ndarray) -> TrainedHMMSignal:
        """
        Generate trading signal from HMM state analysis.

        Args:
            observations: Historical observations

        Returns:
            TrainedHMMSignal with direction and confidence
        """
        if not self.is_trained:
            return TrainedHMMSignal(
                direction=0, confidence=0.0, current_state=-1,
                state_probability=0.0, transition_probs={},
                expected_duration=0.0, features_used=[]
            )

        state, prob = self.get_current_state(observations)
        trans_probs = self.get_transition_probs(state)
        duration = self.get_expected_duration(state)

        # Determine direction based on state label
        label = self.state_labels.get(state, '')

        if 'bullish' in label or 'euphoria' in label:
            direction = 1
        elif 'bearish' in label or 'capitulation' in label:
            direction = -1
        else:
            direction = 0

        # Confidence is state probability * regime clarity
        # High confidence = high prob AND clear trend
        confidence = prob * (1.0 - trans_probs.get(state, 0.5))
        confidence = min(1.0, max(0.0, confidence))

        return TrainedHMMSignal(
            direction=direction,
            confidence=confidence,
            current_state=state,
            state_probability=prob,
            transition_probs=trans_probs,
            expected_duration=duration,
            features_used=self.feature_names,
        )


class _SimpleGaussianHMM:
    """
    Simple Gaussian HMM implementation when hmmlearn not available.

    Implements Baum-Welch (EM) for training and Viterbi for decoding.
    """

    def __init__(self, n_states: int = 5, n_iter: int = 100, tol: float = 1e-4):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol

        # Parameters to learn
        self.means: Optional[np.ndarray] = None
        self.covars: Optional[np.ndarray] = None
        self.trans_mat: Optional[np.ndarray] = None
        self.start_prob: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        """Train HMM using Baum-Welch algorithm."""
        T, n_features = X.shape

        # Initialize parameters
        self._init_params(X, n_features)

        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            # E-step: compute forward-backward probabilities
            alpha, scale = self._forward(X)
            beta = self._backward(X, scale)

            # Compute responsibilities
            gamma = alpha * beta
            gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-10)

            # Compute xi (transition responsibilities)
            xi = self._compute_xi(X, alpha, beta, scale)

            # M-step: update parameters
            self._update_params(X, gamma, xi)

            # Check convergence
            ll = np.sum(np.log(scale + 1e-10))
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def _init_params(self, X: np.ndarray, n_features: int):
        """Initialize HMM parameters using K-means-like approach."""
        T = len(X)

        # Initialize with uniform start probability
        self.start_prob = np.ones(self.n_states) / self.n_states

        # Initialize transition matrix (slight diagonal bias)
        self.trans_mat = np.ones((self.n_states, self.n_states)) / self.n_states
        self.trans_mat += 0.1 * np.eye(self.n_states)
        self.trans_mat /= self.trans_mat.sum(axis=1, keepdims=True)

        # Initialize means using percentiles
        percentiles = np.linspace(0, 100, self.n_states + 2)[1:-1]
        self.means = np.percentile(X, percentiles, axis=0)

        # Initialize covariances
        self.covars = np.array([np.cov(X.T) + 0.1 * np.eye(n_features)
                                for _ in range(self.n_states)])

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward algorithm with scaling."""
        T = len(X)
        alpha = np.zeros((T, self.n_states))
        scale = np.zeros(T)

        # Initial step
        for s in range(self.n_states):
            alpha[0, s] = self.start_prob[s] * self._emission_prob(X[0], s)
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0] + 1e-10

        # Forward pass
        for t in range(1, T):
            for s in range(self.n_states):
                alpha[t, s] = (alpha[t-1] @ self.trans_mat[:, s]) * self._emission_prob(X[t], s)
            scale[t] = alpha[t].sum()
            alpha[t] /= scale[t] + 1e-10

        return alpha, scale

    def _backward(self, X: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Backward algorithm with scaling."""
        T = len(X)
        beta = np.zeros((T, self.n_states))
        beta[-1] = 1.0

        for t in range(T - 2, -1, -1):
            for s in range(self.n_states):
                for s2 in range(self.n_states):
                    beta[t, s] += (self.trans_mat[s, s2] *
                                   self._emission_prob(X[t+1], s2) * beta[t+1, s2])
            beta[t] /= scale[t+1] + 1e-10

        return beta

    def _emission_prob(self, x: np.ndarray, state: int) -> float:
        """Compute Gaussian emission probability."""
        mean = self.means[state]
        cov = self.covars[state]

        d = len(x)
        diff = x - mean

        try:
            cov_inv = np.linalg.inv(cov)
            det = np.linalg.det(cov)

            if det <= 0:
                det = 1e-10

            exp_term = -0.5 * diff @ cov_inv @ diff
            norm = 1.0 / (np.sqrt((2 * np.pi) ** d * det) + 1e-10)

            return norm * np.exp(np.clip(exp_term, -500, 0))
        except np.linalg.LinAlgError:
            return 1e-10

    def _compute_xi(self, X: np.ndarray, alpha: np.ndarray,
                    beta: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Compute transition responsibilities."""
        T = len(X)
        xi = np.zeros((T - 1, self.n_states, self.n_states))

        for t in range(T - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (alpha[t, i] * self.trans_mat[i, j] *
                                   self._emission_prob(X[t+1], j) * beta[t+1, j])
            xi[t] /= xi[t].sum() + 1e-10

        return xi

    def _update_params(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """M-step: update parameters."""
        T = len(X)

        # Update start probability
        self.start_prob = gamma[0]

        # Update transition matrix
        for i in range(self.n_states):
            denom = gamma[:-1, i].sum() + 1e-10
            for j in range(self.n_states):
                self.trans_mat[i, j] = xi[:, i, j].sum() / denom

        # Ensure rows sum to 1
        self.trans_mat /= self.trans_mat.sum(axis=1, keepdims=True) + 1e-10

        # Update means and covariances
        for s in range(self.n_states):
            gamma_s = gamma[:, s:s+1]
            denom = gamma_s.sum() + 1e-10

            # Mean
            self.means[s] = (gamma_s * X).sum(axis=0) / denom

            # Covariance
            diff = X - self.means[s]
            self.covars[s] = (gamma_s * diff).T @ diff / denom

            # Add regularization
            self.covars[s] += 0.01 * np.eye(X.shape[1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding for most likely state sequence."""
        T = len(X)

        # Viterbi algorithm
        V = np.zeros((T, self.n_states))
        path = np.zeros((T, self.n_states), dtype=int)

        # Initial step
        for s in range(self.n_states):
            V[0, s] = np.log(self.start_prob[s] + 1e-10) + np.log(self._emission_prob(X[0], s) + 1e-10)

        # Forward pass
        for t in range(1, T):
            for s in range(self.n_states):
                probs = V[t-1] + np.log(self.trans_mat[:, s] + 1e-10)
                path[t, s] = np.argmax(probs)
                V[t, s] = probs[path[t, s]] + np.log(self._emission_prob(X[t], s) + 1e-10)

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(V[-1])
        for t in range(T - 2, -1, -1):
            states[t] = path[t + 1, states[t + 1]]

        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get state probabilities using forward algorithm."""
        alpha, scale = self._forward(X)
        # Normalize to get probabilities
        probs = alpha / (alpha.sum(axis=1, keepdims=True) + 1e-10)
        return probs

    def score(self, X: np.ndarray) -> float:
        """Compute log-likelihood of observations."""
        _, scale = self._forward(X)
        return np.sum(np.log(scale + 1e-10))


# =============================================================================
# FORMULA IMPLEMENTATIONS (72001-72005)
# =============================================================================

class HMM3StateTrader:
    """
    Formula 72001: 3-State HMM Trader

    Uses a simple 3-state HMM (bullish/neutral/bearish) for regime detection.

    Signals:
    - LONG when in bullish state with high confidence
    - SHORT when in bearish state with high confidence
    - FLAT when neutral or low confidence
    """

    FORMULA_ID = 72001

    def __init__(self, min_confidence: float = 0.6):
        self.hmm = GaussianHMM(HMMConfig(n_states=3))
        self.min_confidence = min_confidence
        self.is_trained = False

    def train(self, features: np.ndarray, feature_names: List[str] = None):
        """Train the HMM on historical data."""
        stats = self.hmm.train(features, feature_names)
        self.is_trained = True
        return stats

    def generate_signal(self, features: np.ndarray) -> TrainedHMMSignal:
        """Generate trading signal."""
        signal = self.hmm.generate_signal(features)

        # Apply confidence threshold
        if signal.confidence < self.min_confidence:
            signal = TrainedHMMSignal(
                direction=0,
                confidence=signal.confidence,
                current_state=signal.current_state,
                state_probability=signal.state_probability,
                transition_probs=signal.transition_probs,
                expected_duration=signal.expected_duration,
                features_used=signal.features_used,
            )

        return signal


class HMM5StateTrader:
    """
    Formula 72002: 5-State HMM Trader

    Uses 5-state HMM capturing: capitulation, bearish, neutral, bullish, euphoria.
    More nuanced than 3-state for capturing extreme market conditions.
    """

    FORMULA_ID = 72002

    def __init__(self, min_confidence: float = 0.5):
        self.hmm = GaussianHMM(HMMConfig(n_states=5))
        self.min_confidence = min_confidence

    def train(self, features: np.ndarray, feature_names: List[str] = None):
        return self.hmm.train(features, feature_names)

    def generate_signal(self, features: np.ndarray) -> TrainedHMMSignal:
        return self.hmm.generate_signal(features)


class HMM7StateTrader:
    """
    Formula 72003: 7-State HMM Trader

    Maximum granularity with 7 states for fine-grained regime detection.
    Best for longer-term analysis with abundant data.
    """

    FORMULA_ID = 72003

    def __init__(self):
        self.hmm = GaussianHMM(HMMConfig(n_states=7))

    def train(self, features: np.ndarray, feature_names: List[str] = None):
        return self.hmm.train(features, feature_names)

    def generate_signal(self, features: np.ndarray) -> TrainedHMMSignal:
        return self.hmm.generate_signal(features)


class HMMOptimalStateTrader:
    """
    Formula 72004: Optimal State Count HMM Trader

    Automatically selects optimal number of states using BIC criterion.
    Tests 3, 5, 7, 10 states and picks best fit.
    """

    FORMULA_ID = 72004

    def __init__(self, state_options: List[int] = None):
        self.state_options = state_options or [3, 5, 7, 10]
        self.best_hmm: Optional[GaussianHMM] = None
        self.best_n_states: int = 5
        self.bic_scores: Dict[int, float] = {}

    def train(self, features: np.ndarray, feature_names: List[str] = None):
        """Train multiple HMMs and select best by BIC."""
        T, n_features = features.shape

        best_bic = np.inf

        for n_states in self.state_options:
            try:
                hmm = GaussianHMM(HMMConfig(n_states=n_states))
                stats = hmm.train(features, feature_names)

                # Compute BIC = -2*LL + k*log(n)
                # k = parameters: (n_states-1) + n_states*(n_states-1) + n_states*n_features + n_states*n_features^2
                k = (n_states - 1) + n_states * (n_states - 1) + n_states * n_features * (1 + n_features)
                bic = -2 * stats['log_likelihood'] + k * np.log(T)

                self.bic_scores[n_states] = bic

                if bic < best_bic:
                    best_bic = bic
                    self.best_hmm = hmm
                    self.best_n_states = n_states

            except Exception as e:
                self.bic_scores[n_states] = np.inf

        return {
            'optimal_states': self.best_n_states,
            'bic_scores': self.bic_scores,
        }

    def generate_signal(self, features: np.ndarray) -> TrainedHMMSignal:
        if self.best_hmm is None:
            raise RuntimeError("Model must be trained first")
        return self.best_hmm.generate_signal(features)


class HMMTransitionTrader:
    """
    Formula 72005: HMM Transition-Based Trader

    Trades based on state TRANSITIONS rather than current state.
    Signals generated when transitioning INTO bullish/bearish states.

    Key insight: The transition moment often has the most edge.
    """

    FORMULA_ID = 72005

    def __init__(self, n_states: int = 5):
        self.hmm = GaussianHMM(HMMConfig(n_states=n_states))
        self.prev_state: int = -1

    def train(self, features: np.ndarray, feature_names: List[str] = None):
        return self.hmm.train(features, feature_names)

    def generate_signal(self, features: np.ndarray) -> TrainedHMMSignal:
        """Generate signal based on state transitions."""
        base_signal = self.hmm.generate_signal(features)
        current_state = base_signal.current_state

        # Check for transition
        if self.prev_state == -1:
            self.prev_state = current_state
            return TrainedHMMSignal(
                direction=0,
                confidence=0.0,
                current_state=current_state,
                state_probability=base_signal.state_probability,
                transition_probs=base_signal.transition_probs,
                expected_duration=base_signal.expected_duration,
                features_used=base_signal.features_used,
            )

        # Signal on transition
        transitioned = current_state != self.prev_state
        self.prev_state = current_state

        if transitioned:
            # Boost confidence on transition
            label = self.hmm.state_labels.get(current_state, '')

            if 'bullish' in label or 'euphoria' in label:
                direction = 1
            elif 'bearish' in label or 'capitulation' in label:
                direction = -1
            else:
                direction = 0

            return TrainedHMMSignal(
                direction=direction,
                confidence=min(1.0, base_signal.confidence * 1.5),  # Boost
                current_state=current_state,
                state_probability=base_signal.state_probability,
                transition_probs=base_signal.transition_probs,
                expected_duration=base_signal.expected_duration,
                features_used=base_signal.features_used,
            )
        else:
            # No transition - reduce confidence
            return TrainedHMMSignal(
                direction=base_signal.direction,
                confidence=base_signal.confidence * 0.5,  # Reduce
                current_state=current_state,
                state_probability=base_signal.state_probability,
                transition_probs=base_signal.transition_probs,
                expected_duration=base_signal.expected_duration,
                features_used=base_signal.features_used,
            )
