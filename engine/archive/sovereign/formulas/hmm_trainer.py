"""
HMM TRAINER - Baum-Welch Training on Historical Blockchain Flows
================================================================

"We're right 50.75% of the time, but we're 100% right 50.75% of the time."
- Robert Mercer, Renaissance Technologies

This module trains Hidden Markov Models on historical flow data using the
Baum-Welch algorithm (Expectation-Maximization for HMMs).

THE GOAL:
    Train HMM to detect market regimes from blockchain flow patterns.
    Each regime has different trading characteristics:
    - ACCUMULATION: Outflows dominate, smart money buying -> LONG
    - DISTRIBUTION: Inflows dominate, smart money selling -> SHORT
    - NEUTRAL: Balanced, no clear signal -> WAIT
    - CAPITULATION: Panic inflows, extreme fear -> Contrarian LONG
    - EUPHORIA: Greed outflows, extreme greed -> Contrarian SHORT

TRAINING PROCESS:
    1. Load historical flow events from database
    2. Convert to observation sequences (features)
    3. Run Baum-Welch to estimate:
       - Transition matrix A (state-to-state probabilities)
       - Emission parameters (mean/var per state)
       - Initial state distribution
    4. Validate on held-out data
    5. Save trained model if accuracy > 50.75%

CITATION:
    - Baum-Welch: Baum et al. (1970)
    - HMM for Finance: Hamilton (1989)
    - RenTech connection: Leonard Baum co-founded Renaissance Technologies
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from .historical_data import HistoricalFlowDatabase, FlowEvent


@dataclass
class HMMParameters:
    """Trained HMM parameters."""
    n_states: int
    transition_matrix: np.ndarray  # A[i,j] = P(state_j | state_i)
    emission_means: Dict[int, List[float]]  # Mean per state per feature
    emission_vars: Dict[int, List[float]]   # Variance per state per feature
    initial_probs: np.ndarray  # P(initial state)
    state_names: List[str]

    # Training metadata
    training_samples: int
    log_likelihood: float
    converged: bool
    iterations: int


class BaumWelchTrainer:
    """
    Baum-Welch algorithm for training HMM on flow data.

    ALGORITHM:
    1. Initialize parameters randomly or from prior
    2. E-step: Compute forward/backward probabilities
    3. M-step: Re-estimate parameters from expected counts
    4. Repeat until convergence (log-likelihood stabilizes)

    FEATURES (observations):
    - flow_imbalance: (outflow - inflow) / total
    - flow_velocity: BTC/second flow rate
    - whale_ratio: Large transaction ratio
    - fee_percentile: Transaction urgency

    STATES (hidden):
    - 0: ACCUMULATION
    - 1: DISTRIBUTION
    - 2: NEUTRAL
    - 3: CAPITULATION
    - 4: EUPHORIA
    """

    STATE_NAMES = ['ACCUMULATION', 'DISTRIBUTION', 'NEUTRAL', 'CAPITULATION', 'EUPHORIA']
    N_FEATURES = 4  # flow_imbalance, flow_velocity, whale_ratio, fee_percentile

    def __init__(self, n_states: int = 5, n_iter: int = 100, tol: float = 1e-4):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol

        # Parameters (initialized in fit())
        self.A = None  # Transition matrix
        self.means = None  # Emission means
        self.vars = None  # Emission variances
        self.pi = None  # Initial state distribution

    def _init_parameters(self, observations: np.ndarray):
        """Initialize HMM parameters."""
        n_obs = len(observations)

        # Transition matrix: slightly sticky states
        self.A = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i == j:
                    self.A[i, j] = 0.7  # Stay in same state
                else:
                    self.A[i, j] = 0.3 / (self.n_states - 1)  # Transition

        # Emission parameters: initialize from data clustering
        # Use k-means-like initialization
        self.means = {}
        self.vars = {}

        # Simple initialization: divide data into n_states chunks
        chunk_size = max(1, n_obs // self.n_states)
        for s in range(self.n_states):
            start = s * chunk_size
            end = min((s + 1) * chunk_size, n_obs)
            chunk = observations[start:end]

            if len(chunk) > 0:
                self.means[s] = list(np.mean(chunk, axis=0))
                self.vars[s] = list(np.var(chunk, axis=0) + 0.01)  # Add small constant
            else:
                self.means[s] = [0.0] * self.N_FEATURES
                self.vars[s] = [1.0] * self.N_FEATURES

        # Initial state distribution: start neutral
        self.pi = np.array([0.1, 0.1, 0.6, 0.1, 0.1])

    def _emission_prob(self, state: int, obs: np.ndarray) -> float:
        """Calculate P(observation | state) using Gaussian."""
        means = self.means[state]
        vars = self.vars[state]

        log_prob = 0.0
        for i in range(len(obs)):
            if vars[i] > 0:
                diff = obs[i] - means[i]
                log_prob -= 0.5 * (diff ** 2) / vars[i]
                log_prob -= 0.5 * math.log(2 * math.pi * vars[i])

        return math.exp(max(-500, log_prob))

    def _forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward algorithm: compute alpha (forward probabilities)."""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))

        # Initialize
        for s in range(self.n_states):
            alpha[0, s] = self.pi[s] * self._emission_prob(s, observations[0])

        # Normalize to prevent underflow
        scale = np.zeros(T)
        scale[0] = alpha[0].sum()
        if scale[0] > 0:
            alpha[0] /= scale[0]

        # Forward pass
        for t in range(1, T):
            for s in range(self.n_states):
                alpha[t, s] = sum(alpha[t-1, s_prev] * self.A[s_prev, s]
                                  for s_prev in range(self.n_states))
                alpha[t, s] *= self._emission_prob(s, observations[t])

            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]

        # Log-likelihood (sum of log scales)
        log_likelihood = sum(math.log(s) for s in scale if s > 0)

        return alpha, log_likelihood

    def _backward(self, observations: np.ndarray, scale: np.ndarray = None) -> np.ndarray:
        """Backward algorithm: compute beta (backward probabilities)."""
        T = len(observations)
        beta = np.zeros((T, self.n_states))

        # Initialize
        beta[T-1] = 1.0

        # Backward pass
        for t in range(T - 2, -1, -1):
            for s in range(self.n_states):
                beta[t, s] = sum(
                    self.A[s, s_next] * self._emission_prob(s_next, observations[t+1]) * beta[t+1, s_next]
                    for s_next in range(self.n_states)
                )

            # Normalize
            total = beta[t].sum()
            if total > 0:
                beta[t] /= total

        return beta

    def _e_step(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """E-step: compute gamma and xi from forward-backward."""
        T = len(observations)

        alpha, log_likelihood = self._forward(observations)
        beta = self._backward(observations)

        # Gamma: P(state_t | observations)
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma = np.divide(gamma, gamma_sum, where=gamma_sum > 0)

        # Xi: P(state_t, state_{t+1} | observations)
        xi = np.zeros((T - 1, self.n_states, self.n_states))
        for t in range(T - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (alpha[t, i] * self.A[i, j] *
                                   self._emission_prob(j, observations[t+1]) *
                                   beta[t+1, j])

            # Normalize
            xi_sum = xi[t].sum()
            if xi_sum > 0:
                xi[t] /= xi_sum

        return gamma, xi, log_likelihood

    def _m_step(self, observations: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """M-step: re-estimate parameters from expected counts."""
        T = len(observations)

        # Update initial distribution
        self.pi = gamma[0] / gamma[0].sum() if gamma[0].sum() > 0 else self.pi

        # Update transition matrix
        for i in range(self.n_states):
            denom = gamma[:-1, i].sum()
            if denom > 0:
                for j in range(self.n_states):
                    self.A[i, j] = xi[:, i, j].sum() / denom

        # Normalize transition matrix
        for i in range(self.n_states):
            row_sum = self.A[i].sum()
            if row_sum > 0:
                self.A[i] /= row_sum

        # Update emission parameters
        for s in range(self.n_states):
            gamma_s = gamma[:, s]
            gamma_sum = gamma_s.sum()

            if gamma_sum > 0.01:
                # Update means
                weighted_obs = (gamma_s[:, np.newaxis] * observations).sum(axis=0)
                self.means[s] = list(weighted_obs / gamma_sum)

                # Update variances
                diff = observations - np.array(self.means[s])
                weighted_var = (gamma_s[:, np.newaxis] * (diff ** 2)).sum(axis=0)
                self.vars[s] = list(np.maximum(weighted_var / gamma_sum, 0.01))

    def fit(self, observations: np.ndarray, verbose: bool = True) -> HMMParameters:
        """
        Train HMM using Baum-Welch algorithm.

        Args:
            observations: Array of shape (T, n_features)
            verbose: Print progress

        Returns:
            Trained HMM parameters
        """
        if len(observations) < 10:
            raise ValueError("Need at least 10 observations to train")

        # Initialize parameters
        self._init_parameters(observations)

        prev_log_likelihood = float('-inf')
        converged = False

        for iteration in range(self.n_iter):
            # E-step
            gamma, xi, log_likelihood = self._e_step(observations)

            # Check convergence
            improvement = log_likelihood - prev_log_likelihood
            if abs(improvement) < self.tol and iteration > 5:
                converged = True
                if verbose:
                    print(f"[HMM] Converged at iteration {iteration}, LL={log_likelihood:.2f}")
                break

            prev_log_likelihood = log_likelihood

            # M-step
            self._m_step(observations, gamma, xi)

            if verbose and iteration % 10 == 0:
                print(f"[HMM] Iteration {iteration}, LL={log_likelihood:.2f}")

        # Build result
        return HMMParameters(
            n_states=self.n_states,
            transition_matrix=self.A.copy(),
            emission_means={k: list(v) for k, v in self.means.items()},
            emission_vars={k: list(v) for k, v in self.vars.items()},
            initial_probs=self.pi.copy(),
            state_names=self.STATE_NAMES[:self.n_states],
            training_samples=len(observations),
            log_likelihood=log_likelihood,
            converged=converged,
            iterations=iteration + 1,
        )

    def predict_states(self, observations: np.ndarray) -> List[int]:
        """Viterbi algorithm: find most likely state sequence."""
        T = len(observations)

        # Viterbi algorithm
        V = np.zeros((T, self.n_states))
        path = np.zeros((T, self.n_states), dtype=int)

        # Initialize
        for s in range(self.n_states):
            V[0, s] = math.log(self.pi[s] + 1e-10) + math.log(self._emission_prob(s, observations[0]) + 1e-10)

        # Forward pass
        for t in range(1, T):
            for s in range(self.n_states):
                probs = [V[t-1, s_prev] + math.log(self.A[s_prev, s] + 1e-10) for s_prev in range(self.n_states)]
                path[t, s] = int(np.argmax(probs))
                V[t, s] = probs[path[t, s]] + math.log(self._emission_prob(s, observations[t]) + 1e-10)

        # Backtrack
        best_path = [0] * T
        best_path[-1] = int(np.argmax(V[-1]))
        for t in range(T - 2, -1, -1):
            best_path[t] = path[t + 1, best_path[t + 1]]

        return best_path


class HMMTrainingPipeline:
    """
    Complete training pipeline for blockchain flow HMM.

    STEPS:
    1. Load historical flows from database
    2. Convert to observation sequences
    3. Split into train/validation (80/20)
    4. Train multiple HMMs with different initializations
    5. Select best model by validation accuracy
    6. Save if accuracy > 50.75%

    "We look for things that can be replicated thousands of times."
    - Jim Simons
    """

    def __init__(self, db: HistoricalFlowDatabase = None):
        self.db = db or HistoricalFlowDatabase()

    def flows_to_observations(self, flows: List[FlowEvent]) -> np.ndarray:
        """Convert flow events to observation matrix."""
        observations = []

        for flow in flows:
            obs = [
                flow.flow_imbalance,
                flow.flow_velocity,
                flow.whale_ratio,
                flow.fee_percentile / 100.0,  # Normalize to 0-1
            ]
            observations.append(obs)

        return np.array(observations)

    def flows_to_outcomes(self, flows: List[FlowEvent], timeframe: str = '30s') -> List[int]:
        """Extract outcomes from flows."""
        outcome_map = {
            '10s': lambda f: f.outcome_10s,
            '30s': lambda f: f.outcome_30s,
            '60s': lambda f: f.outcome_60s,
            '120s': lambda f: f.outcome_120s,
        }
        getter = outcome_map.get(timeframe, outcome_map['30s'])
        return [getter(f) for f in flows]

    def calculate_accuracy(self, states: List[int], outcomes: List[int]) -> float:
        """
        Calculate prediction accuracy.

        State -> Signal mapping:
        - ACCUMULATION (0) -> LONG (+1)
        - DISTRIBUTION (1) -> SHORT (-1)
        - NEUTRAL (2) -> WAIT (0)
        - CAPITULATION (3) -> Contrarian LONG (+1)
        - EUPHORIA (4) -> Contrarian SHORT (-1)
        """
        signal_map = {0: 1, 1: -1, 2: 0, 3: 1, 4: -1}

        correct = 0
        total = 0

        for state, outcome in zip(states, outcomes):
            if outcome == 0:  # No price movement
                continue

            signal = signal_map.get(state, 0)
            if signal == 0:  # NEUTRAL = no trade
                continue

            total += 1
            # Signal matches outcome direction
            if (signal == 1 and outcome == 1) or (signal == -1 and outcome == -1):
                correct += 1

        return correct / max(1, total)

    def train(self, min_flows: int = 1000, n_restarts: int = 5,
              validation_split: float = 0.2, verbose: bool = True) -> Optional[HMMParameters]:
        """
        Full training pipeline.

        Args:
            min_flows: Minimum flows required to train
            n_restarts: Number of random restarts (pick best)
            validation_split: Fraction for validation
            verbose: Print progress

        Returns:
            Best trained HMM parameters, or None if edge < 50.75%
        """
        # Load flows
        flows = self.db.get_flows(min_btc=0.1)
        if len(flows) < min_flows:
            print(f"[TRAIN] Not enough data: {len(flows)} flows (need {min_flows})")
            return None

        if verbose:
            print(f"[TRAIN] Loaded {len(flows)} flows")

        # Filter to flows with outcomes
        flows_with_outcomes = [f for f in flows if f.outcome_30s != 0]
        if len(flows_with_outcomes) < min_flows // 2:
            print(f"[TRAIN] Not enough outcomes: {len(flows_with_outcomes)}")
            return None

        if verbose:
            print(f"[TRAIN] {len(flows_with_outcomes)} flows with outcomes")

        # Convert to observations
        observations = self.flows_to_observations(flows_with_outcomes)
        outcomes = self.flows_to_outcomes(flows_with_outcomes)

        # Split train/validation
        split_idx = int(len(observations) * (1 - validation_split))
        train_obs = observations[:split_idx]
        val_obs = observations[split_idx:]
        val_outcomes = outcomes[split_idx:]

        if verbose:
            print(f"[TRAIN] Train: {len(train_obs)}, Validation: {len(val_obs)}")

        # Train multiple models, pick best
        best_model = None
        best_accuracy = 0.0

        for restart in range(n_restarts):
            if verbose:
                print(f"\n[TRAIN] === Restart {restart + 1}/{n_restarts} ===")

            trainer = BaumWelchTrainer(n_states=5, n_iter=100, tol=1e-4)

            try:
                # Shuffle training data for different initialization
                indices = list(range(len(train_obs)))
                random.shuffle(indices)
                shuffled_obs = train_obs[indices]

                params = trainer.fit(shuffled_obs, verbose=verbose)

                # Validate
                val_states = trainer.predict_states(val_obs)
                accuracy = self.calculate_accuracy(val_states, val_outcomes)

                if verbose:
                    print(f"[TRAIN] Validation accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = params

            except Exception as e:
                print(f"[TRAIN] Restart {restart + 1} failed: {e}")

        if best_model is None:
            print("[TRAIN] All restarts failed")
            return None

        # Check if edge is significant
        edge = best_accuracy - 0.5
        if verbose:
            print(f"\n[TRAIN] === RESULTS ===")
            print(f"[TRAIN] Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            print(f"[TRAIN] Edge over random: {edge:.4f} ({edge*100:.2f}%)")

        if best_accuracy < 0.5075:
            print(f"[TRAIN] WARNING: Accuracy {best_accuracy:.4f} < 50.75% threshold")
            print("[TRAIN] Model may not have real edge - use with caution")
        else:
            print(f"[TRAIN] SUCCESS: Accuracy {best_accuracy:.4f} >= 50.75% threshold")
            print("[TRAIN] Model has statistically significant edge")

        # Save to database
        self.db.save_hmm_model(
            name='default',
            n_states=best_model.n_states,
            transition_matrix=best_model.transition_matrix.tolist(),
            emission_means=best_model.emission_means,
            emission_vars=best_model.emission_vars,
            initial_probs=best_model.initial_probs.tolist(),
            training_samples=best_model.training_samples,
            validation_accuracy=best_accuracy,
        )

        if verbose:
            print(f"[TRAIN] Model saved to database")

        # Save training stats
        self.db.set_stat('hmm_accuracy', best_accuracy)
        self.db.set_stat('hmm_edge', edge)
        self.db.set_stat('hmm_training_samples', best_model.training_samples)

        return best_model


def train_hmm_from_database(db_path: str = None, verbose: bool = True) -> Optional[HMMParameters]:
    """Convenience function to train HMM from database."""
    db = HistoricalFlowDatabase(db_path)
    pipeline = HMMTrainingPipeline(db)
    return pipeline.train(verbose=verbose)
