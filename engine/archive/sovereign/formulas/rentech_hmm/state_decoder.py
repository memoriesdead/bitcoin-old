"""
HMM State Decoder and Analysis
==============================

Formula IDs: 72006-72010

Advanced state decoding and analysis for trading signals.
Builds on Gaussian HMM to extract additional trading insights.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import deque

from .gaussian_hmm import GaussianHMM, HMMConfig, TrainedHMMSignal


@dataclass
class StateAnalysis:
    """Analysis of HMM state characteristics."""
    state_id: int
    label: str
    avg_return: float
    avg_volatility: float
    avg_duration: float
    frequency: float
    transition_matrix_row: Dict[int, float]


class ViterbiDecoder:
    """
    Enhanced Viterbi decoder with confidence metrics.

    Goes beyond simple state assignment to provide:
    - Confidence in the decoded path
    - Alternative paths and their probabilities
    - Smoothed state estimates
    """

    def __init__(self, hmm: GaussianHMM):
        self.hmm = hmm

    def decode_with_confidence(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode state sequence with per-timestep confidence.

        Returns:
            states: Most likely state sequence
            confidence: Confidence (0-1) at each timestep
        """
        states = self.hmm.decode(observations)
        probs = self.hmm.predict_proba(observations)

        # Confidence is max probability minus second highest
        confidence = np.zeros(len(states))
        for t in range(len(states)):
            sorted_probs = np.sort(probs[t])[::-1]
            confidence[t] = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

        return states, confidence

    def get_state_boundaries(self, states: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Find state change boundaries.

        Returns:
            List of (start_idx, end_idx, state) tuples
        """
        boundaries = []
        if len(states) == 0:
            return boundaries

        start_idx = 0
        current_state = states[0]

        for i in range(1, len(states)):
            if states[i] != current_state:
                boundaries.append((start_idx, i, current_state))
                start_idx = i
                current_state = states[i]

        boundaries.append((start_idx, len(states), current_state))
        return boundaries


class OnlineStateInference:
    """
    Online (streaming) state inference for live trading.

    Maintains a sliding window of observations and updates
    state estimates in real-time.
    """

    def __init__(self, hmm: GaussianHMM, window_size: int = 100):
        self.hmm = hmm
        self.window_size = window_size
        self.observation_buffer: deque = deque(maxlen=window_size)
        self.state_history: List[int] = []

    def update(self, observation: np.ndarray) -> Tuple[int, float]:
        """
        Update with new observation and return current state estimate.

        Args:
            observation: New observation vector

        Returns:
            (state, probability)
        """
        self.observation_buffer.append(observation)

        if len(self.observation_buffer) < 10:
            return -1, 0.0

        observations = np.array(list(self.observation_buffer))
        state, prob = self.hmm.get_current_state(observations)

        self.state_history.append(state)
        return state, prob

    def get_regime_stability(self, lookback: int = 20) -> float:
        """
        Measure how stable the current regime is.

        Returns:
            Stability score 0-1 (1 = very stable, same state)
        """
        if len(self.state_history) < lookback:
            return 0.0

        recent = self.state_history[-lookback:]
        mode_count = max(recent.count(s) for s in set(recent))
        return mode_count / lookback


class StateTransitionAnalyzer:
    """
    Analyze state transitions for trading signals.

    Key insight: The TRANSITION between states often contains
    more information than the state itself.
    """

    def __init__(self, hmm: GaussianHMM):
        self.hmm = hmm
        self.transition_counts: Dict[Tuple[int, int], int] = {}
        self.post_transition_returns: Dict[Tuple[int, int], List[float]] = {}

    def analyze_transitions(self, observations: np.ndarray,
                           returns: np.ndarray) -> Dict[str, Any]:
        """
        Analyze transition patterns and their predictive power.

        Args:
            observations: Feature observations
            returns: Forward returns after each timestep

        Returns:
            Analysis results
        """
        states = self.hmm.decode(observations)

        for t in range(len(states) - 1):
            from_state = states[t]
            to_state = states[t + 1]

            key = (from_state, to_state)
            self.transition_counts[key] = self.transition_counts.get(key, 0) + 1

            if t + 1 < len(returns):
                if key not in self.post_transition_returns:
                    self.post_transition_returns[key] = []
                self.post_transition_returns[key].append(returns[t + 1])

        # Compute statistics
        results = {}
        for key, returns_list in self.post_transition_returns.items():
            if len(returns_list) >= 10:
                from_label = self.hmm.state_labels.get(key[0], f's{key[0]}')
                to_label = self.hmm.state_labels.get(key[1], f's{key[1]}')

                results[f"{from_label}->{to_label}"] = {
                    'count': self.transition_counts[key],
                    'avg_return': np.mean(returns_list),
                    'win_rate': sum(1 for r in returns_list if r > 0) / len(returns_list),
                    'sharpe': np.mean(returns_list) / (np.std(returns_list) + 1e-10),
                }

        return results


# =============================================================================
# FORMULA IMPLEMENTATIONS (72006-72010)
# =============================================================================

class ViterbiSignal:
    """
    Formula 72006: Viterbi Confidence Signal

    Trades when Viterbi decoder has high confidence in state assignment.
    Low confidence = uncertain regime = stay flat.
    """

    FORMULA_ID = 72006

    def __init__(self, n_states: int = 5, confidence_threshold: float = 0.3):
        self.hmm = GaussianHMM(HMMConfig(n_states=n_states))
        self.decoder: Optional[ViterbiDecoder] = None
        self.confidence_threshold = confidence_threshold

    def train(self, features: np.ndarray, feature_names: List[str] = None):
        stats = self.hmm.train(features, feature_names)
        self.decoder = ViterbiDecoder(self.hmm)
        return stats

    def generate_signal(self, features: np.ndarray) -> TrainedHMMSignal:
        if self.decoder is None:
            raise RuntimeError("Must train first")

        states, confidence = self.decoder.decode_with_confidence(features)
        current_confidence = confidence[-1]
        current_state = states[-1]

        if current_confidence < self.confidence_threshold:
            direction = 0
        else:
            label = self.hmm.state_labels.get(current_state, '')
            if 'bullish' in label:
                direction = 1
            elif 'bearish' in label:
                direction = -1
            else:
                direction = 0

        return TrainedHMMSignal(
            direction=direction,
            confidence=current_confidence,
            current_state=current_state,
            state_probability=current_confidence,
            transition_probs=self.hmm.get_transition_probs(current_state),
            expected_duration=self.hmm.get_expected_duration(current_state),
            features_used=self.hmm.feature_names,
        )


class TransitionProbSignal:
    """
    Formula 72007: Transition Probability Signal

    Signals based on where we're likely to go NEXT.
    If high probability of transitioning to bullish → LONG.
    """

    FORMULA_ID = 72007

    def __init__(self, n_states: int = 5):
        self.hmm = GaussianHMM(HMMConfig(n_states=n_states))

    def train(self, features: np.ndarray, feature_names: List[str] = None):
        return self.hmm.train(features, feature_names)

    def generate_signal(self, features: np.ndarray) -> TrainedHMMSignal:
        base = self.hmm.generate_signal(features)
        trans_probs = base.transition_probs

        # Compute expected direction from transition probabilities
        expected_direction = 0.0
        for state, prob in trans_probs.items():
            label = self.hmm.state_labels.get(state, '')
            if 'bullish' in label or 'euphoria' in label:
                expected_direction += prob
            elif 'bearish' in label or 'capitulation' in label:
                expected_direction -= prob

        if expected_direction > 0.2:
            direction = 1
        elif expected_direction < -0.2:
            direction = -1
        else:
            direction = 0

        return TrainedHMMSignal(
            direction=direction,
            confidence=abs(expected_direction),
            current_state=base.current_state,
            state_probability=base.state_probability,
            transition_probs=trans_probs,
            expected_duration=base.expected_duration,
            features_used=base.features_used,
        )


class StateDurationSignal:
    """
    Formula 72008: State Duration Signal

    Considers how long we've been in current state.
    - Early in bullish state → stronger LONG
    - Late in state (beyond expected duration) → reduce confidence
    """

    FORMULA_ID = 72008

    def __init__(self, n_states: int = 5):
        self.hmm = GaussianHMM(HMMConfig(n_states=n_states))
        self.state_entry_time: int = 0
        self.prev_state: int = -1
        self.current_duration: int = 0

    def train(self, features: np.ndarray, feature_names: List[str] = None):
        return self.hmm.train(features, feature_names)

    def generate_signal(self, features: np.ndarray) -> TrainedHMMSignal:
        base = self.hmm.generate_signal(features)

        # Track duration
        if base.current_state != self.prev_state:
            self.current_duration = 0
            self.prev_state = base.current_state
        else:
            self.current_duration += 1

        # Adjust confidence based on duration
        expected = base.expected_duration
        duration_ratio = self.current_duration / expected if expected > 0 else 0

        # Early in state = higher confidence, late = lower
        if duration_ratio < 0.5:
            confidence_mult = 1.2
        elif duration_ratio < 1.0:
            confidence_mult = 1.0
        elif duration_ratio < 1.5:
            confidence_mult = 0.7
        else:
            confidence_mult = 0.4  # Overstayed

        adjusted_confidence = min(1.0, base.confidence * confidence_mult)

        return TrainedHMMSignal(
            direction=base.direction,
            confidence=adjusted_confidence,
            current_state=base.current_state,
            state_probability=base.state_probability,
            transition_probs=base.transition_probs,
            expected_duration=base.expected_duration,
            features_used=base.features_used,
        )


class RegimePersistenceSignal:
    """
    Formula 72009: Regime Persistence Signal

    Measures regime stability over recent history.
    - Stable regime → trade with regime
    - Unstable regime → stay flat or fade
    """

    FORMULA_ID = 72009

    def __init__(self, n_states: int = 5, stability_threshold: float = 0.7):
        self.hmm = GaussianHMM(HMMConfig(n_states=n_states))
        self.online_inference: Optional[OnlineStateInference] = None
        self.stability_threshold = stability_threshold

    def train(self, features: np.ndarray, feature_names: List[str] = None):
        stats = self.hmm.train(features, feature_names)
        self.online_inference = OnlineStateInference(self.hmm)
        return stats

    def generate_signal(self, features: np.ndarray) -> TrainedHMMSignal:
        # Update online inference
        for obs in features[-20:]:  # Use last 20 observations
            self.online_inference.update(obs)

        stability = self.online_inference.get_regime_stability()
        base = self.hmm.generate_signal(features)

        if stability < self.stability_threshold:
            # Unstable regime - stay flat
            return TrainedHMMSignal(
                direction=0,
                confidence=stability,
                current_state=base.current_state,
                state_probability=base.state_probability,
                transition_probs=base.transition_probs,
                expected_duration=base.expected_duration,
                features_used=base.features_used,
            )
        else:
            # Stable regime - trade with confidence
            return TrainedHMMSignal(
                direction=base.direction,
                confidence=base.confidence * stability,
                current_state=base.current_state,
                state_probability=base.state_probability,
                transition_probs=base.transition_probs,
                expected_duration=base.expected_duration,
                features_used=base.features_used,
            )


class HMMEnsembleSignal:
    """
    Formula 72010: HMM Ensemble Signal

    Combines multiple HMM configurations:
    - 3-state, 5-state, 7-state
    - Different feature sets
    - Voting/averaging for final signal

    Key insight: Ensemble of HMMs is more robust than single HMM.
    """

    FORMULA_ID = 72010

    def __init__(self):
        self.hmms: List[GaussianHMM] = []
        self.weights: List[float] = []

    def train(self, features: np.ndarray, feature_names: List[str] = None):
        """Train multiple HMM configurations."""
        configs = [
            HMMConfig(n_states=3),
            HMMConfig(n_states=5),
            HMMConfig(n_states=7),
        ]

        total_ll = 0.0
        log_likelihoods = []

        for config in configs:
            hmm = GaussianHMM(config)
            stats = hmm.train(features, feature_names)
            self.hmms.append(hmm)
            log_likelihoods.append(stats['log_likelihood'])
            total_ll += stats['log_likelihood']

        # Weight by relative log-likelihood
        # Better fit = higher weight
        min_ll = min(log_likelihoods)
        self.weights = [(ll - min_ll + 1) for ll in log_likelihoods]
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        return {
            'n_models': len(self.hmms),
            'weights': self.weights,
            'log_likelihoods': log_likelihoods,
        }

    def generate_signal(self, features: np.ndarray) -> TrainedHMMSignal:
        if not self.hmms:
            raise RuntimeError("Must train first")

        # Collect signals from all HMMs
        directions = []
        confidences = []

        for hmm, weight in zip(self.hmms, self.weights):
            signal = hmm.generate_signal(features)
            directions.append(signal.direction * weight)
            confidences.append(signal.confidence * weight)

        # Weighted vote for direction
        avg_direction = sum(directions)
        if avg_direction > 0.3:
            final_direction = 1
        elif avg_direction < -0.3:
            final_direction = -1
        else:
            final_direction = 0

        # Average confidence
        avg_confidence = sum(confidences)

        # Use primary (5-state) HMM for state info
        primary_signal = self.hmms[1].generate_signal(features)

        return TrainedHMMSignal(
            direction=final_direction,
            confidence=avg_confidence,
            current_state=primary_signal.current_state,
            state_probability=primary_signal.state_probability,
            transition_probs=primary_signal.transition_probs,
            expected_duration=primary_signal.expected_duration,
            features_used=primary_signal.features_used,
        )
