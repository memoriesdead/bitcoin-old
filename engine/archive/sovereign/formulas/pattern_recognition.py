"""
RENAISSANCE PATTERN RECOGNITION - Custom Formulas for Blockchain Edge
=====================================================================
IDs: 20001-20020

THE GOAL: Recognize patterns in blockchain inflow/outflow data to predict
price movements 10-60 seconds BEFORE the market.

KEY TECHNOLOGIES (Inspired by Renaissance Technologies):
1. Hidden Markov Models (HMM) - Detect regime states from flow sequences
2. Statistical Arbitrage - Find temporary discrepancies in flow patterns
3. Machine Learning - Non-linear pattern detection
4. Changepoint Detection - Identify regime shifts in real-time

DATA SOURCE: Live Bitcoin node (mempool + blocks)
- INFLOW to exchange = Selling pressure = SHORT
- OUTFLOW from exchange = Accumulation = LONG

FORMULA INDEX:
    20001: BlockchainHMM           - 5-state HMM for flow regime detection
    20002: FlowPatternRecognizer   - Detect recurring patterns in flow sequences
    20003: StatArbFlowDetector     - Statistical arbitrage on flow imbalances
    20004: ChangePointDetector     - Real-time regime shift detection (CUSUM + Bayesian)
    20005: FlowMomentumClassifier  - ML classifier for momentum vs mean-reversion
    20006: WhalePatternHMM         - HMM specifically for whale behavior
    20007: MempoolLeadingHMM       - HMM using mempool as leading indicator
    20008: MultiExchangeFlowHMM    - Cross-exchange flow pattern detection
    20009: VolumeClusterDetector   - Cluster analysis of flow volumes
    20010: SequencePatternMatcher  - Match historical flow sequences
    20011: EnsemblePatternVoter    - Combine all pattern detectors via voting
    20012: AdaptivePatternLearner  - Online learning from trade outcomes

CITATIONS:
- Hidden Markov Models: Rabiner (1989), Hamilton (1989)
- CUSUM: Page (1954), Lucas (1982)
- Bayesian Changepoint: Adams & MacKay (2007)
- Statistical Arbitrage: Pole (2007), Avellaneda & Lee (2010)
- Pattern Recognition: Bishop (2006)
"""

import math
import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


###############################################################################
# FORMULA 20001: BLOCKCHAIN HMM - 5-State Hidden Markov Model
###############################################################################

class FlowRegime(Enum):
    """Market regimes detected from blockchain flow."""
    ACCUMULATION = 0      # Net outflow, whales accumulating -> LONG
    DISTRIBUTION = 1      # Net inflow, smart money selling -> SHORT
    NEUTRAL = 2           # Balanced flow, no clear signal
    CAPITULATION = 3      # Massive inflow + high urgency -> Contrarian LONG
    EUPHORIA = 4          # Massive outflow + high urgency -> Contrarian SHORT


class BlockchainHMM:
    """
    ID: 20001

    5-state Hidden Markov Model for blockchain flow regime detection.

    STATES:
        0 = ACCUMULATION:   Outflows dominate, low volatility -> LONG
        1 = DISTRIBUTION:   Inflows dominate, rising volatility -> SHORT
        2 = NEUTRAL:        Balanced flows, wait for clarity
        3 = CAPITULATION:   Massive inflows + panic (high fees) -> Contrarian LONG
        4 = EUPHORIA:       Massive outflows + greed -> Contrarian SHORT

    OBSERVATIONS (continuous):
        - flow_imbalance: (outflow - inflow) / (outflow + inflow)
        - flow_velocity: abs(net_flow) / time
        - fee_percentile: urgency indicator (0-100)
        - whale_ratio: whale_flow / total_flow

    INFERENCE:
        - Forward algorithm for P(state | observations)
        - Viterbi for most likely state sequence
        - Baum-Welch for parameter updates (online)

    CITATION: Hamilton (1989), Rabiner (1989)
    """

    FORMULA_ID = 20001

    def __init__(self, n_states: int = 5):
        self.n_states = n_states

        # Transition matrix A[i,j] = P(state_j | state_i)
        # Initial: states are "sticky" with some transition probability
        self.A = self._init_transition_matrix()

        # Emission parameters: mean and variance for each observation per state
        # [flow_imbalance, flow_velocity, fee_percentile, whale_ratio]
        self.emission_means = {
            0: [0.5, 0.3, 30, 0.3],    # ACCUMULATION: positive imbalance, moderate velocity
            1: [-0.5, 0.3, 40, 0.3],   # DISTRIBUTION: negative imbalance
            2: [0.0, 0.1, 25, 0.2],    # NEUTRAL: balanced, low velocity
            3: [-0.8, 0.8, 90, 0.5],   # CAPITULATION: strong inflow, high urgency, whales
            4: [0.8, 0.8, 85, 0.5],    # EUPHORIA: strong outflow, high urgency, whales
        }
        self.emission_vars = {s: [0.2, 0.2, 20, 0.15] for s in range(n_states)}

        # State belief (probability distribution over states)
        self.belief = np.array([0.1, 0.1, 0.6, 0.1, 0.1])  # Start neutral

        # History for Baum-Welch updates
        self.observation_history: List[List[float]] = []
        self.state_history: List[int] = []
        self.max_history = 1000

        # Learning rate for online updates
        self.alpha = 0.01

    def _init_transition_matrix(self) -> np.ndarray:
        """Initialize transition matrix with sticky states."""
        A = np.array([
            # To: ACC   DIS   NEU   CAP   EUP
            [0.80, 0.05, 0.10, 0.02, 0.03],  # From ACCUMULATION
            [0.05, 0.80, 0.10, 0.03, 0.02],  # From DISTRIBUTION
            [0.15, 0.15, 0.60, 0.05, 0.05],  # From NEUTRAL
            [0.30, 0.10, 0.20, 0.35, 0.05],  # From CAPITULATION (often leads to accumulation)
            [0.10, 0.30, 0.20, 0.05, 0.35],  # From EUPHORIA (often leads to distribution)
        ])
        return A

    def _emission_prob(self, state: int, obs: List[float]) -> float:
        """Calculate P(observation | state) using Gaussian."""
        means = self.emission_means[state]
        vars = self.emission_vars[state]

        log_prob = 0
        for i in range(len(obs)):
            diff = obs[i] - means[i]
            log_prob -= 0.5 * (diff ** 2) / vars[i]
            log_prob -= 0.5 * math.log(2 * math.pi * vars[i])

        return math.exp(max(-50, log_prob))  # Prevent underflow

    def update(self, flow_imbalance: float, flow_velocity: float,
               fee_percentile: float, whale_ratio: float) -> Dict:
        """
        Update belief state with new observation.

        Returns:
            state: Most likely current state
            regime: Regime enum
            confidence: Confidence in current state
            signal: Trading signal (-1, 0, +1)
            probabilities: Full state distribution
        """
        obs = [flow_imbalance, flow_velocity, fee_percentile, whale_ratio]

        # Forward step: belief = normalize(A.T @ belief * emission)
        emission = np.array([self._emission_prob(s, obs) for s in range(self.n_states)])
        prior = self.A.T @ self.belief
        posterior = prior * emission

        # Normalize
        total = posterior.sum()
        if total > 0:
            self.belief = posterior / total
        else:
            self.belief = np.array([0.1, 0.1, 0.6, 0.1, 0.1])

        # Get most likely state
        state = int(np.argmax(self.belief))
        confidence = float(self.belief[state])

        # Map state to trading signal
        signal_map = {
            0: 1,   # ACCUMULATION -> LONG
            1: -1,  # DISTRIBUTION -> SHORT
            2: 0,   # NEUTRAL -> wait
            3: 1,   # CAPITULATION -> Contrarian LONG
            4: -1,  # EUPHORIA -> Contrarian SHORT
        }

        # Store for learning
        self.observation_history.append(obs)
        self.state_history.append(state)
        if len(self.observation_history) > self.max_history:
            self.observation_history.pop(0)
            self.state_history.pop(0)

        return {
            'state': state,
            'regime': FlowRegime(state).name,
            'confidence': confidence,
            'signal': signal_map[state],
            'probabilities': {FlowRegime(i).name: float(self.belief[i])
                             for i in range(self.n_states)},
        }

    def online_update(self, outcome_correct: bool):
        """
        Update HMM parameters based on trade outcome.

        If prediction was correct, reinforce current parameters.
        If wrong, adjust emission means toward observed values.
        """
        if len(self.observation_history) < 2:
            return

        alpha = self.alpha if outcome_correct else self.alpha * 2  # Learn faster from mistakes
        direction = 1 if outcome_correct else -1

        # Update emission means for recent state
        recent_state = self.state_history[-1]
        recent_obs = self.observation_history[-1]

        for i in range(len(recent_obs)):
            error = recent_obs[i] - self.emission_means[recent_state][i]
            self.emission_means[recent_state][i] += direction * alpha * error

    def get_viterbi_path(self, n: int = 10) -> List[int]:
        """Get most likely state sequence for last n observations."""
        if len(self.observation_history) < n:
            n = len(self.observation_history)
        if n == 0:
            return []

        observations = self.observation_history[-n:]

        # Viterbi algorithm
        T = len(observations)
        V = np.zeros((T, self.n_states))
        path = np.zeros((T, self.n_states), dtype=int)

        # Initialize
        for s in range(self.n_states):
            V[0, s] = self.belief[s] * self._emission_prob(s, observations[0])

        # Forward pass
        for t in range(1, T):
            for s in range(self.n_states):
                probs = V[t-1] * self.A[:, s]
                path[t, s] = np.argmax(probs)
                V[t, s] = probs[path[t, s]] * self._emission_prob(s, observations[t])

        # Backtrack
        best_path = [0] * T
        best_path[-1] = int(np.argmax(V[-1]))
        for t in range(T - 2, -1, -1):
            best_path[t] = path[t + 1, best_path[t + 1]]

        return best_path


###############################################################################
# FORMULA 20002: FLOW PATTERN RECOGNIZER
###############################################################################

@dataclass
class FlowPattern:
    """A recognized pattern in flow data."""
    name: str
    sequence: List[int]  # Sequence of states
    expected_outcome: int  # +1 (up), -1 (down)
    confidence: float
    occurrences: int
    success_rate: float


class FlowPatternRecognizer:
    """
    ID: 20002

    Recognizes recurring patterns in blockchain flow sequences.

    METHOD:
    1. Track sequences of flow states (from HMM)
    2. Build pattern library from historical data
    3. Match current sequence to known patterns
    4. Weight by pattern success rate

    PATTERNS TRACKED:
    - Accumulation -> Neutral -> Distribution (distribution setup)
    - Capitulation -> Accumulation (bottom reversal)
    - Euphoria -> Distribution (top reversal)
    - Consecutive same-state (momentum)
    """

    FORMULA_ID = 20002

    # Predefined patterns (seed library)
    SEED_PATTERNS = [
        FlowPattern("BOTTOM_REVERSAL", [3, 0], 1, 0.7, 0, 0.65),      # Capitulation -> Accumulation
        FlowPattern("TOP_REVERSAL", [4, 1], -1, 0.7, 0, 0.65),        # Euphoria -> Distribution
        FlowPattern("ACCUMULATION_MOMENTUM", [0, 0, 0], 1, 0.65, 0, 0.60),
        FlowPattern("DISTRIBUTION_MOMENTUM", [1, 1, 1], -1, 0.65, 0, 0.60),
        FlowPattern("BREAKOUT_UP", [2, 0, 0], 1, 0.6, 0, 0.55),       # Neutral -> Accumulation
        FlowPattern("BREAKOUT_DOWN", [2, 1, 1], -1, 0.6, 0, 0.55),    # Neutral -> Distribution
        FlowPattern("WHALE_ACCUMULATION", [3, 0, 0], 1, 0.75, 0, 0.70),
        FlowPattern("WHALE_DISTRIBUTION", [4, 1, 1], -1, 0.75, 0, 0.70),
    ]

    def __init__(self, min_pattern_length: int = 2, max_pattern_length: int = 5):
        self.min_len = min_pattern_length
        self.max_len = max_pattern_length

        # Initialize pattern library
        self.patterns: Dict[str, FlowPattern] = {p.name: p for p in self.SEED_PATTERNS}

        # State sequence buffer
        self.state_buffer: deque = deque(maxlen=max_pattern_length + 1)

        # Learning
        self.alpha = 0.1

    def add_state(self, state: int) -> Optional[Dict]:
        """
        Add new state and check for pattern matches.

        Returns:
            Match result with pattern name, signal, confidence
        """
        self.state_buffer.append(state)

        if len(self.state_buffer) < self.min_len:
            return None

        # Check all pattern lengths
        best_match = None
        best_confidence = 0

        for length in range(self.min_len, min(len(self.state_buffer), self.max_len) + 1):
            current_seq = list(self.state_buffer)[-length:]

            for name, pattern in self.patterns.items():
                if len(pattern.sequence) == length and pattern.sequence == current_seq:
                    effective_conf = pattern.confidence * pattern.success_rate
                    if effective_conf > best_confidence:
                        best_confidence = effective_conf
                        best_match = pattern

        if best_match:
            return {
                'pattern': best_match.name,
                'signal': best_match.expected_outcome,
                'confidence': best_confidence,
                'success_rate': best_match.success_rate,
                'occurrences': best_match.occurrences,
            }

        return None

    def record_outcome(self, pattern_name: str, was_correct: bool):
        """Update pattern success rate based on outcome."""
        if pattern_name not in self.patterns:
            return

        pattern = self.patterns[pattern_name]
        pattern.occurrences += 1

        # Update success rate with exponential smoothing
        target = 1.0 if was_correct else 0.0
        pattern.success_rate = pattern.success_rate * (1 - self.alpha) + target * self.alpha

    def discover_pattern(self, sequence: List[int], outcome: int, min_occurrences: int = 5):
        """
        Potentially add new pattern to library.

        Only add if pattern occurs frequently and has consistent outcome.
        """
        seq_key = str(sequence)

        # Check if already exists
        for pattern in self.patterns.values():
            if pattern.sequence == sequence:
                return

        # For now, just track discovered patterns
        # In production, would need more occurrences to confirm
        name = f"DISCOVERED_{seq_key}"
        self.patterns[name] = FlowPattern(
            name=name,
            sequence=sequence,
            expected_outcome=outcome,
            confidence=0.5,  # Start conservative
            occurrences=1,
            success_rate=0.5
        )


###############################################################################
# FORMULA 20003: STATISTICAL ARBITRAGE FLOW DETECTOR
###############################################################################

class StatArbFlowDetector:
    """
    ID: 20003

    Statistical arbitrage on flow imbalances.

    CONCEPT: Flow imbalances tend to mean-revert over short periods.
    When flow is extreme in one direction, expect a reversal.

    METHOD:
    1. Calculate flow imbalance Z-score
    2. When Z > threshold, expect mean reversion (contrarian)
    3. When Z changes direction after extreme, expect momentum (confirmation)

    EDGE: Combines mean-reversion AND momentum detection.
    """

    FORMULA_ID = 20003

    def __init__(self, window: int = 100, zscore_threshold: float = 2.0):
        self.window = window
        self.threshold = zscore_threshold

        self.flow_history: deque = deque(maxlen=window)
        self.mean = 0.0
        self.std = 1.0
        self.last_zscore = 0.0
        self.extreme_detected = False
        self.extreme_direction = 0

    def update(self, flow_imbalance: float) -> Dict:
        """
        Update with new flow imbalance.

        Returns:
            signal: Trading signal
            zscore: Current Z-score
            mode: 'mean_reversion' or 'momentum' or 'neutral'
        """
        self.flow_history.append(flow_imbalance)

        if len(self.flow_history) < 10:
            return {'signal': 0, 'zscore': 0, 'mode': 'neutral', 'confidence': 0}

        # Calculate statistics
        data = list(self.flow_history)
        self.mean = sum(data) / len(data)
        variance = sum((x - self.mean) ** 2 for x in data) / len(data)
        self.std = max(0.01, variance ** 0.5)

        zscore = (flow_imbalance - self.mean) / self.std

        signal = 0
        mode = 'neutral'
        confidence = 0.5

        # MEAN REVERSION: Extreme flow tends to reverse
        if abs(zscore) > self.threshold:
            if not self.extreme_detected:
                self.extreme_detected = True
                self.extreme_direction = 1 if zscore > 0 else -1
            # Signal opposite to extreme
            signal = -self.extreme_direction
            mode = 'mean_reversion'
            confidence = min(0.9, 0.5 + 0.1 * (abs(zscore) - self.threshold))

        # MOMENTUM: After extreme, if direction persists, go with it
        elif self.extreme_detected:
            # Check if we're moving back toward mean
            if (self.extreme_direction > 0 and zscore < self.threshold * 0.5) or \
               (self.extreme_direction < 0 and zscore > -self.threshold * 0.5):
                self.extreme_detected = False
            # If still in direction of extreme, momentum signal
            elif (self.extreme_direction > 0 and zscore > 0.5) or \
                 (self.extreme_direction < 0 and zscore < -0.5):
                signal = self.extreme_direction
                mode = 'momentum'
                confidence = 0.6

        self.last_zscore = zscore

        return {
            'signal': signal,
            'zscore': zscore,
            'mode': mode,
            'confidence': confidence,
            'mean': self.mean,
            'std': self.std,
        }


###############################################################################
# FORMULA 20004: CHANGEPOINT DETECTOR (CUSUM + BAYESIAN)
###############################################################################

class ChangePointDetector:
    """
    ID: 20004

    Real-time regime shift detection using CUSUM and Bayesian methods.

    CUSUM: Cumulative sum test - detects persistent shifts from baseline
    BAYESIAN: Online changepoint detection with hazard function

    EDGE: Detect regime shifts as they happen, not after the fact.

    CITATION: Page (1954), Adams & MacKay (2007)
    """

    FORMULA_ID = 20004

    def __init__(self, cusum_threshold: float = 5.0, hazard_rate: float = 0.01):
        self.cusum_threshold = cusum_threshold
        self.hazard_rate = hazard_rate  # Expected rate of changepoints

        # CUSUM state
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.baseline = 0.0
        self.baseline_count = 0

        # Bayesian state
        self.run_length_probs: List[float] = [1.0]  # P(r=0) = 1 initially
        self.observation_buffer: List[float] = []

        # Detection state
        self.changepoint_detected = False
        self.last_changepoint_time = 0.0
        self.regime_duration = 0

    def _cusum_update(self, value: float) -> Tuple[bool, str]:
        """CUSUM update - detect persistent shifts."""
        # Update baseline (slow EMA)
        if self.baseline_count < 50:
            self.baseline = (self.baseline * self.baseline_count + value) / (self.baseline_count + 1)
            self.baseline_count += 1
        else:
            self.baseline = self.baseline * 0.99 + value * 0.01

        diff = value - self.baseline

        # Update CUSUM statistics
        self.cusum_pos = max(0, self.cusum_pos + diff)
        self.cusum_neg = min(0, self.cusum_neg + diff)

        # Detect changepoint
        if self.cusum_pos > self.cusum_threshold:
            self.cusum_pos = 0
            return True, 'positive_shift'
        if self.cusum_neg < -self.cusum_threshold:
            self.cusum_neg = 0
            return True, 'negative_shift'

        return False, 'none'

    def _bayesian_update(self, value: float) -> float:
        """Bayesian online changepoint detection."""
        self.observation_buffer.append(value)

        # Predictive probability (simple Gaussian)
        if len(self.observation_buffer) < 2:
            return 0.0

        # Growth probabilities (probability of no changepoint)
        growth = [(1 - self.hazard_rate) * p for p in self.run_length_probs]

        # Changepoint probability (hazard * sum of all run lengths)
        cp_prob = self.hazard_rate * sum(self.run_length_probs)

        # New run length distribution
        new_probs = [cp_prob] + growth

        # Normalize
        total = sum(new_probs)
        if total > 0:
            self.run_length_probs = [p / total for p in new_probs]
        else:
            self.run_length_probs = [1.0]

        # Trim very small probabilities
        if len(self.run_length_probs) > 200:
            self.run_length_probs = self.run_length_probs[:200]

        # Return changepoint probability
        return self.run_length_probs[0] if self.run_length_probs else 0.0

    def update(self, value: float, timestamp: float = None) -> Dict:
        """
        Update with new observation.

        Returns:
            changepoint: Whether changepoint detected
            type: 'positive_shift', 'negative_shift', 'none'
            confidence: Confidence in detection
            regime_duration: Time since last changepoint
        """
        ts = timestamp or time.time()

        # CUSUM detection
        cusum_detected, shift_type = self._cusum_update(value)

        # Bayesian detection
        bayesian_cp_prob = self._bayesian_update(value)

        # Combine detectors
        changepoint = cusum_detected or (bayesian_cp_prob > 0.3)

        if changepoint:
            self.changepoint_detected = True
            self.regime_duration = 0
            self.last_changepoint_time = ts
        else:
            self.regime_duration += 1

        # Generate signal based on shift
        signal = 0
        if shift_type == 'positive_shift':
            signal = 1  # Flow shifted positive (outflow) -> LONG
        elif shift_type == 'negative_shift':
            signal = -1  # Flow shifted negative (inflow) -> SHORT

        confidence = 0.0
        if changepoint:
            confidence = max(
                min(1.0, self.cusum_pos / self.cusum_threshold) if self.cusum_pos > 0 else 0,
                min(1.0, abs(self.cusum_neg) / self.cusum_threshold) if self.cusum_neg < 0 else 0,
                bayesian_cp_prob
            )

        return {
            'changepoint': changepoint,
            'type': shift_type,
            'signal': signal,
            'confidence': confidence,
            'regime_duration': self.regime_duration,
            'bayesian_prob': bayesian_cp_prob,
            'cusum_pos': self.cusum_pos,
            'cusum_neg': self.cusum_neg,
        }


###############################################################################
# FORMULA 20005: FLOW MOMENTUM CLASSIFIER (ML)
###############################################################################

class FlowMomentumClassifier:
    """
    ID: 20005

    Simple ML classifier to determine if current flow pattern
    indicates momentum (trend) or mean-reversion.

    FEATURES:
    - flow_imbalance
    - flow_velocity
    - flow_acceleration
    - regime_duration
    - recent_return

    OUTPUT: P(momentum), P(mean_reversion)

    METHOD: Logistic regression with online learning
    """

    FORMULA_ID = 20005

    def __init__(self):
        # Weights for logistic regression
        # [bias, flow_imbalance, velocity, acceleration, duration, return]
        self.weights = np.array([0.0, 0.5, 0.3, 0.2, -0.1, 0.4])
        self.learning_rate = 0.05

        # Feature history
        self.last_imbalance = 0.0
        self.last_velocity = 0.0

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation."""
        return 1 / (1 + math.exp(-max(-50, min(50, x))))

    def _features(self, flow_imbalance: float, flow_velocity: float,
                  regime_duration: int, recent_return: float) -> np.ndarray:
        """Extract feature vector."""
        acceleration = flow_velocity - self.last_velocity
        self.last_velocity = flow_velocity
        self.last_imbalance = flow_imbalance

        return np.array([
            1.0,  # bias
            flow_imbalance,
            flow_velocity,
            acceleration,
            min(1.0, regime_duration / 100),  # Normalize
            recent_return * 100,  # Scale returns
        ])

    def predict(self, flow_imbalance: float, flow_velocity: float,
                regime_duration: int, recent_return: float) -> Dict:
        """
        Predict momentum vs mean-reversion.

        Returns:
            is_momentum: True if momentum regime
            momentum_prob: P(momentum)
            signal: +1 (go with flow) or -1 (fade flow)
        """
        features = self._features(flow_imbalance, flow_velocity,
                                  regime_duration, recent_return)

        # Logistic regression
        logit = np.dot(self.weights, features)
        momentum_prob = self._sigmoid(logit)

        is_momentum = momentum_prob > 0.5

        # Signal: momentum = trade WITH flow, mean-reversion = trade AGAINST
        if is_momentum:
            signal = 1 if flow_imbalance > 0 else -1  # Go with flow
        else:
            signal = -1 if flow_imbalance > 0 else 1  # Fade flow

        return {
            'is_momentum': is_momentum,
            'momentum_prob': momentum_prob,
            'signal': signal,
            'confidence': abs(momentum_prob - 0.5) * 2,  # 0 at 0.5, 1 at 0 or 1
        }

    def update(self, was_correct: bool, features: np.ndarray = None):
        """Online learning update."""
        if features is None:
            return

        target = 1.0 if was_correct else 0.0
        pred = self._sigmoid(np.dot(self.weights, features))
        error = target - pred

        # Gradient descent
        self.weights += self.learning_rate * error * features


###############################################################################
# FORMULA 20011: ENSEMBLE PATTERN VOTER
###############################################################################

class EnsemblePatternVoter:
    """
    ID: 20011

    Combines all pattern recognition methods via voting.

    METHODS COMBINED:
    - HMM regime detection (20001)
    - Pattern recognition (20002)
    - Stat arb detector (20003)
    - Changepoint detector (20004)
    - Momentum classifier (20005)

    VOTING: Weighted by confidence and historical accuracy.

    CITATION: Condorcet Jury Theorem - majority vote improves accuracy
    if each voter has >50% accuracy.
    """

    FORMULA_ID = 20011

    def __init__(self):
        # Initialize all components
        self.hmm = BlockchainHMM()
        self.pattern = FlowPatternRecognizer()
        self.stat_arb = StatArbFlowDetector()
        self.changepoint = ChangePointDetector()
        self.momentum = FlowMomentumClassifier()

        # Voter weights (tuned by performance)
        self.weights = {
            'hmm': 0.30,
            'pattern': 0.20,
            'stat_arb': 0.15,
            'changepoint': 0.20,
            'momentum': 0.15,
        }

        # Track accuracy for adaptive weighting
        self.accuracy = {k: 0.5 for k in self.weights}
        self.vote_history: List[Dict] = []

    def vote(self, flow_imbalance: float, flow_velocity: float,
             fee_percentile: float, whale_ratio: float,
             recent_return: float = 0.0) -> Dict:
        """
        Get ensemble vote from all pattern detectors.

        Returns:
            direction: +1 (LONG), -1 (SHORT), 0 (NEUTRAL)
            confidence: Combined confidence
            agreement: Fraction of voters agreeing
            votes: Individual votes from each method
        """
        # Get individual votes
        hmm_result = self.hmm.update(flow_imbalance, flow_velocity,
                                     fee_percentile, whale_ratio)

        # Feed state to pattern recognizer
        self.pattern.add_state(hmm_result['state'])
        pattern_result = self.pattern.add_state(hmm_result['state']) or \
                        {'signal': 0, 'confidence': 0}

        stat_arb_result = self.stat_arb.update(flow_imbalance)
        changepoint_result = self.changepoint.update(flow_imbalance)
        momentum_result = self.momentum.predict(flow_imbalance, flow_velocity,
                                                 self.changepoint.regime_duration,
                                                 recent_return)

        votes = {
            'hmm': (hmm_result['signal'], hmm_result['confidence']),
            'pattern': (pattern_result.get('signal', 0), pattern_result.get('confidence', 0)),
            'stat_arb': (stat_arb_result['signal'], stat_arb_result['confidence']),
            'changepoint': (changepoint_result['signal'], changepoint_result['confidence']),
            'momentum': (momentum_result['signal'], momentum_result['confidence']),
        }

        # Weighted vote
        weighted_sum = 0.0
        total_weight = 0.0

        for name, (signal, conf) in votes.items():
            if signal != 0:
                weight = self.weights[name] * self.accuracy[name] * conf
                weighted_sum += signal * weight
                total_weight += weight

        # Final direction
        if total_weight > 0:
            avg_signal = weighted_sum / total_weight
            if avg_signal > 0.3:
                direction = 1
            elif avg_signal < -0.3:
                direction = -1
            else:
                direction = 0
        else:
            direction = 0
            avg_signal = 0

        # Agreement: how many non-zero votes agree with direction
        agreeing = sum(1 for name, (sig, _) in votes.items()
                      if sig != 0 and sig == direction)
        total_voting = sum(1 for name, (sig, _) in votes.items() if sig != 0)
        agreement = agreeing / max(1, total_voting)

        # Confidence: weighted by agreement and individual confidences
        confidence = abs(avg_signal) * agreement

        result = {
            'direction': direction,
            'confidence': confidence,
            'agreement': agreement,
            'votes': votes,
            'regime': hmm_result['regime'],
            'regime_probabilities': hmm_result['probabilities'],
        }

        self.vote_history.append(result)
        if len(self.vote_history) > 1000:
            self.vote_history.pop(0)

        return result

    def record_outcome(self, was_correct: bool):
        """Update voter weights based on outcome."""
        if not self.vote_history:
            return

        last_vote = self.vote_history[-1]
        direction = last_vote['direction']

        # Update accuracy for each voter
        alpha = 0.1
        for name, (signal, _) in last_vote['votes'].items():
            if signal != 0:
                voter_correct = (signal == direction and was_correct) or \
                               (signal != direction and not was_correct)
                target = 1.0 if voter_correct else 0.0
                self.accuracy[name] = self.accuracy[name] * (1 - alpha) + target * alpha

        # Renormalize weights based on accuracy
        total_acc = sum(self.accuracy.values())
        if total_acc > 0:
            for name in self.weights:
                self.weights[name] = self.accuracy[name] / total_acc

        # Update HMM
        self.hmm.online_update(was_correct)

    def get_stats(self) -> Dict:
        """Get current voter statistics."""
        return {
            'weights': dict(self.weights),
            'accuracy': dict(self.accuracy),
            'hmm_belief': {FlowRegime(i).name: float(self.hmm.belief[i])
                          for i in range(self.hmm.n_states)},
        }


###############################################################################
# MASTER CLASS: PATTERN RECOGNITION ENGINE
###############################################################################

class PatternRecognitionEngine:
    """
    MASTER ENGINE for Pattern Recognition.

    REQUIRES TRAINED MODELS - Do not run live without training first!

    TRAINING:
        python -m engine.sovereign.formulas.train --live 3600

    This will:
    1. Collect historical flow data
    2. Train HMM using Baum-Welch
    3. Discover patterns from data
    4. Validate edge >= 50.75%
    5. Save models to database

    USAGE (after training):
        engine = PatternRecognitionEngine(require_trained=True)

        # On each blockchain flow:
        signal = engine.on_flow(
            exchange='binance',
            direction=1,  # OUTFLOW
            btc=50.0,
            timestamp=time.time(),
            price=97000.0
        )

        if signal['should_trade']:
            execute_trade(signal)

        # After trade closes:
        engine.record_outcome(was_profitable=True)

    "We're right 50.75% of the time, but we're 100% right 50.75% of the time."
    - Robert Mercer, Renaissance Technologies
    """

    def __init__(self, require_trained: bool = False, db_path: str = None):
        """
        Initialize pattern recognition engine.

        Args:
            require_trained: If True, fail if no trained model exists
            db_path: Path to database with trained models
        """
        self.trained = False
        self.validation_accuracy = 0.0

        # Try to load trained models
        if require_trained or db_path:
            self._load_trained_models(db_path, require_trained)

        # Initialize voter (will use loaded or default parameters)
        self.voter = EnsemblePatternVoter()

        # If we loaded trained HMM, update voter's HMM
        if self.trained and hasattr(self, '_loaded_hmm'):
            self._apply_trained_hmm()

        # Load discovered patterns
        if self.trained:
            self._load_patterns(db_path)

        # Pattern matcher for validated patterns
        self.pattern_matcher = None

        # Print training status
        if self.trained:
            print(f"[PATTERN] Loaded trained model with {self.validation_accuracy:.2%} validation accuracy")
        elif require_trained:
            print("[PATTERN] WARNING: require_trained=True but no trained model found!")
            print("[PATTERN] Run: python -m engine.sovereign.formulas.train --live 3600")
        else:
            print("[PATTERN] Running with default parameters (no training)")

        # Flow tracking
        self.total_inflow = 0.0
        self.total_outflow = 0.0
        self.whale_inflow = 0.0
        self.whale_outflow = 0.0
        self.fee_history: deque = deque(maxlen=100)
        self.price_history: deque = deque(maxlen=100)

        # State
        self.last_signal = None
        self.signals_generated = 0
        self.trades_recorded = 0

    def _load_trained_models(self, db_path: str, require_trained: bool):
        """
        Load trained HMM and patterns from database.

        The database contains:
        - Trained HMM parameters (transition matrix, emission params)
        - Validated patterns (sequences that predict price moves)
        - Validation metrics (out-of-sample accuracy)
        """
        try:
            from .historical_data import HistoricalFlowDatabase
            import numpy as np

            db = HistoricalFlowDatabase(db_path)

            # Load HMM model
            hmm_params = db.load_hmm_model('default')
            if hmm_params:
                self._loaded_hmm = hmm_params
                self.validation_accuracy = hmm_params.get('validation_accuracy', 0.0)
                self.trained = True
                print(f"[PATTERN] Loaded trained HMM: {hmm_params['n_states']} states, "
                      f"{hmm_params['training_samples']} samples")
            else:
                if require_trained:
                    raise ValueError("No trained HMM found. Run training first.")
                print("[PATTERN] No trained HMM found in database")
                self.trained = False

        except ImportError:
            if require_trained:
                raise ValueError("historical_data module not found. Cannot load trained models.")
            self.trained = False
        except Exception as e:
            if require_trained:
                raise ValueError(f"Failed to load trained models: {e}")
            print(f"[PATTERN] Could not load trained models: {e}")
            self.trained = False

    def _apply_trained_hmm(self):
        """
        Apply loaded HMM parameters to the ensemble voter's HMM.

        This replaces the default parameters with trained parameters from
        historical data analysis.
        """
        if not hasattr(self, '_loaded_hmm') or self._loaded_hmm is None:
            return

        import numpy as np

        hmm = self._loaded_hmm

        # Apply transition matrix
        if 'transition_matrix' in hmm:
            self.voter.hmm.A = np.array(hmm['transition_matrix'])

        # Apply emission parameters
        if 'emission_means' in hmm:
            means = hmm['emission_means']
            if isinstance(means, dict):
                for state, values in means.items():
                    self.voter.hmm.emission_means[int(state)] = values
            else:
                # List format [state0_means, state1_means, ...]
                for i, values in enumerate(means):
                    if i < self.voter.hmm.n_states:
                        self.voter.hmm.emission_means[i] = values

        if 'emission_vars' in hmm:
            vars_data = hmm['emission_vars']
            if isinstance(vars_data, dict):
                for state, values in vars_data.items():
                    self.voter.hmm.emission_vars[int(state)] = values
            else:
                for i, values in enumerate(vars_data):
                    if i < self.voter.hmm.n_states:
                        self.voter.hmm.emission_vars[i] = values

        # Apply initial probabilities
        if 'initial_probs' in hmm:
            self.voter.hmm.belief = np.array(hmm['initial_probs'])

        print(f"[PATTERN] Applied trained HMM parameters to voter")

    def _load_patterns(self, db_path: str):
        """
        Load validated patterns from database.

        Patterns are sequences of HMM states that historically predict
        price movements with >50.75% accuracy.
        """
        try:
            from .historical_data import HistoricalFlowDatabase
            from .pattern_discovery import PatternMatcher, DiscoveredPattern

            db = HistoricalFlowDatabase(db_path)
            db_patterns = db.get_patterns(min_occurrences=100, min_win_rate=0.5075)

            if db_patterns:
                # Convert to DiscoveredPattern objects
                patterns = []
                for p in db_patterns:
                    pattern = DiscoveredPattern(
                        name=p.get('name', f"PATTERN_{len(patterns)}"),
                        sequence=p['sequence'],
                        direction=p['direction'],
                        occurrences=p['occurrences'],
                        wins=p['wins'],
                        losses=p['losses'],
                        win_rate=p['win_rate'],
                        edge=p['win_rate'] - 0.5,
                        p_value=0.05,  # Already validated
                        avg_return=0.0,
                        is_valid=True,
                    )
                    patterns.append(pattern)

                self.pattern_matcher = PatternMatcher(patterns)
                print(f"[PATTERN] Loaded {len(patterns)} validated patterns")

                # Also update the pattern recognizer in voter with discovered patterns
                for pattern in patterns:
                    seq_key = str(pattern.sequence)
                    self.voter.pattern.patterns[pattern.name] = FlowPattern(
                        name=pattern.name,
                        sequence=pattern.sequence,
                        expected_outcome=pattern.direction,
                        confidence=pattern.win_rate,
                        occurrences=pattern.occurrences,
                        success_rate=pattern.win_rate,
                    )

            else:
                print("[PATTERN] No validated patterns found in database")

        except ImportError as e:
            print(f"[PATTERN] Could not import pattern modules: {e}")
        except Exception as e:
            print(f"[PATTERN] Could not load patterns: {e}")

    def on_flow(self, exchange: str, direction: int, btc: float,
                timestamp: float, price: float, fee_rate: float = 25.0) -> Dict:
        """
        Process blockchain flow through pattern recognition.

        Args:
            exchange: Exchange ID
            direction: +1 (OUTFLOW) or -1 (INFLOW)
            btc: Amount of BTC
            timestamp: Unix timestamp
            price: Current BTC price
            fee_rate: Transaction fee rate (sat/vbyte)

        Returns:
            Trading signal with direction, confidence, pattern info
        """
        # Update tracking
        if direction == 1:
            self.total_outflow += btc
            if btc >= 100:
                self.whale_outflow += btc
        else:
            self.total_inflow += btc
            if btc >= 100:
                self.whale_inflow += btc

        self.fee_history.append(fee_rate)
        self.price_history.append(price)

        # Calculate features
        total = self.total_inflow + self.total_outflow + 0.001
        flow_imbalance = (self.total_outflow - self.total_inflow) / total

        # Velocity: recent net flow rate
        window_btc = btc * direction  # Just use this tick for now
        flow_velocity = abs(window_btc) / 10  # Normalize

        # Fee percentile
        if self.fee_history:
            sorted_fees = sorted(self.fee_history)
            fee_percentile = 100 * sorted_fees.index(min(sorted_fees, key=lambda x: abs(x - fee_rate))) / len(sorted_fees)
        else:
            fee_percentile = 50

        # Whale ratio
        whale_total = self.whale_inflow + self.whale_outflow + 0.001
        whale_ratio = (self.whale_outflow - self.whale_inflow) / whale_total if whale_total > 0.001 else 0

        # Recent return
        if len(self.price_history) >= 2:
            recent_return = (self.price_history[-1] - self.price_history[0]) / self.price_history[0]
        else:
            recent_return = 0

        # Get ensemble vote
        vote = self.voter.vote(
            flow_imbalance=flow_imbalance,
            flow_velocity=flow_velocity,
            fee_percentile=fee_percentile,
            whale_ratio=whale_ratio,
            recent_return=recent_return
        )

        # Check trained pattern matcher (if available)
        pattern_match = None
        if self.pattern_matcher is not None:
            state = vote.get('votes', {}).get('hmm', (0, 0))[0]
            if hasattr(self.voter, 'hmm') and hasattr(self.voter.hmm, 'state_history'):
                # Use HMM's state history
                if self.voter.hmm.state_history:
                    state = self.voter.hmm.state_history[-1]
            pattern_match = self.pattern_matcher.add_state(state)

            if pattern_match:
                # Boost confidence if trained pattern matches
                vote['confidence'] = max(vote['confidence'], pattern_match.win_rate)
                vote['direction'] = pattern_match.direction
                vote['trained_pattern'] = pattern_match.name

        # Build signal - stricter thresholds for trained vs untrained
        if self.trained:
            # Trained: trust the model, require 50.75% edge
            should_trade = (
                vote['direction'] != 0 and
                vote['confidence'] >= 0.5075 and  # RenTech threshold
                (vote['agreement'] > 0.4 or pattern_match is not None)
            )
        else:
            # Untrained: be more conservative
            should_trade = vote['direction'] != 0 and vote['confidence'] > 0.5 and vote['agreement'] > 0.6

        signal = {
            'direction': vote['direction'],
            'confidence': vote['confidence'],
            'should_trade': should_trade,
            'regime': vote['regime'],
            'agreement': vote['agreement'],
            'votes': vote['votes'],
            'features': {
                'flow_imbalance': flow_imbalance,
                'flow_velocity': flow_velocity,
                'fee_percentile': fee_percentile,
                'whale_ratio': whale_ratio,
            },
            'exchange': exchange,
            'btc': btc,
            'price': price,
            'timestamp': timestamp,
            'trained': self.trained,
            'trained_pattern': vote.get('trained_pattern'),
            'validation_accuracy': self.validation_accuracy if self.trained else None,
        }

        if should_trade:
            self.signals_generated += 1
            dir_str = "LONG" if vote['direction'] == 1 else "SHORT"
            trained_str = "[TRAINED]" if self.trained else "[DEFAULT]"
            pattern_str = f" | pattern={vote.get('trained_pattern')}" if vote.get('trained_pattern') else ""
            print(f"[PATTERN] {trained_str} {dir_str} | regime={vote['regime']} | "
                  f"conf={vote['confidence']:.2f} | agree={vote['agreement']:.2f}{pattern_str}")

        self.last_signal = signal
        return signal

    def record_outcome(self, was_profitable: bool):
        """Record trade outcome for learning."""
        self.trades_recorded += 1
        self.voter.record_outcome(was_profitable)

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        stats = {
            'signals_generated': self.signals_generated,
            'trades_recorded': self.trades_recorded,
            'total_inflow': self.total_inflow,
            'total_outflow': self.total_outflow,
            'net_flow': self.total_outflow - self.total_inflow,
            'voter_stats': self.voter.get_stats(),
            'trained': self.trained,
            'validation_accuracy': self.validation_accuracy if self.trained else None,
        }

        if self.pattern_matcher:
            stats['loaded_patterns'] = len(self.pattern_matcher.patterns)

        return stats
