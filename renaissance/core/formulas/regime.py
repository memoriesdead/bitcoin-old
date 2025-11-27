"""
Regime Detection Formulas (IDs 171-190)
=======================================
HMM, CUSUM, changepoint detection, and market state classification.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from .base import BaseFormula, FormulaRegistry


# =============================================================================
# HIDDEN MARKOV MODELS (171-180)
# =============================================================================

@FormulaRegistry.register(171)
class HMMRegimeDetector(BaseFormula):
    """ID 171: Hidden Markov Model for regime detection"""

    CATEGORY = "regime_detection"
    NAME = "HMMRegimeDetector"
    DESCRIPTION = "2-state HMM for bull/bear regimes"

    def __init__(self, lookback: int = 100, n_states: int = 2, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_states = n_states
        self.transition_matrix = np.array([[0.95, 0.05], [0.05, 0.95]])
        self.emission_means = np.array([0.001, -0.001])
        self.emission_stds = np.array([0.01, 0.02])
        self.state_probs = np.array([0.5, 0.5])
        self.current_state = 0

    def _forward_step(self, observation: float) -> None:
        emission_probs = np.zeros(self.n_states)
        for s in range(self.n_states):
            z = (observation - self.emission_means[s]) / (self.emission_stds[s] + 1e-10)
            emission_probs[s] = np.exp(-0.5 * z**2) / (self.emission_stds[s] * np.sqrt(2*np.pi))
        predicted = np.dot(self.state_probs, self.transition_matrix)
        self.state_probs = predicted * emission_probs
        self.state_probs /= (np.sum(self.state_probs) + 1e-10)
        self.current_state = np.argmax(self.state_probs)

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        observation = self.returns[-1]
        self._forward_step(observation)
        if self.current_state == 0:
            self.signal = 1
            self.confidence = self.state_probs[0]
        else:
            self.signal = -1
            self.confidence = self.state_probs[1]


@FormulaRegistry.register(172)
class BaumWelchHMM(BaseFormula):
    """ID 172: Baum-Welch trained HMM"""

    CATEGORY = "regime_detection"
    NAME = "BaumWelchHMM"
    DESCRIPTION = "HMM with online parameter updates"

    def __init__(self, lookback: int = 100, n_states: int = 2,
                 learning_rate: float = 0.01, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_states = n_states
        self.lr = learning_rate
        self.A = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.means = np.array([0.001, -0.001])
        self.stds = np.array([0.01, 0.015])
        self.pi = np.array([0.5, 0.5])
        self.gamma = np.array([0.5, 0.5])

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        obs = returns[-1]
        emission = np.zeros(self.n_states)
        for s in range(self.n_states):
            z = (obs - self.means[s]) / (self.stds[s] + 1e-10)
            emission[s] = np.exp(-0.5 * z**2)
        alpha = self.pi * emission
        alpha /= (alpha.sum() + 1e-10)
        self.gamma = alpha
        for s in range(self.n_states):
            if self.gamma[s] > 0.1:
                self.means[s] += self.lr * self.gamma[s] * (obs - self.means[s])
                self.stds[s] += self.lr * self.gamma[s] * (abs(obs - self.means[s]) - self.stds[s])
                self.stds[s] = max(self.stds[s], 0.001)
        best_state = np.argmax(self.gamma)
        if best_state == 0:
            self.signal = 1
            self.confidence = self.gamma[0]
        else:
            self.signal = -1
            self.confidence = self.gamma[1]


@FormulaRegistry.register(173)
class ViterbiDecoder(BaseFormula):
    """ID 173: Viterbi algorithm for optimal path"""

    CATEGORY = "regime_detection"
    NAME = "ViterbiDecoder"
    DESCRIPTION = "Most likely state sequence"

    def __init__(self, lookback: int = 100, n_states: int = 2, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_states = n_states
        self.A = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]) + 1e-10)
        self.means = np.array([0.001, -0.001])
        self.stds = np.array([0.01, 0.015])
        self.pi = np.log(np.array([0.5, 0.5]) + 1e-10)
        self.delta = self.pi.copy()
        self.path = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        obs = self.returns[-1]
        log_emission = np.zeros(self.n_states)
        for s in range(self.n_states):
            z = (obs - self.means[s]) / (self.stds[s] + 1e-10)
            log_emission[s] = -0.5 * z**2 - np.log(self.stds[s] * np.sqrt(2*np.pi))
        new_delta = np.zeros(self.n_states)
        for j in range(self.n_states):
            candidates = self.delta + self.A[:, j]
            new_delta[j] = np.max(candidates) + log_emission[j]
        self.delta = new_delta
        best_state = np.argmax(self.delta)
        self.path.append(best_state)
        if best_state == 0:
            self.signal = 1
        else:
            self.signal = -1
        prob = np.exp(self.delta - np.max(self.delta))
        prob /= prob.sum()
        self.confidence = prob[best_state]


@FormulaRegistry.register(174)
class GaussianHMM(BaseFormula):
    """ID 174: Gaussian emission HMM"""

    CATEGORY = "regime_detection"
    NAME = "GaussianHMM"
    DESCRIPTION = "HMM with Gaussian observations"

    def __init__(self, lookback: int = 100, n_states: int = 3, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_states = n_states
        self.means = np.array([-0.002, 0.0, 0.002])
        self.stds = np.array([0.015, 0.008, 0.015])
        self.pi = np.ones(n_states) / n_states
        self.A = np.array([
            [0.8, 0.15, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.15, 0.8]
        ])
        self.alpha = self.pi.copy()

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        obs = self.returns[-1]
        emission = np.zeros(self.n_states)
        for s in range(self.n_states):
            z = (obs - self.means[s]) / (self.stds[s] + 1e-10)
            emission[s] = np.exp(-0.5 * z**2) / (self.stds[s] * np.sqrt(2*np.pi) + 1e-10)
        self.alpha = np.dot(self.alpha, self.A) * emission
        self.alpha /= (self.alpha.sum() + 1e-10)
        best_state = np.argmax(self.alpha)
        if best_state == 0:
            self.signal = -1
        elif best_state == 2:
            self.signal = 1
        else:
            self.signal = 0
        self.confidence = self.alpha[best_state]


@FormulaRegistry.register(175)
class StudentTHMM(BaseFormula):
    """ID 175: Student-t emission HMM"""

    CATEGORY = "regime_detection"
    NAME = "StudentTHMM"
    DESCRIPTION = "HMM robust to fat tails"

    def __init__(self, lookback: int = 100, n_states: int = 2, df: float = 5.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_states = n_states
        self.df = df
        self.means = np.array([0.001, -0.001])
        self.scales = np.array([0.01, 0.02])
        self.pi = np.array([0.5, 0.5])
        self.A = np.array([[0.95, 0.05], [0.05, 0.95]])
        self.alpha = self.pi.copy()

    def _student_t_pdf(self, x: float, mu: float, scale: float) -> float:
        z = (x - mu) / (scale + 1e-10)
        nu = self.df
        from math import lgamma
        coef = np.exp(lgamma((nu+1)/2) - lgamma(nu/2))
        coef /= np.sqrt(nu * np.pi) * scale
        return coef * (1 + z**2/nu)**(-(nu+1)/2)

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        obs = self.returns[-1]
        emission = np.array([
            self._student_t_pdf(obs, self.means[s], self.scales[s])
            for s in range(self.n_states)
        ])
        self.alpha = np.dot(self.alpha, self.A) * emission
        self.alpha /= (self.alpha.sum() + 1e-10)
        best_state = np.argmax(self.alpha)
        if best_state == 0:
            self.signal = 1
        else:
            self.signal = -1
        self.confidence = self.alpha[best_state]


@FormulaRegistry.register(176)
class MarkovSwitching(BaseFormula):
    """ID 176: Markov Switching Regression"""

    CATEGORY = "regime_detection"
    NAME = "MarkovSwitching"
    DESCRIPTION = "Regime-dependent parameters"

    def __init__(self, lookback: int = 100, n_states: int = 2, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_states = n_states
        self.betas = np.array([0.1, -0.1])
        self.sigmas = np.array([0.01, 0.02])
        self.P = np.array([[0.95, 0.05], [0.05, 0.95]])
        self.filtered_prob = np.array([0.5, 0.5])

    def _compute(self) -> None:
        if len(self.returns) < 5:
            return
        returns = self._returns_array()
        y = returns[-1]
        x = returns[-2] if len(returns) >= 2 else 0
        likelihoods = np.zeros(self.n_states)
        for s in range(self.n_states):
            predicted = self.betas[s] * x
            residual = y - predicted
            z = residual / (self.sigmas[s] + 1e-10)
            likelihoods[s] = np.exp(-0.5 * z**2) / (self.sigmas[s] * np.sqrt(2*np.pi) + 1e-10)
        pred_prob = np.dot(self.filtered_prob, self.P)
        self.filtered_prob = pred_prob * likelihoods
        self.filtered_prob /= (self.filtered_prob.sum() + 1e-10)
        best_state = np.argmax(self.filtered_prob)
        if self.betas[best_state] > 0:
            self.signal = 1 if returns[-1] > 0 else -1
        else:
            self.signal = -1 if returns[-1] > 0 else 1
        self.confidence = self.filtered_prob[best_state]


@FormulaRegistry.register(177)
class ThreeStateHMM(BaseFormula):
    """ID 177: Three-State HMM (Bull/Neutral/Bear)"""

    CATEGORY = "regime_detection"
    NAME = "ThreeStateHMM"
    DESCRIPTION = "Bull, neutral, bear regime detection"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.means = np.array([0.002, 0.0, -0.002])
        self.stds = np.array([0.01, 0.005, 0.015])
        self.P = np.array([
            [0.85, 0.10, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.10, 0.85]
        ])
        self.state_probs = np.array([0.33, 0.34, 0.33])
        self.regime_names = ['bull', 'neutral', 'bear']

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        obs = self.returns[-1]
        emission = np.zeros(3)
        for s in range(3):
            z = (obs - self.means[s]) / (self.stds[s] + 1e-10)
            emission[s] = np.exp(-0.5 * z**2)
        predicted = np.dot(self.state_probs, self.P)
        self.state_probs = predicted * emission
        self.state_probs /= (self.state_probs.sum() + 1e-10)
        best_state = np.argmax(self.state_probs)
        if best_state == 0:
            self.signal = 1
        elif best_state == 2:
            self.signal = -1
        else:
            self.signal = 0
        self.confidence = self.state_probs[best_state]


@FormulaRegistry.register(178)
class VolatilityRegimeHMM(BaseFormula):
    """ID 178: Volatility Regime HMM"""

    CATEGORY = "regime_detection"
    NAME = "VolatilityRegimeHMM"
    DESCRIPTION = "Low/High volatility regime detection"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.vol_means = np.array([0.005, 0.015])
        self.vol_stds = np.array([0.002, 0.005])
        self.P = np.array([[0.95, 0.05], [0.10, 0.90]])
        self.state_probs = np.array([0.7, 0.3])

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        current_vol = np.std(returns[-10:])
        emission = np.zeros(2)
        for s in range(2):
            z = (current_vol - self.vol_means[s]) / (self.vol_stds[s] + 1e-10)
            emission[s] = np.exp(-0.5 * z**2)
        predicted = np.dot(self.state_probs, self.P)
        self.state_probs = predicted * emission
        self.state_probs /= (self.state_probs.sum() + 1e-10)
        best_state = np.argmax(self.state_probs)
        if best_state == 0:
            self.signal = 1
            self.confidence = self.state_probs[0]
        else:
            self.signal = -1
            self.confidence = self.state_probs[1]


@FormulaRegistry.register(179)
class TrendRegimeHMM(BaseFormula):
    """ID 179: Trend/Mean-Reversion Regime HMM"""

    CATEGORY = "regime_detection"
    NAME = "TrendRegimeHMM"
    DESCRIPTION = "Detect trending vs mean-reverting regimes"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.autocorr_means = np.array([0.3, -0.2])
        self.autocorr_stds = np.array([0.1, 0.1])
        self.P = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.state_probs = np.array([0.5, 0.5])
        self.regime = 'unknown'

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        r1 = returns[:-1]
        r2 = returns[1:]
        if np.std(r1) > 0 and np.std(r2) > 0:
            autocorr = np.corrcoef(r1[-19:], r2[-19:])[0, 1]
        else:
            autocorr = 0
        emission = np.zeros(2)
        for s in range(2):
            z = (autocorr - self.autocorr_means[s]) / (self.autocorr_stds[s] + 1e-10)
            emission[s] = np.exp(-0.5 * z**2)
        predicted = np.dot(self.state_probs, self.P)
        self.state_probs = predicted * emission
        self.state_probs /= (self.state_probs.sum() + 1e-10)
        best_state = np.argmax(self.state_probs)
        if best_state == 0:
            self.regime = 'trending'
            momentum = np.mean(returns[-5:])
            self.signal = 1 if momentum > 0 else -1
        else:
            self.regime = 'mean_reverting'
            z_score = returns[-1] / (np.std(returns) + 1e-10)
            self.signal = -1 if z_score > 1 else (1 if z_score < -1 else 0)
        self.confidence = self.state_probs[best_state]


@FormulaRegistry.register(180)
class DurationDependentHMM(BaseFormula):
    """ID 180: Duration-Dependent HMM"""

    CATEGORY = "regime_detection"
    NAME = "DurationDependentHMM"
    DESCRIPTION = "Transition depends on time in state"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.current_state = 0
        self.duration = 0
        self.max_duration = 20
        self.means = np.array([0.001, -0.001])
        self.stds = np.array([0.01, 0.015])
        self.base_transition = 0.05

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        obs = self.returns[-1]
        duration_factor = min(self.duration / self.max_duration, 1.0)
        switch_prob = self.base_transition + 0.15 * duration_factor
        emission = np.zeros(2)
        for s in range(2):
            z = (obs - self.means[s]) / (self.stds[s] + 1e-10)
            emission[s] = np.exp(-0.5 * z**2)
        P = np.array([
            [1 - switch_prob, switch_prob],
            [switch_prob, 1 - switch_prob]
        ])
        probs = np.zeros(2)
        probs[self.current_state] = 1.0
        new_probs = np.dot(probs, P) * emission
        new_probs /= (new_probs.sum() + 1e-10)
        new_state = np.argmax(new_probs)
        if new_state == self.current_state:
            self.duration += 1
        else:
            self.current_state = new_state
            self.duration = 0
        if self.current_state == 0:
            self.signal = 1
        else:
            self.signal = -1
        self.confidence = new_probs[self.current_state] * (1 - duration_factor * 0.3)


# =============================================================================
# CHANGEPOINT AND CUSUM (181-190)
# =============================================================================

@FormulaRegistry.register(181)
class CUSUMDetector(BaseFormula):
    """ID 181: CUSUM Change Detection"""

    CATEGORY = "regime_detection"
    NAME = "CUSUMDetector"
    DESCRIPTION = "Cumulative sum changepoint detection"

    def __init__(self, lookback: int = 100, k: float = 0.5, h: float = 4.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.k = k
        self.h = h
        self.S_pos = 0.0
        self.S_neg = 0.0
        self.target_mean = 0.0
        self.target_std = 0.01
        self.changepoint_detected = False

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        if len(returns) >= 50:
            self.target_mean = np.mean(returns[:50])
            self.target_std = np.std(returns[:50]) + 1e-10
        x = returns[-1]
        z = (x - self.target_mean) / self.target_std
        self.S_pos = max(0, self.S_pos + z - self.k)
        self.S_neg = max(0, self.S_neg - z - self.k)
        if self.S_pos > self.h:
            self.changepoint_detected = True
            self.signal = 1
            self.confidence = min(self.S_pos / (self.h * 2), 1.0)
            self.S_pos = 0
        elif self.S_neg > self.h:
            self.changepoint_detected = True
            self.signal = -1
            self.confidence = min(self.S_neg / (self.h * 2), 1.0)
            self.S_neg = 0
        else:
            self.changepoint_detected = False
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(182)
class PageHinkley(BaseFormula):
    """ID 182: Page-Hinkley Test"""

    CATEGORY = "regime_detection"
    NAME = "PageHinkley"
    DESCRIPTION = "Online changepoint detection"

    def __init__(self, lookback: int = 100, delta: float = 0.005,
                 lambda_: float = 50, alpha: float = 0.01, **kwargs):
        super().__init__(lookback, **kwargs)
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.m = 0.0
        self.M = 0.0
        self.sum_ = 0.0
        self.n = 0

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        x = self.returns[-1]
        self.n += 1
        self.sum_ += x
        self.m = self.sum_ / self.n
        self.M = max(self.M, self.sum_ - self.n * (self.m + self.delta))
        PH = self.sum_ - self.n * self.m - self.M
        if PH > self.lambda_:
            self.signal = 1 if x > self.m else -1
            self.confidence = min(PH / self.lambda_ / 2, 1.0)
            self.M = 0
            self.sum_ = 0
            self.n = 0
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(183)
class BayesianChangepoint(BaseFormula):
    """ID 183: Bayesian Online Changepoint Detection"""

    CATEGORY = "regime_detection"
    NAME = "BayesianChangepoint"
    DESCRIPTION = "BOCD with hazard function"

    def __init__(self, lookback: int = 100, hazard_rate: float = 0.01, **kwargs):
        super().__init__(lookback, **kwargs)
        self.hazard_rate = hazard_rate
        self.run_length_probs = np.array([1.0])
        self.mean_params = [0.0]
        self.var_params = [0.01]

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        x = self.returns[-1]
        max_run = min(len(self.run_length_probs) + 1, self.lookback)
        predictive_probs = np.zeros(len(self.run_length_probs))
        for r in range(len(self.run_length_probs)):
            mu = self.mean_params[r] if r < len(self.mean_params) else 0
            var = self.var_params[r] if r < len(self.var_params) else 0.01
            z = (x - mu) / (np.sqrt(var) + 1e-10)
            predictive_probs[r] = np.exp(-0.5 * z**2)
        growth_probs = self.run_length_probs * predictive_probs * (1 - self.hazard_rate)
        cp_prob = np.sum(self.run_length_probs * predictive_probs * self.hazard_rate)
        new_probs = np.zeros(min(len(growth_probs) + 1, max_run))
        new_probs[0] = cp_prob
        new_probs[1:len(growth_probs)+1] = growth_probs[:len(new_probs)-1]
        new_probs /= (new_probs.sum() + 1e-10)
        self.run_length_probs = new_probs
        new_mean = list(self.mean_params)
        new_var = list(self.var_params)
        new_mean = [0.0] + [m + 0.1 * (x - m) for m in new_mean[:max_run-1]]
        new_var = [0.01] + [v + 0.1 * ((x - m)**2 - v) for v, m in zip(new_var[:max_run-1], self.mean_params[:max_run-1])]
        self.mean_params = new_mean
        self.var_params = new_var
        if new_probs[0] > 0.5:
            self.signal = 1 if x > 0 else -1
            self.confidence = new_probs[0]
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(184)
class PELT(BaseFormula):
    """ID 184: Pruned Exact Linear Time (simplified)"""

    CATEGORY = "regime_detection"
    NAME = "PELT"
    DESCRIPTION = "Fast exact changepoint detection"

    def __init__(self, lookback: int = 100, penalty: float = 10.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.penalty = penalty
        self.last_changepoint = 0
        self.segment_mean = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        n = len(returns)
        costs = np.zeros(n + 1)
        changepoints = [0] * (n + 1)
        for t in range(1, n + 1):
            min_cost = np.inf
            best_tau = 0
            for tau in range(max(0, t - 50), t):
                segment = returns[tau:t]
                if len(segment) > 0:
                    cost = np.sum((segment - np.mean(segment))**2)
                else:
                    cost = 0
                total_cost = costs[tau] + cost + self.penalty
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_tau = tau
            costs[t] = min_cost - self.penalty
            changepoints[t] = best_tau
        last_cp = changepoints[n]
        if last_cp > self.last_changepoint and last_cp > n - 10:
            self.signal = 1 if returns[-1] > np.mean(returns[last_cp:]) else -1
            self.confidence = 0.7
            self.last_changepoint = last_cp
        else:
            self.signal = 0
            self.confidence = 0.3
        if last_cp < n:
            self.segment_mean = np.mean(returns[last_cp:])


@FormulaRegistry.register(185)
class BinarySegmentation(BaseFormula):
    """ID 185: Binary Segmentation"""

    CATEGORY = "regime_detection"
    NAME = "BinarySegmentation"
    DESCRIPTION = "Recursive changepoint splitting"

    def __init__(self, lookback: int = 100, min_segment: int = 10, **kwargs):
        super().__init__(lookback, **kwargs)
        self.min_segment = min_segment
        self.changepoints = []

    def _find_changepoint(self, data: np.ndarray) -> Tuple[int, float]:
        n = len(data)
        if n < 2 * self.min_segment:
            return -1, 0.0
        best_score = 0.0
        best_idx = -1
        total_var = np.var(data) * n
        for i in range(self.min_segment, n - self.min_segment):
            left_var = np.var(data[:i]) * i if i > 0 else 0
            right_var = np.var(data[i:]) * (n - i) if n - i > 0 else 0
            score = total_var - left_var - right_var
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx, best_score

    def _compute(self) -> None:
        if len(self.returns) < 30:
            return
        returns = self._returns_array()
        idx, score = self._find_changepoint(returns[-50:] if len(returns) >= 50 else returns)
        threshold = np.var(returns) * len(returns) * 0.1
        if idx > 0 and score > threshold:
            actual_idx = len(returns) - 50 + idx if len(returns) >= 50 else idx
            if not self.changepoints or actual_idx > self.changepoints[-1] + 5:
                self.changepoints.append(actual_idx)
                self.signal = 1 if returns[-1] > 0 else -1
                self.confidence = min(score / threshold / 2, 1.0)
            else:
                self.signal = 0
                self.confidence = 0.3
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(186)
class EWMAChangeDetector(BaseFormula):
    """ID 186: EWMA Control Chart"""

    CATEGORY = "regime_detection"
    NAME = "EWMAChangeDetector"
    DESCRIPTION = "Exponentially weighted control limits"

    def __init__(self, lookback: int = 100, lambda_: float = 0.2, L: float = 3.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.lambda_ = lambda_
        self.L = L
        self.ewma = 0.0
        self.target = 0.0
        self.sigma = 0.01

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        x = returns[-1]
        if len(returns) >= 50:
            self.target = np.mean(returns[:50])
            self.sigma = np.std(returns[:50]) + 1e-10
        self.ewma = self.lambda_ * x + (1 - self.lambda_) * self.ewma
        n = len(returns)
        ewma_std = self.sigma * np.sqrt(self.lambda_ / (2 - self.lambda_) * (1 - (1 - self.lambda_)**(2*n)))
        ucl = self.target + self.L * ewma_std
        lcl = self.target - self.L * ewma_std
        if self.ewma > ucl:
            self.signal = 1
            self.confidence = min((self.ewma - ucl) / ewma_std, 1.0)
        elif self.ewma < lcl:
            self.signal = -1
            self.confidence = min((lcl - self.ewma) / ewma_std, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(187)
class AdaptiveCUSUM(BaseFormula):
    """ID 187: Adaptive CUSUM"""

    CATEGORY = "regime_detection"
    NAME = "AdaptiveCUSUM"
    DESCRIPTION = "CUSUM with adaptive thresholds"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.S_pos = 0.0
        self.S_neg = 0.0
        self.k = 0.5
        self.h = 4.0
        self.recent_vol = deque(maxlen=50)

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        current_vol = np.std(returns[-10:])
        self.recent_vol.append(current_vol)
        if len(self.recent_vol) >= 20:
            avg_vol = np.mean(self.recent_vol)
            vol_ratio = current_vol / (avg_vol + 1e-10)
            self.k = 0.3 + 0.4 * min(vol_ratio, 2)
            self.h = 3.0 + 2.0 * min(vol_ratio, 2)
        mean = np.mean(returns[-20:]) if len(returns) >= 20 else 0
        std = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
        z = (returns[-1] - mean) / (std + 1e-10)
        self.S_pos = max(0, self.S_pos + z - self.k)
        self.S_neg = max(0, self.S_neg - z - self.k)
        if self.S_pos > self.h:
            self.signal = 1
            self.confidence = min(self.S_pos / self.h / 2, 1.0)
            self.S_pos = 0
        elif self.S_neg > self.h:
            self.signal = -1
            self.confidence = min(self.S_neg / self.h / 2, 1.0)
            self.S_neg = 0
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(188)
class HurstExponent(BaseFormula):
    """ID 188: Hurst Exponent for regime"""

    CATEGORY = "regime_detection"
    NAME = "HurstExponent"
    DESCRIPTION = "H < 0.5 mean-reverting, H > 0.5 trending"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.hurst = 0.5
        self.window_sizes = [8, 16, 32, 64]

    def _compute(self) -> None:
        if len(self.returns) < 70:
            return
        returns = self._returns_array()
        rs_values = []
        for n in self.window_sizes:
            if len(returns) < n:
                continue
            rs_list = []
            for start in range(0, len(returns) - n + 1, n // 2):
                segment = returns[start:start + n]
                mean = np.mean(segment)
                std = np.std(segment)
                if std < 1e-10:
                    continue
                cumdev = np.cumsum(segment - mean)
                R = np.max(cumdev) - np.min(cumdev)
                rs_list.append(R / std)
            if rs_list:
                rs_values.append((np.log(n), np.log(np.mean(rs_list))))
        if len(rs_values) >= 2:
            log_n = np.array([v[0] for v in rs_values])
            log_rs = np.array([v[1] for v in rs_values])
            slope, _ = np.polyfit(log_n, log_rs, 1)
            self.hurst = slope
        if self.hurst < 0.45:
            z = (returns[-1] - np.mean(returns)) / (np.std(returns) + 1e-10)
            self.signal = -1 if z > 0 else 1
            self.confidence = min(0.5 - self.hurst, 0.5) * 2
        elif self.hurst > 0.55:
            self.signal = 1 if returns[-1] > 0 else -1
            self.confidence = min(self.hurst - 0.5, 0.5) * 2
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(189)
class StructuralBreakTest(BaseFormula):
    """ID 189: Chow Test for structural breaks"""

    CATEGORY = "regime_detection"
    NAME = "StructuralBreakTest"
    DESCRIPTION = "Test for parameter stability"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.break_detected = False
        self.break_location = 0

    def _compute(self) -> None:
        if len(self.returns) < 40:
            return
        returns = self._returns_array()
        n = len(returns)
        best_f = 0.0
        best_break = n // 2
        for bp in range(20, n - 20):
            r1 = returns[:bp]
            r2 = returns[bp:]
            ssr1 = np.sum((r1 - np.mean(r1))**2)
            ssr2 = np.sum((r2 - np.mean(r2))**2)
            ssr_unrestricted = ssr1 + ssr2
            ssr_restricted = np.sum((returns - np.mean(returns))**2)
            k = 2
            f_stat = ((ssr_restricted - ssr_unrestricted) / k) / (ssr_unrestricted / (n - 2*k) + 1e-10)
            if f_stat > best_f:
                best_f = f_stat
                best_break = bp
        critical_value = 3.0
        if best_f > critical_value:
            self.break_detected = True
            self.break_location = best_break
            post_break = returns[best_break:]
            self.signal = 1 if np.mean(post_break) > 0 else -1
            self.confidence = min(best_f / critical_value / 3, 1.0)
        else:
            self.break_detected = False
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(190)
class RegimeClassifier(BaseFormula):
    """ID 190: Multi-factor regime classifier"""

    CATEGORY = "regime_detection"
    NAME = "RegimeClassifier"
    DESCRIPTION = "Combine multiple regime indicators"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.regime = 'unknown'
        self.regime_confidence = 0.0

    def _calculate_momentum(self, returns: np.ndarray) -> float:
        if len(returns) < 10:
            return 0.0
        return np.mean(returns[-5:]) / (np.std(returns) + 1e-10)

    def _calculate_volatility_regime(self, returns: np.ndarray) -> float:
        if len(returns) < 30:
            return 0.0
        recent_vol = np.std(returns[-10:])
        historical_vol = np.std(returns[-30:])
        return recent_vol / (historical_vol + 1e-10)

    def _calculate_trend_strength(self, returns: np.ndarray) -> float:
        if len(returns) < 20:
            return 0.0
        x = np.arange(20)
        y = returns[-20:]
        slope = np.polyfit(x, y, 1)[0]
        return slope / (np.std(returns) + 1e-10)

    def _compute(self) -> None:
        if len(self.returns) < 30:
            return
        returns = self._returns_array()
        momentum = self._calculate_momentum(returns)
        vol_ratio = self._calculate_volatility_regime(returns)
        trend = self._calculate_trend_strength(returns)
        if vol_ratio > 1.5:
            self.regime = 'high_volatility'
            self.signal = -1
            self.confidence = min(vol_ratio / 3, 1.0)
        elif abs(trend) > 1.0:
            self.regime = 'trending'
            self.signal = 1 if trend > 0 else -1
            self.confidence = min(abs(trend) / 2, 1.0)
        elif vol_ratio < 0.7:
            self.regime = 'low_volatility'
            self.signal = 0
            self.confidence = 0.5
        else:
            self.regime = 'normal'
            if momentum > 0.5:
                self.signal = 1
            elif momentum < -0.5:
                self.signal = -1
            else:
                self.signal = 0
            self.confidence = min(abs(momentum), 1.0)
        self.regime_confidence = self.confidence


__all__ = [
    'HMMRegimeDetector', 'BaumWelchHMM', 'ViterbiDecoder', 'GaussianHMM',
    'StudentTHMM', 'MarkovSwitching', 'ThreeStateHMM', 'VolatilityRegimeHMM',
    'TrendRegimeHMM', 'DurationDependentHMM',
    'CUSUMDetector', 'PageHinkley', 'BayesianChangepoint', 'PELT',
    'BinarySegmentation', 'EWMAChangeDetector', 'AdaptiveCUSUM',
    'HurstExponent', 'StructuralBreakTest', 'RegimeClassifier',
]
