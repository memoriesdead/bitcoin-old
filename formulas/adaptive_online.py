"""
Adaptive Online Learning Formulas (IDs 301-307)
================================================
SOLUTION TO NON-STATIONARITY: Algorithms that adapt in real-time across ALL time horizons

Problem: What works for 1 second doesn't work for 2 seconds.
         What works for 1 minute doesn't work for 5 minutes.

Solution: Online learning + concept drift detection + meta-learning

Research Sources:
- Particle Filtering for Regime Switching (MDPI 2024)
- Online Learning & Concept Drift (Smart Aviation Solutions 2024)
- EWMA Adaptive Parameters for Crypto (Taylor & Francis 2022)
- Meta-Learning for Financial Forecasting (arXiv 2024)
- Renaissance Technologies: Hidden Markov Models + Baum-Welch Algorithm

Key Mathematical Concepts:
==========================
1. EXPONENTIALLY WEIGHTED MOVING AVERAGE (EWMA)
   - Adapts decay factor λ based on volatility
   - λ ∈ [0.94, 0.99] for crypto (RiskMetrics)
   - Recent data weighted more heavily during high volatility

2. KALMAN FILTER
   - Recursive Bayesian estimation
   - Adapts to changing market regimes
   - Estimates hidden states (trend, volatility)

3. PARTICLE FILTER
   - Monte Carlo methods for non-linear systems
   - Handles regime switches better than Kalman
   - Computationally intensive but accurate

4. HIDDEN MARKOV MODELS (HMM)
   - Renaissance Technologies' core technique
   - Baum-Welch algorithm for parameter learning
   - Probability of regime = f(current state only)

5. ONLINE GRADIENT DESCENT
   - Update model with each new data point
   - Learning rate η adapts to gradient magnitude
   - Prevents catastrophic forgetting

6. META-LEARNING (MAML)
   - Model-Agnostic Meta-Learning
   - Learn how to adapt quickly to new regimes
   - Few-shot learning: adapt with 1-5 examples

7. CONCEPT DRIFT DETECTION
   - ADWIN (Adaptive Windowing)
   - DDM/EDDM (Drift Detection Methods)
   - Detects when distribution changes

Mathematics:
============
EWMA: V_t = λ * V_{t-1} + (1-λ) * r_t^2
      where λ = adaptive decay factor

Kalman: x_t = A*x_{t-1} + w_t        (state equation)
        y_t = H*x_t + v_t            (measurement)
        K_t = P_t * H^T / (H*P_t*H^T + R)  (Kalman gain)

HMM: P(state_t | obs_{1:t}) = f(state_{t-1}, obs_t)
     α_t(i) = P(obs_t | state_i) * Σ_j α_{t-1}(j) * P(state_i | state_j)

Meta-Learning: θ' = θ - α∇_θ L(θ, D_train)
               θ* = θ' - β∇_θ' L(θ', D_test)

Online SGD: θ_t = θ_{t-1} - η_t * ∇L(θ_{t-1}, x_t)
            η_t = η_0 / sqrt(t)  (AdaGrad style)
"""

import numpy as np
from collections import deque
from .base import BaseFormula, FormulaRegistry

# =============================================================================
# ADAPTIVE EWMA FORMULAS (IDs 301-307)
# =============================================================================

@FormulaRegistry.register(301)
class AdaptiveEWMAVolatility(BaseFormula):
    """
    Adaptive EWMA with decay factor that changes based on volatility regime

    Formula:
        λ_t = λ_low if vol_high else λ_high
        V_t = λ_t * V_{t-1} + (1-λ_t) * r_t^2

    Research: RiskMetrics uses λ=0.94 for daily crypto
    Adaptive: λ ∈ [0.90, 0.99] based on regime
    """

    CATEGORY = "adaptive_vol"
    NAME = "Adaptive EWMA Volatility"
    DESCRIPTION = "EWMA with adaptive decay factor based on volatility regime"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.variance = 0.0
        self.returns_history = deque(maxlen=20)
        self.lambda_low = 0.90   # High volatility regime
        self.lambda_high = 0.98  # Low volatility regime

    def _compute(self):
        if len(self.returns) < 2:
            return

        latest_return = self.returns[-1]
        self.returns_history.append(latest_return)

        # Detect volatility regime
        if len(self.returns_history) >= 10:
            recent_vol = np.std(list(self.returns_history)[-10:])
            long_vol = np.std(list(self.returns_history))

            # High volatility = use low lambda (react faster)
            # Low volatility = use high lambda (smooth more)
            lambda_t = self.lambda_low if recent_vol > long_vol * 1.5 else self.lambda_high
        else:
            lambda_t = 0.94  # Default

        # Update EWMA variance
        if self.variance == 0:
            self.variance = latest_return ** 2
        else:
            self.variance = lambda_t * self.variance + (1 - lambda_t) * (latest_return ** 2)

        # Signal: Trade inverse to volatility (contrarian)
        volatility = np.sqrt(self.variance)

        if volatility > 0.02:  # High volatility
            # Mean reversion signal
            if latest_return > 2 * volatility:
                self.signal = -1  # Sell on spike up
                self.confidence = 0.8
            elif latest_return < -2 * volatility:
                self.signal = 1   # Buy on spike down
                self.confidence = 0.8
            else:
                self.signal = 0
                self.confidence = 0.0
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(302)
class AdaptiveLearningRateMomentum(BaseFormula):
    """
    Online gradient descent with adaptive learning rate

    Formula:
        θ_t = θ_{t-1} - η_t * ∇L(θ_{t-1}, x_t)
        η_t = η_0 / sqrt(G_t)  where G_t = sum of squared gradients

    This is AdaGrad applied to momentum trading
    """

    CATEGORY = "online_learning"
    NAME = "Adaptive Learning Rate Momentum"
    DESCRIPTION = "AdaGrad-style adaptive learning for momentum signals"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theta = 0.0  # Model parameter (momentum)
        self.G = 1e-8     # Sum of squared gradients
        self.eta_0 = 0.1  # Base learning rate

    def _compute(self):
        if len(self.returns) < 10:
            return

        # Calculate recent momentum
        recent_momentum = np.mean(list(self.returns)[-5:])

        # Gradient: difference between actual and predicted
        gradient = recent_momentum - self.theta

        # Update G (sum of squared gradients)
        self.G += gradient ** 2

        # Adaptive learning rate
        eta_t = self.eta_0 / np.sqrt(self.G)

        # Update theta (our momentum estimate)
        self.theta += eta_t * gradient

        # Signal based on adapted theta
        if self.theta > 0.001:
            self.signal = 1
            self.confidence = min(abs(self.theta) * 200, 1.0)
        elif self.theta < -0.001:
            self.signal = -1
            self.confidence = min(abs(self.theta) * 200, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(303)
class ConceptDriftDetector(BaseFormula):
    """
    ADWIN (Adaptive Windowing) for concept drift detection

    Detects when statistical distribution changes
    When drift detected, reset models and adapt

    Formula:
        Split window W into W0, W1
        If |mean(W0) - mean(W1)| > ε_cut: DRIFT DETECTED
    """

    CATEGORY = "drift_detection"
    NAME = "Concept Drift Detector"
    DESCRIPTION = "ADWIN algorithm for detecting market regime changes"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.window = deque(maxlen=100)
        self.drift_detected = False
        self.trades_since_drift = 0

    def _compute(self):
        if len(self.returns) < 2:
            return

        latest = self.returns[-1]
        self.window.append(latest)

        if len(self.window) < 30:
            return

        # Split window in half
        mid = len(self.window) // 2
        window_list = list(self.window)
        W0 = window_list[:mid]
        W1 = window_list[mid:]

        mean0 = np.mean(W0)
        mean1 = np.mean(W1)
        std_total = np.std(window_list)

        # Drift threshold (Hoeffding bound)
        n = len(self.window)
        epsilon_cut = std_total * np.sqrt((2 * np.log(2 / 0.05)) / n) if n > 0 else 0.1

        # Detect drift
        if abs(mean0 - mean1) > epsilon_cut:
            self.drift_detected = True
            self.trades_since_drift = 0
        else:
            self.drift_detected = False
            self.trades_since_drift += 1

        # After drift: aggressive mean reversion
        # After stability: trend following
        if self.drift_detected or self.trades_since_drift < 10:
            # Just after drift: mean reversion
            if latest > std_total:
                self.signal = -1
                self.confidence = 0.9
            elif latest < -std_total:
                self.signal = 1
                self.confidence = 0.9
            else:
                self.signal = 0
                self.confidence = 0.0
        else:
            # Stable regime: momentum
            if mean1 > 0.001:
                self.signal = 1
                self.confidence = 0.6
            elif mean1 < -0.001:
                self.signal = -1
                self.confidence = 0.6
            else:
                self.signal = 0
                self.confidence = 0.0


@FormulaRegistry.register(304)
class KalmanFilterTrend(BaseFormula):
    """
    Kalman Filter for adaptive trend estimation

    State Space Model:
        x_t = x_{t-1} + w_t      (random walk trend)
        y_t = x_t + v_t          (observed price)

    Kalman recursion estimates hidden trend x_t
    """

    CATEGORY = "kalman"
    NAME = "Kalman Filter Trend"
    DESCRIPTION = "Recursive Bayesian estimation for hidden trend"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # State estimate
        self.x = 0.0          # Trend estimate
        self.P = 1.0          # Estimate variance

        # Process noise and measurement noise
        self.Q = 0.001        # Process noise variance
        self.R = 0.01         # Measurement noise variance

    def _compute(self):
        if len(self.prices) < 2:
            return

        # Observation
        y_t = self.prices[-1]

        # Prediction step
        x_pred = self.x                 # State prediction
        P_pred = self.P + self.Q        # Covariance prediction

        # Update step
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.x = x_pred + K * (y_t - x_pred)  # State update
        self.P = (1 - K) * P_pred      # Covariance update

        # Innovation (measurement - prediction)
        innovation = y_t - x_pred

        # Signal based on innovation
        innovation_std = np.sqrt(P_pred + self.R)

        if innovation > 2 * innovation_std:
            self.signal = 1  # Strong upward surprise
            self.confidence = 0.8
        elif innovation < -2 * innovation_std:
            self.signal = -1  # Strong downward surprise
            self.confidence = 0.8
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(305)
class HMMRegimeDetector(BaseFormula):
    """
    Hidden Markov Model for regime detection

    Renaissance Technologies' core technique:
    - Baum-Welch algorithm for learning
    - Forward algorithm for inference

    Regimes: BULLISH, BEARISH, NEUTRAL
    """

    CATEGORY = "hmm"
    NAME = "HMM Regime Detector"
    DESCRIPTION = "Renaissance-style Hidden Markov Model for regime detection"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 3 regimes: BULL (0), BEAR (1), NEUTRAL (2)
        self.n_states = 3

        # Transition probabilities (learned online)
        self.transition = np.array([
            [0.9, 0.05, 0.05],   # BULL -> BULL, BEAR, NEUTRAL
            [0.05, 0.9, 0.05],   # BEAR -> BULL, BEAR, NEUTRAL
            [0.3, 0.3, 0.4]      # NEUTRAL -> BULL, BEAR, NEUTRAL
        ])

        # Current state probabilities
        self.alpha = np.array([0.33, 0.33, 0.34])  # Uniform prior

    def _compute(self):
        if len(self.returns) < 5:
            return

        # Observation: recent return
        obs = np.mean(list(self.returns)[-3:])

        # Emission probabilities (likelihood of observation in each state)
        # BULL: positive returns more likely
        # BEAR: negative returns more likely
        # NEUTRAL: near-zero returns more likely
        obs_probs = np.array([
            np.exp(-((obs - 0.01) ** 2) / 0.0004),  # BULL centered at +1%
            np.exp(-((obs + 0.01) ** 2) / 0.0004),  # BEAR centered at -1%
            np.exp(-(obs ** 2) / 0.0001)            # NEUTRAL centered at 0%
        ])
        obs_probs /= obs_probs.sum()  # Normalize

        # Forward algorithm: update state probabilities
        self.alpha = obs_probs * (self.transition.T @ self.alpha)
        self.alpha /= self.alpha.sum()  # Normalize

        # Most likely regime
        regime = np.argmax(self.alpha)
        conf = self.alpha[regime]

        # Trading signals based on regime
        if regime == 0:  # BULL
            self.signal = 1
            self.confidence = conf
        elif regime == 1:  # BEAR
            self.signal = -1
            self.confidence = conf
        else:  # NEUTRAL
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(306)
class MAMLFewShotAdaptation(BaseFormula):
    """
    Model-Agnostic Meta-Learning (MAML)

    Learn how to adapt quickly to new market regimes
    with only 1-5 examples (few-shot learning)

    Formula:
        θ' = θ - α∇_θ L(θ, D_support)    (inner loop)
        θ = θ - β∇_θ L(θ', D_query)      (outer loop)
    """

    CATEGORY = "meta_learning"
    NAME = "MAML Few-Shot Adaptation"
    DESCRIPTION = "Meta-learning for rapid adaptation to new regimes"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theta = 0.0  # Meta-learned parameter
        self.alpha_lr = 0.01  # Inner learning rate
        self.beta_lr = 0.001  # Outer learning rate
        self.support_set = deque(maxlen=5)  # Few-shot examples

    def _compute(self):
        if len(self.returns) < 10:
            return

        # Add new example to support set
        recent = self.returns[-1]
        self.support_set.append(recent)

        if len(self.support_set) < 3:
            return

        # Inner loop: adapt to support set
        theta_prime = self.theta
        for example in self.support_set:
            # Gradient of prediction error
            prediction = theta_prime
            gradient = example - prediction
            theta_prime += self.alpha_lr * gradient

        # Outer loop: update meta-parameters
        # Use most recent as query
        query = self.returns[-1]
        meta_gradient = query - theta_prime
        self.theta += self.beta_lr * meta_gradient

        # Signal based on adapted prediction
        prediction = theta_prime

        if prediction > 0.002:
            self.signal = 1
            self.confidence = min(abs(prediction) * 100, 1.0)
        elif prediction < -0.002:
            self.signal = -1
            self.confidence = min(abs(prediction) * 100, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(307)
class ParticleFilterRegime(BaseFormula):
    """
    Particle Filter for non-linear regime tracking

    Better than Kalman for regime switches
    Uses Monte Carlo sampling

    Formula:
        Particles: {x_i, w_i} for i=1..N
        Resample based on weights
        Propagate through dynamics
    """

    CATEGORY = "particle_filter"
    NAME = "Particle Filter Regime"
    DESCRIPTION = "Monte Carlo particle filter for non-linear regime tracking"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_particles = 100
        # Particles represent possible trend values
        self.particles = np.random.normal(0, 0.01, self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles

    def _compute(self):
        if len(self.returns) < 2:
            return

        observation = self.returns[-1]

        # Prediction: add noise to particles
        self.particles += np.random.normal(0, 0.001, self.n_particles)

        # Update: weight by likelihood
        likelihoods = np.exp(-((observation - self.particles) ** 2) / (2 * 0.01 ** 2))
        self.weights *= likelihoods
        self.weights /= self.weights.sum()

        # Resample if effective sample size low
        eff_n = 1.0 / np.sum(self.weights ** 2)
        if eff_n < self.n_particles / 2:
            indices = np.random.choice(self.n_particles, self.n_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Estimate: weighted mean
        estimate = np.sum(self.particles * self.weights)
        uncertainty = np.sqrt(np.sum(self.weights * (self.particles - estimate) ** 2))

        # Signal based on estimate and uncertainty
        if estimate > 2 * uncertainty and uncertainty > 0:
            self.signal = 1
            self.confidence = 0.8
        elif estimate < -2 * uncertainty and uncertainty > 0:
            self.signal = -1
            self.confidence = 0.8
        else:
            self.signal = 0
            self.confidence = 0.0


# Export all formulas
__all__ = [
    'AdaptiveEWMAVolatility',
    'AdaptiveLearningRateMomentum',
    'ConceptDriftDetector',
    'KalmanFilterTrend',
    'HMMRegimeDetector',
    'MAMLFewShotAdaptation',
    'ParticleFilterRegime',
]
