"""
Bayesian Model Averaging
========================

Formula IDs: 72091-72095

Combines models weighted by their posterior probability of being correct.
Models that have been right recently get more weight.

RenTech insight: The best model changes over time. Bayesian averaging
automatically adapts to regime changes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
import math


@dataclass
class BayesianOutput:
    """Output from Bayesian model averaging."""
    direction: int
    confidence: float
    posterior_weights: Dict[str, float]
    model_agreement: float
    evidence: float  # Log marginal likelihood


class BayesianModelAverager:
    """
    Bayesian Model Averaging for combining predictions.

    Updates model weights based on observed outcomes using Bayes' rule:
    P(model | data) ∝ P(data | model) * P(model)
    """

    def __init__(self, prior_weight: float = 1.0, decay: float = 0.99):
        self.prior_weight = prior_weight
        self.decay = decay

        # Model performance tracking
        self.model_log_likelihoods: Dict[str, float] = {}
        self.model_priors: Dict[str, float] = {}
        self.model_counts: Dict[str, int] = {}

    def initialize_model(self, model_id: str):
        """Initialize a new model with uniform prior."""
        if model_id not in self.model_log_likelihoods:
            self.model_log_likelihoods[model_id] = 0.0
            self.model_priors[model_id] = self.prior_weight
            self.model_counts[model_id] = 0

    def update(self, model_predictions: Dict[str, float], actual_return: float):
        """
        Update posterior weights based on prediction accuracy.

        Uses Gaussian likelihood: P(return | prediction) = N(prediction, sigma)
        """
        sigma = 0.02  # Assumed prediction noise

        for model_id, prediction in model_predictions.items():
            self.initialize_model(model_id)

            # Log-likelihood of observation given prediction
            error = actual_return - prediction
            log_lik = -0.5 * (error / sigma) ** 2 - 0.5 * np.log(2 * np.pi * sigma ** 2)

            # Update with decay (recent observations matter more)
            self.model_log_likelihoods[model_id] *= self.decay
            self.model_log_likelihoods[model_id] += log_lik
            self.model_counts[model_id] += 1

    def get_posterior_weights(self) -> Dict[str, float]:
        """Compute normalized posterior weights."""
        if not self.model_log_likelihoods:
            return {}

        # Log posterior = log likelihood + log prior
        log_posteriors = {}
        for model_id in self.model_log_likelihoods:
            log_prior = np.log(self.model_priors.get(model_id, 1.0) + 1e-10)
            log_posteriors[model_id] = self.model_log_likelihoods[model_id] + log_prior

        # Normalize using log-sum-exp for numerical stability
        max_log = max(log_posteriors.values())
        sum_exp = sum(np.exp(lp - max_log) for lp in log_posteriors.values())
        log_normalizer = max_log + np.log(sum_exp)

        weights = {}
        for model_id, log_post in log_posteriors.items():
            weights[model_id] = np.exp(log_post - log_normalizer)

        return weights

    def predict(self, model_predictions: Dict[str, float]) -> BayesianOutput:
        """Generate Bayesian model average prediction."""
        weights = self.get_posterior_weights()

        # Weight predictions by posterior
        weighted_pred = 0.0
        total_weight = 0.0
        directions = []

        for model_id, prediction in model_predictions.items():
            w = weights.get(model_id, 1.0 / len(model_predictions))
            weighted_pred += w * prediction
            total_weight += w

            if prediction > 0.001:
                directions.append(1)
            elif prediction < -0.001:
                directions.append(-1)

        if total_weight > 0:
            weighted_pred /= total_weight

        # Model agreement
        if directions:
            agreement = abs(sum(directions)) / len(directions)
        else:
            agreement = 0.0

        direction = 1 if weighted_pred > 0.001 else (-1 if weighted_pred < -0.001 else 0)
        confidence = min(1.0, abs(weighted_pred) * 10 * agreement)

        # Evidence approximation
        evidence = sum(self.model_log_likelihoods.values())

        return BayesianOutput(
            direction=direction,
            confidence=confidence,
            posterior_weights=weights,
            model_agreement=agreement,
            evidence=evidence
        )


class ThompsonSamplingCombiner:
    """
    Thompson Sampling for model selection.

    Each model has a Beta distribution over win rate.
    Sample from posteriors to select which model to follow.
    """

    def __init__(self):
        # Beta(alpha, beta) for each model
        self.model_alphas: Dict[str, float] = {}  # Wins + 1
        self.model_betas: Dict[str, float] = {}   # Losses + 1

    def initialize_model(self, model_id: str):
        """Initialize with uniform prior Beta(1, 1)."""
        if model_id not in self.model_alphas:
            self.model_alphas[model_id] = 1.0
            self.model_betas[model_id] = 1.0

    def update(self, model_id: str, prediction: float, actual_return: float):
        """Update Beta distribution based on prediction outcome."""
        self.initialize_model(model_id)

        # Win = predicted direction matches actual
        predicted_dir = 1 if prediction > 0 else (-1 if prediction < 0 else 0)
        actual_dir = 1 if actual_return > 0 else (-1 if actual_return < 0 else 0)

        if predicted_dir == actual_dir and predicted_dir != 0:
            self.model_alphas[model_id] += 1  # Win
        elif predicted_dir != 0:
            self.model_betas[model_id] += 1   # Loss

    def sample_win_rates(self) -> Dict[str, float]:
        """Sample win rate from each model's Beta posterior."""
        samples = {}
        for model_id in self.model_alphas:
            alpha = self.model_alphas[model_id]
            beta = self.model_betas[model_id]
            samples[model_id] = np.random.beta(alpha, beta)
        return samples

    def get_expected_win_rates(self) -> Dict[str, float]:
        """Get expected win rate (mean of Beta)."""
        expected = {}
        for model_id in self.model_alphas:
            alpha = self.model_alphas[model_id]
            beta = self.model_betas[model_id]
            expected[model_id] = alpha / (alpha + beta)
        return expected

    def select_model(self) -> str:
        """Select best model via Thompson sampling."""
        samples = self.sample_win_rates()
        if not samples:
            return None
        return max(samples, key=samples.get)


class OnlineBayesianEnsemble:
    """
    Online Bayesian ensemble that updates continuously.
    No batch retraining needed.
    """

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.model_weights: Dict[str, float] = {}
        self.model_squared_errors: Dict[str, deque] = {}
        self.window_size = 50

    def initialize_model(self, model_id: str):
        """Initialize model tracking."""
        if model_id not in self.model_weights:
            self.model_weights[model_id] = 1.0
            self.model_squared_errors[model_id] = deque(maxlen=self.window_size)

    def update(self, model_predictions: Dict[str, float], actual_return: float):
        """Online update of model weights."""
        for model_id, prediction in model_predictions.items():
            self.initialize_model(model_id)

            # Track squared error
            error = (prediction - actual_return) ** 2
            self.model_squared_errors[model_id].append(error)

            # Update weight inversely proportional to recent MSE
            if len(self.model_squared_errors[model_id]) >= 10:
                mse = np.mean(self.model_squared_errors[model_id])
                target_weight = 1.0 / (mse + 0.001)

                # Exponential moving average update
                self.model_weights[model_id] = (
                    (1 - self.learning_rate) * self.model_weights[model_id] +
                    self.learning_rate * target_weight
                )

    def get_normalized_weights(self) -> Dict[str, float]:
        """Get normalized weights."""
        total = sum(self.model_weights.values())
        if total == 0:
            return {m: 1.0 / len(self.model_weights) for m in self.model_weights}
        return {m: w / total for m, w in self.model_weights.items()}

    def predict(self, model_predictions: Dict[str, float]) -> float:
        """Weighted prediction."""
        weights = self.get_normalized_weights()

        weighted_sum = 0.0
        for model_id, pred in model_predictions.items():
            w = weights.get(model_id, 1.0 / len(model_predictions))
            weighted_sum += w * pred

        return weighted_sum


# =============================================================================
# FORMULA IMPLEMENTATIONS (72091-72095)
# =============================================================================

class BayesianAverageSignal:
    """
    Formula 72091: Bayesian Model Averaging Signal

    Weights models by posterior probability of being correct.
    Automatically downweights underperforming models.
    """

    FORMULA_ID = 72091

    def __init__(self, decay: float = 0.99):
        self.averager = BayesianModelAverager(decay=decay)

    def update(self, model_predictions: Dict[str, float], actual_return: float):
        """Update posteriors with new observation."""
        self.averager.update(model_predictions, actual_return)

    def generate_signal(self, model_predictions: Dict[str, float]) -> BayesianOutput:
        """Generate Bayesian averaged signal."""
        return self.averager.predict(model_predictions)


class ThompsonSamplingSignal:
    """
    Formula 72092: Thompson Sampling Signal

    Exploration-exploitation balance in model selection.
    Occasionally tries underperforming models to check if they've improved.
    """

    FORMULA_ID = 72092

    def __init__(self):
        self.sampler = ThompsonSamplingCombiner()
        self.selected_model: str = None

    def update(self, model_predictions: Dict[str, float], actual_return: float):
        """Update model win rates."""
        for model_id, pred in model_predictions.items():
            self.sampler.update(model_id, pred, actual_return)

    def generate_signal(self, model_predictions: Dict[str, float]) -> BayesianOutput:
        """Generate signal via Thompson sampling."""
        # Initialize any new models
        for model_id in model_predictions:
            self.sampler.initialize_model(model_id)

        # Thompson sample to select model
        self.selected_model = self.sampler.select_model()

        if self.selected_model and self.selected_model in model_predictions:
            prediction = model_predictions[self.selected_model]
        else:
            prediction = np.mean(list(model_predictions.values()))

        direction = 1 if prediction > 0.001 else (-1 if prediction < -0.001 else 0)

        # Confidence from expected win rate
        expected_wr = self.sampler.get_expected_win_rates()
        if self.selected_model:
            conf = expected_wr.get(self.selected_model, 0.5)
        else:
            conf = 0.5

        confidence = max(0, (conf - 0.5) * 2)  # Scale 0.5-1.0 to 0-1

        return BayesianOutput(
            direction=direction,
            confidence=confidence,
            posterior_weights=expected_wr,
            model_agreement=0.0,
            evidence=0.0
        )


class OnlineBayesianSignal:
    """
    Formula 72093: Online Bayesian Ensemble Signal

    Continuously updates weights without batch retraining.
    Adapts quickly to regime changes.
    """

    FORMULA_ID = 72093

    def __init__(self, learning_rate: float = 0.1):
        self.ensemble = OnlineBayesianEnsemble(learning_rate=learning_rate)

    def update(self, model_predictions: Dict[str, float], actual_return: float):
        """Online weight update."""
        self.ensemble.update(model_predictions, actual_return)

    def generate_signal(self, model_predictions: Dict[str, float]) -> BayesianOutput:
        """Generate online Bayesian signal."""
        prediction = self.ensemble.predict(model_predictions)
        weights = self.ensemble.get_normalized_weights()

        direction = 1 if prediction > 0.001 else (-1 if prediction < -0.001 else 0)
        confidence = min(1.0, abs(prediction) * 10)

        # Agreement: how concentrated are weights?
        weight_values = list(weights.values())
        if weight_values:
            max_weight = max(weight_values)
            agreement = max_weight  # High if one model dominates
        else:
            agreement = 0.0

        return BayesianOutput(
            direction=direction,
            confidence=confidence,
            posterior_weights=weights,
            model_agreement=agreement,
            evidence=0.0
        )


class BayesianSpikeAndSlab:
    """
    Formula 72094: Spike-and-Slab Bayesian Signal Selection

    Models can have zero weight (spike) or positive weight (slab).
    Automatically identifies which models are useful.
    """

    FORMULA_ID = 72094

    def __init__(self, prior_inclusion: float = 0.5):
        self.prior_inclusion = prior_inclusion
        self.inclusion_probs: Dict[str, float] = {}
        self.model_performance: Dict[str, deque] = {}

    def update(self, model_predictions: Dict[str, float], actual_return: float):
        """Update inclusion probabilities."""
        for model_id, pred in model_predictions.items():
            if model_id not in self.inclusion_probs:
                self.inclusion_probs[model_id] = self.prior_inclusion
                self.model_performance[model_id] = deque(maxlen=100)

            # Track if prediction was correct direction
            correct = (pred > 0 and actual_return > 0) or (pred < 0 and actual_return < 0)
            self.model_performance[model_id].append(1 if correct else 0)

            # Update inclusion probability
            if len(self.model_performance[model_id]) >= 20:
                accuracy = np.mean(self.model_performance[model_id])

                # Bayes update: P(include | data) ∝ P(data | include) * P(include)
                if accuracy > 0.5:
                    # Model is useful
                    lik_ratio = accuracy / 0.5
                else:
                    # Model is not useful
                    lik_ratio = accuracy / 0.5

                prior_odds = self.inclusion_probs[model_id] / (1 - self.inclusion_probs[model_id] + 1e-10)
                posterior_odds = prior_odds * lik_ratio
                self.inclusion_probs[model_id] = posterior_odds / (1 + posterior_odds)

    def generate_signal(self, model_predictions: Dict[str, float]) -> BayesianOutput:
        """Generate signal using only included models."""
        # Select models with high inclusion probability
        included = {m: p for m, p in model_predictions.items()
                   if self.inclusion_probs.get(m, self.prior_inclusion) > 0.5}

        if not included:
            included = model_predictions  # Fall back to all

        # Average included predictions
        avg_pred = np.mean(list(included.values()))
        direction = 1 if avg_pred > 0.001 else (-1 if avg_pred < -0.001 else 0)

        # Confidence from inclusion certainty
        if included:
            avg_inclusion = np.mean([self.inclusion_probs.get(m, 0.5) for m in included])
            confidence = min(1.0, abs(avg_pred) * 10 * avg_inclusion)
        else:
            confidence = 0.0

        return BayesianOutput(
            direction=direction,
            confidence=confidence,
            posterior_weights=self.inclusion_probs,
            model_agreement=len(included) / max(1, len(model_predictions)),
            evidence=0.0
        )


class BayesianRegimeSwitch:
    """
    Formula 72095: Bayesian Regime-Switching Signal

    Detects regime changes and adapts model weights accordingly.
    Uses change-point detection to reset beliefs.
    """

    FORMULA_ID = 72095

    def __init__(self, change_threshold: float = 2.0):
        self.change_threshold = change_threshold
        self.averager = BayesianModelAverager(decay=0.95)

        # Regime tracking
        self.return_history: deque = deque(maxlen=100)
        self.regime_start_idx = 0
        self.current_regime = 'normal'

    def detect_regime_change(self) -> bool:
        """Detect if regime has changed using CUSUM-like statistic."""
        if len(self.return_history) < 30:
            return False

        returns = np.array(self.return_history)
        recent = returns[-10:]
        historical = returns[-30:-10]

        recent_mean = np.mean(recent)
        hist_mean = np.mean(historical)
        hist_std = np.std(historical) + 1e-6

        z_score = abs(recent_mean - hist_mean) / hist_std

        return z_score > self.change_threshold

    def update(self, model_predictions: Dict[str, float], actual_return: float):
        """Update with regime change detection."""
        self.return_history.append(actual_return)

        # Check for regime change
        if self.detect_regime_change():
            # Reset model weights
            self.averager = BayesianModelAverager(decay=0.95)
            self.current_regime = 'new_regime'
        else:
            self.current_regime = 'normal'

        # Normal Bayesian update
        self.averager.update(model_predictions, actual_return)

    def generate_signal(self, model_predictions: Dict[str, float]) -> BayesianOutput:
        """Generate regime-aware Bayesian signal."""
        output = self.averager.predict(model_predictions)

        # Reduce confidence during regime transitions
        if self.current_regime == 'new_regime':
            output = BayesianOutput(
                direction=output.direction,
                confidence=output.confidence * 0.5,  # Halve confidence
                posterior_weights=output.posterior_weights,
                model_agreement=output.model_agreement,
                evidence=output.evidence
            )

        return output
