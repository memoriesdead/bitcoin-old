"""
Renaissance Trading System - Master Probability Engine
Combines all probability-based models into single trading probability
"""
import numpy as np
from .platt_scaling import PlattScaling
from .isotonic import IsotonicCalibration
from .bayesian import BayesianWinRate
from .regime import RegimeConditionalProbability
from .ensemble import EnsembleSignalFusion
from .soft_filter import SoftFilterProbability


class MasterProbabilityEngine:
    """
    Combines all probability-based models into single trading probability.

    This REPLACES the old filter-based approach with probability scaling.

    Key difference:
    - OLD: Filter blocks trade entirely (binary)
    - NEW: Probability scales position size (continuous)

    Components:
    1. Platt Scaling (signal → probability)
    2. Bayesian Win Rate (confidence adjustment)
    3. Regime Probability (market condition)
    4. Ensemble Fusion (multi-signal)
    5. Soft Filters (relaxed conditions)
    """

    def __init__(self):
        # Initialize all probability components
        self.platt = PlattScaling()
        self.isotonic = IsotonicCalibration()
        self.bayesian_wr = BayesianWinRate(alpha_prior=5, beta_prior=5)
        self.regime_prob = RegimeConditionalProbability()
        self.ensemble = EnsembleSignalFusion(n_signals=5)
        self.soft_filter = SoftFilterProbability()

        # Track overall performance
        self.total_trades = 0
        self.total_wins = 0

    def update_price(self, price):
        """Update all price-dependent components."""
        self.regime_prob.update(price)

    def record_trade(self, signal_score, won, signals=None):
        """
        Record trade outcome for online learning.

        Parameters:
        -----------
        signal_score : float
            The main signal score used for this trade
        won : bool
            Did the trade win?
        signals : list (optional)
            Individual signal values for ensemble learning
        """
        self.total_trades += 1
        if won:
            self.total_wins += 1

        # Update all learners
        self.platt.update_online(signal_score, won)
        self.isotonic.update_online(signal_score, won)
        self.bayesian_wr.update(won)

        # Update regime
        regime = self.regime_prob.current_regime
        self.regime_prob.record_trade_outcome(regime, won)

        # Update ensemble if signals provided
        if signals is not None:
            for i, sig in enumerate(signals[:5]):
                self.ensemble.update_signal_performance(i, np.sign(sig), won)

    def get_trade_probability(self, main_signal, signals=None, filter_conditions=None):
        """
        Get overall trade success probability.

        Parameters:
        -----------
        main_signal : float
            Primary signal score (e.g., momentum)
        signals : list (optional)
            Additional signal values for ensemble
        filter_conditions : list (optional)
            List of (value, threshold, 'above'/'below') for soft filtering

        Returns:
        --------
        dict with:
            probability : float [0, 1]
            position_mult : float [0, 2]
            components : dict of individual probabilities
        """
        components = {}

        # 1. Platt calibrated probability
        platt_prob = self.platt.predict_proba(main_signal)
        components['platt'] = platt_prob

        # 2. Isotonic calibrated probability
        iso_prob = self.isotonic.predict_proba(main_signal)
        components['isotonic'] = iso_prob

        # 3. Bayesian win rate confidence
        bayes_mult = self.bayesian_wr.get_confidence_mult()
        bayes_prob = self.bayesian_wr.mean()
        components['bayesian_wr'] = bayes_prob
        components['bayesian_mult'] = bayes_mult

        # 4. Regime conditional probability
        regime_prob = self.regime_prob.get_regime_trade_prob(main_signal)
        components['regime'] = regime_prob
        components['current_regime'] = self.regime_prob.current_regime

        # 5. Ensemble probability (if signals provided)
        if signals is not None and len(signals) >= 5:
            ensemble_prob, direction = self.ensemble.combine_signals(signals[:5])
            agreement = self.ensemble.get_signal_quality(signals[:5])
            components['ensemble'] = ensemble_prob
            components['agreement'] = agreement
        else:
            ensemble_prob = 0.5
            agreement = 0.5

        # 6. Soft filter probability (if conditions provided)
        if filter_conditions is not None and len(filter_conditions) > 0:
            soft_prob = self.soft_filter.multi_condition_soft_filter(filter_conditions)
            components['soft_filter'] = soft_prob
        else:
            soft_prob = 1.0

        # Combine all probabilities
        # Weighted average with emphasis on learned models
        weights = {
            'platt': 0.20 if self.platt.fitted else 0.05,
            'isotonic': 0.15 if self.isotonic.fitted else 0.05,
            'bayesian': 0.25,
            'regime': 0.20,
            'ensemble': 0.10 if signals else 0.05,
            'soft': 0.10
        }

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}

        # Weighted combination
        final_prob = (
            weights['platt'] * platt_prob +
            weights['isotonic'] * iso_prob +
            weights['bayesian'] * bayes_prob +
            weights['regime'] * regime_prob +
            weights['ensemble'] * ensemble_prob +
            weights['soft'] * soft_prob
        )

        # Apply Bayesian confidence multiplier
        final_prob = final_prob * (0.7 + bayes_mult * 0.3)
        final_prob = np.clip(final_prob, 0.1, 0.95)

        # Calculate position multiplier
        position_mult = self._prob_to_position_mult(final_prob, agreement)

        components['final_probability'] = final_prob
        components['position_mult'] = position_mult

        return {
            'probability': final_prob,
            'position_mult': position_mult,
            'components': components
        }

    def _prob_to_position_mult(self, prob, agreement=0.5):
        """
        Convert probability to position size multiplier.

        Key insight: Scale position, don't block trade entirely.
        """
        # Base multiplier from probability
        if prob < 0.40:
            base_mult = 0.2
        elif prob < 0.45:
            base_mult = 0.4
        elif prob < 0.50:
            base_mult = 0.6
        elif prob < 0.55:
            base_mult = 0.8
        elif prob < 0.60:
            base_mult = 1.0
        elif prob < 0.65:
            base_mult = 1.15
        elif prob < 0.70:
            base_mult = 1.3
        elif prob < 0.75:
            base_mult = 1.4
        else:
            base_mult = 1.5

        # Agreement bonus
        agreement_mult = 0.8 + agreement * 0.4  # 0.8 to 1.2

        return base_mult * agreement_mult

    def should_trade(self, main_signal, signals=None, min_prob=0.45):
        """
        Should we enter this trade?

        Note: Even if False, you could still trade with reduced size.
        This is more of a "full size trade?" check.
        """
        result = self.get_trade_probability(main_signal, signals)
        return result['probability'] > min_prob

    def get_stats(self):
        """Get performance statistics."""
        wr = self.total_wins / self.total_trades if self.total_trades > 0 else 0
        return {
            'total_trades': self.total_trades,
            'win_rate': wr,
            'bayesian_wr': self.bayesian_wr.mean(),
            'bayesian_ci': self.bayesian_wr.credible_interval(),
            'platt_fitted': self.platt.fitted,
            'isotonic_fitted': self.isotonic.fitted,
            'current_regime': self.regime_prob.current_regime
        }
