"""
Parameter Controller - Timeframe-Adaptive Mathematical Engine
=============================================================

Controls trading parameters in real-time:
1. Ornstein-Uhlenbeck decay (mean-reversion to priors)
2. Bayesian updating (Kalman filter)
3. Kelly sizing (uncertainty-adjusted)

This module answers: "What parameters should we use right now?"
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import math

from .math_primitives import (
    tae_003_ou_decay,
    tae_003_batch_ou_decay,
    tae_006_uncertain_kelly,
    tae_006_adaptive_kelly,
    get_decay_rate_for_regime,
    OUState,
)


@dataclass
class TradingParameters:
    """Current trading parameters."""
    delay: float = 60.0          # Entry delay in seconds
    hold_time: float = 300.0     # Hold duration in seconds
    threshold: float = 0.7       # Signal threshold
    position_size: float = 0.1   # Position size fraction
    stop_loss: float = 0.02      # Stop loss percentage
    take_profit: float = 0.04    # Take profit percentage

    def as_dict(self) -> Dict[str, float]:
        return {
            'delay': self.delay,
            'hold_time': self.hold_time,
            'threshold': self.threshold,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'TradingParameters':
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


@dataclass
class ParameterPrior:
    """Prior (historical optimal) for a parameter."""
    mean: float              # Historical mean
    std: float              # Historical standard deviation
    min_val: float          # Minimum valid value
    max_val: float          # Maximum valid value


# Default priors (based on historical optimization)
DEFAULT_PRIORS = {
    'delay': ParameterPrior(mean=60.0, std=30.0, min_val=1.0, max_val=300.0),
    'hold_time': ParameterPrior(mean=300.0, std=120.0, min_val=30.0, max_val=3600.0),
    'threshold': ParameterPrior(mean=0.7, std=0.1, min_val=0.5, max_val=0.95),
    'position_size': ParameterPrior(mean=0.1, std=0.05, min_val=0.01, max_val=0.25),
    'stop_loss': ParameterPrior(mean=0.02, std=0.01, min_val=0.005, max_val=0.10),
    'take_profit': ParameterPrior(mean=0.04, std=0.02, min_val=0.01, max_val=0.20),
}

# 20x LEVERAGE PRIORS - Optimized for high win rate with tight risk control
# Based on analysis: need 70%+ WR, tight stops, quick profits
HIGH_LEVERAGE_PRIORS = {
    'delay': ParameterPrior(mean=30.0, std=15.0, min_val=1.0, max_val=120.0),      # Faster entry
    'hold_time': ParameterPrior(mean=180.0, std=60.0, min_val=30.0, max_val=600.0), # Shorter holds
    'threshold': ParameterPrior(mean=0.70, std=0.05, min_val=0.65, max_val=0.90),   # Higher confidence
    'position_size': ParameterPrior(mean=0.05, std=0.02, min_val=0.02, max_val=0.10), # Smaller (Kelly safe)
    'stop_loss': ParameterPrior(mean=0.002, std=0.001, min_val=0.001, max_val=0.005), # TIGHT 0.2% stops
    'take_profit': ParameterPrior(mean=0.005, std=0.002, min_val=0.003, max_val=0.01), # 0.5% quick profits
}


class BayesianUpdater:
    """
    Updates parameter estimates using Kalman filter.

    State: θ (parameter estimate)
    Observation: performance metric (PnL, win rate, etc.)
    """

    def __init__(self, prior: ParameterPrior):
        self.prior = prior

        # Kalman state
        self.estimate = prior.mean
        self.variance = prior.std ** 2

        # Process noise (how much parameter drifts)
        self.process_noise = (prior.std * 0.1) ** 2

        # Observation noise (measurement uncertainty)
        self.observation_noise = prior.std ** 2

    def update(self, observation: float, observation_weight: float = 1.0) -> float:
        """
        Bayesian update with new observation.

        Args:
            observation: New observed optimal value
            observation_weight: Weight of observation (0-1)

        Returns:
            Updated parameter estimate
        """
        # Prediction step
        predicted_variance = self.variance + self.process_noise

        # Kalman gain
        effective_obs_noise = self.observation_noise / (observation_weight + 1e-10)
        kalman_gain = predicted_variance / (predicted_variance + effective_obs_noise)

        # Update step
        innovation = observation - self.estimate
        self.estimate = self.estimate + kalman_gain * innovation
        self.variance = (1 - kalman_gain) * predicted_variance

        # Constrain to valid range
        self.estimate = np.clip(self.estimate, self.prior.min_val, self.prior.max_val)

        return self.estimate

    def get_uncertainty(self) -> float:
        """Get current uncertainty (std dev) in estimate."""
        return math.sqrt(self.variance)

    def reset(self) -> None:
        """Reset to prior."""
        self.estimate = self.prior.mean
        self.variance = self.prior.std ** 2


class OUProcessController:
    """
    Controls parameter mean-reversion using Ornstein-Uhlenbeck process.

    When evidence is weak, parameters decay toward priors.
    """

    def __init__(self, priors: Dict[str, ParameterPrior] = None):
        self.priors = priors or DEFAULT_PRIORS

        # Current OU states
        self.states: Dict[str, OUState] = {}
        for name, prior in self.priors.items():
            self.states[name] = OUState(
                current=prior.mean,
                prior_mean=prior.mean,
                decay_rate=0.05,  # Default, updated by regime
                volatility=prior.std * 0.1
            )

        self.regime = 'unknown'
        self.last_update_time = time.time()

    def set_regime(self, regime: str) -> None:
        """Update regime and adjust decay rates."""
        self.regime = regime
        decay_rate = get_decay_rate_for_regime(regime)

        for name in self.states:
            self.states[name].decay_rate = decay_rate

    def update_parameter(
        self,
        name: str,
        new_value: float,
        evidence_weight: float = 1.0
    ) -> float:
        """
        Update a parameter with OU decay.

        Args:
            name: Parameter name
            new_value: Proposed new value
            evidence_weight: How strongly to weight new value (0-1)

        Returns:
            Decayed parameter value
        """
        if name not in self.states:
            return new_value

        state = self.states[name]
        prior = self.priors.get(name)

        # Time since last update
        now = time.time()
        dt = now - self.last_update_time

        # Apply OU decay
        if evidence_weight < 1.0:
            # Blend toward prior when evidence is weak
            state.current = tae_003_ou_decay(
                current_theta=new_value,
                prior_theta=state.prior_mean,
                decay_kappa=state.decay_rate * (1 - evidence_weight),
                dt=dt,
                volatility_sigma=state.volatility * (1 - evidence_weight)
            )
        else:
            state.current = new_value

        # Constrain to valid range
        if prior:
            state.current = np.clip(state.current, prior.min_val, prior.max_val)

        return state.current

    def decay_all(self, dt: float = 1.0) -> Dict[str, float]:
        """
        Apply decay to all parameters (e.g., when no new observations).

        Returns:
            Dict of parameter name -> decayed value
        """
        result = {}
        for name, state in self.states.items():
            prior = self.priors.get(name)
            state.current = tae_003_ou_decay(
                current_theta=state.current,
                prior_theta=state.prior_mean,
                decay_kappa=state.decay_rate,
                dt=dt,
                volatility_sigma=state.volatility,
                random_shock=0  # No random shock during pure decay
            )
            if prior:
                state.current = np.clip(state.current, prior.min_val, prior.max_val)
            result[name] = state.current
        return result

    def get_current_values(self) -> Dict[str, float]:
        """Get current parameter values."""
        return {name: state.current for name, state in self.states.items()}

    def get_as_trading_params(self) -> TradingParameters:
        """Get parameters as TradingParameters object."""
        values = self.get_current_values()
        return TradingParameters.from_dict(values)


class KellySizer:
    """
    Position sizing using uncertainty-adjusted Kelly criterion.
    """

    def __init__(self, max_position: float = 0.25):
        self.max_position = max_position
        self.win_history: List[bool] = []
        self.pnl_history: List[float] = []

    def add_trade(self, won: bool, pnl: float) -> None:
        """Add trade outcome to history."""
        self.win_history.append(won)
        self.pnl_history.append(pnl)

        # Keep last 500 trades
        if len(self.win_history) > 500:
            self.win_history = self.win_history[-500:]
            self.pnl_history = self.pnl_history[-500:]

    def calculate_size(
        self,
        param_uncertainty: float = 0.0,
        regime: str = 'unknown'
    ) -> float:
        """
        Calculate position size using uncertain Kelly.

        Args:
            param_uncertainty: Uncertainty in parameters (0-1)
            regime: Current market regime

        Returns:
            Position size fraction (0 to max_position)
        """
        if len(self.win_history) < 10:
            # Minimum size with insufficient history
            return 0.01

        return tae_006_adaptive_kelly(
            win_history=self.win_history,
            pnl_history=self.pnl_history,
            regime=regime
        )

    def quick_size(
        self,
        win_prob: float,
        win_loss_ratio: float,
        uncertainty: float = 0.0
    ) -> float:
        """
        Quick Kelly calculation without history.

        Args:
            win_prob: Win probability
            win_loss_ratio: Average win / average loss
            uncertainty: Parameter uncertainty

        Returns:
            Position size fraction
        """
        return tae_006_uncertain_kelly(
            win_prob=win_prob,
            win_loss_ratio=win_loss_ratio,
            param_uncertainty=uncertainty,
            sample_size=len(self.win_history) if self.win_history else 10
        )


class ParameterController:
    """
    Main parameter control engine.

    Combines:
    - Bayesian updating for parameter learning
    - OU decay for mean-reversion
    - Kelly sizing for position control
    """

    def __init__(self, priors: Dict[str, ParameterPrior] = None):
        self.priors = priors or DEFAULT_PRIORS

        # Sub-components
        self.ou_controller = OUProcessController(self.priors)
        self.kelly_sizer = KellySizer()

        # Bayesian updaters for each parameter
        self.bayesian_updaters: Dict[str, BayesianUpdater] = {}
        for name, prior in self.priors.items():
            self.bayesian_updaters[name] = BayesianUpdater(prior)

        # Current regime
        self.regime = 'unknown'

        # Performance tracking
        self.recent_pnl: List[float] = []
        self.recent_wins: List[bool] = []

    def set_regime(self, regime: str) -> None:
        """Update current regime."""
        self.regime = regime
        self.ou_controller.set_regime(regime)

    def update_from_trade(self, pnl: float, params_used: Dict[str, float]) -> None:
        """
        Update parameters from trade outcome.

        Args:
            pnl: Trade PnL
            params_used: Parameters that were used for this trade
        """
        won = pnl > 0
        self.recent_pnl.append(pnl)
        self.recent_wins.append(won)
        self.kelly_sizer.add_trade(won, pnl)

        # Trim history
        if len(self.recent_pnl) > 100:
            self.recent_pnl = self.recent_pnl[-100:]
            self.recent_wins = self.recent_wins[-100:]

        # Update Bayesian estimators if trade was successful
        weight = 0.7 if won else 0.3
        for name, value in params_used.items():
            if name in self.bayesian_updaters:
                self.bayesian_updaters[name].update(value, weight)

    def get_optimal_parameters(
        self,
        proposed: Optional[Dict[str, float]] = None,
        evidence_weight: float = 0.5
    ) -> TradingParameters:
        """
        Get optimal parameters, blending proposed with priors.

        Args:
            proposed: Proposed parameter values (e.g., from optimization)
            evidence_weight: How much to trust proposed values (0-1)

        Returns:
            Optimal TradingParameters
        """
        result = {}

        for name in self.priors:
            # Get Bayesian estimate
            bayesian_val = self.bayesian_updaters[name].estimate

            # Get proposed value if available
            if proposed and name in proposed:
                prop_val = proposed[name]
            else:
                prop_val = bayesian_val

            # Blend with evidence weight
            blended = evidence_weight * prop_val + (1 - evidence_weight) * bayesian_val

            # Apply OU decay
            final = self.ou_controller.update_parameter(name, blended, evidence_weight)
            result[name] = final

        # Calculate Kelly-based position size
        uncertainty = self._get_average_uncertainty()
        kelly_size = self.kelly_sizer.calculate_size(uncertainty, self.regime)
        result['position_size'] = kelly_size

        return TradingParameters.from_dict(result)

    def decay_step(self, dt: float = 1.0) -> TradingParameters:
        """
        Apply one step of decay (when no new observations).

        Call this periodically to let parameters drift toward priors.
        """
        decayed = self.ou_controller.decay_all(dt)
        return TradingParameters.from_dict(decayed)

    def _get_average_uncertainty(self) -> float:
        """Get average uncertainty across parameters."""
        uncertainties = []
        for name, updater in self.bayesian_updaters.items():
            prior = self.priors.get(name)
            if prior:
                # Normalize uncertainty by prior std
                norm_unc = updater.get_uncertainty() / prior.std
                uncertainties.append(norm_unc)
        return np.mean(uncertainties) if uncertainties else 0.5

    def get_parameter_uncertainty(self, name: str) -> float:
        """Get uncertainty for a specific parameter."""
        if name in self.bayesian_updaters:
            return self.bayesian_updaters[name].get_uncertainty()
        return 0.5

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information."""
        return {
            'regime': self.regime,
            'parameters': self.ou_controller.get_current_values(),
            'uncertainties': {
                name: self.get_parameter_uncertainty(name)
                for name in self.priors
            },
            'recent_win_rate': sum(self.recent_wins) / len(self.recent_wins)
                if self.recent_wins else 0.5,
            'kelly_size': self.kelly_sizer.calculate_size(
                self._get_average_uncertainty(), self.regime
            ),
        }

    def reset(self) -> None:
        """Reset all state to priors."""
        self.ou_controller = OUProcessController(self.priors)
        for updater in self.bayesian_updaters.values():
            updater.reset()
        self.kelly_sizer = KellySizer()
        self.recent_pnl = []
        self.recent_wins = []
        self.regime = 'unknown'
