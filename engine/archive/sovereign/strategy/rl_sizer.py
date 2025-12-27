"""
RL Position Sizer - Sovereign Engine
=====================================

Reinforcement Learning based position sizing.

Integrates with:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- Kelly Criterion fallback

Position sizing adapts based on:
- Signal confidence
- Current regime
- Recent performance
- Risk metrics
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import numpy as np
import time

from ..core.types import Signal, SizedOrder, Order, OrderSide, OrderType, TradeOutcome
from ..core.config import RLConfig


@dataclass
class SizerState:
    """State for position sizing decision."""
    # Signal features
    signal_direction: int
    signal_confidence: float
    regime: str

    # Performance features
    recent_win_rate: float
    recent_pnl: float
    consecutive_losses: int
    current_drawdown: float

    # Market features
    volatility: float
    trend_strength: float

    # Capital
    current_capital: float
    available_margin: float


@dataclass
class SizingResult:
    """Result of position sizing."""
    position_size_pct: float       # As percentage of capital
    position_size_usd: float       # In USD
    kelly_fraction: float          # Kelly optimal
    rl_adjustment: float           # RL multiplier
    reason: str                    # Explanation


class BaseSizer(ABC):
    """Abstract base class for position sizers."""

    @abstractmethod
    def size(self, signal: Signal, state: SizerState) -> SizingResult:
        """Calculate position size."""
        pass

    @abstractmethod
    def learn(self, outcome: TradeOutcome):
        """Learn from trade outcome."""
        pass


# =============================================================================
# KELLY CRITERION
# =============================================================================

class KellySizer(BaseSizer):
    """
    Kelly Criterion position sizing.

    Kelly formula: f* = (bp - q) / b
    where:
        f* = fraction of capital to bet
        b = odds (win/loss ratio)
        p = probability of winning
        q = probability of losing (1 - p)
    """

    def __init__(self, fraction: float = 0.25, min_size: float = 0.01, max_size: float = 0.10):
        """
        Initialize Kelly sizer.

        Args:
            fraction: Kelly fraction (0.25 = quarter Kelly for safety)
            min_size: Minimum position size
            max_size: Maximum position size
        """
        self.fraction = fraction
        self.min_size = min_size
        self.max_size = max_size

        # Track performance
        self.wins = 0
        self.losses = 0
        self.total_win_pnl = 0.0
        self.total_loss_pnl = 0.0

    def size(self, signal: Signal, state: SizerState) -> SizingResult:
        """Calculate Kelly-optimal position size."""
        # Get win probability and payoff ratio
        p = state.recent_win_rate if state.recent_win_rate > 0 else 0.5

        # Adjust probability by signal confidence
        p = p * signal.confidence

        # Calculate payoff ratio
        if self.losses > 0 and self.wins > 0:
            avg_win = self.total_win_pnl / self.wins
            avg_loss = abs(self.total_loss_pnl / self.losses)
            b = avg_win / avg_loss if avg_loss > 0 else 2.0
        else:
            b = 2.0  # Default 2:1 risk-reward

        # Kelly formula
        q = 1 - p
        kelly = (b * p - q) / b if b > 0 else 0

        # Apply fraction for safety
        kelly *= self.fraction

        # Clamp to bounds
        kelly = max(self.min_size, min(self.max_size, kelly))

        # Convert to USD
        position_usd = state.current_capital * kelly

        return SizingResult(
            position_size_pct=kelly,
            position_size_usd=position_usd,
            kelly_fraction=kelly / self.fraction,
            rl_adjustment=1.0,
            reason=f"Kelly: p={p:.2f}, b={b:.2f}"
        )

    def learn(self, outcome: TradeOutcome):
        """Update Kelly parameters from outcome."""
        if outcome.was_profitable:
            self.wins += 1
            self.total_win_pnl += outcome.pnl
        else:
            self.losses += 1
            self.total_loss_pnl += outcome.pnl


# =============================================================================
# RL POSITION SIZER (PPO/SAC Interface)
# =============================================================================

class RLPositionSizer(BaseSizer):
    """
    RL-based position sizing using PPO or SAC.

    Falls back to Kelly when not enough training data.
    """

    def __init__(self, config: RLConfig):
        self.config = config
        self.kelly = KellySizer(
            fraction=config.kelly_fraction,
            min_size=config.min_position_pct,
            max_size=config.max_position_pct,
        )

        # Training data
        self.experiences: List[Dict[str, Any]] = []
        self.model = None
        self._trained = False

        # Initialize agent if enabled
        if config.enabled:
            self._init_agent()

    def _init_agent(self):
        """Initialize RL agent."""
        try:
            if self.config.agent == "sac":
                self.model = self._create_sac_agent()
            else:
                self.model = self._create_ppo_agent()

            # Load pre-trained model if available
            if self.config.model_path:
                self._load_model()

        except ImportError as e:
            print(f"[RLSizer] RL libraries not available: {e}")
            self.model = None

    def _create_sac_agent(self):
        """Create SAC agent for continuous action space."""
        # This is a placeholder - actual implementation would use
        # stable-baselines3 or similar
        return None

    def _create_ppo_agent(self):
        """Create PPO agent."""
        return None

    def _load_model(self):
        """Load pre-trained model."""
        pass

    def size(self, signal: Signal, state: SizerState) -> SizingResult:
        """Calculate position size using RL or Kelly fallback."""
        # Use Kelly if RL not available or not trained
        if not self.config.enabled or not self._trained or self.model is None:
            result = self.kelly.size(signal, state)
            result.reason = f"Kelly fallback: {result.reason}"
            return result

        # Build state vector for RL
        state_vector = self._build_state_vector(signal, state)

        try:
            # Get action from RL agent
            action = self.model.predict(state_vector)

            # Action is position size in range [0, 1]
            position_pct = float(action[0]) * self.config.max_position_pct
            position_pct = max(self.config.min_position_pct, min(self.config.max_position_pct, position_pct))

            # Get Kelly for comparison
            kelly_result = self.kelly.size(signal, state)

            # Calculate RL adjustment
            rl_adjustment = position_pct / kelly_result.position_size_pct if kelly_result.position_size_pct > 0 else 1.0

            return SizingResult(
                position_size_pct=position_pct,
                position_size_usd=state.current_capital * position_pct,
                kelly_fraction=kelly_result.kelly_fraction,
                rl_adjustment=rl_adjustment,
                reason=f"RL ({self.config.agent}): adj={rl_adjustment:.2f}"
            )

        except Exception as e:
            print(f"[RLSizer] Prediction error: {e}")
            return self.kelly.size(signal, state)

    def _build_state_vector(self, signal: Signal, state: SizerState) -> np.ndarray:
        """Build state vector for RL agent."""
        return np.array([
            signal.direction,
            signal.confidence,
            state.recent_win_rate,
            state.recent_pnl / max(1, state.current_capital),  # Normalized
            state.consecutive_losses / 10.0,  # Normalized
            state.current_drawdown,
            state.volatility,
            state.trend_strength,
        ], dtype=np.float32)

    def learn(self, outcome: TradeOutcome):
        """Learn from trade outcome."""
        # Always update Kelly
        self.kelly.learn(outcome)

        if not self.config.enabled or not self.config.train_on_outcomes:
            return

        # Store experience
        self.experiences.append({
            'signal_direction': outcome.signal.direction,
            'signal_confidence': outcome.signal.confidence,
            'pnl': outcome.pnl,
            'pnl_pct': outcome.pnl_pct,
            'hold_duration': outcome.hold_duration,
            'exit_reason': outcome.exit_reason,
        })

        # Retrain periodically
        if len(self.experiences) >= self.config.min_samples_for_training:
            if len(self.experiences) % self.config.retrain_interval == 0:
                self._train()

    def _train(self):
        """Train RL agent on collected experiences."""
        if self.model is None:
            return

        print(f"[RLSizer] Training on {len(self.experiences)} experiences")

        # Actual training would happen here using the experiences
        # For now, just mark as trained
        self._trained = True


# =============================================================================
# VOLATILITY-ADJUSTED SIZER
# =============================================================================

class VolatilityAdjustedSizer(BaseSizer):
    """
    Position sizing adjusted for volatility.

    Higher volatility = smaller positions.
    """

    def __init__(
        self,
        base_size: float = 0.02,
        vol_target: float = 0.01,  # 1% target volatility
        min_size: float = 0.01,
        max_size: float = 0.10
    ):
        self.base_size = base_size
        self.vol_target = vol_target
        self.min_size = min_size
        self.max_size = max_size

    def size(self, signal: Signal, state: SizerState) -> SizingResult:
        """Calculate volatility-adjusted position size."""
        # Adjust for volatility
        if state.volatility > 0:
            vol_scalar = self.vol_target / state.volatility
        else:
            vol_scalar = 1.0

        # Calculate size
        size = self.base_size * vol_scalar * signal.confidence

        # Clamp
        size = max(self.min_size, min(self.max_size, size))

        return SizingResult(
            position_size_pct=size,
            position_size_usd=state.current_capital * size,
            kelly_fraction=0.0,
            rl_adjustment=vol_scalar,
            reason=f"Vol-adj: vol={state.volatility:.4f}, scalar={vol_scalar:.2f}"
        )

    def learn(self, outcome: TradeOutcome):
        """No learning for volatility sizer."""
        pass


# =============================================================================
# UNIFIED POSITION SIZER
# =============================================================================

class UnifiedPositionSizer:
    """
    Unified position sizer that combines multiple methods.

    Aggregates:
    - Kelly Criterion
    - RL-based sizing
    - Volatility adjustment
    """

    def __init__(self, config: RLConfig):
        self.config = config

        # Initialize sizers
        self.kelly = KellySizer(
            fraction=config.kelly_fraction,
            min_size=config.min_position_pct,
            max_size=config.max_position_pct,
        )

        self.rl = RLPositionSizer(config) if config.enabled else None
        self.vol_sizer = VolatilityAdjustedSizer()

        # Track performance
        self.recent_outcomes: List[TradeOutcome] = []
        self.max_history = 100

    def size(
        self,
        signal: Signal,
        capital: float,
        volatility: float = 0.01,
        **kwargs
    ) -> SizedOrder:
        """
        Calculate position size for a signal.

        Args:
            signal: Trading signal
            capital: Current capital
            volatility: Current market volatility
            **kwargs: Additional state info

        Returns:
            SizedOrder ready for execution
        """
        # Build state
        state = SizerState(
            signal_direction=signal.direction,
            signal_confidence=signal.confidence,
            regime=signal.regime,
            recent_win_rate=self._get_recent_win_rate(),
            recent_pnl=self._get_recent_pnl(),
            consecutive_losses=self._get_consecutive_losses(),
            current_drawdown=kwargs.get('drawdown', 0.0),
            volatility=volatility,
            trend_strength=kwargs.get('trend_strength', 0.5),
            current_capital=capital,
            available_margin=capital * 0.9,  # 90% of capital
        )

        # Get sizing from primary method
        if self.rl and self.config.enabled:
            result = self.rl.size(signal, state)
        else:
            result = self.kelly.size(signal, state)

        # Apply volatility adjustment
        vol_result = self.vol_sizer.size(signal, state)
        vol_factor = min(1.0, vol_result.rl_adjustment)

        # Combine
        final_size_pct = result.position_size_pct * vol_factor
        final_size_pct = max(self.config.min_position_pct, min(self.config.max_position_pct, final_size_pct))
        final_size_usd = capital * final_size_pct

        # Create order
        side = OrderSide.BUY if signal.direction > 0 else OrderSide.SELL
        price = signal.price_at_signal if signal.price_at_signal > 0 else 100000.0
        amount = final_size_usd / price

        order = Order(
            symbol="BTC/USDT",  # Could be from signal
            side=side,
            order_type=OrderType.MARKET,
            amount=amount,
            signal=signal,
        )

        return SizedOrder(
            order=order,
            position_size_pct=final_size_pct,
            position_size_usd=final_size_usd,
            kelly_fraction=result.kelly_fraction,
            rl_adjustment=result.rl_adjustment * vol_factor,
        )

    def learn(self, outcome: TradeOutcome):
        """Learn from trade outcome."""
        # Store outcome
        self.recent_outcomes.append(outcome)
        if len(self.recent_outcomes) > self.max_history:
            self.recent_outcomes.pop(0)

        # Update all sizers
        self.kelly.learn(outcome)
        if self.rl:
            self.rl.learn(outcome)

    def _get_recent_win_rate(self) -> float:
        """Calculate recent win rate."""
        if not self.recent_outcomes:
            return 0.5
        wins = sum(1 for o in self.recent_outcomes if o.was_profitable)
        return wins / len(self.recent_outcomes)

    def _get_recent_pnl(self) -> float:
        """Calculate recent total PnL."""
        return sum(o.pnl for o in self.recent_outcomes)

    def _get_consecutive_losses(self) -> int:
        """Count consecutive losses from end."""
        count = 0
        for outcome in reversed(self.recent_outcomes):
            if outcome.was_profitable:
                break
            count += 1
        return count


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_position_sizer(config: RLConfig = None) -> UnifiedPositionSizer:
    """
    Create a position sizer.

    Args:
        config: RL configuration

    Returns:
        Configured UnifiedPositionSizer
    """
    if config is None:
        config = RLConfig()

    return UnifiedPositionSizer(config)
