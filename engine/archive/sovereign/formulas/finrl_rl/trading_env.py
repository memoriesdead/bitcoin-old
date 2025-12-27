"""
Trading Environment
===================

Gymnasium-compatible trading environment.
Ported from FinRL's StockTradingEnv.

The environment provides:
- State: market features + portfolio state
- Action: position size adjustment
- Reward: risk-adjusted returns
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


# Try to import gymnasium, fallback to simple interface
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    gym = None
    spaces = None


@dataclass
class TradingState:
    """
    State representation for trading agent.

    FinRL pattern: Combine market features with portfolio state.
    """
    # Market features
    price: float
    price_change: float  # Percentage change
    volatility: float    # Recent volatility
    flow: float          # Blockchain flow signal
    flow_momentum: float
    spread: float        # Bid-ask spread

    # Portfolio state
    position: float      # Current position (-1 to 1)
    unrealized_pnl: float
    cash_ratio: float    # Cash / Total value
    drawdown: float      # Current drawdown

    # Time features
    hour: int            # Hour of day (0-23)
    day_of_week: int     # Day of week (0-6)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for RL input."""
        return np.array([
            self.price_change,
            self.volatility,
            self.flow,
            self.flow_momentum,
            self.spread,
            self.position,
            self.unrealized_pnl,
            self.cash_ratio,
            self.drawdown,
            self.hour / 24.0,  # Normalize
            self.day_of_week / 7.0,
        ], dtype=np.float32)

    @staticmethod
    def dim() -> int:
        """State dimension."""
        return 11


@dataclass
class TradingAction:
    """
    Action representation.

    Continuous action: target position from -1 to 1
    -1 = max short, 0 = neutral, 1 = max long
    """
    target_position: float  # -1 to 1

    @staticmethod
    def from_array(arr: np.ndarray) -> 'TradingAction':
        """Create from numpy array."""
        return TradingAction(
            target_position=float(np.clip(arr[0], -1, 1))
        )


class RewardShaper:
    """
    Shapes rewards to encourage desired trading behavior.

    FinRL pattern: Combine multiple reward components.
    """

    def __init__(self,
                 pnl_weight: float = 1.0,
                 sharpe_weight: float = 0.5,
                 drawdown_penalty: float = 0.3,
                 turnover_penalty: float = 0.1):
        """
        Initialize reward shaper.

        Args:
            pnl_weight: Weight for raw PnL
            sharpe_weight: Weight for risk-adjusted returns
            drawdown_penalty: Penalty for drawdowns
            turnover_penalty: Penalty for excessive trading
        """
        self.pnl_weight = pnl_weight
        self.sharpe_weight = sharpe_weight
        self.drawdown_penalty = drawdown_penalty
        self.turnover_penalty = turnover_penalty

        # Track returns for Sharpe calculation
        self.returns_history: List[float] = []
        self.max_portfolio_value: float = 0.0

    def compute_reward(self,
                       pnl: float,
                       portfolio_value: float,
                       position_change: float,
                       volatility: float = 0.01) -> float:
        """
        Compute shaped reward.

        Args:
            pnl: Period PnL (percentage)
            portfolio_value: Current portfolio value
            position_change: Absolute change in position
            volatility: Recent volatility

        Returns:
            Shaped reward value
        """
        reward = 0.0

        # 1. Raw PnL component
        reward += self.pnl_weight * pnl

        # 2. Sharpe-like component (risk-adjusted)
        self.returns_history.append(pnl)
        if len(self.returns_history) > 20:
            self.returns_history = self.returns_history[-100:]
            returns_std = np.std(self.returns_history)
            if returns_std > 0.001:
                sharpe_component = pnl / returns_std
                reward += self.sharpe_weight * sharpe_component * 0.1

        # 3. Drawdown penalty
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        if self.max_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
            reward -= self.drawdown_penalty * drawdown

        # 4. Turnover penalty (discourage excessive trading)
        reward -= self.turnover_penalty * abs(position_change)

        return float(reward)

    def reset(self):
        """Reset for new episode."""
        self.returns_history = []
        self.max_portfolio_value = 0.0


class TradingEnvironment:
    """
    Trading environment for RL agents.

    FinRL pattern: Gymnasium-compatible interface.

    If gymnasium is installed, this is a proper Gym env.
    Otherwise, it provides a compatible interface.
    """

    formula_id = 71001
    name = "TradingEnvironment"

    def __init__(self,
                 initial_capital: float = 10000.0,
                 max_position: float = 1.0,
                 transaction_cost: float = 0.001,
                 reward_scaling: float = 1.0):
        """
        Initialize environment.

        Args:
            initial_capital: Starting capital
            max_position: Maximum position size (1 = 100% of capital)
            transaction_cost: Cost per transaction (0.001 = 0.1%)
            reward_scaling: Scale rewards for numerical stability
        """
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling

        # State
        self.capital = initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.current_price = 0.0

        # History for state construction
        self.price_history: List[float] = []
        self.flow_history: List[float] = []

        # Reward shaper
        self.reward_shaper = RewardShaper()

        # Episode tracking
        self.step_count = 0
        self.total_pnl = 0.0

        # Define spaces if gymnasium available
        if HAS_GYM:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(TradingState.dim(),),
                dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1, high=1,
                shape=(1,),
                dtype=np.float32
            )

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.

        Args:
            seed: Random seed

        Returns:
            (initial_state, info_dict)
        """
        if seed is not None:
            np.random.seed(seed)

        self.capital = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.current_price = 0.0
        self.price_history = []
        self.flow_history = []
        self.reward_shaper.reset()
        self.step_count = 0
        self.total_pnl = 0.0

        # Return zero state
        state = TradingState(
            price=0.0, price_change=0.0, volatility=0.0,
            flow=0.0, flow_momentum=0.0, spread=0.0,
            position=0.0, unrealized_pnl=0.0, cash_ratio=1.0,
            drawdown=0.0, hour=0, day_of_week=0
        )

        return state.to_array(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take action and return new state.

        Args:
            action: Target position array

        Returns:
            (state, reward, terminated, truncated, info)
        """
        self.step_count += 1

        # Parse action
        target_position = float(np.clip(action[0], -1, 1)) * self.max_position

        # Calculate position change
        position_change = target_position - self.position
        old_position = self.position

        # Transaction cost
        cost = abs(position_change) * self.current_price * self.transaction_cost

        # Update position
        self.position = target_position

        if position_change != 0 and self.current_price > 0:
            self.entry_price = self.current_price

        # Calculate PnL
        if old_position != 0 and len(self.price_history) >= 2:
            price_change = (self.current_price - self.price_history[-2]) / self.price_history[-2]
            pnl = old_position * price_change * self.capital - cost
            pnl_pct = pnl / self.capital
        else:
            pnl = -cost
            pnl_pct = -cost / self.capital

        self.capital += pnl
        self.total_pnl += pnl

        # Compute reward
        reward = self.reward_shaper.compute_reward(
            pnl=pnl_pct,
            portfolio_value=self.capital,
            position_change=abs(position_change),
        ) * self.reward_scaling

        # Build state
        state = self._build_state()

        # Check termination
        terminated = self.capital <= 0
        truncated = self.step_count >= 10000

        info = {
            'capital': self.capital,
            'position': self.position,
            'pnl': pnl,
            'total_pnl': self.total_pnl,
            'step': self.step_count,
        }

        return state.to_array(), float(reward), terminated, truncated, info

    def update_market(self, price: float, flow: float,
                      spread: float = 0.0001,
                      hour: int = 12, day_of_week: int = 0):
        """
        Update market state.

        Call this before step() with new market data.

        Args:
            price: Current price
            flow: Current blockchain flow signal
            spread: Current spread
            hour: Hour of day
            day_of_week: Day of week
        """
        self.current_price = price
        self.price_history.append(price)
        self.flow_history.append(flow)

        # Keep limited history
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
        if len(self.flow_history) > 100:
            self.flow_history = self.flow_history[-100:]

        self._current_spread = spread
        self._current_hour = hour
        self._current_dow = day_of_week

    def _build_state(self) -> TradingState:
        """Build current state from history."""
        # Price change
        if len(self.price_history) >= 2:
            price_change = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
        else:
            price_change = 0.0

        # Volatility
        if len(self.price_history) >= 20:
            returns = np.diff(self.price_history[-20:]) / np.array(self.price_history[-20:-1])
            volatility = np.std(returns)
        else:
            volatility = 0.01

        # Flow and momentum
        flow = self.flow_history[-1] if self.flow_history else 0.0
        if len(self.flow_history) >= 5:
            flow_momentum = np.mean(self.flow_history[-3:]) - np.mean(self.flow_history[-10:-3])
        else:
            flow_momentum = 0.0

        # Unrealized PnL
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl = self.position * (self.current_price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl = 0.0

        # Cash ratio
        position_value = abs(self.position) * self.capital
        total_value = self.capital
        cash_ratio = 1 - abs(self.position)

        # Drawdown
        max_capital = max(self.initial_capital, self.capital)
        drawdown = (max_capital - self.capital) / max_capital if max_capital > 0 else 0.0

        return TradingState(
            price=self.current_price,
            price_change=price_change,
            volatility=volatility,
            flow=flow,
            flow_momentum=flow_momentum,
            spread=getattr(self, '_current_spread', 0.0001),
            position=self.position,
            unrealized_pnl=unrealized_pnl,
            cash_ratio=cash_ratio,
            drawdown=drawdown,
            hour=getattr(self, '_current_hour', 12),
            day_of_week=getattr(self, '_current_dow', 0),
        )

    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self.capital

    def render(self):
        """Render environment state."""
        print(f"Step {self.step_count}: Capital=${self.capital:.2f}, "
              f"Position={self.position:.2f}, PnL=${self.total_pnl:.2f}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("Trading Environment Demo")
    print("=" * 50)
    print(f"Gymnasium available: {HAS_GYM}")

    env = TradingEnvironment(
        initial_capital=10000.0,
        max_position=1.0,
        transaction_cost=0.001,
    )

    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")

    # Simulate some trading
    np.random.seed(42)
    price = 42000.0

    for step in range(20):
        # Update market with random walk
        price *= (1 + np.random.randn() * 0.001)
        flow = np.random.randn() * 0.5
        env.update_market(price, flow)

        # Random action
        action = np.array([np.random.randn() * 0.2])

        state, reward, terminated, truncated, info = env.step(action)

        if step % 5 == 0:
            print(f"Step {step}: reward={reward:.4f}, capital=${info['capital']:.2f}")

    print(f"\nFinal: Capital=${env.capital:.2f}, Total PnL=${env.total_pnl:.2f}")
