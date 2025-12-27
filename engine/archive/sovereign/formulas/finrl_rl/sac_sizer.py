"""
SAC Position Sizer
==================

Soft Actor-Critic for position sizing.
Ported from FinRL's SAC implementation.

SAC is ideal for trading because:
1. Continuous action space (position sizing)
2. Entropy regularization (exploration)
3. Sample efficient (important with limited data)
4. Stable training
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import pickle

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Normal
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


@dataclass
class SACConfig:
    """Configuration for SAC agent."""
    # Network architecture
    hidden_dim: int = 256
    n_hidden_layers: int = 2

    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99          # Discount factor
    tau: float = 0.005           # Soft update coefficient
    alpha: float = 0.2           # Entropy coefficient
    auto_alpha: bool = True      # Auto-tune alpha

    # Buffer
    buffer_size: int = 100000
    batch_size: int = 256
    min_buffer_size: int = 1000  # Min samples before training

    # Training frequency
    update_frequency: int = 1
    target_update_frequency: int = 1


class ExperienceBuffer:
    """
    Experience replay buffer for SAC.

    Stores (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, state: np.ndarray, action: np.ndarray,
            reward: float, next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch from buffer."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size


if HAS_TORCH:
    class PolicyNetwork(nn.Module):
        """
        Gaussian policy network for SAC.

        Outputs mean and log_std for action distribution.
        """

        def __init__(self, state_dim: int, action_dim: int,
                     hidden_dim: int = 256, n_layers: int = 2):
            super().__init__()

            layers = []
            input_dim = state_dim

            for _ in range(n_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim

            self.trunk = nn.Sequential(*layers)
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)

            # Initialize
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Get mean and log_std."""
            h = self.trunk(state)
            mean = self.mean_head(h)
            log_std = self.log_std_head(h)
            log_std = torch.clamp(log_std, -20, 2)  # Stability
            return mean, log_std

        def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Sample action and compute log probability."""
            mean, log_std = self.forward(state)
            std = log_std.exp()

            # Reparameterization trick
            dist = Normal(mean, std)
            x = dist.rsample()

            # Squash to [-1, 1]
            action = torch.tanh(x)

            # Log prob with squashing correction
            log_prob = dist.log_prob(x)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

            return action, log_prob

        def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
            """Get action for inference."""
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                mean, log_std = self.forward(state_t)

                if deterministic:
                    action = torch.tanh(mean)
                else:
                    action, _ = self.sample(state_t)

                return action.squeeze(0).numpy()

    class QNetwork(nn.Module):
        """
        Q-value network for SAC.

        Takes (state, action) and outputs Q-value.
        """

        def __init__(self, state_dim: int, action_dim: int,
                     hidden_dim: int = 256, n_layers: int = 2):
            super().__init__()

            layers = []
            input_dim = state_dim + action_dim

            for _ in range(n_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim

            layers.append(nn.Linear(hidden_dim, 1))
            self.net = nn.Sequential(*layers)

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

        def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            x = torch.cat([state, action], dim=-1)
            return self.net(x)


class SACPositionSizer:
    """
    SAC-based position sizer (Formula 71002).

    Learns optimal position sizing from trading experience.
    """

    formula_id = 71002
    name = "SACPositionSizer"

    def __init__(self, state_dim: int = 11, action_dim: int = 1,
                 config: Optional[SACConfig] = None):
        """
        Initialize SAC agent.

        Args:
            state_dim: State vector dimension
            action_dim: Action vector dimension (1 for position size)
            config: SAC configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or SACConfig()

        self.is_trained = False
        self.train_steps = 0

        # Initialize if torch available
        if HAS_TORCH:
            self._init_networks()
        else:
            self.policy = None
            self.q1 = None
            self.q2 = None

        # Experience buffer
        self.buffer = ExperienceBuffer(
            self.config.buffer_size,
            state_dim,
            action_dim
        )

        # Stats
        self.stats = {
            'train_steps': 0,
            'policy_loss': 0.0,
            'q_loss': 0.0,
            'alpha': self.config.alpha,
        }

    def _init_networks(self):
        """Initialize neural networks."""
        cfg = self.config

        # Policy
        self.policy = PolicyNetwork(
            self.state_dim, self.action_dim,
            cfg.hidden_dim, cfg.n_hidden_layers
        )

        # Twin Q-networks
        self.q1 = QNetwork(
            self.state_dim, self.action_dim,
            cfg.hidden_dim, cfg.n_hidden_layers
        )
        self.q2 = QNetwork(
            self.state_dim, self.action_dim,
            cfg.hidden_dim, cfg.n_hidden_layers
        )

        # Target networks
        self.q1_target = QNetwork(
            self.state_dim, self.action_dim,
            cfg.hidden_dim, cfg.n_hidden_layers
        )
        self.q2_target = QNetwork(
            self.state_dim, self.action_dim,
            cfg.hidden_dim, cfg.n_hidden_layers
        )
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=cfg.learning_rate
        )
        self.q1_optimizer = optim.Adam(
            self.q1.parameters(), lr=cfg.learning_rate
        )
        self.q2_optimizer = optim.Adam(
            self.q2.parameters(), lr=cfg.learning_rate
        )

        # Auto-alpha
        if cfg.auto_alpha:
            self.target_entropy = -self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = cfg.alpha

    def get_position_size(self, state: np.ndarray,
                          deterministic: bool = False) -> float:
        """
        Get optimal position size for state.

        Args:
            state: Current state vector
            deterministic: Use mean action (no exploration)

        Returns:
            Position size in [-1, 1]
        """
        if not HAS_TORCH or self.policy is None:
            # Fallback: simple heuristic
            return self._fallback_position_size(state)

        action = self.policy.get_action(state, deterministic)
        return float(action[0])

    def _fallback_position_size(self, state: np.ndarray) -> float:
        """
        Simple heuristic position sizing when torch unavailable.

        Uses flow signal and volatility.
        """
        # State indices (from TradingState)
        # 0: price_change, 1: volatility, 2: flow, 3: flow_momentum
        # 4: spread, 5: position, 6: unrealized_pnl, 7: cash_ratio
        # 8: drawdown

        flow = state[2] if len(state) > 2 else 0
        volatility = state[1] if len(state) > 1 else 0.01
        drawdown = state[8] if len(state) > 8 else 0

        # Base position from flow signal
        base_position = np.tanh(flow * 2)  # Scale and squash

        # Reduce position in high volatility
        vol_adjustment = 1.0 / (1 + volatility * 10)

        # Reduce position in drawdown
        dd_adjustment = 1.0 - drawdown

        position = base_position * vol_adjustment * dd_adjustment
        return float(np.clip(position, -1, 1))

    def add_experience(self, state: np.ndarray, action: np.ndarray,
                       reward: float, next_state: np.ndarray, done: bool):
        """
        Add experience to replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        self.buffer.add(state, action, reward, next_state, done)

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns:
            Training metrics
        """
        if not HAS_TORCH:
            return {'error': 'torch not available'}

        if len(self.buffer) < self.config.min_buffer_size:
            return {'error': 'insufficient samples'}

        cfg = self.config

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(cfg.batch_size)

        # Convert to tensors
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(-1)

        # Update Q-networks
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states_t)
            q1_next = self.q1_target(next_states_t, next_actions)
            q2_next = self.q2_target(next_states_t, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards_t + cfg.gamma * (1 - dones_t) * q_next

        q1_loss = F.mse_loss(self.q1(states_t, actions_t), q_target)
        q2_loss = F.mse_loss(self.q2(states_t, actions_t), q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update policy
        new_actions, log_probs = self.policy.sample(states_t)
        q1_new = self.q1(states_t, new_actions)
        q2_new = self.q2(states_t, new_actions)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update alpha
        if cfg.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft update targets
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

        self.train_steps += 1
        self.is_trained = True

        metrics = {
            'policy_loss': policy_loss.item(),
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'alpha': self.alpha,
        }

        self.stats.update(metrics)
        self.stats['train_steps'] = self.train_steps

        return metrics

    def save(self, path: str):
        """Save model to file."""
        if not HAS_TORCH:
            return

        data = {
            'policy_state': self.policy.state_dict(),
            'q1_state': self.q1.state_dict(),
            'q2_state': self.q2.state_dict(),
            'q1_target_state': self.q1_target.state_dict(),
            'q2_target_state': self.q2_target.state_dict(),
            'config': self.config,
            'stats': self.stats,
        }
        torch.save(data, path)

    def load(self, path: str):
        """Load model from file."""
        if not HAS_TORCH:
            return

        data = torch.load(path)
        self.policy.load_state_dict(data['policy_state'])
        self.q1.load_state_dict(data['q1_state'])
        self.q2.load_state_dict(data['q2_state'])
        self.q1_target.load_state_dict(data['q1_target_state'])
        self.q2_target.load_state_dict(data['q2_target_state'])
        self.stats = data.get('stats', self.stats)
        self.is_trained = True

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            **self.stats,
            'buffer_size': len(self.buffer),
            'is_trained': self.is_trained,
            'has_torch': HAS_TORCH,
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("SAC Position Sizer Demo")
    print("=" * 50)
    print(f"PyTorch available: {HAS_TORCH}")

    sizer = SACPositionSizer(state_dim=11, action_dim=1)

    # Test position sizing
    np.random.seed(42)

    for i in range(10):
        state = np.random.randn(11).astype(np.float32)
        position = sizer.get_position_size(state)
        print(f"State sample {i}: position = {position:.3f}")

    # Test training (if torch available)
    if HAS_TORCH:
        print("\nTraining test:")

        # Add some experiences
        for _ in range(2000):
            state = np.random.randn(11).astype(np.float32)
            action = np.random.randn(1).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(11).astype(np.float32)
            done = np.random.random() < 0.01

            sizer.add_experience(state, action, reward, next_state, done)

        # Train
        for step in range(5):
            metrics = sizer.train_step()
            print(f"Step {step}: {metrics}")

    print(f"\nStats: {sizer.get_stats()}")
