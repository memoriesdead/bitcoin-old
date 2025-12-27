"""
PPO Position Sizer
==================

Proximal Policy Optimization for position sizing.
Ported from FinRL's PPO implementation.

PPO is a good backup to SAC because:
1. More stable training
2. Works well with limited data
3. Better for on-policy learning
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    # Network architecture
    hidden_dim: int = 256
    n_hidden_layers: int = 2

    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99          # Discount factor
    gae_lambda: float = 0.95     # GAE lambda
    clip_ratio: float = 0.2      # PPO clip ratio
    value_coef: float = 0.5      # Value loss coefficient
    entropy_coef: float = 0.01   # Entropy bonus coefficient

    # Training schedule
    n_epochs: int = 10           # Epochs per update
    batch_size: int = 64
    rollout_length: int = 2048   # Steps before update


class RolloutBuffer:
    """
    Buffer for storing rollout data.

    PPO uses on-policy data, so we store full trajectories.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Computed during finalize
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0

    def add(self, state: np.ndarray, action: np.ndarray,
            reward: float, value: float, log_prob: float, done: bool):
        """Add step to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)
        self.ptr += 1

    def finalize(self, last_value: float, gamma: float, gae_lambda: float):
        """
        Compute advantages and returns using GAE.

        Args:
            last_value: Value estimate for state after rollout
            gamma: Discount factor
            gae_lambda: GAE lambda
        """
        gae = 0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_done = 0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - next_done) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values[:self.ptr]

        # Normalize advantages
        adv = self.advantages[:self.ptr]
        self.advantages[:self.ptr] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def get_batches(self, batch_size: int):
        """
        Yield batches for training.

        Yields:
            Tuple of (states, actions, old_log_probs, advantages, returns)
        """
        indices = np.random.permutation(self.ptr)

        for start in range(0, self.ptr, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                self.states[batch_indices],
                self.actions[batch_indices],
                self.log_probs[batch_indices],
                self.advantages[batch_indices],
                self.returns[batch_indices],
            )

    def reset(self):
        """Reset buffer for new rollout."""
        self.ptr = 0

    def __len__(self):
        return self.ptr


if HAS_TORCH:
    class ActorCriticNetwork(nn.Module):
        """
        Combined actor-critic network for PPO.

        Shared trunk with separate policy and value heads.
        """

        def __init__(self, state_dim: int, action_dim: int,
                     hidden_dim: int = 256, n_layers: int = 2):
            super().__init__()

            # Shared trunk
            layers = []
            input_dim = state_dim

            for _ in range(n_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.Tanh())  # Tanh often better for PPO
                input_dim = hidden_dim

            self.trunk = nn.Sequential(*layers)

            # Policy head
            self.policy_mean = nn.Linear(hidden_dim, action_dim)
            self.policy_log_std = nn.Parameter(torch.zeros(action_dim))

            # Value head
            self.value_head = nn.Linear(hidden_dim, 1)

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0)

            # Smaller init for policy output
            nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)

        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Returns:
                (action_mean, action_log_std, value)
            """
            h = self.trunk(state)
            mean = self.policy_mean(h)
            log_std = self.policy_log_std.expand_as(mean)
            value = self.value_head(h)
            return mean, log_std, value

        def get_action_and_value(self, state: torch.Tensor,
                                 action: Optional[torch.Tensor] = None):
            """
            Get action, log_prob, entropy, and value.

            If action is provided, compute log_prob for that action.
            Otherwise, sample new action.
            """
            mean, log_std, value = self.forward(state)
            std = log_std.exp()
            dist = Normal(mean, std)

            if action is None:
                action = dist.sample()

            # Squash action
            squashed_action = torch.tanh(action)

            # Log prob with tanh correction
            log_prob = dist.log_prob(action)
            log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)

            entropy = dist.entropy().sum(dim=-1)

            return squashed_action, log_prob, entropy, value.squeeze(-1)

        def get_value(self, state: torch.Tensor) -> torch.Tensor:
            """Get value only."""
            h = self.trunk(state)
            return self.value_head(h).squeeze(-1)


class PPOPositionSizer:
    """
    PPO-based position sizer (Formula 71003).

    Alternative to SAC, better for on-policy learning.
    """

    formula_id = 71003
    name = "PPOPositionSizer"

    def __init__(self, state_dim: int = 11, action_dim: int = 1,
                 config: Optional[PPOConfig] = None):
        """
        Initialize PPO agent.

        Args:
            state_dim: State vector dimension
            action_dim: Action vector dimension
            config: PPO configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or PPOConfig()

        self.is_trained = False
        self.train_steps = 0

        # Initialize if torch available
        if HAS_TORCH:
            self._init_network()
        else:
            self.network = None

        # Rollout buffer
        self.buffer = RolloutBuffer(
            self.config.rollout_length,
            state_dim,
            action_dim
        )

        # Stats
        self.stats = {
            'train_steps': 0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
        }

    def _init_network(self):
        """Initialize neural network."""
        cfg = self.config

        self.network = ActorCriticNetwork(
            self.state_dim, self.action_dim,
            cfg.hidden_dim, cfg.n_hidden_layers
        )

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=cfg.learning_rate
        )

    def get_position_size(self, state: np.ndarray,
                          deterministic: bool = False) -> Tuple[float, float, float]:
        """
        Get optimal position size for state.

        Args:
            state: Current state vector
            deterministic: Use mean action

        Returns:
            (position_size, log_prob, value)
        """
        if not HAS_TORCH or self.network is None:
            # Fallback
            pos = self._fallback_position_size(state)
            return pos, 0.0, 0.0

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)

            if deterministic:
                mean, _, value = self.network(state_t)
                action = torch.tanh(mean)
                log_prob = torch.tensor(0.0)
            else:
                action, log_prob, _, value = self.network.get_action_and_value(state_t)

            return (
                float(action.squeeze().numpy()),
                float(log_prob.numpy()),
                float(value.numpy())
            )

    def _fallback_position_size(self, state: np.ndarray) -> float:
        """Simple heuristic when torch unavailable."""
        flow = state[2] if len(state) > 2 else 0
        volatility = state[1] if len(state) > 1 else 0.01
        drawdown = state[8] if len(state) > 8 else 0

        base_position = np.tanh(flow * 2)
        vol_adjustment = 1.0 / (1 + volatility * 10)
        dd_adjustment = 1.0 - drawdown

        return float(np.clip(base_position * vol_adjustment * dd_adjustment, -1, 1))

    def add_step(self, state: np.ndarray, action: np.ndarray,
                 reward: float, value: float, log_prob: float, done: bool):
        """Add step to rollout buffer."""
        self.buffer.add(state, action, reward, value, log_prob, done)

    def train(self, last_value: float) -> Dict[str, float]:
        """
        Train on collected rollout.

        Args:
            last_value: Value estimate for final state

        Returns:
            Training metrics
        """
        if not HAS_TORCH:
            return {'error': 'torch not available'}

        if len(self.buffer) == 0:
            return {'error': 'empty buffer'}

        cfg = self.config

        # Compute advantages
        self.buffer.finalize(last_value, cfg.gamma, cfg.gae_lambda)

        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(cfg.n_epochs):
            for batch in self.buffer.get_batches(cfg.batch_size):
                states, actions, old_log_probs, advantages, returns = batch

                # Convert to tensors
                states_t = torch.FloatTensor(states)
                actions_t = torch.FloatTensor(actions)
                old_log_probs_t = torch.FloatTensor(old_log_probs)
                advantages_t = torch.FloatTensor(advantages)
                returns_t = torch.FloatTensor(returns)

                # Get current policy outputs
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    states_t, actions_t
                )

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs_t)
                clipped_ratio = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
                policy_loss = -torch.min(ratio * advantages_t, clipped_ratio * advantages_t).mean()

                # Value loss
                value_loss = ((values - returns_t) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # Reset buffer
        self.buffer.reset()

        self.train_steps += 1
        self.is_trained = True

        metrics = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }

        self.stats.update(metrics)
        self.stats['train_steps'] = self.train_steps

        return metrics

    def save(self, path: str):
        """Save model to file."""
        if not HAS_TORCH:
            return

        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'stats': self.stats,
        }, path)

    def load(self, path: str):
        """Load model from file."""
        if not HAS_TORCH:
            return

        data = torch.load(path)
        self.network.load_state_dict(data['network_state'])
        self.optimizer.load_state_dict(data['optimizer_state'])
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
    print("PPO Position Sizer Demo")
    print("=" * 50)
    print(f"PyTorch available: {HAS_TORCH}")

    sizer = PPOPositionSizer(state_dim=11, action_dim=1)

    np.random.seed(42)

    # Collect rollout
    for step in range(100):
        state = np.random.randn(11).astype(np.float32)
        position, log_prob, value = sizer.get_position_size(state)
        reward = np.random.randn() * 0.01

        sizer.add_step(
            state=state,
            action=np.array([position]),
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=step == 99
        )

        if step % 20 == 0:
            print(f"Step {step}: position={position:.3f}, value={value:.3f}")

    # Train
    if HAS_TORCH:
        print("\nTraining:")
        last_state = np.random.randn(11).astype(np.float32)
        _, _, last_value = sizer.get_position_size(last_state)
        metrics = sizer.train(last_value)
        print(f"Metrics: {metrics}")

    print(f"\nStats: {sizer.get_stats()}")
