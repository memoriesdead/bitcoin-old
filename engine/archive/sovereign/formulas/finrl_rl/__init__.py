"""
FinRL - Reinforcement Learning Position Sizing
===============================================

Ported from AI4Finance FinRL (https://github.com/AI4Finance-Foundation/FinRL)

Provides:
- Trading environment (Gymnasium compatible)
- SAC agent for position sizing
- PPO agent as backup
- Reward shaping for trading

Formula IDs: 71001-71010
"""

from .trading_env import (
    TradingEnvironment,
    TradingState,
    TradingAction,
)

from .sac_sizer import (
    SACPositionSizer,
    SACConfig,
)

from .ppo_sizer import (
    PPOPositionSizer,
    PPOConfig,
)

__all__ = [
    # Environment
    'TradingEnvironment',
    'TradingState',
    'TradingAction',

    # SAC
    'SACPositionSizer',
    'SACConfig',

    # PPO
    'PPOPositionSizer',
    'PPOConfig',
]

# Formula ID allocation
FINRL_FORMULA_IDS = {
    71001: 'TradingEnvironment',
    71002: 'SACPositionSizer',
    71003: 'PPOPositionSizer',
    71004: 'RewardShaper',
    71005: 'RLEnsemble',
    71006: 'StateNormalizer',
    71007: 'ActionNormalizer',
    71008: 'ExperienceBuffer',
    71009: 'PolicyNetwork',
    71010: 'ValueNetwork',
}
