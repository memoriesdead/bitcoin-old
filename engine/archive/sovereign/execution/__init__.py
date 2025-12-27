"""
Execution Module
================

Real exchange execution via CCXT with Freqtrade patterns.

Provides:
- CCXT exchange client
- Order state machine
- Execution modes (PAPER, DRY_RUN, LIVE)
- Safety mechanisms
- Telegram control
"""

from .ccxt_client import (
    CCXTClient,
    ExchangeConfig,
    get_supported_exchanges,
)

from .order_manager import (
    OrderManager,
    OrderState,
    ManagedOrder,
)

from .execution_engine import (
    ExecutionEngine,
    ExecutionMode,
    ExecutionResult,
)

from .dry_run import (
    DryRunExecutor,
    DryRunFill,
)

from .safety import (
    SafetyManager,
    SafetyConfig,
    CircuitBreaker,
    PositionLimiter,
)

from .config import (
    ConfigManager,
    TradingConfig,
    load_config,
)

from .telegram_bot import (
    TelegramBot,
    TelegramNotifier,
    TradingStatus,
    create_telegram_bot,
)

__all__ = [
    # CCXT Client
    'CCXTClient',
    'ExchangeConfig',
    'get_supported_exchanges',

    # Order Manager
    'OrderManager',
    'OrderState',
    'ManagedOrder',

    # Execution Engine
    'ExecutionEngine',
    'ExecutionMode',
    'ExecutionResult',

    # Dry Run
    'DryRunExecutor',
    'DryRunFill',

    # Safety
    'SafetyManager',
    'SafetyConfig',
    'CircuitBreaker',
    'PositionLimiter',

    # Config
    'ConfigManager',
    'TradingConfig',
    'load_config',

    # Telegram
    'TelegramBot',
    'TelegramNotifier',
    'TradingStatus',
    'create_telegram_bot',
]
