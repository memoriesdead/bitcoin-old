"""
Core Module - Sovereign Engine
===============================

Unified types, configuration, and main engine.
"""
# Types
from .types import (
    # Enums
    ExecutionMode,
    DataSource,
    OrderSide,
    OrderType,
    # Core types
    Tick,
    Signal,
    Order,
    SizedOrder,
    ExecutionResult,
    TradeOutcome,
    # Legacy types
    TradeSignal,
    LeadingSignal,
    ExchangeFlow,
    Position,
    TradeResult,
    EngineStats,
    SignalAggregation,
)

# Configuration
from .config import (
    SovereignConfig,
    DataConfig,
    EnginesConfig,
    EngineConfig,
    EnsembleConfig,
    SafetyConfig,
    ExecutionConfig,
    RLConfig,
    TelegramConfig,
    LoggingConfig,
    load_config,
    create_paper_config,
    create_live_config,
)

# Main engine - lazy import to avoid circular dependency
# Import these directly from .sovereign_engine when needed
def __getattr__(name):
    """Lazy import for SovereignEngine to avoid circular imports."""
    if name in ('SovereignEngine', 'create_engine', 'run_paper', 'run_dry'):
        from .sovereign_engine import SovereignEngine, create_engine, run_paper, run_dry
        globals()[name] = locals()[name]
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Enums
    'ExecutionMode',
    'DataSource',
    'OrderSide',
    'OrderType',
    # Types
    'Tick',
    'Signal',
    'Order',
    'SizedOrder',
    'ExecutionResult',
    'TradeOutcome',
    # Legacy
    'TradeSignal',
    'LeadingSignal',
    'ExchangeFlow',
    'Position',
    'TradeResult',
    'EngineStats',
    'SignalAggregation',
    # Config
    'SovereignConfig',
    'DataConfig',
    'EnginesConfig',
    'EngineConfig',
    'EnsembleConfig',
    'SafetyConfig',
    'ExecutionConfig',
    'RLConfig',
    'TelegramConfig',
    'LoggingConfig',
    'load_config',
    'create_paper_config',
    'create_live_config',
    # Engine
    'SovereignEngine',
    'create_engine',
    'run_paper',
    'run_dry',
]
