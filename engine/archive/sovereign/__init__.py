"""
Sovereign Engine - Simple Blockchain Flow Trading
=================================================
INFLOW to exchange = SHORT (they will sell)
OUTFLOW from exchange = LONG (they are accumulating)

USAGE:
    python -m engine.sovereign.run --capital 100 --duration 300

RenTech Integration (v9.0.0):
    - QLib: Point-in-time data, alpha expressions, LightGBM
    - FinRL: RL position sizing (SAC/PPO)
    - CCXT: Exchange connectivity
    - Freqtrade: Order management, safety, Telegram
    - hftbacktest: Order book simulation

    python -m engine.sovereign.validate  # Run validation
"""

from .run import BlockchainFlowRunner
from .strategy.signal_engine import TradeSignal

# RenTech Integration
from .integration import (
    IntegratedTradingSystem,
    TradingMode,
    IntegratedSignal,
    create_trading_system,
)

__all__ = [
    # Legacy
    'BlockchainFlowRunner',
    'TradeSignal',
    # RenTech Integration
    'IntegratedTradingSystem',
    'TradingMode',
    'IntegratedSignal',
    'create_trading_system',
]
__version__ = '9.0.0'
