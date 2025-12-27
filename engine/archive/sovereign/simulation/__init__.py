"""
RENTECH 1:1 SIMULATION ENGINE
=============================

Pure math trading simulation with full audit trail.

Features:
- Historical replay (2009-2025)
- Live paper trading
- Every trade logged with timestamps
- Exchange cross-reference verification
- Quarter Kelly position sizing
- 85% LONG / 15% SHORT bias

Usage:
    # Historical mode
    python -m engine.sovereign.simulation.run --mode historical

    # Live mode
    python -m engine.sovereign.simulation.run --mode live --duration 3600

Formula IDs Used:
    31001: MACDHistogramLong (Sharpe 2.36)
    31002: GoldenCrossLong (Sharpe 2.03)
    31003: Momentum3dLong
    31004: Momentum10dLong
    31005: MeanRevSMA7pctLong
    31006: StreakDownReversalLong
    31007: RSIOversoldLong
    31050: ExtremeSpikeShort (ONLY SHORT)
    31199: ProductionEnsemble
"""

from .types import (
    SimulationTrade,
    SimulationSession,
    FormulaSignal,
    Position,
    TradeResult,
    Direction,
    ExitReason,
)
from .database import SimulationDatabase
from .trade_logger import TradeLogger
from .formula_engine import ProductionFormulaEngine, PRODUCTION_FORMULA_IDS, FORMULA_CONFIGS
from .historical import HistoricalReplayer, HistoricalTick, IntraHistoricalReplayer
from .live import LivePaperTrader, LiveTick, ExchangeFeed, MultiExchangeFeed
from .verifier import ExchangeVerifier, HistoricalVerifier, PriceSnapshot
from .engine import SimulationEngine, EngineConfig
from .fees import EXCHANGE_FEES, ExchangeFees, get_slippage_estimate, calculate_breakeven_winrate
from .engine_with_fees import SimulationEngineWithFees, EngineConfigWithFees
from .blockchain_runner import BlockchainLiveRunner

__all__ = [
    # Main Engine
    'SimulationEngine',
    'EngineConfig',

    # Formula Engine
    'ProductionFormulaEngine',
    'PRODUCTION_FORMULA_IDS',
    'FORMULA_CONFIGS',

    # Logging
    'TradeLogger',
    'SimulationDatabase',

    # Data Sources
    'HistoricalReplayer',
    'HistoricalTick',
    'IntraHistoricalReplayer',
    'LivePaperTrader',
    'LiveTick',
    'ExchangeFeed',
    'MultiExchangeFeed',

    # Verification
    'ExchangeVerifier',
    'HistoricalVerifier',
    'PriceSnapshot',

    # Types
    'SimulationTrade',
    'SimulationSession',
    'FormulaSignal',
    'Position',
    'TradeResult',
    'Direction',
    'ExitReason',

    # Fees
    'EXCHANGE_FEES',
    'ExchangeFees',
    'get_slippage_estimate',
    'calculate_breakeven_winrate',
    'SimulationEngineWithFees',
    'EngineConfigWithFees',

    # Blockchain Runner
    'BlockchainLiveRunner',
]
