# Renaissance Trading System - Core Module V5
# Modular architecture for scalability (supports 100K+ lines of code)
#
# Directory Structure:
# core/
# ├── data/              # WebSocket feeds and data handling
# ├── filters/           # Signal filtering (Kalman, VPIN, etc.)
# ├── probability/       # Probability models (Platt, Bayesian, etc.)
# ├── exits/             # Exit management (Triple Barrier, Laufer)
# ├── indicators/        # Market indicators (Avellaneda-Stoikov)
# ├── config.py          # Configuration
# └── strategy_base.py   # Base strategy

# Filters Module
from .filters import (
    AdaptiveKalmanFilter,
    AntiWhipsawFilter,
    MasterFilter,
    VPINFilter,
    TimeSeriesMomentum,
    ou_mean_reversion_speed,
    reversal_probability,
    multi_timeframe_momentum,
    momentum_acceleration,
    realized_volatility,
    volatility_regime,
    atr,
    adx_simple,
    hurst_exponent
)

# Probability Module
from .probability import (
    PlattScaling,
    IsotonicCalibration,
    BayesianWinRate,
    RegimeConditionalProbability,
    EnsembleSignalFusion,
    SoftFilterProbability,
    MasterProbabilityEngine
)

# Exits Module
from .exits import (
    TripleBarrierMethod,
    LauferDynamicBetting,
    EnhancedExitManager,
    DynamicKelly,
    kelly_fraction
)

# Indicators Module
from .indicators import (
    AvellanedaStoikov
)

# Config
from .config import CONFIGS, get_config, STARTING_CAPITAL, print_config_summary

# Strategy
from .strategy_base import BaseStrategy

# Formula Engine - 217 Renaissance Formulas
from .formula_engine import FormulaEngine, create_formula_engine, VERSION_FORMULA_SETS

__all__ = [
    # Filters
    'AdaptiveKalmanFilter',
    'AntiWhipsawFilter',
    'MasterFilter',
    'VPINFilter',
    'TimeSeriesMomentum',
    'ou_mean_reversion_speed',
    'reversal_probability',
    'multi_timeframe_momentum',
    'momentum_acceleration',
    'realized_volatility',
    'volatility_regime',
    'atr',
    'adx_simple',
    'hurst_exponent',

    # Probability
    'PlattScaling',
    'IsotonicCalibration',
    'BayesianWinRate',
    'RegimeConditionalProbability',
    'EnsembleSignalFusion',
    'SoftFilterProbability',
    'MasterProbabilityEngine',

    # Exits
    'TripleBarrierMethod',
    'LauferDynamicBetting',
    'EnhancedExitManager',
    'DynamicKelly',
    'kelly_fraction',

    # Indicators
    'AvellanedaStoikov',

    # Config
    'CONFIGS',
    'get_config',
    'STARTING_CAPITAL',
    'print_config_summary',

    # Strategy
    'BaseStrategy',

    # Formula Engine
    'FormulaEngine',
    'create_formula_engine',
    'VERSION_FORMULA_SETS',
]
