"""
Timeframe-Adaptive Mathematical Engine
=======================================

A first-principles approach to the core problem:
"What works for 1 second may not work for 2 seconds."

This engine treats cryptocurrency trading as a pure mathematical problem,
deriving custom formulas that:
1. Automatically detect optimal timeframes
2. Adjust parameters in real-time
3. Maintain consistent edge across regime changes
4. Monitor edge validity and decay

Custom Formulas (TAE-001 to TAE-006):
- TAE-001: Timeframe Validity Score
- TAE-002: Mutual Information for Signal Quality
- TAE-003: Parameter Decay (Ornstein-Uhlenbeck)
- TAE-004: Multi-Scale Consensus
- TAE-005: Edge Half-Life Estimation
- TAE-006: Uncertain Kelly Sizing

Usage:
    from engine.sovereign.formulas.timeframe_adaptive import (
        TimeframeAdaptiveEngine,
        create_engine,
    )

    # Create engine
    engine = create_engine()

    # Set regime from HMM
    engine.set_regime('trending_up', {'trending_up': 0.8, 'consolidation': 0.2})

    # Process tick
    for price in price_stream:
        signal = engine.process(price)
        if signal.tradeable:
            # Execute trade with signal.parameters
            pass

    # Learn from outcome
    engine.learn(pnl=0.01, params_used=signal.parameters.as_dict())
"""

# Core engine
from .core import (
    TimeframeAdaptiveEngine,
    AdaptiveSignal,
    EngineConfig,
    create_engine,
)

# Math primitives
from .math_primitives import (
    tae_001_timeframe_validity,
    tae_001_batch_validity,
    tae_002_mutual_information,
    tae_002_shannon_entropy,
    tae_003_ou_decay,
    tae_003_batch_ou_decay,
    tae_004_consensus,
    tae_004_weighted_consensus,
    tae_005_edge_halflife,
    tae_005_rolling_edge_strength,
    tae_006_uncertain_kelly,
    tae_006_adaptive_kelly,
    get_optimal_timeframe_for_regime,
    get_decay_rate_for_regime,
    compute_timeframe_score,
)

# Components
from .timeframe_selector import (
    TimeframeSelector,
    TimeframeSelection,
    EntropyMeasurer,
    MutualInfoCalculator,
)

from .parameter_controller import (
    ParameterController,
    TradingParameters,
    OUProcessController,
    BayesianUpdater,
    KellySizer,
)

from .signal_aggregator import (
    SignalAggregator,
    AggregatedSignal,
    WaveletDecomposer,
    ConfidenceWeighter,
    ConsensusBuilder,
)

from .validity_monitor import (
    ValidityMonitor,
    EdgeEstimate,
    RegimeState,
    PerformanceSnapshot,
    HalfLifeEstimator,
    RegimeChangeDetector,
    PerformanceTracker,
)

__all__ = [
    # Core
    'TimeframeAdaptiveEngine',
    'AdaptiveSignal',
    'EngineConfig',
    'create_engine',

    # Math primitives
    'tae_001_timeframe_validity',
    'tae_001_batch_validity',
    'tae_002_mutual_information',
    'tae_002_shannon_entropy',
    'tae_003_ou_decay',
    'tae_003_batch_ou_decay',
    'tae_004_consensus',
    'tae_004_weighted_consensus',
    'tae_005_edge_halflife',
    'tae_005_rolling_edge_strength',
    'tae_006_uncertain_kelly',
    'tae_006_adaptive_kelly',
    'get_optimal_timeframe_for_regime',
    'get_decay_rate_for_regime',
    'compute_timeframe_score',

    # Timeframe selection
    'TimeframeSelector',
    'TimeframeSelection',
    'EntropyMeasurer',
    'MutualInfoCalculator',

    # Parameter control
    'ParameterController',
    'TradingParameters',
    'OUProcessController',
    'BayesianUpdater',
    'KellySizer',

    # Signal aggregation
    'SignalAggregator',
    'AggregatedSignal',
    'WaveletDecomposer',
    'ConfidenceWeighter',
    'ConsensusBuilder',

    # Validity monitoring
    'ValidityMonitor',
    'EdgeEstimate',
    'RegimeState',
    'PerformanceSnapshot',
    'HalfLifeEstimator',
    'RegimeChangeDetector',
    'PerformanceTracker',
]
