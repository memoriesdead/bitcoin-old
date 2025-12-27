"""
SOVEREIGN FORMULA LIBRARY - MASTER INDEX
========================================

Complete formula library with 50+ validated trading formulas.
All formulas backtested on 16 years of Bitcoin data (2009-2025).

ID ALLOCATION:
--------------
    10000-19999: ADAPTIVE       - Real-time learning formulas
    20000-29999: PATTERN        - HMM, regime detection
    30000-30099: VALIDATED      - Walk-forward tested
    30100-30199: EXHAUSTIVE     - Microstructure patterns
    31000-31999: PRODUCTION     - Ready for live trading
    40000-49999: BLOCKCHAIN     - Mempool, difficulty (reserved)
    50000-59999: ENSEMBLE       - Combined signals (reserved)
    60000-69999: EXPERIMENTAL   - Research only (reserved)

KEY FINDINGS (from backtesting):
-------------------------------
    - Bitcoin is LONG-biased: 90% of strategies favor LONG
    - Only 1.6% of strategies favor SHORT
    - Best SHORT: Fade 7%+ daily spikes (31050)
    - Best LONG: MACD Histogram (31001), Sharpe 2.36

QUICK START:
-----------
    from engine.sovereign.formulas import (
        get_formula,
        get_production_formulas,
        PRODUCTION_IDS,
        ProductionEnsemble
    )

    # Get a specific formula
    formula = get_formula(31001)  # MACDHistogramLong

    # Get all production-ready formulas
    prod = get_production_formulas()

    # Use the production ensemble
    ensemble = ProductionEnsemble()
    ensemble.update(price)
    signal = ensemble.get_signal()

THE EDGE:
---------
    INFLOW to exchange  = Depositing to SELL = SHORT
    OUTFLOW from exchange = Withdrawing to HOLD = LONG

    We see blockchain flow 10-60 seconds BEFORE it hits exchanges.
"""

# ============================================================================
# ADAPTIVE FORMULAS (10001-10005)
# ============================================================================
from .adaptive import (
    AdaptiveFlowImpactEstimator,   # 10001
    AdaptiveTimingOptimizer,       # 10002
    UniversalRegimeDetector,       # 10003
    BayesianParameterUpdater,      # 10004
    MultiTimescaleAggregator,      # 10005
    BayesianParam,
    AdaptiveTradingEngine,
)

# ============================================================================
# PATTERN RECOGNITION (20001-20011)
# ============================================================================
from .pattern_recognition import (
    BlockchainHMM as HMMRegimeTrader,              # 20001
    StatArbFlowDetector as StatisticalArbitrage,  # 20002
    FlowPatternRecognizer as PatternMatcher,      # 20003
    FlowRegime as VolatilityRegime,               # 20004
    FlowMomentumClassifier as TrendClassifier,    # 20005
    EnsemblePatternVoter as PatternEnsemble,      # 20011
    PatternRecognitionEngine,                     # Master engine
)

# ============================================================================
# RENTECH VALIDATED (30001-30020)
# ============================================================================
from .rentech_validated import (
    MeanRevSMALong as TxZScoreMeanReversion,      # 30001
    TxMomentumHighLong as HighTxMomentum,         # 30002
    ValidationResult as ValidatedFormula,         # Base class
    RenTechEnsemble,                              # Ensemble
    get_formula as get_validated_formula,
    get_all_formula_ids as get_validated_ids,
    get_implementable_formulas,
    get_validation_summary,
)

# ============================================================================
# RENTECH EXHAUSTIVE (30100-30199)
# ============================================================================
from .rentech_exhaustive import (
    StreakDownReversalLong as StreakDownReversal,           # 30100
    HalvingCycleEarlyLong as HalvingCycleEarly,             # 30110
    RenTechExhaustiveEnsemble as ExhaustiveEnsemble,        # 30199
    EXHAUSTIVE_REGISTRY,                                     # Registry
    EXHAUSTIVE_RESULTS,                                      # Results dict
)

# ============================================================================
# RENTECH PRODUCTION (31001-31199) - READY FOR LIVE TRADING
# ============================================================================
from .rentech_production import (
    # LONG formulas - IMPLEMENT
    MACDHistogramLong,            # 31001 - Sharpe 2.36
    GoldenCrossLong,              # 31002 - Sharpe 2.03
    StreakDownReversalLong,       # 31006 - WR 56.3%

    # SHORT formulas - CAUTION
    ExtremeSpikeShort,            # 31050 - Only working SHORT

    # HIGH SHARPE - MONITOR
    HalvingCycleEarlyLong,        # 31101 - Sharpe 5.74

    # ENSEMBLE
    ProductionEnsemble,           # 31199 - Master ensemble

    # Types and registry
    Signal,
    ProductionFormula,
    PRODUCTION_FORMULAS,
    PRODUCTION_REGISTRY,
    get_production_formula,
    get_production_summary,
)

# ============================================================================
# MASTER FORMULA INDEX
# ============================================================================
from .FORMULA_INDEX import (
    # Enums
    FormulaCategory,
    Direction,
    Status,

    # Data class
    FormulaEntry,

    # Registry
    FORMULA_REGISTRY,

    # Lookup functions
    get_formula,
    get_formulas_by_category,
    get_formulas_by_status,
    get_formulas_by_direction,
    get_production_formulas,
    get_long_formulas,
    get_short_formulas,
    get_top_sharpe,
    get_top_win_rate,
    get_formula_class,

    # Summary functions
    print_formula_index,
    print_production_summary,
    get_summary_stats,

    # Quick reference
    PRODUCTION_IDS,
    MONITOR_IDS,
    ID_RANGES,
)


# ============================================================================
# LEGACY FORMULA_IDS (backwards compatibility)
# ============================================================================
FORMULA_IDS = {
    # Adaptive (10001-10005)
    10001: AdaptiveFlowImpactEstimator,
    10002: AdaptiveTimingOptimizer,
    10003: UniversalRegimeDetector,
    10004: BayesianParameterUpdater,
    10005: MultiTimescaleAggregator,

    # Pattern Recognition (20001-20011)
    20001: HMMRegimeTrader,
    20002: StatisticalArbitrage,
    20003: PatternMatcher,
    20004: VolatilityRegime,
    20005: TrendClassifier,
    20011: PatternEnsemble,

    # Production (31001-31199)
    31001: MACDHistogramLong,
    31002: GoldenCrossLong,
    31006: StreakDownReversalLong,
    31050: ExtremeSpikeShort,
    31101: HalvingCycleEarlyLong,
    31199: ProductionEnsemble,
}


# ============================================================================
# __all__ EXPORTS
# ============================================================================
__all__ = [
    # === Adaptive ===
    'AdaptiveFlowImpactEstimator',
    'AdaptiveTimingOptimizer',
    'UniversalRegimeDetector',
    'BayesianParameterUpdater',
    'MultiTimescaleAggregator',
    'BayesianParam',
    'AdaptiveTradingEngine',

    # === Pattern Recognition ===
    'HMMRegimeTrader',
    'StatisticalArbitrage',
    'PatternMatcher',
    'VolatilityRegime',
    'TrendClassifier',
    'PatternEnsemble',
    'PatternRecognitionEngine',

    # === RenTech Validated ===
    'TxZScoreMeanReversion',
    'HighTxMomentum',
    'ValidatedFormula',
    'RenTechEnsemble',
    'get_validated_formula',
    'get_validated_ids',
    'get_implementable_formulas',
    'get_validation_summary',

    # === RenTech Exhaustive ===
    'StreakDownReversal',
    'HalvingCycleEarly',
    'ExhaustiveEnsemble',
    'EXHAUSTIVE_REGISTRY',
    'EXHAUSTIVE_RESULTS',

    # === RenTech Production ===
    'MACDHistogramLong',
    'GoldenCrossLong',
    'StreakDownReversalLong',
    'ExtremeSpikeShort',
    'HalvingCycleEarlyLong',
    'ProductionEnsemble',
    'Signal',
    'ProductionFormula',
    'PRODUCTION_FORMULAS',
    'PRODUCTION_REGISTRY',
    'get_production_formula',
    'get_production_summary',

    # === Master Index ===
    'FormulaCategory',
    'Direction',
    'Status',
    'FormulaEntry',
    'FORMULA_REGISTRY',
    'get_formula',
    'get_formulas_by_category',
    'get_formulas_by_status',
    'get_formulas_by_direction',
    'get_production_formulas',
    'get_long_formulas',
    'get_short_formulas',
    'get_top_sharpe',
    'get_top_win_rate',
    'get_formula_class',
    'print_formula_index',
    'print_production_summary',
    'get_summary_stats',
    'PRODUCTION_IDS',
    'MONITOR_IDS',
    'ID_RANGES',

    # === Legacy ===
    'FORMULA_IDS',
]


# ============================================================================
# QUICK REFERENCE - Print on import if run directly
# ============================================================================
def show_index():
    """Show the complete formula index."""
    print_formula_index()


def show_production():
    """Show production-ready formulas."""
    print_production_summary()
