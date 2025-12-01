"""
TRADING FORMULAS - MINIMAL SERVER VERSION
==========================================
Only imports what's needed for Renaissance Compounding Framework.
"""

from .base import BaseFormula, FormulaRegistry, FORMULA_REGISTRY

# Renaissance Compounding Framework (IDs 801-810) - $100 -> $10,000 IN 46 DAYS
# Master equation: Capital(t) = Capital(0) x (1 + f x edge)^n
try:
    from .renaissance_compounding import (
        MasterGrowthEquation,           # 801
        NetEdgeCalculator,              # 802
        SharpeThresholdFormula,         # 803
        WinRateThresholdFormula,        # 804
        QuarterKellyPositionSizer,      # 805
        TradeFrequencyOptimizer,        # 806
        TimeToTargetCalculator,         # 807
        DrawdownConstrainedGrowth,      # 808
        CompoundProgressTracker,        # 809
        RenaissanceMasterController,    # 810
        RenaissanceSignal,
        create_renaissance_system,
        calculate_trades_to_100x,
        calculate_days_to_target,
    )
    RENAISSANCE_ENABLED = True
except ImportError as e:
    RENAISSANCE_ENABLED = False
    print(f"Renaissance import failed: {e}")

__all__ = [
    "BaseFormula",
    "FormulaRegistry",
    "FORMULA_REGISTRY",
    "RENAISSANCE_ENABLED",
]

# Add Renaissance exports if available
if RENAISSANCE_ENABLED:
    __all__.extend([
        "MasterGrowthEquation",
        "NetEdgeCalculator",
        "SharpeThresholdFormula",
        "WinRateThresholdFormula",
        "QuarterKellyPositionSizer",
        "TradeFrequencyOptimizer",
        "TimeToTargetCalculator",
        "DrawdownConstrainedGrowth",
        "CompoundProgressTracker",
        "RenaissanceMasterController",
        "RenaissanceSignal",
        "create_renaissance_system",
        "calculate_trades_to_100x",
        "calculate_days_to_target",
    ])
