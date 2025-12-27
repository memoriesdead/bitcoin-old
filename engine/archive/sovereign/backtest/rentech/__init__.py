"""
RenTech-Style Historical Backtesting System

Tests trading strategies on 16 years of Bitcoin data (2009-2025) using
Renaissance Technologies principles:
- Find tiny edges (win rate >50.75%)
- Statistical rigor (p < 0.01)
- Walk-forward validation (no look-ahead bias)
- High trade volume for compounding

Usage:
    python -m engine.sovereign.backtest.rentech.run_backtest --full
"""

from .data_loader import RentechDataLoader
from .feature_engine import FeatureEngine
from .strategy_factory import StrategyFactory, Strategy
from .walk_forward import WalkForwardEngine, WFConfig
from .statistical_tests import StatisticalValidator, StrategyResult
from .advanced_validation import (
    BootstrapValidator,
    PermutationTester,
    MultipleHypothesisCorrector,
    AlphaDecayAnalyzer,
    ComprehensiveValidator,
    BootstrapResult,
    PermutationResult,
    MultipleTestResult,
    DecayAnalysis,
    ComprehensiveValidation,
)

__all__ = [
    # Data & Features
    'RentechDataLoader',
    'FeatureEngine',
    # Strategy
    'StrategyFactory',
    'Strategy',
    # Walk-Forward
    'WalkForwardEngine',
    'WFConfig',
    # Statistical Tests
    'StatisticalValidator',
    'StrategyResult',
    # Advanced Validation
    'BootstrapValidator',
    'PermutationTester',
    'MultipleHypothesisCorrector',
    'AlphaDecayAnalyzer',
    'ComprehensiveValidator',
    'BootstrapResult',
    'PermutationResult',
    'MultipleTestResult',
    'DecayAnalysis',
    'ComprehensiveValidation',
]
