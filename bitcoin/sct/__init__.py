#!/usr/bin/env python3
"""
SCT - Statistical Certainty Trading

100% certain of 50.75%+ win rate using Wilson confidence intervals.
Only trades when mathematically proven edge exists.
"""

from .wilson import wilson_lower_bound, wilson_interval
from .certainty import CertaintyChecker, CertaintyStatus
from .strategy_tracker import StrategyStats, StrategyTracker
from .validator import StrategyValidator
from .position_sizer import KellyPositionSizer
from .config import SCTConfig, get_config

__all__ = [
    'wilson_lower_bound',
    'wilson_interval',
    'CertaintyChecker',
    'CertaintyStatus',
    'StrategyStats',
    'StrategyTracker',
    'StrategyValidator',
    'KellyPositionSizer',
    'SCTConfig',
    'get_config',
]

__version__ = '1.0.0'
