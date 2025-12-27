"""
RenTech Micro-Pattern Discovery Module
======================================

Formula IDs: 72051-72080

This module implements micro-pattern discovery - finding the "50.75% edge"
patterns that RenTech exploits with high frequency.

Key insight: Many small edges combined > one large edge.
RenTech doesn't need 60% win rate. They need 50.75% at scale.

Components:
- Streak Analysis - Exact sequences (N down days → probability)
- GARCH Volatility - Conditional volatility patterns
- Calendar Effects - Hour, day, week, month patterns
- Whale Sequencing - What happens after large transactions?
"""

from .streak_patterns import (
    StreakAnalyzer,
    SequencePattern,
    ConditionalProbability,
    # Formula implementations
    Streak2DownSignal,       # 72051
    Streak3DownSignal,       # 72052
    Streak2UpSignal,         # 72053
    Streak3UpSignal,         # 72054
    MixedStreakSignal,       # 72055
    StreakBreakSignal,       # 72056
    StreakContinueSignal,    # 72057
    ConditionalStreakSignal, # 72058
    VolatilityStreakSignal,  # 72059
    StreakEnsembleSignal,    # 72060
)

from .garch_signals import (
    GARCHModel,
    VolatilityForecast,
    ConditionalVolatility,
    # Formula implementations
    GARCHBreakoutSignal,     # 72061
    GARCHMeanRevSignal,      # 72062
    VolClusterSignal,        # 72063
    VolRegimeSignal,         # 72064
    GARCHEnsembleSignal,     # 72065
)

from .calendar_micro import (
    CalendarAnalyzer,
    TimePattern,
    SeasonalDecomposition,
    # Formula implementations
    HourOfDaySignal,         # 72066
    DayOfWeekSignal,         # 72067
    WeekOfMonthSignal,       # 72068
    MonthOfYearSignal,       # 72069
    QuarterEndSignal,        # 72070
    YearEndSignal,           # 72071
    HalvingPhaseSignal,      # 72072
    PostHalvingSignal,       # 72073
    CyclePositionSignal,     # 72074
    CalendarEnsembleSignal,  # 72075
)

from .whale_sequences import (
    WhaleTracker,
    FlowSequence,
    LargeTransactionPattern,
    # Formula implementations
    WhaleAccumSignal,        # 72076
    WhaleDistribSignal,      # 72077
    WhaleSequenceSignal,     # 72078
    FlowMomentumSignal,      # 72079
    WhaleEnsembleSignal,     # 72080
)

__all__ = [
    # Streaks
    'StreakAnalyzer', 'SequencePattern', 'ConditionalProbability',
    'Streak2DownSignal', 'Streak3DownSignal', 'Streak2UpSignal',
    'Streak3UpSignal', 'MixedStreakSignal', 'StreakBreakSignal',
    'StreakContinueSignal', 'ConditionalStreakSignal',
    'VolatilityStreakSignal', 'StreakEnsembleSignal',
    # GARCH
    'GARCHModel', 'VolatilityForecast', 'ConditionalVolatility',
    'GARCHBreakoutSignal', 'GARCHMeanRevSignal', 'VolClusterSignal',
    'VolRegimeSignal', 'GARCHEnsembleSignal',
    # Calendar
    'CalendarAnalyzer', 'TimePattern', 'SeasonalDecomposition',
    'HourOfDaySignal', 'DayOfWeekSignal', 'WeekOfMonthSignal',
    'MonthOfYearSignal', 'QuarterEndSignal', 'YearEndSignal',
    'HalvingPhaseSignal', 'PostHalvingSignal', 'CyclePositionSignal',
    'CalendarEnsembleSignal',
    # Whale
    'WhaleTracker', 'FlowSequence', 'LargeTransactionPattern',
    'WhaleAccumSignal', 'WhaleDistribSignal', 'WhaleSequenceSignal',
    'FlowMomentumSignal', 'WhaleEnsembleSignal',
]
