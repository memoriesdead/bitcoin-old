"""
Hyperopt Parameter Spaces for Renaissance Strategy
===================================================

Parameters for Freqtrade's optimizer to tune formula thresholds.
Uses REAL backtest results - no simulation.
"""

try:
    from freqtrade.optimize.space import Integer, Real
except ImportError:
    class Integer:
        def __init__(self, low, high, **kwargs):
            self.low, self.high = low, high
    class Real:
        def __init__(self, low, high, **kwargs):
            self.low, self.high = low, high


def buy_space():
    """Entry signal parameters."""
    return [
        Real(0.1, 0.6, name='entry_threshold'),
        Real(0.2, 0.7, name='min_confidence'),
        Integer(10, 200, name='min_signals'),
        Real(0.5, 2.0, name='bullish_ratio'),
    ]


def sell_space():
    """Exit signal parameters."""
    return [
        Real(-0.5, 0.0, name='exit_threshold'),
        Real(1.0, 3.0, name='exit_bearish_ratio'),
    ]


def stoploss_space():
    """Stoploss parameters."""
    return [
        Real(-0.10, -0.01, name='stoploss'),
    ]
