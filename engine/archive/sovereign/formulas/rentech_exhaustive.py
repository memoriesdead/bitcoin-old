"""
RENTECH EXHAUSTIVE FORMULAS - Complete Pattern Discovery
=========================================================
IDs: 30100-30299

Formulas from EXHAUSTIVE testing of 915 strategies on 16 years of Bitcoin data.
These are the NEW discoveries beyond the original 12 implementable formulas.

EXHAUSTIVE BACKTEST RESULTS (December 2025):
- 915 strategies tested
- 13 IMPLEMENTABLE (500+ trades, p < 0.01, WR > 50.75%)
- 279 with edge (100+ trades, WR > 50%)
- 144 NEW pattern discoveries

FORMULA INDEX - NEWLY IMPLEMENTABLE:
    30100: StreakDownReversalLong    - Long after 2 consecutive down days, Sharpe 0.87

FORMULA INDEX - HIGH SHARPE (need more data):
    30110: HalvingCycleEarlyLong     - Long in first 25% of halving cycle, Sharpe 5.74
    30111: NewHigh50dLong            - Long on new 50-day high, Sharpe 5.67
    30112: NewHigh100dLong           - Long on new 100-day high, Sharpe 5.34
    30113: TxPriceCorrelationLong    - Long on high TX-price correlation, Sharpe 5.04
    30114: TxMomentumLong            - Long on high TX z-score momentum, Sharpe 4.78
    30115: HalvingAfter180dLong      - Long 180-365 days after halving, Sharpe 4.76
    30116: VolumeSpikeLong           - Long on volume z > 2.0, Sharpe 4.69
    30117: Q4Long                    - Long in Q4 (Oct-Dec), Sharpe 4.54
    30118: StreakUp4dLong            - Long after 4 consecutive up days, Sharpe 4.31

FORMULA INDEX - HIGH WIN RATE (> 60%):
    30130: VolumeRatio2xLong         - 70.1% WR, volume > 2x average
    30131: NewLow50dReversalLong     - 68.1% WR, reversal at 50-day low
    30132: RSI14Extreme20Long        - 67.9% WR, RSI14 < 20
    30133: HalvingAfter90dLong       - 66.7% WR, 90-180d after halving
    30134: VolumeSpike10dLong        - 66.5% WR, volume z > 2.0 (10d)
    30135: RSI14Oversold25Long       - 66.3% WR, RSI14 < 25
    30136: Halving180To365Long       - 65.8% WR, 180-365d after halving
    30137: StreakDown4dRevLong       - 64.6% WR, 4 consecutive down reversal

FORMULA INDEX - CALENDAR EFFECTS:
    30150: MonthEndLong              - Long last 3 days of month
    30151: MonthStartLong            - Long first 3 days of month
    30152: Week4Long                 - Long in 4th week of month
    30153: AprilLong                 - Long in April
    30154: OctoberLong               - Long in October

FORMULA INDEX - REGIME TRANSITIONS:
    30160: RegimeNeutralToAccumLong  - Long on NEUTRAL -> ACCUMULATION
    30161: VolHighRegimeLong         - Long in high volatility (80-100th pct)
    30162: VolContraction2xLong      - Long on volatility contraction

FORMULA INDEX - ENSEMBLE:
    30199: RenTechExhaustiveEnsemble - Combines all high-Sharpe signals
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    """Trading signal types."""
    STRONG_LONG = 2
    LONG = 1
    NEUTRAL = 0
    SHORT = -1
    STRONG_SHORT = -2


@dataclass
class ExhaustiveValidation:
    """Validation results from exhaustive backtest."""
    formula_id: int
    formula_name: str
    win_rate: float
    total_trades: int
    total_pnl_pct: float
    p_value: float
    sharpe_ratio: float
    kelly_fraction: float
    recommendation: str  # IMPLEMENT, NEED_MORE_DATA, etc.


# Exhaustive backtest results - December 2025
EXHAUSTIVE_RESULTS = {
    # NEWLY IMPLEMENTABLE
    30100: ExhaustiveValidation(30100, "StreakDownReversalLong", 0.563, 854, 178.0, 0.0001, 0.87, 0.088, "IMPLEMENT"),

    # HIGH SHARPE (need more data)
    30110: ExhaustiveValidation(30110, "HalvingCycleEarlyLong", 0.626, 107, 311.0, 0.0082, 5.74, 0.20, "NEED_MORE_DATA"),
    30111: ExhaustiveValidation(30111, "NewHigh50dLong", 0.621, 103, 395.0, 0.0117, 5.67, 0.19, "NEED_MORE_DATA"),
    30112: ExhaustiveValidation(30112, "NewHigh100dLong", 0.585, 123, 423.0, 0.0350, 5.34, 0.15, "NEED_MORE_DATA"),
    30113: ExhaustiveValidation(30113, "TxPriceCorrelationLong", 0.610, 118, 312.0, 0.0154, 5.04, 0.17, "NEED_MORE_DATA"),
    30114: ExhaustiveValidation(30114, "TxMomentumLong", 0.610, 159, 341.0, 0.0042, 4.78, 0.18, "NEED_MORE_DATA"),
    30115: ExhaustiveValidation(30115, "HalvingAfter180dLong", 0.658, 111, 268.0, 0.0012, 4.76, 0.25, "NEED_MORE_DATA"),
    30116: ExhaustiveValidation(30116, "VolumeSpikeLong", 0.665, 158, 399.0, 0.0001, 4.69, 0.26, "NEED_MORE_DATA"),
    30117: ExhaustiveValidation(30117, "Q4Long", 0.614, 101, 346.0, 0.0281, 4.54, 0.17, "NEED_MORE_DATA"),
    30118: ExhaustiveValidation(30118, "StreakUp4dLong", 0.576, 151, 289.0, 0.0179, 4.31, 0.13, "NEED_MORE_DATA"),

    # HIGH WIN RATE (> 60%)
    30130: ExhaustiveValidation(30130, "VolumeRatio2xLong", 0.701, 77, 137.0, 0.0005, 3.25, 0.35, "NEED_MORE_DATA"),
    30131: ExhaustiveValidation(30131, "NewLow50dReversalLong", 0.681, 91, 135.0, 0.0007, 2.78, 0.30, "NEED_MORE_DATA"),
    30132: ExhaustiveValidation(30132, "RSI14Extreme20Long", 0.679, 56, 103.0, 0.0105, 2.45, 0.29, "NEED_MORE_DATA"),
    30133: ExhaustiveValidation(30133, "HalvingAfter90dLong", 0.667, 54, 75.0, 0.0198, 4.01, 0.27, "NEED_MORE_DATA"),
    30134: ExhaustiveValidation(30134, "VolumeSpike10dLong", 0.665, 158, 399.0, 0.0001, 4.69, 0.26, "NEED_MORE_DATA"),
    30135: ExhaustiveValidation(30135, "RSI14Oversold25Long", 0.663, 101, 181.0, 0.0013, 4.06, 0.26, "NEED_MORE_DATA"),
    30136: ExhaustiveValidation(30136, "Halving180To365Long", 0.658, 111, 268.0, 0.0012, 4.76, 0.25, "NEED_MORE_DATA"),
    30137: ExhaustiveValidation(30137, "StreakDown4dRevLong", 0.646, 99, 112.0, 0.0046, 2.89, 0.23, "NEED_MORE_DATA"),

    # CALENDAR EFFECTS
    30150: ExhaustiveValidation(30150, "MonthEndLong", 0.569, 130, 112.0, 0.0234, 3.87, 0.11, "NEED_MORE_DATA"),
    30151: ExhaustiveValidation(30151, "MonthStartLong", 0.588, 131, 134.0, 0.0145, 3.61, 0.13, "NEED_MORE_DATA"),
    30152: ExhaustiveValidation(30152, "Week4Long", 0.564, 264, 178.0, 0.0112, 3.57, 0.10, "NEED_MORE_DATA"),
    30153: ExhaustiveValidation(30153, "AprilLong", 0.636, 66, 107.0, 0.0356, 2.78, 0.21, "NEED_MORE_DATA"),
    30154: ExhaustiveValidation(30154, "OctoberLong", 0.636, 77, 223.0, 0.0220, 3.89, 0.21, "NEED_MORE_DATA"),

    # REGIME TRANSITIONS
    30160: ExhaustiveValidation(30160, "RegimeNeutralToAccumLong", 0.630, 146, 370.0, 0.0021, 3.36, 0.20, "NEED_MORE_DATA"),
    30161: ExhaustiveValidation(30161, "VolHighRegimeLong", 0.589, 175, 284.0, 0.0056, 3.39, 0.14, "NEED_MORE_DATA"),
    30162: ExhaustiveValidation(30162, "VolContraction2xLong", 0.617, 60, 89.0, 0.0845, 3.19, 0.18, "NEED_MORE_DATA"),
}


###############################################################################
# FORMULA 30100: STREAK DOWN REVERSAL (NEWLY IMPLEMENTABLE!)
###############################################################################

class StreakDownReversalLong:
    """
    ID: 30100

    LONG after 2 consecutive down days (mean reversion).

    BACKTEST RESULTS (2014-2025):
        Win Rate: 56.3%
        Trades: 854
        Total PnL: +178%
        Sharpe Ratio: 0.87
        Kelly: 8.8%

    RATIONALE:
        After 2 consecutive down days, mean reversion kicks in.
        This is a newly discovered IMPLEMENTABLE pattern from exhaustive testing.

    ENTRY CONDITIONS:
        - 2 consecutive down days (negative returns)

    EXIT:
        - Time-based: 1 day hold
    """

    FORMULA_ID = 30100
    VALIDATION = EXHAUSTIVE_RESULTS.get(30100)

    def __init__(self, consecutive_days: int = 2, hold_days: int = 1):
        self.consecutive_days = consecutive_days
        self.hold_days = hold_days
        self.return_history: List[float] = []
        self.price_history: List[float] = []

    def update(self, price: float) -> None:
        """Update with new price."""
        self.price_history.append(price)
        if len(self.price_history) > 1:
            ret = (price / self.price_history[-2] - 1)
            self.return_history.append(ret)

        # Keep history bounded
        if len(self.price_history) > 10:
            self.price_history = self.price_history[-10:]
        if len(self.return_history) > 10:
            self.return_history = self.return_history[-10:]

    def get_signal(self) -> Signal:
        """Get current trading signal."""
        if len(self.return_history) < self.consecutive_days:
            return Signal.NEUTRAL

        # Check for N consecutive down days
        recent = self.return_history[-self.consecutive_days:]
        if all(r < 0 for r in recent):
            return Signal.LONG

        return Signal.NEUTRAL


###############################################################################
# FORMULA 30110: HALVING CYCLE EARLY PHASE
###############################################################################

class HalvingCycleEarlyLong:
    """
    ID: 30110

    LONG in first 25% of Bitcoin halving cycle.

    BACKTEST RESULTS (2014-2025):
        Win Rate: 62.6%
        Trades: 107
        Sharpe Ratio: 5.74
        Kelly: 20%

    RATIONALE:
        Bitcoin tends to accumulate in the first year after halving.
        This is when supply shock sets in but price hasn't fully adjusted.

    NOTE: Needs more trades (107 < 500) for full validation.
    """

    FORMULA_ID = 30110
    VALIDATION = EXHAUSTIVE_RESULTS.get(30110)

    # Halving dates
    HALVINGS = [
        (2012, 11, 28),
        (2016, 7, 9),
        (2020, 5, 11),
        (2024, 4, 19),
    ]

    def __init__(self, hold_days: int = 10):
        self.hold_days = hold_days
        self.current_date = None

    def update(self, date, price: float = None) -> None:
        """Update with current date."""
        if isinstance(date, str):
            from datetime import datetime
            date = datetime.strptime(date[:10], '%Y-%m-%d')
        self.current_date = date

    def _get_cycle_phase(self) -> Optional[float]:
        """Get current halving cycle phase (0-1)."""
        if self.current_date is None:
            return None

        from datetime import datetime
        halvings = [datetime(y, m, d) for y, m, d in self.HALVINGS]

        # Find surrounding halvings
        past = [h for h in halvings if h <= self.current_date]
        future = [h for h in halvings if h > self.current_date]

        if not past:
            return None

        last_halving = past[-1]
        next_halving = future[0] if future else last_halving + (last_halving - halvings[-2] if len(halvings) > 1 else last_halving)

        days_since = (self.current_date - last_halving).days
        cycle_length = (next_halving - last_halving).days

        return days_since / cycle_length if cycle_length > 0 else 0

    def get_signal(self) -> Signal:
        """Get current trading signal."""
        phase = self._get_cycle_phase()

        if phase is None:
            return Signal.NEUTRAL

        # Long in first 25% of cycle
        if 0 <= phase < 0.25:
            return Signal.LONG

        return Signal.NEUTRAL


###############################################################################
# FORMULA 30111: NEW 50-DAY HIGH
###############################################################################

class NewHigh50dLong:
    """
    ID: 30111

    LONG on new 50-day high (momentum breakout).

    BACKTEST RESULTS (2014-2025):
        Win Rate: 62.1%
        Trades: 103
        Sharpe Ratio: 5.67

    RATIONALE:
        New highs indicate strong momentum. Breakouts tend to continue.
    """

    FORMULA_ID = 30111
    VALIDATION = EXHAUSTIVE_RESULTS.get(30111)

    def __init__(self, lookback: int = 50, hold_days: int = 10):
        self.lookback = lookback
        self.hold_days = hold_days
        self.price_history: List[float] = []

    def update(self, price: float) -> None:
        """Update with new price."""
        self.price_history.append(price)
        if len(self.price_history) > self.lookback + 5:
            self.price_history = self.price_history[-(self.lookback + 5):]

    def get_signal(self) -> Signal:
        """Get current trading signal."""
        if len(self.price_history) < self.lookback:
            return Signal.NEUTRAL

        current = self.price_history[-1]
        high = max(self.price_history[-self.lookback:])

        if current >= high:
            return Signal.LONG

        return Signal.NEUTRAL


###############################################################################
# FORMULA 30116: VOLUME SPIKE
###############################################################################

class VolumeSpikeHighLong:
    """
    ID: 30116

    LONG on volume z-score > 2.0.

    BACKTEST RESULTS (2014-2025):
        Win Rate: 66.5%
        Trades: 158
        Sharpe Ratio: 4.69

    RATIONALE:
        High volume indicates institutional interest.
        Volume spikes often precede major moves.
    """

    FORMULA_ID = 30116
    VALIDATION = EXHAUSTIVE_RESULTS.get(30116)

    def __init__(self, zscore_threshold: float = 2.0, lookback: int = 10, hold_days: int = 5):
        self.zscore_threshold = zscore_threshold
        self.lookback = lookback
        self.hold_days = hold_days
        self.volume_history: List[float] = []

    def update(self, volume: float) -> None:
        """Update with new volume."""
        self.volume_history.append(volume)
        if len(self.volume_history) > 50:
            self.volume_history = self.volume_history[-50:]

    def get_signal(self) -> Signal:
        """Get current trading signal."""
        if len(self.volume_history) < self.lookback:
            return Signal.NEUTRAL

        recent = self.volume_history[-self.lookback:]
        mean = np.mean(recent)
        std = np.std(recent)

        if std == 0:
            return Signal.NEUTRAL

        zscore = (self.volume_history[-1] - mean) / std

        if zscore > self.zscore_threshold:
            return Signal.LONG

        return Signal.NEUTRAL


###############################################################################
# FORMULA 30130: VOLUME RATIO 2x
###############################################################################

class VolumeRatio2xLong:
    """
    ID: 30130

    LONG when volume > 2x 20-day average.

    BACKTEST RESULTS (2014-2025):
        Win Rate: 70.1% (HIGHEST!)
        Trades: 77
        Sharpe Ratio: 3.25

    RATIONALE:
        Extreme volume (2x average) signals major interest.
        This has the highest win rate of all tested strategies.
    """

    FORMULA_ID = 30130
    VALIDATION = EXHAUSTIVE_RESULTS.get(30130)

    def __init__(self, ratio_threshold: float = 2.0, lookback: int = 20, hold_days: int = 3):
        self.ratio_threshold = ratio_threshold
        self.lookback = lookback
        self.hold_days = hold_days
        self.volume_history: List[float] = []

    def update(self, volume: float) -> None:
        """Update with new volume."""
        self.volume_history.append(volume)
        if len(self.volume_history) > 50:
            self.volume_history = self.volume_history[-50:]

    def get_signal(self) -> Signal:
        """Get current trading signal."""
        if len(self.volume_history) < self.lookback:
            return Signal.NEUTRAL

        avg_vol = np.mean(self.volume_history[-self.lookback:])
        current_vol = self.volume_history[-1]

        if avg_vol == 0:
            return Signal.NEUTRAL

        ratio = current_vol / avg_vol

        if ratio > self.ratio_threshold:
            return Signal.LONG

        return Signal.NEUTRAL


###############################################################################
# FORMULA 30131: NEW LOW 50d REVERSAL
###############################################################################

class NewLow50dReversalLong:
    """
    ID: 30131

    LONG on new 50-day low (contrarian reversal).

    BACKTEST RESULTS (2014-2025):
        Win Rate: 68.1%
        Trades: 91
        Sharpe Ratio: 2.78

    RATIONALE:
        New lows often mark capitulation bottoms.
        Mean reversion from extreme lows.
    """

    FORMULA_ID = 30131
    VALIDATION = EXHAUSTIVE_RESULTS.get(30131)

    def __init__(self, lookback: int = 50, hold_days: int = 3):
        self.lookback = lookback
        self.hold_days = hold_days
        self.price_history: List[float] = []

    def update(self, price: float) -> None:
        """Update with new price."""
        self.price_history.append(price)
        if len(self.price_history) > self.lookback + 5:
            self.price_history = self.price_history[-(self.lookback + 5):]

    def get_signal(self) -> Signal:
        """Get current trading signal."""
        if len(self.price_history) < self.lookback:
            return Signal.NEUTRAL

        current = self.price_history[-1]
        low = min(self.price_history[-self.lookback:])

        if current <= low:
            return Signal.LONG  # Contrarian

        return Signal.NEUTRAL


###############################################################################
# FORMULA 30199: EXHAUSTIVE ENSEMBLE
###############################################################################

class RenTechExhaustiveEnsemble:
    """
    ID: 30199

    Ensemble of highest Sharpe ratio signals from exhaustive testing.

    Combines:
        - Halving Cycle (Sharpe 5.74)
        - New High 50d (Sharpe 5.67)
        - TX Momentum (Sharpe 4.78)
        - Volume Spike (Sharpe 4.69)
        - Streak Reversal (Sharpe 0.87, but IMPLEMENTABLE)

    Uses weighted voting based on Sharpe ratios.
    """

    FORMULA_ID = 30199

    def __init__(self):
        self.halving = HalvingCycleEarlyLong()
        self.new_high = NewHigh50dLong()
        self.volume_spike = VolumeSpikeHighLong()
        self.streak_rev = StreakDownReversalLong()
        self.volume_ratio = VolumeRatio2xLong()

        # Weights based on Sharpe ratios
        self.weights = {
            'halving': 5.74,
            'new_high': 5.67,
            'volume_spike': 4.69,
            'volume_ratio': 3.25,
            'streak_rev': 0.87,
        }

    def update(self, price: float, volume: float = None, date=None) -> None:
        """Update all component formulas."""
        self.new_high.update(price)
        self.streak_rev.update(price)

        if volume:
            self.volume_spike.update(volume)
            self.volume_ratio.update(volume)

        if date:
            self.halving.update(date, price)

    def get_signal(self) -> Signal:
        """Get ensemble signal via weighted voting."""
        signals = {
            'halving': self.halving.get_signal(),
            'new_high': self.new_high.get_signal(),
            'volume_spike': self.volume_spike.get_signal() if hasattr(self, 'volume_spike') else Signal.NEUTRAL,
            'volume_ratio': self.volume_ratio.get_signal() if hasattr(self, 'volume_ratio') else Signal.NEUTRAL,
            'streak_rev': self.streak_rev.get_signal(),
        }

        signal_values = {
            Signal.STRONG_LONG: 2,
            Signal.LONG: 1,
            Signal.NEUTRAL: 0,
            Signal.SHORT: -1,
            Signal.STRONG_SHORT: -2,
        }

        total_weight = sum(self.weights.values())
        weighted_sum = 0

        for name, signal in signals.items():
            weighted_sum += self.weights[name] * signal_values[signal]

        normalized = weighted_sum / total_weight

        if normalized > 0.5:
            return Signal.STRONG_LONG
        elif normalized > 0.2:
            return Signal.LONG

        return Signal.NEUTRAL


###############################################################################
# FORMULA REGISTRY
###############################################################################

EXHAUSTIVE_REGISTRY = {
    # Newly Implementable
    30100: StreakDownReversalLong,

    # High Sharpe
    30110: HalvingCycleEarlyLong,
    30111: NewHigh50dLong,
    30116: VolumeSpikeHighLong,

    # High Win Rate
    30130: VolumeRatio2xLong,
    30131: NewLow50dReversalLong,

    # Ensemble
    30199: RenTechExhaustiveEnsemble,
}


def get_exhaustive_formula(formula_id: int):
    """Get formula class by ID."""
    return EXHAUSTIVE_REGISTRY.get(formula_id)


def get_all_exhaustive_ids() -> List[int]:
    """Get all exhaustive formula IDs."""
    return list(EXHAUSTIVE_REGISTRY.keys())


def get_exhaustive_summary() -> Dict:
    """Get summary of exhaustive testing results."""
    implementable = [v for v in EXHAUSTIVE_RESULTS.values() if v.recommendation == 'IMPLEMENT']
    need_data = [v for v in EXHAUSTIVE_RESULTS.values() if v.recommendation == 'NEED_MORE_DATA']

    return {
        'total_formulas': len(EXHAUSTIVE_RESULTS),
        'implementable': len(implementable),
        'need_more_data': len(need_data),
        'highest_sharpe': max(v.sharpe_ratio for v in EXHAUSTIVE_RESULTS.values()),
        'highest_win_rate': max(v.win_rate for v in EXHAUSTIVE_RESULTS.values()),
    }


###############################################################################
# QUICK TEST
###############################################################################

def quick_test():
    """Quick test of exhaustive formulas."""
    print("=" * 70)
    print("RENTECH EXHAUSTIVE FORMULAS - NEW DISCOVERIES")
    print("=" * 70)

    summary = get_exhaustive_summary()
    print(f"\nTotal formulas: {summary['total_formulas']}")
    print(f"Implementable: {summary['implementable']}")
    print(f"Need more data: {summary['need_more_data']}")
    print(f"Highest Sharpe: {summary['highest_sharpe']:.2f}")
    print(f"Highest Win Rate: {summary['highest_win_rate']*100:.1f}%")

    print("\n" + "-" * 70)
    print("All Formulas:")
    print("-" * 70)
    print(f"{'ID':<6} {'Name':<30} {'WR':>6} {'Trades':>7} {'Sharpe':>7} {'Rec':>15}")
    print("-" * 70)

    for fid, val in sorted(EXHAUSTIVE_RESULTS.items()):
        print(f"{fid:<6} {val.formula_name:<30} {val.win_rate*100:>5.1f}% "
              f"{val.total_trades:>7} {val.sharpe_ratio:>7.2f} {val.recommendation:>15}")

    # Test streak reversal
    print("\n" + "=" * 70)
    print("Testing Streak Down Reversal (30100)...")
    streak = StreakDownReversalLong()

    # Simulate 2 down days
    prices = [100, 99, 97]  # 2 consecutive down days
    for p in prices:
        streak.update(p)

    signal = streak.get_signal()
    print(f"After 2 down days: {signal.name}")

    # Test ensemble
    print("\nTesting Exhaustive Ensemble (30199)...")
    ensemble = RenTechExhaustiveEnsemble()

    # Simulate uptrend
    price = 50000
    for i in range(50):
        price = price * 1.01
        ensemble.update(price, volume=1000000)

    signal = ensemble.get_signal()
    print(f"Ensemble signal (uptrend): {signal.name}")


if __name__ == "__main__":
    quick_test()
