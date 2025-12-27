"""
Calendar and Time-Based Micro-Patterns
======================================

Formula IDs: 72066-72075

Analyzes time-based patterns: hour, day, week, month, and
Bitcoin-specific patterns like halving cycles.

RenTech insight: Calendar effects are real but small.
You need statistical rigor to exploit them.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta


@dataclass
class TimePattern:
    """Detected time-based pattern."""
    time_unit: str  # hour, day, week, month
    value: int  # e.g., Monday = 0
    avg_return: float
    win_rate: float
    sample_count: int
    significance: float


@dataclass
class SeasonalDecomposition:
    """Seasonal decomposition results."""
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray


@dataclass
class CalendarSignal:
    """Signal from calendar analysis."""
    direction: int
    confidence: float
    time_factor: str
    expected_effect: float
    pattern: Optional[TimePattern]


class CalendarAnalyzer:
    """
    Analyzes calendar effects in returns.
    """

    def __init__(self):
        self.day_of_week_stats: Dict[int, TimePattern] = {}
        self.month_stats: Dict[int, TimePattern] = {}
        self.day_of_month_stats: Dict[int, TimePattern] = {}

    def build_statistics(self, returns: np.ndarray, dates: List[datetime]):
        """Build calendar statistics from historical data."""
        if len(returns) != len(dates):
            raise ValueError("Returns and dates must have same length")

        # Day of week
        dow_returns: Dict[int, List[float]] = {i: [] for i in range(7)}
        for r, d in zip(returns, dates):
            dow_returns[d.weekday()].append(r)

        for dow, rets in dow_returns.items():
            if len(rets) >= 30:
                avg = np.mean(rets)
                std = np.std(rets)
                t_stat = avg / (std / np.sqrt(len(rets)) + 1e-10)

                self.day_of_week_stats[dow] = TimePattern(
                    time_unit='day_of_week',
                    value=dow,
                    avg_return=avg,
                    win_rate=sum(1 for r in rets if r > 0) / len(rets),
                    sample_count=len(rets),
                    significance=abs(t_stat),
                )

        # Month
        month_returns: Dict[int, List[float]] = {i: [] for i in range(1, 13)}
        for r, d in zip(returns, dates):
            month_returns[d.month].append(r)

        for month, rets in month_returns.items():
            if len(rets) >= 10:
                avg = np.mean(rets)
                std = np.std(rets)
                t_stat = avg / (std / np.sqrt(len(rets)) + 1e-10)

                self.month_stats[month] = TimePattern(
                    time_unit='month',
                    value=month,
                    avg_return=avg,
                    win_rate=sum(1 for r in rets if r > 0) / len(rets),
                    sample_count=len(rets),
                    significance=abs(t_stat),
                )

    def get_day_of_week_effect(self, weekday: int) -> Optional[TimePattern]:
        """Get day-of-week effect."""
        return self.day_of_week_stats.get(weekday)

    def get_month_effect(self, month: int) -> Optional[TimePattern]:
        """Get month effect."""
        return self.month_stats.get(month)


# Bitcoin-specific: Halving cycles
HALVING_DATES = [
    datetime(2012, 11, 28),  # First halving
    datetime(2016, 7, 9),    # Second halving
    datetime(2020, 5, 11),   # Third halving
    datetime(2024, 4, 20),   # Fourth halving (approximate)
]

HALVING_CYCLE_DAYS = 4 * 365  # Approximately 4 years


def get_halving_cycle_position(date: datetime) -> Tuple[float, int]:
    """
    Get position in halving cycle (0-1) and which cycle.

    Returns:
        (position_in_cycle, cycle_number)
    """
    for i, halving in enumerate(HALVING_DATES):
        if date < halving:
            if i == 0:
                # Before first halving
                return 0.5, 0
            prev_halving = HALVING_DATES[i - 1]
            days_since = (date - prev_halving).days
            cycle_length = (halving - prev_halving).days
            return days_since / cycle_length, i

    # After last known halving
    last_halving = HALVING_DATES[-1]
    days_since = (date - last_halving).days
    return min(1.0, days_since / HALVING_CYCLE_DAYS), len(HALVING_DATES)


# =============================================================================
# FORMULA IMPLEMENTATIONS (72066-72075)
# =============================================================================

class HourOfDaySignal:
    """
    Formula 72066: Hour of Day Signal

    Trades based on hour-of-day patterns.
    Note: Bitcoin trades 24/7, so patterns differ from stocks.
    """

    FORMULA_ID = 72066

    def __init__(self):
        self.hour_stats: Dict[int, TimePattern] = {}

    def fit(self, returns: np.ndarray, timestamps: List[datetime]):
        """Build hour statistics."""
        hour_returns: Dict[int, List[float]] = {i: [] for i in range(24)}

        for r, ts in zip(returns, timestamps):
            hour_returns[ts.hour].append(r)

        for hour, rets in hour_returns.items():
            if len(rets) >= 30:
                avg = np.mean(rets)
                std = np.std(rets)
                t_stat = avg / (std / np.sqrt(len(rets)) + 1e-10)

                self.hour_stats[hour] = TimePattern(
                    time_unit='hour',
                    value=hour,
                    avg_return=avg,
                    win_rate=sum(1 for r in rets if r > 0) / len(rets),
                    sample_count=len(rets),
                    significance=abs(t_stat),
                )

    def generate_signal(self, current_hour: int) -> CalendarSignal:
        pattern = self.hour_stats.get(current_hour)

        if pattern is None or pattern.significance < 2.0:
            return CalendarSignal(0, 0.0, f'hour_{current_hour}', 0.0, pattern)

        direction = 1 if pattern.avg_return > 0 else -1
        confidence = min(1.0, (pattern.significance - 2.0) / 2.0)

        return CalendarSignal(
            direction=direction,
            confidence=confidence,
            time_factor=f'hour_{current_hour}',
            expected_effect=pattern.avg_return,
            pattern=pattern,
        )


class DayOfWeekSignal:
    """
    Formula 72067: Day of Week Signal

    Classic calendar anomaly - Monday effect, weekend effect, etc.
    """

    FORMULA_ID = 72067

    def __init__(self):
        self.analyzer = CalendarAnalyzer()

    def fit(self, returns: np.ndarray, dates: List[datetime]):
        self.analyzer.build_statistics(returns, dates)

    def generate_signal(self, current_date: datetime) -> CalendarSignal:
        weekday = current_date.weekday()
        pattern = self.analyzer.get_day_of_week_effect(weekday)

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        if pattern is None or pattern.significance < 2.0:
            return CalendarSignal(0, 0.0, day_names[weekday], 0.0, pattern)

        direction = 1 if pattern.avg_return > 0 else -1
        confidence = min(1.0, (pattern.significance - 2.0) / 2.0)

        return CalendarSignal(
            direction=direction,
            confidence=confidence,
            time_factor=day_names[weekday],
            expected_effect=pattern.avg_return,
            pattern=pattern,
        )


class WeekOfMonthSignal:
    """
    Formula 72068: Week of Month Signal

    Beginning/end of month effects.
    """

    FORMULA_ID = 72068

    def __init__(self):
        self.week_stats: Dict[int, TimePattern] = {}

    def fit(self, returns: np.ndarray, dates: List[datetime]):
        week_returns: Dict[int, List[float]] = {i: [] for i in range(1, 6)}

        for r, d in zip(returns, dates):
            week = min(5, (d.day - 1) // 7 + 1)
            week_returns[week].append(r)

        for week, rets in week_returns.items():
            if len(rets) >= 20:
                avg = np.mean(rets)
                std = np.std(rets)
                t_stat = avg / (std / np.sqrt(len(rets)) + 1e-10)

                self.week_stats[week] = TimePattern(
                    time_unit='week_of_month',
                    value=week,
                    avg_return=avg,
                    win_rate=sum(1 for r in rets if r > 0) / len(rets),
                    sample_count=len(rets),
                    significance=abs(t_stat),
                )

    def generate_signal(self, current_date: datetime) -> CalendarSignal:
        week = min(5, (current_date.day - 1) // 7 + 1)
        pattern = self.week_stats.get(week)

        if pattern is None or pattern.significance < 2.0:
            return CalendarSignal(0, 0.0, f'week_{week}', 0.0, pattern)

        direction = 1 if pattern.avg_return > 0 else -1
        confidence = min(1.0, (pattern.significance - 2.0) / 2.0)

        return CalendarSignal(
            direction=direction,
            confidence=confidence,
            time_factor=f'week_{week}_of_month',
            expected_effect=pattern.avg_return,
            pattern=pattern,
        )


class MonthOfYearSignal:
    """
    Formula 72069: Month of Year Signal

    Seasonal patterns by month.
    """

    FORMULA_ID = 72069

    def __init__(self):
        self.analyzer = CalendarAnalyzer()

    def fit(self, returns: np.ndarray, dates: List[datetime]):
        self.analyzer.build_statistics(returns, dates)

    def generate_signal(self, current_date: datetime) -> CalendarSignal:
        month = current_date.month
        pattern = self.analyzer.get_month_effect(month)

        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        if pattern is None or pattern.significance < 2.0:
            return CalendarSignal(0, 0.0, month_names[month], 0.0, pattern)

        direction = 1 if pattern.avg_return > 0 else -1
        confidence = min(1.0, (pattern.significance - 2.0) / 2.0)

        return CalendarSignal(
            direction=direction,
            confidence=confidence,
            time_factor=month_names[month],
            expected_effect=pattern.avg_return,
            pattern=pattern,
        )


class QuarterEndSignal:
    """
    Formula 72070: Quarter End Signal

    Trades around quarter-end effects.
    """

    FORMULA_ID = 72070

    def __init__(self, days_before: int = 5, days_after: int = 5):
        self.days_before = days_before
        self.days_after = days_after
        self.qe_stats: Optional[TimePattern] = None

    def _is_near_quarter_end(self, date: datetime) -> Tuple[bool, int]:
        """Check if near quarter end, return distance."""
        # Quarter end months: March, June, September, December
        qe_months = [3, 6, 9, 12]

        for qe_month in qe_months:
            # Last day of quarter end month
            if qe_month == 12:
                qe_date = datetime(date.year, 12, 31)
            else:
                next_month = qe_month + 1
                qe_date = datetime(date.year, next_month, 1) - timedelta(days=1)

            distance = (date - qe_date).days

            if -self.days_before <= distance <= self.days_after:
                return True, distance

        return False, 999

    def fit(self, returns: np.ndarray, dates: List[datetime]):
        qe_returns = []

        for r, d in zip(returns, dates):
            is_near, _ = self._is_near_quarter_end(d)
            if is_near:
                qe_returns.append(r)

        if len(qe_returns) >= 20:
            avg = np.mean(qe_returns)
            std = np.std(qe_returns)
            t_stat = avg / (std / np.sqrt(len(qe_returns)) + 1e-10)

            self.qe_stats = TimePattern(
                time_unit='quarter_end',
                value=0,
                avg_return=avg,
                win_rate=sum(1 for r in qe_returns if r > 0) / len(qe_returns),
                sample_count=len(qe_returns),
                significance=abs(t_stat),
            )

    def generate_signal(self, current_date: datetime) -> CalendarSignal:
        is_near, distance = self._is_near_quarter_end(current_date)

        if not is_near or self.qe_stats is None or self.qe_stats.significance < 2.0:
            return CalendarSignal(0, 0.0, 'quarter_end', 0.0, self.qe_stats)

        direction = 1 if self.qe_stats.avg_return > 0 else -1
        confidence = min(1.0, (self.qe_stats.significance - 2.0) / 2.0)

        return CalendarSignal(
            direction=direction,
            confidence=confidence,
            time_factor=f'quarter_end_d{distance:+d}',
            expected_effect=self.qe_stats.avg_return,
            pattern=self.qe_stats,
        )


class YearEndSignal:
    """
    Formula 72071: Year End Signal

    End of year effects (Santa rally, tax-loss selling, etc.)
    """

    FORMULA_ID = 72071

    def __init__(self, days_before: int = 10, days_after: int = 5):
        self.days_before = days_before
        self.days_after = days_after
        self.ye_stats: Optional[TimePattern] = None

    def _is_near_year_end(self, date: datetime) -> Tuple[bool, int]:
        ye_date = datetime(date.year, 12, 31)
        distance = (date - ye_date).days

        # Handle early January
        if date.month == 1 and date.day <= self.days_after:
            prev_ye = datetime(date.year - 1, 12, 31)
            distance = (date - prev_ye).days

        if -self.days_before <= distance <= self.days_after:
            return True, distance

        return False, 999

    def fit(self, returns: np.ndarray, dates: List[datetime]):
        ye_returns = []

        for r, d in zip(returns, dates):
            is_near, _ = self._is_near_year_end(d)
            if is_near:
                ye_returns.append(r)

        if len(ye_returns) >= 10:
            avg = np.mean(ye_returns)
            std = np.std(ye_returns)
            t_stat = avg / (std / np.sqrt(len(ye_returns)) + 1e-10)

            self.ye_stats = TimePattern(
                time_unit='year_end',
                value=0,
                avg_return=avg,
                win_rate=sum(1 for r in ye_returns if r > 0) / len(ye_returns),
                sample_count=len(ye_returns),
                significance=abs(t_stat),
            )

    def generate_signal(self, current_date: datetime) -> CalendarSignal:
        is_near, distance = self._is_near_year_end(current_date)

        if not is_near or self.ye_stats is None or self.ye_stats.significance < 1.5:
            return CalendarSignal(0, 0.0, 'year_end', 0.0, self.ye_stats)

        direction = 1 if self.ye_stats.avg_return > 0 else -1
        confidence = min(1.0, (self.ye_stats.significance - 1.5) / 2.0)

        return CalendarSignal(
            direction=direction,
            confidence=confidence,
            time_factor=f'year_end_d{distance:+d}',
            expected_effect=self.ye_stats.avg_return,
            pattern=self.ye_stats,
        )


class HalvingPhaseSignal:
    """
    Formula 72072: Bitcoin Halving Phase Signal

    Trades based on position in ~4 year halving cycle.
    Early cycle typically bullish, late cycle mixed.
    """

    FORMULA_ID = 72072

    def __init__(self):
        self.phase_stats: Dict[str, TimePattern] = {}

    def _get_phase(self, position: float) -> str:
        """Convert position to phase name."""
        if position < 0.25:
            return 'early'
        elif position < 0.5:
            return 'mid_early'
        elif position < 0.75:
            return 'mid_late'
        else:
            return 'late'

    def fit(self, returns: np.ndarray, dates: List[datetime]):
        phase_returns: Dict[str, List[float]] = {
            'early': [], 'mid_early': [], 'mid_late': [], 'late': []
        }

        for r, d in zip(returns, dates):
            pos, _ = get_halving_cycle_position(d)
            phase = self._get_phase(pos)
            phase_returns[phase].append(r)

        for phase, rets in phase_returns.items():
            if len(rets) >= 30:
                avg = np.mean(rets)
                std = np.std(rets)
                t_stat = avg / (std / np.sqrt(len(rets)) + 1e-10)

                self.phase_stats[phase] = TimePattern(
                    time_unit='halving_phase',
                    value=['early', 'mid_early', 'mid_late', 'late'].index(phase),
                    avg_return=avg,
                    win_rate=sum(1 for r in rets if r > 0) / len(rets),
                    sample_count=len(rets),
                    significance=abs(t_stat),
                )

    def generate_signal(self, current_date: datetime) -> CalendarSignal:
        pos, cycle = get_halving_cycle_position(current_date)
        phase = self._get_phase(pos)
        pattern = self.phase_stats.get(phase)

        if pattern is None or pattern.significance < 2.0:
            return CalendarSignal(0, 0.0, f'halving_{phase}', 0.0, pattern)

        direction = 1 if pattern.avg_return > 0 else -1
        confidence = min(1.0, (pattern.significance - 2.0) / 2.0)

        return CalendarSignal(
            direction=direction,
            confidence=confidence,
            time_factor=f'halving_cycle_{cycle}_{phase}',
            expected_effect=pattern.avg_return,
            pattern=pattern,
        )


class PostHalvingSignal:
    """
    Formula 72073: Post-Halving Signal

    Specific focus on days immediately after halving.
    Historically strong performance 30-300 days post-halving.
    """

    FORMULA_ID = 72073

    def __init__(self, early_days: int = 30, peak_start: int = 200, peak_end: int = 400):
        self.early_days = early_days
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.post_halving_stats: Dict[str, TimePattern] = {}

    def _days_since_halving(self, date: datetime) -> int:
        """Get days since most recent halving."""
        for i in range(len(HALVING_DATES) - 1, -1, -1):
            if date >= HALVING_DATES[i]:
                return (date - HALVING_DATES[i]).days
        return -1  # Before first halving

    def fit(self, returns: np.ndarray, dates: List[datetime]):
        period_returns: Dict[str, List[float]] = {
            'immediate': [],  # 0-30 days
            'early': [],      # 30-200 days
            'peak': [],       # 200-400 days
            'late': [],       # 400+ days
        }

        for r, d in zip(returns, dates):
            days = self._days_since_halving(d)
            if days < 0:
                continue

            if days <= self.early_days:
                period_returns['immediate'].append(r)
            elif days <= self.peak_start:
                period_returns['early'].append(r)
            elif days <= self.peak_end:
                period_returns['peak'].append(r)
            else:
                period_returns['late'].append(r)

        for period, rets in period_returns.items():
            if len(rets) >= 20:
                avg = np.mean(rets)
                std = np.std(rets)
                t_stat = avg / (std / np.sqrt(len(rets)) + 1e-10)

                self.post_halving_stats[period] = TimePattern(
                    time_unit='post_halving',
                    value=list(period_returns.keys()).index(period),
                    avg_return=avg,
                    win_rate=sum(1 for r in rets if r > 0) / len(rets),
                    sample_count=len(rets),
                    significance=abs(t_stat),
                )

    def generate_signal(self, current_date: datetime) -> CalendarSignal:
        days = self._days_since_halving(current_date)

        if days < 0:
            return CalendarSignal(0, 0.0, 'pre_halving', 0.0, None)

        if days <= self.early_days:
            period = 'immediate'
        elif days <= self.peak_start:
            period = 'early'
        elif days <= self.peak_end:
            period = 'peak'
        else:
            period = 'late'

        pattern = self.post_halving_stats.get(period)

        if pattern is None or pattern.significance < 1.5:
            return CalendarSignal(0, 0.0, f'post_halving_{period}', 0.0, pattern)

        direction = 1 if pattern.avg_return > 0 else -1
        confidence = min(1.0, (pattern.significance - 1.5) / 2.0)

        return CalendarSignal(
            direction=direction,
            confidence=confidence,
            time_factor=f'post_halving_{period}_d{days}',
            expected_effect=pattern.avg_return,
            pattern=pattern,
        )


class CyclePositionSignal:
    """
    Formula 72074: Continuous Cycle Position Signal

    Uses continuous cycle position (0-1) as feature.
    """

    FORMULA_ID = 72074

    def __init__(self):
        self.position_returns: Dict[int, List[float]] = {}  # Decile -> returns

    def fit(self, returns: np.ndarray, dates: List[datetime]):
        self.position_returns = {i: [] for i in range(10)}

        for r, d in zip(returns, dates):
            pos, _ = get_halving_cycle_position(d)
            decile = min(9, int(pos * 10))
            self.position_returns[decile].append(r)

    def generate_signal(self, current_date: datetime) -> CalendarSignal:
        pos, cycle = get_halving_cycle_position(current_date)
        decile = min(9, int(pos * 10))

        rets = self.position_returns.get(decile, [])

        if len(rets) < 20:
            return CalendarSignal(0, 0.0, f'cycle_decile_{decile}', 0.0, None)

        avg = np.mean(rets)
        std = np.std(rets)
        t_stat = avg / (std / np.sqrt(len(rets)) + 1e-10)

        if abs(t_stat) < 1.5:
            return CalendarSignal(0, 0.0, f'cycle_decile_{decile}', avg, None)

        direction = 1 if avg > 0 else -1
        confidence = min(1.0, (abs(t_stat) - 1.5) / 2.0)

        return CalendarSignal(
            direction=direction,
            confidence=confidence,
            time_factor=f'cycle_pos_{pos:.2f}',
            expected_effect=avg,
            pattern=None,
        )


class CalendarEnsembleSignal:
    """
    Formula 72075: Calendar Ensemble Signal

    Combines all calendar-based signals.
    """

    FORMULA_ID = 72075

    def __init__(self):
        self.signals = [
            DayOfWeekSignal(),
            MonthOfYearSignal(),
            HalvingPhaseSignal(),
            PostHalvingSignal(),
        ]

    def fit(self, returns: np.ndarray, dates: List[datetime]):
        for s in self.signals:
            s.fit(returns, dates)

    def generate_signal(self, current_date: datetime) -> CalendarSignal:
        results = [s.generate_signal(current_date) for s in self.signals]

        # Filter active signals
        active = [r for r in results if r.direction != 0]

        if not active:
            return CalendarSignal(0, 0.0, 'no_calendar_signal', 0.0, None)

        # Weighted vote
        total_dir = sum(r.direction * r.confidence for r in active)
        total_conf = sum(r.confidence for r in active)

        if total_conf > 0:
            avg_dir = total_dir / total_conf
            direction = 1 if avg_dir > 0.3 else (-1 if avg_dir < -0.3 else 0)
            confidence = total_conf / len(self.signals)
        else:
            direction = 0
            confidence = 0.0

        return CalendarSignal(
            direction=direction,
            confidence=confidence,
            time_factor='calendar_ensemble',
            expected_effect=sum(r.expected_effect for r in active) / (len(active) + 1),
            pattern=None,
        )
