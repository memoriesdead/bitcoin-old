"""
Production Formula Engine - Integrates 31001-31199.

Uses the validated RenTech formulas for signal generation.
85% LONG bias, 15% SHORT opportunity.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .types import FormulaSignal


class Signal(Enum):
    """Trading signal types."""
    STRONG_LONG = 2
    LONG = 1
    NEUTRAL = 0
    SHORT = -1
    STRONG_SHORT = -2


# Production formula IDs
PRODUCTION_FORMULA_IDS = [
    # LONG formulas - high Sharpe
    31001,  # MACDHistogramLong (Sharpe 2.36)
    31002,  # GoldenCrossLong (Sharpe 2.03)
    31003,  # Momentum3dLong (Sharpe 1.88)
    31004,  # Momentum10dLong (Sharpe 1.77)
    31005,  # MeanRevSMA7pctLong (Sharpe 1.16)
    31006,  # StreakDownReversalLong (Sharpe 0.87)
    31007,  # RSIOversoldLong (Sharpe 0.67)

    # SHORT formula - ONLY ONE THAT WORKS
    31050,  # ExtremeSpikeShort (Sharpe 1.78)
]


@dataclass
class FormulaConfig:
    """Configuration for a formula."""
    formula_id: int
    name: str
    direction: str  # 'LONG' or 'SHORT'
    sharpe: float
    win_rate: float
    kelly: float
    stop_loss: float
    take_profit: float
    hold_days: int


# Formula configurations from backtesting
FORMULA_CONFIGS = {
    31001: FormulaConfig(31001, "MACDHistogramLong", "LONG", 2.36, 0.553, 0.197, 0.10, 0.20, 3),
    31002: FormulaConfig(31002, "GoldenCrossLong", "LONG", 2.03, 0.556, 0.172, 0.10, 0.20, 3),
    31003: FormulaConfig(31003, "Momentum3dLong", "LONG", 1.88, 0.537, 0.157, 0.10, 0.20, 3),
    31004: FormulaConfig(31004, "Momentum10dLong", "LONG", 1.77, 0.546, 0.151, 0.10, 0.20, 5),
    31005: FormulaConfig(31005, "MeanRevSMA7pctLong", "LONG", 1.16, 0.569, 0.110, 0.10, 0.20, 3),
    31006: FormulaConfig(31006, "StreakDownReversalLong", "LONG", 0.87, 0.563, 0.088, 0.10, 0.15, 1),
    31007: FormulaConfig(31007, "RSIOversoldLong", "LONG", 0.67, 0.558, 0.069, 0.10, 0.15, 3),
    31050: FormulaConfig(31050, "ExtremeSpikeShort", "SHORT", 1.78, 0.551, 0.080, 0.05, 0.10, 1),
}


class MACDCalculator:
    """MACD indicator calculator."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.prices: List[float] = []

    def update(self, price: float):
        self.prices.append(price)
        if len(self.prices) > 100:
            self.prices = self.prices[-100:]

    def _ema(self, data: List[float], span: int) -> np.ndarray:
        """Calculate EMA."""
        if len(data) < span:
            return np.array([])
        alpha = 2 / (span + 1)
        result = np.zeros(len(data))
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

    def get_histogram(self) -> Optional[float]:
        """Get current MACD histogram value."""
        if len(self.prices) < self.slow + self.signal_period:
            return None

        prices = np.array(self.prices)
        ema_fast = self._ema(self.prices, self.fast)
        ema_slow = self._ema(self.prices, self.slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line.tolist(), self.signal_period)

        if len(signal_line) == 0:
            return None

        return macd_line[-1] - signal_line[-1]


class RSICalculator:
    """RSI indicator calculator."""

    def __init__(self, period: int = 7):
        self.period = period
        self.prices: List[float] = []

    def update(self, price: float):
        self.prices.append(price)
        if len(self.prices) > 50:
            self.prices = self.prices[-50:]

    def get_rsi(self) -> Optional[float]:
        """Get current RSI value."""
        if len(self.prices) < self.period + 1:
            return None

        changes = np.diff(self.prices[-self.period-1:])
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


class ProductionFormulaEngine:
    """
    Integrates production formulas (31001-31199).

    Features:
    - MACD Histogram (31001)
    - Golden Cross (31002)
    - Momentum (31003, 31004)
    - Mean Reversion (31005)
    - Streak Reversal (31006)
    - RSI Oversold (31007)
    - Extreme Spike Short (31050)

    Direction Bias: 85% LONG, 15% SHORT
    """

    def __init__(
        self,
        formula_ids: List[int] = None,
        kelly_fraction: float = 0.25,  # Quarter Kelly
    ):
        self.formula_ids = formula_ids or PRODUCTION_FORMULA_IDS
        self.kelly_fraction = kelly_fraction

        # Price history
        self.prices: List[float] = []
        self.returns: List[float] = []

        # Technical indicators
        self.macd = MACDCalculator()
        self.rsi7 = RSICalculator(7)

        # SMA tracking
        self.sma10: Optional[float] = None
        self.sma20: Optional[float] = None

        # Streak tracking
        self.consecutive_down = 0
        self.consecutive_up = 0

        # Last signals (avoid duplicates)
        self.last_signals: Dict[int, int] = {}  # formula_id -> last direction

    def update(self, price: float, data: Dict = None) -> List[FormulaSignal]:
        """
        Update all formulas with new price.

        Args:
            price: Current BTC price
            data: Additional data (OHLCV, blockchain features)

        Returns:
            List of signals from formulas that triggered
        """
        # Update price history
        if self.prices:
            ret = (price / self.prices[-1]) - 1
            self.returns.append(ret)

            # Track streaks
            if ret < 0:
                self.consecutive_down += 1
                self.consecutive_up = 0
            elif ret > 0:
                self.consecutive_up += 1
                self.consecutive_down = 0
            else:
                self.consecutive_down = 0
                self.consecutive_up = 0

        self.prices.append(price)

        # Keep history bounded
        if len(self.prices) > 100:
            self.prices = self.prices[-100:]
        if len(self.returns) > 100:
            self.returns = self.returns[-100:]

        # Update SMAs
        if len(self.prices) >= 10:
            self.sma10 = np.mean(self.prices[-10:])
        if len(self.prices) >= 20:
            self.sma20 = np.mean(self.prices[-20:])

        # Update indicators
        self.macd.update(price)
        self.rsi7.update(price)

        # Check each formula
        signals = []

        for formula_id in self.formula_ids:
            signal = self._check_formula(formula_id, price, data)
            if signal:
                signals.append(signal)

        return signals

    def _check_formula(
        self,
        formula_id: int,
        price: float,
        data: Dict = None
    ) -> Optional[FormulaSignal]:
        """Check if formula should signal."""
        config = FORMULA_CONFIGS.get(formula_id)
        if not config:
            return None

        direction = 0  # 0 = no signal

        # Check formula-specific conditions
        if formula_id == 31001:  # MACDHistogramLong
            direction = self._check_macd_histogram()

        elif formula_id == 31002:  # GoldenCrossLong
            direction = self._check_golden_cross()

        elif formula_id == 31003:  # Momentum3dLong
            direction = self._check_momentum(3, 0.02)

        elif formula_id == 31004:  # Momentum10dLong
            direction = self._check_momentum(10, 0.05)

        elif formula_id == 31005:  # MeanRevSMA7pctLong
            direction = self._check_mean_reversion(0.07)

        elif formula_id == 31006:  # StreakDownReversalLong
            direction = self._check_streak_reversal()

        elif formula_id == 31007:  # RSIOversoldLong
            direction = self._check_rsi_oversold()

        elif formula_id == 31050:  # ExtremeSpikeShort
            direction = self._check_extreme_spike()

        # Only signal if direction changed from last time
        if direction != 0:
            last_dir = self.last_signals.get(formula_id, 0)
            if direction == last_dir:
                return None  # Same signal, skip

            self.last_signals[formula_id] = direction

            # Calculate position size (Quarter Kelly)
            position_size = config.kelly * self.kelly_fraction

            return FormulaSignal(
                formula_id=formula_id,
                formula_name=config.name,
                direction=direction,
                confidence=config.win_rate,
                position_size_pct=position_size,
                stop_loss_pct=config.stop_loss,
                take_profit_pct=config.take_profit,
                max_hold_seconds=config.hold_days * 86400,
            )

        # Clear last signal if no longer active
        if formula_id in self.last_signals:
            del self.last_signals[formula_id]

        return None

    def _check_macd_histogram(self) -> int:
        """31001: LONG when MACD histogram > 0."""
        histogram = self.macd.get_histogram()
        if histogram is not None and histogram > 0:
            return 1  # LONG
        return 0

    def _check_golden_cross(self) -> int:
        """31002: LONG when SMA10 > SMA20."""
        if self.sma10 and self.sma20 and self.sma10 > self.sma20:
            return 1  # LONG
        return 0

    def _check_momentum(self, days: int, threshold: float) -> int:
        """31003/31004: LONG when N-day return > threshold."""
        if len(self.prices) < days + 1:
            return 0

        ret = (self.prices[-1] / self.prices[-days-1]) - 1
        if ret > threshold:
            return 1  # LONG
        return 0

    def _check_mean_reversion(self, threshold: float) -> int:
        """31005: LONG when price is X% below SMA20."""
        if not self.sma20:
            return 0

        deviation = (self.prices[-1] / self.sma20) - 1
        if deviation < -threshold:
            return 1  # LONG (buy the dip)
        return 0

    def _check_streak_reversal(self) -> int:
        """31006: LONG after 2+ consecutive down days."""
        if self.consecutive_down >= 2:
            return 1  # LONG (mean reversion)
        return 0

    def _check_rsi_oversold(self) -> int:
        """31007: LONG when RSI7 < 30."""
        rsi = self.rsi7.get_rsi()
        if rsi is not None and rsi < 30:
            return 1  # LONG
        return 0

    def _check_extreme_spike(self) -> int:
        """31050: SHORT after 7%+ daily gain."""
        if len(self.returns) < 1:
            return 0

        daily_return = self.returns[-1]
        if daily_return > 0.07:  # 7%+ spike
            return -1  # SHORT (fade the spike)
        return 0

    def get_ensemble_signal(self, price: float) -> Optional[FormulaSignal]:
        """
        Get combined ensemble signal.

        Weights formulas by Sharpe ratio.
        """
        signals = self.update(price)

        if not signals:
            return None

        # Weight by Sharpe
        total_weight = 0
        weighted_direction = 0

        for signal in signals:
            config = FORMULA_CONFIGS.get(signal.formula_id)
            if config:
                weight = config.sharpe
                total_weight += weight
                weighted_direction += signal.direction * weight

        if total_weight == 0:
            return None

        avg_direction = weighted_direction / total_weight

        # Determine ensemble direction
        if avg_direction > 0.3:
            direction = 1  # LONG
        elif avg_direction < -0.3:
            direction = -1  # SHORT
        else:
            return None  # No clear signal

        # Use best signal's parameters
        best_signal = max(signals, key=lambda s: FORMULA_CONFIGS[s.formula_id].sharpe)

        return FormulaSignal(
            formula_id=31199,  # Ensemble ID
            formula_name="ProductionEnsemble",
            direction=direction,
            confidence=best_signal.confidence,
            position_size_pct=best_signal.position_size_pct,
            stop_loss_pct=best_signal.stop_loss_pct,
            take_profit_pct=best_signal.take_profit_pct,
            max_hold_seconds=best_signal.max_hold_seconds,
        )

    def reset(self):
        """Reset all state."""
        self.prices = []
        self.returns = []
        self.macd = MACDCalculator()
        self.rsi7 = RSICalculator(7)
        self.sma10 = None
        self.sma20 = None
        self.consecutive_down = 0
        self.consecutive_up = 0
        self.last_signals = {}
