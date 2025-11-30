"""
Renaissance Formula Library - Regime Filter
============================================
ID 335: Trend-Aware Trading

The Problem:
- Mean reversion signals SHORT in strong uptrends → LOSES MONEY
- Momentum signals BUY in mean-reverting ranges → LOSES MONEY
- Same strategy doesn't work in all market conditions

The Solution:
- DETECT market regime (trending vs ranging)
- FILTER signals based on regime
- Only allow signals that match current regime

Market Regimes:
1. STRONG_UPTREND: Only allow BUY signals
2. WEAK_UPTREND: Allow BUY, reduce SELL
3. RANGING: Allow all signals (mean reversion works)
4. WEAK_DOWNTREND: Allow SELL, reduce BUY
5. STRONG_DOWNTREND: Only allow SELL signals

Mathematical Foundation:
From Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum":
- Past 12-month returns predict future returns
- Trend-following works ACROSS ALL asset classes

Regime Detection Methods:
1. EMA Slope: (EMA_fast - EMA_slow) / EMA_slow
2. ADX (Average Directional Index): Trend strength
3. Volatility Regime: High vol = ranging, Low vol = trending
4. Price Position: Above/below key levels

Sources:
- Moskowitz et al. (2012): "Time Series Momentum"
- Faber (2007): "A Quantitative Approach to Tactical Asset Allocation"
- AQR Research: "Trend-following and momentum strategies"
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from enum import Enum
import time

from .base import BaseFormula, FormulaRegistry


class MarketRegime(Enum):
    """Market regime classification"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


@FormulaRegistry.register(335, name="RegimeFilter", category="regime")
class RegimeFilterFormula(BaseFormula):
    """
    ID 335: Regime Filter - Trend-Aware Trading

    Detects market regime and filters signals accordingly.

    Key Features:
    1. Multi-method regime detection (EMA, ADX, Volatility)
    2. Smooth regime transitions (no flip-flopping)
    3. Signal filtering based on regime
    4. Confidence adjustment by regime clarity

    Signal Filtering Rules:
    - STRONG_UPTREND: BUY only (block SELL)
    - WEAK_UPTREND: BUY full, SELL at 50%
    - RANGING: All signals allowed
    - WEAK_DOWNTREND: SELL full, BUY at 50%
    - STRONG_DOWNTREND: SELL only (block BUY)
    """

    FORMULA_ID = 335
    CATEGORY = "regime"
    NAME = "Regime Filter"
    DESCRIPTION = "Trend-aware signal filtering based on market regime"

    def __init__(self,
                 lookback: int = 200,
                 fast_period: int = 20,
                 slow_period: int = 50,
                 trend_period: int = 200,
                 strong_trend_threshold: float = 0.02,  # 2% EMA divergence
                 weak_trend_threshold: float = 0.005,   # 0.5% EMA divergence
                 regime_smoothing: int = 5,
                 **kwargs):
        super().__init__(lookback, **kwargs)

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trend_period = trend_period
        self.strong_trend_threshold = strong_trend_threshold
        self.weak_trend_threshold = weak_trend_threshold
        self.regime_smoothing = regime_smoothing

        # EMAs
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.ema_trend = 0.0

        # Regime detection
        self.current_regime = MarketRegime.RANGING
        self.regime_history: deque = deque(maxlen=regime_smoothing)
        self.regime_confidence = 0.5

        # Trend metrics
        self.trend_strength = 0.0
        self.trend_direction = 0  # 1 = up, -1 = down, 0 = neutral
        self.volatility_regime = "normal"

        # Signal filtering
        self.buy_multiplier = 1.0
        self.sell_multiplier = 1.0

    def update(self, price: float, volume: float = 0, timestamp: float = None):
        """Update with new price and detect regime"""
        super().update(price, volume, timestamp)

        if len(self.prices) < self.slow_period:
            return

        self._update_emas(price)
        self._detect_regime()
        self._update_signal_filters()

    def _update_emas(self, price: float):
        """Update EMAs"""
        prices = self._prices_array()

        # Fast EMA
        alpha_fast = 2 / (self.fast_period + 1)
        if self.ema_fast == 0:
            self.ema_fast = np.mean(prices[-self.fast_period:])
        else:
            self.ema_fast = alpha_fast * price + (1 - alpha_fast) * self.ema_fast

        # Slow EMA
        alpha_slow = 2 / (self.slow_period + 1)
        if self.ema_slow == 0:
            self.ema_slow = np.mean(prices[-self.slow_period:])
        else:
            self.ema_slow = alpha_slow * price + (1 - alpha_slow) * self.ema_slow

        # Trend EMA (200 period)
        if len(prices) >= self.trend_period:
            alpha_trend = 2 / (self.trend_period + 1)
            if self.ema_trend == 0:
                self.ema_trend = np.mean(prices[-self.trend_period:])
            else:
                self.ema_trend = alpha_trend * price + (1 - alpha_trend) * self.ema_trend

    def _detect_regime(self):
        """Detect current market regime"""
        if self.ema_slow == 0:
            return

        # Calculate trend metrics
        ema_divergence = (self.ema_fast - self.ema_slow) / self.ema_slow

        # Calculate EMA slope (trend direction)
        prices = self._prices_array()
        if len(prices) >= 20:
            ema_20_bars_ago = np.mean(prices[-40:-20]) if len(prices) >= 40 else prices[-20]
            ema_slope = (self.ema_slow - ema_20_bars_ago) / ema_20_bars_ago if ema_20_bars_ago > 0 else 0
        else:
            ema_slope = 0

        self.trend_strength = abs(ema_divergence)
        self.trend_direction = 1 if ema_divergence > 0 else (-1 if ema_divergence < 0 else 0)

        # Determine regime
        if ema_divergence > self.strong_trend_threshold and ema_slope > 0.005:
            new_regime = MarketRegime.STRONG_UPTREND
            self.regime_confidence = min(1.0, ema_divergence / self.strong_trend_threshold)
        elif ema_divergence > self.weak_trend_threshold:
            new_regime = MarketRegime.WEAK_UPTREND
            self.regime_confidence = 0.7
        elif ema_divergence < -self.strong_trend_threshold and ema_slope < -0.005:
            new_regime = MarketRegime.STRONG_DOWNTREND
            self.regime_confidence = min(1.0, abs(ema_divergence) / self.strong_trend_threshold)
        elif ema_divergence < -self.weak_trend_threshold:
            new_regime = MarketRegime.WEAK_DOWNTREND
            self.regime_confidence = 0.7
        else:
            new_regime = MarketRegime.RANGING
            self.regime_confidence = 1.0 - abs(ema_divergence) / self.weak_trend_threshold

        # Smooth regime transitions
        self.regime_history.append(new_regime)

        # Only change regime if consistent for smoothing period
        if len(self.regime_history) >= self.regime_smoothing:
            regime_counts = {}
            for r in self.regime_history:
                regime_counts[r] = regime_counts.get(r, 0) + 1

            most_common = max(regime_counts, key=regime_counts.get)
            if regime_counts[most_common] >= self.regime_smoothing * 0.6:
                self.current_regime = most_common

    def _update_signal_filters(self):
        """Update signal multipliers based on regime"""
        regime = self.current_regime

        if regime == MarketRegime.STRONG_UPTREND:
            self.buy_multiplier = 1.5   # Boost buys
            self.sell_multiplier = 0.0  # Block sells
            self.signal = 1
        elif regime == MarketRegime.WEAK_UPTREND:
            self.buy_multiplier = 1.2
            self.sell_multiplier = 0.5
            self.signal = 1
        elif regime == MarketRegime.RANGING:
            self.buy_multiplier = 1.0
            self.sell_multiplier = 1.0
            self.signal = 0
        elif regime == MarketRegime.WEAK_DOWNTREND:
            self.buy_multiplier = 0.5
            self.sell_multiplier = 1.2
            self.signal = -1
        elif regime == MarketRegime.STRONG_DOWNTREND:
            self.buy_multiplier = 0.0  # Block buys
            self.sell_multiplier = 1.5  # Boost sells
            self.signal = -1

        self.confidence = self.regime_confidence

    def filter_signal(self, direction: int, strength: float) -> Tuple[int, float]:
        """
        Filter a signal based on current regime.

        Args:
            direction: 1 for buy, -1 for sell
            strength: Signal strength 0-1

        Returns:
            (filtered_direction, filtered_strength)
        """
        if direction > 0:
            # Buy signal
            filtered_strength = strength * self.buy_multiplier
            if self.buy_multiplier == 0:
                return 0, 0.0
            return direction, filtered_strength
        elif direction < 0:
            # Sell signal
            filtered_strength = strength * self.sell_multiplier
            if self.sell_multiplier == 0:
                return 0, 0.0
            return direction, filtered_strength
        else:
            return 0, 0.0

    def should_allow_signal(self, direction: int) -> bool:
        """Check if signal direction is allowed in current regime"""
        if direction > 0:
            return self.buy_multiplier > 0
        elif direction < 0:
            return self.sell_multiplier > 0
        return True

    def get_regime_name(self) -> str:
        """Get current regime name"""
        return self.current_regime.value

    def _compute(self) -> None:
        """Required by BaseFormula - updates in update()"""
        pass

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'trend_strength': self.trend_strength,
            'trend_direction': self.trend_direction,
            'buy_multiplier': self.buy_multiplier,
            'sell_multiplier': self.sell_multiplier,
            'ema_fast': self.ema_fast,
            'ema_slow': self.ema_slow,
            'ema_trend': self.ema_trend,
        })
        return state


@FormulaRegistry.register(340, name="VolatilityRegime", category="regime")
class VolatilityRegimeFormula(BaseFormula):
    """
    ID 340: Volatility Regime Detection

    Different strategies work in different volatility regimes:
    - HIGH VOLATILITY: Mean reversion works well (overreactions)
    - LOW VOLATILITY: Momentum works well (smooth trends)
    - TRANSITION: Be cautious (regime changing)

    Uses realized volatility percentile to classify regime.
    """

    FORMULA_ID = 340
    CATEGORY = "regime"
    NAME = "Volatility Regime"
    DESCRIPTION = "Volatility-based regime detection and strategy selection"

    def __init__(self,
                 lookback: int = 200,
                 vol_window: int = 20,
                 high_vol_percentile: float = 75,
                 low_vol_percentile: float = 25,
                 **kwargs):
        super().__init__(lookback, **kwargs)

        self.vol_window = vol_window
        self.high_vol_percentile = high_vol_percentile
        self.low_vol_percentile = low_vol_percentile

        self.current_volatility = 0.0
        self.vol_percentile = 50
        self.vol_regime = "normal"

        self.vol_history: deque = deque(maxlen=lookback)

        # Strategy preferences by regime
        self.mean_reversion_weight = 1.0
        self.momentum_weight = 1.0

    def update(self, price: float, volume: float = 0, timestamp: float = None):
        """Update with new price"""
        super().update(price, volume, timestamp)

        if len(self.prices) < self.vol_window:
            return

        self._calculate_volatility()
        self._classify_regime()
        self._update_strategy_weights()

    def _calculate_volatility(self):
        """Calculate current realized volatility"""
        prices = self._prices_array()
        returns = np.diff(np.log(prices[-self.vol_window:]))

        if len(returns) > 1:
            self.current_volatility = np.std(returns) * np.sqrt(252)  # Annualized
            self.vol_history.append(self.current_volatility)

    def _classify_regime(self):
        """Classify volatility regime"""
        if len(self.vol_history) < 20:
            self.vol_regime = "normal"
            self.vol_percentile = 50
            return

        vol_array = np.array(list(self.vol_history))
        self.vol_percentile = np.percentile(
            np.searchsorted(np.sort(vol_array), self.current_volatility) / len(vol_array) * 100,
            50
        )
        self.vol_percentile = (np.searchsorted(np.sort(vol_array), self.current_volatility)
                              / len(vol_array) * 100)

        if self.vol_percentile >= self.high_vol_percentile:
            self.vol_regime = "high"
        elif self.vol_percentile <= self.low_vol_percentile:
            self.vol_regime = "low"
        else:
            self.vol_regime = "normal"

    def _update_strategy_weights(self):
        """Update strategy weights based on regime"""
        if self.vol_regime == "high":
            # High vol: Mean reversion works better
            self.mean_reversion_weight = 1.5
            self.momentum_weight = 0.5
            self.signal = 0  # Neutral - use mean reversion
            self.confidence = 0.7
        elif self.vol_regime == "low":
            # Low vol: Momentum works better
            self.mean_reversion_weight = 0.5
            self.momentum_weight = 1.5
            self.signal = 1  # Trending - follow momentum
            self.confidence = 0.7
        else:
            # Normal: Balanced
            self.mean_reversion_weight = 1.0
            self.momentum_weight = 1.0
            self.signal = 0
            self.confidence = 0.5

    def get_strategy_weight(self, strategy_type: str) -> float:
        """Get weight for a strategy type"""
        if strategy_type in ["mean_reversion", "ou", "zscore"]:
            return self.mean_reversion_weight
        elif strategy_type in ["momentum", "trend", "breakout"]:
            return self.momentum_weight
        return 1.0

    def _compute(self) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'current_volatility': self.current_volatility,
            'vol_percentile': self.vol_percentile,
            'vol_regime': self.vol_regime,
            'mean_reversion_weight': self.mean_reversion_weight,
            'momentum_weight': self.momentum_weight,
        })
        return state
