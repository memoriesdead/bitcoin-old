"""
Mean Reversion Formulas (IDs 131-150)
=====================================
Ornstein-Uhlenbeck, Z-Score, Cointegration, and pairs trading signals.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from .base import BaseFormula, FormulaRegistry


# =============================================================================
# ORNSTEIN-UHLENBECK PROCESSES (131-140)
# =============================================================================

@FormulaRegistry.register(131)
class OrnsteinUhlenbeck(BaseFormula):
    """ID 131: Ornstein-Uhlenbeck Process"""

    CATEGORY = "mean_reversion"
    NAME = "OrnsteinUhlenbeck"
    DESCRIPTION = "dX = θ(μ - X)dt + σdW"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.theta = 0.1
        self.mu = 0.0
        self.sigma = 0.01
        self.half_life = 0.0

    def _compute(self) -> None:
        if len(self.prices) < 30:
            return
        prices = self._prices_array()
        log_prices = np.log(prices)
        y = log_prices[1:]
        x = log_prices[:-1]
        n = len(x)
        if n < 10:
            return
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return
        a = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - a * sum_x) / n
        if a <= 0 or a >= 1:
            a = 0.99
        self.theta = -np.log(a)
        self.mu = b / (1 - a)
        residuals = y - a * x - b
        self.sigma = np.std(residuals) * np.sqrt(2 * self.theta)
        self.half_life = np.log(2) / self.theta if self.theta > 0 else 100
        current_log = log_prices[-1]
        deviation = current_log - self.mu
        z_score = deviation / (self.sigma / np.sqrt(2 * self.theta) + 1e-10)
        if z_score < -2:
            self.signal = 1
            self.confidence = min(abs(z_score) / 4, 1.0)
        elif z_score > 2:
            self.signal = -1
            self.confidence = min(abs(z_score) / 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(132)
class OUHalfLife(BaseFormula):
    """ID 132: OU Half-Life for mean reversion timing"""

    CATEGORY = "mean_reversion"
    NAME = "OUHalfLife"
    DESCRIPTION = "half_life = ln(2) / θ"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.half_life = 50.0
        self.theta = 0.01

    def _compute(self) -> None:
        if len(self.prices) < 30:
            return
        prices = self._prices_array()
        log_prices = np.log(prices)
        y = log_prices[1:]
        x = log_prices[:-1]
        n = len(x)
        if n < 10:
            return
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov_xy = np.sum((x - x_mean) * (y - y_mean))
        var_x = np.sum((x - x_mean) ** 2)
        if var_x < 1e-10:
            return
        a = cov_xy / var_x
        if a <= 0:
            a = 0.01
        elif a >= 1:
            a = 0.99
        self.theta = -np.log(a)
        self.half_life = np.log(2) / self.theta if self.theta > 0 else 100
        if self.half_life < 10:
            self.signal = 1
            self.confidence = min(10 / self.half_life / 5, 1.0)
        elif self.half_life > 50:
            self.signal = -1
            self.confidence = min(self.half_life / 100, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.5


@FormulaRegistry.register(133)
class OUMeanLevel(BaseFormula):
    """ID 133: OU Mean Level Estimation"""

    CATEGORY = "mean_reversion"
    NAME = "OUMeanLevel"
    DESCRIPTION = "Estimate long-term mean μ"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.mean_level = 0.0
        self.mean_history = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.prices) < 30:
            return
        prices = self._prices_array()
        log_prices = np.log(prices)
        y = log_prices[1:]
        x = log_prices[:-1]
        n = len(x)
        if n < 10:
            return
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov_xy = np.sum((x - x_mean) * (y - y_mean))
        var_x = np.sum((x - x_mean) ** 2)
        if var_x < 1e-10:
            return
        a = cov_xy / var_x
        b = y_mean - a * x_mean
        if abs(1 - a) < 1e-10:
            self.mean_level = np.mean(log_prices)
        else:
            self.mean_level = b / (1 - a)
        self.mean_history.append(self.mean_level)
        current_log = log_prices[-1]
        deviation = current_log - self.mean_level
        std_price = np.std(log_prices)
        z = deviation / (std_price + 1e-10)
        if z < -1.5:
            self.signal = 1
            self.confidence = min(abs(z) / 3, 1.0)
        elif z > 1.5:
            self.signal = -1
            self.confidence = min(abs(z) / 3, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(134)
class OUVolatility(BaseFormula):
    """ID 134: OU Volatility Parameter"""

    CATEGORY = "mean_reversion"
    NAME = "OUVolatility"
    DESCRIPTION = "Estimate σ in OU process"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.ou_sigma = 0.01

    def _compute(self) -> None:
        if len(self.prices) < 30:
            return
        prices = self._prices_array()
        log_prices = np.log(prices)
        y = log_prices[1:]
        x = log_prices[:-1]
        n = len(x)
        if n < 10:
            return
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov_xy = np.sum((x - x_mean) * (y - y_mean))
        var_x = np.sum((x - x_mean) ** 2)
        if var_x < 1e-10:
            return
        a = cov_xy / var_x
        b = y_mean - a * x_mean
        residuals = y - a * x - b
        residual_std = np.std(residuals)
        theta = -np.log(max(a, 0.01))
        self.ou_sigma = residual_std * np.sqrt(2 * theta)
        historical_vol = np.std(np.diff(log_prices))
        if self.ou_sigma > historical_vol * 1.5:
            self.signal = -1
            self.confidence = min(self.ou_sigma / historical_vol / 3, 1.0)
        elif self.ou_sigma < historical_vol * 0.7:
            self.signal = 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(135)
class OUSpeedOfReversion(BaseFormula):
    """ID 135: OU Speed of Reversion"""

    CATEGORY = "mean_reversion"
    NAME = "OUSpeedOfReversion"
    DESCRIPTION = "θ - rate of mean reversion"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.theta = 0.01
        self.theta_history = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.prices) < 30:
            return
        prices = self._prices_array()
        log_prices = np.log(prices)
        y = log_prices[1:]
        x = log_prices[:-1]
        x_mean = np.mean(x)
        cov_xy = np.sum((x - x_mean) * (y - np.mean(y)))
        var_x = np.sum((x - x_mean) ** 2)
        if var_x < 1e-10:
            return
        a = cov_xy / var_x
        if a <= 0:
            a = 0.01
        elif a >= 1:
            a = 0.99
        self.theta = -np.log(a)
        self.theta_history.append(self.theta)
        if len(self.theta_history) < 10:
            return
        avg_theta = np.mean(self.theta_history)
        if self.theta > avg_theta * 1.5:
            self.signal = 1
            self.confidence = min(self.theta / avg_theta / 3, 1.0)
        elif self.theta < avg_theta * 0.5:
            self.signal = -1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(136)
class OUEquilibrium(BaseFormula):
    """ID 136: OU Equilibrium Detection"""

    CATEGORY = "mean_reversion"
    NAME = "OUEquilibrium"
    DESCRIPTION = "Detect when price at equilibrium"

    def __init__(self, lookback: int = 100, equilibrium_threshold: float = 0.5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.equilibrium_threshold = equilibrium_threshold
        self.at_equilibrium = False

    def _compute(self) -> None:
        if len(self.prices) < 30:
            return
        prices = self._prices_array()
        log_prices = np.log(prices)
        y = log_prices[1:]
        x = log_prices[:-1]
        x_mean = np.mean(x)
        cov_xy = np.sum((x - x_mean) * (y - np.mean(y)))
        var_x = np.sum((x - x_mean) ** 2)
        if var_x < 1e-10:
            return
        a = cov_xy / var_x
        b = np.mean(y) - a * x_mean
        if abs(1 - a) < 1e-10:
            mu = np.mean(log_prices)
        else:
            mu = b / (1 - a)
        current = log_prices[-1]
        deviation = abs(current - mu)
        std = np.std(log_prices)
        z = deviation / (std + 1e-10)
        self.at_equilibrium = z < self.equilibrium_threshold
        if self.at_equilibrium:
            self.signal = 0
            self.confidence = 1 - z / self.equilibrium_threshold
        else:
            self.signal = 1 if current < mu else -1
            self.confidence = min(z / 3, 1.0)


@FormulaRegistry.register(137)
class OUOptimalEntry(BaseFormula):
    """ID 137: OU Optimal Entry Point"""

    CATEGORY = "mean_reversion"
    NAME = "OUOptimalEntry"
    DESCRIPTION = "Calculate optimal entry based on OU dynamics"

    def __init__(self, lookback: int = 100, entry_z: float = 2.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.entry_z = entry_z
        self.optimal_entry = False

    def _compute(self) -> None:
        if len(self.prices) < 30:
            return
        prices = self._prices_array()
        log_prices = np.log(prices)
        mu = np.mean(log_prices)
        std = np.std(log_prices)
        current = log_prices[-1]
        z = (current - mu) / (std + 1e-10)
        y = log_prices[1:]
        x = log_prices[:-1]
        x_mean = np.mean(x)
        cov_xy = np.sum((x - x_mean) * (y - np.mean(y)))
        var_x = np.sum((x - x_mean) ** 2)
        if var_x < 1e-10:
            return
        a = cov_xy / var_x
        if a <= 0 or a >= 1:
            a = 0.9
        theta = -np.log(a)
        half_life = np.log(2) / theta
        self.optimal_entry = abs(z) > self.entry_z and half_life < 50
        if self.optimal_entry:
            self.signal = 1 if z < 0 else -1
            self.confidence = min(abs(z) / 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(138)
class OUOptimalExit(BaseFormula):
    """ID 138: OU Optimal Exit Point"""

    CATEGORY = "mean_reversion"
    NAME = "OUOptimalExit"
    DESCRIPTION = "Calculate optimal exit based on OU dynamics"

    def __init__(self, lookback: int = 100, exit_z: float = 0.5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.exit_z = exit_z
        self.should_exit = False

    def _compute(self) -> None:
        if len(self.prices) < 30:
            return
        prices = self._prices_array()
        log_prices = np.log(prices)
        mu = np.mean(log_prices)
        std = np.std(log_prices)
        current = log_prices[-1]
        z = (current - mu) / (std + 1e-10)
        self.should_exit = abs(z) < self.exit_z
        if self.should_exit:
            self.signal = 0
            self.confidence = 1 - abs(z) / self.exit_z
        else:
            self.signal = 1 if z < 0 else -1
            self.confidence = min(abs(z) / 3, 1.0)


@FormulaRegistry.register(139)
class VasicekModel(BaseFormula):
    """ID 139: Vasicek Interest Rate Model (OU variant)"""

    CATEGORY = "mean_reversion"
    NAME = "VasicekModel"
    DESCRIPTION = "dr = a(b - r)dt + σdW"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.a = 0.1
        self.b = 0.0
        self.sigma = 0.01

    def _compute(self) -> None:
        if len(self.returns) < 30:
            return
        returns = self._returns_array()
        y = returns[1:]
        x = returns[:-1]
        n = len(x)
        if n < 10:
            return
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov = np.sum((x - x_mean) * (y - y_mean))
        var = np.sum((x - x_mean) ** 2)
        if var < 1e-10:
            return
        phi = cov / var
        c = y_mean - phi * x_mean
        self.a = 1 - phi
        self.b = c / self.a if self.a > 0.01 else 0
        residuals = y - phi * x - c
        self.sigma = np.std(residuals)
        current = returns[-1]
        z = (current - self.b) / (self.sigma + 1e-10)
        if z < -2:
            self.signal = 1
            self.confidence = min(abs(z) / 4, 1.0)
        elif z > 2:
            self.signal = -1
            self.confidence = min(abs(z) / 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(140)
class CIRModel(BaseFormula):
    """ID 140: Cox-Ingersoll-Ross Model"""

    CATEGORY = "mean_reversion"
    NAME = "CIRModel"
    DESCRIPTION = "dr = a(b - r)dt + σ√r dW"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.a = 0.1
        self.b = 0.01
        self.sigma = 0.05

    def _compute(self) -> None:
        if len(self.prices) < 30:
            return
        prices = self._prices_array()
        volatility = np.abs(np.diff(np.log(prices)))
        y = volatility[1:]
        x = volatility[:-1]
        n = len(x)
        if n < 10:
            return
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov = np.sum((x - x_mean) * (y - y_mean))
        var = np.sum((x - x_mean) ** 2)
        if var < 1e-10:
            return
        phi = cov / var
        c = y_mean - phi * x_mean
        self.a = 1 - phi
        self.b = c / self.a if self.a > 0.01 else np.mean(volatility)
        current_vol = volatility[-1]
        vol_z = (current_vol - self.b) / (np.std(volatility) + 1e-10)
        if vol_z > 2:
            self.signal = -1
            self.confidence = min(vol_z / 4, 1.0)
        elif vol_z < -1:
            self.signal = 1
            self.confidence = min(abs(vol_z) / 3, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


# =============================================================================
# Z-SCORE AND COINTEGRATION (141-150)
# =============================================================================

@FormulaRegistry.register(141)
class ZScoreSignal(BaseFormula):
    """ID 141: Z-Score Mean Reversion Signal"""

    CATEGORY = "mean_reversion"
    NAME = "ZScoreSignal"
    DESCRIPTION = "z = (x - μ) / σ"

    def __init__(self, lookback: int = 100, entry_z: float = 2.0, exit_z: float = 0.5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.z_score = 0.0

    def _compute(self) -> None:
        if len(self.prices) < 20:
            return
        prices = self._prices_array()
        mean = np.mean(prices)
        std = np.std(prices)
        current = prices[-1]
        self.z_score = (current - mean) / (std + 1e-10)
        if self.z_score < -self.entry_z:
            self.signal = 1
            self.confidence = min(abs(self.z_score) / 4, 1.0)
        elif self.z_score > self.entry_z:
            self.signal = -1
            self.confidence = min(abs(self.z_score) / 4, 1.0)
        elif abs(self.z_score) < self.exit_z:
            self.signal = 0
            self.confidence = 1 - abs(self.z_score) / self.exit_z
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(142)
class AdaptiveZScore(BaseFormula):
    """ID 142: Adaptive Z-Score with dynamic thresholds"""

    CATEGORY = "mean_reversion"
    NAME = "AdaptiveZScore"
    DESCRIPTION = "Z-score with regime-adjusted thresholds"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.adaptive_entry = 2.0
        self.adaptive_exit = 0.5

    def _compute(self) -> None:
        if len(self.prices) < 30:
            return
        prices = self._prices_array()
        mean = np.mean(prices)
        std = np.std(prices)
        current = prices[-1]
        z = (current - mean) / (std + 1e-10)
        recent_range = np.max(prices[-20:]) - np.min(prices[-20:])
        historical_range = np.max(prices) - np.min(prices)
        range_ratio = recent_range / (historical_range + 1e-10)
        self.adaptive_entry = 2.0 * range_ratio + 1.5 * (1 - range_ratio)
        self.adaptive_exit = 0.5 * range_ratio + 0.3 * (1 - range_ratio)
        if z < -self.adaptive_entry:
            self.signal = 1
            self.confidence = min(abs(z) / 4, 1.0)
        elif z > self.adaptive_entry:
            self.signal = -1
            self.confidence = min(abs(z) / 4, 1.0)
        elif abs(z) < self.adaptive_exit:
            self.signal = 0
            self.confidence = 0.7
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(143)
class BollingerBandReversion(BaseFormula):
    """ID 143: Bollinger Band Mean Reversion"""

    CATEGORY = "mean_reversion"
    NAME = "BollingerBandReversion"
    DESCRIPTION = "Reversion from Bollinger Band extremes"

    def __init__(self, lookback: int = 100, window: int = 20, num_std: float = 2.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.window = window
        self.num_std = num_std
        self.percent_b = 0.5

    def _compute(self) -> None:
        if len(self.prices) < self.window:
            return
        prices = self._prices_array()
        ma = np.mean(prices[-self.window:])
        std = np.std(prices[-self.window:])
        upper = ma + self.num_std * std
        lower = ma - self.num_std * std
        current = prices[-1]
        band_width = upper - lower
        if band_width > 0:
            self.percent_b = (current - lower) / band_width
        else:
            self.percent_b = 0.5
        if self.percent_b < 0.1:
            self.signal = 1
            self.confidence = min((0.5 - self.percent_b) * 2, 1.0)
        elif self.percent_b > 0.9:
            self.signal = -1
            self.confidence = min((self.percent_b - 0.5) * 2, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(144)
class KeltnerChannelReversion(BaseFormula):
    """ID 144: Keltner Channel Mean Reversion"""

    CATEGORY = "mean_reversion"
    NAME = "KeltnerChannelReversion"
    DESCRIPTION = "Reversion from ATR-based channel"

    def __init__(self, lookback: int = 100, window: int = 20, atr_mult: float = 2.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.window = window
        self.atr_mult = atr_mult

    def _compute(self) -> None:
        if len(self.prices) < self.window + 1:
            return
        prices = self._prices_array()
        ema = self._ema(prices, self.window)[-1]
        tr_values = []
        for i in range(1, len(prices)):
            high_low = abs(prices[i] - prices[i-1])
            tr_values.append(high_low)
        atr = np.mean(tr_values[-self.window:]) if len(tr_values) >= self.window else np.mean(tr_values)
        upper = ema + self.atr_mult * atr
        lower = ema - self.atr_mult * atr
        current = prices[-1]
        if current < lower:
            self.signal = 1
            distance = (lower - current) / (atr + 1e-10)
            self.confidence = min(distance / 2, 1.0)
        elif current > upper:
            self.signal = -1
            distance = (current - upper) / (atr + 1e-10)
            self.confidence = min(distance / 2, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(145)
class RSIMeanReversion(BaseFormula):
    """ID 145: RSI Mean Reversion"""

    CATEGORY = "mean_reversion"
    NAME = "RSIMeanReversion"
    DESCRIPTION = "Mean reversion from RSI extremes"

    def __init__(self, lookback: int = 100, period: int = 14,
                 oversold: float = 30, overbought: float = 70, **kwargs):
        super().__init__(lookback, **kwargs)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.rsi = 50.0

    def _compute(self) -> None:
        if len(self.prices) < self.period + 1:
            return
        prices = self._prices_array()
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-self.period:])
        avg_loss = np.mean(losses[-self.period:])
        if avg_loss == 0:
            self.rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            self.rsi = 100 - (100 / (1 + rs))
        if self.rsi < self.oversold:
            self.signal = 1
            self.confidence = min((self.oversold - self.rsi) / 30, 1.0)
        elif self.rsi > self.overbought:
            self.signal = -1
            self.confidence = min((self.rsi - self.overbought) / 30, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(146)
class StochasticReversion(BaseFormula):
    """ID 146: Stochastic Oscillator Reversion"""

    CATEGORY = "mean_reversion"
    NAME = "StochasticReversion"
    DESCRIPTION = "Mean reversion from stochastic extremes"

    def __init__(self, lookback: int = 100, k_period: int = 14, d_period: int = 3, **kwargs):
        super().__init__(lookback, **kwargs)
        self.k_period = k_period
        self.d_period = d_period
        self.k_value = 50.0
        self.d_value = 50.0

    def _compute(self) -> None:
        if len(self.prices) < self.k_period:
            return
        prices = self._prices_array()
        high = np.max(prices[-self.k_period:])
        low = np.min(prices[-self.k_period:])
        current = prices[-1]
        if high - low > 0:
            self.k_value = 100 * (current - low) / (high - low)
        else:
            self.k_value = 50.0
        if len(self.prices) >= self.k_period + self.d_period:
            k_values = []
            for i in range(self.d_period):
                idx = len(prices) - 1 - i
                h = np.max(prices[max(0,idx-self.k_period+1):idx+1])
                l = np.min(prices[max(0,idx-self.k_period+1):idx+1])
                k = 100 * (prices[idx] - l) / (h - l + 1e-10)
                k_values.append(k)
            self.d_value = np.mean(k_values)
        if self.k_value < 20 and self.d_value < 20:
            self.signal = 1
            self.confidence = min((20 - self.k_value) / 20, 1.0)
        elif self.k_value > 80 and self.d_value > 80:
            self.signal = -1
            self.confidence = min((self.k_value - 80) / 20, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(147)
class EngleGrangerCoint(BaseFormula):
    """ID 147: Engle-Granger Cointegration Test"""

    CATEGORY = "mean_reversion"
    NAME = "EngleGrangerCoint"
    DESCRIPTION = "Test for cointegration (simplified)"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.is_cointegrated = False
        self.spread = 0.0
        self.hedge_ratio = 1.0

    def _compute(self) -> None:
        if len(self.prices) < 50:
            return
        prices = self._prices_array()
        log_prices = np.log(prices)
        y = log_prices[:-1]
        x = np.arange(len(y))
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov = np.sum((x - x_mean) * (y - y_mean))
        var = np.sum((x - x_mean) ** 2)
        if var < 1e-10:
            return
        beta = cov / var
        alpha = y_mean - beta * x_mean
        residuals = y - (alpha + beta * x)
        resid_diff = np.diff(residuals)
        resid_lag = residuals[:-1]
        if len(resid_lag) < 10:
            return
        r_mean = np.mean(resid_lag)
        d_mean = np.mean(resid_diff)
        cov_rd = np.sum((resid_lag - r_mean) * (resid_diff - d_mean))
        var_r = np.sum((resid_lag - r_mean) ** 2)
        if var_r < 1e-10:
            return
        gamma = cov_rd / var_r
        adf_stat = gamma / (np.std(resid_diff) / np.sqrt(var_r) + 1e-10)
        self.is_cointegrated = adf_stat < -2.86
        self.spread = residuals[-1] if len(residuals) > 0 else 0
        spread_std = np.std(residuals)
        z = self.spread / (spread_std + 1e-10)
        if self.is_cointegrated and z < -2:
            self.signal = 1
            self.confidence = min(abs(z) / 4, 1.0)
        elif self.is_cointegrated and z > 2:
            self.signal = -1
            self.confidence = min(abs(z) / 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(148)
class JohansenCoint(BaseFormula):
    """ID 148: Johansen Cointegration (simplified)"""

    CATEGORY = "mean_reversion"
    NAME = "JohansenCoint"
    DESCRIPTION = "Vector cointegration test"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.rank = 0
        self.eigenvalue = 0.0

    def _compute(self) -> None:
        if len(self.prices) < 50 or len(self.volumes) < 50:
            return
        prices = self._prices_array()
        volumes = self._volumes_array()
        log_p = np.log(prices)
        log_v = np.log(volumes + 1)
        dp = np.diff(log_p)
        dv = np.diff(log_v)
        p_lag = log_p[:-1]
        v_lag = log_v[:-1]
        X = np.column_stack([p_lag, v_lag])
        Y = np.column_stack([dp, dv])
        if len(X) < 10:
            return
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        X_c = X - X_mean
        Y_c = Y - Y_mean
        try:
            cov_xx = np.dot(X_c.T, X_c) / len(X)
            cov_yy = np.dot(Y_c.T, Y_c) / len(Y)
            cov_xy = np.dot(X_c.T, Y_c) / len(X)
            inv_xx = np.linalg.inv(cov_xx + np.eye(2) * 1e-6)
            inv_yy = np.linalg.inv(cov_yy + np.eye(2) * 1e-6)
            M = np.dot(np.dot(np.dot(inv_yy, cov_xy.T), inv_xx), cov_xy)
            eigenvalues = np.linalg.eigvals(M)
            self.eigenvalue = np.max(np.abs(eigenvalues))
            self.rank = np.sum(np.abs(eigenvalues) > 0.1)
        except:
            return
        spread = log_p[-1] - np.mean(log_p)
        std = np.std(log_p)
        z = spread / (std + 1e-10)
        if self.rank > 0 and z < -2:
            self.signal = 1
            self.confidence = min(abs(z) / 4, 1.0)
        elif self.rank > 0 and z > 2:
            self.signal = -1
            self.confidence = min(abs(z) / 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(149)
class HedgeRatioEstimator(BaseFormula):
    """ID 149: Hedge Ratio for Pairs Trading"""

    CATEGORY = "mean_reversion"
    NAME = "HedgeRatioEstimator"
    DESCRIPTION = "Optimal hedge ratio estimation"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.hedge_ratio = 1.0
        self.r_squared = 0.0

    def _compute(self) -> None:
        if len(self.prices) < 30 or len(self.volumes) < 30:
            return
        prices = self._prices_array()
        volumes = self._volumes_array()
        log_p = np.log(prices)
        log_v = np.log(volumes + 1)
        p_mean = np.mean(log_p)
        v_mean = np.mean(log_v)
        cov = np.sum((log_p - p_mean) * (log_v - v_mean))
        var_v = np.sum((log_v - v_mean) ** 2)
        if var_v < 1e-10:
            return
        self.hedge_ratio = cov / var_v
        spread = log_p - self.hedge_ratio * log_v
        ss_res = np.sum((spread - np.mean(spread)) ** 2)
        ss_tot = np.sum((log_p - p_mean) ** 2)
        self.r_squared = 1 - ss_res / (ss_tot + 1e-10)
        spread_z = (spread[-1] - np.mean(spread)) / (np.std(spread) + 1e-10)
        if abs(spread_z) > 2 and self.r_squared > 0.5:
            self.signal = 1 if spread_z < 0 else -1
            self.confidence = min(abs(spread_z) / 4, 1.0) * self.r_squared
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(150)
class KalmanHedgeRatio(BaseFormula):
    """ID 150: Kalman Filter Hedge Ratio"""

    CATEGORY = "mean_reversion"
    NAME = "KalmanHedgeRatio"
    DESCRIPTION = "Dynamic hedge ratio via Kalman filter"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.beta = 1.0
        self.P = 1.0
        self.Q = 0.001
        self.R = 0.1

    def _compute(self) -> None:
        if len(self.prices) < 10 or len(self.volumes) < 10:
            return
        prices = self._prices_array()
        volumes = self._volumes_array()
        for i in range(len(prices)):
            x = np.log(volumes[i] + 1)
            y = np.log(prices[i])
            P_pred = self.P + self.Q
            K = P_pred * x / (x * P_pred * x + self.R + 1e-10)
            innovation = y - self.beta * x
            self.beta = self.beta + K * innovation
            self.P = (1 - K * x) * P_pred
        spread = np.log(prices[-1]) - self.beta * np.log(volumes[-1] + 1)
        spread_history = np.log(prices) - self.beta * np.log(volumes + 1)
        spread_std = np.std(spread_history)
        spread_mean = np.mean(spread_history)
        z = (spread - spread_mean) / (spread_std + 1e-10)
        if z < -2:
            self.signal = 1
            self.confidence = min(abs(z) / 4, 1.0)
        elif z > 2:
            self.signal = -1
            self.confidence = min(abs(z) / 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


__all__ = [
    'OrnsteinUhlenbeck', 'OUHalfLife', 'OUMeanLevel', 'OUVolatility',
    'OUSpeedOfReversion', 'OUEquilibrium', 'OUOptimalEntry', 'OUOptimalExit',
    'VasicekModel', 'CIRModel',
    'ZScoreSignal', 'AdaptiveZScore', 'BollingerBandReversion',
    'KeltnerChannelReversion', 'RSIMeanReversion', 'StochasticReversion',
    'EngleGrangerCoint', 'JohansenCoint', 'HedgeRatioEstimator', 'KalmanHedgeRatio',
]
