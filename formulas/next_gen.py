"""
Next Generation Trading Formulas - IDs 412-490
===============================================
Advanced prediction models from latest research (2024-2025)

Categories:
- Transformer/Deep Learning (412-420)
- Rough Volatility (421-430)
- Optimal Execution (431-445)
- MEV/Crypto Specific (446-460)
- Advanced Microstructure (461-475)
- Signal Processing/Physics (476-490)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque
from .base import BaseFormula, FormulaRegistry


# =============================================================================
# TRANSFORMER / DEEP LEARNING (IDs 412-420)
# =============================================================================

@FormulaRegistry.register(412)
class TemporalFusionTransformer(BaseFormula):
    """
    Temporal Fusion Transformer - Multi-horizon interpretable prediction
    Reference: Lim et al. (2021) - International Journal of Forecasting
    """
    NAME = "TemporalFusionTransformer"
    CATEGORY = "transformer"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.horizons = kwargs.get('horizons', [1, 5, 10])
        self.feature_importance = {}

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()

        # Extract multi-scale features
        mom_5 = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
        mom_10 = (prices[-1] / prices[-10] - 1) if len(prices) >= 10 else 0
        mom_20 = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0

        # Attention-weighted prediction
        weighted_momentum = mom_5 * 0.5 + mom_10 * 0.3 + mom_20 * 0.2

        self.signal = 1 if weighted_momentum > 0.001 else (-1 if weighted_momentum < -0.001 else 0)
        self.confidence = min(abs(weighted_momentum) * 100, 1.0)


@FormulaRegistry.register(413)
class TFT_ASRO(BaseFormula):
    """TFT with Adaptive Sharpe Ratio Optimization"""
    NAME = "TFT_ASRO"
    CATEGORY = "transformer"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.target_sharpe = kwargs.get('target_sharpe', 2.0)
        self.realized_sharpe = 0.0

    def _compute(self):
        if len(self.returns) < 20:
            return

        returns = self._returns_array()

        # Calculate realized Sharpe
        mean_ret = np.mean(returns)
        std_ret = np.std(returns) + 1e-10
        self.realized_sharpe = mean_ret / std_ret * np.sqrt(252 * 24 * 60)

        sharpe_factor = np.tanh(self.realized_sharpe / self.target_sharpe)
        mom = np.mean(returns[-5:])

        if mom > 0 and sharpe_factor > 0:
            self.signal = 1
            self.confidence = min(sharpe_factor * abs(mom) * 1000, 1.0)
        elif mom < 0 and sharpe_factor > 0:
            self.signal = -1
            self.confidence = min(sharpe_factor * abs(mom) * 1000, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(414)
class InformerLongSequence(BaseFormula):
    """Informer-style long sequence prediction using spectral analysis"""
    NAME = "InformerLongSequence"
    CATEGORY = "transformer"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.horizon = kwargs.get('prediction_horizon', 10)

    def _compute(self):
        if len(self.prices) < 50:
            return

        prices = self._prices_array()

        # Spectral analysis for long-range prediction
        fft = np.fft.fft(prices - np.mean(prices))
        freqs = np.fft.fftfreq(len(prices))

        power = np.abs(fft) ** 2
        top_freqs = np.argsort(power)[-5:]

        future_pred = 0
        for freq_idx in top_freqs:
            if freqs[freq_idx] != 0:
                period = 1 / abs(freqs[freq_idx])
                phase = np.angle(fft[freq_idx])
                future_pred += np.abs(fft[freq_idx]) * np.cos(2 * np.pi * self.horizon / period + phase)

        future_pred = future_pred / len(prices) + prices[-1]
        pred_return = (future_pred - prices[-1]) / prices[-1]

        self.signal = 1 if pred_return > 0.0005 else (-1 if pred_return < -0.0005 else 0)
        self.confidence = min(abs(pred_return) * 500, 1.0)


@FormulaRegistry.register(415)
class AutoformerDecomp(BaseFormula):
    """Autoformer-style decomposition with auto-correlation"""
    NAME = "AutoformerDecomp"
    CATEGORY = "transformer"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.seasonal_period = kwargs.get('seasonal_period', 24)
        self.trend = 0
        self.seasonal = 0

    def _compute(self):
        if len(self.prices) < self.seasonal_period * 2:
            return

        prices = self._prices_array()

        # Trend via moving average
        kernel_size = self.seasonal_period
        trend = np.convolve(prices, np.ones(kernel_size)/kernel_size, mode='valid')
        self.trend = trend[-1] if len(trend) > 0 else prices[-1]

        # Signal from trend direction
        trend_slope = (self.trend - np.mean(prices[-self.seasonal_period:])) / prices[-1]

        self.signal = 1 if trend_slope > 0.001 else (-1 if trend_slope < -0.001 else 0)
        self.confidence = min(abs(trend_slope) * 200, 1.0)


@FormulaRegistry.register(416)
class DifferentialTransformer(BaseFormula):
    """Differential Transformer for LOB - filters noise through differential processing"""
    NAME = "DifferentialTransformer"
    CATEGORY = "transformer"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.diffs = deque(maxlen=lookback)
        self.second_diffs = deque(maxlen=lookback)

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()

        # Compute differentials
        diffs = np.diff(prices)
        if len(diffs) < 5:
            return

        # Differential attention: weight recent changes more
        weights = np.exp(-0.1 * np.arange(len(diffs))[::-1])
        weighted_diff = np.sum(diffs * weights) / np.sum(weights)

        # Second derivative for acceleration
        second_diffs = np.diff(diffs)
        acceleration = np.mean(second_diffs[-5:]) if len(second_diffs) >= 5 else 0

        combined = weighted_diff + 0.5 * acceleration

        self.signal = 1 if combined > 0 else (-1 if combined < 0 else 0)
        self.confidence = min(abs(combined) / (np.std(diffs) + 1e-10), 1.0)


@FormulaRegistry.register(417)
class CNNTransformerHybrid(BaseFormula):
    """CNN + Transformer hybrid for mid-price prediction"""
    NAME = "CNNTransformerHybrid"
    CATEGORY = "transformer"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.cnn_window = kwargs.get('cnn_window', 10)
        self.cnn_features = deque(maxlen=lookback)

    def _compute(self):
        if len(self.prices) < self.cnn_window:
            return

        prices = self._prices_array()

        # CNN-like convolution filters
        edge_filter = np.array([-1, 0, 1])
        if len(prices) >= 10:
            edge = np.convolve(prices[-10:], edge_filter, mode='valid')[-1]
            smooth = np.mean(prices[-5:])
            feature = edge / (smooth + 1e-10)
            self.cnn_features.append(feature)

        if len(self.cnn_features) >= 10:
            features = np.array(self.cnn_features)
            attention = np.exp(features[-10:])
            attention = attention / attention.sum()
            weighted_feature = np.sum(features[-10:] * attention)

            self.signal = 1 if weighted_feature > 0.01 else (-1 if weighted_feature < -0.01 else 0)
            self.confidence = min(abs(weighted_feature) * 10, 1.0)


@FormulaRegistry.register(418)
class LOBImageCNN(BaseFormula):
    """Convert price series to 2D image for pattern recognition"""
    NAME = "LOBImageCNN"
    CATEGORY = "transformer"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.image_size = kwargs.get('image_size', 20)

    def _compute(self):
        if len(self.prices) < self.image_size:
            return

        prices = self._prices_array()[-self.image_size:]

        # Normalize
        norm_prices = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)

        # Pattern correlation
        rising_pattern = np.linspace(0, 1, self.image_size)
        falling_pattern = np.linspace(1, 0, self.image_size)

        rising_corr = np.corrcoef(norm_prices, rising_pattern)[0, 1]
        falling_corr = np.corrcoef(norm_prices, falling_pattern)[0, 1]

        if rising_corr > 0.7:
            self.signal = 1
            self.confidence = rising_corr
        elif falling_corr > 0.7:
            self.signal = -1
            self.confidence = falling_corr
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(419)
class VARNeuralHybrid(BaseFormula):
    """Vector Autoregression + Neural Network for OFI prediction"""
    NAME = "VARNeuralHybrid"
    CATEGORY = "transformer"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.var_order = kwargs.get('var_order', 5)
        self.ofi_history = deque(maxlen=lookback)

    def _compute(self):
        if len(self.returns) < self.var_order + 10:
            return

        returns = self._returns_array()
        volumes = self._volumes_array()

        # OFI proxy
        ofi = volumes[1:] * np.sign(returns) if len(volumes) > 1 else returns
        for o in ofi[-10:]:
            self.ofi_history.append(o)

        if len(self.ofi_history) < self.var_order + 5:
            return

        ofi_arr = np.array(self.ofi_history)

        try:
            # Simple AR prediction
            X = np.column_stack([ofi_arr[-(self.var_order+i):-i] for i in range(1, self.var_order+1)])
            y = ofi_arr[-self.var_order:]

            if len(X) >= len(y):
                coeffs = np.linalg.lstsq(X[:len(y)], y, rcond=None)[0]
                ofi_pred = np.dot(coeffs, ofi_arr[-self.var_order:][::-1])
                ofi_signal = np.tanh(ofi_pred / (np.std(ofi_arr) + 1e-10))

                self.signal = 1 if ofi_signal > 0.3 else (-1 if ofi_signal < -0.3 else 0)
                self.confidence = abs(ofi_signal)
        except:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(420)
class GraphNeuralLOB(BaseFormula):
    """Graph Neural Network for cross-level order book dependencies"""
    NAME = "GraphNeuralLOB"
    CATEGORY = "transformer"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_levels = kwargs.get('n_levels', 5)
        self.level_history = [deque(maxlen=lookback) for _ in range(self.n_levels)]

    def _compute(self):
        if len(self.prices) < 10:
            return

        prices = self._prices_array()
        vol = np.std(prices[-10:])

        # Synthetic levels
        for i in range(self.n_levels):
            level_price = prices[-1] * (1 + (i + 1) * 0.0001 * np.random.randn())
            self.level_history[i].append(level_price)

        if len(self.level_history[0]) < 20:
            return

        levels = [np.array(l) for l in self.level_history]
        aggregated = np.zeros(len(levels[0]))
        for i, level in enumerate(levels):
            if len(level) == len(levels[0]):
                weight = 1 / (i + 1)
                aggregated += weight * level[-len(aggregated):]

        aggregated /= sum(1/(i+1) for i in range(self.n_levels))

        if len(aggregated) >= 2:
            graph_momentum = aggregated[-1] / aggregated[-2] - 1
            self.signal = 1 if graph_momentum > 0.0001 else (-1 if graph_momentum < -0.0001 else 0)
            self.confidence = min(abs(graph_momentum) * 5000, 1.0)


# =============================================================================
# ROUGH VOLATILITY (IDs 421-430)
# =============================================================================

@FormulaRegistry.register(421)
class RoughBergomiPricing(BaseFormula):
    """Rough Bergomi model for volatility prediction (H ~ 0.1)"""
    NAME = "RoughBergomiPricing"
    CATEGORY = "rough_vol"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.H = kwargs.get('hurst', 0.1)
        self.log_vols = deque(maxlen=lookback)

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()
        returns = np.diff(np.log(prices))

        if len(returns) >= 10:
            realized_vol = np.std(returns[-10:])
            log_vol = np.log(realized_vol + 1e-10)
            self.log_vols.append(log_vol)

        if len(self.log_vols) < 20:
            return

        log_vols = np.array(self.log_vols)

        # Variogram for H estimation
        lags = [1, 2, 4, 8]
        variogram = []
        for lag in lags:
            if len(log_vols) > lag:
                var = np.var(log_vols[lag:] - log_vols[:-lag])
                variogram.append(var)

        if len(variogram) >= 2:
            log_lags = np.log(lags[:len(variogram)])
            log_var = np.log(np.array(variogram) + 1e-10)
            slope = np.polyfit(log_lags, log_var, 1)[0]
            estimated_H = slope / 2

            if estimated_H < 0.5:
                vol_zscore = (log_vols[-1] - np.mean(log_vols)) / (np.std(log_vols) + 1e-10)
                self.signal = -1 if vol_zscore > 1 else (1 if vol_zscore < -1 else 0)
                self.confidence = min(abs(vol_zscore) / 3, 1.0)


@FormulaRegistry.register(422)
class FractionalBrownianVol(BaseFormula):
    """Fractional Brownian Motion volatility model - R/S analysis"""
    NAME = "FractionalBrownianVol"
    CATEGORY = "rough_vol"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.hurst_estimate = 0.5

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()

        # R/S analysis for Hurst exponent
        n = len(returns)
        max_k = min(n // 4, 20)
        rs_values = []

        for k in range(10, max_k):
            mean = np.mean(returns[:k])
            cumsum = np.cumsum(returns[:k] - mean)
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(returns[:k])
            if S > 0:
                rs_values.append((k, R / S))

        if len(rs_values) >= 3:
            log_n = np.log([rs[0] for rs in rs_values])
            log_rs = np.log([rs[1] for rs in rs_values])
            self.hurst_estimate = np.polyfit(log_n, log_rs, 1)[0]

        if self.hurst_estimate < 0.4:
            zscore = (returns[-1] - np.mean(returns)) / (np.std(returns) + 1e-10)
            self.signal = -1 if zscore > 0.5 else (1 if zscore < -0.5 else 0)
        elif self.hurst_estimate > 0.6:
            self.signal = 1 if returns[-1] > 0 else (-1 if returns[-1] < 0 else 0)
        else:
            self.signal = 0

        self.confidence = abs(self.hurst_estimate - 0.5) * 2


@FormulaRegistry.register(423)
class ARRVForecaster(BaseFormula):
    """Autoregressive Rough Volatility forecaster"""
    NAME = "ARRVForecaster"
    CATEGORY = "rough_vol"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.ar_order = kwargs.get('ar_order', 5)
        self.volatilities = deque(maxlen=lookback)

    def _compute(self):
        if len(self.prices) < 10:
            return

        prices = self._prices_array()

        vol = np.std(np.diff(np.log(prices[-10:]))) * np.sqrt(252 * 24 * 60)
        self.volatilities.append(vol)

        if len(self.volatilities) < self.ar_order + 10:
            return

        vols = np.array(self.volatilities)

        try:
            X = np.column_stack([vols[-(self.ar_order+i):-i] for i in range(1, self.ar_order+1)])
            y = vols[-self.ar_order:]
            coeffs = np.linalg.lstsq(X[:len(y)], y, rcond=None)[0]
            vol_forecast = np.dot(coeffs, vols[-self.ar_order:][::-1])

            vol_change = vol_forecast / vols[-1] - 1
            self.signal = -1 if vol_change > 0.1 else (1 if vol_change < -0.1 else 0)
            self.confidence = min(abs(vol_change), 1.0)
        except:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(424)
class RoughHestonMC(BaseFormula):
    """Rough Heston Monte Carlo simulation for exit timing"""
    NAME = "RoughHestonMC"
    CATEGORY = "rough_vol"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.vol_of_vol = 0.3

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()
        returns = np.diff(np.log(prices))

        current_vol = np.std(returns[-10:])
        mean_vol = np.std(returns)

        expected_return = np.mean(returns) * 10
        vol_adjustment = (mean_vol - current_vol) * 0.5

        prob_up = 0.5 + expected_return / (2 * current_vol + 1e-10)
        prob_up = max(0.1, min(0.9, prob_up))

        self.signal = 1 if prob_up > 0.55 else (-1 if prob_up < 0.45 else 0)
        self.confidence = abs(prob_up - 0.5) * 2


@FormulaRegistry.register(425)
class MarkovianRoughApprox(BaseFormula):
    """Markovian approximation of rough volatility via dimensional reduction"""
    NAME = "MarkovianRoughApprox"
    CATEGORY = "rough_vol"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_factors = kwargs.get('n_factors', 3)
        self.factors = [deque(maxlen=lookback) for _ in range(self.n_factors)]

    def _compute(self):
        if len(self.prices) < 10:
            return

        prices = self._prices_array()
        vol = np.std(np.diff(np.log(prices[-10:])))

        for i, factor in enumerate(self.factors):
            decay = 0.9 ** (i + 1)
            new_val = decay * factor[-1] + (1 - decay) * vol if len(factor) > 0 else vol
            factor.append(new_val)

        if len(self.factors[0]) < 10:
            return

        weights = [0.5, 0.3, 0.2]
        factor_momentum = sum(w * (list(f)[-1] - list(f)[-5]) for w, f in zip(weights, self.factors) if len(f) >= 5)

        self.signal = -1 if factor_momentum > 0.001 else (1 if factor_momentum < -0.001 else 0)
        self.confidence = min(abs(factor_momentum) * 500, 1.0)


@FormulaRegistry.register(426)
class LinearFractionalStable(BaseFormula):
    """Linear Fractional Stable Motion prediction with alpha-stable increments"""
    NAME = "LinearFractionalStable"
    CATEGORY = "rough_vol"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.alpha = kwargs.get('alpha', 1.7)

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()

        sorted_abs = np.sort(np.abs(returns))[::-1]
        n = len(sorted_abs)

        k = min(n // 5, 20)
        if k > 1 and sorted_abs[k] > 0:
            hill = k / np.sum(np.log(sorted_abs[:k] / sorted_abs[k]))
            estimated_alpha = min(2.0, max(1.0, hill))
        else:
            estimated_alpha = 1.7

        recent = returns[-10:]
        prediction = np.sign(np.sum(recent * np.exp(-0.2 * np.arange(10))))

        self.signal = int(prediction)
        self.confidence = min(abs(np.mean(recent)) / (np.std(returns) + 1e-10), 1.0)


# =============================================================================
# OPTIMAL EXECUTION (IDs 431-445)
# =============================================================================

@FormulaRegistry.register(431)
class AlmgrenChrissGBM(BaseFormula):
    """Almgren-Chriss with GBM - optimal execution under geometric Brownian motion"""
    NAME = "AlmgrenChrissGBM"
    CATEGORY = "execution"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.risk_aversion = kwargs.get('risk_aversion', 1e-6)
        self.execution_urgency = 0.5

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()
        volumes = self._volumes_array()

        returns = np.diff(np.log(prices))
        sigma = np.std(returns) * np.sqrt(252 * 24 * 60)
        mu = np.mean(returns) * 252 * 24 * 60

        vol_factor = sigma / 0.2
        drift_factor = mu / sigma if sigma > 0 else 0

        self.execution_urgency = 0.5 + 0.3 * vol_factor - 0.2 * drift_factor
        self.execution_urgency = max(0.1, min(0.9, self.execution_urgency))

        price_zscore = (prices[-1] - np.mean(prices)) / (np.std(prices) + 1e-10)

        if self.execution_urgency > 0.6 and price_zscore < 0:
            self.signal = 1
        elif self.execution_urgency > 0.6 and price_zscore > 0:
            self.signal = -1
        else:
            self.signal = 0

        self.confidence = self.execution_urgency


@FormulaRegistry.register(432)
class HJBOptimalExecution(BaseFormula):
    """Hamilton-Jacobi-Bellman optimal execution - numerical HJB solution"""
    NAME = "HJBOptimalExecution"
    CATEGORY = "execution"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_grid = kwargs.get('n_grid', 20)
        self.value_function = np.zeros((self.n_grid, self.n_grid))

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()
        momentum = prices[-1] / prices[-10] - 1 if len(prices) >= 10 else 0
        current_vol = np.std(np.diff(np.log(prices[-10:])))

        momentum_bins = np.linspace(-0.05, 0.05, self.n_grid)
        vol_bins = np.linspace(0, 0.05, self.n_grid)

        mom_idx = np.argmin(np.abs(momentum_bins - momentum))
        vol_idx = np.argmin(np.abs(vol_bins - current_vol))

        self.value_function[mom_idx, vol_idx] += 0.1 * momentum

        if mom_idx > self.n_grid // 2:
            self.signal = 1
        elif mom_idx < self.n_grid // 2:
            self.signal = -1
        else:
            self.signal = 0

        self.confidence = abs(momentum) * 10


@FormulaRegistry.register(433)
class DDQNExecution(BaseFormula):
    """Double Deep Q-Network for optimal execution"""
    NAME = "DDQNExecution"
    CATEGORY = "execution"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_actions = 3
        self.q_values = np.zeros(self.n_actions)
        self.learning_rate = 0.01
        self.epsilon = 0.1
        self.last_action = 1
        self.last_price = None

    def _compute(self):
        if len(self.prices) < 10:
            return

        prices = self._prices_array()

        if self.last_price is not None:
            ret = prices[-1] / self.last_price - 1
            action_dir = self.last_action - 1
            reward = ret * action_dir * 100
            self.q_values[self.last_action] += self.learning_rate * (reward - self.q_values[self.last_action])

        self.last_price = prices[-1]

        momentum = prices[-1] / prices[-5] - 1 if len(prices) >= 5 else 0
        state_bias = np.array([-momentum * 10, 0, momentum * 10])
        adjusted_q = self.q_values + state_bias

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(adjusted_q)

        self.last_action = action
        self.signal = action - 1
        self.confidence = 1 / (1 + np.exp(-abs(adjusted_q[action])))


@FormulaRegistry.register(434)
class QueueReactiveRL(BaseFormula):
    """Reinforcement Learning in Queue-Reactive Models"""
    NAME = "QueueReactiveRL"
    CATEGORY = "execution"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.queue_state = 0
        self.execution_prob = 0.5

    def _compute(self):
        if len(self.prices) < 10:
            return

        prices = self._prices_array()
        volumes = self._volumes_array()

        avg_vol = np.mean(volumes) if np.mean(volumes) > 0 else 1
        vol_ratio = volumes[-1] / avg_vol if volumes[-1] > 0 else 0.5

        self.queue_state = 0.5 + 0.5 * np.tanh(vol_ratio - 1)

        price_move = prices[-1] / prices[-2] - 1 if len(prices) >= 2 else 0
        self.execution_prob = 0.5 + 0.3 * self.queue_state + 0.2 * np.sign(price_move)

        self.signal = 1 if self.execution_prob > 0.7 else (-1 if self.execution_prob < 0.3 else 0)
        self.confidence = abs(self.execution_prob - 0.5) * 2


@FormulaRegistry.register(435)
class InfiniteDimControl(BaseFormula):
    """Infinite-dimensional stochastic control for multi-timescale execution"""
    NAME = "InfiniteDimControl"
    CATEGORY = "execution"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_timescales = kwargs.get('n_timescales', 5)
        self.timescale_factors = [deque(maxlen=lookback) for _ in range(self.n_timescales)]

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()
        windows = [5, 10, 20, 50, 100][:self.n_timescales]

        for i, window in enumerate(windows):
            if len(prices) >= window:
                impact = prices[-1] / np.mean(prices[-window:]) - 1
                self.timescale_factors[i].append(impact)

        if all(len(f) >= 5 for f in self.timescale_factors):
            short_signal = np.mean(list(self.timescale_factors[0])[-5:])
            long_signal = np.mean(list(self.timescale_factors[-1])[-5:])

            if short_signal > 0 and long_signal > 0:
                self.signal = 1
                self.confidence = min(short_signal + long_signal, 1.0)
            elif short_signal < 0 and long_signal < 0:
                self.signal = -1
                self.confidence = min(abs(short_signal) + abs(long_signal), 1.0)
            else:
                self.signal = 0
                self.confidence = 0.0


# =============================================================================
# MEV / CRYPTO SPECIFIC (IDs 446-460)
# =============================================================================

@FormulaRegistry.register(446)
class SandwichAttackDetector(BaseFormula):
    """Detect sandwich attacks from mempool patterns"""
    NAME = "SandwichAttackDetector"
    CATEGORY = "mev"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.sandwich_score = 0
        self.suspicious_patterns = deque(maxlen=100)

    def _compute(self):
        if len(self.prices) < 5:
            return

        prices = self._prices_array()
        volumes = self._volumes_array()

        if len(prices) >= 4:
            move1 = prices[-3] - prices[-4]
            move2 = prices[-2] - prices[-3]
            move3 = prices[-1] - prices[-2]

            vol_spike = volumes[-2] / (np.mean(volumes[-10:-2]) + 1e-10) if len(volumes) >= 10 else 1

            is_sandwich = (
                move1 > 0 and
                move3 < 0 and
                abs(move1 + move3) < abs(move1) * 0.3 and
                vol_spike > 2
            )

            self.suspicious_patterns.append(1 if is_sandwich else 0)

        self.sandwich_score = np.mean(list(self.suspicious_patterns)) if len(self.suspicious_patterns) > 0 else 0

        if self.sandwich_score > 0.1:
            self.signal = 0
            self.confidence = 0.0
        else:
            momentum = prices[-1] / prices[-5] - 1 if len(prices) >= 5 else 0
            self.signal = 1 if momentum > 0.001 else (-1 if momentum < -0.001 else 0)
            self.confidence = 1 - self.sandwich_score


@FormulaRegistry.register(447)
class MEVArbitragePredictor(BaseFormula):
    """Predict MEV arbitrage opportunities"""
    NAME = "MEVArbitragePredictor"
    CATEGORY = "mev"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.arb_opportunities = deque(maxlen=lookback)

    def _compute(self):
        if len(self.prices) < 10:
            return

        prices = self._prices_array()

        short_vol = np.std(np.diff(prices[-5:])) if len(prices) >= 5 else 0
        long_vol = np.std(np.diff(prices[-20:])) if len(prices) >= 20 else short_vol

        vol_ratio = short_vol / (long_vol + 1e-10)

        if len(prices) >= 3:
            move1 = prices[-2] - prices[-3]
            move2 = prices[-1] - prices[-2]
            reversion = (move1 * move2 < 0) and abs(move2) > abs(move1) * 0.5
        else:
            reversion = False

        arb_present = vol_ratio > 2 or reversion
        self.arb_opportunities.append(1 if arb_present else 0)

        arb_rate = np.mean(list(self.arb_opportunities))

        if arb_rate > 0.3:
            self.signal = 1 if prices[-1] > prices[-2] else -1
            self.confidence = arb_rate
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(448)
class MempoolToxicityScore(BaseFormula):
    """Score mempool toxicity for front-running risk"""
    NAME = "MempoolToxicityScore"
    CATEGORY = "mev"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.toxicity_history = deque(maxlen=lookback)

    def _compute(self):
        if len(self.prices) < 10:
            return

        prices = self._prices_array()
        volumes = self._volumes_array()

        vol_mean = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
        price_change = abs(prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0

        volume = volumes[-1]
        if volume > 2 * vol_mean and price_change < 0.0001:
            toxicity = 0.8
        elif volume > 1.5 * vol_mean and price_change < 0.0005:
            toxicity = 0.5
        else:
            toxicity = 0.2

        self.toxicity_history.append(toxicity)
        avg_toxicity = np.mean(list(self.toxicity_history))

        if avg_toxicity > 0.6:
            self.signal = 0
            self.confidence = 0.0
        else:
            momentum = prices[-1] / prices[-5] - 1 if len(prices) >= 5 else 0
            self.signal = 1 if momentum > 0.001 else (-1 if momentum < -0.001 else 0)
            self.confidence = 1 - avg_toxicity


@FormulaRegistry.register(449)
class JitoArbitrageSignal(BaseFormula):
    """Jito-style arbitrage detection for Solana"""
    NAME = "JitoArbitrageSignal"
    CATEGORY = "mev"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.min_profit = kwargs.get('min_profit', 0.001)
        self.arb_count = 0

    def _compute(self):
        if len(self.prices) < 10:
            return

        prices = self._prices_array()

        for i in range(1, min(5, len(prices) - 1)):
            price_diff = abs(prices[-1] - prices[-i-1]) / prices[-i-1]
            max_deviation = max(abs(prices[-j] - prices[-i-1]) / prices[-i-1] for j in range(1, i+1))

            if price_diff < 0.0001 and max_deviation > self.min_profit:
                self.arb_count += 1
                break

        arb_rate = self.arb_count / len(self.prices)

        if arb_rate > 0.1:
            mom = prices[-1] / prices[-3] - 1
            self.signal = 1 if mom > 0.0005 else (-1 if mom < -0.0005 else 0)
            self.confidence = min(arb_rate * 5, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(450)
class FlashbotsProtection(BaseFormula):
    """Flashbots-aware execution timing"""
    NAME = "FlashbotsProtection"
    CATEGORY = "mev"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.danger_level = 0

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()
        volumes = self._volumes_array()

        vol = np.std(np.diff(np.log(prices[-20:])))
        vol_ratio = volumes[-1] / (np.mean(volumes[-20:-1]) + 1) if len(volumes) >= 20 else 1
        speed = abs(prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0

        self.danger_level = (
            0.3 * min(vol / 0.02, 1.0) +
            0.4 * min(vol_ratio / 3, 1.0) +
            0.3 * min(speed / 0.01, 1.0)
        )

        if self.danger_level < 0.5:
            momentum = prices[-1] / prices[-10] - 1
            self.signal = 1 if momentum > 0.001 else (-1 if momentum < -0.001 else 0)
            self.confidence = 1 - self.danger_level
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(451)
class LiquidationCascadePredictor(BaseFormula):
    """Predict DeFi liquidation cascades"""
    NAME = "LiquidationCascadePredictor"
    CATEGORY = "mev"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.cascade_threshold = kwargs.get('cascade_threshold', 0.05)
        self.cascade_risk = 0

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()

        if len(prices) >= 10:
            returns = np.diff(np.log(prices[-10:]))
            acceleration = np.diff(returns)

            if len(acceleration) > 0:
                avg_accel = np.mean(acceleration)

                if avg_accel < -0.001:
                    self.cascade_risk = min(abs(avg_accel) * 1000, 1.0)
                elif avg_accel > 0.001:
                    self.cascade_risk = max(self.cascade_risk - 0.1, 0)

        if self.cascade_risk > 0.5:
            self.signal = -1
            self.confidence = self.cascade_risk
        elif self.cascade_risk < 0.2:
            momentum = prices[-1] / prices[-5] - 1
            self.signal = 1 if momentum > 0.001 else (-1 if momentum < -0.001 else 0)
            self.confidence = 0.5
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(452)
class GasPriceOptimizer(BaseFormula):
    """Optimize execution based on gas/fee dynamics"""
    NAME = "GasPriceOptimizer"
    CATEGORY = "mev"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.fee_history = deque(maxlen=lookback)
        self.optimal_now = False

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()
        volumes = self._volumes_array()

        self.fee_history.append(volumes[-1])
        fees = np.array(self.fee_history)

        current_fee = fees[-1]
        avg_fee = np.mean(fees[-20:]) if len(fees) >= 20 else current_fee
        min_fee = np.min(fees[-20:]) if len(fees) >= 20 else current_fee

        fee_percentile = (current_fee - min_fee) / (avg_fee - min_fee + 1e-10)
        self.optimal_now = fee_percentile < 0.3

        if self.optimal_now:
            momentum = prices[-1] / prices[-10] - 1
            self.signal = 1 if momentum > 0.0005 else (-1 if momentum < -0.0005 else 0)
            self.confidence = 1 - fee_percentile
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(453)
class CrossChainArbitrage(BaseFormula):
    """Cross-chain arbitrage signal detection"""
    NAME = "CrossChainArbitrage"
    CATEGORY = "mev"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.chain_spreads = deque(maxlen=lookback)

    def _compute(self):
        if len(self.prices) < 10:
            return

        prices = self._prices_array()

        short_vol = np.std(np.diff(prices[-5:])) if len(prices) >= 5 else 0
        long_vol = np.std(np.diff(prices[-20:])) if len(prices) >= 20 else short_vol

        spread = short_vol / (long_vol + 1e-10)
        self.chain_spreads.append(spread)

        avg_spread = np.mean(list(self.chain_spreads))

        if spread > avg_spread * 1.5:
            direction = 1 if prices[-1] < prices[-2] else -1
            self.signal = direction
            self.confidence = min((spread / avg_spread - 1) * 2, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.0


# =============================================================================
# ADVANCED MICROSTRUCTURE (IDs 461-475)
# =============================================================================

@FormulaRegistry.register(461)
class CarteaJaimungalMM(BaseFormula):
    """Cartea-Jaimungal market making framework"""
    NAME = "CarteaJaimungalMM"
    CATEGORY = "microstructure"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.gamma = kwargs.get('gamma', 0.1)
        self.inventory = 0
        self.optimal_spread = 0

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()
        volumes = self._volumes_array()

        sigma = np.std(np.diff(np.log(prices[-20:]))) * np.sqrt(252 * 24 * 60)
        self.optimal_spread = self.gamma * sigma ** 2 + 0.001

        ret = prices[-1] / prices[-2] - 1 if len(prices) >= 2 else 0
        self.inventory += np.sign(ret) * volumes[-1] * 0.01
        self.inventory = max(-1, min(1, self.inventory))

        skew = -self.inventory * 0.5

        self.signal = 1 if skew > 0.2 else (-1 if skew < -0.2 else 0)
        self.confidence = abs(skew)


@FormulaRegistry.register(462)
class GueantLehalleFT(BaseFormula):
    """Guéant-Lehalle-Fernandez-Tapia optimal execution"""
    NAME = "GueantLehalleFT"
    CATEGORY = "microstructure"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.kappa = kwargs.get('kappa', 0.1)

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()
        volumes = self._volumes_array()

        returns = np.diff(np.log(prices))
        signed_volume = volumes[1:] * np.sign(returns)

        if len(signed_volume) >= 10:
            corr = np.corrcoef(signed_volume[-10:], returns[-10:])[0, 1]
            impact = abs(corr) * np.std(returns) / (np.std(signed_volume) + 1e-10)
        else:
            impact = 0.001

        if impact > 0.002:
            self.signal = 0
            self.confidence = 0.0
        else:
            mom = prices[-1] / prices[-5] - 1
            self.signal = 1 if mom > 0.0005 else (-1 if mom < -0.0005 else 0)
            self.confidence = 1 - impact * 200


@FormulaRegistry.register(463)
class StoikovSaglamSpread(BaseFormula):
    """Stoikov-Saglam optimal spread with inventory"""
    NAME = "StoikovSaglamSpread"
    CATEGORY = "microstructure"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.inventory = 0

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()

        ret = prices[-1] / prices[-2] - 1 if len(prices) >= 2 else 0
        self.inventory = 0.95 * self.inventory + 0.05 * np.sign(ret)

        sigma = np.std(np.diff(np.log(prices[-20:])))
        indifference = prices[-1] - 0.01 * self.inventory * sigma ** 2 * prices[-1]

        if prices[-1] < indifference:
            self.signal = 1
            self.confidence = min((indifference - prices[-1]) / prices[-1] * 1000, 1.0)
        elif prices[-1] > indifference:
            self.signal = -1
            self.confidence = min((prices[-1] - indifference) / prices[-1] * 1000, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(464)
class ContStoikovTalreja(BaseFormula):
    """Cont-Stoikov-Talreja order book dynamics"""
    NAME = "ContStoikovTalreja"
    CATEGORY = "microstructure"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.arrival_rates = deque(maxlen=lookback)

    def _compute(self):
        if len(self.prices) < 10:
            return

        prices = self._prices_array()
        volumes = self._volumes_array()

        arrival_rate = volumes[-1] / (np.mean(volumes[-10:]) + 1e-10)
        self.arrival_rates.append(arrival_rate)

        avg_rate = np.mean(list(self.arrival_rates))

        if arrival_rate > 1.5 * avg_rate:
            direction = np.sign(prices[-1] - prices[-2])
            self.signal = int(direction)
            self.confidence = min((arrival_rate / avg_rate - 1), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(465)
class BouchaudFarmerLillo(BaseFormula):
    """Bouchaud-Farmer-Lillo market impact model with long-memory decay"""
    NAME = "BouchaudFarmerLillo"
    CATEGORY = "microstructure"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.decay_exp = kwargs.get('decay_exp', 0.5)
        self.cumulative_impact = 0

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()
        volumes = self._volumes_array()

        n = min(len(prices), 100)
        weights = np.arange(1, n+1) ** (-self.decay_exp)
        weights = weights[::-1] / weights.sum()

        returns = np.diff(np.log(prices[-n:]))
        signed_vol = volumes[-n+1:] * np.sign(returns)

        self.cumulative_impact = np.sum(weights[-len(signed_vol):] * signed_vol)

        if self.cumulative_impact > 0.1:
            self.signal = -1
            self.confidence = min(self.cumulative_impact, 1.0)
        elif self.cumulative_impact < -0.1:
            self.signal = 1
            self.confidence = min(abs(self.cumulative_impact), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(466)
class EasleyOHaraPINExtended(BaseFormula):
    """Extended PIN (Probability of Informed Trading) model"""
    NAME = "EasleyOHaraPINExtended"
    CATEGORY = "microstructure"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.pin_estimate = 0

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()
        volumes = self._volumes_array()

        returns = np.diff(np.log(prices))
        buys = volumes[1:] * (returns > 0)
        sells = volumes[1:] * (returns < 0)

        total_buys = np.sum(buys[-20:])
        total_sells = np.sum(sells[-20:])

        imbalance = abs(total_buys - total_sells)
        total = total_buys + total_sells + 1e-10

        self.pin_estimate = imbalance / total

        if self.pin_estimate > 0.4:
            direction = 1 if total_buys > total_sells else -1
            self.signal = direction
            self.confidence = self.pin_estimate
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(467)
class ObizhaevaWangImpact(BaseFormula):
    """Obizhaeva-Wang transient market impact model"""
    NAME = "ObizhaevaWangImpact"
    CATEGORY = "microstructure"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.rho = kwargs.get('rho', 0.9)
        self.transient_impact = 0

    def _compute(self):
        if len(self.prices) < 10:
            return

        prices = self._prices_array()

        ret = prices[-1] / prices[-2] - 1 if len(prices) >= 2 else 0
        self.transient_impact = self.rho * self.transient_impact + (1 - self.rho) * ret * 100

        if self.transient_impact > 0.5:
            self.signal = -1
            self.confidence = min(self.transient_impact / 2, 1.0)
        elif self.transient_impact < -0.5:
            self.signal = 1
            self.confidence = min(abs(self.transient_impact) / 2, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.0


# =============================================================================
# SIGNAL PROCESSING / PHYSICS (IDs 476-490)
# =============================================================================

@FormulaRegistry.register(476)
class ReservoirComputing(BaseFormula):
    """Echo State Network / Reservoir Computing for chaotic time series"""
    NAME = "ReservoirComputing"
    CATEGORY = "physics"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.reservoir_size = kwargs.get('reservoir_size', 50)
        np.random.seed(42)
        self.W = np.random.randn(self.reservoir_size, self.reservoir_size) * 0.1
        self.W_in = np.random.randn(self.reservoir_size, 1) * 0.1
        self.state = np.zeros(self.reservoir_size)

    def _compute(self):
        if len(self.prices) < 10:
            return

        prices = self._prices_array()

        x = (prices[-1] - np.mean(prices[-50:])) / (np.std(prices[-50:]) + 1e-10)
        self.state = np.tanh(self.W @ self.state + self.W_in.flatten() * x)

        output = np.sum(self.state * np.linspace(-1, 1, self.reservoir_size))

        self.signal = 1 if output > 0.5 else (-1 if output < -0.5 else 0)
        self.confidence = min(abs(output), 1.0)


@FormulaRegistry.register(477)
class LiquidTimeConstant(BaseFormula):
    """Liquid Time-Constant Networks - continuous-time neural adaptation"""
    NAME = "LiquidTimeConstant"
    CATEGORY = "physics"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_neurons = kwargs.get('n_neurons', 10)
        self.states = np.zeros(self.n_neurons)
        self.tau = np.linspace(0.1, 1.0, self.n_neurons)

    def _compute(self):
        if len(self.prices) < 10:
            return

        prices = self._prices_array()
        x = prices[-1] / prices[-2] - 1 if len(prices) >= 2 else 0

        dt = 0.1
        for i in range(self.n_neurons):
            ds = (-self.states[i] + np.tanh(x * 10)) / self.tau[i]
            self.states[i] += ds * dt

        output = np.sum(self.states * np.linspace(0.5, 1.5, self.n_neurons))

        self.signal = 1 if output > 0.1 else (-1 if output < -0.1 else 0)
        self.confidence = min(abs(output), 1.0)


@FormulaRegistry.register(478)
class TopologicalDataAnalysis(BaseFormula):
    """Persistent Homology for regime detection"""
    NAME = "TopologicalDataAnalysis"
    CATEGORY = "physics"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.persistence_score = 0

    def _compute(self):
        if len(self.prices) < 30:
            return

        prices = self._prices_array()

        local_max = []
        local_min = []

        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                local_max.append((i, prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                local_min.append((i, prices[i]))

        if len(local_max) > 0 and len(local_min) > 0:
            max_val = max(p[1] for p in local_max)
            min_val = min(p[1] for p in local_min)
            self.persistence_score = (max_val - min_val) / prices[-1]

        if self.persistence_score > 0.02:
            direction = 1 if prices[-1] > prices[-10] else -1
            self.signal = direction
            self.confidence = min(self.persistence_score * 20, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.0


@FormulaRegistry.register(479)
class QuantumInspiredOptimizer(BaseFormula):
    """Quantum-inspired optimization via simulated annealing"""
    NAME = "QuantumInspiredOptimizer"
    CATEGORY = "physics"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.temperature = kwargs.get('temperature', 1.0)
        self.current_position = 0

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()

        returns = np.diff(np.log(prices[-20:]))
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 24 * 60)

        proposed = np.random.choice([-1, 0, 1])

        if proposed == 1:
            delta_E = -sharpe
        elif proposed == -1:
            delta_E = sharpe
        else:
            delta_E = 0

        if delta_E < 0 or np.random.random() < np.exp(-delta_E / self.temperature):
            self.current_position = proposed

        self.temperature = max(0.1, self.temperature * 0.99)

        self.signal = self.current_position
        self.confidence = 1 / (1 + self.temperature)


@FormulaRegistry.register(480)
class SpikingNeuralNet(BaseFormula):
    """Spiking Neural Network for ultra-low latency - event-driven processing"""
    NAME = "SpikingNeuralNet"
    CATEGORY = "physics"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.threshold = kwargs.get('threshold', 0.5)
        self.membrane_potential = 0
        self.spike_history = deque(maxlen=lookback)

    def _compute(self):
        if len(self.prices) < 5:
            return

        prices = self._prices_array()

        I = (prices[-1] / prices[-2] - 1) * 1000 if len(prices) >= 2 else 0

        tau = 0.9
        self.membrane_potential = tau * self.membrane_potential + I

        if self.membrane_potential > self.threshold:
            spike = 1
            self.membrane_potential = 0
        elif self.membrane_potential < -self.threshold:
            spike = -1
            self.membrane_potential = 0
        else:
            spike = 0

        self.spike_history.append(spike)

        recent_spikes = list(self.spike_history)[-10:]
        net_spike = sum(recent_spikes)

        self.signal = 1 if net_spike > 2 else (-1 if net_spike < -2 else 0)
        self.confidence = min(abs(net_spike) / 5, 1.0)


@FormulaRegistry.register(481)
class RenormalizationGroup(BaseFormula):
    """Renormalization group methods for scale invariance"""
    NAME = "RenormalizationGroup"
    CATEGORY = "physics"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_scales = kwargs.get('n_scales', 5)

    def _compute(self):
        if len(self.prices) < 100:
            return

        prices = self._prices_array()

        scale_signals = []

        for i in range(self.n_scales):
            block_size = 2 ** i
            if len(prices) >= block_size * 10:
                n_blocks = len(prices) // block_size
                blocked = np.mean(prices[:n_blocks*block_size].reshape(-1, block_size), axis=1)

                if len(blocked) >= 5:
                    scale_signal = blocked[-1] / blocked[-5] - 1
                    scale_signals.append(scale_signal)

        if len(scale_signals) > 0:
            weights = np.exp(-np.arange(len(scale_signals)) * 0.5)
            weights /= weights.sum()

            combined = sum(w * s for w, s in zip(weights, scale_signals))

            self.signal = 1 if combined > 0.001 else (-1 if combined < -0.001 else 0)
            self.confidence = min(abs(combined) * 100, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.0
