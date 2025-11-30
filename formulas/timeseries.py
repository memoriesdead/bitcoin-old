"""
Time Series Formulas (IDs 31-60)
================================
ARIMA, GARCH variants, filters, and time series specific models.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, Any, Optional
from .base import BaseFormula, FormulaRegistry


@FormulaRegistry.register(31)
class ARModel(BaseFormula):
    """ID 31: Autoregressive Model AR(p)"""
    NAME = "ARModel"
    CATEGORY = "time_series"
    DESCRIPTION = "AR(1) to AR(5) autoregressive model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = kwargs.get('ar_order', 3)
        self.coefficients = np.zeros(self.order)

    def _compute(self):
        if len(self.returns) < self.order + 10:
            return

        returns = self._returns_array()

        # Fit AR model using Yule-Walker
        acf = np.correlate(returns - np.mean(returns), returns - np.mean(returns), mode='full')
        acf = acf[len(acf)//2:] / acf[len(acf)//2]

        # Levinson-Durbin recursion approximation
        r = acf[1:self.order+1]
        R = np.zeros((self.order, self.order))
        for i in range(self.order):
            for j in range(self.order):
                R[i, j] = acf[abs(i-j)]

        try:
            self.coefficients = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            self.coefficients = np.zeros(self.order)

        # Forecast
        forecast = np.sum(self.coefficients * returns[-self.order:][::-1])

        self.confidence = min(np.sum(np.abs(self.coefficients)), 1.0)
        self.signal = self._clip_signal(forecast * 100, threshold=0.05)


@FormulaRegistry.register(32)
class MAModel(BaseFormula):
    """ID 32: Moving Average Model MA(q)"""
    NAME = "MAModel"
    CATEGORY = "time_series"
    DESCRIPTION = "MA(1) to MA(3) moving average model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = kwargs.get('ma_order', 2)
        self.residuals = []

    def _compute(self):
        if len(self.returns) < 20:
            return

        returns = self._returns_array()
        mean = np.mean(returns)

        # Calculate residuals
        self.residuals = list(returns - mean)

        if len(self.residuals) >= self.order:
            # Simple MA estimation
            recent_residuals = self.residuals[-self.order:]
            ma_weights = np.array([0.5 ** (i+1) for i in range(self.order)])
            ma_weights = ma_weights / ma_weights.sum()

            forecast = mean + np.sum(ma_weights * recent_residuals)

            self.confidence = min(np.std(self.residuals[-20:]) * 10, 1.0)
            self.signal = self._clip_signal(forecast * 100, threshold=0.05)


@FormulaRegistry.register(33)
class ARMAModel(BaseFormula):
    """ID 33: ARMA(p,q) Combined Model"""
    NAME = "ARMAModel"
    CATEGORY = "time_series"
    DESCRIPTION = "Combined autoregressive moving average"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ar_order = kwargs.get('ar_order', 2)
        self.ma_order = kwargs.get('ma_order', 1)

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()
        mean = np.mean(returns)
        centered = returns - mean

        # Simple AR component
        ar_forecast = 0
        if self.ar_order > 0:
            ar_coef = np.corrcoef(centered[:-1], centered[1:])[0, 1]
            ar_forecast = ar_coef * centered[-1]

        # Simple MA component (innovation)
        residuals = centered[1:] - ar_forecast * centered[:-1]
        ma_term = np.mean(residuals[-self.ma_order:]) if len(residuals) >= self.ma_order else 0

        forecast = mean + ar_forecast + 0.3 * ma_term

        self.confidence = min(abs(ar_forecast) + abs(ma_term), 1.0)
        self.signal = self._clip_signal(forecast * 100, threshold=0.03)


@FormulaRegistry.register(34)
class ARCHModel(BaseFormula):
    """ID 34: ARCH(q) Autoregressive Conditional Heteroskedasticity"""
    NAME = "ARCHModel"
    CATEGORY = "time_series"
    DESCRIPTION = "Volatility clustering model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = kwargs.get('arch_order', 3)
        self.omega = 0.0001
        self.alpha = np.ones(self.order) / self.order * 0.3

    def _compute(self):
        if len(self.returns) < self.order + 10:
            return

        returns = self._returns_array()
        squared_returns = returns ** 2

        # ARCH variance forecast
        conditional_var = self.omega + np.sum(self.alpha * squared_returns[-self.order:][::-1])

        # Compare to historical
        hist_var = np.var(returns[-30:])

        vol_ratio = np.sqrt(conditional_var) / np.sqrt(hist_var) if hist_var > 0 else 1

        self.confidence = min(abs(vol_ratio - 1), 1.0)

        # High vol forecast - reduce exposure
        if vol_ratio > 1.5:
            self.signal = 0  # Stay out
        elif vol_ratio < 0.7:
            # Low vol - follow trend
            self.signal = 1 if returns[-1] > 0 else -1
        else:
            self.signal = 0


@FormulaRegistry.register(35)
class GARCHModel(BaseFormula):
    """ID 35: GARCH(1,1) Model"""
    NAME = "GARCHModel"
    CATEGORY = "time_series"
    DESCRIPTION = "Generalized ARCH with persistence"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.omega = kwargs.get('omega', 0.0001)
        self.alpha = kwargs.get('alpha', 0.1)
        self.beta = kwargs.get('beta', 0.85)
        self.sigma2 = self.omega / (1 - self.alpha - self.beta)

    def _compute(self):
        if len(self.returns) < 20:
            return

        returns = self._returns_array()

        # Update variance
        self.sigma2 = self.omega + self.alpha * returns[-1]**2 + self.beta * self.sigma2

        # Forecast volatility
        forecast_vol = np.sqrt(self.sigma2)
        hist_vol = np.std(returns[-20:])

        vol_ratio = forecast_vol / hist_vol if hist_vol > 0 else 1

        self.confidence = min(abs(vol_ratio - 1), 1.0)

        # Vol-based signal
        if vol_ratio > 1.3:
            # High vol - mean reversion more likely
            self.signal = -np.sign(returns[-1])
        elif vol_ratio < 0.8:
            # Low vol - trend more likely
            self.signal = np.sign(np.mean(returns[-5:]))
        else:
            self.signal = 0


@FormulaRegistry.register(36)
class EGARCHModel(BaseFormula):
    """ID 36: Exponential GARCH for asymmetry"""
    NAME = "EGARCHModel"
    CATEGORY = "time_series"
    DESCRIPTION = "Asymmetric volatility response"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.omega = kwargs.get('omega', -0.5)
        self.alpha = kwargs.get('alpha', 0.2)
        self.gamma = kwargs.get('gamma', -0.1)  # Asymmetry
        self.beta = kwargs.get('beta', 0.9)
        self.log_sigma2 = -5

    def _compute(self):
        if len(self.returns) < 20:
            return

        returns = self._returns_array()
        sigma = np.exp(self.log_sigma2 / 2)

        if sigma > 0:
            z = returns[-1] / sigma
        else:
            z = 0

        # EGARCH update
        self.log_sigma2 = (self.omega +
                          self.alpha * (abs(z) - np.sqrt(2/np.pi)) +
                          self.gamma * z +
                          self.beta * self.log_sigma2)

        new_sigma = np.exp(self.log_sigma2 / 2)
        hist_sigma = np.std(returns[-20:])

        # Asymmetry: negative returns increase vol more
        asymmetry = self.gamma * z

        self.confidence = min(abs(asymmetry), 1.0)

        if asymmetry < -0.1:
            # Negative return increased vol - expect continuation
            self.signal = -1
        elif asymmetry > 0.1:
            # Positive return increased vol less - bullish
            self.signal = 1
        else:
            self.signal = 0


@FormulaRegistry.register(37)
class GJRGARCHModel(BaseFormula):
    """ID 37: GJR-GARCH Threshold Model"""
    NAME = "GJRGARCHModel"
    CATEGORY = "time_series"
    DESCRIPTION = "Threshold GARCH for leverage effect"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.omega = kwargs.get('omega', 0.0001)
        self.alpha = kwargs.get('alpha', 0.05)
        self.gamma = kwargs.get('gamma', 0.1)  # Leverage
        self.beta = kwargs.get('beta', 0.85)
        self.sigma2 = 0.0001

    def _compute(self):
        if len(self.returns) < 20:
            return

        returns = self._returns_array()
        ret = returns[-1]

        # Indicator for negative returns
        I = 1 if ret < 0 else 0

        # GJR update
        self.sigma2 = self.omega + (self.alpha + self.gamma * I) * ret**2 + self.beta * self.sigma2

        leverage_effect = self.gamma * I * ret**2

        self.confidence = min(leverage_effect * 1000, 1.0)

        if leverage_effect > 0:
            # Leverage effect active - bearish
            self.signal = -1
        else:
            self.signal = 0


@FormulaRegistry.register(38)
class STARModel(BaseFormula):
    """ID 38: Smooth Transition AR Model"""
    NAME = "STARModel"
    CATEGORY = "time_series"
    DESCRIPTION = "Regime-switching AR with smooth transition"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gamma = kwargs.get('gamma', 5.0)  # Transition speed
        self.c = kwargs.get('c', 0.0)  # Threshold

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()

        # Transition function (logistic)
        G = 1 / (1 + np.exp(-self.gamma * (returns[-2] - self.c)))

        # Two regime AR(1) coefficients
        phi1 = np.corrcoef(returns[:-1][returns[:-1] < 0], returns[1:][returns[:-1] < 0])[0, 1] if np.sum(returns[:-1] < 0) > 5 else 0
        phi2 = np.corrcoef(returns[:-1][returns[:-1] >= 0], returns[1:][returns[:-1] >= 0])[0, 1] if np.sum(returns[:-1] >= 0) > 5 else 0

        # Weighted forecast
        forecast = (1 - G) * phi1 * returns[-1] + G * phi2 * returns[-1]

        self.confidence = abs(G - 0.5) * 2
        self.signal = self._clip_signal(forecast * 100, threshold=0.05)


@FormulaRegistry.register(39)
class TARModel(BaseFormula):
    """ID 39: Threshold AR Model"""
    NAME = "TARModel"
    CATEGORY = "time_series"
    DESCRIPTION = "Discrete regime-switching AR"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = kwargs.get('threshold', 0.0)

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()

        # Split by threshold
        below = returns[returns < self.threshold]
        above = returns[returns >= self.threshold]

        # Regime-specific AR(1)
        if len(below) > 5:
            phi_below = np.corrcoef(below[:-1], below[1:])[0, 1] if len(below) > 2 else 0
        else:
            phi_below = 0

        if len(above) > 5:
            phi_above = np.corrcoef(above[:-1], above[1:])[0, 1] if len(above) > 2 else 0
        else:
            phi_above = 0

        # Current regime
        if returns[-1] < self.threshold:
            forecast = phi_below * returns[-1]
            self.confidence = abs(phi_below)
        else:
            forecast = phi_above * returns[-1]
            self.confidence = abs(phi_above)

        self.signal = self._clip_signal(forecast * 100, threshold=0.03)


@FormulaRegistry.register(40)
class MarkovSwitching(BaseFormula):
    """ID 40: Markov Switching Model"""
    NAME = "MarkovSwitching"
    CATEGORY = "time_series"
    DESCRIPTION = "Hamilton-style regime switching"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p11 = 0.95  # Stay in state 1
        self.p22 = 0.90  # Stay in state 2
        self.prob_state1 = 0.5

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()

        # Two-state parameters
        mu1 = np.percentile(returns, 25)  # Low return state
        mu2 = np.percentile(returns, 75)  # High return state
        sigma = np.std(returns)

        # Likelihood of current return in each state
        if sigma > 0:
            l1 = np.exp(-0.5 * ((returns[-1] - mu1) / sigma) ** 2)
            l2 = np.exp(-0.5 * ((returns[-1] - mu2) / sigma) ** 2)
        else:
            l1 = l2 = 0.5

        # Update state probability
        pred_prob1 = self.p11 * self.prob_state1 + (1 - self.p22) * (1 - self.prob_state1)

        if l1 * pred_prob1 + l2 * (1 - pred_prob1) > 0:
            self.prob_state1 = (l1 * pred_prob1) / (l1 * pred_prob1 + l2 * (1 - pred_prob1))
        else:
            self.prob_state1 = 0.5

        self.confidence = abs(self.prob_state1 - 0.5) * 2

        if self.prob_state1 > 0.7:
            self.signal = -1  # In low state
        elif self.prob_state1 < 0.3:
            self.signal = 1  # In high state
        else:
            self.signal = 0


@FormulaRegistry.register(41)
class KalmanFilter(BaseFormula):
    """ID 41: Kalman Filter for state estimation"""
    NAME = "KalmanFilter"
    CATEGORY = "time_series"
    DESCRIPTION = "Optimal linear state estimation"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = 0
        self.variance = 1
        self.Q = kwargs.get('process_noise', 0.01)  # Process noise
        self.R = kwargs.get('measurement_noise', 0.1)  # Measurement noise

    def _compute(self):
        if len(self.returns) < 5:
            return

        returns = self._returns_array()
        measurement = returns[-1]

        # Predict
        pred_state = self.state
        pred_var = self.variance + self.Q

        # Update
        K = pred_var / (pred_var + self.R)  # Kalman gain
        self.state = pred_state + K * (measurement - pred_state)
        self.variance = (1 - K) * pred_var

        self.confidence = 1 - K  # High K = more trust in measurement

        self.signal = self._clip_signal(self.state * 100, threshold=0.05)


@FormulaRegistry.register(42)
class ParticleFilter(BaseFormula):
    """ID 42: Particle Filter for nonlinear state estimation"""
    NAME = "ParticleFilter"
    CATEGORY = "time_series"
    DESCRIPTION = "Sequential Monte Carlo estimation"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_particles = kwargs.get('n_particles', 100)
        self.particles = np.random.randn(self.n_particles) * 0.01
        self.weights = np.ones(self.n_particles) / self.n_particles

    def _compute(self):
        if len(self.returns) < 5:
            return

        returns = self._returns_array()
        measurement = returns[-1]

        # Predict (random walk transition)
        self.particles = self.particles + np.random.randn(self.n_particles) * 0.01

        # Update weights
        sigma = 0.02
        likelihoods = np.exp(-0.5 * ((measurement - self.particles) / sigma) ** 2)
        self.weights = likelihoods * self.weights
        self.weights = self.weights / (self.weights.sum() + 1e-10)

        # Resample if effective sample size too low
        ess = 1 / (self.weights ** 2).sum()
        if ess < self.n_particles / 2:
            indices = np.random.choice(self.n_particles, self.n_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Estimate
        state_estimate = np.sum(self.weights * self.particles)

        self.confidence = min(1 / np.var(self.particles) if np.var(self.particles) > 0 else 1, 1.0)
        self.signal = self._clip_signal(state_estimate * 100, threshold=0.02)


@FormulaRegistry.register(43)
class UnscentedKalman(BaseFormula):
    """ID 43: Unscented Kalman Filter"""
    NAME = "UnscentedKalman"
    CATEGORY = "time_series"
    DESCRIPTION = "Nonlinear Kalman using sigma points"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = 0
        self.variance = 0.01
        self.alpha = 0.001
        self.beta = 2
        self.kappa = 0

    def _compute(self):
        if len(self.returns) < 10:
            return

        returns = self._returns_array()
        measurement = returns[-1]

        # Simple approximation using sigma points
        sigma_points = np.array([
            self.state - np.sqrt(3 * self.variance),
            self.state,
            self.state + np.sqrt(3 * self.variance)
        ])

        # Weights
        w = np.array([1/6, 2/3, 1/6])

        # Predict (identity transition for simplicity)
        pred_mean = np.sum(w * sigma_points)
        pred_var = np.sum(w * (sigma_points - pred_mean)**2) + 0.001

        # Update
        K = pred_var / (pred_var + 0.01)
        self.state = pred_mean + K * (measurement - pred_mean)
        self.variance = (1 - K) * pred_var

        self.confidence = 1 - K
        self.signal = self._clip_signal(self.state * 100, threshold=0.03)


@FormulaRegistry.register(44)
class ExponentialFilter(BaseFormula):
    """ID 44: Exponential Smoothing Filter"""
    NAME = "ExponentialFilter"
    CATEGORY = "time_series"
    DESCRIPTION = "Simple exponential smoothing"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = kwargs.get('alpha', 0.2)
        self.smoothed = None

    def _compute(self):
        prices = self._prices_array()
        if len(prices) < 5:
            return

        current = prices[-1]

        if self.smoothed is None:
            self.smoothed = current
        else:
            self.smoothed = self.alpha * current + (1 - self.alpha) * self.smoothed

        deviation = (current - self.smoothed) / self.smoothed if self.smoothed != 0 else 0

        self.confidence = min(abs(deviation) * 10, 1.0)
        self.signal = self._clip_signal(deviation * 100, threshold=0.1)


@FormulaRegistry.register(45)
class DoubleExponential(BaseFormula):
    """ID 45: Double Exponential Smoothing (Holt)"""
    NAME = "DoubleExponential"
    CATEGORY = "time_series"
    DESCRIPTION = "Trend-capturing exponential smoothing"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = kwargs.get('alpha', 0.3)
        self.beta = kwargs.get('beta', 0.1)
        self.level = None
        self.trend = 0

    def _compute(self):
        prices = self._prices_array()
        if len(prices) < 5:
            return

        current = prices[-1]

        if self.level is None:
            self.level = current
            self.trend = 0
        else:
            prev_level = self.level
            self.level = self.alpha * current + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend

        # Forecast
        forecast = self.level + self.trend

        self.confidence = min(abs(self.trend) / np.std(prices[-20:]) if len(prices) >= 20 else 0.5, 1.0)
        self.signal = self._clip_signal(self.trend * 1000, threshold=1)


@FormulaRegistry.register(46)
class TripleExponential(BaseFormula):
    """ID 46: Triple Exponential (Holt-Winters)"""
    NAME = "TripleExponential"
    CATEGORY = "time_series"
    DESCRIPTION = "Trend and seasonality smoothing"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = kwargs.get('alpha', 0.3)
        self.beta = kwargs.get('beta', 0.1)
        self.gamma = kwargs.get('gamma', 0.1)
        self.season_length = kwargs.get('season_length', 20)
        self.level = None
        self.trend = 0
        self.seasonals = []

    def _compute(self):
        prices = self._prices_array()
        if len(prices) < self.season_length + 5:
            return

        current = prices[-1]

        if self.level is None:
            self.level = np.mean(prices[:self.season_length])
            self.trend = (np.mean(prices[self.season_length:2*self.season_length]) -
                         np.mean(prices[:self.season_length])) / self.season_length if len(prices) >= 2*self.season_length else 0
            self.seasonals = list(prices[:self.season_length] / self.level)

        if len(self.seasonals) >= self.season_length:
            idx = len(prices) % self.season_length
            season = self.seasonals[idx]

            prev_level = self.level
            self.level = self.alpha * (current / season) + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
            self.seasonals[idx] = self.gamma * (current / self.level) + (1 - self.gamma) * season

        self.confidence = min(abs(self.trend) * 100, 1.0)
        self.signal = self._clip_signal(self.trend * 1000, threshold=0.5)


@FormulaRegistry.register(47)
class DickeyFullerTest(BaseFormula):
    """ID 47: Augmented Dickey-Fuller Unit Root Test"""
    NAME = "DickeyFullerTest"
    CATEGORY = "time_series"
    DESCRIPTION = "Test for stationarity"

    def _compute(self):
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # Simple ADF approximation
        # y_t = phi*y_{t-1} + e_t
        y = returns[1:]
        y_lag = returns[:-1]

        if np.var(y_lag) > 0:
            phi = np.cov(y, y_lag)[0, 1] / np.var(y_lag)
        else:
            phi = 0

        # ADF test statistic approximation
        se = np.std(y - phi * y_lag) / np.sqrt(np.sum(y_lag**2))
        if se > 0:
            adf_stat = (phi - 1) / se
        else:
            adf_stat = 0

        # Critical value approximately -2.86 at 5%
        is_stationary = adf_stat < -2.86

        self.confidence = min(abs(adf_stat) / 3, 1.0)

        if is_stationary:
            # Mean reverting
            self.signal = -np.sign(returns[-1])
        else:
            # Unit root - momentum
            self.signal = np.sign(returns[-1])


@FormulaRegistry.register(48)
class PhillipsPerronTest(BaseFormula):
    """ID 48: Phillips-Perron Unit Root Test"""
    NAME = "PhillipsPerronTest"
    CATEGORY = "time_series"
    DESCRIPTION = "Robust unit root test"

    def _compute(self):
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # Newey-West corrected test
        y = returns[1:]
        y_lag = returns[:-1]

        if np.var(y_lag) > 0:
            phi = np.cov(y, y_lag)[0, 1] / np.var(y_lag)
            residuals = y - phi * y_lag
        else:
            phi = 0
            residuals = y

        # Long-run variance approximation
        gamma0 = np.var(residuals)
        gamma1 = np.cov(residuals[:-1], residuals[1:])[0, 1] if len(residuals) > 1 else 0
        omega = gamma0 + 2 * 0.5 * gamma1  # Simplified Newey-West

        # PP statistic
        if omega > 0 and np.var(y_lag) > 0:
            pp_stat = (phi - 1) * np.sqrt(np.sum(y_lag**2)) / np.sqrt(omega)
        else:
            pp_stat = 0

        is_stationary = pp_stat < -2.86

        self.confidence = min(abs(pp_stat) / 3, 1.0)

        if is_stationary:
            self.signal = -np.sign(returns[-1])
        else:
            self.signal = np.sign(returns[-1])


@FormulaRegistry.register(49)
class KPSSTest(BaseFormula):
    """ID 49: KPSS Stationarity Test"""
    NAME = "KPSSTest"
    CATEGORY = "time_series"
    DESCRIPTION = "Test for trend stationarity"

    def _compute(self):
        if len(self.returns) < 50:
            return

        returns = self._returns_array()
        n = len(returns)

        # Partial sums of demeaned series
        demeaned = returns - np.mean(returns)
        S = np.cumsum(demeaned)

        # Estimate variance
        sigma2 = np.var(demeaned)

        # KPSS statistic
        if sigma2 > 0:
            kpss_stat = np.sum(S**2) / (n**2 * sigma2)
        else:
            kpss_stat = 0

        # Critical value approximately 0.463 at 5%
        is_stationary = kpss_stat < 0.463

        self.confidence = min(kpss_stat / 0.5, 1.0)

        if is_stationary:
            # Trend stationary - mean reversion
            self.signal = -np.sign(returns[-1])
        else:
            # Not trend stationary
            self.signal = np.sign(np.mean(returns[-10:]))


@FormulaRegistry.register(50)
class SpectralDensity(BaseFormula):
    """ID 50: Spectral Density Estimation"""
    NAME = "SpectralDensity"
    CATEGORY = "time_series"
    DESCRIPTION = "Frequency domain analysis"

    def _compute(self):
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # FFT-based spectral density
        fft = np.fft.fft(returns)
        psd = np.abs(fft[:len(fft)//2])**2

        # Find dominant frequency
        dominant_freq = np.argmax(psd[1:]) + 1  # Skip DC component
        period = len(returns) / dominant_freq if dominant_freq > 0 else len(returns)

        # Phase of dominant frequency
        phase = np.angle(fft[dominant_freq])

        # Signal based on phase
        self.confidence = psd[dominant_freq] / np.sum(psd)

        if phase > 0:
            self.signal = 1
        elif phase < 0:
            self.signal = -1
        else:
            self.signal = 0


@FormulaRegistry.register(51)
class BandpassFilter(BaseFormula):
    """ID 51: Bandpass Filter for cycle extraction"""
    NAME = "BandpassFilter"
    CATEGORY = "time_series"
    DESCRIPTION = "Extract specific frequency band"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.low_period = kwargs.get('low_period', 10)
        self.high_period = kwargs.get('high_period', 30)

    def _compute(self):
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # Simple bandpass using difference of EMAs
        fast_ema = self._ema(returns, self.low_period)
        slow_ema = self._ema(returns, self.high_period)

        cycle = fast_ema - slow_ema

        self.confidence = min(abs(cycle[-1]) / np.std(cycle), 1.0)
        self.signal = self._clip_signal(cycle[-1] * 100, threshold=0.1)


@FormulaRegistry.register(52)
class HodrickPrescott(BaseFormula):
    """ID 52: Hodrick-Prescott Filter"""
    NAME = "HodrickPrescott"
    CATEGORY = "time_series"
    DESCRIPTION = "Trend-cycle decomposition"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lamb = kwargs.get('lambda', 1600)  # Standard for daily data

    def _compute(self):
        if len(self.prices) < 30:
            return

        prices = self._prices_array()
        n = len(prices)

        # HP filter approximation using double-sided EMA
        alpha = 2 / (1 + np.sqrt(1 + 4*self.lamb))

        # Forward EMA
        forward = self._ema(prices, int(1/alpha))

        # Backward EMA
        backward = self._ema(prices[::-1], int(1/alpha))[::-1]

        trend = (forward + backward) / 2
        cycle = prices - trend

        self.confidence = min(abs(cycle[-1]) / np.std(cycle), 1.0)
        self.signal = self._clip_signal(-cycle[-1] / np.std(cycle), threshold=1.5)


@FormulaRegistry.register(53)
class ChristianFitzgerald(BaseFormula):
    """ID 53: Christiano-Fitzgerald Bandpass Filter"""
    NAME = "ChristianFitzgerald"
    CATEGORY = "time_series"
    DESCRIPTION = "Asymmetric bandpass filter"

    def _compute(self):
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # CF filter approximation
        # Use weighted moving average with sinc weights
        periods = [10, 30]  # Passband
        weights = []

        for j in range(-10, 11):
            if j == 0:
                w = 2 * (1/periods[0] - 1/periods[1])
            else:
                w = (np.sin(2*np.pi*j/periods[0]) - np.sin(2*np.pi*j/periods[1])) / (np.pi*j)
            weights.append(w)

        weights = np.array(weights)
        weights = weights / weights.sum()

        if len(returns) >= len(weights):
            cycle = np.convolve(returns, weights, mode='valid')
            self.confidence = min(abs(cycle[-1]) / np.std(cycle), 1.0)
            self.signal = self._clip_signal(cycle[-1] * 100, threshold=0.05)


@FormulaRegistry.register(54)
class BeveridgeNelson(BaseFormula):
    """ID 54: Beveridge-Nelson Decomposition"""
    NAME = "BeveridgeNelson"
    CATEGORY = "time_series"
    DESCRIPTION = "Permanent vs transitory component"

    def _compute(self):
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # BN decomposition approximation
        # Permanent component = long-run mean
        long_run_mean = np.mean(returns)

        # Transitory = deviation from long-run
        transitory = returns[-1] - long_run_mean

        # Estimate persistence
        persistence = np.corrcoef(returns[:-1], returns[1:])[0, 1]

        # Permanent shock
        permanent = long_run_mean / (1 - persistence) if persistence < 1 else long_run_mean

        self.confidence = abs(persistence)

        if transitory > np.std(returns):
            self.signal = -1  # Large transitory - expect reversion
        elif transitory < -np.std(returns):
            self.signal = 1
        else:
            self.signal = 0


@FormulaRegistry.register(55)
class WaveletDecomposition(BaseFormula):
    """ID 55: Wavelet Multi-Resolution Analysis"""
    NAME = "WaveletDecomposition"
    CATEGORY = "time_series"
    DESCRIPTION = "Time-frequency decomposition"

    def _compute(self):
        if len(self.prices) < 64:
            return

        prices = self._prices_array()[-64:]

        # Simple Haar wavelet approximation
        levels = 4
        coeffs = []
        signal_data = prices.copy()

        for _ in range(levels):
            n = len(signal_data)
            if n < 2:
                break
            # Approximation and detail
            approx = (signal_data[::2] + signal_data[1::2]) / 2
            detail = (signal_data[::2] - signal_data[1::2]) / 2
            coeffs.append(detail)
            signal_data = approx

        # Use detail coefficients for signal
        if len(coeffs) > 0:
            high_freq = coeffs[0][-1] if len(coeffs[0]) > 0 else 0
            self.confidence = min(abs(high_freq) / np.std(prices), 1.0)
            self.signal = self._clip_signal(-high_freq * 10, threshold=0.5)


@FormulaRegistry.register(56)
class EMDDecomposition(BaseFormula):
    """ID 56: Empirical Mode Decomposition"""
    NAME = "EMDDecomposition"
    CATEGORY = "time_series"
    DESCRIPTION = "Data-driven decomposition"

    def _compute(self):
        if len(self.prices) < 50:
            return

        prices = self._prices_array()

        # Simplified EMD - extract one IMF
        # Find local extrema
        maxima_idx = []
        minima_idx = []

        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                maxima_idx.append(i)
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                minima_idx.append(i)

        if len(maxima_idx) < 2 or len(minima_idx) < 2:
            self.signal = 0
            return

        # Interpolate envelopes
        upper_env = np.interp(range(len(prices)), maxima_idx, prices[maxima_idx])
        lower_env = np.interp(range(len(prices)), minima_idx, prices[minima_idx])

        mean_env = (upper_env + lower_env) / 2
        imf = prices - mean_env

        self.confidence = min(abs(imf[-1]) / np.std(imf), 1.0)
        self.signal = self._clip_signal(-imf[-1] / np.std(imf), threshold=1.0)


@FormulaRegistry.register(57)
class SingularSpectrum(BaseFormula):
    """ID 57: Singular Spectrum Analysis"""
    NAME = "SingularSpectrum"
    CATEGORY = "time_series"
    DESCRIPTION = "SSA for trend extraction"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.window = kwargs.get('window', 20)

    def _compute(self):
        if len(self.prices) < self.window * 2:
            return

        prices = self._prices_array()
        N = len(prices)
        L = self.window
        K = N - L + 1

        # Trajectory matrix
        X = np.zeros((L, K))
        for i in range(K):
            X[:, i] = prices[i:i+L]

        # SVD
        try:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            self.signal = 0
            return

        # Reconstruct using first component (trend)
        trend_matrix = s[0] * np.outer(U[:, 0], Vt[0, :])

        # Average anti-diagonals for reconstruction
        trend = np.zeros(N)
        counts = np.zeros(N)

        for i in range(L):
            for j in range(K):
                idx = i + j
                trend[idx] += trend_matrix[i, j]
                counts[idx] += 1

        trend = trend / counts

        deviation = (prices[-1] - trend[-1]) / np.std(prices - trend)

        self.confidence = min(abs(deviation) / 2, 1.0)
        self.signal = self._clip_signal(-deviation, threshold=1.5)


@FormulaRegistry.register(58)
class VARModel(BaseFormula):
    """ID 58: Vector Autoregression (single variable approx)"""
    NAME = "VARModel"
    CATEGORY = "time_series"
    DESCRIPTION = "VAR(1) for return and volatility"

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()
        vol = np.abs(returns)

        # VAR(1) coefficients
        y1 = returns[1:]
        y2 = vol[1:]
        x1 = returns[:-1]
        x2 = vol[:-1]

        # Simple OLS approximation
        A11 = np.corrcoef(x1, y1)[0, 1] if np.std(x1) > 0 else 0
        A12 = np.corrcoef(x2, y1)[0, 1] if np.std(x2) > 0 else 0
        A21 = np.corrcoef(x1, y2)[0, 1] if np.std(x1) > 0 else 0
        A22 = np.corrcoef(x2, y2)[0, 1] if np.std(x2) > 0 else 0

        # Forecast
        ret_forecast = A11 * returns[-1] + A12 * vol[-1]
        vol_forecast = A21 * returns[-1] + A22 * vol[-1]

        self.confidence = min((abs(A11) + abs(A12)) / 2, 1.0)
        self.signal = self._clip_signal(ret_forecast * 100, threshold=0.05)


@FormulaRegistry.register(59)
class StateSpaceModel(BaseFormula):
    """ID 59: General State Space Model"""
    NAME = "StateSpaceModel"
    CATEGORY = "time_series"
    DESCRIPTION = "Hidden state dynamics"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = np.array([0.0, 0.0])  # [level, trend]
        self.P = np.eye(2) * 0.1

    def _compute(self):
        if len(self.returns) < 10:
            return

        returns = self._returns_array()
        y = returns[-1]

        # State transition
        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * 0.001
        R = np.array([[0.01]])

        # Predict
        x_pred = F @ self.state
        P_pred = F @ self.P @ F.T + Q

        # Update
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T / S
        self.state = x_pred + K.flatten() * (y - H @ x_pred)
        self.P = (np.eye(2) - K @ H) @ P_pred

        level, trend = self.state

        self.confidence = min(abs(trend) * 100, 1.0)
        self.signal = self._clip_signal(trend * 1000, threshold=0.5)


@FormulaRegistry.register(60)
class DynamicFactorModel(BaseFormula):
    """ID 60: Dynamic Factor Model"""
    NAME = "DynamicFactorModel"
    CATEGORY = "time_series"
    DESCRIPTION = "Common factor extraction"

    def _compute(self):
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # Create pseudo-observable (returns at different lags)
        X = np.column_stack([
            returns[3:],
            returns[2:-1],
            returns[1:-2],
            returns[:-3]
        ])

        # PCA for factor
        X_centered = X - X.mean(axis=0)
        cov = X_centered.T @ X_centered / len(X_centered)

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            self.signal = 0
            return

        # First principal component (factor)
        factor_loadings = eigenvectors[:, -1]
        factor = X_centered @ factor_loadings

        # Factor signal
        factor_z = (factor[-1] - np.mean(factor)) / np.std(factor)

        self.confidence = eigenvalues[-1] / eigenvalues.sum()
        self.signal = self._clip_signal(factor_z, threshold=1.0)


# Export all classes
__all__ = [
    'ARModel', 'MAModel', 'ARMAModel', 'ARCHModel', 'GARCHModel',
    'EGARCHModel', 'GJRGARCHModel', 'STARModel', 'TARModel', 'MarkovSwitching',
    'KalmanFilter', 'ParticleFilter', 'UnscentedKalman', 'ExponentialFilter',
    'DoubleExponential', 'TripleExponential', 'DickeyFullerTest', 'PhillipsPerronTest',
    'KPSSTest', 'SpectralDensity', 'BandpassFilter', 'HodrickPrescott',
    'ChristianFitzgerald', 'BeveridgeNelson', 'WaveletDecomposition',
    'EMDDecomposition', 'SingularSpectrum', 'VARModel', 'StateSpaceModel',
    'DynamicFactorModel'
]
