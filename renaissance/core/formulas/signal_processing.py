"""
Signal Processing Formulas (IDs 191-210)
========================================
FFT, Wavelet, Kalman filters, and spectral analysis.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from .base import BaseFormula, FormulaRegistry


# =============================================================================
# FOURIER AND SPECTRAL ANALYSIS (191-200)
# =============================================================================

@FormulaRegistry.register(191)
class FFTCycleDetector(BaseFormula):
    """ID 191: FFT for cycle detection"""

    CATEGORY = "signal_processing"
    NAME = "FFTCycleDetector"
    DESCRIPTION = "Detect dominant frequencies in price"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.dominant_period = 0
        self.dominant_amplitude = 0.0
        self.phase = 0.0

    def _compute(self) -> None:
        if len(self.prices) < 64:
            return
        prices = self._prices_array()
        n = min(64, len(prices))
        data = prices[-n:]
        data = data - np.mean(data)
        fft_result = np.fft.fft(data)
        freqs = np.fft.fftfreq(n)
        magnitudes = np.abs(fft_result[1:n//2])
        if len(magnitudes) == 0:
            return
        dominant_idx = np.argmax(magnitudes)
        self.dominant_amplitude = magnitudes[dominant_idx] / n
        if freqs[dominant_idx + 1] != 0:
            self.dominant_period = abs(1 / freqs[dominant_idx + 1])
        else:
            self.dominant_period = n
        self.phase = np.angle(fft_result[dominant_idx + 1])
        if self.dominant_amplitude > np.std(prices) * 0.5:
            cycle_position = (len(prices) % self.dominant_period) / self.dominant_period
            if 0.1 < cycle_position < 0.4:
                self.signal = 1
            elif 0.6 < cycle_position < 0.9:
                self.signal = -1
            else:
                self.signal = 0
            self.confidence = min(self.dominant_amplitude / np.std(prices), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(192)
class SpectralDensity(BaseFormula):
    """ID 192: Power Spectral Density"""

    CATEGORY = "signal_processing"
    NAME = "SpectralDensity"
    DESCRIPTION = "Power distribution across frequencies"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.low_freq_power = 0.0
        self.high_freq_power = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 64:
            return
        returns = self._returns_array()
        n = min(64, len(returns))
        data = returns[-n:]
        fft_result = np.fft.fft(data)
        psd = np.abs(fft_result[:n//2])**2 / n
        mid_point = n // 4
        self.low_freq_power = np.sum(psd[1:mid_point])
        self.high_freq_power = np.sum(psd[mid_point:])
        total_power = self.low_freq_power + self.high_freq_power + 1e-10
        low_ratio = self.low_freq_power / total_power
        if low_ratio > 0.7:
            momentum = np.mean(returns[-10:])
            self.signal = 1 if momentum > 0 else -1
            self.confidence = low_ratio
        elif low_ratio < 0.3:
            z = returns[-1] / (np.std(returns) + 1e-10)
            self.signal = -1 if z > 0 else 1
            self.confidence = 1 - low_ratio
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(193)
class HilbertTransform(BaseFormula):
    """ID 193: Hilbert Transform for instantaneous phase"""

    CATEGORY = "signal_processing"
    NAME = "HilbertTransform"
    DESCRIPTION = "Extract amplitude and phase envelope"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.amplitude = 0.0
        self.phase = 0.0
        self.inst_freq = 0.0

    def _compute(self) -> None:
        if len(self.prices) < 32:
            return
        prices = self._prices_array()
        n = len(prices)
        data = prices - np.mean(prices)
        fft_result = np.fft.fft(data)
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = 1
            h[1:n//2] = 2
            h[n//2] = 1
        else:
            h[0] = 1
            h[1:(n+1)//2] = 2
        analytic = np.fft.ifft(fft_result * h)
        self.amplitude = np.abs(analytic[-1])
        self.phase = np.angle(analytic[-1])
        if len(analytic) > 1:
            phase_diff = np.angle(analytic[-1]) - np.angle(analytic[-2])
            if phase_diff > np.pi:
                phase_diff -= 2 * np.pi
            elif phase_diff < -np.pi:
                phase_diff += 2 * np.pi
            self.inst_freq = phase_diff / (2 * np.pi)
        if -np.pi/4 < self.phase < np.pi/4:
            self.signal = 1
        elif np.pi*3/4 < self.phase or self.phase < -np.pi*3/4:
            self.signal = -1
        else:
            self.signal = 0
        self.confidence = min(self.amplitude / (np.std(prices) + 1e-10) / 3, 1.0)


@FormulaRegistry.register(194)
class BandpassFilter(BaseFormula):
    """ID 194: Bandpass Filter for specific frequencies"""

    CATEGORY = "signal_processing"
    NAME = "BandpassFilter"
    DESCRIPTION = "Extract signals in frequency band"

    def __init__(self, lookback: int = 100, low_period: int = 10,
                 high_period: int = 40, **kwargs):
        super().__init__(lookback, **kwargs)
        self.low_period = low_period
        self.high_period = high_period
        self.filtered_signal = 0.0

    def _compute(self) -> None:
        if len(self.prices) < max(self.high_period * 2, 32):
            return
        prices = self._prices_array()
        n = len(prices)
        data = prices - np.mean(prices)
        fft_result = np.fft.fft(data)
        freqs = np.fft.fftfreq(n)
        low_freq = 1.0 / self.high_period
        high_freq = 1.0 / self.low_period
        mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
        filtered_fft = fft_result * mask
        filtered = np.real(np.fft.ifft(filtered_fft))
        self.filtered_signal = filtered[-1]
        signal_std = np.std(filtered)
        z = self.filtered_signal / (signal_std + 1e-10)
        if z > 1.5:
            self.signal = -1
            self.confidence = min(abs(z) / 3, 1.0)
        elif z < -1.5:
            self.signal = 1
            self.confidence = min(abs(z) / 3, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(195)
class LowpassFilter(BaseFormula):
    """ID 195: Lowpass Filter for trend extraction"""

    CATEGORY = "signal_processing"
    NAME = "LowpassFilter"
    DESCRIPTION = "Extract low-frequency trend"

    def __init__(self, lookback: int = 100, cutoff_period: int = 20, **kwargs):
        super().__init__(lookback, **kwargs)
        self.cutoff_period = cutoff_period
        self.trend = 0.0

    def _compute(self) -> None:
        if len(self.prices) < 32:
            return
        prices = self._prices_array()
        n = len(prices)
        fft_result = np.fft.fft(prices)
        freqs = np.fft.fftfreq(n)
        cutoff_freq = 1.0 / self.cutoff_period
        mask = np.abs(freqs) <= cutoff_freq
        filtered_fft = fft_result * mask
        filtered = np.real(np.fft.ifft(filtered_fft))
        self.trend = filtered[-1]
        trend_direction = filtered[-1] - filtered[-2] if len(filtered) > 1 else 0
        price_vs_trend = prices[-1] - self.trend
        if trend_direction > 0 and price_vs_trend > 0:
            self.signal = 1
            self.confidence = 0.7
        elif trend_direction < 0 and price_vs_trend < 0:
            self.signal = -1
            self.confidence = 0.7
        elif abs(price_vs_trend) > np.std(prices) * 2:
            self.signal = -1 if price_vs_trend > 0 else 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(196)
class HighpassFilter(BaseFormula):
    """ID 196: Highpass Filter for noise"""

    CATEGORY = "signal_processing"
    NAME = "HighpassFilter"
    DESCRIPTION = "Extract high-frequency components"

    def __init__(self, lookback: int = 100, cutoff_period: int = 10, **kwargs):
        super().__init__(lookback, **kwargs)
        self.cutoff_period = cutoff_period
        self.noise = 0.0

    def _compute(self) -> None:
        if len(self.prices) < 32:
            return
        prices = self._prices_array()
        n = len(prices)
        fft_result = np.fft.fft(prices)
        freqs = np.fft.fftfreq(n)
        cutoff_freq = 1.0 / self.cutoff_period
        mask = np.abs(freqs) > cutoff_freq
        filtered_fft = fft_result * mask
        filtered = np.real(np.fft.ifft(filtered_fft))
        self.noise = filtered[-1]
        noise_power = np.std(filtered)
        total_power = np.std(prices)
        noise_ratio = noise_power / (total_power + 1e-10)
        if noise_ratio > 0.5:
            self.signal = 0
            self.confidence = 0.3
        else:
            momentum = np.mean(self._returns_array()[-5:]) if len(self.returns) >= 5 else 0
            self.signal = 1 if momentum > 0 else (-1 if momentum < 0 else 0)
            self.confidence = 1 - noise_ratio


@FormulaRegistry.register(197)
class WaveletDecomposition(BaseFormula):
    """ID 197: Haar Wavelet Decomposition"""

    CATEGORY = "signal_processing"
    NAME = "WaveletDecomposition"
    DESCRIPTION = "Multi-scale wavelet analysis"

    def __init__(self, lookback: int = 100, levels: int = 3, **kwargs):
        super().__init__(lookback, **kwargs)
        self.levels = levels
        self.approximation = 0.0
        self.details = []

    def _haar_decompose(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(data) // 2
        approx = np.zeros(n)
        detail = np.zeros(n)
        for i in range(n):
            approx[i] = (data[2*i] + data[2*i + 1]) / np.sqrt(2)
            detail[i] = (data[2*i] - data[2*i + 1]) / np.sqrt(2)
        return approx, detail

    def _compute(self) -> None:
        if len(self.prices) < 2**self.levels:
            return
        prices = self._prices_array()
        n = 2**int(np.log2(len(prices)))
        data = prices[-n:]
        self.details = []
        current = data.copy()
        for _ in range(self.levels):
            if len(current) < 2:
                break
            approx, detail = self._haar_decompose(current)
            self.details.append(detail)
            current = approx
        self.approximation = current[-1] if len(current) > 0 else prices[-1]
        if len(self.details) > 0:
            high_freq_energy = np.sum([np.sum(d**2) for d in self.details[:2]])
            low_freq_energy = np.sum(current**2) if len(current) > 0 else 1
            energy_ratio = high_freq_energy / (low_freq_energy + 1e-10)
            if energy_ratio > 2:
                self.signal = 0
                self.confidence = 0.3
            else:
                trend = self.approximation - np.mean(current) if len(current) > 1 else 0
                self.signal = 1 if trend > 0 else (-1 if trend < 0 else 0)
                self.confidence = 1 / (1 + energy_ratio)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(198)
class MorletWavelet(BaseFormula):
    """ID 198: Morlet Wavelet Transform"""

    CATEGORY = "signal_processing"
    NAME = "MorletWavelet"
    DESCRIPTION = "Complex wavelet for time-frequency"

    def __init__(self, lookback: int = 100, omega0: float = 6.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.omega0 = omega0
        self.scales = [4, 8, 16, 32]
        self.dominant_scale = 0

    def _morlet(self, t: np.ndarray, scale: float) -> np.ndarray:
        normalized_t = t / scale
        wavelet = np.exp(1j * self.omega0 * normalized_t) * np.exp(-0.5 * normalized_t**2)
        return wavelet / np.sqrt(scale)

    def _compute(self) -> None:
        if len(self.prices) < 64:
            return
        prices = self._prices_array()
        n = len(prices)
        t = np.arange(n) - n // 2
        data = prices - np.mean(prices)
        max_power = 0
        self.dominant_scale = self.scales[0]
        for scale in self.scales:
            wavelet = self._morlet(t, scale)
            coef = np.abs(np.sum(data * wavelet))
            if coef > max_power:
                max_power = coef
                self.dominant_scale = scale
        dominant_wavelet = self._morlet(t, self.dominant_scale)
        coef = np.sum(data * dominant_wavelet)
        phase = np.angle(coef)
        if -np.pi/3 < phase < np.pi/3:
            self.signal = 1
        elif abs(phase) > 2*np.pi/3:
            self.signal = -1
        else:
            self.signal = 0
        self.confidence = min(max_power / (np.std(prices) * np.sqrt(n) + 1e-10), 1.0)


@FormulaRegistry.register(199)
class EMDDecomposition(BaseFormula):
    """ID 199: Empirical Mode Decomposition (simplified)"""

    CATEGORY = "signal_processing"
    NAME = "EMDDecomposition"
    DESCRIPTION = "Data-driven signal decomposition"

    def __init__(self, lookback: int = 100, n_imfs: int = 3, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_imfs = n_imfs
        self.residual = 0.0

    def _find_extrema(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        maxima_idx = []
        minima_idx = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                maxima_idx.append(i)
            elif data[i] < data[i-1] and data[i] < data[i+1]:
                minima_idx.append(i)
        return np.array(maxima_idx), np.array(minima_idx)

    def _compute(self) -> None:
        if len(self.prices) < 30:
            return
        prices = self._prices_array()
        residual = prices.copy()
        imfs = []
        for _ in range(self.n_imfs):
            max_idx, min_idx = self._find_extrema(residual)
            if len(max_idx) < 2 or len(min_idx) < 2:
                break
            upper_env = np.interp(np.arange(len(residual)), max_idx, residual[max_idx])
            lower_env = np.interp(np.arange(len(residual)), min_idx, residual[min_idx])
            mean_env = (upper_env + lower_env) / 2
            imf = residual - mean_env
            imfs.append(imf)
            residual = mean_env
        self.residual = residual[-1] if len(residual) > 0 else prices[-1]
        if len(imfs) > 0:
            high_freq_energy = np.sum(imfs[0]**2)
            total_energy = np.sum(prices**2)
            noise_ratio = high_freq_energy / (total_energy + 1e-10)
            if noise_ratio > 0.3:
                self.signal = 0
                self.confidence = 0.3
            else:
                trend = self.residual - np.mean(residual)
                self.signal = 1 if trend > 0 else (-1 if trend < 0 else 0)
                self.confidence = 1 - noise_ratio
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(200)
class SingularSpectrum(BaseFormula):
    """ID 200: Singular Spectrum Analysis"""

    CATEGORY = "signal_processing"
    NAME = "SingularSpectrum"
    DESCRIPTION = "SSA for trend and periodicity"

    def __init__(self, lookback: int = 100, window: int = 20, n_components: int = 3, **kwargs):
        super().__init__(lookback, **kwargs)
        self.window = window
        self.n_components = n_components
        self.trend = 0.0

    def _compute(self) -> None:
        if len(self.prices) < self.window * 2:
            return
        prices = self._prices_array()
        n = len(prices)
        L = self.window
        K = n - L + 1
        X = np.zeros((L, K))
        for i in range(K):
            X[:, i] = prices[i:i+L]
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            reconstructed = np.zeros(n)
            weights = np.zeros(n)
            for comp in range(min(self.n_components, len(S))):
                X_comp = S[comp] * np.outer(U[:, comp], Vt[comp, :])
                for i in range(K):
                    reconstructed[i:i+L] += X_comp[:, i]
                    weights[i:i+L] += 1
            reconstructed /= (weights + 1e-10)
            self.trend = reconstructed[-1]
            trend_direction = reconstructed[-1] - reconstructed[-2] if len(reconstructed) > 1 else 0
            if trend_direction > np.std(prices) * 0.1:
                self.signal = 1
                self.confidence = min(abs(trend_direction) / np.std(prices), 1.0)
            elif trend_direction < -np.std(prices) * 0.1:
                self.signal = -1
                self.confidence = min(abs(trend_direction) / np.std(prices), 1.0)
            else:
                self.signal = 0
                self.confidence = 0.4
        except:
            self.signal = 0
            self.confidence = 0.3


# =============================================================================
# KALMAN AND ADAPTIVE FILTERS (201-210)
# =============================================================================

@FormulaRegistry.register(201)
class KalmanFilter(BaseFormula):
    """ID 201: Kalman Filter for state estimation"""

    CATEGORY = "signal_processing"
    NAME = "KalmanFilter"
    DESCRIPTION = "Optimal linear state estimator"

    def __init__(self, lookback: int = 100, Q: float = 0.001, R: float = 0.1, **kwargs):
        super().__init__(lookback, **kwargs)
        self.Q = Q
        self.R = R
        self.state = 0.0
        self.P = 1.0
        self.K = 0.0

    def _compute(self) -> None:
        if len(self.prices) < 2:
            return
        measurement = self.prices[-1]
        pred_state = self.state
        pred_P = self.P + self.Q
        self.K = pred_P / (pred_P + self.R)
        innovation = measurement - pred_state
        self.state = pred_state + self.K * innovation
        self.P = (1 - self.K) * pred_P
        innovation_std = np.sqrt(pred_P + self.R)
        z = innovation / (innovation_std + 1e-10)
        if z > 2:
            self.signal = 1
            self.confidence = min(abs(z) / 4, 1.0)
        elif z < -2:
            self.signal = -1
            self.confidence = min(abs(z) / 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(202)
class ExtendedKalman(BaseFormula):
    """ID 202: Extended Kalman Filter (linearized)"""

    CATEGORY = "signal_processing"
    NAME = "ExtendedKalman"
    DESCRIPTION = "EKF for nonlinear dynamics"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.state = np.array([0.0, 0.0])
        self.P = np.eye(2) * 1.0
        self.Q = np.eye(2) * 0.001
        self.R = 0.1

    def _compute(self) -> None:
        if len(self.prices) < 2:
            return
        measurement = self.prices[-1]
        F = np.array([[1, 1], [0, 1]])
        pred_state = np.dot(F, self.state)
        pred_P = np.dot(np.dot(F, self.P), F.T) + self.Q
        H = np.array([[1, 0]])
        S = np.dot(np.dot(H, pred_P), H.T) + self.R
        K = np.dot(np.dot(pred_P, H.T), 1.0 / S)
        innovation = measurement - np.dot(H, pred_state)
        self.state = pred_state + K.flatten() * innovation
        self.P = np.dot(np.eye(2) - np.outer(K.flatten(), H), pred_P)
        velocity = self.state[1]
        if velocity > 0.5:
            self.signal = 1
            self.confidence = min(abs(velocity) / 2, 1.0)
        elif velocity < -0.5:
            self.signal = -1
            self.confidence = min(abs(velocity) / 2, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(203)
class UnscentedKalman(BaseFormula):
    """ID 203: Unscented Kalman Filter"""

    CATEGORY = "signal_processing"
    NAME = "UnscentedKalman"
    DESCRIPTION = "UKF with sigma points"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.state = 0.0
        self.P = 1.0
        self.Q = 0.001
        self.R = 0.1
        self.alpha = 0.001
        self.beta = 2.0
        self.kappa = 0.0

    def _compute(self) -> None:
        if len(self.prices) < 2:
            return
        measurement = self.prices[-1]
        n = 1
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        sigma_points = np.array([
            self.state,
            self.state + np.sqrt((n + lambda_) * self.P),
            self.state - np.sqrt((n + lambda_) * self.P)
        ])
        W_m = np.array([lambda_ / (n + lambda_), 0.5 / (n + lambda_), 0.5 / (n + lambda_)])
        W_c = W_m.copy()
        W_c[0] += (1 - self.alpha**2 + self.beta)
        pred_sigma = sigma_points
        pred_mean = np.sum(W_m * pred_sigma)
        pred_cov = np.sum(W_c * (pred_sigma - pred_mean)**2) + self.Q
        meas_sigma = pred_sigma
        meas_mean = np.sum(W_m * meas_sigma)
        meas_cov = np.sum(W_c * (meas_sigma - meas_mean)**2) + self.R
        cross_cov = np.sum(W_c * (pred_sigma - pred_mean) * (meas_sigma - meas_mean))
        K = cross_cov / (meas_cov + 1e-10)
        innovation = measurement - meas_mean
        self.state = pred_mean + K * innovation
        self.P = pred_cov - K**2 * meas_cov
        z = innovation / np.sqrt(meas_cov + 1e-10)
        if z > 2:
            self.signal = 1
            self.confidence = min(abs(z) / 4, 1.0)
        elif z < -2:
            self.signal = -1
            self.confidence = min(abs(z) / 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(204)
class ParticleFilter(BaseFormula):
    """ID 204: Particle Filter (Sequential Monte Carlo)"""

    CATEGORY = "signal_processing"
    NAME = "ParticleFilter"
    DESCRIPTION = "Nonlinear non-Gaussian state estimation"

    def __init__(self, lookback: int = 100, n_particles: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_particles = n_particles
        self.particles = np.random.randn(n_particles) * 10
        self.weights = np.ones(n_particles) / n_particles
        self.estimate = 0.0

    def _compute(self) -> None:
        if len(self.prices) < 2:
            return
        measurement = self.prices[-1]
        self.particles += np.random.randn(self.n_particles) * 0.5
        sigma = np.std(self.particles) + 1e-10
        likelihoods = np.exp(-0.5 * ((measurement - self.particles) / sigma)**2)
        self.weights = self.weights * likelihoods
        self.weights /= (np.sum(self.weights) + 1e-10)
        self.estimate = np.sum(self.weights * self.particles)
        n_eff = 1.0 / (np.sum(self.weights**2) + 1e-10)
        if n_eff < self.n_particles / 2:
            indices = np.random.choice(self.n_particles, self.n_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles
        innovation = measurement - self.estimate
        std = np.std(self.particles)
        z = innovation / (std + 1e-10)
        if z > 2:
            self.signal = 1
            self.confidence = min(abs(z) / 4, 1.0)
        elif z < -2:
            self.signal = -1
            self.confidence = min(abs(z) / 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(205)
class AdaptiveFilter(BaseFormula):
    """ID 205: LMS Adaptive Filter"""

    CATEGORY = "signal_processing"
    NAME = "AdaptiveFilter"
    DESCRIPTION = "Least Mean Squares adaptive filter"

    def __init__(self, lookback: int = 100, filter_length: int = 10,
                 mu: float = 0.01, **kwargs):
        super().__init__(lookback, **kwargs)
        self.filter_length = filter_length
        self.mu = mu
        self.weights = np.zeros(filter_length)
        self.error = 0.0

    def _compute(self) -> None:
        if len(self.prices) < self.filter_length + 1:
            return
        prices = self._prices_array()
        x = prices[-self.filter_length-1:-1]
        d = prices[-1]
        y = np.dot(self.weights, x)
        self.error = d - y
        self.weights += self.mu * self.error * x / (np.dot(x, x) + 1e-10)
        if self.error > np.std(prices) * 2:
            self.signal = 1
            self.confidence = min(abs(self.error) / np.std(prices) / 4, 1.0)
        elif self.error < -np.std(prices) * 2:
            self.signal = -1
            self.confidence = min(abs(self.error) / np.std(prices) / 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(206)
class RLSFilter(BaseFormula):
    """ID 206: Recursive Least Squares Filter"""

    CATEGORY = "signal_processing"
    NAME = "RLSFilter"
    DESCRIPTION = "RLS adaptive filter with forgetting"

    def __init__(self, lookback: int = 100, filter_length: int = 10,
                 lambda_: float = 0.99, delta: float = 0.1, **kwargs):
        super().__init__(lookback, **kwargs)
        self.filter_length = filter_length
        self.lambda_ = lambda_
        self.weights = np.zeros(filter_length)
        self.P = np.eye(filter_length) / delta
        self.error = 0.0

    def _compute(self) -> None:
        if len(self.prices) < self.filter_length + 1:
            return
        prices = self._prices_array()
        x = prices[-self.filter_length-1:-1]
        d = prices[-1]
        pi = np.dot(self.P, x)
        k = pi / (self.lambda_ + np.dot(x, pi))
        y = np.dot(self.weights, x)
        self.error = d - y
        self.weights += k * self.error
        self.P = (self.P - np.outer(k, pi)) / self.lambda_
        if self.error > np.std(prices) * 2:
            self.signal = 1
            self.confidence = min(abs(self.error) / np.std(prices) / 4, 1.0)
        elif self.error < -np.std(prices) * 2:
            self.signal = -1
            self.confidence = min(abs(self.error) / np.std(prices) / 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(207)
class SavitzkyGolay(BaseFormula):
    """ID 207: Savitzky-Golay Smoothing Filter"""

    CATEGORY = "signal_processing"
    NAME = "SavitzkyGolay"
    DESCRIPTION = "Polynomial smoothing filter"

    def __init__(self, lookback: int = 100, window: int = 11, poly_order: int = 3, **kwargs):
        super().__init__(lookback, **kwargs)
        self.window = window
        self.poly_order = poly_order
        self.smoothed = 0.0
        self.derivative = 0.0

    def _compute(self) -> None:
        if len(self.prices) < self.window:
            return
        prices = self._prices_array()
        data = prices[-self.window:]
        x = np.arange(self.window) - self.window // 2
        coeffs = np.polyfit(x, data, self.poly_order)
        poly = np.poly1d(coeffs)
        self.smoothed = poly(0)
        deriv_coeffs = np.polyder(coeffs)
        self.derivative = np.polyval(deriv_coeffs, 0)
        if self.derivative > np.std(prices) * 0.5:
            self.signal = 1
            self.confidence = min(abs(self.derivative) / np.std(prices), 1.0)
        elif self.derivative < -np.std(prices) * 0.5:
            self.signal = -1
            self.confidence = min(abs(self.derivative) / np.std(prices), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(208)
class MedianFilter(BaseFormula):
    """ID 208: Median Filter for outlier removal"""

    CATEGORY = "signal_processing"
    NAME = "MedianFilter"
    DESCRIPTION = "Robust nonlinear smoothing"

    def __init__(self, lookback: int = 100, window: int = 5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.window = window
        self.filtered = 0.0
        self.is_outlier = False

    def _compute(self) -> None:
        if len(self.prices) < self.window:
            return
        prices = self._prices_array()
        data = prices[-self.window:]
        self.filtered = np.median(data)
        current = prices[-1]
        deviation = abs(current - self.filtered)
        mad = np.median(np.abs(data - self.filtered))
        z = deviation / (mad * 1.4826 + 1e-10)
        self.is_outlier = z > 3
        if self.is_outlier:
            self.signal = -1 if current > self.filtered else 1
            self.confidence = min(z / 5, 1.0)
        else:
            trend = self.filtered - np.median(prices[-self.window*2:-self.window]) if len(prices) >= self.window*2 else 0
            self.signal = 1 if trend > 0 else (-1 if trend < 0 else 0)
            self.confidence = 0.5


@FormulaRegistry.register(209)
class ExponentialSmoothing(BaseFormula):
    """ID 209: Double Exponential Smoothing"""

    CATEGORY = "signal_processing"
    NAME = "ExponentialSmoothing"
    DESCRIPTION = "Holt's linear trend method"

    def __init__(self, lookback: int = 100, alpha: float = 0.3, beta: float = 0.1, **kwargs):
        super().__init__(lookback, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.level = 0.0
        self.trend = 0.0
        self.initialized = False

    def _compute(self) -> None:
        if len(self.prices) < 2:
            return
        prices = self._prices_array()
        if not self.initialized:
            self.level = prices[-1]
            self.trend = prices[-1] - prices[-2] if len(prices) >= 2 else 0
            self.initialized = True
            return
        prev_level = self.level
        self.level = self.alpha * prices[-1] + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
        forecast = self.level + self.trend
        forecast_error = prices[-1] - (prev_level + self.trend)
        if self.trend > np.std(prices) * 0.3:
            self.signal = 1
            self.confidence = min(abs(self.trend) / np.std(prices), 1.0)
        elif self.trend < -np.std(prices) * 0.3:
            self.signal = -1
            self.confidence = min(abs(self.trend) / np.std(prices), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(210)
class TripleExponentialSmoothing(BaseFormula):
    """ID 210: Holt-Winters Triple Exponential"""

    CATEGORY = "signal_processing"
    NAME = "TripleExponentialSmoothing"
    DESCRIPTION = "Level, trend, and seasonality"

    def __init__(self, lookback: int = 100, alpha: float = 0.3,
                 beta: float = 0.1, gamma: float = 0.1, season_length: int = 10, **kwargs):
        super().__init__(lookback, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_length = season_length
        self.level = 0.0
        self.trend = 0.0
        self.seasonal = np.zeros(season_length)
        self.initialized = False
        self.t = 0

    def _compute(self) -> None:
        if len(self.prices) < self.season_length + 2:
            return
        prices = self._prices_array()
        if not self.initialized:
            self.level = np.mean(prices[-self.season_length:])
            self.trend = (np.mean(prices[-self.season_length:]) -
                         np.mean(prices[-2*self.season_length:-self.season_length])) / self.season_length \
                         if len(prices) >= 2*self.season_length else 0
            for i in range(self.season_length):
                idx = -self.season_length + i
                self.seasonal[i] = prices[idx] - self.level
            self.initialized = True
            return
        season_idx = self.t % self.season_length
        prev_level = self.level
        self.level = self.alpha * (prices[-1] - self.seasonal[season_idx]) + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
        self.seasonal[season_idx] = self.gamma * (prices[-1] - self.level) + (1 - self.gamma) * self.seasonal[season_idx]
        self.t += 1
        next_season_idx = self.t % self.season_length
        forecast = self.level + self.trend + self.seasonal[next_season_idx]
        if self.trend > np.std(prices) * 0.3:
            self.signal = 1
            self.confidence = min(abs(self.trend) / np.std(prices), 1.0)
        elif self.trend < -np.std(prices) * 0.3:
            self.signal = -1
            self.confidence = min(abs(self.trend) / np.std(prices), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


__all__ = [
    'FFTCycleDetector', 'SpectralDensity', 'HilbertTransform', 'BandpassFilter',
    'LowpassFilter', 'HighpassFilter', 'WaveletDecomposition', 'MorletWavelet',
    'EMDDecomposition', 'SingularSpectrum',
    'KalmanFilter', 'ExtendedKalman', 'UnscentedKalman', 'ParticleFilter',
    'AdaptiveFilter', 'RLSFilter', 'SavitzkyGolay', 'MedianFilter',
    'ExponentialSmoothing', 'TripleExponentialSmoothing',
]
