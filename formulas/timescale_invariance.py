"""
Time-Scale Invariance Formulas (IDs 347-360)
=============================================
Academic gold standard formulas for multi-timeframe trading.
These formulas detect which time scales work and adapt accordingly.

References:
    - Mandelbrot (1963): Self-affinity and fractional Brownian motion
    - Lo (1991): Long-term memory in stock market prices
    - Peters (1994): Fractal Market Hypothesis
    - Percival & Walden (2000): Wavelet Methods for Time Series Analysis
    - Bandi & Russell (2006): Optimal sampling frequency
    - Barndorff-Nielsen & Shephard (2002): Realized volatility
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from .base import BaseFormula, FormulaRegistry


# =============================================================================
# HURST EXPONENT VARIANTS (347-349)
# =============================================================================

@FormulaRegistry.register(347)
class AdaptiveHurstExponent(BaseFormula):
    """
    ID 347: Adaptive Hurst Exponent with Rolling Window

    Academic Reference: Lo (1991) "Long-term memory in stock market prices"

    The Hurst exponent H determines the market regime:
        H > 0.5: Trending (persistent) - momentum strategies work
        H = 0.5: Random walk - no predictability
        H < 0.5: Mean-reverting (anti-persistent) - mean reversion works

    This adaptive version:
        1. Computes H over rolling windows of different sizes
        2. Detects regime changes when H crosses 0.5
        3. Adapts trading strategy to current regime

    Formula:
        R/S = (max(cumdev) - min(cumdev)) / std(returns)
        log(R/S) = H * log(n) + c
    """

    NAME = "AdaptiveHurstExponent"
    CATEGORY = "timescale_invariance"
    DESCRIPTION = "Rolling Hurst exponent for adaptive regime detection"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.window_sizes = [16, 32, 64, 128]  # Multiple scales
        self.hurst_history = deque(maxlen=50)
        self.current_hurst = 0.5
        self.regime = 'random'  # 'trending', 'random', 'mean_reverting'
        self.hurst_std = 0.1  # For confidence calculation

    def _compute_rs(self, data: np.ndarray) -> float:
        """Compute R/S statistic for a data series"""
        n = len(data)
        if n < 4:
            return np.nan

        mean = np.mean(data)
        cumdev = np.cumsum(data - mean)
        R = np.max(cumdev) - np.min(cumdev)
        S = np.std(data, ddof=1)

        if S < 1e-10:
            return np.nan
        return R / S

    def _compute_hurst(self, returns: np.ndarray) -> float:
        """Compute Hurst exponent using R/S analysis"""
        rs_values = []

        for n in self.window_sizes:
            if len(returns) < n:
                continue

            # Compute R/S for non-overlapping windows
            rs_list = []
            for start in range(0, len(returns) - n + 1, n // 2):
                segment = returns[start:start + n]
                rs = self._compute_rs(segment)
                if not np.isnan(rs) and rs > 0:
                    rs_list.append(rs)

            if rs_list:
                avg_rs = np.mean(rs_list)
                if avg_rs > 0:
                    rs_values.append((np.log(n), np.log(avg_rs)))

        if len(rs_values) >= 2:
            log_n = np.array([v[0] for v in rs_values])
            log_rs = np.array([v[1] for v in rs_values])
            # OLS regression: log(R/S) = H * log(n) + c
            H = np.polyfit(log_n, log_rs, 1)[0]
            return np.clip(H, 0, 1)

        return 0.5

    def _compute(self) -> None:
        if len(self.returns) < 70:
            return

        returns = self._returns_array()

        # Compute current Hurst exponent
        self.current_hurst = self._compute_hurst(returns)
        self.hurst_history.append(self.current_hurst)

        # Update regime
        if self.current_hurst > 0.55:
            self.regime = 'trending'
        elif self.current_hurst < 0.45:
            self.regime = 'mean_reverting'
        else:
            self.regime = 'random'

        # Compute Hurst stability (std of recent Hurst values)
        if len(self.hurst_history) >= 10:
            self.hurst_std = np.std(list(self.hurst_history)[-10:])

        # Generate signal based on regime
        if self.regime == 'trending':
            # Momentum: follow recent direction
            recent_return = np.mean(returns[-5:])
            self.signal = 1 if recent_return > 0 else -1
            self.confidence = min((self.current_hurst - 0.5) * 4, 1.0)

        elif self.regime == 'mean_reverting':
            # Mean reversion: fade recent direction
            z_score = returns[-1] / (np.std(returns) + 1e-10)
            self.signal = -1 if z_score > 1 else (1 if z_score < -1 else 0)
            self.confidence = min((0.5 - self.current_hurst) * 4, 1.0)

        else:
            # Random walk: no signal
            self.signal = 0
            self.confidence = 0.3

        # Reduce confidence if Hurst is unstable
        if self.hurst_std > 0.1:
            self.confidence *= 0.5


@FormulaRegistry.register(348)
class MultifractalDFA(BaseFormula):
    """
    ID 348: Multifractal Detrended Fluctuation Analysis (MF-DFA)

    Academic Reference:
        - Kantelhardt et al. (2002) "Multifractal DFA of nonstationary time series"
        - Di Matteo et al. (2005) "Multi-scaling in finance"

    MF-DFA extends Hurst exponent to multiple moments q:
        h(q) = generalized Hurst exponent for moment q

    Multifractal Width:
        Delta_h = h(q_min) - h(q_max)

    Interpretation:
        - Delta_h ~ 0: Monofractal (simple scaling)
        - Delta_h > 0.1: Multifractal (complex scaling, fat tails)

    This formula detects when simple vs complex models are needed.
    """

    NAME = "MultifractalDFA"
    CATEGORY = "timescale_invariance"
    DESCRIPTION = "Multi-scale Hurst analysis for complex market regimes"

    def __init__(self, lookback: int = 256, **kwargs):
        super().__init__(lookback, **kwargs)
        self.q_values = [-2, -1, 0.5, 1, 2, 3, 4]  # Moment orders
        self.scales = [8, 16, 32, 64, 128]
        self.h_q = {}  # Generalized Hurst for each q
        self.multifractal_width = 0.0
        self.complexity = 'low'  # 'low', 'medium', 'high'

    def _detrend_segment(self, segment: np.ndarray) -> np.ndarray:
        """Remove linear trend from segment"""
        n = len(segment)
        x = np.arange(n)
        coeffs = np.polyfit(x, segment, 1)
        trend = np.polyval(coeffs, x)
        return segment - trend

    def _compute_fluctuation(self, data: np.ndarray, scale: int, q: float) -> float:
        """Compute q-th order fluctuation function F_q(s)"""
        n = len(data)
        n_segments = n // scale

        if n_segments < 2:
            return np.nan

        # Cumulative sum (integration)
        profile = np.cumsum(data - np.mean(data))

        fluctuations = []
        for v in range(n_segments):
            segment = profile[v*scale:(v+1)*scale]
            if len(segment) == scale:
                detrended = self._detrend_segment(segment)
                rms = np.sqrt(np.mean(detrended**2))
                if rms > 0:
                    fluctuations.append(rms)

        if len(fluctuations) < 2:
            return np.nan

        F = np.array(fluctuations)

        # q-th order fluctuation
        if q == 0:
            return np.exp(np.mean(np.log(F + 1e-10)))
        else:
            return np.power(np.mean(np.power(F, q)), 1.0/q)

    def _compute(self) -> None:
        if len(self.returns) < 130:
            return

        returns = self._returns_array()

        # Compute h(q) for each q value
        self.h_q = {}
        for q in self.q_values:
            fq_values = []
            for scale in self.scales:
                fq = self._compute_fluctuation(returns, scale, q)
                if not np.isnan(fq) and fq > 0:
                    fq_values.append((np.log(scale), np.log(fq)))

            if len(fq_values) >= 2:
                log_s = np.array([v[0] for v in fq_values])
                log_fq = np.array([v[1] for v in fq_values])
                h = np.polyfit(log_s, log_fq, 1)[0]
                self.h_q[q] = np.clip(h, 0, 2)

        # Calculate multifractal width
        if len(self.h_q) >= 3:
            h_values = list(self.h_q.values())
            self.multifractal_width = max(h_values) - min(h_values)

            # Classify complexity
            if self.multifractal_width < 0.1:
                self.complexity = 'low'
            elif self.multifractal_width < 0.25:
                self.complexity = 'medium'
            else:
                self.complexity = 'high'

        # Get main Hurst exponent (q=2)
        h2 = self.h_q.get(2, 0.5)

        # Generate signal based on Hurst and complexity
        if self.complexity == 'high':
            # High complexity = volatile, risky - reduce exposure
            self.signal = 0
            self.confidence = 0.3
        elif h2 > 0.55:
            # Trending regime
            recent_return = np.mean(returns[-5:])
            self.signal = 1 if recent_return > 0 else -1
            self.confidence = min((h2 - 0.5) * 3, 1.0)
        elif h2 < 0.45:
            # Mean reverting
            z = returns[-1] / (np.std(returns) + 1e-10)
            self.signal = -1 if z > 1 else (1 if z < -1 else 0)
            self.confidence = min((0.5 - h2) * 3, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(349)
class TimeVaryingHurst(BaseFormula):
    """
    ID 349: Time-Varying Hurst with Regime Change Detection

    Academic Reference:
        - Corazza & Malliaris (2002) "Multi-Fractality in Foreign Currency Markets"
        - Alvarez-Ramirez et al. (2008) "Time-varying Hurst exponent for financial time series"

    This formula tracks H(t) - Hurst exponent as a function of time.
    Key insight: Regime changes are detected when H(t) crosses 0.5.

    Applications:
        - Detect when momentum starts working
        - Detect when mean reversion starts working
        - Avoid trading during random walk periods
    """

    NAME = "TimeVaryingHurst"
    CATEGORY = "timescale_invariance"
    DESCRIPTION = "H(t) tracking with regime change alerts"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.hurst_window = kwargs.get('hurst_window', 64)
        self.hurst_series = deque(maxlen=100)
        self.regime_change_count = 0
        self.last_regime = 'random'
        self.regime_duration = 0

    def _quick_hurst(self, data: np.ndarray) -> float:
        """Fast Hurst estimation using variance ratio method"""
        n = len(data)
        if n < 16:
            return 0.5

        # Variance at different lags
        var_1 = np.var(data)

        # Aggregate to different scales
        scales = [2, 4, 8]
        log_scale = []
        log_var = []

        for s in scales:
            if n >= s * 4:
                # Aggregate returns
                n_agg = n // s
                agg_data = np.array([np.sum(data[i*s:(i+1)*s]) for i in range(n_agg)])
                var_s = np.var(agg_data)
                if var_s > 0 and var_1 > 0:
                    log_scale.append(np.log(s))
                    log_var.append(np.log(var_s / var_1))

        if len(log_scale) >= 2:
            # H = slope/2 + 0.5
            slope = np.polyfit(log_scale, log_var, 1)[0]
            H = slope / 2 + 0.5
            return np.clip(H, 0, 1)

        return 0.5

    def _compute(self) -> None:
        if len(self.returns) < self.hurst_window:
            return

        returns = self._returns_array()

        # Compute current Hurst
        current_h = self._quick_hurst(returns[-self.hurst_window:])
        self.hurst_series.append(current_h)

        # Determine current regime
        if current_h > 0.55:
            current_regime = 'trending'
        elif current_h < 0.45:
            current_regime = 'mean_reverting'
        else:
            current_regime = 'random'

        # Check for regime change
        if current_regime != self.last_regime:
            self.regime_change_count += 1
            self.regime_duration = 0
        else:
            self.regime_duration += 1

        self.last_regime = current_regime

        # Calculate Hurst trend
        if len(self.hurst_series) >= 10:
            h_recent = np.mean(list(self.hurst_series)[-5:])
            h_older = np.mean(list(self.hurst_series)[-10:-5])
            hurst_trend = h_recent - h_older
        else:
            hurst_trend = 0

        # Generate signal
        # Strong signal if regime is stable (duration > 5)
        regime_stability = min(self.regime_duration / 10, 1.0)

        if current_regime == 'trending':
            recent = np.mean(returns[-5:])
            self.signal = 1 if recent > 0 else -1
            self.confidence = min((current_h - 0.5) * 3, 1.0) * (0.5 + 0.5 * regime_stability)

        elif current_regime == 'mean_reverting':
            z = returns[-1] / (np.std(returns) + 1e-10)
            self.signal = -1 if z > 1 else (1 if z < -1 else 0)
            self.confidence = min((0.5 - current_h) * 3, 1.0) * (0.5 + 0.5 * regime_stability)

        else:
            # Random walk with trend - only trade if Hurst is trending up or down
            if hurst_trend > 0.02:
                # Hurst rising toward trending
                recent = np.mean(returns[-3:])
                self.signal = 1 if recent > 0 else -1
                self.confidence = 0.4
            elif hurst_trend < -0.02:
                # Hurst falling toward mean-reversion
                z = returns[-1] / (np.std(returns) + 1e-10)
                self.signal = -1 if z > 0.5 else (1 if z < -0.5 else 0)
                self.confidence = 0.4
            else:
                self.signal = 0
                self.confidence = 0.2


# =============================================================================
# WAVELET ANALYSIS (350-352)
# =============================================================================

@FormulaRegistry.register(350)
class MODWTWavelet(BaseFormula):
    """
    ID 350: Maximal Overlap Discrete Wavelet Transform (MODWT)

    Academic Reference:
        - Percival & Walden (2000) "Wavelet Methods for Time Series Analysis"
        - Gencay et al. (2002) "An Introduction to Wavelets and Other Filtering Methods"

    MODWT Advantages over DWT:
        1. Shift-invariant (no boundary effects)
        2. Works with any sample size (not just powers of 2)
        3. Produces aligned outputs at each scale

    Trading Application:
        - Decompose price into independent time scales
        - Trade signals at scales with highest energy
        - Ignore noise at high-frequency scales
    """

    NAME = "MODWTWavelet"
    CATEGORY = "timescale_invariance"
    DESCRIPTION = "MODWT for multi-scale signal decomposition"

    def __init__(self, lookback: int = 256, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_levels = kwargs.get('n_levels', 5)
        self.wavelet_filter = 'la8'  # Least asymmetric with 8 taps
        self.scale_energies = []
        self.dominant_scale = 0

    def _la8_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Least Asymmetric wavelet filter with 8 coefficients (Daubechies)"""
        # LA8 scaling coefficients
        h = np.array([
            -0.0757657147893407,
            -0.0296355276459541,
            0.4976186676324578,
            0.8037387518052163,
            0.2978577956055422,
            -0.0992195435769354,
            -0.0126039672622612,
            0.0322231006040713
        ])
        # Wavelet coefficients (QMF)
        g = np.array([(-1)**k * h[7-k] for k in range(8)])
        return h, g

    def _modwt_level(self, data: np.ndarray, h: np.ndarray, g: np.ndarray,
                     level: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute one level of MODWT"""
        n = len(data)
        L = len(h)

        # Scale filter for MODWT
        scale = 2 ** (level - 1)
        h_j = h / np.sqrt(2 ** level)
        g_j = g / np.sqrt(2 ** level)

        # Wavelet and scaling coefficients
        W = np.zeros(n)
        V = np.zeros(n)

        for t in range(n):
            for l in range(L):
                k = (t - l * scale) % n
                W[t] += h_j[l] * data[k]
                V[t] += g_j[l] * data[k]

        return W, V

    def _compute(self) -> None:
        if len(self.returns) < 64:
            return

        returns = self._returns_array()
        h, g = self._la8_filter()

        # MODWT decomposition
        V = returns.copy()
        details = []
        self.scale_energies = []

        for j in range(1, self.n_levels + 1):
            if len(V) < 16:
                break
            W, V = self._modwt_level(V, h, g, j)
            details.append(W)

            # Energy at this scale
            energy = np.mean(W[-32:]**2)
            self.scale_energies.append(energy)

        if not self.scale_energies:
            self.signal = 0
            self.confidence = 0.3
            return

        # Find dominant scale (highest energy in recent data)
        self.dominant_scale = np.argmax(self.scale_energies) + 1

        # Total energy for normalization
        total_energy = sum(self.scale_energies)
        if total_energy == 0:
            self.signal = 0
            self.confidence = 0.3
            return

        # Energy ratio at dominant scale
        dominant_energy_ratio = self.scale_energies[self.dominant_scale - 1] / total_energy

        # Get signal from dominant scale
        if self.dominant_scale <= len(details):
            dominant_detail = details[self.dominant_scale - 1]

            # Recent trend at dominant scale
            recent_trend = np.mean(dominant_detail[-10:])

            if abs(recent_trend) > np.std(dominant_detail) * 0.5:
                self.signal = 1 if recent_trend > 0 else -1
                self.confidence = min(dominant_energy_ratio * 2, 1.0)
            else:
                self.signal = 0
                self.confidence = 0.4
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(351)
class WaveletVarianceAnalysis(BaseFormula):
    """
    ID 351: Wavelet Variance by Scale

    Academic Reference:
        - Whitcher et al. (2000) "Wavelet analysis of covariance with applications"
        - In, Sangbae & Kim, Sang (2006) "The hedge ratio and the empirical performance of
          the minimum variance hedge"

    Key Insight:
        Variance at each wavelet scale corresponds to a time horizon.
        - Scale 1: 2^1 = 2 periods (shortest)
        - Scale 2: 2^2 = 4 periods
        - Scale J: 2^J periods (longest)

    Trading Application:
        - High variance at short scales = noise-dominated
        - High variance at long scales = trend-dominated
        - Adapt holding period to match dominant variance scale
    """

    NAME = "WaveletVarianceAnalysis"
    CATEGORY = "timescale_invariance"
    DESCRIPTION = "Variance decomposition by time scale"

    def __init__(self, lookback: int = 256, **kwargs):
        super().__init__(lookback, **kwargs)
        self.scale_variances = []
        self.optimal_holding = 1  # Optimal holding period in samples
        self.noise_ratio = 0.5

    def _haar_modwt(self, data: np.ndarray, level: int) -> np.ndarray:
        """Simplified Haar MODWT for efficiency"""
        n = len(data)
        scale = 2 ** (level - 1)
        W = np.zeros(n)

        for t in range(n):
            idx1 = t
            idx2 = (t - scale) % n
            W[t] = (data[idx1] - data[idx2]) / np.sqrt(2 ** level)

        return W

    def _compute(self) -> None:
        if len(self.returns) < 64:
            return

        returns = self._returns_array()
        n_levels = int(np.log2(len(returns))) - 2
        n_levels = min(n_levels, 6)

        # Compute wavelet variance at each scale
        self.scale_variances = []
        total_variance = np.var(returns)

        for j in range(1, n_levels + 1):
            W = self._haar_modwt(returns, j)
            var_j = np.var(W[-32:])  # Recent variance
            self.scale_variances.append(var_j)

        if not self.scale_variances or total_variance == 0:
            self.signal = 0
            self.confidence = 0.3
            return

        # Calculate noise ratio (variance at scale 1 / total)
        self.noise_ratio = self.scale_variances[0] / (sum(self.scale_variances) + 1e-10)

        # Find scale with maximum variance
        max_scale_idx = np.argmax(self.scale_variances)
        self.optimal_holding = 2 ** (max_scale_idx + 1)

        # Trend-to-noise ratio
        long_scale_var = sum(self.scale_variances[2:]) if len(self.scale_variances) > 2 else 0
        short_scale_var = sum(self.scale_variances[:2])

        trend_noise_ratio = long_scale_var / (short_scale_var + 1e-10)

        # Generate signal
        if trend_noise_ratio > 1.5:
            # Trend-dominated: momentum signal
            recent_return = np.sum(returns[-self.optimal_holding:])
            self.signal = 1 if recent_return > 0 else -1
            self.confidence = min(trend_noise_ratio / 3, 1.0)

        elif trend_noise_ratio < 0.5:
            # Noise-dominated: mean reversion
            z = returns[-1] / (np.std(returns) + 1e-10)
            self.signal = -1 if z > 1 else (1 if z < -1 else 0)
            self.confidence = min(1 / (trend_noise_ratio + 0.1), 1.0) * 0.5

        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(352)
class WaveletCoherence(BaseFormula):
    """
    ID 352: Wavelet Coherence for Multi-Scale Correlation

    Academic Reference:
        - Torrence & Compo (1998) "A Practical Guide to Wavelet Analysis"
        - Grinsted et al. (2004) "Application of the cross wavelet transform"

    Coherence measures correlation at each time scale.
    Used to find which scales are predictable vs random.

    Application:
        - High coherence at scale J = predictable at that horizon
        - Trade only at scales with high coherence
    """

    NAME = "WaveletCoherence"
    CATEGORY = "timescale_invariance"
    DESCRIPTION = "Scale-dependent predictability via coherence"

    def __init__(self, lookback: int = 256, **kwargs):
        super().__init__(lookback, **kwargs)
        self.coherence_by_scale = []
        self.most_predictable_scale = 1
        self.max_coherence = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 80:
            return

        returns = self._returns_array()

        # Create lagged series for coherence calculation
        if len(returns) < 40:
            self.signal = 0
            self.confidence = 0.3
            return

        # Split into past and future
        mid = len(returns) // 2
        past = returns[:mid]
        future = returns[mid:]

        min_len = min(len(past), len(future))
        past = past[-min_len:]
        future = future[:min_len]

        # Coherence at different scales using simple aggregation
        scales = [2, 4, 8, 16]
        self.coherence_by_scale = []

        for s in scales:
            if min_len < s * 4:
                continue

            # Aggregate returns at scale s
            n_agg = min_len // s
            past_agg = np.array([np.sum(past[i*s:(i+1)*s]) for i in range(n_agg)])
            future_agg = np.array([np.sum(future[i*s:(i+1)*s]) for i in range(n_agg)])

            if len(past_agg) > 3:
                # Correlation between past and future at this scale
                corr = np.corrcoef(past_agg[:-1], future_agg[1:])[0, 1]
                if not np.isnan(corr):
                    self.coherence_by_scale.append((s, abs(corr)))

        if not self.coherence_by_scale:
            self.signal = 0
            self.confidence = 0.3
            return

        # Find most predictable scale
        best_scale, best_coherence = max(self.coherence_by_scale, key=lambda x: x[1])
        self.most_predictable_scale = best_scale
        self.max_coherence = best_coherence

        # Trade at most predictable scale
        if self.max_coherence > 0.3:
            # Aggregate recent returns at best scale
            recent_agg = np.sum(returns[-best_scale:])
            self.signal = 1 if recent_agg > 0 else -1
            self.confidence = min(self.max_coherence * 2, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


# =============================================================================
# OPTIMAL SAMPLING (353-354)
# =============================================================================

@FormulaRegistry.register(353)
class VolatilitySignaturePlot(BaseFormula):
    """
    ID 353: Volatility Signature Plot for Optimal Sampling Frequency

    Academic Reference:
        - Andersen et al. (2000) "The distribution of realized stock return volatility"
        - Bandi & Russell (2006) "Separating microstructure noise from volatility"
        - Zhang et al. (2005) "A tale of two time scales"

    The Signature Plot:
        Plot realized variance vs sampling frequency.
        - At high frequencies: microstructure noise inflates RV
        - At low frequencies: estimation error increases
        - Optimal frequency: where plot flattens

    Typical Results:
        - Stocks: 5-15 minutes optimal
        - FX: 15-30 seconds optimal
        - Crypto: 10-60 seconds optimal (market dependent)
    """

    NAME = "VolatilitySignaturePlot"
    CATEGORY = "timescale_invariance"
    DESCRIPTION = "Find optimal sampling frequency for volatility"

    def __init__(self, lookback: int = 1000, **kwargs):
        super().__init__(lookback, **kwargs)
        self.sampling_intervals = [1, 2, 5, 10, 20, 30, 60]  # In ticks/samples
        self.realized_variances = {}
        self.optimal_interval = 10
        self.noise_variance = 0.0
        self.true_variance = 0.0

    def _realized_variance(self, returns: np.ndarray, interval: int) -> float:
        """Compute realized variance at given sampling interval"""
        n = len(returns)
        if n < interval * 3:
            return np.nan

        # Aggregate returns at the interval
        n_samples = n // interval
        agg_returns = np.array([np.sum(returns[i*interval:(i+1)*interval])
                                for i in range(n_samples)])

        # Realized variance (sum of squared returns)
        rv = np.sum(agg_returns**2)

        # Annualize (assuming each sample is 1 second, adjust as needed)
        return rv

    def _compute(self) -> None:
        if len(self.returns) < 200:
            return

        returns = self._returns_array()

        # Compute signature plot
        self.realized_variances = {}
        for interval in self.sampling_intervals:
            rv = self._realized_variance(returns, interval)
            if not np.isnan(rv):
                self.realized_variances[interval] = rv

        if len(self.realized_variances) < 3:
            self.signal = 0
            self.confidence = 0.3
            return

        # Find where the plot stabilizes (derivative approaches zero)
        intervals = sorted(self.realized_variances.keys())
        rvs = [self.realized_variances[i] for i in intervals]

        # Calculate derivatives
        derivatives = []
        for i in range(1, len(rvs)):
            d = abs(rvs[i] - rvs[i-1]) / (rvs[i-1] + 1e-10)
            derivatives.append(d)

        # Find first interval where derivative is small
        threshold = 0.1  # 10% change
        self.optimal_interval = intervals[-1]  # Default to longest

        for i, d in enumerate(derivatives):
            if d < threshold:
                self.optimal_interval = intervals[i + 1]
                break

        # Estimate noise variance (difference between RV at interval 1 and optimal)
        if 1 in self.realized_variances and self.optimal_interval in self.realized_variances:
            rv_1 = self.realized_variances[1]
            rv_opt = self.realized_variances[self.optimal_interval]
            self.noise_variance = max(rv_1 - rv_opt, 0)
            self.true_variance = rv_opt

        # Noise ratio
        total_var = self.noise_variance + self.true_variance
        noise_ratio = self.noise_variance / (total_var + 1e-10)

        # Generate signal
        # Aggregate returns at optimal interval for cleaner signal
        opt = self.optimal_interval
        if len(returns) >= opt * 5:
            recent_agg = np.sum(returns[-opt:])
            prev_agg = np.sum(returns[-2*opt:-opt])

            # Momentum at optimal scale
            if abs(recent_agg) > np.std(returns) * np.sqrt(opt) * 1.5:
                self.signal = 1 if recent_agg > 0 else -1
                # Confidence inversely related to noise
                self.confidence = 1 - noise_ratio
            else:
                self.signal = 0
                self.confidence = 0.4
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(354)
class OptimalHoldingPeriod(BaseFormula):
    """
    ID 354: Optimal Holding Period via Variance Ratio

    Academic Reference:
        - Lo & MacKinlay (1988) "Stock market prices do not follow random walks"
        - Campbell et al. (1997) "The Econometrics of Financial Markets"

    The variance ratio VR(q) = Var(r_t + ... + r_{t+q-1}) / (q * Var(r_t))

    Interpretation:
        - VR(q) = 1: Random walk at horizon q
        - VR(q) > 1: Momentum at horizon q
        - VR(q) < 1: Mean reversion at horizon q

    Optimal holding = horizon q where |VR(q) - 1| is maximized
    """

    NAME = "OptimalHoldingPeriod"
    CATEGORY = "timescale_invariance"
    DESCRIPTION = "Find optimal trade duration via variance ratio"

    def __init__(self, lookback: int = 500, **kwargs):
        super().__init__(lookback, **kwargs)
        self.horizons = [2, 4, 8, 16, 32, 64]
        self.variance_ratios = {}
        self.optimal_horizon = 8
        self.market_type = 'random'  # 'momentum', 'mean_reverting', 'random'

    def _variance_ratio(self, returns: np.ndarray, q: int) -> float:
        """Compute variance ratio for horizon q"""
        n = len(returns)
        if n < q * 3:
            return np.nan

        # Single period variance
        var_1 = np.var(returns)
        if var_1 == 0:
            return np.nan

        # q-period returns
        q_returns = np.array([np.sum(returns[i:i+q]) for i in range(n - q + 1)])
        var_q = np.var(q_returns)

        # Variance ratio
        vr = var_q / (q * var_1)
        return vr

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()

        # Compute variance ratios
        self.variance_ratios = {}
        for q in self.horizons:
            vr = self._variance_ratio(returns, q)
            if not np.isnan(vr):
                self.variance_ratios[q] = vr

        if len(self.variance_ratios) < 2:
            self.signal = 0
            self.confidence = 0.3
            return

        # Find optimal horizon (max deviation from 1)
        max_deviation = 0
        best_q = self.horizons[0]
        best_vr = 1.0

        for q, vr in self.variance_ratios.items():
            deviation = abs(vr - 1)
            if deviation > max_deviation:
                max_deviation = deviation
                best_q = q
                best_vr = vr

        self.optimal_horizon = best_q

        # Determine market type at optimal horizon
        if best_vr > 1.15:
            self.market_type = 'momentum'
        elif best_vr < 0.85:
            self.market_type = 'mean_reverting'
        else:
            self.market_type = 'random'

        # Generate signal based on market type and optimal horizon
        if self.market_type == 'momentum':
            # Momentum strategy at optimal horizon
            recent_return = np.sum(returns[-best_q:])
            self.signal = 1 if recent_return > 0 else -1
            self.confidence = min((best_vr - 1) * 3, 1.0)

        elif self.market_type == 'mean_reverting':
            # Mean reversion at optimal horizon
            cumulative = np.sum(returns[-best_q:])
            expected_std = np.std(returns) * np.sqrt(best_q)
            z = cumulative / (expected_std + 1e-10)

            self.signal = -1 if z > 1.5 else (1 if z < -1.5 else 0)
            self.confidence = min((1 - best_vr) * 3, 1.0)

        else:
            self.signal = 0
            self.confidence = 0.3


# =============================================================================
# ADAPTIVE STRATEGIES (355-357)
# =============================================================================

@FormulaRegistry.register(355)
class AdaptiveOUHalfLife(BaseFormula):
    """
    ID 355: Adaptive Ornstein-Uhlenbeck Half-Life

    Academic Reference:
        - Uhlenbeck & Ornstein (1930) "On the Theory of Brownian Motion"
        - Elliott et al. (2005) "Pairs trading"
        - Cummins & Bucca (2012) "Quantitative spread trading on crude oil and refined products"

    The OU process: dX = theta*(mu - X)*dt + sigma*dW

    Half-life: tau = ln(2) / theta
        - Time for spread to revert halfway to mean
        - Used to determine optimal holding period for mean reversion

    This formula adapts the half-life estimate in real-time.
    """

    NAME = "AdaptiveOUHalfLife"
    CATEGORY = "timescale_invariance"
    DESCRIPTION = "Rolling OU half-life for optimal mean-reversion timing"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.estimation_window = kwargs.get('estimation_window', 100)
        self.half_life = 20  # Default in samples
        self.theta = 0.0
        self.mu = 0.0
        self.sigma = 0.0
        self.half_life_history = deque(maxlen=20)

    def _estimate_ou_params(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Estimate OU parameters using OLS regression"""
        n = len(prices)
        if n < 20:
            return 0, np.mean(prices), np.std(prices)

        # OU regression: dX = theta*(mu - X)*dt + noise
        # X[t] - X[t-1] = theta*(mu - X[t-1]) + noise
        # Rearrange: X[t] = a + b*X[t-1] + noise
        # where a = theta*mu, b = 1 - theta

        y = prices[1:]
        x = prices[:-1]

        # OLS regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean)**2)

        if denominator == 0:
            return 0, np.mean(prices), np.std(prices)

        b = numerator / denominator
        a = y_mean - b * x_mean

        # Extract OU parameters
        theta = 1 - b
        if theta <= 0 or theta >= 2:
            theta = 0.1  # Default if unreasonable

        mu = a / theta if theta != 0 else np.mean(prices)

        # Estimate sigma from residuals
        residuals = y - (a + b * x)
        sigma = np.std(residuals)

        return theta, mu, sigma

    def _compute(self) -> None:
        if len(self.prices) < self.estimation_window:
            return

        prices = self._prices_array()

        # Estimate OU parameters on rolling window
        window = prices[-self.estimation_window:]
        self.theta, self.mu, self.sigma = self._estimate_ou_params(window)

        # Calculate half-life
        if self.theta > 0.01:
            self.half_life = np.log(2) / self.theta
            self.half_life = np.clip(self.half_life, 1, 200)  # Reasonable bounds
        else:
            self.half_life = 200  # Very slow mean reversion

        self.half_life_history.append(self.half_life)

        # Current deviation from mean
        current_price = prices[-1]
        deviation = (current_price - self.mu) / (self.sigma + 1e-10)

        # Time since price crossed equilibrium
        # (simplified: use z-score threshold crossing)

        # Trading signal
        if abs(deviation) > 2.0:
            # Strong deviation - mean reversion opportunity
            self.signal = -1 if deviation > 0 else 1

            # Confidence based on half-life
            # Short half-life = faster reversion = higher confidence
            if self.half_life < 10:
                self.confidence = 0.8
            elif self.half_life < 30:
                self.confidence = 0.6
            elif self.half_life < 60:
                self.confidence = 0.4
            else:
                self.confidence = 0.2

        elif abs(deviation) > 1.0:
            self.signal = -1 if deviation > 0 else 1
            self.confidence = 0.4 if self.half_life < 30 else 0.2

        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(356)
class RollingKellyCriterion(BaseFormula):
    """
    ID 356: Rolling Kelly Criterion for Adaptive Position Sizing

    Academic Reference:
        - Kelly (1956) "A New Interpretation of Information Rate"
        - Thorp (2006) "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
        - MacLean et al. (2010) "Good and bad properties of the Kelly criterion"

    Kelly Fraction: f* = (p*b - q) / b
        where p = win probability, q = 1-p, b = win/loss ratio

    For trading: f* = (mu - r) / sigma^2
        - mu = expected return
        - r = risk-free rate
        - sigma = volatility

    This formula:
        1. Estimates Kelly over rolling window
        2. Applies half-Kelly for safety
        3. Adapts to changing market conditions
    """

    NAME = "RollingKellyCriterion"
    CATEGORY = "timescale_invariance"
    DESCRIPTION = "Adaptive Kelly fraction for position sizing"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.kelly_window = kwargs.get('kelly_window', 100)
        self.risk_free_rate = kwargs.get('risk_free_rate', 0.0)
        self.kelly_fraction = 0.0
        self.half_kelly = 0.0
        self.kelly_history = deque(maxlen=20)

    def _compute(self) -> None:
        if len(self.returns) < self.kelly_window:
            return

        returns = self._returns_array()
        window = returns[-self.kelly_window:]

        # Estimate expected return and variance
        mu = np.mean(window)
        sigma_sq = np.var(window)

        if sigma_sq < 1e-10:
            self.kelly_fraction = 0
            self.signal = 0
            self.confidence = 0.3
            return

        # Kelly fraction: f* = (mu - r) / sigma^2
        self.kelly_fraction = (mu - self.risk_free_rate) / sigma_sq

        # Clip to reasonable range (-2 to 2)
        self.kelly_fraction = np.clip(self.kelly_fraction, -2, 2)

        # Half-Kelly for safety
        self.half_kelly = self.kelly_fraction / 2

        self.kelly_history.append(self.kelly_fraction)

        # Kelly stability
        if len(self.kelly_history) >= 5:
            kelly_std = np.std(list(self.kelly_history))
        else:
            kelly_std = 1.0

        # Generate signal from Kelly direction
        if abs(self.kelly_fraction) > 0.5:
            self.signal = 1 if self.kelly_fraction > 0 else -1

            # Confidence based on Kelly magnitude and stability
            magnitude_conf = min(abs(self.kelly_fraction) / 2, 1.0)
            stability_conf = 1 / (1 + kelly_std)
            self.confidence = magnitude_conf * stability_conf

        elif abs(self.kelly_fraction) > 0.1:
            self.signal = 1 if self.kelly_fraction > 0 else -1
            self.confidence = 0.4

        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(357)
class MultiFractionalBrownian(BaseFormula):
    """
    ID 357: Multi-Fractional Brownian Motion H(t)

    Academic Reference:
        - Peltier & Levy Vehel (1995) "Multifractional Brownian motion"
        - Coeurjolly (2005) "Identification of multifractional Brownian motion"

    Extension of fBm where H varies with time:
        B_H(t)(t) - Hurst exponent is a function of time

    This captures:
        - Regime changes (H jumps between values)
        - Gradual market structure changes
        - Time-varying predictability

    Trading Application:
        - Track H(t) trajectory
        - Predict future regime from H(t) trend
    """

    NAME = "MultiFractionalBrownian"
    CATEGORY = "timescale_invariance"
    DESCRIPTION = "Time-varying Hurst H(t) for regime prediction"

    def __init__(self, lookback: int = 300, **kwargs):
        super().__init__(lookback, **kwargs)
        self.local_window = kwargs.get('local_window', 50)
        self.h_t_series = deque(maxlen=100)  # H(t) over time
        self.h_derivative = 0.0  # Rate of change of H
        self.predicted_h = 0.5

    def _local_hurst(self, data: np.ndarray) -> float:
        """Estimate local Hurst using DFA"""
        n = len(data)
        if n < 16:
            return 0.5

        # Profile (cumulative sum of mean-adjusted)
        profile = np.cumsum(data - np.mean(data))

        # DFA at multiple scales
        scales = [4, 8, 16]
        log_scale = []
        log_fluct = []

        for s in scales:
            if n < s * 2:
                continue

            n_segments = n // s
            fluctuations = []

            for v in range(n_segments):
                segment = profile[v*s:(v+1)*s]
                # Detrend (linear)
                x = np.arange(s)
                trend = np.polyfit(x, segment, 1)
                detrended = segment - np.polyval(trend, x)
                fluct = np.sqrt(np.mean(detrended**2))
                fluctuations.append(fluct)

            if fluctuations:
                avg_fluct = np.mean(fluctuations)
                if avg_fluct > 0:
                    log_scale.append(np.log(s))
                    log_fluct.append(np.log(avg_fluct))

        if len(log_scale) >= 2:
            H = np.polyfit(log_scale, log_fluct, 1)[0]
            return np.clip(H, 0, 1)

        return 0.5

    def _compute(self) -> None:
        if len(self.returns) < self.local_window + 20:
            return

        returns = self._returns_array()

        # Compute local H for current window
        current_h = self._local_hurst(returns[-self.local_window:])
        self.h_t_series.append(current_h)

        # Compute H derivative (trend in H)
        if len(self.h_t_series) >= 10:
            h_array = np.array(list(self.h_t_series)[-10:])
            x = np.arange(len(h_array))
            self.h_derivative = np.polyfit(x, h_array, 1)[0]

            # Predict next H
            self.predicted_h = current_h + self.h_derivative * 5
            self.predicted_h = np.clip(self.predicted_h, 0.3, 0.7)
        else:
            self.predicted_h = current_h

        # Trading decision based on current and predicted H
        if self.predicted_h > 0.55:
            # Predicting trending regime
            recent = np.mean(returns[-5:])
            self.signal = 1 if recent > 0 else -1

            # Higher confidence if H is rising
            if self.h_derivative > 0.01:
                self.confidence = 0.7
            else:
                self.confidence = 0.5

        elif self.predicted_h < 0.45:
            # Predicting mean-reverting regime
            z = returns[-1] / (np.std(returns) + 1e-10)
            self.signal = -1 if z > 1 else (1 if z < -1 else 0)

            # Higher confidence if H is falling
            if self.h_derivative < -0.01:
                self.confidence = 0.7
            else:
                self.confidence = 0.5

        else:
            # Transitional - trade cautiously based on H direction
            if abs(self.h_derivative) > 0.02:
                if self.h_derivative > 0:
                    # H rising = becoming trendy
                    recent = np.mean(returns[-3:])
                    self.signal = 1 if recent > 0 else -1
                else:
                    # H falling = becoming mean-reverting
                    z = returns[-1] / (np.std(returns) + 1e-10)
                    self.signal = -1 if z > 0.5 else (1 if z < -0.5 else 0)
                self.confidence = 0.4
            else:
                self.signal = 0
                self.confidence = 0.2


# =============================================================================
# INTEGRATED SYSTEMS (358-360)
# =============================================================================

@FormulaRegistry.register(358)
class AdaptiveTimeScale(BaseFormula):
    """
    ID 358: Adaptive Time-Scale Selection

    This formula combines multiple time-scale analysis methods to:
        1. Detect optimal trading frequency
        2. Adapt strategy to current market regime
        3. Dynamically adjust holding period

    Combines:
        - Hurst exponent (regime detection)
        - Variance ratio (optimal horizon)
        - Wavelet energy (dominant scale)
    """

    NAME = "AdaptiveTimeScale"
    CATEGORY = "timescale_invariance"
    DESCRIPTION = "Master controller for time-scale adaptation"

    def __init__(self, lookback: int = 300, **kwargs):
        super().__init__(lookback, **kwargs)
        self.optimal_scale = 10
        self.regime = 'random'
        self.confidence_by_scale = {}

    def _quick_hurst(self, data: np.ndarray) -> float:
        """Fast Hurst via variance ratio"""
        if len(data) < 20:
            return 0.5
        var1 = np.var(data)
        if var1 == 0:
            return 0.5
        agg = np.array([np.sum(data[i*4:(i+1)*4]) for i in range(len(data)//4)])
        var4 = np.var(agg) if len(agg) > 2 else var1 * 4
        H = np.log(var4 / var1 / 4) / (2 * np.log(4)) + 0.5 if var1 > 0 else 0.5
        return np.clip(H, 0, 1)

    def _variance_ratio(self, data: np.ndarray, q: int) -> float:
        """Variance ratio for horizon q"""
        if len(data) < q * 3:
            return 1.0
        var1 = np.var(data)
        if var1 == 0:
            return 1.0
        q_rets = np.array([np.sum(data[i:i+q]) for i in range(len(data)-q+1)])
        var_q = np.var(q_rets)
        return var_q / (q * var1)

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()

        # 1. Hurst exponent for regime
        H = self._quick_hurst(returns[-64:])

        if H > 0.55:
            self.regime = 'trending'
        elif H < 0.45:
            self.regime = 'mean_reverting'
        else:
            self.regime = 'random'

        # 2. Variance ratios for optimal scale
        scales = [2, 4, 8, 16, 32]
        self.confidence_by_scale = {}

        for s in scales:
            vr = self._variance_ratio(returns, s)
            # Higher deviation from 1 = more predictable
            predictability = abs(vr - 1)
            self.confidence_by_scale[s] = predictability

        # Find optimal scale
        if self.confidence_by_scale:
            self.optimal_scale = max(self.confidence_by_scale,
                                    key=self.confidence_by_scale.get)

        # 3. Generate signal based on regime and optimal scale
        s = self.optimal_scale
        recent_return = np.sum(returns[-s:]) if len(returns) >= s else np.sum(returns)
        expected_std = np.std(returns) * np.sqrt(s)
        z = recent_return / (expected_std + 1e-10)

        if self.regime == 'trending':
            self.signal = 1 if recent_return > 0 else -1
            self.confidence = min((H - 0.5) * 3 + self.confidence_by_scale.get(s, 0), 1.0)

        elif self.regime == 'mean_reverting':
            self.signal = -1 if z > 1.5 else (1 if z < -1.5 else 0)
            self.confidence = min((0.5 - H) * 3 + self.confidence_by_scale.get(s, 0), 1.0)

        else:
            # Random but check if any scale is predictable
            best_pred = max(self.confidence_by_scale.values()) if self.confidence_by_scale else 0
            if best_pred > 0.2:
                vr_at_best = self._variance_ratio(returns, self.optimal_scale)
                if vr_at_best > 1.1:
                    self.signal = 1 if recent_return > 0 else -1
                elif vr_at_best < 0.9:
                    self.signal = -1 if z > 1 else (1 if z < -1 else 0)
                else:
                    self.signal = 0
                self.confidence = best_pred
            else:
                self.signal = 0
                self.confidence = 0.2


@FormulaRegistry.register(359)
class ScaleInvariantMomentum(BaseFormula):
    """
    ID 359: Scale-Invariant Momentum

    Traditional momentum works at one time scale.
    This formula computes momentum at multiple scales and:
        1. Identifies scales where momentum is significant
        2. Combines signals weighted by scale reliability
        3. Produces scale-invariant momentum signal
    """

    NAME = "ScaleInvariantMomentum"
    CATEGORY = "timescale_invariance"
    DESCRIPTION = "Momentum that works across all time scales"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.scales = [2, 4, 8, 16, 32, 64]
        self.momentum_by_scale = {}
        self.combined_momentum = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 70:
            return

        returns = self._returns_array()

        # Compute momentum at each scale
        self.momentum_by_scale = {}
        weights = []
        signals = []

        for s in self.scales:
            if len(returns) < s * 2:
                continue

            # Aggregate returns at scale s
            n = len(returns) // s
            agg = np.array([np.sum(returns[i*s:(i+1)*s]) for i in range(n)])

            if len(agg) < 3:
                continue

            # Momentum = normalized recent aggregate return
            recent = agg[-1]
            mean = np.mean(agg[:-1])
            std = np.std(agg[:-1])

            if std > 0:
                z = (recent - mean) / std
                self.momentum_by_scale[s] = z

                # Weight by significance
                weight = min(abs(z) / 2, 1.0)
                weights.append(weight)
                signals.append(np.sign(z) * weight)

        if not signals:
            self.signal = 0
            self.confidence = 0.3
            return

        # Combined momentum (weighted average of signals)
        total_weight = sum(weights)
        if total_weight > 0:
            self.combined_momentum = sum(signals) / total_weight
        else:
            self.combined_momentum = 0

        # Generate signal
        if abs(self.combined_momentum) > 0.5:
            self.signal = 1 if self.combined_momentum > 0 else -1
            self.confidence = min(abs(self.combined_momentum), 1.0)
        elif abs(self.combined_momentum) > 0.2:
            self.signal = 1 if self.combined_momentum > 0 else -1
            self.confidence = 0.4
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(360)
class UnifiedTimeScaleAnalyzer(BaseFormula):
    """
    ID 360: Unified Time-Scale Analyzer

    Master formula that combines all time-scale invariance concepts:
        1. Hurst exponent for regime detection
        2. Wavelet variance for scale analysis
        3. Optimal holding period calculation
        4. Adaptive strategy selection

    Outputs:
        - Optimal trading time scale
        - Regime (trending/mean-reverting/random)
        - Confidence-weighted signal
    """

    NAME = "UnifiedTimeScaleAnalyzer"
    CATEGORY = "timescale_invariance"
    DESCRIPTION = "Complete time-scale analysis system"

    def __init__(self, lookback: int = 300, **kwargs):
        super().__init__(lookback, **kwargs)

        # State
        self.hurst = 0.5
        self.regime = 'random'
        self.optimal_scale = 8
        self.strategy = 'neutral'  # 'momentum', 'mean_reversion', 'neutral'

        # Analysis outputs
        self.variance_ratios = {}
        self.scale_reliability = {}

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()

        # === 1. Hurst Exponent ===
        # Using variance ratio method for speed
        var1 = np.var(returns[-64:])
        if var1 > 0:
            agg_8 = np.array([np.sum(returns[-64:][i*8:(i+1)*8]) for i in range(8)])
            var_8 = np.var(agg_8) if len(agg_8) > 1 else var1 * 8
            self.hurst = np.log(var_8 / var1 / 8) / (2 * np.log(8)) + 0.5
            self.hurst = np.clip(self.hurst, 0.2, 0.8)

        # Determine regime
        if self.hurst > 0.55:
            self.regime = 'trending'
            self.strategy = 'momentum'
        elif self.hurst < 0.45:
            self.regime = 'mean_reverting'
            self.strategy = 'mean_reversion'
        else:
            self.regime = 'random'
            self.strategy = 'neutral'

        # === 2. Optimal Scale via Variance Ratio ===
        scales = [2, 4, 8, 16, 32]
        best_scale = 8
        best_deviation = 0

        self.variance_ratios = {}
        for s in scales:
            if len(returns) < s * 4:
                continue

            q_rets = np.array([np.sum(returns[i:i+s]) for i in range(len(returns)-s+1)])
            vr = np.var(q_rets) / (s * var1) if var1 > 0 else 1
            self.variance_ratios[s] = vr

            deviation = abs(vr - 1)
            if deviation > best_deviation:
                best_deviation = deviation
                best_scale = s

        self.optimal_scale = best_scale

        # === 3. Scale Reliability ===
        self.scale_reliability = {}
        for s, vr in self.variance_ratios.items():
            if self.regime == 'trending' and vr > 1:
                reliability = min((vr - 1) * 2, 1.0)
            elif self.regime == 'mean_reverting' and vr < 1:
                reliability = min((1 - vr) * 2, 1.0)
            else:
                reliability = 0.3
            self.scale_reliability[s] = reliability

        # === 4. Generate Signal ===
        s = self.optimal_scale
        recent = np.sum(returns[-s:]) if len(returns) >= s else np.sum(returns)
        std = np.std(returns) * np.sqrt(s)
        z = recent / (std + 1e-10)

        if self.strategy == 'momentum':
            self.signal = 1 if recent > 0 else -1
            self.confidence = min(
                (self.hurst - 0.5) * 3 * self.scale_reliability.get(s, 0.5),
                1.0
            )

        elif self.strategy == 'mean_reversion':
            if abs(z) > 1.5:
                self.signal = -1 if z > 0 else 1
                self.confidence = min(
                    (0.5 - self.hurst) * 3 * self.scale_reliability.get(s, 0.5),
                    1.0
                )
            else:
                self.signal = 0
                self.confidence = 0.3

        else:
            # Neutral - only trade if scale shows strong predictability
            if best_deviation > 0.2:
                vr = self.variance_ratios.get(s, 1.0)
                if vr > 1.1:
                    self.signal = 1 if recent > 0 else -1
                elif vr < 0.9:
                    self.signal = -1 if z > 1 else (1 if z < -1 else 0)
                else:
                    self.signal = 0
                self.confidence = min(best_deviation, 1.0) * 0.7
            else:
                self.signal = 0
                self.confidence = 0.2


# Export all classes
__all__ = [
    # Hurst Variants (347-349)
    'AdaptiveHurstExponent',
    'MultifractalDFA',
    'TimeVaryingHurst',

    # Wavelet Analysis (350-352)
    'MODWTWavelet',
    'WaveletVarianceAnalysis',
    'WaveletCoherence',

    # Optimal Sampling (353-354)
    'VolatilitySignaturePlot',
    'OptimalHoldingPeriod',

    # Adaptive Strategies (355-357)
    'AdaptiveOUHalfLife',
    'RollingKellyCriterion',
    'MultiFractionalBrownian',

    # Integrated Systems (358-360)
    'AdaptiveTimeScale',
    'ScaleInvariantMomentum',
    'UnifiedTimeScaleAnalyzer',
]
