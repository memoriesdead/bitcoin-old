"""
Wavelet Decomposition for Multi-Scale Pattern Detection
=======================================================

Formula IDs: 72021-72030

Wavelets decompose signals into multiple time scales simultaneously.
Unlike FFT which loses time information, wavelets preserve both
time and frequency localization.

RenTech insight: Patterns exist at multiple timescales. A 5-day pattern
can be embedded in a 30-day pattern. Wavelets capture both.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


@dataclass
class WaveletCoefficients:
    """Wavelet decomposition coefficients."""
    approximation: np.ndarray  # Low-frequency trend
    details: List[np.ndarray]  # High-frequency details at each level
    levels: int


@dataclass
class MultiScaleFeatures:
    """Features extracted from wavelet analysis."""
    trend_direction: int  # From approximation
    trend_strength: float
    detail_energy: List[float]  # Energy at each scale
    dominant_scale: int  # Scale with most energy
    cross_scale_correlation: float


@dataclass
class WaveletSignal:
    """Signal from wavelet analysis."""
    direction: int
    confidence: float
    trend_direction: int
    detail_direction: int
    scale_alignment: float
    features: MultiScaleFeatures


class WaveletDecomposer:
    """
    Discrete Wavelet Transform for multi-scale analysis.

    Implements Haar wavelet (simplest but effective) and
    Daubechies wavelets.
    """

    def __init__(self, wavelet: str = 'haar', max_level: int = 5):
        self.wavelet = wavelet
        self.max_level = max_level

    def _haar_lowpass(self, data: np.ndarray) -> np.ndarray:
        """Haar wavelet low-pass filter."""
        n = len(data)
        if n % 2 != 0:
            data = np.append(data, data[-1])
        return (data[::2] + data[1::2]) / np.sqrt(2)

    def _haar_highpass(self, data: np.ndarray) -> np.ndarray:
        """Haar wavelet high-pass filter."""
        n = len(data)
        if n % 2 != 0:
            data = np.append(data, data[-1])
        return (data[::2] - data[1::2]) / np.sqrt(2)

    def _db4_coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Daubechies-4 filter coefficients."""
        h = np.array([
            (1 + np.sqrt(3)) / (4 * np.sqrt(2)),
            (3 + np.sqrt(3)) / (4 * np.sqrt(2)),
            (3 - np.sqrt(3)) / (4 * np.sqrt(2)),
            (1 - np.sqrt(3)) / (4 * np.sqrt(2)),
        ])
        g = np.array([h[3], -h[2], h[1], -h[0]])
        return h, g

    def decompose(self, data: np.ndarray) -> WaveletCoefficients:
        """
        Perform wavelet decomposition.

        Returns approximation (trend) and detail coefficients.
        """
        coeffs = []
        current = data.copy()

        for level in range(self.max_level):
            if len(current) < 4:
                break

            if self.wavelet == 'haar':
                detail = self._haar_highpass(current)
                current = self._haar_lowpass(current)
            else:
                # Daubechies-like decomposition
                h, g = self._db4_coeffs()
                n = len(current)
                padded = np.pad(current, (len(h) - 1, len(h) - 1), mode='edge')
                detail = np.convolve(padded, g[::-1], mode='valid')[::2][:n // 2]
                current = np.convolve(padded, h[::-1], mode='valid')[::2][:n // 2]

            coeffs.append(detail)

        return WaveletCoefficients(
            approximation=current,
            details=coeffs,
            levels=len(coeffs),
        )

    def extract_features(self, data: np.ndarray) -> MultiScaleFeatures:
        """Extract multi-scale features from wavelet decomposition."""
        wc = self.decompose(data)

        # Trend from approximation
        trend = wc.approximation
        if len(trend) >= 2:
            trend_direction = 1 if trend[-1] > trend[0] else -1
            trend_strength = abs(trend[-1] - trend[0]) / (np.std(trend) + 1e-10)
        else:
            trend_direction = 0
            trend_strength = 0.0

        # Energy at each scale
        detail_energy = [np.sum(d ** 2) for d in wc.details]

        # Dominant scale (most energy)
        dominant_scale = np.argmax(detail_energy) if detail_energy else 0

        # Cross-scale correlation
        if len(wc.details) >= 2:
            # Correlation between adjacent scales
            correlations = []
            for i in range(len(wc.details) - 1):
                d1 = wc.details[i]
                d2 = wc.details[i + 1]
                # Align lengths
                min_len = min(len(d1), len(d2))
                if min_len > 0:
                    corr = np.corrcoef(d1[:min_len], d2[:min_len])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            cross_scale_corr = np.mean(correlations) if correlations else 0.0
        else:
            cross_scale_corr = 0.0

        return MultiScaleFeatures(
            trend_direction=trend_direction,
            trend_strength=min(5.0, trend_strength),
            detail_energy=detail_energy,
            dominant_scale=dominant_scale,
            cross_scale_correlation=cross_scale_corr,
        )

    def generate_signal(self, data: np.ndarray) -> WaveletSignal:
        """Generate trading signal from wavelet analysis."""
        features = self.extract_features(data)
        wc = self.decompose(data)

        # Detail direction from finest scale
        if wc.details:
            finest = wc.details[0]
            detail_direction = 1 if np.mean(finest[-5:]) > 0 else -1
        else:
            detail_direction = 0

        # Scale alignment: trend and detail same direction
        scale_alignment = 1.0 if features.trend_direction == detail_direction else 0.0

        # Direction: follow trend when aligned, detail when not
        if scale_alignment > 0.5:
            direction = features.trend_direction
            confidence = 0.7 * features.trend_strength / 5.0
        else:
            direction = detail_direction
            confidence = 0.4

        return WaveletSignal(
            direction=direction,
            confidence=min(1.0, confidence),
            trend_direction=features.trend_direction,
            detail_direction=detail_direction,
            scale_alignment=scale_alignment,
            features=features,
        )


# =============================================================================
# FORMULA IMPLEMENTATIONS (72021-72030)
# =============================================================================

class WaveletTrendSignal:
    """
    Formula 72021: Wavelet Trend Signal

    Trades based on wavelet approximation (trend).
    Filters out noise to find underlying direction.
    """

    FORMULA_ID = 72021

    def __init__(self, level: int = 4):
        self.decomposer = WaveletDecomposer(max_level=level)

    def generate_signal(self, prices: np.ndarray) -> WaveletSignal:
        returns = np.diff(np.log(prices + 1e-10))
        base = self.decomposer.generate_signal(returns)

        # Focus on trend only
        if base.trend_direction != 0:
            direction = base.trend_direction
            confidence = base.features.trend_strength / 5.0
        else:
            direction = 0
            confidence = 0.0

        return WaveletSignal(
            direction=direction,
            confidence=min(1.0, confidence),
            trend_direction=base.trend_direction,
            detail_direction=base.detail_direction,
            scale_alignment=base.scale_alignment,
            features=base.features,
        )


class WaveletNoiseSignal:
    """
    Formula 72022: Wavelet Noise Signal

    Uses detail coefficients to measure market noise.
    High noise = stay flat, low noise = trade trend.
    """

    FORMULA_ID = 72022

    def __init__(self):
        self.decomposer = WaveletDecomposer()

    def generate_signal(self, prices: np.ndarray) -> WaveletSignal:
        returns = np.diff(np.log(prices + 1e-10))
        features = self.decomposer.extract_features(returns)
        wc = self.decomposer.decompose(returns)

        # Noise level from high-frequency details
        if features.detail_energy:
            noise_level = sum(features.detail_energy[:2]) / (sum(features.detail_energy) + 1e-10)
        else:
            noise_level = 1.0

        if noise_level < 0.3:
            # Low noise - trade the trend
            direction = features.trend_direction
            confidence = (0.3 - noise_level) * 2
        else:
            # High noise - stay flat
            direction = 0
            confidence = 0.0

        return WaveletSignal(
            direction=direction,
            confidence=confidence,
            trend_direction=features.trend_direction,
            detail_direction=0,
            scale_alignment=1.0 - noise_level,
            features=features,
        )


class WaveletBreakoutSignal:
    """
    Formula 72023: Wavelet Breakout Signal

    Detects breakouts using sudden energy increase in details.
    """

    FORMULA_ID = 72023

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.decomposer = WaveletDecomposer()
        self.energy_history: List[float] = []

    def generate_signal(self, prices: np.ndarray) -> WaveletSignal:
        returns = np.diff(np.log(prices + 1e-10))
        features = self.decomposer.extract_features(returns)

        current_energy = sum(features.detail_energy)
        self.energy_history.append(current_energy)

        if len(self.energy_history) > self.lookback:
            self.energy_history = self.energy_history[-self.lookback:]

        if len(self.energy_history) < 5:
            return WaveletSignal(
                direction=0, confidence=0.0,
                trend_direction=0, detail_direction=0,
                scale_alignment=0.0, features=features
            )

        # Breakout = energy spike
        avg_energy = np.mean(self.energy_history[:-1])
        std_energy = np.std(self.energy_history[:-1])

        energy_zscore = (current_energy - avg_energy) / (std_energy + 1e-10)

        if energy_zscore > 2.0:
            # Breakout detected - trade in trend direction
            direction = features.trend_direction
            confidence = min(1.0, energy_zscore / 4.0)
        else:
            direction = 0
            confidence = 0.0

        return WaveletSignal(
            direction=direction,
            confidence=confidence,
            trend_direction=features.trend_direction,
            detail_direction=0,
            scale_alignment=energy_zscore / 4.0,
            features=features,
        )


class WaveletMomentumSignal:
    """
    Formula 72024: Wavelet Momentum Signal

    Measures momentum across multiple scales.
    Aligned momentum across scales = stronger signal.
    """

    FORMULA_ID = 72024

    def __init__(self):
        self.decomposer = WaveletDecomposer()

    def generate_signal(self, prices: np.ndarray) -> WaveletSignal:
        returns = np.diff(np.log(prices + 1e-10))
        wc = self.decomposer.decompose(returns)
        features = self.decomposer.extract_features(returns)

        # Direction at each scale
        scale_directions = []

        # Trend direction
        if len(wc.approximation) >= 2:
            scale_directions.append(1 if wc.approximation[-1] > wc.approximation[0] else -1)

        # Detail directions
        for detail in wc.details:
            if len(detail) >= 2:
                scale_directions.append(1 if np.mean(detail[-3:]) > 0 else -1)

        if not scale_directions:
            return WaveletSignal(
                direction=0, confidence=0.0,
                trend_direction=0, detail_direction=0,
                scale_alignment=0.0, features=features
            )

        # Alignment = fraction of scales agreeing
        avg_direction = np.mean(scale_directions)
        alignment = abs(avg_direction)

        if alignment > 0.5:
            direction = 1 if avg_direction > 0 else -1
            confidence = alignment
        else:
            direction = 0
            confidence = 0.0

        return WaveletSignal(
            direction=direction,
            confidence=confidence,
            trend_direction=features.trend_direction,
            detail_direction=scale_directions[-1] if scale_directions else 0,
            scale_alignment=alignment,
            features=features,
        )


class MultiScaleSignal:
    """
    Formula 72025: Multi-Scale Signal

    Trades based on dominant scale energy.
    Different scales = different holding periods.
    """

    FORMULA_ID = 72025

    def __init__(self):
        self.decomposer = WaveletDecomposer(max_level=6)

    def generate_signal(self, prices: np.ndarray) -> WaveletSignal:
        returns = np.diff(np.log(prices + 1e-10))
        features = self.decomposer.extract_features(returns)
        wc = self.decomposer.decompose(returns)

        dominant = features.dominant_scale

        # Direction from dominant scale
        if dominant < len(wc.details):
            detail = wc.details[dominant]
            direction = 1 if np.mean(detail[-3:]) > 0 else -1
        else:
            direction = features.trend_direction

        # Confidence from relative energy
        if features.detail_energy:
            total_energy = sum(features.detail_energy)
            dominant_ratio = features.detail_energy[dominant] / (total_energy + 1e-10)
            confidence = dominant_ratio
        else:
            confidence = 0.0

        return WaveletSignal(
            direction=direction,
            confidence=confidence,
            trend_direction=features.trend_direction,
            detail_direction=direction,
            scale_alignment=confidence,
            features=features,
        )


class CrossScaleSignal:
    """
    Formula 72026: Cross-Scale Correlation Signal

    Trades when scales are correlated (regime stable).
    Stays flat when scales diverge (regime change).
    """

    FORMULA_ID = 72026

    def __init__(self):
        self.decomposer = WaveletDecomposer()

    def generate_signal(self, prices: np.ndarray) -> WaveletSignal:
        returns = np.diff(np.log(prices + 1e-10))
        features = self.decomposer.extract_features(returns)

        corr = features.cross_scale_correlation

        if corr > 0.5:
            # Scales correlated - regime stable, trade trend
            direction = features.trend_direction
            confidence = corr
        elif corr < -0.3:
            # Negative correlation - regime change, fade
            direction = -features.trend_direction
            confidence = abs(corr)
        else:
            direction = 0
            confidence = 0.0

        return WaveletSignal(
            direction=direction,
            confidence=confidence,
            trend_direction=features.trend_direction,
            detail_direction=0,
            scale_alignment=corr,
            features=features,
        )


class WaveletRegimeSignal:
    """
    Formula 72027: Wavelet Regime Signal

    Identifies regime changes using wavelet energy distribution.
    """

    FORMULA_ID = 72027

    def __init__(self, lookback: int = 30):
        self.lookback = lookback
        self.decomposer = WaveletDecomposer()
        self.energy_dist_history: List[np.ndarray] = []

    def generate_signal(self, prices: np.ndarray) -> WaveletSignal:
        returns = np.diff(np.log(prices + 1e-10))
        features = self.decomposer.extract_features(returns)

        if features.detail_energy:
            total = sum(features.detail_energy) + 1e-10
            dist = np.array([e / total for e in features.detail_energy])
            self.energy_dist_history.append(dist)

        if len(self.energy_dist_history) > self.lookback:
            self.energy_dist_history = self.energy_dist_history[-self.lookback:]

        if len(self.energy_dist_history) < 5:
            return WaveletSignal(
                direction=0, confidence=0.0,
                trend_direction=0, detail_direction=0,
                scale_alignment=0.0, features=features
            )

        # Regime change = shift in energy distribution
        recent = np.mean(self.energy_dist_history[-3:], axis=0)
        older = np.mean(self.energy_dist_history[:-3], axis=0)

        # KL divergence as regime change measure
        kl_div = np.sum(recent * np.log((recent + 1e-10) / (older + 1e-10)))

        if kl_div > 0.5:
            # Regime changing - stay flat
            direction = 0
            confidence = 0.0
        else:
            # Stable regime - trade trend
            direction = features.trend_direction
            confidence = 1.0 - kl_div

        return WaveletSignal(
            direction=direction,
            confidence=confidence,
            trend_direction=features.trend_direction,
            detail_direction=0,
            scale_alignment=1.0 - kl_div,
            features=features,
        )


class WaveletVolatilitySignal:
    """
    Formula 72028: Wavelet Volatility Signal

    Uses wavelet energy as volatility measure.
    Low volatility = trade trend, high volatility = cautious.
    """

    FORMULA_ID = 72028

    def __init__(self):
        self.decomposer = WaveletDecomposer()
        self.vol_history: List[float] = []

    def generate_signal(self, prices: np.ndarray) -> WaveletSignal:
        returns = np.diff(np.log(prices + 1e-10))
        features = self.decomposer.extract_features(returns)

        wavelet_vol = np.sqrt(sum(features.detail_energy))
        self.vol_history.append(wavelet_vol)

        if len(self.vol_history) > 100:
            self.vol_history = self.vol_history[-100:]

        if len(self.vol_history) < 10:
            return WaveletSignal(
                direction=0, confidence=0.0,
                trend_direction=0, detail_direction=0,
                scale_alignment=0.0, features=features
            )

        vol_percentile = np.sum(np.array(self.vol_history) < wavelet_vol) / len(self.vol_history)

        if vol_percentile < 0.3:
            # Low vol - trade trend confidently
            direction = features.trend_direction
            confidence = 0.8
        elif vol_percentile < 0.7:
            # Medium vol - trade trend cautiously
            direction = features.trend_direction
            confidence = 0.4
        else:
            # High vol - reduce exposure
            direction = features.trend_direction
            confidence = 0.2

        return WaveletSignal(
            direction=direction,
            confidence=confidence,
            trend_direction=features.trend_direction,
            detail_direction=0,
            scale_alignment=1.0 - vol_percentile,
            features=features,
        )


class WaveletCorrelationSignal:
    """
    Formula 72029: Wavelet Correlation Signal

    Correlates wavelet coefficients with price changes.
    Uses this for prediction.
    """

    FORMULA_ID = 72029

    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.decomposer = WaveletDecomposer()

    def generate_signal(self, prices: np.ndarray) -> WaveletSignal:
        if len(prices) < self.lookback + 5:
            features = MultiScaleFeatures(0, 0.0, [], 0, 0.0)
            return WaveletSignal(
                direction=0, confidence=0.0,
                trend_direction=0, detail_direction=0,
                scale_alignment=0.0, features=features
            )

        returns = np.diff(np.log(prices + 1e-10))
        features = self.decomposer.extract_features(returns)
        wc = self.decomposer.decompose(returns)

        # Correlate wavelet energy with forward returns
        # (In practice, you'd do this over training data)
        # Here we use a simple heuristic

        if wc.details:
            # Recent high-frequency energy predicts volatility
            recent_hf_energy = np.sum(wc.details[0][-5:] ** 2)
            trend_energy = np.sum(wc.approximation ** 2)

            ratio = recent_hf_energy / (trend_energy + 1e-10)

            if ratio < 0.3:
                # Trend dominates - follow trend
                direction = features.trend_direction
                confidence = 0.6
            else:
                # Noise dominates - mean revert
                direction = -features.trend_direction
                confidence = 0.4
        else:
            direction = 0
            confidence = 0.0

        return WaveletSignal(
            direction=direction,
            confidence=confidence,
            trend_direction=features.trend_direction,
            detail_direction=0,
            scale_alignment=0.0,
            features=features,
        )


class WaveletEnsembleSignal:
    """
    Formula 72030: Wavelet Ensemble Signal

    Combines all wavelet-based signals.
    """

    FORMULA_ID = 72030

    def __init__(self):
        self.signals = [
            WaveletTrendSignal(),
            WaveletNoiseSignal(),
            WaveletMomentumSignal(),
            MultiScaleSignal(),
            CrossScaleSignal(),
        ]

    def generate_signal(self, prices: np.ndarray) -> WaveletSignal:
        results = [s.generate_signal(prices) for s in self.signals]

        # Weighted vote
        total_dir = sum(r.direction * r.confidence for r in results)
        total_conf = sum(r.confidence for r in results)

        if total_conf > 0:
            avg_dir = total_dir / total_conf
            if avg_dir > 0.3:
                direction = 1
            elif avg_dir < -0.3:
                direction = -1
            else:
                direction = 0
            confidence = total_conf / len(self.signals)
        else:
            direction = 0
            confidence = 0.0

        return WaveletSignal(
            direction=direction,
            confidence=min(1.0, confidence),
            trend_direction=results[0].trend_direction,
            detail_direction=results[0].detail_direction,
            scale_alignment=np.mean([r.scale_alignment for r in results]),
            features=results[0].features,
        )
