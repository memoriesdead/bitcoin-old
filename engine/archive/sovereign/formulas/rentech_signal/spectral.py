"""
Spectral Analysis (FFT) for Pattern Detection
=============================================

Formula IDs: 72016-72020

Uses Fourier Transform to detect cyclical patterns in price data.
RenTech insight: Markets have hidden periodicities that repeat.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


@dataclass
class FFTFeatures:
    """Features extracted from FFT analysis."""
    dominant_frequency: float
    dominant_period: float  # In time units
    dominant_amplitude: float
    top_frequencies: List[Tuple[float, float]]  # (freq, amplitude)
    spectral_entropy: float
    spectral_centroid: float
    spectral_rolloff: float


@dataclass
class CyclicalPattern:
    """Detected cyclical pattern."""
    period: float
    amplitude: float
    phase: float
    significance: float  # Statistical significance


@dataclass
class SpectralSignal:
    """Signal from spectral analysis."""
    direction: int
    confidence: float
    current_phase: float  # Where in the cycle we are
    cycle_position: str  # 'bottom', 'rising', 'top', 'falling'
    dominant_period: float
    features: FFTFeatures


class SpectralAnalyzer:
    """
    FFT-based spectral analysis for cycle detection.

    Identifies dominant frequencies and predicts cycle position.
    """

    def __init__(self, min_period: int = 5, max_period: int = 100):
        self.min_period = min_period
        self.max_period = max_period
        self.dominant_cycles: List[CyclicalPattern] = []

    def compute_fft(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of data.

        Returns:
            frequencies, amplitudes
        """
        n = len(data)

        # Detrend data
        detrended = data - np.linspace(data[0], data[-1], n)

        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(n)
        windowed = detrended * window

        # Compute FFT
        fft = np.fft.fft(windowed)
        freqs = np.fft.fftfreq(n)

        # Get positive frequencies only
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        amplitudes = np.abs(fft[pos_mask]) * 2 / n

        return freqs, amplitudes

    def extract_features(self, data: np.ndarray) -> FFTFeatures:
        """Extract spectral features from data."""
        freqs, amps = self.compute_fft(data)

        if len(freqs) == 0:
            return FFTFeatures(
                dominant_frequency=0,
                dominant_period=0,
                dominant_amplitude=0,
                top_frequencies=[],
                spectral_entropy=0,
                spectral_centroid=0,
                spectral_rolloff=0,
            )

        # Filter by period range
        periods = 1 / (freqs + 1e-10)
        mask = (periods >= self.min_period) & (periods <= self.max_period)
        freqs_filtered = freqs[mask]
        amps_filtered = amps[mask]

        if len(freqs_filtered) == 0:
            freqs_filtered = freqs
            amps_filtered = amps

        # Dominant frequency
        max_idx = np.argmax(amps_filtered)
        dominant_freq = freqs_filtered[max_idx]
        dominant_amp = amps_filtered[max_idx]
        dominant_period = 1 / dominant_freq if dominant_freq > 0 else 0

        # Top frequencies
        top_indices = np.argsort(amps_filtered)[-5:][::-1]
        top_freqs = [(freqs_filtered[i], amps_filtered[i]) for i in top_indices]

        # Spectral entropy (measure of randomness)
        amps_norm = amps_filtered / (np.sum(amps_filtered) + 1e-10)
        spectral_entropy = -np.sum(amps_norm * np.log(amps_norm + 1e-10))

        # Spectral centroid (center of mass)
        spectral_centroid = np.sum(freqs_filtered * amps_filtered) / (np.sum(amps_filtered) + 1e-10)

        # Spectral rolloff (frequency below which 85% of energy)
        cumsum = np.cumsum(amps_filtered)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        spectral_rolloff = freqs_filtered[min(rolloff_idx, len(freqs_filtered) - 1)]

        return FFTFeatures(
            dominant_frequency=dominant_freq,
            dominant_period=dominant_period,
            dominant_amplitude=dominant_amp,
            top_frequencies=top_freqs,
            spectral_entropy=spectral_entropy,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
        )

    def detect_cycles(self, data: np.ndarray, significance_threshold: float = 2.0) -> List[CyclicalPattern]:
        """
        Detect significant cyclical patterns.

        Uses amplitude relative to noise floor for significance.
        """
        freqs, amps = self.compute_fft(data)

        # Noise floor estimate (median amplitude)
        noise_floor = np.median(amps)

        # Find peaks above threshold
        cycles = []
        for i in range(1, len(amps) - 1):
            if amps[i] > amps[i-1] and amps[i] > amps[i+1]:
                significance = amps[i] / (noise_floor + 1e-10)
                if significance >= significance_threshold:
                    period = 1 / freqs[i] if freqs[i] > 0 else 0
                    if self.min_period <= period <= self.max_period:
                        # Compute phase
                        phase = np.angle(np.fft.fft(data)[i])
                        cycles.append(CyclicalPattern(
                            period=period,
                            amplitude=amps[i],
                            phase=phase,
                            significance=significance,
                        ))

        # Sort by significance
        cycles.sort(key=lambda x: x.significance, reverse=True)
        self.dominant_cycles = cycles[:5]
        return self.dominant_cycles

    def get_cycle_position(self, data: np.ndarray) -> Tuple[float, str]:
        """
        Determine current position in dominant cycle.

        Returns:
            (phase_0_to_1, position_label)
        """
        if not self.dominant_cycles:
            self.detect_cycles(data)

        if not self.dominant_cycles:
            return 0.0, 'unknown'

        cycle = self.dominant_cycles[0]
        n = len(data)

        # Current phase in cycle
        phase = (n % cycle.period) / cycle.period

        # Map phase to position
        if phase < 0.25:
            position = 'bottom'
        elif phase < 0.5:
            position = 'rising'
        elif phase < 0.75:
            position = 'top'
        else:
            position = 'falling'

        return phase, position

    def generate_signal(self, data: np.ndarray) -> SpectralSignal:
        """Generate trading signal from spectral analysis."""
        features = self.extract_features(data)
        cycles = self.detect_cycles(data)
        phase, position = self.get_cycle_position(data)

        # Direction based on cycle position
        if position == 'bottom':
            direction = 1  # Buy at bottom
            confidence = 0.7
        elif position == 'rising':
            direction = 1  # Hold long
            confidence = 0.5
        elif position == 'top':
            direction = -1  # Sell at top
            confidence = 0.6
        elif position == 'falling':
            direction = -1  # Stay short
            confidence = 0.4
        else:
            direction = 0
            confidence = 0.0

        # Adjust confidence by cycle significance
        if cycles:
            confidence *= min(1.0, cycles[0].significance / 5.0)
        else:
            confidence = 0.0

        return SpectralSignal(
            direction=direction,
            confidence=confidence,
            current_phase=phase,
            cycle_position=position,
            dominant_period=features.dominant_period,
            features=features,
        )


# =============================================================================
# FORMULA IMPLEMENTATIONS (72016-72020)
# =============================================================================

class FFTCycleSignal:
    """
    Formula 72016: FFT Cycle Signal

    Basic cycle detection using FFT.
    Trades based on position in dominant cycle.
    """

    FORMULA_ID = 72016

    def __init__(self, min_period: int = 7, max_period: int = 60):
        self.analyzer = SpectralAnalyzer(min_period=min_period, max_period=max_period)

    def generate_signal(self, prices: np.ndarray) -> SpectralSignal:
        returns = np.diff(np.log(prices + 1e-10))
        return self.analyzer.generate_signal(returns)


class DominantFrequencySignal:
    """
    Formula 72017: Dominant Frequency Signal

    Trades based on dominant frequency characteristics.
    Strong dominant frequency = trade the cycle.
    Weak/no dominant = stay flat.
    """

    FORMULA_ID = 72017

    def __init__(self):
        self.analyzer = SpectralAnalyzer()

    def generate_signal(self, prices: np.ndarray) -> SpectralSignal:
        returns = np.diff(np.log(prices + 1e-10))
        features = self.analyzer.extract_features(returns)

        # Strong dominant frequency = tradeable cycle
        if features.dominant_amplitude > 0.05:
            base_signal = self.analyzer.generate_signal(returns)
            return base_signal
        else:
            # No clear cycle - stay flat
            return SpectralSignal(
                direction=0,
                confidence=0.0,
                current_phase=0.0,
                cycle_position='unclear',
                dominant_period=features.dominant_period,
                features=features,
            )


class SpectralMomentumSignal:
    """
    Formula 72018: Spectral Momentum Signal

    Combines spectral analysis with momentum.
    Uses spectral centroid shift as momentum indicator.
    """

    FORMULA_ID = 72018

    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.analyzer = SpectralAnalyzer()
        self.prev_centroid: float = 0.0

    def generate_signal(self, prices: np.ndarray) -> SpectralSignal:
        returns = np.diff(np.log(prices + 1e-10))

        if len(returns) < self.lookback:
            return SpectralSignal(
                direction=0, confidence=0.0, current_phase=0.0,
                cycle_position='insufficient_data', dominant_period=0,
                features=FFTFeatures(0, 0, 0, [], 0, 0, 0)
            )

        features = self.analyzer.extract_features(returns[-self.lookback:])

        # Centroid shift indicates momentum
        centroid_change = features.spectral_centroid - self.prev_centroid
        self.prev_centroid = features.spectral_centroid

        if centroid_change > 0.001:
            direction = 1  # Higher frequency activity = bullish momentum
            confidence = min(1.0, abs(centroid_change) * 100)
        elif centroid_change < -0.001:
            direction = -1
            confidence = min(1.0, abs(centroid_change) * 100)
        else:
            direction = 0
            confidence = 0.0

        return SpectralSignal(
            direction=direction,
            confidence=confidence,
            current_phase=0.0,
            cycle_position='momentum',
            dominant_period=features.dominant_period,
            features=features,
        )


class PhaseAnalysisSignal:
    """
    Formula 72019: Phase Analysis Signal

    Analyzes phase relationships across multiple cycles.
    Trades when multiple cycles align.
    """

    FORMULA_ID = 72019

    def __init__(self):
        self.analyzer = SpectralAnalyzer()

    def generate_signal(self, prices: np.ndarray) -> SpectralSignal:
        returns = np.diff(np.log(prices + 1e-10))
        cycles = self.analyzer.detect_cycles(returns)

        if len(cycles) < 2:
            return SpectralSignal(
                direction=0, confidence=0.0, current_phase=0.0,
                cycle_position='no_cycles', dominant_period=0,
                features=self.analyzer.extract_features(returns)
            )

        # Check phase alignment
        phases = [c.phase for c in cycles[:3]]
        phase_std = np.std(phases)

        # Low phase std = cycles aligned
        if phase_std < 0.5:
            # Determine direction from average phase
            avg_phase = np.mean(phases)
            if -np.pi/2 < avg_phase < np.pi/2:
                direction = 1  # Cycles aligned bullish
            else:
                direction = -1  # Cycles aligned bearish
            confidence = 1.0 - phase_std
        else:
            direction = 0
            confidence = 0.0

        return SpectralSignal(
            direction=direction,
            confidence=confidence,
            current_phase=phases[0] if phases else 0.0,
            cycle_position='aligned' if direction != 0 else 'misaligned',
            dominant_period=cycles[0].period,
            features=self.analyzer.extract_features(returns),
        )


class SpectralEnsembleSignal:
    """
    Formula 72020: Spectral Ensemble Signal

    Combines multiple spectral analysis approaches.
    """

    FORMULA_ID = 72020

    def __init__(self):
        self.cycle_signal = FFTCycleSignal()
        self.freq_signal = DominantFrequencySignal()
        self.momentum_signal = SpectralMomentumSignal()
        self.phase_signal = PhaseAnalysisSignal()

    def generate_signal(self, prices: np.ndarray) -> SpectralSignal:
        signals = [
            self.cycle_signal.generate_signal(prices),
            self.freq_signal.generate_signal(prices),
            self.momentum_signal.generate_signal(prices),
            self.phase_signal.generate_signal(prices),
        ]

        # Weighted voting
        total_direction = sum(s.direction * s.confidence for s in signals)
        total_confidence = sum(s.confidence for s in signals)

        if total_confidence > 0:
            avg_direction = total_direction / total_confidence
            if avg_direction > 0.3:
                direction = 1
            elif avg_direction < -0.3:
                direction = -1
            else:
                direction = 0
            confidence = total_confidence / 4
        else:
            direction = 0
            confidence = 0.0

        return SpectralSignal(
            direction=direction,
            confidence=min(1.0, confidence),
            current_phase=signals[0].current_phase,
            cycle_position='ensemble',
            dominant_period=signals[0].dominant_period,
            features=signals[0].features,
        )
