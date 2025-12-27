"""
Signal Aggregator - Timeframe-Adaptive Mathematical Engine
===========================================================

Aggregates signals across multiple scales:
1. Wavelet decomposition for scale separation
2. Entropy-based confidence weighting
3. Consensus building across scales

This module answers: "What is the signal and how confident are we?"
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

from .math_primitives import (
    tae_004_consensus,
    tae_004_weighted_consensus,
    tae_002_shannon_entropy,
)


@dataclass
class ScaleSignal:
    """Signal at a specific wavelet scale."""
    scale: int                    # Wavelet scale index
    direction: int               # +1, -1, 0
    strength: float              # Signal strength
    entropy: float               # Signal entropy at this scale
    confidence: float            # Confidence (1 - normalized_entropy)


@dataclass
class AggregatedSignal:
    """Final aggregated signal across all scales."""
    direction: int               # +1 LONG, -1 SHORT, 0 HOLD
    confidence: float            # Confidence (0-1)
    consensus: float             # Multi-scale consensus (0-1)
    scale_signals: List[ScaleSignal] = field(default_factory=list)
    dominant_scale: int = -1     # Scale with highest confidence

    @property
    def tradeable(self) -> bool:
        return self.direction != 0 and self.confidence > 0.5 and self.consensus > 0.5


class WaveletDecomposer:
    """
    Decomposes price/signal into multiple resolution levels using DWT.

    Separates:
    - High frequency (noise/HFT)
    - Medium frequency (intraday)
    - Low frequency (trend)
    """

    def __init__(self, wavelet: str = 'db4', levels: int = 5):
        """
        Initialize wavelet decomposer.

        Args:
            wavelet: Wavelet family ('db4', 'haar', 'sym4', etc.)
            levels: Number of decomposition levels
        """
        self.wavelet = wavelet
        self.levels = levels

        # Scale meanings (approximate for 1-second data)
        self.scale_names = {
            0: 'trend',      # Approximation coefficients
            1: 'position',   # ~32 candles
            2: 'swing',      # ~16 candles
            3: 'intraday',   # ~8 candles
            4: 'scalping',   # ~4 candles
            5: 'noise',      # ~2 candles
        }

    def decompose(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose data into multiple scales.

        Args:
            data: Time series data

        Returns:
            Dict of scale name -> coefficients
        """
        if not HAS_PYWT:
            # Fallback: simple moving average decomposition
            return self._simple_decompose(data)

        if len(data) < 2 ** self.levels:
            # Not enough data for full decomposition
            actual_levels = max(1, int(np.log2(len(data))))
        else:
            actual_levels = self.levels

        try:
            coeffs = pywt.wavedec(data, self.wavelet, level=actual_levels)
        except Exception:
            return self._simple_decompose(data)

        result = {'trend': coeffs[0]}

        # Detail coefficients (from coarse to fine)
        detail_names = ['position', 'swing', 'intraday', 'scalping', 'noise']
        for i, name in enumerate(detail_names):
            if i + 1 < len(coeffs):
                result[name] = coeffs[i + 1]
            else:
                result[name] = np.array([])

        return result

    def _simple_decompose(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Fallback decomposition using moving averages."""
        result = {}

        # Trend: long MA
        if len(data) >= 32:
            result['trend'] = self._rolling_mean(data, 32)
        else:
            result['trend'] = data

        # Different scale MAs
        for name, window in [('position', 16), ('swing', 8), ('intraday', 4), ('scalping', 2)]:
            if len(data) >= window:
                smoothed = self._rolling_mean(data, window)
                # Detail = difference from next smoother level
                result[name] = data[-len(smoothed):] - smoothed
            else:
                result[name] = np.array([])

        if len(data) >= 2:
            noise_smoothed = self._rolling_mean(data, 2)
            result['noise'] = data[-len(noise_smoothed):] - noise_smoothed
        else:
            result['noise'] = data

        return result

    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Simple rolling mean."""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def reconstruct(self, coeffs: Dict[str, np.ndarray]) -> np.ndarray:
        """Reconstruct signal from coefficients."""
        if not HAS_PYWT:
            # Fallback: sum all components
            return sum(c for c in coeffs.values() if len(c) > 0)

        # Prepare coefficients list
        coeff_list = [coeffs.get('trend', np.array([]))]
        for name in ['position', 'swing', 'intraday', 'scalping', 'noise']:
            if name in coeffs and len(coeffs[name]) > 0:
                coeff_list.append(coeffs[name])

        try:
            return pywt.waverec(coeff_list, self.wavelet)
        except Exception:
            return sum(c for c in coeffs.values() if len(c) > 0)


class ConfidenceWeighter:
    """
    Weights signals by entropy-based confidence.

    Low entropy scales get higher weight.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def calculate_confidence(self, coefficients: np.ndarray) -> float:
        """
        Calculate confidence from wavelet coefficients.

        Confidence = 1 - normalized_entropy.
        """
        if len(coefficients) < 5:
            return 0.5

        entropy = tae_002_shannon_entropy(coefficients, self.n_bins)
        max_entropy = np.log(self.n_bins)
        normalized = entropy / max_entropy if max_entropy > 0 else 1.0

        return 1.0 - min(1.0, normalized)

    def get_scale_weights(
        self,
        decomposed: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Get weights for each scale based on confidence.

        Returns normalized weights summing to 1.
        """
        weights = {}
        for name, coeffs in decomposed.items():
            if len(coeffs) > 0:
                weights[name] = self.calculate_confidence(coeffs)
            else:
                weights[name] = 0.0

        # Normalize
        total = sum(weights.values())
        if total > 1e-10:
            weights = {k: v/total for k, v in weights.items()}

        return weights


class ConsensusBuilder:
    """
    Builds consensus signal across scales.

    When scales agree -> high confidence.
    When scales conflict -> low confidence.
    """

    def __init__(self):
        pass

    def extract_direction(self, coefficients: np.ndarray) -> int:
        """
        Extract direction from wavelet coefficients.

        Uses recent trend in coefficients.
        IMPROVED: More sensitive threshold for faster signal detection.
        """
        if len(coefficients) < 2:
            return 0

        # Use last few coefficients
        recent = coefficients[-min(5, len(coefficients)):]

        # Direction based on trend
        if len(recent) >= 2:
            trend = np.polyfit(range(len(recent)), recent, 1)[0]
            # More sensitive threshold (0.001 instead of 0.01)
            # Normalized by coefficient std for scale-invariance
            std = np.std(coefficients)
            threshold = 0.001 * (1 + std) if std > 0 else 0.001
            if trend > threshold:
                return 1   # Bullish
            elif trend < -threshold:
                return -1  # Bearish

        # Also check recent mean vs historical mean
        if len(coefficients) >= 10:
            recent_mean = np.mean(coefficients[-5:])
            hist_mean = np.mean(coefficients[:-5])
            if recent_mean > hist_mean * 1.05:
                return 1
            elif recent_mean < hist_mean * 0.95:
                return -1

        return 0  # Neutral

    def build_consensus(
        self,
        decomposed: Dict[str, np.ndarray],
        weights: Dict[str, float]
    ) -> Tuple[float, int, List[ScaleSignal]]:
        """
        Build consensus across all scales.

        Returns:
            Tuple of (consensus_score, final_direction, scale_signals)
        """
        directions = []
        weight_list = []
        scale_signals = []

        for name, coeffs in decomposed.items():
            if len(coeffs) < 2:
                continue

            direction = self.extract_direction(coeffs)
            weight = weights.get(name, 0.0)

            # Calculate strength from coefficient magnitude
            strength = np.abs(coeffs[-1]) / (np.std(coeffs) + 1e-10)
            strength = min(1.0, strength / 3.0)  # Normalize

            # Calculate entropy
            entropy = tae_002_shannon_entropy(coeffs, 10)
            max_entropy = np.log(10)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

            scale_signal = ScaleSignal(
                scale=list(decomposed.keys()).index(name),
                direction=direction,
                strength=strength,
                entropy=normalized_entropy,
                confidence=1.0 - normalized_entropy
            )
            scale_signals.append(scale_signal)

            if direction != 0:
                directions.append(direction)
                weight_list.append(weight * scale_signal.confidence)

        if not directions:
            return 0.0, 0, scale_signals

        # Calculate weighted consensus
        consensus, final_direction = tae_004_weighted_consensus(
            np.array(directions),
            np.array(weight_list)
        )

        return consensus, final_direction, scale_signals


class SignalAggregator:
    """
    Main signal aggregation engine.

    Combines wavelet decomposition, confidence weighting,
    and consensus building to produce final trading signal.
    """

    def __init__(self, wavelet: str = 'db4', levels: int = 5):
        """
        Initialize signal aggregator.

        Args:
            wavelet: Wavelet type for decomposition
            levels: Number of decomposition levels
        """
        self.decomposer = WaveletDecomposer(wavelet, levels)
        self.weighter = ConfidenceWeighter()
        self.consensus_builder = ConsensusBuilder()

        # History
        self.signal_history: List[AggregatedSignal] = []

    def aggregate(self, data: np.ndarray) -> AggregatedSignal:
        """
        Aggregate signals from multi-scale decomposition.

        Args:
            data: Time series data (price or returns)

        Returns:
            AggregatedSignal with direction, confidence, consensus
        """
        if len(data) < 4:
            return AggregatedSignal(
                direction=0,
                confidence=0.0,
                consensus=0.0
            )

        # Step 1: Wavelet decomposition
        decomposed = self.decomposer.decompose(data)

        # Step 2: Calculate confidence weights
        weights = self.weighter.get_scale_weights(decomposed)

        # Step 3: Build consensus
        consensus, direction, scale_signals = self.consensus_builder.build_consensus(
            decomposed, weights
        )

        # Step 4: Calculate overall confidence
        # IMPROVED: More aggressive confidence for 20x leverage
        confidences = [s.confidence for s in scale_signals]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # Previous formula was too conservative (consensus * avg = very low)
        # New formula: base confidence from avg_confidence, boosted by consensus
        # This produces more reasonable confidence levels (0.4-0.7 range)
        if consensus > 0.5 and avg_confidence > 0.3:
            # Strong consensus: boost confidence
            final_confidence = 0.5 + (avg_confidence * 0.3) + (consensus * 0.15)
        elif consensus > 0.3:
            # Moderate consensus: moderate confidence
            final_confidence = 0.4 + (avg_confidence * 0.2) + (consensus * 0.1)
        else:
            # Low consensus: use dampened formula
            final_confidence = consensus * avg_confidence * 1.5

        # Cap at 0.85 (room for QUIET_WHALE boost to push to 0.95)
        final_confidence = min(0.85, max(0.1, final_confidence))

        # Find dominant scale
        dominant_scale = -1
        if scale_signals:
            dominant = max(scale_signals, key=lambda s: s.confidence)
            dominant_scale = dominant.scale

        signal = AggregatedSignal(
            direction=direction,
            confidence=final_confidence,
            consensus=consensus,
            scale_signals=scale_signals,
            dominant_scale=dominant_scale
        )

        self.signal_history.append(signal)
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-500:]

        return signal

    def aggregate_returns(
        self,
        prices: np.ndarray,
        window: int = 100
    ) -> AggregatedSignal:
        """
        Aggregate signals from price returns.

        Args:
            prices: Price time series
            window: Window of recent data to use

        Returns:
            AggregatedSignal
        """
        if len(prices) < 10:
            return AggregatedSignal(direction=0, confidence=0.0, consensus=0.0)

        # Calculate returns
        recent = prices[-window:] if len(prices) > window else prices
        returns = np.diff(recent) / recent[:-1]

        return self.aggregate(returns)

    def get_consistency(self, window: int = 50) -> float:
        """
        Get signal consistency over recent history.

        Returns:
            Consistency score (0-1)
        """
        if len(self.signal_history) < 10:
            return 0.5

        recent = self.signal_history[-window:]
        directions = [s.direction for s in recent]

        # Use consensus formula
        return tae_004_consensus(np.array(directions))

    def get_scale_strengths(self) -> Dict[str, float]:
        """Get average signal strength per scale from history."""
        if not self.signal_history:
            return {}

        scale_strengths: Dict[str, List[float]] = {}

        for signal in self.signal_history[-100:]:
            for ss in signal.scale_signals:
                scale_name = self.decomposer.scale_names.get(ss.scale, f'scale_{ss.scale}')
                if scale_name not in scale_strengths:
                    scale_strengths[scale_name] = []
                scale_strengths[scale_name].append(ss.strength)

        return {k: np.mean(v) for k, v in scale_strengths.items()}

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information."""
        latest = self.signal_history[-1] if self.signal_history else None
        return {
            'latest_direction': latest.direction if latest else 0,
            'latest_confidence': latest.confidence if latest else 0.0,
            'latest_consensus': latest.consensus if latest else 0.0,
            'consistency': self.get_consistency(),
            'scale_strengths': self.get_scale_strengths(),
            'history_length': len(self.signal_history),
        }
