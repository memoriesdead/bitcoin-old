"""
RenTech Signal Processing Module
================================

Formula IDs: 72011-72030

This module applies speech recognition and signal processing techniques
to financial time series - the core innovation that made RenTech famous.

Key insight: Price charts are waveforms. The same techniques used to
recognize spoken words can identify patterns in market data.

Components:
- Dynamic Time Warping (DTW) - Pattern matching regardless of speed
- Spectral Analysis (FFT) - Detect cyclical patterns
- Wavelet Decomposition - Multi-scale pattern detection
- Autocorrelation/Cross-correlation - Lead/lag relationships
"""

from .dtw_matcher import (
    DTWMatcher,
    PatternLibrary,
    SimilarityScore,
    # Formula implementations
    DTWPatternSignal,        # 72011
    DTWBreakoutSignal,       # 72012
    DTWReversalSignal,       # 72013
    DTWMomentumSignal,       # 72014
    DTWEnsembleSignal,       # 72015
)

from .spectral import (
    SpectralAnalyzer,
    FFTFeatures,
    CyclicalPattern,
    # Formula implementations
    FFTCycleSignal,          # 72016
    DominantFrequencySignal, # 72017
    SpectralMomentumSignal,  # 72018
    PhaseAnalysisSignal,     # 72019
    SpectralEnsembleSignal,  # 72020
)

from .wavelet import (
    WaveletDecomposer,
    MultiScaleFeatures,
    WaveletCoefficients,
    # Formula implementations
    WaveletTrendSignal,      # 72021
    WaveletNoiseSignal,      # 72022
    WaveletBreakoutSignal,   # 72023
    WaveletMomentumSignal,   # 72024
    MultiScaleSignal,        # 72025
    CrossScaleSignal,        # 72026
    WaveletRegimeSignal,     # 72027
    WaveletVolatilitySignal, # 72028
    WaveletCorrelationSignal,# 72029
    WaveletEnsembleSignal,   # 72030
)

__all__ = [
    # DTW
    'DTWMatcher', 'PatternLibrary', 'SimilarityScore',
    'DTWPatternSignal', 'DTWBreakoutSignal', 'DTWReversalSignal',
    'DTWMomentumSignal', 'DTWEnsembleSignal',
    # Spectral
    'SpectralAnalyzer', 'FFTFeatures', 'CyclicalPattern',
    'FFTCycleSignal', 'DominantFrequencySignal', 'SpectralMomentumSignal',
    'PhaseAnalysisSignal', 'SpectralEnsembleSignal',
    # Wavelet
    'WaveletDecomposer', 'MultiScaleFeatures', 'WaveletCoefficients',
    'WaveletTrendSignal', 'WaveletNoiseSignal', 'WaveletBreakoutSignal',
    'WaveletMomentumSignal', 'MultiScaleSignal', 'CrossScaleSignal',
    'WaveletRegimeSignal', 'WaveletVolatilitySignal',
    'WaveletCorrelationSignal', 'WaveletEnsembleSignal',
]
