"""
RenTech Non-Linear Pattern Detection Module
===========================================

Formula IDs: 72031-72050

This module implements non-linear pattern recognition techniques that
go beyond simple linear z-scores to capture complex market relationships.

Key insight: Markets are non-linear systems. Linear statistics miss
the most profitable patterns.

Components:
- Kernel PCA - Non-linear dimensionality reduction
- Isolation Forest - Anomaly detection for unusual market states
- DBSCAN Clustering - Find natural regime clusters
- Polynomial Features - Interaction terms
"""

from .kernel_features import (
    KernelPCA,
    KernelFeatureExtractor,
    PolynomialInteractions,
    # Formula implementations
    KernelPCASignal,         # 72031
    RBFKernelSignal,         # 72032
    PolynomialKernelSignal,  # 72033
    InteractionSignal,       # 72034
    NonlinearMomentumSignal, # 72035
    NonlinearMeanRevSignal,  # 72036
    KernelRegimeSignal,      # 72037
    NonlinearTrendSignal,    # 72038
    KernelVolatilitySignal,  # 72039
    KernelEnsembleSignal,    # 72040
)

from .anomaly_detector import (
    IsolationForestDetector,
    LocalOutlierDetector,
    AnomalyScorer,
    # Formula implementations
    IsolationAnomalySignal,  # 72041
    LOFAnomalySignal,        # 72042
    ExtremeMoveSignal,       # 72043
    StructuralBreakSignal,   # 72044
    AnomalyRegimeSignal,     # 72045
    ClusterAnomalySignal,    # 72046
    DistributionShiftSignal, # 72047
    TailRiskSignal,          # 72048
    BlackSwanSignal,         # 72049
    AnomalyEnsembleSignal,   # 72050
)

__all__ = [
    # Kernel
    'KernelPCA', 'KernelFeatureExtractor', 'PolynomialInteractions',
    'KernelPCASignal', 'RBFKernelSignal', 'PolynomialKernelSignal',
    'InteractionSignal', 'NonlinearMomentumSignal', 'NonlinearMeanRevSignal',
    'KernelRegimeSignal', 'NonlinearTrendSignal', 'KernelVolatilitySignal',
    'KernelEnsembleSignal',
    # Anomaly
    'IsolationForestDetector', 'LocalOutlierDetector', 'AnomalyScorer',
    'IsolationAnomalySignal', 'LOFAnomalySignal', 'ExtremeMoveSignal',
    'StructuralBreakSignal', 'AnomalyRegimeSignal', 'ClusterAnomalySignal',
    'DistributionShiftSignal', 'TailRiskSignal', 'BlackSwanSignal',
    'AnomalyEnsembleSignal',
]
