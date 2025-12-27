"""
QLib Alpha - Point-in-Time ML Pipeline
=======================================

Concepts ported from Microsoft QLib (https://github.com/microsoft/qlib):
- Point-in-time data handling (no lookahead bias)
- Alpha expression framework
- LightGBM flow classifier
- Online incremental learning

Formula IDs: 70001-70010
"""

from .pit_handler import (
    PointInTimeHandler,
    PITFlowDatabase,
    validate_no_lookahead,
)

from .expression import (
    AlphaExpression,
    FlowMomentum,
    FlowAcceleration,
    FlowZScore,
    FlowSkew,
    FlowAutoCorr,
    FlowRegimeDetector,
    create_alpha_features,
)

from .lightgbm_flow import (
    LightGBMFlowClassifier,
    OnlineLightGBM,
)

from .online_learner import (
    OnlineLearner,
    IncrementalUpdater,
    EnsembleOnlineLearner,
)

__all__ = [
    # PIT Handler
    'PointInTimeHandler',
    'PITFlowDatabase',
    'validate_no_lookahead',

    # Alpha Expressions
    'AlphaExpression',
    'FlowMomentum',
    'FlowAcceleration',
    'FlowZScore',
    'FlowSkew',
    'FlowAutoCorr',
    'FlowRegimeDetector',
    'create_alpha_features',

    # LightGBM
    'LightGBMFlowClassifier',
    'OnlineLightGBM',

    # Online Learning
    'OnlineLearner',
    'IncrementalUpdater',
    'EnsembleOnlineLearner',
]

# Formula ID allocation
QLIB_FORMULA_IDS = {
    70001: 'FlowMomentum',
    70002: 'FlowAcceleration',
    70003: 'FlowZScore',
    70004: 'FlowSkew',
    70005: 'FlowAutoCorr',
    70006: 'LightGBMFlowClassifier',
    70007: 'OnlineLightGBM',
    70008: 'PointInTimeHandler',
    70009: 'AlphaEnsemble',
    70010: 'OnlineLearner',
}
