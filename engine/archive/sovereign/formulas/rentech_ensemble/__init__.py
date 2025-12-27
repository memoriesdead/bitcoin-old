"""
RenTech Ensemble Combination Module
===================================

Formula IDs: 72081-72099

This module implements RenTech's core insight: combine many weak signals
into one strong signal. No single pattern wins - the edge comes from
combining thousands of 50.5% edges into one 55%+ edge.

Key insight: The ensemble IS the strategy. Individual signals are
just ingredients.

Components:
- Gradient Boosted Ensemble - Decision tree boosting on formula outputs
- Stacking - Meta-learner on base model predictions
- Bayesian Model Averaging - Weight by posterior probability
- Master Ensemble - Final signal combination with regime adaptation
"""

from .gradient_ensemble import (
    SignalInput,
    EnsembleOutput,
    TreeNode,
    SimpleDecisionTree,
    GradientBoostingEnsemble,
    # Formula implementations
    GradientEnsembleSignal,      # 72081
    AdaptiveGradientEnsemble,    # 72082
    RegimeAwareEnsemble,         # 72083
    FeatureSelectedEnsemble,     # 72084
    GradientEnsembleWithDecay,   # 72085
)

from .stacked_meta import (
    BaseModelPrediction,
    StackedOutput,
    LinearMetaLearner,
    NeuralMetaLearner,
    StackedEnsemble,
    # Formula implementations
    LinearStackedSignal,                # 72086
    NeuralStackedSignal,                # 72087
    CrossValidatedStacker,              # 72088
    HierarchicalStacker,                # 72089
    StackedEnsembleWithUncertainty,     # 72090
)

from .bayesian_combiner import (
    BayesianOutput,
    BayesianModelAverager,
    ThompsonSamplingCombiner,
    OnlineBayesianEnsemble,
    # Formula implementations
    BayesianAverageSignal,      # 72091
    ThompsonSamplingSignal,     # 72092
    OnlineBayesianSignal,       # 72093
    BayesianSpikeAndSlab,       # 72094
    BayesianRegimeSwitch,       # 72095
)

from .master_ensemble import (
    MasterSignal,
    SignalGroup,
    MasterEnsemble,
    # Formula implementations
    MasterEnsembleSignal,       # 72096
    ConservativeMasterSignal,   # 72097
    AggressiveMasterSignal,     # 72098
    AdaptiveMasterSignal,       # 72099 - THE FINAL ADAPTIVE ENSEMBLE
)

__all__ = [
    # Data classes
    'SignalInput', 'EnsembleOutput', 'BaseModelPrediction', 'StackedOutput',
    'BayesianOutput', 'MasterSignal', 'SignalGroup',

    # Core components
    'TreeNode', 'SimpleDecisionTree', 'GradientBoostingEnsemble',
    'LinearMetaLearner', 'NeuralMetaLearner', 'StackedEnsemble',
    'BayesianModelAverager', 'ThompsonSamplingCombiner', 'OnlineBayesianEnsemble',
    'MasterEnsemble',

    # Gradient Ensemble (72081-72085)
    'GradientEnsembleSignal', 'AdaptiveGradientEnsemble', 'RegimeAwareEnsemble',
    'FeatureSelectedEnsemble', 'GradientEnsembleWithDecay',

    # Stacking (72086-72090)
    'LinearStackedSignal', 'NeuralStackedSignal', 'CrossValidatedStacker',
    'HierarchicalStacker', 'StackedEnsembleWithUncertainty',

    # Bayesian (72091-72095)
    'BayesianAverageSignal', 'ThompsonSamplingSignal', 'OnlineBayesianSignal',
    'BayesianSpikeAndSlab', 'BayesianRegimeSwitch',

    # Master Ensemble (72096-72099)
    'MasterEnsembleSignal', 'ConservativeMasterSignal', 'AggressiveMasterSignal',
    'AdaptiveMasterSignal',
]

# Formula ID to Class mapping
FORMULA_IDS = {
    # Gradient Ensemble
    72081: GradientEnsembleSignal,
    72082: AdaptiveGradientEnsemble,
    72083: RegimeAwareEnsemble,
    72084: FeatureSelectedEnsemble,
    72085: GradientEnsembleWithDecay,

    # Stacked Meta-Learners
    72086: LinearStackedSignal,
    72087: NeuralStackedSignal,
    72088: CrossValidatedStacker,
    72089: HierarchicalStacker,
    72090: StackedEnsembleWithUncertainty,

    # Bayesian Combiners
    72091: BayesianAverageSignal,
    72092: ThompsonSamplingSignal,
    72093: OnlineBayesianSignal,
    72094: BayesianSpikeAndSlab,
    72095: BayesianRegimeSwitch,

    # Master Ensemble
    72096: MasterEnsembleSignal,
    72097: ConservativeMasterSignal,
    72098: AggressiveMasterSignal,
    72099: AdaptiveMasterSignal,
}
