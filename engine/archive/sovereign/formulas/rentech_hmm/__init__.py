"""
RenTech HMM Module - Trained Hidden Markov Models
=================================================

Formula IDs: 72001-72010

This module implements TRUE Hidden Markov Models trained with the
Baum-Welch algorithm, replacing rule-based regime detection.

RenTech Insight: Jim Simons hired speech recognition experts because
the statistical patterns in price data mirror those in audio signals.
HMMs were originally designed for speech recognition.

Components:
- Gaussian HMM with configurable states
- Baum-Welch training (EM algorithm)
- Viterbi decoding for state sequences
- Online state inference for live trading
"""

from .gaussian_hmm import (
    GaussianHMM,
    HMMConfig,
    HMMState,
    TrainedHMMSignal,
    # Formula implementations
    HMM3StateTrader,       # 72001
    HMM5StateTrader,       # 72002
    HMM7StateTrader,       # 72003
    HMMOptimalStateTrader, # 72004
    HMMTransitionTrader,   # 72005
)

from .state_decoder import (
    ViterbiDecoder,
    OnlineStateInference,
    StateTransitionAnalyzer,
    # Formula implementations
    ViterbiSignal,          # 72006
    TransitionProbSignal,   # 72007
    StateDurationSignal,    # 72008
    RegimePersistenceSignal,# 72009
    HMMEnsembleSignal,      # 72010
)

__all__ = [
    # Core
    'GaussianHMM',
    'HMMConfig',
    'HMMState',
    'TrainedHMMSignal',
    'ViterbiDecoder',
    'OnlineStateInference',
    'StateTransitionAnalyzer',
    # Formulas 72001-72010
    'HMM3StateTrader',
    'HMM5StateTrader',
    'HMM7StateTrader',
    'HMMOptimalStateTrader',
    'HMMTransitionTrader',
    'ViterbiSignal',
    'TransitionProbSignal',
    'StateDurationSignal',
    'RegimePersistenceSignal',
    'HMMEnsembleSignal',
]
