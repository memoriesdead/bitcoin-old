"""
TRADING FORMULAS - 481 Academic Formulas (70 Next-Gen Added)
=============================================================
Organized by category with unique IDs

ID Ranges:
    1-30:    Statistical (Bayesian, MLE, entropy)
    31-60:   Time Series (ARIMA, GARCH)
    61-100:  Machine Learning (ensemble, neural)
    101-130: Microstructure (Kyle, VPIN, OFI)
    131-150: Mean Reversion (OU, Z-score)
    151-170: Volatility (GARCH, rough vol)
    171-190: Regime Detection (HMM, CUSUM)
    191-210: Signal Processing (Kalman, wavelet)
    211-222: Risk Management (Kelly, VaR)
    239-258: Advanced HFT (MicroPrice, tick bars)
    259-268: Bitcoin Specific (OBI, cross-exchange)
    269-276: Bitcoin Derivatives (funding rate)
    277-282: Bitcoin Timing (session filters)
    283-284: Market Making (Avellaneda-Stoikov)
    285-290: Execution (Almgren-Chriss)
    295-299: Volume Scaling
    300-310: Academic Research
    311-319: Transaction Costs
    320-322: Exit Strategies (Leung, Trailing Stop)
    323-330: Compounding & Growth (Kelly, Optimal F, Almgren-Chriss, Avellaneda-Stoikov)
    331-340: Profitability Fixes (Edge, Frequency, Confluence, Drawdown, Regime)
    341-346: Advanced Microstructure (Quant-Level Research)
    347-360: Time-Scale Invariance (Academic Gold Standard)
        347: AdaptiveHurstExponent - Rolling H for regime detection
        348: MultifractalDFA - Multi-scale Hurst analysis
        349: TimeVaryingHurst - H(t) with regime change alerts
        350: MODWTWavelet - Maximal overlap wavelet decomposition
        351: WaveletVarianceAnalysis - Variance by time scale
        352: WaveletCoherence - Scale-dependent predictability
        353: VolatilitySignaturePlot - Optimal sampling frequency
        354: OptimalHoldingPeriod - Variance ratio optimal horizon
        355: AdaptiveOUHalfLife - Rolling OU mean-reversion timing
        356: RollingKellyCriterion - Adaptive position sizing
        357: MultiFractionalBrownian - Time-varying H(t) prediction
        358: AdaptiveTimeScale - Master time-scale controller
        359: ScaleInvariantMomentum - Momentum across all scales
        360: UnifiedTimeScaleAnalyzer - Complete time-scale system
    361-380: Multi-Scale Advanced (Complete Time-Scale Coverage)
        361: BaiPerronBreakDetector - Multiple structural break detection
        362: CUSUMScaleDetector - Multi-scale CUSUM change detection
        363: WBSChangepoint - Wild Binary Segmentation
        364: EntropyScaleSelector - Information-theoretic scale selection
        365: TransferEntropyScale - Cross-scale information flow
        366: DCCACoefficient - Detrended cross-correlation coefficient
        367: MultiscaleDCCA - Multifractal cross-correlation
        368: EppsEffectCorrector - Scale-dependent correlation correction
        369: TurbulentCascade - Turbulence-inspired volatility cascade
        370: RoughVolatilityEstimator - Rough volatility Hurst (H << 0.5)
        371: ContinuousVRFunction - VR(τ) as continuous function
        372: AutocorrDecayRate - Autocorrelation decay analysis
        373: ScaleDependentSharpe - Sharpe ratio scaling anomalies
        374: OptimalStoppingMR - Leung-Li optimal entry/exit
        375: GHETrading - Generalized Hurst Exponent signals
        376: CEEMDANDecomposition - CEEMDAN multi-scale decomposition
        377: WaveletPacketBestBasis - Entropy-based best basis
        378: LocallyStationaryWavelet - LSW non-stationary model
        379: SpectralDensityWhittle - Whittle MLE spectral estimation
        380: ACDDurationModel - ACD-inspired activity clustering
    381-400: Multi-Scale Advanced Part 2 (Complete Coverage)
        381: HorizonKelly - Horizon-dependent Kelly fraction
        382: ContinuousKellyHJB - HJB continuous Kelly
        383: FractionalKelly - Multi-scale fractional Kelly
        384: DrawdownConstrainedKelly - Kelly with DD constraint
        385: VolatilityScaledKelly - Volatility-scaled Kelly
        386: KyleLambdaScaling - Kyle lambda price impact
        387: AlmgrenChrissTiming - Optimal execution timing
        388: TransientImpactDecay - Impact decay analysis
        389: SquareRootImpact - Square-root impact law
        390: MarketResiliency - Market recovery speed
        391: MomentScaling - Moment scaling zeta(q)
        392: TailIndexStability - Tail index across scales
        393: ReturnAggregationTest - Aggregation bias test
        394: ScalePredictabilityLoss - Predictability decay rate
        395: UniversalityClass - Market universality class
        396: AdaptiveBandwidth - Adaptive smoothing bandwidth
        397: ScaleCoherence - Signal coherence across scales
        398: MultiScaleEnsemble - Ensemble from all scales
        399: TimeScaleFilter - Adaptive time-scale filter
        400: UnifiedScaleAnalyzer - MASTER formula for all scales
    401-402: Volume-Based Frequency (Dynamic from 24h volume)
    403-411: Bidirectional Trading (SHORT SELLING ENABLED)
        403: BidirectionalOUReversion - Mean reversion LONG and SHORT signals
        404: ExchangeInflowBearish - SHORT on exchange inflow spikes
        405: NVTOverboughtSignal - SHORT when NVT indicates overbought
        406: VPINToxicShort - SHORT on high VPIN (toxic flow)
        407: WhaleDistributionSignal - SHORT on whale distribution
        408: FundingRateArbitrage - SHORT on high positive funding
        409: MempoolPressureInversion - SHORT on high mempool (panic)
        410: FeeSpikeShortSignal - SHORT on fee spikes
        411: BidirectionalSignalAggregator - Combines all bidirectional signals
    412-481: Next-Gen Prediction Models (CUTTING EDGE AI/ML)
        412-420: Transformer/Deep Learning (TFT, Informer, Autoformer, CNN-Transformer)
        421-430: Rough Volatility (rBergomi, fBm, ARRV, Rough Heston)
        431-445: Optimal Execution (Almgren-Chriss GBM, HJB, DDQN, Queue-Reactive RL)
        446-460: MEV/Crypto Specific (Sandwich Detection, MEV Arbitrage, Liquidation Cascade)
        461-475: Advanced Microstructure (Cartea-Jaimungal, Guéant-Lehalle, Stoikov-Saglam)
        476-481: Signal Processing/Physics (Reservoir Computing, Liquid Time Constant, TDA, SNN)
    501-508: Universal Time-Scale Invariance (WORKS AT ANY TIMEFRAME)
        501: DirectionalChangeIntrinsicTime - Guillaume et al. (1997) event-based time
        502: PathSignatureTrading - Lyons et al. (2014) rough path signatures
        503: VariableLagCausality - Amornbunchornvej (2021) DTW-based causal discovery
        504: MultifractalDFATrading - Kantelhardt (2002) multi-scale Hurst
        505: ContinuousRegimeSwitching - Hamilton (1989) continuous regime probabilities
        506: WaveletMultiResolutionFusion - Daubechies (1992) multi-scale signal fusion
        507: RecursiveBayesianAdaptive - Kalman (1960) online parameter learning
        508: UniversalTimescaleController - MASTER controller combining ALL above
"""

from .base import BaseFormula, FormulaRegistry, FORMULA_REGISTRY

# Import all formula modules
from . import statistical
from . import timeseries
from . import machine_learning
from . import microstructure
from . import mean_reversion
from . import volatility
from . import regime
from . import signal_processing
from . import risk
from . import advanced_hft
from . import bitcoin_specific
from . import bitcoin_derivatives
from . import bitcoin_timing
from . import market_making
from . import execution
from . import volume_scaling
from . import academic_research
from . import adaptive_online
from . import advanced_prediction
from . import hft_volume
from . import gap_analysis
from . import transaction_costs
from . import renaissance_strategies
from . import exit_strategies
from . import compounding_strategies

# Profitability Fixes (IDs 331-340) - THE KEY TO MAKING MONEY
from . import edge_measurement      # 331, 336: Real edge from actual outcomes
from . import optimal_frequency     # 332, 337: High freq + high quality
from . import signal_confluence     # 333, 338: Condorcet voting
from . import drawdown_control      # 334, 339: Position sizing with DD limits
from . import regime_filter         # 335, 340: Trend-aware filtering

# Advanced Microstructure (IDs 341-346) - QUANT-LEVEL RESEARCH
from . import advanced_microstructure  # 341-346: Research-backed edge

# Time-Scale Invariance (IDs 347-360) - ACADEMIC GOLD STANDARD
from . import timescale_invariance     # 347-360: Multi-timeframe adaptation

# Multi-Scale Advanced (IDs 361-400) - COMPLETE TIME-SCALE COVERAGE
from . import multiscale_advanced      # 361-380: Structural breaks, DCCA, turbulence, Kelly
from . import multiscale_advanced_2    # 381-400: Price impact, return scaling, ensemble

# Volume-Based Frequency (IDs 401-402) - LIVE DATA, NO HARDCODING
from . import volume_frequency         # 401-402: Dynamic freq from 24h volume

# Bidirectional Trading (IDs 403-411) - SHORT SELLING ENABLED
from . import bidirectional            # 403-411: SHORT signals from blockchain bearish indicators

# Next-Gen Prediction Models (IDs 412-481) - CUTTING EDGE AI/ML
from . import next_gen                 # 412-481: Transformers, Rough Vol, MEV, RL Execution, Physics

# Universal Time-Scale Invariance (IDs 501-508) - WORKS AT ANY TIMEFRAME
from . import universal_timescale      # 501-508: Event-time, Signatures, MFDFA, Regime, Wavelet, Bayesian

# Blockchain Pipeline Signals (IDs 520-560) - ACADEMIC PEER-REVIEWED RESEARCH
# Based on: Kyle (1985), Easley/OHara (2012), Cont/Stoikov (2010), Almgren/Chriss (2001)
from . import blockchain_signals       # 520-560: Kyle Lambda, VPIN, OFI, NVT, MVRV, SOPR, Kelly, HMM, TRUE Price

__all__ = [
    "BaseFormula",
    "FormulaRegistry", 
    "FORMULA_REGISTRY",
]
