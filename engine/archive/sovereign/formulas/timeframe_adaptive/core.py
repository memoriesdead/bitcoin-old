"""
Core Engine - Timeframe-Adaptive Mathematical Engine
=====================================================

Main orchestration engine that combines all components:
- TimeframeSelector: Finds optimal timeframe
- ParameterController: Manages trading parameters
- SignalAggregator: Multi-scale signal combination
- ValidityMonitor: Edge decay detection

This is the primary interface for the adaptive system.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import logging

from .math_primitives import (
    compute_timeframe_score,
    get_optimal_timeframe_for_regime,
    get_decay_rate_for_regime,
)
from .timeframe_selector import TimeframeSelector, TimeframeSelection
from .parameter_controller import ParameterController, TradingParameters, DEFAULT_PRIORS
from .signal_aggregator import SignalAggregator, AggregatedSignal
from .validity_monitor import ValidityMonitor, EdgeEstimate, RegimeState, PerformanceSnapshot

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSignal:
    """Complete adaptive trading signal."""
    timestamp: float
    direction: int               # +1 LONG, -1 SHORT, 0 HOLD
    confidence: float            # Overall confidence (0-1)

    # Timeframe
    optimal_timeframe: float     # Selected τ* in seconds
    timeframe_confidence: float  # Confidence in timeframe selection

    # Parameters
    parameters: TradingParameters

    # Consensus
    consensus: float             # Multi-scale consensus (0-1)

    # Validity
    edge_strength: float         # Current edge strength
    is_valid: bool              # Should we trade?

    # Regime
    regime: str
    regime_confidence: float

    # Raw components
    aggregated_signal: Optional[AggregatedSignal] = None
    timeframe_selection: Optional[TimeframeSelection] = None

    @property
    def tradeable(self) -> bool:
        """Check if signal is tradeable."""
        return (
            self.direction != 0 and
            self.confidence > 0.5 and
            self.consensus > 0.5 and
            self.is_valid
        )

    @property
    def strength(self) -> float:
        """Signal strength = direction * confidence * consensus."""
        return self.direction * self.confidence * self.consensus


@dataclass
class EngineConfig:
    """Configuration for TimeframeAdaptiveEngine."""
    # Candidate timeframes
    candidate_timeframes: List[float] = field(default_factory=lambda: [
        1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0
    ])

    # Wavelet settings
    wavelet_type: str = 'db4'
    wavelet_levels: int = 5

    # Validity thresholds
    min_edge_strength: float = 0.3
    min_regime_duration: float = 60.0

    # Decay settings
    parameter_decay_interval: float = 60.0  # Decay params every N seconds

    # Minimum data requirements
    min_data_points: int = 50


class TimeframeAdaptiveEngine:
    """
    Main Timeframe-Adaptive Mathematical Engine.

    This engine solves: "What works for 1 second may not work for 2 seconds."

    Key innovations:
    1. Automatic optimal timeframe detection using information theory
    2. Parameter mean-reversion using Ornstein-Uhlenbeck process
    3. Multi-scale signal consensus from wavelet decomposition
    4. Edge half-life monitoring for validity detection
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        """
        Initialize the engine.

        Args:
            config: Engine configuration (uses defaults if None)
        """
        self.config = config or EngineConfig()

        # Core components
        self.timeframe_selector = TimeframeSelector(self.config.candidate_timeframes)
        self.parameter_controller = ParameterController(DEFAULT_PRIORS)
        self.signal_aggregator = SignalAggregator(
            self.config.wavelet_type,
            self.config.wavelet_levels
        )
        self.validity_monitor = ValidityMonitor(
            self.config.min_edge_strength,
            self.config.min_regime_duration
        )

        # State
        self.current_regime = 'unknown'
        self.regime_probabilities: Dict[str, float] = {}
        self.last_decay_time = time.time()
        self.last_signal: Optional[AdaptiveSignal] = None

        # Data buffers
        self.price_buffer: List[float] = []
        self.return_buffer: List[float] = []
        self.signal_buffer: List[float] = []

        # Statistics
        self.signals_generated = 0
        self.trades_processed = 0

        self._initialized = False

    def initialize(self) -> None:
        """Initialize the engine."""
        logger.info("Initializing TimeframeAdaptiveEngine")
        self._initialized = True

    def set_regime(
        self,
        regime: str,
        probabilities: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update current market regime from HMM.

        Args:
            regime: Current regime name
            probabilities: Optional probability distribution
        """
        self.current_regime = regime
        self.regime_probabilities = probabilities or {regime: 1.0}

        # Update components
        self.parameter_controller.set_regime(regime)
        self.validity_monitor.update_regime(regime, self.regime_probabilities)

    def add_price(self, price: float) -> None:
        """
        Add new price to buffer.

        Args:
            price: New price observation
        """
        self.price_buffer.append(price)

        # Calculate return
        if len(self.price_buffer) >= 2:
            ret = (self.price_buffer[-1] - self.price_buffer[-2]) / self.price_buffer[-2]
            self.return_buffer.append(ret)

        # Limit buffer size
        max_buffer = 5000
        if len(self.price_buffer) > max_buffer:
            self.price_buffer = self.price_buffer[-max_buffer:]
            self.return_buffer = self.return_buffer[-max_buffer:]

    def add_signal(self, signal_value: float) -> None:
        """
        Add external signal value to buffer.

        Args:
            signal_value: Signal from another engine (e.g., blockchain features)
        """
        self.signal_buffer.append(signal_value)
        if len(self.signal_buffer) > 5000:
            self.signal_buffer = self.signal_buffer[-5000:]

    def process(
        self,
        price: Optional[float] = None,
        external_signal: Optional[float] = None
    ) -> AdaptiveSignal:
        """
        Process new data and generate adaptive signal.

        Args:
            price: New price (optional if already added via add_price)
            external_signal: External signal value (optional)

        Returns:
            AdaptiveSignal with direction, confidence, and parameters
        """
        if not self._initialized:
            self.initialize()

        # Add new data
        if price is not None:
            self.add_price(price)
        if external_signal is not None:
            self.add_signal(external_signal)

        # Check minimum data requirements
        if len(self.price_buffer) < self.config.min_data_points:
            return self._create_hold_signal("Insufficient data")

        # Step 1: Apply parameter decay
        self._apply_decay_if_needed()

        # Step 2: Aggregate signal across scales
        prices = np.array(self.price_buffer)
        aggregated = self.signal_aggregator.aggregate_returns(prices)

        # Step 3: Select optimal timeframe
        tf_selection = self._select_timeframe()

        # Step 4: Get optimal parameters
        params = self.parameter_controller.get_optimal_parameters()

        # Step 5: Check validity
        is_valid, reasons = self.validity_monitor.check_validity()
        edge = self.validity_monitor.get_edge_estimate()

        # Step 6: Determine final direction and confidence
        direction = aggregated.direction
        confidence = aggregated.confidence

        # Adjust confidence based on validity
        if not is_valid:
            confidence *= 0.5
        if edge.current_strength < 0.5:
            confidence *= edge.current_strength

        # Create signal
        signal = AdaptiveSignal(
            timestamp=time.time(),
            direction=direction,
            confidence=confidence,
            optimal_timeframe=tf_selection.optimal_tau if tf_selection else 15.0,
            timeframe_confidence=tf_selection.confidence if tf_selection else 0.5,
            parameters=params,
            consensus=aggregated.consensus,
            edge_strength=edge.current_strength,
            is_valid=is_valid,
            regime=self.current_regime,
            regime_confidence=self.regime_probabilities.get(self.current_regime, 0.5),
            aggregated_signal=aggregated,
            timeframe_selection=tf_selection
        )

        self.last_signal = signal
        self.signals_generated += 1

        return signal

    def learn(self, pnl: float, params_used: Optional[Dict[str, float]] = None) -> None:
        """
        Learn from trade outcome.

        Args:
            pnl: Trade PnL
            params_used: Parameters used for the trade
        """
        self.trades_processed += 1

        # Update validity monitor
        self.validity_monitor.add_trade(pnl)

        # Update parameter controller
        if params_used:
            self.parameter_controller.update_from_trade(pnl, params_used)

    def _select_timeframe(self) -> Optional[TimeframeSelection]:
        """Select optimal timeframe."""
        if len(self.return_buffer) < 50:
            return None

        # Prepare signals and returns at different timeframes
        # For simplicity, use same data - in production, would resample
        signals_by_tau: Dict[float, np.ndarray] = {}
        returns_by_tau: Dict[float, np.ndarray] = {}

        returns = np.array(self.return_buffer)
        signals = np.array(self.signal_buffer) if self.signal_buffer else returns

        for tau in self.config.candidate_timeframes:
            # In production, would resample to tau-second intervals
            # For now, use raw data with different window sizes
            window = int(min(tau * 10, len(returns)))
            if window >= 10:
                signals_by_tau[tau] = signals[-window:]
                returns_by_tau[tau] = returns[-window:]

        if not signals_by_tau:
            return None

        return self.timeframe_selector.select(
            signals_by_tau=signals_by_tau,
            returns_by_tau=returns_by_tau,
            regime=self.current_regime
        )

    def _apply_decay_if_needed(self) -> None:
        """Apply parameter decay if interval has passed."""
        now = time.time()
        if now - self.last_decay_time >= self.config.parameter_decay_interval:
            self.parameter_controller.decay_step(
                dt=(now - self.last_decay_time) / self.config.parameter_decay_interval
            )
            self.last_decay_time = now

    def _create_hold_signal(self, reason: str = "") -> AdaptiveSignal:
        """Create a HOLD signal."""
        return AdaptiveSignal(
            timestamp=time.time(),
            direction=0,
            confidence=0.0,
            optimal_timeframe=get_optimal_timeframe_for_regime(self.current_regime),
            timeframe_confidence=0.0,
            parameters=TradingParameters(),
            consensus=0.0,
            edge_strength=1.0,
            is_valid=True,
            regime=self.current_regime,
            regime_confidence=0.5
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        return {
            'engine': {
                'initialized': self._initialized,
                'signals_generated': self.signals_generated,
                'trades_processed': self.trades_processed,
                'buffer_size': len(self.price_buffer),
            },
            'regime': {
                'current': self.current_regime,
                'probabilities': self.regime_probabilities,
            },
            'timeframe': self.timeframe_selector.get_consistency_score()
                if self.timeframe_selector.selection_history else 0.5,
            'parameters': self.parameter_controller.get_diagnostics(),
            'signal': self.signal_aggregator.get_diagnostics(),
            'validity': self.validity_monitor.get_diagnostics(),
            'last_signal': {
                'direction': self.last_signal.direction if self.last_signal else 0,
                'confidence': self.last_signal.confidence if self.last_signal else 0.0,
                'tradeable': self.last_signal.tradeable if self.last_signal else False,
            }
        }

    def reset(self) -> None:
        """Reset all state."""
        self.timeframe_selector = TimeframeSelector(self.config.candidate_timeframes)
        self.parameter_controller = ParameterController(DEFAULT_PRIORS)
        self.signal_aggregator = SignalAggregator(
            self.config.wavelet_type,
            self.config.wavelet_levels
        )
        self.validity_monitor = ValidityMonitor(
            self.config.min_edge_strength,
            self.config.min_regime_duration
        )

        self.current_regime = 'unknown'
        self.regime_probabilities = {}
        self.last_decay_time = time.time()
        self.last_signal = None

        self.price_buffer = []
        self.return_buffer = []
        self.signal_buffer = []

        self.signals_generated = 0
        self.trades_processed = 0
        self._initialized = False


def create_engine(
    candidate_timeframes: Optional[List[float]] = None,
    wavelet_type: str = 'db4',
    wavelet_levels: int = 5,
    min_edge_strength: float = 0.3
) -> TimeframeAdaptiveEngine:
    """
    Factory function to create configured engine.

    Args:
        candidate_timeframes: List of candidate timeframes in seconds
        wavelet_type: Wavelet type for decomposition
        wavelet_levels: Number of wavelet decomposition levels
        min_edge_strength: Minimum edge strength for validity

    Returns:
        Configured TimeframeAdaptiveEngine
    """
    config = EngineConfig(
        candidate_timeframes=candidate_timeframes or [1, 2, 5, 10, 15, 20, 30, 45, 60],
        wavelet_type=wavelet_type,
        wavelet_levels=wavelet_levels,
        min_edge_strength=min_edge_strength
    )
    return TimeframeAdaptiveEngine(config)
