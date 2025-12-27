"""
Formula Registry - Sovereign Engine
====================================

Central registry for all formula engines with ensemble voting.

Manages:
- Engine registration and initialization
- Tick distribution to all engines
- Ensemble voting on signals
- Feedback distribution for learning
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time
import numpy as np

from .base import BaseEngine, EngineState
from ..core.types import Tick, Signal, TradeOutcome
from ..core.config import EnginesConfig, EnsembleConfig


@dataclass
class EnsembleResult:
    """Result of ensemble voting."""
    final_signal: Signal
    agreement_level: str           # "unanimous", "majority", "minority", "conflict", "none"
    num_engines: int
    num_agreeing: int
    engine_signals: Dict[str, Signal] = field(default_factory=dict)
    engine_votes: Dict[str, int] = field(default_factory=dict)
    confidence_boost: float = 1.0


class EnsembleVoter:
    """
    Ensemble voting for combining signals from multiple engines.

    Voting Methods:
    - weighted_vote: Weight by engine configuration
    - majority: Simple majority wins
    - confidence_weighted: Weight by signal confidence
    """

    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.weights: Dict[str, float] = {}

    def set_weights(self, weights: Dict[str, float]):
        """Set engine weights for voting."""
        self.weights = weights

    def vote(
        self,
        signals: Dict[str, Signal],
        method: str = None
    ) -> EnsembleResult:
        """
        Combine signals from multiple engines.

        Args:
            signals: Dict of engine_name -> Signal
            method: Voting method override

        Returns:
            EnsembleResult with final signal and metadata
        """
        method = method or self.config.method

        if not signals:
            return EnsembleResult(
                final_signal=Signal(timestamp=time.time(), direction=0, confidence=0.0),
                agreement_level="none",
                num_engines=0,
                num_agreeing=0,
            )

        # Extract votes (direction) and confidences
        votes = {}
        confidences = {}
        for name, signal in signals.items():
            if signal.direction != 0:  # Only count non-HOLD signals
                votes[name] = signal.direction
                confidences[name] = signal.confidence

        # Count directions
        long_count = sum(1 for v in votes.values() if v > 0)
        short_count = sum(1 for v in votes.values() if v < 0)
        total_votes = long_count + short_count

        # Determine agreement level and final direction
        num_engines = len(signals)

        if total_votes == 0:
            # All HOLD
            return EnsembleResult(
                final_signal=Signal(timestamp=time.time(), direction=0, confidence=0.0),
                agreement_level="none",
                num_engines=num_engines,
                num_agreeing=0,
                engine_signals=signals,
                engine_votes={n: s.direction for n, s in signals.items()},
            )

        # Voting by method
        if method == "weighted_vote":
            direction, confidence = self._weighted_vote(signals, votes, confidences)
        elif method == "confidence_weighted":
            direction, confidence = self._confidence_weighted_vote(signals, votes, confidences)
        else:  # majority
            direction, confidence = self._majority_vote(signals, votes, confidences)

        # Determine agreement level
        agreeing = long_count if direction > 0 else short_count
        if agreeing == num_engines and total_votes == num_engines:
            agreement_level = "unanimous"
            confidence_boost = self.config.boost_unanimous
        elif agreeing >= self.config.min_agreement:
            agreement_level = "majority"
            confidence_boost = self.config.boost_majority
        elif total_votes == 1:
            agreement_level = "minority"
            confidence_boost = 1.0
        else:
            agreement_level = "conflict"
            confidence_boost = 0.8  # Reduce confidence on conflict

        # Apply confidence threshold
        if confidence < self.config.confidence_threshold:
            direction = 0
            confidence = 0.0
            agreement_level = "below_threshold"
            confidence_boost = 1.0

        # Build final signal
        final_confidence = min(1.0, confidence * confidence_boost)

        # Aggregate hints from signals
        avg_sl = np.mean([s.stop_loss for s in signals.values() if s.stop_loss > 0]) if signals else 0.005
        avg_tp = np.mean([s.take_profit for s in signals.values() if s.take_profit > 0]) if signals else 0.01
        avg_hold = np.mean([s.hold_seconds for s in signals.values() if s.hold_seconds > 0]) if signals else 60.0

        final_signal = Signal(
            timestamp=time.time(),
            direction=direction,
            confidence=final_confidence,
            source_engine="ensemble",
            stop_loss=float(avg_sl) if not np.isnan(avg_sl) else 0.005,
            take_profit=float(avg_tp) if not np.isnan(avg_tp) else 0.01,
            hold_seconds=float(avg_hold) if not np.isnan(avg_hold) else 60.0,
            votes={n: s.direction for n, s in signals.items()},
            vote_confidences={n: s.confidence for n, s in signals.items()},
        )

        return EnsembleResult(
            final_signal=final_signal,
            agreement_level=agreement_level,
            num_engines=num_engines,
            num_agreeing=agreeing,
            engine_signals=signals,
            engine_votes={n: s.direction for n, s in signals.items()},
            confidence_boost=confidence_boost,
        )

    def _weighted_vote(
        self,
        signals: Dict[str, Signal],
        votes: Dict[str, int],
        confidences: Dict[str, float]
    ) -> tuple:
        """Weighted voting by engine weights."""
        weighted_sum = 0.0
        total_weight = 0.0

        for name, direction in votes.items():
            weight = self.weights.get(name, 1.0)
            confidence = confidences.get(name, 0.5)
            weighted_sum += direction * weight * confidence
            total_weight += weight

        if total_weight == 0:
            return 0, 0.0

        avg_score = weighted_sum / total_weight
        direction = 1 if avg_score > 0 else (-1 if avg_score < 0 else 0)
        confidence = min(1.0, abs(avg_score))

        return direction, confidence

    def _majority_vote(
        self,
        signals: Dict[str, Signal],
        votes: Dict[str, int],
        confidences: Dict[str, float]
    ) -> tuple:
        """Simple majority voting."""
        long_count = sum(1 for v in votes.values() if v > 0)
        short_count = sum(1 for v in votes.values() if v < 0)

        if long_count > short_count:
            direction = 1
            agreeing_confidences = [c for n, c in confidences.items() if votes.get(n, 0) > 0]
        elif short_count > long_count:
            direction = -1
            agreeing_confidences = [c for n, c in confidences.items() if votes.get(n, 0) < 0]
        else:
            # Tie - go with highest confidence
            if confidences:
                best = max(confidences.items(), key=lambda x: x[1])
                direction = votes.get(best[0], 0)
                agreeing_confidences = [best[1]]
            else:
                return 0, 0.0

        confidence = np.mean(agreeing_confidences) if agreeing_confidences else 0.5
        return direction, float(confidence)

    def _confidence_weighted_vote(
        self,
        signals: Dict[str, Signal],
        votes: Dict[str, int],
        confidences: Dict[str, float]
    ) -> tuple:
        """Vote weighted by signal confidence."""
        long_weight = sum(confidences[n] for n, v in votes.items() if v > 0)
        short_weight = sum(confidences[n] for n, v in votes.items() if v < 0)

        if long_weight > short_weight:
            direction = 1
            confidence = long_weight / (long_weight + short_weight) if (long_weight + short_weight) > 0 else 0.5
        elif short_weight > long_weight:
            direction = -1
            confidence = short_weight / (long_weight + short_weight) if (long_weight + short_weight) > 0 else 0.5
        else:
            return 0, 0.0

        return direction, float(confidence)


class FormulaRegistry:
    """
    Central registry for all formula engines.

    Manages engine lifecycle:
    - Registration
    - Initialization
    - Tick processing
    - Ensemble voting
    - Learning feedback
    """

    def __init__(self, engines_config: EnginesConfig = None, ensemble_config: EnsembleConfig = None):
        self.engines_config = engines_config or EnginesConfig()
        self.ensemble_config = ensemble_config or EnsembleConfig()

        self.engines: Dict[str, BaseEngine] = {}
        self.voter = EnsembleVoter(ensemble_config)

        # Stats
        self.ticks_processed = 0
        self.signals_generated = 0
        self.last_result: Optional[EnsembleResult] = None

    def register(self, name: str, engine: BaseEngine):
        """
        Register an engine.

        Args:
            name: Engine name (e.g., "adaptive", "pattern", "rentech")
            engine: Engine instance implementing BaseEngine
        """
        self.engines[name] = engine

        # Set weight from config
        engine_cfg = getattr(self.engines_config, name, None)
        if engine_cfg:
            self.voter.weights[name] = engine_cfg.weight

    def initialize(self, configs: Dict[str, Dict[str, Any]] = None):
        """
        Initialize all registered engines.

        Args:
            configs: Dict of engine_name -> config dict
        """
        configs = configs or {}

        for name, engine in self.engines.items():
            config = configs.get(name, {})
            engine_cfg = getattr(self.engines_config, name, None)

            # Merge with engine config
            if engine_cfg:
                config['enabled'] = engine_cfg.enabled
                config['formulas'] = engine_cfg.formulas

            engine.initialize(config)

    def process(self, tick: Tick) -> Signal:
        """
        Process a tick through all engines and vote.

        Args:
            tick: Market data tick

        Returns:
            Final voted signal
        """
        self.ticks_processed += 1
        signals: Dict[str, Signal] = {}

        # Process through enabled engines
        for name, engine in self.engines.items():
            engine_cfg = getattr(self.engines_config, name, None)

            # Skip disabled engines
            if engine_cfg and not engine_cfg.enabled:
                continue

            try:
                signal = engine.process(tick)
                signals[name] = signal
            except Exception as e:
                print(f"[Registry] Error in {name}: {e}")
                # Continue with other engines

        # Vote
        result = self.voter.vote(signals)
        self.last_result = result

        if result.final_signal.tradeable:
            self.signals_generated += 1

        return result.final_signal

    def learn(self, outcome: TradeOutcome):
        """
        Distribute learning feedback to all engines.

        Args:
            outcome: Complete trade outcome
        """
        for name, engine in self.engines.items():
            try:
                engine.learn(outcome)
            except Exception as e:
                print(f"[Registry] Error learning in {name}: {e}")

    def get_engine(self, name: str) -> Optional[BaseEngine]:
        """Get engine by name."""
        return self.engines.get(name)

    def get_all_states(self) -> Dict[str, EngineState]:
        """Get states of all engines."""
        return {name: engine.get_state() for name, engine in self.engines.items()}

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        return {
            "ticks_processed": self.ticks_processed,
            "signals_generated": self.signals_generated,
            "engines": {
                name: engine.get_diagnostics()
                for name, engine in self.engines.items()
            },
            "last_result": {
                "agreement_level": self.last_result.agreement_level if self.last_result else None,
                "num_agreeing": self.last_result.num_agreeing if self.last_result else 0,
            }
        }

    def reset(self):
        """Reset all engines."""
        self.ticks_processed = 0
        self.signals_generated = 0
        self.last_result = None

        for engine in self.engines.values():
            engine.reset()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_registry(
    engines_config: EnginesConfig = None,
    ensemble_config: EnsembleConfig = None,
    include_engines: List[str] = None
) -> FormulaRegistry:
    """
    Create a FormulaRegistry with default engines.

    Args:
        engines_config: Engine configuration
        ensemble_config: Ensemble voting configuration
        include_engines: List of engines to include (None = all)

    Returns:
        Configured FormulaRegistry
    """
    registry = FormulaRegistry(engines_config, ensemble_config)

    # Default engines to include
    if include_engines is None:
        include_engines = ["adaptive", "pattern", "rentech"]

    # Import engines dynamically to avoid circular imports
    if "adaptive" in include_engines:
        try:
            from .adaptive_wrapper import AdaptiveEngineWrapper
            registry.register("adaptive", AdaptiveEngineWrapper())
        except ImportError:
            print("[Registry] Adaptive engine not available")

    if "pattern" in include_engines:
        try:
            from .pattern_wrapper import PatternEngineWrapper
            registry.register("pattern", PatternEngineWrapper())
        except ImportError:
            print("[Registry] Pattern engine not available")

    if "rentech" in include_engines:
        try:
            from .rentech_wrapper import RenTechEngineWrapper
            registry.register("rentech", RenTechEngineWrapper())
        except ImportError:
            print("[Registry] RenTech engine not available")

    if "qlib" in include_engines:
        try:
            from .qlib_wrapper import QLibEngineWrapper
            registry.register("qlib", QLibEngineWrapper())
        except ImportError:
            print("[Registry] QLib engine not available")

    return registry
