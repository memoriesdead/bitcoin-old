"""
Base Engine Interface - Sovereign Engine
=========================================

Abstract base class that all formula engines must implement.
Ensures consistent interface for:
- Processing ticks
- Generating signals
- Learning from outcomes
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time

from ..core.types import Tick, Signal, TradeOutcome


@dataclass
class EngineState:
    """Current state of an engine."""
    name: str
    enabled: bool = True
    ticks_processed: int = 0
    signals_generated: int = 0
    trades_learned: int = 0
    last_signal_time: float = 0.0
    win_rate: float = 0.5
    avg_pnl: float = 0.0

    # Performance tracking
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0

    def update_from_outcome(self, outcome: TradeOutcome):
        """Update state from trade outcome."""
        self.trades_learned += 1
        self.total_pnl += outcome.pnl

        if outcome.was_profitable:
            self.wins += 1
        else:
            self.losses += 1

        total = self.wins + self.losses
        if total > 0:
            self.win_rate = self.wins / total
            self.avg_pnl = self.total_pnl / total


class BaseEngine(ABC):
    """
    Abstract base class for formula engines.

    All engines (Adaptive, Pattern, RenTech, QLib) must implement this interface.
    """

    def __init__(self, name: str, formula_ids: List[int] = None):
        """
        Initialize engine.

        Args:
            name: Engine name (e.g., "adaptive", "pattern", "rentech")
            formula_ids: List of formula IDs this engine uses
        """
        self.name = name
        self.formula_ids = formula_ids or []
        self.state = EngineState(name=name)
        self._initialized = False

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize engine with configuration.

        Called once before processing starts.

        Args:
            config: Engine-specific configuration
        """
        pass

    @abstractmethod
    def process(self, tick: Tick) -> Signal:
        """
        Process a tick and generate a signal.

        This is the main entry point for each market data event.

        Args:
            tick: Market data tick

        Returns:
            Signal with direction, confidence, and metadata
        """
        pass

    @abstractmethod
    def learn(self, outcome: TradeOutcome) -> None:
        """
        Learn from a trade outcome.

        Called after each trade completes to update internal parameters.

        Args:
            outcome: Complete trade outcome with PnL
        """
        pass

    def reset(self) -> None:
        """Reset engine state. Override if engine has additional state."""
        self.state = EngineState(name=self.name)
        self._initialized = False

    def get_state(self) -> EngineState:
        """Get current engine state."""
        return self.state

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information for debugging.

        Override to provide engine-specific diagnostics.
        """
        return {
            "name": self.name,
            "formula_ids": self.formula_ids,
            "ticks_processed": self.state.ticks_processed,
            "signals_generated": self.state.signals_generated,
            "win_rate": self.state.win_rate,
            "avg_pnl": self.state.avg_pnl,
        }

    def _create_signal(
        self,
        direction: int,
        confidence: float,
        price: float = 0.0,
        formula_ids: List[int] = None,
        **kwargs
    ) -> Signal:
        """
        Helper to create a Signal with common fields filled.

        Args:
            direction: +1 LONG, -1 SHORT, 0 HOLD
            confidence: 0.0 to 1.0
            price: Current price
            formula_ids: List of formula IDs that contributed
            **kwargs: Additional Signal fields

        Returns:
            Properly formatted Signal
        """
        self.state.signals_generated += 1
        self.state.last_signal_time = time.time()

        return Signal(
            timestamp=time.time(),
            direction=direction,
            confidence=confidence,
            source_engine=self.name,
            formula_ids=formula_ids or self.formula_ids,
            price_at_signal=price,
            **kwargs
        )

    def _no_signal(self) -> Signal:
        """Create a HOLD signal (no action)."""
        return self._create_signal(direction=0, confidence=0.0)


class PassthroughEngine(BaseEngine):
    """
    Simple passthrough engine for testing.

    Always returns HOLD signal.
    """

    def __init__(self):
        super().__init__(name="passthrough", formula_ids=[])

    def initialize(self, config: Dict[str, Any]) -> None:
        self._initialized = True

    def process(self, tick: Tick) -> Signal:
        self.state.ticks_processed += 1
        return self._no_signal()

    def learn(self, outcome: TradeOutcome) -> None:
        self.state.update_from_outcome(outcome)


class MockEngine(BaseEngine):
    """
    Mock engine for testing.

    Can be configured to return specific signals.
    """

    def __init__(self, name: str = "mock"):
        super().__init__(name=name, formula_ids=[99999])
        self._mock_signals: List[Signal] = []
        self._signal_index = 0

    def initialize(self, config: Dict[str, Any]) -> None:
        self._initialized = True

    def set_signals(self, signals: List[Signal]):
        """Set mock signals to return."""
        self._mock_signals = signals
        self._signal_index = 0

    def process(self, tick: Tick) -> Signal:
        self.state.ticks_processed += 1

        if self._mock_signals and self._signal_index < len(self._mock_signals):
            signal = self._mock_signals[self._signal_index]
            self._signal_index += 1
            self.state.signals_generated += 1
            return signal

        return self._no_signal()

    def learn(self, outcome: TradeOutcome) -> None:
        self.state.update_from_outcome(outcome)


# =============================================================================
# ENGINE WRAPPER
# =============================================================================

class EngineWrapper(BaseEngine):
    """
    Wrapper to adapt existing engines to BaseEngine interface.

    Use this to wrap legacy engines that don't implement the new interface.
    """

    def __init__(
        self,
        name: str,
        process_fn,
        learn_fn=None,
        formula_ids: List[int] = None
    ):
        """
        Wrap a function-based engine.

        Args:
            name: Engine name
            process_fn: Function(tick) -> Signal
            learn_fn: Optional function(outcome) -> None
            formula_ids: List of formula IDs
        """
        super().__init__(name=name, formula_ids=formula_ids or [])
        self._process_fn = process_fn
        self._learn_fn = learn_fn

    def initialize(self, config: Dict[str, Any]) -> None:
        self._initialized = True

    def process(self, tick: Tick) -> Signal:
        self.state.ticks_processed += 1
        try:
            result = self._process_fn(tick)
            if isinstance(result, Signal):
                self.state.signals_generated += 1
                return result
            else:
                # Assume legacy format, convert
                return self._convert_legacy_signal(result)
        except Exception as e:
            print(f"[{self.name}] Error processing tick: {e}")
            return self._no_signal()

    def learn(self, outcome: TradeOutcome) -> None:
        self.state.update_from_outcome(outcome)
        if self._learn_fn:
            try:
                self._learn_fn(outcome)
            except Exception as e:
                print(f"[{self.name}] Error learning from outcome: {e}")

    def _convert_legacy_signal(self, result: Any) -> Signal:
        """Convert legacy signal format to Signal."""
        if hasattr(result, 'direction') and hasattr(result, 'confidence'):
            return self._create_signal(
                direction=result.direction,
                confidence=getattr(result, 'confidence', 0.5),
                price=getattr(result, 'price', 0.0),
            )
        elif isinstance(result, dict):
            return self._create_signal(
                direction=result.get('direction', 0),
                confidence=result.get('confidence', 0.5),
                price=result.get('price', 0.0),
            )
        elif isinstance(result, (int, float)):
            # Assume it's just a direction
            return self._create_signal(
                direction=int(result),
                confidence=0.5 if result != 0 else 0.0,
            )
        else:
            return self._no_signal()
