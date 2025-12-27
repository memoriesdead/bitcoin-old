"""
RenTech Engine Wrapper - Sovereign Engine
==========================================

Wraps the existing RenTechPatternEngine to implement BaseEngine interface.
"""
from typing import Dict, Any, List
import time
import numpy as np

from .base import BaseEngine
from ..core.types import Tick, Signal, TradeOutcome


class RenTechEngineWrapper(BaseEngine):
    """
    Wrapper for RenTechPatternEngine (IDs 72001-72099).

    Adapts the 99 RenTech patterns to the BaseEngine interface.
    """

    def __init__(self):
        super().__init__(
            name="rentech",
            formula_ids=list(range(72001, 72100))  # 99 formulas
        )
        self._engine = None
        self._sub_engines = {}
        self._price_history = []
        self._max_history = 500
        self._current_regime = "neutral"

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the RenTech engine."""
        try:
            from .rentech_engine import (
                HMMSubEngine,
                SignalSubEngine,
                NonlinearSubEngine,
                MicroSubEngine,
                EnsembleSubEngine,
            )

            self._sub_engines = {
                'hmm': HMMSubEngine(),
                'signal': SignalSubEngine(),
                'nonlinear': NonlinearSubEngine(),
                'micro': MicroSubEngine(),
                'ensemble': EnsembleSubEngine(),
            }
            self._initialized = True
            print(f"[RenTech] Initialized {len(self._sub_engines)} sub-engines")

        except ImportError as e:
            print(f"[RenTech] Engine not available: {e}")
            self._initialized = False

    def _get_regime(self) -> str:
        """Get current market regime from HMM engine."""
        if 'hmm' in self._sub_engines:
            try:
                self._current_regime = self._sub_engines['hmm'].regime
            except Exception:
                pass
        return self._current_regime

    def process(self, tick: Tick) -> Signal:
        """Process tick through RenTech patterns."""
        self.state.ticks_processed += 1

        if not self._initialized:
            return self._no_signal()

        # Update price history
        self._price_history.append(tick.price)
        if len(self._price_history) > self._max_history:
            self._price_history.pop(0)

        # Need minimum history
        if len(self._price_history) < 50:
            return self._no_signal()

        try:
            # Build features dict from tick
            features = {
                'volume_z': tick.features.get('volume_z', 0.0),
                'whale_accumulation': tick.features.get('whale_accumulation', 0.0),
                'whale_distribution': tick.features.get('whale_distribution', 0.0),
                'exchange_inflow': tick.features.get('exchange_inflow', 0.0),
                'exchange_outflow': tick.features.get('exchange_outflow', 0.0),
                'mempool_congestion': tick.features.get('mempool_congestion', 0.0),
                'fee_spike': tick.features.get('fee_spike', 0.0),
                'hashrate_momentum': tick.features.get('hashrate_momentum', 0.0),
                'difficulty_signal': tick.features.get('difficulty_signal', 0.0),
                'halving_cycle': tick.features.get('halving_cycle', 0.3),
                'open_interest': tick.features.get('open_interest', 0.0),
                'funding_rate': tick.features.get('funding_rate', 0.0),
                'liquidation_signal': tick.features.get('liquidation_signal', 0.0),
                'spot_premium': tick.features.get('spot_premium', 0.0),
                'orderbook_imbalance': tick.features.get('orderbook_imbalance', 0.0),
                'trade_flow': tick.features.get('trade_flow', 0.0),
                'market_maker': tick.features.get('market_maker', 0.0),
                'spread_regime': tick.features.get('spread_regime', 0.0),
                'tick_rule': tick.features.get('tick_rule', 0.0),
            }

            # Collect signals from all sub-engines
            # Sub-engines expect (price, features) interface
            all_signals = {}

            for name, engine in self._sub_engines.items():
                try:
                    if name == 'ensemble':
                        # Ensemble engine needs all_signals from other engines
                        result = engine.process(all_signals, self._get_regime())
                    else:
                        result = engine.process(tick.price, features)

                    # Result is Dict[int, float] - formula_id -> signal
                    if isinstance(result, dict):
                        all_signals.update(result)
                except Exception as e:
                    # Sub-engine error, skip
                    pass

            # Get master signal (72099) from ensemble
            master_signal = all_signals.get(72099, 0.0)

            # Convert to direction
            if abs(master_signal) < 0.2:
                return self._no_signal()

            direction = 1 if master_signal > 0 else -1
            confidence = min(1.0, abs(master_signal))

            # Get regime from HMM engine
            regime = self._get_regime()

            return self._create_signal(
                direction=direction,
                confidence=confidence,
                price=tick.price,
                formula_ids=[72099],  # Master ensemble
                regime=regime,
                stop_loss=0.005,
                take_profit=0.01,
                hold_seconds=60.0,
                votes={fid: sig for fid, sig in all_signals.items() if abs(sig) > 0.1},
                vote_confidences={fid: abs(sig) for fid, sig in all_signals.items() if abs(sig) > 0.1},
            )

        except Exception as e:
            print(f"[RenTech] Error: {e}")
            return self._no_signal()

    def learn(self, outcome: TradeOutcome) -> None:
        """Learn from trade outcome."""
        self.state.update_from_outcome(outcome)

        # Update sub-engines if they support learning
        for name, engine in self._sub_engines.items():
            if hasattr(engine, 'record_outcome'):
                try:
                    engine.record_outcome(outcome)
                except Exception:
                    pass
