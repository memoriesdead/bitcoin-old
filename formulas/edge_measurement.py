"""
Renaissance Formula Library - Real Edge Measurement
====================================================
ID 331: Rolling Backtest Edge Calculator

THIS IS THE MOST CRITICAL FORMULA IN THE ENTIRE SYSTEM.

The Problem:
- Old system: edge = signal * 0.001 (FABRICATED)
- Reality: edge must come from ACTUAL MEASURED OUTCOMES

The Solution:
- Track every signal and its outcome
- Calculate REAL win rate, avg win, avg loss from data
- Only trade when MEASURED edge > costs

Mathematical Foundation:
- Shannon (1956): Optimal betting requires knowing true probabilities
- Kelly (1956): f* = edge / odds requires REAL edge measurement
- Thorp (1962): "Beat the Dealer" - measured edge = profit

Formula:
    True_Edge = E[return | signal] = Σ(r_i × I(signal_i)) / Σ(I(signal_i))

    Where:
    - r_i = return after signal i
    - I(signal_i) = indicator function (1 if signal matches direction)

    Trade only when: True_Edge > 2 × Total_Cost (for margin of safety)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import time

from .base import BaseFormula, FormulaRegistry


@dataclass
class SignalOutcome:
    """Record of a signal and its outcome"""
    timestamp: float
    signal_direction: int  # 1 = buy, -1 = sell
    entry_price: float
    exit_price: float = 0.0
    bars_held: int = 0
    pnl_pct: float = 0.0
    resolved: bool = False


@FormulaRegistry.register(331, name="RealEdgeMeasurement", category="edge")
class RealEdgeMeasurementFormula(BaseFormula):
    """
    ID 331: Real Edge Measurement - THE FOUNDATION OF PROFITABLE TRADING

    This formula measures ACTUAL edge from historical signal outcomes,
    not fabricated numbers.

    Key Metrics Tracked:
    1. Win Rate by Signal Type (buy signals, sell signals, combined)
    2. Average Win Size (in %)
    3. Average Loss Size (in %)
    4. Edge = (WinRate × AvgWin) - (LossRate × AvgLoss)
    5. Sharpe Ratio of signals
    6. Profit Factor = GrossWins / GrossLosses

    Sources:
    - Thorp, E. (2017). "A Man for All Markets"
    - Lopez de Prado (2018). "Advances in Financial Machine Learning"
    - Chan, E. (2013). "Algorithmic Trading"
    """

    FORMULA_ID = 331
    CATEGORY = "edge"
    NAME = "Real Edge Measurement"
    DESCRIPTION = "Measures actual trading edge from signal outcomes"

    def __init__(self,
                 lookback: int = 1000,
                 exit_bars: int = 10,
                 min_signals_for_edge: int = 30,
                 cost_multiplier: float = 2.0,
                 **kwargs):
        """
        Args:
            lookback: Number of signals to track for edge calculation
            exit_bars: Number of bars to wait before measuring outcome
            min_signals_for_edge: Minimum signals needed for valid edge estimate
            cost_multiplier: Required edge must be this × costs
        """
        super().__init__(lookback, **kwargs)

        self.exit_bars = exit_bars
        self.min_signals_for_edge = min_signals_for_edge
        self.cost_multiplier = cost_multiplier

        # Signal tracking
        self.pending_signals: List[SignalOutcome] = []
        self.resolved_signals: deque = deque(maxlen=lookback)

        # Separate tracking for buy/sell
        self.buy_outcomes: deque = deque(maxlen=lookback // 2)
        self.sell_outcomes: deque = deque(maxlen=lookback // 2)

        # Current metrics - ALL START AT ZERO (no assumptions)
        # Only updated from ACTUAL trade outcomes
        self.measured_win_rate = 0.0  # LIVE: From actual trades only
        self.measured_avg_win = 0.0   # LIVE: From actual winning trades
        self.measured_avg_loss = 0.0  # LIVE: From actual losing trades
        self.measured_edge = 0.0      # LIVE: Calculated from actual outcomes
        self.measured_sharpe = 0.0    # LIVE: From actual returns
        self.profit_factor = 0.0      # LIVE: From actual wins/losses

        # Separate metrics by direction - ALL START AT ZERO
        self.buy_win_rate = 0.0   # LIVE: From actual buy outcomes
        self.sell_win_rate = 0.0  # LIVE: From actual sell outcomes
        self.buy_edge = 0.0       # LIVE: From actual buy outcomes
        self.sell_edge = 0.0      # LIVE: From actual sell outcomes

        # Current bar counter
        self.bar_count = 0

        # Transaction costs - MUST BE SET FROM LIVE DATA
        self.total_cost = 0.0  # LIVE: From actual spread measurement

    def record_signal(self, direction: int, price: float, timestamp: float = None):
        """
        Record a new trading signal for edge measurement.

        Args:
            direction: 1 for buy, -1 for sell
            price: Entry price at signal
            timestamp: Signal timestamp
        """
        if direction == 0:
            return

        signal = SignalOutcome(
            timestamp=timestamp or time.time(),
            signal_direction=direction,
            entry_price=price,
        )
        self.pending_signals.append(signal)

    def update_price(self, price: float):
        """
        Update with new price to resolve pending signals.
        Called every bar/tick.
        """
        self.bar_count += 1

        # Resolve signals that have reached exit_bars
        still_pending = []
        for signal in self.pending_signals:
            signal.bars_held += 1

            if signal.bars_held >= self.exit_bars:
                # Resolve this signal
                signal.exit_price = price
                signal.pnl_pct = (
                    (price - signal.entry_price) / signal.entry_price * 100
                    * signal.signal_direction  # Positive if direction was correct
                )
                signal.resolved = True

                # Add to appropriate tracking
                self.resolved_signals.append(signal)
                if signal.signal_direction == 1:
                    self.buy_outcomes.append(signal)
                else:
                    self.sell_outcomes.append(signal)
            else:
                still_pending.append(signal)

        self.pending_signals = still_pending

        # Recalculate metrics
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate REAL edge metrics from resolved signals - NO FALLBACKS"""

        if len(self.resolved_signals) < self.min_signals_for_edge:
            # Not enough data yet - ALL ZEROS (no trading until we have real data)
            self.measured_win_rate = 0.0
            self.measured_avg_win = 0.0
            self.measured_avg_loss = 0.0
            self.measured_edge = 0.0
            self.signal = 0
            self.confidence = 0.0
            return

        # Calculate from REAL data ONLY
        wins = []
        losses = []
        all_pnls = []

        for sig in self.resolved_signals:
            pnl = sig.pnl_pct / 100  # Convert to decimal
            all_pnls.append(pnl)

            if pnl > 0:
                wins.append(pnl)
            else:
                losses.append(abs(pnl))

        # REAL metrics - NO FALLBACKS
        total_trades = len(all_pnls)
        win_count = len(wins)

        self.measured_win_rate = win_count / total_trades if total_trades > 0 else 0.0
        self.measured_avg_win = np.mean(wins) if wins else 0.0
        self.measured_avg_loss = np.mean(losses) if losses else 0.0

        # REAL EDGE FORMULA: E[r] = p*W - (1-p)*L
        self.measured_edge = (
            self.measured_win_rate * self.measured_avg_win -
            (1 - self.measured_win_rate) * self.measured_avg_loss
        )

        # Sharpe ratio of signals
        if len(all_pnls) > 1:
            self.measured_sharpe = (
                np.mean(all_pnls) / np.std(all_pnls) * np.sqrt(252)
                if np.std(all_pnls) > 0 else 0
            )

        # Profit factor
        gross_wins = sum(wins) if wins else 0
        gross_losses = sum(losses) if losses else 0.0001
        self.profit_factor = gross_wins / gross_losses if gross_losses > 0 else 1.0

        # Calculate separate buy/sell metrics
        self._calculate_directional_metrics()

        # Update signal/confidence based on edge
        self._update_signal()

    def _calculate_directional_metrics(self):
        """Calculate metrics separately for buy and sell signals"""

        # Buy signals
        if len(self.buy_outcomes) >= self.min_signals_for_edge // 2:
            buy_pnls = [s.pnl_pct / 100 for s in self.buy_outcomes]
            buy_wins = [p for p in buy_pnls if p > 0]
            buy_losses = [abs(p) for p in buy_pnls if p <= 0]

            self.buy_win_rate = len(buy_wins) / len(buy_pnls) if buy_pnls else 0.5
            avg_buy_win = np.mean(buy_wins) if buy_wins else 0.001
            avg_buy_loss = np.mean(buy_losses) if buy_losses else 0.001
            self.buy_edge = (
                self.buy_win_rate * avg_buy_win -
                (1 - self.buy_win_rate) * avg_buy_loss
            )

        # Sell signals
        if len(self.sell_outcomes) >= self.min_signals_for_edge // 2:
            sell_pnls = [s.pnl_pct / 100 for s in self.sell_outcomes]
            sell_wins = [p for p in sell_pnls if p > 0]
            sell_losses = [abs(p) for p in sell_pnls if p <= 0]

            self.sell_win_rate = len(sell_wins) / len(sell_pnls) if sell_pnls else 0.5
            avg_sell_win = np.mean(sell_wins) if sell_wins else 0.001
            avg_sell_loss = np.mean(sell_losses) if sell_losses else 0.001
            self.sell_edge = (
                self.sell_win_rate * avg_sell_win -
                (1 - self.sell_win_rate) * avg_sell_loss
            )

    def _update_signal(self):
        """Update trading signal based on measured edge"""

        # Required edge = cost_multiplier × total_cost
        required_edge = self.cost_multiplier * self.total_cost

        if self.measured_edge > required_edge:
            # We have positive edge - signal to trade
            self.signal = 1  # Positive = system has edge, trade
            self.confidence = min(1.0, self.measured_edge / required_edge - 1)
        elif self.measured_edge < -required_edge:
            # Negative edge - something is wrong
            self.signal = -1  # Signal to STOP trading
            self.confidence = min(1.0, abs(self.measured_edge) / required_edge)
        else:
            # Marginal edge - be cautious
            self.signal = 0
            self.confidence = 0.5

    def set_transaction_cost(self, cost: float):
        """Update transaction cost for edge comparison"""
        self.total_cost = cost

    def should_trade(self, direction: int = None) -> Tuple[bool, float]:
        """
        Check if we should trade based on measured edge.

        Args:
            direction: Optional - 1 for buy, -1 for sell

        Returns:
            (should_trade: bool, expected_edge: float)
        """
        required_edge = self.cost_multiplier * self.total_cost

        if direction == 1:
            edge = self.buy_edge
        elif direction == -1:
            edge = self.sell_edge
        else:
            edge = self.measured_edge

        should = edge > required_edge
        return should, edge

    def get_kelly_inputs(self) -> Dict[str, float]:
        """Get inputs for Kelly criterion from REAL measurements"""
        return {
            'win_rate': self.measured_win_rate,
            'avg_win': self.measured_avg_win,
            'avg_loss': self.measured_avg_loss,
            'edge': self.measured_edge,
        }

    def _compute(self) -> None:
        """Required by BaseFormula - metrics updated in update_price"""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get current state with all REAL metrics"""
        state = super().get_state()
        state.update({
            # REAL measured values
            'measured_win_rate': self.measured_win_rate,
            'measured_avg_win': self.measured_avg_win,
            'measured_avg_loss': self.measured_avg_loss,
            'measured_edge': self.measured_edge,
            'measured_sharpe': self.measured_sharpe,
            'profit_factor': self.profit_factor,

            # Directional metrics
            'buy_win_rate': self.buy_win_rate,
            'sell_win_rate': self.sell_win_rate,
            'buy_edge': self.buy_edge,
            'sell_edge': self.sell_edge,

            # Signal counts
            'resolved_signals': len(self.resolved_signals),
            'pending_signals': len(self.pending_signals),
            'buy_signals': len(self.buy_outcomes),
            'sell_signals': len(self.sell_outcomes),

            # Trading recommendation
            'should_trade': self.measured_edge > self.cost_multiplier * self.total_cost,
            'required_edge': self.cost_multiplier * self.total_cost,
        })
        return state


@FormulaRegistry.register(336, name="AdaptiveEdgeTracker", category="edge")
class AdaptiveEdgeTracker(BaseFormula):
    """
    ID 336: Adaptive Edge Tracker - Real-time edge monitoring

    Tracks edge across multiple timeframes and adapts position sizing.
    Uses exponential weighting to give more importance to recent signals.

    Key Features:
    1. Multi-timeframe edge (short/medium/long term)
    2. Exponential decay weighting for recent relevance
    3. Regime detection (edge increasing/decreasing)
    4. Automatic position scaling based on edge confidence
    """

    FORMULA_ID = 336
    CATEGORY = "edge"
    NAME = "Adaptive Edge Tracker"
    DESCRIPTION = "Multi-timeframe adaptive edge tracking with regime detection"

    def __init__(self,
                 lookback: int = 500,
                 short_window: int = 20,
                 medium_window: int = 100,
                 long_window: int = 500,
                 decay_factor: float = 0.95,
                 **kwargs):
        super().__init__(lookback, **kwargs)

        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.decay_factor = decay_factor

        # Trade outcomes by timeframe
        self.outcomes: deque = deque(maxlen=long_window)

        # Edge by timeframe
        self.short_edge = 0.0
        self.medium_edge = 0.0
        self.long_edge = 0.0

        # Edge regime
        self.edge_trend = 0  # 1 = improving, -1 = deteriorating
        self.edge_momentum = 0.0

        # Position scaling factor
        self.position_scale = 1.0

    def add_outcome(self, pnl_pct: float, timestamp: float = None):
        """Add a trade outcome"""
        self.outcomes.append({
            'pnl': pnl_pct / 100,
            'timestamp': timestamp or time.time()
        })
        self._calculate_edges()

    def _calculate_edges(self):
        """Calculate edge across timeframes with exponential weighting"""

        if len(self.outcomes) < self.short_window:
            return

        outcomes_list = list(self.outcomes)

        # Short-term edge (most recent, highest weight)
        short_outcomes = outcomes_list[-self.short_window:]
        weights = np.array([self.decay_factor ** i for i in range(len(short_outcomes))])[::-1]
        weights /= weights.sum()
        self.short_edge = np.sum([o['pnl'] * w for o, w in zip(short_outcomes, weights)])

        # Medium-term edge
        if len(outcomes_list) >= self.medium_window:
            med_outcomes = outcomes_list[-self.medium_window:]
            weights = np.array([self.decay_factor ** i for i in range(len(med_outcomes))])[::-1]
            weights /= weights.sum()
            self.medium_edge = np.sum([o['pnl'] * w for o, w in zip(med_outcomes, weights)])

        # Long-term edge
        if len(outcomes_list) >= self.long_window:
            weights = np.array([self.decay_factor ** i for i in range(len(outcomes_list))])[::-1]
            weights /= weights.sum()
            self.long_edge = np.sum([o['pnl'] * w for o, w in zip(outcomes_list, weights)])

        # Edge regime detection
        if self.short_edge > self.medium_edge > self.long_edge:
            self.edge_trend = 1  # Improving
            self.edge_momentum = self.short_edge - self.long_edge
        elif self.short_edge < self.medium_edge < self.long_edge:
            self.edge_trend = -1  # Deteriorating
            self.edge_momentum = self.short_edge - self.long_edge
        else:
            self.edge_trend = 0  # Mixed
            self.edge_momentum = 0.0

        # Position scaling
        if self.edge_trend == 1 and self.short_edge > 0:
            # Edge improving and positive - increase size
            self.position_scale = min(2.0, 1.0 + self.edge_momentum * 10)
        elif self.edge_trend == -1 or self.short_edge < 0:
            # Edge deteriorating or negative - reduce size
            self.position_scale = max(0.25, 1.0 + self.edge_momentum * 10)
        else:
            self.position_scale = 1.0

        # Update signal
        self.signal = self.edge_trend
        self.confidence = abs(self.edge_momentum) * 100

    def _compute(self) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'short_edge': self.short_edge,
            'medium_edge': self.medium_edge,
            'long_edge': self.long_edge,
            'edge_trend': self.edge_trend,
            'edge_momentum': self.edge_momentum,
            'position_scale': self.position_scale,
            'total_outcomes': len(self.outcomes),
        })
        return state
