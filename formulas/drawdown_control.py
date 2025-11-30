"""
Renaissance Formula Library - Drawdown Control
===============================================
ID 334: Position Sizing with Drawdown Limits

The Problem:
- Kelly criterion maximizes long-term growth but has HIGH VARIANCE
- Full Kelly can have 50%+ drawdowns
- A 50% drawdown requires 100% gain to recover
- A 70% drawdown requires 233% gain to recover

The Solution:
- Use FRACTIONAL Kelly (1/4 to 1/2)
- SCALE DOWN position size during drawdowns
- HARD STOPS at drawdown thresholds
- GRADUAL RECOVERY as performance improves

Mathematical Foundation:
From Thorp (2006) "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market":

Optimal Growth Rate: g = μ - σ²/2

With full Kelly: E[drawdown] ≈ 50%
With half Kelly: E[drawdown] ≈ 25%
With quarter Kelly: E[drawdown] ≈ 12.5%

Drawdown Recovery Time:
t_recovery = DD / g

If g = 1% daily and DD = 20%, recovery = 20 days
If g = 0.5% daily and DD = 20%, recovery = 40 days

Position Scaling Formula:
scale = min(1.0, (1 - DD/max_DD)^2)

This creates smooth reduction as drawdown increases.

Sources:
- Kelly (1956): "A New Interpretation of Information Rate"
- Thorp (2006): "The Kelly Criterion in Blackjack..."
- Vince (1992): "The Mathematics of Money Management"
"""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass
import time

from .base import BaseFormula, FormulaRegistry


@dataclass
class EquityPoint:
    """A point on the equity curve"""
    timestamp: float
    equity: float
    pnl: float
    trade_count: int


@FormulaRegistry.register(334, name="DrawdownControl", category="risk")
class DrawdownControlFormula(BaseFormula):
    """
    ID 334: Drawdown Control - Position Sizing with DD Limits

    Key Features:
    1. Tracks real-time drawdown from peak equity
    2. Scales position size inversely with drawdown
    3. Hard stops at configurable thresholds
    4. Gradual recovery scaling as equity recovers

    Position Scaling Levels:
    - DD < 5%:   100% of target size
    - DD 5-10%:  75% of target size
    - DD 10-15%: 50% of target size
    - DD 15-20%: 25% of target size
    - DD > 20%:  STOP TRADING

    This ensures survival during bad periods while still
    compounding during good periods.
    """

    FORMULA_ID = 334
    CATEGORY = "risk"
    NAME = "Drawdown Control"
    DESCRIPTION = "Position sizing with drawdown-based scaling"

    def __init__(self,
                 lookback: int = 1000,
                 max_drawdown_pct: float = 20.0,
                 warning_drawdown_pct: float = 10.0,
                 kelly_fraction: float = 0.25,
                 min_position_scale: float = 0.1,
                 recovery_rate: float = 0.5,
                 **kwargs):
        """
        Args:
            max_drawdown_pct: Stop trading at this drawdown %
            warning_drawdown_pct: Start reducing size at this %
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
            min_position_scale: Minimum position scale before stopping
            recovery_rate: How fast to increase size after recovery
        """
        super().__init__(lookback, **kwargs)

        self.max_drawdown_pct = max_drawdown_pct
        self.warning_drawdown_pct = warning_drawdown_pct
        self.kelly_fraction = kelly_fraction
        self.min_position_scale = min_position_scale
        self.recovery_rate = recovery_rate

        # Equity tracking
        self.initial_capital = 0.0
        self.current_equity = 0.0
        self.peak_equity = 0.0
        self.trough_equity = float('inf')

        # Drawdown metrics
        self.current_drawdown_pct = 0.0
        self.max_drawdown_experienced = 0.0

        # Position scaling
        self.position_scale = 1.0
        self.can_trade = True
        self.in_recovery_mode = False

        # History
        self.equity_history: deque = deque(maxlen=lookback)
        self.drawdown_history: deque = deque(maxlen=lookback)

        # Trade tracking
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.recent_pnl: deque = deque(maxlen=50)

    def initialize(self, capital: float):
        """Initialize with starting capital"""
        self.initial_capital = capital
        self.current_equity = capital
        self.peak_equity = capital
        self.trough_equity = capital

    def update_equity(self, new_equity: float, pnl: float = None, timestamp: float = None):
        """
        Update equity and recalculate drawdown.

        Args:
            new_equity: Current account equity
            pnl: P&L from last trade (optional)
            timestamp: Update time
        """
        now = timestamp or time.time()

        self.current_equity = new_equity

        # Update peak
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
            self.in_recovery_mode = False

        # Update trough
        if new_equity < self.trough_equity:
            self.trough_equity = new_equity

        # Calculate current drawdown
        if self.peak_equity > 0:
            self.current_drawdown_pct = (
                (self.peak_equity - new_equity) / self.peak_equity * 100
            )

        # Track max drawdown
        if self.current_drawdown_pct > self.max_drawdown_experienced:
            self.max_drawdown_experienced = self.current_drawdown_pct

        # Record history
        self.equity_history.append(EquityPoint(
            timestamp=now,
            equity=new_equity,
            pnl=pnl or 0,
            trade_count=len(self.equity_history)
        ))
        self.drawdown_history.append(self.current_drawdown_pct)

        # Track consecutive wins/losses
        if pnl is not None:
            self.recent_pnl.append(pnl)
            if pnl > 0:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
            elif pnl < 0:
                self.consecutive_losses += 1
                self.consecutive_wins = 0

        # Update position scale and trading status
        self._update_position_scale()

    def _update_position_scale(self):
        """Calculate position scale based on drawdown"""

        # Check if we should stop trading
        if self.current_drawdown_pct >= self.max_drawdown_pct:
            self.can_trade = False
            self.position_scale = 0.0
            self.signal = -1  # Signal to stop
            self.confidence = 1.0
            return

        # Smooth scaling based on drawdown
        # Formula: scale = (1 - DD/max_DD)^2 for smooth reduction
        if self.current_drawdown_pct <= self.warning_drawdown_pct:
            # Below warning - full size
            base_scale = 1.0
        else:
            # Between warning and max - scale down smoothly
            dd_range = self.max_drawdown_pct - self.warning_drawdown_pct
            dd_in_range = self.current_drawdown_pct - self.warning_drawdown_pct
            reduction_factor = dd_in_range / dd_range
            base_scale = max(self.min_position_scale, (1 - reduction_factor) ** 2)

        # Additional scaling for consecutive losses
        if self.consecutive_losses >= 5:
            base_scale *= 0.5  # Halve size after 5 consecutive losses
        elif self.consecutive_losses >= 3:
            base_scale *= 0.75  # Reduce 25% after 3 consecutive losses

        # Recovery mode bonus (gradual)
        if self.in_recovery_mode and self.consecutive_wins >= 3:
            recovery_bonus = min(0.25, self.consecutive_wins * 0.05)
            base_scale = min(1.0, base_scale + recovery_bonus)

        self.position_scale = base_scale
        self.can_trade = base_scale >= self.min_position_scale

        # Update signal
        if self.current_drawdown_pct < self.warning_drawdown_pct:
            self.signal = 1  # OK to trade full size
            self.confidence = 1.0 - self.current_drawdown_pct / 100
        elif self.can_trade:
            self.signal = 0  # Caution - reduced size
            self.confidence = 0.5
        else:
            self.signal = -1  # Stop trading
            self.confidence = 0.0

    def get_position_size(self,
                         target_size: float,
                         win_rate: float = 0.55,
                         avg_win: float = 0.002,
                         avg_loss: float = 0.001) -> float:
        """
        Get position size adjusted for drawdown and Kelly.

        Args:
            target_size: Desired position size before adjustments
            win_rate: Current win rate
            avg_win: Average winning trade %
            avg_loss: Average losing trade %

        Returns:
            Adjusted position size
        """
        if not self.can_trade:
            return 0.0

        # Calculate Kelly fraction
        if avg_loss > 0:
            b = avg_win / avg_loss  # Win/loss ratio
            q = 1 - win_rate
            kelly = (win_rate * b - q) / b
        else:
            kelly = 0.1  # Default small fraction

        # Apply fractional Kelly
        kelly_adjusted = kelly * self.kelly_fraction

        # Clamp Kelly to reasonable range
        kelly_adjusted = max(0.01, min(0.5, kelly_adjusted))

        # Apply drawdown scaling
        final_scale = kelly_adjusted * self.position_scale

        # Calculate final position size
        position_size = target_size * final_scale

        return position_size

    def get_max_loss_per_trade(self, capital: float = None) -> float:
        """Get maximum allowed loss per trade"""
        equity = capital or self.current_equity or self.initial_capital
        if equity <= 0:
            return 0

        # Base max loss: 2% of equity
        base_max_loss = equity * 0.02

        # Reduce during drawdown
        adjusted_max_loss = base_max_loss * self.position_scale

        return adjusted_max_loss

    def should_stop_trading(self) -> bool:
        """Check if we should stop trading entirely"""
        return not self.can_trade

    def get_recovery_target(self) -> float:
        """Get equity target to exit recovery mode"""
        return self.peak_equity * 0.95  # Recover to 95% of peak

    def reset_peak(self):
        """Reset peak equity (use carefully)"""
        self.peak_equity = self.current_equity
        self.current_drawdown_pct = 0.0
        self.in_recovery_mode = False

    def _compute(self) -> None:
        """Required by BaseFormula"""
        pass

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'current_drawdown_pct': self.current_drawdown_pct,
            'max_drawdown_experienced': self.max_drawdown_experienced,
            'position_scale': self.position_scale,
            'can_trade': self.can_trade,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'kelly_fraction': self.kelly_fraction,
            'in_recovery_mode': self.in_recovery_mode,
        })
        return state


@FormulaRegistry.register(339, name="AntiMartingale", category="risk")
class AntiMartingaleFormula(BaseFormula):
    """
    ID 339: Anti-Martingale Position Sizing

    The opposite of Martingale (which doubles down on losses).
    Anti-Martingale INCREASES size after wins, DECREASES after losses.

    Mathematical Basis:
    - After a win: size = base_size × (1 + win_streak × increment)
    - After a loss: size = base_size × (1 - loss_streak × decrement)

    This naturally:
    - Compounds during winning streaks (when edge is working)
    - Protects capital during losing streaks (when edge may be gone)

    Combined with Kelly for optimal growth rate.
    """

    FORMULA_ID = 339
    CATEGORY = "risk"
    NAME = "Anti-Martingale"
    DESCRIPTION = "Increase size on wins, decrease on losses"

    def __init__(self,
                 lookback: int = 100,
                 win_increment: float = 0.2,
                 loss_decrement: float = 0.3,
                 max_multiplier: float = 3.0,
                 min_multiplier: float = 0.2,
                 **kwargs):
        super().__init__(lookback, **kwargs)

        self.win_increment = win_increment
        self.loss_decrement = loss_decrement
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier

        self.win_streak = 0
        self.loss_streak = 0
        self.current_multiplier = 1.0

        self.trade_history: deque = deque(maxlen=lookback)

    def record_trade(self, pnl: float):
        """Record trade outcome and update multiplier"""
        self.trade_history.append(pnl)

        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
            # Increase size after win
            self.current_multiplier = min(
                self.max_multiplier,
                1.0 + self.win_streak * self.win_increment
            )
        else:
            self.loss_streak += 1
            self.win_streak = 0
            # Decrease size after loss
            self.current_multiplier = max(
                self.min_multiplier,
                1.0 - self.loss_streak * self.loss_decrement
            )

        self._update_signal()

    def _update_signal(self):
        """Update signal based on streak"""
        if self.win_streak >= 3:
            self.signal = 1  # Strong momentum, increase size
            self.confidence = min(1.0, self.win_streak / 5)
        elif self.loss_streak >= 3:
            self.signal = -1  # Losing streak, reduce size
            self.confidence = min(1.0, self.loss_streak / 5)
        else:
            self.signal = 0
            self.confidence = 0.5

    def get_size_multiplier(self) -> float:
        """Get current position size multiplier"""
        return self.current_multiplier

    def reset(self):
        """Reset streaks"""
        self.win_streak = 0
        self.loss_streak = 0
        self.current_multiplier = 1.0

    def _compute(self) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'win_streak': self.win_streak,
            'loss_streak': self.loss_streak,
            'current_multiplier': self.current_multiplier,
            'trade_count': len(self.trade_history),
        })
        return state
