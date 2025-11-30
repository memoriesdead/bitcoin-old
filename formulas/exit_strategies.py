#!/usr/bin/env python3
"""
EXIT STRATEGY FORMULAS (IDs 320-322)
====================================
Academic research-based exit strategies for optimal position closing.

Based on:
- Leung & Li (2015): "Optimal mean reversion trading with transaction costs and stop-loss exit"
- Leung (2021): "Optimal Trading with a Trailing Stop" (Applied Math & Optimization)
- arXiv 1706.07021: "Stop-loss and Leverage in optimal Statistical Arbitrage"

These formulas determine WHEN to exit a position to lock in profits.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass

from formulas.base import BaseFormula, FormulaRegistry


@dataclass
class ExitSignal:
    """Exit decision from formulas"""
    should_exit: bool
    exit_type: str  # 'take_profit', 'stop_loss', 'trailing_stop', 'time_decay', 'mean_reversion'
    confidence: float
    expected_pnl: float
    reason: str


# =============================================================================
# FORMULA 320: OPTIMAL STOPPING FORMULA
# =============================================================================
# Based on Leung & Li (2015) - Optimal Mean Reversion Trading
#
# For an Ornstein-Uhlenbeck process:
#   dX_t = theta(mu - X_t)dt + sigma*dW_t
#
# The optimal exit boundary U* satisfies:
#   U* = mu + sigma * sqrt(2/theta) * Phi^(-1)(1 - c/V)
#
# Where:
#   theta = mean reversion speed
#   mu = long-term mean
#   sigma = volatility
#   c = transaction cost
#   V = expected value of trade
# =============================================================================

@FormulaRegistry.register(320, name="OptimalStoppingFormula", category="exit")
class OptimalStoppingFormula(BaseFormula):
    """
    Formula 320: Optimal Stopping Exit

    Based on Leung & Li (2015) optimal double-stopping problem.
    Determines optimal exit boundary for mean-reverting positions.

    Academic Reference:
        Leung, T., & Li, X. (2015). Optimal mean reversion trading with
        transaction costs and stop-loss exit. International Journal of
        Theoretical and Applied Finance.
    """

    FORMULA_ID = 320
    CATEGORY = "exit"
    NAME = "OptimalStoppingFormula"
    DESCRIPTION = "Optimal exit boundary for mean-reverting positions (Leung & Li 2015)"

    def __init__(
        self,
        theta: float = 0.5,           # Mean reversion speed (higher = faster)
        mu: float = 0.0,              # Long-term mean (log returns)
        sigma: float = 0.02,          # Volatility
        transaction_cost: float = 0.001,  # Transaction cost as fraction
        take_profit_multiple: float = 2.0,  # Take profit at 2x expected edge
        stop_loss_multiple: float = 1.0,    # Stop loss at 1x expected edge
        **kwargs
    ):
        super().__init__(**kwargs)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.transaction_cost = transaction_cost
        self.take_profit_multiple = take_profit_multiple
        self.stop_loss_multiple = stop_loss_multiple

        # Position tracking
        self.entry_price = 0.0
        self.entry_time = 0.0
        self.position_side = 0  # 1 = long, -1 = short, 0 = flat
        self.current_price = 0.0

        # Computed boundaries
        self.take_profit_level = 0.0
        self.stop_loss_level = 0.0
        self.optimal_exit_boundary = 0.0

        # Statistics
        self.log_returns = deque(maxlen=100)

    def set_position(self, entry_price: float, side: int, entry_time: float = 0.0):
        """Set current position for exit monitoring"""
        self.entry_price = entry_price
        self.position_side = side
        self.entry_time = entry_time
        self._compute_boundaries()

    def _compute_boundaries(self):
        """Compute optimal exit boundaries based on OU theory"""
        if self.entry_price <= 0 or self.position_side == 0:
            return

        # Expected value from mean reversion (simplified)
        # V = sigma * sqrt(2/theta) based on OU expected gain
        expected_value = self.sigma * np.sqrt(2 / self.theta) if self.theta > 0 else 0.01

        # Optimal take profit (Leung & Li formula simplified)
        # Take profit when unrealized gain exceeds transaction costs + buffer
        optimal_tp = expected_value * self.take_profit_multiple

        # Stop loss based on acceptable loss
        optimal_sl = expected_value * self.stop_loss_multiple

        if self.position_side == 1:  # Long position
            self.take_profit_level = self.entry_price * (1 + optimal_tp)
            self.stop_loss_level = self.entry_price * (1 - optimal_sl)
        else:  # Short position
            self.take_profit_level = self.entry_price * (1 - optimal_tp)
            self.stop_loss_level = self.entry_price * (1 + optimal_sl)

        self.optimal_exit_boundary = optimal_tp

    def _compute(self) -> None:
        """Compute exit signal"""
        if len(self.prices) < 2:
            return

        self.current_price = self.prices[-1]

        # Track log returns for volatility estimation
        if len(self.prices) >= 2:
            log_ret = np.log(self.prices[-1] / self.prices[-2])
            self.log_returns.append(log_ret)

            # Update sigma estimate from recent returns
            if len(self.log_returns) >= 10:
                self.sigma = np.std(list(self.log_returns)) * np.sqrt(252 * 24 * 60)  # Annualized

    def should_exit(self) -> ExitSignal:
        """Check if position should be exited"""
        if self.position_side == 0 or self.entry_price <= 0:
            return ExitSignal(False, 'none', 0.0, 0.0, 'No position')

        current_price = self.current_price if self.current_price > 0 else self.prices[-1] if self.prices else 0

        if current_price <= 0:
            return ExitSignal(False, 'none', 0.0, 0.0, 'No price data')

        # Calculate unrealized PnL
        if self.position_side == 1:  # Long
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # Short
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        # Check take profit
        if self.position_side == 1 and current_price >= self.take_profit_level:
            return ExitSignal(
                True, 'take_profit', 0.95,
                pnl_pct,
                f'Take profit hit: {current_price:.2f} >= {self.take_profit_level:.2f}'
            )
        elif self.position_side == -1 and current_price <= self.take_profit_level:
            return ExitSignal(
                True, 'take_profit', 0.95,
                pnl_pct,
                f'Take profit hit: {current_price:.2f} <= {self.take_profit_level:.2f}'
            )

        # Check stop loss
        if self.position_side == 1 and current_price <= self.stop_loss_level:
            return ExitSignal(
                True, 'stop_loss', 0.99,
                pnl_pct,
                f'Stop loss hit: {current_price:.2f} <= {self.stop_loss_level:.2f}'
            )
        elif self.position_side == -1 and current_price >= self.stop_loss_level:
            return ExitSignal(
                True, 'stop_loss', 0.99,
                pnl_pct,
                f'Stop loss hit: {current_price:.2f} >= {self.stop_loss_level:.2f}'
            )

        return ExitSignal(False, 'hold', 0.5, pnl_pct, 'Within boundaries')

    def get_signal(self) -> int:
        """Return exit signal: -1 = exit long, +1 = exit short, 0 = hold"""
        exit_signal = self.should_exit()
        if exit_signal.should_exit:
            return -self.position_side  # Opposite of position to close
        return 0

    def get_confidence(self) -> float:
        return self.should_exit().confidence


# =============================================================================
# FORMULA 321: TRAILING STOP FORMULA
# =============================================================================
# Based on Leung (2021) - Optimal Trading with a Trailing Stop
#
# A trailing stop is characterized by a stochastic floor M_t = max(S_s : 0<=s<=t)
# The stop is triggered when S_t <= (1-delta) * M_t
#
# Optimal trailing percentage delta* minimizes expected shortfall while maximizing gain
# =============================================================================

@FormulaRegistry.register(321, name="TrailingStopFormula", category="exit")
class TrailingStopFormula(BaseFormula):
    """
    Formula 321: Trailing Stop Exit

    Based on Leung (2021) - Optimal Trading with a Trailing Stop.
    Uses a dynamic floor based on running maximum.

    Academic Reference:
        Leung, T. (2021). Optimal Trading with a Trailing Stop.
        Applied Mathematics & Optimization.
    """

    FORMULA_ID = 321
    CATEGORY = "exit"
    NAME = "TrailingStopFormula"
    DESCRIPTION = "Dynamic trailing stop based on running maximum (Leung 2021)"

    def __init__(
        self,
        trail_pct: float = 0.002,     # 0.2% trailing stop (tight for HFT)
        min_profit_to_trail: float = 0.001,  # Only trail after 0.1% profit
        acceleration_factor: float = 0.02,   # Parabolic SAR acceleration
        max_acceleration: float = 0.2,       # Max acceleration
        **kwargs
    ):
        super().__init__(**kwargs)
        self.trail_pct = trail_pct
        self.min_profit_to_trail = min_profit_to_trail
        self.acceleration_factor = acceleration_factor
        self.max_acceleration = max_acceleration

        # Position tracking
        self.entry_price = 0.0
        self.position_side = 0

        # Trailing state
        self.running_max = 0.0
        self.running_min = float('inf')
        self.trail_stop = 0.0
        self.is_trailing = False

        # Parabolic SAR state (for acceleration)
        self.sar = 0.0
        self.af = 0.02
        self.ep = 0.0  # Extreme point

    def set_position(self, entry_price: float, side: int, entry_time: float = 0.0):
        """Set position for trailing"""
        self.entry_price = entry_price
        self.position_side = side
        self.running_max = entry_price
        self.running_min = entry_price
        self.is_trailing = False
        self.trail_stop = 0.0
        self.af = self.acceleration_factor
        self.sar = entry_price
        self.ep = entry_price

    def _compute(self) -> None:
        """Update trailing stop"""
        if len(self.prices) < 1 or self.position_side == 0:
            return

        current_price = self.prices[-1]

        if self.position_side == 1:  # Long position
            # Update running max
            if current_price > self.running_max:
                self.running_max = current_price
                # Accelerate trailing when making new highs
                self.af = min(self.af + self.acceleration_factor, self.max_acceleration)
                self.ep = current_price

            # Check if we should start trailing
            profit_pct = (self.running_max - self.entry_price) / self.entry_price
            if profit_pct >= self.min_profit_to_trail:
                self.is_trailing = True
                # Dynamic trail: tighter as profit grows
                dynamic_trail = self.trail_pct * (1 - min(profit_pct * 10, 0.5))
                self.trail_stop = self.running_max * (1 - dynamic_trail)

            # Update Parabolic SAR
            self.sar = self.sar + self.af * (self.ep - self.sar)
            self.trail_stop = max(self.trail_stop, self.sar)

        else:  # Short position
            # Update running min
            if current_price < self.running_min:
                self.running_min = current_price
                self.af = min(self.af + self.acceleration_factor, self.max_acceleration)
                self.ep = current_price

            profit_pct = (self.entry_price - self.running_min) / self.entry_price
            if profit_pct >= self.min_profit_to_trail:
                self.is_trailing = True
                dynamic_trail = self.trail_pct * (1 - min(profit_pct * 10, 0.5))
                self.trail_stop = self.running_min * (1 + dynamic_trail)

            self.sar = self.sar - self.af * (self.sar - self.ep)
            self.trail_stop = min(self.trail_stop, self.sar) if self.trail_stop > 0 else self.sar

    def should_exit(self) -> ExitSignal:
        """Check if trailing stop triggered"""
        if self.position_side == 0 or not self.is_trailing:
            return ExitSignal(False, 'none', 0.0, 0.0, 'Not trailing')

        current_price = self.prices[-1] if self.prices else 0

        if current_price <= 0:
            return ExitSignal(False, 'none', 0.0, 0.0, 'No price')

        if self.position_side == 1:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            if current_price <= self.trail_stop:
                return ExitSignal(
                    True, 'trailing_stop', 0.90,
                    pnl_pct,
                    f'Trail stop hit: {current_price:.2f} <= {self.trail_stop:.2f}'
                )
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price
            if current_price >= self.trail_stop:
                return ExitSignal(
                    True, 'trailing_stop', 0.90,
                    pnl_pct,
                    f'Trail stop hit: {current_price:.2f} >= {self.trail_stop:.2f}'
                )

        return ExitSignal(False, 'hold', 0.5, pnl_pct, 'Trailing active')

    def get_signal(self) -> int:
        exit_signal = self.should_exit()
        if exit_signal.should_exit:
            return -self.position_side
        return 0

    def get_confidence(self) -> float:
        return self.should_exit().confidence


# =============================================================================
# FORMULA 322: FIRST EXIT TIME FORMULA
# =============================================================================
# Based on arXiv 1706.07021 - Expected First-Exit-Time for OU process
#
# For OU process with entry band D and exit band U:
# E[tau] = (2/theta) * ln(U/D) for symmetrical bounds
#
# This gives expected time to exit, used for time-decay exits
# =============================================================================

@FormulaRegistry.register(322, name="FirstExitTimeFormula", category="exit")
class FirstExitTimeFormula(BaseFormula):
    """
    Formula 322: First Exit Time Exit

    Based on expected first-exit-time theory for OU processes.
    Triggers exit if position held too long without hitting target.

    Academic Reference:
        arXiv 1706.07021 - Stop-loss and Leverage in optimal Statistical Arbitrage
    """

    FORMULA_ID = 322
    CATEGORY = "exit"
    NAME = "FirstExitTimeFormula"
    DESCRIPTION = "Time-based exit using expected first-exit-time (arXiv 1706.07021)"

    def __init__(
        self,
        theta: float = 0.5,           # Mean reversion speed
        max_holding_periods: int = 60,  # Max periods to hold (e.g., 60 seconds)
        decay_start_pct: float = 0.5,   # Start decaying confidence at 50% of max time
        target_profit: float = 0.001,   # Target profit (0.1%)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.theta = theta
        self.max_holding_periods = max_holding_periods
        self.decay_start_pct = decay_start_pct
        self.target_profit = target_profit

        # Position tracking
        self.entry_price = 0.0
        self.entry_time = 0.0
        self.position_side = 0
        self.holding_periods = 0

    def set_position(self, entry_price: float, side: int, entry_time: float = 0.0):
        """Set position"""
        self.entry_price = entry_price
        self.position_side = side
        self.entry_time = entry_time
        self.holding_periods = 0

    def _compute(self) -> None:
        """Update holding time"""
        if self.position_side != 0:
            self.holding_periods += 1

    def should_exit(self) -> ExitSignal:
        """Check time-based exit"""
        if self.position_side == 0 or self.entry_price <= 0:
            return ExitSignal(False, 'none', 0.0, 0.0, 'No position')

        current_price = self.prices[-1] if self.prices else 0
        if current_price <= 0:
            return ExitSignal(False, 'none', 0.0, 0.0, 'No price')

        # Calculate PnL
        if self.position_side == 1:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        # Check if target profit reached
        if pnl_pct >= self.target_profit:
            return ExitSignal(
                True, 'take_profit', 0.95,
                pnl_pct,
                f'Target profit reached: {pnl_pct*100:.3f}% >= {self.target_profit*100:.3f}%'
            )

        # Time decay logic
        time_pct = self.holding_periods / self.max_holding_periods

        if time_pct >= 1.0:
            # Max time exceeded - force exit
            return ExitSignal(
                True, 'time_decay', 0.85,
                pnl_pct,
                f'Max holding time exceeded: {self.holding_periods} periods'
            )

        # If past decay start and still in profit, take it
        if time_pct >= self.decay_start_pct and pnl_pct > 0:
            # Confidence increases as time passes (more urgent to exit)
            decay_confidence = 0.5 + 0.4 * ((time_pct - self.decay_start_pct) / (1 - self.decay_start_pct))
            return ExitSignal(
                True, 'time_decay', decay_confidence,
                pnl_pct,
                f'Time decay exit at {time_pct*100:.0f}% with {pnl_pct*100:.3f}% profit'
            )

        return ExitSignal(False, 'hold', 0.5, pnl_pct, f'Holding: {self.holding_periods}/{self.max_holding_periods}')

    def get_signal(self) -> int:
        exit_signal = self.should_exit()
        if exit_signal.should_exit:
            return -self.position_side
        return 0

    def get_confidence(self) -> float:
        return self.should_exit().confidence


# =============================================================================
# COMBINED EXIT MANAGER
# =============================================================================

class ExitManager:
    """
    Combines all exit formulas for comprehensive exit strategy.
    Uses IDs 320, 321, 322.
    """

    def __init__(
        self,
        take_profit_pct: float = 0.002,  # 0.2% take profit
        stop_loss_pct: float = 0.001,    # 0.1% stop loss
        trail_pct: float = 0.001,        # 0.1% trailing
        max_holding_seconds: int = 60,   # Max 60 seconds per trade
    ):
        # Initialize all exit formulas
        self.optimal_stop = OptimalStoppingFormula(
            take_profit_multiple=take_profit_pct / 0.001,  # Scale to edge
            stop_loss_multiple=stop_loss_pct / 0.001,
        )
        self.trailing_stop = TrailingStopFormula(
            trail_pct=trail_pct,
            min_profit_to_trail=take_profit_pct * 0.5,  # Trail after 50% of TP
        )
        self.time_exit = FirstExitTimeFormula(
            max_holding_periods=max_holding_seconds,
            target_profit=take_profit_pct,
        )

        self.position_side = 0
        self.entry_price = 0.0

    def set_position(self, entry_price: float, side: int, entry_time: float = 0.0):
        """Set position on all exit formulas"""
        self.entry_price = entry_price
        self.position_side = side
        self.optimal_stop.set_position(entry_price, side, entry_time)
        self.trailing_stop.set_position(entry_price, side, entry_time)
        self.time_exit.set_position(entry_price, side, entry_time)

    def clear_position(self):
        """Clear position state"""
        self.position_side = 0
        self.entry_price = 0.0
        self.optimal_stop.position_side = 0
        self.trailing_stop.position_side = 0
        self.time_exit.position_side = 0

    def update(self, price: float, volume: float = 0.0, timestamp: float = 0.0):
        """Update all formulas"""
        self.optimal_stop.update(price, volume, timestamp)
        self.trailing_stop.update(price, volume, timestamp)
        self.time_exit.update(price, volume, timestamp)

    def should_exit(self) -> ExitSignal:
        """
        Check all exit conditions and return highest priority exit signal.
        Priority: stop_loss > time_decay > trailing_stop > take_profit
        """
        if self.position_side == 0:
            return ExitSignal(False, 'none', 0.0, 0.0, 'No position')

        signals = [
            self.optimal_stop.should_exit(),
            self.trailing_stop.should_exit(),
            self.time_exit.should_exit(),
        ]

        # Priority order for exit types
        priority = {
            'stop_loss': 5,
            'time_decay': 4,
            'trailing_stop': 3,
            'take_profit': 2,
            'hold': 1,
            'none': 0,
        }

        # Find highest priority exit signal
        exit_signals = [s for s in signals if s.should_exit]
        if exit_signals:
            best_signal = max(
                exit_signals,
                key=lambda s: priority.get(s.exit_type, 0)
            )
            return best_signal

        return ExitSignal(False, 'hold', 0.5, 0.0, 'All conditions hold')

    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return {
            'position_side': self.position_side,
            'entry_price': self.entry_price,
            'optimal_stop': {
                'tp_level': self.optimal_stop.take_profit_level,
                'sl_level': self.optimal_stop.stop_loss_level,
            },
            'trailing_stop': {
                'is_trailing': self.trailing_stop.is_trailing,
                'trail_stop': self.trailing_stop.trail_stop,
                'running_max': self.trailing_stop.running_max,
            },
            'time_exit': {
                'holding_periods': self.time_exit.holding_periods,
                'max_periods': self.time_exit.max_holding_periods,
            },
        }
