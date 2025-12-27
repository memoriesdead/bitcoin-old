"""
Safety Module
=============

Safety mechanisms for trading.

Features:
- Position limits
- Daily loss limits
- Circuit breakers
- Kill switch
"""

import time
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from collections import deque


@dataclass
class SafetyConfig:
    """Safety configuration."""
    # Position limits
    max_position_usd: float = 1000.0      # Per-trade limit
    max_total_exposure_usd: float = 5000.0  # Total open positions
    max_position_pct: float = 0.2         # Max % of capital per trade

    # Daily limits
    max_daily_trades: int = 100
    max_daily_loss_usd: float = 200.0
    max_daily_loss_pct: float = 0.10      # 10% of capital

    # Circuit breakers
    consecutive_loss_limit: int = 5       # Pause after N losses
    max_drawdown_pct: float = 0.10        # Pause if drawdown > 10%
    api_error_limit: int = 3              # Errors in 5 min
    api_error_window: int = 300           # 5 minutes

    # Timeouts
    order_timeout_seconds: float = 60.0
    position_timeout_hours: float = 24.0  # Close positions older than this


class CircuitBreaker:
    """
    Circuit breaker for automatic trading pause.

    Monitors conditions and triggers pause when thresholds exceeded.
    """

    def __init__(self, name: str, threshold: float, reset_after: float = 3600):
        """
        Initialize circuit breaker.

        Args:
            name: Breaker identifier
            threshold: Value that triggers break
            reset_after: Seconds before auto-reset
        """
        self.name = name
        self.threshold = threshold
        self.reset_after = reset_after

        self.current_value = 0.0
        self.triggered = False
        self.triggered_at: Optional[float] = None
        self.trigger_count = 0

    def update(self, value: float) -> bool:
        """
        Update value and check threshold.

        Returns:
            True if breaker just triggered
        """
        self.current_value = value

        if not self.triggered and value >= self.threshold:
            self.triggered = True
            self.triggered_at = time.time()
            self.trigger_count += 1
            return True

        # Check for auto-reset
        if self.triggered and self.triggered_at:
            if time.time() - self.triggered_at > self.reset_after:
                self.reset()

        return False

    def reset(self):
        """Manually reset breaker."""
        self.triggered = False
        self.triggered_at = None

    def is_triggered(self) -> bool:
        """Check if breaker is currently triggered."""
        return self.triggered

    def get_status(self) -> Dict[str, Any]:
        """Get breaker status."""
        return {
            'name': self.name,
            'threshold': self.threshold,
            'current_value': self.current_value,
            'triggered': self.triggered,
            'triggered_at': self.triggered_at,
            'trigger_count': self.trigger_count,
        }


class PositionLimiter:
    """
    Tracks and limits positions.
    """

    def __init__(self, max_per_trade: float, max_total: float):
        self.max_per_trade = max_per_trade
        self.max_total = max_total

        self.positions: Dict[str, float] = {}

    def can_open(self, symbol: str, size_usd: float) -> tuple:
        """
        Check if position can be opened.

        Returns:
            (allowed, reason)
        """
        if size_usd > self.max_per_trade:
            return False, f"Size {size_usd} exceeds per-trade limit {self.max_per_trade}"

        current_total = sum(self.positions.values())
        if current_total + size_usd > self.max_total:
            return False, f"Would exceed total exposure limit {self.max_total}"

        return True, "OK"

    def open_position(self, symbol: str, size_usd: float):
        """Record position opening."""
        self.positions[symbol] = self.positions.get(symbol, 0) + size_usd

    def close_position(self, symbol: str, size_usd: float):
        """Record position closing."""
        if symbol in self.positions:
            self.positions[symbol] -= size_usd
            if self.positions[symbol] <= 0:
                del self.positions[symbol]

    def get_total_exposure(self) -> float:
        """Get total exposure."""
        return sum(self.positions.values())


class SafetyManager:
    """
    Main safety manager.

    Coordinates all safety mechanisms.
    """

    def __init__(self, config: SafetyConfig):
        """
        Initialize safety manager.

        Args:
            config: Safety configuration
        """
        self.config = config

        # State
        self.enabled = True
        self.kill_switch = False

        # Daily tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.daily_reset_time = time.time()

        # Capital tracking
        self.initial_capital = 0.0
        self.current_capital = 0.0
        self.peak_capital = 0.0

        # Position limiter
        self.position_limiter = PositionLimiter(
            config.max_position_usd,
            config.max_total_exposure_usd
        )

        # Circuit breakers
        self.breakers = {
            'consecutive_losses': CircuitBreaker(
                'consecutive_losses',
                config.consecutive_loss_limit,
                reset_after=1800,  # 30 min
            ),
            'drawdown': CircuitBreaker(
                'drawdown',
                config.max_drawdown_pct,
                reset_after=3600,  # 1 hour
            ),
            'api_errors': CircuitBreaker(
                'api_errors',
                config.api_error_limit,
                reset_after=config.api_error_window,
            ),
        }

        # Recent trades for pattern detection
        self.recent_trades: deque = deque(maxlen=100)
        self.api_errors: deque = deque(maxlen=100)

        # Consecutive loss tracking
        self.consecutive_losses = 0

    def set_capital(self, capital: float):
        """Set current capital for calculations."""
        if self.initial_capital == 0:
            self.initial_capital = capital
        self.current_capital = capital
        self.peak_capital = max(self.peak_capital, capital)

        # Update drawdown breaker
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - capital) / self.peak_capital
            self.breakers['drawdown'].update(drawdown)

    def check_order(self, symbol: str, side: str, amount: float,
                    price: float) -> Dict[str, Any]:
        """
        Check if order is allowed.

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            amount: Order amount
            price: Order price

        Returns:
            Dict with 'allowed' bool and 'reason'
        """
        # Kill switch
        if self.kill_switch:
            return {'allowed': False, 'reason': 'Kill switch active'}

        # Disabled
        if not self.enabled:
            return {'allowed': False, 'reason': 'Safety disabled'}

        # Circuit breakers
        for name, breaker in self.breakers.items():
            if breaker.is_triggered():
                return {'allowed': False, 'reason': f'Circuit breaker: {name}'}

        # Check daily reset
        self._check_daily_reset()

        # Daily trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            return {'allowed': False, 'reason': 'Daily trade limit reached'}

        # Daily loss limit
        if self.daily_pnl <= -self.config.max_daily_loss_usd:
            return {'allowed': False, 'reason': 'Daily loss limit reached'}

        # Position size
        size_usd = amount * price
        allowed, reason = self.position_limiter.can_open(symbol, size_usd)
        if not allowed:
            return {'allowed': False, 'reason': reason}

        # Max position as % of capital
        if self.current_capital > 0:
            pct = size_usd / self.current_capital
            if pct > self.config.max_position_pct:
                return {'allowed': False, 'reason': f'Position size {pct*100:.1f}% exceeds limit'}

        return {'allowed': True, 'reason': 'OK'}

    def record_trade(self, pnl: float, is_win: bool):
        """
        Record trade outcome.

        Args:
            pnl: Trade PnL
            is_win: Whether trade was profitable
        """
        self.daily_trades += 1
        self.daily_pnl += pnl

        # Track consecutive losses
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.breakers['consecutive_losses'].update(self.consecutive_losses)

        self.recent_trades.append({
            'time': time.time(),
            'pnl': pnl,
            'is_win': is_win,
        })

    def record_api_error(self, error: str):
        """Record API error for monitoring."""
        now = time.time()
        self.api_errors.append({'time': now, 'error': error})

        # Count recent errors
        cutoff = now - self.config.api_error_window
        recent_count = sum(1 for e in self.api_errors if e['time'] > cutoff)

        self.breakers['api_errors'].update(recent_count)

    def activate_kill_switch(self, reason: str = "Manual"):
        """Activate kill switch to stop all trading."""
        self.kill_switch = True
        print(f"[KILL SWITCH] Activated: {reason}")

    def deactivate_kill_switch(self):
        """Deactivate kill switch."""
        self.kill_switch = False
        print("[KILL SWITCH] Deactivated")

    def reset_breaker(self, name: str):
        """Reset a specific circuit breaker."""
        if name in self.breakers:
            self.breakers[name].reset()

    def _check_daily_reset(self):
        """Check if daily counters should reset."""
        now = time.time()
        # Reset at midnight UTC
        if now - self.daily_reset_time > 86400:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.daily_reset_time = now

    def get_stats(self) -> Dict[str, Any]:
        """Get safety statistics."""
        return {
            'enabled': self.enabled,
            'kill_switch': self.kill_switch,
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'total_exposure': self.position_limiter.get_total_exposure(),
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'breakers': {
                name: breaker.get_status()
                for name, breaker in self.breakers.items()
            },
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("Safety Manager Demo")
    print("=" * 50)

    config = SafetyConfig(
        max_position_usd=1000,
        max_total_exposure_usd=5000,
        max_daily_trades=10,
        max_daily_loss_usd=200,
        consecutive_loss_limit=3,
    )

    safety = SafetyManager(config)
    safety.set_capital(10000)

    # Test order checks
    print("\nOrder Checks:")

    # Valid order
    result = safety.check_order("BTC/USDT", "buy", 0.02, 42000)
    print(f"  0.02 BTC ($840): {result}")

    # Too large
    result = safety.check_order("BTC/USDT", "buy", 0.05, 42000)
    print(f"  0.05 BTC ($2100): {result}")

    # Simulate losses
    print("\nSimulating losses:")
    for i in range(4):
        safety.record_trade(-50, is_win=False)
        print(f"  Loss {i+1}: consecutive_losses={safety.consecutive_losses}")

        result = safety.check_order("BTC/USDT", "buy", 0.01, 42000)
        print(f"    Can trade: {result['allowed']}")

    print(f"\nStats: {safety.get_stats()}")
