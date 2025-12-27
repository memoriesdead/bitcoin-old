"""
RenTech Signal Engine - ONE EDGE, EXECUTE, COMPOUND
====================================================
"We're right 50.75% of the time, but we're 100% right 50.75% of the time."

THE EDGE: Blockchain ZMQ shows exchange flows 10-60 seconds before price moves.
- INFLOW to exchange = They will SELL = SHORT
- OUTFLOW from exchange = They are accumulating = LONG

That's it. No 12 formulas. No priority chains. No override logic.
One signal. One decision. Compound.
"""

import time
import threading
from typing import Optional, Dict, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from ..config.exchanges import EXCHANGE_ENDPOINTS

if TYPE_CHECKING:
    from ..ai.claude_adapter import ClaudeAdapter


@dataclass
class TradeSignal:
    timestamp: float
    direction: int  # 1 = LONG, -1 = SHORT
    probability: float
    position_size: float
    price: float
    regime: str = "blockchain"
    strategy: str = "flow"
    regime_confidence: float = 0.5075
    signal_confidence: float = 0.5075
    gate_passed: bool = True
    exchange_id: str = ""
    adaptive_sl: float = 0.003  # 0.3% stop loss
    adaptive_tp: float = 0.005  # 0.5% take profit
    adaptive_hold: float = 60   # 60 second max hold


class ExchangeSignalEngine:
    """
    RenTech Style: ONE EDGE, EXECUTE, COMPOUND

    The only signal source is blockchain ZMQ exchange flows.
    Everything else was noise causing 0% win rate.
    """

    def __init__(self, exchange_id: str, capital: float = 5.0,
                 claude: "ClaudeAdapter" = None):
        self.exchange_id = exchange_id
        self.capital = capital
        self.initial_capital = capital
        self.claude = claude

        # Exchange config
        self.config = EXCHANGE_ENDPOINTS.get(exchange_id, {})
        self.maker_fee = self.config.get("maker_fee", 0.001)
        self.taker_fee = self.config.get("taker_fee", 0.002)
        self.total_fees = self.maker_fee + self.taker_fee

        # Simple parameters - no "adaptive" complexity
        self.stop_loss_pct = 0.003    # 0.3%
        self.take_profit_pct = 0.005  # 0.5%
        self.max_hold_seconds = 10    # 10 seconds - rapid trading
        self.min_confidence = 0.5075  # RenTech: just above 50%

        # Position state
        self.current_price = 0.0
        self.position = 0.0
        self.position_direction = 0
        self.entry_price = 0.0
        self.entry_time = 0.0

        # Stats
        self.total_trades = 0
        self.closed_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

        self.exits_by_sl = 0
        self.exits_by_tp = 0
        self.exits_by_time = 0
        self.exits_by_reversal = 0

        # Cooldown
        self.last_trade_time = 0.0
        self.cooldown = 1.0  # 1 second cooldown - rapid trading

        self.lock = threading.Lock()

        print(f"[{exchange_id.upper()}] RenTech engine: blockchain flow only")

    def update_price(self, price: float, volume: float = None) -> None:
        """Update current price."""
        with self.lock:
            self.current_price = price

    def check_exit_conditions(self) -> Optional[Dict]:
        """Check if we should exit current position."""
        if self.position_direction == 0 or self.entry_price == 0:
            return None

        now = time.time()
        price = self.current_price

        # Calculate PnL
        if self.position_direction == 1:  # LONG
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:  # SHORT
            pnl_pct = (self.entry_price - price) / self.entry_price

        # Check exits
        if pnl_pct <= -self.stop_loss_pct:
            return {"reason": "stop_loss", "pnl_pct": pnl_pct}
        if pnl_pct >= self.take_profit_pct:
            return {"reason": "take_profit", "pnl_pct": pnl_pct}
        if now - self.entry_time >= self.max_hold_seconds:
            return {"reason": "time_exit", "pnl_pct": pnl_pct}

        return None

    def generate_signal(self, mempool_momentum: float = 0.0, blockchain_signal: dict = None) -> Optional[TradeSignal]:
        """
        RenTech Style: ONE signal source. ONE decision.

        blockchain_signal comes from per_exchange_feed.get_aggregated_signal()
        It contains real-time exchange inflow/outflow data from Bitcoin Core ZMQ.

        That's our edge. Nothing else.
        """
        with self.lock:
            if self.current_price == 0:
                return None

            now = time.time()

            # Cooldown
            if now - self.last_trade_time < self.cooldown:
                return None

            # Already in position - don't open another
            if self.position_direction != 0:
                return None

            # ===========================================
            # THE ONLY SIGNAL: BLOCKCHAIN EXCHANGE FLOWS
            # ===========================================
            if not blockchain_signal:
                return None

            if not blockchain_signal.get("should_trade", False):
                return None

            # Get direction (FIXED: was "signal", should be "direction")
            direction = blockchain_signal.get("direction", 0)
            if direction == 0:
                return None

            confidence = blockchain_signal.get("confidence", 0.5)

            # Gate: must beat fees (50.75% minimum)
            if confidence < self.min_confidence:
                return None

            # Position size: Use formula's Kelly-optimal size if provided
            position_size = blockchain_signal.get('position_size', 0.02)
            # Fallback: Kelly criterion simplified
            if position_size <= 0:
                edge = confidence - 0.5
                position_size = min(0.25, max(0.02, edge * 2))  # 2-25% of capital

            return TradeSignal(
                timestamp=now,
                direction=direction,
                probability=confidence,
                position_size=position_size,
                price=self.current_price,
                regime="blockchain",
                strategy="flow",
                regime_confidence=confidence,
                signal_confidence=confidence,
                gate_passed=True,
                exchange_id=self.exchange_id,
                adaptive_sl=self.stop_loss_pct,
                adaptive_tp=self.take_profit_pct,
                adaptive_hold=self.max_hold_seconds,
            )

    def execute_trade(self, signal: TradeSignal) -> Optional[Dict]:
        """Execute a trade signal."""
        with self.lock:
            if signal.direction == 0:
                return None

            # CLAUDE AI CONFIRMATION (if enabled)
            if self.claude:
                approval = self.claude.confirm_trade(
                    {
                        'direction': signal.direction,
                        'position_size': signal.position_size,
                        'price': signal.price,
                        'stop_loss': signal.adaptive_sl,
                        'take_profit': signal.adaptive_tp,
                    },
                    {
                        'capital': self.capital,
                        'drawdown': self._get_drawdown(),
                        'win_rate': self.get_win_rate(),
                        'recent_trades': self._get_recent_pnl_pct(),
                    }
                )
                if approval.success:
                    if approval.action == "REJECT":
                        print(f"[{self.exchange_id.upper()}] Claude REJECTED trade: {approval.reasoning}")
                        return None
                    elif approval.action == "ADJUST":
                        signal.position_size *= approval.size_adjustment
                        print(f"[{self.exchange_id.upper()}] Claude ADJUSTED size: *{approval.size_adjustment:.2f}")

            pnl = 0.0
            exit_reason = None

            # Close existing position if reversing
            if self.position_direction != 0:
                pnl, exit_reason = self._close_position(signal.price, "reversal")

            # Open new position
            position_value = self.capital * signal.position_size
            self.position = position_value / signal.price
            self.position_direction = signal.direction
            self.entry_price = signal.price
            self.entry_time = signal.timestamp
            self.last_trade_time = signal.timestamp
            self.total_trades += 1

            return {
                "exchange": self.exchange_id,
                "type": "LONG" if signal.direction == 1 else "SHORT",
                "price": signal.price,
                "size": self.position,
                "value": position_value,
                "probability": signal.probability,
                "regime": signal.regime,
                "strategy": signal.strategy,
                "regime_confidence": signal.regime_confidence,
                "timestamp": signal.timestamp,
                "pnl": pnl,
                "exit_reason": exit_reason,
            }

    def force_exit(self, reason: str) -> Optional[Dict]:
        """Force exit current position."""
        with self.lock:
            if self.position_direction == 0:
                return None

            pnl, _ = self._close_position(self.current_price, reason)

            return {
                "exchange": self.exchange_id,
                "type": "EXIT",
                "reason": reason,
                "price": self.current_price,
                "pnl": pnl,
                "timestamp": time.time(),
            }

    def _close_position(self, current_price: float, reason: str) -> Tuple[float, str]:
        """Close current position and calculate PnL."""
        if self.position == 0 or self.entry_price == 0:
            return 0.0, reason

        # Calculate PnL
        if self.position_direction == 1:  # LONG
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        # Subtract fees
        pnl_pct -= self.total_fees
        pnl = self.position * self.entry_price * pnl_pct

        # Update stats
        self.capital += pnl
        self.total_pnl += pnl
        self.closed_trades += 1

        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Track exit reasons
        if reason == "stop_loss":
            self.exits_by_sl += 1
        elif reason == "take_profit":
            self.exits_by_tp += 1
        elif reason == "time_exit":
            self.exits_by_time += 1
        elif reason == "reversal":
            self.exits_by_reversal += 1

        # Reset position
        self.position = 0.0
        self.position_direction = 0
        self.entry_price = 0.0
        self.entry_time = 0.0

        return pnl, reason

    def get_win_rate(self) -> float:
        """Get current win rate."""
        if self.closed_trades == 0:
            return 0.0
        return self.wins / self.closed_trades * 100

    def _get_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        peak = max(self.initial_capital, self.capital + abs(self.total_pnl))
        if peak <= 0:
            return 0.0
        return max(0, (peak - self.capital) / peak)

    def _get_recent_pnl_pct(self) -> list:
        """Get recent trade PnL percentages."""
        # This is a simplified implementation - in production would track actual trade results
        return []

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            "exchange": self.exchange_id,
            "capital": self.capital,
            "pnl": self.total_pnl,
            "pnl_pct": ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            "trades_opened": self.total_trades,
            "trades_closed": self.closed_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.get_win_rate(),
            "drawdown": self._get_drawdown() * 100,
            "exits_sl": self.exits_by_sl,
            "exits_tp": self.exits_by_tp,
            "exits_time": self.exits_by_time,
            "exits_reversal": self.exits_by_reversal,
        }
