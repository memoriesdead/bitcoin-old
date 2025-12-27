#!/usr/bin/env python3
"""
DETERMINISTIC TRADER
====================

Single responsibility: Manage positions based on signals.

FEATURES:
- Uses TradingConfig for all settings
- Time-based exits (no flow reversal exits)
- Per-exchange P&L tracking
- Clean position management
"""

import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from enum import Enum

from config import TradingConfig, get_config
from correlation_formula import Signal, SignalType


class PositionStatus(Enum):
    """Position status."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    STOPPED_OUT = "STOPPED_OUT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TIMED_OUT = "TIMED_OUT"


@dataclass
class Position:
    """A trading position."""
    id: int
    exchange: str
    direction: SignalType  # SHORT or LONG
    entry_price: float
    entry_time: datetime
    size_usd: float
    size_btc: float
    leverage: int
    stop_loss: float
    take_profit: float
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class TraderStats:
    """Trader statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_usd: float = 0.0
    current_capital: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_capital: float = 0.0
    per_exchange_pnl: Dict[str, float] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades


class DeterministicTrader:
    """
    Deterministic position manager.

    Uses signals from CorrelationFormula to open positions.
    Uses time-based exits (not flow reversal).
    Tracks P&L per exchange.
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or get_config()
        self.lock = threading.Lock()

        # Active positions
        self.positions: Dict[int, Position] = {}
        self.position_counter = 0

        # Statistics
        self.stats = TraderStats(current_capital=self.config.initial_capital)
        self.stats.peak_capital = self.config.initial_capital

        # Initialize database
        self._init_db()

        # Load historical stats
        self._load_stats()

    def _init_db(self):
        """Initialize trades database."""
        conn = sqlite3.connect(self.config.trades_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exchange TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_price REAL,
                exit_time TEXT,
                size_usd REAL NOT NULL,
                size_btc REAL NOT NULL,
                leverage INTEGER NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                status TEXT NOT NULL,
                pnl_usd REAL DEFAULT 0.0,
                pnl_pct REAL DEFAULT 0.0,
                exit_reason TEXT,
                signal_correlation REAL,
                signal_win_rate REAL,
                signal_samples INTEGER
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                capital REAL NOT NULL,
                open_positions INTEGER NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def _load_stats(self):
        """Load historical statistics from database."""
        try:
            conn = sqlite3.connect(self.config.trades_db_path)
            cursor = conn.cursor()

            # Get totals
            cursor.execute("""
                SELECT COUNT(*),
                       SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END),
                       SUM(pnl_usd)
                FROM trades
                WHERE status != 'OPEN'
            """)
            row = cursor.fetchone()
            if row and row[0]:
                self.stats.total_trades = row[0]
                self.stats.winning_trades = row[1] or 0
                self.stats.losing_trades = self.stats.total_trades - self.stats.winning_trades
                self.stats.total_pnl_usd = row[2] or 0.0
                self.stats.current_capital = self.config.initial_capital + self.stats.total_pnl_usd

            # Get per-exchange P&L
            cursor.execute("""
                SELECT exchange, SUM(pnl_usd)
                FROM trades
                WHERE status != 'OPEN'
                GROUP BY exchange
            """)
            for exchange, pnl in cursor.fetchall():
                self.stats.per_exchange_pnl[exchange] = pnl

            conn.close()
        except Exception:
            pass

    def can_open_position(self, exchange: str) -> bool:
        """Check if we can open a new position."""
        with self.lock:
            # Check max positions
            if len(self.positions) >= self.config.max_positions:
                return False

            # Check if we already have a position on this exchange
            for pos in self.positions.values():
                if pos.exchange.lower() == exchange.lower():
                    return False

            # Check if exchange is tradeable
            if not self.config.is_tradeable(exchange):
                return False

            return True

    def open_position(
        self,
        signal: Signal,
        current_price: float
    ) -> Optional[Position]:
        """
        Open a position based on signal.

        Returns Position if opened, None otherwise.
        """
        if not self.can_open_position(signal.exchange):
            return None

        if not signal.is_tradeable:
            return None

        with self.lock:
            # Calculate position size
            position_capital = self.stats.current_capital * self.config.position_size_pct
            size_usd = position_capital * self.config.max_leverage
            size_btc = size_usd / current_price

            # Calculate stop loss and take profit
            if signal.direction == SignalType.SHORT:
                stop_loss = current_price * (1 + self.config.stop_loss_pct)
                take_profit = current_price * (1 - self.config.take_profit_pct)
            else:  # LONG
                stop_loss = current_price * (1 - self.config.stop_loss_pct)
                take_profit = current_price * (1 + self.config.take_profit_pct)

            # Create position
            self.position_counter += 1
            position = Position(
                id=self.position_counter,
                exchange=signal.exchange.lower(),
                direction=signal.direction,
                entry_price=current_price,
                entry_time=signal.timestamp,
                size_usd=size_usd,
                size_btc=size_btc,
                leverage=self.config.max_leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

            self.positions[position.id] = position

            # Save to database
            self._save_position(position, signal)

            return position

    def _save_position(self, position: Position, signal: Signal):
        """Save position to database."""
        conn = sqlite3.connect(self.config.trades_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO trades (
                exchange, direction, entry_price, entry_time,
                size_usd, size_btc, leverage, stop_loss, take_profit,
                status, signal_correlation, signal_win_rate, signal_samples
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position.exchange,
            position.direction.value,
            position.entry_price,
            position.entry_time.isoformat(),
            position.size_usd,
            position.size_btc,
            position.leverage,
            position.stop_loss,
            position.take_profit,
            position.status.value,
            signal.correlation,
            signal.win_rate,
            signal.sample_count
        ))

        position.id = cursor.lastrowid
        conn.commit()
        conn.close()

    def check_exits(self, current_price: float, current_time: datetime) -> List[Position]:
        """
        Check all positions for exit conditions.

        Exit conditions (in priority order):
        1. Stop loss hit
        2. Take profit hit
        3. Time-based exit (5 minutes)

        Returns list of closed positions.
        """
        closed = []

        with self.lock:
            for position in list(self.positions.values()):
                exit_reason = None

                # Check stop loss
                if position.direction == SignalType.SHORT:
                    if current_price >= position.stop_loss:
                        exit_reason = "STOP_LOSS"
                        position.status = PositionStatus.STOPPED_OUT
                else:  # LONG
                    if current_price <= position.stop_loss:
                        exit_reason = "STOP_LOSS"
                        position.status = PositionStatus.STOPPED_OUT

                # Check take profit
                if not exit_reason:
                    if position.direction == SignalType.SHORT:
                        if current_price <= position.take_profit:
                            exit_reason = "TAKE_PROFIT"
                            position.status = PositionStatus.TAKE_PROFIT
                    else:  # LONG
                        if current_price >= position.take_profit:
                            exit_reason = "TAKE_PROFIT"
                            position.status = PositionStatus.TAKE_PROFIT

                # Check timeout
                if not exit_reason:
                    age_seconds = (current_time - position.entry_time).total_seconds()
                    if age_seconds >= self.config.exit_timeout_seconds:
                        exit_reason = "TIMEOUT"
                        position.status = PositionStatus.TIMED_OUT

                # Close position if exit triggered
                if exit_reason:
                    self._close_position(position, current_price, current_time, exit_reason)
                    closed.append(position)

        return closed

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ):
        """Close a position and calculate P&L."""
        position.exit_price = exit_price
        position.exit_time = exit_time
        position.exit_reason = exit_reason

        # Calculate P&L
        if position.direction == SignalType.SHORT:
            # SHORT: profit when price drops
            price_change_pct = (position.entry_price - exit_price) / position.entry_price
        else:
            # LONG: profit when price rises
            price_change_pct = (exit_price - position.entry_price) / position.entry_price

        # Apply leverage
        position.pnl_pct = price_change_pct * position.leverage

        # Deduct fees
        fee = self.config.get_fee(position.exchange)
        position.pnl_pct -= (fee * 2)  # Entry + exit fees

        # Calculate USD P&L (on collateral, not leveraged amount)
        collateral = position.size_usd / position.leverage
        position.pnl_usd = collateral * position.pnl_pct

        # Update stats
        self.stats.total_trades += 1
        self.stats.total_pnl_usd += position.pnl_usd
        self.stats.current_capital += position.pnl_usd

        if position.pnl_usd > 0:
            self.stats.winning_trades += 1
        else:
            self.stats.losing_trades += 1

        # Update per-exchange P&L
        if position.exchange not in self.stats.per_exchange_pnl:
            self.stats.per_exchange_pnl[position.exchange] = 0.0
        self.stats.per_exchange_pnl[position.exchange] += position.pnl_usd

        # Update peak and drawdown
        if self.stats.current_capital > self.stats.peak_capital:
            self.stats.peak_capital = self.stats.current_capital
        else:
            drawdown = (self.stats.peak_capital - self.stats.current_capital) / self.stats.peak_capital
            if drawdown > self.stats.max_drawdown_pct:
                self.stats.max_drawdown_pct = drawdown

        # Remove from active positions
        if position.id in self.positions:
            del self.positions[position.id]

        # Update database
        self._update_position(position)
        self._record_equity(exit_time)

    def _update_position(self, position: Position):
        """Update position in database."""
        conn = sqlite3.connect(self.config.trades_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE trades SET
                exit_price = ?,
                exit_time = ?,
                status = ?,
                pnl_usd = ?,
                pnl_pct = ?,
                exit_reason = ?
            WHERE id = ?
        """, (
            position.exit_price,
            position.exit_time.isoformat() if position.exit_time else None,
            position.status.value,
            position.pnl_usd,
            position.pnl_pct,
            position.exit_reason,
            position.id
        ))

        conn.commit()
        conn.close()

    def _record_equity(self, timestamp: datetime):
        """Record equity curve point."""
        conn = sqlite3.connect(self.config.trades_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO equity_curve (timestamp, capital, open_positions)
            VALUES (?, ?, ?)
        """, (
            timestamp.isoformat(),
            self.stats.current_capital,
            len(self.positions)
        ))

        conn.commit()
        conn.close()

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_stats(self) -> Dict:
        """Get trader statistics."""
        return {
            "capital": f"${self.stats.current_capital:.2f}",
            "total_trades": self.stats.total_trades,
            "win_rate": f"{self.stats.win_rate:.1%}",
            "winning": self.stats.winning_trades,
            "losing": self.stats.losing_trades,
            "total_pnl": f"${self.stats.total_pnl_usd:+.2f}",
            "max_drawdown": f"{self.stats.max_drawdown_pct:.1%}",
            "open_positions": len(self.positions),
            "per_exchange": {
                ex: f"${pnl:+.2f}"
                for ex, pnl in self.stats.per_exchange_pnl.items()
            }
        }


def format_position_open(position: Position) -> str:
    """Format position opening for logging."""
    return (
        f"[OPEN] {position.direction.value} {position.exchange.upper()} "
        f"@ ${position.entry_price:,.2f} | Size: ${position.size_usd:,.0f} "
        f"({position.size_btc:.4f} BTC) | SL: ${position.stop_loss:,.2f} "
        f"| TP: ${position.take_profit:,.2f}"
    )


def format_position_close(position: Position) -> str:
    """Format position closing for logging."""
    return (
        f"[CLOSE] {position.direction.value} {position.exchange.upper()} "
        f"| Entry: ${position.entry_price:,.2f} -> Exit: ${position.exit_price:,.2f} "
        f"| P&L: ${position.pnl_usd:+.2f} ({position.pnl_pct:+.1%}) "
        f"| Reason: {position.exit_reason}"
    )


def main():
    """Test the trader."""
    print("=" * 70)
    print("DETERMINISTIC TRADER")
    print("=" * 70)
    print()

    config = get_config()
    trader = DeterministicTrader(config)

    print(f"Config:")
    print(f"  - Initial capital: ${config.initial_capital}")
    print(f"  - Max leverage: {config.max_leverage}x")
    print(f"  - Max positions: {config.max_positions}")
    print(f"  - Position size: {config.position_size_pct:.0%}")
    print(f"  - Exit timeout: {config.exit_timeout_seconds}s")
    print(f"  - Stop loss: {config.stop_loss_pct:.1%}")
    print(f"  - Take profit: {config.take_profit_pct:.1%}")
    print()

    stats = trader.get_stats()
    print(f"Stats: {stats}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
