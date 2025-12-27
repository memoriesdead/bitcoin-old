#!/usr/bin/env python3
"""
SHORT TRADER
============
INFLOW to exchange = Sellers depositing = Price DOWN = SHORT

Simple. Clean. One direction.
"""

import time
import sqlite3
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import CONFIG
from shared.price_feed import PriceFeed


class ExitReason(Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TIMEOUT = "timeout"
    MANUAL = "manual"


@dataclass
class Position:
    """A short position."""
    id: int
    exchange: str
    entry_price: float
    entry_time: float
    size_usd: float
    leverage: int
    stop_loss: float
    take_profit: float
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    pnl_usd: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_price is None


class ShortTrader:
    """
    SHORT only trader.

    Logic: INFLOW detected -> Open SHORT -> Exit on SL/TP/timeout
    """

    def __init__(self, db_path: str = None):
        self.config = CONFIG
        self.price_feed = PriceFeed()
        self.positions: Dict[int, Position] = {}
        self.next_id = 1
        self.capital = CONFIG.initial_capital
        self.db_path = db_path or CONFIG.db_path

        # Stats
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

    def on_inflow(self, exchange: str, flow_btc: float) -> Optional[Position]:
        """
        INFLOW detected - open SHORT position.

        Args:
            exchange: Which exchange saw the inflow
            flow_btc: Amount of BTC flowing in

        Returns:
            Position if opened, None if skipped
        """
        # Skip if flow too small
        if flow_btc < self.config.min_flow_btc:
            return None

        # Skip if exchange not tradeable
        if exchange.lower() not in self.config.tradeable:
            return None

        # Get current price
        price = self.price_feed.get(exchange)
        if not price:
            return None

        # Calculate position size
        size_usd = self.capital * self.config.position_size_pct
        leverage = self.config.max_leverage

        # Calculate exits
        stop_loss = price * (1 + self.config.stop_loss_pct)    # Price UP = loss for short
        take_profit = price * (1 - self.config.take_profit_pct)  # Price DOWN = profit

        # Create position
        pos = Position(
            id=self.next_id,
            exchange=exchange,
            entry_price=price,
            entry_time=time.time(),
            size_usd=size_usd,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.positions[self.next_id] = pos
        self.next_id += 1

        print(f"[SHORT] Opened #{pos.id} on {exchange} @ ${price:.2f}")
        print(f"        SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")

        return pos

    def check_exits(self) -> List[Position]:
        """Check all open positions for exit conditions."""
        closed = []
        now = time.time()

        for pos in list(self.positions.values()):
            if not pos.is_open:
                continue

            price = self.price_feed.get(pos.exchange)
            if not price:
                continue

            exit_reason = None

            # Stop loss - price went UP (bad for short)
            if price >= pos.stop_loss:
                exit_reason = ExitReason.STOP_LOSS

            # Take profit - price went DOWN (good for short)
            elif price <= pos.take_profit:
                exit_reason = ExitReason.TAKE_PROFIT

            # Timeout
            elif (now - pos.entry_time) >= self.config.exit_timeout_seconds:
                exit_reason = ExitReason.TIMEOUT

            if exit_reason:
                self._close_position(pos, price, exit_reason)
                closed.append(pos)

        return closed

    def _close_position(self, pos: Position, exit_price: float, reason: ExitReason):
        """Close a position and calculate P&L."""
        pos.exit_price = exit_price
        pos.exit_time = time.time()
        pos.exit_reason = reason

        # P&L for SHORT: profit when price goes DOWN
        price_change_pct = (pos.entry_price - exit_price) / pos.entry_price
        pos.pnl_usd = pos.size_usd * pos.leverage * price_change_pct

        # Subtract fees
        fee = self.config.get_fee(pos.exchange)
        pos.pnl_usd -= pos.size_usd * fee * 2  # Entry + exit

        # Update stats
        self.total_trades += 1
        self.total_pnl += pos.pnl_usd
        self.capital += pos.pnl_usd

        if pos.pnl_usd > 0:
            self.wins += 1
        else:
            self.losses += 1

        print(f"[SHORT] Closed #{pos.id} @ ${exit_price:.2f} ({reason.value})")
        print(f"        P&L: ${pos.pnl_usd:.2f} | Capital: ${self.capital:.2f}")

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [p for p in self.positions.values() if p.is_open]

    def get_stats(self) -> Dict:
        """Get trading statistics."""
        win_rate = self.wins / self.total_trades if self.total_trades > 0 else 0
        return {
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'capital': self.capital,
            'open_positions': len(self.get_open_positions()),
        }

    def start(self):
        """Start price feed."""
        self.price_feed.start()

    def stop(self):
        """Stop price feed."""
        self.price_feed.stop()


# Simple test
if __name__ == "__main__":
    trader = ShortTrader()
    trader.start()

    print("SHORT Trader initialized")
    print(f"Capital: ${trader.capital}")
    print(f"Exchanges: {trader.config.tradeable}")

    # Simulate an inflow signal
    pos = trader.on_inflow('coinbase', 50.0)

    if pos:
        print(f"\nPosition opened: {pos}")

    # Check stats
    print(f"\nStats: {trader.get_stats()}")

    trader.stop()
