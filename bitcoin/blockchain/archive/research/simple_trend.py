#!/usr/bin/env python3
"""
SIMPLE TREND FOLLOWING - NO COMPLEXITY
=======================================

Rules:
  - Price going UP   → LONG
  - Price going DOWN → SHORT

That's it. No quant formulas. No correlation databases.
No liquidity ratios. No 170-formula ensembles.

Just follow the trend.
"""

import ccxt
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional
import sqlite3
import os


@dataclass
class Position:
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: datetime
    size_usd: float
    stop_loss: float
    take_profit: float


class SimpleTrendFollower:
    """
    Dead simple trend following.

    If price went up in last N minutes → LONG
    If price went down in last N minutes → SHORT
    """

    def __init__(
        self,
        lookback_minutes: int = 5,
        min_move_pct: float = 0.1,  # 0.1% minimum move to trigger
        capital: float = 100.0,
        leverage: int = 10,
        stop_loss_pct: float = 0.5,  # 0.5% stop loss
        take_profit_pct: float = 1.0,  # 1% take profit
    ):
        self.lookback_minutes = lookback_minutes
        self.min_move_pct = min_move_pct
        self.capital = capital
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Price history: [(timestamp, price), ...]
        self.price_history = []

        # Current position
        self.position: Optional[Position] = None

        # Stats
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

        # Exchange connection
        self.exchange = ccxt.kraken()

        # Trade log
        self.db_path = os.environ.get('TRADE_DB', 'simple_trades.db')
        self._init_db()

    def _init_db(self):
        """Initialize trade database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                size_usd REAL,
                pnl REAL,
                pnl_pct REAL,
                exit_reason TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def get_price(self) -> float:
        """Get current BTC price."""
        ticker = self.exchange.fetch_ticker('BTC/USD')
        return ticker['last']

    def get_trend(self) -> Optional[str]:
        """
        Determine trend direction.

        Returns 'LONG', 'SHORT', or None (no clear trend).
        """
        now = datetime.now(timezone.utc)
        current_price = self.get_price()

        # Add to history
        self.price_history.append((now, current_price))

        # Keep only last 60 minutes of data
        cutoff = now.timestamp() - (60 * 60)
        self.price_history = [
            (t, p) for t, p in self.price_history
            if t.timestamp() > cutoff
        ]

        # Need enough history
        lookback_seconds = self.lookback_minutes * 60
        old_prices = [
            p for t, p in self.price_history
            if (now - t).total_seconds() >= lookback_seconds - 30
            and (now - t).total_seconds() <= lookback_seconds + 30
        ]

        if not old_prices:
            return None

        old_price = sum(old_prices) / len(old_prices)
        change_pct = (current_price - old_price) / old_price * 100

        if change_pct >= self.min_move_pct:
            return 'LONG'
        elif change_pct <= -self.min_move_pct:
            return 'SHORT'
        else:
            return None

    def open_position(self, direction: str, price: float):
        """Open a new position."""
        size_usd = self.capital * self.leverage

        if direction == 'LONG':
            stop_loss = price * (1 - self.stop_loss_pct / 100)
            take_profit = price * (1 + self.take_profit_pct / 100)
        else:  # SHORT
            stop_loss = price * (1 + self.stop_loss_pct / 100)
            take_profit = price * (1 - self.take_profit_pct / 100)

        self.position = Position(
            direction=direction,
            entry_price=price,
            entry_time=datetime.now(timezone.utc),
            size_usd=size_usd,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        print(f"\n[OPEN] {direction} @ ${price:,.2f}")
        print(f"       Size: ${size_usd:,.0f} | SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}")

    def check_exit(self, current_price: float) -> Optional[str]:
        """Check if position should be closed."""
        if not self.position:
            return None

        p = self.position

        if p.direction == 'LONG':
            if current_price <= p.stop_loss:
                return 'STOP_LOSS'
            if current_price >= p.take_profit:
                return 'TAKE_PROFIT'
        else:  # SHORT
            if current_price >= p.stop_loss:
                return 'STOP_LOSS'
            if current_price <= p.take_profit:
                return 'TAKE_PROFIT'

        return None

    def close_position(self, exit_price: float, reason: str):
        """Close current position."""
        if not self.position:
            return

        p = self.position

        # Calculate P&L
        if p.direction == 'LONG':
            pnl_pct = (exit_price - p.entry_price) / p.entry_price * 100
        else:  # SHORT
            pnl_pct = (p.entry_price - exit_price) / p.entry_price * 100

        pnl_usd = p.size_usd * (pnl_pct / 100)

        # Update stats
        self.total_pnl += pnl_usd
        if pnl_usd > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Log trade
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO trades (timestamp, direction, entry_price, exit_price,
                              size_usd, pnl, pnl_pct, exit_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(timezone.utc).isoformat(),
            p.direction,
            p.entry_price,
            exit_price,
            p.size_usd,
            pnl_usd,
            pnl_pct,
            reason
        ))
        conn.commit()
        conn.close()

        # Print result
        emoji = "+" if pnl_usd > 0 else ""
        print(f"\n[CLOSE] {p.direction} | {reason}")
        print(f"        Entry: ${p.entry_price:,.2f} -> Exit: ${exit_price:,.2f}")
        print(f"        P&L: {emoji}${pnl_usd:,.2f} ({emoji}{pnl_pct:.2f}%)")
        print(f"        Total P&L: ${self.total_pnl:,.2f} | Wins: {self.wins} | Losses: {self.losses}")

        self.position = None

    def run(self):
        """Main loop - dead simple."""
        print("=" * 60)
        print("SIMPLE TREND FOLLOWER")
        print("=" * 60)
        print(f"Lookback:    {self.lookback_minutes} minutes")
        print(f"Min move:    {self.min_move_pct}%")
        print(f"Capital:     ${self.capital}")
        print(f"Leverage:    {self.leverage}x")
        print(f"Stop Loss:   {self.stop_loss_pct}%")
        print(f"Take Profit: {self.take_profit_pct}%")
        print("=" * 60)
        print("\nWaiting for price data...\n")

        while True:
            try:
                price = self.get_price()
                now = datetime.now(timezone.utc).strftime('%H:%M:%S')

                # Check existing position
                if self.position:
                    exit_reason = self.check_exit(price)
                    if exit_reason:
                        self.close_position(price, exit_reason)
                    else:
                        # Show position status every 30 seconds
                        p = self.position
                        if p.direction == 'LONG':
                            pnl_pct = (price - p.entry_price) / p.entry_price * 100
                        else:
                            pnl_pct = (p.entry_price - price) / p.entry_price * 100
                        print(f"[{now}] ${price:,.2f} | {p.direction} | P&L: {pnl_pct:+.2f}%", end='\r')

                # Look for new trend if no position
                else:
                    trend = self.get_trend()
                    if trend:
                        self.open_position(trend, price)
                    else:
                        print(f"[{now}] ${price:,.2f} | Waiting for trend...", end='\r')

                time.sleep(10)  # Check every 10 seconds

            except KeyboardInterrupt:
                print("\n\nStopping...")
                if self.position:
                    price = self.get_price()
                    self.close_position(price, 'MANUAL')
                break
            except Exception as e:
                print(f"\nError: {e}")
                time.sleep(5)


def main():
    trader = SimpleTrendFollower(
        lookback_minutes=5,    # 5 minute trend
        min_move_pct=0.1,      # 0.1% minimum move
        capital=100.0,         # $100
        leverage=10,           # 10x
        stop_loss_pct=0.5,     # 0.5% stop
        take_profit_pct=1.0,   # 1% profit target
    )
    trader.run()


if __name__ == "__main__":
    main()
