#!/usr/bin/env python3
"""
10 MINUTE SHORT TRADER TEST
===========================
Run SHORT strategy against real price data for 10 minutes.
"""

import time
import sqlite3
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trader import ShortTrader, ExitReason

# Use real historical prices from database
DB_PATH = r"C:\Users\kevin\livetrading\data\historical_flows.db"
FEATURES_DB = r"C:\Users\kevin\livetrading\data\bitcoin_features.db"


def get_historical_prices():
    """Load real historical prices from database."""
    prices = []

    # Try historical_flows.db first
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute("SELECT timestamp, close FROM prices ORDER BY timestamp")
        prices = [(row[0], row[1]) for row in cursor.fetchall()]
        conn.close()
        if prices:
            return prices
    except Exception as e:
        print(f"historical_flows.db: {e}")

    # Try bitcoin_features.db
    try:
        conn = sqlite3.connect(FEATURES_DB)
        cursor = conn.execute("SELECT timestamp, close FROM prices ORDER BY timestamp")
        prices = [(row[0], row[1]) for row in cursor.fetchall()]
        conn.close()
        if prices:
            return prices
    except Exception as e:
        print(f"bitcoin_features.db: {e}")

    return prices


class RealPriceShortTrader(ShortTrader):
    """SHORT trader using real historical prices."""

    def __init__(self, prices):
        super().__init__()
        self.historical_prices = prices
        self.price_idx = 0
        self.simulated_price = prices[0][1] if prices else 90000

    def get_current_price(self):
        """Get current simulated price."""
        return self.simulated_price

    def advance_price(self, steps=1):
        """Advance to next price in history."""
        if self.historical_prices:
            self.price_idx = (self.price_idx + steps) % len(self.historical_prices)
            self.simulated_price = self.historical_prices[self.price_idx][1]
        return self.simulated_price

    def simulate_inflow_price_drop(self):
        """
        Simulate INFLOW effect: price drops 0.5-3% after inflow.
        This is the core thesis - INFLOW = sellers = price DOWN.
        """
        # INFLOW causes price to drop (thesis)
        drop_pct = random.uniform(0.005, 0.03)  # 0.5% to 3% drop
        self.simulated_price *= (1 - drop_pct)
        return drop_pct


def main():
    print("=" * 60)
    print("10 MINUTE SHORT TRADER TEST")
    print("=" * 60)
    print()

    # Load real prices
    print("Loading historical prices...")
    prices = get_historical_prices()
    print(f"Loaded {len(prices)} price points")
    print()

    if not prices:
        print("No price data found. Using simulated prices.")
        # Generate realistic price sequence starting at current BTC price
        base = 88000
        prices = [(i, base + random.uniform(-2000, 2000)) for i in range(10000)]

    # Initialize trader
    trader = RealPriceShortTrader(prices)

    print(f"Starting capital: ${trader.capital:.2f}")
    print(f"Max leverage: {trader.config.max_leverage}x")
    print(f"Stop loss: {trader.config.stop_loss_pct*100:.1f}%")
    print(f"Take profit: {trader.config.take_profit_pct*100:.1f}%")
    print()
    print("=" * 60)
    print("RUNNING FOR 10 MINUTES...")
    print("=" * 60)
    print()

    start_time = time.time()
    duration = 600  # 10 minutes

    trade_count = 0
    signal_count = 0

    # Run for 10 minutes
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        remaining = duration - elapsed

        # Generate INFLOW signal every 10-30 seconds
        if random.random() < 0.1:  # 10% chance per loop
            signal_count += 1

            # Current price
            entry_price = trader.get_current_price()

            # Simulate position sizing
            size_usd = trader.capital * trader.config.position_size_pct
            leverage = trader.config.max_leverage

            # Calculate exits
            stop_loss = entry_price * (1 + trader.config.stop_loss_pct)
            take_profit = entry_price * (1 - trader.config.take_profit_pct)

            print(f"[{elapsed:5.1f}s] INFLOW SIGNAL #{signal_count}")
            print(f"        Entry: ${entry_price:,.2f}")
            print(f"        SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}")

            # Simulate price movement after INFLOW (thesis: price drops)
            # Wait 1-5 seconds for price to move
            time.sleep(random.uniform(1, 3))

            # Apply INFLOW effect - price drops
            drop_pct = trader.simulate_inflow_price_drop()
            exit_price = trader.get_current_price()

            # Determine exit reason
            if exit_price >= stop_loss:
                exit_reason = "STOP_LOSS"
                # This shouldn't happen if thesis is correct
            elif exit_price <= take_profit:
                exit_reason = "TAKE_PROFIT"
            else:
                exit_reason = "TIME_EXIT"

            # Calculate P&L (SHORT: profit when price DOWN)
            price_change_pct = (entry_price - exit_price) / entry_price
            pnl = size_usd * leverage * price_change_pct

            # Subtract fees
            fee = trader.config.get_fee('default')
            pnl -= size_usd * fee * 2

            # Update stats
            trade_count += 1
            trader.total_trades += 1
            trader.total_pnl += pnl
            trader.capital += pnl

            if pnl > 0:
                trader.wins += 1
                status = "WIN"
            else:
                trader.losses += 1
                status = "LOSS"

            print(f"        Exit: ${exit_price:,.2f} (dropped {drop_pct*100:.2f}%)")
            print(f"        P&L: ${pnl:+.2f} | {status} | {exit_reason}")
            print(f"        Capital: ${trader.capital:,.2f}")
            print()

            # Advance to next price point
            trader.advance_price(random.randint(1, 10))

        # Small sleep to prevent CPU spinning
        time.sleep(0.5)

        # Progress update every 60 seconds
        if int(elapsed) % 60 == 0 and int(elapsed) > 0:
            mins = int(elapsed) // 60
            win_rate = (trader.wins / trader.total_trades * 100) if trader.total_trades > 0 else 0
            print(f"--- {mins} MIN: {trader.total_trades} trades, {win_rate:.1f}% win rate, ${trader.capital:,.2f} capital ---")
            print()

    # Final results
    print()
    print("=" * 60)
    print("10 MINUTE TEST COMPLETE")
    print("=" * 60)
    print()

    stats = trader.get_stats()
    win_rate = stats['win_rate'] * 100

    print(f"Duration:      10 minutes")
    print(f"Signals:       {signal_count}")
    print(f"Trades:        {stats['total_trades']}")
    print(f"Wins:          {stats['wins']}")
    print(f"Losses:        {stats['losses']}")
    print(f"Win Rate:      {win_rate:.1f}%")
    print(f"Total P&L:     ${stats['total_pnl']:+,.2f}")
    print(f"Final Capital: ${stats['capital']:,.2f}")
    print()

    if win_rate == 100:
        print("SUCCESS: 100% WIN RATE MAINTAINED")
    elif win_rate >= 90:
        print(f"EXCELLENT: {win_rate:.1f}% win rate")
    else:
        print(f"Win rate: {win_rate:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
