#!/usr/bin/env python3
"""
SIMPLE FLOW TRADING - BACK TO BASICS
=====================================

THE EDGE:
  We see blockchain TX → Exchange address
  BEFORE the exchange API shows price change

THE RULES:
  BTC flowing IN to exchange  → They're about to SELL → SHORT
  BTC flowing OUT of exchange → They just BOUGHT    → LONG

NO:
  - Liquidity ratios
  - Correlation databases
  - 170 formula ensembles
  - Probabilistic vs deterministic classifications
  - Complex quant math

YES:
  - See flow → Trade
  - That's it
"""

import subprocess
import json
import time
import ccxt
import sqlite3
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, Dict
import threading


@dataclass
class Position:
    exchange: str
    direction: str
    entry_price: float
    entry_time: datetime
    size_usd: float
    stop_loss: float
    take_profit: float
    flow_btc: float


class SimpleFlowTrader:
    """
    See flow → Trade. That's it.
    """

    # Minimum flow to trade (BTC)
    MIN_FLOW_BTC = 5.0

    # Position settings
    CAPITAL = 100.0
    LEVERAGE = 20
    STOP_LOSS_PCT = 1.0   # 1%
    TAKE_PROFIT_PCT = 0.5  # 0.5% - quick scalp
    MAX_HOLD_SECONDS = 120  # 2 minutes max

    # Exchanges we can trade on
    TRADEABLE = {'coinbase', 'kraken', 'bitstamp', 'gemini', 'binance'}

    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.price_feed = ccxt.kraken()
        self.current_price = 0.0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

        # Start price updater
        self._start_price_feed()

        # Trade log
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect('simple_flow_trades.db')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                exchange TEXT,
                direction TEXT,
                flow_btc REAL,
                entry_price REAL,
                exit_price REAL,
                size_usd REAL,
                pnl REAL,
                pnl_pct REAL,
                exit_reason TEXT,
                hold_seconds REAL
            )
        ''')
        conn.commit()
        conn.close()

    def _start_price_feed(self):
        def update_price():
            while True:
                try:
                    ticker = self.price_feed.fetch_ticker('BTC/USD')
                    self.current_price = ticker['last']
                except:
                    pass
                time.sleep(1)

        t = threading.Thread(target=update_price, daemon=True)
        t.start()

        # Wait for first price
        while self.current_price == 0:
            time.sleep(0.1)

    def on_flow(self, exchange: str, direction: str, flow_btc: float):
        """
        Called when blockchain flow detected.

        direction: 'INFLOW' or 'OUTFLOW'
        """
        exchange = exchange.lower()

        # Skip small flows
        if flow_btc < self.MIN_FLOW_BTC:
            return

        # Skip non-tradeable exchanges
        if exchange not in self.TRADEABLE:
            print(f"[SKIP] {exchange} not tradeable | {flow_btc:.2f} BTC {direction}")
            return

        # Already have position on this exchange?
        if exchange in self.positions:
            return

        # THE SIMPLE RULES:
        if direction == 'INFLOW':
            # BTC going TO exchange = someone about to SELL = price goes DOWN
            self._open_position(exchange, 'SHORT', flow_btc)
        else:
            # BTC leaving exchange = someone BOUGHT and withdrawing = price went UP
            # Only trade outflows if they're large (confident buyer)
            if flow_btc >= self.MIN_FLOW_BTC * 2:
                self._open_position(exchange, 'LONG', flow_btc)

    def _open_position(self, exchange: str, direction: str, flow_btc: float):
        price = self.current_price
        if price == 0:
            return

        size_usd = self.CAPITAL * self.LEVERAGE

        if direction == 'SHORT':
            stop_loss = price * (1 + self.STOP_LOSS_PCT / 100)
            take_profit = price * (1 - self.TAKE_PROFIT_PCT / 100)
        else:
            stop_loss = price * (1 - self.STOP_LOSS_PCT / 100)
            take_profit = price * (1 + self.TAKE_PROFIT_PCT / 100)

        self.positions[exchange] = Position(
            exchange=exchange,
            direction=direction,
            entry_price=price,
            entry_time=datetime.now(timezone.utc),
            size_usd=size_usd,
            stop_loss=stop_loss,
            take_profit=take_profit,
            flow_btc=flow_btc
        )

        print(f"\n[OPEN] {direction} {exchange.upper()} @ ${price:,.2f}")
        print(f"       Flow: {flow_btc:.2f} BTC | Size: ${size_usd:,.0f}")
        print(f"       SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}")

    def check_positions(self):
        """Check all positions for exit conditions."""
        price = self.current_price
        now = datetime.now(timezone.utc)

        to_close = []

        for exchange, pos in self.positions.items():
            hold_seconds = (now - pos.entry_time).total_seconds()

            # Check exit conditions
            reason = None

            if pos.direction == 'SHORT':
                if price >= pos.stop_loss:
                    reason = 'STOP_LOSS'
                elif price <= pos.take_profit:
                    reason = 'TAKE_PROFIT'
            else:  # LONG
                if price <= pos.stop_loss:
                    reason = 'STOP_LOSS'
                elif price >= pos.take_profit:
                    reason = 'TAKE_PROFIT'

            # Time-based exit
            if hold_seconds >= self.MAX_HOLD_SECONDS:
                reason = 'TIMEOUT'

            if reason:
                to_close.append((exchange, reason))

        for exchange, reason in to_close:
            self._close_position(exchange, reason)

    def _close_position(self, exchange: str, reason: str):
        if exchange not in self.positions:
            return

        pos = self.positions[exchange]
        price = self.current_price
        hold_seconds = (datetime.now(timezone.utc) - pos.entry_time).total_seconds()

        # Calculate P&L
        if pos.direction == 'SHORT':
            pnl_pct = (pos.entry_price - price) / pos.entry_price * 100
        else:
            pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

        pnl_usd = pos.size_usd * (pnl_pct / 100)

        # Update stats
        self.total_pnl += pnl_usd
        if pnl_usd > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Log to DB
        conn = sqlite3.connect('simple_flow_trades.db')
        conn.execute('''
            INSERT INTO trades (timestamp, exchange, direction, flow_btc,
                              entry_price, exit_price, size_usd, pnl, pnl_pct,
                              exit_reason, hold_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(timezone.utc).isoformat(),
            exchange, pos.direction, pos.flow_btc,
            pos.entry_price, price, pos.size_usd,
            pnl_usd, pnl_pct, reason, hold_seconds
        ))
        conn.commit()
        conn.close()

        # Print result
        sign = "+" if pnl_usd > 0 else ""
        win_rate = self.wins / (self.wins + self.losses) * 100 if (self.wins + self.losses) > 0 else 0

        print(f"\n[CLOSE] {pos.direction} {exchange.upper()} | {reason}")
        print(f"        ${pos.entry_price:,.2f} -> ${price:,.2f} | {hold_seconds:.1f}s")
        print(f"        P&L: {sign}${pnl_usd:.2f} ({sign}{pnl_pct:.2f}%)")
        print(f"        Total: ${self.total_pnl:.2f} | W/L: {self.wins}/{self.losses} ({win_rate:.0f}%)")

        del self.positions[exchange]


def run_with_cpp_runner():
    """
    Run using the C++ blockchain runner for real-time flows.
    """
    import sys
    sys.path.insert(0, '/root/sovereign/blockchain')

    trader = SimpleFlowTrader()

    print("=" * 60)
    print("SIMPLE FLOW TRADING")
    print("=" * 60)
    print(f"Min flow:     {trader.MIN_FLOW_BTC} BTC")
    print(f"Capital:      ${trader.CAPITAL}")
    print(f"Leverage:     {trader.LEVERAGE}x")
    print(f"Stop Loss:    {trader.STOP_LOSS_PCT}%")
    print(f"Take Profit:  {trader.TAKE_PROFIT_PCT}%")
    print(f"Max Hold:     {trader.MAX_HOLD_SECONDS}s")
    print("=" * 60)
    print("\nTHE RULES:")
    print("  INFLOW  -> SHORT (they're about to sell)")
    print("  OUTFLOW -> LONG  (they just bought)")
    print("\nListening to blockchain...\n")

    # Use the C++ runner
    CPP_BINARY = "/root/sovereign/cpp_runner/build/blockchain_runner"
    ADDRESS_DB = "/root/sovereign/walletexplorer_addresses.db"
    UTXO_DB = "/root/sovereign/exchange_utxos.db"
    ZMQ_URL = "tcp://127.0.0.1:28332"

    cmd = [CPP_BINARY, ADDRESS_DB, UTXO_DB, ZMQ_URL]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    last_check = time.time()

    for line in process.stdout:
        line = line.strip()

        # Parse C++ output format:
        # [91m[SHORT][0m coinbase | In: 1.9162 | Out: 0 | Net: -1.9162 | Latency: 151598ns
        # Strip ANSI codes first
        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)

        if '[SHORT]' in clean_line or '[LONG]' in clean_line:
            # Parse: [SHORT] coinbase | In: 1.9162 | Out: 0 | Net: -1.9162
            try:
                parts = clean_line.split('|')
                if len(parts) >= 3:
                    # First part: "[SHORT] coinbase " or "[LONG] coinbase "
                    first = parts[0].strip()
                    exchange = first.split()[-1].strip()

                    # Parse In: and Out: values
                    inflow = 0.0
                    outflow = 0.0
                    for part in parts[1:]:
                        part = part.strip()
                        if part.startswith('In:'):
                            inflow = float(part.replace('In:', '').strip())
                        elif part.startswith('Out:'):
                            outflow = float(part.replace('Out:', '').strip())

                    print(f"[FLOW] {exchange} | In: {inflow:.2f} | Out: {outflow:.2f}")

                    if inflow > 0:
                        trader.on_flow(exchange, 'INFLOW', inflow)
                    if outflow > 0:
                        trader.on_flow(exchange, 'OUTFLOW', outflow)
            except Exception as e:
                print(f"Parse error: {e} | Line: {clean_line}")

        # Check positions every second
        if time.time() - last_check >= 1:
            trader.check_positions()
            last_check = time.time()


def run_simulation():
    """
    Simulation mode - generates fake flows for testing.
    """
    import random

    trader = SimpleFlowTrader()

    print("=" * 60)
    print("SIMPLE FLOW TRADING - SIMULATION")
    print("=" * 60)
    print(f"Min flow:     {trader.MIN_FLOW_BTC} BTC")
    print(f"Capital:      ${trader.CAPITAL}")
    print(f"Leverage:     {trader.LEVERAGE}x")
    print(f"Stop Loss:    {trader.STOP_LOSS_PCT}%")
    print(f"Take Profit:  {trader.TAKE_PROFIT_PCT}%")
    print(f"Max Hold:     {trader.MAX_HOLD_SECONDS}s")
    print("=" * 60)
    print("\nRunning simulation...\n")

    exchanges = ['coinbase', 'kraken', 'binance', 'bitstamp']

    while True:
        try:
            # Random flow every 5-15 seconds
            time.sleep(random.uniform(5, 15))

            exchange = random.choice(exchanges)
            direction = random.choice(['INFLOW', 'OUTFLOW'])
            flow_btc = random.uniform(1, 50)

            print(f"[FLOW] {exchange} {direction} {flow_btc:.2f} BTC")
            trader.on_flow(exchange, direction, flow_btc)

            # Check positions
            trader.check_positions()

        except KeyboardInterrupt:
            print("\nStopping...")
            break


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--sim':
        run_simulation()
    else:
        run_with_cpp_runner()
