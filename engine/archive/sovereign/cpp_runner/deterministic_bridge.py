#!/usr/bin/env python3
"""
DETERMINISTIC SIGNAL BRIDGE - C++ to CCXT Trading

Captures deterministic signals from C++ runner and executes trades.

SIGNALS:
    SHORT_INTERNAL: Exchange consolidating → about to sell → SHORT
    LONG_EXTERNAL:  Customer withdrawal → already bought → LONG
    INFLOW_SHORT:   Deposit to exchange → about to sell → SHORT

Usage:
    python3 deterministic_bridge.py --paper  # Paper trading mode
    python3 deterministic_bridge.py          # Live trading (careful!)
"""

import subprocess
import sys
import re
import time
import sqlite3
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    print("[WARN] ccxt not available, paper mode only")


@dataclass
class DeterministicSignal:
    """Deterministic signal from C++ runner."""
    timestamp: str
    signal_type: str      # SHORT_INTERNAL, LONG_EXTERNAL, INFLOW_SHORT
    action: str           # SHORT or LONG
    source: str           # Source exchange or 'deposit'
    outflow_btc: float
    internal_btc: float
    external_btc: float
    internal_pct: int
    external_pct: int
    dest_exchanges: str
    reason: str
    txid: str
    latency_ns: int


@dataclass
class Position:
    """Open trading position."""
    entry_time: datetime
    exchange: str
    direction: str        # 'SHORT' or 'LONG'
    entry_price: float
    size_btc: float
    size_usd: float
    signal_type: str
    txid: str


class DeterministicBridge:
    """
    Bridge C++ deterministic signals to CCXT trading.

    100% deterministic signals based on ACTUAL blockchain behavior.
    """

    # Trading config
    POSITION_SIZE_USD = 100.0      # Fixed position size per trade
    MAX_POSITIONS = 8              # Max concurrent positions (increased to catch LONG signals)
    POSITION_TIMEOUT_SEC = 300     # 5 min timeout
    STOP_LOSS_PCT = 0.01           # 1% stop loss
    TAKE_PROFIT_PCT = 0.02         # 2% take profit

    # Signal patterns from C++ output
    SIGNAL_START = re.compile(r'\[(\w+)\]\s*(SHORT|LONG)')
    SOURCE_PATTERN = re.compile(r'Source:\s*(.+)')
    OUTFLOW_PATTERN = re.compile(r'Outflow:\s*([\d.]+)\s*BTC')
    INTERNAL_PATTERN = re.compile(r'Internal:\s*([\d.]+)\s*BTC\s*\((\d+)%\)')
    EXTERNAL_PATTERN = re.compile(r'External:\s*([\d.]+)\s*BTC\s*\((\d+)%\)')
    DEST_PATTERN = re.compile(r'Dest Exch:\s*(.+)')
    REASON_PATTERN = re.compile(r'Reason:\s*(.+)')
    TXID_PATTERN = re.compile(r'TXID:\s*(\w+)')
    LATENCY_PATTERN = re.compile(r'Latency:\s*(\d+)\s*ns')

    def __init__(self, cpp_binary: str, paper: bool = True,
                 db_path: str = "/root/sovereign/deterministic_trades.db"):
        self.cpp_binary = cpp_binary
        self.paper = paper
        self.db_path = db_path

        # State
        self.positions: Dict[str, Position] = {}  # txid -> Position
        self.current_price = 0.0
        self.signal_count = 0
        self.trade_count = 0
        self.win_count = 0
        self.total_pnl = 0.0

        # Parsing state
        self._in_signal = False
        self._current_lines = []

        # Initialize
        self._init_db()
        if HAS_CCXT:
            self._start_price_feed()

    def _init_db(self):
        """Initialize trade database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                signal_type TEXT,
                action TEXT,
                source TEXT,
                outflow_btc REAL,
                entry_price REAL,
                exit_price REAL,
                size_usd REAL,
                pnl REAL,
                pnl_pct REAL,
                result TEXT,
                txid TEXT UNIQUE
            )
        """)
        conn.commit()
        conn.close()

    def _start_price_feed(self):
        """Start background price feed."""
        def update_price():
            feed = ccxt.kraken()
            while True:
                try:
                    ticker = feed.fetch_ticker('BTC/USD')
                    self.current_price = ticker['last']
                except:
                    pass
                time.sleep(1)

        t = threading.Thread(target=update_price, daemon=True)
        t.start()

        # Wait for first price
        for _ in range(100):
            if self.current_price > 0:
                print(f"[PRICE] ${self.current_price:,.2f}")
                break
            time.sleep(0.1)

    def process_line(self, line: str):
        """Process a line from C++ output."""
        # Strip ANSI codes
        clean = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()

        # Detect signal header (e.g., "[INFLOW_SHORT] SHORT")
        if self.SIGNAL_START.search(clean):
            # Found signal type - start collecting
            self._in_signal = True
            self._sep_after_header = True  # Next separator is part of header
            self._current_lines = [clean]
            return

        # If we're in a signal block, collect until we hit the closing separator
        if self._in_signal:
            if '========================================' in clean and len(clean) == 40:
                if self._sep_after_header:
                    # Skip the separator right after header
                    self._sep_after_header = False
                else:
                    # End of signal block - parse it
                    if self._current_lines:
                        signal = self._parse_signal_block()
                        if signal:
                            self._handle_signal(signal)
                    self._in_signal = False
                    self._current_lines = []
            else:
                # Collect detail lines
                self._current_lines.append(clean)

    def _parse_signal_block(self) -> Optional[DeterministicSignal]:
        """Parse a signal block from collected lines."""
        if not self._current_lines:
            return None

        text = '\n'.join(self._current_lines)

        # Extract signal type and action
        match = self.SIGNAL_START.search(text)
        if not match:
            return None

        signal_type = match.group(1)
        action = match.group(2)

        # Extract fields
        source = ""
        outflow = 0.0
        internal_btc = 0.0
        external_btc = 0.0
        internal_pct = 0
        external_pct = 0
        dest_exchanges = ""
        reason = ""
        txid = ""
        latency = 0

        for line in self._current_lines:
            if m := self.SOURCE_PATTERN.search(line):
                source = m.group(1).strip()
            elif m := self.OUTFLOW_PATTERN.search(line):
                outflow = float(m.group(1))
            elif m := self.INTERNAL_PATTERN.search(line):
                internal_btc = float(m.group(1))
                internal_pct = int(m.group(2))
            elif m := self.EXTERNAL_PATTERN.search(line):
                external_btc = float(m.group(1))
                external_pct = int(m.group(2))
            elif m := self.DEST_PATTERN.search(line):
                dest_exchanges = m.group(1).strip()
            elif m := self.REASON_PATTERN.search(line):
                reason = m.group(1).strip()
            elif m := self.TXID_PATTERN.search(line):
                txid = m.group(1).strip()
            elif m := self.LATENCY_PATTERN.search(line):
                latency = int(m.group(1))

        # Debug: print what we parsed
        print(f"[DEBUG] Parsed: signal={signal_type}, action={action}, txid={txid[:16] if txid else 'NONE'}")

        if not signal_type or not txid:
            print(f"[DEBUG] SKIPPED: signal_type={signal_type}, txid={txid}")
            return None

        return DeterministicSignal(
            timestamp=datetime.now(timezone.utc).isoformat(),
            signal_type=signal_type,
            action=action,
            source=source,
            outflow_btc=outflow,
            internal_btc=internal_btc,
            external_btc=external_btc,
            internal_pct=internal_pct,
            external_pct=external_pct,
            dest_exchanges=dest_exchanges,
            reason=reason,
            txid=txid,
            latency_ns=latency
        )

    def _handle_signal(self, signal: DeterministicSignal):
        """Handle a deterministic signal - execute trade."""
        self.signal_count += 1

        # Skip if already have position for this txid
        if signal.txid in self.positions:
            return

        # Skip if max positions reached
        if len(self.positions) >= self.MAX_POSITIONS:
            print(f"[SKIP] Max positions reached ({self.MAX_POSITIONS})")
            return

        # Get current price
        price = self.current_price
        if price <= 0:
            print(f"[SKIP] No price feed")
            return

        # Calculate position size
        size_usd = self.POSITION_SIZE_USD
        size_btc = size_usd / price

        # Create position
        position = Position(
            entry_time=datetime.now(timezone.utc),
            exchange=signal.source,
            direction=signal.action,
            entry_price=price,
            size_btc=size_btc,
            size_usd=size_usd,
            signal_type=signal.signal_type,
            txid=signal.txid
        )

        self.positions[signal.txid] = position
        self.trade_count += 1

        # Print trade
        mode = "PAPER" if self.paper else "LIVE"
        emoji = "🔴" if signal.action == "SHORT" else "🟢"

        print()
        print(f"{emoji} [{mode}] {signal.action} @ ${price:,.2f}")
        print(f"    Signal:  {signal.signal_type}")
        print(f"    Source:  {signal.source}")
        print(f"    Size:    ${size_usd:.2f} ({size_btc:.6f} BTC)")
        print(f"    Reason:  {signal.reason}")
        print(f"    Latency: {signal.latency_ns:,} ns")
        print()

        # TODO: Execute real trade via CCXT in live mode
        # if not self.paper:
        #     exchange = ccxt.bybit(...)
        #     order = exchange.create_market_order('BTC/USDT', ...)

    def _check_positions(self):
        """Check and close positions based on timeout/P&L."""
        if not self.positions:
            return

        price = self.current_price
        if price <= 0:
            return

        now = datetime.now(timezone.utc)
        to_close = []

        for txid, pos in self.positions.items():
            # Calculate P&L
            if pos.direction == "SHORT":
                pnl_pct = (pos.entry_price - price) / pos.entry_price
            else:
                pnl_pct = (price - pos.entry_price) / pos.entry_price

            pnl_usd = pos.size_usd * pnl_pct

            # Check exit conditions
            elapsed = (now - pos.entry_time).total_seconds()
            close_reason = None

            if pnl_pct >= self.TAKE_PROFIT_PCT:
                close_reason = "TP"
            elif pnl_pct <= -self.STOP_LOSS_PCT:
                close_reason = "SL"
            elif elapsed >= self.POSITION_TIMEOUT_SEC:
                close_reason = "TIMEOUT"

            if close_reason:
                to_close.append((txid, pnl_usd, pnl_pct, close_reason, price))

        # Close positions
        for txid, pnl_usd, pnl_pct, reason, exit_price in to_close:
            pos = self.positions.pop(txid)

            self.total_pnl += pnl_usd
            if pnl_usd > 0:
                self.win_count += 1

            emoji = "✅" if pnl_usd > 0 else "❌"
            print()
            print(f"{emoji} CLOSE [{reason}] {pos.direction}")
            print(f"    Entry: ${pos.entry_price:,.2f} → Exit: ${exit_price:,.2f}")
            print(f"    P&L:   ${pnl_usd:+.2f} ({pnl_pct:+.2%})")
            print(f"    Total: ${self.total_pnl:+.2f}")
            print()

            # Save to database
            self._save_trade(pos, exit_price, pnl_usd, pnl_pct, reason)

    def _save_trade(self, pos: Position, exit_price: float,
                    pnl: float, pnl_pct: float, result: str):
        """Save completed trade to database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR IGNORE INTO trades
            (timestamp, signal_type, action, source, outflow_btc,
             entry_price, exit_price, size_usd, pnl, pnl_pct, result, txid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pos.entry_time.isoformat(),
            pos.signal_type,
            pos.direction,
            pos.exchange,
            0.0,  # Would need to store this in Position
            pos.entry_price,
            exit_price,
            pos.size_usd,
            pnl,
            pnl_pct,
            result,
            pos.txid
        ))
        conn.commit()
        conn.close()

    def run(self):
        """Run the deterministic trading bridge."""
        mode = "PAPER" if self.paper else "LIVE"

        print()
        print("=" * 70)
        print(f"DETERMINISTIC TRADING BRIDGE - {mode} MODE")
        print("=" * 70)
        print()
        print("DETERMINISTIC SIGNALS (100% based on blockchain behavior):")
        print("  SHORT_INTERNAL: Consolidation → about to sell → SHORT")
        print("  LONG_EXTERNAL:  Withdrawal → already bought → LONG")
        print("  INFLOW_SHORT:   Deposit → about to sell → SHORT")
        print()
        print(f"Position size: ${self.POSITION_SIZE_USD}")
        print(f"Max positions: {self.MAX_POSITIONS}")
        print(f"Timeout:       {self.POSITION_TIMEOUT_SEC}s")
        print(f"Stop loss:     {self.STOP_LOSS_PCT:.0%}")
        print(f"Take profit:   {self.TAKE_PROFIT_PCT:.0%}")
        print()
        print("=" * 70)
        print()

        # Start C++ process
        print(f"Starting C++ runner: {self.cpp_binary}")
        process = subprocess.Popen(
            [self.cpp_binary],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        last_check = time.time()

        try:
            for line in process.stdout:
                print(line, end='')  # Echo C++ output
                self.process_line(line)

                # Check positions every second
                if time.time() - last_check >= 1.0:
                    self._check_positions()
                    last_check = time.time()

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            process.terminate()

        # Final stats
        self._print_stats()

    def _print_stats(self):
        """Print trading statistics."""
        win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0

        print()
        print("=" * 50)
        print("TRADING STATS")
        print("=" * 50)
        print(f"Signals received:  {self.signal_count}")
        print(f"Trades executed:   {self.trade_count}")
        print(f"Wins:              {self.win_count}")
        print(f"Win rate:          {win_rate:.1f}%")
        print(f"Total P&L:         ${self.total_pnl:+.2f}")
        print(f"Open positions:    {len(self.positions)}")
        print("=" * 50)
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deterministic Trading Bridge")
    parser.add_argument("--binary", default="/root/sovereign/cpp_runner/build/blockchain_runner",
                       help="Path to C++ blockchain runner binary")
    parser.add_argument("--db", default="/root/sovereign/deterministic_trades.db",
                       help="Path to trades database")
    parser.add_argument("--paper", action="store_true", default=True,
                       help="Paper trading mode (default)")
    parser.add_argument("--live", action="store_true",
                       help="Live trading mode (careful!)")
    args = parser.parse_args()

    paper = not args.live

    bridge = DeterministicBridge(
        cpp_binary=args.binary,
        paper=paper,
        db_path=args.db
    )
    bridge.run()
