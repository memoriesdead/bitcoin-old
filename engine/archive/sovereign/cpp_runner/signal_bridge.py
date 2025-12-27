#!/usr/bin/env python3
"""
C++ Signal Bridge - Lightweight Python wrapper for C++ blockchain runner

Reads signals from C++ runner and forwards to trading system.
C++ does the heavy lifting (nanosecond latency), Python handles trading logic.
"""

import subprocess
import sys
import re
import time
import sqlite3
from datetime import datetime
from pathlib import Path

# Signal pattern from C++ output
# [SHORT] coinbase | In: 8.17221 | Out: 0 | Net: -8.17221 | Latency: 6420ns
SIGNAL_PATTERN = re.compile(
    r'\[(SHORT|LONG)\]\s*'
    r'([^|]+)\s*\|\s*'
    r'In:\s*([\d.]+)\s*\|\s*'
    r'Out:\s*([\d.]+)\s*\|\s*'
    r'Net:\s*([+-]?[\d.]+)\s*\|\s*'
    r'Latency:\s*(\d+)ns'
)

class SignalBridge:
    def __init__(self, cpp_binary: str, db_path: str = "/root/sovereign/cpp_signals.db"):
        self.cpp_binary = cpp_binary
        self.db_path = db_path
        self.signal_count = 0
        self.start_time = time.time()
        self._init_db()

    def _init_db(self):
        """Initialize signal database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                direction TEXT,
                exchanges TEXT,
                inflow_btc REAL,
                outflow_btc REAL,
                net_flow REAL,
                latency_ns INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def _save_signal(self, direction: str, exchanges: str, inflow: float,
                     outflow: float, net: float, latency_ns: int):
        """Save signal to database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO signals (timestamp, direction, exchanges, inflow_btc,
                                outflow_btc, net_flow, latency_ns)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), direction, exchanges,
              inflow, outflow, net, latency_ns))
        conn.commit()
        conn.close()

    def process_line(self, line: str):
        """Process a line from C++ output"""
        # Strip ANSI color codes
        clean = re.sub(r'\x1b\[[0-9;]*m', '', line)

        match = SIGNAL_PATTERN.search(clean)
        if match:
            direction = match.group(1)
            exchanges = match.group(2).strip()
            inflow = float(match.group(3))
            outflow = float(match.group(4))
            net = float(match.group(5))
            latency_ns = int(match.group(6))

            self.signal_count += 1
            self._save_signal(direction, exchanges, inflow, outflow, net, latency_ns)

            # Forward signal (can be extended to trading system)
            self.on_signal(direction, exchanges, inflow, outflow, net, latency_ns)

    def on_signal(self, direction: str, exchanges: str, inflow: float,
                  outflow: float, net: float, latency_ns: int):
        """Handle signal - override for trading integration"""
        # Print with timestamp
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}] {direction:5} | {exchanges:20} | Net: {net:+.4f} BTC | {latency_ns}ns")

    def run(self):
        """Run the bridge, reading from C++ process"""
        print("=" * 70)
        print("C++ SIGNAL BRIDGE - Nanosecond Latency")
        print("=" * 70)
        print(f"Binary: {self.cpp_binary}")
        print(f"Database: {self.db_path}")
        print("=" * 70)
        print()

        # Start C++ process
        process = subprocess.Popen(
            [self.cpp_binary],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        try:
            for line in process.stdout:
                print(line, end='')  # Echo C++ output
                self.process_line(line)
        except KeyboardInterrupt:
            print("\nShutting down...")
            process.terminate()

        elapsed = time.time() - self.start_time
        print(f"\nProcessed {self.signal_count} signals in {elapsed:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="/root/sovereign/cpp_runner/build/blockchain_runner")
    parser.add_argument("--db", default="/root/sovereign/cpp_signals.db")
    args = parser.parse_args()

    bridge = SignalBridge(args.binary, args.db)
    bridge.run()
