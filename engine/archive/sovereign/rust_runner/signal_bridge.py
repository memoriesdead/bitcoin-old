#!/usr/bin/env python3
"""
SIGNAL BRIDGE - Receives signals from Rust runner, feeds to Paper Trader.

The Rust runner outputs signals to stdout:
  [LONG] coinbase | In: 0.0000 | Out: 15.5000 | Net: +15.5000 | Latency: 850ns
  [SHORT] binance | In: 42.0000 | Out: 0.0000 | Net: -42.0000 | Latency: 1200ns

This bridge:
1. Parses Rust runner output
2. Gets current price from exchange
3. Sends signal to paper trader
"""

import subprocess
import sys
import re
import time
import json
from dataclasses import dataclass
from typing import Optional
import threading

# Import paper trader
sys.path.insert(0, '/root/sovereign/blockchain')

try:
    from paper_trader import PaperTrader, Signal
except ImportError:
    print("Warning: Could not import PaperTrader")
    Signal = None

try:
    from multi_price_feed import MultiExchangePriceFeed
except ImportError:
    print("Warning: Could not import MultiExchangePriceFeed")
    MultiExchangePriceFeed = None


@dataclass
class RustSignal:
    direction: str  # "LONG" or "SHORT"
    exchanges: list
    inflow_btc: float
    outflow_btc: float
    net_flow: float
    latency_ns: int


def parse_rust_signal(line: str) -> Optional[RustSignal]:
    """Parse a signal line from Rust runner output."""
    # Pattern: [LONG] exchange1, exchange2 | In: 0.0000 | Out: 15.5000 | Net: +15.5000 | Latency: 850ns
    pattern = r'\[(LONG|SHORT)\]\x1b\[0m\s+(.+?)\s*\|\s*In:\s*([\d.]+)\s*\|\s*Out:\s*([\d.]+)\s*\|\s*Net:\s*([+-]?[\d.]+)\s*\|\s*Latency:\s*(\d+)ns'

    # Also try without ANSI codes
    if '\x1b' not in line:
        pattern = r'\[(LONG|SHORT)\]\s+(.+?)\s*\|\s*In:\s*([\d.]+)\s*\|\s*Out:\s*([\d.]+)\s*\|\s*Net:\s*([+-]?[\d.]+)\s*\|\s*Latency:\s*(\d+)ns'

    match = re.search(pattern, line)
    if not match:
        return None

    direction = match.group(1)
    exchanges = [e.strip() for e in match.group(2).split(',')]
    inflow = float(match.group(3))
    outflow = float(match.group(4))
    net = float(match.group(5))
    latency = int(match.group(6))

    return RustSignal(
        direction=direction,
        exchanges=exchanges,
        inflow_btc=inflow,
        outflow_btc=outflow,
        net_flow=net,
        latency_ns=latency
    )


class SignalBridge:
    def __init__(self, capital: float = 400.0):
        self.capital = capital

        # Initialize paper trader
        if Signal:
            self.trader = PaperTrader(capital=capital)
        else:
            self.trader = None
            print("Running without paper trader")

        # Initialize price feed
        if MultiExchangePriceFeed:
            self.price_feed = MultiExchangePriceFeed()
            self.price_feed.start()
        else:
            self.price_feed = None
            print("Running without price feed")

        # Stats
        self.signals_received = 0
        self.trades_executed = 0
        self.total_latency_ns = 0

    def process_signal(self, signal: RustSignal):
        """Process a signal from Rust runner."""
        self.signals_received += 1
        self.total_latency_ns += signal.latency_ns

        # Get price for first exchange
        exchange = signal.exchanges[0] if signal.exchanges else 'unknown'
        price = self.get_price(exchange)

        if not price:
            return

        # Create Signal object for paper trader
        if self.trader and Signal:
            trader_signal = Signal(
                exchange=exchange,
                direction=signal.direction,
                amount_btc=abs(signal.net_flow),
                price=price,
                confidence=0.85,  # High confidence from Rust runner
                timestamp=time.time()
            )

            self.trader.on_signal(trader_signal)
            self.trades_executed += 1

        # Print signal info
        avg_latency = self.total_latency_ns / self.signals_received if self.signals_received else 0
        print(f"[BRIDGE] {signal.direction} {exchange} | {abs(signal.net_flow):.2f} BTC | "
              f"Price: ${price:,.2f} | Latency: {signal.latency_ns}ns (avg: {avg_latency:.0f}ns)")

    def get_price(self, exchange: str) -> Optional[float]:
        """Get current price for exchange."""
        if self.price_feed:
            prices = self.price_feed.get_all_prices()
            return prices.get(exchange)

        # Fallback: use a default price
        return 95000.0  # Approximate current BTC price

    def run_rust_runner(self, rust_binary: str = "./target/release/blockchain_runner"):
        """Run Rust binary and process its output."""
        print("=" * 70)
        print("SIGNAL BRIDGE - Rust Runner → Paper Trader")
        print("=" * 70)
        print(f"Rust binary: {rust_binary}")
        print(f"Capital: ${self.capital:.2f}")
        print("=" * 70)
        print()

        # Start Rust process
        process = subprocess.Popen(
            [rust_binary],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        # Process output
        for line in process.stdout:
            line = line.strip()

            # Parse signals
            signal = parse_rust_signal(line)
            if signal:
                self.process_signal(signal)
            else:
                # Print non-signal lines (stats, etc.)
                print(line)

    def run_stdin(self):
        """Read signals from stdin (for piping from Rust runner)."""
        print("=" * 70)
        print("SIGNAL BRIDGE - Reading from stdin")
        print("=" * 70)

        for line in sys.stdin:
            line = line.strip()
            signal = parse_rust_signal(line)
            if signal:
                self.process_signal(signal)
            else:
                print(line)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Signal Bridge: Rust Runner → Paper Trader")
    parser.add_argument("--capital", type=float, default=400.0, help="Starting capital")
    parser.add_argument("--rust-binary", type=str, default="./target/release/blockchain_runner",
                        help="Path to Rust binary")
    parser.add_argument("--stdin", action="store_true", help="Read signals from stdin")

    args = parser.parse_args()

    bridge = SignalBridge(capital=args.capital)

    if args.stdin:
        bridge.run_stdin()
    else:
        bridge.run_rust_runner(args.rust_binary)


if __name__ == "__main__":
    main()
