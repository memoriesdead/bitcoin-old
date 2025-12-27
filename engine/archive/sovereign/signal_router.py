#!/usr/bin/env python3
"""
SIGNAL ROUTER
=============
Routes C++ pipeline signals to SHORT or LONG traders.

C++ detects flows -> This routes to correct trader -> Trader executes

Simple. Clean. Fast.
"""

import subprocess
import time
import json
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from short.trader import ShortTrader
from long.trader import LongTrader
from shared.config import CONFIG


class SignalType(Enum):
    INFLOW = "INFLOW"    # -> SHORT
    OUTFLOW = "OUTFLOW"  # -> LONG


@dataclass
class Signal:
    """Signal from C++ pipeline."""
    exchange: str
    signal_type: SignalType
    flow_btc: float
    latency_ns: int


class SignalRouter:
    """
    Routes signals from C++ pipeline to appropriate trader.

    INFLOW  -> ShortTrader.on_inflow()
    OUTFLOW -> LongTrader.on_outflow()
    """

    def __init__(self, paper: bool = True):
        self.paper = paper
        self.short_trader = ShortTrader()
        self.long_trader = LongTrader()
        self.cpp_process: Optional[subprocess.Popen] = None
        self.running = False

        # Stats
        self.signals_received = 0
        self.inflows = 0
        self.outflows = 0

    def start(self):
        """Start traders and C++ pipeline."""
        print("=" * 50)
        print("SIGNAL ROUTER")
        print("=" * 50)
        print(f"Mode: {'PAPER' if self.paper else 'LIVE'}")
        print(f"C++ Runner: {CONFIG.cpp_runner}")
        print(f"ZMQ: {CONFIG.zmq_endpoint}")
        print("=" * 50)

        # Start price feeds
        self.short_trader.start()
        self.long_trader.start()

        # Start C++ pipeline
        self._start_cpp()

        self.running = True

    def _start_cpp(self):
        """Start C++ blockchain runner."""
        try:
            self.cpp_process = subprocess.Popen(
                [CONFIG.cpp_runner],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            print(f"[CPP] Started blockchain_runner (PID: {self.cpp_process.pid})")
        except Exception as e:
            print(f"[CPP] Failed to start: {e}")
            print("[CPP] Running in simulation mode")

    def run(self):
        """Main loop - read signals from C++ and route."""
        print("\n[ROUTER] Listening for signals...")

        while self.running:
            try:
                # Check for exits on both traders
                self.short_trader.check_exits()
                self.long_trader.check_exits()

                # Read from C++ pipeline
                if self.cpp_process and self.cpp_process.stdout:
                    line = self.cpp_process.stdout.readline()
                    if line:
                        signal = self._parse_signal(line.strip())
                        if signal:
                            self._route_signal(signal)

                time.sleep(0.001)  # 1ms loop

            except KeyboardInterrupt:
                print("\n[ROUTER] Shutting down...")
                break
            except Exception as e:
                print(f"[ROUTER] Error: {e}")
                time.sleep(1)

        self.stop()

    def _parse_signal(self, line: str) -> Optional[Signal]:
        """Parse signal from C++ output."""
        # Expected format: SIGNAL|exchange|INFLOW/OUTFLOW|btc_amount|latency_ns
        try:
            if not line.startswith("SIGNAL|"):
                return None

            parts = line.split("|")
            if len(parts) < 5:
                return None

            return Signal(
                exchange=parts[1],
                signal_type=SignalType(parts[2]),
                flow_btc=float(parts[3]),
                latency_ns=int(parts[4]),
            )
        except Exception:
            return None

    def _route_signal(self, signal: Signal):
        """Route signal to appropriate trader."""
        self.signals_received += 1

        print(f"\n[SIGNAL] {signal.signal_type.value} | {signal.exchange} | "
              f"{signal.flow_btc:.2f} BTC | {signal.latency_ns/1000:.1f}us")

        if signal.signal_type == SignalType.INFLOW:
            self.inflows += 1
            self.short_trader.on_inflow(signal.exchange, signal.flow_btc)

        elif signal.signal_type == SignalType.OUTFLOW:
            self.outflows += 1
            self.long_trader.on_outflow(signal.exchange, signal.flow_btc)

    def stop(self):
        """Stop everything."""
        self.running = False

        # Stop C++ process
        if self.cpp_process:
            self.cpp_process.terminate()
            print("[CPP] Stopped")

        # Stop traders
        self.short_trader.stop()
        self.long_trader.stop()

        # Print final stats
        self._print_stats()

    def _print_stats(self):
        """Print final statistics."""
        print("\n" + "=" * 50)
        print("FINAL STATISTICS")
        print("=" * 50)

        print(f"\nSignals Received: {self.signals_received}")
        print(f"  INFLOWS:  {self.inflows}")
        print(f"  OUTFLOWS: {self.outflows}")

        print("\n--- SHORT TRADER ---")
        short_stats = self.short_trader.get_stats()
        for k, v in short_stats.items():
            print(f"  {k}: {v}")

        print("\n--- LONG TRADER ---")
        long_stats = self.long_trader.get_stats()
        for k, v in long_stats.items():
            print(f"  {k}: {v}")

        # Combined
        total_pnl = short_stats['total_pnl'] + long_stats['total_pnl']
        total_trades = short_stats['total_trades'] + long_stats['total_trades']
        print(f"\n--- COMBINED ---")
        print(f"  Total Trades: {total_trades}")
        print(f"  Total P&L: ${total_pnl:.2f}")

        print("=" * 50)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Signal Router")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode")
    parser.add_argument("--live", action="store_true", help="Live trading mode")
    args = parser.parse_args()

    paper = not args.live

    router = SignalRouter(paper=paper)
    router.start()
    router.run()


if __name__ == "__main__":
    main()
