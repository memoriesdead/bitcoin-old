#!/usr/bin/env python3
"""
SCT Live Runner - Statistical Certainty Trading

Tracks strategy performance, calculates Wilson CI in real-time,
only trades when 99% confident of 50.75%+ win rate.
"""

import asyncio
import time
import sys
import json
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sct_bridge import SCTBridge, CertaintyStatus, CertaintyResult


@dataclass
class StrategyRecord:
    """Strategy performance record."""
    name: str
    wins: int = 0
    losses: int = 0
    last_result: str = ""  # "win" or "loss"
    last_trade_time: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0

    def to_dict(self) -> dict:
        return asdict(self)


class SCTLiveRunner:
    """
    Live SCT Strategy Tracker.

    Monitors strategy performance in real-time,
    uses C++ for nanosecond Wilson CI calculations.
    """

    def __init__(self, data_file: str = "sct_strategies.json",
                 min_wr: float = 0.5075,
                 confidence: float = 0.99):
        """
        Initialize live runner.

        Args:
            data_file: File to persist strategy data
            min_wr: Minimum win rate threshold (50.75%)
            confidence: Confidence level (99%)
        """
        self.data_file = Path(data_file)
        self.bridge = SCTBridge(min_wr=min_wr, confidence=confidence)
        self.strategies: Dict[str, StrategyRecord] = {}
        self.running = False

        # Load existing data
        self.load_data()

    def load_data(self):
        """Load strategy data from file."""
        if self.data_file.exists():
            try:
                with open(self.data_file) as f:
                    data = json.load(f)

                for name, record in data.items():
                    self.strategies[name] = StrategyRecord(**record)

                print(f"[SCT] Loaded {len(self.strategies)} strategies from {self.data_file}")
            except Exception as e:
                print(f"[SCT] Could not load data: {e}")

    def save_data(self):
        """Save strategy data to file."""
        data = {name: record.to_dict() for name, record in self.strategies.items()}

        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def record_trade(self, strategy: str, won: bool):
        """
        Record a trade outcome.

        Args:
            strategy: Strategy name
            won: Whether the trade was profitable
        """
        if strategy not in self.strategies:
            self.strategies[strategy] = StrategyRecord(name=strategy)

        record = self.strategies[strategy]

        if won:
            record.wins += 1
            record.last_result = "win"
        else:
            record.losses += 1
            record.last_result = "loss"

        record.last_trade_time = time.time_ns()

        # Update bridge tracking
        self.bridge.record_trade(strategy, won)

        # Save to file
        self.save_data()

        # Print update
        self._print_strategy_update(strategy)

    def _print_strategy_update(self, strategy: str):
        """Print strategy update after trade."""
        record = self.strategies[strategy]
        result = self.check_strategy(strategy)

        print(f"\n{'='*60}")
        print(f"TRADE RECORDED: {strategy}")
        print(f"{'='*60}")
        print(f"  Result: {'WIN' if record.last_result == 'win' else 'LOSS'}")
        print(f"  Record: {record.wins}W / {record.losses}L ({record.total} total)")
        print(f"  Win Rate: {record.win_rate*100:.2f}%")
        print(f"  Wilson CI: [{result.lower_bound*100:.2f}%, {result.upper_bound*100:.2f}%]")
        print(f"  Status: {result.status.value.upper()}")

        if result.status == CertaintyStatus.CERTAIN:
            size = self.bridge.size_from_stats(record.wins, record.total)
            print(f"\n  TRADEABLE - Position: {size.capital_pct:.2f}% of capital")
        elif result.trades_needed > 0:
            print(f"\n  Need {result.trades_needed} more trades for certainty")

        print(f"{'='*60}")

    def check_strategy(self, strategy: str) -> CertaintyResult:
        """Check certainty for a strategy."""
        if strategy not in self.strategies:
            return CertaintyResult(
                observed_wr=0,
                lower_bound=0,
                upper_bound=0,
                target_wr=self.bridge._min_wr,
                confidence=self.bridge._confidence,
                trades_needed=-1,
                status=CertaintyStatus.NEED_MORE_DATA,
                calc_time_ns=0
            )

        record = self.strategies[strategy]
        return self.bridge.check(record.wins, record.total)

    def get_tradeable(self) -> List[str]:
        """Get strategies we're certain about."""
        tradeable = []

        for name, record in self.strategies.items():
            result = self.bridge.check(record.wins, record.total)
            if result.status == CertaintyStatus.CERTAIN:
                tradeable.append(name)

        return tradeable

    def get_pending(self) -> List[str]:
        """Get strategies needing more data."""
        pending = []

        for name, record in self.strategies.items():
            result = self.bridge.check(record.wins, record.total)
            if result.status == CertaintyStatus.NEED_MORE_DATA:
                pending.append(name)

        return pending

    def print_summary(self):
        """Print summary of all strategies."""
        print("\n" + "=" * 80)
        print("SCT STRATEGY SUMMARY")
        print("=" * 80)

        if not self.strategies:
            print("\n  No strategies tracked yet.")
            print("  Use: runner.record_trade('STRATEGY_NAME', True/False)")
            return

        print(f"\n{'Strategy':<20} {'W/L':>10} {'WR':>8} {'CI Lower':>10} {'Status':<15}")
        print("-" * 80)

        for name, record in sorted(self.strategies.items()):
            result = self.bridge.check(record.wins, record.total)

            wl = f"{record.wins}/{record.losses}"
            wr = f"{record.win_rate*100:.1f}%"
            ci = f"{result.lower_bound*100:.2f}%"
            status = result.status.value.upper()

            if result.status == CertaintyStatus.CERTAIN:
                status = f"TRADEABLE"
            elif result.trades_needed > 0:
                status = f"Need {result.trades_needed}"

            print(f"{name:<20} {wl:>10} {wr:>8} {ci:>10} {status:<15}")

        print("-" * 80)

        tradeable = self.get_tradeable()
        pending = self.get_pending()

        print(f"\nTradeable: {len(tradeable)} | Pending: {len(pending)} | Total: {len(self.strategies)}")

        if tradeable:
            print(f"\nReady to trade:")
            for name in tradeable:
                record = self.strategies[name]
                size = self.bridge.size_from_stats(record.wins, record.total)
                print(f"  - {name}: {size.capital_pct:.2f}% position size")

        print("=" * 80)

    async def interactive_mode(self):
        """Run in interactive mode for recording trades."""
        print("\n" + "=" * 60)
        print("SCT INTERACTIVE MODE")
        print("=" * 60)
        print(f"\nC++ available: {self.bridge.cpp_available}")
        print(f"Min win rate: {self.bridge._min_wr*100:.2f}%")
        print(f"Confidence: {self.bridge._confidence*100:.0f}%")

        self.print_summary()

        print("\nCommands:")
        print("  win STRATEGY   - Record winning trade")
        print("  loss STRATEGY  - Record losing trade")
        print("  check STRATEGY - Check strategy status")
        print("  summary        - Show all strategies")
        print("  exit           - Quit")

        self.running = True

        try:
            while self.running:
                try:
                    cmd = input("\n> ").strip().lower()
                except EOFError:
                    break

                if not cmd:
                    continue

                parts = cmd.split(maxsplit=1)
                action = parts[0]

                if action == "exit":
                    break
                elif action == "summary":
                    self.print_summary()
                elif action == "win" and len(parts) > 1:
                    self.record_trade(parts[1], True)
                elif action == "loss" and len(parts) > 1:
                    self.record_trade(parts[1], False)
                elif action == "check" and len(parts) > 1:
                    name = parts[1]
                    if name in self.strategies:
                        self._print_strategy_update(name)
                    else:
                        print(f"Strategy '{name}' not found")
                else:
                    print("Unknown command. Use: win/loss STRATEGY, check STRATEGY, summary, exit")

        except KeyboardInterrupt:
            pass

        print("\n[SCT] Goodbye.")

    async def run_demo(self):
        """Run demo showing Wilson CI convergence."""
        print("\n" + "=" * 70)
        print("SCT - STATISTICAL CERTAINTY TRADING")
        print("Demonstrating Wilson CI Convergence")
        print("=" * 70)

        # Simulate 58% true win rate over increasing sample sizes
        true_wr = 0.58
        strategy = "DEMO_58PCT"

        print(f"\nSimulating {true_wr*100:.0f}% win rate strategy...")
        print(f"Target: Lower bound >= 50.75%\n")

        sample_sizes = [25, 50, 100, 150, 200, 300, 500, 750, 1000]

        print(f"{'Trades':>8} {'Wins':>8} {'Obs WR':>10} {'Lower':>10} {'Upper':>10} {'Status':<12} {'Time':>10}")
        print("-" * 80)

        for n in sample_sizes:
            wins = int(n * true_wr)

            start = time.time_ns()
            result = self.bridge.check(wins, n)
            calc_ns = time.time_ns() - start

            status_str = 'CERTAIN' if result.status == CertaintyStatus.CERTAIN else result.status.value.upper()

            print(f"{n:>8} {wins:>8} {result.observed_wr*100:>9.1f}% "
                  f"{result.lower_bound*100:>9.2f}% {result.upper_bound*100:>9.2f}% "
                  f"{status_str:<12} {calc_ns:>7} ns")

        print("\n" + "=" * 70)
        print("INTERPRETATION:")
        print("- CI narrows as sample size increases")
        print("- Lower bound rises toward true win rate")
        print("- CERTAIN when lower bound >= 50.75%")
        print("=" * 70)


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='SCT Live Strategy Tracker')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--summary', action='store_true', help='Show strategy summary')
    parser.add_argument('--data', type=str, default='sct_strategies.json', help='Data file')

    args = parser.parse_args()

    runner = SCTLiveRunner(data_file=args.data)

    if args.demo:
        await runner.run_demo()
    elif args.summary:
        runner.print_summary()
    else:
        await runner.interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
