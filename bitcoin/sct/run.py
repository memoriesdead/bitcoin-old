#!/usr/bin/env python3
"""
SCT Runner - Statistical Certainty Trading

Main entry point for the SCT system.
Validates strategies and manages certainty checks.

Usage:
    python -m bitcoin.sct.run --check STRATEGY --wins 580 --total 1000
    python -m bitcoin.sct.run --demo
    python -m bitcoin.sct.run --table
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bitcoin.sct.config import SCTConfig, get_config
from bitcoin.sct.wilson import wilson_lower_bound, wilson_interval, trades_needed_for_certainty
from bitcoin.sct.certainty import CertaintyChecker, CertaintyStatus
from bitcoin.sct.validator import StrategyValidator, ValidationDecision
from bitcoin.sct.position_sizer import KellyPositionSizer


def check_strategy(name: str, wins: int, total: int):
    """Check if a strategy meets certainty threshold."""
    validator = StrategyValidator()
    sizer = KellyPositionSizer()

    result = validator.validate(name, wins, total)
    validator.print_result(result)

    if result.decision == ValidationDecision.TRADE:
        size = sizer.size_from_stats(wins, total)
        sizer.print_size(size)


def print_sample_size_table():
    """Print table of sample sizes needed for various win rates."""
    print("\n" + "=" * 70)
    print("SAMPLE SIZE REQUIREMENTS")
    print("For 99% confidence that lower bound >= 50.75%")
    print("=" * 70)

    print(f"\n{'Observed WR':>12} {'Min Trades':>12} {'CI Lower':>12} {'Status':<15}")
    print("-" * 70)

    test_wrs = [0.52, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60, 0.65, 0.70, 0.80, 0.90]

    for wr in test_wrs:
        trades = trades_needed_for_certainty(wr, 0.5075, 0.99)
        if trades > 0:
            wins = int(trades * wr)
            lower = wilson_lower_bound(wins, trades, 0.99)
            status = "POSSIBLE"
        else:
            lower = 0.0
            status = "IMPOSSIBLE"
            trades = -1

        trades_str = str(trades) if trades > 0 else "N/A"
        lower_str = f"{lower*100:.2f}%" if trades > 0 else "N/A"

        print(f"{wr*100:>11.0f}% {trades_str:>12} {lower_str:>12} {status:<15}")

    print("=" * 70)


def demo():
    """Run demo showing CI convergence."""
    print("\n" + "=" * 70)
    print("SCT - STATISTICAL CERTAINTY TRADING")
    print("Demonstrating Wilson CI Convergence")
    print("=" * 70)

    checker = CertaintyChecker()

    # Simulate 58% true win rate over increasing sample sizes
    true_wr = 0.58

    print(f"\nSimulating {true_wr*100:.0f}% win rate strategy...")
    print(f"Target: Lower bound >= 50.75%\n")

    sample_sizes = [25, 50, 100, 150, 200, 300, 500, 750, 1000]

    print(f"{'Trades':>8} {'Wins':>8} {'Obs WR':>10} {'Lower':>10} {'Upper':>10} {'Status':<12}")
    print("-" * 70)

    for n in sample_sizes:
        wins = int(n * true_wr)
        result = checker.check(wins, n)

        status_str = 'CERTAIN' if result.status == CertaintyStatus.CERTAIN else result.status.value.upper()

        print(f"{n:>8} {wins:>8} {result.observed_wr*100:>9.1f}% "
              f"{result.lower_bound*100:>9.2f}% {result.upper_bound*100:>9.2f}% "
              f"{status_str:<12}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("- CI narrows as sample size increases")
    print("- Lower bound rises toward true win rate")
    print("- CERTAIN when lower bound >= 50.75%")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SCT - Statistical Certainty Trading'
    )
    parser.add_argument(
        '--check', type=str, metavar='STRATEGY',
        help='Check if strategy meets certainty threshold'
    )
    parser.add_argument(
        '--wins', type=int, default=0,
        help='Number of winning trades'
    )
    parser.add_argument(
        '--total', type=int, default=0,
        help='Total number of trades'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run demo showing CI convergence'
    )
    parser.add_argument(
        '--table', action='store_true',
        help='Print sample size requirements table'
    )

    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.table:
        print_sample_size_table()
    elif args.check:
        if args.total == 0:
            print("Error: --total required with --check")
            return
        check_strategy(args.check, args.wins, args.total)
    else:
        # Default: show help
        parser.print_help()
        print("\n" + "=" * 50)
        print("Quick Start:")
        print("  python -m bitcoin.sct.run --demo")
        print("  python -m bitcoin.sct.run --table")
        print("  python -m bitcoin.sct.run --check MY_STRAT --wins 580 --total 1000")
        print("=" * 50)


if __name__ == "__main__":
    main()
