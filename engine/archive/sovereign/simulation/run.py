#!/usr/bin/env python3
"""
RenTech 1:1 Simulation Engine - CLI Entry Point.

Usage:
    # Historical - full 16 years
    python -m engine.sovereign.simulation.run --mode historical

    # Historical - specific dates
    python -m engine.sovereign.simulation.run --mode historical --start 2020-01-01 --end 2024-12-31

    # Live paper trading for 1 hour
    python -m engine.sovereign.simulation.run --mode live --duration 3600

    # Custom capital and Kelly
    python -m engine.sovereign.simulation.run --capital 50000 --kelly 0.25
"""

import argparse
import sys
from datetime import datetime

from .engine import SimulationEngine, EngineConfig
from .formula_engine import PRODUCTION_FORMULA_IDS


def parse_args():
    parser = argparse.ArgumentParser(
        description="RenTech 1:1 Simulation Trading Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Historical replay (all data):
    python -m engine.sovereign.simulation.run --mode historical

  Historical with date range:
    python -m engine.sovereign.simulation.run --mode historical --start 2020-01-01 --end 2024-12-31

  Live paper trading (1 hour):
    python -m engine.sovereign.simulation.run --mode live --duration 3600

  Custom settings:
    python -m engine.sovereign.simulation.run --mode historical --capital 50000 --kelly 0.5
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['historical', 'live'],
        default='historical',
        help='Simulation mode: historical replay or live paper trading'
    )

    parser.add_argument(
        '--start',
        type=str,
        default=None,
        help='Start date for historical mode (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date for historical mode (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Duration in seconds for live mode (default: 3600 = 1 hour)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='Initial capital in USD (default: 10000)'
    )

    parser.add_argument(
        '--kelly',
        type=float,
        default=0.25,
        help='Kelly fraction (default: 0.25 = Quarter Kelly)'
    )

    parser.add_argument(
        '--max-positions',
        type=int,
        default=5,
        help='Maximum concurrent positions (default: 5)'
    )

    parser.add_argument(
        '--speed',
        type=float,
        default=0,
        help='Replay speed: 0=instant, 1=realtime, N=Nx speed'
    )

    parser.add_argument(
        '--poll-interval',
        type=float,
        default=1.0,
        help='Price poll interval for live mode in seconds (default: 1.0)'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default='data/simulation_trades.db',
        help='Path to simulation trades database'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='data/unified_bitcoin.db',
        help='Path to historical data database'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("RENTECH 1:1 SIMULATION ENGINE")
    print("=" * 70)
    print(f"Mode:           {args.mode.upper()}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Kelly Fraction: {args.kelly} (Quarter Kelly = {args.kelly * 0.25})")
    print(f"Max Positions:  {args.max_positions}")
    print(f"Formulas:       {len(PRODUCTION_FORMULA_IDS)} production formulas")
    print("=" * 70)

    # Create config
    config = EngineConfig(
        initial_capital=args.capital,
        kelly_fraction=args.kelly,
        max_positions=args.max_positions,
        db_path=args.db_path,
    )

    # Create engine
    engine = SimulationEngine(config)

    try:
        if args.mode == 'historical':
            print(f"\nStarting HISTORICAL simulation...")
            if args.start:
                print(f"Start Date: {args.start}")
            if args.end:
                print(f"End Date:   {args.end}")
            print(f"Speed:      {'instant' if args.speed == 0 else f'{args.speed}x'}")
            print()

            results = engine.run_historical(
                start_date=args.start,
                end_date=args.end,
                speed=args.speed,
                db_path=args.data_path,
            )

        else:  # live mode
            print(f"\nStarting LIVE PAPER TRADING...")
            print(f"Duration:       {args.duration}s ({args.duration/3600:.1f} hours)")
            print(f"Poll Interval:  {args.poll_interval}s")
            print()

            results = engine.run_live(
                duration_seconds=args.duration,
                poll_interval=args.poll_interval,
            )

        # Print final summary
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"Initial Capital: ${config.initial_capital:,.2f}")
        print(f"Final Capital:   ${results['final_capital']:,.2f}")
        print(f"Total PnL:       ${results['total_pnl_usd']:+,.2f} ({results['total_pnl_pct']:+.2f}%)")
        print(f"Total Trades:    {results['total_trades']}")
        print(f"Win Rate:        {results['win_rate']*100:.1f}%" if results['total_trades'] > 0 else "Win Rate: N/A")
        print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:    {results['max_drawdown_pct']:.2f}%")
        print("=" * 70)

        return 0

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Make sure the data file exists at the specified path.")
        return 1

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
