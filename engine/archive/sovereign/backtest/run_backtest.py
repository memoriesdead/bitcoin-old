#!/usr/bin/env python3
"""
RenTech-Style Blockchain Backtesting - Main Runner

Complete pipeline:
1. Scan blockchain for historical exchange flows
2. Download price data
3. Test hypotheses
4. Find proven edges

Usage:
    # Full pipeline (on VPS with synced node)
    python -m engine.sovereign.backtest.run_backtest --full

    # Just download prices
    python -m engine.sovereign.backtest.run_backtest --prices

    # Just test hypotheses (if data exists)
    python -m engine.sovereign.backtest.run_backtest --test

    # Scan specific block range
    python -m engine.sovereign.backtest.run_backtest --scan --start 700000 --end 800000
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def run_scanner(args):
    """Run historical blockchain scanner."""
    from engine.sovereign.backtest.historical_scanner import HistoricalBlockchainScanner

    scanner = HistoricalBlockchainScanner(
        rpc_host=args.host,
        rpc_port=args.port,
        rpc_user=args.user,
        rpc_password=args.password
    )

    scanner.scan(
        start_block=args.start,
        end_block=args.end
    )

    stats = scanner.get_flow_stats()
    print("\nFLOW DATABASE STATS:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def run_price_download(args):
    """Download price data from Binance."""
    from engine.sovereign.backtest.price_downloader import PriceDownloader

    downloader = PriceDownloader()
    downloader.download(
        start_date=args.price_start or "2020-01-01",
        end_date=args.price_end
    )

    stats = downloader.get_stats()
    print("\nPRICE DATABASE STATS:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def run_hypothesis_testing(args):
    """Run hypothesis testing on historical data."""
    from engine.sovereign.backtest.hypothesis_tester import HypothesisTester

    tester = HypothesisTester()
    results = tester.run_all_tests()
    tester.save_results()

    # Summary
    implementable = [r for r in results if r.recommendation == "IMPLEMENT"]
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total hypotheses tested: {len(results)}")
    print(f"Implementable strategies: {len(implementable)}")

    if implementable:
        print(f"\nBEST STRATEGY:")
        best = implementable[0]
        print(f"  {best.hypothesis_id}")
        print(f"  Win Rate: {best.win_rate:.2%}")
        print(f"  Sharpe: {best.sharpe_ratio:.2f}")
        print(f"  Total PnL: {best.total_pnl_pct:.2f}%")


def run_full_pipeline(args):
    """Run complete backtesting pipeline."""
    print("\n" + "=" * 80)
    print("RENTECH-STYLE BACKTESTING PIPELINE")
    print("=" * 80)

    # Step 1: Scan blockchain (if node available)
    print("\n[STEP 1/3] Scanning blockchain for exchange flows...")
    try:
        run_scanner(args)
    except Exception as e:
        print(f"[!] Scanner error: {e}")
        print("[!] Make sure Bitcoin Core is running and synced")

    # Step 2: Download prices
    print("\n[STEP 2/3] Downloading price data...")
    try:
        run_price_download(args)
    except Exception as e:
        print(f"[!] Price download error: {e}")

    # Step 3: Test hypotheses
    print("\n[STEP 3/3] Testing hypotheses...")
    try:
        run_hypothesis_testing(args)
    except Exception as e:
        print(f"[!] Testing error: {e}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review data/hypothesis_results.json")
    print("2. Implement strategies with 'IMPLEMENT' recommendation")
    print("3. Run out-of-sample validation on recent data")


def main():
    parser = argparse.ArgumentParser(
        description='RenTech-Style Blockchain Backtesting',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='Run full pipeline')
    mode.add_argument('--scan', action='store_true', help='Only scan blockchain')
    mode.add_argument('--prices', action='store_true', help='Only download prices')
    mode.add_argument('--test', action='store_true', help='Only test hypotheses')

    # Bitcoin RPC settings
    parser.add_argument('--host', default='127.0.0.1', help='Bitcoin RPC host')
    parser.add_argument('--port', type=int, default=8332, help='Bitcoin RPC port')
    parser.add_argument('--user', default='bitcoin', help='RPC username')
    parser.add_argument('--password', default='bitcoin123secure', help='RPC password')

    # Scan settings
    parser.add_argument('--start', type=int, help='Start block for scanning')
    parser.add_argument('--end', type=int, help='End block for scanning')

    # Price settings
    parser.add_argument('--price-start', default='2020-01-01', help='Price data start date')
    parser.add_argument('--price-end', help='Price data end date')

    args = parser.parse_args()

    if args.full:
        run_full_pipeline(args)
    elif args.scan:
        run_scanner(args)
    elif args.prices:
        run_price_download(args)
    elif args.test:
        run_hypothesis_testing(args)
    else:
        # Default: show status
        from engine.sovereign.backtest.hypothesis_tester import HypothesisTester
        import sqlite3

        db_path = Path("data/historical_flows.db")
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            c = conn.cursor()

            c.execute('SELECT COUNT(*) FROM flows')
            flows = c.fetchone()[0]

            c.execute('SELECT COUNT(*) FROM prices')
            prices = c.fetchone()[0]

            conn.close()

            print(f"\nCurrent data status:")
            print(f"  Flows in database: {flows:,}")
            print(f"  Price candles: {prices:,}")

            if flows == 0:
                print(f"\n[!] No flow data. Run: python -m engine.sovereign.backtest.run_backtest --scan")
            if prices == 0:
                print(f"\n[!] No price data. Run: python -m engine.sovereign.backtest.run_backtest --prices")
            if flows > 0 and prices > 0:
                print(f"\n[+] Ready to test! Run: python -m engine.sovereign.backtest.run_backtest --test")
        else:
            print("\nNo database found. Run full pipeline:")
            print("  python -m engine.sovereign.backtest.run_backtest --full")


if __name__ == '__main__':
    main()
