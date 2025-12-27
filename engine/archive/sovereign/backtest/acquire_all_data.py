#!/usr/bin/env python3
"""
RenTech-Style Data Acquisition Pipeline

Complete pipeline to get ALL Bitcoin blockchain data:
1. ORBITAAL dataset (2009-2021) - 156GB from Zenodo
2. Exchange addresses (30M labeled) - From EntityAddressBitcoin
3. Gap fill (2021-2025) - From mempool.space API
4. Price data (2020-2025) - From Binance

Usage:
    # Full acquisition (takes several hours)
    python -m engine.sovereign.backtest.acquire_all_data --full

    # Just download ORBITAAL
    python -m engine.sovereign.backtest.acquire_all_data --orbitaal

    # Just download exchange addresses
    python -m engine.sovereign.backtest.acquire_all_data --exchanges

    # Just fill gap with mempool.space
    python -m engine.sovereign.backtest.acquire_all_data --mempool

    # Check status
    python -m engine.sovereign.backtest.acquire_all_data --status
"""
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def check_status():
    """Check current data status."""
    import sqlite3

    print("\n" + "=" * 70)
    print("DATA ACQUISITION STATUS")
    print("=" * 70)

    data_dir = Path("data")

    # ORBITAAL
    orbitaal_dir = data_dir / "orbitaal"
    orbitaal_files = list(orbitaal_dir.glob("*.tar.gz")) if orbitaal_dir.exists() else []
    print(f"\n[ORBITAAL Dataset]")
    if orbitaal_files:
        total_size = sum(f.stat().st_size for f in orbitaal_files)
        print(f"  Status: DOWNLOADED")
        print(f"  Files: {len(orbitaal_files)}")
        print(f"  Size: {total_size / (1024**3):.1f} GB")
    else:
        print(f"  Status: NOT DOWNLOADED")
        print(f"  Run: python -m engine.sovereign.backtest.acquire_all_data --orbitaal")

    # Exchange addresses
    exchanges_file = data_dir / "exchanges.json"
    print(f"\n[Exchange Addresses]")
    if exchanges_file.exists():
        import json
        with open(exchanges_file) as f:
            data = json.load(f)
        total_addrs = sum(len(v) for v in data.values())
        print(f"  Status: LOADED")
        print(f"  Exchanges: {len(data)}")
        print(f"  Addresses: {total_addrs:,}")
    else:
        print(f"  Status: NOT DOWNLOADED")
        print(f"  Run: python -m engine.sovereign.backtest.acquire_all_data --exchanges")

    # Historical flows database
    db_path = data_dir / "historical_flows.db"
    print(f"\n[Historical Flows Database]")
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Main flows table
        try:
            c.execute('SELECT COUNT(*) FROM flows')
            local_flows = c.fetchone()[0]
            print(f"  Local node flows: {local_flows:,}")
        except:
            local_flows = 0

        # Mempool flows table
        try:
            c.execute('SELECT COUNT(*) FROM mempool_flows')
            mempool_flows = c.fetchone()[0]
            print(f"  Mempool.space flows: {mempool_flows:,}")
        except:
            mempool_flows = 0

        # Prices
        try:
            c.execute('SELECT COUNT(*) FROM prices')
            prices = c.fetchone()[0]
            c.execute('SELECT MIN(timestamp), MAX(timestamp) FROM prices')
            row = c.fetchone()
            if row[0]:
                start = datetime.fromtimestamp(row[0]).strftime('%Y-%m-%d')
                end = datetime.fromtimestamp(row[1]).strftime('%Y-%m-%d')
                print(f"  Price candles: {prices:,} ({start} to {end})")
            else:
                print(f"  Price candles: {prices:,}")
        except:
            prices = 0
            print(f"  Price candles: 0")

        conn.close()

        print(f"\n  Total flows available: {local_flows + mempool_flows:,}")
    else:
        print(f"  Status: NOT INITIALIZED")
        print(f"  Run: python -m engine.sovereign.backtest.acquire_all_data --full")

    print("\n" + "=" * 70)

    # What's needed
    print("\nDATA REQUIREMENTS FOR RENTECH-STYLE TESTING:")
    print("-" * 70)

    requirements = [
        ("ORBITAAL (2009-2021)", bool(orbitaal_files), "156 GB download"),
        ("Exchange addresses", exchanges_file.exists(), "30M+ addresses"),
        ("Price data (Binance)", prices > 0 if 'prices' in dir() else False, "1-min candles"),
        ("2021-2025 gap fill", mempool_flows > 0 if 'mempool_flows' in dir() else False, "mempool.space"),
    ]

    all_ready = True
    for name, ready, note in requirements:
        status = "✓ READY" if ready else "✗ MISSING"
        if not ready:
            all_ready = False
        print(f"  [{status}] {name} ({note})")

    print("-" * 70)

    if all_ready:
        print("\n[+] ALL DATA READY! Run hypothesis testing:")
        print("    python -m engine.sovereign.backtest.run_backtest --test")
    else:
        print("\n[!] Missing data. Run full acquisition:")
        print("    python -m engine.sovereign.backtest.acquire_all_data --full")

    print()


def run_orbitaal_download():
    """Download ORBITAAL dataset."""
    from engine.sovereign.backtest.download_orbitaal import main
    main()


def run_exchanges_download():
    """Download exchange addresses."""
    from engine.sovereign.backtest.download_exchanges import main
    main()


def run_mempool_scan():
    """Scan mempool.space for 2021-2025 data."""
    from engine.sovereign.backtest.mempool_scanner import MempoolScanner
    scanner = MempoolScanner()
    scanner.scan()


def run_price_download():
    """Download price data."""
    from engine.sovereign.backtest.price_downloader import PriceDownloader
    downloader = PriceDownloader()
    downloader.download(start_date="2020-01-01")


def run_full_acquisition():
    """Run complete data acquisition pipeline."""
    print("\n" + "=" * 70)
    print("RENTECH-STYLE COMPLETE DATA ACQUISITION")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Step 1: Exchange addresses (fast, needed for scanning)
    print("\n" + "-" * 70)
    print("[STEP 1/4] Downloading Exchange Addresses")
    print("-" * 70)
    try:
        run_exchanges_download()
    except Exception as e:
        print(f"[!] Error: {e}")

    # Step 2: Price data (needed for correlation)
    print("\n" + "-" * 70)
    print("[STEP 2/4] Downloading Price Data")
    print("-" * 70)
    try:
        run_price_download()
    except Exception as e:
        print(f"[!] Error: {e}")

    # Step 3: ORBITAAL (large download)
    print("\n" + "-" * 70)
    print("[STEP 3/4] Downloading ORBITAAL Dataset (156 GB)")
    print("-" * 70)
    print("[!] This is a large download. You can skip and use mempool.space instead.")
    try:
        run_orbitaal_download()
    except Exception as e:
        print(f"[!] Error: {e}")

    # Step 4: Fill gap with mempool.space
    print("\n" + "-" * 70)
    print("[STEP 4/4] Scanning 2021-2025 via Mempool.space")
    print("-" * 70)
    print("[!] This is slow (~1 block/sec). Can run in background.")
    try:
        run_mempool_scan()
    except KeyboardInterrupt:
        print("[!] Scan interrupted. Progress saved.")
    except Exception as e:
        print(f"[!] Error: {e}")

    print("\n" + "=" * 70)
    print("ACQUISITION COMPLETE")
    print("=" * 70)
    check_status()


def main():
    parser = argparse.ArgumentParser(
        description='RenTech-Style Data Acquisition',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='Run full acquisition pipeline')
    mode.add_argument('--orbitaal', action='store_true', help='Download ORBITAAL dataset')
    mode.add_argument('--exchanges', action='store_true', help='Download exchange addresses')
    mode.add_argument('--mempool', action='store_true', help='Scan mempool.space for 2021-2025')
    mode.add_argument('--prices', action='store_true', help='Download price data')
    mode.add_argument('--status', action='store_true', help='Check current status')

    args = parser.parse_args()

    if args.full:
        run_full_acquisition()
    elif args.orbitaal:
        run_orbitaal_download()
    elif args.exchanges:
        run_exchanges_download()
    elif args.mempool:
        run_mempool_scan()
    elif args.prices:
        run_price_download()
    else:
        check_status()


if __name__ == '__main__':
    main()
