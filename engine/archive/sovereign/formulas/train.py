#!/usr/bin/env python3
"""
MASTER TRAINING SCRIPT - RenTech Style
======================================

"We don't start with models. We start with data."
"We look for things that can be replicated thousands of times."
- Jim Simons, Renaissance Technologies

This script runs the complete training pipeline:

1. COLLECT: Scan historical blockchain data for exchange flows
2. TRAIN: Train HMM using Baum-Welch algorithm
3. DISCOVER: Find patterns that predict price movements
4. VALIDATE: Prove edge is real (>50.75%) with statistical significance
5. SAVE: Save trained models for live trading

USAGE:
    # Full training pipeline
    python -m engine.sovereign.formulas.train

    # With options
    python -m engine.sovereign.formulas.train --blocks 1000 --min-btc 1.0

    # Validation only (if already trained)
    python -m engine.sovereign.formulas.train --validate-only

REQUIREMENTS:
    - Bitcoin Core running with txindex=1
    - Historical price data (or API access)
    - exchanges.json with known exchange addresses

TARGET:
    - Win rate >= 50.75% (RenTech threshold)
    - Statistical significance (p < 0.05)
    - Edge persists in out-of-sample data
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Add parent directories
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from engine.sovereign.formulas.historical_data import (
    HistoricalFlowDatabase,
    HistoricalFlowScanner,
    FlowEvent,
)
from engine.sovereign.formulas.hmm_trainer import (
    HMMTrainingPipeline,
    BaumWelchTrainer,
    train_hmm_from_database,
)
from engine.sovereign.formulas.pattern_discovery import (
    PatternDiscoveryEngine,
    discover_patterns_from_database,
)
from engine.sovereign.formulas.validation import (
    FullValidationSuite,
    validate_model,
)


def print_banner():
    """Print startup banner."""
    print("""
================================================================================
                    RENAISSANCE PATTERN RECOGNITION TRAINER
================================================================================

"We're right 50.75% of the time, but we're 100% right 50.75% of the time."
- Robert Mercer, Renaissance Technologies

PIPELINE:
  1. COLLECT  - Scan historical blockchain for exchange flows
  2. TRAIN    - Train HMM using Baum-Welch algorithm
  3. DISCOVER - Find patterns that predict price movements
  4. VALIDATE - Prove edge is statistically significant
  5. SAVE     - Save models for live trading

TARGET: Win rate >= 50.75% on out-of-sample data
================================================================================
""")


def collect_historical_data(db: HistoricalFlowDatabase, n_blocks: int = 1000,
                           verbose: bool = True) -> int:
    """
    Phase 1: Collect historical blockchain flow data.

    Scans past blocks for exchange flows and stores in database.
    """
    print("\n" + "="*60)
    print("PHASE 1: COLLECT HISTORICAL DATA")
    print("="*60)

    scanner = HistoricalFlowScanner(db)

    # Get current block height
    try:
        rpc = scanner._get_rpc()
        current_height = rpc.call('getblockcount')
        print(f"[COLLECT] Current block height: {current_height}")
    except Exception as e:
        print(f"[COLLECT] Error connecting to Bitcoin Core: {e}")
        print("[COLLECT] Make sure Bitcoin Core is running with:")
        print("  - txindex=1 (transaction index)")
        print("  - server=1 (RPC server)")
        return 0

    # Scan from (current - n_blocks) to current
    start_height = max(0, current_height - n_blocks)

    print(f"[COLLECT] Scanning blocks {start_height} to {current_height}")
    print(f"[COLLECT] Exchange addresses loaded: {len(scanner.exchange_addresses)}")

    total_events = scanner.scan_range(start_height, current_height)

    print(f"\n[COLLECT] Total flow events: {total_events}")

    return total_events


def collect_from_live_feed(db: HistoricalFlowDatabase, duration: int = 3600,
                          verbose: bool = True) -> int:
    """
    Alternative: Collect from live feed for specified duration.

    This captures real-time flows with accurate timestamps and prices.
    Better for training because we get exact timing.
    """
    print("\n" + "="*60)
    print("PHASE 1b: COLLECT FROM LIVE FEED")
    print(f"Duration: {duration} seconds")
    print("="*60)

    from engine.sovereign.blockchain import PerExchangeBlockchainFeed, ExchangeTick
    import json
    from urllib.request import urlopen, Request

    event_count = 0

    def get_price():
        try:
            url = 'https://api.exchange.coinbase.com/products/BTC-USD/ticker'
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=5) as response:
                return float(json.loads(response.read().decode())['price'])
        except:
            return 0.0

    def on_tick(tick: ExchangeTick):
        nonlocal event_count

        price = get_price()
        if price <= 0:
            return

        event = FlowEvent(
            timestamp=tick.timestamp,
            exchange=tick.exchange,
            direction=tick.direction,
            btc_amount=tick.volume,
            tx_hash=f"{tick.exchange}_{tick.timestamp}_{event_count}",
            block_height=0,
            price_at_flow=price,
        )

        db.add_flow(event)
        db.add_price(tick.timestamp, price)
        event_count += 1

        if event_count % 10 == 0:
            dir_str = "OUT" if tick.direction == 1 else "IN"
            print(f"[LIVE] {event_count}: {tick.exchange} {dir_str} {tick.volume:.2f} BTC @ ${price:,.2f}")

    feed = PerExchangeBlockchainFeed(on_tick=on_tick)
    price = get_price()
    if price > 0:
        feed.set_reference_price(price)

    print("[LIVE] Starting feed...")
    if not feed.start():
        print("[LIVE] Failed to start feed")
        return 0

    print(f"[LIVE] Collecting for {duration} seconds...")
    start = time.time()

    try:
        while time.time() - start < duration:
            time.sleep(1)
            elapsed = int(time.time() - start)
            if elapsed % 60 == 0:
                print(f"[LIVE] {elapsed}s / {duration}s - {event_count} events")
    except KeyboardInterrupt:
        print("\n[LIVE] Interrupted")
    finally:
        feed.stop()

    print(f"\n[LIVE] Collected {event_count} events")
    return event_count


def train_hmm(db: HistoricalFlowDatabase, verbose: bool = True):
    """
    Phase 2: Train HMM using Baum-Welch algorithm.
    """
    print("\n" + "="*60)
    print("PHASE 2: TRAIN HMM")
    print("="*60)

    pipeline = HMMTrainingPipeline(db)

    params = pipeline.train(
        min_flows=100,  # Minimum flows required
        n_restarts=5,   # Try multiple random starts
        validation_split=0.2,
        verbose=verbose,
    )

    if params:
        print(f"\n[TRAIN] HMM trained successfully")
        print(f"[TRAIN] States: {params.n_states}")
        print(f"[TRAIN] Training samples: {params.training_samples}")
        print(f"[TRAIN] Converged: {params.converged}")
        return True
    else:
        print("[TRAIN] HMM training failed")
        return False


def discover_patterns(db: HistoricalFlowDatabase, verbose: bool = True):
    """
    Phase 3: Discover patterns from historical data.
    """
    print("\n" + "="*60)
    print("PHASE 3: DISCOVER PATTERNS")
    print("="*60)

    patterns = discover_patterns_from_database(db.db_path, verbose=verbose)

    if patterns:
        print(f"\n[DISCOVER] Found {len(patterns)} valid patterns")
        return True
    else:
        print("[DISCOVER] No valid patterns found")
        return False


def validate_models(db: HistoricalFlowDatabase, verbose: bool = True) -> bool:
    """
    Phase 4: Validate models with rigorous statistical tests.
    """
    print("\n" + "="*60)
    print("PHASE 4: VALIDATE MODELS")
    print("="*60)

    suite = FullValidationSuite(db)
    results = suite.run_all(verbose=verbose)

    all_passed = all(r.passed for r in results.values())

    # Save validation results
    for name, result in results.items():
        db.set_stat(f'validation_{name}_passed', 1.0 if result.passed else 0.0)
        db.set_stat(f'validation_{name}_value', result.value)
        db.set_stat(f'validation_{name}_pvalue', result.p_value)

    db.set_stat('validation_all_passed', 1.0 if all_passed else 0.0)

    return all_passed


def print_summary(db: HistoricalFlowDatabase):
    """Print training summary."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    stats = db.get_stats_summary()

    print(f"\nDATA:")
    print(f"  Total flows: {stats['total_flows']:,}")
    print(f"  Flows with outcomes: {stats['flows_with_outcomes']:,}")

    if stats['time_range_start'] > 0:
        start = datetime.fromtimestamp(stats['time_range_start'])
        end = datetime.fromtimestamp(stats['time_range_end'])
        print(f"  Time range: {start} to {end}")

    # Win rate
    win_stats = db.get_win_rate(timeframe='30s')
    print(f"\nPERFORMANCE:")
    print(f"  Overall win rate: {win_stats['win_rate']:.4f} ({win_stats['win_rate']*100:.2f}%)")
    print(f"  Edge over random: {win_stats['edge']:.4f} ({win_stats['edge']*100:.2f}%)")

    # HMM
    hmm = db.load_hmm_model('default')
    if hmm:
        print(f"\nHMM MODEL:")
        print(f"  States: {hmm['n_states']}")
        print(f"  Training samples: {hmm['training_samples']}")
        print(f"  Validation accuracy: {hmm['validation_accuracy']:.4f}")

    # Patterns
    patterns = db.get_patterns(min_occurrences=100, min_win_rate=0.5075)
    print(f"\nPATTERNS:")
    print(f"  Valid patterns: {len(patterns)}")

    if patterns:
        print(f"  Top patterns:")
        for p in patterns[:5]:
            dir_str = "LONG" if p['direction'] == 1 else "SHORT"
            print(f"    {p['name']}: {dir_str} | WR={p['win_rate']:.2%} | N={p['occurrences']}")

    # Validation
    val_passed = db.get_stat('validation_all_passed')
    print(f"\nVALIDATION:")
    if val_passed is not None:
        status = "PASSED" if val_passed > 0 else "FAILED"
        print(f"  All tests: {status}")

        for test in ['out_of_sample', 'walk_forward', 'monte_carlo']:
            passed = db.get_stat(f'validation_{test}_passed')
            value = db.get_stat(f'validation_{test}_value')
            if passed is not None:
                status = "PASS" if passed > 0 else "FAIL"
                print(f"  {test}: {status} (value={value:.4f})")
    else:
        print("  Not yet validated")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Train Renaissance Pattern Recognition')
    parser.add_argument('--blocks', type=int, default=1000,
                       help='Number of historical blocks to scan')
    parser.add_argument('--live', type=int, default=0,
                       help='Collect from live feed for N seconds instead of blocks')
    parser.add_argument('--min-btc', type=float, default=0.1,
                       help='Minimum BTC for flow to be recorded')
    parser.add_argument('--db', type=str, default=None,
                       help='Path to database file')
    parser.add_argument('--validate-only', action='store_true',
                       help='Skip training, only validate existing model')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary only')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output')

    args = parser.parse_args()
    verbose = not args.quiet

    print_banner()

    # Initialize database
    db = HistoricalFlowDatabase(args.db)
    print(f"[INIT] Database: {db.db_path}")

    # Summary only
    if args.summary:
        print_summary(db)
        return

    # Validate only
    if args.validate_only:
        passed = validate_models(db, verbose=verbose)
        print_summary(db)
        sys.exit(0 if passed else 1)

    # Full pipeline
    start_time = time.time()

    # Phase 1: Collect data
    if args.live > 0:
        events = collect_from_live_feed(db, duration=args.live, verbose=verbose)
    else:
        events = collect_historical_data(db, n_blocks=args.blocks, verbose=verbose)

    if events == 0:
        print("\n[ERROR] No data collected. Cannot proceed.")
        print("[ERROR] Options:")
        print("  1. Run with --live 3600 to collect from live feed")
        print("  2. Ensure Bitcoin Core is running with txindex=1")
        sys.exit(1)

    # Phase 2: Train HMM
    if not train_hmm(db, verbose=verbose):
        print("\n[WARNING] HMM training failed or weak edge")
        print("[WARNING] Need more data or better patterns")

    # Phase 3: Discover patterns
    discover_patterns(db, verbose=verbose)

    # Phase 4: Validate
    passed = validate_models(db, verbose=verbose)

    # Summary
    print_summary(db)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Total time: {elapsed:.1f} seconds")

    if passed:
        print("\n>>> READY FOR LIVE TRADING <<<")
        print("Run: python -m engine.sovereign.run --capital 100")
    else:
        print("\n>>> MORE DATA NEEDED <<<")
        print("Collect more data and retrain")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
