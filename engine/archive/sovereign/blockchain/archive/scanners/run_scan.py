#!/usr/bin/env python3
"""
Full Blockchain Scanner - Renaissance Style.
Scan entire blockchain to discover ALL exchange addresses.

USAGE:
    # Scan last 10,000 blocks (quick test)
    python run_scan.py --recent 10000

    # Scan specific range
    python run_scan.py --start 800000 --end 870000

    # Full scan from genesis (takes 6-12 hours)
    python run_scan.py --full

    # Resume from last checkpoint
    python run_scan.py --resume
"""
import argparse
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from engine.sovereign.blockchain.rpc import BitcoinRPC, get_rpc_from_env
from engine.sovereign.blockchain.scanner import FullChainScanner
from engine.sovereign.blockchain.classifier import AddressClassifier, EntityLabeler
from engine.sovereign.blockchain.entity import EntityClusterer, ConsolidationTracker
from engine.sovereign.blockchain.database import AddressDatabase
from engine.sovereign.blockchain.exchange_wallets import EXCHANGE_SEEDS


def run_scan(args):
    """Run blockchain scan and persist to database."""
    print("=" * 60)
    print("RENAISSANCE BLOCKCHAIN SCANNER")
    print("=" * 60)

    # Initialize RPC
    rpc = BitcoinRPC(
        host=args.rpc_host,
        port=args.rpc_port,
        user=args.rpc_user,
        password=args.rpc_pass
    )

    print(f"[RPC] Testing connection to {args.rpc_host}:{args.rpc_port}...")
    if not rpc.test_connection():
        print("[ERROR] Cannot connect to Bitcoin Core RPC")
        print("Ensure bitcoind is running with server=1")
        return

    info = rpc.getblockchaininfo()
    current_height = info.get('blocks', 0)
    print(f"[RPC] Connected. Chain: {info.get('chain')} | Height: {current_height:,}")

    # Initialize database
    db = AddressDatabase(args.db_path)
    progress = db.get_scan_progress()
    last_scanned = progress.get('last_block_scanned', 0)
    print(f"[DB] Last scanned block: {last_scanned:,}")

    # Determine scan range
    if args.full:
        start_block = 0
        end_block = current_height
    elif args.resume:
        start_block = last_scanned + 1
        end_block = current_height
        if start_block >= end_block:
            print("[INFO] Already up to date!")
            db.print_stats()
            return
    elif args.recent:
        start_block = max(0, current_height - args.recent)
        end_block = current_height
    else:
        start_block = args.start
        end_block = args.end if args.end else current_height

    print(f"[SCAN] Range: {start_block:,} -> {end_block:,} ({end_block - start_block + 1:,} blocks)")

    # Initialize scanner
    scanner = FullChainScanner(rpc)

    # Run scan
    scan_start = time.time()
    scanner.scan_range(start_block, end_block, progress_interval=args.progress)

    # Process results
    print("\n[PROCESSING] Building entity clusters...")
    clusterer = EntityClusterer()
    cons_tracker = ConsolidationTracker()

    # Process entity links from scanner
    for addr1, addr2 in scanner.entity_links:
        clusterer.add_link(addr1, addr2)

    # Process consolidations
    for cons in scanner.consolidations:
        cons_tracker.process_transaction(
            cons.txid,
            cons.input_addresses,
            cons.output_addresses,
            cons.total_btc
        )
        clusterer.process_consolidation(
            cons.input_addresses,
            cons.output_addresses
        )

    # Classify addresses
    print("[PROCESSING] Classifying addresses...")
    classifier = AddressClassifier(
        seed_addresses={ex_id: info.addresses for ex_id, info in EXCHANGE_SEEDS.items()}
    )

    profiles = []
    for addr, p in scanner.addresses.items():
        profiles.append({
            'address': addr,
            'tx_count': p.tx_count,
            'receive_count': p.receive_count,
            'send_count': p.send_count,
            'total_received': p.total_received,
            'total_sent': p.total_sent,
            'consolidation_count': p.consolidation_count,
            'is_hot_wallet': p.is_hot_wallet,
            'active_hours': len(p.active_hours),
            'first_seen': p.first_seen,
            'last_seen': p.last_seen,
        })

    results = classifier.classify_batch(profiles)

    # Save to database
    print("[PROCESSING] Saving to database...")
    saved = 0
    for profile, result in zip(profiles, results):
        entity_id = clusterer.get_entity_id(profile['address'])

        db.upsert_address(
            profile['address'],
            entity_id=entity_id,
            classification=result.classification.value,
            exchange_id=result.exchange_id,
            confidence=result.confidence,
            first_seen=profile['first_seen'],
            last_seen=profile['last_seen'],
            tx_count=profile['tx_count'],
            receive_count=profile['receive_count'],
            send_count=profile['send_count'],
            total_received=profile['total_received'],
            total_sent=profile['total_sent'],
            consolidation_count=profile['consolidation_count'],
            is_hot_wallet=1 if profile['is_hot_wallet'] else 0,
            active_hours=profile['active_hours']
        )
        saved += 1

        if saved % 10000 == 0:
            print(f"[SAVE] {saved:,} addresses saved...")

    # Save consolidations
    for cons in scanner.consolidations:
        db.add_consolidation(
            cons.txid,
            cons.block_height,
            cons.input_count,
            cons.output_count,
            cons.total_btc,
            cons.input_addresses,
            cons.output_addresses
        )

    # Propagate labels from seeds
    labeler = EntityLabeler({ex_id: info.addresses for ex_id, info in EXCHANGE_SEEDS.items()})
    all_entities = clusterer.get_all_entities()
    labeled = labeler.propagate_labels(all_entities)

    for addr, ex_id in labeled.items():
        db.upsert_address(addr, exchange_id=ex_id)

    # Update scan progress
    db.update_scan_progress(
        end_block,
        total_addresses=len(scanner.addresses),
        total_consolidations=len(scanner.consolidations)
    )

    # Final stats
    elapsed = time.time() - scan_start
    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Blocks: {scanner.blocks_scanned:,}")
    print(f"Transactions: {scanner.txs_scanned:,}")
    print(f"Addresses: {len(scanner.addresses):,}")
    print(f"Consolidations: {len(scanner.consolidations):,}")
    print(f"Entity Links: {len(scanner.entity_links):,}")
    print(f"Labeled Addresses: {len(labeled):,}")
    print("=" * 60)

    db.print_stats()

    # Export files if requested
    if args.export:
        scanner.export_addresses(args.export)
        scanner.export_consolidation_addresses(args.export.replace('.json', '_exchanges.json'))


def main():
    parser = argparse.ArgumentParser(
        description="Renaissance Blockchain Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_scan.py --recent 10000       # Quick test: last 10k blocks
  python run_scan.py --recent 100000      # Medium: last 100k blocks
  python run_scan.py --full               # Full scan (6-12 hours)
  python run_scan.py --resume             # Continue from last checkpoint
        """
    )

    # Scan mode
    parser.add_argument('--full', action='store_true', help='Scan entire blockchain from genesis')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--recent', type=int, help='Scan last N blocks')
    parser.add_argument('--start', type=int, default=0, help='Start block')
    parser.add_argument('--end', type=int, default=None, help='End block')

    # RPC settings
    parser.add_argument('--rpc-host', default='127.0.0.1', help='Bitcoin RPC host')
    parser.add_argument('--rpc-port', type=int, default=8332, help='Bitcoin RPC port')
    parser.add_argument('--rpc-user', default='bitcoin', help='Bitcoin RPC user')
    parser.add_argument('--rpc-pass', default='bitcoin', help='Bitcoin RPC password')

    # Output
    parser.add_argument('--db-path', default=None, help='Database path (default: data/addresses.db)')
    parser.add_argument('--export', type=str, help='Export addresses to JSON file')
    parser.add_argument('--progress', type=int, default=100, help='Progress interval (blocks)')

    args = parser.parse_args()

    # Default to recent 1000 if no mode specified
    if not (args.full or args.resume or args.recent or args.end):
        args.recent = 1000
        print("[INFO] No mode specified, defaulting to --recent 1000")

    run_scan(args)


if __name__ == '__main__':
    main()
