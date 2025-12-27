#!/usr/bin/env python3
"""
COMPLETE UTXO CACHE SEED - 100% COVERAGE
=========================================
Seeds ALL exchange addresses with retry logic.
Ensures every exchange we can trade on is covered.

Run on VPS:
    cd /root/sovereign && python3 blockchain/complete_seed.py
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')

from address_collector import KNOWN_COLD_WALLETS
from exchange_utxo_cache import ExchangeUTXOCache

BITCOIN_CLI = "/usr/local/bin/bitcoin-cli"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 600  # 10 minutes per address


def scan_address_with_retry(address: str, retries: int = MAX_RETRIES) -> dict:
    """Scan UTXO set for a single address with retries."""
    for attempt in range(retries):
        try:
            cmd = [BITCOIN_CLI, "scantxoutset", "start", f'["addr({address})"]']
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS
            )

            if result.returncode != 0:
                print(f"      Attempt {attempt+1} failed: {result.stderr[:100]}")
                time.sleep(5)
                continue

            data = json.loads(result.stdout)
            if data.get("success", False):
                return data
            else:
                print(f"      Attempt {attempt+1}: scan returned success=false")
                time.sleep(5)

        except subprocess.TimeoutExpired:
            print(f"      Attempt {attempt+1} timed out after {TIMEOUT_SECONDS}s")
            # Abort any running scan
            subprocess.run([BITCOIN_CLI, "scantxoutset", "abort"], capture_output=True)
            time.sleep(10)
        except json.JSONDecodeError as e:
            print(f"      Attempt {attempt+1} JSON error: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"      Attempt {attempt+1} error: {e}")
            time.sleep(5)

    return {"success": False, "error": "All retries failed"}


def get_current_coverage(cache: ExchangeUTXOCache) -> dict:
    """Get current UTXO coverage by exchange."""
    import sqlite3
    conn = sqlite3.connect(cache.db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT exchange, COUNT(*), SUM(value_sat)/1e8
        FROM utxos
        GROUP BY exchange
    """)
    coverage = {}
    for row in cursor.fetchall():
        coverage[row[0]] = {"utxos": row[1], "btc": row[2]}
    conn.close()
    return coverage


def main():
    print("=" * 70)
    print("COMPLETE UTXO CACHE SEED - 100% EXCHANGE COVERAGE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize cache
    cache = ExchangeUTXOCache("/root/sovereign/exchange_utxos.db")

    # Get current coverage
    current = get_current_coverage(cache)
    print("CURRENT COVERAGE:")
    for ex, stats in sorted(current.items(), key=lambda x: -x[1]["btc"]):
        print(f"  {ex:<15} {stats['utxos']:>8,} UTXOs  {stats['btc']:>15,.2f} BTC")
    print()

    # Count total addresses
    total_addresses = sum(len(addrs) for addrs in KNOWN_COLD_WALLETS.values())
    print(f"SCANNING {total_addresses} addresses across {len(KNOWN_COLD_WALLETS)} exchanges")
    print()

    # Track results
    results = {}
    failed_addresses = []

    for exchange, addresses in KNOWN_COLD_WALLETS.items():
        print(f"\n{'='*50}")
        print(f"EXCHANGE: {exchange.upper()}")
        print(f"{'='*50}")

        exchange_utxos = 0
        exchange_btc = 0.0

        for i, address in enumerate(addresses):
            print(f"  [{i+1}/{len(addresses)}] {address[:25]}...", end=" ", flush=True)

            result = scan_address_with_retry(address)

            if not result.get("success", False):
                print("FAILED")
                failed_addresses.append((exchange, address))
                continue

            unspents = result.get("unspents", [])
            amount = result.get("total_amount", 0)

            # Add each UTXO to cache
            added = 0
            for utxo in unspents:
                txid = utxo["txid"]
                vout = utxo["vout"]
                value_sat = int(utxo["amount"] * 1e8)
                cache.add_utxo(txid, vout, value_sat, exchange, address)
                added += 1

            exchange_utxos += len(unspents)
            exchange_btc += amount

            print(f"{len(unspents):,} UTXOs, {amount:,.4f} BTC")

        results[exchange] = {
            "utxos": exchange_utxos,
            "btc": exchange_btc,
        }
        print(f"\n  TOTAL {exchange}: {exchange_utxos:,} UTXOs, {exchange_btc:,.4f} BTC")

    # Final summary
    print()
    print("=" * 70)
    print("SEED COMPLETE - FINAL COVERAGE")
    print("=" * 70)
    print()

    final_coverage = get_current_coverage(cache)
    total_utxos = 0
    total_btc = 0.0

    print(f"{'Exchange':<15} {'UTXOs':>12} {'BTC':>20}")
    print("-" * 50)
    for ex, stats in sorted(final_coverage.items(), key=lambda x: -x[1]["btc"]):
        print(f"{ex:<15} {stats['utxos']:>12,} {stats['btc']:>20,.2f}")
        total_utxos += stats['utxos']
        total_btc += stats['btc']
    print("-" * 50)
    print(f"{'TOTAL':<15} {total_utxos:>12,} {total_btc:>20,.2f}")
    print()

    if failed_addresses:
        print(f"FAILED ADDRESSES ({len(failed_addresses)}):")
        for ex, addr in failed_addresses:
            print(f"  {ex}: {addr}")
    else:
        print("ALL ADDRESSES SCANNED SUCCESSFULLY - 100% COVERAGE!")

    print()
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
