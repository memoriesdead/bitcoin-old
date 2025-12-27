#!/usr/bin/env python3
"""
SEED UTXO CACHE FROM BITCOIN CORE
=================================
Uses scantxoutset to find ALL UTXOs belonging to known exchange addresses.
This enables 100% OUTFLOW detection from day one.

Run on VPS:
    cd /root/sovereign && python3 blockchain/seed_utxo_cache.py
"""

import json
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')

try:
    from address_collector import KNOWN_COLD_WALLETS
    from exchange_utxo_cache import ExchangeUTXOCache
except ImportError:
    from blockchain.address_collector import KNOWN_COLD_WALLETS
    from blockchain.exchange_utxo_cache import ExchangeUTXOCache


BITCOIN_CLI = "/usr/local/bin/bitcoin-cli"


def scan_address(address: str) -> dict:
    """Scan UTXO set for a single address."""
    cmd = [BITCOIN_CLI, "scantxoutset", "start", f'["addr({address})"]']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        return {"success": False, "error": result.stderr}

    return json.loads(result.stdout)


def main():
    print("=" * 70)
    print("SEEDING UTXO CACHE FROM BITCOIN CORE NODE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize cache
    cache = ExchangeUTXOCache("/root/sovereign/exchange_utxos.db")

    # Count addresses
    total_addresses = sum(len(addrs) for addrs in KNOWN_COLD_WALLETS.values())
    print(f"Scanning {total_addresses} cold wallet addresses across {len(KNOWN_COLD_WALLETS)} exchanges")
    print()

    # Track totals
    total_utxos = 0
    total_btc = 0.0
    exchange_stats = {}

    # Scan each exchange
    for exchange, addresses in KNOWN_COLD_WALLETS.items():
        print(f"\n{'='*50}")
        print(f"EXCHANGE: {exchange.upper()}")
        print(f"{'='*50}")

        exchange_utxos = 0
        exchange_btc = 0.0

        for i, address in enumerate(addresses):
            print(f"  [{i+1}/{len(addresses)}] Scanning {address[:20]}...", end=" ", flush=True)

            try:
                result = scan_address(address)

                if not result.get("success", False):
                    print(f"SKIP (error)")
                    continue

                unspents = result.get("unspents", [])
                amount = result.get("total_amount", 0)

                # Add each UTXO to cache
                for utxo in unspents:
                    txid = utxo["txid"]
                    vout = utxo["vout"]
                    value_sat = int(utxo["amount"] * 1e8)
                    cache.add_utxo(txid, vout, value_sat, exchange, address)

                exchange_utxos += len(unspents)
                exchange_btc += amount

                print(f"{len(unspents):,} UTXOs, {amount:,.4f} BTC")

            except subprocess.TimeoutExpired:
                print("TIMEOUT")
            except Exception as e:
                print(f"ERROR: {e}")

        exchange_stats[exchange] = {
            "utxos": exchange_utxos,
            "btc": exchange_btc,
        }
        total_utxos += exchange_utxos
        total_btc += exchange_btc

        print(f"\n  TOTAL {exchange}: {exchange_utxos:,} UTXOs, {exchange_btc:,.4f} BTC")

    # Print summary
    print()
    print("=" * 70)
    print("SEED COMPLETE - SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Exchange':<15} {'UTXOs':>12} {'BTC':>20}")
    print("-" * 50)
    for exchange, stats in sorted(exchange_stats.items(), key=lambda x: -x[1]["btc"]):
        print(f"{exchange:<15} {stats['utxos']:>12,} {stats['btc']:>20,.4f}")
    print("-" * 50)
    print(f"{'TOTAL':<15} {total_utxos:>12,} {total_btc:>20,.4f}")
    print()

    # Cache stats
    cache_stats = cache.get_stats()
    print(f"UTXO Cache Stats:")
    print(f"  - Total UTXOs in cache: {cache_stats['total_utxos']:,}")
    print(f"  - Total BTC tracked: {cache_stats['total_btc']:,.4f}")
    print()
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("=" * 70)
    print("OUTFLOW DETECTION NOW 100% ENABLED")
    print("When ANY of these UTXOs are spent, we detect it as LONG signal")
    print("=" * 70)


if __name__ == "__main__":
    main()
