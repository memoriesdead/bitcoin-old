#!/usr/bin/env python3
"""
FAST SCANNER - Works WITHOUT txindex.

Only scans OUTPUTS and detects consolidations by input COUNT.
Hot wallets = outputs of 50+ input transactions.

This gives us exchange HOT WALLETS which is what we need for trading.
"""
import json
import os
import sys
import time
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from engine.sovereign.blockchain.rpc import BitcoinRPC
from engine.sovereign.blockchain.exchange_wallets import EXCHANGE_SEEDS


class FastScanner:
    """
    Fast scanner that works WITHOUT txindex.

    Detects exchange hot wallets via consolidation pattern.
    50+ inputs in one tx = outputs are hot wallets.
    """

    CONSOLIDATION_THRESHOLD = 50

    def __init__(self, rpc_host: str = "127.0.0.1", rpc_port: int = 8332,
                 rpc_user: str = "bitcoin", rpc_pass: str = "bitcoin",
                 workers: int = 8, output_path: str = None):

        self.rpc_config = {
            "host": rpc_host,
            "port": rpc_port,
            "user": rpc_user,
            "password": rpc_pass
        }
        self.workers = workers

        if output_path is None:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
            os.makedirs(data_dir, exist_ok=True)
            output_path = os.path.join(data_dir, "exchanges.json")
        self.output_path = output_path

        # Results
        self.hot_wallets: Set[str] = set()
        self.consolidations: List[Dict] = []
        self.high_volume_addresses: Dict[str, int] = defaultdict(int)  # addr -> tx_count

        # Progress
        self.blocks_scanned = 0
        self.txs_scanned = 0
        self.lock = threading.Lock()

        # Load seeds
        self.seed_addresses = set()
        for ex_id, info in EXCHANGE_SEEDS.items():
            self.seed_addresses.update(info.addresses)

    def _get_rpc(self) -> BitcoinRPC:
        return BitcoinRPC(
            self.rpc_config["host"],
            self.rpc_config["port"],
            self.rpc_config["user"],
            self.rpc_config["password"]
        )

    def scan_block_range(self, start: int, end: int) -> Dict:
        """Scan a range of blocks."""
        rpc = self._get_rpc()
        local_hot_wallets = set()
        local_consolidations = []
        local_high_volume = defaultdict(int)
        blocks_done = 0
        txs_done = 0

        for height in range(start, end + 1):
            try:
                block = rpc.getblockbyheight(height, verbosity=2)

                for tx in block.get("tx", []):
                    txid = tx.get("txid", "")
                    vin_count = len(tx.get("vin", []))
                    vouts = tx.get("vout", [])

                    # Extract output addresses
                    output_addrs = []
                    for vout in vouts:
                        script = vout.get("scriptPubKey", {})
                        addr = script.get("address")
                        if not addr and script.get("addresses"):
                            addr = script["addresses"][0]
                        if addr:
                            output_addrs.append(addr)
                            local_high_volume[addr] += 1

                    # CONSOLIDATION DETECTION: 50+ inputs
                    if vin_count >= self.CONSOLIDATION_THRESHOLD:
                        total_btc = sum(v.get("value", 0) for v in vouts)

                        local_consolidations.append({
                            "txid": txid,
                            "block": height,
                            "inputs": vin_count,
                            "outputs": len(output_addrs),
                            "btc": total_btc
                        })

                        # All outputs are HOT WALLETS
                        local_hot_wallets.update(output_addrs)

                    txs_done += 1

                blocks_done += 1

            except Exception as e:
                continue

        return {
            "hot_wallets": local_hot_wallets,
            "consolidations": local_consolidations,
            "high_volume": dict(local_high_volume),
            "blocks": blocks_done,
            "txs": txs_done
        }

    def scan_full(self, start_block: int = 0, end_block: int = None):
        """Full parallel scan."""
        rpc = self._get_rpc()

        if end_block is None:
            end_block = rpc.getblockcount()

        total_blocks = end_block - start_block + 1
        chunk_size = total_blocks // self.workers

        print("=" * 60)
        print("FAST SCANNER (no txindex required)")
        print("=" * 60)
        print(f"Blocks: {start_block:,} -> {end_block:,} ({total_blocks:,} total)")
        print(f"Workers: {self.workers}")
        print(f"Chunk size: {chunk_size:,} blocks/worker")
        print("=" * 60)
        sys.stdout.flush()

        # Create chunks
        chunks = []
        for i in range(self.workers):
            chunk_start = start_block + (i * chunk_size)
            chunk_end = chunk_start + chunk_size - 1
            if i == self.workers - 1:
                chunk_end = end_block
            chunks.append((chunk_start, chunk_end))

        start_time = time.time()
        completed = 0

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self.scan_block_range, s, e): (s, e)
                for s, e in chunks
            }

            for future in as_completed(futures):
                try:
                    result = future.result()

                    with self.lock:
                        self.hot_wallets.update(result["hot_wallets"])
                        self.consolidations.extend(result["consolidations"])
                        for addr, count in result["high_volume"].items():
                            self.high_volume_addresses[addr] += count
                        self.blocks_scanned += result["blocks"]
                        self.txs_scanned += result["txs"]

                        completed += 1
                        elapsed = time.time() - start_time
                        rate = self.blocks_scanned / elapsed if elapsed > 0 else 0
                        eta = (total_blocks - self.blocks_scanned) / rate if rate > 0 else 0

                        print(f"[{completed}/{self.workers}] "
                              f"Blocks: {self.blocks_scanned:,} | "
                              f"Hot Wallets: {len(self.hot_wallets):,} | "
                              f"Consolidations: {len(self.consolidations):,} | "
                              f"Rate: {rate:.0f} blk/s | "
                              f"ETA: {eta/60:.1f}min")
                        sys.stdout.flush()

                except Exception as e:
                    print(f"[ERROR] {e}")

        # Add high-volume addresses (likely exchanges)
        high_volume_threshold = 1000  # 1000+ transactions
        high_volume_exchanges = {
            addr for addr, count in self.high_volume_addresses.items()
            if count >= high_volume_threshold
        }

        print(f"[INFO] Found {len(high_volume_exchanges):,} high-volume addresses (1000+ txs)")

        # Save results
        self.save(high_volume_exchanges)

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("SCAN COMPLETE")
        print("=" * 60)
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Blocks: {self.blocks_scanned:,}")
        print(f"Transactions: {self.txs_scanned:,}")
        print(f"Hot Wallets: {len(self.hot_wallets):,}")
        print(f"High Volume Addresses: {len(high_volume_exchanges):,}")
        print(f"Consolidations: {len(self.consolidations):,}")
        print(f"Output: {self.output_path}")
        print("=" * 60)

    def save(self, high_volume: Set[str] = None):
        """Save to JSON."""
        # Combine all exchange addresses
        all_exchange_addrs = self.hot_wallets.copy()
        if high_volume:
            all_exchange_addrs.update(high_volume)
        all_exchange_addrs.update(self.seed_addresses)

        # Group by known exchanges where possible
        output = defaultdict(list)

        # First, add seeds
        for ex_id, info in EXCHANGE_SEEDS.items():
            output[ex_id] = list(info.addresses)

        # Add discovered addresses
        for addr in all_exchange_addrs:
            if addr not in self.seed_addresses:
                output["discovered_exchange"].append(addr)

        # Sort lists
        for key in output:
            output[key] = sorted(set(output[key]))

        # Add metadata
        output["_metadata"] = {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "blocks_scanned": self.blocks_scanned,
            "hot_wallets": len(self.hot_wallets),
            "consolidations_found": len(self.consolidations),
            "total_addresses": sum(len(v) for k, v in output.items() if k != "_metadata")
        }

        with open(self.output_path, "w") as f:
            json.dump(dict(output), f, indent=2)

        print(f"[SAVE] {self.output_path}")
        print(f"[SAVE] Total addresses: {output['_metadata']['total_addresses']:,}")


def main():
    parser = argparse.ArgumentParser(description="Fast Scanner (no txindex)")

    parser.add_argument("--full", action="store_true", help="Full scan")
    parser.add_argument("--recent", type=int, help="Scan last N blocks")
    parser.add_argument("--workers", type=int, default=8, help="Workers (default: 8)")

    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=8332)
    parser.add_argument("--rpc-user", default="bitcoin")
    parser.add_argument("--rpc-pass", default="bitcoin")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    scanner = FastScanner(
        rpc_host=args.rpc_host,
        rpc_port=args.rpc_port,
        rpc_user=args.rpc_user,
        rpc_pass=args.rpc_pass,
        workers=args.workers,
        output_path=args.output
    )

    if args.full:
        scanner.scan_full()
    elif args.recent:
        rpc = scanner._get_rpc()
        end = rpc.getblockcount()
        start = max(0, end - args.recent)
        scanner.scan_full(start, end)
    else:
        # Default: recent 50k blocks
        rpc = scanner._get_rpc()
        end = rpc.getblockcount()
        start = max(0, end - 50000)
        scanner.scan_full(start, end)


if __name__ == "__main__":
    main()
