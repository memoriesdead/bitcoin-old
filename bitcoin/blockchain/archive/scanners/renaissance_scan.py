#!/usr/bin/env python3
"""
RENAISSANCE BLOCKCHAIN SCANNER - MAX SPEED
==========================================
Parallel RPC processing. Maxes out KVM8.
One scan → exchanges.json → O(1) forever.
Live updates via ZMQ block subscriber.

USAGE:
    # Full scan (parallel, ~30-60 min on KVM8)
    python renaissance_scan.py --full --workers 16

    # Then run live updater
    python renaissance_scan.py --live
"""
import json
import os
import sys
import time
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from queue import Queue

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from engine.sovereign.blockchain.rpc import BitcoinRPC
from engine.sovereign.blockchain.exchange_wallets import EXCHANGE_SEEDS


class RenaissanceScanner:
    """
    Max-speed parallel blockchain scanner.

    Identifies ALL exchange addresses via consolidation pattern.
    Outputs single JSON file for O(1) runtime lookup.
    """

    CONSOLIDATION_THRESHOLD = 50  # 50+ inputs = exchange

    def __init__(self, rpc_host: str = "127.0.0.1", rpc_port: int = 8332,
                 rpc_user: str = "bitcoin", rpc_pass: str = "bitcoin",
                 workers: int = 16, output_path: str = None):

        self.rpc_config = {
            "host": rpc_host,
            "port": rpc_port,
            "user": rpc_user,
            "password": rpc_pass
        }
        self.workers = workers

        # Output path
        if output_path is None:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
            os.makedirs(data_dir, exist_ok=True)
            output_path = os.path.join(data_dir, "exchanges.json")
        self.output_path = output_path

        # Results
        self.exchange_addresses: Dict[str, Set[str]] = defaultdict(set)
        self.hot_wallets: Set[str] = set()
        self.consolidations: List[Dict] = []
        self.entity_links: List[Tuple[str, str]] = []

        # Progress tracking
        self.blocks_scanned = 0
        self.txs_scanned = 0
        self.lock = threading.Lock()

        # Load existing seeds
        for ex_id, info in EXCHANGE_SEEDS.items():
            for addr in info.addresses:
                self.exchange_addresses[ex_id].add(addr)

    def _get_rpc(self) -> BitcoinRPC:
        """Create new RPC connection (thread-safe)."""
        return BitcoinRPC(
            self.rpc_config["host"],
            self.rpc_config["port"],
            self.rpc_config["user"],
            self.rpc_config["password"]
        )

    def scan_block_range(self, start: int, end: int) -> Dict:
        """Scan a range of blocks (for one worker)."""
        rpc = self._get_rpc()
        local_exchanges = defaultdict(set)
        local_hot_wallets = set()
        local_consolidations = []
        local_links = []
        blocks_done = 0
        txs_done = 0

        for height in range(start, end + 1):
            try:
                block = rpc.getblockbyheight(height, verbosity=2)

                for tx in block.get("tx", []):
                    txid = tx.get("txid", "")
                    inputs = self._extract_inputs(tx, rpc)
                    outputs = self._extract_outputs(tx)

                    input_addrs = [i["address"] for i in inputs if i.get("address")]
                    output_addrs = [o["address"] for o in outputs if o.get("address")]

                    # CONSOLIDATION = EXCHANGE
                    if len(inputs) >= self.CONSOLIDATION_THRESHOLD:
                        total_btc = sum(o.get("btc", 0) for o in outputs)

                        local_consolidations.append({
                            "txid": txid,
                            "block": height,
                            "inputs": len(inputs),
                            "outputs": len(outputs),
                            "btc": total_btc
                        })

                        # All inputs = exchange deposit addresses
                        for addr in input_addrs:
                            local_exchanges["unknown_exchange"].add(addr)

                        # All outputs = hot wallets
                        for addr in output_addrs:
                            local_hot_wallets.add(addr)
                            local_exchanges["unknown_exchange"].add(addr)

                    # Common-input-ownership = same entity
                    if len(input_addrs) > 1:
                        first = input_addrs[0]
                        for addr in input_addrs[1:]:
                            local_links.append((first, addr))

                    txs_done += 1

                blocks_done += 1

            except Exception as e:
                continue

        return {
            "exchanges": local_exchanges,
            "hot_wallets": local_hot_wallets,
            "consolidations": local_consolidations,
            "links": local_links,
            "blocks": blocks_done,
            "txs": txs_done
        }

    def _extract_inputs(self, tx: Dict, rpc: BitcoinRPC) -> List[Dict]:
        """Extract inputs with address lookup."""
        inputs = []
        for vin in tx.get("vin", []):
            if "coinbase" in vin:
                continue

            prev_txid = vin.get("txid")
            prev_vout = vin.get("vout", 0)

            if prev_txid:
                try:
                    prev_tx = rpc.getrawtransaction(prev_txid, True)
                    prev_out = prev_tx.get("vout", [])[prev_vout]
                    script = prev_out.get("scriptPubKey", {})
                    addr = script.get("address") or (script.get("addresses", [None])[0] if script.get("addresses") else None)
                    btc = prev_out.get("value", 0)
                    if addr:
                        inputs.append({"address": addr, "btc": btc})
                except:
                    pass

        return inputs

    def _extract_outputs(self, tx: Dict) -> List[Dict]:
        """Extract outputs."""
        outputs = []
        for vout in tx.get("vout", []):
            script = vout.get("scriptPubKey", {})
            addr = script.get("address") or (script.get("addresses", [None])[0] if script.get("addresses") else None)
            btc = vout.get("value", 0)
            if addr:
                outputs.append({"address": addr, "btc": btc})
        return outputs

    def scan_full(self, start_block: int = 0, end_block: int = None):
        """
        Full parallel scan of blockchain.

        Splits work across workers, merges results.
        """
        rpc = self._get_rpc()

        if end_block is None:
            end_block = rpc.getblockcount()

        total_blocks = end_block - start_block + 1
        chunk_size = total_blocks // self.workers

        print("=" * 60)
        print("RENAISSANCE SCANNER - MAX SPEED")
        print("=" * 60)
        print(f"Blocks: {start_block:,} -> {end_block:,} ({total_blocks:,} total)")
        print(f"Workers: {self.workers}")
        print(f"Chunk size: {chunk_size:,} blocks/worker")
        print("=" * 60)

        # Create work chunks
        chunks = []
        for i in range(self.workers):
            chunk_start = start_block + (i * chunk_size)
            chunk_end = chunk_start + chunk_size - 1
            if i == self.workers - 1:
                chunk_end = end_block  # Last worker gets remainder
            chunks.append((chunk_start, chunk_end))

        # Progress tracking
        start_time = time.time()
        completed = 0

        # Parallel execution
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self.scan_block_range, s, e): (s, e)
                for s, e in chunks
            }

            for future in as_completed(futures):
                chunk_range = futures[future]
                try:
                    result = future.result()

                    # Merge results
                    with self.lock:
                        for ex_id, addrs in result["exchanges"].items():
                            self.exchange_addresses[ex_id].update(addrs)
                        self.hot_wallets.update(result["hot_wallets"])
                        self.consolidations.extend(result["consolidations"])
                        self.entity_links.extend(result["links"])
                        self.blocks_scanned += result["blocks"]
                        self.txs_scanned += result["txs"]

                        completed += 1
                        elapsed = time.time() - start_time
                        rate = self.blocks_scanned / elapsed if elapsed > 0 else 0
                        eta = (total_blocks - self.blocks_scanned) / rate if rate > 0 else 0

                        total_addrs = sum(len(a) for a in self.exchange_addresses.values())

                        print(f"[{completed}/{self.workers}] "
                              f"Blocks: {self.blocks_scanned:,} | "
                              f"Addresses: {total_addrs:,} | "
                              f"Consolidations: {len(self.consolidations):,} | "
                              f"Rate: {rate:.0f} blk/s | "
                              f"ETA: {eta/60:.1f}min")

                except Exception as e:
                    print(f"[ERROR] Chunk {chunk_range}: {e}")

        # Propagate labels from seeds
        self._propagate_labels()

        # Save results
        self.save()

        # Print final stats
        elapsed = time.time() - start_time
        total_addrs = sum(len(a) for a in self.exchange_addresses.values())

        print("\n" + "=" * 60)
        print("SCAN COMPLETE")
        print("=" * 60)
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Blocks: {self.blocks_scanned:,}")
        print(f"Transactions: {self.txs_scanned:,}")
        print(f"Exchange Addresses: {total_addrs:,}")
        print(f"Hot Wallets: {len(self.hot_wallets):,}")
        print(f"Consolidations: {len(self.consolidations):,}")
        print(f"Entity Links: {len(self.entity_links):,}")
        print(f"Output: {self.output_path}")
        print("=" * 60)

    def _propagate_labels(self):
        """Propagate exchange labels via entity links."""
        # Build union-find from links
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Process all links
        for a, b in self.entity_links:
            union(a, b)

        # Find which entities contain seed addresses
        seed_to_exchange = {}
        for ex_id, info in EXCHANGE_SEEDS.items():
            for addr in info.addresses:
                seed_to_exchange[addr] = ex_id

        # For each known seed, label entire entity
        entity_labels = {}
        for addr, ex_id in seed_to_exchange.items():
            root = find(addr)
            if root not in entity_labels:
                entity_labels[root] = ex_id

        # Relabel addresses based on entity
        relabeled = 0
        unknown_addrs = list(self.exchange_addresses.get("unknown_exchange", set()))

        for addr in unknown_addrs:
            root = find(addr)
            if root in entity_labels:
                ex_id = entity_labels[root]
                self.exchange_addresses["unknown_exchange"].discard(addr)
                self.exchange_addresses[ex_id].add(addr)
                relabeled += 1

        print(f"[LABEL] Propagated labels to {relabeled:,} addresses")

    def save(self):
        """Save to JSON file."""
        # Convert sets to sorted lists for JSON
        output = {
            ex_id: sorted(list(addrs))
            for ex_id, addrs in self.exchange_addresses.items()
            if addrs  # Skip empty
        }

        # Add metadata
        output["_metadata"] = {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "blocks_scanned": self.blocks_scanned,
            "consolidations_found": len(self.consolidations),
            "total_addresses": sum(len(a) for a in self.exchange_addresses.values())
        }

        with open(self.output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[SAVE] {self.output_path}")

    def load(self) -> Dict[str, List[str]]:
        """Load from JSON file."""
        if os.path.exists(self.output_path):
            with open(self.output_path) as f:
                data = json.load(f)
            # Remove metadata for lookup
            data.pop("_metadata", None)
            return data
        return {}


class LiveBlockUpdater:
    """
    Subscribes to new blocks via ZMQ.
    Updates exchanges.json with new consolidations.
    """

    def __init__(self, scanner: RenaissanceScanner, zmq_endpoint: str = "tcp://127.0.0.1:28332"):
        self.scanner = scanner
        self.zmq_endpoint = zmq_endpoint
        self.running = False

    def start(self):
        """Start live block subscription."""
        try:
            import zmq
        except ImportError:
            print("[ERROR] pyzmq not installed. Run: pip install pyzmq")
            return

        print("[LIVE] Starting block subscriber...")
        print(f"[LIVE] ZMQ endpoint: {self.zmq_endpoint}")

        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(self.zmq_endpoint.replace("28332", "28333"))  # hashblock port
        socket.setsockopt_string(zmq.SUBSCRIBE, "hashblock")

        self.running = True
        rpc = self.scanner._get_rpc()

        print("[LIVE] Listening for new blocks...")

        while self.running:
            try:
                msg = socket.recv_multipart()
                topic = msg[0].decode()

                if topic == "hashblock":
                    blockhash = msg[1].hex()

                    # Scan the new block
                    block = rpc.getblock(blockhash, 2)
                    height = block.get("height", 0)

                    new_addrs = 0
                    new_cons = 0

                    for tx in block.get("tx", []):
                        inputs = self.scanner._extract_inputs(tx, rpc)
                        outputs = self.scanner._extract_outputs(tx)

                        # Check for consolidation
                        if len(inputs) >= self.scanner.CONSOLIDATION_THRESHOLD:
                            new_cons += 1

                            for i in inputs:
                                if i.get("address"):
                                    addr = i["address"]
                                    if addr not in self.scanner.exchange_addresses.get("unknown_exchange", set()):
                                        self.scanner.exchange_addresses["unknown_exchange"].add(addr)
                                        new_addrs += 1

                            for o in outputs:
                                if o.get("address"):
                                    addr = o["address"]
                                    self.scanner.hot_wallets.add(addr)
                                    if addr not in self.scanner.exchange_addresses.get("unknown_exchange", set()):
                                        self.scanner.exchange_addresses["unknown_exchange"].add(addr)
                                        new_addrs += 1

                    if new_addrs > 0 or new_cons > 0:
                        self.scanner.save()
                        print(f"[BLOCK {height}] +{new_addrs} addresses, +{new_cons} consolidations")
                    else:
                        print(f"[BLOCK {height}] No new exchange activity")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(1)

        print("[LIVE] Stopped")

    def stop(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Renaissance Blockchain Scanner")

    parser.add_argument("--full", action="store_true", help="Full blockchain scan")
    parser.add_argument("--recent", type=int, help="Scan last N blocks")
    parser.add_argument("--live", action="store_true", help="Run live block updater")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers (default: 16)")

    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=8332)
    parser.add_argument("--rpc-user", default="bitcoin")
    parser.add_argument("--rpc-pass", default="bitcoin")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--zmq", default="tcp://127.0.0.1:28332", help="ZMQ endpoint")

    args = parser.parse_args()

    scanner = RenaissanceScanner(
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
    elif args.live:
        # Load existing data first
        existing = scanner.load()
        for ex_id, addrs in existing.items():
            scanner.exchange_addresses[ex_id].update(addrs)
        print(f"[LOAD] {sum(len(a) for a in scanner.exchange_addresses.values()):,} addresses from {scanner.output_path}")

        # Start live updater
        updater = LiveBlockUpdater(scanner, args.zmq)
        updater.start()
    else:
        # Default: recent 10k blocks
        rpc = scanner._get_rpc()
        end = rpc.getblockcount()
        start = max(0, end - 10000)
        scanner.scan_full(start, end)


if __name__ == "__main__":
    main()
