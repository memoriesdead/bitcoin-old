#!/usr/bin/env python3
"""
LIVE SCANNER - Continuously scans new blocks via ZMQ.

Subscribes to new blocks and updates exchanges.json in real-time.
Handles pruned nodes by only scanning available blocks.
"""
import json
import os
import sys
import time
import zmq
import struct
from collections import defaultdict
from typing import Dict, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from engine.sovereign.blockchain.rpc import BitcoinRPC
from engine.sovereign.blockchain.exchange_wallets import EXCHANGE_SEEDS


class LiveScanner:
    """
    Live blockchain scanner with ZMQ subscription.

    Continuously monitors new blocks and updates exchange database.
    Works with pruned nodes - only scans available blocks.
    """

    CONSOLIDATION_THRESHOLD = 50
    HIGH_VOLUME_THRESHOLD = 10  # 10+ txs per scan period

    def __init__(self, rpc_host: str = "127.0.0.1", rpc_port: int = 8332,
                 rpc_user: str = "bitcoin", rpc_pass: str = "bitcoin",
                 zmq_host: str = "127.0.0.1", zmq_port: int = 28335,
                 output_path: str = None):

        self.rpc = BitcoinRPC(rpc_host, rpc_port, rpc_user, rpc_pass)
        self.zmq_endpoint = f"tcp://{zmq_host}:{zmq_port}"

        if output_path is None:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
            os.makedirs(data_dir, exist_ok=True)
            output_path = os.path.join(data_dir, "exchanges.json")
        self.output_path = output_path

        # Load existing data
        self.hot_wallets: Set[str] = set()
        self.high_volume: Dict[str, int] = defaultdict(int)
        self.seed_addresses: Set[str] = set()
        self.last_block = 0
        self.consolidation_count = 0

        self._load_existing()
        self._load_seeds()

    def _load_seeds(self):
        """Load seed addresses."""
        for ex_id, info in EXCHANGE_SEEDS.items():
            self.seed_addresses.update(info.addresses)

    def _load_existing(self):
        """Load existing exchanges.json if present."""
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path) as f:
                    data = json.load(f)

                meta = data.get("_metadata", {})
                self.consolidation_count = meta.get("consolidations_found", 0)

                # Extract block range
                block_range = meta.get("block_range", "0-0")
                if "-" in str(block_range):
                    self.last_block = int(block_range.split("-")[1])

                # Load hot wallets
                if "discovered_hot_wallet" in data:
                    self.hot_wallets.update(data["discovered_hot_wallet"])

                print(f"[LOAD] Existing: {len(self.hot_wallets)} hot wallets, last block {self.last_block}")

            except Exception as e:
                print(f"[LOAD] Error loading existing: {e}")

    def scan_block(self, height: int) -> Dict:
        """Scan a single block for exchange patterns."""
        result = {
            "hot_wallets": set(),
            "high_volume": defaultdict(int),
            "consolidations": 0,
            "txs": 0
        }

        try:
            block = self.rpc.getblockbyheight(height, verbosity=2)

            for tx in block.get("tx", []):
                vin_count = len(tx.get("vin", []))
                vouts = tx.get("vout", [])

                # Extract output addresses
                for vout in vouts:
                    script = vout.get("scriptPubKey", {})
                    addr = script.get("address")
                    if not addr and script.get("addresses"):
                        addr = script["addresses"][0]
                    if addr:
                        result["high_volume"][addr] += 1
                        if vin_count >= self.CONSOLIDATION_THRESHOLD:
                            result["hot_wallets"].add(addr)

                if vin_count >= self.CONSOLIDATION_THRESHOLD:
                    result["consolidations"] += 1

                result["txs"] += 1

        except Exception as e:
            if "pruned" not in str(e).lower():
                print(f"[ERROR] Block {height}: {e}")

        return result

    def initial_scan(self):
        """Scan all available blocks on startup."""
        height = self.rpc.getblockcount()

        # Binary search for oldest available block
        low, high = 0, height
        oldest = height
        while low <= high:
            mid = (low + high) // 2
            try:
                self.rpc.getblockbyheight(mid, verbosity=1)
                oldest = mid
                high = mid - 1
            except:
                low = mid + 1

        # Only scan from last_block if we have data
        start = max(oldest, self.last_block + 1) if self.last_block > oldest else oldest

        print(f"[SCAN] Blocks {start} to {height} ({height - start + 1} blocks)")

        for h in range(start, height + 1):
            result = self.scan_block(h)

            self.hot_wallets.update(result["hot_wallets"])
            for addr, count in result["high_volume"].items():
                self.high_volume[addr] += count
            self.consolidation_count += result["consolidations"]

            if h % 50 == 0:
                print(f"[SCAN] Block {h}: {len(self.hot_wallets)} hot wallets")

        self.last_block = height
        self.save()

        print(f"[SCAN] Complete: {len(self.hot_wallets)} hot wallets, {self.consolidation_count} consolidations")

    def save(self):
        """Save to JSON."""
        output = defaultdict(list)

        # Add seeds
        for ex_id, info in EXCHANGE_SEEDS.items():
            output[ex_id] = list(info.addresses)

        # Add discovered hot wallets
        for addr in self.hot_wallets:
            if addr not in self.seed_addresses:
                output["discovered_hot_wallet"].append(addr)

        # Add high volume addresses
        high_vol = {addr for addr, count in self.high_volume.items()
                    if count >= self.HIGH_VOLUME_THRESHOLD}
        for addr in high_vol:
            if addr not in self.seed_addresses and addr not in self.hot_wallets:
                output["discovered_high_volume"].append(addr)

        # Sort
        for key in output:
            output[key] = sorted(set(output[key]))

        # Metadata
        output["_metadata"] = {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_block": self.last_block,
            "block_range": f"pruned-{self.last_block}",
            "hot_wallets": len(self.hot_wallets),
            "high_volume": len(high_vol),
            "consolidations_found": self.consolidation_count,
            "total_addresses": sum(len(v) for k, v in output.items() if k != "_metadata")
        }

        with open(self.output_path, "w") as f:
            json.dump(dict(output), f, indent=2)

        print(f"[SAVE] {output['_metadata']['total_addresses']} addresses to {self.output_path}")

    def run_zmq(self):
        """Subscribe to new blocks via ZMQ."""
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(self.zmq_endpoint)
        socket.setsockopt_string(zmq.SUBSCRIBE, "hashblock")

        print(f"[ZMQ] Subscribed to {self.zmq_endpoint}")
        print(f"[ZMQ] Waiting for new blocks...")

        while True:
            try:
                msg = socket.recv_multipart()
                topic = msg[0].decode()

                if topic == "hashblock":
                    blockhash = msg[1].hex()
                    height = self.rpc.getblockcount()

                    print(f"[NEW] Block {height}: {blockhash[:16]}...")

                    result = self.scan_block(height)

                    new_hot = len(result["hot_wallets"] - self.hot_wallets)
                    self.hot_wallets.update(result["hot_wallets"])
                    for addr, count in result["high_volume"].items():
                        self.high_volume[addr] += count
                    self.consolidation_count += result["consolidations"]
                    self.last_block = height

                    if result["consolidations"] > 0:
                        print(f"[FOUND] {result['consolidations']} consolidations, {new_hot} new hot wallets")
                        self.save()

            except KeyboardInterrupt:
                print("\n[EXIT] Shutting down...")
                break
            except Exception as e:
                print(f"[ERROR] ZMQ: {e}")
                time.sleep(1)

    def run(self):
        """Run initial scan then subscribe to new blocks."""
        print("=" * 60)
        print("LIVE SCANNER")
        print("=" * 60)

        # Initial scan
        self.initial_scan()

        # ZMQ subscription
        self.run_zmq()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live Blockchain Scanner")

    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=8332)
    parser.add_argument("--rpc-user", default="bitcoin")
    parser.add_argument("--rpc-pass", default="bitcoin")
    parser.add_argument("--zmq-host", default="127.0.0.1")
    parser.add_argument("--zmq-port", type=int, default=28335)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    scanner = LiveScanner(
        rpc_host=args.rpc_host,
        rpc_port=args.rpc_port,
        rpc_user=args.rpc_user,
        rpc_pass=args.rpc_pass,
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        output_path=args.output
    )

    scanner.run()


if __name__ == "__main__":
    main()
