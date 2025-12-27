"""
Full Chain Scanner - Renaissance Style.
Scan EVERY block. Detect EVERY pattern. Miss NOTHING.
"""
import time
import json
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .rpc import BitcoinRPC, get_rpc_from_env


@dataclass
class AddressProfile:
    """Complete profile of a Bitcoin address."""
    address: str
    first_seen: int = 0          # Block height
    last_seen: int = 0           # Block height
    tx_count: int = 0            # Total transactions
    receive_count: int = 0       # Times received
    send_count: int = 0          # Times sent
    total_received: float = 0.0  # Total BTC received
    total_sent: float = 0.0      # Total BTC sent
    consolidation_count: int = 0 # Times in 50+ input tx
    is_hot_wallet: bool = False  # Output of consolidation
    active_hours: Set[int] = field(default_factory=set)  # Hours active (0-23)
    tx_timestamps: List[int] = field(default_factory=list)  # For frequency calc


@dataclass
class ConsolidationTx:
    """A consolidation transaction (50+ inputs)."""
    txid: str
    block_height: int
    input_count: int
    output_count: int
    input_addresses: List[str]
    output_addresses: List[str]
    total_btc: float


class FullChainScanner:
    """
    Scan entire blockchain. Build complete address database.

    PATTERNS DETECTED:
    1. Consolidation (50+ inputs) = Exchange
    2. High-frequency addresses = Hot wallets
    3. Common-input-ownership = Same entity
    4. Large balance flows = Exchange/Whale
    """

    CONSOLIDATION_THRESHOLD = 50  # Inputs to qualify as consolidation
    BATCH_SIZE = 100              # Blocks per batch

    def __init__(self, rpc: BitcoinRPC = None):
        self.rpc = rpc or get_rpc_from_env()
        self.addresses: Dict[str, AddressProfile] = {}
        self.consolidations: List[ConsolidationTx] = []
        self.entity_links: List[Tuple[str, str]] = []  # (addr1, addr2) pairs

        # Stats
        self.blocks_scanned = 0
        self.txs_scanned = 0
        self.start_time = 0

    def scan_range(self, start_block: int, end_block: int, progress_interval: int = 100):
        """
        Scan a range of blocks.

        Args:
            start_block: Starting block height
            end_block: Ending block height (inclusive)
            progress_interval: Print progress every N blocks
        """
        self.start_time = time.time()
        print(f"[SCANNER] Scanning blocks {start_block} to {end_block}")

        for height in range(start_block, end_block + 1):
            try:
                self._scan_block(height)
                self.blocks_scanned += 1

                if height % progress_interval == 0:
                    self._print_progress(height, end_block)

            except Exception as e:
                print(f"[ERROR] Block {height}: {e}")
                continue

        self._print_final_stats()

    def scan_recent(self, num_blocks: int = 10000):
        """Scan most recent N blocks."""
        current = self.rpc.getblockcount()
        start = max(0, current - num_blocks)
        self.scan_range(start, current)

    def scan_all(self):
        """Scan entire blockchain from genesis."""
        current = self.rpc.getblockcount()
        self.scan_range(0, current)

    def _scan_block(self, height: int):
        """Scan a single block."""
        block = self.rpc.getblockbyheight(height, verbosity=2)
        block_time = block.get("time", 0)
        hour = (block_time // 3600) % 24

        for tx in block.get("tx", []):
            self._process_transaction(tx, height, hour)
            self.txs_scanned += 1

    def _process_transaction(self, tx: Dict, height: int, hour: int):
        """Process a single transaction."""
        txid = tx.get("txid", "")

        # Extract inputs and outputs
        inputs = self._extract_inputs(tx)
        outputs = self._extract_outputs(tx)

        input_addrs = [i["address"] for i in inputs if i.get("address")]
        output_addrs = [o["address"] for o in outputs if o.get("address")]

        # PATTERN 1: Consolidation detection
        if len(inputs) >= self.CONSOLIDATION_THRESHOLD:
            self._record_consolidation(txid, height, inputs, outputs)

        # PATTERN 2: Update address profiles
        for inp in inputs:
            addr = inp.get("address")
            if addr:
                self._update_address(addr, height, hour, "send", inp.get("btc", 0))

        for out in outputs:
            addr = out.get("address")
            if addr:
                self._update_address(addr, height, hour, "receive", out.get("btc", 0))

        # PATTERN 3: Common-input-ownership (entity linking)
        if len(input_addrs) > 1:
            # All inputs belong to same entity
            first = input_addrs[0]
            for addr in input_addrs[1:]:
                self.entity_links.append((first, addr))

    def _extract_inputs(self, tx: Dict) -> List[Dict]:
        """Extract input addresses and values from transaction."""
        inputs = []
        for vin in tx.get("vin", []):
            # Coinbase transactions have no inputs
            if "coinbase" in vin:
                continue

            # For inputs, we need to look up the previous output
            # This is expensive - in production, use txindex or cache
            prev_txid = vin.get("txid")
            prev_vout = vin.get("vout", 0)

            if prev_txid:
                try:
                    prev_tx = self.rpc.getrawtransaction(prev_txid, True)
                    prev_out = prev_tx.get("vout", [])[prev_vout]
                    script = prev_out.get("scriptPubKey", {})
                    addr = script.get("address") or (script.get("addresses", [None])[0])
                    btc = prev_out.get("value", 0)
                    inputs.append({"address": addr, "btc": btc})
                except:
                    pass

        return inputs

    def _extract_outputs(self, tx: Dict) -> List[Dict]:
        """Extract output addresses and values from transaction."""
        outputs = []
        for vout in tx.get("vout", []):
            script = vout.get("scriptPubKey", {})
            addr = script.get("address") or (script.get("addresses", [None])[0] if script.get("addresses") else None)
            btc = vout.get("value", 0)
            if addr:
                outputs.append({"address": addr, "btc": btc})
        return outputs

    def _update_address(self, addr: str, height: int, hour: int, direction: str, btc: float):
        """Update address profile."""
        if addr not in self.addresses:
            self.addresses[addr] = AddressProfile(address=addr, first_seen=height)

        p = self.addresses[addr]
        p.last_seen = height
        p.tx_count += 1
        p.active_hours.add(hour)

        if direction == "send":
            p.send_count += 1
            p.total_sent += btc
        else:
            p.receive_count += 1
            p.total_received += btc

    def _record_consolidation(self, txid: str, height: int, inputs: List, outputs: List):
        """Record a consolidation transaction."""
        input_addrs = [i["address"] for i in inputs if i.get("address")]
        output_addrs = [o["address"] for o in outputs if o.get("address")]
        total_btc = sum(o.get("btc", 0) for o in outputs)

        self.consolidations.append(ConsolidationTx(
            txid=txid,
            block_height=height,
            input_count=len(inputs),
            output_count=len(outputs),
            input_addresses=input_addrs,
            output_addresses=output_addrs,
            total_btc=total_btc
        ))

        # Mark all inputs as consolidation participants
        for addr in input_addrs:
            if addr in self.addresses:
                self.addresses[addr].consolidation_count += 1

        # Mark outputs as hot wallets
        for addr in output_addrs:
            if addr in self.addresses:
                self.addresses[addr].is_hot_wallet = True
            else:
                self.addresses[addr] = AddressProfile(
                    address=addr,
                    first_seen=height,
                    is_hot_wallet=True
                )

    def _print_progress(self, current: int, total: int):
        """Print scan progress."""
        elapsed = time.time() - self.start_time
        blocks_per_sec = self.blocks_scanned / max(1, elapsed)
        remaining = (total - current) / max(0.1, blocks_per_sec)

        print(f"[{current}/{total}] "
              f"Addresses: {len(self.addresses):,} | "
              f"Consolidations: {len(self.consolidations):,} | "
              f"Speed: {blocks_per_sec:.1f} blk/s | "
              f"ETA: {remaining/60:.1f}min")

    def _print_final_stats(self):
        """Print final statistics."""
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print("SCAN COMPLETE")
        print(f"{'='*60}")
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Blocks: {self.blocks_scanned:,}")
        print(f"Transactions: {self.txs_scanned:,}")
        print(f"Addresses: {len(self.addresses):,}")
        print(f"Consolidations: {len(self.consolidations):,}")
        print(f"Entity Links: {len(self.entity_links):,}")
        print(f"{'='*60}")

    def get_exchange_candidates(self) -> List[str]:
        """Get addresses that look like exchanges."""
        candidates = []
        for addr, p in self.addresses.items():
            # High activity = exchange
            if p.tx_count >= 100 and len(p.active_hours) >= 20:
                candidates.append(addr)
            # Consolidation participant = exchange
            elif p.consolidation_count >= 1:
                candidates.append(addr)
            # Hot wallet = exchange
            elif p.is_hot_wallet:
                candidates.append(addr)
        return candidates

    def export_addresses(self, filepath: str):
        """Export addresses to JSON."""
        data = {
            "addresses": {
                addr: {
                    "first_seen": p.first_seen,
                    "last_seen": p.last_seen,
                    "tx_count": p.tx_count,
                    "receive_count": p.receive_count,
                    "send_count": p.send_count,
                    "total_received": p.total_received,
                    "total_sent": p.total_sent,
                    "consolidation_count": p.consolidation_count,
                    "is_hot_wallet": p.is_hot_wallet,
                    "active_hours": len(p.active_hours)
                }
                for addr, p in self.addresses.items()
            },
            "consolidations": [
                {
                    "txid": c.txid,
                    "block": c.block_height,
                    "inputs": c.input_count,
                    "outputs": c.output_count,
                    "btc": c.total_btc
                }
                for c in self.consolidations
            ],
            "stats": {
                "total_addresses": len(self.addresses),
                "total_consolidations": len(self.consolidations),
                "total_entity_links": len(self.entity_links)
            }
        }

        with open(filepath, "w") as f:
            json.dump(data, f)

        print(f"[EXPORT] Saved to {filepath}")

    def export_consolidation_addresses(self, filepath: str):
        """Export just addresses from consolidations (definitely exchanges)."""
        exchange_addrs = set()

        for c in self.consolidations:
            exchange_addrs.update(c.input_addresses)
            exchange_addrs.update(c.output_addresses)

        with open(filepath, "w") as f:
            json.dump(list(exchange_addrs), f)

        print(f"[EXPORT] {len(exchange_addrs)} consolidation addresses -> {filepath}")
        return exchange_addrs


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Full Chain Scanner")
    parser.add_argument("--start", type=int, default=0, help="Start block")
    parser.add_argument("--end", type=int, default=None, help="End block (default: current)")
    parser.add_argument("--recent", type=int, default=None, help="Scan last N blocks")
    parser.add_argument("--output", type=str, default="/tmp/addresses.json", help="Output file")
    parser.add_argument("--rpc-host", type=str, default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=8332)
    parser.add_argument("--rpc-user", type=str, default="bitcoin")
    parser.add_argument("--rpc-pass", type=str, default="bitcoin")

    args = parser.parse_args()

    rpc = BitcoinRPC(args.rpc_host, args.rpc_port, args.rpc_user, args.rpc_pass)

    print("[SCANNER] Testing RPC connection...")
    if not rpc.test_connection():
        print("[ERROR] Cannot connect to Bitcoin Core RPC")
        print("Ensure bitcoind is running with server=1 in bitcoin.conf")
        return

    scanner = FullChainScanner(rpc)

    if args.recent:
        scanner.scan_recent(args.recent)
    elif args.end:
        scanner.scan_range(args.start, args.end)
    else:
        # Default: scan last 1000 blocks
        scanner.scan_recent(1000)

    scanner.export_addresses(args.output)
    scanner.export_consolidation_addresses(args.output.replace(".json", "_exchanges.json"))


if __name__ == "__main__":
    main()
