#!/usr/bin/env python3
"""
REAL-TIME ADDRESS CLUSTERING
=============================
Captures addresses from detected flows in real-time.

When we detect an OUTFLOW (withdrawal):
- Input addresses belong to the exchange
- Change outputs (smaller values) belong to the exchange
- We add these new addresses to our database

When we detect an INFLOW (deposit):
- The output receiving the deposit is the exchange
- Other outputs might be change from user (not exchange)

This runs continuously, growing our address database with every flow.
"""

import sys
import time
import json
import sqlite3
import zmq
from datetime import datetime
from typing import Dict, Set, Optional, List, Tuple
from collections import defaultdict

sys.path.insert(0, '/root/sovereign')


class RealtimeCluster:
    """
    Real-time address clustering from ZMQ transaction feed.
    Captures addresses as flows are detected.
    """

    def __init__(self, db_path: str = "/root/sovereign/address_clusters.db"):
        print("=" * 70)
        print("REAL-TIME ADDRESS CLUSTERING")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        print()

        self.db_path = db_path
        self._init_db()

        # Load known addresses
        self.addresses: Set[str] = set()
        self.addr_to_exchange: Dict[str, str] = {}
        self._load_existing()

        # Stats
        self.flows_seen = 0
        self.addresses_added = 0
        self.start_time = time.time()

        # Exchange patterns for detection
        self.exchange_patterns = self._load_exchange_patterns()

        print(f"Starting addresses: {len(self.addresses):,}")
        print()

    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS addresses (
                address TEXT PRIMARY KEY,
                exchange TEXT NOT NULL,
                discovered_at TEXT,
                source TEXT DEFAULT 'scan'
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_ex ON addresses(exchange)")

        # Track real-time discoveries
        c.execute("""
            CREATE TABLE IF NOT EXISTS realtime_discoveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                txid TEXT,
                exchange TEXT,
                direction TEXT,
                btc_amount REAL,
                addresses_found INTEGER,
                timestamp TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _load_existing(self):
        """Load all known addresses."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT address, exchange FROM addresses")
        for row in c.fetchall():
            self.addresses.add(row[0])
            self.addr_to_exchange[row[0]] = row[1]
        conn.close()

    def _load_exchange_patterns(self) -> Dict[str, List[str]]:
        """Load address prefixes/patterns for each exchange."""
        # Common address prefixes for major exchanges
        return {
            'binance': ['34xp4', '3LYJf', 'bc1qm34', 'bc1qgdj'],
            'coinbase': ['3Kzh9', '395xM', 'bc1q7y', '3LCGs'],
            'okx': ['3LvppK', 'bc1qjas', '3FHNBLo'],
            'bybit': ['bc1q8wtz', '32yvQ'],
            'bitfinex': ['bc1qgdj', '3D2oet', '385cR5'],
            'kraken': ['bc1qcup', '3FWBZ'],
        }

    def _save_address(self, addr: str, exchange: str, source: str) -> bool:
        """Save new address to database."""
        if addr in self.addresses:
            return False

        self.addresses.add(addr)
        self.addr_to_exchange[addr] = exchange

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT OR IGNORE INTO addresses (address, exchange, discovered_at, source)
            VALUES (?, ?, ?, ?)
        """, (addr, exchange, datetime.now().isoformat(), source))
        conn.commit()
        conn.close()

        self.addresses_added += 1
        return True

    def _save_batch(self, entries: List[Tuple[str, str, str]]):
        """Batch save addresses."""
        if not entries:
            return

        new_entries = []
        for addr, exchange, source in entries:
            if addr not in self.addresses:
                self.addresses.add(addr)
                self.addr_to_exchange[addr] = exchange
                new_entries.append((addr, exchange, datetime.now().isoformat(), source))
                self.addresses_added += 1

        if new_entries:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.executemany("""
                INSERT OR IGNORE INTO addresses (address, exchange, discovered_at, source)
                VALUES (?, ?, ?, ?)
            """, new_entries)
            conn.commit()
            conn.close()

    def process_transaction(self, tx: Dict, detected_exchange: str = None,
                           direction: str = None) -> int:
        """
        Process a transaction and extract exchange addresses.

        Args:
            tx: Raw transaction data
            detected_exchange: Exchange if already detected
            direction: 'inflow' or 'outflow' if known

        Returns:
            Number of new addresses found
        """
        txid = tx.get('txid', '')

        # Extract all addresses
        input_addrs = []
        output_addrs = []
        output_values = []

        # Get inputs
        for vin in tx.get('vin', []):
            if 'coinbase' in vin:
                continue
            prevout = vin.get('prevout', {})
            if prevout:
                addr = prevout.get('scriptPubKey', {}).get('address')
                if addr:
                    input_addrs.append(addr)

        # Get outputs with values
        for vout in tx.get('vout', []):
            addr = vout.get('scriptPubKey', {}).get('address')
            value = vout.get('value', 0)
            if addr:
                output_addrs.append(addr)
                output_values.append(value)

        # Find known exchange in inputs or outputs
        known_exchange = detected_exchange
        known_in_inputs = False
        known_in_outputs = False

        for addr in input_addrs:
            if addr in self.addr_to_exchange:
                known_exchange = self.addr_to_exchange[addr]
                known_in_inputs = True
                break

        for addr in output_addrs:
            if addr in self.addr_to_exchange:
                if not known_exchange:
                    known_exchange = self.addr_to_exchange[addr]
                known_in_outputs = True

        if not known_exchange:
            return 0

        found = 0
        entries = []

        # === OUTFLOW CLUSTERING ===
        # If exchange is in inputs, all other inputs = exchange
        # Also, change outputs (smaller values) = exchange
        if known_in_inputs:
            # Cluster all inputs
            for addr in input_addrs:
                if addr not in self.addresses:
                    entries.append((addr, known_exchange, 'realtime_input'))
                    found += 1

            # Find change output (smaller value in 2-output tx)
            if len(output_addrs) == 2 and len(output_values) == 2:
                # Smaller output is likely change
                if output_values[0] < output_values[1]:
                    change_addr = output_addrs[0]
                else:
                    change_addr = output_addrs[1]

                if change_addr not in self.addresses:
                    entries.append((change_addr, known_exchange, 'realtime_change'))
                    found += 1

            # For 3+ outputs, the ones closest to known addresses might be internal
            if len(output_addrs) >= 3:
                known_outputs = [a for a in output_addrs if a in self.addr_to_exchange]
                if len(known_outputs) >= 2:
                    # Multiple outputs to same exchange = internal
                    for addr in output_addrs:
                        if addr not in self.addresses:
                            entries.append((addr, known_exchange, 'realtime_internal'))
                            found += 1

        # === INFLOW CLUSTERING ===
        # If exchange is in outputs (receiving deposit), those outputs = exchange
        # But don't cluster inputs (those are user wallets)
        if known_in_outputs and not known_in_inputs:
            # Only cluster the output addresses that look like exchange addresses
            for i, addr in enumerate(output_addrs):
                if addr in self.addr_to_exchange:
                    continue
                # Check if it looks like an exchange address (same prefix pattern)
                for prefix in self.exchange_patterns.get(known_exchange, []):
                    if addr.startswith(prefix[:5]):
                        entries.append((addr, known_exchange, 'realtime_deposit'))
                        found += 1
                        break

        # Save batch
        self._save_batch(entries)

        if found > 0:
            self.flows_seen += 1
            print(f"[CLUSTER] {known_exchange}: +{found} addrs from {direction or 'flow'} "
                  f"(total: {len(self.addresses):,})")

            # Log discovery
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""
                INSERT INTO realtime_discoveries
                (txid, exchange, direction, btc_amount, addresses_found, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (txid, known_exchange, direction, sum(output_values), found,
                  datetime.now().isoformat()))
            conn.commit()
            conn.close()

        return found

    def run_zmq_loop(self, zmq_address: str = "tcp://127.0.0.1:28332"):
        """
        Run continuous ZMQ loop to capture transactions.
        """
        import subprocess
        import json

        print(f"Connecting to ZMQ at {zmq_address}")
        print("Monitoring for transactions...")
        print()

        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(zmq_address)
        socket.setsockopt(zmq.SUBSCRIBE, b"rawtx")

        while True:
            try:
                msg = socket.recv_multipart()
                if len(msg) >= 2:
                    topic = msg[0].decode('utf-8', errors='ignore')
                    raw_tx = msg[1].hex()

                    # Decode transaction
                    result = subprocess.run(
                        ['bitcoin-cli', 'decoderawtransaction', raw_tx],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        tx = json.loads(result.stdout)
                        self.process_transaction(tx)

                    # Status every 100 flows
                    if self.flows_seen > 0 and self.flows_seen % 100 == 0:
                        self._print_status()

            except Exception as e:
                print(f"[ERR] {e}")
                time.sleep(1)

    def _print_status(self):
        """Print current status."""
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600

        # Count per exchange
        exchange_counts = defaultdict(int)
        for addr, ex in self.addr_to_exchange.items():
            exchange_counts[ex] += 1

        print()
        print("=" * 50)
        print(f"REAL-TIME CLUSTERING STATUS ({hours:.2f}h)")
        print("=" * 50)
        print(f"Flows processed: {self.flows_seen:,}")
        print(f"Addresses added: {self.addresses_added:,}")
        print(f"Total addresses: {len(self.addresses):,}")
        print()
        print("Major exchanges:")
        for ex in ['binance', 'coinbase', 'okx', 'bybit', 'bitfinex', 'kraken']:
            print(f"  {ex:<15} {exchange_counts.get(ex, 0):>10,}")
        print("=" * 50)
        print()


def run_with_pipeline():
    """
    Hook into the sovereign pipeline to cluster from detected flows.
    """
    import subprocess
    import threading

    cluster = RealtimeCluster()

    def monitor_flows():
        """Monitor flow log for new detections."""
        import subprocess

        # Tail the pipeline log and process flows
        process = subprocess.Popen(
            ['tail', '-f', '/tmp/sovereign_pipeline.log'],
            stdout=subprocess.PIPE,
            text=True
        )

        for line in process.stdout:
            if '[INFLOW]' in line or '[OUTFLOW]' in line:
                # Parse the flow line
                # Format: [INFLOW] binance +7.6102 BTC @ $89,179 -> SHORT
                parts = line.strip().split()
                if len(parts) >= 3:
                    direction = 'inflow' if 'INFLOW' in line else 'outflow'
                    exchange = parts[1]

                    # We don't have the full tx here, but we can use this
                    # to trigger more aggressive scanning
                    print(f"[DETECT] {direction} {exchange}")

    # Start flow monitor
    thread = threading.Thread(target=monitor_flows, daemon=True)
    thread.start()

    # Also run ZMQ loop
    cluster.run_zmq_loop()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Real-time address clustering')
    parser.add_argument('--zmq', action='store_true', help='Run ZMQ loop')
    parser.add_argument('--pipeline', action='store_true', help='Hook into pipeline')
    args = parser.parse_args()

    if args.pipeline:
        run_with_pipeline()
    else:
        cluster = RealtimeCluster()
        cluster.run_zmq_loop()


if __name__ == '__main__':
    main()
