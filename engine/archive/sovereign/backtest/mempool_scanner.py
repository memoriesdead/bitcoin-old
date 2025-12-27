#!/usr/bin/env python3
"""
Mempool.space Scanner - Fill gap from 2021 to present

Uses FREE mempool.space API to get Bitcoin transactions from Jan 2021 to now.
No API key required.

Usage:
    python -m engine.sovereign.backtest.mempool_scanner
"""
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from typing import Dict, List, Optional
import sys

# Mempool.space API (FREE, no auth)
MEMPOOL_API = "https://mempool.space/api"

# Block heights for date reference:
# Jan 1, 2021 = block ~663,000
# Jan 1, 2022 = block ~716,000
# Jan 1, 2023 = block ~769,000
# Jan 1, 2024 = block ~822,000
# Dec 13, 2025 = block ~927,000+ (current tip)

START_BLOCK = 663000  # Jan 2021 (where ORBITAAL ends)
BATCH_SIZE = 10       # Blocks per batch


class MempoolScanner:
    """
    Scan Bitcoin blockchain via mempool.space API.
    FREE, no API key, rate limited to ~10 req/sec.
    """

    def __init__(self, db_path: str = "data/historical_flows.db"):
        self.db_path = Path(db_path)
        self.exchange_addresses: Dict[str, str] = {}
        self._load_exchanges()
        self._init_db()

        # Stats
        self.blocks_scanned = 0
        self.flows_found = 0

    def _load_exchanges(self):
        """Load exchange addresses."""
        exchanges_file = Path("data/exchanges.json")
        if exchanges_file.exists():
            with open(exchanges_file) as f:
                data = json.load(f)
            for exchange, addresses in data.items():
                if isinstance(addresses, list):
                    for addr in addresses:
                        self.exchange_addresses[addr] = exchange
            print(f"[+] Loaded {len(self.exchange_addresses):,} exchange addresses")

    def _init_db(self):
        """Initialize database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS mempool_flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_height INTEGER,
                block_time INTEGER,
                txid TEXT,
                exchange TEXT,
                direction INTEGER,
                amount_btc REAL,
                address TEXT,
                UNIQUE(txid, address)
            )
        ''')

        c.execute('CREATE INDEX IF NOT EXISTS idx_mempool_time ON mempool_flows(block_time)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_mempool_exchange ON mempool_flows(exchange)')

        # Progress tracking
        c.execute('''
            CREATE TABLE IF NOT EXISTS mempool_progress (
                id INTEGER PRIMARY KEY,
                last_block INTEGER,
                updated_at TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def _api_get(self, endpoint: str) -> Optional[dict]:
        """Make API request with rate limiting."""
        url = f"{MEMPOOL_API}/{endpoint}"
        try:
            req = Request(url, headers={'User-Agent': 'RenTech-Scanner/1.0'})
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            if e.code == 429:  # Rate limited
                print(f"[!] Rate limited, waiting 10s...")
                time.sleep(10)
                return self._api_get(endpoint)
            print(f"[!] HTTP error {e.code}: {endpoint}")
            return None
        except Exception as e:
            print(f"[!] API error: {e}")
            return None

    def get_block(self, height: int) -> Optional[dict]:
        """Get block by height."""
        # First get block hash
        block_hash = self._api_get(f"block-height/{height}")
        if not block_hash:
            return None

        # Then get block details
        block = self._api_get(f"block/{block_hash}")
        return block

    def get_block_txids(self, block_hash: str) -> List[str]:
        """Get all transaction IDs in a block."""
        txids = self._api_get(f"block/{block_hash}/txids")
        return txids or []

    def get_transaction(self, txid: str) -> Optional[dict]:
        """Get transaction details."""
        return self._api_get(f"tx/{txid}")

    def _get_last_scanned_block(self) -> int:
        """Get last scanned block."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT last_block FROM mempool_progress WHERE id = 1')
        row = c.fetchone()
        conn.close()
        return row[0] if row else START_BLOCK - 1

    def _save_progress(self, block_height: int):
        """Save progress."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO mempool_progress (id, last_block, updated_at)
            VALUES (1, ?, ?)
        ''', (block_height, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def _save_flows(self, flows: List[dict]):
        """Save flows to database."""
        if not flows:
            return

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        for flow in flows:
            try:
                c.execute('''
                    INSERT OR IGNORE INTO mempool_flows
                    (block_height, block_time, txid, exchange, direction, amount_btc, address)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    flow['block_height'],
                    flow['block_time'],
                    flow['txid'],
                    flow['exchange'],
                    flow['direction'],
                    flow['amount_btc'],
                    flow['address']
                ))
            except:
                pass

        conn.commit()
        conn.close()
        self.flows_found += len(flows)

    def _process_transaction(self, tx: dict, block_height: int, block_time: int) -> List[dict]:
        """Process a single transaction for exchange flows."""
        flows = []
        txid = tx.get('txid', '')

        # Extract input addresses
        inputs = set()
        for vin in tx.get('vin', []):
            if 'prevout' in vin:
                addr = vin['prevout'].get('scriptpubkey_address')
                if addr:
                    inputs.add(addr)

        # Extract output addresses and values
        outputs = {}
        for vout in tx.get('vout', []):
            addr = vout.get('scriptpubkey_address')
            value = vout.get('value', 0) / 100_000_000  # satoshi to BTC
            if addr:
                outputs[addr] = outputs.get(addr, 0) + value

        # Check for INFLOWS (deposits to exchange)
        for addr, amount in outputs.items():
            if addr in self.exchange_addresses:
                exchange = self.exchange_addresses[addr]
                input_exchanges = {self.exchange_addresses.get(a) for a in inputs}
                if exchange not in input_exchanges and amount >= 0.01:
                    flows.append({
                        'block_height': block_height,
                        'block_time': block_time,
                        'txid': txid,
                        'exchange': exchange,
                        'direction': -1,  # INFLOW = SHORT signal
                        'amount_btc': amount,
                        'address': addr
                    })

        # Check for OUTFLOWS (withdrawals from exchange)
        for addr in inputs:
            if addr in self.exchange_addresses:
                exchange = self.exchange_addresses[addr]
                output_exchanges = {self.exchange_addresses.get(a) for a in outputs.keys()}
                if exchange not in output_exchanges:
                    amount = sum(v for a, v in outputs.items() if a not in self.exchange_addresses)
                    if amount >= 0.01:
                        flows.append({
                            'block_height': block_height,
                            'block_time': block_time,
                            'txid': txid,
                            'exchange': exchange,
                            'direction': 1,  # OUTFLOW = LONG signal
                            'amount_btc': amount,
                            'address': addr
                        })

        return flows

    def scan(self, start_block: int = None, end_block: int = None):
        """
        Scan blockchain for exchange flows using mempool.space API.

        Note: This is slow (~1 block/sec) due to API rate limits.
        For faster scanning, use local node when synced.
        """
        # Get current tip
        tip = self._api_get("blocks/tip/height")
        if not tip:
            print("[!] Could not get chain tip")
            return

        # Determine range
        if start_block is None:
            start_block = self._get_last_scanned_block() + 1

        if end_block is None:
            end_block = tip

        total_blocks = end_block - start_block + 1

        print("\n" + "=" * 60)
        print("MEMPOOL.SPACE SCANNER (2021-2025)")
        print("=" * 60)
        print(f"Blocks: {start_block:,} to {end_block:,} ({total_blocks:,} blocks)")
        print(f"Exchange addresses: {len(self.exchange_addresses):,}")
        print(f"Rate: ~1 block/sec (API limited)")
        print(f"ETA: ~{total_blocks / 3600:.1f} hours")
        print("=" * 60 + "\n")

        if not self.exchange_addresses:
            print("[!] No exchange addresses loaded!")
            print("[!] Run: python -m engine.sovereign.backtest.download_exchanges")
            return

        start_time = time.time()
        batch_flows = []

        for height in range(start_block, end_block + 1):
            try:
                # Get block
                block = self.get_block(height)
                if not block:
                    continue

                block_hash = block['id']
                block_time = block['timestamp']

                # Get all txids
                txids = self.get_block_txids(block_hash)

                # Process each transaction (rate limited)
                for txid in txids[:100]:  # Limit to first 100 txs per block for speed
                    tx = self.get_transaction(txid)
                    if tx:
                        flows = self._process_transaction(tx, height, block_time)
                        batch_flows.extend(flows)
                    time.sleep(0.1)  # Rate limiting

                self.blocks_scanned += 1

                # Progress
                if self.blocks_scanned % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = self.blocks_scanned / elapsed if elapsed > 0 else 0
                    eta = (total_blocks - self.blocks_scanned) / rate / 3600 if rate > 0 else 0

                    block_date = datetime.fromtimestamp(block_time).strftime('%Y-%m-%d')
                    print(f"[{height:,}] {block_date} | "
                          f"Scanned: {self.blocks_scanned:,} | "
                          f"Flows: {self.flows_found:,} | "
                          f"ETA: {eta:.1f}h")

                    # Save batch
                    self._save_flows(batch_flows)
                    self._save_progress(height)
                    batch_flows = []

            except KeyboardInterrupt:
                print("\n[!] Interrupted by user")
                break
            except Exception as e:
                print(f"[!] Error at block {height}: {e}")
                time.sleep(1)

        # Final save
        if batch_flows:
            self._save_flows(batch_flows)
        self._save_progress(height)

        print("\n" + "=" * 60)
        print("SCAN COMPLETE")
        print("=" * 60)
        print(f"Blocks scanned: {self.blocks_scanned:,}")
        print(f"Flows found: {self.flows_found:,}")
        print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Scan blockchain via mempool.space')
    parser.add_argument('--start', type=int, default=None, help='Start block')
    parser.add_argument('--end', type=int, default=None, help='End block')
    args = parser.parse_args()

    scanner = MempoolScanner()
    scanner.scan(start_block=args.start, end_block=args.end)


if __name__ == '__main__':
    main()
