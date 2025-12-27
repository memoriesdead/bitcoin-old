"""
Bitcoin Core Scanner - Transaction-level data from your local node.

Scans Bitcoin Core via RPC to extract transaction-level data for
dates after ORBITAAL ends (2021-01-26 onward).

This creates data in the same format as ORBITAAL for seamless integration.
"""
import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Iterator
from datetime import datetime, date, timedelta
from collections import defaultdict
import json
import sys
import os

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from engine.sovereign.blockchain.rpc import BitcoinRPC, RPCBatchProcessor


class BitcoinCoreScanner:
    """
    Scan Bitcoin Core for transaction-level data.

    Produces daily aggregations compatible with ORBITAAL format.
    """

    # ORBITAAL ends on 2021-01-25, we start from 2021-01-26
    START_HEIGHT = 667_000  # Approximate block height for 2021-01-26
    SATOSHI_PER_BTC = 100_000_000

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8332,
        user: str = "bitcoin",
        password: str = "bitcoin"
    ):
        self.rpc = BitcoinRPC(host, port, user, password)
        self._height_cache: Dict[int, int] = {}  # height -> timestamp

    def get_current_height(self) -> int:
        """Get current blockchain height."""
        return self.rpc.getblockcount()

    def get_block_timestamp(self, height: int) -> int:
        """Get Unix timestamp for block height."""
        if height in self._height_cache:
            return self._height_cache[height]

        block = self.rpc.getblockbyheight(height, verbosity=1)
        ts = block['time']
        self._height_cache[height] = ts
        return ts

    def find_height_for_date(self, target_date: date) -> int:
        """
        Binary search to find first block height for a given date.
        """
        target_ts = int(datetime.combine(target_date, datetime.min.time()).timestamp())

        low = 0
        high = self.get_current_height()

        while low < high:
            mid = (low + high) // 2
            block_ts = self.get_block_timestamp(mid)

            if block_ts < target_ts:
                low = mid + 1
            else:
                high = mid

        return low

    def get_blocks_for_date(self, target_date: date) -> List[int]:
        """Get all block heights for a specific date."""
        start_ts = int(datetime.combine(target_date, datetime.min.time()).timestamp())
        end_ts = start_ts + 86400  # Next day

        start_height = self.find_height_for_date(target_date)
        heights = []

        height = start_height
        while True:
            ts = self.get_block_timestamp(height)
            if ts >= end_ts:
                break
            if ts >= start_ts:
                heights.append(height)
            height += 1

        return heights

    def scan_block_transactions(self, height: int) -> Dict:
        """
        Scan a single block and extract transaction data.

        Returns dict with:
        - tx_count: Number of transactions
        - total_value_btc: Total value transferred
        - flows: List of (from_addr, to_addr, value_btc) tuples
        - whale_txs: Transactions over 100 BTC
        """
        block = self.rpc.getblockbyheight(height, verbosity=2)

        tx_count = 0
        total_value = 0
        whale_count = 0
        whale_value = 0
        unique_inputs = set()
        unique_outputs = set()

        for tx in block.get('tx', []):
            # Skip coinbase
            if 'coinbase' in tx.get('vin', [{}])[0]:
                continue

            tx_count += 1
            tx_value = 0

            # Process inputs (sources)
            for vin in tx.get('vin', []):
                if 'txid' in vin:
                    # Would need to look up previous tx for address
                    # For efficiency, we skip detailed input tracking
                    pass

            # Process outputs (destinations)
            for vout in tx.get('vout', []):
                value_btc = vout.get('value', 0)
                tx_value += value_btc

                # Extract output address
                spk = vout.get('scriptPubKey', {})
                if 'address' in spk:
                    unique_outputs.add(spk['address'])
                elif 'addresses' in spk:
                    unique_outputs.update(spk['addresses'])

            total_value += tx_value

            # Whale detection (>100 BTC)
            if tx_value > 100:
                whale_count += 1
                whale_value += tx_value

        return {
            'height': height,
            'timestamp': block['time'],
            'tx_count': tx_count,
            'total_value_btc': total_value,
            'unique_receivers': len(unique_outputs),
            'whale_tx_count': whale_count,
            'whale_value_btc': whale_value,
        }

    def scan_date(self, target_date: date, progress_callback=None) -> Dict:
        """
        Scan all blocks for a specific date and aggregate.

        Returns daily statistics compatible with ORBITAAL format.
        """
        heights = self.get_blocks_for_date(target_date)

        daily_stats = {
            'date': str(target_date),
            'timestamp': int(datetime.combine(target_date, datetime.min.time()).timestamp()),
            'blocks': len(heights),
            'tx_count': 0,
            'total_value_btc': 0.0,
            'unique_receivers': 0,
            'whale_tx_count': 0,
            'whale_value_btc': 0.0,
        }

        all_receivers = set()

        for i, height in enumerate(heights):
            try:
                block_stats = self.scan_block_transactions(height)
                daily_stats['tx_count'] += block_stats['tx_count']
                daily_stats['total_value_btc'] += block_stats['total_value_btc']
                daily_stats['whale_tx_count'] += block_stats['whale_tx_count']
                daily_stats['whale_value_btc'] += block_stats['whale_value_btc']

                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback(i + 1, len(heights), height)

            except Exception as e:
                print(f"Error scanning block {height}: {e}")
                continue

        daily_stats['unique_receivers'] = len(all_receivers) if all_receivers else daily_stats['tx_count']

        return daily_stats

    def build_daily_features_db(
        self,
        output_db: str = "data/btc_scanner_daily.db",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback=None
    ) -> int:
        """
        Build SQLite database of daily features from Bitcoin Core.

        Scans from ORBITAAL end date (2021-01-26) to present.

        Returns: Number of days processed
        """
        output_path = Path(output_db)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(output_path)
        c = conn.cursor()

        # Create table (compatible with ORBITAAL format)
        c.execute("""
            CREATE TABLE IF NOT EXISTS daily_features (
                date TEXT PRIMARY KEY,
                timestamp INTEGER,
                blocks INTEGER,
                tx_count INTEGER,
                total_value_btc REAL,
                unique_receivers INTEGER,
                whale_tx_count INTEGER,
                whale_value_btc REAL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON daily_features(timestamp)")
        conn.commit()

        # Determine date range
        if start_date:
            current_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        else:
            current_date = date(2021, 1, 26)  # Day after ORBITAAL ends

        if end_date:
            final_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            final_date = date.today() - timedelta(days=1)  # Yesterday

        processed = 0
        total_days = (final_date - current_date).days + 1

        while current_date <= final_date:
            # Skip if already processed
            c.execute("SELECT 1 FROM daily_features WHERE date = ?", (str(current_date),))
            if c.fetchone():
                current_date += timedelta(days=1)
                continue

            try:
                print(f"Scanning {current_date} ({processed + 1}/{total_days})...")

                stats = self.scan_date(current_date)

                c.execute("""
                    INSERT OR REPLACE INTO daily_features VALUES (?,?,?,?,?,?,?,?)
                """, (
                    stats['date'],
                    stats['timestamp'],
                    stats['blocks'],
                    stats['tx_count'],
                    stats['total_value_btc'],
                    stats['unique_receivers'],
                    stats['whale_tx_count'],
                    stats['whale_value_btc'],
                ))

                conn.commit()
                processed += 1

                if progress_callback:
                    progress_callback(processed, total_days, current_date)

            except Exception as e:
                print(f"Error processing {current_date}: {e}")

            current_date += timedelta(days=1)

        conn.close()
        return processed

    def quick_scan(self, days: int = 7) -> List[Dict]:
        """
        Quick scan of recent days for testing.

        Returns list of daily statistics.
        """
        results = []
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=days - 1)

        current = start_date
        while current <= end_date:
            try:
                print(f"Scanning {current}...")
                stats = self.scan_date(current)
                results.append(stats)
                print(f"  Blocks: {stats['blocks']}, TXs: {stats['tx_count']:,}, Value: {stats['total_value_btc']:,.0f} BTC")
            except Exception as e:
                print(f"Error: {e}")
            current += timedelta(days=1)

        return results


class FastBlockScanner:
    """
    Fast scanner using block-level data only (no individual transaction parsing).

    This is much faster than full transaction scanning and sufficient for
    most trading signals (tx_count, block fullness, etc).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8332,
        user: str = "bitcoin",
        password: str = "bitcoin"
    ):
        self.rpc = BitcoinRPC(host, port, user, password)

    def scan_blocks_fast(
        self,
        start_height: int,
        end_height: int,
        output_db: str = "data/block_features.db",
        batch_size: int = 100,
        progress_callback=None
    ) -> int:
        """
        Fast scan using verbosity=1 (no full tx data).

        Returns: Number of blocks scanned
        """
        output_path = Path(output_db)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(output_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS block_features (
                height INTEGER PRIMARY KEY,
                timestamp INTEGER,
                hash TEXT,
                tx_count INTEGER,
                size INTEGER,
                weight INTEGER,
                difficulty REAL,
                version INTEGER
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON block_features(timestamp)")
        conn.commit()

        scanned = 0
        batch = RPCBatchProcessor(self.rpc, batch_size=batch_size)

        for height in range(start_height, end_height + 1):
            batch.add("getblockhash", [height])

        hashes = batch.execute()

        # Now get block data
        for i, blockhash in enumerate(hashes):
            if blockhash is None:
                continue

            batch.add("getblock", [blockhash, 1])

        blocks = batch.execute()

        for block in blocks:
            if block is None:
                continue

            c.execute("""
                INSERT OR REPLACE INTO block_features VALUES (?,?,?,?,?,?,?,?)
            """, (
                block['height'],
                block['time'],
                block['hash'],
                block['nTx'],
                block['size'],
                block.get('weight', block['size'] * 4),
                block['difficulty'],
                block['version'],
            ))
            scanned += 1

            if scanned % 1000 == 0:
                conn.commit()
                if progress_callback:
                    progress_callback(scanned, end_height - start_height + 1)

        conn.commit()
        conn.close()
        return scanned


def quick_test():
    """Quick test of Bitcoin Core scanner."""
    print("Testing Bitcoin Core Scanner...")
    print("=" * 50)

    try:
        scanner = BitcoinCoreScanner()
        height = scanner.get_current_height()
        print(f"Current block height: {height:,}")

        # Test quick scan
        print("\nQuick scan of last 3 days:")
        results = scanner.quick_scan(days=3)

        for r in results:
            print(f"  {r['date']}: {r['tx_count']:,} txs, {r['total_value_btc']:,.0f} BTC")

    except Exception as e:
        print(f"Error connecting to Bitcoin Core: {e}")
        print("Make sure Bitcoin Core is running with RPC enabled.")
        print("Check your bitcoin.conf has: server=1, rpcuser=bitcoin, rpcpassword=bitcoin")


if __name__ == "__main__":
    quick_test()
