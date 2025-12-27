#!/usr/bin/env python3
"""
HISTORICAL DATA FETCHER - GENESIS TO PRESENT
==============================================

Fetches historical Bitcoin blockchain data from public APIs
since our node is pruned.

DATA SOURCES:
1. Blockchain.com API - Block data, transactions
2. Blockstream API - Block details
3. Mempool.space API - Fee estimates, block data
4. CoinGecko/CoinMarketCap - Historical prices

This allows us to build the full historical dataset needed
for RenTech-style pattern discovery without requiring 700GB
for a full node.

USAGE:
    python -m engine.sovereign.formulas.historical_fetcher --start 0 --end -1

    # Fetch specific range:
    python -m engine.sovereign.formulas.historical_fetcher --start 500000 --end 600000
"""

import os
import sys
import time
import json
import sqlite3
import argparse
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np

# Rate limiting
REQUEST_DELAY = 0.1  # seconds between requests
MAX_RETRIES = 3

# Scales for aggregation (in blocks)
BLOCK_SCALES = [1, 3, 6, 36, 144]

# Block milestones
GENESIS_TIMESTAMP = 1231006505
HALVINGS = [210000, 420000, 630000, 840000]
SEGWIT_ACTIVATION = 481824
TAPROOT_ACTIVATION = 709632


@dataclass
class BlockData:
    """Raw block data from API."""
    height: int
    hash: str
    time: int
    size: int
    weight: int
    tx_count: int
    total_fees: float
    difficulty: float
    prev_hash: str


@dataclass
class HistoricalFeatures:
    """Feature vector for HMM training."""
    block_height: int
    block_time: int
    scale_blocks: int

    # Block features
    block_time_delta: float = 0.0
    block_size: int = 0
    block_weight: int = 0
    block_fullness: float = 0.0
    block_tx_count: int = 0
    block_fees_btc: float = 0.0

    # Fee features
    fee_density: float = 0.0

    # Network features
    difficulty: float = 0.0
    hash_rate_estimate: float = 0.0
    subsidy_btc: float = 0.0

    # Derived
    halving_epoch: int = 0
    days_since_halving: float = 0.0
    market_cycle_phase: float = 0.0
    block_interval_zscore: float = 0.0

    # Price (if available)
    price_usd: float = 0.0
    price_change_24h: float = 0.0


class BlockchainAPI:
    """Fetches data from public blockchain APIs."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research Bot)'
        })
        self.last_request = 0
        self.lock = threading.Lock()

    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        with self.lock:
            elapsed = time.time() - self.last_request
            if elapsed < REQUEST_DELAY:
                time.sleep(REQUEST_DELAY - elapsed)
            self.last_request = time.time()

    def _get(self, url: str, retries: int = MAX_RETRIES) -> Optional[dict]:
        """Make rate-limited GET request."""
        self._rate_limit()

        for attempt in range(retries):
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:  # Rate limited
                    time.sleep(5 * (attempt + 1))
                else:
                    return None
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    return None
        return None

    def get_block_hash(self, height: int) -> Optional[str]:
        """Get block hash by height."""
        # Try Blockstream first
        url = f"https://blockstream.info/api/block-height/{height}"
        try:
            self._rate_limit()
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.text.strip()
        except:
            pass

        # Fallback to blockchain.com
        data = self._get(f"https://blockchain.info/block-height/{height}?format=json")
        if data and 'blocks' in data and data['blocks']:
            return data['blocks'][0].get('hash')
        return None

    def get_block_blockstream(self, hash_or_height) -> Optional[BlockData]:
        """Get block from Blockstream API."""
        if isinstance(hash_or_height, int):
            block_hash = self.get_block_hash(hash_or_height)
            if not block_hash:
                return None
        else:
            block_hash = hash_or_height

        url = f"https://blockstream.info/api/block/{block_hash}"
        data = self._get(url)

        if not data:
            return None

        try:
            return BlockData(
                height=data['height'],
                hash=data['id'],
                time=data['timestamp'],
                size=data['size'],
                weight=data['weight'],
                tx_count=data['tx_count'],
                total_fees=data.get('totalfee', 0) / 1e8,  # satoshis to BTC
                difficulty=float(data.get('difficulty', 0)),
                prev_hash=data.get('previousblockhash', '')
            )
        except Exception as e:
            return None

    def get_block_blockchain_com(self, height: int) -> Optional[BlockData]:
        """Get block from Blockchain.com API."""
        data = self._get(f"https://blockchain.info/block-height/{height}?format=json")

        if not data or 'blocks' not in data or not data['blocks']:
            return None

        block = data['blocks'][0]
        try:
            return BlockData(
                height=block['height'],
                hash=block['hash'],
                time=block['time'],
                size=block['size'],
                weight=block.get('weight', block['size'] * 4),
                tx_count=block['n_tx'],
                total_fees=block.get('fee', 0) / 1e8,
                difficulty=0,  # Not in this API
                prev_hash=block.get('prev_block', '')
            )
        except Exception as e:
            return None

    def get_block(self, height: int) -> Optional[BlockData]:
        """Get block data, trying multiple APIs."""
        # Try Blockstream first (faster, more reliable)
        block = self.get_block_blockstream(height)
        if block:
            return block

        # Fallback to Blockchain.com
        return self.get_block_blockchain_com(height)

    def get_current_height(self) -> int:
        """Get current blockchain height."""
        try:
            self._rate_limit()
            resp = self.session.get("https://blockstream.info/api/blocks/tip/height", timeout=30)
            if resp.status_code == 200:
                return int(resp.text.strip())
        except:
            pass

        data = self._get("https://blockchain.info/latestblock")
        if data:
            return data.get('height', 0)
        return 0


class HistoricalPriceFetcher:
    """Fetches historical BTC prices."""

    def __init__(self):
        self.session = requests.Session()
        self.price_cache: Dict[str, float] = {}  # date -> price

    def get_price_on_date(self, timestamp: int) -> float:
        """Get BTC price on a specific date."""
        date_str = datetime.utcfromtimestamp(timestamp).strftime('%d-%m-%Y')

        if date_str in self.price_cache:
            return self.price_cache[date_str]

        try:
            # CoinGecko historical price
            url = f"https://api.coingecko.com/api/v3/coins/bitcoin/history?date={date_str}"
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                price = data.get('market_data', {}).get('current_price', {}).get('usd', 0)
                self.price_cache[date_str] = price
                return price
        except:
            pass

        return 0.0

    def preload_prices(self, start_timestamp: int, end_timestamp: int):
        """Preload prices for a date range."""
        # This would batch-fetch prices
        pass


class HistoricalFetcher:
    """Main historical data fetcher and processor."""

    def __init__(self, db_path: str = None):
        self.api = BlockchainAPI()
        self.price_fetcher = HistoricalPriceFetcher()

        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', '..', 'data', 'historical_features.db'
        )

        self.prev_block_time: Dict[int, int] = {}  # height -> time

        self._init_db()

    def _init_db(self):
        """Initialize database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS block_features (
                block_height INTEGER,
                scale_blocks INTEGER,
                block_time INTEGER,
                features TEXT,
                PRIMARY KEY (block_height, scale_blocks)
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS raw_blocks (
                height INTEGER PRIMARY KEY,
                hash TEXT,
                time INTEGER,
                size INTEGER,
                weight INTEGER,
                tx_count INTEGER,
                fees REAL,
                difficulty REAL
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS fetch_progress (
                id INTEGER PRIMARY KEY,
                last_block INTEGER,
                timestamp REAL
            )
        ''')

        c.execute('CREATE INDEX IF NOT EXISTS idx_block_time ON block_features(block_time)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_raw_time ON raw_blocks(time)')

        conn.commit()
        conn.close()
        print(f"[DB] Database: {self.db_path}")

    def get_last_fetched(self) -> int:
        """Get last successfully fetched block."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT MAX(height) FROM raw_blocks')
        row = c.fetchone()
        conn.close()
        return row[0] if row and row[0] else -1

    def save_raw_block(self, block: BlockData):
        """Save raw block data."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute('''
                INSERT OR REPLACE INTO raw_blocks
                (height, hash, time, size, weight, tx_count, fees, difficulty)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                block.height, block.hash, block.time, block.size,
                block.weight, block.tx_count, block.total_fees, block.difficulty
            ))
            conn.commit()
        finally:
            conn.close()

    def save_features(self, features: HistoricalFeatures):
        """Save processed features."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute('''
                INSERT OR REPLACE INTO block_features
                (block_height, scale_blocks, block_time, features)
                VALUES (?, ?, ?, ?)
            ''', (
                features.block_height,
                features.scale_blocks,
                features.block_time,
                json.dumps(asdict(features))
            ))
            conn.commit()
        finally:
            conn.close()

    def _get_halving_info(self, height: int) -> Tuple[int, float, float]:
        """Get halving epoch and cycle info."""
        epoch = sum(1 for h in HALVINGS if height >= h)
        last_halving = HALVINGS[epoch - 1] if epoch > 0 else 0
        blocks_since = height - last_halving
        days_since = blocks_since * 10 / 60 / 24
        cycle_phase = (height % 210000) / 210000
        return epoch, days_since, cycle_phase

    def process_block(self, block: BlockData, prev_time: int = None) -> Dict:
        """Process block into features dict."""
        time_delta = block.time - prev_time if prev_time else 600

        epoch, days_since_halving, cycle_phase = self._get_halving_info(block.height)
        subsidy = 50 / (2 ** epoch)

        # Estimate hash rate: difficulty * 2^32 / 600
        hash_rate = block.difficulty * (2**32) / 600 / 1e18 if block.difficulty > 0 else 0

        return {
            'height': block.height,
            'time': block.time,
            'time_delta': time_delta,
            'size': block.size,
            'weight': block.weight,
            'fullness': block.weight / 4000000 if block.weight else block.size / 1000000,
            'tx_count': block.tx_count,
            'total_fees': block.total_fees,
            'fee_density': (block.total_fees * 1e8) / block.weight if block.weight > 0 else 0,
            'difficulty': block.difficulty,
            'hash_rate': hash_rate,
            'subsidy': subsidy,
            'epoch': epoch,
            'days_since_halving': days_since_halving,
            'cycle_phase': cycle_phase,
        }

    def aggregate_features(self, blocks: List[Dict], scale: int) -> HistoricalFeatures:
        """Aggregate block data into feature vector."""
        if not blocks:
            return None

        last = blocks[-1]

        features = HistoricalFeatures(
            block_height=last['height'],
            block_time=last['time'],
            scale_blocks=scale,
        )

        features.block_time_delta = np.mean([b['time_delta'] for b in blocks])
        features.block_size = int(np.mean([b['size'] for b in blocks]))
        features.block_weight = int(np.mean([b['weight'] for b in blocks]))
        features.block_fullness = np.mean([b['fullness'] for b in blocks])
        features.block_tx_count = int(np.mean([b['tx_count'] for b in blocks]))
        features.block_fees_btc = sum([b['total_fees'] for b in blocks])
        features.fee_density = np.mean([b['fee_density'] for b in blocks])
        features.difficulty = last['difficulty']
        features.hash_rate_estimate = last['hash_rate']
        features.subsidy_btc = last['subsidy']
        features.halving_epoch = last['epoch']
        features.days_since_halving = last['days_since_halving']
        features.market_cycle_phase = last['cycle_phase']
        features.block_interval_zscore = (features.block_time_delta - 600) / 300

        return features

    def fetch_range(self, start: int, end: int, workers: int = 4):
        """
        Fetch blocks in range using multiple workers.

        Args:
            start: Starting block height
            end: Ending block height
            workers: Number of parallel workers
        """
        print(f"""
================================================================================
                    HISTORICAL DATA FETCHER
================================================================================

"The data speaks for itself."
- Jim Simons

FETCHING: Block {start:,} to {end:,}
TOTAL: {end - start + 1:,} blocks
WORKERS: {workers}

================================================================================
""")

        # Fetch blocks
        scale_buffers: Dict[int, List[Dict]] = {s: [] for s in BLOCK_SCALES}
        prev_time = None
        blocks_fetched = 0
        start_time = time.time()
        errors = 0

        for height in range(start, end + 1):
            block = self.api.get_block(height)

            if block is None:
                errors += 1
                if errors > 10:
                    print(f"[WARN] Too many errors at height {height}, slowing down...")
                    time.sleep(5)
                continue

            errors = 0

            # Save raw block
            self.save_raw_block(block)

            # Process into features
            processed = self.process_block(block, prev_time)
            prev_time = block.time

            # Add to scale buffers
            for scale in BLOCK_SCALES:
                scale_buffers[scale].append(processed)

                if len(scale_buffers[scale]) >= scale:
                    features = self.aggregate_features(scale_buffers[scale], scale)
                    if features:
                        self.save_features(features)
                    scale_buffers[scale] = []

            blocks_fetched += 1

            # Progress
            if blocks_fetched % 100 == 0:
                elapsed = time.time() - start_time
                rate = blocks_fetched / elapsed
                remaining = (end - height) / rate if rate > 0 else 0

                print(f"[FETCH] Block {height:,} / {end:,} | "
                      f"{blocks_fetched:,} done | "
                      f"{rate:.1f} blk/s | "
                      f"ETA: {remaining/3600:.1f}h")

        elapsed = time.time() - start_time
        print(f"""
================================================================================
                         FETCH COMPLETE
================================================================================

Blocks fetched: {blocks_fetched:,}
Time: {elapsed/3600:.2f} hours
Rate: {blocks_fetched/elapsed:.1f} blocks/second

================================================================================
""")


def main():
    parser = argparse.ArgumentParser(description='Historical Data Fetcher')
    parser.add_argument('--start', type=int, default=0, help='Starting block')
    parser.add_argument('--end', type=int, default=-1, help='Ending block (-1 = current)')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')

    args = parser.parse_args()

    fetcher = HistoricalFetcher()

    # Get current height if needed
    if args.end == -1:
        args.end = fetcher.api.get_current_height()
        print(f"[INFO] Current height: {args.end:,}")

    # Check for resume
    last_fetched = fetcher.get_last_fetched()
    if last_fetched >= args.start:
        args.start = last_fetched + 1
        print(f"[RESUME] Continuing from block {args.start:,}")

    fetcher.fetch_range(args.start, args.end, args.workers)


if __name__ == '__main__':
    main()
