#!/usr/bin/env python3
"""
HISTORICAL BLOCKCHAIN SCANNER - GENESIS TO PRESENT
====================================================

"We don't start with models. We start with data."
- Jim Simons, Renaissance Technologies

Scans the ENTIRE Bitcoin blockchain from block 0 (genesis) to present,
extracting all 43 features for pattern discovery.

This is the foundation of the RenTech approach:
1. Gather ALL historical data
2. Find patterns that worked across different market conditions
3. Validate patterns statistically
4. Only then deploy live

FEATURES EXTRACTED (43 total):
- Block structure (timing, size, fees, fullness)
- Transaction patterns (segwit, taproot, batching, consolidation)
- UTXO economics (coin days destroyed, dormancy)
- Network state (difficulty, hash rate)
- Lightning Network (channel opens/closes) - from 2018
- Ordinals/Inscriptions - from 2023
- Miner behavior (empty blocks, timestamps, pools)
- Address clustering patterns

USAGE:
    python -m engine.sovereign.formulas.historical_scanner --start 0 --end -1

    # Resume from specific block:
    python -m engine.sovereign.formulas.historical_scanner --start 500000

    # Scan specific range:
    python -m engine.sovereign.formulas.historical_scanner --start 0 --end 100000
"""

import os
import sys
import time
import json
import sqlite3
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import struct

import numpy as np

# Bitcoin block timestamps
GENESIS_TIMESTAMP = 1231006505  # Jan 3, 2009
SEGWIT_ACTIVATION = 481824  # Aug 24, 2017
TAPROOT_ACTIVATION = 709632  # Nov 14, 2021
LIGHTNING_START = 506000  # ~Early 2018
ORDINALS_START = 767430  # Jan 2023

# Halving blocks
HALVINGS = [210000, 420000, 630000, 840000]

# Scales for aggregation (in blocks, not seconds for historical)
# 1 block ≈ 10 min, so:
# 1 block, 3 blocks (30 min), 6 blocks (1 hr), 36 blocks (6 hr), 144 blocks (1 day)
BLOCK_SCALES = [1, 3, 6, 36, 144]


@dataclass
class HistoricalFeatures:
    """Feature vector extracted from historical block data."""
    block_height: int
    block_time: int
    scale_blocks: int

    # Block features (6)
    block_time_delta: float = 0.0  # seconds since last block
    block_size: int = 0
    block_weight: int = 0
    block_fullness: float = 0.0  # weight / 4M
    block_tx_count: int = 0
    block_fees_btc: float = 0.0

    # Fee features (4)
    fee_density: float = 0.0  # sat/vB average
    fee_min: float = 0.0
    fee_max: float = 0.0
    fee_spread: float = 0.0

    # Transaction features (8)
    tx_avg_size: float = 0.0
    tx_avg_inputs: float = 0.0
    tx_avg_outputs: float = 0.0
    tx_consolidation_ratio: float = 0.0  # many inputs -> few outputs
    tx_batch_ratio: float = 0.0  # few inputs -> many outputs
    tx_segwit_ratio: float = 0.0
    tx_taproot_ratio: float = 0.0
    tx_multisig_ratio: float = 0.0

    # UTXO features (4)
    coin_days_destroyed: float = 0.0
    avg_utxo_age_blocks: float = 0.0
    old_coin_ratio: float = 0.0  # coins > 1 year old moving
    new_coin_ratio: float = 0.0  # coins < 1 day old moving

    # Network features (4)
    difficulty: float = 0.0
    difficulty_change: float = 0.0  # vs previous adjustment
    hash_rate_estimate: float = 0.0  # EH/s
    subsidy_btc: float = 0.0

    # Miner features (5)
    miner_pool: str = ""
    empty_block: int = 0  # 1 if < 10 tx
    timestamp_delta: float = 0.0  # block time - median past time
    coinbase_size: int = 0
    fees_vs_subsidy: float = 0.0  # fee revenue ratio

    # Lightning features (3) - only after LIGHTNING_START
    ln_channel_opens: int = 0
    ln_channel_closes: int = 0
    ln_funding_volume: float = 0.0

    # Ordinals features (3) - only after ORDINALS_START
    inscription_count: int = 0
    inscription_bytes: int = 0
    op_return_count: int = 0

    # Derived (4)
    halving_epoch: int = 0  # 0, 1, 2, 3, 4
    days_since_halving: float = 0.0
    market_cycle_phase: float = 0.0  # 0-1 within halving cycle
    block_interval_zscore: float = 0.0  # deviation from 600s

    # Price data (if available from external source)
    price_usd: float = 0.0
    price_change: float = 0.0


class HistoricalScanner:
    """
    Scans entire blockchain history extracting features.

    Uses Bitcoin Core RPC to iterate through all blocks.
    """

    def __init__(self, db_path: str = None):
        self.rpc = None
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', '..', 'data', 'historical_features.db'
        )

        # Caches
        self.utxo_ages: Dict[str, int] = {}  # txid:vout -> block_height created
        self.recent_blocks: deque = deque(maxlen=2016)  # For difficulty calc
        self.block_times: deque = deque(maxlen=11)  # For median time

        # Mining pool identification
        self.pool_tags = {
            b'/AntPool/': 'AntPool',
            b'/F2Pool/': 'F2Pool',
            b'/ViaBTC/': 'ViaBTC',
            b'/Foundry/': 'Foundry',
            b'/Poolin/': 'Poolin',
            b'/BTC.com/': 'BTC.com',
            b'/SlushPool/': 'SlushPool',
            b'/Braiins/': 'Braiins',
            b'/MARA Pool/': 'MARA',
            b'/SBI Crypto/': 'SBI',
            b'/Binance/': 'Binance',
            b'/Luxor/': 'Luxor',
        }

        # Historical price data (would load from file)
        self.price_data: Dict[int, float] = {}

        self._init_db()

    def _init_db(self):
        """Initialize database for historical features."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Main features table
        c.execute('''
            CREATE TABLE IF NOT EXISTS block_features (
                block_height INTEGER,
                scale_blocks INTEGER,
                block_time INTEGER,
                features TEXT,
                PRIMARY KEY (block_height, scale_blocks)
            )
        ''')

        # Progress tracking
        c.execute('''
            CREATE TABLE IF NOT EXISTS scan_progress (
                id INTEGER PRIMARY KEY,
                last_block INTEGER,
                timestamp REAL
            )
        ''')

        # Index for fast queries
        c.execute('CREATE INDEX IF NOT EXISTS idx_block_time ON block_features(block_time)')

        conn.commit()
        conn.close()
        print(f"[DB] Historical database: {self.db_path}")

    def _get_rpc(self):
        """Get RPC connection."""
        if self.rpc is None:
            try:
                from bitcoinrpc.authproxy import AuthServiceProxy
                rpc_user = os.environ.get('BITCOIN_RPC_USER', 'bitcoin')
                rpc_pass = os.environ.get('BITCOIN_RPC_PASS', 'bitcoin')
                rpc_host = os.environ.get('BITCOIN_RPC_HOST', '127.0.0.1')
                rpc_port = os.environ.get('BITCOIN_RPC_PORT', '8332')
                self.rpc = AuthServiceProxy(
                    f"http://{rpc_user}:{rpc_pass}@{rpc_host}:{rpc_port}",
                    timeout=120
                )
            except Exception as e:
                print(f"[ERROR] RPC connection failed: {e}")
                sys.exit(1)
        return self.rpc

    def get_last_scanned_block(self) -> int:
        """Get last successfully scanned block."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT last_block FROM scan_progress ORDER BY id DESC LIMIT 1')
        row = c.fetchone()
        conn.close()
        return row[0] if row else -1

    def save_progress(self, block_height: int):
        """Save scan progress."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('INSERT INTO scan_progress (last_block, timestamp) VALUES (?, ?)',
                  (block_height, time.time()))
        conn.commit()
        conn.close()

    def _identify_pool(self, coinbase_hex: str) -> str:
        """Identify mining pool from coinbase."""
        try:
            coinbase_bytes = bytes.fromhex(coinbase_hex)
            for tag, pool in self.pool_tags.items():
                if tag in coinbase_bytes:
                    return pool
            return "Unknown"
        except:
            return "Unknown"

    def _get_halving_info(self, height: int) -> Tuple[int, float, float]:
        """Get halving epoch and cycle position."""
        epoch = 0
        for h in HALVINGS:
            if height >= h:
                epoch += 1

        # Days since last halving
        if epoch == 0:
            blocks_since = height
            last_halving = 0
        else:
            last_halving = HALVINGS[epoch - 1]
            blocks_since = height - last_halving

        days_since = blocks_since * 10 / 60 / 24  # ~10 min per block

        # Cycle phase (0-1)
        blocks_in_cycle = 210000
        cycle_phase = (height % blocks_in_cycle) / blocks_in_cycle

        return epoch, days_since, cycle_phase

    def scan_block(self, height: int) -> Optional[Dict]:
        """
        Scan a single block and extract features.

        Returns raw data for aggregation.
        """
        rpc = self._get_rpc()

        try:
            # Get block hash and full block data
            block_hash = rpc.getblockhash(height)
            block = rpc.getblock(block_hash, 2)  # verbosity 2 = full tx data

            # Basic block info
            block_time = block['time']
            block_size = block['size']
            block_weight = block['weight']
            tx_count = len(block['tx'])

            # Time since last block
            if height > 0:
                prev_hash = block['previousblockhash']
                prev_block = rpc.getblockheader(prev_hash)
                time_delta = block_time - prev_block['time']
            else:
                time_delta = 0

            # Update median time tracking
            self.block_times.append(block_time)
            median_time = sorted(self.block_times)[len(self.block_times) // 2]
            timestamp_delta = block_time - median_time

            # Difficulty
            difficulty = float(str(block['difficulty']))

            # Coinbase analysis
            coinbase_tx = block['tx'][0]
            coinbase_hex = coinbase_tx['vin'][0].get('coinbase', '')
            pool = self._identify_pool(coinbase_hex)
            coinbase_size = len(coinbase_hex) // 2

            # Calculate subsidy
            epoch, days_since_halving, cycle_phase = self._get_halving_info(height)
            subsidy = 50 / (2 ** epoch)

            # Analyze all transactions
            total_fees = 0.0
            total_inputs = 0
            total_outputs = 0
            segwit_count = 0
            taproot_count = 0
            consolidations = 0
            batches = 0
            op_returns = 0
            ln_opens = 0
            ln_closes = 0
            inscription_count = 0
            inscription_bytes = 0
            coin_days = 0.0
            old_coins = 0.0
            new_coins = 0.0
            total_value = 0.0
            fee_rates = []

            for tx in block['tx'][1:]:  # Skip coinbase
                # Count inputs/outputs
                n_inputs = len(tx.get('vin', []))
                n_outputs = len(tx.get('vout', []))
                total_inputs += n_inputs
                total_outputs += n_outputs

                # Consolidation vs batch detection
                if n_inputs > 3 and n_outputs <= 2:
                    consolidations += 1
                elif n_inputs <= 2 and n_outputs > 5:
                    batches += 1

                # Fee calculation
                vsize = tx.get('vsize', tx.get('size', 1))
                fee = tx.get('fee', 0)
                if isinstance(fee, (int, float)) and fee > 0:
                    total_fees += fee
                    fee_rate = (fee * 1e8) / vsize if vsize > 0 else 0
                    fee_rates.append(fee_rate)

                # Segwit/Taproot detection
                for vout in tx.get('vout', []):
                    script_type = vout.get('scriptPubKey', {}).get('type', '')
                    if script_type in ['witness_v0_keyhash', 'witness_v0_scripthash']:
                        segwit_count += 1
                    elif script_type == 'witness_v1_taproot':
                        taproot_count += 1
                    elif script_type == 'nulldata':
                        op_returns += 1

                    # Lightning channel detection (P2WSH in typical range)
                    if script_type == 'witness_v0_scripthash':
                        value = vout.get('value', 0)
                        if 0.001 <= value <= 0.5:
                            ln_opens += 1

                # Check for inscriptions (witness data patterns)
                if height >= ORDINALS_START:
                    for vin in tx.get('vin', []):
                        witness = vin.get('txinwitness', [])
                        for w in witness:
                            if '0063' in w or '036f7264' in w:
                                inscription_count += 1
                                inscription_bytes += len(w) // 2

                # UTXO age tracking (simplified)
                for vin in tx.get('vin', []):
                    prev_txid = vin.get('txid', '')
                    prev_vout = vin.get('vout', 0)
                    key = f"{prev_txid}:{prev_vout}"

                    if key in self.utxo_ages:
                        created_height = self.utxo_ages[key]
                        age_blocks = height - created_height

                        # Estimate value (simplified)
                        estimated_value = 0.01  # placeholder
                        coin_days += age_blocks * estimated_value / 144
                        total_value += estimated_value

                        if age_blocks > 144 * 365:  # > 1 year
                            old_coins += estimated_value
                        elif age_blocks < 144:  # < 1 day
                            new_coins += estimated_value

                        del self.utxo_ages[key]

                # Track new UTXOs (limit memory usage)
                txid = tx.get('txid', '')
                for i, vout in enumerate(tx.get('vout', [])):
                    if len(self.utxo_ages) < 1000000:  # Cap at 1M
                        self.utxo_ages[f"{txid}:{i}"] = height

            # Calculate ratios
            non_coinbase = tx_count - 1 if tx_count > 1 else 1

            return {
                'height': height,
                'time': block_time,
                'time_delta': time_delta,
                'size': block_size,
                'weight': block_weight,
                'fullness': block_weight / 4000000,
                'tx_count': tx_count,
                'total_fees': total_fees,
                'fee_density': np.mean(fee_rates) if fee_rates else 0,
                'fee_min': min(fee_rates) if fee_rates else 0,
                'fee_max': max(fee_rates) if fee_rates else 0,
                'fee_spread': (max(fee_rates) - min(fee_rates)) if fee_rates else 0,
                'avg_inputs': total_inputs / non_coinbase,
                'avg_outputs': total_outputs / non_coinbase,
                'consolidation_ratio': consolidations / non_coinbase,
                'batch_ratio': batches / non_coinbase,
                'segwit_ratio': segwit_count / max(1, total_outputs),
                'taproot_ratio': taproot_count / max(1, total_outputs),
                'difficulty': difficulty,
                'subsidy': subsidy,
                'pool': pool,
                'empty': 1 if tx_count < 10 else 0,
                'timestamp_delta': timestamp_delta,
                'coinbase_size': coinbase_size,
                'fees_vs_subsidy': total_fees / subsidy if subsidy > 0 else 0,
                'ln_opens': ln_opens,
                'ln_closes': ln_closes,  # Would need to track closures
                'inscription_count': inscription_count,
                'inscription_bytes': inscription_bytes,
                'op_returns': op_returns,
                'coin_days': coin_days,
                'old_coin_ratio': old_coins / total_value if total_value > 0 else 0,
                'new_coin_ratio': new_coins / total_value if total_value > 0 else 0,
                'epoch': epoch,
                'days_since_halving': days_since_halving,
                'cycle_phase': cycle_phase,
            }

        except Exception as e:
            print(f"[ERROR] Block {height}: {e}")
            return None

    def aggregate_features(self, blocks: List[Dict], scale: int) -> HistoricalFeatures:
        """Aggregate block data into feature vector."""
        if not blocks:
            return None

        last_block = blocks[-1]

        # Aggregate over window
        features = HistoricalFeatures(
            block_height=last_block['height'],
            block_time=last_block['time'],
            scale_blocks=scale,
        )

        # Block features (averages/sums over window)
        features.block_time_delta = np.mean([b['time_delta'] for b in blocks])
        features.block_size = int(np.mean([b['size'] for b in blocks]))
        features.block_weight = int(np.mean([b['weight'] for b in blocks]))
        features.block_fullness = np.mean([b['fullness'] for b in blocks])
        features.block_tx_count = int(np.mean([b['tx_count'] for b in blocks]))
        features.block_fees_btc = sum([b['total_fees'] for b in blocks])

        # Fee features
        features.fee_density = np.mean([b['fee_density'] for b in blocks])
        features.fee_min = np.mean([b['fee_min'] for b in blocks])
        features.fee_max = np.mean([b['fee_max'] for b in blocks])
        features.fee_spread = np.mean([b['fee_spread'] for b in blocks])

        # Transaction features
        features.tx_avg_inputs = np.mean([b['avg_inputs'] for b in blocks])
        features.tx_avg_outputs = np.mean([b['avg_outputs'] for b in blocks])
        features.tx_consolidation_ratio = np.mean([b['consolidation_ratio'] for b in blocks])
        features.tx_batch_ratio = np.mean([b['batch_ratio'] for b in blocks])
        features.tx_segwit_ratio = np.mean([b['segwit_ratio'] for b in blocks])
        features.tx_taproot_ratio = np.mean([b['taproot_ratio'] for b in blocks])

        # Network features
        features.difficulty = last_block['difficulty']
        features.subsidy_btc = last_block['subsidy']

        # Estimate hash rate from difficulty
        # hash_rate = difficulty * 2^32 / 600
        features.hash_rate_estimate = features.difficulty * (2**32) / 600 / 1e18

        # Miner features
        features.miner_pool = last_block['pool']
        features.empty_block = sum([b['empty'] for b in blocks])
        features.timestamp_delta = np.mean([b['timestamp_delta'] for b in blocks])
        features.coinbase_size = int(np.mean([b['coinbase_size'] for b in blocks]))
        features.fees_vs_subsidy = np.mean([b['fees_vs_subsidy'] for b in blocks])

        # Lightning
        features.ln_channel_opens = sum([b['ln_opens'] for b in blocks])
        features.ln_channel_closes = sum([b['ln_closes'] for b in blocks])

        # Ordinals
        features.inscription_count = sum([b['inscription_count'] for b in blocks])
        features.inscription_bytes = sum([b['inscription_bytes'] for b in blocks])
        features.op_return_count = sum([b['op_returns'] for b in blocks])

        # UTXO
        features.coin_days_destroyed = sum([b['coin_days'] for b in blocks])
        features.old_coin_ratio = np.mean([b['old_coin_ratio'] for b in blocks])
        features.new_coin_ratio = np.mean([b['new_coin_ratio'] for b in blocks])

        # Derived
        features.halving_epoch = last_block['epoch']
        features.days_since_halving = last_block['days_since_halving']
        features.market_cycle_phase = last_block['cycle_phase']
        features.block_interval_zscore = (features.block_time_delta - 600) / 300

        return features

    def save_features(self, features: HistoricalFeatures):
        """Save features to database."""
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
        except Exception as e:
            print(f"[DB ERROR] {e}")
        finally:
            conn.close()

    def scan(self, start_block: int = 0, end_block: int = -1, batch_size: int = 100):
        """
        Scan blockchain from start to end, extracting features.

        Args:
            start_block: Starting block height (0 = genesis)
            end_block: Ending block (-1 = current tip)
            batch_size: Blocks to process before saving progress
        """
        rpc = self._get_rpc()

        # Get current tip
        if end_block == -1:
            end_block = rpc.getblockcount()

        # Check for resume
        last_scanned = self.get_last_scanned_block()
        if last_scanned >= start_block:
            start_block = last_scanned + 1
            print(f"[RESUME] Continuing from block {start_block}")

        print(f"""
================================================================================
                    HISTORICAL BLOCKCHAIN SCANNER
================================================================================

"The data speaks for itself."
- Jim Simons

SCANNING: Block {start_block:,} to {end_block:,}
TOTAL BLOCKS: {end_block - start_block + 1:,}
SCALES: {BLOCK_SCALES} blocks

FEATURES: 43 dimensions per observation
ESTIMATED TIME: {(end_block - start_block) * 0.5 / 3600:.1f} hours

================================================================================
""")

        # Buffers for each scale
        scale_buffers: Dict[int, List[Dict]] = {s: [] for s in BLOCK_SCALES}

        start_time = time.time()
        blocks_processed = 0

        for height in range(start_block, end_block + 1):
            # Scan block
            block_data = self.scan_block(height)

            if block_data is None:
                continue

            # Add to all scale buffers
            for scale in BLOCK_SCALES:
                scale_buffers[scale].append(block_data)

                # If buffer full, aggregate and save
                if len(scale_buffers[scale]) >= scale:
                    features = self.aggregate_features(scale_buffers[scale], scale)
                    if features:
                        self.save_features(features)
                    scale_buffers[scale] = []

            blocks_processed += 1

            # Progress update
            if blocks_processed % batch_size == 0:
                elapsed = time.time() - start_time
                rate = blocks_processed / elapsed
                remaining = (end_block - height) / rate if rate > 0 else 0

                print(f"[SCAN] Block {height:,} / {end_block:,} | "
                      f"{blocks_processed:,} done | "
                      f"{rate:.1f} blk/s | "
                      f"ETA: {remaining/3600:.1f}h")

                self.save_progress(height)

        # Final progress save
        self.save_progress(end_block)

        elapsed = time.time() - start_time
        print(f"""
================================================================================
                         SCAN COMPLETE
================================================================================

Blocks scanned: {blocks_processed:,}
Time elapsed: {elapsed/3600:.2f} hours
Average rate: {blocks_processed/elapsed:.1f} blocks/second

Database: {self.db_path}
================================================================================
""")


def main():
    parser = argparse.ArgumentParser(description='Historical Blockchain Scanner')
    parser.add_argument('--start', type=int, default=0, help='Starting block')
    parser.add_argument('--end', type=int, default=-1, help='Ending block (-1 = current)')
    parser.add_argument('--batch', type=int, default=100, help='Progress save interval')

    args = parser.parse_args()

    scanner = HistoricalScanner()
    scanner.scan(args.start, args.end, args.batch)


if __name__ == '__main__':
    main()
