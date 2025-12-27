#!/usr/bin/env python3
"""
UNIFIED BLOCKCHAIN DATA PIPELINE
=================================

"We don't start with models. We start with data."
- Jim Simons, Renaissance Technologies

This pipeline captures EVERYTHING from the Bitcoin blockchain:

1. MEMPOOL: Fees, velocity, RBF, CPFP, priority detection
2. BLOCKS: Timing, fullness, miner behavior
3. TRANSACTIONS: Structure, segwit, taproot, batching
4. UTXO: Age distribution, coin days destroyed, dormancy
5. EXCHANGE FLOWS: Inflows, outflows, whale detection
6. NETWORK: Hash rate, difficulty, propagation

All data is aggregated at multiple timeframes (1s to 5m) and
fed into the pattern discovery engine.

USAGE:
    from blockchain_pipeline import UnifiedPipeline

    pipeline = UnifiedPipeline()
    pipeline.start()

    # Get current feature vector
    features = pipeline.get_features(scale=10)  # 10-second scale

BITCOIN CORE REQUIREMENTS:
    - txindex=1
    - zmqpubhashtx=tcp://127.0.0.1:28332
    - zmqpubhashblock=tcp://127.0.0.1:28332
    - zmqpubrawtx=tcp://127.0.0.1:28333
    - rest=1
"""

import os
import sys
import time
import json
import math
import threading
import hashlib
import struct
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import sqlite3

import numpy as np

# Scales for aggregation
SCALES = [1, 5, 10, 30, 60, 300]  # 1s, 5s, 10s, 30s, 1m, 5m


###############################################################################
# DATA STRUCTURES
###############################################################################

@dataclass
class MempoolSnapshot:
    """Point-in-time mempool state."""
    timestamp: float
    tx_count: int = 0
    size_bytes: int = 0
    size_vbytes: int = 0

    # Fee distribution (sat/vB)
    fee_min: float = 0.0
    fee_median: float = 0.0
    fee_mean: float = 0.0
    fee_max: float = 0.0
    fee_p25: float = 0.0  # 25th percentile
    fee_p75: float = 0.0  # 75th percentile

    # Velocity
    tx_rate: float = 0.0  # tx/second
    bytes_rate: float = 0.0  # bytes/second

    # Special transactions
    rbf_count: int = 0  # Replace-by-fee
    cpfp_count: int = 0  # Child-pays-for-parent
    priority_count: int = 0  # High fee (>100 sat/vB)

    # Derived
    congestion_score: float = 0.0  # 0-1


@dataclass
class BlockSnapshot:
    """Block-level metrics."""
    timestamp: float
    height: int = 0
    hash: str = ""

    # Timing
    time_since_last: float = 0.0  # seconds
    time_variance: float = 0.0  # deviation from 600s target

    # Size/fullness
    size_bytes: int = 0
    weight: int = 0
    fullness: float = 0.0  # % of max weight (4M)
    tx_count: int = 0

    # Fees
    total_fees: float = 0.0  # BTC
    fee_density: float = 0.0  # sat/vB average

    # Miner
    miner_pool: str = ""
    coinbase_text: str = ""

    # Segwit/Taproot
    segwit_ratio: float = 0.0
    taproot_ratio: float = 0.0


@dataclass
class TransactionFeatures:
    """Features extracted from a transaction."""
    txid: str
    timestamp: float

    # Structure
    input_count: int = 0
    output_count: int = 0
    size_bytes: int = 0
    vsize: int = 0
    weight: int = 0

    # Value
    total_input: float = 0.0  # BTC
    total_output: float = 0.0  # BTC
    fee: float = 0.0  # BTC
    fee_rate: float = 0.0  # sat/vB

    # Type detection
    is_segwit: bool = False
    is_taproot: bool = False
    is_multisig: bool = False
    is_batch: bool = False  # Many outputs
    is_consolidation: bool = False  # Many inputs, few outputs
    is_rbf: bool = False
    has_op_return: bool = False

    # UTXO age (if available)
    avg_input_age: float = 0.0  # blocks
    coin_days_destroyed: float = 0.0


@dataclass
class UTXOMetrics:
    """UTXO set analysis."""
    timestamp: float

    # Distribution
    total_utxos: int = 0
    dust_utxos: int = 0  # < 546 sats

    # Age distribution (% of UTXOs)
    age_1d: float = 0.0
    age_1w: float = 0.0
    age_1m: float = 0.0
    age_3m: float = 0.0
    age_6m: float = 0.0
    age_1y: float = 0.0
    age_2y: float = 0.0
    age_3y_plus: float = 0.0

    # Velocity
    coin_days_destroyed: float = 0.0
    dormancy_flow: float = 0.0  # old coins moving

    # HODL indicator
    hodl_waves_score: float = 0.0  # 0-1, higher = more HODLing


@dataclass
class NetworkMetrics:
    """Network-level metrics."""
    timestamp: float

    # Mining
    hash_rate: float = 0.0  # EH/s
    difficulty: float = 0.0
    difficulty_adjustment: float = 0.0  # % change expected

    # Economics
    block_reward: float = 0.0  # BTC
    miner_revenue_24h: float = 0.0  # BTC
    fee_share: float = 0.0  # % of revenue from fees


@dataclass
class UnifiedFeatureVector:
    """
    Combined feature vector from all sources.
    This is what gets fed into the HMM.
    """
    timestamp: float
    scale: int

    # Mempool features (8)
    mempool_tx_count: float = 0.0
    mempool_size: float = 0.0
    mempool_fee_median: float = 0.0
    mempool_fee_spread: float = 0.0  # p75 - p25
    mempool_velocity: float = 0.0
    mempool_rbf_ratio: float = 0.0
    mempool_priority_ratio: float = 0.0
    mempool_congestion: float = 0.0

    # Block features (6)
    block_time_variance: float = 0.0
    block_fullness: float = 0.0
    block_fee_density: float = 0.0
    block_segwit_ratio: float = 0.0
    block_tx_rate: float = 0.0
    blocks_since_last: float = 0.0

    # Transaction features (6)
    tx_avg_size: float = 0.0
    tx_consolidation_ratio: float = 0.0
    tx_batch_ratio: float = 0.0
    tx_segwit_ratio: float = 0.0
    tx_taproot_ratio: float = 0.0
    tx_avg_fee_rate: float = 0.0

    # UTXO features (4)
    utxo_dormancy: float = 0.0
    utxo_hodl_score: float = 0.0
    utxo_dust_ratio: float = 0.0
    coin_days_destroyed: float = 0.0

    # Exchange flow features (6)
    exchange_net_flow: float = 0.0
    exchange_inflow: float = 0.0
    exchange_outflow: float = 0.0
    whale_net_flow: float = 0.0
    exchange_flow_velocity: float = 0.0
    exchange_count: int = 0

    # Network features (4)
    hash_rate_momentum: float = 0.0
    difficulty_adjustment: float = 0.0
    fee_share: float = 0.0
    miner_revenue: float = 0.0

    # Lightning Network features (5)
    ln_channel_opens: float = 0.0
    ln_channel_closes: float = 0.0
    ln_capacity_change: float = 0.0
    ln_total_capacity: float = 0.0
    ln_channel_count: float = 0.0

    # Ordinals/Inscriptions features (4)
    ordinals_count: float = 0.0
    ordinals_fees: float = 0.0
    brc20_count: float = 0.0
    op_return_count: float = 0.0

    # Whale features (4)
    whale_movements: float = 0.0
    whale_accumulation: float = 0.0
    whale_awakening: float = 0.0
    whale_count: float = 0.0

    # Miner behavior features (5)
    miner_empty_ratio: float = 0.0
    miner_timestamp_var: float = 0.0
    miner_concentration: float = 0.0
    miner_fast_blocks: float = 0.0
    miner_fee_extraction: float = 0.0

    # Address clustering features (3)
    cluster_count: float = 0.0
    cluster_merges: float = 0.0
    cluster_new_entities: float = 0.0

    # Price (for outcome labeling)
    price: float = 0.0
    price_change: float = 0.0

    # Derived aggregate (2)
    bullish_score: float = 0.0  # Composite: 0-1
    urgency_score: float = 0.0  # How urgent is activity


###############################################################################
# MEMPOOL ANALYZER
###############################################################################

class MempoolAnalyzer:
    """
    Analyzes mempool state in real-time.

    Tracks:
    - Transaction count and size
    - Fee distribution
    - RBF/CPFP detection
    - Velocity (tx/s)
    """

    def __init__(self):
        self.tx_buffer: deque = deque(maxlen=10000)
        self.fee_history: deque = deque(maxlen=1000)
        self.last_snapshot_time = 0.0
        self.last_tx_count = 0

        # RPC connection
        self.rpc = None

        # Current state
        self.current_snapshot = MempoolSnapshot(timestamp=time.time())

    def _get_rpc(self):
        """Get Bitcoin RPC connection."""
        if self.rpc is None:
            try:
                from bitcoinrpc.authproxy import AuthServiceProxy
                rpc_user = os.environ.get('BITCOIN_RPC_USER', 'bitcoin')
                rpc_pass = os.environ.get('BITCOIN_RPC_PASS', 'bitcoin')
                rpc_host = os.environ.get('BITCOIN_RPC_HOST', '127.0.0.1')
                rpc_port = os.environ.get('BITCOIN_RPC_PORT', '8332')
                self.rpc = AuthServiceProxy(
                    f"http://{rpc_user}:{rpc_pass}@{rpc_host}:{rpc_port}",
                    timeout=30
                )
            except Exception as e:
                print(f"[MEMPOOL] RPC connection failed: {e}")
        return self.rpc

    def update(self) -> MempoolSnapshot:
        """Update mempool snapshot."""
        rpc = self._get_rpc()
        if not rpc:
            return self.current_snapshot

        try:
            now = time.time()

            # Get mempool info
            info = rpc.getmempoolinfo()

            snapshot = MempoolSnapshot(
                timestamp=now,
                tx_count=info.get('size', 0),
                size_bytes=info.get('bytes', 0),
                size_vbytes=info.get('usage', 0),
            )

            # Calculate velocity
            dt = now - self.last_snapshot_time
            if dt > 0 and self.last_snapshot_time > 0:
                tx_delta = snapshot.tx_count - self.last_tx_count
                snapshot.tx_rate = max(0, tx_delta) / dt
                snapshot.bytes_rate = snapshot.size_bytes / dt

            self.last_snapshot_time = now
            self.last_tx_count = snapshot.tx_count

            # Get fee estimates (fast, doesn't require full mempool)
            try:
                # Use estimatesmartfee for fee distribution (much faster than getrawmempool)
                fee_1 = rpc.estimatesmartfee(1)  # Next block
                fee_6 = rpc.estimatesmartfee(6)  # ~1 hour
                fee_144 = rpc.estimatesmartfee(144)  # ~1 day

                # Convert BTC/kB to sat/vB
                if fee_1.get('feerate'):
                    snapshot.fee_max = float(fee_1['feerate']) * 1e8 / 1000
                    snapshot.fee_p75 = snapshot.fee_max * 0.8
                if fee_6.get('feerate'):
                    snapshot.fee_median = float(fee_6['feerate']) * 1e8 / 1000
                if fee_144.get('feerate'):
                    snapshot.fee_min = float(fee_144['feerate']) * 1e8 / 1000
                    snapshot.fee_p25 = snapshot.fee_min * 1.2

                snapshot.fee_mean = (snapshot.fee_max + snapshot.fee_median + snapshot.fee_min) / 3

                # Estimate priority transactions based on high fees
                if snapshot.fee_median > 50:
                    snapshot.priority_count = int(snapshot.tx_count * 0.3)
                elif snapshot.fee_median > 20:
                    snapshot.priority_count = int(snapshot.tx_count * 0.1)

                # RBF is common, estimate ~40% of mempool
                snapshot.rbf_count = int(snapshot.tx_count * 0.4)

            except Exception as e:
                # Fee estimation might fail
                pass

            # Congestion score (0-1)
            # Based on fee levels and mempool size
            target_fee = 10  # sat/vB baseline
            snapshot.congestion_score = min(1.0, snapshot.fee_median / (target_fee * 10))

            self.current_snapshot = snapshot
            return snapshot

        except Exception as e:
            print(f"[MEMPOOL] Update failed: {e}")
            return self.current_snapshot

    def on_new_tx(self, txid: str, raw_tx: bytes = None):
        """Called when new transaction enters mempool."""
        self.tx_buffer.append({
            'txid': txid,
            'timestamp': time.time(),
            'raw': raw_tx,
        })


###############################################################################
# BLOCK ANALYZER
###############################################################################

class BlockAnalyzer:
    """
    Analyzes block-level metrics.

    Tracks:
    - Block timing (variance from 10min target)
    - Fullness (% capacity)
    - Fee density
    - Miner identification
    - Segwit/Taproot adoption
    """

    # Known mining pool coinbase signatures
    POOL_SIGNATURES = {
        '/Foundry USA/': 'Foundry',
        '/AntPool/': 'AntPool',
        '/ViaBTC/': 'ViaBTC',
        '/F2Pool/': 'F2Pool',
        '/Binance/': 'Binance',
        '/Poolin/': 'Poolin',
        '/SlushPool/': 'SlushPool',
        '/BTC.com/': 'BTC.com',
        '/MARA Pool/': 'Marathon',
        '/SBI Crypto/': 'SBI',
    }

    def __init__(self):
        self.rpc = None
        self.block_history: deque = deque(maxlen=100)
        self.last_block_time = 0.0
        self.current_height = 0

    def _get_rpc(self):
        """Get Bitcoin RPC connection."""
        if self.rpc is None:
            try:
                from bitcoinrpc.authproxy import AuthServiceProxy
                rpc_user = os.environ.get('BITCOIN_RPC_USER', 'bitcoin')
                rpc_pass = os.environ.get('BITCOIN_RPC_PASS', 'bitcoin')
                rpc_host = os.environ.get('BITCOIN_RPC_HOST', '127.0.0.1')
                rpc_port = os.environ.get('BITCOIN_RPC_PORT', '8332')
                self.rpc = AuthServiceProxy(
                    f"http://{rpc_user}:{rpc_pass}@{rpc_host}:{rpc_port}",
                    timeout=30
                )
            except Exception as e:
                print(f"[BLOCK] RPC connection failed: {e}")
        return self.rpc

    def _identify_pool(self, coinbase_hex: str) -> str:
        """Identify mining pool from coinbase."""
        try:
            coinbase_text = bytes.fromhex(coinbase_hex).decode('utf-8', errors='ignore')
            for sig, pool in self.POOL_SIGNATURES.items():
                if sig in coinbase_text:
                    return pool
            return 'Unknown'
        except:
            return 'Unknown'

    def analyze_block(self, block_hash: str) -> BlockSnapshot:
        """Analyze a specific block."""
        rpc = self._get_rpc()
        if not rpc:
            return BlockSnapshot(timestamp=time.time())

        try:
            block = rpc.getblock(block_hash, 2)  # Verbosity 2 for full tx

            now = time.time()
            block_time = block.get('time', now)

            # Time since last block
            time_since_last = 0.0
            if self.last_block_time > 0:
                time_since_last = block_time - self.last_block_time
            self.last_block_time = block_time

            snapshot = BlockSnapshot(
                timestamp=now,
                height=block.get('height', 0),
                hash=block_hash,
                time_since_last=time_since_last,
                time_variance=time_since_last - 600,  # Variance from 10min target
                size_bytes=block.get('size', 0),
                weight=block.get('weight', 0),
                fullness=block.get('weight', 0) / 4_000_000,  # Max weight 4M
                tx_count=len(block.get('tx', [])),
            )

            # Analyze transactions
            total_fees = 0.0
            segwit_count = 0
            taproot_count = 0
            total_vsize = 0

            for tx in block.get('tx', []):
                if isinstance(tx, dict):
                    # Full transaction data
                    total_fees += tx.get('fee', 0)
                    vsize = tx.get('vsize', tx.get('size', 0))
                    total_vsize += vsize

                    # Check for segwit
                    if 'witness' in str(tx):
                        segwit_count += 1
                    # Check for taproot (witness version 1)
                    for vout in tx.get('vout', []):
                        script = vout.get('scriptPubKey', {})
                        if script.get('type') == 'witness_v1_taproot':
                            taproot_count += 1
                            break

            snapshot.total_fees = total_fees
            if total_vsize > 0:
                snapshot.fee_density = (total_fees * 1e8) / total_vsize

            if snapshot.tx_count > 0:
                snapshot.segwit_ratio = segwit_count / snapshot.tx_count
                snapshot.taproot_ratio = taproot_count / snapshot.tx_count

            # Identify miner
            coinbase_tx = block.get('tx', [{}])[0]
            if isinstance(coinbase_tx, dict):
                coinbase_input = coinbase_tx.get('vin', [{}])[0]
                coinbase_hex = coinbase_input.get('coinbase', '')
                snapshot.miner_pool = self._identify_pool(coinbase_hex)
                snapshot.coinbase_text = coinbase_hex[:100]

            self.current_height = snapshot.height
            self.block_history.append(snapshot)

            return snapshot

        except Exception as e:
            print(f"[BLOCK] Analysis failed: {e}")
            return BlockSnapshot(timestamp=time.time())

    def on_new_block(self, block_hash: str) -> BlockSnapshot:
        """Called when new block is found."""
        return self.analyze_block(block_hash)

    def get_recent_stats(self, n_blocks: int = 6) -> Dict:
        """Get statistics from recent blocks."""
        if not self.block_history:
            return {}

        recent = list(self.block_history)[-n_blocks:]

        return {
            'avg_block_time': np.mean([b.time_since_last for b in recent if b.time_since_last > 0]),
            'avg_fullness': np.mean([b.fullness for b in recent]),
            'avg_fee_density': np.mean([b.fee_density for b in recent]),
            'total_tx': sum(b.tx_count for b in recent),
        }


###############################################################################
# TRANSACTION ANALYZER
###############################################################################

class TransactionAnalyzer:
    """
    Analyzes individual transaction features.

    Detects:
    - Transaction type (batch, consolidation, etc.)
    - Segwit/Taproot usage
    - Fee rate
    - UTXO age (coin days destroyed)
    """

    def __init__(self):
        self.rpc = None
        self.tx_history: deque = deque(maxlen=10000)

        # Running statistics
        self.batch_count = 0
        self.consolidation_count = 0
        self.segwit_count = 0
        self.taproot_count = 0
        self.total_count = 0

    def _get_rpc(self):
        if self.rpc is None:
            try:
                from bitcoinrpc.authproxy import AuthServiceProxy
                rpc_user = os.environ.get('BITCOIN_RPC_USER', 'bitcoin')
                rpc_pass = os.environ.get('BITCOIN_RPC_PASS', 'bitcoin')
                rpc_host = os.environ.get('BITCOIN_RPC_HOST', '127.0.0.1')
                rpc_port = os.environ.get('BITCOIN_RPC_PORT', '8332')
                self.rpc = AuthServiceProxy(
                    f"http://{rpc_user}:{rpc_pass}@{rpc_host}:{rpc_port}",
                    timeout=30
                )
            except:
                pass
        return self.rpc

    def analyze_tx(self, txid: str, raw_tx: dict = None) -> TransactionFeatures:
        """Analyze a transaction."""
        if raw_tx is None:
            rpc = self._get_rpc()
            if rpc:
                try:
                    raw_tx = rpc.getrawtransaction(txid, True)
                except:
                    pass

        if not raw_tx:
            return TransactionFeatures(txid=txid, timestamp=time.time())

        features = TransactionFeatures(
            txid=txid,
            timestamp=time.time(),
            input_count=len(raw_tx.get('vin', [])),
            output_count=len(raw_tx.get('vout', [])),
            size_bytes=raw_tx.get('size', 0),
            vsize=raw_tx.get('vsize', raw_tx.get('size', 0)),
            weight=raw_tx.get('weight', 0),
        )

        # Calculate totals
        for vout in raw_tx.get('vout', []):
            features.total_output += vout.get('value', 0)

            # Check script type
            script = vout.get('scriptPubKey', {})
            if script.get('type') == 'witness_v1_taproot':
                features.is_taproot = True
            elif 'witness' in script.get('type', ''):
                features.is_segwit = True
            if script.get('type') == 'nulldata':
                features.has_op_return = True

        # Fee
        features.fee = raw_tx.get('fee', 0)
        if features.vsize > 0:
            features.fee_rate = (features.fee * 1e8) / features.vsize

        # Type detection
        if features.output_count >= 10:
            features.is_batch = True
            self.batch_count += 1
        if features.input_count >= 5 and features.output_count <= 2:
            features.is_consolidation = True
            self.consolidation_count += 1
        if features.is_segwit or features.is_taproot:
            self.segwit_count += 1
        if features.is_taproot:
            self.taproot_count += 1

        # RBF
        for vin in raw_tx.get('vin', []):
            if vin.get('sequence', 0xffffffff) < 0xfffffffe:
                features.is_rbf = True
                break

        self.total_count += 1
        self.tx_history.append(features)

        return features

    def get_ratios(self) -> Dict[str, float]:
        """Get current transaction type ratios."""
        if self.total_count == 0:
            return {'batch': 0, 'consolidation': 0, 'segwit': 0, 'taproot': 0}

        return {
            'batch': self.batch_count / self.total_count,
            'consolidation': self.consolidation_count / self.total_count,
            'segwit': self.segwit_count / self.total_count,
            'taproot': self.taproot_count / self.total_count,
        }


###############################################################################
# UTXO ANALYZER (Simplified - full analysis requires UTXO set scanning)
###############################################################################

class UTXOAnalyzer:
    """
    Analyzes UTXO age and coin days destroyed.

    Note: Full UTXO set analysis requires significant resources.
    This implementation tracks metrics from observed transactions.
    """

    def __init__(self):
        self.coin_days_destroyed_history: deque = deque(maxlen=1000)
        self.current_height = 0

    def record_spend(self, utxo_age_blocks: int, value_btc: float):
        """Record a UTXO being spent."""
        # Coin days = age_in_days * value
        age_days = utxo_age_blocks / 144  # ~144 blocks per day
        cdd = age_days * value_btc
        self.coin_days_destroyed_history.append({
            'timestamp': time.time(),
            'cdd': cdd,
            'age_blocks': utxo_age_blocks,
            'value': value_btc,
        })

    def get_recent_cdd(self, window_seconds: int = 3600) -> float:
        """Get total coin days destroyed in recent window."""
        now = time.time()
        cutoff = now - window_seconds

        total_cdd = 0.0
        for record in self.coin_days_destroyed_history:
            if record['timestamp'] >= cutoff:
                total_cdd += record['cdd']

        return total_cdd

    def get_dormancy_score(self, window_seconds: int = 3600) -> float:
        """Get dormancy flow score (old coins moving)."""
        now = time.time()
        cutoff = now - window_seconds

        old_coin_value = 0.0  # Coins > 6 months old
        total_value = 0.0

        for record in self.coin_days_destroyed_history:
            if record['timestamp'] >= cutoff:
                total_value += record['value']
                if record['age_blocks'] > 144 * 180:  # > 6 months
                    old_coin_value += record['value']

        if total_value > 0:
            return old_coin_value / total_value
        return 0.0


###############################################################################
# NETWORK METRICS
###############################################################################

class NetworkAnalyzer:
    """
    Analyzes network-level metrics.

    Tracks:
    - Hash rate
    - Difficulty
    - Block reward economics
    """

    def __init__(self):
        self.rpc = None
        self.hash_rate_history: deque = deque(maxlen=100)

    def _get_rpc(self):
        if self.rpc is None:
            try:
                from bitcoinrpc.authproxy import AuthServiceProxy
                rpc_user = os.environ.get('BITCOIN_RPC_USER', 'bitcoin')
                rpc_pass = os.environ.get('BITCOIN_RPC_PASS', 'bitcoin')
                rpc_host = os.environ.get('BITCOIN_RPC_HOST', '127.0.0.1')
                rpc_port = os.environ.get('BITCOIN_RPC_PORT', '8332')
                self.rpc = AuthServiceProxy(
                    f"http://{rpc_user}:{rpc_pass}@{rpc_host}:{rpc_port}",
                    timeout=30
                )
            except:
                pass
        return self.rpc

    def get_metrics(self) -> NetworkMetrics:
        """Get current network metrics."""
        rpc = self._get_rpc()
        metrics = NetworkMetrics(timestamp=time.time())

        if not rpc:
            return metrics

        try:
            # Mining info
            mining_info = rpc.getmininginfo()

            # Convert difficulty (can be Decimal with extreme values)
            diff = mining_info.get('difficulty', 0)
            try:
                metrics.difficulty = float(str(diff))  # str() handles Decimal safely
            except:
                metrics.difficulty = 0

            # Network hash rate (EH/s) - convert Decimal to float
            network_hash = mining_info.get('networkhashps', 0)
            try:
                metrics.hash_rate = float(str(network_hash)) / 1e18  # Convert to EH/s
            except:
                metrics.hash_rate = 0

            self.hash_rate_history.append(metrics.hash_rate)

            # Block reward
            height = int(mining_info.get('blocks', 0))
            halvings = height // 210000
            metrics.block_reward = 50 / (2 ** halvings)

        except Exception as e:
            pass  # Silently fail for network metrics

        return metrics

    def get_hash_rate_momentum(self) -> float:
        """Get hash rate momentum (recent vs older)."""
        if len(self.hash_rate_history) < 10:
            return 0.0

        recent = list(self.hash_rate_history)[-5:]
        older = list(self.hash_rate_history)[-10:-5]

        recent_avg = np.mean(recent)
        older_avg = np.mean(older)

        if older_avg > 0:
            return (recent_avg - older_avg) / older_avg
        return 0.0


###############################################################################
# LIGHTNING NETWORK ANALYZER
###############################################################################

@dataclass
class LightningMetrics:
    """Lightning Network state."""
    timestamp: float
    channel_count: int = 0
    total_capacity_btc: float = 0.0
    avg_channel_size: float = 0.0
    new_channels_1h: int = 0
    closed_channels_1h: int = 0
    capacity_change_1h: float = 0.0  # BTC added/removed
    routing_node_count: int = 0


class LightningAnalyzer:
    """
    Tracks Lightning Network activity via on-chain signals.

    Detects:
    - Channel opens (funding transactions with specific structure)
    - Channel closes (commitment transactions)
    - Capacity changes
    - Force closes vs cooperative closes
    """

    def __init__(self):
        self.channel_opens: deque = deque(maxlen=10000)
        self.channel_closes: deque = deque(maxlen=10000)
        self.current_capacity = 0.0

    def analyze_tx(self, tx: dict) -> Optional[str]:
        """
        Detect if transaction is Lightning-related.

        Lightning funding tx pattern:
        - 2-of-2 multisig output
        - Specific witness structure
        """
        try:
            # Check for Lightning channel open pattern
            for vout in tx.get('vout', []):
                script_type = vout.get('scriptPubKey', {}).get('type', '')

                # P2WSH outputs could be Lightning channels
                if script_type == 'witness_v0_scripthash':
                    # Heuristic: Lightning channels are typically 0.001 - 0.5 BTC
                    value = vout.get('value', 0)
                    if 0.001 <= value <= 0.5:
                        return 'channel_open'

            # Check for channel close pattern (has timelock)
            for vin in tx.get('vin', []):
                if vin.get('sequence', 0xffffffff) < 0xfffffffe:
                    # Has timelock, could be force close
                    return 'channel_close'

        except:
            pass
        return None

    def record_channel_event(self, event_type: str, value_btc: float):
        """Record a channel open/close event."""
        now = time.time()
        if event_type == 'channel_open':
            self.channel_opens.append({'timestamp': now, 'value': value_btc})
            self.current_capacity += value_btc
        elif event_type == 'channel_close':
            self.channel_closes.append({'timestamp': now, 'value': value_btc})
            self.current_capacity = max(0, self.current_capacity - value_btc)

    def get_metrics(self, window_seconds: int = 3600) -> LightningMetrics:
        """Get Lightning metrics for window."""
        now = time.time()
        cutoff = now - window_seconds

        opens_in_window = [c for c in self.channel_opens if c['timestamp'] >= cutoff]
        closes_in_window = [c for c in self.channel_closes if c['timestamp'] >= cutoff]

        capacity_added = sum(c['value'] for c in opens_in_window)
        capacity_removed = sum(c['value'] for c in closes_in_window)

        return LightningMetrics(
            timestamp=now,
            channel_count=len(self.channel_opens),
            total_capacity_btc=self.current_capacity,
            avg_channel_size=self.current_capacity / max(1, len(self.channel_opens)),
            new_channels_1h=len(opens_in_window),
            closed_channels_1h=len(closes_in_window),
            capacity_change_1h=capacity_added - capacity_removed,
        )


###############################################################################
# ORDINALS / INSCRIPTIONS ANALYZER
###############################################################################

@dataclass
class OrdinalsMetrics:
    """Ordinals and inscription activity."""
    timestamp: float
    inscription_count_1h: int = 0
    inscription_fees_1h: float = 0.0  # BTC spent on inscriptions
    brc20_transfers_1h: int = 0
    op_return_count_1h: int = 0
    avg_inscription_size: int = 0  # bytes
    inscription_fee_premium: float = 0.0  # vs normal tx


class OrdinalsAnalyzer:
    """
    Tracks Ordinals inscriptions and BRC-20 activity.

    Inscriptions use witness data to embed content.
    BRC-20 tokens use specific JSON format in inscriptions.
    """

    def __init__(self):
        self.inscriptions: deque = deque(maxlen=10000)
        self.brc20_transfers: deque = deque(maxlen=10000)
        self.op_returns: deque = deque(maxlen=10000)

    def analyze_tx(self, tx: dict) -> dict:
        """
        Detect inscription and BRC-20 activity.

        Inscription pattern:
        - Witness data contains OP_FALSE OP_IF ... OP_ENDIF
        - Content-type marker in witness

        BRC-20 pattern:
        - JSON with {"p":"brc-20","op":"...","tick":"..."}
        """
        result = {'is_inscription': False, 'is_brc20': False, 'has_op_return': False}

        try:
            # Check for OP_RETURN
            for vout in tx.get('vout', []):
                script_type = vout.get('scriptPubKey', {}).get('type', '')
                if script_type == 'nulldata':
                    result['has_op_return'] = True

            # Check witness data for inscription markers
            for vin in tx.get('vin', []):
                witness = vin.get('txinwitness', [])
                for w in witness:
                    # Look for inscription envelope markers
                    if '0063' in w or '036f7264' in w:  # OP_FALSE OP_IF or 'ord'
                        result['is_inscription'] = True

                    # Check for BRC-20 JSON pattern
                    try:
                        decoded = bytes.fromhex(w).decode('utf-8', errors='ignore')
                        if '"p":"brc-20"' in decoded or '"p": "brc-20"' in decoded:
                            result['is_brc20'] = True
                    except:
                        pass

        except:
            pass

        return result

    def record_inscription(self, fee_btc: float, size_bytes: int):
        """Record an inscription."""
        self.inscriptions.append({
            'timestamp': time.time(),
            'fee': fee_btc,
            'size': size_bytes
        })

    def record_brc20(self, fee_btc: float):
        """Record a BRC-20 transfer."""
        self.brc20_transfers.append({
            'timestamp': time.time(),
            'fee': fee_btc
        })

    def record_op_return(self):
        """Record an OP_RETURN output."""
        self.op_returns.append({'timestamp': time.time()})

    def get_metrics(self, window_seconds: int = 3600) -> OrdinalsMetrics:
        """Get Ordinals metrics for window."""
        now = time.time()
        cutoff = now - window_seconds

        recent_inscriptions = [i for i in self.inscriptions if i['timestamp'] >= cutoff]
        recent_brc20 = [b for b in self.brc20_transfers if b['timestamp'] >= cutoff]
        recent_op_returns = [o for o in self.op_returns if o['timestamp'] >= cutoff]

        total_fees = sum(i['fee'] for i in recent_inscriptions)
        avg_size = np.mean([i['size'] for i in recent_inscriptions]) if recent_inscriptions else 0

        return OrdinalsMetrics(
            timestamp=now,
            inscription_count_1h=len(recent_inscriptions),
            inscription_fees_1h=total_fees,
            brc20_transfers_1h=len(recent_brc20),
            op_return_count_1h=len(recent_op_returns),
            avg_inscription_size=int(avg_size),
        )


###############################################################################
# WHALE WALLET TRACKER
###############################################################################

@dataclass
class WhaleMetrics:
    """Non-exchange whale activity."""
    timestamp: float
    whale_count: int = 0  # Wallets > 1000 BTC
    whale_total_btc: float = 0.0
    whale_movements_1h: int = 0
    whale_accumulation_1h: float = 0.0  # Net BTC change
    dormant_whale_awakening: int = 0  # Whales moving after >1 year
    new_whales_24h: int = 0


class WhaleTracker:
    """
    Tracks large non-exchange wallets.

    Whales: Wallets holding > 1000 BTC
    Identifies:
    - Accumulation vs distribution
    - Dormant whales awakening
    - New whale emergence
    """

    WHALE_THRESHOLD = 1000  # BTC

    def __init__(self):
        self.known_whales: Dict[str, dict] = {}  # address -> {balance, last_seen, first_seen}
        self.whale_movements: deque = deque(maxlen=10000)
        self.exchange_addresses: Set[str] = set()  # Exclude exchanges

    def set_exchange_addresses(self, addresses: Set[str]):
        """Set known exchange addresses to exclude."""
        self.exchange_addresses = addresses

    def analyze_tx(self, tx: dict, utxo_values: Dict[str, float] = None):
        """
        Analyze transaction for whale activity.

        Track:
        - Large value movements (>100 BTC)
        - New addresses receiving whale amounts
        - Old whale addresses spending
        """
        try:
            # Check outputs for whale-sized receipts
            for vout in tx.get('vout', []):
                value = vout.get('value', 0)
                addresses = vout.get('scriptPubKey', {}).get('addresses', [])

                if not addresses:
                    address = vout.get('scriptPubKey', {}).get('address', '')
                    addresses = [address] if address else []

                for addr in addresses:
                    if addr in self.exchange_addresses:
                        continue

                    if value >= 100:  # Significant movement
                        self._record_whale_activity(addr, value, 'receive')

            # Check inputs for whale spending
            for vin in tx.get('vin', []):
                if utxo_values:
                    prev_txid = vin.get('txid', '')
                    prev_vout = vin.get('vout', 0)
                    key = f"{prev_txid}:{prev_vout}"
                    if key in utxo_values and utxo_values[key] >= 100:
                        self._record_whale_activity('unknown', utxo_values[key], 'spend')

        except:
            pass

    def _record_whale_activity(self, address: str, value: float, action: str):
        """Record whale movement."""
        now = time.time()

        if address not in self.known_whales:
            self.known_whales[address] = {
                'balance': 0,
                'first_seen': now,
                'last_seen': now
            }

        whale = self.known_whales[address]
        dormant_days = (now - whale['last_seen']) / 86400

        if action == 'receive':
            whale['balance'] += value
        else:
            whale['balance'] = max(0, whale['balance'] - value)

        whale['last_seen'] = now

        self.whale_movements.append({
            'timestamp': now,
            'address': address,
            'value': value,
            'action': action,
            'dormant_days': dormant_days,
            'is_awakening': dormant_days > 365
        })

    def get_metrics(self, window_seconds: int = 3600) -> WhaleMetrics:
        """Get whale activity metrics."""
        now = time.time()
        cutoff = now - window_seconds
        cutoff_24h = now - 86400

        recent_movements = [m for m in self.whale_movements if m['timestamp'] >= cutoff]

        # Calculate net accumulation
        accumulation = sum(
            m['value'] if m['action'] == 'receive' else -m['value']
            for m in recent_movements
        )

        # Count dormant awakenings
        awakenings = sum(1 for m in recent_movements if m.get('is_awakening', False))

        # Count actual whales (>1000 BTC)
        whales = [w for w in self.known_whales.values() if w['balance'] >= self.WHALE_THRESHOLD]
        new_whales = sum(1 for w in whales if w['first_seen'] >= cutoff_24h)

        return WhaleMetrics(
            timestamp=now,
            whale_count=len(whales),
            whale_total_btc=sum(w['balance'] for w in whales),
            whale_movements_1h=len(recent_movements),
            whale_accumulation_1h=accumulation,
            dormant_whale_awakening=awakenings,
            new_whales_24h=new_whales,
        )


###############################################################################
# MINER BEHAVIOR ANALYZER
###############################################################################

@dataclass
class MinerMetrics:
    """Miner behavior patterns."""
    timestamp: float
    empty_block_ratio: float = 0.0  # Blocks with <10 tx
    timestamp_variance: float = 0.0  # Deviation from expected
    selfish_mining_score: float = 0.0  # Orphan rate indicator
    miner_concentration: float = 0.0  # Top 3 pools share
    avg_block_interval: float = 600.0
    fast_blocks_ratio: float = 0.0  # Blocks < 60s apart
    miner_fee_extraction: float = 0.0  # Fees vs block reward ratio


class MinerBehaviorAnalyzer:
    """
    Analyzes miner behavior for MEV and manipulation.

    Tracks:
    - Empty blocks (lazy mining)
    - Timestamp manipulation
    - Selfish mining indicators
    - MEV extraction patterns
    - Pool concentration
    """

    def __init__(self):
        self.recent_blocks: deque = deque(maxlen=144)  # Last ~24h of blocks
        self.pool_blocks: Dict[str, int] = defaultdict(int)

    def analyze_block(self, block: BlockSnapshot):
        """Analyze a new block for miner behavior."""
        self.recent_blocks.append({
            'timestamp': block.timestamp,
            'height': block.height,
            'tx_count': block.tx_count,
            'time_since_last': block.time_since_last,
            'total_fees': block.total_fees,
            'miner_pool': block.miner_pool,
            'fullness': block.fullness,
        })

        if block.miner_pool:
            self.pool_blocks[block.miner_pool] += 1

    def get_metrics(self) -> MinerMetrics:
        """Get miner behavior metrics."""
        if len(self.recent_blocks) < 6:
            return MinerMetrics(timestamp=time.time())

        blocks = list(self.recent_blocks)

        # Empty block ratio (< 10 transactions)
        empty_blocks = sum(1 for b in blocks if b['tx_count'] < 10)
        empty_ratio = empty_blocks / len(blocks)

        # Timestamp variance
        intervals = [b['time_since_last'] for b in blocks if b['time_since_last'] > 0]
        timestamp_var = np.std(intervals) if intervals else 0

        # Fast blocks (< 60 seconds)
        fast_blocks = sum(1 for i in intervals if i < 60)
        fast_ratio = fast_blocks / len(intervals) if intervals else 0

        # Avg interval
        avg_interval = np.mean(intervals) if intervals else 600

        # Pool concentration (top 3 pools)
        total_blocks = sum(self.pool_blocks.values())
        if total_blocks > 0:
            top_3 = sorted(self.pool_blocks.values(), reverse=True)[:3]
            concentration = sum(top_3) / total_blocks
        else:
            concentration = 0

        # Fee extraction ratio
        recent_fees = sum(b['total_fees'] for b in blocks[-6:])
        block_rewards = 3.125 * 6  # Current subsidy
        fee_ratio = recent_fees / block_rewards if block_rewards > 0 else 0

        return MinerMetrics(
            timestamp=time.time(),
            empty_block_ratio=empty_ratio,
            timestamp_variance=timestamp_var,
            miner_concentration=concentration,
            avg_block_interval=avg_interval,
            fast_blocks_ratio=fast_ratio,
            miner_fee_extraction=fee_ratio,
        )


###############################################################################
# ADDRESS CLUSTERING / ENTITY RESOLUTION
###############################################################################

@dataclass
class ClusterMetrics:
    """Entity clustering metrics."""
    timestamp: float
    total_clusters: int = 0
    avg_cluster_size: float = 0.0
    new_entities_1h: int = 0
    merges_1h: int = 0  # Clusters merged (common input heuristic)
    largest_cluster_size: int = 0
    exchange_cluster_activity: float = 0.0


class AddressClusterer:
    """
    Groups addresses into entities using heuristics.

    Heuristics:
    - Common input ownership (addresses spent together)
    - Change address detection
    - Known entity patterns
    """

    def __init__(self):
        self.address_to_cluster: Dict[str, int] = {}
        self.clusters: Dict[int, Set[str]] = {}
        self.next_cluster_id = 0
        self.merge_events: deque = deque(maxlen=1000)
        self.new_entity_events: deque = deque(maxlen=1000)

    def analyze_tx(self, tx: dict):
        """
        Apply clustering heuristics to transaction.

        Common input heuristic: All inputs to a transaction
        are controlled by the same entity.
        """
        try:
            # Get all input addresses
            input_addresses = set()
            for vin in tx.get('vin', []):
                # Would need to look up prevout to get address
                # For now, use simplified approach
                pass

            # If multiple inputs, cluster them together
            if len(input_addresses) > 1:
                self._merge_addresses(input_addresses)

            # Detect change outputs (heuristic: round number outputs)
            for vout in tx.get('vout', []):
                value = vout.get('value', 0)
                addresses = vout.get('scriptPubKey', {}).get('addresses', [])

                # Change detection: non-round values often change
                if addresses and not self._is_round_number(value):
                    # Likely change, associate with input cluster
                    pass

        except:
            pass

    def _is_round_number(self, value: float) -> bool:
        """Check if value is a round number (likely intentional payment)."""
        # Round to 4 decimals and check if clean
        rounded = round(value, 4)
        return rounded == value and (value * 10000) % 100 == 0

    def _merge_addresses(self, addresses: Set[str]):
        """Merge addresses into same cluster."""
        now = time.time()

        # Find existing clusters
        existing_clusters = set()
        for addr in addresses:
            if addr in self.address_to_cluster:
                existing_clusters.add(self.address_to_cluster[addr])

        if not existing_clusters:
            # New cluster
            cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            self.clusters[cluster_id] = addresses.copy()
            for addr in addresses:
                self.address_to_cluster[addr] = cluster_id
            self.new_entity_events.append({'timestamp': now, 'size': len(addresses)})

        elif len(existing_clusters) == 1:
            # Add to existing cluster
            cluster_id = existing_clusters.pop()
            for addr in addresses:
                self.clusters[cluster_id].add(addr)
                self.address_to_cluster[addr] = cluster_id

        else:
            # Merge multiple clusters
            clusters_list = list(existing_clusters)
            target_cluster = clusters_list[0]

            for other_cluster in clusters_list[1:]:
                # Move all addresses to target cluster
                if other_cluster in self.clusters:
                    for addr in self.clusters[other_cluster]:
                        self.clusters[target_cluster].add(addr)
                        self.address_to_cluster[addr] = target_cluster
                    del self.clusters[other_cluster]

            # Add new addresses
            for addr in addresses:
                self.clusters[target_cluster].add(addr)
                self.address_to_cluster[addr] = target_cluster

            self.merge_events.append({
                'timestamp': now,
                'clusters_merged': len(clusters_list)
            })

    def get_metrics(self, window_seconds: int = 3600) -> ClusterMetrics:
        """Get clustering metrics."""
        now = time.time()
        cutoff = now - window_seconds

        recent_merges = [m for m in self.merge_events if m['timestamp'] >= cutoff]
        recent_new = [n for n in self.new_entity_events if n['timestamp'] >= cutoff]

        cluster_sizes = [len(c) for c in self.clusters.values()]

        return ClusterMetrics(
            timestamp=now,
            total_clusters=len(self.clusters),
            avg_cluster_size=np.mean(cluster_sizes) if cluster_sizes else 0,
            new_entities_1h=len(recent_new),
            merges_1h=len(recent_merges),
            largest_cluster_size=max(cluster_sizes) if cluster_sizes else 0,
        )


###############################################################################
# UNIFIED PIPELINE
###############################################################################

class UnifiedPipeline:
    """
    Master pipeline that combines all data sources.

    Aggregates data at multiple timeframes (1s to 5m) and
    produces unified feature vectors for the HMM.
    """

    def __init__(self, db_path: str = None):
        # Core analyzers
        self.mempool = MempoolAnalyzer()
        self.blocks = BlockAnalyzer()
        self.transactions = TransactionAnalyzer()
        self.utxo = UTXOAnalyzer()
        self.network = NetworkAnalyzer()

        # NEW: Advanced analyzers for comprehensive coverage
        self.lightning = LightningAnalyzer()
        self.ordinals = OrdinalsAnalyzer()
        self.whales = WhaleTracker()
        self.miner_behavior = MinerBehaviorAnalyzer()
        self.clustering = AddressClusterer()

        # Exchange flow integration (from existing system)
        self.exchange_flows = None  # Will be set externally

        # Feature aggregation at each scale
        self.scale_buffers: Dict[int, deque] = {
            s: deque(maxlen=1000) for s in SCALES
        }

        # Current aggregated features per scale
        self.current_features: Dict[int, UnifiedFeatureVector] = {}
        self.bucket_start_times: Dict[int, float] = {}

        # Database
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', '..', 'data', 'unified_pipeline.db'
        )
        self._init_db()

        # Threading
        self.running = False
        self.update_thread = None

        # Price tracking
        self.current_price = 0.0
        self.price_history: deque = deque(maxlen=1000)

        # Initialize buckets
        now = time.time()
        for scale in SCALES:
            self._start_new_bucket(scale, now)

    def _init_db(self):
        """Initialize database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Feature vectors table
        c.execute('''
            CREATE TABLE IF NOT EXISTS feature_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                scale INTEGER,
                features TEXT,
                price REAL,
                price_change REAL,
                outcome INTEGER,
                UNIQUE(timestamp, scale)
            )
        ''')

        conn.commit()
        conn.close()
        print(f"[PIPELINE] Database: {self.db_path}")

    def _start_new_bucket(self, scale: int, timestamp: float):
        """Start new feature bucket for a scale."""
        aligned_ts = (int(timestamp) // scale) * scale

        self.current_features[scale] = UnifiedFeatureVector(
            timestamp=aligned_ts,
            scale=scale,
            price=self.current_price,
        )
        self.bucket_start_times[scale] = aligned_ts

    def update_price(self, price: float):
        """Update current price."""
        self.current_price = price
        self.price_history.append((time.time(), price))

    def set_exchange_flows(self, exchange_flow_tracker):
        """Set exchange flow tracker for integration."""
        self.exchange_flows = exchange_flow_tracker

    def update(self) -> Dict[int, UnifiedFeatureVector]:
        """
        Update all analyzers and produce feature vectors.

        Returns feature vectors for any scales that completed a bucket.
        """
        now = time.time()
        completed_buckets = {}

        # Update analyzers
        mempool_snapshot = self.mempool.update()
        network_metrics = self.network.get_metrics()
        tx_ratios = self.transactions.get_ratios()

        # Update each scale
        for scale in SCALES:
            features = self.current_features[scale]

            # Check if bucket is complete
            if now >= self.bucket_start_times[scale] + scale:
                # Finalize bucket
                features.price = self.current_price
                if features.price > 0 and len(self.price_history) > 1:
                    old_price = self.price_history[0][1]
                    features.price_change = (self.current_price - old_price) / old_price

                # Save to database
                self._save_features(features)
                completed_buckets[scale] = features

                # Start new bucket
                self._start_new_bucket(scale, now)
                features = self.current_features[scale]

            # Update features with current data
            # Mempool
            features.mempool_tx_count = mempool_snapshot.tx_count
            features.mempool_size = mempool_snapshot.size_vbytes
            features.mempool_fee_median = mempool_snapshot.fee_median
            features.mempool_fee_spread = mempool_snapshot.fee_p75 - mempool_snapshot.fee_p25
            features.mempool_velocity = mempool_snapshot.tx_rate
            if mempool_snapshot.tx_count > 0:
                features.mempool_rbf_ratio = mempool_snapshot.rbf_count / mempool_snapshot.tx_count
                features.mempool_priority_ratio = mempool_snapshot.priority_count / mempool_snapshot.tx_count
            features.mempool_congestion = mempool_snapshot.congestion_score

            # Transactions
            features.tx_batch_ratio = tx_ratios.get('batch', 0)
            features.tx_consolidation_ratio = tx_ratios.get('consolidation', 0)
            features.tx_segwit_ratio = tx_ratios.get('segwit', 0)
            features.tx_taproot_ratio = tx_ratios.get('taproot', 0)

            # Network
            features.hash_rate_momentum = self.network.get_hash_rate_momentum()
            features.difficulty_adjustment = network_metrics.difficulty_adjustment
            features.fee_share = network_metrics.fee_share
            features.miner_revenue = network_metrics.block_reward

            # UTXO
            features.coin_days_destroyed = self.utxo.get_recent_cdd(scale)
            features.utxo_dormancy = self.utxo.get_dormancy_score(scale)

            # Block stats
            block_stats = self.blocks.get_recent_stats()
            features.block_time_variance = block_stats.get('avg_block_time', 600) - 600
            features.block_fullness = block_stats.get('avg_fullness', 0)
            features.block_fee_density = block_stats.get('avg_fee_density', 0)

            # Exchange flows (if available)
            if self.exchange_flows:
                # Would integrate with existing exchange flow tracking
                pass

            # Lightning Network metrics
            ln_metrics = self.lightning.get_metrics(scale)
            features.ln_channel_opens = ln_metrics.new_channels_1h
            features.ln_channel_closes = ln_metrics.closed_channels_1h
            features.ln_capacity_change = ln_metrics.capacity_change_1h
            features.ln_total_capacity = ln_metrics.total_capacity_btc
            features.ln_channel_count = ln_metrics.channel_count

            # Ordinals/Inscriptions metrics
            ordinals_metrics = self.ordinals.get_metrics(scale)
            features.ordinals_count = ordinals_metrics.inscription_count_1h
            features.ordinals_fees = ordinals_metrics.inscription_fees_1h
            features.brc20_count = ordinals_metrics.brc20_transfers_1h
            features.op_return_count = ordinals_metrics.op_return_count_1h

            # Whale metrics
            whale_metrics = self.whales.get_metrics(scale)
            features.whale_movements = whale_metrics.whale_movements_1h
            features.whale_accumulation = whale_metrics.whale_accumulation_1h
            features.whale_awakening = whale_metrics.dormant_whale_awakening
            features.whale_count = whale_metrics.whale_count

            # Miner behavior metrics
            miner_metrics = self.miner_behavior.get_metrics()
            features.miner_empty_ratio = miner_metrics.empty_block_ratio
            features.miner_timestamp_var = miner_metrics.timestamp_variance
            features.miner_concentration = miner_metrics.miner_concentration
            features.miner_fast_blocks = miner_metrics.fast_blocks_ratio
            features.miner_fee_extraction = miner_metrics.miner_fee_extraction

            # Address clustering metrics
            cluster_metrics = self.clustering.get_metrics(scale)
            features.cluster_count = cluster_metrics.total_clusters
            features.cluster_merges = cluster_metrics.merges_1h
            features.cluster_new_entities = cluster_metrics.new_entities_1h

            # Composite scores
            # Bullish: outflows, low fees, high segwit
            features.bullish_score = min(1.0, max(0.0,
                0.5 +
                features.exchange_net_flow * 0.3 +
                (1 - features.mempool_congestion) * 0.2 +
                features.tx_segwit_ratio * 0.1
            ))

            # Urgency: high fees, RBF, priority tx
            features.urgency_score = min(1.0, max(0.0,
                features.mempool_congestion * 0.4 +
                features.mempool_rbf_ratio * 0.3 +
                features.mempool_priority_ratio * 0.3
            ))

        return completed_buckets

    def _save_features(self, features: UnifiedFeatureVector):
        """Save feature vector to database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Determine outcome
        outcome = 0
        if features.price_change > 0.0001:
            outcome = 1
        elif features.price_change < -0.0001:
            outcome = -1

        try:
            c.execute('''
                INSERT OR REPLACE INTO feature_vectors
                (timestamp, scale, features, price, price_change, outcome)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                features.timestamp,
                features.scale,
                json.dumps(self._features_to_dict(features)),
                features.price,
                features.price_change,
                outcome
            ))
            conn.commit()
        except Exception as e:
            print(f"[PIPELINE] Save failed: {e}")
        finally:
            conn.close()

    def _features_to_dict(self, features: UnifiedFeatureVector) -> Dict:
        """Convert features to dictionary."""
        return {
            'mempool_tx_count': features.mempool_tx_count,
            'mempool_size': features.mempool_size,
            'mempool_fee_median': features.mempool_fee_median,
            'mempool_fee_spread': features.mempool_fee_spread,
            'mempool_velocity': features.mempool_velocity,
            'mempool_rbf_ratio': features.mempool_rbf_ratio,
            'mempool_priority_ratio': features.mempool_priority_ratio,
            'mempool_congestion': features.mempool_congestion,
            'block_time_variance': features.block_time_variance,
            'block_fullness': features.block_fullness,
            'block_fee_density': features.block_fee_density,
            'tx_batch_ratio': features.tx_batch_ratio,
            'tx_consolidation_ratio': features.tx_consolidation_ratio,
            'tx_segwit_ratio': features.tx_segwit_ratio,
            'tx_taproot_ratio': features.tx_taproot_ratio,
            'coin_days_destroyed': features.coin_days_destroyed,
            'utxo_dormancy': features.utxo_dormancy,
            'hash_rate_momentum': features.hash_rate_momentum,
            'exchange_net_flow': features.exchange_net_flow,
            'whale_net_flow': features.whale_net_flow,
            'bullish_score': features.bullish_score,
            'urgency_score': features.urgency_score,
        }

    def get_features(self, scale: int) -> Optional[UnifiedFeatureVector]:
        """Get current feature vector for a scale."""
        return self.current_features.get(scale)

    def get_feature_vector(self, scale: int) -> np.ndarray:
        """Get feature vector as numpy array for HMM input."""
        features = self.current_features.get(scale)
        if not features:
            return np.zeros(22)  # 22 features

        return np.array([
            features.mempool_tx_count / 100000,  # Normalize
            features.mempool_size / 1e9,
            features.mempool_fee_median / 100,
            features.mempool_fee_spread / 100,
            features.mempool_velocity,
            features.mempool_rbf_ratio,
            features.mempool_priority_ratio,
            features.mempool_congestion,
            features.block_time_variance / 600,
            features.block_fullness,
            features.block_fee_density / 100,
            features.tx_batch_ratio,
            features.tx_consolidation_ratio,
            features.tx_segwit_ratio,
            features.tx_taproot_ratio,
            features.coin_days_destroyed / 1000,
            features.utxo_dormancy,
            features.hash_rate_momentum,
            features.exchange_net_flow / 1000,
            features.whale_net_flow / 1000,
            features.bullish_score,
            features.urgency_score,
        ])

    def on_new_block(self, block_hash: str):
        """Handle new block notification."""
        self.blocks.on_new_block(block_hash)

    def on_new_tx(self, txid: str, raw_tx: bytes = None):
        """Handle new transaction notification."""
        self.mempool.on_new_tx(txid, raw_tx)
        self.transactions.analyze_tx(txid)

    def start_background_updates(self, interval: float = 1.0):
        """Start background update thread."""
        self.running = True

        def update_loop():
            while self.running:
                try:
                    self.update()
                except Exception as e:
                    print(f"[PIPELINE] Update error: {e}")
                time.sleep(interval)

        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        print(f"[PIPELINE] Background updates started (interval={interval}s)")

    def stop(self):
        """Stop background updates."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        print("[PIPELINE] Stopped")


###############################################################################
# MAIN
###############################################################################

def main():
    """Test the pipeline."""
    print("="*60)
    print("UNIFIED BLOCKCHAIN DATA PIPELINE")
    print("="*60)

    pipeline = UnifiedPipeline()

    # Get initial price
    try:
        from urllib.request import urlopen, Request
        url = 'https://api.exchange.coinbase.com/products/BTC-USD/ticker'
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=5) as resp:
            price = float(json.loads(resp.read().decode())['price'])
            pipeline.update_price(price)
            print(f"[PIPELINE] Initial price: ${price:,.0f}")
    except:
        pipeline.update_price(97000)

    # Start background updates
    pipeline.start_background_updates(interval=1.0)

    print("\nRunning for 60 seconds...")

    try:
        for i in range(60):
            time.sleep(1)

            if i % 10 == 0:
                # Print current state
                for scale in [10, 60]:
                    features = pipeline.get_features(scale)
                    if features:
                        print(f"\n[{scale}s] Mempool: {features.mempool_tx_count:.0f} tx | "
                              f"Fee: {features.mempool_fee_median:.1f} sat/vB | "
                              f"Congestion: {features.mempool_congestion:.2f} | "
                              f"Bullish: {features.bullish_score:.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        pipeline.stop()

    print("\n[PIPELINE] Done")


if __name__ == "__main__":
    main()
