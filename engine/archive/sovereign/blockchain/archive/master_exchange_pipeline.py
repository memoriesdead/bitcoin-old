#!/usr/bin/env python3
"""
MASTER EXCHANGE PIPELINE - 100% DETERMINISTIC TRADING
======================================================
Real-time per-exchange flow detection and signal generation.

Architecture:
  Bitcoin Core ZMQ (rawtx)
         │
         ▼
  Transaction Decoder (parse raw TX)
         │
         ▼
  Address Matcher (8.6M addresses → exchange)
         │
         ▼
  Per-Exchange Flow Aggregator
         │
         ├── Binance Pipeline
         ├── Coinbase Pipeline
         ├── Kraken Pipeline
         ├── Bitfinex Pipeline
         ├── Bitstamp Pipeline
         ├── Huobi Pipeline
         └── ... all exchanges
         │
         ▼
  Signal Generator (SHORT on inflow, LONG on exhaustion)
         │
         ▼
  Correlation DB (track accuracy per exchange)

Usage:
    python3 master_exchange_pipeline.py          # Live mode
    python3 master_exchange_pipeline.py --test   # Test mode (no trading)
"""

import os
import sys
import time
import json
import sqlite3
import threading
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Add paths
sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')

# Import components
try:
    from zmq_subscriber import BlockchainZMQ
    from tx_decoder import TransactionDecoder
    from multi_price_feed import MultiExchangePriceFeed
    from correlation_db import CorrelationDatabase
    from exchange_utxo_cache import ExchangeUTXOCache
except ImportError:
    from blockchain.zmq_subscriber import BlockchainZMQ
    from blockchain.tx_decoder import TransactionDecoder
    from blockchain.multi_price_feed import MultiExchangePriceFeed
    from blockchain.correlation_db import CorrelationDatabase
    from blockchain.exchange_utxo_cache import ExchangeUTXOCache


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    # Database
    db_path: str = "/root/sovereign/walletexplorer_addresses.db"
    correlation_db_path: str = "/root/sovereign/correlation.db"
    utxo_cache_path: str = "/root/sovereign/exchange_utxos.db"

    # ZMQ
    zmq_endpoint: str = "tcp://127.0.0.1:28332"

    # Signal thresholds
    min_flow_btc: float = 0.1         # Minimum flow to track
    min_signal_btc: float = 10.0      # Minimum net flow for signal

    # Aggregation window
    window_blocks: int = 10           # Aggregate flows over N blocks (if using blocks)
    window_seconds: int = 60          # Aggregate flows over N seconds (time-based)

    # Exhaustion pattern (LONG signals)
    exhaustion_ratio: float = 0.4     # Inflow < 40% of average = exhaustion
    rolling_window: int = 10          # Rolling average over N windows
    min_exhaustion_outflow: float = 2.0  # Minimum net outflow for exhaustion
    min_consecutive: int = 2          # Consecutive windows of exhaustion

    # USA-legal exchanges only (trade on these)
    major_exchanges: Set[str] = field(default_factory=lambda: {
        'coinbase', 'kraken', 'bitstamp', 'gemini', 'crypto.com'
    })

    # Trading
    test_mode: bool = True            # Don't execute real trades
    short_only: bool = False          # Enable BOTH LONG and SHORT signals
    usa_only: bool = True             # Only generate signals for USA exchanges


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExchangeFlow:
    """Flow data for a single exchange."""
    exchange: str
    inflow_btc: float = 0.0
    outflow_btc: float = 0.0
    inflow_count: int = 0
    outflow_count: int = 0

    @property
    def net_flow(self) -> float:
        """Positive = net outflow, Negative = net inflow."""
        return self.outflow_btc - self.inflow_btc


@dataclass
class Signal:
    """Trading signal."""
    exchange: str
    direction: str          # 'LONG' or 'SHORT'
    amount_btc: float       # Net flow amount
    confidence: float       # 0.0 - 1.0
    reason: str
    price: float
    timestamp: float
    pattern: str            # 'inflow', 'exhaustion', etc.


# =============================================================================
# PER-EXCHANGE PIPELINE
# =============================================================================

class ExchangePipeline:
    """
    Individual pipeline for one exchange.

    Tracks:
    - Real-time inflows/outflows
    - Rolling averages for exhaustion detection
    - Signal generation
    """

    def __init__(self, exchange: str, config: PipelineConfig):
        self.exchange = exchange
        self.config = config

        # Current window flows
        self.current_window = ExchangeFlow(exchange=exchange)

        # Rolling history for exhaustion detection
        self.inflow_history: deque = deque(maxlen=config.rolling_window)
        self.outflow_history: deque = deque(maxlen=config.rolling_window)

        # Exhaustion tracking
        self.consecutive_exhaustion = 0

        # Statistics
        self.total_inflow_btc = 0.0
        self.total_outflow_btc = 0.0
        self.signals_generated = 0

    def add_inflow(self, amount_btc: float):
        """Record an inflow to this exchange."""
        self.current_window.inflow_btc += amount_btc
        self.current_window.inflow_count += 1
        self.total_inflow_btc += amount_btc

    def add_outflow(self, amount_btc: float):
        """Record an outflow from this exchange."""
        self.current_window.outflow_btc += amount_btc
        self.current_window.outflow_count += 1
        self.total_outflow_btc += amount_btc

    def close_window(self) -> Optional[Signal]:
        """
        Close current window and check for signals.

        Returns:
            Signal if conditions are met, None otherwise.
        """
        window = self.current_window
        signal = None

        # Add to history
        self.inflow_history.append(window.inflow_btc)
        self.outflow_history.append(window.outflow_btc)

        # Skip non-USA exchanges if usa_only is enabled
        if self.config.usa_only and self.exchange not in self.config.major_exchanges:
            self.current_window = ExchangeFlow(exchange=self.exchange)
            return None

        # Check for signals
        net_flow = window.net_flow  # Positive = net outflow

        # === SHORT SIGNAL: Net inflow (deposit to sell) ===
        if net_flow < -self.config.min_signal_btc:
            signal = Signal(
                exchange=self.exchange,
                direction='SHORT',
                amount_btc=abs(net_flow),
                confidence=min(abs(net_flow) / 100, 1.0),  # Scale by size
                reason=f"Net inflow {abs(net_flow):.1f} BTC",
                price=0.0,  # Filled later
                timestamp=time.time(),
                pattern='inflow'
            )
            self.signals_generated += 1

        # === LONG SIGNAL: Seller exhaustion pattern ===
        if len(self.inflow_history) >= self.config.rolling_window:
            avg_inflow = sum(self.inflow_history) / len(self.inflow_history)

            # Exhaustion: current inflow < 40% of average AND net outflow
            is_exhaustion = (
                window.inflow_btc < avg_inflow * self.config.exhaustion_ratio and
                net_flow > self.config.min_exhaustion_outflow
            )

            if is_exhaustion:
                self.consecutive_exhaustion += 1
            else:
                self.consecutive_exhaustion = 0

            # Signal after sustained exhaustion
            if self.consecutive_exhaustion >= self.config.min_consecutive:
                if not self.config.short_only:  # Only if LONG enabled
                    signal = Signal(
                        exchange=self.exchange,
                        direction='LONG',
                        amount_btc=net_flow,
                        confidence=min(self.consecutive_exhaustion * 0.3, 1.0),
                        reason=f"Seller exhaustion ({self.consecutive_exhaustion} windows)",
                        price=0.0,
                        timestamp=time.time(),
                        pattern='exhaustion'
                    )
                    self.signals_generated += 1

        # Reset window
        self.current_window = ExchangeFlow(exchange=self.exchange)

        return signal

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            'exchange': self.exchange,
            'total_inflow_btc': self.total_inflow_btc,
            'total_outflow_btc': self.total_outflow_btc,
            'net_btc': self.total_outflow_btc - self.total_inflow_btc,
            'signals_generated': self.signals_generated,
            'consecutive_exhaustion': self.consecutive_exhaustion,
        }


# =============================================================================
# MASTER PIPELINE
# =============================================================================

class MasterExchangePipeline:
    """
    Master pipeline that coordinates all per-exchange pipelines.

    Flow:
    1. ZMQ receives raw transaction
    2. Decode transaction → inputs/outputs
    3. Match addresses → exchanges
    4. Route to per-exchange pipelines
    5. Aggregate over window
    6. Generate signals
    7. Log to correlation DB
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

        print("=" * 70)
        print("MASTER EXCHANGE PIPELINE")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        print(f"Mode: {'TEST' if self.config.test_mode else 'LIVE'}")
        print(f"SHORT only: {self.config.short_only}")
        print()

        # Load exchange addresses
        self.addresses: Dict[str, str] = {}  # address → exchange
        self._load_addresses()

        # Per-exchange pipelines
        self.pipelines: Dict[str, ExchangePipeline] = {}

        # Components
        self.decoder = TransactionDecoder()
        self.price_feed = MultiExchangePriceFeed(refresh_interval=1.0)
        self.correlation_db = CorrelationDatabase(self.config.correlation_db_path)
        self.utxo_cache = ExchangeUTXOCache(self.config.utxo_cache_path)
        self.zmq: Optional[BlockchainZMQ] = None

        # Block tracking
        self.blocks_in_window = 0
        self.last_block_time = time.time()
        self.last_window_close = time.time()

        # Signal tracking
        self.signals: List[Signal] = []
        self.signals_lock = threading.Lock()

        # Running state
        self.running = False
        self.window_thread = None

    def _load_addresses(self):
        """INSTANT startup - use SQLite directly, no memory loading."""
        print(f"Connecting to address database: {self.config.db_path}")

        # Keep connection open for fast lookups
        self.addr_db = sqlite3.connect(self.config.db_path, check_same_thread=False)
        self.addr_cursor = self.addr_db.cursor()

        # SPEED PRAGMAS - 10-50x faster lookups
        self.addr_cursor.execute("PRAGMA journal_mode = WAL")      # Concurrent reads
        self.addr_cursor.execute("PRAGMA synchronous = normal")    # Faster writes
        self.addr_cursor.execute("PRAGMA temp_store = memory")     # Temp in RAM
        self.addr_cursor.execute("PRAGMA mmap_size = 1000000000")  # 1GB memory-mapped
        self.addr_cursor.execute("PRAGMA cache_size = -500000")    # 500MB page cache

        # Ensure index exists for instant lookups
        self.addr_cursor.execute("CREATE INDEX IF NOT EXISTS idx_addr ON addresses(address)")
        self.addr_db.commit()

        # Get stats
        self.addr_cursor.execute("SELECT COUNT(*) FROM addresses")
        total = self.addr_cursor.fetchone()[0]

        self.addr_cursor.execute("SELECT COUNT(DISTINCT exchange) FROM addresses")
        num_exchanges = self.addr_cursor.fetchone()[0]

        print(f"Ready: {total:,} addresses across {num_exchanges} exchanges")
        print()

        # addresses dict is now empty - we use lookup_address() instead
        self.addresses = {}

    def lookup_address(self, addr: str) -> Optional[str]:
        """O(1) lookup of address to exchange using SQLite index."""
        self.addr_cursor.execute("SELECT exchange FROM addresses WHERE address = ?", (addr,))
        row = self.addr_cursor.fetchone()
        return row[0].lower() if row else None

    def _get_or_create_pipeline(self, exchange: str) -> ExchangePipeline:
        """Get or create pipeline for an exchange."""
        if exchange not in self.pipelines:
            self.pipelines[exchange] = ExchangePipeline(exchange, self.config)
        return self.pipelines[exchange]

    def _on_transaction(self, raw_tx: bytes):
        """Handle incoming transaction from ZMQ."""
        try:
            tx = self.decoder.decode(raw_tx)
            if not tx:
                return

            self._process_transaction(tx)
        except Exception as e:
            print(f"[ERROR] TX processing: {e}")

    def _process_transaction(self, tx: Dict):
        """
        Process decoded transaction for exchange flows.

        Logic:
        - Output to exchange address = INFLOW (deposit) + cache UTXO
        - Input spending cached UTXO = OUTFLOW (withdrawal)

        CRITICAL FIX: Use UTXO cache for outflow detection!
        Raw TX inputs don't contain BTC value - must look up from cache.
        """
        txid = tx.get('txid', '')
        tx_inflows: Dict[str, float] = defaultdict(float)
        tx_outflows: Dict[str, float] = defaultdict(float)

        # Track TX count for debugging
        self.tx_count = getattr(self, 'tx_count', 0) + 1
        inputs_list = tx.get('inputs', [])

        # Debug: print first 5 TXs to see input structure
        if self.tx_count <= 5:
            print(f"[TRACE] TX#{self.tx_count} has {len(inputs_list)} inputs")
            if inputs_list:
                sample = inputs_list[0]
                print(f"[TRACE]   Input[0]: prev_txid={sample.get('prev_txid', 'NONE')[:16] if sample.get('prev_txid') else 'NONE'}..., prev_vout={sample.get('prev_vout')}")

        # Check INPUTS for OUTFLOWS (spending cached exchange UTXOs)
        # This is the FIX: use utxo_cache.spend_utxo() to get BTC value
        inputs_checked = 0
        for inp in inputs_list:
            prev_txid = inp.get('prev_txid')
            prev_vout = inp.get('prev_vout')

            if prev_txid and prev_vout is not None:
                inputs_checked += 1
                # Look up in UTXO cache - this returns (value_sat, exchange, address)
                result = self.utxo_cache.spend_utxo(prev_txid, prev_vout)
                if result:
                    value_sat, exchange, address = result
                    btc = value_sat / 1e8
                    print(f"[OUTFLOW] {exchange}: {btc:.4f} BTC spent from {address[:16]}...")
                    if btc >= self.config.min_flow_btc:
                        tx_outflows[exchange] += btc

        # Debug: log input stats every 1000 TXs
        self.inputs_checked = getattr(self, 'inputs_checked', 0) + inputs_checked
        if self.tx_count % 1000 == 0:
            print(f"[DEBUG] TX #{self.tx_count:,}: checked {self.inputs_checked:,} inputs, cache: {len(self.utxo_cache.cache):,}")

        # Check OUTPUTS for INFLOWS (deposits to exchange addresses)
        for i, output in enumerate(tx.get('outputs', [])):
            addr = output.get('address')
            btc = output.get('btc', 0)

            if addr and btc >= self.config.min_flow_btc:
                # Use SQLite lookup instead of in-memory dict
                exchange = self.lookup_address(addr)
                if exchange:
                    tx_inflows[exchange] += btc

                    # Cache this UTXO for future outflow detection
                    value_sat = int(btc * 1e8)
                    self.utxo_cache.add_utxo(txid, i, value_sat, exchange, addr)

        # Skip internal transfers (same exchange in inputs and outputs)
        all_exchanges = set(tx_inflows.keys()) | set(tx_outflows.keys())

        for exchange in all_exchanges:
            inflow = tx_inflows.get(exchange, 0)
            outflow = tx_outflows.get(exchange, 0)

            # Skip if both present (internal transfer)
            if inflow > 0 and outflow > 0:
                continue

            pipeline = self._get_or_create_pipeline(exchange)

            if inflow > 0:
                pipeline.add_inflow(inflow)
            if outflow > 0:
                pipeline.add_outflow(outflow)

    def _on_block(self, raw_block: bytes):
        """Handle new block - triggers window check."""
        self.blocks_in_window += 1

        if self.blocks_in_window >= self.config.window_blocks:
            self._close_windows()
            self.blocks_in_window = 0

    def _close_windows(self):
        """Close all pipeline windows and check for signals."""
        now = time.time()

        for exchange, pipeline in self.pipelines.items():
            signal = pipeline.close_window()

            if signal:
                # Get price for this exchange
                price = self.price_feed.get_price(exchange)
                if price:
                    signal.price = price

                # Add to signals
                with self.signals_lock:
                    self.signals.append(signal)

                # Log to correlation DB
                direction = 'INFLOW' if signal.direction == 'SHORT' else 'OUTFLOW'
                self.correlation_db.record_flow(
                    exchange=exchange,
                    direction=direction,
                    amount_btc=signal.amount_btc,
                    txid=f"window_{int(now)}",
                    price_now=signal.price
                )

                # Print signal
                self._print_signal(signal)

        # Periodic stats
        elapsed = now - self.last_block_time
        self.last_block_time = now

    def _print_signal(self, signal: Signal):
        """Print signal with color coding."""
        if signal.direction == 'SHORT':
            color = '\033[91m'  # Red
        else:
            color = '\033[92m'  # Green
        reset = '\033[0m'

        major = '*' if signal.exchange in self.config.major_exchanges else ''

        print(f"{color}[{signal.direction}]{reset} {signal.exchange.upper()}{major} "
              f"{signal.amount_btc:.1f} BTC @ ${signal.price:,.0f} | "
              f"conf:{signal.confidence:.0%} | {signal.reason}")

    def start(self):
        """Start the pipeline."""
        if self.running:
            return

        self.running = True

        # Start price feed
        self.price_feed.start()
        print("[PRICE] Price feed started")

        # Start ZMQ
        self.zmq = BlockchainZMQ(
            rawtx_endpoint=self.config.zmq_endpoint,
            on_transaction=self._on_transaction,
            on_block=self._on_block
        )

        if self.zmq.start():
            print("[ZMQ] Connected to Bitcoin Core")
        else:
            print("[ZMQ] Failed to connect!")
            self.running = False
            return

        # Start correlation verification thread
        self.verify_thread = threading.Thread(
            target=self._verification_loop,
            daemon=True
        )
        self.verify_thread.start()

        # Start time-based window thread
        self.window_thread = threading.Thread(
            target=self._window_loop,
            daemon=True
        )
        self.window_thread.start()
        print(f"[WINDOW] Time-based windows: every {self.config.window_seconds}s")

        print()
        print("=" * 70)
        print("PIPELINE RUNNING - Waiting for transactions...")
        print("=" * 70)
        print()

    def _verification_loop(self):
        """Background loop to verify price predictions."""
        while self.running:
            try:
                self.correlation_db.check_pending_verifications()
            except Exception as e:
                print(f"[VERIFY] Error: {e}")
            time.sleep(5)

    def _window_loop(self):
        """Time-based window closing (every N seconds)."""
        while self.running:
            try:
                time.sleep(self.config.window_seconds)
                if self.running:
                    self._close_windows()
            except Exception as e:
                print(f"[WINDOW] Error: {e}")

    def stop(self):
        """Stop the pipeline."""
        self.running = False

        if self.zmq:
            self.zmq.stop()
        self.price_feed.stop()

        print()
        print("[PIPELINE] Stopped")

    def print_stats(self):
        """Print pipeline statistics."""
        print()
        print("=" * 70)
        print("PIPELINE STATISTICS")
        print("=" * 70)

        print(f"\n{'Exchange':<15} {'Inflow':>12} {'Outflow':>12} {'Net':>12} {'Signals':>8}")
        print("-" * 65)

        total_inflow = 0
        total_outflow = 0
        total_signals = 0

        for exchange in sorted(self.pipelines.keys()):
            stats = self.pipelines[exchange].get_stats()
            major = '*' if exchange in self.config.major_exchanges else ''

            print(f"{exchange:<15} {stats['total_inflow_btc']:>11.1f} "
                  f"{stats['total_outflow_btc']:>11.1f} {stats['net_btc']:>+11.1f} "
                  f"{stats['signals_generated']:>8} {major}")

            total_inflow += stats['total_inflow_btc']
            total_outflow += stats['total_outflow_btc']
            total_signals += stats['signals_generated']

        print("-" * 65)
        print(f"{'TOTAL':<15} {total_inflow:>11.1f} {total_outflow:>11.1f} "
              f"{total_outflow - total_inflow:>+11.1f} {total_signals:>8}")

        # Recent signals
        with self.signals_lock:
            if self.signals:
                print(f"\n\nRECENT SIGNALS (last 10):")
                print("-" * 65)
                for sig in self.signals[-10:]:
                    major = '*' if sig.exchange in self.config.major_exchanges else ''
                    ts = datetime.fromtimestamp(sig.timestamp).strftime('%H:%M:%S')
                    print(f"  {ts} {sig.direction:5} {sig.exchange}{major:1} "
                          f"{sig.amount_btc:>8.1f} BTC @ ${sig.price:>9,.0f}")

        print()

    def run_interactive(self):
        """Run with interactive controls."""
        self.start()

        try:
            while self.running:
                time.sleep(30)
                self.print_stats()
        except KeyboardInterrupt:
            print("\n[Ctrl+C] Stopping...")
        finally:
            self.stop()
            self.print_stats()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Master Exchange Pipeline')
    parser.add_argument('--test', action='store_true', help='Test mode (no trading)')
    parser.add_argument('--live', action='store_true', help='Live mode (execute trades)')
    parser.add_argument('--long', action='store_true', help='Enable LONG signals')
    parser.add_argument('--window', type=int, default=10, help='Window size in blocks')
    parser.add_argument('--min-signal', type=float, default=10.0, help='Min BTC for signal')
    args = parser.parse_args()

    config = PipelineConfig(
        test_mode=not args.live,
        short_only=not args.long,
        window_blocks=args.window,
        min_signal_btc=args.min_signal,
    )

    pipeline = MasterExchangePipeline(config)
    pipeline.run_interactive()


if __name__ == '__main__':
    main()
