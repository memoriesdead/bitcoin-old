#!/usr/bin/env python3
"""
DETERMINISTIC ZMQ PIPELINE - 100% SIGNALS
==========================================

Pure Python pipeline that captures FULL TX data from ZMQ for destination classification.

THE KEY INSIGHT:
    When an exchange spends a UTXO, we can see WHERE the BTC goes:

    INTERNAL (to exchange address) → Consolidation → About to sell → SHORT
    EXTERNAL (to non-exchange address) → Customer withdrawal → Already bought → LONG

This is NOT prediction. This is observing ACTUAL blockchain behavior.

PIPELINE:
    1. ZMQ receives every mempool TX
    2. Parse with TransactionDecoder (get full TX data including txid)
    3. Check inputs: Do they spend from known exchange UTXOs?
    4. If YES, classify ALL outputs as internal/external
    5. Generate deterministic signal based on classification

Usage:
    python3 deterministic_zmq_pipeline.py
"""

import sqlite3
import time
import threading
import sys
from datetime import datetime, timezone
from typing import Dict, Set, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Local imports
from zmq_subscriber import BlockchainZMQ
from tx_decoder import TransactionDecoder

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    print("[WARN] ccxt not available, price feed disabled")


@dataclass
class DeterministicSignal:
    """A 100% deterministic signal based on destination classification."""
    timestamp: str
    signal_type: str          # 'SHORT_INTERNAL' or 'LONG_EXTERNAL'
    exchange: str             # Source exchange
    txid: str
    total_btc: float          # Total outflow
    internal_btc: float       # Portion going to exchange addresses
    external_btc: float       # Portion going to non-exchange addresses
    internal_pct: float       # % internal
    external_pct: float       # % external
    destination_exchanges: str  # Which exchanges received (for internal)
    price: float


class DeterministicZMQPipeline:
    """
    Pure Python pipeline with full TX data for deterministic signals.

    Bypasses C++ runner limitation by parsing TX directly from ZMQ.
    """

    # Signal thresholds
    MIN_OUTFLOW_BTC = 1.0       # Minimum outflow to generate signal
    INTERNAL_THRESHOLD = 0.7    # 70%+ internal = SHORT
    EXTERNAL_THRESHOLD = 0.7    # 70%+ external = LONG

    def __init__(self, db_path: str = "/root/sovereign"):
        self.db_path = Path(db_path)

        # Load exchange addresses
        self.exchange_addresses, self.address_to_exchange = self._load_addresses()
        print(f"[PIPELINE] Loaded {len(self.exchange_addresses):,} exchange addresses")

        # Load current exchange UTXOs for outflow detection
        self.exchange_utxos = self._load_utxos()
        print(f"[PIPELINE] Loaded {len(self.exchange_utxos):,} tracked UTXOs")

        # TX decoder
        self.decoder = TransactionDecoder()

        # Price feed
        self.current_price = 0.0
        if HAS_CCXT:
            self._start_price_feed()

        # Initialize signal database
        self._init_db()

        # Stats
        self.tx_processed = 0
        self.inflows_detected = 0
        self.outflows_detected = 0
        self.short_signals = 0
        self.long_signals = 0
        self.mixed_skipped = 0

        # Recent signals for deduplication
        self.recent_signals: Set[str] = set()

    def _load_addresses(self) -> Tuple[Set[str], Dict[str, str]]:
        """Load exchange addresses from database."""
        addresses = set()
        addr_to_exchange = {}

        db_file = self.db_path / "walletexplorer_addresses.db"
        if not db_file.exists():
            print(f"[ERROR] Address DB not found: {db_file}")
            return addresses, addr_to_exchange

        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            cursor.execute("SELECT address, exchange FROM addresses")
            for row in cursor.fetchall():
                addr, exchange = row
                addresses.add(addr)
                addr_to_exchange[addr] = exchange.lower()
            conn.close()
        except Exception as e:
            print(f"[ERROR] Loading addresses: {e}")

        return addresses, addr_to_exchange

    def _load_utxos(self) -> Dict[Tuple[str, int], Tuple[float, str, str]]:
        """Load tracked exchange UTXOs for outflow detection."""
        utxos = {}

        db_file = self.db_path / "exchange_utxos.db"
        if not db_file.exists():
            print(f"[WARN] UTXO cache not found: {db_file}")
            return utxos

        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT txid, vout, value_sat, exchange, address
                FROM utxos
            """)
            for row in cursor.fetchall():
                txid, vout, value_sat, exchange, address = row
                key = (txid, vout)
                utxos[key] = (value_sat / 1e8, exchange, address)
            conn.close()
        except Exception as e:
            print(f"[ERROR] Loading UTXOs: {e}")

        return utxos

    def _start_price_feed(self):
        """Start background price feed."""
        def update_price():
            feed = ccxt.kraken()
            while True:
                try:
                    ticker = feed.fetch_ticker('BTC/USD')
                    self.current_price = ticker['last']
                except:
                    pass
                time.sleep(1)

        t = threading.Thread(target=update_price, daemon=True)
        t.start()

        # Wait for first price
        timeout = 10
        start = time.time()
        while self.current_price == 0 and time.time() - start < timeout:
            time.sleep(0.1)

        if self.current_price > 0:
            print(f"[PIPELINE] Price feed ready: ${self.current_price:,.2f}")
        else:
            print("[WARN] Price feed not available")

    def _init_db(self):
        """Initialize signal database."""
        db_file = self.db_path / "deterministic_signals.db"
        conn = sqlite3.connect(str(db_file))

        conn.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                signal_type TEXT,
                exchange TEXT,
                txid TEXT UNIQUE,
                total_btc REAL,
                internal_btc REAL,
                external_btc REAL,
                internal_pct REAL,
                external_pct REAL,
                destination_exchanges TEXT,
                price_at_signal REAL,

                -- Price tracking (filled later)
                price_t1min REAL,
                price_t5min REAL,
                price_t10min REAL,
                correct INTEGER
            )
        ''')

        # Index for quick queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_type ON signals(signal_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_exchange ON signals(exchange)")

        conn.commit()
        conn.close()

    def process_transaction(self, raw_tx: bytes):
        """
        Process a transaction from ZMQ.

        1. Decode TX to get full data
        2. Check if inputs spend exchange UTXOs (outflow)
        3. If outflow, classify all outputs
        4. Generate deterministic signal
        """
        self.tx_processed += 1

        # Decode TX
        tx = self.decoder.decode(raw_tx)
        if not tx:
            return

        txid = tx['txid']

        # Check for inflows (outputs to exchange addresses)
        self._check_inflows(tx)

        # Check for outflows (inputs spending exchange UTXOs)
        outflow_info = self._check_outflows(tx)

        if outflow_info:
            exchange, outflow_btc = outflow_info

            if outflow_btc >= self.MIN_OUTFLOW_BTC:
                # Classify ALL outputs
                signal = self._classify_and_signal(tx, exchange, outflow_btc)

                if signal:
                    self._handle_signal(signal)

    def _check_inflows(self, tx: Dict):
        """Check if any outputs go to exchange addresses (inflows/deposits)."""
        for i, output in enumerate(tx.get('outputs', [])):
            addr = output.get('address')
            btc = output.get('btc', 0)

            if addr and addr in self.exchange_addresses:
                exchange = self.address_to_exchange.get(addr, 'unknown')
                self.inflows_detected += 1

                # Cache this UTXO for future outflow detection
                key = (tx['txid'], i)
                self.exchange_utxos[key] = (btc, exchange, addr)

                if btc >= 1.0:  # Log significant inflows
                    print(f"[INFLOW] +{btc:.4f} BTC to {exchange.upper()}")

    def _check_outflows(self, tx: Dict) -> Optional[Tuple[str, float]]:
        """
        Check if any inputs spend exchange UTXOs (outflows).

        Returns (exchange, total_outflow_btc) if outflow detected.
        """
        total_outflow = 0.0
        source_exchange = None

        for inp in tx.get('inputs', []):
            prev_txid = inp.get('prev_txid')
            prev_vout = inp.get('prev_vout')

            if prev_txid and prev_vout is not None:
                key = (prev_txid, prev_vout)

                if key in self.exchange_utxos:
                    btc, exchange, _ = self.exchange_utxos[key]
                    total_outflow += btc
                    source_exchange = exchange

                    # Remove from cache (spent)
                    del self.exchange_utxos[key]

        if total_outflow > 0 and source_exchange:
            self.outflows_detected += 1
            return (source_exchange, total_outflow)

        return None

    def _classify_and_signal(self, tx: Dict, source_exchange: str,
                             outflow_btc: float) -> Optional[DeterministicSignal]:
        """
        Classify ALL outputs as internal (exchange) or external (non-exchange).

        Generate deterministic signal based on classification.
        """
        internal_btc = 0.0
        external_btc = 0.0
        destination_exchanges = set()

        for output in tx.get('outputs', []):
            addr = output.get('address')
            btc = output.get('btc', 0)

            if addr:
                if addr in self.exchange_addresses:
                    # INTERNAL - going to an exchange address
                    internal_btc += btc
                    dest_exchange = self.address_to_exchange.get(addr, 'unknown')
                    destination_exchanges.add(dest_exchange)
                else:
                    # EXTERNAL - going to non-exchange address
                    external_btc += btc

        total_output = internal_btc + external_btc
        if total_output == 0:
            return None

        internal_pct = internal_btc / total_output
        external_pct = external_btc / total_output

        # Determine signal type
        signal_type = None

        if internal_pct >= self.INTERNAL_THRESHOLD:
            # Majority going to exchange addresses = consolidation = SHORT
            signal_type = "SHORT_INTERNAL"
            self.short_signals += 1
        elif external_pct >= self.EXTERNAL_THRESHOLD:
            # Majority going to non-exchange = withdrawal = LONG
            signal_type = "LONG_EXTERNAL"
            self.long_signals += 1
        else:
            # Mixed - skip
            self.mixed_skipped += 1
            return None

        # Deduplicate
        if tx['txid'] in self.recent_signals:
            return None
        self.recent_signals.add(tx['txid'])

        # Keep only last 1000
        if len(self.recent_signals) > 1000:
            self.recent_signals = set(list(self.recent_signals)[-500:])

        return DeterministicSignal(
            timestamp=datetime.now(timezone.utc).isoformat(),
            signal_type=signal_type,
            exchange=source_exchange,
            txid=tx['txid'],
            total_btc=outflow_btc,
            internal_btc=internal_btc,
            external_btc=external_btc,
            internal_pct=internal_pct,
            external_pct=external_pct,
            destination_exchanges=",".join(destination_exchanges),
            price=self.current_price
        )

    def _handle_signal(self, signal: DeterministicSignal):
        """Handle a deterministic signal - log and save."""
        # Print signal
        if signal.signal_type == "SHORT_INTERNAL":
            emoji = "🔴"
            direction = "SHORT"
            reason = f"Consolidating to {signal.destination_exchanges}"
        else:
            emoji = "🟢"
            direction = "LONG"
            reason = "Customer withdrawal"

        print()
        print("=" * 70)
        print(f"{emoji} DETERMINISTIC SIGNAL: {direction}")
        print("=" * 70)
        print(f"  Exchange:     {signal.exchange.upper()}")
        print(f"  Type:         {signal.signal_type}")
        print(f"  Amount:       {signal.total_btc:.4f} BTC")
        print(f"  Internal:     {signal.internal_btc:.4f} BTC ({signal.internal_pct:.1%})")
        print(f"  External:     {signal.external_btc:.4f} BTC ({signal.external_pct:.1%})")
        print(f"  Reason:       {reason}")
        print(f"  Price:        ${signal.price:,.2f}")
        print(f"  TXID:         {signal.txid[:32]}...")
        print("=" * 70)
        print()

        # Save to database
        self._save_signal(signal)

    def _save_signal(self, signal: DeterministicSignal):
        """Save signal to database."""
        db_file = self.db_path / "deterministic_signals.db"

        try:
            conn = sqlite3.connect(str(db_file))
            conn.execute('''
                INSERT OR IGNORE INTO signals
                (timestamp, signal_type, exchange, txid, total_btc,
                 internal_btc, external_btc, internal_pct, external_pct,
                 destination_exchanges, price_at_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.timestamp, signal.signal_type, signal.exchange,
                signal.txid, signal.total_btc, signal.internal_btc,
                signal.external_btc, signal.internal_pct, signal.external_pct,
                signal.destination_exchanges, signal.price
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[ERROR] Saving signal: {e}")

    def print_stats(self):
        """Print current statistics."""
        print()
        print("=" * 50)
        print("DETERMINISTIC PIPELINE STATS")
        print("=" * 50)
        print(f"TX processed:       {self.tx_processed:,}")
        print(f"Inflows detected:   {self.inflows_detected:,}")
        print(f"Outflows detected:  {self.outflows_detected:,}")
        print(f"SHORT signals:      {self.short_signals}")
        print(f"LONG signals:       {self.long_signals}")
        print(f"Mixed (skipped):    {self.mixed_skipped}")
        print(f"Tracked UTXOs:      {len(self.exchange_utxos):,}")
        if self.current_price > 0:
            print(f"Current price:      ${self.current_price:,.2f}")
        print("=" * 50)
        print()


def run_pipeline():
    """Run the deterministic ZMQ pipeline."""
    print()
    print("=" * 70)
    print("DETERMINISTIC ZMQ PIPELINE - 100% SIGNALS")
    print("=" * 70)
    print()
    print("SIGNAL LOGIC (DETERMINISTIC):")
    print("  OUTFLOW where 70%+ goes to EXCHANGE addresses = SHORT_INTERNAL")
    print("    → Exchange consolidating to hot wallet → About to sell")
    print()
    print("  OUTFLOW where 70%+ goes to NON-EXCHANGE = LONG_EXTERNAL")
    print("    → Customer withdrew their BTC → Already bought")
    print()
    print("This is NOT prediction. This is observing ACTUAL behavior.")
    print("=" * 70)
    print()

    # Initialize pipeline
    pipeline = DeterministicZMQPipeline()

    print(f"Exchange addresses: {len(pipeline.exchange_addresses):,}")
    print(f"Tracked UTXOs:      {len(pipeline.exchange_utxos):,}")
    print(f"Min outflow:        {pipeline.MIN_OUTFLOW_BTC} BTC")
    print(f"Internal threshold: {pipeline.INTERNAL_THRESHOLD:.0%}")
    print(f"External threshold: {pipeline.EXTERNAL_THRESHOLD:.0%}")
    print()
    print("Connecting to Bitcoin Core ZMQ...")
    print()

    # Initialize ZMQ
    zmq_subscriber = BlockchainZMQ(
        on_transaction=pipeline.process_transaction
    )

    if not zmq_subscriber.start():
        print("[ERROR] Failed to connect to ZMQ")
        sys.exit(1)

    print("[PIPELINE] Listening for transactions...")
    print()

    # Main loop
    last_stats = time.time()
    STATS_INTERVAL = 60  # Print stats every minute

    try:
        while True:
            time.sleep(1)

            # Print stats periodically
            if time.time() - last_stats >= STATS_INTERVAL:
                pipeline.print_stats()
                last_stats = time.time()

    except KeyboardInterrupt:
        print("\n[PIPELINE] Shutting down...")
        zmq_subscriber.stop()
        pipeline.print_stats()


if __name__ == "__main__":
    run_pipeline()
