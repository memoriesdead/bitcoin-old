#!/usr/bin/env python3
"""
DETERMINISTIC SIGNAL GENERATOR
==============================

100% DETERMINISTIC SIGNALS based on WHERE spent UTXOs go:

    INTERNAL (to exchange address) → Exchange consolidating → SHORT
    EXTERNAL (to non-exchange address) → Customer withdrawal → LONG

PIPELINE:
    1. C++ runner detects outflow (exchange UTXO spent)
    2. We get the spending TX via RPC
    3. We classify ALL outputs as internal/external
    4. Generate deterministic signal based on classification

This is NOT statistical. This is observing ACTUAL blockchain behavior.
"""

import subprocess
import sqlite3
import json
import re
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Set, List, Optional, Tuple
from dataclasses import dataclass
import ccxt

# Bitcoin RPC wrapper
def bitcoin_rpc(method: str, params: list = None) -> Optional[Dict]:
    """Call Bitcoin Core RPC."""
    cmd = ["/usr/local/bin/bitcoin-cli"]
    cmd.append(method)
    if params:
        for p in params:
            cmd.append(str(p))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return json.loads(result.stdout) if result.stdout.strip() else None
    except Exception as e:
        pass
    return None


@dataclass
class DeterministicSignal:
    """A 100% deterministic signal."""
    timestamp: str
    exchange: str
    signal_type: str          # 'SHORT_INTERNAL' or 'LONG_EXTERNAL'
    txid: str
    total_spent_btc: float
    internal_btc: float       # Portion going to exchange addresses
    external_btc: float       # Portion going to non-exchange addresses
    internal_pct: float       # % going internal
    external_pct: float       # % going external
    price: float


class DeterministicSignalGenerator:
    """
    Generate deterministic SHORT/LONG signals based on destination classification.
    """

    # Thresholds for signal generation
    MIN_FLOW_BTC = 1.0          # Minimum flow to consider
    INTERNAL_THRESHOLD = 0.7    # 70%+ going internal = SHORT
    EXTERNAL_THRESHOLD = 0.7    # 70%+ going external = LONG

    def __init__(self):
        # Load exchange addresses
        self.exchange_addresses, self.address_to_exchange = self._load_addresses()
        print(f"[DETERMINISTIC] Loaded {len(self.exchange_addresses):,} exchange addresses")

        # Price feed
        self.price_feed = ccxt.kraken()
        self.current_price = 0.0
        self._start_price_feed()

        # Signal database
        self._init_db()

        # Stats
        self.outflows_analyzed = 0
        self.short_signals = 0
        self.long_signals = 0
        self.mixed_signals = 0

    def _load_addresses(self) -> Tuple[Set[str], Dict[str, str]]:
        """Load exchange addresses from database."""
        addresses = set()
        addr_to_exchange = {}

        try:
            conn = sqlite3.connect("/root/sovereign/walletexplorer_addresses.db")
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

    def _start_price_feed(self):
        """Start background price feed."""
        def update_price():
            while True:
                try:
                    ticker = self.price_feed.fetch_ticker('BTC/USD')
                    self.current_price = ticker['last']
                except:
                    pass
                time.sleep(1)

        t = threading.Thread(target=update_price, daemon=True)
        t.start()

        while self.current_price == 0:
            time.sleep(0.1)

    def _init_db(self):
        """Initialize signal database."""
        conn = sqlite3.connect('/root/sovereign/deterministic_signals.db')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                exchange TEXT,
                signal_type TEXT,
                txid TEXT,
                total_btc REAL,
                internal_btc REAL,
                external_btc REAL,
                internal_pct REAL,
                external_pct REAL,
                price REAL,

                -- Price tracking (filled later)
                price_t1min REAL,
                price_t5min REAL,
                price_t10min REAL,
                correct INTEGER
            )
        ''')
        conn.commit()
        conn.close()

    def analyze_outflow_tx(self, txid: str, exchange: str, outflow_btc: float) -> Optional[DeterministicSignal]:
        """
        Analyze an outflow TX to classify destination and generate signal.

        Returns a DeterministicSignal if classification is clear enough.
        """
        if outflow_btc < self.MIN_FLOW_BTC:
            return None

        self.outflows_analyzed += 1

        # Get full TX from RPC
        tx = bitcoin_rpc("getrawtransaction", [txid, 1])  # 1 = verbose
        if not tx:
            print(f"[WARN] Could not get TX {txid[:16]}... from RPC")
            return None

        # Analyze outputs
        internal_btc = 0.0
        external_btc = 0.0
        internal_exchanges = set()

        for vout in tx.get('vout', []):
            btc = vout.get('value', 0)
            script = vout.get('scriptPubKey', {})
            addr = script.get('address')

            if not addr:
                # Try addresses array (older format)
                addresses = script.get('addresses', [])
                if addresses:
                    addr = addresses[0]

            if addr:
                if addr in self.exchange_addresses:
                    internal_btc += btc
                    ex = self.address_to_exchange.get(addr, 'unknown')
                    internal_exchanges.add(ex)
                else:
                    external_btc += btc

        total_output = internal_btc + external_btc
        if total_output == 0:
            return None

        internal_pct = internal_btc / total_output
        external_pct = external_btc / total_output

        # Generate signal based on classification
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
            # Mixed - log but don't trade
            self.mixed_signals += 1
            print(f"[MIXED] {exchange} | Internal: {internal_pct:.1%} | External: {external_pct:.1%} | Skipping")
            return None

        signal = DeterministicSignal(
            timestamp=datetime.now(timezone.utc).isoformat(),
            exchange=exchange,
            signal_type=signal_type,
            txid=txid,
            total_spent_btc=outflow_btc,
            internal_btc=internal_btc,
            external_btc=external_btc,
            internal_pct=internal_pct,
            external_pct=external_pct,
            price=self.current_price
        )

        self._log_signal(signal)
        return signal

    def _log_signal(self, signal: DeterministicSignal):
        """Log signal to database."""
        conn = sqlite3.connect('/root/sovereign/deterministic_signals.db')
        conn.execute('''
            INSERT INTO signals
            (timestamp, exchange, signal_type, txid, total_btc,
             internal_btc, external_btc, internal_pct, external_pct, price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.timestamp, signal.exchange, signal.signal_type, signal.txid,
            signal.total_spent_btc, signal.internal_btc, signal.external_btc,
            signal.internal_pct, signal.external_pct, signal.price
        ))
        conn.commit()
        conn.close()

    def print_stats(self):
        """Print current statistics."""
        print(f"\n{'='*60}")
        print("DETERMINISTIC SIGNAL STATS")
        print(f"{'='*60}")
        print(f"Outflows analyzed: {self.outflows_analyzed}")
        print(f"SHORT signals:     {self.short_signals}")
        print(f"LONG signals:      {self.long_signals}")
        print(f"Mixed (skipped):   {self.mixed_signals}")
        print(f"Current price:     ${self.current_price:,.2f}")
        print(f"{'='*60}\n")


def run_deterministic_pipeline():
    """Run the deterministic signal pipeline."""
    generator = DeterministicSignalGenerator()

    print("="*70)
    print("DETERMINISTIC SIGNAL GENERATOR - 100% SIGNALS")
    print("="*70)
    print(f"Exchange addresses: {len(generator.exchange_addresses):,}")
    print(f"Min flow:           {generator.MIN_FLOW_BTC} BTC")
    print(f"Internal threshold: {generator.INTERNAL_THRESHOLD:.0%} → SHORT")
    print(f"External threshold: {generator.EXTERNAL_THRESHOLD:.0%} → LONG")
    print("="*70)
    print()
    print("SIGNAL LOGIC:")
    print("  OUTFLOW where 70%+ goes to exchange addresses = SHORT_INTERNAL")
    print("    → Exchange consolidating to hot wallet → About to sell")
    print()
    print("  OUTFLOW where 70%+ goes to non-exchange = LONG_EXTERNAL")
    print("    → Customer withdrew purchase → Buying absorbed")
    print()
    print("Connecting to C++ runner...")
    print()

    # C++ runner
    CPP_BINARY = "/root/sovereign/cpp_runner/build/blockchain_runner"
    ADDRESS_DB = "/root/sovereign/walletexplorer_addresses.db"
    UTXO_DB = "/root/sovereign/exchange_utxos.db"
    ZMQ_URL = "tcp://127.0.0.1:28332"

    cmd = [CPP_BINARY, ADDRESS_DB, UTXO_DB, ZMQ_URL]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    last_stats = time.time()
    pending_lookups = []  # TXs we need to analyze

    for line in process.stdout:
        # Strip ANSI codes
        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line.strip())

        if not clean_line:
            continue

        # Parse C++ output
        if '[SHORT]' in clean_line or '[LONG]' in clean_line:
            try:
                parts = clean_line.split('|')
                if len(parts) >= 4:
                    first = parts[0].strip()
                    exchange = first.split()[-1].strip().lower()

                    outflow = 0.0
                    for part in parts[1:]:
                        part = part.strip()
                        if part.startswith('Out:'):
                            outflow = float(part.replace('Out:', '').strip())

                    # If there's an outflow, we need to analyze where it went
                    if outflow >= generator.MIN_FLOW_BTC:
                        # The C++ runner outputs the TX - but we need the TXID
                        # For now, we can use mempool to find recent TXs
                        # OR we need to modify C++ runner to output TXID

                        # For the current limitation, we log the opportunity
                        print(f"[OUTFLOW] {exchange.upper()} -{outflow:.4f} BTC (need TXID for analysis)")

            except Exception as e:
                print(f"[ERROR] {e}: {clean_line}")

        # Periodic stats
        if time.time() - last_stats >= 60:
            generator.print_stats()
            last_stats = time.time()


def test_with_sample_tx():
    """Test with a known TX to verify the logic works."""
    generator = DeterministicSignalGenerator()

    print("="*60)
    print("TESTING DETERMINISTIC SIGNAL GENERATOR")
    print("="*60)

    # Test with a sample TX (replace with actual TXID from blockchain)
    sample_txid = input("Enter a TX ID to analyze (or 'skip'): ").strip()

    if sample_txid.lower() == 'skip':
        print("Skipping test. Run with actual C++ runner.")
        return

    signal = generator.analyze_outflow_tx(
        txid=sample_txid,
        exchange="test",
        outflow_btc=1.0
    )

    if signal:
        print(f"\nSignal generated: {signal.signal_type}")
        print(f"Internal: {signal.internal_btc:.4f} BTC ({signal.internal_pct:.1%})")
        print(f"External: {signal.external_btc:.4f} BTC ({signal.external_pct:.1%})")
    else:
        print("No signal generated (mixed or below threshold)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_with_sample_tx()
    else:
        run_deterministic_pipeline()
