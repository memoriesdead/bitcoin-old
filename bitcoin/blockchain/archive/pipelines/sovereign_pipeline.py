#!/usr/bin/env python3
"""
SOVEREIGN PIPELINE - 100% DATA COVERAGE
========================================
The complete data pipeline for deterministic trading.

Components:
1. Address Clustering - Discover exchange addresses from blockchain
2. UTXO Cache - Track every UTXO belonging to exchanges
3. Flow Detection - Detect INFLOW/OUTFLOW in real-time
4. Multi-Exchange Prices - Track price per exchange
5. Correlation Database - Track flow → price relationship

Goal: 100% data coverage = 100% win rate

Run on VPS:
    cd /root/sovereign && python3 blockchain/sovereign_pipeline.py
"""

import sys
import time
import json
import threading
from datetime import datetime
from typing import Dict, Set, Optional

sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')

try:
    from zmq_subscriber import BlockchainZMQ
    from tx_decoder import TransactionDecoder
    from address_collector import KNOWN_COLD_WALLETS
    from cluster_runner import SQLiteAddressCluster
    from exchange_utxo_cache import ExchangeUTXOCache
    from multi_price_feed import MultiExchangePriceFeed
    from correlation_db import CorrelationDatabase
except ImportError:
    from blockchain.zmq_subscriber import BlockchainZMQ
    from blockchain.tx_decoder import TransactionDecoder
    from blockchain.address_collector import KNOWN_COLD_WALLETS
    from blockchain.cluster_runner import SQLiteAddressCluster
    from blockchain.exchange_utxo_cache import ExchangeUTXOCache
    from blockchain.multi_price_feed import MultiExchangePriceFeed
    from blockchain.correlation_db import CorrelationDatabase


class SovereignPipeline:
    """
    Complete data pipeline for 100% coverage.

    Flow:
    1. ZMQ receives raw transaction
    2. TX decoder extracts inputs/outputs
    3. Clustering discovers new exchange addresses
    4. UTXO cache tracks exchange UTXOs
    5. Flow detector identifies INFLOW/OUTFLOW
    6. Correlation DB tracks price movement
    7. Patterns analyzed for causation
    """

    def __init__(self,
                 cluster_db: str = "/root/sovereign/address_clusters.db",
                 utxo_db: str = "/root/sovereign/exchange_utxos.db",
                 correlation_db: str = "/root/sovereign/correlation.db"):

        print("=" * 70)
        print("SOVEREIGN PIPELINE - 100% DATA COVERAGE")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Initialize components
        print("[1/5] Initializing address clustering...")
        self.cluster = SQLiteAddressCluster(cluster_db)

        print("[2/5] Initializing UTXO cache...")
        self.utxo_cache = ExchangeUTXOCache(utxo_db)

        print("[3/5] Initializing multi-exchange price feed...")
        self.price_feed = MultiExchangePriceFeed(refresh_interval=2.0)

        print("[4/5] Initializing correlation database...")
        self.correlation_db = CorrelationDatabase(correlation_db)

        print("[5/5] Initializing TX decoder...")
        self.decoder = TransactionDecoder()

        # Stats
        self.start_time = 0
        self.tx_count = 0
        self.flow_count = 0
        self.inflow_count = 0
        self.outflow_count = 0
        self.inflow_btc = 0.0
        self.outflow_btc = 0.0
        self.last_stats_time = 0

        # ZMQ subscriber
        self.zmq = BlockchainZMQ(
            rawtx_endpoint="tcp://127.0.0.1:28332",
            on_transaction=self._on_transaction
        )

        # Background threads
        self.running = False

    def _on_transaction(self, raw_tx: bytes):
        """Process incoming transaction through the full pipeline."""
        self.tx_count += 1

        try:
            tx = self.decoder.decode(raw_tx)
            if not tx:
                return
        except Exception:
            return

        txid = tx.get('txid', '')

        # Step 1: Process through clustering
        discovered = self.cluster.process_transaction(tx)

        # Step 2: Detect flows
        inflow = 0.0
        outflow = 0.0
        flow_exchanges = set()

        # Check INPUTS for OUTFLOWS (spending exchange UTXOs)
        for inp in tx.get('inputs', []):
            prev_txid = inp.get('prev_txid')
            prev_vout = inp.get('prev_vout')

            if prev_txid and prev_vout is not None:
                result = self.utxo_cache.spend_utxo(prev_txid, prev_vout)
                if result:
                    value_sat, exchange, address = result
                    btc = value_sat / 1e8
                    outflow += btc
                    flow_exchanges.add(exchange)

        # Check OUTPUTS for INFLOWS (to exchange addresses)
        for i, out in enumerate(tx.get('outputs', [])):
            addr = out.get('address')
            btc = out.get('btc', 0)

            if addr and self.cluster.is_exchange_address(addr):
                exchange = self.cluster.get_exchange(addr)
                inflow += btc
                flow_exchanges.add(exchange)

                # Cache this UTXO for future outflow detection
                value_sat = int(btc * 1e8)
                self.utxo_cache.add_utxo(txid, i, value_sat, exchange, addr)

        # Step 3: Record flows in correlation database
        if inflow > 0.1:  # Minimum 0.1 BTC to reduce noise
            for exchange in flow_exchanges:
                price = self.price_feed.get_price(exchange)
                self.correlation_db.record_flow(
                    exchange=exchange,
                    direction='INFLOW',
                    amount_btc=inflow,
                    txid=txid,
                    price_now=price
                )
                self.inflow_count += 1
                self.inflow_btc += inflow
                self.flow_count += 1

                # Print significant flows
                if inflow >= 1.0:
                    price_str = f"${price:,.0f}" if price else "N/A"
                    print(f"[INFLOW] {exchange} +{inflow:.4f} BTC @ {price_str} -> SHORT")

        if outflow > 0.1:
            for exchange in flow_exchanges:
                price = self.price_feed.get_price(exchange)
                self.correlation_db.record_flow(
                    exchange=exchange,
                    direction='OUTFLOW',
                    amount_btc=outflow,
                    txid=txid,
                    price_now=price
                )
                self.outflow_count += 1
                self.outflow_btc += outflow
                self.flow_count += 1

                # Print significant flows
                if outflow >= 1.0:
                    price_str = f"${price:,.0f}" if price else "N/A"
                    print(f"[OUTFLOW] {exchange} -{outflow:.4f} BTC @ {price_str} -> LONG")

    def _verification_loop(self):
        """Background thread to verify pending price correlations."""
        while self.running:
            try:
                self.correlation_db.check_pending_verifications()
            except Exception:
                pass
            time.sleep(5)

    def _print_stats(self):
        """Print current statistics."""
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600

        cluster_stats = self.cluster.get_stats()
        utxo_stats = self.utxo_cache.get_stats()

        print()
        print("=" * 70)
        print(f"SOVEREIGN PIPELINE STATS ({hours:.1f} hours)")
        print("=" * 70)

        print(f"\nTRANSACTIONS:")
        print(f"  Processed: {self.tx_count:,}")
        print(f"  Rate: {self.tx_count / max(elapsed, 1):.1f}/sec")

        print(f"\nADDRESS CLUSTERING:")
        print(f"  Total addresses: {cluster_stats['total_addresses']:,}")
        print(f"  Session discovered: {cluster_stats['session_discovered']:,}")
        print(f"  Discovery rate: {cluster_stats['session_discovered'] / max(hours, 0.01):.1f}/hour")

        print(f"\nUTXO CACHE:")
        print(f"  Total UTXOs: {utxo_stats['total_utxos']:,}")
        print(f"  Total BTC tracked: {utxo_stats['total_btc']:,.2f}")

        print(f"\nFLOW DETECTION:")
        print(f"  Total flows: {self.flow_count:,}")
        print(f"  Inflows: {self.inflow_count} ({self.inflow_btc:,.2f} BTC)")
        print(f"  Outflows: {self.outflow_count} ({self.outflow_btc:,.2f} BTC)")
        print(f"  Net flow: {self.outflow_btc - self.inflow_btc:+,.2f} BTC")

        # Per-exchange breakdown
        print(f"\nADDRESSES PER EXCHANGE:")
        for ex, count in sorted(cluster_stats['per_exchange'].items(), key=lambda x: -x[1]):
            print(f"  {ex:<15} {count:>10,}")

        # Correlation summary
        print(f"\nCORRELATION ACCURACY:")
        correlations = self.correlation_db.get_all_correlations()
        for ex, data in sorted(correlations.items()):
            if data['total_verified'] > 0:
                acc = data['overall_accuracy']
                acc_str = f"{acc:.1f}%" if acc else "-"
                print(f"  {ex:<15} {data['total_verified']:>6} verified, {acc_str:>8} accuracy")

        print()
        print("=" * 70)

    def run(self):
        """Run the complete pipeline."""
        print()
        print("Starting pipeline components...")

        # Start price feed
        self.price_feed.start()

        # Fetch initial prices
        print("Fetching initial prices...")
        prices = self.price_feed.fetch_all()
        print(f"  Got prices from {len(prices)} exchanges")

        # Start verification thread
        self.running = True
        verify_thread = threading.Thread(target=self._verification_loop, daemon=True)
        verify_thread.start()

        # Start ZMQ
        print("Connecting to Bitcoin Core ZMQ...")
        self.start_time = time.time()
        self.last_stats_time = self.start_time
        self.zmq.start()

        print()
        print("Pipeline running. Waiting for transactions...")
        print("-" * 70)

        try:
            while True:
                time.sleep(60)

                # Print stats every 10 minutes
                if time.time() - self.last_stats_time >= 600:
                    self._print_stats()
                    self.last_stats_time = time.time()

                    # Save clustering stats
                    self.cluster.save_stats()

        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
        finally:
            self.running = False
            self.zmq.stop()
            self.price_feed.stop()
            self._print_stats()

            # Print final correlation report
            self.correlation_db.print_report()

            print("\nPipeline stopped.")


def main():
    """Entry point."""
    pipeline = SovereignPipeline()
    pipeline.run()


if __name__ == '__main__':
    main()
