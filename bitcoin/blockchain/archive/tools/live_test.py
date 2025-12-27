#!/usr/bin/env python3
"""
LIVE BLOCKCHAIN SIGNAL TEST - BIDIRECTIONAL FLOW DETECTION
============================================================
Runs for N minutes, captures all LONG/SHORT signals from Bitcoin Core ZMQ.

Uses UTXO cache for complete flow detection:
- INFLOW (to exchange): Detected from transaction outputs
- OUTFLOW (from exchange): Detected when cached UTXOs are spent

Run on VPS:
    cd /root/sovereign && python3 blockchain/live_test.py [minutes]
"""

import time
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Set

# Add paths for VPS structure
sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')

# Try different import paths
try:
    from zmq_subscriber import BlockchainZMQ
    from tx_decoder import TransactionDecoder
    from address_collector import CompleteAddressDatabase, KNOWN_COLD_WALLETS
    from exchange_utxo_cache import ExchangeUTXOCache, FlowDetectorWithCache
except ImportError:
    from blockchain.zmq_subscriber import BlockchainZMQ
    from blockchain.tx_decoder import TransactionDecoder
    from blockchain.address_collector import CompleteAddressDatabase, KNOWN_COLD_WALLETS
    from blockchain.exchange_utxo_cache import ExchangeUTXOCache, FlowDetectorWithCache


class LiveSignalTest:
    """Live test of blockchain signal detection with UTXO cache."""

    def __init__(self, duration_minutes: int = 10):
        self.duration = duration_minutes * 60
        self.start_time = 0

        # Load address database
        print("=" * 60)
        print("LOADING ADDRESS DATABASE...")
        print("=" * 60)

        self.db = CompleteAddressDatabase()
        self.db.load_cold_wallets()

        # Try to load existing exchanges.json
        try:
            self.db.load_existing()
        except Exception as e:
            print(f"[WARN] Could not load exchanges.json: {e}")

        self.exchange_addresses = set(self.db.address_to_exchange.keys())
        self.address_to_exchange = self.db.address_to_exchange

        print(f"\nLoaded {len(self.exchange_addresses):,} exchange addresses")

        # Initialize flow detector with UTXO cache
        print("\n" + "=" * 60)
        print("INITIALIZING UTXO CACHE...")
        print("=" * 60)

        self.flow_detector = FlowDetectorWithCache(
            exchange_addresses=self.exchange_addresses,
            address_to_exchange=self.address_to_exchange,
            cache_path="/root/sovereign/exchange_utxos.db"
        )

        # Transaction decoder
        self.decoder = TransactionDecoder()

        # Results
        self.signals: List[Dict] = []
        self.tx_count = 0
        self.long_count = 0
        self.short_count = 0

        # ZMQ subscriber
        self.zmq = BlockchainZMQ(
            rawtx_endpoint="tcp://127.0.0.1:28332",
            on_transaction=self._on_transaction
        )

    def _on_transaction(self, raw_tx: bytes):
        """Process each transaction from ZMQ."""
        self.tx_count += 1

        # Decode transaction
        try:
            tx = self.decoder.decode(raw_tx)
            if not tx:
                return
        except Exception as e:
            return

        # Process with flow detector (handles both INFLOW and OUTFLOW)
        result = self.flow_detector.process_transaction(tx)

        # Skip if no exchange flow
        if result['inflow'] == 0 and result['outflow'] == 0:
            return

        # Skip neutral signals
        if result['direction'] == 0:
            return

        # Record signal
        direction = result['signal']
        if direction == 'LONG':
            self.long_count += 1
        elif direction == 'SHORT':
            self.short_count += 1

        signal = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'txid': tx.get('txid', '')[:16] + '...',
            'direction': direction,
            'net_flow': round(result['net_flow'], 4),
            'inflow': round(result['inflow'], 4),
            'outflow': round(result['outflow'], 4),
            'exchanges': result['exchanges'],
        }
        self.signals.append(signal)

        # Print signal
        print(f"[{signal['time']}] {direction:5} | Net: {result['net_flow']:+.4f} BTC | "
              f"In: {result['inflow']:.4f} | Out: {result['outflow']:.4f} | {', '.join(result['exchanges'])}")

    def run(self):
        """Run the test."""
        print()
        print("=" * 60)
        print(f"LIVE BLOCKCHAIN SIGNAL TEST - {self.duration // 60} MINUTES")
        print("=" * 60)
        print(f"ZMQ: tcp://127.0.0.1:28332")
        print(f"Addresses: {len(self.exchange_addresses):,}")
        print(f"UTXO Cache: {self.flow_detector.utxo_cache.get_stats()}")
        print()
        print("Waiting for transactions...")
        print("-" * 60)

        self.start_time = time.time()

        # Start ZMQ subscriber
        self.zmq.start()

        try:
            # Run for duration
            while time.time() - self.start_time < self.duration:
                time.sleep(1)

                # Progress every 60 seconds
                elapsed = int(time.time() - self.start_time)
                if elapsed > 0 and elapsed % 60 == 0:
                    remaining = (self.duration - elapsed) // 60
                    stats = self.flow_detector.get_stats()
                    print(f"\n--- {remaining} minutes remaining | "
                          f"TXs: {self.tx_count} | LONG: {self.long_count} | SHORT: {self.short_count} | "
                          f"Cache: {stats['cache']['total_utxos']} UTXOs ---\n")

        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")

        finally:
            self.zmq.stop()

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print test summary."""
        elapsed = time.time() - self.start_time
        stats = self.flow_detector.get_stats()

        print()
        print("=" * 60)
        print("TEST COMPLETE - SUMMARY")
        print("=" * 60)
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Transactions processed: {self.tx_count:,}")
        print()
        print(f"LONG signals:  {self.long_count} (outflows from exchanges)")
        print(f"SHORT signals: {self.short_count} (inflows to exchanges)")
        print(f"Total signals: {self.long_count + self.short_count}")
        print()
        print(f"Total INFLOW:  {stats['inflow_btc']:.4f} BTC (to exchanges)")
        print(f"Total OUTFLOW: {stats['outflow_btc']:.4f} BTC (from exchanges)")
        print(f"NET FLOW:      {stats['net_btc']:+.4f} BTC")
        print()

        if stats['net_btc'] > 0:
            print("OVERALL SIGNAL: LONG (more BTC leaving exchanges)")
        elif stats['net_btc'] < 0:
            print("OVERALL SIGNAL: SHORT (more BTC entering exchanges)")
        else:
            print("OVERALL SIGNAL: NEUTRAL")

        print()
        print(f"UTXO Cache Stats:")
        print(f"  - Tracked UTXOs: {stats['cache']['total_utxos']:,}")
        print(f"  - Spent UTXOs: {stats['cache']['spent_utxos']:,}")
        print(f"  - Cached BTC: {stats['cache']['total_btc']:.4f}")
        print()
        print("=" * 60)

        # Save results
        results = {
            'duration_seconds': elapsed,
            'tx_count': self.tx_count,
            'long_count': self.long_count,
            'short_count': self.short_count,
            'total_inflow_btc': stats['inflow_btc'],
            'total_outflow_btc': stats['outflow_btc'],
            'net_flow_btc': stats['net_btc'],
            'utxo_cache': stats['cache'],
            'signals': self.signals,
        }

        result_path = '/root/sovereign/test_results.json'
        try:
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {result_path}")
        except Exception as e:
            print(f"Could not save results: {e}")


if __name__ == '__main__':
    # Default 10 minutes, or pass argument
    minutes = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    test = LiveSignalTest(duration_minutes=minutes)
    test.run()
