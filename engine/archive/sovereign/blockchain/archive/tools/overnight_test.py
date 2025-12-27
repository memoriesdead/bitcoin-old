#!/usr/bin/env python3
"""
OVERNIGHT DETERMINISTIC SIGNAL TEST
====================================
Runs for N hours, tracks LONG/SHORT signals with price verification.

Every signal is 100% deterministic:
- INFLOW to exchange -> SHORT (they will sell)
- OUTFLOW from exchange -> LONG (they are accumulating)

Tracks price at signal time and 60s/5min after to measure accuracy.

Run on VPS:
    cd /root/sovereign && python3 blockchain/overnight_test.py 8
"""

import time
import json
import sys
import os
import requests
from datetime import datetime
from typing import Dict, List, Set

sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')

from zmq_subscriber import BlockchainZMQ
from tx_decoder import TransactionDecoder
from address_collector import CompleteAddressDatabase, KNOWN_COLD_WALLETS
from exchange_utxo_cache import ExchangeUTXOCache, FlowDetectorWithCache


def get_btc_price():
    """Get current BTC price from multiple sources."""
    try:
        r = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT', timeout=5)
        return float(r.json()['price'])
    except:
        try:
            r = requests.get('https://api.coinbase.com/v2/prices/BTC-USD/spot', timeout=5)
            return float(r.json()['data']['amount'])
        except:
            return None


class OvernightSignalTest:
    """Overnight test with price tracking."""

    def __init__(self, duration_hours: int = 8):
        self.duration = duration_hours * 3600
        self.start_time = 0
        self.last_price_check = 0

        # Load address database
        print("=" * 70)
        print("OVERNIGHT DETERMINISTIC SIGNAL TEST")
        print("=" * 70)
        print(f"Duration: {duration_hours} hours")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        self.db = CompleteAddressDatabase()
        self.db.load_cold_wallets()
        try:
            self.db.load_existing()
        except Exception as e:
            print(f"[WARN] Could not load exchanges.json: {e}")

        self.exchange_addresses = set(self.db.address_to_exchange.keys())
        self.address_to_exchange = self.db.address_to_exchange
        print(f"Loaded {len(self.exchange_addresses):,} exchange addresses")

        # Initialize flow detector
        self.flow_detector = FlowDetectorWithCache(
            exchange_addresses=self.exchange_addresses,
            address_to_exchange=self.address_to_exchange,
            cache_path="/root/sovereign/exchange_utxos.db"
        )

        self.decoder = TransactionDecoder()

        # Results tracking
        self.signals: List[Dict] = []
        self.tx_count = 0
        self.long_count = 0
        self.short_count = 0
        self.current_price = get_btc_price()

        # Per-exchange stats
        self.exchange_stats: Dict[str, Dict] = {}

        # Price verification queue (signals to check after delay)
        self.pending_verification: List[Dict] = []

        # ZMQ
        self.zmq = BlockchainZMQ(
            rawtx_endpoint="tcp://127.0.0.1:28332",
            on_transaction=self._on_transaction
        )

    def _on_transaction(self, raw_tx: bytes):
        self.tx_count += 1

        try:
            tx = self.decoder.decode(raw_tx)
            if not tx:
                return
        except:
            return

        result = self.flow_detector.process_transaction(tx)

        if result['inflow'] == 0 and result['outflow'] == 0:
            return
        if result['direction'] == 0:
            return

        # Get current price for this signal
        price_now = self.current_price or get_btc_price()

        direction = result['signal']
        if direction == 'LONG':
            self.long_count += 1
        elif direction == 'SHORT':
            self.short_count += 1

        # Record signal with price
        signal = {
            'timestamp': datetime.now().isoformat(),
            'time': datetime.now().strftime('%H:%M:%S'),
            'txid': tx.get('txid', '')[:16],
            'direction': direction,
            'net_flow': round(result['net_flow'], 4),
            'inflow': round(result['inflow'], 4),
            'outflow': round(result['outflow'], 4),
            'exchanges': result['exchanges'],
            'price_at_signal': price_now,
            'price_60s': None,
            'price_5min': None,
            'correct_60s': None,
            'correct_5min': None,
        }
        self.signals.append(signal)

        # Track per-exchange
        for ex in result['exchanges']:
            if ex not in self.exchange_stats:
                self.exchange_stats[ex] = {
                    'long': 0, 'short': 0,
                    'inflow_btc': 0, 'outflow_btc': 0,
                    'correct_60s': 0, 'correct_5min': 0,
                    'total_verified': 0
                }
            if direction == 'LONG':
                self.exchange_stats[ex]['long'] += 1
                self.exchange_stats[ex]['outflow_btc'] += result['outflow']
            else:
                self.exchange_stats[ex]['short'] += 1
                self.exchange_stats[ex]['inflow_btc'] += result['inflow']

        # Add to verification queue
        self.pending_verification.append({
            'signal_idx': len(self.signals) - 1,
            'signal_time': time.time(),
            'direction': direction,
            'price_at_signal': price_now,
        })

        # Print signal
        print(f"[{signal['time']}] {direction:5} | Net: {result['net_flow']:+8.4f} BTC | "
              f"Price: ${price_now:,.0f} | {', '.join(result['exchanges'])}")

    def _verify_prices(self):
        """Check price movement for signals after delay."""
        now = time.time()
        current_price = get_btc_price()
        if not current_price:
            return

        self.current_price = current_price

        still_pending = []
        for pending in self.pending_verification:
            elapsed = now - pending['signal_time']
            signal = self.signals[pending['signal_idx']]

            # 60 second check
            if elapsed >= 60 and signal['price_60s'] is None:
                signal['price_60s'] = current_price
                price_change = current_price - pending['price_at_signal']

                # LONG expects price UP, SHORT expects price DOWN
                if pending['direction'] == 'LONG':
                    signal['correct_60s'] = price_change > 0
                else:
                    signal['correct_60s'] = price_change < 0

                for ex in signal['exchanges']:
                    if ex in self.exchange_stats:
                        if signal['correct_60s']:
                            self.exchange_stats[ex]['correct_60s'] += 1
                        self.exchange_stats[ex]['total_verified'] += 1

            # 5 minute check
            if elapsed >= 300 and signal['price_5min'] is None:
                signal['price_5min'] = current_price
                price_change = current_price - pending['price_at_signal']

                if pending['direction'] == 'LONG':
                    signal['correct_5min'] = price_change > 0
                else:
                    signal['correct_5min'] = price_change < 0

                for ex in signal['exchanges']:
                    if ex in self.exchange_stats:
                        if signal['correct_5min']:
                            self.exchange_stats[ex]['correct_5min'] += 1

            # Keep if not fully verified
            if signal['price_5min'] is None:
                still_pending.append(pending)

        self.pending_verification = still_pending

    def run(self):
        print()
        print(f"UTXO Cache: {self.flow_detector.utxo_cache.get_stats()}")
        print()
        print("Waiting for transactions...")
        print("-" * 70)

        self.start_time = time.time()
        self.zmq.start()

        try:
            while time.time() - self.start_time < self.duration:
                time.sleep(10)

                # Verify pending prices
                self._verify_prices()

                # Progress every 10 minutes
                elapsed = int(time.time() - self.start_time)
                if elapsed > 0 and elapsed % 600 == 0:
                    hours_left = (self.duration - elapsed) / 3600
                    self._print_progress(hours_left)

        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
        finally:
            self.zmq.stop()

        self._print_summary()
        self._save_results()

    def _print_progress(self, hours_left: float):
        stats = self.flow_detector.get_stats()
        verified = sum(1 for s in self.signals if s['correct_60s'] is not None)
        correct = sum(1 for s in self.signals if s.get('correct_60s') == True)
        accuracy = (correct / verified * 100) if verified > 0 else 0

        print(f"\n{'='*70}")
        print(f"[{hours_left:.1f}h remaining] TXs: {self.tx_count:,} | "
              f"LONG: {self.long_count} | SHORT: {self.short_count} | "
              f"Accuracy: {accuracy:.1f}% ({correct}/{verified})")
        print(f"{'='*70}\n")

    def _print_summary(self):
        elapsed = time.time() - self.start_time
        stats = self.flow_detector.get_stats()

        verified = sum(1 for s in self.signals if s['correct_60s'] is not None)
        correct_60s = sum(1 for s in self.signals if s.get('correct_60s') == True)
        correct_5min = sum(1 for s in self.signals if s.get('correct_5min') == True)

        print()
        print("=" * 70)
        print("OVERNIGHT TEST COMPLETE - RESULTS")
        print("=" * 70)
        print(f"Duration: {elapsed/3600:.2f} hours")
        print(f"Transactions: {self.tx_count:,}")
        print()
        print(f"DETERMINISTIC SIGNALS:")
        print(f"  LONG:  {self.long_count} (outflows from exchanges)")
        print(f"  SHORT: {self.short_count} (inflows to exchanges)")
        print(f"  Total: {self.long_count + self.short_count}")
        print()
        print(f"FLOW AMOUNTS:")
        print(f"  Total INFLOW:  {stats['inflow_btc']:,.2f} BTC")
        print(f"  Total OUTFLOW: {stats['outflow_btc']:,.2f} BTC")
        print(f"  NET FLOW:      {stats['net_btc']:+,.2f} BTC")
        print()
        print(f"PRICE PREDICTION ACCURACY:")
        if verified > 0:
            print(f"  60-second:  {correct_60s}/{verified} = {correct_60s/verified*100:.1f}%")
            print(f"  5-minute:   {correct_5min}/{verified} = {correct_5min/verified*100:.1f}%")
        else:
            print(f"  No verified signals yet")
        print()
        print(f"PER-EXCHANGE BREAKDOWN:")
        print(f"  {'Exchange':<12} {'LONG':>6} {'SHORT':>6} {'Outflow':>12} {'Inflow':>12} {'Accuracy':>10}")
        print("-" * 65)
        for ex, s in sorted(self.exchange_stats.items(), key=lambda x: -(x[1]['outflow_btc']+x[1]['inflow_btc'])):
            acc = (s['correct_60s']/s['total_verified']*100) if s['total_verified'] > 0 else 0
            print(f"  {ex:<12} {s['long']:>6} {s['short']:>6} {s['outflow_btc']:>12.2f} {s['inflow_btc']:>12.2f} {acc:>9.1f}%")
        print()
        print("=" * 70)

    def _save_results(self):
        results = {
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'duration_hours': (time.time() - self.start_time) / 3600,
            'tx_count': self.tx_count,
            'long_count': self.long_count,
            'short_count': self.short_count,
            'total_inflow_btc': self.flow_detector.inflow_btc,
            'total_outflow_btc': self.flow_detector.outflow_btc,
            'exchange_stats': self.exchange_stats,
            'signals': self.signals,
            'utxo_cache': self.flow_detector.utxo_cache.get_stats(),
        }

        path = f'/root/sovereign/overnight_results_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {path}")


if __name__ == '__main__':
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    test = OvernightSignalTest(duration_hours=hours)
    test.run()
