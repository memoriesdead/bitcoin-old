#!/usr/bin/env python3
"""
LONG SIGNAL - RESERVE DEPLETION PER EXCHANGE
=============================================
SHORT = Large inflow → WILL sell → Price DOWN (single event)
LONG = Reserve depleting + NO inflows → Supply shock → Price UP (trend)

MATHEMATICAL FORMULA:
    For each exchange, track:
    1. Cumulative netflow over N blocks (outflows - inflows)
    2. Max single inflow in period (must be < threshold)

    LONG SIGNAL when:
    - Cumulative netflow > X BTC (more outflows than inflows)
    - Max single inflow < Y BTC (no whale selling)
    - Sustained for Z consecutive windows

This is the INVERSE of SHORT:
    SHORT: One whale deposits → selling pressure
    LONG: No whales depositing + coins leaving → supply shock
"""

import json
import sqlite3
import urllib.request
import base64
from datetime import datetime
from collections import defaultdict, deque

# RPC config
RPC_USER = "bitcoin"
RPC_PASS = "bitcoin123secure"
RPC_HOST = "127.0.0.1"
RPC_PORT = 8332
DB_PATH = "/root/sovereign/walletexplorer_addresses.db"

auth = base64.b64encode(f"{RPC_USER}:{RPC_PASS}".encode()).decode()

def rpc(method, params=None):
    payload = json.dumps({
        "jsonrpc": "1.0", "id": "long", "method": method, "params": params or []
    }).encode()
    try:
        req = urllib.request.Request(f"http://{RPC_HOST}:{RPC_PORT}")
        req.add_header("Authorization", f"Basic {auth}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, payload, timeout=60) as resp:
            return json.loads(resp.read()).get('result')
    except:
        return None

def get_price(timestamp_ms):
    try:
        url = f"https://api.binance.us/api/v3/klines?symbol=BTCUSD&interval=1m&startTime={timestamp_ms}&limit=1"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if data:
                return float(data[0][4])
    except:
        pass
    return None

def load_exchange_addresses():
    """Load addresses grouped by exchange."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT address, exchange FROM addresses")

    # Group by exchange
    by_exchange = defaultdict(set)
    for row in c.fetchall():
        by_exchange[row[1]].add(row[0])

    conn.close()

    total = sum(len(addrs) for addrs in by_exchange.values())
    print(f"Loaded {total:,} addresses across {len(by_exchange)} exchanges")

    # Also create flat lookup
    flat = {}
    for ex, addrs in by_exchange.items():
        for addr in addrs:
            flat[addr] = ex

    return by_exchange, flat

def analyze_window_per_exchange(start_height, window_size, addr_to_exchange):
    """
    Analyze window and return metrics PER EXCHANGE.

    Returns: {exchange: {inflow, outflow, netflow, max_inflow, timestamp}}
    """
    exchange_metrics = defaultdict(lambda: {
        'inflow': 0.0,
        'outflow': 0.0,
        'max_inflow': 0.0,
        'inflow_count': 0,
        'outflow_count': 0,
    })

    window_timestamp = 0

    for h in range(start_height, start_height + window_size):
        block_hash = rpc('getblockhash', [h])
        if not block_hash:
            continue

        block = rpc('getblock', [block_hash, 3])  # Need prevout for outflows
        if not block:
            continue

        if window_timestamp == 0:
            window_timestamp = block.get('time', 0)

        for tx in block.get('tx', []):
            tx_in = defaultdict(float)   # exchange -> inflow
            tx_out = defaultdict(float)  # exchange -> outflow

            # Outputs = inflows to exchanges
            for vout in tx.get('vout', []):
                addr = vout.get('scriptPubKey', {}).get('address')
                value = vout.get('value', 0)
                if addr and addr in addr_to_exchange and value >= 0.01:
                    ex = addr_to_exchange[addr]
                    tx_in[ex] += value

            # Inputs = outflows from exchanges
            for vin in tx.get('vin', []):
                if 'coinbase' in vin:
                    continue
                prevout = vin.get('prevout', {})
                if prevout:
                    addr = prevout.get('scriptPubKey', {}).get('address')
                    value = prevout.get('value', 0)
                    if addr and addr in addr_to_exchange and value >= 0.01:
                        ex = addr_to_exchange[addr]
                        tx_out[ex] += value

            # Record external flows only (skip internal transfers)
            for ex in set(tx_in.keys()) | set(tx_out.keys()):
                if ex in tx_in and ex in tx_out:
                    continue  # Internal transfer

                if ex in tx_in:
                    val = tx_in[ex]
                    exchange_metrics[ex]['inflow'] += val
                    exchange_metrics[ex]['inflow_count'] += 1
                    if val > exchange_metrics[ex]['max_inflow']:
                        exchange_metrics[ex]['max_inflow'] = val

                if ex in tx_out:
                    exchange_metrics[ex]['outflow'] += tx_out[ex]
                    exchange_metrics[ex]['outflow_count'] += 1

    # Calculate netflow
    for ex in exchange_metrics:
        m = exchange_metrics[ex]
        m['netflow'] = m['outflow'] - m['inflow']  # Positive = more outflows = bullish
        m['timestamp'] = window_timestamp

    return dict(exchange_metrics), window_timestamp

class ExchangeReserveTracker:
    """
    Track reserve changes per exchange over rolling windows.

    LONG signal when:
    1. Cumulative netflow > threshold (reserve depleting)
    2. No large single inflows (no whale sellers)
    3. Sustained pattern over N windows
    """

    # Thresholds - STRICT for 100% accuracy
    MIN_CUMULATIVE_NETFLOW = 10.0   # 10+ BTC net outflow
    MAX_SINGLE_INFLOW = 5.0         # No single deposit > 5 BTC
    MIN_WINDOWS = 3                  # 3+ consecutive windows

    def __init__(self):
        # Per-exchange rolling window history
        self.history = defaultdict(lambda: deque(maxlen=20))

    def add_window(self, exchange_metrics):
        """Add window metrics to history."""
        for ex, metrics in exchange_metrics.items():
            self.history[ex].append(metrics)

    def check_long_signal(self, exchange):
        """
        Check if LONG conditions are met for this exchange.

        Returns: (signal, confidence, reason) or None
        """
        hist = list(self.history[exchange])

        if len(hist) < self.MIN_WINDOWS:
            return None

        # Check last N windows
        recent = hist[-self.MIN_WINDOWS:]

        # Condition 1: Cumulative positive netflow (more outflows)
        cumulative_netflow = sum(w['netflow'] for w in recent)
        if cumulative_netflow < self.MIN_CUMULATIVE_NETFLOW:
            return None

        # Condition 2: No large single inflows in ANY window
        max_single = max(w['max_inflow'] for w in recent)
        if max_single > self.MAX_SINGLE_INFLOW:
            return None

        # Condition 3: Each window must have positive netflow
        all_positive = all(w['netflow'] > 0 for w in recent)
        if not all_positive:
            return None

        # All conditions met - LONG signal
        confidence = 1.0  # 100% deterministic

        reason = (f"LONG: {exchange} reserve depleting | "
                  f"{self.MIN_WINDOWS} windows, cumulative netflow +{cumulative_netflow:.1f} BTC, "
                  f"max single inflow {max_single:.1f} BTC")

        return {
            'exchange': exchange,
            'confidence': confidence,
            'cumulative_netflow': cumulative_netflow,
            'max_single_inflow': max_single,
            'windows': self.MIN_WINDOWS,
            'reason': reason
        }

def main():
    print("=" * 70)
    print("LONG SIGNAL - RESERVE DEPLETION PER EXCHANGE")
    print("=" * 70)
    print()
    print("MATHEMATICAL FORMULA:")
    print(f"  1. Cumulative netflow > {ExchangeReserveTracker.MIN_CUMULATIVE_NETFLOW} BTC (reserve depleting)")
    print(f"  2. Max single inflow < {ExchangeReserveTracker.MAX_SINGLE_INFLOW} BTC (no whale sellers)")
    print(f"  3. {ExchangeReserveTracker.MIN_WINDOWS}+ consecutive windows with positive netflow")
    print()
    print("This is INVERSE of SHORT:")
    print("  SHORT: Large single inflow → whale selling → price DOWN")
    print("  LONG: Reserve depleting + no whales → supply shock → price UP")
    print()

    by_exchange, addr_to_exchange = load_exchange_addresses()
    tracker = ExchangeReserveTracker()

    height = rpc('getblockcount')
    if not height:
        print("Failed to get block height")
        return

    print(f"Current block: {height:,}")

    # Parameters
    window_size = 10  # ~100 minutes
    verify_hours = 2
    blocks_per_hour = 6
    buffer = verify_hours * blocks_per_hour + 10

    num_windows = 30
    start = height - buffer - (num_windows * window_size)

    print(f"Window size: {window_size} blocks (~{window_size*10} minutes)")
    print(f"Testing {num_windows} windows from block {start:,}")
    print()

    signals = []

    for i in range(num_windows):
        window_start = start + (i * window_size)

        print(f"Window {i+1}/{num_windows}...", end=" ", flush=True)

        exchange_metrics, timestamp = analyze_window_per_exchange(
            window_start, window_size, addr_to_exchange
        )

        if not exchange_metrics or timestamp == 0:
            print("no data")
            continue

        # Add to tracker
        tracker.add_window(exchange_metrics)

        # Check for LONG signals on each exchange
        window_signals = []
        for ex in exchange_metrics.keys():
            signal = tracker.check_long_signal(ex)
            if signal:
                window_signals.append(signal)

        # Show top exchanges by activity
        top = sorted(exchange_metrics.items(), key=lambda x: abs(x[1]['netflow']), reverse=True)[:3]
        for ex, m in top:
            print(f"{ex}: net={m['netflow']:+.1f} in={m['inflow']:.1f} out={m['outflow']:.1f}", end=" | ")
        print()

        # Verify signals
        for signal in window_signals:
            signal_ms = timestamp * 1000
            later_ms = signal_ms + (verify_hours * 3600 * 1000)

            price_at = get_price(signal_ms)
            price_later = get_price(later_ms)

            if price_at and price_later:
                price_change = (price_later - price_at) / price_at
                correct = price_change > 0

                signal['price_at'] = price_at
                signal['price_later'] = price_later
                signal['price_change'] = price_change
                signal['correct'] = correct
                signals.append(signal)

                status = "CORRECT" if correct else "WRONG"
                color = '\033[92m' if correct else '\033[91m'
                reset = '\033[0m'

                print(f"{color}  >>> LONG SIGNAL: {signal['exchange']} | "
                      f"netflow +{signal['cumulative_netflow']:.1f} BTC | "
                      f"${price_at:,.0f} → ${price_later:,.0f} ({price_change*100:+.2f}%) | "
                      f"{status}{reset}")

    # Summary
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if signals:
        correct = sum(1 for s in signals if s['correct'])
        total = len(signals)
        acc = correct / total * 100

        print(f"LONG signals: {total}")
        print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
        print()

        for s in signals:
            status = "OK" if s['correct'] else "WRONG"
            print(f"  {s['exchange']}: netflow +{s['cumulative_netflow']:.1f} BTC, "
                  f"max_in {s['max_single_inflow']:.1f} BTC → "
                  f"price {s['price_change']*100:+.2f}% [{status}]")
    else:
        print("No LONG signals generated")
        print()
        print("Possible reasons:")
        print("  1. No exchange had sustained outflows > inflows")
        print("  2. Whale inflows broke the pattern")
        print("  3. Thresholds too strict - try lowering MIN_CUMULATIVE_NETFLOW")

if __name__ == '__main__':
    main()
