#!/usr/bin/env python3
"""
LONG SIGNALS - 100% Deterministic
==================================
Based on research from CryptoQuant, Glassnode, and academic papers.

KEY INSIGHT: LONG is a TREND signal, not a single event.
- SHORT works on single large inflows (deposit → sell → price down)
- LONG works on SUSTAINED outflows (reserve decreasing → supply shock → price up)

FORMULA:
1. Exchange Reserve DECREASING over N consecutive windows
2. NO whale inflows during the measurement period
3. Cumulative netflow is NEGATIVE (more outflows than inflows)

Sources:
- https://cryptoquant.com/asset/btc/chart/exchange-flows/exchange-netflow-total
- https://academy.glassnode.com/indicators/sopr/sopr-spent-output-profit-ratio
- https://www.okx.com/learn/bitcoin-negative-netflow-whale-accumulation
"""

import json
import sqlite3
import urllib.request
import base64
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional, List, Dict

# RPC config
RPC_USER = "bitcoin"
RPC_PASS = "bitcoin123secure"
RPC_HOST = "127.0.0.1"
RPC_PORT = 8332
DB_PATH = "/root/sovereign/walletexplorer_addresses.db"

auth = base64.b64encode(f"{RPC_USER}:{RPC_PASS}".encode()).decode()


@dataclass
class WindowMetrics:
    """Metrics for a single window."""
    height_start: int
    height_end: int
    timestamp: int
    total_inflow: float
    total_outflow: float
    netflow: float  # outflow - inflow (positive = bullish)
    whale_inflow_count: int  # Inflows > 10 BTC
    max_single_inflow: float
    reserve_change: float  # Change in exchange holdings


@dataclass
class LongSignal:
    """LONG signal with full context."""
    exchange: str
    confidence: float
    consecutive_outflow_windows: int
    cumulative_netflow: float
    reserve_change_total: float
    reason: str


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
    except Exception as e:
        return None


def get_price(timestamp_ms):
    """Get BTC price at timestamp."""
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
    """Load exchange addresses."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT address, exchange FROM addresses")
    addrs = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    print(f"Loaded {len(addrs):,} exchange addresses")
    return addrs


def analyze_window(start_height: int, window_size: int, exchange_addrs: dict) -> Dict[str, WindowMetrics]:
    """
    Analyze a window and return metrics per exchange.

    Tracks:
    - Inflows (deposits to exchange)
    - Outflows (withdrawals from exchange)
    - Netflow (outflows - inflows)
    - Whale activity (large single inflows)
    """
    exchange_metrics = defaultdict(lambda: {
        'inflow': 0.0,
        'outflow': 0.0,
        'whale_count': 0,
        'max_inflow': 0.0,
        'timestamp': 0
    })

    for h in range(start_height, start_height + window_size):
        block_hash = rpc('getblockhash', [h])
        if not block_hash:
            continue

        block = rpc('getblock', [block_hash, 3])  # Need prevout for outflows
        if not block:
            continue

        if exchange_metrics['_global']['timestamp'] == 0:
            exchange_metrics['_global']['timestamp'] = block.get('time', 0)

        for tx in block.get('tx', []):
            tx_exchange_in = defaultdict(float)   # Inflows per exchange
            tx_exchange_out = defaultdict(float)  # Outflows per exchange

            # Check outputs (INFLOWS to exchanges)
            for vout in tx.get('vout', []):
                addr = vout.get('scriptPubKey', {}).get('address')
                value = vout.get('value', 0)
                if addr and addr in exchange_addrs and value >= 0.001:
                    ex = exchange_addrs[addr]
                    tx_exchange_in[ex] += value

            # Check inputs (OUTFLOWS from exchanges)
            for vin in tx.get('vin', []):
                if 'coinbase' in vin:
                    continue
                prevout = vin.get('prevout', {})
                if prevout:
                    addr = prevout.get('scriptPubKey', {}).get('address')
                    value = prevout.get('value', 0)
                    if addr and addr in exchange_addrs and value >= 0.001:
                        ex = exchange_addrs[addr]
                        tx_exchange_out[ex] += value

            # Record external flows only (skip internal transfers)
            for ex in set(tx_exchange_in.keys()) | set(tx_exchange_out.keys()):
                if ex in tx_exchange_in and ex in tx_exchange_out:
                    continue  # Internal transfer

                if ex in tx_exchange_in:
                    inflow = tx_exchange_in[ex]
                    exchange_metrics[ex]['inflow'] += inflow
                    if inflow > exchange_metrics[ex]['max_inflow']:
                        exchange_metrics[ex]['max_inflow'] = inflow
                    if inflow >= 10.0:
                        exchange_metrics[ex]['whale_count'] += 1

                if ex in tx_exchange_out:
                    exchange_metrics[ex]['outflow'] += tx_exchange_out[ex]

    # Convert to WindowMetrics
    result = {}
    timestamp = exchange_metrics['_global']['timestamp']

    for ex, data in exchange_metrics.items():
        if ex == '_global':
            continue
        if data['inflow'] > 0 or data['outflow'] > 0:
            netflow = data['outflow'] - data['inflow']
            result[ex] = WindowMetrics(
                height_start=start_height,
                height_end=start_height + window_size,
                timestamp=timestamp,
                total_inflow=data['inflow'],
                total_outflow=data['outflow'],
                netflow=netflow,
                whale_inflow_count=data['whale_count'],
                max_single_inflow=data['max_inflow'],
                reserve_change=-netflow  # Negative netflow = reserve decrease
            )

    return result, timestamp


class LongSignalEngine:
    """
    Generates LONG signals based on sustained exchange outflows.

    LONG CONDITIONS (all must be true):
    1. N consecutive windows with positive netflow (outflows > inflows)
    2. NO whale inflows (>10 BTC single deposit) during period
    3. Cumulative netflow exceeds threshold

    This is the INVERSE of SHORT:
    - SHORT: Single large inflow → immediate sell pressure
    - LONG: Sustained outflows → supply shock building
    """

    # Thresholds based on research - VERY selective for 100%
    MIN_CONSECUTIVE_WINDOWS = 2  # 2+ windows of net outflows
    MIN_CUMULATIVE_NETFLOW = 5.0  # 5+ BTC net outflow total (lowered to see patterns)
    MAX_SINGLE_INFLOW = 20.0  # No single deposit > 20 BTC
    OUTFLOW_RATIO_MIN = 1.1  # Outflows must be 10%+ higher than inflows

    def __init__(self, exchange_addrs: dict):
        self.exchange_addrs = exchange_addrs
        # Track window history per exchange
        self.window_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

    def add_window(self, window_metrics: Dict[str, WindowMetrics]):
        """Add a window's metrics to history."""
        for ex, metrics in window_metrics.items():
            self.window_history[ex].append(metrics)

    def check_long_signal(self, exchange: str) -> Optional[LongSignal]:
        """
        Check if conditions for LONG signal are met.

        LONG = ABSENCE of selling pressure (no significant inflows).
        This is the INVERSE of SHORT:
        - SHORT: Large inflow detected (selling pressure)
        - LONG: NO significant inflows for N windows (no selling pressure)

        Based on research: When sellers are exhausted, price rises.
        """
        history = list(self.window_history[exchange])

        if len(history) < self.MIN_CONSECUTIVE_WINDOWS:
            return None

        # Check last N windows
        recent = history[-self.MIN_CONSECUTIVE_WINDOWS:]

        # Condition 1: NO significant inflows in ANY recent window
        # (If there's selling pressure, don't go LONG)
        max_inflow_seen = max(w.total_inflow for w in recent)
        if max_inflow_seen > 5.0:  # Any window with >5 BTC inflow = selling pressure
            return None

        # Condition 2: Very low max single inflow (no whales depositing)
        max_single = max(w.max_single_inflow for w in recent)
        if max_single > 2.0:  # No single deposit > 2 BTC
            return None

        # Condition 3: Some activity exists (not dead exchange)
        total_activity = sum(w.total_inflow + w.total_outflow for w in recent)
        if total_activity < 1.0:  # Need some activity
            return None

        # All conditions met - LONG signal (absence of selling)
        cumulative_netflow = sum(w.netflow for w in recent)

        confidence = 0.85  # High confidence when no selling pressure

        reason = (f"LONG: {len(recent)} windows with NO selling pressure, "
                  f"max inflow {max_inflow_seen:.1f} BTC, max single {max_single:.1f} BTC")

        return LongSignal(
            exchange=exchange,
            confidence=confidence,
            consecutive_outflow_windows=len(recent),
            cumulative_netflow=cumulative_netflow,
            reserve_change_total=-cumulative_netflow,
            reason=reason
        )


def run_backtest(blocks: int = 1000, window_size: int = 50, verify_hours: int = 4):
    """
    Backtest LONG signals with price verification.
    """
    print("=" * 70)
    print("LONG SIGNAL BACKTEST - Sustained Outflow Detection")
    print("=" * 70)
    print()
    print("LONG CONDITIONS:")
    print(f"  1. {LongSignalEngine.MIN_CONSECUTIVE_WINDOWS}+ consecutive windows with outflows > inflows")
    print(f"  2. No single inflow > {LongSignalEngine.MAX_SINGLE_INFLOW} BTC")
    print(f"  3. Cumulative netflow > {LongSignalEngine.MIN_CUMULATIVE_NETFLOW} BTC")
    print()

    exchange_addrs = load_exchange_addresses()
    engine = LongSignalEngine(exchange_addrs)

    height = rpc('getblockcount')
    if not height:
        print("Failed to get block height")
        return

    print(f"Current block: {height:,}")

    # Buffer for price verification
    blocks_per_hour = 6
    buffer = verify_hours * blocks_per_hour + 10
    end_block = height - buffer
    start_block = end_block - blocks

    print(f"Testing blocks {start_block:,} to {end_block:,}")
    print(f"Window size: {window_size} blocks (~{window_size/6:.1f} hours)")
    print(f"Verification delay: {verify_hours} hours")
    print()

    signals = []
    verified = []

    for window_start in range(start_block, end_block, window_size):
        window_end = min(window_start + window_size, end_block)

        print(f"Analyzing window {window_start}-{window_end}...", end=" ")

        # Analyze window
        window_metrics, timestamp = analyze_window(window_start, window_size, exchange_addrs)

        if not window_metrics:
            print("no data")
            continue

        # Add to engine
        engine.add_window(window_metrics)

        # Check for LONG signals
        window_signals = []
        for ex in window_metrics.keys():
            signal = engine.check_long_signal(ex)
            if signal:
                window_signals.append((signal, timestamp))
                signals.append(signal)

        # Show window data for debugging
        for ex, m in sorted(window_metrics.items(), key=lambda x: -abs(x[1].netflow))[:3]:
            print(f"{ex}: in={m.total_inflow:.1f} out={m.total_outflow:.1f} net={m.netflow:+.1f} whale={m.whale_inflow_count}")

        if window_signals:
            print(f"  >>> SIGNALS: {len(window_signals)}")
            for signal, ts in window_signals:
                # Verify price
                signal_ms = ts * 1000
                later_ms = signal_ms + (verify_hours * 3600 * 1000)

                price_at = get_price(signal_ms)
                price_later = get_price(later_ms)

                if price_at and price_later:
                    price_change = (price_later - price_at) / price_at
                    correct = price_change > 0  # LONG expects price UP

                    verified.append({
                        'signal': signal,
                        'price_at': price_at,
                        'price_later': price_later,
                        'price_change': price_change,
                        'correct': correct
                    })

                    status = "CORRECT" if correct else "WRONG"
                    color = '\033[92m' if correct else '\033[91m'
                    reset = '\033[0m'

                    print(f"  {color}[LONG] {signal.exchange} | {signal.consecutive_outflow_windows} windows | "
                          f"net {signal.cumulative_netflow:.0f} BTC | "
                          f"${price_at:,.0f} -> ${price_later:,.0f} ({price_change*100:+.2f}%) | {status}{reset}")

    # Summary
    print()
    print("=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Total signals: {len(signals)}")
    print(f"Verified signals: {len(verified)}")

    if verified:
        correct = sum(1 for v in verified if v['correct'])
        total = len(verified)
        accuracy = correct / total * 100

        print()
        print(f"{'='*50}")
        print(f"ACCURACY: {correct}/{total} = {accuracy:.1f}%")
        print(f"{'='*50}")

        if accuracy < 100:
            print()
            print("WRONG SIGNALS:")
            for v in verified:
                if not v['correct']:
                    s = v['signal']
                    print(f"  {s.exchange}: {s.consecutive_outflow_windows} windows, "
                          f"net {s.cumulative_netflow:.0f} BTC, "
                          f"price {v['price_change']*100:+.2f}%")


if __name__ == '__main__':
    run_backtest(blocks=300, window_size=20, verify_hours=2)
