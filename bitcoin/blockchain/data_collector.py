#!/usr/bin/env python3
"""
DATA COLLECTOR FOR 100% WIN RATE DISCOVERY
===========================================

PURPOSE:
  Collect ALL available blockchain data for each exchange flow.
  Track price movements at T+0, T+1m, T+5m, T+10m.
  NO TRADING - just collect data for analysis.

AFTER 24-48 HOURS:
  Analyze which factors correlate with price movements.
  Find the 100% predictive patterns.

DATA COLLECTED PER FLOW:
  From C++ runner:
    - txid, exchange, direction, flow_btc, latency_ns

  From bitcoin-cli (for significant flows):
    - fee_sat_vb (urgency indicator)
    - input_count (consolidation signal)
    - output_count (distribution signal)
    - tx_size_bytes
    - is_segwit
    - oldest_utxo_blocks (holder conviction)
    - locktime
    - rbf_enabled (sequence < 0xFFFFFFFE)

  From price feed:
    - price_t0, price_t1m, price_t5m, price_t10m
    - price_moved_expected (did SHORT lead to drop?)
    - max_move_pct (biggest move in 10 min)
"""

import subprocess
import json
import time
import sqlite3
import re
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
from collections import deque
import ccxt


@dataclass
class FlowData:
    """Complete data for a single exchange flow."""
    # Basic flow info
    txid: str
    timestamp: str
    exchange: str
    direction: str  # INFLOW or OUTFLOW
    flow_btc: float
    latency_ns: int

    # TX metadata (from bitcoin-cli)
    fee_sat_vb: Optional[float] = None
    input_count: Optional[int] = None
    output_count: Optional[int] = None
    tx_size_bytes: Optional[int] = None
    is_segwit: Optional[bool] = None
    oldest_utxo_blocks: Optional[int] = None
    locktime: Optional[int] = None
    rbf_enabled: Optional[bool] = None

    # Price tracking
    price_t0: Optional[float] = None
    price_t1m: Optional[float] = None
    price_t5m: Optional[float] = None
    price_t10m: Optional[float] = None

    # Outcome analysis
    price_moved_expected: Optional[bool] = None  # For INFLOW: did price drop?
    max_down_move_pct: Optional[float] = None
    max_up_move_pct: Optional[float] = None


class BitcoinRPC:
    """Bitcoin Core RPC for TX metadata."""

    def __init__(self, cli_path: str = "/usr/local/bin/bitcoin-cli"):
        self.cli_path = cli_path

    def get_tx_data(self, txid: str) -> Optional[Dict]:
        """Get full TX data including fee and UTXO ages."""
        try:
            # Get raw transaction with details
            result = subprocess.run(
                [self.cli_path, "getrawtransaction", txid, "true"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return None

            tx = json.loads(result.stdout)

            # Calculate fee (sum of inputs - sum of outputs)
            input_value = 0
            oldest_blocks = 0
            current_height = self._get_block_height()

            for vin in tx.get('vin', []):
                if 'coinbase' in vin:
                    continue  # Skip coinbase inputs

                # Look up input TX to get value
                prev_txid = vin.get('txid')
                prev_vout = vin.get('vout')
                if prev_txid:
                    prev_tx = self._get_tx(prev_txid)
                    if prev_tx and 'vout' in prev_tx:
                        if prev_vout < len(prev_tx['vout']):
                            input_value += prev_tx['vout'][prev_vout].get('value', 0)

                        # Get UTXO age
                        if 'blockhash' in prev_tx:
                            block_height = self._get_block_height_for_hash(prev_tx['blockhash'])
                            if block_height and current_height:
                                age = current_height - block_height
                                oldest_blocks = max(oldest_blocks, age)

            output_value = sum(vout.get('value', 0) for vout in tx.get('vout', []))
            fee_btc = input_value - output_value

            # Calculate fee rate
            tx_size = tx.get('vsize', tx.get('size', 0))
            fee_sat = int(fee_btc * 100_000_000)
            fee_sat_vb = fee_sat / tx_size if tx_size > 0 else 0

            # Check RBF (any sequence < 0xFFFFFFFE)
            rbf_enabled = any(
                vin.get('sequence', 0xFFFFFFFF) < 0xFFFFFFFE
                for vin in tx.get('vin', [])
            )

            return {
                'fee_sat_vb': round(fee_sat_vb, 2),
                'input_count': len(tx.get('vin', [])),
                'output_count': len(tx.get('vout', [])),
                'tx_size_bytes': tx_size,
                'is_segwit': 'txinwitness' in str(tx),
                'oldest_utxo_blocks': oldest_blocks if oldest_blocks > 0 else None,
                'locktime': tx.get('locktime', 0),
                'rbf_enabled': rbf_enabled
            }

        except Exception as e:
            print(f"[RPC ERROR] {e}")
            return None

    def _get_tx(self, txid: str) -> Optional[Dict]:
        try:
            result = subprocess.run(
                [self.cli_path, "getrawtransaction", txid, "true"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except:
            pass
        return None

    def _get_block_height(self) -> Optional[int]:
        try:
            result = subprocess.run(
                [self.cli_path, "getblockcount"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except:
            pass
        return None

    def _get_block_height_for_hash(self, blockhash: str) -> Optional[int]:
        try:
            result = subprocess.run(
                [self.cli_path, "getblockheader", blockhash],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                header = json.loads(result.stdout)
                return header.get('height')
        except:
            pass
        return None


class PriceTracker:
    """Track prices for flows at T+0, T+1m, T+5m, T+10m."""

    def __init__(self):
        self.exchange = ccxt.kraken()
        self.current_price = 0.0
        self.price_history: deque = deque(maxlen=1200)  # 20 min of seconds

        # Pending flows waiting for price updates
        self.pending_flows: Dict[str, FlowData] = {}
        self.lock = threading.Lock()

        # Start price feed
        self._start_price_feed()

    def _start_price_feed(self):
        def update():
            while True:
                try:
                    ticker = self.exchange.fetch_ticker('BTC/USD')
                    price = ticker['last']
                    now = time.time()

                    self.current_price = price
                    self.price_history.append((now, price))

                    # Check pending flows for price updates
                    self._check_pending()

                except Exception as e:
                    print(f"[PRICE ERROR] {e}")

                time.sleep(1)

        t = threading.Thread(target=update, daemon=True)
        t.start()

        # Wait for first price
        while self.current_price == 0:
            time.sleep(0.1)

    def get_price(self) -> float:
        return self.current_price

    def track_flow(self, flow: FlowData, callback):
        """Start tracking price for a flow. Calls callback when complete."""
        flow.price_t0 = self.current_price

        with self.lock:
            self.pending_flows[flow.txid] = {
                'flow': flow,
                'callback': callback,
                'start_time': time.time(),
                't1m_done': False,
                't5m_done': False,
                't10m_done': False
            }

    def _check_pending(self):
        """Check pending flows and update prices at checkpoints."""
        now = time.time()
        completed = []

        with self.lock:
            for txid, data in self.pending_flows.items():
                elapsed = now - data['start_time']
                flow = data['flow']

                # T+1 minute
                if elapsed >= 60 and not data['t1m_done']:
                    flow.price_t1m = self.current_price
                    data['t1m_done'] = True

                # T+5 minutes
                if elapsed >= 300 and not data['t5m_done']:
                    flow.price_t5m = self.current_price
                    data['t5m_done'] = True

                # T+10 minutes - complete
                if elapsed >= 600 and not data['t10m_done']:
                    flow.price_t10m = self.current_price
                    data['t10m_done'] = True

                    # Calculate outcome
                    self._calculate_outcome(flow)

                    # Callback
                    data['callback'](flow)
                    completed.append(txid)

            # Remove completed
            for txid in completed:
                del self.pending_flows[txid]

    def _calculate_outcome(self, flow: FlowData):
        """Calculate if price moved in expected direction."""
        if flow.price_t0 is None:
            return

        prices = [flow.price_t0]
        if flow.price_t1m: prices.append(flow.price_t1m)
        if flow.price_t5m: prices.append(flow.price_t5m)
        if flow.price_t10m: prices.append(flow.price_t10m)

        min_price = min(prices)
        max_price = max(prices)

        flow.max_down_move_pct = (flow.price_t0 - min_price) / flow.price_t0 * 100
        flow.max_up_move_pct = (max_price - flow.price_t0) / flow.price_t0 * 100

        # For INFLOW (SHORT signal): did price drop?
        # For OUTFLOW (LONG signal): did price rise?
        if flow.direction == 'INFLOW':
            # Expected: price drops
            flow.price_moved_expected = flow.max_down_move_pct > flow.max_up_move_pct
        else:
            # Expected: price rises
            flow.price_moved_expected = flow.max_up_move_pct > flow.max_down_move_pct


class DataCollector:
    """Main data collector - wraps C++ runner and collects all data."""

    # Only lookup TX details for flows >= this (to avoid RPC spam)
    MIN_RPC_FLOW_BTC = 1.0

    def __init__(self, db_path: str = "flow_data.db"):
        self.db_path = db_path
        self.rpc = BitcoinRPC()
        self.price_tracker = PriceTracker()

        self._init_db()

        self.flows_collected = 0
        self.flows_completed = 0

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS flows (
                id INTEGER PRIMARY KEY,
                txid TEXT UNIQUE,
                timestamp TEXT,
                exchange TEXT,
                direction TEXT,
                flow_btc REAL,
                latency_ns INTEGER,

                -- TX metadata
                fee_sat_vb REAL,
                input_count INTEGER,
                output_count INTEGER,
                tx_size_bytes INTEGER,
                is_segwit INTEGER,
                oldest_utxo_blocks INTEGER,
                locktime INTEGER,
                rbf_enabled INTEGER,

                -- Price tracking
                price_t0 REAL,
                price_t1m REAL,
                price_t5m REAL,
                price_t10m REAL,

                -- Outcome
                price_moved_expected INTEGER,
                max_down_move_pct REAL,
                max_up_move_pct REAL
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_exchange ON flows(exchange)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_direction ON flows(direction)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_flow_btc ON flows(flow_btc)')
        conn.commit()
        conn.close()

    def _save_flow(self, flow: FlowData):
        """Save completed flow to database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO flows VALUES (
                    NULL, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?
                )
            ''', (
                flow.txid, flow.timestamp, flow.exchange, flow.direction,
                flow.flow_btc, flow.latency_ns,
                flow.fee_sat_vb, flow.input_count, flow.output_count,
                flow.tx_size_bytes, 1 if flow.is_segwit else 0 if flow.is_segwit is not None else None,
                flow.oldest_utxo_blocks, flow.locktime,
                1 if flow.rbf_enabled else 0 if flow.rbf_enabled is not None else None,
                flow.price_t0, flow.price_t1m, flow.price_t5m, flow.price_t10m,
                1 if flow.price_moved_expected else 0 if flow.price_moved_expected is not None else None,
                flow.max_down_move_pct, flow.max_up_move_pct
            ))
            conn.commit()
            self.flows_completed += 1

            # Print result
            expected = "YES" if flow.price_moved_expected else "NO"
            down = f"{flow.max_down_move_pct:.3f}%" if flow.max_down_move_pct else "?"
            up = f"{flow.max_up_move_pct:.3f}%" if flow.max_up_move_pct else "?"

            print(f"\n[COMPLETE] {flow.exchange} {flow.direction} {flow.flow_btc:.2f} BTC")
            print(f"           Expected move: {expected} | Down: {down} | Up: {up}")
            print(f"           Fee: {flow.fee_sat_vb} sat/vB | Inputs: {flow.input_count} | UTXO age: {flow.oldest_utxo_blocks}")
            print(f"           Total collected: {self.flows_completed}")

        except Exception as e:
            print(f"[DB ERROR] {e}")
        finally:
            conn.close()

    def on_flow(self, txid: str, exchange: str, direction: str, flow_btc: float, latency_ns: int):
        """Called when C++ runner detects a flow."""

        flow = FlowData(
            txid=txid,
            timestamp=datetime.now(timezone.utc).isoformat(),
            exchange=exchange,
            direction=direction,
            flow_btc=flow_btc,
            latency_ns=latency_ns
        )

        # Get TX metadata for significant flows
        if flow_btc >= self.MIN_RPC_FLOW_BTC:
            tx_data = self.rpc.get_tx_data(txid)
            if tx_data:
                flow.fee_sat_vb = tx_data['fee_sat_vb']
                flow.input_count = tx_data['input_count']
                flow.output_count = tx_data['output_count']
                flow.tx_size_bytes = tx_data['tx_size_bytes']
                flow.is_segwit = tx_data['is_segwit']
                flow.oldest_utxo_blocks = tx_data['oldest_utxo_blocks']
                flow.locktime = tx_data['locktime']
                flow.rbf_enabled = tx_data['rbf_enabled']

        self.flows_collected += 1

        print(f"\n[FLOW {self.flows_collected}] {exchange} {direction} {flow_btc:.4f} BTC")
        print(f"           TXID: {txid[:16]}... | Latency: {latency_ns}ns")
        if flow.fee_sat_vb:
            print(f"           Fee: {flow.fee_sat_vb} sat/vB | Inputs: {flow.input_count} | Outputs: {flow.output_count}")
            if flow.oldest_utxo_blocks:
                print(f"           Oldest UTXO: {flow.oldest_utxo_blocks} blocks | RBF: {flow.rbf_enabled}")

        # Track price for 10 minutes
        self.price_tracker.track_flow(flow, self._save_flow)

    def run(self):
        """Run the data collector with C++ runner."""
        CPP_BINARY = "/root/sovereign/cpp_runner/build/blockchain_runner"

        print("=" * 70)
        print("DATA COLLECTOR FOR 100% WIN RATE DISCOVERY")
        print("=" * 70)
        print(f"Current price: ${self.price_tracker.get_price():,.2f}")
        print(f"RPC lookup for flows >= {self.MIN_RPC_FLOW_BTC} BTC")
        print("Tracking prices at T+0, T+1m, T+5m, T+10m")
        print("=" * 70)
        print("\nNO TRADING - Just collecting data...")
        print("Let this run for 24-48 hours, then analyze.\n")

        process = subprocess.Popen(
            [CPP_BINARY],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            line = line.strip()

            # Strip ANSI codes
            clean = re.sub(r'\x1b\[[0-9;]*m', '', line)

            # Parse C++ output: [SHORT] coinbase | In: 1.9162 | Out: 0 | Net: -1.9162 | Latency: 151598ns
            if '[SHORT]' in clean or '[LONG]' in clean:
                try:
                    parts = clean.split('|')
                    if len(parts) >= 4:
                        # Extract exchange
                        first = parts[0].strip()
                        exchange = first.split()[-1].strip()

                        # Extract inflow/outflow
                        inflow = 0.0
                        outflow = 0.0
                        latency_ns = 0

                        for part in parts[1:]:
                            part = part.strip()
                            if part.startswith('In:'):
                                inflow = float(part.replace('In:', '').strip())
                            elif part.startswith('Out:'):
                                outflow = float(part.replace('Out:', '').strip())
                            elif part.startswith('Latency:'):
                                latency_str = part.replace('Latency:', '').replace('ns', '').strip()
                                latency_ns = int(latency_str)

                        # Generate a pseudo-txid (we don't have real one in current output)
                        # TODO: Modify C++ to output txid
                        pseudo_txid = f"{exchange}_{int(time.time()*1000000)}"

                        if inflow > 0:
                            self.on_flow(pseudo_txid, exchange, 'INFLOW', inflow, latency_ns)
                        if outflow > 0:
                            self.on_flow(pseudo_txid, exchange, 'OUTFLOW', outflow, latency_ns)

                except Exception as e:
                    print(f"[PARSE ERROR] {e}")


def main():
    collector = DataCollector(db_path="/root/sovereign/flow_data.db")
    collector.run()


if __name__ == "__main__":
    main()
