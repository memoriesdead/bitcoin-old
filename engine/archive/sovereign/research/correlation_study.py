"""
BLOCKCHAIN-EXCHANGE CORRELATION STUDY

Goal: Find the mathematical relationship between blockchain events and price movements.

We have the TIMING ADVANTAGE (see transactions 5-60 seconds before anyone else).
Now we need to find WHAT PREDICTS WHAT.

This collector captures:
1. Every blockchain event (flows, patterns, volumes)
2. Price at T=0, T+10s, T+30s, T+60s, T+2min, T+5min
3. Builds dataset for correlation analysis

Run this for 24-48 hours, then analyze.
"""
import os
import sys
import time
import json
import struct
import sqlite3
import threading
import hashlib
from datetime import datetime
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set
import gzip

import zmq
import websocket

# =============================================================================
# CONFIGURATION
# =============================================================================

ZMQ_ENDPOINT = "tcp://127.0.0.1:28332"
EXCHANGES_JSON = "/root/validation/exchanges.json"
DB_PATH = "/root/research/correlation_study.db"
PRICE_WS = "wss://ws-feed.exchange.coinbase.com"

# Time windows to capture price changes (seconds)
PRICE_WINDOWS = [10, 30, 60, 120, 300]  # 10s, 30s, 1min, 2min, 5min

# Minimum BTC for tracking
MIN_BTC_TRACK = 1.0  # Track flows >= 1 BTC
MIN_BTC_MEGA = 100.0  # Mega flows >= 100 BTC

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BlockchainEvent:
    """A single blockchain event to correlate with price."""
    event_id: str
    timestamp: float
    event_type: str  # 'inflow', 'outflow', 'consolidation', 'fanout', 'whale', 'mega'
    btc_amount: float
    exchange: str  # 'binance', 'coinbase', 'unknown', etc.
    tx_inputs: int
    tx_outputs: int
    price_at_event: float
    # Filled in later
    price_10s: float = 0.0
    price_30s: float = 0.0
    price_60s: float = 0.0
    price_120s: float = 0.0
    price_300s: float = 0.0
    return_10s: float = 0.0
    return_30s: float = 0.0
    return_60s: float = 0.0
    return_120s: float = 0.0
    return_300s: float = 0.0


# =============================================================================
# EXCHANGE ADDRESS LOADER
# =============================================================================

class ExchangeAddresses:
    """Load and match exchange addresses."""

    def __init__(self, json_path: str):
        self.address_to_exchange: Dict[str, str] = {}
        self.exchange_set: Set[str] = set()
        self._load(json_path)

    def _load(self, path: str):
        """Load exchange addresses from JSON."""
        if not os.path.exists(path):
            # Try gzipped version
            gz_path = path + '.gz' if not path.endswith('.gz') else path
            if os.path.exists(gz_path):
                path = gz_path
            else:
                print(f"[ADDR] No address file found at {path}")
                return

        print(f"[ADDR] Loading {path}...")
        try:
            if path.endswith('.gz'):
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(path, 'r') as f:
                    data = json.load(f)

            # Handle different JSON formats
            if isinstance(data, dict):
                for exchange, addrs in data.items():
                    if isinstance(addrs, list):
                        for addr in addrs:
                            self.address_to_exchange[addr] = exchange.lower()
                            self.exchange_set.add(addr)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        addr = item.get('address', '')
                        ex = item.get('exchange', 'unknown')
                        if addr:
                            self.address_to_exchange[addr] = ex.lower()
                            self.exchange_set.add(addr)

            print(f"[ADDR] Loaded {len(self.exchange_set):,} addresses")
        except Exception as e:
            print(f"[ADDR] Error loading: {e}")

    def lookup(self, address: str) -> Optional[str]:
        """Return exchange name if address is known."""
        return self.address_to_exchange.get(address)

    def is_exchange(self, address: str) -> bool:
        return address in self.exchange_set


# =============================================================================
# BITCOIN ADDRESS ENCODING
# =============================================================================

# Base58 alphabet
B58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def b58encode(data: bytes) -> str:
    """Encode bytes to base58."""
    num = int.from_bytes(data, 'big')
    result = ''
    while num > 0:
        num, rem = divmod(num, 58)
        result = B58_ALPHABET[rem] + result
    # Add leading 1s for leading zero bytes
    for byte in data:
        if byte == 0:
            result = '1' + result
        else:
            break
    return result

def b58encode_check(payload: bytes) -> str:
    """Encode with checksum."""
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return b58encode(payload + checksum)

def hash160_to_p2pkh(hash160: bytes) -> str:
    """Convert hash160 to P2PKH address (1...)."""
    return b58encode_check(b'\x00' + hash160)

def hash160_to_p2sh(hash160: bytes) -> str:
    """Convert hash160 to P2SH address (3...)."""
    return b58encode_check(b'\x05' + hash160)

# Bech32 encoding
BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

def bech32_polymod(values):
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = ((chk & 0x1ffffff) << 5) ^ v
        for i in range(5):
            chk ^= GEN[i] if ((b >> i) & 1) else 0
    return chk

def bech32_hrp_expand(hrp):
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]

def bech32_create_checksum(hrp, data):
    values = bech32_hrp_expand(hrp) + data
    polymod = bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ 1
    return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]

def bech32_encode(hrp, witver, witprog):
    """Encode to bech32 address."""
    data = [witver] + convertbits(witprog, 8, 5)
    combined = data + bech32_create_checksum(hrp, data)
    return hrp + '1' + ''.join([BECH32_CHARSET[d] for d in combined])

def convertbits(data, frombits, tobits, pad=True):
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    for value in data:
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    return ret


# =============================================================================
# TRANSACTION DECODER
# =============================================================================

def decode_varint(data: bytes, offset: int) -> tuple:
    val = data[offset]
    if val < 0xfd:
        return val, offset + 1
    elif val == 0xfd:
        return struct.unpack_from('<H', data, offset + 1)[0], offset + 3
    elif val == 0xfe:
        return struct.unpack_from('<I', data, offset + 1)[0], offset + 5
    else:
        return struct.unpack_from('<Q', data, offset + 1)[0], offset + 9


def decode_transaction(raw: bytes) -> Optional[Dict]:
    """Decode raw transaction to extract flow information."""
    try:
        offset = 0
        version = struct.unpack_from('<I', raw, offset)[0]
        offset += 4

        # SegWit marker
        is_segwit = False
        if raw[offset] == 0x00 and raw[offset + 1] == 0x01:
            is_segwit = True
            offset += 2

        # Inputs
        in_count, offset = decode_varint(raw, offset)
        inputs = []
        for _ in range(in_count):
            prev_txid = raw[offset:offset+32][::-1].hex()
            offset += 32
            vout = struct.unpack_from('<I', raw, offset)[0]
            offset += 4
            script_len, offset = decode_varint(raw, offset)
            script_sig = raw[offset:offset+script_len]
            offset += script_len
            sequence = struct.unpack_from('<I', raw, offset)[0]
            offset += 4
            inputs.append({'prev_txid': prev_txid, 'vout': vout})

        # Outputs
        out_count, offset = decode_varint(raw, offset)
        outputs = []
        total_value = 0
        for _ in range(out_count):
            value = struct.unpack_from('<Q', raw, offset)[0]
            total_value += value
            offset += 8
            script_len, offset = decode_varint(raw, offset)
            script = raw[offset:offset+script_len]
            offset += script_len

            # Extract address from script
            address = extract_address(script)
            outputs.append({
                'value_sat': value,
                'btc': value / 1e8,
                'address': address
            })

        # Calculate txid
        txid = hashlib.sha256(hashlib.sha256(raw).digest()).digest()[::-1].hex()

        return {
            'txid': txid[:16],  # Short txid
            'inputs': inputs,
            'outputs': outputs,
            'input_count': in_count,
            'output_count': out_count,
            'total_btc': total_value / 1e8,
            'is_segwit': is_segwit
        }
    except Exception:
        return None


def extract_address(script: bytes) -> str:
    """Extract address from output script and convert to proper Bitcoin address format."""
    try:
        # P2PKH: OP_DUP OP_HASH160 <20 bytes> OP_EQUALVERIFY OP_CHECKSIG
        if len(script) == 25 and script[0] == 0x76 and script[1] == 0xa9:
            hash160 = script[3:23]
            return hash160_to_p2pkh(hash160)
        # P2SH: OP_HASH160 <20 bytes> OP_EQUAL
        elif len(script) == 23 and script[0] == 0xa9:
            hash160 = script[2:22]
            return hash160_to_p2sh(hash160)
        # P2WPKH: OP_0 <20 bytes> - bech32 address (bc1q...)
        elif len(script) == 22 and script[0] == 0x00 and script[1] == 0x14:
            witprog = list(script[2:22])
            return bech32_encode('bc', 0, witprog)
        # P2WSH: OP_0 <32 bytes> - bech32 address (bc1q...)
        elif len(script) == 34 and script[0] == 0x00 and script[1] == 0x20:
            witprog = list(script[2:34])
            return bech32_encode('bc', 0, witprog)
        # P2TR: OP_1 <32 bytes> - bech32m address (bc1p...)
        elif len(script) == 34 and script[0] == 0x51 and script[1] == 0x20:
            witprog = list(script[2:34])
            # Use witness version 1 for taproot
            return bech32_encode('bc', 1, witprog)
        return ""
    except:
        return ""


# =============================================================================
# PRICE FEED
# =============================================================================

class PriceFeed:
    """Real-time price from Coinbase WebSocket."""

    def __init__(self):
        self.price = 0.0
        self.bid = 0.0
        self.ask = 0.0
        self.last_update = 0.0
        self.ws = None
        self.running = False
        self._lock = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()

    def get_price(self) -> float:
        with self._lock:
            return self.price

    def _run(self):
        while self.running:
            try:
                self.ws = websocket.create_connection(PRICE_WS, timeout=10)
                subscribe = {
                    "type": "subscribe",
                    "channels": [{"name": "ticker", "product_ids": ["BTC-USD"]}]
                }
                self.ws.send(json.dumps(subscribe))

                while self.running:
                    msg = self.ws.recv()
                    data = json.loads(msg)
                    if data.get('type') == 'ticker':
                        with self._lock:
                            self.price = float(data.get('price', 0))
                            self.bid = float(data.get('best_bid', 0))
                            self.ask = float(data.get('best_ask', 0))
                            self.last_update = time.time()
            except Exception as e:
                print(f"[PRICE] Error: {e}, reconnecting...")
                time.sleep(2)


# =============================================================================
# CORRELATION COLLECTOR
# =============================================================================

class CorrelationCollector:
    """
    Collects blockchain events and correlates with price movements.
    """

    def __init__(self):
        self.addresses = ExchangeAddresses(EXCHANGES_JSON)
        self.price_feed = PriceFeed()
        self.db = self._init_db()

        # ZMQ connection
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)

        # Pending events waiting for price updates
        self.pending_events: List[BlockchainEvent] = []
        self.pending_lock = threading.Lock()

        # Stats
        self.events_captured = 0
        self.events_completed = 0
        self.inflows = 0
        self.outflows = 0
        self.consolidations = 0
        self.fanouts = 0
        self.whales = 0
        self.megas = 0

        self.running = False
        self.start_time = 0

    def _init_db(self) -> sqlite3.Connection:
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute('''CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            timestamp REAL,
            event_type TEXT,
            btc_amount REAL,
            exchange TEXT,
            tx_inputs INTEGER,
            tx_outputs INTEGER,
            price_at_event REAL,
            price_10s REAL,
            price_30s REAL,
            price_60s REAL,
            price_120s REAL,
            price_300s REAL,
            return_10s REAL,
            return_30s REAL,
            return_60s REAL,
            return_120s REAL,
            return_300s REAL,
            completed INTEGER DEFAULT 0
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_type ON events(event_type)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ts ON events(timestamp)')
        conn.commit()
        return conn

    def start(self):
        """Start the collector."""
        print("=" * 60)
        print("BLOCKCHAIN-EXCHANGE CORRELATION STUDY")
        print("=" * 60)
        print(f"Database: {DB_PATH}")
        print(f"Addresses: {len(self.addresses.exchange_set):,}")
        print(f"Price windows: {PRICE_WINDOWS}")
        print()

        self.running = True
        self.start_time = time.time()

        # Start price feed
        self.price_feed.start()
        time.sleep(2)  # Wait for initial price

        # Start ZMQ
        self.sock.connect(ZMQ_ENDPOINT)
        self.sock.setsockopt(zmq.SUBSCRIBE, b'rawtx')
        print(f"[ZMQ] Connected to {ZMQ_ENDPOINT}")

        # Start background threads
        threading.Thread(target=self._price_updater, daemon=True).start()
        threading.Thread(target=self._stats_printer, daemon=True).start()

        # Main loop
        self._main_loop()

    def _main_loop(self):
        """Process incoming transactions."""
        print("[LIVE] Collecting blockchain events...")
        print()

        while self.running:
            try:
                msg = self.sock.recv_multipart()
                if len(msg) >= 2 and msg[0].decode('utf-8', errors='ignore') == 'rawtx':
                    raw_tx = msg[1]
                    self._process_tx(raw_tx)
            except KeyboardInterrupt:
                print("\n[STOP] Shutting down...")
                self.running = False
            except Exception as e:
                print(f"[ERROR] {e}")

    def _process_tx(self, raw_tx: bytes):
        """Process a raw transaction."""
        tx = decode_transaction(raw_tx)
        if not tx:
            return

        ts = time.time()
        price = self.price_feed.get_price()
        if price <= 0:
            return

        inputs = tx['input_count']
        outputs = tx['output_count']
        total_btc = tx['total_btc']

        # Skip small transactions
        if total_btc < MIN_BTC_TRACK:
            return

        # Classify the event
        event_type = None
        exchange = 'unknown'
        direction = 0  # 1 = outflow (bullish), -1 = inflow (bearish)

        # Check for exchange address matches
        for out in tx['outputs']:
            addr = out.get('address', '')
            ex = self.addresses.lookup(addr)
            if ex:
                # Output to exchange = INFLOW (bearish)
                event_type = 'inflow'
                exchange = ex
                direction = -1
                self.inflows += 1
                break

        if not event_type:
            # No output match, check inputs (would need prev tx lookup - skip for now)
            pass

        # Pattern-based detection
        if not event_type:
            # Consolidation: many inputs -> few outputs = exchange gathering (bullish)
            if inputs >= 10 and outputs <= 3:
                event_type = 'consolidation'
                direction = 1
                self.consolidations += 1
            # Fan-out: few inputs -> many outputs = exchange distributing (bullish)
            elif inputs <= 3 and outputs >= 20:
                event_type = 'fanout'
                direction = 1
                self.fanouts += 1
            # Mega flow
            elif total_btc >= MIN_BTC_MEGA:
                event_type = 'mega'
                self.megas += 1
            # Whale flow
            elif total_btc >= 10.0:
                event_type = 'whale'
                self.whales += 1

        if not event_type:
            return

        # Create event
        event = BlockchainEvent(
            event_id=f"{tx['txid']}_{ts:.0f}",
            timestamp=ts,
            event_type=event_type,
            btc_amount=total_btc,
            exchange=exchange,
            tx_inputs=inputs,
            tx_outputs=outputs,
            price_at_event=price
        )

        # Save to DB
        self._save_event(event)
        self.events_captured += 1

        # Add to pending for price tracking
        with self.pending_lock:
            self.pending_events.append(event)

        # Log significant events
        if total_btc >= 10 or event_type in ['inflow', 'consolidation', 'fanout']:
            dir_str = "LONG" if direction == 1 else "SHORT" if direction == -1 else "?"
            print(f"[{event_type.upper()}] {total_btc:.2f} BTC | {exchange} | {dir_str} | ${price:,.0f}")

    def _save_event(self, event: BlockchainEvent):
        """Save event to database."""
        try:
            self.db.execute('''INSERT OR REPLACE INTO events
                (event_id, timestamp, event_type, btc_amount, exchange,
                 tx_inputs, tx_outputs, price_at_event, completed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)''',
                (event.event_id, event.timestamp, event.event_type,
                 event.btc_amount, event.exchange, event.tx_inputs,
                 event.tx_outputs, event.price_at_event))
            self.db.commit()
        except Exception as e:
            print(f"[DB] Error: {e}")

    def _update_event_prices(self, event: BlockchainEvent):
        """Update event with final prices."""
        try:
            # Calculate returns
            p0 = event.price_at_event
            if p0 > 0:
                event.return_10s = (event.price_10s - p0) / p0 if event.price_10s > 0 else 0
                event.return_30s = (event.price_30s - p0) / p0 if event.price_30s > 0 else 0
                event.return_60s = (event.price_60s - p0) / p0 if event.price_60s > 0 else 0
                event.return_120s = (event.price_120s - p0) / p0 if event.price_120s > 0 else 0
                event.return_300s = (event.price_300s - p0) / p0 if event.price_300s > 0 else 0

            self.db.execute('''UPDATE events SET
                price_10s = ?, price_30s = ?, price_60s = ?, price_120s = ?, price_300s = ?,
                return_10s = ?, return_30s = ?, return_60s = ?, return_120s = ?, return_300s = ?,
                completed = 1
                WHERE event_id = ?''',
                (event.price_10s, event.price_30s, event.price_60s, event.price_120s, event.price_300s,
                 event.return_10s, event.return_30s, event.return_60s, event.return_120s, event.return_300s,
                 event.event_id))
            self.db.commit()
            self.events_completed += 1
        except Exception as e:
            print(f"[DB] Update error: {e}")

    def _price_updater(self):
        """Background thread to update prices for pending events."""
        while self.running:
            time.sleep(1)
            now = time.time()
            price = self.price_feed.get_price()

            with self.pending_lock:
                still_pending = []
                for event in self.pending_events:
                    age = now - event.timestamp

                    # Update prices at each window
                    if age >= 10 and event.price_10s == 0:
                        event.price_10s = price
                    if age >= 30 and event.price_30s == 0:
                        event.price_30s = price
                    if age >= 60 and event.price_60s == 0:
                        event.price_60s = price
                    if age >= 120 and event.price_120s == 0:
                        event.price_120s = price
                    if age >= 300 and event.price_300s == 0:
                        event.price_300s = price

                    # Check if complete
                    if age >= 300 and event.price_300s > 0:
                        self._update_event_prices(event)
                    else:
                        still_pending.append(event)

                self.pending_events = still_pending

    def _stats_printer(self):
        """Print stats periodically."""
        while self.running:
            time.sleep(60)
            elapsed = (time.time() - self.start_time) / 3600
            price = self.price_feed.get_price()

            print(f"\n[{elapsed:.2f}h] Events: {self.events_captured} | "
                  f"Completed: {self.events_completed} | "
                  f"Inflows: {self.inflows} | Outflows: {self.outflows} | "
                  f"Consol: {self.consolidations} | Fanout: {self.fanouts} | "
                  f"BTC: ${price:,.0f}")

            # Quick correlation check
            self._quick_correlation()

    def _quick_correlation(self):
        """Quick correlation analysis on collected data."""
        try:
            cur = self.db.execute('''
                SELECT event_type, COUNT(*) as n,
                       AVG(return_60s) * 100 as avg_return_60s,
                       AVG(return_300s) * 100 as avg_return_300s
                FROM events
                WHERE completed = 1 AND return_60s != 0
                GROUP BY event_type
            ''')
            results = cur.fetchall()

            if results:
                print("  Correlation check:")
                for row in results:
                    event_type, n, ret_60, ret_300 = row
                    if n >= 5:
                        print(f"    {event_type}: n={n}, 60s={ret_60:+.4f}%, 5min={ret_300:+.4f}%")
        except Exception as e:
            pass


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    collector = CorrelationCollector()
    collector.start()
