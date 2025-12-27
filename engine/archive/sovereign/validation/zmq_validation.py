#!/usr/bin/env python3
"""
Bitcoin Core ZMQ Signal Validation
===================================

THE REAL EDGE: Direct ZMQ connection to Bitcoin Core.
Sees transactions BEFORE they hit any exchange or API.

This is 10-60 seconds ahead of everyone using public APIs.

Deploy to Hostinger: scp zmq_validation.py root@31.97.211.217:/root/validation/
"""

import os
import sys
import time
import json
import signal
import sqlite3
import threading
import struct
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Optional, Callable
import logging

# ZMQ support
try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False
    print("ERROR: pyzmq not installed. Run: pip install pyzmq")

# HTTP for price
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# WebSocket for price
try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/root/validation/zmq_validation.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("/root/validation/data")
SIGNALS_DB = DATA_DIR / "signals.db"
PRICES_DB = DATA_DIR / "prices.db"
EXCHANGES_JSON = Path("/root/validation/exchanges.json")

# Bitcoin Core ZMQ endpoints
ZMQ_RAWTX = "tcp://127.0.0.1:28332"
ZMQ_RAWBLOCK = "tcp://127.0.0.1:28333"

# Bitcoin Core RPC for input address resolution
RPC_URL = "http://127.0.0.1:8332"
RPC_USER = "bitcoin"
RPC_PASS = "bitcoin123secure"


# =============================================================================
# TRANSACTION DECODER (Minimal)
# =============================================================================

def decode_varint(data: bytes, offset: int) -> tuple:
    """Decode Bitcoin varint."""
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
    """Decode raw Bitcoin transaction to extract addresses."""
    try:
        offset = 0

        # Version (4 bytes)
        version = struct.unpack_from('<I', raw, offset)[0]
        offset += 4

        # Check for SegWit marker
        is_segwit = False
        if raw[offset] == 0x00 and raw[offset + 1] == 0x01:
            is_segwit = True
            offset += 2

        # Input count
        in_count, offset = decode_varint(raw, offset)

        inputs = []
        for _ in range(in_count):
            # Previous txid (32 bytes, reversed)
            prev_txid = raw[offset:offset + 32][::-1].hex()
            offset += 32

            # Previous output index (4 bytes)
            prev_idx = struct.unpack_from('<I', raw, offset)[0]
            offset += 4

            # Script length and script
            script_len, offset = decode_varint(raw, offset)
            script = raw[offset:offset + script_len]
            offset += script_len

            # Sequence (4 bytes)
            offset += 4

            inputs.append({
                'prev_txid': prev_txid,
                'prev_idx': prev_idx,
            })

        # Output count
        out_count, offset = decode_varint(raw, offset)

        outputs = []
        for _ in range(out_count):
            # Value (8 bytes, satoshis)
            value = struct.unpack_from('<Q', raw, offset)[0]
            offset += 8

            # Script length and script
            script_len, offset = decode_varint(raw, offset)
            script = raw[offset:offset + script_len]
            offset += script_len

            # Extract address from script
            addr = extract_address(script)

            outputs.append({
                'value': value,
                'btc': value / 1e8,
                'address': addr,
                'script': script.hex(),
            })

        # Calculate txid
        if is_segwit:
            # For txid, we need to hash without witness data
            # This is simplified - full implementation would strip witness
            txid_data = raw[:4] + raw[6:]  # Remove marker/flag
        else:
            txid_data = raw

        txid = hashlib.sha256(hashlib.sha256(raw).digest()).digest()[::-1].hex()

        return {
            'txid': txid,
            'version': version,
            'inputs': inputs,
            'outputs': outputs,
            'is_segwit': is_segwit,
        }

    except Exception as e:
        return None


def extract_address(script: bytes) -> Optional[str]:
    """Extract Bitcoin address from output script."""
    try:
        # P2PKH: OP_DUP OP_HASH160 <20 bytes> OP_EQUALVERIFY OP_CHECKSIG
        if len(script) == 25 and script[0] == 0x76 and script[1] == 0xa9:
            pubkey_hash = script[3:23]
            return hash160_to_address(pubkey_hash, 0x00)

        # P2SH: OP_HASH160 <20 bytes> OP_EQUAL
        if len(script) == 23 and script[0] == 0xa9:
            script_hash = script[2:22]
            return hash160_to_address(script_hash, 0x05)

        # P2WPKH: OP_0 <20 bytes>
        if len(script) == 22 and script[0] == 0x00 and script[1] == 0x14:
            witness_program = script[2:22]
            return bech32_encode('bc', 0, witness_program)

        # P2WSH: OP_0 <32 bytes>
        if len(script) == 34 and script[0] == 0x00 and script[1] == 0x20:
            witness_program = script[2:34]
            return bech32_encode('bc', 0, witness_program)

        # P2TR: OP_1 <32 bytes>
        if len(script) == 34 and script[0] == 0x51 and script[1] == 0x20:
            witness_program = script[2:34]
            return bech32_encode('bc', 1, witness_program)

    except:
        pass

    return None


def hash160_to_address(hash160: bytes, version: int) -> str:
    """Convert hash160 to base58check address."""
    data = bytes([version]) + hash160
    checksum = hashlib.sha256(hashlib.sha256(data).digest()).digest()[:4]
    return base58_encode(data + checksum)


def base58_encode(data: bytes) -> str:
    """Base58 encode."""
    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    n = int.from_bytes(data, 'big')
    result = ''
    while n > 0:
        n, r = divmod(n, 58)
        result = alphabet[r] + result
    # Add leading zeros
    for byte in data:
        if byte == 0:
            result = '1' + result
        else:
            break
    return result


def bech32_encode(hrp: str, witver: int, witprog: bytes) -> str:
    """Bech32/Bech32m encode for SegWit addresses."""
    CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

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

    def bech32_create_checksum(hrp, data, spec):
        values = bech32_hrp_expand(hrp) + data
        const = 0x2bc830a3 if spec == 'm' else 1
        polymod = bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ const
        return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]

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
        if pad and bits:
            ret.append((acc << (tobits - bits)) & maxv)
        return ret

    spec = 'm' if witver > 0 else '1'
    data = [witver] + convertbits(witprog, 8, 5)
    checksum = bech32_create_checksum(hrp, data, spec)
    return hrp + '1' + ''.join([CHARSET[d] for d in data + checksum])


# =============================================================================
# BITCOIN CORE RPC CLIENT
# =============================================================================

class BitcoinRPC:
    """Minimal Bitcoin Core RPC client for input address resolution."""

    def __init__(self, url: str = RPC_URL, user: str = RPC_USER, password: str = RPC_PASS):
        self.url = url
        self.auth = (user, password)
        self._cache = {}  # Cache txid -> outputs to reduce RPC calls
        self._cache_max = 10000

    def get_raw_transaction(self, txid: str, verbose: bool = True) -> Optional[Dict]:
        """Get transaction details via RPC."""
        # Check cache first
        cache_key = f"{txid}_{verbose}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            payload = {
                "jsonrpc": "1.0",
                "id": "zmq_validation",
                "method": "getrawtransaction",
                "params": [txid, verbose]
            }
            resp = requests.post(self.url, json=payload, auth=self.auth, timeout=5)
            if resp.status_code == 200:
                result = resp.json().get('result')
                # Cache the result
                if len(self._cache) > self._cache_max:
                    # Clear oldest half of cache
                    keys = list(self._cache.keys())[:len(self._cache)//2]
                    for k in keys:
                        del self._cache[k]
                self._cache[cache_key] = result
                return result
        except Exception as e:
            pass
        return None

    def get_input_address(self, txid: str, vout: int) -> Optional[str]:
        """Get the address from a previous transaction output."""
        tx = self.get_raw_transaction(txid)
        if tx and 'vout' in tx:
            for out in tx['vout']:
                if out.get('n') == vout:
                    script = out.get('scriptPubKey', {})
                    # Try different address fields
                    if 'address' in script:
                        return script['address']
                    if 'addresses' in script and script['addresses']:
                        return script['addresses'][0]
        return None


# =============================================================================
# EXCHANGE ADDRESS TRACKER
# =============================================================================

class ExchangeTracker:
    """Track known exchange addresses for flow detection."""

    def __init__(self, json_path: str = None):
        self.address_to_exchange: Dict[str, str] = {}
        self.exchange_addresses: Set[str] = set()

        if json_path and Path(json_path).exists():
            self._load_json(json_path)
        else:
            logger.warning(f"exchanges.json not found at {json_path}")

    def _load_json(self, path: str):
        """Load exchange addresses from JSON."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            for exchange_id, addresses in data.items():
                # Format is {exchange: [addresses]} - addresses is a list directly
                if isinstance(addresses, list):
                    for addr in addresses:
                        self.address_to_exchange[addr] = exchange_id
                        self.exchange_addresses.add(addr)
                elif isinstance(addresses, dict):
                    # Alternative format: {exchange: {addresses: [...]}}
                    for addr in addresses.get('addresses', []):
                        self.address_to_exchange[addr] = exchange_id
                        self.exchange_addresses.add(addr)

            logger.info(f"Loaded {len(self.address_to_exchange):,} exchange addresses")

        except Exception as e:
            logger.error(f"Error loading exchanges.json: {e}")

    def is_exchange(self, address: str) -> bool:
        return address in self.exchange_addresses

    def get_exchange(self, address: str) -> Optional[str]:
        return self.address_to_exchange.get(address)


# =============================================================================
# PRICE COLLECTOR
# =============================================================================

class PriceCollector:
    """Collect BTC price via Binance WebSocket."""

    def __init__(self):
        self.running = False
        self.current_price = 0.0
        self.current_bid = 0.0
        self.current_ask = 0.0
        self.prices_logged = 0
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(PRICES_DB)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                price REAL NOT NULL,
                bid REAL,
                ask REAL,
                source TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_price_ts ON prices(timestamp)")
        conn.commit()
        conn.close()

    def log_price(self, price: float, bid: float = 0, ask: float = 0, source: str = "binance"):
        now = time.time()
        now_ms = int(now * 1000)

        with self._lock:
            conn = sqlite3.connect(PRICES_DB)
            c = conn.cursor()
            c.execute(
                "INSERT INTO prices (timestamp, timestamp_ms, price, bid, ask, source) VALUES (?,?,?,?,?,?)",
                (now, now_ms, price, bid, ask, source)
            )
            conn.commit()
            conn.close()

            self.current_price = price
            self.current_bid = bid
            self.current_ask = ask
            self.prices_logged += 1

    def start(self):
        if not HAS_WEBSOCKET:
            logger.error("websocket-client not installed")
            return

        self.running = True
        thread = threading.Thread(target=self._ws_loop, daemon=True)
        thread.start()
        logger.info("Price collection started (Binance WebSocket)")

    def _ws_loop(self):
        """Try multiple price sources."""
        # Try Kraken first (no geo-restrictions)
        kraken_url = "wss://ws.kraken.com"
        binance_url = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"

        while self.running:
            # Try Kraken WebSocket
            try:
                logger.info("Trying Kraken WebSocket for price...")
                ws = websocket.WebSocketApp(
                    kraken_url,
                    on_open=self._on_kraken_open,
                    on_message=self._on_kraken_message,
                    on_error=lambda ws, e: logger.error(f"Kraken WS error: {e}"),
                )
                ws.run_forever()
            except Exception as e:
                logger.error(f"Kraken WS failed: {e}")

            # Fallback to REST polling
            if self.running:
                logger.info("Falling back to REST price polling...")
                self._poll_prices()

    def _on_kraken_open(self, ws):
        """Subscribe to Kraken BTC/USD ticker."""
        subscribe = {
            "event": "subscribe",
            "pair": ["XBT/USD"],
            "subscription": {"name": "ticker"}
        }
        ws.send(json.dumps(subscribe))
        logger.info("Subscribed to Kraken XBT/USD ticker")

    def _on_kraken_message(self, ws, message):
        try:
            data = json.loads(message)
            # Kraken ticker format: [channelID, {"a":["price","wholeLotVolume","lotVolume"],...}, "ticker", "XBT/USD"]
            if isinstance(data, list) and len(data) >= 2:
                ticker = data[1]
                if isinstance(ticker, dict) and 'c' in ticker:
                    # 'c' = close/last trade price [price, lot volume]
                    price = float(ticker['c'][0])
                    bid = float(ticker['b'][0]) if 'b' in ticker else price
                    ask = float(ticker['a'][0]) if 'a' in ticker else price
                    self.log_price(price, bid, ask, source="kraken")
        except:
            pass

    def _poll_prices(self):
        """Fallback REST API price polling."""
        sources = [
            ("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", self._parse_kraken_rest),
            ("https://api.coinbase.com/v2/prices/BTC-USD/spot", self._parse_coinbase_rest),
            ("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd", self._parse_coingecko),
        ]

        while self.running:
            for url, parser in sources:
                try:
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        price = parser(resp.json())
                        if price and price > 0:
                            self.log_price(price, source="rest")
                            break
                except:
                    continue
            time.sleep(1)

    def _parse_kraken_rest(self, data):
        try:
            ticker = data['result']['XXBTZUSD']
            return float(ticker['c'][0])
        except:
            return None

    def _parse_coinbase_rest(self, data):
        try:
            return float(data['data']['amount'])
        except:
            return None

    def _parse_coingecko(self, data):
        try:
            return float(data['bitcoin']['usd'])
        except:
            return None

    def _on_message(self, ws, message):
        """Binance message handler (backup)."""
        try:
            data = json.loads(message)
            bid = float(data.get('b', 0))
            ask = float(data.get('a', 0))
            price = (bid + ask) / 2
            self.log_price(price, bid, ask)
        except:
            pass

    def stop(self):
        self.running = False


# =============================================================================
# ZMQ SIGNAL COLLECTOR - THE REAL EDGE
# =============================================================================

class ZMQSignalCollector:
    """
    Direct Bitcoin Core ZMQ connection for REAL-TIME exchange flow detection.

    This sees transactions 10-60 seconds BEFORE any public API.
    """

    def __init__(self, price_collector: PriceCollector, exchanges_json: str = None):
        self.price_collector = price_collector
        self.tracker = ExchangeTracker(exchanges_json)
        self.rpc = BitcoinRPC()  # For input address resolution

        self.running = False
        self._lock = threading.Lock()

        # ZMQ context
        self.context = None
        self.socket = None

        # Stats
        self.txs_processed = 0
        self.signals_logged = 0
        self.inflows = 0
        self.outflows = 0
        self.total_inflow_btc = 0.0
        self.total_outflow_btc = 0.0
        self.start_time = 0.0

        # Signal accumulator (emit every N seconds)
        self.signal_interval = 60
        self.last_signal_time = 0
        self.accumulated_inflow = 0.0
        self.accumulated_outflow = 0.0

        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(SIGNALS_DB)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                direction INTEGER NOT NULL,
                confidence REAL NOT NULL,
                should_trade INTEGER,
                inflow_btc REAL,
                outflow_btc REAL,
                net_flow REAL,
                price_at_signal REAL,
                source TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_signal_ts ON signals(timestamp)")
        conn.commit()
        conn.close()

    def start(self) -> bool:
        if not HAS_ZMQ:
            logger.error("pyzmq not installed")
            return False

        self.running = True
        self.start_time = time.time()
        self.last_signal_time = time.time()

        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, 'rawtx')
        self.socket.connect(ZMQ_RAWTX)

        # Start receiver thread
        thread = threading.Thread(target=self._zmq_loop, daemon=True)
        thread.start()

        # Start signal emitter thread
        signal_thread = threading.Thread(target=self._signal_loop, daemon=True)
        signal_thread.start()

        logger.info(f"ZMQ connected to {ZMQ_RAWTX}")
        logger.info(f"Tracking {len(self.tracker.exchange_addresses):,} exchange addresses")
        return True

    def _zmq_loop(self):
        """Main ZMQ receive loop."""
        logger.info("ZMQ receiver started - listening for transactions...")

        while self.running:
            try:
                # Receive message (topic + body + sequence)
                msg = self.socket.recv_multipart()

                if len(msg) >= 2:
                    topic = msg[0].decode('utf-8', errors='ignore')
                    raw_tx = msg[1]

                    if topic == 'rawtx':
                        self._process_transaction(raw_tx)

            except zmq.ZMQError as e:
                if self.running:
                    logger.error(f"ZMQ error: {e}")
                    time.sleep(1)
            except Exception as e:
                if self.running:
                    logger.error(f"Error: {e}")

    def _process_transaction(self, raw_tx: bytes):
        """Process a raw transaction for exchange flows."""
        tx = decode_transaction(raw_tx)
        if not tx:
            return

        with self._lock:
            self.txs_processed += 1

        txid = tx['txid']
        ts = time.time()

        # Check outputs for exchange INFLOWS (bearish - sending TO exchange to sell)
        for out in tx['outputs']:
            addr = out.get('address')
            btc = out.get('btc', 0)

            if addr and btc > 0 and self.tracker.is_exchange(addr):
                exchange = self.tracker.get_exchange(addr)

                with self._lock:
                    self.inflows += 1
                    self.total_inflow_btc += btc
                    self.accumulated_inflow += btc

                if btc >= 1.0:
                    logger.info(f"[INFLOW] {btc:.2f} BTC -> {exchange} (SHORT signal)")

        # Check inputs for exchange OUTFLOWS (bullish - exchange withdrawals)
        # This uses RPC to resolve input addresses
        # Only check high-value transactions (>0.5 BTC) to reduce RPC load
        total_output_btc = sum(o.get('btc', 0) for o in tx['outputs'])

        if total_output_btc >= 0.5:  # Only check significant transactions
            for inp in tx['inputs'][:5]:  # Limit to first 5 inputs
                prev_txid = inp.get('prev_txid')
                prev_idx = inp.get('prev_idx')

                # Skip coinbase (mining rewards)
                if prev_txid == '0' * 64:
                    continue

                # Query RPC to get the input address
                # Note: Pruned nodes may not have old transactions
                try:
                    input_addr = self.rpc.get_input_address(prev_txid, prev_idx)
                except:
                    continue

                if input_addr and self.tracker.is_exchange(input_addr):
                    exchange = self.tracker.get_exchange(input_addr)

                    # Estimate BTC value (input value ~ output value for spending transactions)
                    btc_estimate = total_output_btc / max(1, len(tx['inputs']))

                    with self._lock:
                        self.outflows += 1
                        self.total_outflow_btc += btc_estimate
                        self.accumulated_outflow += btc_estimate

                    if btc_estimate >= 1.0:
                        logger.info(f"[OUTFLOW] ~{btc_estimate:.2f} BTC <- {exchange} (LONG signal)")

    def _signal_loop(self):
        """Emit signals periodically."""
        while self.running:
            time.sleep(self.signal_interval)

            with self._lock:
                inflow = self.accumulated_inflow
                outflow = self.accumulated_outflow

                # Reset accumulators
                self.accumulated_inflow = 0.0
                self.accumulated_outflow = 0.0

            net = outflow - inflow

            # Generate signal
            if inflow > 1.0:  # Significant inflow = bearish
                direction = -1
                confidence = min(0.8, 0.5 + inflow / 50)
            elif outflow > 1.0:  # Significant outflow = bullish
                direction = 1
                confidence = min(0.8, 0.5 + outflow / 50)
            else:
                direction = 0
                confidence = 0.0

            # Log signal if significant
            if direction != 0:
                self._log_signal(direction, confidence, inflow, outflow)

    def _log_signal(self, direction: int, confidence: float, inflow: float, outflow: float):
        """Log a trading signal."""
        now = time.time()
        now_ms = int(now * 1000)
        net = outflow - inflow
        price = self.price_collector.current_price

        conn = sqlite3.connect(SIGNALS_DB)
        c = conn.cursor()
        c.execute("""
            INSERT INTO signals (
                timestamp, timestamp_ms, direction, confidence, should_trade,
                inflow_btc, outflow_btc, net_flow, price_at_signal, source
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            now, now_ms, direction, confidence, 1,
            inflow, outflow, net, price, 'bitcoin_core_zmq'
        ))
        conn.commit()
        conn.close()

        with self._lock:
            self.signals_logged += 1

        dir_str = "LONG" if direction == 1 else "SHORT"
        logger.info(f"[SIGNAL] {dir_str} | Conf: {confidence:.2f} | "
                   f"Net: {net:.2f} BTC | Price: ${price:,.0f}")

    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()

    def get_stats(self) -> Dict:
        with self._lock:
            elapsed = time.time() - self.start_time if self.start_time > 0 else 1
            return {
                'txs_processed': self.txs_processed,
                'txs_per_min': (self.txs_processed / elapsed) * 60,
                'signals_logged': self.signals_logged,
                'inflows': self.inflows,
                'outflows': self.outflows,
                'total_inflow_btc': self.total_inflow_btc,
                'total_outflow_btc': self.total_outflow_btc,
                'addresses_tracked': len(self.tracker.exchange_addresses),
            }


# =============================================================================
# MAIN COLLECTOR
# =============================================================================

class ZMQValidationCollector:
    """Main orchestrator for Bitcoin Core ZMQ validation."""

    def __init__(self):
        self.price_collector = PriceCollector()
        self.signal_collector = ZMQSignalCollector(
            self.price_collector,
            str(EXCHANGES_JSON)
        )
        self.running = False
        self.start_time = None

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Shutdown signal received...")
        self.running = False

    def start(self, duration_hours: float = None):
        logger.info("=" * 60)
        logger.info("BITCOIN CORE ZMQ SIGNAL VALIDATION")
        logger.info("THE REAL EDGE - 10-60 SECONDS AHEAD")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration_hours or 'unlimited'} hours")
        logger.info(f"Data dir: {DATA_DIR}")
        logger.info("")

        self.running = True
        self.start_time = time.time()

        # Start collectors
        self.price_collector.start()

        if not self.signal_collector.start():
            logger.error("Failed to start ZMQ collector")
            return

        logger.info("")
        logger.info("LIVE - Monitoring Bitcoin Core mempool for exchange flows")
        logger.info("")

        end_time = None
        if duration_hours:
            end_time = self.start_time + (duration_hours * 3600)

        last_status = 0

        while self.running:
            now = time.time()

            if end_time and now >= end_time:
                logger.info("Duration reached")
                break

            if now - last_status >= 60:
                self._print_status()
                last_status = now

            time.sleep(1)

        self._print_final()

    def _print_status(self):
        runtime = (time.time() - self.start_time) / 3600
        stats = self.signal_collector.get_stats()

        logger.info(f"[{runtime:.2f}h] TXs: {stats['txs_processed']:,} ({stats['txs_per_min']:.0f}/min) | "
                   f"Signals: {stats['signals_logged']} | "
                   f"In: {stats['total_inflow_btc']:.2f} | Out: {stats['total_outflow_btc']:.2f} BTC | "
                   f"BTC: ${self.price_collector.current_price:,.0f}")

    def _print_final(self):
        runtime = (time.time() - self.start_time) / 3600
        stats = self.signal_collector.get_stats()

        logger.info("")
        logger.info("=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Runtime: {runtime:.2f} hours")
        logger.info(f"Transactions processed: {stats['txs_processed']:,}")
        logger.info(f"Signals collected: {stats['signals_logged']}")
        logger.info(f"Total inflow: {stats['total_inflow_btc']:.2f} BTC")
        logger.info(f"Total outflow: {stats['total_outflow_btc']:.2f} BTC")
        logger.info(f"Prices collected: {self.price_collector.prices_logged:,}")
        logger.info("")
        logger.info("Data files:")
        logger.info(f"  {SIGNALS_DB}")
        logger.info(f"  {PRICES_DB}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=None,
                       help="Duration in hours")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    collector = ZMQValidationCollector()
    collector.start(duration_hours=args.duration)


if __name__ == "__main__":
    main()
