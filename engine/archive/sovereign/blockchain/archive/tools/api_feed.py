"""
API-Based Blockchain Feed - NO BITCOIN CORE REQUIRED
=====================================================

Uses mempool.space WebSocket API for real-time transaction detection.
Detects exchange flows (INFLOW = SHORT, OUTFLOW = LONG) via API.

This is the PRACTICAL solution for VPS deployment without 500GB disk.

Mempool.space WebSocket: wss://mempool.space/api/v1/ws
"""

import time
import json
import threading
from typing import Dict, Callable, Optional, Set
from pathlib import Path

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .exchange_wallets import ExchangeWalletTracker
from .types import FlowType, ExchangeTick


class APIBlockchainFeed:
    """
    API-based blockchain feed using mempool.space WebSocket.

    NO BITCOIN CORE REQUIRED - works on any VPS with internet.

    Detects:
    - Exchange inflows (BTC sent TO exchange = bearish, SHORT signal)
    - Exchange outflows (BTC sent FROM exchange = bullish, LONG signal)
    """

    MEMPOOL_WS_URL = "wss://mempool.space/api/v1/ws"
    MEMPOOL_API_URL = "https://mempool.space/api"

    TRADING_EXCHANGES = ['coinbase', 'kraken', 'bitstamp', 'gemini', 'binance']

    def __init__(self,
                 on_tick: Callable[[ExchangeTick], None] = None,
                 on_signal: Callable[[Dict], None] = None,
                 json_path: str = None):
        """
        Initialize API-based feed.

        Args:
            on_tick: Callback for each exchange flow tick
            on_signal: Callback for aggregated trading signals
            json_path: Path to exchanges.json with wallet addresses
        """
        self.on_tick = on_tick
        self.on_signal = on_signal

        # Load exchange addresses
        self.wallet_tracker = ExchangeWalletTracker(json_path=json_path)
        self.address_to_exchange = self.wallet_tracker.address_to_exchange
        self.exchange_addresses_set = self.wallet_tracker.exchange_addresses_set

        # WebSocket state
        self._ws = None
        self._ws_thread = None
        self.running = False
        self._lock = threading.Lock()

        # Price tracking
        self.reference_price = 0.0
        self._price_thread = None

        # Statistics
        self.start_time = 0.0
        self.txs_processed = 0
        self.ticks_emitted = 0
        self.inflows = 0
        self.outflows = 0
        self.total_inflow_btc = 0.0
        self.total_outflow_btc = 0.0

        # Address tracking for new blocks
        self._tracked_addresses = set()

        print(f"[API-FEED] Initialized with {len(self.address_to_exchange):,} exchange addresses")
        print(f"[API-FEED] Source: mempool.space WebSocket (NO Bitcoin Core needed)")

    def start(self) -> bool:
        """Start the API feed."""
        if not HAS_WEBSOCKET:
            print("[API-FEED] ERROR: websocket-client not installed")
            print("[API-FEED] Run: pip install websocket-client")
            return False

        print("[API-FEED] Connecting to mempool.space WebSocket...")
        self.running = True
        self.start_time = time.time()

        # Start price tracker
        self._start_price_tracker()

        # Start WebSocket
        self._ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._ws_thread.start()

        # Give it a moment to connect
        time.sleep(2)

        if self.running:
            print("[API-FEED] LIVE - Monitoring exchange flows via API")
            return True
        return False

    def stop(self):
        """Stop the feed."""
        self.running = False
        if self._ws:
            try:
                self._ws.close()
            except:
                pass
        print("[API-FEED] Stopped")

    def _ws_loop(self):
        """WebSocket connection loop with auto-reconnect."""
        while self.running:
            try:
                self._ws = websocket.WebSocketApp(
                    self.MEMPOOL_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)

                if self.running:
                    print("[API-FEED] Disconnected, reconnecting in 5s...")
                    time.sleep(5)

            except Exception as e:
                print(f"[API-FEED] Error: {e}")
                if self.running:
                    time.sleep(5)

    def _on_open(self, ws):
        """Handle WebSocket open - subscribe to relevant data."""
        print("[API-FEED] WebSocket connected")

        # Subscribe to blocks (for new transaction announcements)
        ws.send(json.dumps({"action": "want", "data": ["blocks", "stats", "mempool-blocks"]}))

        # Track specific exchange addresses (limited set for real-time)
        # We'll focus on high-volume addresses
        high_volume_addresses = self._get_high_volume_addresses()
        for addr in high_volume_addresses[:100]:  # Limit to top 100
            ws.send(json.dumps({"track-address": addr}))
            self._tracked_addresses.add(addr)

        print(f"[API-FEED] Tracking {len(self._tracked_addresses)} high-volume addresses")

    def _get_high_volume_addresses(self) -> list:
        """Get high-volume exchange addresses to track."""
        # Priority exchanges for trading signals
        priority = ['binance', 'coinbase', 'kraken', 'bitstamp', 'gemini', 'okx', 'bybit']

        addresses = []
        for exchange in priority:
            ex_addrs = [addr for addr, ex in self.address_to_exchange.items() if ex == exchange]
            addresses.extend(ex_addrs[:20])  # Top 20 per exchange

        return addresses

    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            # Handle new block
            if 'block' in data:
                self._handle_block(data['block'])

            # Handle address activity
            if 'address-transactions' in data:
                for tx in data['address-transactions']:
                    self._process_transaction(tx)

            # Handle new transactions in tracked addresses
            if 'txs' in data:
                for tx in data['txs']:
                    self._process_transaction(tx)

        except Exception as e:
            pass  # Silently ignore parse errors

    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        print(f"[API-FEED] WebSocket error: {error}")

    def _on_close(self, ws, close_status, close_msg):
        """Handle WebSocket close."""
        print(f"[API-FEED] WebSocket closed: {close_status}")

    def _handle_block(self, block_data: Dict):
        """Handle new block announcement."""
        height = block_data.get('height', 0)
        tx_count = block_data.get('tx_count', 0)

        # Fetch block transactions for exchange detection
        if tx_count > 0:
            threading.Thread(
                target=self._fetch_block_txs,
                args=(block_data.get('id', ''),),
                daemon=True
            ).start()

    def _fetch_block_txs(self, block_hash: str):
        """Fetch and process block transactions."""
        if not HAS_REQUESTS or not block_hash:
            return

        try:
            # Get first page of transactions (up to 25)
            resp = requests.get(
                f"{self.MEMPOOL_API_URL}/block/{block_hash}/txs/0",
                timeout=10
            )
            if resp.status_code == 200:
                txs = resp.json()
                for tx in txs:
                    self._process_transaction(tx)
        except:
            pass

    def _process_transaction(self, tx: Dict):
        """Process a transaction for exchange flows."""
        with self._lock:
            self.txs_processed += 1

        txid = tx.get('txid', '')
        ts = time.time()

        # Extract inputs and outputs
        vin = tx.get('vin', [])
        vout = tx.get('vout', [])

        # Check inputs (OUTFLOW from exchange = LONG signal)
        for inp in vin:
            prevout = inp.get('prevout', {})
            addr = prevout.get('scriptpubkey_address', '')
            value_sat = prevout.get('value', 0)
            btc = value_sat / 1e8

            if addr and btc > 0 and addr in self.exchange_addresses_set:
                ex_id = self.address_to_exchange.get(addr, 'unknown_exchange')
                self._emit_tick(ex_id, btc, FlowType.OUTFLOW, txid, ts)

        # Check outputs (INFLOW to exchange = SHORT signal)
        for out in vout:
            addr = out.get('scriptpubkey_address', '')
            value_sat = out.get('value', 0)
            btc = value_sat / 1e8

            if addr and btc > 0 and addr in self.exchange_addresses_set:
                ex_id = self.address_to_exchange.get(addr, 'unknown_exchange')
                self._emit_tick(ex_id, btc, FlowType.INFLOW, txid, ts)

    def _emit_tick(self, ex_id: str, btc: float, flow_type: FlowType, txid: str, ts: float):
        """Emit a flow tick."""
        direction = 1 if flow_type == FlowType.OUTFLOW else -1
        price = self.reference_price if self.reference_price > 0 else 100000.0

        tick = ExchangeTick(
            exchange=ex_id, timestamp=ts, price=price,
            bid=price * 0.9999, ask=price * 1.0001, spread=price * 0.0002,
            volume=btc, volume_1m=btc, volume_5m=btc, volume_1h=btc,
            buy_volume=btc if direction == 1 else 0,
            sell_volume=btc if direction == -1 else 0,
            direction=direction, pressure=min(1.0, btc / 100),
            tx_count=1, source='mempool_api', txid=txid, flow_type=flow_type.value,
        )

        with self._lock:
            self.ticks_emitted += 1
            if direction == 1:
                self.outflows += 1
                self.total_outflow_btc += btc
            else:
                self.inflows += 1
                self.total_inflow_btc += btc

        # Log significant flows
        if btc >= 1.0:
            flow_str = 'OUTFLOW->LONG' if direction == 1 else 'INFLOW->SHORT'
            print(f"[{ex_id.upper()}] {btc:.2f} BTC {flow_str}")

        # Emit tick callback
        if self.on_tick:
            self.on_tick(tick)

        # Emit signal if significant
        if btc >= 0.5:
            self._emit_signal()

    def _emit_signal(self):
        """Emit aggregated trading signal."""
        signal = self.get_aggregated_signal()

        if self.on_signal and signal['should_trade']:
            self.on_signal(signal)

    def get_aggregated_signal(self) -> Dict:
        """Get aggregated trading signal."""
        with self._lock:
            net = self.total_outflow_btc - self.total_inflow_btc

            if net > 1.0:
                direction = 1
                strength = min(1.0, net / 50)
            elif net < -1.0:
                direction = -1
                strength = min(1.0, abs(net) / 50)
            else:
                direction = 0
                strength = 0.0

            confidence = min(0.9, strength * 0.8 + 0.1)

            return {
                'timestamp': time.time(),
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'net_flow': net,
                'total_inflow': self.total_inflow_btc,
                'total_outflow': self.total_outflow_btc,
                'inflow_btc': self.total_inflow_btc,
                'outflow_btc': self.total_outflow_btc,
                'should_trade': direction != 0,
                'source': 'mempool_api',
            }

    def set_reference_price(self, price: float):
        """Set reference price for USD calculations."""
        self.reference_price = price

    def _start_price_tracker(self):
        """Start background price tracking."""
        if not HAS_REQUESTS:
            return

        def price_loop():
            while self.running:
                try:
                    resp = requests.get(
                        "https://api.binance.com/api/v3/ticker/price",
                        params={"symbol": "BTCUSDT"},
                        timeout=5
                    )
                    if resp.status_code == 200:
                        self.reference_price = float(resp.json()['price'])
                except:
                    pass
                time.sleep(5)

        self._price_thread = threading.Thread(target=price_loop, daemon=True)
        self._price_thread.start()

    def get_stats(self) -> Dict:
        """Get feed statistics."""
        with self._lock:
            elapsed = time.time() - self.start_time if self.start_time > 0 else 1
            return {
                'running': self.running,
                'source': 'mempool.space API',
                'txs_processed': self.txs_processed,
                'ticks_emitted': self.ticks_emitted,
                'ticks_per_min': (self.ticks_emitted / elapsed) * 60,
                'inflows': self.inflows,
                'outflows': self.outflows,
                'total_inflow_btc': self.total_inflow_btc,
                'total_outflow_btc': self.total_outflow_btc,
                'net_flow_btc': self.total_outflow_btc - self.total_inflow_btc,
                'addresses_tracked': len(self._tracked_addresses),
                'total_addresses': len(self.address_to_exchange),
                'reference_price': self.reference_price,
            }

    def reset_counters(self):
        """Reset flow counters (for periodic signal generation)."""
        with self._lock:
            self.total_inflow_btc = 0.0
            self.total_outflow_btc = 0.0
            self.inflows = 0
            self.outflows = 0


class PollingBlockchainFeed:
    """
    Polling-based blockchain feed for environments without WebSocket.

    Uses REST API polling - slower but more reliable.
    """

    MEMPOOL_API_URL = "https://mempool.space/api"

    def __init__(self,
                 on_signal: Callable[[Dict], None] = None,
                 poll_interval: float = 10.0,
                 json_path: str = None):
        """
        Initialize polling feed.

        Args:
            on_signal: Callback for signals
            poll_interval: Seconds between polls
            json_path: Path to exchanges.json
        """
        self.on_signal = on_signal
        self.poll_interval = poll_interval

        self.wallet_tracker = ExchangeWalletTracker(json_path=json_path)
        self.address_to_exchange = self.wallet_tracker.address_to_exchange
        self.exchange_addresses_set = self.wallet_tracker.exchange_addresses_set

        self.running = False
        self._thread = None
        self._lock = threading.Lock()

        self.last_block_height = 0
        self.reference_price = 0.0

        # Stats
        self.start_time = 0.0
        self.blocks_processed = 0
        self.total_inflow_btc = 0.0
        self.total_outflow_btc = 0.0

        print(f"[POLL-FEED] Initialized with {len(self.address_to_exchange):,} addresses")

    def start(self) -> bool:
        """Start polling."""
        if not HAS_REQUESTS:
            print("[POLL-FEED] ERROR: requests not installed")
            return False

        self.running = True
        self.start_time = time.time()

        # Get current block height
        try:
            resp = requests.get(f"{self.MEMPOOL_API_URL}/blocks/tip/height", timeout=10)
            self.last_block_height = int(resp.text)
        except:
            self.last_block_height = 0

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

        print(f"[POLL-FEED] Started polling every {self.poll_interval}s")
        return True

    def stop(self):
        """Stop polling."""
        self.running = False
        print("[POLL-FEED] Stopped")

    def _poll_loop(self):
        """Main polling loop."""
        while self.running:
            try:
                self._check_new_blocks()
                self._update_price()
            except Exception as e:
                print(f"[POLL-FEED] Error: {e}")

            time.sleep(self.poll_interval)

    def _check_new_blocks(self):
        """Check for new blocks and process them."""
        try:
            resp = requests.get(f"{self.MEMPOOL_API_URL}/blocks/tip/height", timeout=10)
            current_height = int(resp.text)

            if current_height > self.last_block_height:
                # Process new blocks
                for height in range(self.last_block_height + 1, current_height + 1):
                    self._process_block(height)
                    self.blocks_processed += 1

                self.last_block_height = current_height

                # Emit signal after processing
                self._emit_signal()

        except:
            pass

    def _process_block(self, height: int):
        """Process a single block for exchange flows."""
        try:
            # Get block hash
            resp = requests.get(f"{self.MEMPOOL_API_URL}/block-height/{height}", timeout=10)
            if resp.status_code != 200:
                return
            block_hash = resp.text

            # Get transactions (first page)
            resp = requests.get(f"{self.MEMPOOL_API_URL}/block/{block_hash}/txs/0", timeout=15)
            if resp.status_code != 200:
                return

            txs = resp.json()
            for tx in txs:
                self._process_tx(tx)

        except:
            pass

    def _process_tx(self, tx: Dict):
        """Process transaction for flows."""
        for inp in tx.get('vin', []):
            prevout = inp.get('prevout', {})
            addr = prevout.get('scriptpubkey_address', '')
            value = prevout.get('value', 0) / 1e8

            if addr in self.exchange_addresses_set and value > 0:
                with self._lock:
                    self.total_outflow_btc += value

        for out in tx.get('vout', []):
            addr = out.get('scriptpubkey_address', '')
            value = out.get('value', 0) / 1e8

            if addr in self.exchange_addresses_set and value > 0:
                with self._lock:
                    self.total_inflow_btc += value

    def _update_price(self):
        """Update reference price."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": "BTCUSDT"},
                timeout=5
            )
            if resp.status_code == 200:
                self.reference_price = float(resp.json()['price'])
        except:
            pass

    def _emit_signal(self):
        """Emit trading signal."""
        signal = self.get_aggregated_signal()
        if self.on_signal:
            self.on_signal(signal)

    def get_aggregated_signal(self) -> Dict:
        """Get current signal."""
        with self._lock:
            net = self.total_outflow_btc - self.total_inflow_btc

            if net > 1.0:
                direction = 1
                confidence = min(0.8, 0.5 + net / 100)
            elif net < -1.0:
                direction = -1
                confidence = min(0.8, 0.5 + abs(net) / 100)
            else:
                direction = 0
                confidence = 0.0

            return {
                'timestamp': time.time(),
                'direction': direction,
                'confidence': confidence,
                'net_flow': net,
                'inflow_btc': self.total_inflow_btc,
                'outflow_btc': self.total_outflow_btc,
                'should_trade': direction != 0,
                'source': 'mempool_polling',
            }

    def get_stats(self) -> Dict:
        """Get stats."""
        with self._lock:
            return {
                'running': self.running,
                'blocks_processed': self.blocks_processed,
                'total_inflow_btc': self.total_inflow_btc,
                'total_outflow_btc': self.total_outflow_btc,
                'net_flow_btc': self.total_outflow_btc - self.total_inflow_btc,
                'reference_price': self.reference_price,
            }

    def reset_counters(self):
        """Reset counters."""
        with self._lock:
            self.total_inflow_btc = 0.0
            self.total_outflow_btc = 0.0


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("API-BASED BLOCKCHAIN FEED TEST")
    print("NO BITCOIN CORE REQUIRED")
    print("=" * 60)
    print()

    def on_signal(signal):
        dir_str = "LONG" if signal['direction'] == 1 else "SHORT" if signal['direction'] == -1 else "NEUTRAL"
        print(f"[SIGNAL] {dir_str} | Confidence: {signal['confidence']:.2f} | "
              f"Net Flow: {signal['net_flow']:.2f} BTC")

    # Use WebSocket feed
    feed = APIBlockchainFeed(on_signal=on_signal)

    if feed.start():
        print("\nFeed running. Press Ctrl+C to stop.\n")

        try:
            while True:
                time.sleep(30)
                stats = feed.get_stats()
                print(f"[STATS] TXs: {stats['txs_processed']} | "
                      f"Ticks: {stats['ticks_emitted']} | "
                      f"Net: {stats['net_flow_btc']:.2f} BTC | "
                      f"BTC: ${stats['reference_price']:,.0f}")
        except KeyboardInterrupt:
            pass

        feed.stop()
    else:
        print("Failed to start feed")
