"""
Price Logger
============

Captures BTC price every second for signal validation.
Uses multiple sources for redundancy.

Sources:
1. Binance WebSocket (primary)
2. Coinbase API (backup)
3. Kraken API (backup)
"""

import sqlite3
import time
import json
import threading
import requests
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebSocket support
try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


@dataclass
class PriceRecord:
    """A single price record."""
    timestamp: float
    timestamp_ms: int
    price: float
    bid: float
    ask: float
    spread: float
    volume_24h: float
    source: str


class PriceLogger:
    """
    Logs BTC price every second.

    Uses WebSocket for real-time data, falls back to REST API.
    """

    def __init__(self, db_path: str = "data/prices.db"):
        """
        Initialize price logger.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        self._lock = threading.Lock()

        # State
        self.running = False
        self.current_price = 0.0
        self.current_bid = 0.0
        self.current_ask = 0.0
        self.last_update = 0.0

        # WebSocket
        self._ws = None
        self._ws_thread = None

        # Stats
        self.prices_logged = 0
        self.start_time = time.time()

        logger.info(f"PriceLogger initialized: {self.db_path}")

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                price REAL NOT NULL,
                bid REAL,
                ask REAL,
                spread REAL,
                volume_24h REAL,
                source TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        # Index for fast lookups
        c.execute("CREATE INDEX IF NOT EXISTS idx_price_timestamp ON prices(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_price_timestamp_ms ON prices(timestamp_ms)")

        conn.commit()
        conn.close()

    def log_price(self, price: float, bid: float = 0, ask: float = 0,
                  volume: float = 0, source: str = "manual") -> int:
        """
        Log a price to the database.

        Args:
            price: Mid price
            bid: Best bid
            ask: Best ask
            volume: 24h volume
            source: Data source

        Returns:
            Price record ID
        """
        with self._lock:
            now = time.time()
            now_ms = int(now * 1000)

            # Calculate spread
            spread = ask - bid if bid > 0 and ask > 0 else 0

            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute("""
                INSERT INTO prices (
                    timestamp, timestamp_ms, price, bid, ask, spread, volume_24h, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (now, now_ms, price, bid, ask, spread, volume, source))

            price_id = c.lastrowid
            conn.commit()
            conn.close()

            # Update current state
            self.current_price = price
            self.current_bid = bid
            self.current_ask = ask
            self.last_update = now

            self.prices_logged += 1

            if self.prices_logged % 1000 == 0:
                logger.info(f"Prices logged: {self.prices_logged}")

            return price_id

    # =========================================================================
    # BINANCE WEBSOCKET (Primary Source)
    # =========================================================================

    def start_binance_ws(self):
        """Start Binance WebSocket for real-time prices."""
        if not HAS_WEBSOCKET:
            logger.error("websocket-client not installed. Run: pip install websocket-client")
            return False

        self.running = True
        self._ws_thread = threading.Thread(target=self._binance_ws_loop, daemon=True)
        self._ws_thread.start()
        logger.info("Binance WebSocket started")
        return True

    def _binance_ws_loop(self):
        """WebSocket connection loop."""
        url = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"

        while self.running:
            try:
                self._ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_binance_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close,
                )
                self._ws.run_forever()

                if self.running:
                    logger.warning("WebSocket disconnected, reconnecting in 5s...")
                    time.sleep(5)

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                time.sleep(5)

    def _on_binance_message(self, ws, message):
        """Handle Binance WebSocket message."""
        try:
            data = json.loads(message)
            bid = float(data.get('b', 0))
            ask = float(data.get('a', 0))
            price = (bid + ask) / 2

            self.log_price(price, bid, ask, source="binance_ws")

        except Exception as e:
            logger.error(f"Message parse error: {e}")

    def _on_ws_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")

    def _on_ws_close(self, ws, close_status, close_msg):
        """Handle WebSocket close."""
        logger.info(f"WebSocket closed: {close_status} {close_msg}")

    def stop_ws(self):
        """Stop WebSocket connection."""
        self.running = False
        if self._ws:
            self._ws.close()
        logger.info("WebSocket stopped")

    # =========================================================================
    # REST API FALLBACKS
    # =========================================================================

    def fetch_binance_rest(self) -> Optional[float]:
        """Fetch price from Binance REST API."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/bookTicker",
                params={"symbol": "BTCUSDT"},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                bid = float(data['bidPrice'])
                ask = float(data['askPrice'])
                price = (bid + ask) / 2
                self.log_price(price, bid, ask, source="binance_rest")
                return price
        except Exception as e:
            logger.error(f"Binance REST error: {e}")
        return None

    def fetch_coinbase_rest(self) -> Optional[float]:
        """Fetch price from Coinbase REST API."""
        try:
            resp = requests.get(
                "https://api.coinbase.com/v2/prices/BTC-USD/spot",
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                price = float(data['data']['amount'])
                self.log_price(price, source="coinbase_rest")
                return price
        except Exception as e:
            logger.error(f"Coinbase REST error: {e}")
        return None

    def fetch_kraken_rest(self) -> Optional[float]:
        """Fetch price from Kraken REST API."""
        try:
            resp = requests.get(
                "https://api.kraken.com/0/public/Ticker",
                params={"pair": "XBTUSD"},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                ticker = data['result']['XXBTZUSD']
                bid = float(ticker['b'][0])
                ask = float(ticker['a'][0])
                price = (bid + ask) / 2
                self.log_price(price, bid, ask, source="kraken_rest")
                return price
        except Exception as e:
            logger.error(f"Kraken REST error: {e}")
        return None

    # =========================================================================
    # POLLING MODE (Backup)
    # =========================================================================

    def start_polling(self, interval: float = 1.0):
        """
        Start polling mode (if WebSocket unavailable).

        Args:
            interval: Polling interval in seconds
        """
        self.running = True
        self._poll_thread = threading.Thread(
            target=self._polling_loop,
            args=(interval,),
            daemon=True
        )
        self._poll_thread.start()
        logger.info(f"Polling started: {interval}s interval")

    def _polling_loop(self, interval: float):
        """Polling loop."""
        sources = [
            self.fetch_binance_rest,
            self.fetch_coinbase_rest,
            self.fetch_kraken_rest,
        ]

        while self.running:
            success = False
            for fetch in sources:
                price = fetch()
                if price:
                    success = True
                    break

            if not success:
                logger.warning("All price sources failed")

            time.sleep(interval)

    def stop_polling(self):
        """Stop polling."""
        self.running = False
        logger.info("Polling stopped")

    # =========================================================================
    # DATA ACCESS
    # =========================================================================

    def get_price_at(self, timestamp: float, tolerance_ms: int = 1000) -> Optional[float]:
        """
        Get price at a specific timestamp.

        Args:
            timestamp: Unix timestamp
            tolerance_ms: How far to look (milliseconds)

        Returns:
            Price or None
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        target_ms = int(timestamp * 1000)

        # Find closest price within tolerance
        row = c.execute("""
            SELECT price, ABS(timestamp_ms - ?) as diff
            FROM prices
            WHERE timestamp_ms BETWEEN ? AND ?
            ORDER BY diff
            LIMIT 1
        """, (target_ms, target_ms - tolerance_ms, target_ms + tolerance_ms)).fetchone()

        conn.close()

        return row[0] if row else None

    def get_prices_range(self, start: float, end: float) -> list:
        """
        Get all prices in a time range.

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            List of (timestamp, price) tuples
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        rows = c.execute("""
            SELECT timestamp, price
            FROM prices
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """, (start, end)).fetchall()

        conn.close()
        return rows

    def get_current_price(self) -> float:
        """Get most recent price."""
        return self.current_price

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        total = c.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        first = c.execute("SELECT MIN(timestamp) FROM prices").fetchone()[0]
        last = c.execute("SELECT MAX(timestamp) FROM prices").fetchone()[0]
        avg_price = c.execute("SELECT AVG(price) FROM prices").fetchone()[0]

        conn.close()

        runtime = time.time() - self.start_time
        rate = total / runtime if runtime > 0 else 0

        return {
            'total_prices': total,
            'first_price': first,
            'last_price': last,
            'avg_price': avg_price,
            'current_price': self.current_price,
            'last_update': self.last_update,
            'runtime_hours': runtime / 3600,
            'prices_per_second': rate,
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PRICE LOGGER TEST")
    print("=" * 60)

    # Create logger
    price_logger = PriceLogger("test_prices.db")

    # Test REST APIs
    print("\nTesting REST APIs:")

    print("  Binance:", price_logger.fetch_binance_rest())
    print("  Coinbase:", price_logger.fetch_coinbase_rest())
    print("  Kraken:", price_logger.fetch_kraken_rest())

    # Show stats
    print("\nStats:")
    stats = price_logger.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Test WebSocket (brief)
    if HAS_WEBSOCKET:
        print("\nTesting WebSocket (5 seconds)...")
        price_logger.start_binance_ws()
        time.sleep(5)
        price_logger.stop_ws()

        print(f"Prices collected: {price_logger.prices_logged}")
        print(f"Current price: ${price_logger.current_price:,.2f}")
    else:
        print("\nWebSocket not available. Install: pip install websocket-client")

    # Cleanup
    Path("test_prices.db").unlink(missing_ok=True)
