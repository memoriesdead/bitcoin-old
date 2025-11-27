"""
Renaissance Trading System - WebSocket Data Feed
High-frequency real-time data for scalable trading

Supports:
- Kraken WebSocket API (USA-friendly)
- Tick buffering and aggregation
- OHLCV bar construction
- Multiple symbol support
- Auto-reconnection
"""
import asyncio
import json
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Any
import numpy as np

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("Warning: websockets not installed. Run: pip install websockets")


@dataclass
class Tick:
    """Single trade tick"""
    price: float
    volume: float
    timestamp: float
    side: str  # 'buy' or 'sell'


@dataclass
class OHLCV:
    """OHLCV bar"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: float
    tick_count: int = 0


class TickBuffer:
    """
    High-performance tick buffer with O(1) operations

    Features:
    - Circular buffer for memory efficiency
    - Real-time statistics (VWAP, spread, etc.)
    - Thread-safe operations
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.ticks: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()

        # Running statistics
        self.total_volume = 0.0
        self.total_value = 0.0  # price * volume
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.tick_count = 0

    def add(self, tick: Tick):
        """Add tick to buffer (thread-safe)"""
        with self.lock:
            # If buffer full, remove oldest and adjust stats
            if len(self.ticks) >= self.max_size:
                old = self.ticks[0]
                self.total_volume -= old.volume
                self.total_value -= old.price * old.volume
                if old.side == 'buy':
                    self.buy_volume -= old.volume
                else:
                    self.sell_volume -= old.volume

            self.ticks.append(tick)
            self.total_volume += tick.volume
            self.total_value += tick.price * tick.volume
            self.tick_count += 1

            if tick.side == 'buy':
                self.buy_volume += tick.volume
            else:
                self.sell_volume += tick.volume

    def get_latest(self, n: int = 1) -> List[Tick]:
        """Get latest n ticks"""
        with self.lock:
            return list(self.ticks)[-n:]

    def get_vwap(self) -> float:
        """Volume Weighted Average Price"""
        with self.lock:
            if self.total_volume == 0:
                return 0.0
            return self.total_value / self.total_volume

    def get_order_flow_imbalance(self) -> float:
        """
        Order flow imbalance: (buy - sell) / total
        Positive = buying pressure, Negative = selling pressure
        """
        with self.lock:
            total = self.buy_volume + self.sell_volume
            if total == 0:
                return 0.0
            return (self.buy_volume - self.sell_volume) / total

    def get_tick_rate(self, window_sec: float = 1.0) -> float:
        """Ticks per second in recent window"""
        with self.lock:
            if len(self.ticks) < 2:
                return 0.0
            now = time.time()
            count = 0
            for tick in reversed(self.ticks):
                if now - tick.timestamp <= window_sec:
                    count += 1
                else:
                    break
            return count / window_sec

    def get_prices(self, n: Optional[int] = None) -> np.ndarray:
        """Get price array for analysis"""
        with self.lock:
            if n is None:
                return np.array([t.price for t in self.ticks])
            return np.array([t.price for t in list(self.ticks)[-n:]])

    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.ticks.clear()
            self.total_volume = 0.0
            self.total_value = 0.0
            self.buy_volume = 0.0
            self.sell_volume = 0.0


class OHLCVAggregator:
    """
    Aggregates ticks into OHLCV bars

    Supports multiple timeframes simultaneously
    """

    def __init__(self, intervals: List[int] = None):
        """
        Args:
            intervals: Bar intervals in seconds (default: [1, 5, 15, 60])
        """
        self.intervals = intervals or [1, 5, 15, 60]
        self.current_bars: Dict[int, OHLCV] = {}
        self.completed_bars: Dict[int, deque] = {}
        self.callbacks: Dict[int, List[Callable]] = {}

        for interval in self.intervals:
            self.current_bars[interval] = None
            self.completed_bars[interval] = deque(maxlen=1000)
            self.callbacks[interval] = []

    def on_bar_complete(self, interval: int, callback: Callable[[OHLCV], None]):
        """Register callback for bar completion"""
        if interval in self.callbacks:
            self.callbacks[interval].append(callback)

    def process_tick(self, tick: Tick):
        """Process tick and update bars"""
        for interval in self.intervals:
            bar_start = int(tick.timestamp // interval) * interval

            current = self.current_bars[interval]

            if current is None or current.timestamp != bar_start:
                # Complete previous bar
                if current is not None:
                    self.completed_bars[interval].append(current)
                    for cb in self.callbacks[interval]:
                        try:
                            cb(current)
                        except Exception as e:
                            print(f"Bar callback error: {e}")

                # Start new bar
                self.current_bars[interval] = OHLCV(
                    open=tick.price,
                    high=tick.price,
                    low=tick.price,
                    close=tick.price,
                    volume=tick.volume,
                    timestamp=bar_start,
                    tick_count=1
                )
            else:
                # Update current bar
                current.high = max(current.high, tick.price)
                current.low = min(current.low, tick.price)
                current.close = tick.price
                current.volume += tick.volume
                current.tick_count += 1

    def get_bars(self, interval: int, n: Optional[int] = None) -> List[OHLCV]:
        """Get completed bars for interval"""
        if interval not in self.completed_bars:
            return []
        bars = list(self.completed_bars[interval])
        if n is not None:
            bars = bars[-n:]
        return bars


class KrakenWebSocket:
    """
    Kraken WebSocket client for real-time market data

    USA-friendly! No geo-restrictions.

    Features:
    - Auto-reconnection
    - Multiple symbol support
    - Trade and ticker streams
    - Thread-safe callbacks
    """

    # Kraken WebSocket endpoints
    WS_URL = "wss://ws.kraken.com"
    WS_URL_V2 = "wss://ws.kraken.com/v2"

    # Symbol mapping (standard -> Kraken)
    SYMBOL_MAP = {
        'BTCUSD': 'XBT/USD',
        'BTCUSDT': 'XBT/USDT',
        'ETHUSD': 'ETH/USD',
        'ETHUSDT': 'ETH/USDT',
    }

    def __init__(self, symbols: List[str] = None):
        """
        Args:
            symbols: List of symbols to subscribe (default: ['BTCUSD'])
        """
        self.symbols = symbols or ['BTCUSD']
        self.kraken_symbols = [self.SYMBOL_MAP.get(s, s) for s in self.symbols]

        self.ws = None
        self.running = False
        self.connected = False
        self.loop = None
        self.thread = None

        # Callbacks
        self.on_tick: Optional[Callable[[str, Tick], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

        # Stats
        self.message_count = 0
        self.tick_count = 0
        self.last_tick_time = 0
        self.connect_time = 0

    def _parse_trade(self, data: dict, symbol: str) -> List[Tick]:
        """Parse Kraken trade message into Ticks"""
        ticks = []

        # Kraken V1 trade format: [channelID, [[price, volume, time, side, orderType, misc], ...], channelName, pair]
        if isinstance(data, list) and len(data) >= 2:
            trades = data[1]
            if isinstance(trades, list):
                for trade in trades:
                    if len(trade) >= 4:
                        price = float(trade[0])
                        volume = float(trade[1])
                        timestamp = float(trade[2])
                        side = 'buy' if trade[3] == 'b' else 'sell'

                        ticks.append(Tick(
                            price=price,
                            volume=volume,
                            timestamp=timestamp,
                            side=side
                        ))
        return ticks

    def _parse_ticker(self, data: dict, symbol: str) -> Optional[Tick]:
        """Parse Kraken ticker message into Tick (uses last trade price)"""
        # Kraken ticker format: [channelID, {a: [ask], b: [bid], c: [close], ...}, channelName, pair]
        if isinstance(data, list) and len(data) >= 2:
            ticker = data[1]
            if isinstance(ticker, dict) and 'c' in ticker:
                # c = last trade [price, lot volume]
                close_data = ticker['c']
                if len(close_data) >= 2:
                    price = float(close_data[0])
                    volume = float(close_data[1])

                    # Determine side from bid/ask proximity
                    bid = float(ticker.get('b', [price])[0])
                    ask = float(ticker.get('a', [price])[0])
                    mid = (bid + ask) / 2
                    side = 'buy' if price >= mid else 'sell'

                    return Tick(
                        price=price,
                        volume=volume,
                        timestamp=time.time(),
                        side=side
                    )
        return None

    async def _connect(self):
        """Connect to WebSocket"""
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets library required: pip install websockets")

        try:
            self.ws = await websockets.connect(
                self.WS_URL,
                ping_interval=30,
                ping_timeout=10
            )
            self.connected = True
            self.connect_time = time.time()

            if self.on_connect:
                self.on_connect()

            # Subscribe to trades
            subscribe_msg = {
                "event": "subscribe",
                "pair": self.kraken_symbols,
                "subscription": {"name": "trade"}
            }
            await self.ws.send(json.dumps(subscribe_msg))

            # Also subscribe to ticker for spread data
            ticker_msg = {
                "event": "subscribe",
                "pair": self.kraken_symbols,
                "subscription": {"name": "ticker"}
            }
            await self.ws.send(json.dumps(ticker_msg))

            print(f"[WS] Connected to Kraken, subscribed to {self.kraken_symbols}")

        except Exception as e:
            self.connected = False
            if self.on_error:
                self.on_error(e)
            raise

    async def _listen(self):
        """Listen for messages"""
        while self.running and self.ws:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=30)
                self.message_count += 1

                data = json.loads(msg)

                # Skip system messages
                if isinstance(data, dict):
                    event = data.get('event', '')
                    if event in ['systemStatus', 'subscriptionStatus', 'heartbeat']:
                        continue

                # Parse trade and ticker messages
                if isinstance(data, list) and len(data) >= 4:
                    channel_name = data[-2] if isinstance(data[-2], str) else ''
                    pair = data[-1] if isinstance(data[-1], str) else ''

                    # Convert Kraken symbol back to standard
                    std_symbol = next(
                        (k for k, v in self.SYMBOL_MAP.items() if v == pair),
                        pair
                    )

                    if channel_name == 'trade':
                        ticks = self._parse_trade(data, pair)
                        for tick in ticks:
                            self.tick_count += 1
                            self.last_tick_time = time.time()
                            if self.on_tick:
                                self.on_tick(std_symbol, tick)

                    elif channel_name == 'ticker':
                        tick = self._parse_ticker(data, pair)
                        if tick:
                            self.tick_count += 1
                            self.last_tick_time = time.time()
                            if self.on_tick:
                                self.on_tick(std_symbol, tick)

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                if self.ws:
                    try:
                        pong = await self.ws.ping()
                        await asyncio.wait_for(pong, timeout=10)
                    except:
                        break
            except websockets.exceptions.ConnectionClosed:
                print("[WS] Connection closed, reconnecting...")
                break
            except Exception as e:
                if self.on_error:
                    self.on_error(e)

    async def _run(self):
        """Main run loop with auto-reconnect"""
        while self.running:
            try:
                await self._connect()
                await self._listen()
            except Exception as e:
                print(f"[WS] Error: {e}")
                if self.on_error:
                    self.on_error(e)

            self.connected = False
            if self.on_disconnect:
                self.on_disconnect()

            if self.running:
                print("[WS] Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    def _thread_run(self):
        """Run in separate thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run())

    def start(self):
        """Start WebSocket in background thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._thread_run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop WebSocket"""
        self.running = False
        if self.ws and self.loop:
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
        if self.thread:
            self.thread.join(timeout=5)

    def get_stats(self) -> dict:
        """Get connection statistics"""
        uptime = time.time() - self.connect_time if self.connect_time else 0
        return {
            'connected': self.connected,
            'uptime_sec': uptime,
            'message_count': self.message_count,
            'tick_count': self.tick_count,
            'ticks_per_sec': self.tick_count / uptime if uptime > 0 else 0,
            'last_tick_age': time.time() - self.last_tick_time if self.last_tick_time else None
        }


class WebSocketFeed:
    """
    High-level WebSocket feed manager

    Combines:
    - WebSocket client (Kraken)
    - Tick buffer
    - OHLCV aggregation
    - Strategy callbacks
    """

    def __init__(self, symbols: List[str] = None, buffer_size: int = 10000):
        """
        Args:
            symbols: Symbols to subscribe (default: ['BTCUSD'])
            buffer_size: Tick buffer size
        """
        self.symbols = symbols or ['BTCUSD']

        # Components
        self.ws = KrakenWebSocket(self.symbols)
        self.buffers: Dict[str, TickBuffer] = {s: TickBuffer(buffer_size) for s in self.symbols}
        self.aggregator = OHLCVAggregator([1, 5, 15, 60])

        # Callbacks
        self.on_tick_callbacks: List[Callable[[str, Tick], None]] = []
        self.on_bar_callbacks: Dict[int, List[Callable[[str, OHLCV], None]]] = {}

        # Wire up internal callbacks
        self.ws.on_tick = self._handle_tick
        self.ws.on_connect = self._on_connect
        self.ws.on_disconnect = self._on_disconnect

        # State
        self.running = False
        self.primary_symbol = self.symbols[0]

    def _handle_tick(self, symbol: str, tick: Tick):
        """Internal tick handler"""
        # Add to buffer
        if symbol in self.buffers:
            self.buffers[symbol].add(tick)

        # Aggregate to bars
        self.aggregator.process_tick(tick)

        # Call user callbacks
        for cb in self.on_tick_callbacks:
            try:
                cb(symbol, tick)
            except Exception as e:
                print(f"Tick callback error: {e}")

    def _on_connect(self):
        """Handle connection"""
        print(f"[Feed] Connected - {len(self.symbols)} symbols")

    def _on_disconnect(self):
        """Handle disconnection"""
        print("[Feed] Disconnected")

    def on_tick(self, callback: Callable[[str, Tick], None]):
        """Register tick callback"""
        self.on_tick_callbacks.append(callback)

    def on_bar(self, interval: int, callback: Callable[[str, OHLCV], None]):
        """Register bar completion callback"""
        if interval not in self.on_bar_callbacks:
            self.on_bar_callbacks[interval] = []
        self.on_bar_callbacks[interval].append(callback)

        # Wire up to aggregator
        def bar_wrapper(bar: OHLCV):
            callback(self.primary_symbol, bar)
        self.aggregator.on_bar_complete(interval, bar_wrapper)

    def start(self):
        """Start feed"""
        self.running = True
        self.ws.start()

    def stop(self):
        """Stop feed"""
        self.running = False
        self.ws.stop()

    def get_latest_price(self, symbol: str = None) -> Optional[float]:
        """Get latest price for symbol"""
        symbol = symbol or self.primary_symbol
        if symbol in self.buffers:
            ticks = self.buffers[symbol].get_latest(1)
            if ticks:
                return ticks[0].price
        return None

    def get_prices(self, symbol: str = None, n: Optional[int] = None) -> np.ndarray:
        """Get price array"""
        symbol = symbol or self.primary_symbol
        if symbol in self.buffers:
            return self.buffers[symbol].get_prices(n)
        return np.array([])

    def get_vwap(self, symbol: str = None) -> float:
        """Get VWAP"""
        symbol = symbol or self.primary_symbol
        if symbol in self.buffers:
            return self.buffers[symbol].get_vwap()
        return 0.0

    def get_order_flow_imbalance(self, symbol: str = None) -> float:
        """Get order flow imbalance"""
        symbol = symbol or self.primary_symbol
        if symbol in self.buffers:
            return self.buffers[symbol].get_order_flow_imbalance()
        return 0.0

    def get_tick_rate(self, symbol: str = None) -> float:
        """Get ticks per second"""
        symbol = symbol or self.primary_symbol
        if symbol in self.buffers:
            return self.buffers[symbol].get_tick_rate()
        return 0.0

    def get_stats(self) -> dict:
        """Get feed statistics"""
        ws_stats = self.ws.get_stats()
        buffer_stats = {}
        for symbol, buffer in self.buffers.items():
            buffer_stats[symbol] = {
                'tick_count': len(buffer.ticks),
                'vwap': buffer.get_vwap(),
                'order_flow': buffer.get_order_flow_imbalance(),
                'tick_rate': buffer.get_tick_rate()
            }
        return {
            'websocket': ws_stats,
            'buffers': buffer_stats
        }


# Convenience function for quick setup
def create_btc_feed(buffer_size: int = 10000) -> WebSocketFeed:
    """Create BTC/USD WebSocket feed"""
    return WebSocketFeed(symbols=['BTCUSD'], buffer_size=buffer_size)
