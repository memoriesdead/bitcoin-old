"""
Renaissance Trading System - Coinbase WebSocket Client
High-frequency data from Coinbase (USA's largest crypto exchange)

Uses Coinbase Advanced Trade API for MAXIMUM tick data.
- market_trades channel: EVERY trade in real-time
- No authentication needed for market data
- FREE tier supports this

Reference: https://docs.cdp.coinbase.com/advanced-trade/docs/ws-overview
"""
import asyncio
import json
import time
import threading
from typing import Callable, Optional, List

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

from .websocket_feed import Tick


class CoinbaseWebSocket:
    """
    Coinbase Advanced Trade WebSocket client for MAXIMUM tick data

    USA-friendly with highest liquidity!
    Uses Advanced Trade API (newest) for more data than old Exchange API.

    Features:
    - market_trades channel: EVERY trade with millisecond timestamps
    - No auth needed for market data (FREE)
    - Auto-reconnection

    Endpoint: wss://advanced-trade-ws.coinbase.com
    """

    # Use Advanced Trade API (newest, most data)
    WS_URL = "wss://advanced-trade-ws.coinbase.com"
    # Fallback to old Exchange API
    WS_URL_EXCHANGE = "wss://ws-feed.exchange.coinbase.com"

    # Symbol mapping
    SYMBOL_MAP = {
        'BTCUSD': 'BTC-USD',
        'ETHUSD': 'ETH-USD',
        'BTCUSDT': 'BTC-USDT',
        'ETHUSDT': 'ETH-USDT',
    }

    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['BTCUSD']
        self.coinbase_symbols = [self.SYMBOL_MAP.get(s, s) for s in self.symbols]

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

    def _parse_market_trades(self, data: dict) -> list:
        """Parse Advanced Trade API market_trades message - EVERY TRADE"""
        ticks = []
        if data.get('channel') == 'market_trades':
            events = data.get('events', [])
            for event in events:
                trades = event.get('trades', [])
                for trade in trades:
                    try:
                        price = float(trade['price'])
                        volume = float(trade['size'])

                        # Parse timestamp (ISO format with milliseconds)
                        time_str = trade.get('time', '')
                        if time_str:
                            from datetime import datetime
                            try:
                                # Handle ISO format: 2024-01-01T00:00:00.000000Z
                                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                                timestamp = dt.timestamp()
                            except:
                                timestamp = time.time()
                        else:
                            timestamp = time.time()

                        side = 'buy' if trade.get('side', '').upper() == 'BUY' else 'sell'

                        ticks.append(Tick(
                            price=price,
                            volume=volume,
                            timestamp=timestamp,
                            side=side
                        ))
                    except (KeyError, ValueError, TypeError):
                        pass
        return ticks

    def _parse_match(self, data: dict) -> Optional[Tick]:
        """Parse old Exchange API match (trade) message - fallback"""
        if data.get('type') == 'match' or data.get('type') == 'last_match':
            try:
                price = float(data['price'])
                volume = float(data['size'])

                # Parse timestamp
                time_str = data.get('time', '')
                if time_str:
                    from datetime import datetime
                    try:
                        dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                        timestamp = dt.timestamp()
                    except:
                        timestamp = time.time()
                else:
                    timestamp = time.time()

                side = 'buy' if data.get('side') == 'buy' else 'sell'

                return Tick(
                    price=price,
                    volume=volume,
                    timestamp=timestamp,
                    side=side
                )
            except (KeyError, ValueError) as e:
                pass
        return None

    def _parse_ticker(self, data: dict) -> Optional[Tick]:
        """Parse Coinbase ticker message"""
        if data.get('type') == 'ticker':
            try:
                price = float(data['price'])
                volume = float(data.get('last_size', 0.001))

                # Determine side from bid/ask
                bid = float(data.get('best_bid', price))
                ask = float(data.get('best_ask', price))
                mid = (bid + ask) / 2
                side = 'buy' if price >= mid else 'sell'

                return Tick(
                    price=price,
                    volume=volume,
                    timestamp=time.time(),
                    side=side
                )
            except (KeyError, ValueError):
                pass
        return None

    async def _connect(self):
        """Connect to Coinbase Advanced Trade WebSocket"""
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets library required")

        try:
            # Try Advanced Trade API first (more data)
            self.ws = await websockets.connect(
                self.WS_URL,
                ping_interval=30,
                ping_timeout=10
            )
            self.connected = True
            self.connect_time = time.time()
            self.use_advanced_api = True

            if self.on_connect:
                self.on_connect()

            # Advanced Trade API subscription format
            # market_trades = EVERY trade (highest frequency)
            subscribe_msg = {
                "type": "subscribe",
                "product_ids": self.coinbase_symbols,
                "channel": "market_trades"
            }
            await self.ws.send(json.dumps(subscribe_msg))

            print(f"[CB-WS] Connected to Coinbase Advanced Trade API, subscribed to market_trades for {self.coinbase_symbols}")

        except Exception as e:
            # Fallback to old Exchange API
            print(f"[CB-WS] Advanced API failed ({e}), trying Exchange API...")
            try:
                self.ws = await websockets.connect(
                    self.WS_URL_EXCHANGE,
                    ping_interval=30,
                    ping_timeout=10
                )
                self.connected = True
                self.connect_time = time.time()
                self.use_advanced_api = False

                if self.on_connect:
                    self.on_connect()

                # Old Exchange API subscription
                subscribe_msg = {
                    "type": "subscribe",
                    "product_ids": self.coinbase_symbols,
                    "channels": ["matches", "ticker"]
                }
                await self.ws.send(json.dumps(subscribe_msg))

                print(f"[CB-WS] Connected to Coinbase Exchange API (fallback), subscribed to {self.coinbase_symbols}")

            except Exception as e2:
                self.connected = False
                if self.on_error:
                    self.on_error(e2)
                raise

    async def _listen(self):
        """Listen for messages - handles both Advanced Trade and Exchange APIs"""
        while self.running and self.ws:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=30)
                self.message_count += 1

                data = json.loads(msg)

                # Advanced Trade API format
                if getattr(self, 'use_advanced_api', False):
                    channel = data.get('channel', '')
                    msg_type = data.get('type', '')

                    # Skip subscription confirmations and heartbeats
                    if msg_type in ['subscriptions', 'heartbeat', 'snapshot']:
                        continue

                    # market_trades channel - EVERY TRADE
                    if channel == 'market_trades':
                        ticks = self._parse_market_trades(data)
                        for tick in ticks:
                            self.tick_count += 1
                            self.last_tick_time = time.time()
                            # Get product ID from events
                            events = data.get('events', [])
                            product_id = ''
                            if events and events[0].get('trades'):
                                product_id = events[0]['trades'][0].get('product_id', 'BTC-USD')
                            std_symbol = next(
                                (k for k, v in self.SYMBOL_MAP.items() if v == product_id),
                                product_id.replace('-', '')
                            )
                            if self.on_tick:
                                self.on_tick(std_symbol, tick)

                # Old Exchange API format (fallback)
                else:
                    msg_type = data.get('type', '')

                    # Skip subscription confirmations
                    if msg_type in ['subscriptions', 'heartbeat']:
                        continue

                    product_id = data.get('product_id', '')

                    # Convert symbol back to standard
                    std_symbol = next(
                        (k for k, v in self.SYMBOL_MAP.items() if v == product_id),
                        product_id.replace('-', '')
                    )

                    # Parse matches (trades) - highest priority
                    if msg_type in ['match', 'last_match']:
                        tick = self._parse_match(data)
                        if tick:
                            self.tick_count += 1
                            self.last_tick_time = time.time()
                            if self.on_tick:
                                self.on_tick(std_symbol, tick)

                    # Parse ticker updates
                    elif msg_type == 'ticker':
                        tick = self._parse_ticker(data)
                        if tick:
                            self.tick_count += 1
                            self.last_tick_time = time.time()
                            if self.on_tick:
                                self.on_tick(std_symbol, tick)

            except asyncio.TimeoutError:
                if self.ws:
                    try:
                        pong = await self.ws.ping()
                        await asyncio.wait_for(pong, timeout=10)
                    except:
                        break
            except websockets.exceptions.ConnectionClosed:
                print("[CB-WS] Connection closed, reconnecting...")
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
                print(f"[CB-WS] Error: {e}")
                if self.on_error:
                    self.on_error(e)

            self.connected = False
            if self.on_disconnect:
                self.on_disconnect()

            if self.running:
                print("[CB-WS] Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    def _thread_run(self):
        """Run in separate thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run())

    def start(self):
        """Start WebSocket"""
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
        }
