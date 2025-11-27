"""
Renaissance Trading System - Gemini WebSocket Client
US-regulated exchange with institutional-grade reliability

Gemini Features:
- New York Trust Company (NYDFS regulated)
- SOC 2 Type 2 certified
- Digital asset insurance
- All 50 US states supported

Reference: https://docs.gemini.com/websocket-api/
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


class GeminiWebSocket:
    """
    Gemini WebSocket client for real-time market data

    US-REGULATED - New York Trust Company
    - SOC 2 Type 2 certified
    - NYDFS regulated
    - Available in all 50 US states

    Endpoints:
    - V1: wss://api.gemini.com/v1/marketdata/{symbol}
    - V2: wss://api.gemini.com/v2/marketdata (multi-symbol)
    """

    # Use V2 for better multi-symbol support
    WS_URL = "wss://api.gemini.com/v2/marketdata"
    WS_URL_V1 = "wss://api.gemini.com/v1/marketdata"

    # Symbol mapping
    SYMBOL_MAP = {
        'BTCUSD': 'BTCUSD',
        'ETHUSD': 'ETHUSD',
        'BTCUSDT': 'BTCUSD',  # Gemini uses USD, not USDT
        'ETHUSDT': 'ETHUSD',
    }

    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['BTCUSD']
        self.gemini_symbols = [self.SYMBOL_MAP.get(s, s) for s in self.symbols]

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

        # Last known prices for each symbol
        self.last_prices = {}

    def _parse_trade(self, event: dict) -> Optional[Tick]:
        """Parse Gemini trade event"""
        try:
            # V2 trade event format
            if event.get('type') == 'trade':
                price = float(event['price'])
                quantity = float(event['quantity'])
                timestamp = event.get('timestamp', time.time() * 1000) / 1000  # ms to sec
                side = 'buy' if event.get('makerSide') == 'ask' else 'sell'

                return Tick(
                    price=price,
                    volume=quantity,
                    timestamp=timestamp,
                    side=side
                )
        except (KeyError, ValueError, TypeError):
            pass
        return None

    def _parse_l2_update(self, data: dict, symbol: str) -> Optional[Tick]:
        """Parse L2 update for price changes (ticker-like functionality)"""
        try:
            changes = data.get('changes', [])
            for change in changes:
                if len(change) >= 3:
                    side, price, qty = change[0], float(change[1]), float(change[2])
                    if qty > 0:  # Only non-zero quantities
                        self.last_prices[symbol] = price
                        return Tick(
                            price=price,
                            volume=qty,
                            timestamp=time.time(),
                            side='buy' if side == 'buy' else 'sell'
                        )
        except (KeyError, ValueError, TypeError):
            pass
        return None

    async def _connect(self):
        """Connect to Gemini WebSocket"""
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets library required")

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

            # V2 subscription message - subscribe to TRADES (candles has trades)
            # Gemini V2 doesn't have a direct trade channel, use l2 + candles
            subscribe_msg = {
                "type": "subscribe",
                "subscriptions": [
                    {
                        "name": "l2",
                        "symbols": self.gemini_symbols
                    },
                    {
                        "name": "candles_1m",  # 1-minute candles include trade data
                        "symbols": self.gemini_symbols
                    }
                ]
            }
            await self.ws.send(json.dumps(subscribe_msg))

            print(f"[GEMINI-WS] Connected, subscribed to {self.gemini_symbols}")

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
                msg_type = data.get('type', '')

                # Skip system messages
                if msg_type in ['subscription_ack', 'heartbeat']:
                    continue

                symbol = data.get('symbol', self.gemini_symbols[0] if self.gemini_symbols else 'BTCUSD')

                # Convert to standard symbol
                std_symbol = next(
                    (k for k, v in self.SYMBOL_MAP.items() if v == symbol),
                    symbol
                )

                # Parse trade events
                if msg_type == 'trade':
                    tick = self._parse_trade(data)
                    if tick:
                        self.tick_count += 1
                        self.last_tick_time = time.time()
                        if self.on_tick:
                            self.on_tick(std_symbol, tick)

                # Parse L2 updates (order book changes)
                elif msg_type == 'l2_updates':
                    tick = self._parse_l2_update(data, symbol)
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
                print("[GEMINI-WS] Connection closed, reconnecting...")
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
                print(f"[GEMINI-WS] Error: {e}")
                if self.on_error:
                    self.on_error(e)

            self.connected = False
            if self.on_disconnect:
                self.on_disconnect()

            if self.running:
                print("[GEMINI-WS] Reconnecting in 5 seconds...")
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
