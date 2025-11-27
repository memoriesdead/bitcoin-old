"""
Renaissance Trading System - Bitstamp WebSocket Client
Oldest running crypto exchange (since 2011) with proven reliability

Bitstamp Features:
- Founded 2011 (longest-running exchange)
- Luxembourg licensed
- NY BitLicense holder
- Available in all US states
- Institutional-grade reliability

Reference: https://www.bitstamp.net/websocket/v2/
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


class BitstampWebSocket:
    """
    Bitstamp WebSocket client for real-time market data

    OLDEST EXCHANGE - Operating since 2011
    - NY BitLicense holder
    - Luxembourg EU license
    - Available in all US states
    - Proven 13+ year reliability track record

    Endpoint: wss://ws.bitstamp.net
    Channels: live_trades_{pair}, order_book_{pair}, diff_order_book_{pair}
    """

    WS_URL = "wss://ws.bitstamp.net"

    # Symbol mapping (Bitstamp uses lowercase)
    SYMBOL_MAP = {
        'BTCUSD': 'btcusd',
        'ETHUSD': 'ethusd',
        'BTCUSDT': 'btcusd',  # Bitstamp uses USD
        'ETHUSDT': 'ethusd',
        'BTCEUR': 'btceur',
        'ETHEUR': 'etheur',
    }

    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['BTCUSD']
        self.bitstamp_symbols = [self.SYMBOL_MAP.get(s, s.lower()) for s in self.symbols]

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

    def _parse_trade(self, data: dict) -> Optional[Tick]:
        """Parse Bitstamp trade message"""
        try:
            trade_data = data.get('data', {})

            price = float(trade_data['price'])
            amount = float(trade_data['amount'])
            timestamp = float(trade_data.get('timestamp', time.time()))

            # Bitstamp: type 0 = buy, type 1 = sell
            trade_type = trade_data.get('type', 0)
            side = 'buy' if trade_type == 0 else 'sell'

            return Tick(
                price=price,
                volume=amount,
                timestamp=timestamp,
                side=side
            )
        except (KeyError, ValueError, TypeError):
            pass
        return None

    async def _connect(self):
        """Connect to Bitstamp WebSocket"""
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

            # Subscribe to live trades for each symbol
            for symbol in self.bitstamp_symbols:
                subscribe_msg = {
                    "event": "bts:subscribe",
                    "data": {
                        "channel": f"live_trades_{symbol}"
                    }
                }
                await self.ws.send(json.dumps(subscribe_msg))

                # Also subscribe to order book for ticker-like updates
                order_book_msg = {
                    "event": "bts:subscribe",
                    "data": {
                        "channel": f"diff_order_book_{symbol}"
                    }
                }
                await self.ws.send(json.dumps(order_book_msg))

            print(f"[BITSTAMP-WS] Connected, subscribed to {self.bitstamp_symbols}")

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
                event = data.get('event', '')
                channel = data.get('channel', '')

                # Skip system messages
                if event in ['bts:subscription_succeeded', 'bts:heartbeat', 'bts:request_reconnect']:
                    if event == 'bts:request_reconnect':
                        print("[BITSTAMP-WS] Server requested reconnect")
                        break
                    continue

                # Extract symbol from channel name
                # Format: live_trades_btcusd or diff_order_book_btcusd
                symbol = None
                for sym in self.bitstamp_symbols:
                    if sym in channel:
                        symbol = sym
                        break

                if not symbol:
                    continue

                # Convert to standard symbol
                std_symbol = next(
                    (k for k, v in self.SYMBOL_MAP.items() if v == symbol),
                    symbol.upper()
                )

                # Parse trade events
                if event == 'trade' and 'live_trades' in channel:
                    tick = self._parse_trade(data)
                    if tick:
                        self.tick_count += 1
                        self.last_tick_time = time.time()
                        if self.on_tick:
                            self.on_tick(std_symbol, tick)

                # Parse order book updates (for price changes)
                elif event == 'data' and 'order_book' in channel:
                    order_data = data.get('data', {})
                    bids = order_data.get('bids', [])
                    asks = order_data.get('asks', [])

                    # Use best bid/ask as price indicator
                    if bids and asks:
                        try:
                            best_bid = float(bids[0][0]) if bids else 0
                            best_ask = float(asks[0][0]) if asks else 0
                            if best_bid > 0 and best_ask > 0:
                                mid_price = (best_bid + best_ask) / 2
                                tick = Tick(
                                    price=mid_price,
                                    volume=0.001,  # Nominal volume for price updates
                                    timestamp=time.time(),
                                    side='buy' if best_ask < best_bid else 'sell'
                                )
                                self.tick_count += 1
                                self.last_tick_time = time.time()
                                if self.on_tick:
                                    self.on_tick(std_symbol, tick)
                        except (IndexError, ValueError):
                            pass

            except asyncio.TimeoutError:
                if self.ws:
                    try:
                        pong = await self.ws.ping()
                        await asyncio.wait_for(pong, timeout=10)
                    except:
                        break
            except websockets.exceptions.ConnectionClosed:
                print("[BITSTAMP-WS] Connection closed, reconnecting...")
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
                print(f"[BITSTAMP-WS] Error: {e}")
                if self.on_error:
                    self.on_error(e)

            self.connected = False
            if self.on_disconnect:
                self.on_disconnect()

            if self.running:
                print("[BITSTAMP-WS] Reconnecting in 5 seconds...")
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
