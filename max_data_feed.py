"""
Renaissance Trading - MAXIMUM USA Data Feed
============================================
Every US-accessible exchange, every tick, millisecond precision.

Target: 50+ ticks/second aggregated from all sources.
"""

import asyncio
import json
import time
from collections import deque
import websockets

class DataFeed:
    """Maximum speed multi-exchange data aggregator."""

    def __init__(self, symbol: str = 'BTC/USD'):
        self.symbol = symbol
        self.connections = []
        self.price = 0.0
        self.bid = 0.0
        self.ask = 0.0
        self.spread = 0.0
        self.prices = deque(maxlen=50000)
        self.volumes = deque(maxlen=50000)
        self.timestamps = deque(maxlen=50000)
        self.tick_count = 0
        self.connected = False
        self.last_update_ms = 0

        # ALL US-ACCESSIBLE EXCHANGES
        self.exchanges = {
            'kraken': {
                'url': 'wss://ws.kraken.com',
                'subscribe': {'event': 'subscribe', 'pair': ['XBT/USD'], 'subscription': {'name': 'trade'}}
            },
            'kraken_book': {
                'url': 'wss://ws.kraken.com',
                'subscribe': {'event': 'subscribe', 'pair': ['XBT/USD'], 'subscription': {'name': 'spread'}}
            },
            'coinbase': {
                'url': 'wss://ws-feed.exchange.coinbase.com',
                'subscribe': {'type': 'subscribe', 'product_ids': ['BTC-USD'], 'channels': ['ticker', 'matches']}
            },
            'gemini': {
                'url': 'wss://api.gemini.com/v1/marketdata/BTCUSD',
                'subscribe': None
            },
            'bitstamp': {
                'url': 'wss://ws.bitstamp.net',
                'subscribe': {'event': 'bts:subscribe', 'data': {'channel': 'live_trades_btcusd'}}
            },
            'bitstamp_book': {
                'url': 'wss://ws.bitstamp.net',
                'subscribe': {'event': 'bts:subscribe', 'data': {'channel': 'live_orders_btcusd'}}
            },
            'bitfinex': {
                'url': 'wss://api-pub.bitfinex.com/ws/2',
                'subscribe': {'event': 'subscribe', 'channel': 'trades', 'symbol': 'tBTCUSD'}
            },
            'okcoin': {
                'url': 'wss://real.okcoin.com:8443/ws/v5/public',
                'subscribe': {'op': 'subscribe', 'args': [{'channel': 'trades', 'instId': 'BTC-USD'}]}
            },
        }

    async def connect(self):
        """Connect to ALL exchanges simultaneously."""
        self.connected = True

        tasks = [self._connect_exchange(name, config) for name, config in self.exchanges.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        connected = sum(1 for r in results if r is True)
        print(f'Connected to {connected}/{len(self.exchanges)} exchanges')

        for _ in range(100):
            if self.price > 0:
                break
            await asyncio.sleep(0.05)

    async def _connect_exchange(self, name: str, config: dict) -> bool:
        """Connect to single exchange."""
        try:
            ws = await asyncio.wait_for(
                websockets.connect(config['url'], ping_interval=20, ping_timeout=10),
                timeout=5
            )
            self.connections.append((name, ws))

            if config['subscribe']:
                await ws.send(json.dumps(config['subscribe']))

            asyncio.create_task(self._listen_fast(name, ws))
            print(f'  [{name}] OK')
            return True

        except Exception as e:
            print(f'  [{name}] FAIL: {str(e)[:30]}')
            return False

    async def _listen_fast(self, name: str, ws):
        """Ultra-fast listener."""
        try:
            while self.connected:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    now_ms = int(time.time() * 1000)

                    price, volume, bid, ask = self._parse_fast(name, msg)

                    if price and price > 0:
                        self.price = price
                        self.prices.append(price)
                        self.volumes.append(volume or 0)
                        self.timestamps.append(now_ms)
                        self.tick_count += 1
                        self.last_update_ms = now_ms

                    if bid and ask:
                        self.bid = bid
                        self.ask = ask
                        self.spread = ask - bid

                except asyncio.TimeoutError:
                    continue

        except Exception:
            pass

    def _parse_fast(self, name: str, msg: str) -> tuple:
        """Ultra-fast message parsing."""
        try:
            data = json.loads(msg)

            if name == 'kraken':
                if isinstance(data, list) and len(data) >= 4:
                    trades = data[1]
                    if isinstance(trades, list) and trades:
                        t = trades[-1]
                        return float(t[0]), float(t[1]), None, None

            elif name == 'kraken_book':
                if isinstance(data, list) and len(data) >= 4:
                    spread = data[1]
                    if isinstance(spread, list) and len(spread) >= 2:
                        return None, None, float(spread[0]), float(spread[1])

            elif name == 'coinbase':
                if data.get('type') in ('ticker', 'match'):
                    p = data.get('price')
                    if p:
                        return float(p), float(data.get('last_size', data.get('size', 0))), None, None

            elif name == 'gemini':
                if 'events' in data:
                    for e in data['events']:
                        if e.get('type') == 'trade':
                            return float(e['price']), float(e.get('amount', 0)), None, None

            elif name in ('bitstamp', 'bitstamp_book'):
                if 'data' in data:
                    d = data['data']
                    if 'price' in d:
                        return float(d['price']), float(d.get('amount', 0)), None, None

            elif name == 'bitfinex':
                if isinstance(data, list) and len(data) >= 3:
                    if data[1] == 'te':
                        return float(data[2][3]), abs(float(data[2][2])), None, None

            elif name == 'okcoin':
                if 'data' in data:
                    for d in data['data']:
                        if 'px' in d:
                            return float(d['px']), float(d.get('sz', 0)), None, None

        except:
            pass

        return None, None, None, None

    async def get_price(self) -> float:
        return self.price

    def get_prices(self) -> list:
        return list(self.prices)

    def get_volumes(self) -> list:
        return list(self.volumes)

    def get_spread(self) -> float:
        return self.spread

    def get_tick_rate(self) -> float:
        if len(self.timestamps) < 2:
            return 0
        elapsed = (self.timestamps[-1] - self.timestamps[0]) / 1000
        return len(self.timestamps) / elapsed if elapsed > 0 else 0

    def get_stats(self) -> dict:
        return {
            'price': self.price,
            'bid': self.bid,
            'ask': self.ask,
            'spread': self.spread,
            'ticks': self.tick_count,
            'rate': self.get_tick_rate(),
            'exchanges': len(self.connections)
        }

    async def close(self):
        self.connected = False
        for name, ws in self.connections:
            try:
                await ws.close()
            except:
                pass
        self.connections = []
