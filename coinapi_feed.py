#!/usr/bin/env python3
"""
COINAPI ULTRA-FAST DATA FEED
============================
One-stop API capturing EVERYTHING:
- 400+ exchanges through single endpoint
- Full order book (L1, L2, L3)
- WebSocket real-time + REST historical
- 99.9% uptime SLA

Target: 100+ ticks/second with sub-millisecond precision
Perfect for 300,000 - 1,000,000 trades
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable
import websockets
import aiohttp

@dataclass
class Tick:
    """Single price tick with full metadata."""
    price: float
    volume: float
    timestamp_ms: int
    source: str
    bid: float = 0
    ask: float = 0
    side: str = ""
    sequence: int = 0

class CoinAPIFeed:
    """
    CoinAPI WebSocket feed - captures EVERYTHING.

    Pricing tiers:
    - Free: 100 requests/day (testing only)
    - Startup: $79/mo - 100,000 requests/day
    - Pro: $199/mo - 500,000 requests/day
    - Business: $599/mo - unlimited WebSocket

    For 300K-1M trades: Business tier recommended
    """

    def __init__(self, api_key: str = None, symbols: list = None):
        # CoinAPI key - get free trial at coinapi.io
        self.api_key = api_key or "YOUR_COINAPI_KEY"

        # Default to BTC across major exchanges
        self.symbols = symbols or [
            "COINBASE_SPOT_BTC_USD",
            "KRAKEN_SPOT_XBT_USD",
            "GEMINI_SPOT_BTC_USD",
            "BITSTAMP_SPOT_BTC_USD",
            "BITFINEX_SPOT_BTC_USD",
            "BINANCE_SPOT_BTC_USDT",  # USDT pair
            "OKX_SPOT_BTC_USDT",
        ]

        # State
        self.price = 0.0
        self.bid = 0.0
        self.ask = 0.0
        self.spread = 0.0
        self.vwap = 0.0  # Volume-weighted average price

        # Tick storage (massive for HFT)
        self.ticks = deque(maxlen=500000)
        self.prices = deque(maxlen=100000)
        self.volumes = deque(maxlen=100000)
        self.timestamps = deque(maxlen=100000)

        # Per-exchange prices for arbitrage detection
        self.exchange_prices = {}

        # Stats
        self.tick_count = 0
        self.tick_rate = 0.0
        self.sequence = 0
        self.last_tick_ms = 0

        # Connection
        self.ws = None
        self.running = False
        self.connected_exchanges = set()

        # Callbacks for real-time processing
        self.on_tick: Optional[Callable] = None
        self.on_spread_change: Optional[Callable] = None
        self.on_arbitrage: Optional[Callable] = None

    async def connect(self):
        """Connect to CoinAPI WebSocket."""
        self.running = True

        print("=" * 60)
        print("COINAPI ULTRA-FAST FEED")
        print("=" * 60)
        print(f"Symbols: {len(self.symbols)}")
        print("Connecting to CoinAPI WebSocket...")

        try:
            # CoinAPI WebSocket endpoint
            ws_url = "wss://ws.coinapi.io/v1/"

            self.ws = await asyncio.wait_for(
                websockets.connect(
                    ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    max_size=50_000_000,  # Large for order book
                    extra_headers={"X-CoinAPI-Key": self.api_key}
                ),
                timeout=15
            )

            # Subscribe to trades for all symbols
            subscribe_msg = {
                "type": "hello",
                "apikey": self.api_key,
                "heartbeat": False,
                "subscribe_data_type": ["trade", "quote"],  # trades + bid/ask
                "subscribe_filter_symbol_id": self.symbols
            }

            await self.ws.send(json.dumps(subscribe_msg))
            print("Subscription sent...")

            # Start listener
            asyncio.create_task(self._listen())

            # Wait for first price
            for _ in range(100):
                if self.price > 0:
                    break
                await asyncio.sleep(0.1)

            print(f"Connected! First price: ${self.price:,.2f}")
            print(f"Exchanges: {len(self.connected_exchanges)}")
            print("=" * 60)
            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def _listen(self):
        """Ultra-fast message listener."""
        try:
            while self.running and self.ws:
                try:
                    msg = await asyncio.wait_for(self.ws.recv(), timeout=30)
                    now_ms = int(time.time() * 1000)

                    data = json.loads(msg)
                    msg_type = data.get("type", "")

                    if msg_type == "trade":
                        tick = self._parse_trade(data, now_ms)
                        if tick:
                            self._process_tick(tick)

                    elif msg_type == "quote":
                        self._parse_quote(data, now_ms)

                    elif msg_type == "error":
                        print(f"CoinAPI Error: {data.get('message', 'Unknown')}")

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("CoinAPI disconnected, reconnecting...")
                    await self._reconnect()

        except Exception as e:
            print(f"Listener error: {e}")

    async def _reconnect(self):
        """Reconnect to CoinAPI."""
        for attempt in range(5):
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            if await self.connect():
                return
        print("Failed to reconnect after 5 attempts")

    def _parse_trade(self, data: dict, now_ms: int) -> Optional[Tick]:
        """Parse CoinAPI trade message."""
        try:
            symbol = data.get("symbol_id", "")
            exchange = symbol.split("_")[0] if symbol else "unknown"

            # Extract timestamp (CoinAPI uses ISO format)
            ts_str = data.get("time_exchange", "") or data.get("time_coinapi", "")
            if ts_str:
                # Parse ISO timestamp to ms
                from datetime import datetime
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                timestamp_ms = int(dt.timestamp() * 1000)
            else:
                timestamp_ms = now_ms

            self.sequence += 1
            self.connected_exchanges.add(exchange)

            return Tick(
                price=float(data.get("price", 0)),
                volume=float(data.get("size", data.get("base_amount", 0))),
                timestamp_ms=timestamp_ms,
                source=exchange,
                side=data.get("taker_side", ""),
                sequence=self.sequence
            )
        except Exception:
            return None

    def _parse_quote(self, data: dict, now_ms: int):
        """Parse CoinAPI quote (bid/ask) message."""
        try:
            symbol = data.get("symbol_id", "")
            exchange = symbol.split("_")[0] if symbol else "unknown"

            bid = float(data.get("bid_price", 0))
            ask = float(data.get("ask_price", 0))

            if bid > 0 and ask > 0:
                self.bid = bid
                self.ask = ask
                self.spread = ask - bid

                # Store per-exchange for arbitrage
                self.exchange_prices[exchange] = {
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2,
                    "time": now_ms
                }

                # Detect arbitrage opportunities
                self._check_arbitrage()

                if self.on_spread_change:
                    self.on_spread_change(self.spread)

        except Exception:
            pass

    def _process_tick(self, tick: Tick):
        """Process incoming tick."""
        self.price = tick.price
        self.prices.append(tick.price)
        self.volumes.append(tick.volume)
        self.timestamps.append(tick.timestamp_ms)
        self.ticks.append(tick)
        self.tick_count += 1
        self.last_tick_ms = tick.timestamp_ms

        # Store per-exchange price
        self.exchange_prices[tick.source] = {
            "price": tick.price,
            "time": tick.timestamp_ms
        }

        # Calculate VWAP (Volume-Weighted Average Price)
        if len(self.prices) >= 100:
            recent_prices = list(self.prices)[-100:]
            recent_volumes = list(self.volumes)[-100:]
            total_vol = sum(recent_volumes)
            if total_vol > 0:
                self.vwap = sum(p * v for p, v in zip(recent_prices, recent_volumes)) / total_vol

        # Trigger callback
        if self.on_tick:
            self.on_tick(tick)

    def _check_arbitrage(self):
        """Check for arbitrage opportunities across exchanges."""
        if len(self.exchange_prices) < 2:
            return

        now = int(time.time() * 1000)
        recent = {k: v for k, v in self.exchange_prices.items()
                  if now - v.get("time", 0) < 5000}  # Last 5 seconds

        if len(recent) < 2:
            return

        # Find best bid and best ask across exchanges
        best_bid = 0
        best_ask = float('inf')
        bid_exchange = ""
        ask_exchange = ""

        for exchange, data in recent.items():
            if "bid" in data and data["bid"] > best_bid:
                best_bid = data["bid"]
                bid_exchange = exchange
            if "ask" in data and data["ask"] < best_ask:
                best_ask = data["ask"]
                ask_exchange = exchange

        # Arbitrage exists if we can buy at ask_exchange and sell at bid_exchange
        if best_bid > best_ask and bid_exchange != ask_exchange:
            profit_pct = (best_bid - best_ask) / best_ask * 100
            if self.on_arbitrage and profit_pct > 0.01:  # > 0.01% profit
                self.on_arbitrage({
                    "buy_exchange": ask_exchange,
                    "buy_price": best_ask,
                    "sell_exchange": bid_exchange,
                    "sell_price": best_bid,
                    "profit_pct": profit_pct
                })

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_tick_rate(self) -> float:
        """Calculate ticks per second."""
        if len(self.timestamps) < 10:
            return 0

        ts = list(self.timestamps)
        elapsed_ms = ts[-1] - ts[-min(100, len(ts))]
        if elapsed_ms > 0:
            return len(ts[-100:]) / (elapsed_ms / 1000)
        return 0

    def get_price(self) -> float:
        return self.price

    def get_prices(self) -> list:
        return list(self.prices)

    def get_volumes(self) -> list:
        return list(self.volumes)

    def get_spread(self) -> float:
        return self.spread

    def get_vwap(self) -> float:
        return self.vwap

    def get_exchange_prices(self) -> dict:
        """Get per-exchange prices for arbitrage."""
        return self.exchange_prices.copy()

    def get_stats(self) -> dict:
        return {
            "price": self.price,
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
            "spread_pct": (self.spread / self.price * 100) if self.price > 0 else 0,
            "vwap": self.vwap,
            "ticks": self.tick_count,
            "tick_rate": self.get_tick_rate(),
            "exchanges": len(self.connected_exchanges),
            "sequence": self.sequence,
            "last_tick_ms": self.last_tick_ms
        }

    async def close(self):
        """Close connection."""
        self.running = False
        if self.ws:
            await self.ws.close()
            self.ws = None


class FallbackMultiFeed:
    """
    Fallback: Direct exchange WebSockets (free, no API key needed).
    Use this if you don't have CoinAPI key yet.
    """

    def __init__(self):
        self.price = 0.0
        self.bid = 0.0
        self.ask = 0.0
        self.spread = 0.0

        self.prices = deque(maxlen=100000)
        self.volumes = deque(maxlen=100000)
        self.timestamps = deque(maxlen=100000)
        self.ticks = deque(maxlen=500000)

        self.tick_count = 0
        self.running = False
        self.connections = {}

        # All free WebSocket sources
        self.exchanges = {
            "coinbase": {
                "url": "wss://ws-feed.exchange.coinbase.com",
                "subscribe": {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["matches", "ticker"]}
            },
            "kraken": {
                "url": "wss://ws.kraken.com",
                "subscribe": {"event": "subscribe", "pair": ["XBT/USD"], "subscription": {"name": "trade"}}
            },
            "kraken_spread": {
                "url": "wss://ws.kraken.com",
                "subscribe": {"event": "subscribe", "pair": ["XBT/USD"], "subscription": {"name": "spread"}}
            },
            "gemini": {
                "url": "wss://api.gemini.com/v1/marketdata/BTCUSD?trades=true",
                "subscribe": None
            },
            "bitstamp": {
                "url": "wss://ws.bitstamp.net",
                "subscribe": {"event": "bts:subscribe", "data": {"channel": "live_trades_btcusd"}}
            },
            "bitfinex": {
                "url": "wss://api-pub.bitfinex.com/ws/2",
                "subscribe": {"event": "subscribe", "channel": "trades", "symbol": "tBTCUSD"}
            },
            "okx": {
                "url": "wss://ws.okx.com:8443/ws/v5/public",
                "subscribe": {"op": "subscribe", "args": [{"channel": "trades", "instId": "BTC-USDT"}]}
            },
        }

    async def connect(self):
        """Connect to all exchanges."""
        self.running = True
        print("=" * 60)
        print("MULTI-EXCHANGE FALLBACK FEED (Free)")
        print("=" * 60)

        tasks = []
        for name, config in self.exchanges.items():
            tasks.append(self._connect_one(name, config))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        connected = sum(1 for r in results if r is True)
        print(f"\nConnected: {connected}/{len(self.exchanges)}")

        for _ in range(100):
            if self.price > 0:
                break
            await asyncio.sleep(0.05)

        print(f"First price: ${self.price:,.2f}")
        print("=" * 60)
        return connected > 0

    async def _connect_one(self, name: str, config: dict) -> bool:
        try:
            ws = await asyncio.wait_for(
                websockets.connect(config["url"], ping_interval=20, ping_timeout=10),
                timeout=10
            )
            self.connections[name] = ws

            if config["subscribe"]:
                await ws.send(json.dumps(config["subscribe"]))

            asyncio.create_task(self._listen(name, ws))
            print(f"  [{name}] OK")
            return True
        except Exception as e:
            print(f"  [{name}] FAIL: {str(e)[:30]}")
            return False

    async def _listen(self, name: str, ws):
        try:
            while self.running:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    now_ms = int(time.time() * 1000)

                    price, volume, bid, ask = self._parse(name, msg)

                    if price and price > 0:
                        self.price = price
                        self.prices.append(price)
                        self.volumes.append(volume or 0)
                        self.timestamps.append(now_ms)
                        self.tick_count += 1

                    if bid and ask:
                        self.bid = bid
                        self.ask = ask
                        self.spread = ask - bid

                except asyncio.TimeoutError:
                    continue
        except Exception:
            pass

    def _parse(self, name: str, msg: str) -> tuple:
        try:
            data = json.loads(msg)

            if name == "coinbase":
                if data.get("type") in ("match", "ticker"):
                    p = data.get("price")
                    if p:
                        return float(p), float(data.get("size", 0)), None, None

            elif name == "kraken":
                if isinstance(data, list) and len(data) >= 4:
                    trades = data[1]
                    if isinstance(trades, list) and trades:
                        t = trades[-1]
                        return float(t[0]), float(t[1]), None, None

            elif name == "kraken_spread":
                if isinstance(data, list) and len(data) >= 4:
                    spread = data[1]
                    if isinstance(spread, list) and len(spread) >= 2:
                        return None, None, float(spread[0]), float(spread[1])

            elif name == "gemini":
                if "events" in data:
                    for e in data["events"]:
                        if e.get("type") == "trade":
                            return float(e["price"]), float(e.get("amount", 0)), None, None

            elif name == "bitstamp":
                if "data" in data and "price" in data["data"]:
                    return float(data["data"]["price"]), float(data["data"].get("amount", 0)), None, None

            elif name == "bitfinex":
                if isinstance(data, list) and len(data) >= 3 and data[1] == "te":
                    return float(data[2][3]), abs(float(data[2][2])), None, None

            elif name == "okx":
                if "data" in data:
                    for d in data["data"]:
                        if "px" in d:
                            return float(d["px"]), float(d.get("sz", 0)), None, None

        except Exception:
            pass
        return None, None, None, None

    def get_price(self) -> float:
        return self.price

    def get_prices(self) -> list:
        return list(self.prices)

    def get_volumes(self) -> list:
        return list(self.volumes)

    def get_spread(self) -> float:
        return self.spread

    def get_tick_rate(self) -> float:
        if len(self.timestamps) < 10:
            return 0
        ts = list(self.timestamps)
        elapsed_ms = ts[-1] - ts[-min(100, len(ts))]
        if elapsed_ms > 0:
            return len(ts[-100:]) / (elapsed_ms / 1000)
        return 0

    def get_stats(self) -> dict:
        return {
            "price": self.price,
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
            "spread_pct": (self.spread / self.price * 100) if self.price > 0 else 0,
            "ticks": self.tick_count,
            "tick_rate": self.get_tick_rate(),
            "exchanges": len(self.connections)
        }

    async def close(self):
        self.running = False
        for name, ws in self.connections.items():
            try:
                await ws.close()
            except:
                pass
        self.connections = {}


async def get_best_feed(api_key: str = None):
    """Get the best available feed - CoinAPI if key provided, else fallback."""
    if api_key and api_key != "YOUR_COINAPI_KEY":
        feed = CoinAPIFeed(api_key)
        if await feed.connect():
            return feed

    # Fallback to free multi-exchange
    feed = FallbackMultiFeed()
    await feed.connect()
    return feed


async def test_feed():
    """Test the data feed."""
    # Try CoinAPI first, fall back to free WebSockets
    feed = await get_best_feed()

    print("\nMonitoring for 30 seconds...")
    print("-" * 60)

    start = time.time()
    last_report = start

    while time.time() - start < 30:
        await asyncio.sleep(0.1)

        if time.time() - last_report >= 3:
            stats = feed.get_stats()
            print(f"Price: ${stats['price']:,.2f} | "
                  f"Bid: ${stats['bid']:,.2f} | "
                  f"Ask: ${stats['ask']:,.2f} | "
                  f"Spread: {stats['spread_pct']:.4f}% | "
                  f"Ticks: {stats['ticks']} | "
                  f"Rate: {stats['tick_rate']:.1f}/sec")
            last_report = time.time()

    await feed.close()

    print("-" * 60)
    print(f"FINAL: {feed.tick_count} ticks in 30s = {feed.tick_count/30:.1f} ticks/sec")


if __name__ == "__main__":
    asyncio.run(test_feed())
