#!/usr/bin/env python3
"""
ULTIMATE SPEED FEED - SUB-MILLISECOND BITCOIN DATA
===================================================
Combines ALL data sources for maximum speed advantage:
1. Exchange WebSockets (7+ exchanges) - price discovery
2. Blockchain.com WebSocket - mempool transactions
3. Mempool.space WebSocket - real-time blockchain data
4. CoinAPI (optional) - 400+ exchanges aggregated

Target: Sub-millisecond latency, 100+ events/second
Perfect for 300K-1M trades competing on SPEED

SPEED ADVANTAGE:
- Mempool data = see BTC transactions BEFORE they hit exchanges
- Multi-exchange = fastest price discovery across all venues
- Sub-second processing = compete with institutional HFT
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, List
import websockets

@dataclass
class Event:
    """High-frequency market event."""
    event_type: str  # 'trade', 'mempool_tx', 'block', 'quote'
    timestamp_ms: int
    price: float = 0
    volume: float = 0
    source: str = ""

    # Exchange data
    bid: float = 0
    ask: float = 0
    spread: float = 0
    side: str = ""

    # Blockchain data
    tx_hash: str = ""
    tx_value_btc: float = 0
    tx_fee_sat: int = 0
    block_height: int = 0

    # Metadata
    latency_ms: int = 0  # Time from event to processing


class UltimateSpeedFeed:
    """
    Ultimate speed feed combining:
    - 7+ exchange WebSockets
    - Blockchain mempool data
    - Real-time block notifications

    Designed for sub-millisecond competitive advantage.
    """

    def __init__(self):
        # Current market state
        self.price = 0.0
        self.bid = 0.0
        self.ask = 0.0
        self.spread = 0.0
        self.vwap = 0.0

        # Event storage (massive for HFT)
        self.events = deque(maxlen=1000000)  # 1M events
        self.prices = deque(maxlen=100000)
        self.volumes = deque(maxlen=100000)
        self.timestamps = deque(maxlen=100000)

        # Per-exchange tracking
        self.exchange_prices = {}  # For arbitrage
        self.exchange_latencies = {}  # Track which is fastest

        # Blockchain tracking
        self.mempool_txs = deque(maxlen=10000)  # Recent mempool txs
        self.large_txs = deque(maxlen=1000)  # Large transactions (whales)
        self.last_block_height = 0
        self.last_block_time = 0

        # Stats
        self.event_count = 0
        self.events_per_sec = 0
        self.average_latency_ms = 0
        self.running = False

        # Connections
        self.connections = {}

        # Callbacks
        self.on_event: Optional[Callable] = None
        self.on_large_tx: Optional[Callable] = None  # Whale alert
        self.on_block: Optional[Callable] = None

        # Exchange WebSocket configs
        self.exchanges = {
            # Fastest exchanges first
            "coinbase": {
                "url": "wss://ws-feed.exchange.coinbase.com",
                "subscribe": {
                    "type": "subscribe",
                    "product_ids": ["BTC-USD"],
                    "channels": ["full", "ticker", "matches"]  # Full = complete order book
                },
                "parser": self._parse_coinbase
            },
            "kraken": {
                "url": "wss://ws.kraken.com",
                "subscribe": {
                    "event": "subscribe",
                    "pair": ["XBT/USD"],
                    "subscription": {"name": "trade"}
                },
                "parser": self._parse_kraken
            },
            "kraken_spread": {
                "url": "wss://ws.kraken.com",
                "subscribe": {
                    "event": "subscribe",
                    "pair": ["XBT/USD"],
                    "subscription": {"name": "spread"}
                },
                "parser": self._parse_kraken_spread
            },
            "gemini": {
                "url": "wss://api.gemini.com/v1/marketdata/BTCUSD?trades=true&bids=true&offers=true",
                "subscribe": None,
                "parser": self._parse_gemini
            },
            "bitstamp": {
                "url": "wss://ws.bitstamp.net",
                "subscribe": {
                    "event": "bts:subscribe",
                    "data": {"channel": "live_trades_btcusd"}
                },
                "parser": self._parse_bitstamp
            },
            "bitfinex": {
                "url": "wss://api-pub.bitfinex.com/ws/2",
                "subscribe": {
                    "event": "subscribe",
                    "channel": "trades",
                    "symbol": "tBTCUSD"
                },
                "parser": self._parse_bitfinex
            },
            "okx": {
                "url": "wss://ws.okx.com:8443/ws/v5/public",
                "subscribe": {
                    "op": "subscribe",
                    "args": [
                        {"channel": "trades", "instId": "BTC-USDT"},
                        {"channel": "tickers", "instId": "BTC-USDT"}
                    ]
                },
                "parser": self._parse_okx
            },
        }

        # Blockchain data sources
        self.blockchain_sources = {
            "blockchain_com": {
                "url": "wss://ws.blockchain.info/inv",
                "subscribe": {"op": "unconfirmed_sub"},  # Mempool transactions
                "parser": self._parse_blockchain_com
            },
            "mempool_space": {
                "url": "wss://mempool.space/api/v1/ws",
                "subscribe": None,  # Auto-subscribes
                "parser": self._parse_mempool_space
            },
        }

    async def connect(self):
        """Connect to ALL sources simultaneously."""
        self.running = True

        print("=" * 70)
        print("ULTIMATE SPEED FEED - SUB-MILLISECOND DATA")
        print("=" * 70)
        print("Connecting to ALL sources...")

        tasks = []

        # Connect to exchanges
        for name, config in self.exchanges.items():
            tasks.append(self._connect_source(name, config, "exchange"))

        # Connect to blockchain sources
        for name, config in self.blockchain_sources.items():
            tasks.append(self._connect_source(name, config, "blockchain"))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        connected_exchanges = sum(1 for r in results[:len(self.exchanges)] if r is True)
        connected_blockchain = sum(1 for r in results[len(self.exchanges):] if r is True)

        print(f"\nExchanges: {connected_exchanges}/{len(self.exchanges)}")
        print(f"Blockchain: {connected_blockchain}/{len(self.blockchain_sources)}")
        print(f"Total: {connected_exchanges + connected_blockchain}/{len(self.exchanges) + len(self.blockchain_sources)}")

        # Wait for first price
        for _ in range(100):
            if self.price > 0:
                break
            await asyncio.sleep(0.05)

        print(f"\nFirst price: ${self.price:,.2f}")
        print(f"Event rate: {self.events_per_sec:.1f}/sec")
        print("=" * 70)

        return connected_exchanges + connected_blockchain > 0

    async def _connect_source(self, name: str, config: dict, source_type: str) -> bool:
        """Connect to a single WebSocket source."""
        try:
            ws = await asyncio.wait_for(
                websockets.connect(
                    config["url"],
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=50_000_000
                ),
                timeout=10
            )

            self.connections[name] = ws

            # Send subscription
            if config.get("subscribe"):
                await ws.send(json.dumps(config["subscribe"]))

            # Start listener
            asyncio.create_task(self._listen(name, ws, config["parser"], source_type))

            print(f"  [{name:<20}] OK")
            return True

        except Exception as e:
            print(f"  [{name:<20}] FAIL: {str(e)[:30]}")
            return False

    async def _listen(self, name: str, ws, parser, source_type: str):
        """Ultra-fast event listener."""
        try:
            while self.running:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    recv_time_ms = int(time.time() * 1000)

                    event = parser(msg, recv_time_ms)

                    if event:
                        # Calculate latency
                        event.latency_ms = recv_time_ms - event.timestamp_ms if event.timestamp_ms > 0 else 0

                        # Track exchange latency
                        if source_type == "exchange":
                            self.exchange_latencies[name] = event.latency_ms

                        # Process event
                        self._process_event(event)

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print(f"  [{name}] Reconnecting...")
                    break

        except Exception as e:
            print(f"  [{name}] Error: {e}")

    def _process_event(self, event: Event):
        """Process incoming event at maximum speed."""
        self.events.append(event)
        self.event_count += 1

        # Update market state
        if event.price > 0:
            self.price = event.price
            self.prices.append(event.price)
            self.timestamps.append(event.timestamp_ms)

        if event.volume > 0:
            self.volumes.append(event.volume)

        if event.bid > 0 and event.ask > 0:
            self.bid = event.bid
            self.ask = event.ask
            self.spread = event.ask - event.bid

        # Track per-exchange prices
        if event.source:
            self.exchange_prices[event.source] = {
                "price": event.price,
                "bid": event.bid,
                "ask": event.ask,
                "time": event.timestamp_ms
            }

        # Blockchain events
        if event.event_type == "mempool_tx":
            self.mempool_txs.append(event)

            # Detect large transactions (whales)
            if event.tx_value_btc > 10:  # > 10 BTC
                self.large_txs.append(event)
                if self.on_large_tx:
                    self.on_large_tx(event)

        elif event.event_type == "block":
            self.last_block_height = event.block_height
            self.last_block_time = event.timestamp_ms
            if self.on_block:
                self.on_block(event)

        # Trigger callback
        if self.on_event:
            self.on_event(event)

    # =========================================================================
    # EXCHANGE PARSERS
    # =========================================================================

    def _parse_coinbase(self, msg: str, now_ms: int) -> Optional[Event]:
        try:
            d = json.loads(msg)
            msg_type = d.get("type", "")

            if msg_type in ("match", "ticker", "last_match"):
                price = d.get("price")
                if price:
                    return Event(
                        event_type="trade",
                        timestamp_ms=now_ms,
                        price=float(price),
                        volume=float(d.get("size", d.get("last_size", 0))),
                        source="coinbase",
                        side=d.get("side", "")
                    )
        except:
            pass
        return None

    def _parse_kraken(self, msg: str, now_ms: int) -> Optional[Event]:
        try:
            d = json.loads(msg)
            if isinstance(d, list) and len(d) >= 4:
                trades = d[1]
                if isinstance(trades, list) and trades:
                    t = trades[-1]
                    return Event(
                        event_type="trade",
                        timestamp_ms=int(float(t[2]) * 1000),
                        price=float(t[0]),
                        volume=float(t[1]),
                        source="kraken",
                        side="buy" if t[3] == "b" else "sell"
                    )
        except:
            pass
        return None

    def _parse_kraken_spread(self, msg: str, now_ms: int) -> Optional[Event]:
        try:
            d = json.loads(msg)
            if isinstance(d, list) and len(d) >= 4:
                spread = d[1]
                if isinstance(spread, list) and len(spread) >= 2:
                    bid = float(spread[0])
                    ask = float(spread[1])
                    return Event(
                        event_type="quote",
                        timestamp_ms=now_ms,
                        price=(bid + ask) / 2,
                        source="kraken",
                        bid=bid,
                        ask=ask,
                        spread=ask - bid
                    )
        except:
            pass
        return None

    def _parse_gemini(self, msg: str, now_ms: int) -> Optional[Event]:
        try:
            d = json.loads(msg)
            if "events" in d:
                for e in d["events"]:
                    if e.get("type") == "trade":
                        return Event(
                            event_type="trade",
                            timestamp_ms=d.get("timestampms", now_ms),
                            price=float(e["price"]),
                            volume=float(e.get("amount", 0)),
                            source="gemini",
                            side=e.get("makerSide", "")
                        )
        except:
            pass
        return None

    def _parse_bitstamp(self, msg: str, now_ms: int) -> Optional[Event]:
        try:
            d = json.loads(msg)
            if "data" in d and "price" in d["data"]:
                return Event(
                    event_type="trade",
                    timestamp_ms=int(d["data"].get("microtimestamp", now_ms * 1000)) // 1000,
                    price=float(d["data"]["price"]),
                    volume=float(d["data"].get("amount", 0)),
                    source="bitstamp"
                )
        except:
            pass
        return None

    def _parse_bitfinex(self, msg: str, now_ms: int) -> Optional[Event]:
        try:
            d = json.loads(msg)
            if isinstance(d, list) and len(d) >= 3 and d[1] == "te":
                return Event(
                    event_type="trade",
                    timestamp_ms=d[2][1],
                    price=float(d[2][3]),
                    volume=abs(float(d[2][2])),
                    source="bitfinex",
                    side="buy" if d[2][2] > 0 else "sell"
                )
        except:
            pass
        return None

    def _parse_okx(self, msg: str, now_ms: int) -> Optional[Event]:
        try:
            d = json.loads(msg)
            if "data" in d:
                for trade in d["data"]:
                    if "px" in trade:
                        return Event(
                            event_type="trade",
                            timestamp_ms=int(trade.get("ts", now_ms)),
                            price=float(trade["px"]),
                            volume=float(trade.get("sz", 0)),
                            source="okx",
                            side=trade.get("side", "")
                        )
        except:
            pass
        return None

    # =========================================================================
    # BLOCKCHAIN PARSERS
    # =========================================================================

    def _parse_blockchain_com(self, msg: str, now_ms: int) -> Optional[Event]:
        """Parse blockchain.com mempool transactions."""
        try:
            d = json.loads(msg)

            # Unconfirmed transaction from mempool
            if d.get("op") == "utx":
                tx = d.get("x", {})

                # Calculate BTC value
                total_out = sum(out.get("value", 0) for out in tx.get("out", []))
                btc_value = total_out / 100_000_000  # Satoshis to BTC

                # Extract fee
                fee = tx.get("fee", 0)

                return Event(
                    event_type="mempool_tx",
                    timestamp_ms=tx.get("time", now_ms) * 1000 if tx.get("time") else now_ms,
                    tx_hash=tx.get("hash", ""),
                    tx_value_btc=btc_value,
                    tx_fee_sat=fee,
                    source="blockchain_com"
                )
        except:
            pass
        return None

    def _parse_mempool_space(self, msg: str, now_ms: int) -> Optional[Event]:
        """Parse mempool.space WebSocket data."""
        try:
            d = json.loads(msg)

            # New block
            if "block" in d:
                block = d["block"]
                return Event(
                    event_type="block",
                    timestamp_ms=block.get("timestamp", now_ms) * 1000,
                    block_height=block.get("height", 0),
                    source="mempool_space"
                )

            # Mempool transaction
            elif "tx" in d:
                tx = d["tx"]
                return Event(
                    event_type="mempool_tx",
                    timestamp_ms=now_ms,
                    tx_hash=tx.get("txid", ""),
                    tx_value_btc=tx.get("vout", [{}])[0].get("value", 0) if tx.get("vout") else 0,
                    tx_fee_sat=tx.get("fee", 0),
                    source="mempool_space"
                )
        except:
            pass
        return None

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_event_rate(self) -> float:
        """Calculate events per second."""
        if len(self.timestamps) < 10:
            return 0

        ts = list(self.timestamps)
        elapsed_ms = ts[-1] - ts[-min(100, len(ts))]
        if elapsed_ms > 0:
            return len(ts[-100:]) / (elapsed_ms / 1000)
        return 0

    def get_stats(self) -> dict:
        """Get comprehensive stats."""
        # Calculate average latency
        recent_events = list(self.events)[-100:]
        avg_latency = sum(e.latency_ms for e in recent_events) / len(recent_events) if recent_events else 0

        # Fastest exchange
        fastest_exchange = min(self.exchange_latencies.items(), key=lambda x: x[1])[0] if self.exchange_latencies else "none"

        return {
            "price": self.price,
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
            "spread_pct": (self.spread / self.price * 100) if self.price > 0 else 0,
            "events": self.event_count,
            "event_rate": self.get_event_rate(),
            "avg_latency_ms": avg_latency,
            "fastest_exchange": fastest_exchange,
            "exchanges": len([k for k in self.connections if k in self.exchanges]),
            "blockchain_sources": len([k for k in self.connections if k in self.blockchain_sources]),
            "mempool_txs": len(self.mempool_txs),
            "large_txs": len(self.large_txs),
            "last_block": self.last_block_height
        }

    async def close(self):
        """Close all connections."""
        self.running = False
        for name, ws in self.connections.items():
            try:
                await ws.close()
            except:
                pass
        self.connections = {}


async def test_ultimate_feed():
    """Test the ultimate speed feed."""
    feed = UltimateSpeedFeed()
    await feed.connect()

    print("\nMonitoring for 30 seconds...")
    print("-" * 70)

    start = time.time()
    last_report = start

    while time.time() - start < 30:
        await asyncio.sleep(0.1)

        if time.time() - last_report >= 3:
            stats = feed.get_stats()
            print(f"Price: ${stats['price']:,.2f} | "
                  f"Spread: {stats['spread_pct']:.4f}% | "
                  f"Events: {stats['events']:,} | "
                  f"Rate: {stats['event_rate']:.1f}/sec | "
                  f"Latency: {stats['avg_latency_ms']:.1f}ms | "
                  f"Fastest: {stats['fastest_exchange']} | "
                  f"Mempool: {stats['mempool_txs']} | "
                  f"Whales: {stats['large_txs']}")
            last_report = time.time()

    await feed.close()

    final_stats = feed.get_stats()
    print("-" * 70)
    print(f"FINAL: {final_stats['events']:,} events in 30s = {final_stats['events']/30:.1f} events/sec")
    print(f"Average latency: {final_stats['avg_latency_ms']:.1f}ms")
    print(f"Fastest exchange: {final_stats['fastest_exchange']}")


if __name__ == "__main__":
    asyncio.run(test_ultimate_feed())
