#!/usr/bin/env python3
"""
MULTI-EXCHANGE PRICE FEED
=========================
Real-time price feeds from TOP 3 exchanges for HFT trading.

Exchanges (99.3% of flow):
1. Bitfinex - 55% (detect only, US blocked)
2. Binance.US - 26% (detect + execute)
3. Coinbase - 20% (detect + execute)

Usage:
    feed = MultiExchangePriceFeed()
    await feed.connect()
    prices = await feed.get_prices()  # All exchanges
    price = await feed.get_best_price()  # Best bid/ask
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque

try:
    import ccxt.async_support as ccxt_async
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    ccxt_async = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


@dataclass
class ExchangePrice:
    """Price data from a single exchange."""
    exchange: str
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    timestamp: float
    latency_ms: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return (self.spread / self.mid) * 10000 if self.mid > 0 else 0


@dataclass
class AggregatedPrice:
    """Aggregated price across exchanges."""
    best_bid: float
    best_ask: float
    best_bid_exchange: str
    best_ask_exchange: str
    prices: Dict[str, ExchangePrice]
    timestamp: float

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def cross_spread(self) -> float:
        """Spread between best bid/ask (arbitrage indicator)."""
        return self.best_ask - self.best_bid


class ExchangeFeed:
    """Single exchange price feed."""

    def __init__(self, exchange_id: str, symbol: str = "BTC/USDT"):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.exchange = None
        self.connected = False
        self.last_price: Optional[ExchangePrice] = None
        self.latencies: deque = deque(maxlen=100)

    async def connect(self) -> bool:
        """Connect to exchange."""
        if not HAS_CCXT:
            log.error("CCXT not installed")
            return False

        try:
            exchange_class = getattr(ccxt_async, self.exchange_id, None)
            if exchange_class is None:
                log.error(f"Unknown exchange: {self.exchange_id}")
                return False

            self.exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 10000,
            })

            await self.exchange.load_markets()
            self.connected = True
            log.info(f"Connected to {self.exchange_id}")
            return True

        except Exception as e:
            log.error(f"Failed to connect to {self.exchange_id}: {e}")
            return False

    async def disconnect(self):
        """Disconnect from exchange."""
        if self.exchange:
            await self.exchange.close()
            self.connected = False

    async def fetch_price(self) -> Optional[ExchangePrice]:
        """Fetch current price."""
        if not self.connected:
            return None

        try:
            start = time.time()
            ticker = await self.exchange.fetch_ticker(self.symbol)
            latency = (time.time() - start) * 1000

            self.latencies.append(latency)

            price = ExchangePrice(
                exchange=self.exchange_id,
                symbol=self.symbol,
                bid=float(ticker.get('bid', 0) or 0),
                ask=float(ticker.get('ask', 0) or 0),
                last=float(ticker.get('last', 0) or 0),
                volume_24h=float(ticker.get('quoteVolume', 0) or 0),
                timestamp=time.time(),
                latency_ms=latency,
            )

            self.last_price = price
            return price

        except Exception as e:
            log.warning(f"{self.exchange_id} fetch error: {e}")
            return None

    @property
    def avg_latency(self) -> float:
        """Average latency in ms."""
        if not self.latencies:
            return 0
        return sum(self.latencies) / len(self.latencies)


class MultiExchangePriceFeed:
    """
    Multi-exchange price aggregator.

    Tracks TOP 3 exchanges for Bitcoin flow:
    - Bitfinex (55% of flow, US blocked - detect only)
    - Binance.US (26% of flow, US accessible)
    - Coinbase (20% of flow, US accessible)
    """

    # Exchange configs for BTC trading
    EXCHANGE_CONFIGS = {
        # Bitfinex - Most flow but US blocked
        'bitfinex': {
            'symbol': 'BTC/USDT',
            'can_execute': False,  # US blocked
            'flow_share': 0.553,
        },
        # Binance.US - Primary execution venue
        'binanceus': {
            'symbol': 'BTC/USD',
            'can_execute': True,
            'flow_share': 0.258,
        },
        # Coinbase - Secondary execution venue
        'coinbase': {
            'symbol': 'BTC/USD',
            'can_execute': True,
            'flow_share': 0.202,
        },
        # Optional: Kraken for diversification
        'kraken': {
            'symbol': 'BTC/USD',
            'can_execute': True,
            'flow_share': 0.01,
        },
    }

    def __init__(self, exchanges: Optional[List[str]] = None):
        """
        Initialize multi-exchange feed.

        Args:
            exchanges: List of exchange IDs (default: top 3)
        """
        if exchanges is None:
            exchanges = ['bitfinex', 'binanceus', 'coinbase']

        self.feeds: Dict[str, ExchangeFeed] = {}
        self.prices: Dict[str, ExchangePrice] = {}
        self._running = False
        self._update_task = None
        self._callbacks: List[Callable] = []

        for ex in exchanges:
            if ex in self.EXCHANGE_CONFIGS:
                config = self.EXCHANGE_CONFIGS[ex]
                self.feeds[ex] = ExchangeFeed(ex, config['symbol'])
            else:
                log.warning(f"Unknown exchange config: {ex}")

    async def connect(self) -> int:
        """
        Connect to all exchanges.

        Returns:
            Number of successful connections
        """
        tasks = [feed.connect() for feed in self.feeds.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        connected = sum(1 for r in results if r is True)
        log.info(f"Connected to {connected}/{len(self.feeds)} exchanges")
        return connected

    async def disconnect(self):
        """Disconnect from all exchanges."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()

        for feed in self.feeds.values():
            await feed.disconnect()

    async def fetch_all_prices(self) -> Dict[str, ExchangePrice]:
        """Fetch prices from all exchanges concurrently."""
        tasks = []
        exchange_ids = []

        for ex_id, feed in self.feeds.items():
            if feed.connected:
                tasks.append(feed.fetch_price())
                exchange_ids.append(ex_id)

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for ex_id, result in zip(exchange_ids, results):
            if isinstance(result, ExchangePrice):
                self.prices[ex_id] = result

        return self.prices

    def get_aggregated_price(self) -> Optional[AggregatedPrice]:
        """Get aggregated price across all exchanges."""
        if not self.prices:
            return None

        valid_prices = {k: v for k, v in self.prices.items()
                        if v.bid > 0 and v.ask > 0}

        if not valid_prices:
            return None

        # Find best bid/ask
        best_bid = 0
        best_bid_ex = ""
        best_ask = float('inf')
        best_ask_ex = ""

        for ex_id, price in valid_prices.items():
            if price.bid > best_bid:
                best_bid = price.bid
                best_bid_ex = ex_id
            if price.ask < best_ask:
                best_ask = price.ask
                best_ask_ex = ex_id

        return AggregatedPrice(
            best_bid=best_bid,
            best_ask=best_ask,
            best_bid_exchange=best_bid_ex,
            best_ask_exchange=best_ask_ex,
            prices=valid_prices,
            timestamp=time.time(),
        )

    def get_execution_venue(self, side: str = "sell") -> Optional[str]:
        """
        Get best execution venue for US traders.

        Args:
            side: "buy" or "sell"

        Returns:
            Exchange ID with best price (US accessible only)
        """
        if not self.prices:
            return None

        best_ex = None
        best_price = 0 if side == "sell" else float('inf')

        for ex_id, price in self.prices.items():
            config = self.EXCHANGE_CONFIGS.get(ex_id, {})
            if not config.get('can_execute', False):
                continue  # Skip US-blocked exchanges

            if side == "sell":
                # Best bid for selling
                if price.bid > best_price:
                    best_price = price.bid
                    best_ex = ex_id
            else:
                # Best ask for buying
                if price.ask < best_price:
                    best_price = price.ask
                    best_ex = ex_id

        return best_ex

    async def start_streaming(self, interval_ms: int = 500):
        """
        Start streaming prices at regular intervals.

        Args:
            interval_ms: Update interval in milliseconds
        """
        self._running = True

        async def _stream():
            while self._running:
                prices = await self.fetch_all_prices()

                # Notify callbacks
                if prices:
                    agg = self.get_aggregated_price()
                    for callback in self._callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(agg)
                            else:
                                callback(agg)
                        except Exception as e:
                            log.error(f"Callback error: {e}")

                await asyncio.sleep(interval_ms / 1000)

        self._update_task = asyncio.create_task(_stream())
        log.info(f"Started price streaming at {interval_ms}ms intervals")

    def on_price_update(self, callback: Callable):
        """Register callback for price updates."""
        self._callbacks.append(callback)

    def get_stats(self) -> Dict:
        """Get feed statistics."""
        stats = {
            'exchanges': {},
            'total_connected': 0,
        }

        for ex_id, feed in self.feeds.items():
            ex_stats = {
                'connected': feed.connected,
                'avg_latency_ms': round(feed.avg_latency, 2),
                'last_price': feed.last_price.last if feed.last_price else None,
            }
            stats['exchanges'][ex_id] = ex_stats
            if feed.connected:
                stats['total_connected'] += 1

        return stats


async def demo():
    """Demo the multi-exchange feed."""
    print("=" * 60)
    print("MULTI-EXCHANGE PRICE FEED DEMO")
    print("=" * 60)

    if not HAS_CCXT:
        print("ERROR: CCXT not installed. Run: pip install ccxt")
        return

    # Create feed for TOP 3 exchanges
    feed = MultiExchangePriceFeed(['bitfinex', 'binanceus', 'coinbase'])

    # Connect
    connected = await feed.connect()
    print(f"\nConnected to {connected} exchanges")

    if connected == 0:
        print("No exchanges connected!")
        return

    # Fetch prices
    print("\nFetching prices...")
    for i in range(5):
        prices = await feed.fetch_all_prices()
        agg = feed.get_aggregated_price()

        if agg:
            print(f"\n--- Update {i+1} ---")
            print(f"Best Bid: ${agg.best_bid:,.2f} ({agg.best_bid_exchange})")
            print(f"Best Ask: ${agg.best_ask:,.2f} ({agg.best_ask_exchange})")
            print(f"Mid:      ${agg.mid:,.2f}")
            print(f"Cross Spread: ${agg.cross_spread:.2f}")

            # Per-exchange prices
            for ex_id, price in agg.prices.items():
                config = feed.EXCHANGE_CONFIGS.get(ex_id, {})
                can_exec = "EXEC" if config.get('can_execute') else "DETECT"
                print(f"  {ex_id:12} | Bid: ${price.bid:,.2f} | Ask: ${price.ask:,.2f} | "
                      f"Spread: {price.spread_bps:.1f}bps | {can_exec}")

        # Best execution venue for SHORT (sell)
        best_venue = feed.get_execution_venue("sell")
        if best_venue:
            print(f"\nBest SELL venue (US): {best_venue}")

        await asyncio.sleep(2)

    # Stats
    print("\n--- Feed Stats ---")
    stats = feed.get_stats()
    for ex_id, ex_stats in stats['exchanges'].items():
        status = "OK" if ex_stats['connected'] else "DISCONNECTED"
        print(f"  {ex_id}: {status} | Latency: {ex_stats['avg_latency_ms']}ms")

    # Cleanup
    await feed.disconnect()
    print("\nDisconnected")


if __name__ == "__main__":
    asyncio.run(demo())
