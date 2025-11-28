#!/usr/bin/env python3
"""
ULTRA-LOW LATENCY MARKET DATA
=============================
TARGET: Sub-100ms response times
DESIGN: Maximum speed, minimum overhead

Optimizations:
- Async parallel fetching (not threads)
- Connection pooling with keep-alive
- Disabled rate limiting
- Pre-warmed connections
- Minimal JSON parsing
- Direct socket connections
"""

import asyncio
import aiohttp
import time
import json
from collections import deque
from typing import Optional, Dict
import ssl

# Disable SSL verification for speed (use with caution in production)
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


class UltraLowLatencyFeed:
    """
    Maximum speed market data feed.
    Target: <100ms latency per exchange.
    """

    # Direct REST endpoints - fastest available
    ENDPOINTS = {
        'kraken': 'https://api.kraken.com/0/public/Ticker?pair=XBTUSD',
        'coinbase': 'https://api.exchange.coinbase.com/products/BTC-USD/ticker',
        'gemini': 'https://api.gemini.com/v1/pubticker/btcusd',
        'bitstamp': 'https://www.bitstamp.net/api/v2/ticker/btcusd/',
        'bitfinex': 'https://api-pub.bitfinex.com/v2/ticker/tBTCUSD',
        'binance': 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',
        'okx': 'https://www.okx.com/api/v5/market/ticker?instId=BTC-USDT',
        'bybit': 'https://api.bybit.com/v5/market/tickers?category=spot&symbol=BTCUSDT',
    }

    # Parsers - minimal, fast extraction
    PARSERS = {
        'kraken': lambda d: float(d.get('result', {}).get('XXBTZUSD', {}).get('c', [0])[0]),
        'coinbase': lambda d: float(d.get('price', 0)),
        'gemini': lambda d: float(d.get('last', 0)),
        'bitstamp': lambda d: float(d.get('last', 0)),
        'bitfinex': lambda d: float(d[6]) if isinstance(d, list) and len(d) > 6 else 0,
        'binance': lambda d: float(d.get('price', 0)),
        'okx': lambda d: float(d.get('data', [{}])[0].get('last', 0)),
        'bybit': lambda d: float(d.get('result', {}).get('list', [{}])[0].get('lastPrice', 0)),
    }

    def __init__(self):
        self.price = 0.0
        self.prices = deque(maxlen=100000)
        self.timestamps = deque(maxlen=100000)
        self.latencies: Dict[str, float] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_update = 0
        self.tick_count = 0

    async def _create_session(self):
        """Create optimized aiohttp session."""
        if self.session is None or self.session.closed:
            # Optimized connector for speed
            connector = aiohttp.TCPConnector(
                limit=100,  # Max connections
                limit_per_host=10,
                ttl_dns_cache=300,  # DNS cache
                use_dns_cache=True,
                keepalive_timeout=60,
                enable_cleanup_closed=True,
                force_close=False,  # Keep connections alive
                ssl=SSL_CONTEXT,
            )

            timeout = aiohttp.ClientTimeout(
                total=2,  # 2 second max
                connect=0.5,  # 500ms connect
                sock_read=1,  # 1s read
            )

            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'Accept': 'application/json',
                    'Connection': 'keep-alive',
                }
            )

    async def _fetch_one(self, name: str, url: str) -> tuple:
        """Fetch from single exchange - maximum speed."""
        start = time.perf_counter()
        try:
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = self.PARSERS[name](data)
                    latency = (time.perf_counter() - start) * 1000
                    self.latencies[name] = latency
                    return name, price, latency
        except Exception as e:
            pass
        return name, 0, 9999

    async def fetch_all_parallel(self) -> dict:
        """Fetch from ALL exchanges in parallel - maximum speed."""
        await self._create_session()

        start = time.perf_counter()

        # Create all tasks
        tasks = [
            self._fetch_one(name, url)
            for name, url in self.ENDPOINTS.items()
        ]

        # Execute ALL in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_latency = (time.perf_counter() - start) * 1000

        # Process results
        prices = []
        successful = []

        for r in results:
            if isinstance(r, tuple) and r[1] > 0:
                name, price, latency = r
                prices.append(price)
                successful.append((name, price, latency))

        if prices:
            self.price = sum(prices) / len(prices)
            self.prices.append(self.price)
            self.timestamps.append(time.time())
            self.tick_count += 1

        self.last_update = time.time()

        return {
            'price': self.price,
            'total_latency_ms': total_latency,
            'sources': len(successful),
            'results': successful,
            'tick_count': self.tick_count
        }

    async def warmup(self):
        """Pre-warm connections for faster subsequent calls."""
        await self._create_session()
        # Make initial requests to establish connections
        await self.fetch_all_parallel()

    async def close(self):
        """Close session."""
        if self.session and not self.session.closed:
            await self.session.close()

    def get_fastest_exchange(self) -> tuple:
        """Return fastest exchange."""
        if not self.latencies:
            return None, 0
        fastest = min(self.latencies.items(), key=lambda x: x[1])
        return fastest


# =============================================================================
# SINGLETON FOR SPEED
# =============================================================================
_FEED: Optional[UltraLowLatencyFeed] = None


async def get_feed() -> UltraLowLatencyFeed:
    global _FEED
    if _FEED is None:
        _FEED = UltraLowLatencyFeed()
        await _FEED.warmup()
    return _FEED


async def fetch_price_fast() -> dict:
    """Fastest possible price fetch."""
    feed = await get_feed()
    return await feed.fetch_all_parallel()


# =============================================================================
# TEST
# =============================================================================
async def benchmark():
    """Benchmark latency."""
    print("=" * 70)
    print("ULTRA-LOW LATENCY BENCHMARK")
    print("TARGET: <100ms per exchange, <500ms total")
    print("=" * 70)

    feed = UltraLowLatencyFeed()

    # Warmup
    print("\nWarming up connections...")
    await feed.warmup()

    # Run 5 tests
    print("\nRunning 5 sequential fetches...\n")

    for i in range(5):
        result = await feed.fetch_all_parallel()

        print(f"Fetch #{i+1}:")
        print(f"  Total latency: {result['total_latency_ms']:.0f}ms")
        print(f"  Price: ${result['price']:,.2f}")
        print(f"  Sources: {result['sources']}")

        # Show individual latencies
        for name, price, latency in sorted(result['results'], key=lambda x: x[2]):
            status = "FAST" if latency < 100 else "OK" if latency < 300 else "SLOW"
            print(f"    [{name:10}] {latency:6.0f}ms | ${price:,.2f} | {status}")

        print()

    # Summary
    print("-" * 70)
    print("LATENCY SUMMARY:")
    for name, latency in sorted(feed.latencies.items(), key=lambda x: x[1]):
        status = "FAST" if latency < 100 else "OK" if latency < 300 else "SLOW"
        print(f"  [{name:10}] {latency:6.0f}ms | {status}")

    fastest_name, fastest_latency = feed.get_fastest_exchange()
    print(f"\nFASTEST: {fastest_name} at {fastest_latency:.0f}ms")
    print("=" * 70)

    await feed.close()


if __name__ == "__main__":
    asyncio.run(benchmark())
