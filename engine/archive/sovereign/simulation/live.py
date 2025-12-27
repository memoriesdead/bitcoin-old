"""
Live paper trading simulation.

Connects to live exchange feeds for real-time paper trading.
No real money - simulation only.
"""

import time
import threading
import requests
from typing import Optional, Dict, Callable, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class LiveTick:
    """Live price tick from exchange."""
    timestamp: float
    price: float
    exchange: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume_24h: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'price': self.price,
            'exchange': self.exchange,
            'bid': self.bid,
            'ask': self.ask,
            'volume_24h': self.volume_24h,
        }


class ExchangeFeed:
    """
    Live price feed from exchanges.

    Supports:
    - Coinbase
    - Kraken
    - Bitstamp
    - Binance US
    """

    ENDPOINTS = {
        'coinbase': {
            'url': 'https://api.exchange.coinbase.com/products/BTC-USD/ticker',
            'parser': lambda r: {
                'price': float(r['price']),
                'bid': float(r['bid']),
                'ask': float(r['ask']),
                'volume_24h': float(r['volume']),
            }
        },
        'kraken': {
            'url': 'https://api.kraken.com/0/public/Ticker?pair=XBTUSD',
            'parser': lambda r: {
                'price': float(r['result']['XXBTZUSD']['c'][0]),
                'bid': float(r['result']['XXBTZUSD']['b'][0]),
                'ask': float(r['result']['XXBTZUSD']['a'][0]),
                'volume_24h': float(r['result']['XXBTZUSD']['v'][1]),
            }
        },
        'bitstamp': {
            'url': 'https://www.bitstamp.net/api/v2/ticker/btcusd/',
            'parser': lambda r: {
                'price': float(r['last']),
                'bid': float(r['bid']),
                'ask': float(r['ask']),
                'volume_24h': float(r['volume']),
            }
        },
        'binance': {
            'url': 'https://api.binance.us/api/v3/ticker/24hr?symbol=BTCUSD',
            'parser': lambda r: {
                'price': float(r['lastPrice']),
                'bid': float(r['bidPrice']),
                'ask': float(r['askPrice']),
                'volume_24h': float(r['volume']),
            }
        },
    }

    def __init__(self, primary_exchange: str = 'coinbase'):
        self.primary_exchange = primary_exchange
        self._cache: Dict[str, LiveTick] = {}
        self._cache_time: Dict[str, float] = {}
        self.cache_ttl = 1.0  # 1 second cache

    def get_price(self, exchange: str = None) -> Optional[LiveTick]:
        """
        Get current price from exchange.

        Args:
            exchange: Exchange name (default: primary)

        Returns:
            LiveTick or None if failed
        """
        exchange = exchange or self.primary_exchange
        now = time.time()

        # Check cache
        if exchange in self._cache:
            if (now - self._cache_time.get(exchange, 0)) < self.cache_ttl:
                return self._cache[exchange]

        # Fetch from exchange
        config = self.ENDPOINTS.get(exchange)
        if not config:
            return None

        try:
            resp = requests.get(config['url'], timeout=5)
            if resp.status_code != 200:
                return None

            data = config['parser'](resp.json())

            tick = LiveTick(
                timestamp=now,
                price=data['price'],
                exchange=exchange,
                bid=data.get('bid'),
                ask=data.get('ask'),
                volume_24h=data.get('volume_24h'),
            )

            # Cache result
            self._cache[exchange] = tick
            self._cache_time[exchange] = now

            return tick

        except Exception as e:
            print(f"[FEED] Error fetching {exchange}: {e}")
            return None

    def get_all_prices(self) -> Dict[str, LiveTick]:
        """Get prices from all exchanges."""
        prices = {}
        for exchange in self.ENDPOINTS.keys():
            tick = self.get_price(exchange)
            if tick:
                prices[exchange] = tick
        return prices

    def get_avg_price(self) -> float:
        """Get average price across all exchanges."""
        prices = self.get_all_prices()
        if not prices:
            return 0.0
        return sum(t.price for t in prices.values()) / len(prices)


class LivePaperTrader:
    """
    Live paper trading engine.

    Features:
    - Real-time price polling
    - Multiple exchange support
    - Callback-based tick processing
    - Graceful shutdown
    """

    def __init__(
        self,
        poll_interval: float = 1.0,
        primary_exchange: str = 'coinbase'
    ):
        self.poll_interval = poll_interval
        self.feed = ExchangeFeed(primary_exchange)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[LiveTick], None]] = []

        self.tick_count = 0
        self.start_time: Optional[float] = None
        self.last_tick: Optional[LiveTick] = None

    def add_callback(self, callback: Callable[[LiveTick], None]):
        """Add callback for price updates."""
        self._callbacks.append(callback)

    def start(self, duration_seconds: float = None):
        """
        Start live paper trading.

        Args:
            duration_seconds: Run duration (None = indefinite)
        """
        if self._running:
            print("[LIVE] Already running")
            return

        self._running = True
        self.start_time = time.time()
        self.tick_count = 0

        print(f"[LIVE] Starting paper trading")
        print(f"[LIVE] Poll interval: {self.poll_interval}s")
        print(f"[LIVE] Duration: {'indefinite' if duration_seconds is None else f'{duration_seconds}s'}")

        self._thread = threading.Thread(
            target=self._run_loop,
            args=(duration_seconds,),
            daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stop live paper trading."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print(f"[LIVE] Stopped after {self.tick_count} ticks")

    def wait(self):
        """Wait for trading to complete."""
        if self._thread:
            self._thread.join()

    def _run_loop(self, duration_seconds: float = None):
        """Main polling loop."""
        end_time = time.time() + duration_seconds if duration_seconds else None

        while self._running:
            # Check duration
            if end_time and time.time() >= end_time:
                print("[LIVE] Duration complete")
                self._running = False
                break

            # Get price
            tick = self.feed.get_price()

            if tick:
                self.last_tick = tick
                self.tick_count += 1

                # Call all callbacks
                for callback in self._callbacks:
                    try:
                        callback(tick)
                    except Exception as e:
                        print(f"[LIVE] Callback error: {e}")

            # Wait for next poll
            time.sleep(self.poll_interval)

    def run_blocking(
        self,
        callback: Callable[[LiveTick], None],
        duration_seconds: float = None,
        progress_interval: int = 60
    ) -> Dict:
        """
        Run paper trading in blocking mode.

        Args:
            callback: Function called for each tick
            duration_seconds: Run duration (None = Ctrl+C to stop)
            progress_interval: Print progress every N seconds

        Returns:
            Dict with session statistics
        """
        self.add_callback(callback)

        print("=" * 60)
        print("LIVE PAPER TRADING")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {'Indefinite (Ctrl+C to stop)' if duration_seconds is None else f'{duration_seconds}s'}")
        print("=" * 60)

        start_time = time.time()
        first_price = None
        last_progress = start_time

        try:
            self.start(duration_seconds)

            while self._running:
                time.sleep(1)

                # Track first price
                if first_price is None and self.last_tick:
                    first_price = self.last_tick.price

                # Progress update
                now = time.time()
                if progress_interval > 0 and (now - last_progress) >= progress_interval:
                    elapsed = now - start_time
                    ticks = self.tick_count
                    if self.last_tick:
                        print(f"[LIVE] {elapsed:.0f}s elapsed | {ticks} ticks | "
                              f"Price: ${self.last_tick.price:,.2f}")
                    last_progress = now

        except KeyboardInterrupt:
            print("\n[LIVE] Interrupted by user")
            self.stop()

        elapsed = time.time() - start_time
        last_price = self.last_tick.price if self.last_tick else 0

        stats = {
            'total_ticks': self.tick_count,
            'elapsed_seconds': elapsed,
            'ticks_per_second': self.tick_count / elapsed if elapsed > 0 else 0,
            'first_price': first_price,
            'last_price': last_price,
            'price_change_pct': ((last_price / first_price) - 1) * 100 if first_price else 0,
        }

        print("=" * 60)
        print("SESSION COMPLETE")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Ticks: {self.tick_count}")
        if first_price and last_price:
            change = ((last_price / first_price) - 1) * 100
            print(f"Price: ${first_price:,.2f} -> ${last_price:,.2f} ({change:+.2f}%)")
        print("=" * 60)

        return stats


class MultiExchangeFeed:
    """
    Aggregated feed from multiple exchanges.

    Provides consensus price and arbitrage detection.
    """

    def __init__(self):
        self.feed = ExchangeFeed()
        self._last_prices: Dict[str, float] = {}

    def get_consensus_price(self) -> Dict:
        """
        Get consensus price across exchanges.

        Returns:
            Dict with avg price, spread, and per-exchange prices
        """
        prices = self.feed.get_all_prices()

        if not prices:
            return {
                'avg_price': 0,
                'spread_pct': 0,
                'prices': {},
            }

        price_values = [t.price for t in prices.values()]
        avg = sum(price_values) / len(price_values)
        spread_pct = ((max(price_values) - min(price_values)) / min(price_values)) * 100

        return {
            'timestamp': time.time(),
            'avg_price': avg,
            'spread_pct': spread_pct,
            'min_price': min(price_values),
            'max_price': max(price_values),
            'prices': {k: v.price for k, v in prices.items()},
        }

    def detect_arbitrage(self, threshold_pct: float = 0.1) -> Optional[Dict]:
        """
        Detect arbitrage opportunities.

        Args:
            threshold_pct: Minimum spread to consider arbitrage

        Returns:
            Dict with arbitrage opportunity or None
        """
        consensus = self.get_consensus_price()

        if consensus['spread_pct'] >= threshold_pct:
            prices = consensus['prices']
            buy_exchange = min(prices, key=prices.get)
            sell_exchange = max(prices, key=prices.get)

            return {
                'buy_exchange': buy_exchange,
                'buy_price': prices[buy_exchange],
                'sell_exchange': sell_exchange,
                'sell_price': prices[sell_exchange],
                'spread_pct': consensus['spread_pct'],
                'potential_profit_pct': consensus['spread_pct'] - 0.1,  # Minus fees
            }

        return None
