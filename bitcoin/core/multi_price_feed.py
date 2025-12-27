#!/usr/bin/env python3
"""
MULTI-EXCHANGE PRICE FEED
=========================
Phase 3: Track price separately for each exchange.

Each exchange has its own price feed. When we detect flow to/from
an exchange, we track THAT exchange's price for correlation.

Supported exchanges:
- Binance
- Coinbase
- Kraken
- Bitfinex
- OKX
- Bybit
- Huobi/HTX
- Bitstamp
- KuCoin
- Gate.io
- Gemini
- Crypto.com
"""

import time
import threading
import requests
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass


@dataclass
class ExchangePrice:
    """Price data from an exchange."""
    exchange: str
    price: float
    timestamp: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume_24h: Optional[float] = None


class MultiExchangePriceFeed:
    """
    Fetch BTC/USD(T) prices from multiple exchanges.

    Each exchange has its own API endpoint and parsing logic.
    Prices are cached and refreshed periodically.
    """

    def __init__(self, refresh_interval: float = 1.0):
        self.refresh_interval = refresh_interval
        self.prices: Dict[str, ExchangePrice] = {}
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable[[str, float], None]] = []

        # Last fetch times
        self.last_fetch: Dict[str, float] = {}

    def add_callback(self, callback: Callable[[str, float], None]):
        """Add callback for price updates. callback(exchange, price)"""
        self.callbacks.append(callback)

    def _fetch_binance(self) -> Optional[ExchangePrice]:
        """Fetch from Binance - tries Binance.US first, then global, then CoinGecko."""
        # Try Binance.US first (not geo-blocked)
        try:
            r = requests.get(
                'https://api.binance.us/api/v3/ticker/bookTicker?symbol=BTCUSD',
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                bid = float(data['bidPrice'])
                ask = float(data['askPrice'])
                return ExchangePrice(
                    exchange='binance',
                    price=(bid + ask) / 2,
                    timestamp=time.time(),
                    bid=bid,
                    ask=ask
                )
        except Exception:
            pass

        # Fallback to global Binance
        try:
            r = requests.get(
                'https://api.binance.com/api/v3/ticker/bookTicker?symbol=BTCUSDT',
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                bid = float(data['bidPrice'])
                ask = float(data['askPrice'])
                return ExchangePrice(
                    exchange='binance',
                    price=(bid + ask) / 2,
                    timestamp=time.time(),
                    bid=bid,
                    ask=ask
                )
        except Exception:
            pass

        # Final fallback: CoinGecko
        try:
            r = requests.get(
                'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd',
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                price = float(data['bitcoin']['usd'])
                return ExchangePrice(
                    exchange='binance',
                    price=price,
                    timestamp=time.time()
                )
        except Exception:
            pass
        return None

    def _fetch_coinbase(self) -> Optional[ExchangePrice]:
        """Fetch from Coinbase."""
        try:
            r = requests.get(
                'https://api.coinbase.com/v2/prices/BTC-USD/spot',
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                price = float(data['data']['amount'])
                return ExchangePrice(
                    exchange='coinbase',
                    price=price,
                    timestamp=time.time()
                )
        except Exception:
            pass
        return None

    def _fetch_kraken(self) -> Optional[ExchangePrice]:
        """Fetch from Kraken."""
        try:
            r = requests.get(
                'https://api.kraken.com/0/public/Ticker?pair=XBTUSD',
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                ticker = data['result']['XXBTZUSD']
                bid = float(ticker['b'][0])
                ask = float(ticker['a'][0])
                return ExchangePrice(
                    exchange='kraken',
                    price=(bid + ask) / 2,
                    timestamp=time.time(),
                    bid=bid,
                    ask=ask
                )
        except Exception:
            pass
        return None

    def _fetch_bitfinex(self) -> Optional[ExchangePrice]:
        """Fetch from Bitfinex."""
        try:
            r = requests.get(
                'https://api-pub.bitfinex.com/v2/ticker/tBTCUSD',
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                # [BID, BID_SIZE, ASK, ASK_SIZE, DAILY_CHANGE, ...]
                bid = float(data[0])
                ask = float(data[2])
                return ExchangePrice(
                    exchange='bitfinex',
                    price=(bid + ask) / 2,
                    timestamp=time.time(),
                    bid=bid,
                    ask=ask
                )
        except Exception:
            pass
        return None

    def _fetch_okx(self) -> Optional[ExchangePrice]:
        """Fetch from OKX."""
        try:
            r = requests.get(
                'https://www.okx.com/api/v5/market/ticker?instId=BTC-USDT',
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                ticker = data['data'][0]
                bid = float(ticker['bidPx'])
                ask = float(ticker['askPx'])
                return ExchangePrice(
                    exchange='okx',
                    price=(bid + ask) / 2,
                    timestamp=time.time(),
                    bid=bid,
                    ask=ask
                )
        except Exception:
            pass
        return None

    def _fetch_bybit(self) -> Optional[ExchangePrice]:
        """Fetch from Bybit."""
        try:
            r = requests.get(
                'https://api.bybit.com/v5/market/tickers?category=spot&symbol=BTCUSDT',
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                ticker = data['result']['list'][0]
                bid = float(ticker['bid1Price'])
                ask = float(ticker['ask1Price'])
                return ExchangePrice(
                    exchange='bybit',
                    price=(bid + ask) / 2,
                    timestamp=time.time(),
                    bid=bid,
                    ask=ask
                )
        except Exception:
            pass
        return None

    def _fetch_huobi(self) -> Optional[ExchangePrice]:
        """Fetch from Huobi/HTX."""
        try:
            r = requests.get(
                'https://api.huobi.pro/market/detail/merged?symbol=btcusdt',
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                tick = data['tick']
                bid = float(tick['bid'][0])
                ask = float(tick['ask'][0])
                return ExchangePrice(
                    exchange='huobi',
                    price=(bid + ask) / 2,
                    timestamp=time.time(),
                    bid=bid,
                    ask=ask
                )
        except Exception:
            pass
        return None

    def _fetch_bitstamp(self) -> Optional[ExchangePrice]:
        """Fetch from Bitstamp."""
        try:
            r = requests.get(
                'https://www.bitstamp.net/api/v2/ticker/btcusd/',
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                bid = float(data['bid'])
                ask = float(data['ask'])
                return ExchangePrice(
                    exchange='bitstamp',
                    price=(bid + ask) / 2,
                    timestamp=time.time(),
                    bid=bid,
                    ask=ask
                )
        except Exception:
            pass
        return None

    def _fetch_kucoin(self) -> Optional[ExchangePrice]:
        """Fetch from KuCoin."""
        try:
            r = requests.get(
                'https://api.kucoin.com/api/v1/market/orderbook/level1?symbol=BTC-USDT',
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                ticker = data['data']
                bid = float(ticker['bestBid'])
                ask = float(ticker['bestAsk'])
                return ExchangePrice(
                    exchange='kucoin',
                    price=(bid + ask) / 2,
                    timestamp=time.time(),
                    bid=bid,
                    ask=ask
                )
        except Exception:
            pass
        return None

    def _fetch_gateio(self) -> Optional[ExchangePrice]:
        """Fetch from Gate.io."""
        try:
            r = requests.get(
                'https://api.gateio.ws/api/v4/spot/tickers?currency_pair=BTC_USDT',
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                ticker = data[0]
                bid = float(ticker['highest_bid'])
                ask = float(ticker['lowest_ask'])
                return ExchangePrice(
                    exchange='gate.io',
                    price=(bid + ask) / 2,
                    timestamp=time.time(),
                    bid=bid,
                    ask=ask
                )
        except Exception:
            pass
        return None

    def _fetch_gemini(self) -> Optional[ExchangePrice]:
        """Fetch from Gemini."""
        try:
            r = requests.get(
                'https://api.gemini.com/v1/pubticker/btcusd',
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                bid = float(data['bid'])
                ask = float(data['ask'])
                return ExchangePrice(
                    exchange='gemini',
                    price=(bid + ask) / 2,
                    timestamp=time.time(),
                    bid=bid,
                    ask=ask
                )
        except Exception:
            pass
        return None

    def _fetch_cryptocom(self) -> Optional[ExchangePrice]:
        """Fetch from Crypto.com."""
        try:
            r = requests.get(
                'https://api.crypto.com/v2/public/get-ticker?instrument_name=BTC_USDT',
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                ticker = data['result']['data']
                bid = float(ticker['b'])
                ask = float(ticker['k'])
                return ExchangePrice(
                    exchange='crypto.com',
                    price=(bid + ask) / 2,
                    timestamp=time.time(),
                    bid=bid,
                    ask=ask
                )
        except Exception:
            pass
        return None

    # Map exchange names to fetch functions
    FETCHERS = {
        'binance': '_fetch_binance',
        'coinbase': '_fetch_coinbase',
        'kraken': '_fetch_kraken',
        'bitfinex': '_fetch_bitfinex',
        'okx': '_fetch_okx',
        'bybit': '_fetch_bybit',
        'huobi': '_fetch_huobi',
        'bitstamp': '_fetch_bitstamp',
        'kucoin': '_fetch_kucoin',
        'gate.io': '_fetch_gateio',
        'gemini': '_fetch_gemini',
        'crypto.com': '_fetch_cryptocom',
    }

    def fetch_price(self, exchange: str) -> Optional[float]:
        """Fetch current price from a specific exchange."""
        if exchange not in self.FETCHERS:
            return None

        fetcher = getattr(self, self.FETCHERS[exchange])
        result = fetcher()

        if result:
            with self.lock:
                self.prices[exchange] = result
                self.last_fetch[exchange] = time.time()

            # Trigger callbacks
            for cb in self.callbacks:
                try:
                    cb(exchange, result.price)
                except Exception:
                    pass

            return result.price
        return None

    def fetch_all(self) -> Dict[str, float]:
        """Fetch prices from all exchanges."""
        prices = {}
        for exchange in self.FETCHERS:
            price = self.fetch_price(exchange)
            if price:
                prices[exchange] = price
        return prices

    def get_price(self, exchange: str, max_age: float = 5.0) -> Optional[float]:
        """
        Get cached price for an exchange.

        Args:
            exchange: Exchange name
            max_age: Max age in seconds (refetch if older)

        Returns:
            Price or None
        """
        with self.lock:
            if exchange in self.prices:
                age = time.time() - self.prices[exchange].timestamp
                if age <= max_age:
                    return self.prices[exchange].price

        # Refetch if stale
        return self.fetch_price(exchange)

    def get_all_prices(self) -> Dict[str, ExchangePrice]:
        """Get all cached prices."""
        with self.lock:
            return dict(self.prices)

    def _run_loop(self):
        """Background thread to refresh prices."""
        while self.running:
            try:
                self.fetch_all()
            except Exception:
                pass
            time.sleep(self.refresh_interval)

    def start(self):
        """Start background price updates."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print(f"[PRICE] Started multi-exchange price feed ({len(self.FETCHERS)} exchanges)")

    def stop(self):
        """Stop background updates."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)


def get_exchange_price(exchange: str) -> Optional[float]:
    """
    Convenience function to get a single exchange price.

    Usage:
        price = get_exchange_price('binance')
    """
    feed = MultiExchangePriceFeed()
    return feed.fetch_price(exchange)


if __name__ == '__main__':
    print("=" * 70)
    print("MULTI-EXCHANGE PRICE FEED TEST")
    print("=" * 70)

    feed = MultiExchangePriceFeed()
    prices = feed.fetch_all()

    print(f"\n{'Exchange':<15} {'Price':>12} {'Bid':>12} {'Ask':>12}")
    print("-" * 55)

    for exchange, price in sorted(prices.items(), key=lambda x: -x[1]):
        p = feed.prices[exchange]
        bid_str = f"${p.bid:,.2f}" if p.bid else "-"
        ask_str = f"${p.ask:,.2f}" if p.ask else "-"
        print(f"{exchange:<15} ${price:>11,.2f} {bid_str:>12} {ask_str:>12}")

    print()
    print(f"Successfully fetched {len(prices)} exchange prices")
