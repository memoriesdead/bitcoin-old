#!/usr/bin/env python3
"""
PRICE FEED
==========
Simple price fetching from exchanges.
"""

import time
import threading
import requests
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class Price:
    exchange: str
    price: float
    bid: float
    ask: float
    timestamp: float


class PriceFeed:
    """Multi-exchange price feed."""

    def __init__(self):
        self.prices: Dict[str, Price] = {}
        self.lock = threading.Lock()
        self.running = False

    def fetch(self, exchange: str) -> Optional[float]:
        """Fetch price from exchange."""
        fetchers = {
            'coinbase': self._coinbase,
            'kraken': self._kraken,
            'bitstamp': self._bitstamp,
            'gemini': self._gemini,
            'binance': self._binance,
        }

        if exchange not in fetchers:
            return None

        try:
            price = fetchers[exchange]()
            if price:
                with self.lock:
                    self.prices[exchange] = price
                return price.price
        except Exception:
            pass
        return None

    def get(self, exchange: str) -> Optional[float]:
        """Get cached price."""
        with self.lock:
            if exchange in self.prices:
                return self.prices[exchange].price
        return self.fetch(exchange)

    def _coinbase(self) -> Optional[Price]:
        r = requests.get('https://api.coinbase.com/v2/prices/BTC-USD/spot', timeout=5)
        if r.ok:
            price = float(r.json()['data']['amount'])
            return Price('coinbase', price, price, price, time.time())
        return None

    def _kraken(self) -> Optional[Price]:
        r = requests.get('https://api.kraken.com/0/public/Ticker?pair=XBTUSD', timeout=5)
        if r.ok:
            data = r.json()['result']['XXBTZUSD']
            bid, ask = float(data['b'][0]), float(data['a'][0])
            return Price('kraken', (bid+ask)/2, bid, ask, time.time())
        return None

    def _bitstamp(self) -> Optional[Price]:
        r = requests.get('https://www.bitstamp.net/api/v2/ticker/btcusd/', timeout=5)
        if r.ok:
            data = r.json()
            bid, ask = float(data['bid']), float(data['ask'])
            return Price('bitstamp', (bid+ask)/2, bid, ask, time.time())
        return None

    def _gemini(self) -> Optional[Price]:
        r = requests.get('https://api.gemini.com/v1/pubticker/btcusd', timeout=5)
        if r.ok:
            data = r.json()
            bid, ask = float(data['bid']), float(data['ask'])
            return Price('gemini', (bid+ask)/2, bid, ask, time.time())
        return None

    def _binance(self) -> Optional[Price]:
        r = requests.get('https://api.binance.us/api/v3/ticker/bookTicker?symbol=BTCUSD', timeout=5)
        if r.ok:
            data = r.json()
            bid, ask = float(data['bidPrice']), float(data['askPrice'])
            return Price('binance', (bid+ask)/2, bid, ask, time.time())
        return None

    def start(self):
        """Start background updates."""
        self.running = True
        def loop():
            while self.running:
                for ex in ['coinbase', 'kraken', 'bitstamp', 'gemini']:
                    self.fetch(ex)
                time.sleep(1)
        threading.Thread(target=loop, daemon=True).start()

    def stop(self):
        self.running = False
