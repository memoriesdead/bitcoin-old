"""
Exchange price verifier for cross-reference.

Fetches prices from multiple exchanges to:
- Verify signal price accuracy
- Calculate slippage estimation
- Confirm prediction accuracy
"""

import time
import requests
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class PriceSnapshot:
    """Multi-exchange price snapshot."""
    timestamp: float
    coinbase: Optional[float] = None
    kraken: Optional[float] = None
    bitstamp: Optional[float] = None
    binance: Optional[float] = None

    @property
    def avg_price(self) -> float:
        """Get average price across exchanges."""
        prices = [p for p in [self.coinbase, self.kraken, self.bitstamp, self.binance] if p]
        return sum(prices) / len(prices) if prices else 0.0

    @property
    def spread_pct(self) -> float:
        """Get spread percentage across exchanges."""
        prices = [p for p in [self.coinbase, self.kraken, self.bitstamp, self.binance] if p]
        if len(prices) < 2:
            return 0.0
        return ((max(prices) - min(prices)) / min(prices)) * 100

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'coinbase': self.coinbase,
            'kraken': self.kraken,
            'bitstamp': self.bitstamp,
            'binance': self.binance,
            'avg_price': self.avg_price,
            'spread_pct': self.spread_pct,
        }


class ExchangeVerifier:
    """
    Cross-reference trade prices with actual exchange prices.

    Features:
    - Multi-exchange price fetching (Coinbase, Kraken, Bitstamp, Binance)
    - Slippage estimation
    - Prediction accuracy verification
    - Price caching (1 second TTL)
    """

    ENDPOINTS = {
        'coinbase': 'https://api.exchange.coinbase.com/products/BTC-USD/ticker',
        'kraken': 'https://api.kraken.com/0/public/Ticker?pair=XBTUSD',
        'bitstamp': 'https://www.bitstamp.net/api/v2/ticker/btcusd/',
        'binance': 'https://api.binance.us/api/v3/ticker/price?symbol=BTCUSD',
    }

    def __init__(self, cache_ttl: float = 1.0):
        self.cache_ttl = cache_ttl
        self._cache: Optional[PriceSnapshot] = None
        self._cache_time: float = 0

    def get_prices(self, force_refresh: bool = False) -> PriceSnapshot:
        """
        Fetch current prices from all exchanges.

        Args:
            force_refresh: Bypass cache

        Returns:
            PriceSnapshot with prices from each exchange
        """
        now = time.time()

        # Return cached if fresh
        if not force_refresh and self._cache and (now - self._cache_time) < self.cache_ttl:
            return self._cache

        snapshot = PriceSnapshot(timestamp=now)

        # Coinbase
        try:
            resp = requests.get(self.ENDPOINTS['coinbase'], timeout=5)
            if resp.status_code == 200:
                snapshot.coinbase = float(resp.json()['price'])
        except Exception:
            pass

        # Kraken
        try:
            resp = requests.get(self.ENDPOINTS['kraken'], timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if 'result' in data and 'XXBTZUSD' in data['result']:
                    snapshot.kraken = float(data['result']['XXBTZUSD']['c'][0])
        except Exception:
            pass

        # Bitstamp
        try:
            resp = requests.get(self.ENDPOINTS['bitstamp'], timeout=5)
            if resp.status_code == 200:
                snapshot.bitstamp = float(resp.json()['last'])
        except Exception:
            pass

        # Binance US
        try:
            resp = requests.get(self.ENDPOINTS['binance'], timeout=5)
            if resp.status_code == 200:
                snapshot.binance = float(resp.json()['price'])
        except Exception:
            pass

        # Cache result
        self._cache = snapshot
        self._cache_time = now

        return snapshot

    def get_current_price(self) -> float:
        """Get average price across exchanges."""
        snapshot = self.get_prices()
        return snapshot.avg_price

    def verify_price(
        self,
        signal_price: float,
        entry_price: float,
        direction: int
    ) -> Dict:
        """
        Verify trade price against exchange prices.

        Args:
            signal_price: Price when signal fired
            entry_price: Actual entry price
            direction: 1 for LONG, -1 for SHORT

        Returns:
            Dict with verification results
        """
        snapshot = self.get_prices()
        exchange_price = snapshot.avg_price

        # Calculate slippage
        slippage_pct = ((entry_price - signal_price) / signal_price) * 100

        # For LONG: positive slippage is bad (paid more)
        # For SHORT: negative slippage is bad (received less)
        slippage_cost = slippage_pct if direction == 1 else -slippage_pct

        return {
            'exchange_price': exchange_price,
            'signal_price': signal_price,
            'entry_price': entry_price,
            'slippage_pct': slippage_pct,
            'slippage_cost_pct': slippage_cost,
            'exchange_spread_pct': snapshot.spread_pct,
            'prices': snapshot.to_dict(),
        }

    def verify_prediction(
        self,
        direction: int,
        entry_price: float,
        exit_price: float = None
    ) -> bool:
        """
        Check if price moved in predicted direction.

        Args:
            direction: 1 for LONG, -1 for SHORT
            entry_price: Entry price
            exit_price: Exit price (or current price if None)

        Returns:
            True if prediction was correct
        """
        if exit_price is None:
            exit_price = self.get_current_price()

        if direction == 1:  # LONG
            return exit_price > entry_price
        else:  # SHORT
            return exit_price < entry_price

    def calculate_slippage(
        self,
        signal_price: float,
        entry_price: float
    ) -> float:
        """
        Calculate slippage percentage.

        Returns:
            Slippage as percentage (positive = paid more, negative = paid less)
        """
        return ((entry_price - signal_price) / signal_price) * 100

    def estimate_execution_price(
        self,
        signal_price: float,
        direction: int,
        size_btc: float = 1.0
    ) -> float:
        """
        Estimate execution price including typical slippage.

        For simulation purposes, applies realistic slippage based on:
        - Direction (LONG typically has higher slippage)
        - Size (larger orders = more slippage)

        Args:
            signal_price: Price when signal fired
            direction: 1 for LONG, -1 for SHORT
            size_btc: Position size in BTC

        Returns:
            Estimated execution price
        """
        # Base slippage: 0.01% for small orders, scales with size
        base_slippage = 0.0001  # 0.01%
        size_factor = 1 + (size_btc / 100)  # Increases with size
        slippage_pct = base_slippage * size_factor

        if direction == 1:  # LONG - pay slightly more
            return signal_price * (1 + slippage_pct)
        else:  # SHORT - receive slightly less
            return signal_price * (1 - slippage_pct)


class HistoricalVerifier:
    """
    Verifier for historical simulation.

    Uses historical price data instead of live exchange feeds.
    """

    def __init__(self):
        self.current_price: float = 0.0
        self.price_history: List[float] = []

    def set_price(self, price: float):
        """Set current price for historical simulation."""
        self.current_price = price
        self.price_history.append(price)
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]

    def get_current_price(self) -> float:
        """Get current historical price."""
        return self.current_price

    def get_prices(self) -> PriceSnapshot:
        """Get price snapshot (all exchanges = historical price)."""
        return PriceSnapshot(
            timestamp=time.time(),
            coinbase=self.current_price,
            kraken=self.current_price,
            bitstamp=self.current_price,
            binance=self.current_price,
        )

    def verify_prediction(
        self,
        direction: int,
        entry_price: float,
        exit_price: float
    ) -> bool:
        """Check if prediction was correct."""
        if direction == 1:  # LONG
            return exit_price > entry_price
        else:  # SHORT
            return exit_price < entry_price

    def estimate_execution_price(
        self,
        signal_price: float,
        direction: int,
        size_btc: float = 1.0
    ) -> float:
        """Estimate execution price with realistic slippage."""
        # Apply small slippage for realism
        base_slippage = 0.0001  # 0.01%
        size_factor = 1 + (size_btc / 100)
        slippage_pct = base_slippage * size_factor

        if direction == 1:
            return signal_price * (1 + slippage_pct)
        else:
            return signal_price * (1 - slippage_pct)
