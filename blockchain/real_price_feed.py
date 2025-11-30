#!/usr/bin/env python3
"""
REAL-TIME PRICE FEED - Coinbase WebSocket
==========================================
REAL prices from exchange, not synthetic.

Architecture:
- PRICE: Real from Coinbase WebSocket (updates every trade)
- SIGNALS: From blockchain data (424 formulas)
- EXECUTION: At real price
"""

import asyncio
import json
import time
from dataclasses import dataclass
from collections import deque
from typing import Optional, Callable
import urllib.request


@dataclass
class RealPrice:
    """Real exchange price"""
    timestamp: float
    price: float
    bid: float
    ask: float
    volume_24h: float
    source: str


class CoinbasePriceFeed:
    """
    Real-time BTC price from Coinbase.

    Uses REST API polling (WebSocket requires auth for some features).
    Updates every 100ms for near real-time prices.
    """

    def __init__(self):
        self.current_price: float = 0.0
        self.bid: float = 0.0
        self.ask: float = 0.0
        self.volume_24h: float = 0.0
        self.last_update: float = 0.0
        self.price_history: deque = deque(maxlen=1000)
        self._running = False
        self._callbacks: list = []

    def get_price(self) -> float:
        """Get current real price."""
        return self.current_price

    def get_real_price(self) -> RealPrice:
        """Get full price data."""
        return RealPrice(
            timestamp=self.last_update,
            price=self.current_price,
            bid=self.bid,
            ask=self.ask,
            volume_24h=self.volume_24h,
            source="coinbase"
        )

    def on_price_update(self, callback: Callable[[float], None]):
        """Register callback for price updates."""
        self._callbacks.append(callback)

    def _fetch_price_sync(self) -> Optional[float]:
        """Synchronous price fetch - runs in thread pool."""
        try:
            # Get spot price
            req = urllib.request.Request(
                'https://api.coinbase.com/v2/prices/BTC-USD/spot',
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                price = float(data['data']['amount'])

            # Get buy/sell prices for spread
            req_buy = urllib.request.Request(
                'https://api.coinbase.com/v2/prices/BTC-USD/buy',
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            req_sell = urllib.request.Request(
                'https://api.coinbase.com/v2/prices/BTC-USD/sell',
                headers={'User-Agent': 'Mozilla/5.0'}
            )

            with urllib.request.urlopen(req_buy, timeout=2) as resp:
                self.ask = float(json.loads(resp.read())['data']['amount'])
            with urllib.request.urlopen(req_sell, timeout=2) as resp:
                self.bid = float(json.loads(resp.read())['data']['amount'])

            return price

        except Exception as e:
            return None

    async def _fetch_price(self) -> Optional[float]:
        """Async wrapper - runs blocking HTTP in thread pool to avoid blocking event loop."""
        import asyncio
        return await asyncio.to_thread(self._fetch_price_sync)

    async def start(self, poll_interval: float = 0.5):
        """Start polling for prices."""
        self._running = True
        print(f"[RealPriceFeed] Starting Coinbase price feed (poll every {poll_interval}s)")

        # Get initial price
        price = await self._fetch_price()
        if price:
            self.current_price = price
            self.last_update = time.time()
            print(f"[RealPriceFeed] Initial BTC price: ${price:,.2f}")

        while self._running:
            try:
                price = await self._fetch_price()
                if price:
                    old_price = self.current_price
                    self.current_price = price
                    self.last_update = time.time()
                    self.price_history.append((self.last_update, price))

                    # Notify callbacks if price changed
                    if abs(price - old_price) > 0.01:
                        for callback in self._callbacks:
                            try:
                                callback(price)
                            except:
                                pass

            except Exception as e:
                pass

            await asyncio.sleep(poll_interval)

    def stop(self):
        """Stop the feed."""
        self._running = False
        print("[RealPriceFeed] Stopped")

    def get_price_change(self, seconds: int = 60) -> float:
        """Get price change over last N seconds."""
        if len(self.price_history) < 2:
            return 0.0

        now = time.time()
        cutoff = now - seconds

        # Find oldest price in window
        oldest_price = self.current_price
        for ts, price in self.price_history:
            if ts >= cutoff:
                oldest_price = price
                break

        return self.current_price - oldest_price


class RealPriceEngine:
    """
    Drop-in replacement for BlockchainPriceEngine.

    Uses REAL Coinbase prices instead of synthetic.
    Blockchain signals still used for trading decisions.
    """

    def __init__(self, calibration_price: float = 0.0):
        self.feed = CoinbasePriceFeed()
        self.calibration_price = calibration_price
        self.current_price = calibration_price
        self._task = None

    async def start(self):
        """Start the real price feed."""
        self._task = asyncio.create_task(self.feed.start(poll_interval=0.5))
        await asyncio.sleep(1)  # Wait for first price

    def get_price(self) -> float:
        """Get current REAL price."""
        price = self.feed.get_price()
        if price > 0:
            self.current_price = price
        return self.current_price

    def update(self, state) -> 'DerivedPrice':
        """
        Update - returns REAL price with blockchain signals.

        Compatible with BlockchainPriceEngine interface.
        """
        from blockchain.blockchain_price_engine import DerivedPrice

        # Get REAL price
        price = self.get_price()

        # Calculate signal from blockchain state (for formula compatibility)
        signal = 0.0
        if hasattr(state, 'fee_fast') and state.fee_fast > 0:
            # Simple signal from fee pressure
            signal = min((state.fee_fast - 1) / 100, 1.0)

        return DerivedPrice(
            timestamp=time.time(),
            composite_price=price,  # REAL PRICE!
            signal=signal,
            confidence=0.8,
            metcalfe_price=price,
            nvt_signal=0,
            fee_pressure_index=signal,
            velocity_price=price,
            price_momentum=0,
            acceleration=0
        )
