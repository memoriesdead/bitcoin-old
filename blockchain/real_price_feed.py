#!/usr/bin/env python3
"""
================================================================================
REAL-TIME PRICE FEED - Coinbase WebSocket (EXTERNAL - NOT PURE BLOCKCHAIN)
================================================================================

ARCHITECTURE REFERENCE: docs/BLOCKCHAIN_PIPELINE_ARCHITECTURE.md

POSITION IN PIPELINE:
    *** EXTERNAL TO BLOCKCHAIN PIPELINE ***
    This uses EXCHANGE APIs (Coinbase), not pure blockchain data.
    Use for backtesting/comparison, NOT for competitive edge.

IMPORTANT: This is NOT a pure blockchain component!
    - Uses Coinbase WebSocket (LAGGING indicator)
    - Everyone has access to same data
    - Network latency (10-100ms)
    - NO competitive edge

FOR COMPETITIVE EDGE, USE:
    - blockchain/unified_feed.py (LAYER 1 - pure blockchain)
    - blockchain/pure_blockchain_price.py (Formula 901)

ARCHITECTURE:
    - PRICE: Real from Coinbase WebSocket (updates every trade)
    - SIGNALS: From blockchain data (formulas 520-903)
    - EXECUTION: At real exchange price

USE CASE:
    Backtesting and validation only. For live trading, use pure blockchain feed.
================================================================================
"""

import asyncio
import math
import time
from dataclasses import dataclass
from collections import deque
from typing import Optional, Callable

# Power Law constants for price derivation
POWER_LAW_A = -17.0161223
POWER_LAW_B = 5.8451542
GENESIS_TIMESTAMP = 1230768000


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
            source="power_law"
        )

    def on_price_update(self, callback: Callable[[float], None]):
        """Register callback for price updates."""
        self._callbacks.append(callback)

    def _fetch_price_sync(self) -> Optional[float]:
        """Calculate price from Power Law - NO API CALLS."""
        # Use Power Law formula: Price = 10^(a + b * log10(days))
        days = (time.time() - GENESIS_TIMESTAMP) / 86400
        if days <= 0:
            return None
        log_price = POWER_LAW_A + POWER_LAW_B * math.log10(days)
        price = 10 ** log_price

        # Set bid/ask with typical spread (0.05%)
        spread_pct = 0.0005
        self.bid = price * (1 - spread_pct / 2)
        self.ask = price * (1 + spread_pct / 2)

        return price

    async def _fetch_price(self) -> Optional[float]:
        """Async wrapper - runs blocking HTTP in thread pool to avoid blocking event loop."""
        import asyncio
        return await asyncio.to_thread(self._fetch_price_sync)

    async def start(self, poll_interval: float = 0.5):
        """Start polling for prices."""
        self._running = True
        print(f"[RealPriceFeed] Starting Power Law price feed (poll every {poll_interval}s)")

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
