"""
SOVEREIGN DATA FEED - BLOCKCHAIN AS DATA SOURCE
================================================
The blockchain nodes provide DATA for price discovery.
All EXECUTION happens internally at unlimited speed.

This module connects:
- Hyperliquid MAINNET (real orderbook)
- Sei (if available)
- Solana (if available)
- Monad (if available)

Data flows: Blockchain -> Internal Orderbook -> Matching Engine
Execution flows: Matching Engine -> Settlement Layer -> Blockchain (only when needed)
"""
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque

from engine.sovereign.matching_engine import SovereignMatchingEngine, OrderSide


@dataclass
class DataFeedStats:
    """Statistics for data feed operations."""
    updates: int = 0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    last_update: float = 0.0
    data_source: str = ""


class SovereignDataFeed:
    """
    Data feed layer for Sovereign Matching Engine.

    Pulls real orderbook data from blockchain nodes.
    Feeds it into the internal matching engine.

    IMPORTANT: This is DATA ONLY.
    No execution happens on the blockchain.
    """

    def __init__(
        self,
        matching_engine: SovereignMatchingEngine,
        update_interval_ms: float = 100.0,  # How often to pull data
    ):
        self.engine = matching_engine
        self.update_interval_ms = update_interval_ms

        # Node connections
        self._hyperliquid = None
        self._connected = False

        # Stats per asset
        self.stats: Dict[str, DataFeedStats] = {}

        # Data history
        self.price_history: deque = deque(maxlen=100000)

        # Initialize connections
        self._init_connections()

    def _init_connections(self):
        """Initialize blockchain node connections for DATA."""
        try:
            # Hyperliquid - Primary data source
            from blockchain.node_data_feed import NodeDataFeed
            self._hyperliquid = NodeDataFeed(use_mainnet=True)

            if self._hyperliquid.is_connected:
                self._connected = True
                print("[SOVEREIGN_DATA] Connected to Hyperliquid MAINNET for DATA")
            else:
                # Try direct SDK
                from hyperliquid.info import Info
                from hyperliquid.utils import constants
                self._hyperliquid_sdk = Info(constants.MAINNET_API_URL, skip_ws=True)
                self._connected = True
                print("[SOVEREIGN_DATA] Connected to Hyperliquid SDK for DATA")

        except ImportError as e:
            print(f"[SOVEREIGN_DATA] Import error: {e}")
            self._connected = False
        except Exception as e:
            print(f"[SOVEREIGN_DATA] Connection error: {e}")
            self._connected = False

    def update_orderbook(self, asset: str = "BTC") -> bool:
        """
        Pull latest orderbook from blockchain and update internal state.

        Returns True if update was successful.
        """
        if not self._connected:
            return False

        start = time.time()

        try:
            # Get data from Hyperliquid
            if hasattr(self, '_hyperliquid') and self._hyperliquid:
                ob = self._hyperliquid.get_orderbook(asset)

                if ob.is_valid:
                    # Update internal orderbook
                    self.engine.update_orderbook(
                        asset=asset,
                        bids=ob.bids,
                        asks=ob.asks,
                    )

                    # Record stats
                    latency_ms = (time.time() - start) * 1000
                    self._update_stats(asset, latency_ms, "hyperliquid")

                    # Record price
                    self.price_history.append({
                        'timestamp': time.time(),
                        'asset': asset,
                        'bid': ob.best_bid,
                        'ask': ob.best_ask,
                        'mid': ob.mid_price,
                        'spread_bps': ob.spread_bps,
                    })

                    return True

            # Fallback to SDK
            if hasattr(self, '_hyperliquid_sdk'):
                l2 = self._hyperliquid_sdk.l2_snapshot(asset)

                if l2 and 'levels' in l2:
                    levels = l2.get('levels', [[], []])
                    bids = [(float(p['px']), float(p['sz'])) for p in levels[0]] if levels[0] else []
                    asks = [(float(p['px']), float(p['sz'])) for p in levels[1]] if len(levels) > 1 and levels[1] else []

                    if bids and asks:
                        self.engine.update_orderbook(
                            asset=asset,
                            bids=bids[:20],
                            asks=asks[:20],
                        )

                        latency_ms = (time.time() - start) * 1000
                        self._update_stats(asset, latency_ms, "hyperliquid_sdk")
                        return True

            return False

        except Exception as e:
            print(f"[SOVEREIGN_DATA] Update error for {asset}: {e}")
            return False

    def _update_stats(self, asset: str, latency_ms: float, source: str):
        """Update feed statistics."""
        if asset not in self.stats:
            self.stats[asset] = DataFeedStats(data_source=source)

        stats = self.stats[asset]
        stats.updates += 1
        stats.last_update = time.time()
        stats.data_source = source

        # Update latency stats
        n = stats.updates
        stats.avg_latency_ms = (stats.avg_latency_ms * (n - 1) + latency_ms) / n
        stats.min_latency_ms = min(stats.min_latency_ms, latency_ms)
        stats.max_latency_ms = max(stats.max_latency_ms, latency_ms)

    def get_current_price(self, asset: str = "BTC") -> Tuple[float, float, float]:
        """
        Get current bid/ask/mid from internal orderbook.

        Returns: (bid, ask, mid)
        """
        ob = self.engine.get_orderbook(asset)
        return ob.best_bid, ob.best_ask, ob.mid_price

    def get_spread_bps(self, asset: str = "BTC") -> float:
        """Get current spread in basis points."""
        ob = self.engine.get_orderbook(asset)
        return ob.spread_bps

    def get_imbalance(self, asset: str = "BTC") -> float:
        """Get order flow imbalance: -1 (sell) to +1 (buy)."""
        ob = self.engine.get_orderbook(asset)
        return ob.get_imbalance()

    async def run_feed_loop(
        self,
        assets: List[str] = None,
        duration_seconds: float = None,
    ):
        """
        Run continuous data feed loop.

        Updates orderbooks at specified interval.
        """
        if assets is None:
            assets = ["BTC"]

        start_time = time.time()

        print(f"[SOVEREIGN_DATA] Starting feed loop for {assets}")

        while True:
            # Update all assets
            for asset in assets:
                self.update_orderbook(asset)

            # Check duration
            if duration_seconds:
                elapsed = time.time() - start_time
                if elapsed >= duration_seconds:
                    break

            # Wait for next update
            await asyncio.sleep(self.update_interval_ms / 1000)

    def print_stats(self):
        """Print data feed statistics."""
        print("\n" + "=" * 60)
        print("SOVEREIGN DATA FEED - STATISTICS")
        print("=" * 60)

        for asset, stats in self.stats.items():
            print(f"\n{asset}:")
            print(f"  Updates: {stats.updates:,}")
            print(f"  Avg Latency: {stats.avg_latency_ms:.2f} ms")
            print(f"  Min Latency: {stats.min_latency_ms:.2f} ms")
            print(f"  Max Latency: {stats.max_latency_ms:.2f} ms")
            print(f"  Data Source: {stats.data_source}")

        print("=" * 60)

    @property
    def is_connected(self) -> bool:
        return self._connected


class SignalGenerator:
    """
    Generate trading signals from data feed.

    Uses orderbook imbalance, price momentum, and volatility.
    """

    def __init__(
        self,
        data_feed: SovereignDataFeed,
        lookback: int = 100,
    ):
        self.feed = data_feed
        self.lookback = lookback

        # Signal history
        self.signals: deque = deque(maxlen=10000)

    def generate_signal(self, asset: str = "BTC") -> float:
        """
        Generate trading signal.

        Returns: -1 to +1 (negative = sell, positive = buy)
        """
        # Get orderbook data
        ob = self.feed.engine.get_orderbook(asset)

        if not ob.bids or not ob.asks:
            return 0.0

        # Imbalance signal: more bids = bullish
        imbalance_signal = ob.get_imbalance() * 0.5

        # Price momentum from history
        momentum_signal = self._calculate_momentum(asset)

        # Spread signal: tight spread = more confident
        spread_mult = 1.0 - min(1.0, ob.spread_bps / 100)  # Max 100 bps

        # Combined signal
        raw_signal = (imbalance_signal + momentum_signal) * spread_mult

        # Bound to [-1, 1]
        signal = max(-1.0, min(1.0, raw_signal))

        self.signals.append({
            'timestamp': time.time(),
            'asset': asset,
            'signal': signal,
            'imbalance': imbalance_signal,
            'momentum': momentum_signal,
        })

        return signal

    def _calculate_momentum(self, asset: str) -> float:
        """Calculate price momentum from history."""
        history = self.feed.price_history

        if len(history) < 2:
            return 0.0

        # Get recent prices for this asset
        recent = [p['mid'] for p in history if p['asset'] == asset][-self.lookback:]

        if len(recent) < 2:
            return 0.0

        # Simple momentum: price change as fraction
        first_price = recent[0]
        last_price = recent[-1]

        if first_price == 0:
            return 0.0

        momentum = (last_price - first_price) / first_price * 100  # Convert to percentage

        # Bound to [-1, 1]
        return max(-1.0, min(1.0, momentum))


__all__ = [
    'SovereignDataFeed',
    'SignalGenerator',
    'DataFeedStats',
]
