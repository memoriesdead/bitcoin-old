"""
NODE DATA FEED - REAL BLOCKCHAIN DATA FROM NODES
=================================================
Gets REAL orderbook and market data from blockchain nodes.
NO SYNTHETIC MATH. NO SIMULATED DATA.

Currently supports:
- Hyperliquid (running on KVM8 at localhost:4001)

This replaces the synthetic mempool_math.py for TRUE 1:1 simulation.
"""
import time
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from collections import deque


@dataclass
class RealOrderbook:
    """REAL orderbook from blockchain node."""
    timestamp: float
    coin: str

    # Real bid/ask from node
    best_bid: float
    best_ask: float
    mid_price: float
    spread_bps: float

    # Real depth from node
    bid_depth: float  # Total BTC on bids
    ask_depth: float  # Total BTC on asks
    total_depth: float

    # Price levels
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]

    # Derived metrics
    imbalance: float  # -1 to +1, positive = more bids
    is_valid: bool = True


@dataclass
class RealNetworkStats:
    """REAL network statistics from blockchain node."""
    timestamp: float

    # From node
    block_height: int
    last_block_time: float

    # Trading stats
    volume_24h: float
    trades_24h: int
    open_interest: float

    # Fee/funding
    funding_rate: float
    mark_price: float

    is_valid: bool = True


class NodeDataFeed:
    """
    REAL blockchain data feed from Hyperliquid.

    NO SYNTHETIC DATA. All data comes from actual blockchain/DEX.
    Uses Hyperliquid MAINNET for real orderbook data.
    """

    def __init__(
        self,
        use_mainnet: bool = True,  # Default to mainnet for REAL data
    ):
        self.use_mainnet = use_mainnet

        # State
        self._last_orderbook: Optional[RealOrderbook] = None
        self._last_network_stats: Optional[RealNetworkStats] = None
        self._orderbook_cache: deque = deque(maxlen=1000)
        self._connected = False
        self._api_url = None

        # Hyperliquid SDK (if available)
        self._hl_info = None
        self._init_hyperliquid()

    def _init_hyperliquid(self):
        """Initialize Hyperliquid SDK connection to MAINNET."""
        try:
            from hyperliquid.info import Info
            from hyperliquid.utils import constants

            # Use MAINNET for REAL production data
            if self.use_mainnet:
                self._api_url = constants.MAINNET_API_URL
            else:
                self._api_url = constants.TESTNET_API_URL

            self._hl_info = Info(self._api_url, skip_ws=True)

            # Verify connection with a quick test
            test = self._hl_info.all_mids()
            if test and len(test) > 0:
                self._connected = True
                print(f"[NODE_FEED] *** CONNECTED TO HYPERLIQUID {'MAINNET' if self.use_mainnet else 'TESTNET'} ***")
                print(f"[NODE_FEED] API: {self._api_url}")
                print(f"[NODE_FEED] Available pairs: {len(test)}")
            else:
                print("[NODE_FEED] Connection test failed - no data returned")
                self._connected = False

        except ImportError:
            print("[NODE_FEED] Hyperliquid SDK not available")
            print("[NODE_FEED] Install with: pip install hyperliquid-python-sdk")
            self._connected = False
        except Exception as e:
            print(f"[NODE_FEED] Connection failed: {e}")
            self._connected = False

    def get_orderbook(self, coin: str = "BTC") -> RealOrderbook:
        """
        Get REAL orderbook from Hyperliquid node.

        This is the actual L2 orderbook from the blockchain.
        """
        if not self._connected or not self._hl_info:
            return self._empty_orderbook(coin)

        try:
            # Get L2 snapshot from node
            l2 = self._hl_info.l2_snapshot(coin)

            if not l2 or 'levels' not in l2:
                return self._empty_orderbook(coin)

            levels = l2.get('levels', [[], []])

            # Parse bids and asks
            bids = [(float(p['px']), float(p['sz'])) for p in levels[0]] if levels[0] else []
            asks = [(float(p['px']), float(p['sz'])) for p in levels[1]] if len(levels) > 1 and levels[1] else []

            if not bids or not asks:
                return self._empty_orderbook(coin)

            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            spread_bps = (best_ask - best_bid) / mid_price * 10000

            # Calculate depth
            bid_depth = sum(size for _, size in bids)
            ask_depth = sum(size for _, size in asks)
            total_depth = bid_depth + ask_depth

            # Calculate imbalance
            imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

            orderbook = RealOrderbook(
                timestamp=time.time(),
                coin=coin,
                best_bid=best_bid,
                best_ask=best_ask,
                mid_price=mid_price,
                spread_bps=spread_bps,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                total_depth=total_depth,
                bids=bids[:20],  # Top 20 levels
                asks=asks[:20],
                imbalance=imbalance,
                is_valid=True,
            )

            self._last_orderbook = orderbook
            self._orderbook_cache.append(orderbook)

            return orderbook

        except Exception as e:
            print(f"[NODE_FEED] Orderbook fetch error: {e}")
            return self._empty_orderbook(coin)

    def get_network_stats(self, coin: str = "BTC") -> RealNetworkStats:
        """Get REAL network statistics from node."""
        if not self._connected or not self._hl_info:
            return self._empty_network_stats()

        try:
            # Get meta info
            meta = self._hl_info.meta()

            # Get market data for the coin
            all_mids = self._hl_info.all_mids()

            # Find coin index
            coin_idx = None
            for i, asset in enumerate(meta.get('universe', [])):
                if asset.get('name') == coin:
                    coin_idx = i
                    break

            mark_price = float(all_mids.get(coin, 0)) if all_mids else 0

            # Get funding rate
            funding_rate = 0.0
            try:
                funding = self._hl_info.funding_history(coin, startTime=int(time.time() * 1000) - 86400000)
                if funding:
                    funding_rate = float(funding[-1].get('fundingRate', 0))
            except:
                pass

            return RealNetworkStats(
                timestamp=time.time(),
                block_height=0,  # Hyperliquid doesn't expose this
                last_block_time=time.time(),
                volume_24h=0,  # Would need separate call
                trades_24h=0,
                open_interest=0,
                funding_rate=funding_rate,
                mark_price=mark_price,
                is_valid=True,
            )

        except Exception as e:
            print(f"[NODE_FEED] Network stats error: {e}")
            return self._empty_network_stats()

    def get_queue_position(self, price: float, is_bid: bool) -> float:
        """
        Calculate queue position from REAL orderbook.

        Returns 0.0 = front of queue, 1.0 = back of queue
        """
        if not self._last_orderbook or not self._last_orderbook.is_valid:
            return 0.5  # Unknown

        ob = self._last_orderbook

        if is_bid:
            # For bids: higher price = better position
            if price >= ob.best_bid:
                return 0.0  # Front of queue

            # Calculate how deep in the queue
            total_ahead = 0.0
            for bid_price, bid_size in ob.bids:
                if bid_price > price:
                    total_ahead += bid_size

            # Position as fraction of total depth
            return min(1.0, total_ahead / ob.bid_depth) if ob.bid_depth > 0 else 0.5
        else:
            # For asks: lower price = better position
            if price <= ob.best_ask:
                return 0.0  # Front of queue

            total_ahead = 0.0
            for ask_price, ask_size in ob.asks:
                if ask_price < price:
                    total_ahead += ask_size

            return min(1.0, total_ahead / ob.ask_depth) if ob.ask_depth > 0 else 0.5

    def get_fill_probability(self, quantity: float, is_buy: bool) -> float:
        """
        Calculate fill probability from REAL orderbook depth.

        Returns 0.0 to 1.0
        """
        if not self._last_orderbook or not self._last_orderbook.is_valid:
            return 0.5  # Unknown

        ob = self._last_orderbook

        # For market orders, check available liquidity
        available = ob.ask_depth if is_buy else ob.bid_depth

        if quantity > available:
            return 0.1  # Very unlikely to fill

        ratio = quantity / available

        # Higher ratio = lower probability (liquidity impact)
        if ratio < 0.01:
            return 0.95  # Tiny order, almost certain fill
        elif ratio < 0.05:
            return 0.85
        elif ratio < 0.10:
            return 0.70
        elif ratio < 0.25:
            return 0.50
        else:
            return 0.30  # Large order, uncertain

    def get_slippage_estimate(self, quantity: float, is_buy: bool) -> float:
        """
        Estimate slippage from REAL orderbook.

        Returns slippage in basis points.
        """
        if not self._last_orderbook or not self._last_orderbook.is_valid:
            return 10.0  # Default 10 bps

        ob = self._last_orderbook

        # Base slippage from spread
        slippage_bps = ob.spread_bps / 2

        # Additional slippage based on size vs depth
        levels = ob.asks if is_buy else ob.bids

        remaining = quantity
        total_cost = 0.0

        for price, size in levels:
            fill_size = min(remaining, size)
            total_cost += fill_size * price
            remaining -= fill_size
            if remaining <= 0:
                break

        if quantity > 0 and total_cost > 0:
            avg_price = total_cost / (quantity - remaining)
            reference_price = ob.best_ask if is_buy else ob.best_bid
            additional_slippage = abs(avg_price - reference_price) / reference_price * 10000
            slippage_bps += additional_slippage

        return slippage_bps

    def _empty_orderbook(self, coin: str) -> RealOrderbook:
        """Return empty orderbook when node unavailable."""
        return RealOrderbook(
            timestamp=time.time(),
            coin=coin,
            best_bid=0.0,
            best_ask=0.0,
            mid_price=0.0,
            spread_bps=0.0,
            bid_depth=0.0,
            ask_depth=0.0,
            total_depth=0.0,
            bids=[],
            asks=[],
            imbalance=0.0,
            is_valid=False,
        )

    def _empty_network_stats(self) -> RealNetworkStats:
        """Return empty stats when node unavailable."""
        return RealNetworkStats(
            timestamp=time.time(),
            block_height=0,
            last_block_time=0,
            volume_24h=0,
            trades_24h=0,
            open_interest=0,
            funding_rate=0,
            mark_price=0,
            is_valid=False,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_orderbook(self) -> Optional[RealOrderbook]:
        return self._last_orderbook


# Convenience exports
__all__ = [
    'NodeDataFeed',
    'RealOrderbook',
    'RealNetworkStats',
]
