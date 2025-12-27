"""
Order Book Types
================

Data structures for order book simulation.
Ported from hftbacktest concepts.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto


class Side(Enum):
    """Order side."""
    BUY = 1
    SELL = -1

    def __neg__(self):
        return Side.SELL if self == Side.BUY else Side.BUY


class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = auto()      # Not yet submitted
    SUBMITTED = auto()    # Sent to exchange
    OPEN = auto()         # Accepted, in order book
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


@dataclass
class Level:
    """
    Single price level in order book.

    hftbacktest concept: Track both price and quantity at each level.
    """
    price: float
    quantity: float
    order_count: int = 1  # Number of orders at this level

    def __post_init__(self):
        if self.price <= 0:
            raise ValueError(f"Price must be positive: {self.price}")
        if self.quantity < 0:
            raise ValueError(f"Quantity cannot be negative: {self.quantity}")


@dataclass
class OrderBookSnapshot:
    """
    Point-in-time order book snapshot.

    hftbacktest concept: Full L2 depth with timestamps.
    """
    timestamp: float  # Unix timestamp (microseconds preferred)
    symbol: str

    # Bid side (highest first)
    bids: List[Level] = field(default_factory=list)

    # Ask side (lowest first)
    asks: List[Level] = field(default_factory=list)

    # Metadata
    exchange: str = "unknown"
    sequence: int = 0  # Exchange sequence number

    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """Get mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """Get spread in basis points."""
        if self.mid_price and self.spread:
            return (self.spread / self.mid_price) * 10000
        return None

    def get_bid_depth(self, levels: int = 10) -> float:
        """Get total bid depth for N levels."""
        return sum(b.quantity for b in self.bids[:levels])

    def get_ask_depth(self, levels: int = 10) -> float:
        """Get total ask depth for N levels."""
        return sum(a.quantity for a in self.asks[:levels])

    def get_imbalance(self, levels: int = 5) -> float:
        """
        Get order book imbalance.

        Returns:
            Positive = more bids (bullish)
            Negative = more asks (bearish)
        """
        bid_depth = self.get_bid_depth(levels)
        ask_depth = self.get_ask_depth(levels)
        total = bid_depth + ask_depth

        if total == 0:
            return 0.0

        return (bid_depth - ask_depth) / total

    def get_vwap_bid(self, quantity: float) -> Optional[float]:
        """
        Get volume-weighted average price to sell quantity.

        Args:
            quantity: Amount to sell

        Returns:
            VWAP price to execute sell order
        """
        if not self.bids:
            return None

        remaining = quantity
        total_value = 0.0
        total_qty = 0.0

        for level in self.bids:
            fill_qty = min(remaining, level.quantity)
            total_value += fill_qty * level.price
            total_qty += fill_qty
            remaining -= fill_qty

            if remaining <= 0:
                break

        if total_qty == 0:
            return None

        return total_value / total_qty

    def get_vwap_ask(self, quantity: float) -> Optional[float]:
        """
        Get volume-weighted average price to buy quantity.

        Args:
            quantity: Amount to buy

        Returns:
            VWAP price to execute buy order
        """
        if not self.asks:
            return None

        remaining = quantity
        total_value = 0.0
        total_qty = 0.0

        for level in self.asks:
            fill_qty = min(remaining, level.quantity)
            total_value += fill_qty * level.price
            total_qty += fill_qty
            remaining -= fill_qty

            if remaining <= 0:
                break

        if total_qty == 0:
            return None

        return total_value / total_qty

    def get_market_impact(self, side: Side, quantity: float) -> float:
        """
        Estimate market impact of order.

        Args:
            side: BUY or SELL
            quantity: Order quantity

        Returns:
            Expected price impact as percentage
        """
        mid = self.mid_price
        if mid is None:
            return 0.0

        if side == Side.BUY:
            vwap = self.get_vwap_ask(quantity)
        else:
            vwap = self.get_vwap_bid(quantity)

        if vwap is None:
            return 0.0

        return abs(vwap - mid) / mid

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'mid_price': self.mid_price,
            'spread_bps': self.spread_bps,
            'bid_depth_10': self.get_bid_depth(10),
            'ask_depth_10': self.get_ask_depth(10),
            'imbalance_5': self.get_imbalance(5),
        }


@dataclass
class Order:
    """
    Order representation.

    hftbacktest concept: Track full order lifecycle.
    """
    order_id: str
    symbol: str
    side: Side
    quantity: float
    price: Optional[float] = None  # None = market order

    # Lifecycle
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0

    # Timing
    created_at: float = 0.0
    submitted_at: Optional[float] = None
    filled_at: Optional[float] = None

    # Queue position (for limit orders)
    queue_position: Optional[int] = None

    # Metadata
    exchange: str = "unknown"
    client_order_id: Optional[str] = None

    @property
    def is_market_order(self) -> bool:
        return self.price is None

    @property
    def is_limit_order(self) -> bool:
        return self.price is not None

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }

    def fill(self, quantity: float, price: float, timestamp: float):
        """
        Record a fill.

        Args:
            quantity: Fill quantity
            price: Fill price
            timestamp: Fill timestamp
        """
        # Update average price
        prev_value = self.filled_quantity * self.average_price
        new_value = quantity * price
        self.filled_quantity += quantity
        self.average_price = (prev_value + new_value) / self.filled_quantity

        # Update status
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
            self.filled_at = timestamp
        else:
            self.status = OrderStatus.PARTIALLY_FILLED


@dataclass
class Fill:
    """
    Execution fill report.

    hftbacktest concept: Detailed fill information.
    """
    order_id: str
    fill_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    timestamp: float

    # Fees
    commission: float = 0.0
    commission_asset: str = "USD"

    # Context
    exchange: str = "unknown"
    is_maker: bool = False  # Maker or taker

    @property
    def value(self) -> float:
        """Total fill value."""
        return self.quantity * self.price

    @property
    def net_value(self) -> float:
        """Fill value after commission."""
        return self.value - self.commission


@dataclass
class Trade:
    """
    Public trade from tape.

    Used for market replay and analysis.
    """
    timestamp: float
    price: float
    quantity: float
    side: Side  # Aggressor side
    trade_id: Optional[str] = None

    @property
    def value(self) -> float:
        return self.price * self.quantity


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_snapshot_from_arrays(
    timestamp: float,
    bid_prices: np.ndarray,
    bid_quantities: np.ndarray,
    ask_prices: np.ndarray,
    ask_quantities: np.ndarray,
    symbol: str = "BTCUSD",
    exchange: str = "unknown"
) -> OrderBookSnapshot:
    """
    Create OrderBookSnapshot from numpy arrays.

    Args:
        timestamp: Snapshot timestamp
        bid_prices: Array of bid prices (descending)
        bid_quantities: Array of bid quantities
        ask_prices: Array of ask prices (ascending)
        ask_quantities: Array of ask quantities
        symbol: Trading symbol
        exchange: Exchange name

    Returns:
        OrderBookSnapshot instance
    """
    bids = [
        Level(price=float(p), quantity=float(q))
        for p, q in zip(bid_prices, bid_quantities)
        if q > 0
    ]

    asks = [
        Level(price=float(p), quantity=float(q))
        for p, q in zip(ask_prices, ask_quantities)
        if q > 0
    ]

    return OrderBookSnapshot(
        timestamp=timestamp,
        symbol=symbol,
        bids=bids,
        asks=asks,
        exchange=exchange,
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Create sample order book
    snapshot = OrderBookSnapshot(
        timestamp=1702500000.0,
        symbol="BTCUSD",
        bids=[
            Level(price=42000.0, quantity=1.5),
            Level(price=41999.0, quantity=2.0),
            Level(price=41998.0, quantity=3.5),
        ],
        asks=[
            Level(price=42001.0, quantity=1.0),
            Level(price=42002.0, quantity=2.5),
            Level(price=42003.0, quantity=4.0),
        ],
        exchange="binance",
    )

    print("Order Book Snapshot")
    print("=" * 40)
    print(f"Symbol: {snapshot.symbol}")
    print(f"Best Bid: {snapshot.best_bid}")
    print(f"Best Ask: {snapshot.best_ask}")
    print(f"Mid Price: {snapshot.mid_price}")
    print(f"Spread: {snapshot.spread:.2f} ({snapshot.spread_bps:.2f} bps)")
    print(f"Imbalance: {snapshot.get_imbalance():.3f}")
    print(f"VWAP to buy 2 BTC: {snapshot.get_vwap_ask(2.0):.2f}")
    print(f"Market impact (buy 5 BTC): {snapshot.get_market_impact(Side.BUY, 5.0)*100:.4f}%")
