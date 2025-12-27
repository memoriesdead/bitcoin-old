"""
Execution Simulator
===================

Realistic order execution simulation.
Ported from hftbacktest execution logic.

Features:
- Queue position tracking
- Depth walking for market orders
- Partial fills
- Market impact estimation
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import time

from .types import (
    OrderBookSnapshot, Order, OrderStatus, Fill, Side, Level
)


@dataclass
class ExecutionResult:
    """Result of an execution attempt."""
    success: bool
    fills: List[Fill] = field(default_factory=list)
    remaining_quantity: float = 0.0
    average_price: float = 0.0
    total_slippage: float = 0.0
    market_impact: float = 0.0
    latency_ms: float = 0.0
    error: Optional[str] = None


class DepthWalker:
    """
    Walks through order book depth to execute orders.

    hftbacktest concept: Simulate execution by consuming liquidity
    at each price level.
    """

    def __init__(self, snapshot: OrderBookSnapshot):
        self.snapshot = snapshot

    def walk_buy(self, quantity: float) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Walk through asks to fill a buy order.

        Args:
            quantity: Amount to buy

        Returns:
            (average_price, list of (price, quantity) fills)
        """
        remaining = quantity
        fills = []
        total_value = 0.0

        for level in self.snapshot.asks:
            if remaining <= 0:
                break

            fill_qty = min(remaining, level.quantity)
            fills.append((level.price, fill_qty))
            total_value += level.price * fill_qty
            remaining -= fill_qty

        total_filled = quantity - remaining
        avg_price = total_value / total_filled if total_filled > 0 else 0

        return avg_price, fills

    def walk_sell(self, quantity: float) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Walk through bids to fill a sell order.

        Args:
            quantity: Amount to sell

        Returns:
            (average_price, list of (price, quantity) fills)
        """
        remaining = quantity
        fills = []
        total_value = 0.0

        for level in self.snapshot.bids:
            if remaining <= 0:
                break

            fill_qty = min(remaining, level.quantity)
            fills.append((level.price, fill_qty))
            total_value += level.price * fill_qty
            remaining -= fill_qty

        total_filled = quantity - remaining
        avg_price = total_value / total_filled if total_filled > 0 else 0

        return avg_price, fills


class QueuePositionTracker:
    """
    Tracks queue position for limit orders.

    hftbacktest concept: Estimate position in queue at a price level.
    Earlier orders have priority.
    """

    def __init__(self):
        # Track orders by (symbol, price)
        self.orders: Dict[Tuple[str, float], deque] = {}
        self.order_positions: Dict[str, Tuple[str, float, int]] = {}

    def add_order(self, order: Order, snapshot: OrderBookSnapshot) -> int:
        """
        Add order to queue and estimate position.

        Args:
            order: The limit order
            snapshot: Current order book state

        Returns:
            Estimated queue position
        """
        if order.price is None:
            raise ValueError("Cannot track queue position for market order")

        key = (order.symbol, order.price)

        # Find quantity ahead of us in the queue
        if order.side == Side.BUY:
            # For buy orders, look at bid side
            queue_ahead = 0
            for level in snapshot.bids:
                if level.price > order.price:
                    queue_ahead += level.quantity
                elif level.price == order.price:
                    # Assume we're at the back of this level
                    queue_ahead += level.quantity
                    break
        else:
            # For sell orders, look at ask side
            queue_ahead = 0
            for level in snapshot.asks:
                if level.price < order.price:
                    queue_ahead += level.quantity
                elif level.price == order.price:
                    queue_ahead += level.quantity
                    break

        # Estimate position as quantity ahead
        position = int(queue_ahead * 100)  # Scale to integer

        # Store position
        self.order_positions[order.order_id] = (order.symbol, order.price, position)

        return position

    def update_position(self, order_id: str, trade_quantity: float,
                        trade_price: float) -> Optional[float]:
        """
        Update queue position after a trade.

        Args:
            order_id: Order to update
            trade_quantity: Quantity traded at this level
            trade_price: Trade price

        Returns:
            New position estimate, or None if order not tracked
        """
        if order_id not in self.order_positions:
            return None

        symbol, price, position = self.order_positions[order_id]

        if trade_price == price:
            # Trade at our price level - reduce position
            new_position = max(0, position - trade_quantity * 100)
            self.order_positions[order_id] = (symbol, price, int(new_position))
            return new_position

        return position

    def check_fillable(self, order_id: str) -> bool:
        """
        Check if order has reached front of queue.

        Args:
            order_id: Order to check

        Returns:
            True if order should be fillable
        """
        if order_id not in self.order_positions:
            return False

        _, _, position = self.order_positions[order_id]
        return position <= 0

    def remove_order(self, order_id: str):
        """Remove order from tracking."""
        self.order_positions.pop(order_id, None)


class ExecutionSimulator:
    """
    Full execution simulation engine.

    Combines:
    - Depth walking for market orders
    - Queue position tracking for limit orders
    - Latency simulation
    - Fee calculation
    """

    def __init__(self,
                 taker_fee: float = 0.001,
                 maker_fee: float = 0.0005,
                 min_latency_ms: float = 10.0,
                 max_latency_ms: float = 100.0):
        """
        Initialize execution simulator.

        Args:
            taker_fee: Taker fee rate (0.001 = 0.1%)
            maker_fee: Maker fee rate
            min_latency_ms: Minimum execution latency
            max_latency_ms: Maximum execution latency
        """
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms

        self.queue_tracker = QueuePositionTracker()

        # Stats
        self.stats = {
            'market_orders': 0,
            'limit_orders': 0,
            'total_fills': 0,
            'total_slippage': 0.0,
        }

    def execute_market_order(self, order: Order,
                             snapshot: OrderBookSnapshot) -> ExecutionResult:
        """
        Execute a market order against order book.

        Args:
            order: Market order to execute
            snapshot: Current order book state

        Returns:
            ExecutionResult with fills
        """
        if not order.is_market_order:
            raise ValueError("Expected market order")

        self.stats['market_orders'] += 1

        # Simulate latency
        latency = np.random.uniform(self.min_latency_ms, self.max_latency_ms)

        # Walk the book
        walker = DepthWalker(snapshot)

        if order.side == Side.BUY:
            avg_price, fill_pairs = walker.walk_buy(order.quantity)
            reference_price = snapshot.best_ask or snapshot.mid_price
        else:
            avg_price, fill_pairs = walker.walk_sell(order.quantity)
            reference_price = snapshot.best_bid or snapshot.mid_price

        if not fill_pairs:
            return ExecutionResult(
                success=False,
                error="Insufficient liquidity",
                latency_ms=latency,
            )

        # Create fills
        fills = []
        total_filled = 0.0
        total_value = 0.0

        for price, qty in fill_pairs:
            commission = price * qty * self.taker_fee

            fill = Fill(
                order_id=order.order_id,
                fill_id=f"{order.order_id}_{len(fills)}",
                symbol=order.symbol,
                side=order.side,
                quantity=qty,
                price=price,
                timestamp=snapshot.timestamp + latency / 1000,
                commission=commission,
                is_maker=False,
            )
            fills.append(fill)
            total_filled += qty
            total_value += price * qty

        # Calculate slippage
        if reference_price and reference_price > 0:
            slippage = abs(avg_price - reference_price) / reference_price
        else:
            slippage = 0.0

        self.stats['total_fills'] += len(fills)
        self.stats['total_slippage'] += slippage

        return ExecutionResult(
            success=True,
            fills=fills,
            remaining_quantity=order.quantity - total_filled,
            average_price=avg_price,
            total_slippage=slippage,
            market_impact=snapshot.get_market_impact(order.side, order.quantity),
            latency_ms=latency,
        )

    def execute_limit_order(self, order: Order,
                            snapshot: OrderBookSnapshot) -> ExecutionResult:
        """
        Execute a limit order against order book.

        Args:
            order: Limit order to execute
            snapshot: Current order book state

        Returns:
            ExecutionResult (may be partially filled or nothing)
        """
        if order.is_market_order:
            raise ValueError("Expected limit order")

        self.stats['limit_orders'] += 1

        latency = np.random.uniform(self.min_latency_ms, self.max_latency_ms)

        # Check if immediately fillable (crosses the spread)
        fills = []

        if order.side == Side.BUY:
            # Buy limit order - fills if price >= best ask
            if snapshot.best_ask and order.price >= snapshot.best_ask:
                # Execute as taker
                walker = DepthWalker(snapshot)
                avg_price, fill_pairs = walker.walk_buy(order.quantity)

                # Only fill up to our limit price
                for price, qty in fill_pairs:
                    if price <= order.price:
                        fill = Fill(
                            order_id=order.order_id,
                            fill_id=f"{order.order_id}_{len(fills)}",
                            symbol=order.symbol,
                            side=order.side,
                            quantity=qty,
                            price=price,
                            timestamp=snapshot.timestamp + latency / 1000,
                            commission=price * qty * self.taker_fee,
                            is_maker=False,
                        )
                        fills.append(fill)
        else:
            # Sell limit order - fills if price <= best bid
            if snapshot.best_bid and order.price <= snapshot.best_bid:
                walker = DepthWalker(snapshot)
                avg_price, fill_pairs = walker.walk_sell(order.quantity)

                for price, qty in fill_pairs:
                    if price >= order.price:
                        fill = Fill(
                            order_id=order.order_id,
                            fill_id=f"{order.order_id}_{len(fills)}",
                            symbol=order.symbol,
                            side=order.side,
                            quantity=qty,
                            price=price,
                            timestamp=snapshot.timestamp + latency / 1000,
                            commission=price * qty * self.taker_fee,
                            is_maker=False,
                        )
                        fills.append(fill)

        if fills:
            total_filled = sum(f.quantity for f in fills)
            avg_price = sum(f.quantity * f.price for f in fills) / total_filled
            self.stats['total_fills'] += len(fills)

            return ExecutionResult(
                success=True,
                fills=fills,
                remaining_quantity=order.quantity - total_filled,
                average_price=avg_price,
                latency_ms=latency,
            )

        # Order didn't cross - add to queue
        position = self.queue_tracker.add_order(order, snapshot)

        return ExecutionResult(
            success=True,  # Order accepted, just not filled
            remaining_quantity=order.quantity,
            latency_ms=latency,
        )

    def update_limit_order(self, order: Order,
                           snapshot: OrderBookSnapshot) -> Optional[Fill]:
        """
        Check if a pending limit order should fill.

        Args:
            order: Pending limit order
            snapshot: Current order book state

        Returns:
            Fill if order filled, None otherwise
        """
        if not self.queue_tracker.check_fillable(order.order_id):
            return None

        # Create maker fill
        fill = Fill(
            order_id=order.order_id,
            fill_id=f"{order.order_id}_maker",
            symbol=order.symbol,
            side=order.side,
            quantity=order.remaining_quantity,
            price=order.price,
            timestamp=snapshot.timestamp,
            commission=order.price * order.remaining_quantity * self.maker_fee,
            is_maker=True,
        )

        self.queue_tracker.remove_order(order.order_id)
        self.stats['total_fills'] += 1

        return fill

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total_orders = self.stats['market_orders'] + self.stats['limit_orders']
        return {
            **self.stats,
            'avg_slippage': (
                self.stats['total_slippage'] / self.stats['market_orders']
                if self.stats['market_orders'] > 0 else 0
            ),
            'total_orders': total_orders,
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    from .types import OrderBookSnapshot, Level, Order, Side

    # Create sample order book
    snapshot = OrderBookSnapshot(
        timestamp=time.time(),
        symbol="BTCUSDT",
        bids=[
            Level(price=42000.0, quantity=1.0),
            Level(price=41999.0, quantity=2.0),
            Level(price=41998.0, quantity=5.0),
        ],
        asks=[
            Level(price=42001.0, quantity=0.5),
            Level(price=42002.0, quantity=1.5),
            Level(price=42003.0, quantity=3.0),
        ],
        exchange="binance",
    )

    print("Execution Simulator Demo")
    print("=" * 50)

    simulator = ExecutionSimulator(
        taker_fee=0.001,
        maker_fee=0.0005,
    )

    # Market buy order
    market_order = Order(
        order_id="MKT_001",
        symbol="BTCUSDT",
        side=Side.BUY,
        quantity=2.0,
    )

    result = simulator.execute_market_order(market_order, snapshot)
    print(f"\nMarket Buy 2 BTC:")
    print(f"  Success: {result.success}")
    print(f"  Avg Price: {result.average_price:.2f}")
    print(f"  Slippage: {result.total_slippage*100:.4f}%")
    print(f"  Fills: {len(result.fills)}")
    for fill in result.fills:
        print(f"    {fill.quantity:.4f} @ {fill.price:.2f}")

    # Limit order
    limit_order = Order(
        order_id="LMT_001",
        symbol="BTCUSDT",
        side=Side.BUY,
        quantity=1.0,
        price=41999.0,  # Below best ask
    )

    result = simulator.execute_limit_order(limit_order, snapshot)
    print(f"\nLimit Buy 1 BTC @ 41999:")
    print(f"  Success: {result.success}")
    print(f"  Fills: {len(result.fills)} (in queue)")

    print(f"\nStats: {simulator.get_stats()}")
