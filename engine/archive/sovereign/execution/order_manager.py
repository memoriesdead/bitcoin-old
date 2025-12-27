"""
Order Manager
=============

Order state machine ported from Freqtrade patterns.

Manages order lifecycle:
- Submission
- Confirmation
- Partial fills
- Full fills
- Cancellation
- Timeout handling
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict


class OrderState(Enum):
    """Order lifecycle states."""
    PENDING = auto()       # Created, not submitted
    SUBMITTED = auto()     # Sent to exchange
    OPEN = auto()          # Confirmed, in orderbook
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLING = auto()    # Cancel requested
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()
    ERROR = auto()


class OrderTransition(Enum):
    """Valid state transitions."""
    SUBMIT = auto()
    CONFIRM = auto()
    PARTIAL_FILL = auto()
    FILL = auto()
    CANCEL = auto()
    CANCEL_CONFIRM = auto()
    REJECT = auto()
    EXPIRE = auto()
    ERROR = auto()


@dataclass
class ManagedOrder:
    """
    Order with full state tracking.

    Freqtrade pattern: Track complete order lifecycle.
    """
    # Identity
    client_order_id: str
    exchange_order_id: Optional[str] = None

    # Order details
    symbol: str = ""
    side: str = ""  # "buy" or "sell"
    order_type: str = ""  # "market" or "limit"
    amount: float = 0.0
    price: Optional[float] = None

    # State
    state: OrderState = OrderState.PENDING

    # Fill tracking
    filled_amount: float = 0.0
    average_price: float = 0.0
    total_cost: float = 0.0
    fee: float = 0.0
    fee_currency: str = ""

    # Timestamps
    created_at: float = field(default_factory=time.time)
    submitted_at: Optional[float] = None
    confirmed_at: Optional[float] = None
    filled_at: Optional[float] = None
    cancelled_at: Optional[float] = None

    # Metadata
    timeout_seconds: float = 60.0
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None

    # Trade link
    trade_id: Optional[str] = None

    @property
    def remaining_amount(self) -> float:
        return self.amount - self.filled_amount

    @property
    def is_complete(self) -> bool:
        return self.state in {
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.ERROR,
        }

    @property
    def is_active(self) -> bool:
        return self.state in {
            OrderState.SUBMITTED,
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
            OrderState.CANCELLING,
        }

    @property
    def fill_percentage(self) -> float:
        if self.amount == 0:
            return 0.0
        return (self.filled_amount / self.amount) * 100

    @property
    def is_timed_out(self) -> bool:
        if self.submitted_at is None:
            return False
        return time.time() - self.submitted_at > self.timeout_seconds


# State transition rules
VALID_TRANSITIONS = {
    OrderState.PENDING: {
        OrderTransition.SUBMIT: OrderState.SUBMITTED,
        OrderTransition.ERROR: OrderState.ERROR,
    },
    OrderState.SUBMITTED: {
        OrderTransition.CONFIRM: OrderState.OPEN,
        OrderTransition.FILL: OrderState.FILLED,
        OrderTransition.REJECT: OrderState.REJECTED,
        OrderTransition.ERROR: OrderState.ERROR,
        OrderTransition.EXPIRE: OrderState.EXPIRED,
    },
    OrderState.OPEN: {
        OrderTransition.PARTIAL_FILL: OrderState.PARTIALLY_FILLED,
        OrderTransition.FILL: OrderState.FILLED,
        OrderTransition.CANCEL: OrderState.CANCELLING,
        OrderTransition.EXPIRE: OrderState.EXPIRED,
    },
    OrderState.PARTIALLY_FILLED: {
        OrderTransition.PARTIAL_FILL: OrderState.PARTIALLY_FILLED,
        OrderTransition.FILL: OrderState.FILLED,
        OrderTransition.CANCEL: OrderState.CANCELLING,
    },
    OrderState.CANCELLING: {
        OrderTransition.CANCEL_CONFIRM: OrderState.CANCELLED,
        OrderTransition.FILL: OrderState.FILLED,  # Might fill before cancel
        OrderTransition.ERROR: OrderState.ERROR,
    },
}


class OrderManager:
    """
    Manages order lifecycle and state transitions.

    Freqtrade pattern: Centralized order management with callbacks.
    """

    def __init__(self):
        self.orders: Dict[str, ManagedOrder] = {}
        self.orders_by_exchange_id: Dict[str, str] = {}
        self._lock = threading.Lock()

        # Callbacks
        self.on_state_change: Optional[Callable[[ManagedOrder, OrderState, OrderState], None]] = None
        self.on_fill: Optional[Callable[[ManagedOrder, float, float], None]] = None
        self.on_complete: Optional[Callable[[ManagedOrder], None]] = None

        # Stats
        self.stats = defaultdict(int)

    def create_order(self, client_order_id: str, symbol: str, side: str,
                     order_type: str, amount: float,
                     price: Optional[float] = None,
                     timeout: float = 60.0,
                     trade_id: Optional[str] = None) -> ManagedOrder:
        """
        Create new managed order.

        Args:
            client_order_id: Unique client ID
            symbol: Trading pair
            side: "buy" or "sell"
            order_type: "market" or "limit"
            amount: Order amount
            price: Limit price (None for market)
            timeout: Order timeout in seconds
            trade_id: Associated trade ID

        Returns:
            New ManagedOrder
        """
        order = ManagedOrder(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
            timeout_seconds=timeout,
            trade_id=trade_id,
        )

        with self._lock:
            self.orders[client_order_id] = order
            self.stats['orders_created'] += 1

        return order

    def transition(self, client_order_id: str, transition: OrderTransition,
                   **kwargs) -> bool:
        """
        Transition order to new state.

        Args:
            client_order_id: Order ID
            transition: Transition to apply
            **kwargs: Additional data (exchange_order_id, filled, price, etc.)

        Returns:
            True if transition successful
        """
        with self._lock:
            order = self.orders.get(client_order_id)
            if order is None:
                return False

            # Check if transition is valid
            valid_transitions = VALID_TRANSITIONS.get(order.state, {})
            new_state = valid_transitions.get(transition)

            if new_state is None:
                self.stats['invalid_transitions'] += 1
                return False

            old_state = order.state
            order.state = new_state

            # Handle transition-specific updates
            self._handle_transition(order, transition, **kwargs)

            # Update stats
            self.stats[f'transition_{transition.name.lower()}'] += 1

            # Fire callbacks
            if self.on_state_change:
                self.on_state_change(order, old_state, new_state)

            if new_state in {OrderState.FILLED, OrderState.PARTIALLY_FILLED}:
                if self.on_fill:
                    self.on_fill(order, kwargs.get('fill_amount', 0), kwargs.get('fill_price', 0))

            if order.is_complete and self.on_complete:
                self.on_complete(order)

            return True

    def _handle_transition(self, order: ManagedOrder, transition: OrderTransition,
                           **kwargs):
        """Handle transition-specific updates."""
        if transition == OrderTransition.SUBMIT:
            order.submitted_at = time.time()
            if 'exchange_order_id' in kwargs:
                order.exchange_order_id = kwargs['exchange_order_id']
                self.orders_by_exchange_id[kwargs['exchange_order_id']] = order.client_order_id

        elif transition == OrderTransition.CONFIRM:
            order.confirmed_at = time.time()
            if 'exchange_order_id' in kwargs:
                order.exchange_order_id = kwargs['exchange_order_id']
                self.orders_by_exchange_id[kwargs['exchange_order_id']] = order.client_order_id

        elif transition in {OrderTransition.PARTIAL_FILL, OrderTransition.FILL}:
            fill_amount = kwargs.get('fill_amount', 0)
            fill_price = kwargs.get('fill_price', order.price or 0)

            # Update average price
            prev_value = order.filled_amount * order.average_price
            new_value = fill_amount * fill_price
            order.filled_amount += fill_amount

            if order.filled_amount > 0:
                order.average_price = (prev_value + new_value) / order.filled_amount

            order.total_cost = order.filled_amount * order.average_price

            if 'fee' in kwargs:
                order.fee += kwargs['fee']
            if 'fee_currency' in kwargs:
                order.fee_currency = kwargs['fee_currency']

            if transition == OrderTransition.FILL:
                order.filled_at = time.time()

        elif transition == OrderTransition.CANCEL_CONFIRM:
            order.cancelled_at = time.time()

        elif transition in {OrderTransition.REJECT, OrderTransition.ERROR}:
            if 'error' in kwargs:
                order.error_message = kwargs['error']

    def get_order(self, client_order_id: str) -> Optional[ManagedOrder]:
        """Get order by client ID."""
        return self.orders.get(client_order_id)

    def get_order_by_exchange_id(self, exchange_order_id: str) -> Optional[ManagedOrder]:
        """Get order by exchange ID."""
        client_id = self.orders_by_exchange_id.get(exchange_order_id)
        if client_id:
            return self.orders.get(client_id)
        return None

    def get_active_orders(self) -> List[ManagedOrder]:
        """Get all active (non-complete) orders."""
        with self._lock:
            return [o for o in self.orders.values() if o.is_active]

    def get_orders_by_trade(self, trade_id: str) -> List[ManagedOrder]:
        """Get all orders for a trade."""
        with self._lock:
            return [o for o in self.orders.values() if o.trade_id == trade_id]

    def check_timeouts(self) -> List[ManagedOrder]:
        """
        Check for timed out orders.

        Returns:
            List of timed out orders
        """
        timed_out = []

        with self._lock:
            for order in self.orders.values():
                if order.is_active and order.is_timed_out:
                    timed_out.append(order)
                    self.stats['timeouts'] += 1

        return timed_out

    def get_stats(self) -> Dict[str, int]:
        """Get order management statistics."""
        with self._lock:
            return dict(self.stats)

    def cleanup_completed(self, max_age_seconds: float = 3600):
        """
        Remove old completed orders.

        Args:
            max_age_seconds: Remove orders older than this
        """
        now = time.time()
        to_remove = []

        with self._lock:
            for order_id, order in self.orders.items():
                if order.is_complete:
                    age = now - order.created_at
                    if age > max_age_seconds:
                        to_remove.append(order_id)

            for order_id in to_remove:
                order = self.orders.pop(order_id)
                if order.exchange_order_id:
                    self.orders_by_exchange_id.pop(order.exchange_order_id, None)

            self.stats['orders_cleaned'] += len(to_remove)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("Order Manager Demo")
    print("=" * 50)

    manager = OrderManager()

    # Callbacks
    def on_state_change(order, old, new):
        print(f"  State: {old.name} -> {new.name}")

    def on_fill(order, amount, price):
        print(f"  Fill: {amount} @ {price}")

    manager.on_state_change = on_state_change
    manager.on_fill = on_fill

    # Create order
    order = manager.create_order(
        client_order_id="test_001",
        symbol="BTC/USDT",
        side="buy",
        order_type="limit",
        amount=0.1,
        price=42000.0,
    )

    print(f"\nCreated order: {order.client_order_id}")
    print(f"Initial state: {order.state.name}")

    # Submit
    print("\nSubmitting...")
    manager.transition("test_001", OrderTransition.SUBMIT,
                       exchange_order_id="EX_123")

    # Confirm
    print("\nConfirming...")
    manager.transition("test_001", OrderTransition.CONFIRM)

    # Partial fill
    print("\nPartial fill...")
    manager.transition("test_001", OrderTransition.PARTIAL_FILL,
                       fill_amount=0.05, fill_price=42000.0)

    print(f"Filled: {order.fill_percentage:.1f}%")

    # Full fill
    print("\nFull fill...")
    manager.transition("test_001", OrderTransition.FILL,
                       fill_amount=0.05, fill_price=41995.0)

    print(f"\nFinal state: {order.state.name}")
    print(f"Average price: {order.average_price:.2f}")
    print(f"Total cost: {order.total_cost:.2f}")

    print(f"\nStats: {manager.get_stats()}")
