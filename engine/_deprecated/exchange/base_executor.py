"""
BASE EXECUTOR - Abstract Exchange Execution Layer
==================================================
All exchanges implement this interface for unified order routing.

The signal engine generates 237K+ signals/second.
The executor layer batches and routes to real exchanges.

ARCHITECTURE:
Signal Engine (237K TPS) → Message Queue → Executor (batched) → Exchange
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict
import asyncio
import time


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Signal:
    """Trading signal from the HFT engine."""
    timestamp: float
    instrument: str
    side: OrderSide
    strength: float  # 0.0 to 1.0
    edge_bps: float  # Expected edge in basis points
    confidence: float  # Signal confidence
    formula_id: int  # Which formula generated this (701=OFI, etc.)

    # Optional price targets
    entry_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None


@dataclass
class Order:
    """Order to be sent to exchange."""
    id: str
    timestamp: float
    instrument: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None

    # Execution details
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    exchange: str = ""
    exchange_order_id: str = ""

    # Risk management
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None


@dataclass
class Position:
    """Current position on an instrument."""
    instrument: str
    side: OrderSide
    quantity: float
    entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    exchange: str


@dataclass
class ExchangeConfig:
    """Exchange connection configuration."""
    name: str
    api_key: str
    api_secret: str
    testnet: bool = True
    rate_limit_per_second: int = 20
    rate_limit_per_minute: int = 1200
    max_position_size: float = 10000.0  # USD
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%


class BaseExecutor(ABC):
    """
    Abstract base class for exchange executors.

    Each exchange (Binance, Bybit, etc.) implements this interface.
    The executor handles:
    - WebSocket connection management
    - Order submission and tracking
    - Position management
    - Rate limiting
    - Error handling and retries
    """

    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []

        # Rate limiting
        self._order_timestamps: List[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # Statistics
        self.stats = {
            'orders_sent': 0,
            'orders_filled': 0,
            'orders_rejected': 0,
            'total_volume': 0.0,
            'total_pnl': 0.0,
            'avg_latency_ms': 0.0,
        }

    @abstractmethod
    async def connect(self) -> bool:
        """Establish WebSocket connection to exchange."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        pass

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """Submit order to exchange, return updated order with exchange ID."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        pass

    @abstractmethod
    async def get_position(self, instrument: str) -> Optional[Position]:
        """Get current position for an instrument."""
        pass

    @abstractmethod
    async def get_orderbook(self, instrument: str) -> Dict:
        """Get current orderbook (top N levels)."""
        pass

    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """Get account balances."""
        pass

    async def can_send_order(self) -> bool:
        """Check if we can send an order (rate limiting)."""
        async with self._rate_limit_lock:
            now = time.time()

            # Clean old timestamps
            self._order_timestamps = [
                ts for ts in self._order_timestamps
                if now - ts < 60
            ]

            # Check per-second limit
            recent_second = sum(1 for ts in self._order_timestamps if now - ts < 1)
            if recent_second >= self.config.rate_limit_per_second:
                return False

            # Check per-minute limit
            if len(self._order_timestamps) >= self.config.rate_limit_per_minute:
                return False

            return True

    async def record_order(self) -> None:
        """Record an order timestamp for rate limiting."""
        async with self._rate_limit_lock:
            self._order_timestamps.append(time.time())

    def signal_to_order(
        self,
        signal: Signal,
        capital: float,
        kelly_fraction: float = 0.25
    ) -> Order:
        """
        Convert a trading signal to an order.

        Uses Kelly Criterion for position sizing:
        f* = (p * b - q) / b

        Where:
        - p = win probability (from signal confidence)
        - q = 1 - p
        - b = win/loss ratio (from TP/SL)
        """
        # Kelly sizing
        win_prob = 0.479 + (signal.confidence * 0.05)  # Base + signal boost
        loss_prob = 1 - win_prob
        win_loss_ratio = 2.0  # 2:1 TP/SL

        kelly = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
        position_fraction = kelly * kelly_fraction  # Quarter-Kelly

        # Calculate position size
        position_usd = capital * position_fraction
        position_usd = min(position_usd, self.config.max_position_size)

        # Convert to quantity (assuming price is available)
        if signal.entry_price:
            quantity = position_usd / signal.entry_price
        else:
            quantity = position_usd  # Will be filled at market

        return Order(
            id=f"{self.config.name}_{int(time.time() * 1000000)}",
            timestamp=signal.timestamp,
            instrument=signal.instrument,
            side=signal.side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
            exchange=self.config.name,
        )

    def calculate_fees(self, order: Order, is_maker: bool = False) -> float:
        """Calculate trading fees for an order."""
        notional = order.filled_quantity * order.filled_price
        fee_rate = self.config.maker_fee if is_maker else self.config.taker_fee
        return notional * fee_rate
