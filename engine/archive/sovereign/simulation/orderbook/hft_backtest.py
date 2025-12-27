"""
HFT Backtester
==============

High-fidelity backtesting with order book simulation.

Ported from hftbacktest patterns:
- Full order book reconstruction
- Queue position tracking
- Realistic latency modeling
- Exchange-specific fills

This provides much more accurate backtest results than
simple OHLCV-based backtesting.
"""

import time
from typing import Dict, List, Optional, Any, Callable, Iterator
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from .types import (
    OrderBookSnapshot, Order, Fill, Trade,
    Side, OrderStatus, Level
)
from .loader import OrderBookLoader, InMemoryOrderBookLoader
from .execution import ExecutionSimulator
from .latency import LatencyModel, create_latency_model


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    # Time settings
    start_time: float = 0.0
    end_time: float = float('inf')

    # Capital
    initial_capital: float = 10000.0
    base_currency: str = "USDT"

    # Execution
    exchange: str = "binance"
    fee_rate: float = 0.001  # 0.1%
    slippage_model: str = "realistic"  # "zero", "fixed", "realistic"

    # Risk
    max_position: float = 1.0  # Max BTC
    max_order_size: float = 0.1

    # Latency
    use_latency: bool = True
    order_latency_ms: float = 50.0
    data_latency_ms: float = 10.0

    # Output
    verbose: bool = False
    log_trades: bool = True


@dataclass
class BacktestState:
    """Current backtest state."""
    timestamp: float = 0.0

    # Capital
    cash: float = 0.0
    position: float = 0.0
    equity: float = 0.0

    # Performance
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Orders
    pending_orders: List[Order] = field(default_factory=list)
    active_orders: Dict[str, Order] = field(default_factory=dict)

    # Tracking
    trades: List[Fill] = field(default_factory=list)
    equity_curve: List[tuple] = field(default_factory=list)

    # Stats
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    total_fees: float = 0.0


@dataclass
class BacktestResult:
    """Backtesting results."""
    # Performance
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0

    # Trading
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Execution
    avg_slippage: float = 0.0
    total_fees: float = 0.0
    fill_rate: float = 0.0

    # Time
    start_time: float = 0.0
    end_time: float = 0.0
    duration_days: float = 0.0

    # Raw data
    equity_curve: List[tuple] = field(default_factory=list)
    trades: List[Fill] = field(default_factory=list)


class HFTBacktester:
    """
    High-fidelity backtester.

    Uses order book data for realistic execution simulation.
    """

    def __init__(self, config: BacktestConfig,
                 loader: Optional[OrderBookLoader] = None):
        """
        Initialize backtester.

        Args:
            config: Backtest configuration
            loader: Order book data loader
        """
        self.config = config
        self.loader = loader or InMemoryOrderBookLoader()

        # Execution simulator
        self.executor = ExecutionSimulator(
            exchange=config.exchange,
            fee_rate=config.fee_rate,
        )

        # Latency model
        self.latency = create_latency_model(
            config.exchange,
            base_latency_ms=config.order_latency_ms
        ) if config.use_latency else None

        # State
        self.state = BacktestState(cash=config.initial_capital)

        # Current market state
        self._current_book: Optional[OrderBookSnapshot] = None
        self._mid_price: float = 0.0
        self._last_trade_price: float = 0.0

        # Order tracking
        self._next_order_id = 1

        # Strategy callback
        self.strategy: Optional[Callable[[OrderBookSnapshot, BacktestState], List[Order]]] = None

    def set_strategy(self, strategy: Callable[[OrderBookSnapshot, BacktestState], List[Order]]):
        """
        Set trading strategy.

        Args:
            strategy: Function that takes (orderbook, state) and returns orders
        """
        self.strategy = strategy

    def run(self, symbol: str = "BTCUSDT") -> BacktestResult:
        """
        Run backtest.

        Args:
            symbol: Trading symbol

        Returns:
            BacktestResult
        """
        if self.strategy is None:
            raise ValueError("No strategy set. Call set_strategy() first.")

        start_time = time.time()

        # Iterate through order book snapshots
        for snapshot in self.loader.iter_snapshots(
            symbol,
            self.config.start_time,
            self.config.end_time
        ):
            self._process_snapshot(snapshot)

        # Finalize
        self._close_positions()
        result = self._compute_results()
        result.start_time = self.config.start_time
        result.end_time = self.state.timestamp

        if self.config.verbose:
            elapsed = time.time() - start_time
            print(f"[BACKTEST] Completed in {elapsed:.2f}s")
            print(f"[BACKTEST] Total trades: {result.total_trades}")
            print(f"[BACKTEST] Total return: {result.total_return*100:.2f}%")

        return result

    def _process_snapshot(self, snapshot: OrderBookSnapshot):
        """Process single order book snapshot."""
        self._current_book = snapshot
        self.state.timestamp = snapshot.timestamp
        self._mid_price = snapshot.get_mid_price()

        # 1. Process pending orders (latency simulation)
        self._process_pending_orders(snapshot)

        # 2. Check for fills on active orders
        self._check_fills(snapshot)

        # 3. Get strategy orders
        new_orders = self.strategy(snapshot, self.state)

        # 4. Submit new orders
        for order in new_orders:
            self._submit_order(order)

        # 5. Update state
        self._update_state(snapshot)

    def _process_pending_orders(self, snapshot: OrderBookSnapshot):
        """Process pending orders that have finished latency period."""
        still_pending = []

        for order in self.state.pending_orders:
            # Check if latency period is over
            if order.metadata.get('submit_time', 0) + self._get_latency_seconds() <= snapshot.timestamp:
                # Order reaches exchange
                order.status = OrderStatus.OPEN
                self.state.active_orders[order.order_id] = order

                if self.config.verbose:
                    print(f"[ORDER] {order.order_id} now active: {order.side.value} {order.amount} @ {order.price}")
            else:
                still_pending.append(order)

        self.state.pending_orders = still_pending

    def _check_fills(self, snapshot: OrderBookSnapshot):
        """Check active orders for fills."""
        filled_ids = []

        for order_id, order in self.state.active_orders.items():
            fill = self.executor.simulate_fill(order, snapshot)

            if fill and fill.amount > 0:
                self._process_fill(order, fill)

                if order.filled >= order.amount:
                    order.status = OrderStatus.FILLED
                    filled_ids.append(order_id)
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED

        # Remove filled orders
        for order_id in filled_ids:
            del self.state.active_orders[order_id]
            self.state.filled_orders += 1

    def _process_fill(self, order: Order, fill: Fill):
        """Process order fill."""
        order.filled += fill.amount
        order.average_price = (
            (order.average_price * (order.filled - fill.amount) + fill.price * fill.amount)
            / order.filled
        ) if order.filled > 0 else fill.price

        # Update position
        if order.side == Side.BUY:
            self.state.position += fill.amount
            self.state.cash -= fill.amount * fill.price + fill.fee
        else:
            self.state.position -= fill.amount
            self.state.cash += fill.amount * fill.price - fill.fee

        self.state.total_fees += fill.fee
        self.state.trades.append(fill)
        self._last_trade_price = fill.price

        if self.config.verbose:
            print(f"[FILL] {order.side.value} {fill.amount:.6f} @ ${fill.price:.2f}, fee: ${fill.fee:.4f}")

    def _submit_order(self, order: Order):
        """Submit new order."""
        # Assign order ID
        order.order_id = f"BT_{self._next_order_id:08d}"
        self._next_order_id += 1

        # Validate order
        if not self._validate_order(order):
            return

        # Apply latency
        order.status = OrderStatus.PENDING
        order.metadata = {'submit_time': self.state.timestamp}
        self.state.pending_orders.append(order)
        self.state.total_orders += 1

        if self.config.verbose:
            print(f"[SUBMIT] {order.order_id}: {order.side.value} {order.amount} @ {order.price}")

    def _validate_order(self, order: Order) -> bool:
        """Validate order against risk limits."""
        # Position limit
        new_position = self.state.position
        if order.side == Side.BUY:
            new_position += order.amount
        else:
            new_position -= order.amount

        if abs(new_position) > self.config.max_position:
            if self.config.verbose:
                print(f"[REJECT] Position limit exceeded")
            return False

        # Order size limit
        if order.amount > self.config.max_order_size:
            if self.config.verbose:
                print(f"[REJECT] Order size limit exceeded")
            return False

        # Capital check for buys
        if order.side == Side.BUY:
            cost = order.amount * order.price * (1 + self.config.fee_rate)
            if cost > self.state.cash:
                if self.config.verbose:
                    print(f"[REJECT] Insufficient capital")
                return False

        return True

    def _update_state(self, snapshot: OrderBookSnapshot):
        """Update backtest state."""
        # Calculate unrealized PnL
        if self.state.position != 0:
            self.state.unrealized_pnl = (
                self.state.position *
                (self._mid_price - self._last_trade_price)
            )

        # Calculate equity
        self.state.equity = (
            self.state.cash +
            self.state.position * self._mid_price
        )

        # Record equity curve
        self.state.equity_curve.append((
            snapshot.timestamp,
            self.state.equity
        ))

    def _close_positions(self):
        """Close any remaining positions at end of backtest."""
        if self.state.position != 0 and self._current_book:
            # Create market order to close
            side = Side.SELL if self.state.position > 0 else Side.BUY
            order = Order(
                order_id="",
                symbol=self._current_book.symbol,
                side=side,
                order_type="market",
                amount=abs(self.state.position),
                price=self._mid_price,
                timestamp=self.state.timestamp,
            )

            # Simulate fill
            fill = self.executor.simulate_fill(order, self._current_book)
            if fill:
                self._process_fill(order, fill)

    def _get_latency_seconds(self) -> float:
        """Get order latency in seconds."""
        if self.latency:
            return self.latency.sample() / 1000.0
        return self.config.order_latency_ms / 1000.0

    def _compute_results(self) -> BacktestResult:
        """Compute backtest results."""
        result = BacktestResult()

        # Equity curve
        result.equity_curve = self.state.equity_curve
        result.trades = self.state.trades

        if len(self.state.equity_curve) < 2:
            return result

        # Convert to numpy for calculations
        times = np.array([e[0] for e in self.state.equity_curve])
        equity = np.array([e[1] for e in self.state.equity_curve])

        # Returns
        returns = np.diff(equity) / equity[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) == 0:
            return result

        # Total return
        result.total_return = (equity[-1] / equity[0]) - 1

        # Duration
        result.duration_days = (times[-1] - times[0]) / 86400

        # Annualized return
        if result.duration_days > 0:
            result.annualized_return = (
                (1 + result.total_return) ** (365 / result.duration_days) - 1
            )

        # Sharpe ratio (assume 252 trading days, 0% risk-free rate)
        if len(returns) > 1 and np.std(returns) > 0:
            daily_factor = 86400 / np.mean(np.diff(times)) if len(times) > 1 else 1
            result.sharpe_ratio = (
                np.mean(returns) / np.std(returns) *
                np.sqrt(252 * daily_factor)
            )

        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            daily_factor = 86400 / np.mean(np.diff(times)) if len(times) > 1 else 1
            result.sortino_ratio = (
                np.mean(returns) / np.std(downside) *
                np.sqrt(252 * daily_factor)
            )

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        result.max_drawdown = np.max(drawdown)

        # Calmar ratio
        if result.max_drawdown > 0:
            result.calmar_ratio = result.annualized_return / result.max_drawdown

        # Trade statistics
        result.total_trades = len(self.state.trades)
        result.total_fees = self.state.total_fees
        result.fill_rate = (
            self.state.filled_orders / self.state.total_orders
            if self.state.total_orders > 0 else 0
        )

        # Win rate and PnL
        if result.total_trades > 0:
            trade_pnls = self._compute_trade_pnls()
            wins = [p for p in trade_pnls if p > 0]
            losses = [p for p in trade_pnls if p < 0]

            result.win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0
            result.avg_trade_pnl = np.mean(trade_pnls) if trade_pnls else 0
            result.avg_win = np.mean(wins) if wins else 0
            result.avg_loss = np.mean(losses) if losses else 0

            # Profit factor
            total_wins = sum(wins)
            total_losses = abs(sum(losses))
            result.profit_factor = (
                total_wins / total_losses if total_losses > 0 else float('inf')
            )

        # Slippage
        if self.state.trades:
            slippages = [t.slippage for t in self.state.trades if t.slippage is not None]
            result.avg_slippage = np.mean(slippages) if slippages else 0

        return result

    def _compute_trade_pnls(self) -> List[float]:
        """Compute PnL for each round-trip trade."""
        # Simple approach: pair buys with sells
        pnls = []
        position = 0.0
        entry_cost = 0.0

        for trade in self.state.trades:
            if trade.side == Side.BUY:
                entry_cost += trade.amount * trade.price + trade.fee
                position += trade.amount
            else:
                if position > 0:
                    # Calculate PnL for this exit
                    exit_value = trade.amount * trade.price - trade.fee
                    avg_entry = entry_cost / position if position > 0 else 0
                    pnl = exit_value - trade.amount * avg_entry
                    pnls.append(pnl)

                    # Update position
                    entry_cost -= trade.amount * avg_entry
                    position -= trade.amount

        return pnls

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        # Check pending
        for i, order in enumerate(self.state.pending_orders):
            if order.order_id == order_id:
                self.state.pending_orders.pop(i)
                self.state.cancelled_orders += 1
                return True

        # Check active
        if order_id in self.state.active_orders:
            del self.state.active_orders[order_id]
            self.state.cancelled_orders += 1
            return True

        return False

    def get_state(self) -> BacktestState:
        """Get current backtest state."""
        return self.state


# =============================================================================
# EXAMPLE STRATEGIES
# =============================================================================

def simple_market_maker(book: OrderBookSnapshot, state: BacktestState) -> List[Order]:
    """
    Simple market making strategy.

    Places orders on both sides of the book.
    """
    orders = []
    mid = book.get_mid_price()

    if mid <= 0:
        return orders

    # Skip if we have active orders
    if state.active_orders:
        return orders

    spread = 0.001  # 0.1% spread
    size = 0.01

    # Buy order
    if state.position < 0.1:
        orders.append(Order(
            order_id="",
            symbol=book.symbol,
            side=Side.BUY,
            order_type="limit",
            amount=size,
            price=mid * (1 - spread/2),
            timestamp=book.timestamp,
        ))

    # Sell order
    if state.position > -0.1:
        orders.append(Order(
            order_id="",
            symbol=book.symbol,
            side=Side.SELL,
            order_type="limit",
            amount=size,
            price=mid * (1 + spread/2),
            timestamp=book.timestamp,
        ))

    return orders


def momentum_strategy(book: OrderBookSnapshot, state: BacktestState) -> List[Order]:
    """
    Simple momentum strategy using order book imbalance.
    """
    orders = []
    imbalance = book.get_imbalance()
    mid = book.get_mid_price()

    if mid <= 0:
        return orders

    # Strong buy imbalance
    if imbalance > 0.3 and state.position < 0.05:
        orders.append(Order(
            order_id="",
            symbol=book.symbol,
            side=Side.BUY,
            order_type="market",
            amount=0.01,
            price=mid,
            timestamp=book.timestamp,
        ))

    # Strong sell imbalance
    elif imbalance < -0.3 and state.position > -0.05:
        orders.append(Order(
            order_id="",
            symbol=book.symbol,
            side=Side.SELL,
            order_type="market",
            amount=0.01,
            price=mid,
            timestamp=book.timestamp,
        ))

    return orders


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("HFT Backtester Demo")
    print("=" * 50)

    from .loader import generate_synthetic_orderbook

    # Create synthetic order book data
    print("\nGenerating synthetic order book data...")
    loader = InMemoryOrderBookLoader()

    base_price = 42000.0
    for i in range(1000):
        # Simulate price movement
        price_change = np.random.normal(0, 10)
        base_price += price_change

        book = generate_synthetic_orderbook(
            "BTCUSDT",
            base_price,
            timestamp=1700000000 + i * 60,  # 1 minute intervals
        )
        loader.add_snapshot(book)

    # Configure backtest
    config = BacktestConfig(
        initial_capital=10000.0,
        exchange="binance",
        fee_rate=0.001,
        max_position=0.5,
        use_latency=True,
        verbose=False,
    )

    # Run backtest
    backtester = HFTBacktester(config, loader)
    backtester.set_strategy(momentum_strategy)

    print("\nRunning backtest...")
    result = backtester.run("BTCUSDT")

    # Print results
    print("\nResults:")
    print(f"  Total Return: {result.total_return*100:.2f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown*100:.2f}%")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate*100:.1f}%")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Total Fees: ${result.total_fees:.2f}")
    print(f"  Fill Rate: {result.fill_rate*100:.1f}%")
