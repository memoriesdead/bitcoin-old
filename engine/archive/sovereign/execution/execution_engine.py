"""
Execution Engine
================

Main execution engine with multiple modes.

Modes:
- PAPER: Simulated execution (current behavior)
- DRY_RUN: Real price, simulated fills (Freqtrade pattern)
- LIVE: Real execution via CCXT

Freqtrade pattern: Separate strategy from execution.
"""

import time
import uuid
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from .order_manager import OrderManager, OrderTransition, ManagedOrder
from .ccxt_client import CCXTClient, ExchangeConfig, OrderSide
from .dry_run import DryRunExecutor
from .safety import SafetyManager, SafetyConfig


class ExecutionMode(Enum):
    """Execution modes."""
    PAPER = "paper"       # Full simulation
    DRY_RUN = "dry_run"   # Real prices, simulated fills
    LIVE = "live"         # Real execution


@dataclass
class ExecutionResult:
    """Result of execution attempt."""
    success: bool
    order_id: Optional[str] = None
    executed_price: float = 0.0
    executed_amount: float = 0.0
    remaining_amount: float = 0.0
    fee: float = 0.0
    slippage: float = 0.0
    mode: ExecutionMode = ExecutionMode.PAPER
    timestamp: float = 0.0
    error: Optional[str] = None
    details: Optional[Dict] = None


class ExecutionEngine:
    """
    Main execution engine.

    Handles order execution in multiple modes with safety checks.
    """

    def __init__(self,
                 mode: ExecutionMode = ExecutionMode.PAPER,
                 exchange_config: Optional[ExchangeConfig] = None,
                 safety_config: Optional[SafetyConfig] = None):
        """
        Initialize execution engine.

        Args:
            mode: Execution mode
            exchange_config: Exchange configuration (for LIVE mode)
            safety_config: Safety parameters
        """
        self.mode = mode
        self.exchange_config = exchange_config

        # Order management
        self.order_manager = OrderManager()

        # Safety
        self.safety = SafetyManager(safety_config or SafetyConfig())

        # Exchange client (for LIVE mode)
        self.exchange: Optional[CCXTClient] = None
        if mode == ExecutionMode.LIVE and exchange_config:
            self.exchange = CCXTClient(exchange_config)

        # Dry run executor
        self.dry_run = DryRunExecutor()

        # Current price (for paper/dry_run)
        self._current_price: Dict[str, float] = {}

        # Stats
        self.stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_volume': 0.0,
            'total_fees': 0.0,
        }

        # Callbacks
        self.on_execution: Optional[Callable[[ExecutionResult], None]] = None

    def set_price(self, symbol: str, price: float):
        """
        Set current price for paper/dry_run modes.

        Args:
            symbol: Trading pair
            price: Current price
        """
        self._current_price[symbol] = price

    def execute(self, symbol: str, side: str, amount: float,
                price: Optional[float] = None,
                order_type: str = "market",
                trade_id: Optional[str] = None) -> ExecutionResult:
        """
        Execute an order.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            side: "buy" or "sell"
            amount: Order amount
            price: Limit price (None for market)
            order_type: "market" or "limit"
            trade_id: Associated trade ID

        Returns:
            ExecutionResult
        """
        self.stats['total_orders'] += 1

        # Safety checks
        safety_result = self.safety.check_order(
            symbol=symbol,
            side=side,
            amount=amount,
            price=price or self._current_price.get(symbol, 0),
        )

        if not safety_result['allowed']:
            return ExecutionResult(
                success=False,
                error=f"Safety check failed: {safety_result['reason']}",
                mode=self.mode,
                timestamp=time.time(),
            )

        # Create managed order
        client_order_id = f"EXE_{uuid.uuid4().hex[:8]}"
        order = self.order_manager.create_order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
            trade_id=trade_id,
        )

        # Execute based on mode
        if self.mode == ExecutionMode.PAPER:
            result = self._execute_paper(order)
        elif self.mode == ExecutionMode.DRY_RUN:
            result = self._execute_dry_run(order)
        else:  # LIVE
            result = self._execute_live(order)

        # Update stats
        if result.success:
            self.stats['successful_orders'] += 1
            self.stats['total_volume'] += result.executed_amount * result.executed_price
            self.stats['total_fees'] += result.fee

            # Update safety manager
            self.safety.record_trade(
                pnl=0,  # Unknown until closed
                is_win=True,  # Placeholder
            )
        else:
            self.stats['failed_orders'] += 1

        # Fire callback
        if self.on_execution:
            self.on_execution(result)

        return result

    def _execute_paper(self, order: ManagedOrder) -> ExecutionResult:
        """Execute in paper mode (full simulation)."""
        # Get current price
        price = self._current_price.get(order.symbol, order.price or 0)

        if price <= 0:
            return ExecutionResult(
                success=False,
                error="No price available",
                mode=ExecutionMode.PAPER,
                timestamp=time.time(),
            )

        # Simulate slippage
        if order.order_type == "market":
            slippage_pct = 0.0001  # 0.01% slippage
            if order.side == "buy":
                price *= (1 + slippage_pct)
            else:
                price *= (1 - slippage_pct)

        # Simulate fee
        fee_rate = 0.001  # 0.1%
        fee = order.amount * price * fee_rate

        # Update order state
        self.order_manager.transition(
            order.client_order_id,
            OrderTransition.SUBMIT,
        )
        self.order_manager.transition(
            order.client_order_id,
            OrderTransition.FILL,
            fill_amount=order.amount,
            fill_price=price,
            fee=fee,
        )

        return ExecutionResult(
            success=True,
            order_id=order.client_order_id,
            executed_price=price,
            executed_amount=order.amount,
            remaining_amount=0,
            fee=fee,
            slippage=abs(price - (order.price or price)) / price if order.price else 0,
            mode=ExecutionMode.PAPER,
            timestamp=time.time(),
        )

    def _execute_dry_run(self, order: ManagedOrder) -> ExecutionResult:
        """Execute in dry-run mode (real price, simulated fill)."""
        # Get real price from exchange or cache
        price = self._current_price.get(order.symbol, order.price or 0)

        # Use dry run executor
        fill = self.dry_run.simulate_fill(
            symbol=order.symbol,
            side=order.side,
            amount=order.amount,
            price=price,
            order_type=order.order_type,
        )

        # Update order state
        self.order_manager.transition(
            order.client_order_id,
            OrderTransition.SUBMIT,
        )
        self.order_manager.transition(
            order.client_order_id,
            OrderTransition.FILL,
            fill_amount=fill.amount,
            fill_price=fill.price,
            fee=fill.fee,
        )

        return ExecutionResult(
            success=True,
            order_id=order.client_order_id,
            executed_price=fill.price,
            executed_amount=fill.amount,
            remaining_amount=order.amount - fill.amount,
            fee=fill.fee,
            slippage=fill.slippage,
            mode=ExecutionMode.DRY_RUN,
            timestamp=time.time(),
        )

    def _execute_live(self, order: ManagedOrder) -> ExecutionResult:
        """Execute in live mode (real exchange)."""
        if self.exchange is None:
            return ExecutionResult(
                success=False,
                error="Exchange not configured",
                mode=ExecutionMode.LIVE,
                timestamp=time.time(),
            )

        try:
            # Submit order
            self.order_manager.transition(
                order.client_order_id,
                OrderTransition.SUBMIT,
            )

            side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL

            if order.order_type == "market":
                result = self.exchange.create_market_order(
                    symbol=order.symbol,
                    side=side,
                    amount=order.amount,
                )
            else:
                result = self.exchange.create_limit_order(
                    symbol=order.symbol,
                    side=side,
                    amount=order.amount,
                    price=order.price,
                )

            if not result.success:
                self.order_manager.transition(
                    order.client_order_id,
                    OrderTransition.REJECT,
                    error=result.error,
                )
                return ExecutionResult(
                    success=False,
                    error=result.error,
                    mode=ExecutionMode.LIVE,
                    timestamp=time.time(),
                )

            # Update order with exchange ID
            self.order_manager.transition(
                order.client_order_id,
                OrderTransition.CONFIRM,
                exchange_order_id=result.order_id,
            )

            # Handle fill
            if result.filled > 0:
                if result.remaining == 0:
                    self.order_manager.transition(
                        order.client_order_id,
                        OrderTransition.FILL,
                        fill_amount=result.filled,
                        fill_price=result.average,
                        fee=result.fee.get('cost', 0) if result.fee else 0,
                    )
                else:
                    self.order_manager.transition(
                        order.client_order_id,
                        OrderTransition.PARTIAL_FILL,
                        fill_amount=result.filled,
                        fill_price=result.average,
                    )

            return ExecutionResult(
                success=True,
                order_id=result.order_id,
                executed_price=result.average,
                executed_amount=result.filled,
                remaining_amount=result.remaining,
                fee=result.fee.get('cost', 0) if result.fee else 0,
                slippage=0,  # TODO: Calculate
                mode=ExecutionMode.LIVE,
                timestamp=time.time(),
                details=result.raw,
            )

        except Exception as e:
            self.order_manager.transition(
                order.client_order_id,
                OrderTransition.ERROR,
                error=str(e),
            )
            return ExecutionResult(
                success=False,
                error=str(e),
                mode=ExecutionMode.LIVE,
                timestamp=time.time(),
            )

    def cancel_order(self, client_order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            client_order_id: Order to cancel

        Returns:
            True if cancelled
        """
        order = self.order_manager.get_order(client_order_id)
        if order is None:
            return False

        # Start cancellation
        self.order_manager.transition(
            client_order_id,
            OrderTransition.CANCEL,
        )

        if self.mode == ExecutionMode.LIVE and self.exchange and order.exchange_order_id:
            success = self.exchange.cancel_order(
                order.exchange_order_id,
                order.symbol,
            )
            if success:
                self.order_manager.transition(
                    client_order_id,
                    OrderTransition.CANCEL_CONFIRM,
                )
            return success
        else:
            # Paper/dry-run: instant cancel
            self.order_manager.transition(
                client_order_id,
                OrderTransition.CANCEL_CONFIRM,
            )
            return True

    def get_active_orders(self):
        """Get all active orders."""
        return self.order_manager.get_active_orders()

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            **self.stats,
            'mode': self.mode.value,
            'safety_stats': self.safety.get_stats(),
            'order_stats': self.order_manager.get_stats(),
        }

    def close(self):
        """Clean up resources."""
        if self.exchange:
            self.exchange.close()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("Execution Engine Demo")
    print("=" * 50)

    # Paper mode
    engine = ExecutionEngine(mode=ExecutionMode.PAPER)

    # Set price
    engine.set_price("BTC/USDT", 42000.0)

    # Execute market buy
    print("\nMarket Buy:")
    result = engine.execute(
        symbol="BTC/USDT",
        side="buy",
        amount=0.1,
        order_type="market",
    )
    print(f"  Success: {result.success}")
    print(f"  Price: {result.executed_price:.2f}")
    print(f"  Amount: {result.executed_amount}")
    print(f"  Fee: {result.fee:.4f}")

    # Execute limit sell
    print("\nLimit Sell:")
    engine.set_price("BTC/USDT", 42500.0)
    result = engine.execute(
        symbol="BTC/USDT",
        side="sell",
        amount=0.1,
        price=42500.0,
        order_type="limit",
    )
    print(f"  Success: {result.success}")
    print(f"  Price: {result.executed_price:.2f}")

    print(f"\nStats: {engine.get_stats()}")
