"""
Execution Router - Sovereign Engine
====================================

Routes trades to correct executor based on mode.

Modes:
- PAPER: Simulated fills with configurable slippage
- DRY_RUN: Real prices, simulated fills
- LIVE: Real CCXT orders
- ONCHAIN: On-chain DEX execution (Solana/EVM)
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import asyncio

from ..core.types import (
    ExecutionMode, Order, SizedOrder, ExecutionResult, OrderSide, OrderType
)
from ..core.config import ExecutionConfig, SafetyConfig


# =============================================================================
# BASE EXECUTOR
# =============================================================================

class BaseExecutor(ABC):
    """Abstract base class for trade executors."""

    def __init__(self, name: str):
        self.name = name
        self.orders_executed = 0
        self.orders_failed = 0
        self.total_fees = 0.0
        self.total_slippage = 0.0

    @abstractmethod
    def execute(self, order: SizedOrder) -> ExecutionResult:
        """Execute an order."""
        pass

    @abstractmethod
    def cancel(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "name": self.name,
            "orders_executed": self.orders_executed,
            "orders_failed": self.orders_failed,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
        }


# =============================================================================
# PAPER EXECUTOR
# =============================================================================

class PaperExecutor(BaseExecutor):
    """
    Paper trading executor.

    Simulates fills with configurable slippage and fees.
    """

    def __init__(
        self,
        slippage_pct: float = 0.0001,  # 0.01%
        fee_pct: float = 0.001,         # 0.1%
        fill_delay_ms: float = 50.0
    ):
        super().__init__("paper")
        self.slippage_pct = slippage_pct
        self.fee_pct = fee_pct
        self.fill_delay_ms = fill_delay_ms

        # Track positions
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.order_counter = 0

    def execute(self, order: SizedOrder) -> ExecutionResult:
        """Execute paper trade with simulated fill."""
        self.order_counter += 1
        order_id = f"PAPER-{self.order_counter:06d}"

        # Get base price
        base_price = order.order.price or 0.0
        if base_price == 0:
            # Use signal price if available
            if order.order.signal and order.order.signal.price_at_signal > 0:
                base_price = order.order.signal.price_at_signal
            else:
                base_price = 100000.0  # Default BTC price

        # Apply slippage
        slippage_direction = 1 if order.side == OrderSide.BUY else -1
        slippage = base_price * self.slippage_pct * slippage_direction
        fill_price = base_price + slippage

        # Calculate fee
        fee = order.amount * fill_price * self.fee_pct

        # Simulate fill delay
        fill_time = time.time() + (self.fill_delay_ms / 1000)

        self.orders_executed += 1
        self.total_fees += fee
        self.total_slippage += abs(slippage) * order.amount

        return ExecutionResult(
            success=True,
            order=order.order,
            fill_price=fill_price,
            fill_amount=order.amount,
            fill_time=fill_time,
            fee=fee,
            slippage=slippage,
            order_id=order_id,
        )

    def cancel(self, order_id: str) -> bool:
        """Paper orders are instant fills, nothing to cancel."""
        return True


# =============================================================================
# DRY RUN EXECUTOR
# =============================================================================

class DryRunExecutor(BaseExecutor):
    """
    Dry run executor.

    Uses real prices but simulates fills with depth walking.
    """

    def __init__(
        self,
        fee_pct: float = 0.001,
        depth_impact_pct: float = 0.0005  # Price impact per 1 BTC
    ):
        super().__init__("dry_run")
        self.fee_pct = fee_pct
        self.depth_impact_pct = depth_impact_pct
        self.order_counter = 0
        self.current_prices: Dict[str, float] = {}

    def set_current_price(self, symbol: str, price: float):
        """Update current price for a symbol."""
        self.current_prices[symbol] = price

    def execute(self, order: SizedOrder) -> ExecutionResult:
        """Execute with depth-based slippage."""
        self.order_counter += 1
        order_id = f"DRY-{self.order_counter:06d}"

        # Get current price
        base_price = self.current_prices.get(order.symbol, 0.0)
        if base_price == 0:
            if order.order.signal and order.order.signal.price_at_signal > 0:
                base_price = order.order.signal.price_at_signal
            else:
                base_price = 100000.0

        # Depth-based slippage (larger orders = more slippage)
        size_impact = order.amount * self.depth_impact_pct
        slippage_direction = 1 if order.side == OrderSide.BUY else -1
        slippage = base_price * size_impact * slippage_direction
        fill_price = base_price + slippage

        # Fee
        fee = order.amount * fill_price * self.fee_pct

        self.orders_executed += 1
        self.total_fees += fee
        self.total_slippage += abs(slippage) * order.amount

        return ExecutionResult(
            success=True,
            order=order.order,
            fill_price=fill_price,
            fill_amount=order.amount,
            fill_time=time.time(),
            fee=fee,
            slippage=slippage,
            order_id=order_id,
        )

    def cancel(self, order_id: str) -> bool:
        return True


# =============================================================================
# CCXT EXECUTOR (LIVE)
# =============================================================================

class CCXTExecutor(BaseExecutor):
    """
    Live trading via CCXT.

    Executes real orders on exchanges.
    """

    def __init__(self, config: ExecutionConfig):
        super().__init__("live")
        self.config = config
        self.exchange = None
        self._initialized = False

    def initialize(self):
        """Initialize CCXT exchange connection."""
        try:
            import ccxt

            exchange_class = getattr(ccxt, self.config.default_exchange)
            self.exchange = exchange_class({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True,
            })

            # Load markets
            self.exchange.load_markets()
            self._initialized = True
            print(f"[CCXT] Initialized {self.config.default_exchange}")

        except Exception as e:
            print(f"[CCXT] Initialization failed: {e}")
            self._initialized = False

    def execute(self, order: SizedOrder) -> ExecutionResult:
        """Execute real order via CCXT."""
        if not self._initialized:
            self.initialize()

        if not self._initialized or not self.exchange:
            return ExecutionResult(
                success=False,
                order=order.order,
                rejected=True,
                reject_reason="Exchange not initialized",
            )

        try:
            side = "buy" if order.side == OrderSide.BUY else "sell"
            order_type = order.order.order_type.value

            if order_type == "market":
                result = self.exchange.create_market_order(
                    order.symbol,
                    side,
                    order.amount,
                )
            else:
                result = self.exchange.create_limit_order(
                    order.symbol,
                    side,
                    order.amount,
                    order.order.price,
                )

            self.orders_executed += 1

            return ExecutionResult(
                success=True,
                order=order.order,
                fill_price=result.get('average', result.get('price', 0)),
                fill_amount=result.get('filled', order.amount),
                fill_time=time.time(),
                fee=result.get('fee', {}).get('cost', 0),
                order_id=result.get('id'),
            )

        except Exception as e:
            self.orders_failed += 1
            return ExecutionResult(
                success=False,
                order=order.order,
                rejected=True,
                reject_reason=str(e),
            )

    def cancel(self, order_id: str) -> bool:
        """Cancel order via CCXT."""
        if not self.exchange:
            return False

        try:
            self.exchange.cancel_order(order_id)
            return True
        except Exception as e:
            print(f"[CCXT] Cancel failed: {e}")
            return False


# =============================================================================
# SAFETY MANAGER
# =============================================================================

class SafetyManager:
    """
    Risk management and safety checks.

    Applied to ALL execution modes.
    """

    def __init__(self, config: SafetyConfig):
        self.config = config

        # State
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.kill_switch_active = False

        # Positions
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.total_exposure = 0.0

    def check(self, order: SizedOrder) -> tuple:
        """
        Check if order passes safety checks.

        Returns:
            (allowed: bool, reason: str)
        """
        # Kill switch
        if self.kill_switch_active:
            return False, "Kill switch active"

        # Daily trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            return False, f"Daily trade limit ({self.config.max_daily_trades})"

        # Position size
        if order.position_size_usd > self.config.max_position_usd:
            return False, f"Position too large (${order.position_size_usd:.0f} > ${self.config.max_position_usd:.0f})"

        # Total exposure
        new_exposure = self.total_exposure + order.position_size_usd
        if new_exposure > self.config.max_exposure_usd:
            return False, f"Exposure limit (${new_exposure:.0f} > ${self.config.max_exposure_usd:.0f})"

        # Daily loss
        if self.daily_pnl < 0 and abs(self.daily_pnl) > self.config.max_daily_loss_usd:
            return False, f"Daily loss limit (${abs(self.daily_pnl):.0f})"

        # Consecutive losses
        if self.consecutive_losses >= self.config.consecutive_loss_limit:
            return False, f"Consecutive loss limit ({self.consecutive_losses})"

        # Drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            if drawdown > self.config.max_drawdown_pct:
                return False, f"Drawdown limit ({drawdown*100:.1f}%)"

        return True, "OK"

    def record_trade(self, result: ExecutionResult):
        """Record a trade result."""
        self.daily_trades += 1

        # Update exposure
        if result.success:
            self.total_exposure += result.fill_amount * result.fill_price

    def record_close(self, pnl: float):
        """Record a position close."""
        self.daily_pnl += pnl

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Update equity
        self.current_equity += pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        # Check kill switch
        if self.config.kill_switch_enabled:
            if self.peak_equity > 0:
                drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
                if drawdown > self.config.kill_switch_loss_pct:
                    self.kill_switch_active = True
                    print(f"[SAFETY] KILL SWITCH ACTIVATED - Drawdown {drawdown*100:.1f}%")

    def reset_daily(self):
        """Reset daily counters (call at start of new day)."""
        self.daily_pnl = 0.0
        self.daily_trades = 0

    def reset_kill_switch(self):
        """Manually reset kill switch."""
        self.kill_switch_active = False
        print("[SAFETY] Kill switch reset")

    def set_equity(self, equity: float):
        """Set current equity level."""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity


# =============================================================================
# EXECUTION ROUTER
# =============================================================================

class ExecutionRouter:
    """
    Routes trades to correct executor based on mode.

    Central point for all trade execution with safety checks.
    """

    def __init__(
        self,
        config: ExecutionConfig,
        safety_config: SafetyConfig = None
    ):
        self.config = config
        self.mode = config.mode
        self.safety = SafetyManager(safety_config or SafetyConfig())

        # Initialize executors
        self.executors: Dict[ExecutionMode, BaseExecutor] = {
            ExecutionMode.PAPER: PaperExecutor(
                slippage_pct=config.paper_slippage_pct,
                fee_pct=config.paper_fee_pct,
            ),
            ExecutionMode.DRY_RUN: DryRunExecutor(
                fee_pct=config.paper_fee_pct,
            ),
            ExecutionMode.LIVE: CCXTExecutor(config),
            ExecutionMode.BACKTEST: PaperExecutor(slippage_pct=0, fee_pct=0.001),
        }

        # On-chain executor (lazy load)
        self._onchain_executor = None

    @property
    def current_executor(self) -> BaseExecutor:
        """Get executor for current mode."""
        return self.executors.get(self.mode) or self.executors[ExecutionMode.PAPER]

    def execute(self, order: SizedOrder) -> ExecutionResult:
        """
        Execute an order with safety checks.

        Args:
            order: Sized order to execute

        Returns:
            ExecutionResult
        """
        # Safety check (ALL modes)
        allowed, reason = self.safety.check(order)
        if not allowed:
            return ExecutionResult(
                success=False,
                order=order.order,
                rejected=True,
                reject_reason=reason,
            )

        # Route to executor
        executor = self.current_executor
        result = executor.execute(order)

        # Record in safety manager
        if result.success:
            self.safety.record_trade(result)

        return result

    def cancel(self, order_id: str) -> bool:
        """Cancel an order."""
        return self.current_executor.cancel(order_id)

    def set_mode(self, mode: ExecutionMode):
        """Change execution mode."""
        self.mode = mode

    def set_current_price(self, symbol: str, price: float):
        """Update current price for dry run executor."""
        if isinstance(self.executors.get(ExecutionMode.DRY_RUN), DryRunExecutor):
            self.executors[ExecutionMode.DRY_RUN].set_current_price(symbol, price)

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "mode": self.mode.value,
            "executor": self.current_executor.get_stats(),
            "safety": {
                "daily_trades": self.safety.daily_trades,
                "daily_pnl": self.safety.daily_pnl,
                "consecutive_losses": self.safety.consecutive_losses,
                "kill_switch": self.safety.kill_switch_active,
            }
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_router(
    mode: ExecutionMode = ExecutionMode.PAPER,
    config: ExecutionConfig = None,
    safety: SafetyConfig = None
) -> ExecutionRouter:
    """
    Create an execution router.

    Args:
        mode: Execution mode
        config: Execution configuration
        safety: Safety configuration

    Returns:
        Configured ExecutionRouter
    """
    if config is None:
        config = ExecutionConfig()
    config.mode = mode

    return ExecutionRouter(config, safety)
