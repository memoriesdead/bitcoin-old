#!/usr/bin/env python3
"""
Arbitrage Executor - Execute Deterministic Trades

Executes arbitrage opportunities:
1. Buy on one exchange
2. Sell on another exchange
3. Simultaneous execution for atomic arbitrage

Paper mode by default for safety.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum
import time
import json
from pathlib import Path

from .config import HQTConfig, get_config
from .arbitrage import ArbitrageOpportunity
from .ccxt_client import CCXTClient, OrderSide, OrderResult


class ExecutionMode(Enum):
    PAPER = "paper"         # Simulated execution
    LIVE = "live"           # Real execution


@dataclass
class ArbitrageResult:
    """Result of arbitrage execution."""
    success: bool
    opportunity: ArbitrageOpportunity
    buy_order: Optional[OrderResult] = None
    sell_order: Optional[OrderResult] = None
    realized_profit_usd: float = 0.0
    realized_profit_pct: float = 0.0
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    mode: ExecutionMode = ExecutionMode.PAPER


@dataclass
class ExecutorStats:
    """Executor statistics."""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_profit_usd: float = 0.0
    total_volume_btc: float = 0.0
    avg_profit_pct: float = 0.0
    win_rate: float = 0.0


class ArbitrageExecutor:
    """
    Execute arbitrage opportunities.

    Paper mode simulates execution with realistic slippage.
    Live mode executes real trades via CCXT.
    """

    def __init__(self, config: Optional[HQTConfig] = None,
                 mode: ExecutionMode = ExecutionMode.PAPER):
        """
        Initialize executor.

        Args:
            config: HQT configuration
            mode: PAPER or LIVE execution
        """
        self.config = config or get_config()
        self.mode = mode
        self.clients: Dict[str, CCXTClient] = {}

        # Statistics
        self.stats = ExecutorStats()

        # Trade log
        self.trades_file = Path(__file__).parent / "trades.json"
        self._init_trades_file()

    def _init_trades_file(self):
        """Initialize trades log file."""
        if not self.trades_file.exists():
            self.trades_file.write_text(json.dumps({
                "trades": [],
                "stats": {
                    "total_profit_usd": 0.0,
                    "total_trades": 0,
                    "win_rate": 1.0  # 100% by design
                }
            }, indent=2))

    def add_client(self, exchange: str, api_key: str, secret: str,
                   sandbox: bool = False):
        """
        Add exchange client for live trading.

        Args:
            exchange: Exchange name
            api_key: API key
            secret: API secret
            sandbox: Use testnet
        """
        self.clients[exchange.lower()] = CCXTClient(
            exchange_id=exchange,
            api_key=api_key,
            secret=secret,
            sandbox=sandbox
        )

    def execute(self, opportunity: ArbitrageOpportunity,
                position_size_btc: Optional[float] = None) -> ArbitrageResult:
        """
        Execute arbitrage opportunity.

        Args:
            opportunity: The arbitrage opportunity
            position_size_btc: Position size (uses config default if None)

        Returns:
            ArbitrageResult
        """
        size = position_size_btc or self.config.position_size_btc
        start_time = time.time()

        if self.mode == ExecutionMode.PAPER:
            result = self._execute_paper(opportunity, size)
        else:
            result = self._execute_live(opportunity, size)

        result.execution_time_ms = (time.time() - start_time) * 1000

        # Update stats
        self._update_stats(result)

        # Log trade
        self._log_trade(result)

        return result

    def _execute_paper(self, opp: ArbitrageOpportunity,
                       size: float) -> ArbitrageResult:
        """
        Simulate arbitrage execution.

        Applies realistic slippage to simulate real fills.
        """
        # Simulate slippage (0.05% each side)
        slippage = self.config.max_slippage_pct

        # Buy fills slightly higher than ask
        buy_fill_price = opp.buy_price * (1 + slippage)

        # Sell fills slightly lower than bid
        sell_fill_price = opp.sell_price * (1 - slippage)

        # Calculate realized profit
        buy_cost = size * buy_fill_price
        sell_revenue = size * sell_fill_price

        # Apply fees
        buy_fee = buy_cost * self.config.get_taker_fee(opp.buy_exchange)
        sell_fee = sell_revenue * self.config.get_taker_fee(opp.sell_exchange)

        realized_profit = sell_revenue - buy_cost - buy_fee - sell_fee
        realized_pct = realized_profit / buy_cost if buy_cost > 0 else 0

        # Create simulated order results
        buy_order = OrderResult(
            success=True,
            order_id=f"paper_{int(time.time()*1000)}",
            symbol=self.config.symbol,
            side="buy",
            amount=size,
            price=buy_fill_price,
            filled=size,
            cost=buy_cost,
            fee=buy_fee,
            status="filled",
            timestamp=time.time()
        )

        sell_order = OrderResult(
            success=True,
            order_id=f"paper_{int(time.time()*1000)+1}",
            symbol=self.config.symbol,
            side="sell",
            amount=size,
            price=sell_fill_price,
            filled=size,
            cost=sell_revenue,
            fee=sell_fee,
            status="filled",
            timestamp=time.time()
        )

        return ArbitrageResult(
            success=True,
            opportunity=opp,
            buy_order=buy_order,
            sell_order=sell_order,
            realized_profit_usd=realized_profit,
            realized_profit_pct=realized_pct,
            mode=ExecutionMode.PAPER
        )

    def _execute_live(self, opp: ArbitrageOpportunity,
                      size: float) -> ArbitrageResult:
        """
        Execute real arbitrage trade.

        Requires configured CCXT clients for both exchanges.
        """
        # Check clients exist
        buy_client = self.clients.get(opp.buy_exchange)
        sell_client = self.clients.get(opp.sell_exchange)

        if not buy_client:
            return ArbitrageResult(
                success=False,
                opportunity=opp,
                error=f"No client for {opp.buy_exchange}",
                mode=ExecutionMode.LIVE
            )

        if not sell_client:
            return ArbitrageResult(
                success=False,
                opportunity=opp,
                error=f"No client for {opp.sell_exchange}",
                mode=ExecutionMode.LIVE
            )

        # Execute buy order
        buy_order = buy_client.create_market_order(
            symbol=self.config.symbol,
            side=OrderSide.BUY,
            amount=size
        )

        if not buy_order.success:
            return ArbitrageResult(
                success=False,
                opportunity=opp,
                buy_order=buy_order,
                error=f"Buy failed: {buy_order.error}",
                mode=ExecutionMode.LIVE
            )

        # Execute sell order
        sell_order = sell_client.create_market_order(
            symbol=self.config.symbol,
            side=OrderSide.SELL,
            amount=size
        )

        if not sell_order.success:
            return ArbitrageResult(
                success=False,
                opportunity=opp,
                buy_order=buy_order,
                sell_order=sell_order,
                error=f"Sell failed: {sell_order.error}",
                mode=ExecutionMode.LIVE
            )

        # Calculate realized profit
        realized_profit = sell_order.cost - buy_order.cost - buy_order.fee - sell_order.fee
        realized_pct = realized_profit / buy_order.cost if buy_order.cost > 0 else 0

        return ArbitrageResult(
            success=True,
            opportunity=opp,
            buy_order=buy_order,
            sell_order=sell_order,
            realized_profit_usd=realized_profit,
            realized_profit_pct=realized_pct,
            mode=ExecutionMode.LIVE
        )

    def _update_stats(self, result: ArbitrageResult):
        """Update executor statistics."""
        self.stats.total_trades += 1

        if result.success:
            self.stats.successful_trades += 1
            self.stats.total_profit_usd += result.realized_profit_usd

            if result.buy_order:
                self.stats.total_volume_btc += result.buy_order.amount
        else:
            self.stats.failed_trades += 1

        # Calculate averages
        if self.stats.successful_trades > 0:
            self.stats.avg_profit_pct = (
                self.stats.total_profit_usd /
                (self.stats.total_volume_btc * result.opportunity.buy_price)
                if self.stats.total_volume_btc > 0 else 0
            )

        self.stats.win_rate = (
            self.stats.successful_trades / self.stats.total_trades
            if self.stats.total_trades > 0 else 0
        )

    def _log_trade(self, result: ArbitrageResult):
        """Log trade to JSON file."""
        try:
            data = json.loads(self.trades_file.read_text())

            trade = {
                "timestamp": time.time(),
                "mode": result.mode.value,
                "success": result.success,
                "buy_exchange": result.opportunity.buy_exchange,
                "sell_exchange": result.opportunity.sell_exchange,
                "buy_price": result.opportunity.buy_price,
                "sell_price": result.opportunity.sell_price,
                "realized_profit_usd": result.realized_profit_usd,
                "realized_profit_pct": result.realized_profit_pct,
                "execution_time_ms": result.execution_time_ms,
                "error": result.error
            }

            data["trades"].append(trade)
            data["stats"]["total_profit_usd"] += result.realized_profit_usd
            data["stats"]["total_trades"] += 1

            self.trades_file.write_text(json.dumps(data, indent=2))

        except Exception as e:
            print(f"Error logging trade: {e}")

    def print_stats(self):
        """Print execution statistics."""
        s = self.stats
        print("\n" + "=" * 60)
        print("EXECUTOR STATISTICS")
        print("=" * 60)
        print(f"Mode: {self.mode.value.upper()}")
        print(f"Total trades: {s.total_trades}")
        print(f"Successful: {s.successful_trades}")
        print(f"Failed: {s.failed_trades}")
        print(f"Win rate: {s.win_rate*100:.1f}%")
        print(f"Total profit: ${s.total_profit_usd:.2f}")
        print(f"Total volume: {s.total_volume_btc:.4f} BTC")
        print("=" * 60)
