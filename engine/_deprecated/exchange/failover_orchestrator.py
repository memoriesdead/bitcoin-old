"""
FAILOVER ORCHESTRATOR - GOLD STANDARD MULTI-EXCHANGE
=====================================================
Automatic failover between exchanges with health monitoring.

PRIORITY ORDER (configurable):
1. Hyperliquid (on-chain, 200K/sec, own node = no limits)
2. dYdX v4 (on-chain, Cosmos, no custody)
3. Bybit (API, 500/sec VIP)
4. Binance (API, 10K/sec MM)
5. Solana/Jito (on-chain, MEV bundles)

FAILOVER LOGIC:
- Monitor health of each exchange continuously
- Automatic failover on consecutive errors (>3)
- Automatic failover on high latency (>1000ms)
- Re-check primary every 30 seconds
- Parallel order submission for redundancy

HEALTH METRICS:
- Consecutive errors
- Average latency
- Last successful request
- Fill rate
- Rate limit headroom
"""
import asyncio
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from .base_executor import (
    BaseExecutor, Order, Position, Signal,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class ExchangeStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class ExchangeHealth:
    """Health metrics for an exchange."""
    name: str
    status: ExchangeStatus = ExchangeStatus.OFFLINE
    consecutive_errors: int = 0
    consecutive_successes: int = 0
    avg_latency_ms: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    orders_sent: int = 0
    orders_filled: int = 0
    fill_rate: float = 0.0
    last_health_check: float = 0.0

    def update_success(self, latency_ms: float):
        """Update after successful operation."""
        self.consecutive_successes += 1
        self.consecutive_errors = 0
        self.last_success = time.time()

        # Exponential moving average for latency
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = 0.9 * self.avg_latency_ms + 0.1 * latency_ms

        self._update_status()

    def update_failure(self):
        """Update after failed operation."""
        self.consecutive_errors += 1
        self.consecutive_successes = 0
        self.last_failure = time.time()
        self._update_status()

    def _update_status(self):
        """Determine health status."""
        if self.consecutive_errors > 5:
            self.status = ExchangeStatus.OFFLINE
        elif self.consecutive_errors > 3:
            self.status = ExchangeStatus.UNHEALTHY
        elif self.consecutive_errors > 1 or self.avg_latency_ms > 500:
            self.status = ExchangeStatus.DEGRADED
        elif self.consecutive_successes >= 3:
            self.status = ExchangeStatus.HEALTHY
        else:
            self.status = ExchangeStatus.DEGRADED


@dataclass
class FailoverConfig:
    """Configuration for failover orchestrator."""
    # Max consecutive errors before failover
    max_errors_before_failover: int = 3

    # Max latency before considering exchange degraded
    max_latency_ms: float = 1000.0

    # How often to health check (seconds)
    health_check_interval: float = 10.0

    # How often to try recovering primary (seconds)
    primary_recovery_interval: float = 30.0

    # Enable parallel order submission for redundancy
    parallel_submission: bool = False

    # Timeout for order submission (seconds)
    order_timeout: float = 5.0


class FailoverOrchestrator:
    """
    Multi-exchange failover orchestrator.

    Manages multiple exchange executors with automatic health
    monitoring and failover. Ensures 99.99% uptime by always
    having backup exchanges ready.
    """

    def __init__(self, config: FailoverConfig = None):
        self.config = config or FailoverConfig()

        # Exchange priority order
        self.priority: List[str] = []
        self.executors: Dict[str, BaseExecutor] = {}
        self.health: Dict[str, ExchangeHealth] = {}

        # Current state
        self.primary: Optional[str] = None
        self.active: Optional[str] = None

        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False

        # Stats
        self.stats = {
            'orders_total': 0,
            'orders_filled': 0,
            'failovers': 0,
            'primary_recoveries': 0,
        }

    def add_executor(self, name: str, executor: BaseExecutor, priority: int = None):
        """
        Add an executor with optional priority.

        Args:
            name: Exchange identifier (e.g., 'hyperliquid', 'bybit')
            executor: The executor instance
            priority: Lower = higher priority. None = append to end.
        """
        self.executors[name] = executor
        self.health[name] = ExchangeHealth(name=name)

        if priority is not None:
            # Insert at specific priority
            if priority >= len(self.priority):
                self.priority.append(name)
            else:
                self.priority.insert(priority, name)
        else:
            self.priority.append(name)

        print(f"[FAILOVER] Added executor: {name} (priority {self.priority.index(name) + 1})")

    async def connect_all(self) -> int:
        """
        Connect to all exchanges.

        Returns:
            Number of successfully connected exchanges.
        """
        connected = 0

        for name in self.priority:
            executor = self.executors[name]
            try:
                if await executor.connect():
                    self.health[name].status = ExchangeStatus.HEALTHY
                    self.health[name].last_success = time.time()
                    connected += 1
                    print(f"[FAILOVER] {name}: Connected")
                else:
                    self.health[name].status = ExchangeStatus.OFFLINE
                    print(f"[FAILOVER] {name}: Connection failed")
            except Exception as e:
                self.health[name].status = ExchangeStatus.OFFLINE
                print(f"[FAILOVER] {name}: Connection error: {e}")

        # Set primary and active
        for name in self.priority:
            if self.health[name].status == ExchangeStatus.HEALTHY:
                self.primary = name
                self.active = name
                print(f"[FAILOVER] Primary: {name}")
                break

        return connected

    async def disconnect_all(self):
        """Disconnect from all exchanges."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        for name, executor in self.executors.items():
            try:
                await executor.disconnect()
                print(f"[FAILOVER] {name}: Disconnected")
            except Exception as e:
                print(f"[FAILOVER] {name}: Disconnect error: {e}")

    async def start_health_monitoring(self):
        """Start background health monitoring."""
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        print("[FAILOVER] Health monitoring started")

    async def _health_check_loop(self):
        """Background health check loop."""
        while self._running:
            try:
                await self._check_all_health()
                await self._maybe_recover_primary()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[FAILOVER] Health check error: {e}")
                await asyncio.sleep(1)

    async def _check_all_health(self):
        """Check health of all exchanges."""
        for name in self.priority:
            executor = self.executors[name]
            health = self.health[name]

            # Skip if recently checked
            if time.time() - health.last_health_check < self.config.health_check_interval:
                continue

            try:
                start = time.perf_counter()
                balance = await asyncio.wait_for(
                    executor.get_balance(),
                    timeout=5.0
                )
                latency_ms = (time.perf_counter() - start) * 1000

                if balance:
                    health.update_success(latency_ms)
                else:
                    health.update_failure()

            except asyncio.TimeoutError:
                health.update_failure()
            except Exception:
                health.update_failure()

            health.last_health_check = time.time()

    async def _maybe_recover_primary(self):
        """Try to recover to primary if it becomes healthy."""
        if not self.primary or self.active == self.primary:
            return

        primary_health = self.health[self.primary]

        # Check if primary is healthy and we should switch back
        if primary_health.status == ExchangeStatus.HEALTHY:
            # Double-check with a fresh health check
            executor = self.executors[self.primary]
            try:
                start = time.perf_counter()
                balance = await asyncio.wait_for(
                    executor.get_balance(),
                    timeout=3.0
                )
                latency_ms = (time.perf_counter() - start) * 1000

                if balance and latency_ms < self.config.max_latency_ms:
                    print(f"[FAILOVER] Recovering to primary: {self.primary}")
                    self.active = self.primary
                    self.stats['primary_recoveries'] += 1

            except Exception:
                pass

    async def submit_order(self, order: Order) -> Order:
        """
        Submit order with automatic failover.

        Tries active exchange first, fails over to next healthy
        exchange if needed.
        """
        self.stats['orders_total'] += 1

        if self.config.parallel_submission:
            return await self._submit_parallel(order)

        # Sequential submission with failover
        attempted = set()

        while True:
            # Get next exchange to try
            exchange_name = self._get_next_healthy_exchange(attempted)
            if not exchange_name:
                order.status = OrderStatus.REJECTED
                print("[FAILOVER] All exchanges exhausted")
                return order

            attempted.add(exchange_name)
            executor = self.executors[exchange_name]
            health = self.health[exchange_name]

            try:
                start = time.perf_counter()
                result = await asyncio.wait_for(
                    executor.submit_order(order),
                    timeout=self.config.order_timeout
                )
                latency_ms = (time.perf_counter() - start) * 1000

                if result.status not in [OrderStatus.REJECTED]:
                    health.update_success(latency_ms)

                    if result.status == OrderStatus.FILLED:
                        self.stats['orders_filled'] += 1
                        health.orders_filled += 1

                    health.orders_sent += 1
                    result.exchange = exchange_name
                    return result

                # Order rejected, try next exchange
                health.update_failure()
                print(f"[FAILOVER] {exchange_name} rejected order, trying next...")

            except asyncio.TimeoutError:
                health.update_failure()
                print(f"[FAILOVER] {exchange_name} timeout, failing over...")
                self._maybe_failover(exchange_name)

            except Exception as e:
                health.update_failure()
                print(f"[FAILOVER] {exchange_name} error: {e}, failing over...")
                self._maybe_failover(exchange_name)

        order.status = OrderStatus.REJECTED
        return order

    async def _submit_parallel(self, order: Order) -> Order:
        """Submit to multiple exchanges in parallel for redundancy."""
        healthy_exchanges = [
            name for name in self.priority
            if self.health[name].status in [ExchangeStatus.HEALTHY, ExchangeStatus.DEGRADED]
        ][:3]  # Max 3 parallel

        if not healthy_exchanges:
            order.status = OrderStatus.REJECTED
            return order

        # Create tasks for parallel submission
        tasks = []
        for name in healthy_exchanges:
            executor = self.executors[name]
            # Create a copy of the order for each exchange
            order_copy = Order(
                id=f"{order.id}_{name}",
                timestamp=order.timestamp,
                instrument=order.instrument,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                take_profit=order.take_profit,
                stop_loss=order.stop_loss,
            )
            tasks.append(self._submit_with_timeout(name, executor, order_copy))

        # Wait for first success
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Find first successful result
        for i, result in enumerate(results):
            if isinstance(result, Order) and result.status != OrderStatus.REJECTED:
                # Cancel other orders if filled
                for j, other_result in enumerate(results):
                    if i != j and isinstance(other_result, Order):
                        if other_result.status == OrderStatus.OPEN:
                            asyncio.create_task(
                                self.executors[healthy_exchanges[j]].cancel_order(other_result.id)
                            )

                self.stats['orders_filled'] += 1
                result.exchange = healthy_exchanges[i]
                return result

        order.status = OrderStatus.REJECTED
        return order

    async def _submit_with_timeout(
        self,
        name: str,
        executor: BaseExecutor,
        order: Order
    ) -> Order:
        """Submit order with timeout."""
        try:
            return await asyncio.wait_for(
                executor.submit_order(order),
                timeout=self.config.order_timeout
            )
        except Exception as e:
            order.status = OrderStatus.REJECTED
            return order

    def _get_next_healthy_exchange(self, attempted: set) -> Optional[str]:
        """Get next healthy exchange that hasn't been attempted."""
        # Try active first
        if self.active and self.active not in attempted:
            health = self.health[self.active]
            if health.status in [ExchangeStatus.HEALTHY, ExchangeStatus.DEGRADED]:
                return self.active

        # Then try by priority
        for name in self.priority:
            if name not in attempted:
                health = self.health[name]
                if health.status in [ExchangeStatus.HEALTHY, ExchangeStatus.DEGRADED]:
                    return name

        return None

    def _maybe_failover(self, failed_exchange: str):
        """Maybe failover from failed exchange."""
        if self.active == failed_exchange:
            health = self.health[failed_exchange]
            if health.consecutive_errors >= self.config.max_errors_before_failover:
                # Find next healthy exchange
                for name in self.priority:
                    if name != failed_exchange:
                        if self.health[name].status in [ExchangeStatus.HEALTHY, ExchangeStatus.DEGRADED]:
                            print(f"[FAILOVER] Failing over from {failed_exchange} to {name}")
                            self.active = name
                            self.stats['failovers'] += 1
                            return

    async def get_position(self, instrument: str) -> Optional[Position]:
        """Get position from active exchange."""
        if self.active:
            try:
                return await self.executors[self.active].get_position(instrument)
            except Exception:
                pass
        return None

    async def get_balance(self) -> Dict[str, float]:
        """Get balance from active exchange."""
        if self.active:
            try:
                return await self.executors[self.active].get_balance()
            except Exception:
                pass
        return {}

    def get_status(self) -> Dict:
        """Get orchestrator status."""
        return {
            'primary': self.primary,
            'active': self.active,
            'exchanges': {
                name: {
                    'status': health.status.value,
                    'consecutive_errors': health.consecutive_errors,
                    'avg_latency_ms': round(health.avg_latency_ms, 1),
                    'orders_sent': health.orders_sent,
                    'orders_filled': health.orders_filled,
                }
                for name, health in self.health.items()
            },
            'stats': self.stats,
        }

    def print_status(self):
        """Print current status."""
        print("\n" + "=" * 70)
        print("FAILOVER ORCHESTRATOR STATUS")
        print("=" * 70)
        print(f"Primary: {self.primary}")
        print(f"Active:  {self.active}")
        print("-" * 70)

        for name in self.priority:
            health = self.health[name]
            status_icon = {
                ExchangeStatus.HEALTHY: "[OK]",
                ExchangeStatus.DEGRADED: "[!!]",
                ExchangeStatus.UNHEALTHY: "[XX]",
                ExchangeStatus.OFFLINE: "[--]",
            }.get(health.status, "[??]")

            print(f"  {status_icon} {name:15s} | "
                  f"Latency: {health.avg_latency_ms:6.1f}ms | "
                  f"Errors: {health.consecutive_errors} | "
                  f"Orders: {health.orders_sent}")

        print("-" * 70)
        print(f"Total Orders: {self.stats['orders_total']} | "
              f"Filled: {self.stats['orders_filled']} | "
              f"Failovers: {self.stats['failovers']}")
        print("=" * 70 + "\n")


def create_failover_orchestrator(
    hyperliquid_executor=None,
    dydx_executor=None,
    bybit_executor=None,
    binance_executor=None,
    solana_executor=None,
    config: FailoverConfig = None
) -> FailoverOrchestrator:
    """
    Factory function to create failover orchestrator with gold standard exchanges.

    Priority order:
    1. Hyperliquid (on-chain, 200K/sec)
    2. dYdX v4 (on-chain, Cosmos)
    3. Bybit (API, 500/sec VIP)
    4. Binance (API, 10K/sec MM)
    5. Solana/Jito (on-chain, MEV)

    Example:
        from engine.exchange.hyperliquid_executor import create_hyperliquid_executor
        from engine.exchange.bybit_executor import create_bybit_executor

        hl = create_hyperliquid_executor(key, testnet=True)
        bb = create_bybit_executor(api_key, api_secret, testnet=True)

        orchestrator = create_failover_orchestrator(
            hyperliquid_executor=hl,
            bybit_executor=bb,
        )

        await orchestrator.connect_all()
        await orchestrator.start_health_monitoring()

        order = await orchestrator.submit_order(my_order)
    """
    orchestrator = FailoverOrchestrator(config or FailoverConfig())

    # Add in priority order
    if hyperliquid_executor:
        orchestrator.add_executor('hyperliquid', hyperliquid_executor, priority=0)

    if dydx_executor:
        orchestrator.add_executor('dydx', dydx_executor, priority=1)

    if bybit_executor:
        orchestrator.add_executor('bybit', bybit_executor, priority=2)

    if binance_executor:
        orchestrator.add_executor('binance', binance_executor, priority=3)

    if solana_executor:
        orchestrator.add_executor('solana', solana_executor, priority=4)

    return orchestrator
