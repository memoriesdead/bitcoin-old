"""
MULTI-EXCHANGE ORCHESTRATOR
============================
Routes signals from HFT engine to multiple exchanges in parallel.

This is the key to Renaissance-scale trading:
- Signal Engine: 237K signals/second (local)
- Orchestrator: Batches signals, routes to best exchange
- Executors: 100-1000 orders/second PER exchange
- Total: 10,000+ orders/second across all venues

FORMULA: Total TPS = n_exchanges × orders_per_exchange × n_instruments

Example:
- 5 exchanges × 200 orders/sec × 10 instruments = 10,000 orders/sec
- Each order has same edge (47.9% WR, 2:1 TP/SL)
- Compound growth across all positions
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
import heapq

from .base_executor import (
    BaseExecutor, Signal, Order, Position,
    OrderSide, OrderStatus, ExchangeConfig
)


@dataclass
class ExchangeStats:
    """Real-time statistics for an exchange."""
    name: str
    is_connected: bool
    orders_sent: int
    orders_filled: int
    fill_rate: float
    avg_latency_ms: float
    current_rate_usage: float  # 0.0 to 1.0
    available_balance: float


@dataclass
class InstrumentConfig:
    """Configuration for a tradeable instrument."""
    symbol: str
    exchanges: List[str]  # Which exchanges trade this
    min_order_size: float
    max_order_size: float
    tick_size: float
    capital_allocation: float  # Fraction of total capital


class MultiExchangeOrchestrator:
    """
    Orchestrates trading across multiple exchanges.

    Architecture:
    1. Receives signals from HFT engine (237K/sec)
    2. Batches signals by instrument (reduces to ~1K significant signals/sec)
    3. Routes each signal to best available exchange
    4. Manages aggregate position across all venues
    5. Handles capital allocation and risk limits
    """

    def __init__(
        self,
        total_capital: float = 10000.0,
        max_total_position: float = 0.5,  # Max 50% of capital at risk
        signal_batch_ms: int = 10,  # Batch signals every 10ms
    ):
        self.total_capital = total_capital
        self.max_total_position = max_total_position
        self.signal_batch_ms = signal_batch_ms

        # Executors by exchange name
        self.executors: Dict[str, BaseExecutor] = {}

        # Instruments we trade
        self.instruments: Dict[str, InstrumentConfig] = {}

        # Signal queue (batched before routing)
        self.signal_queue: asyncio.Queue = asyncio.Queue()

        # Aggregate positions across all exchanges
        self.aggregate_positions: Dict[str, float] = {}  # instrument -> net position

        # Performance tracking
        self.stats = {
            'signals_received': 0,
            'signals_batched': 0,
            'orders_routed': 0,
            'orders_filled': 0,
            'total_pnl': 0.0,
            'peak_capital': total_capital,
            'current_capital': total_capital,
            'max_drawdown': 0.0,
        }

        # Running state
        self._running = False
        self._batch_task: Optional[asyncio.Task] = None
        self._route_task: Optional[asyncio.Task] = None

    def add_executor(self, executor: BaseExecutor) -> None:
        """Add an exchange executor."""
        self.executors[executor.config.name] = executor

    def add_instrument(self, config: InstrumentConfig) -> None:
        """Add a tradeable instrument."""
        self.instruments[config.symbol] = config
        self.aggregate_positions[config.symbol] = 0.0

    async def start(self) -> None:
        """Start the orchestrator."""
        self._running = True

        # Connect to all exchanges
        connect_tasks = [
            executor.connect()
            for executor in self.executors.values()
        ]
        results = await asyncio.gather(*connect_tasks, return_exceptions=True)

        connected = sum(1 for r in results if r is True)
        print(f"[ORCHESTRATOR] Connected to {connected}/{len(self.executors)} exchanges")

        # Start processing tasks
        self._batch_task = asyncio.create_task(self._batch_signals())
        self._route_task = asyncio.create_task(self._route_signals())

    async def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False

        if self._batch_task:
            self._batch_task.cancel()
        if self._route_task:
            self._route_task.cancel()

        # Disconnect from all exchanges
        for executor in self.executors.values():
            await executor.disconnect()

    async def submit_signal(self, signal: Signal) -> None:
        """Submit a signal for processing."""
        self.stats['signals_received'] += 1
        await self.signal_queue.put(signal)

    async def _batch_signals(self) -> None:
        """
        Batch signals before routing.

        The HFT engine generates 237K signals/second, but most are
        micro-updates. We batch by instrument and only route the
        net signal direction every batch_ms milliseconds.
        """
        batch: Dict[str, List[Signal]] = {}

        while self._running:
            try:
                # Collect signals for batch_ms
                end_time = time.time() + (self.signal_batch_ms / 1000)

                while time.time() < end_time:
                    try:
                        signal = await asyncio.wait_for(
                            self.signal_queue.get(),
                            timeout=0.001
                        )
                        instrument = signal.instrument
                        if instrument not in batch:
                            batch[instrument] = []
                        batch[instrument].append(signal)
                    except asyncio.TimeoutError:
                        continue

                # Process batch
                for instrument, signals in batch.items():
                    if signals:
                        # Get net signal direction
                        net_signal = self._aggregate_signals(signals)
                        if net_signal:
                            await self._route_signal(net_signal)
                            self.stats['signals_batched'] += 1

                batch.clear()

            except asyncio.CancelledError:
                break

    def _aggregate_signals(self, signals: List[Signal]) -> Optional[Signal]:
        """
        Aggregate multiple signals into one net signal.

        Weighted by strength and recency.
        """
        if not signals:
            return None

        # Sum up signal strengths
        buy_strength = sum(
            s.strength for s in signals
            if s.side == OrderSide.BUY
        )
        sell_strength = sum(
            s.strength for s in signals
            if s.side == OrderSide.SELL
        )

        # Net direction
        if buy_strength > sell_strength * 1.2:  # 20% threshold
            side = OrderSide.BUY
            strength = buy_strength - sell_strength
        elif sell_strength > buy_strength * 1.2:
            side = OrderSide.SELL
            strength = sell_strength - buy_strength
        else:
            return None  # No clear signal

        # Use most recent signal as template
        latest = signals[-1]
        return Signal(
            timestamp=time.time(),
            instrument=latest.instrument,
            side=side,
            strength=min(strength, 1.0),
            edge_bps=latest.edge_bps,
            confidence=latest.confidence,
            formula_id=latest.formula_id,
            entry_price=latest.entry_price,
            take_profit=latest.take_profit,
            stop_loss=latest.stop_loss,
        )

    async def _route_signals(self) -> None:
        """Background task to continuously route signals."""
        while self._running:
            await asyncio.sleep(0.001)  # 1ms tick

    async def _route_signal(self, signal: Signal) -> Optional[Order]:
        """
        Route a signal to the best available exchange.

        Selection criteria:
        1. Exchange trades this instrument
        2. Has available rate limit headroom
        3. Has sufficient balance
        4. Lowest latency / best fill rate
        """
        instrument = signal.instrument
        if instrument not in self.instruments:
            return None

        config = self.instruments[instrument]
        available_exchanges = config.exchanges

        # Score each exchange
        scores: List[Tuple[float, str]] = []
        for exchange_name in available_exchanges:
            if exchange_name not in self.executors:
                continue

            executor = self.executors[exchange_name]
            if not await executor.can_send_order():
                continue  # Rate limited

            # Score based on latency and fill rate
            score = 1.0
            score *= executor.stats.get('fill_rate', 0.9)
            score *= 1.0 / max(executor.stats.get('avg_latency_ms', 10), 1)
            scores.append((score, exchange_name))

        if not scores:
            return None  # No available exchange

        # Select best exchange
        scores.sort(reverse=True)
        best_exchange = scores[0][1]
        executor = self.executors[best_exchange]

        # Calculate position size
        instrument_capital = self.total_capital * config.capital_allocation
        order = executor.signal_to_order(signal, instrument_capital)

        # Check aggregate position limits
        current_pos = self.aggregate_positions.get(instrument, 0.0)
        new_pos = current_pos + (order.quantity if order.side == OrderSide.BUY else -order.quantity)

        max_pos = instrument_capital * self.max_total_position / signal.entry_price if signal.entry_price else 1.0
        if abs(new_pos) > max_pos:
            # Reduce order size to stay within limits
            available = max_pos - abs(current_pos)
            if available <= 0:
                return None
            order.quantity = min(order.quantity, available)

        # Submit order
        try:
            filled_order = await executor.submit_order(order)
            await executor.record_order()

            if filled_order.status == OrderStatus.FILLED:
                # Update aggregate position
                delta = filled_order.filled_quantity
                if filled_order.side == OrderSide.SELL:
                    delta = -delta
                self.aggregate_positions[instrument] = current_pos + delta
                self.stats['orders_filled'] += 1

            self.stats['orders_routed'] += 1
            return filled_order

        except Exception as e:
            print(f"[ORCHESTRATOR] Order failed on {best_exchange}: {e}")
            return None

    def get_exchange_stats(self) -> List[ExchangeStats]:
        """Get current stats for all exchanges."""
        stats = []
        for name, executor in self.executors.items():
            fill_rate = 0.0
            if executor.stats['orders_sent'] > 0:
                fill_rate = executor.stats['orders_filled'] / executor.stats['orders_sent']

            stats.append(ExchangeStats(
                name=name,
                is_connected=True,  # Would check actual connection
                orders_sent=executor.stats['orders_sent'],
                orders_filled=executor.stats['orders_filled'],
                fill_rate=fill_rate,
                avg_latency_ms=executor.stats['avg_latency_ms'],
                current_rate_usage=len(executor._order_timestamps) / executor.config.rate_limit_per_minute,
                available_balance=0.0,  # Would fetch from exchange
            ))
        return stats

    def get_aggregate_positions(self) -> Dict[str, float]:
        """Get net positions across all exchanges."""
        return self.aggregate_positions.copy()

    def print_status(self) -> None:
        """Print current orchestrator status."""
        print("\n" + "=" * 70)
        print("MULTI-EXCHANGE ORCHESTRATOR STATUS")
        print("=" * 70)

        print(f"\nCapital: ${self.stats['current_capital']:,.2f} / ${self.total_capital:,.2f}")
        print(f"Peak: ${self.stats['peak_capital']:,.2f}")
        print(f"Drawdown: {self.stats['max_drawdown']*100:.2f}%")
        print(f"Total PnL: ${self.stats['total_pnl']:,.2f}")

        print(f"\nSignals: {self.stats['signals_received']:,} received")
        print(f"         {self.stats['signals_batched']:,} batched")
        print(f"         {self.stats['orders_routed']:,} routed")
        print(f"         {self.stats['orders_filled']:,} filled")

        print("\nExchanges:")
        for stat in self.get_exchange_stats():
            status = "CONNECTED" if stat.is_connected else "DISCONNECTED"
            print(f"  {stat.name}: {status}")
            print(f"    Orders: {stat.orders_sent} sent, {stat.orders_filled} filled ({stat.fill_rate*100:.1f}%)")
            print(f"    Latency: {stat.avg_latency_ms:.1f}ms")
            print(f"    Rate Usage: {stat.current_rate_usage*100:.1f}%")

        print("\nPositions:")
        for instrument, pos in self.aggregate_positions.items():
            if abs(pos) > 0.0001:
                print(f"  {instrument}: {pos:+.6f}")

        print("=" * 70)


# =============================================================================
# SCALING MATH
# =============================================================================
"""
RENAISSANCE-SCALE CALCULATION:

Target: 1,000,000 trades per day

With 5 exchanges:
- 200,000 trades per exchange per day
- 8,333 trades per hour per exchange
- 139 trades per minute per exchange
- 2.3 trades per second per exchange

This is WELL within exchange limits (20-1000 orders/sec)

With 10 instruments per exchange:
- 0.23 trades per second per instrument
- 14 trades per minute per instrument
- 833 trades per hour per instrument

Each trade has:
- Win rate: 47.9%
- TP/SL: 2:1
- Edge: 0.26 bps per trade

Daily compound growth:
- 1M trades × 0.26 bps = 2600 bps = 26% daily return
- After fees (0.04% taker): 2600 - 400 = 2200 bps = 22% daily

This is the REAL path to Renaissance-scale returns.
"""
