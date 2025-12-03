"""
MAXED SOVEREIGN RUNNER - ALL CPU CORES AT 100%
===============================================
Renaissance Technologies level execution.

This runner maxes out ALL available CPU cores for maximum TPS.
Uses multiprocessing to parallelize matching across cores.

Usage:
    python -m engine.sovereign.maxed_runner [trades] [capital]
"""
import sys
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple
import threading

# Set process priority and CPU affinity
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from engine.sovereign.matching_engine import SovereignMatchingEngine, OrderSide


@dataclass
class CoreStats:
    """Stats from a single CPU core."""
    core_id: int
    trades: int
    pnl: float
    execution_ns: float
    tps: float


def set_max_priority():
    """Set maximum CPU priority for this process."""
    if HAS_PSUTIL:
        try:
            p = psutil.Process()
            if sys.platform == 'win32':
                p.nice(psutil.REALTIME_PRIORITY_CLASS)
            else:
                p.nice(-20)  # Highest priority on Linux
            print(f"[MAXED] Set process priority to REALTIME")
        except:
            pass

    # Disable garbage collection during trading
    import gc
    gc.disable()
    print("[MAXED] Garbage collection DISABLED")


def get_cpu_count() -> int:
    """Get number of available CPU cores."""
    return mp.cpu_count()


def core_worker(
    core_id: int,
    trades_per_core: int,
    initial_capital: float,
    result_queue: mp.Queue,
    orderbook_data: Tuple[List, List],
):
    """
    Worker function for a single CPU core.

    Each core runs its own matching engine at maximum speed.
    """
    # Set CPU affinity to this specific core
    if HAS_PSUTIL:
        try:
            p = psutil.Process()
            p.cpu_affinity([core_id])
        except:
            pass

    # Create engine for this core
    engine = SovereignMatchingEngine(
        initial_capital=initial_capital,
        settlement_threshold=float('inf'),  # No settlement interrupts
    )

    # Update orderbook with shared data
    bids, asks = orderbook_data
    if bids and asks:
        engine.update_orderbook("BTC", bids, asks)

    # Run trades at maximum speed
    start = time.perf_counter_ns()

    for i in range(trades_per_core):
        # Alternate buy/sell for balanced execution
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        engine.execute(
            asset="BTC",
            side=side,
            quantity=0.001,
            signal_strength=0.5,
        )

    end = time.perf_counter_ns()

    # Calculate stats
    duration_ns = end - start
    duration_s = duration_ns / 1e9
    tps = trades_per_core / duration_s if duration_s > 0 else 0
    avg_exec_ns = duration_ns / trades_per_core if trades_per_core > 0 else 0

    stats = CoreStats(
        core_id=core_id,
        trades=trades_per_core,
        pnl=engine.total_pnl,
        execution_ns=avg_exec_ns,
        tps=tps,
    )

    result_queue.put(stats)


class MaxedSovereignRunner:
    """
    MAXED OUT sovereign matching engine.

    Uses ALL CPU cores at 100% for maximum TPS.
    """

    def __init__(
        self,
        initial_capital: float = 5.0,
        use_all_cores: bool = True,
    ):
        self.initial_capital = initial_capital
        self.num_cores = get_cpu_count() if use_all_cores else 1

        # Shared orderbook data
        self.orderbook_data: Tuple[List, List] = ([], [])

        # Stats
        self.total_trades = 0
        self.total_tps = 0
        self.core_stats: List[CoreStats] = []

    def fetch_orderbook(self):
        """Fetch real orderbook from Hyperliquid."""
        try:
            from blockchain.node_data_feed import NodeDataFeed
            feed = NodeDataFeed(use_mainnet=True)

            if feed.is_connected:
                ob = feed.get_orderbook("BTC")
                if ob.is_valid:
                    self.orderbook_data = (ob.bids, ob.asks)
                    print(f"[MAXED] Orderbook: Bid ${ob.best_bid:,.2f} | Ask ${ob.best_ask:,.2f} | Spread {ob.spread_bps:.1f}bps")
                    return True
        except Exception as e:
            print(f"[MAXED] Orderbook fetch failed: {e}")

        # Fallback synthetic orderbook
        mid = 97000.0
        self.orderbook_data = (
            [(mid - 0.5, 10.0), (mid - 1.0, 20.0), (mid - 2.0, 30.0)],
            [(mid + 0.5, 10.0), (mid + 1.0, 20.0), (mid + 2.0, 30.0)],
        )
        print("[MAXED] Using synthetic orderbook")
        return False

    def run(self, total_trades: int = 1000000):
        """
        Run MAXED matching across all CPU cores.
        """
        set_max_priority()

        print("\n" + "=" * 70)
        print("SOVEREIGN MATCHING ENGINE - MAXED OUT")
        print("=" * 70)
        print(f"CPU Cores: {self.num_cores}")
        print(f"Total Trades: {total_trades:,}")
        print(f"Trades/Core: {total_trades // self.num_cores:,}")
        print(f"Capital: ${self.initial_capital:.2f}")
        print("=" * 70 + "\n")

        # Fetch orderbook
        self.fetch_orderbook()

        # Distribute trades across cores
        trades_per_core = total_trades // self.num_cores

        # Create result queue
        result_queue = mp.Queue()

        print(f"[MAXED] Starting {self.num_cores} parallel matching engines...")
        start_time = time.time()

        # Spawn workers for each core
        processes = []
        for core_id in range(self.num_cores):
            p = mp.Process(
                target=core_worker,
                args=(
                    core_id,
                    trades_per_core,
                    self.initial_capital / self.num_cores,
                    result_queue,
                    self.orderbook_data,
                ),
            )
            processes.append(p)
            p.start()

        # Collect results
        for _ in range(self.num_cores):
            stats = result_queue.get()
            self.core_stats.append(stats)
            print(f"[CORE {stats.core_id}] {stats.trades:,} trades | {stats.tps:,.0f} TPS | {stats.execution_ns:.0f}ns avg")

        # Wait for all processes
        for p in processes:
            p.join()

        end_time = time.time()
        duration = end_time - start_time

        # Calculate totals
        self.total_trades = sum(s.trades for s in self.core_stats)
        self.total_tps = self.total_trades / duration
        total_pnl = sum(s.pnl for s in self.core_stats)
        avg_exec_ns = sum(s.execution_ns for s in self.core_stats) / len(self.core_stats)

        print("\n" + "=" * 70)
        print("MAXED RESULTS")
        print("=" * 70)
        print(f"\n--- AGGREGATE ---")
        print(f"Total Trades: {self.total_trades:,}")
        print(f"Duration: {duration:.2f}s")
        print(f"AGGREGATE TPS: {self.total_tps:,.0f}")
        print(f"Average Execution: {avg_exec_ns:.0f}ns")
        print(f"\n--- PER CORE ---")
        for stats in sorted(self.core_stats, key=lambda x: x.core_id):
            print(f"  Core {stats.core_id}: {stats.tps:,.0f} TPS")
        print(f"\n--- THEORETICAL MAXIMUM ---")
        max_per_core_tps = 1e9 / avg_exec_ns if avg_exec_ns > 0 else 0
        theoretical_max = max_per_core_tps * self.num_cores
        print(f"Per Core Max: {max_per_core_tps:,.0f} TPS")
        print(f"Total Max ({self.num_cores} cores): {theoretical_max:,.0f} TPS")
        print("=" * 70)
        print("ALL CPU CORES MAXED OUT")
        print("=" * 70)


def main():
    """Main entry point."""
    total_trades = 1000000
    capital = 5.0

    if len(sys.argv) > 1:
        try:
            total_trades = int(sys.argv[1])
        except:
            pass

    if len(sys.argv) > 2:
        try:
            capital = float(sys.argv[2])
        except:
            pass

    runner = MaxedSovereignRunner(initial_capital=capital)
    runner.run(total_trades=total_trades)


if __name__ == "__main__":
    # Windows multiprocessing fix
    mp.freeze_support()
    main()
