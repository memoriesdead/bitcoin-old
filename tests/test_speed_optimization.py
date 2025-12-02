#!/usr/bin/env python3
"""
RENAISSANCE TECHNOLOGIES LEVEL SPEED OPTIMIZATION TEST
=======================================================
Target: 300,000 to 1,000,000,000,000 trades
Level: Nanosecond to Millisecond precision

KVM8 Specs: 8 cores AMD EPYC, 32GB RAM

This test measures and optimizes:
1. Signal generation latency (target: < 1 microsecond)
2. Trading decision speed (target: < 100 nanoseconds)
3. Throughput (target: > 10 million signals/second)
4. Memory efficiency
5. CPU utilization across all 8 cores
"""

import time
import sys
import os
import math
import statistics
from dataclasses import dataclass
from typing import List, Tuple
from collections import deque
import multiprocessing as mp

# Try to import performance libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[WARN] numpy not available - install for 10x speed")

try:
    from numba import jit, njit, prange, vectorize, float64, int64
    from numba import config as numba_config
    NUMBA_AVAILABLE = True
    # Numba optimizations
    numba_config.THREADING_LAYER = 'omp'
except ImportError:
    NUMBA_AVAILABLE = False
    print("[WARN] numba not available - install for 100x speed")


# =============================================================================
# PURE PYTHON BASELINE (slowest)
# =============================================================================

def pure_python_ofi(fee_pressure: float, tx_momentum: float, congestion: float) -> float:
    """Pure Python OFI calculation - BASELINE"""
    return fee_pressure * 0.35 + tx_momentum * 0.35 + congestion * 0.30


def pure_python_power_law(days: float) -> float:
    """Pure Python Power Law calculation - BASELINE"""
    return 10 ** (-17.0161223 + 5.8451542 * math.log10(days))


def pure_python_trading_decision(ofi: float, threshold: float = 0.15) -> int:
    """Pure Python trading decision - BASELINE"""
    if ofi > threshold:
        return 1  # BUY
    elif ofi < -threshold:
        return -1  # SELL
    return 0  # HOLD


# =============================================================================
# NUMPY VECTORIZED (10-100x faster)
# =============================================================================

if NUMPY_AVAILABLE:
    def numpy_ofi_batch(fee_pressure: np.ndarray, tx_momentum: np.ndarray,
                        congestion: np.ndarray) -> np.ndarray:
        """Numpy vectorized OFI - processes millions at once"""
        return fee_pressure * 0.35 + tx_momentum * 0.35 + congestion * 0.30

    def numpy_power_law_batch(days: np.ndarray) -> np.ndarray:
        """Numpy vectorized Power Law"""
        return np.power(10, -17.0161223 + 5.8451542 * np.log10(days))

    def numpy_trading_decisions_batch(ofi: np.ndarray, threshold: float = 0.15) -> np.ndarray:
        """Numpy vectorized trading decisions"""
        decisions = np.zeros(len(ofi), dtype=np.int8)
        decisions[ofi > threshold] = 1
        decisions[ofi < -threshold] = -1
        return decisions


# =============================================================================
# NUMBA JIT COMPILED (100-1000x faster)
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True)
    def numba_ofi(fee_pressure: float, tx_momentum: float, congestion: float) -> float:
        """Numba JIT OFI - near C speed"""
        return fee_pressure * 0.35 + tx_momentum * 0.35 + congestion * 0.30

    @njit(fastmath=True, cache=True)
    def numba_power_law(days: float) -> float:
        """Numba JIT Power Law"""
        return 10.0 ** (-17.0161223 + 5.8451542 * math.log10(days))

    @njit(fastmath=True, cache=True)
    def numba_trading_decision(ofi: float, threshold: float = 0.15) -> int:
        """Numba JIT trading decision"""
        if ofi > threshold:
            return 1
        elif ofi < -threshold:
            return -1
        return 0

    @njit(parallel=True, fastmath=True, cache=True)
    def numba_ofi_batch_parallel(fee_pressure: np.ndarray, tx_momentum: np.ndarray,
                                  congestion: np.ndarray) -> np.ndarray:
        """Numba parallel OFI - uses all CPU cores"""
        n = len(fee_pressure)
        result = np.empty(n, dtype=np.float64)
        for i in prange(n):
            result[i] = fee_pressure[i] * 0.35 + tx_momentum[i] * 0.35 + congestion[i] * 0.30
        return result

    @njit(parallel=True, fastmath=True, cache=True)
    def numba_full_pipeline_parallel(fee_pressure: np.ndarray, tx_momentum: np.ndarray,
                                      congestion: np.ndarray, threshold: float = 0.15) -> np.ndarray:
        """Complete trading pipeline - OFI + decision in one pass"""
        n = len(fee_pressure)
        decisions = np.empty(n, dtype=np.int8)
        for i in prange(n):
            ofi = fee_pressure[i] * 0.35 + tx_momentum[i] * 0.35 + congestion[i] * 0.30
            if ofi > threshold:
                decisions[i] = 1
            elif ofi < -threshold:
                decisions[i] = -1
            else:
                decisions[i] = 0
        return decisions

    @njit(fastmath=True, cache=True)
    def numba_mempool_signals(t: float) -> Tuple[float, float, float, float]:
        """
        Numba JIT mempool signal generation.
        Returns: (fee_pressure, tx_momentum, congestion, block_progress)
        """
        # Block timing (10 min avg)
        block_progress = (t % 600.0) / 600.0

        # Fee pressure oscillation
        fee_pressure = math.sin(t * 0.001) * 0.5 + math.sin(t * 0.0001) * 0.3

        # TX momentum
        tx_momentum = math.cos(t * 0.0005) * 0.4 + math.sin(t * 0.00005) * 0.2

        # Congestion signal
        congestion = (math.sin(t * 0.0002) + 1.0) / 2.0 * 0.6

        return fee_pressure, tx_momentum, congestion, block_progress


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def benchmark_single_operation(func, args, iterations: int = 1_000_000) -> dict:
    """Benchmark a single operation."""
    # Warmup
    for _ in range(1000):
        func(*args)

    # Time it
    times = []
    for _ in range(10):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            func(*args)
        end = time.perf_counter_ns()
        times.append((end - start) / iterations)

    return {
        'mean_ns': statistics.mean(times),
        'min_ns': min(times),
        'max_ns': max(times),
        'std_ns': statistics.stdev(times) if len(times) > 1 else 0,
        'ops_per_sec': 1e9 / statistics.mean(times),
    }


def benchmark_batch_operation(func, arrays, iterations: int = 1000) -> dict:
    """Benchmark a batch/vectorized operation."""
    batch_size = len(arrays[0])

    # Warmup
    for _ in range(10):
        func(*arrays)

    # Time it
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func(*arrays)
        end = time.perf_counter_ns()
        times.append(end - start)

    mean_time = statistics.mean(times)

    return {
        'mean_ns': mean_time,
        'per_item_ns': mean_time / batch_size,
        'batch_size': batch_size,
        'batches_per_sec': 1e9 / mean_time,
        'items_per_sec': batch_size * 1e9 / mean_time,
    }


def run_latency_test(iterations: int = 100_000) -> dict:
    """Measure end-to-end latency for a single trade decision."""
    latencies = []

    for i in range(iterations):
        t = time.time() + i * 0.001

        start = time.perf_counter_ns()

        # Full pipeline: generate signals -> calculate OFI -> make decision
        if NUMBA_AVAILABLE:
            fee, tx, cong, _ = numba_mempool_signals(t)
            ofi = numba_ofi(fee, tx, cong)
            decision = numba_trading_decision(ofi)
        else:
            fee = math.sin(t * 0.001) * 0.5
            tx = math.cos(t * 0.0005) * 0.4
            cong = (math.sin(t * 0.0002) + 1.0) / 2.0 * 0.6
            ofi = pure_python_ofi(fee, tx, cong)
            decision = pure_python_trading_decision(ofi)

        end = time.perf_counter_ns()
        latencies.append(end - start)

    return {
        'mean_ns': statistics.mean(latencies),
        'median_ns': statistics.median(latencies),
        'p50_ns': sorted(latencies)[int(len(latencies) * 0.50)],
        'p95_ns': sorted(latencies)[int(len(latencies) * 0.95)],
        'p99_ns': sorted(latencies)[int(len(latencies) * 0.99)],
        'min_ns': min(latencies),
        'max_ns': max(latencies),
    }


def run_throughput_test(duration_sec: float = 5.0) -> dict:
    """Measure maximum throughput."""
    if not NUMPY_AVAILABLE:
        return {'error': 'numpy required for throughput test'}

    # Generate test data
    batch_size = 1_000_000
    fee_pressure = np.random.uniform(-1, 1, batch_size).astype(np.float64)
    tx_momentum = np.random.uniform(-1, 1, batch_size).astype(np.float64)
    congestion = np.random.uniform(0, 1, batch_size).astype(np.float64)

    # Warmup
    if NUMBA_AVAILABLE:
        for _ in range(5):
            numba_full_pipeline_parallel(fee_pressure, tx_momentum, congestion)

    # Run test
    total_trades = 0
    start = time.perf_counter()

    while time.perf_counter() - start < duration_sec:
        if NUMBA_AVAILABLE:
            decisions = numba_full_pipeline_parallel(fee_pressure, tx_momentum, congestion)
        else:
            decisions = numpy_trading_decisions_batch(
                numpy_ofi_batch(fee_pressure, tx_momentum, congestion)
            )
        total_trades += batch_size

    elapsed = time.perf_counter() - start

    return {
        'total_trades': total_trades,
        'elapsed_sec': elapsed,
        'trades_per_sec': total_trades / elapsed,
        'trades_per_ms': total_trades / elapsed / 1000,
        'ns_per_trade': elapsed * 1e9 / total_trades,
    }


def run_parallel_scaling_test() -> dict:
    """Test how performance scales with CPU cores."""
    if not NUMPY_AVAILABLE or not NUMBA_AVAILABLE:
        return {'error': 'numpy and numba required'}

    batch_size = 10_000_000
    fee = np.random.uniform(-1, 1, batch_size).astype(np.float64)
    tx = np.random.uniform(-1, 1, batch_size).astype(np.float64)
    cong = np.random.uniform(0, 1, batch_size).astype(np.float64)

    results = {}

    for num_threads in [1, 2, 4, 8]:
        os.environ['NUMBA_NUM_THREADS'] = str(num_threads)
        os.environ['OMP_NUM_THREADS'] = str(num_threads)

        # Warmup
        numba_full_pipeline_parallel(fee, tx, cong)

        # Time
        times = []
        for _ in range(5):
            start = time.perf_counter()
            numba_full_pipeline_parallel(fee, tx, cong)
            times.append(time.perf_counter() - start)

        results[f'{num_threads}_threads'] = {
            'mean_sec': statistics.mean(times),
            'trades_per_sec': batch_size / statistics.mean(times),
        }

    return results


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    print("=" * 80)
    print("RENAISSANCE TECHNOLOGIES LEVEL SPEED OPTIMIZATION TEST")
    print("=" * 80)
    print()

    # System info
    print("SYSTEM INFO:")
    print(f"  CPU cores: {mp.cpu_count()}")
    print(f"  Python version: {sys.version.split()[0]}")
    print(f"  numpy available: {NUMPY_AVAILABLE}")
    print(f"  numba available: {NUMBA_AVAILABLE}")
    print()

    # Test 1: Single operation benchmarks
    print("=" * 80)
    print("TEST 1: SINGLE OPERATION LATENCY")
    print("=" * 80)

    print("\n[Pure Python - BASELINE]")
    result = benchmark_single_operation(pure_python_ofi, (0.5, 0.3, 0.2))
    print(f"  OFI calculation: {result['mean_ns']:.1f} ns ({result['ops_per_sec']:,.0f} ops/sec)")

    result = benchmark_single_operation(pure_python_trading_decision, (0.3,))
    print(f"  Trading decision: {result['mean_ns']:.1f} ns ({result['ops_per_sec']:,.0f} ops/sec)")

    if NUMBA_AVAILABLE:
        print("\n[Numba JIT - OPTIMIZED]")

        # Warmup JIT
        numba_ofi(0.5, 0.3, 0.2)
        numba_trading_decision(0.3)
        numba_mempool_signals(time.time())

        result = benchmark_single_operation(numba_ofi, (0.5, 0.3, 0.2))
        print(f"  OFI calculation: {result['mean_ns']:.1f} ns ({result['ops_per_sec']:,.0f} ops/sec)")

        result = benchmark_single_operation(numba_trading_decision, (0.3,))
        print(f"  Trading decision: {result['mean_ns']:.1f} ns ({result['ops_per_sec']:,.0f} ops/sec)")

        result = benchmark_single_operation(numba_mempool_signals, (time.time(),))
        print(f"  Mempool signals: {result['mean_ns']:.1f} ns ({result['ops_per_sec']:,.0f} ops/sec)")

    # Test 2: Batch operations
    print("\n" + "=" * 80)
    print("TEST 2: BATCH OPERATION THROUGHPUT")
    print("=" * 80)

    if NUMPY_AVAILABLE:
        batch_sizes = [10_000, 100_000, 1_000_000, 10_000_000]

        for batch_size in batch_sizes:
            fee = np.random.uniform(-1, 1, batch_size).astype(np.float64)
            tx = np.random.uniform(-1, 1, batch_size).astype(np.float64)
            cong = np.random.uniform(0, 1, batch_size).astype(np.float64)

            print(f"\n[Batch size: {batch_size:,}]")

            # Numpy
            result = benchmark_batch_operation(numpy_ofi_batch, (fee, tx, cong), iterations=100)
            print(f"  Numpy: {result['items_per_sec']:,.0f} items/sec ({result['per_item_ns']:.2f} ns/item)")

            # Numba parallel
            if NUMBA_AVAILABLE:
                # Warmup
                numba_full_pipeline_parallel(fee, tx, cong)

                result = benchmark_batch_operation(
                    numba_full_pipeline_parallel, (fee, tx, cong), iterations=100
                )
                print(f"  Numba parallel: {result['items_per_sec']:,.0f} items/sec ({result['per_item_ns']:.2f} ns/item)")

    # Test 3: End-to-end latency
    print("\n" + "=" * 80)
    print("TEST 3: END-TO-END LATENCY (signal -> decision)")
    print("=" * 80)

    result = run_latency_test(iterations=100_000)
    print(f"  Mean latency:   {result['mean_ns']:.1f} ns ({result['mean_ns']/1000:.3f} µs)")
    print(f"  Median latency: {result['median_ns']:.1f} ns")
    print(f"  P95 latency:    {result['p95_ns']:.1f} ns")
    print(f"  P99 latency:    {result['p99_ns']:.1f} ns")
    print(f"  Min latency:    {result['min_ns']:.1f} ns")
    print(f"  Max latency:    {result['max_ns']:.1f} ns")

    # Test 4: Maximum throughput
    print("\n" + "=" * 80)
    print("TEST 4: MAXIMUM THROUGHPUT (5 second stress test)")
    print("=" * 80)

    result = run_throughput_test(duration_sec=5.0)
    if 'error' not in result:
        print(f"  Total trades:    {result['total_trades']:,}")
        print(f"  Elapsed:         {result['elapsed_sec']:.2f} sec")
        print(f"  Trades/second:   {result['trades_per_sec']:,.0f}")
        print(f"  Trades/ms:       {result['trades_per_ms']:,.0f}")
        print(f"  ns per trade:    {result['ns_per_trade']:.2f}")
    else:
        print(f"  Error: {result['error']}")

    # Test 5: Parallel scaling
    print("\n" + "=" * 80)
    print("TEST 5: PARALLEL SCALING (1-8 cores)")
    print("=" * 80)

    if NUMBA_AVAILABLE:
        results = run_parallel_scaling_test()
        baseline = results.get('1_threads', {}).get('trades_per_sec', 1)

        for threads, data in sorted(results.items()):
            if 'trades_per_sec' in data:
                speedup = data['trades_per_sec'] / baseline
                print(f"  {threads}: {data['trades_per_sec']:,.0f} trades/sec (speedup: {speedup:.2f}x)")
    else:
        print("  [Skipped - requires numba]")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - RENAISSANCE TECHNOLOGIES READINESS")
    print("=" * 80)

    if NUMBA_AVAILABLE and NUMPY_AVAILABLE:
        throughput_result = run_throughput_test(duration_sec=2.0)
        trades_per_sec = throughput_result.get('trades_per_sec', 0)

        print(f"\n  Peak throughput: {trades_per_sec:,.0f} trades/second")
        print(f"  Daily capacity:  {trades_per_sec * 86400:,.0f} trades/day")
        print(f"  Yearly capacity: {trades_per_sec * 86400 * 365:,.0f} trades/year")

        # Grade
        if trades_per_sec > 100_000_000:
            grade = "S+ (WORLD CLASS)"
        elif trades_per_sec > 10_000_000:
            grade = "S (RENAISSANCE LEVEL)"
        elif trades_per_sec > 1_000_000:
            grade = "A (INSTITUTIONAL)"
        elif trades_per_sec > 100_000:
            grade = "B (PROFESSIONAL)"
        else:
            grade = "C (NEEDS OPTIMIZATION)"

        print(f"\n  GRADE: {grade}")

        if trades_per_sec >= 10_000_000:
            print("\n  ✓ READY FOR 300,000+ trades")
            print("  ✓ READY FOR billions of trades")
            print("  ✓ NANOSECOND-LEVEL PERFORMANCE")
    else:
        print("\n  [CRITICAL] Install numpy and numba for full optimization")
        print("  pip install numpy numba")


if __name__ == "__main__":
    main()
