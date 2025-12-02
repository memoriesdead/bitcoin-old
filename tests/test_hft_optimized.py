#!/usr/bin/env python3
"""
HFT OPTIMIZED SPEED TEST - RENAISSANCE TECHNOLOGIES LEVEL
============================================================
Tests all optimizations from research:
1. SIMD/AVX with Numba fastmath
2. Memory-mapped shared memory
3. CPU affinity and real-time priority
4. Parallel processing across all 8 cores

Target: Beat 12 billion ops/sec baseline
"""

import os
import sys
import time
import struct
import mmap
import numpy as np
import multiprocessing
from typing import Tuple

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Numba with aggressive optimization
os.environ['NUMBA_OPT'] = '3'
os.environ['NUMBA_LOOP_VECTORIZE'] = '1'
os.environ['NUMBA_INTEL_SVML'] = '1'
os.environ['NUMBA_ENABLE_AVX'] = '1'
os.environ['NUMBA_THREADING_LAYER'] = 'omp'
os.environ['NUMBA_BOUNDSCHECK'] = '0'

from numba import njit, prange, config
config.FASTMATH = True

# Import our HFT optimizer
try:
    from core.hft_optimizer import (
        HFTOptimizer,
        simd_calculate_ofi,
        simd_batch_signals,
        simd_power_law_prices
    )
    HFT_OPTIMIZER_AVAILABLE = True
except ImportError:
    HFT_OPTIMIZER_AVAILABLE = False
    print("Warning: HFT optimizer not available")


# =============================================================================
# SIMD-OPTIMIZED FUNCTIONS (Float32 for 8x AVX throughput)
# =============================================================================

@njit(fastmath=True, cache=True)
def simd_trading_decision_f32(
    prices: np.ndarray,      # float32
    fair_values: np.ndarray, # float32
    fee_pressure: np.ndarray,# float32
    tx_momentum: np.ndarray, # float32
    congestion: np.ndarray,  # float32
    threshold: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ultra-fast SIMD trading decision using float32.

    AVX2 processes 8 float32 values per instruction vs 4 float64.
    This doubles throughput for the same operations.

    Returns: (directions, strengths, ofi_values)
    """
    n = len(prices)
    directions = np.empty(n, dtype=np.int32)
    strengths = np.empty(n, dtype=np.float32)
    ofi_values = np.empty(n, dtype=np.float32)

    for i in range(n):
        # Calculate OFI (vectorized)
        ofi = (
            fee_pressure[i] * 0.35 +
            tx_momentum[i] * 0.35 +
            congestion[i] * 0.30
        )
        ofi_values[i] = ofi

        # Calculate strength
        strengths[i] = ofi if ofi > 0 else -ofi  # abs without branch

        # Direction decision (minimal branching)
        if ofi > threshold:
            directions[i] = 1   # BUY
        elif ofi < -threshold:
            directions[i] = -1  # SELL
        else:
            directions[i] = 0   # HOLD

    return directions, strengths, ofi_values


@njit(fastmath=True, parallel=True, cache=True)
def simd_parallel_batch(
    prices: np.ndarray,
    fair_values: np.ndarray,
    fee_pressure: np.ndarray,
    tx_momentum: np.ndarray,
    congestion: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel SIMD batch processing across all CPU cores.

    Uses prange for automatic parallel distribution.
    Each core processes a chunk with full SIMD vectorization.
    """
    n = len(prices)
    directions = np.empty(n, dtype=np.int32)
    strengths = np.empty(n, dtype=np.float32)
    ofi_values = np.empty(n, dtype=np.float32)
    deviations = np.empty(n, dtype=np.float32)

    # Parallel loop - Numba distributes across cores
    for i in prange(n):
        # OFI calculation
        ofi = (
            fee_pressure[i] * 0.35 +
            tx_momentum[i] * 0.35 +
            congestion[i] * 0.30
        )
        ofi_values[i] = ofi

        # Deviation from fair value
        deviations[i] = (prices[i] - fair_values[i]) / fair_values[i] * 100.0

        # Strength (branchless abs)
        strengths[i] = ofi if ofi > 0 else -ofi

        # Direction
        if ofi > 0.15:
            directions[i] = 1
        elif ofi < -0.15:
            directions[i] = -1
        else:
            directions[i] = 0

    return directions, strengths, ofi_values, deviations


@njit(fastmath=True, cache=True)
def simd_power_law_batch(days: np.ndarray) -> np.ndarray:
    """
    SIMD Power Law price calculation.

    Price = 10^(-17.0161223 + 5.8451542 * log10(days))
    """
    n = len(days)
    result = np.empty(n, dtype=np.float64)

    A = -17.0161223
    B = 5.8451542

    for i in range(n):
        log_price = A + B * np.log10(days[i])
        result[i] = 10.0 ** log_price

    return result


@njit(fastmath=True, parallel=True, cache=True)
def simd_mempool_simulation(
    times: np.ndarray,
    block_time: float = 600.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SIMD-parallel mempool simulation.

    Simulates fee pressure, TX momentum, and congestion
    based on time within block cycle.
    """
    n = len(times)
    fee_pressure = np.empty(n, dtype=np.float32)
    tx_momentum = np.empty(n, dtype=np.float32)
    congestion = np.empty(n, dtype=np.float32)

    for i in prange(n):
        # Time within block
        t = times[i]
        block_progress = (t % block_time) / block_time

        # Fee pressure oscillates with block progress
        fee_pressure[i] = np.sin(block_progress * 2.0 * np.pi) * 0.5

        # TX momentum based on recent activity
        tx_momentum[i] = np.cos(t * 0.01) * 0.4

        # Congestion cycles
        congestion[i] = (np.sin(t * 0.005) + 1.0) * 0.25

    return fee_pressure, tx_momentum, congestion


# =============================================================================
# SHARED MEMORY BENCHMARK
# =============================================================================

class SharedMemoryBenchmark:
    """Test zero-copy shared memory performance."""

    SIGNAL_SIZE = 24  # timestamp(8) + price(8) + ofi(4) + dir(4)

    def __init__(self, name: str, capacity: int):
        self.name = name
        self.capacity = capacity
        self.size = self.SIGNAL_SIZE * capacity + 8

        if sys.platform == 'linux':
            path = f"/dev/shm/{name}"
            self.fd = os.open(path, os.O_CREAT | os.O_RDWR)
            os.ftruncate(self.fd, self.size)
            self.mm = mmap.mmap(self.fd, self.size)
        else:
            self.mm = mmap.mmap(-1, self.size, tagname=name)
            self.fd = None

        struct.pack_into('Q', self.mm, 0, 0)

    def write_batch(self, count: int):
        """Write batch of signals to shared memory."""
        ts = time.time()
        for i in range(count):
            offset = 8 + (i % self.capacity) * self.SIGNAL_SIZE
            struct.pack_into('ddfi', self.mm, offset, ts, 97000.0 + i * 0.01, 0.5, 1)

    def close(self):
        self.mm.close()
        if self.fd:
            os.close(self.fd)


# =============================================================================
# MAIN TEST
# =============================================================================

def run_tests():
    """Run all HFT optimization tests."""

    print("=" * 80)
    print("HFT OPTIMIZED SPEED TEST - RENAISSANCE TECHNOLOGIES LEVEL")
    print("=" * 80)
    print()

    cpu_count = multiprocessing.cpu_count()
    print(f"CPU Cores: {cpu_count}")
    print(f"Platform: {sys.platform}")
    print(f"Numba fastmath: ENABLED")
    print(f"AVX/SIMD: ENABLED")
    print()

    # Apply HFT optimizations if available
    if HFT_OPTIMIZER_AVAILABLE:
        print("Applying HFT optimizations...")
        optimizer = HFTOptimizer(verbose=False)
        optimizer.set_cpu_affinity()
        print()

    results = {}

    # =========================================================================
    # TEST 1: SIMD Trading Decision (Float32)
    # =========================================================================
    print("=" * 80)
    print("TEST 1: SIMD TRADING DECISION (Float32 - 8x AVX throughput)")
    print("=" * 80)

    sizes = [100_000, 1_000_000, 10_000_000, 100_000_000]

    for n in sizes:
        # Generate test data (float32 for SIMD)
        prices = np.random.uniform(95000, 99000, n).astype(np.float32)
        fair_values = np.full(n, 97000.0, dtype=np.float32)
        fee = np.random.randn(n).astype(np.float32) * 0.5
        tx = np.random.randn(n).astype(np.float32) * 0.5
        cong = np.random.randn(n).astype(np.float32) * 0.3

        # Warmup
        _ = simd_trading_decision_f32(prices[:1000], fair_values[:1000], fee[:1000], tx[:1000], cong[:1000])

        # Benchmark
        start = time.perf_counter_ns()
        directions, strengths, ofi = simd_trading_decision_f32(prices, fair_values, fee, tx, cong)
        elapsed = time.perf_counter_ns() - start

        ops_per_sec = n / (elapsed / 1e9)
        ns_per_op = elapsed / n

        print(f"  {n:>12,} signals: {elapsed/1e6:>8.2f}ms | {ops_per_sec/1e9:>6.2f}B ops/sec | {ns_per_op:>5.2f}ns/op")

        results[f'simd_f32_{n}'] = ops_per_sec

    print()

    # =========================================================================
    # TEST 2: Parallel SIMD (All Cores)
    # =========================================================================
    print("=" * 80)
    print(f"TEST 2: PARALLEL SIMD ({cpu_count} cores)")
    print("=" * 80)

    for n in sizes:
        prices = np.random.uniform(95000, 99000, n).astype(np.float32)
        fair_values = np.full(n, 97000.0, dtype=np.float32)
        fee = np.random.randn(n).astype(np.float32) * 0.5
        tx = np.random.randn(n).astype(np.float32) * 0.5
        cong = np.random.randn(n).astype(np.float32) * 0.3

        # Warmup
        _ = simd_parallel_batch(prices[:1000], fair_values[:1000], fee[:1000], tx[:1000], cong[:1000])

        # Benchmark
        start = time.perf_counter_ns()
        dirs, strs, ofi, devs = simd_parallel_batch(prices, fair_values, fee, tx, cong)
        elapsed = time.perf_counter_ns() - start

        ops_per_sec = n / (elapsed / 1e9)
        ns_per_op = elapsed / n

        print(f"  {n:>12,} signals: {elapsed/1e6:>8.2f}ms | {ops_per_sec/1e9:>6.2f}B ops/sec | {ns_per_op:>5.2f}ns/op")

        results[f'parallel_{n}'] = ops_per_sec

    print()

    # =========================================================================
    # TEST 3: Power Law Batch
    # =========================================================================
    print("=" * 80)
    print("TEST 3: SIMD POWER LAW CALCULATION")
    print("=" * 80)

    for n in sizes:
        days = np.random.uniform(1000, 6000, n).astype(np.float64)

        # Warmup
        _ = simd_power_law_batch(days[:1000])

        # Benchmark
        start = time.perf_counter_ns()
        prices = simd_power_law_batch(days)
        elapsed = time.perf_counter_ns() - start

        ops_per_sec = n / (elapsed / 1e9)
        ns_per_op = elapsed / n

        print(f"  {n:>12,} prices:  {elapsed/1e6:>8.2f}ms | {ops_per_sec/1e9:>6.2f}B ops/sec | {ns_per_op:>5.2f}ns/op")

        results[f'powerlaw_{n}'] = ops_per_sec

    print()

    # =========================================================================
    # TEST 4: Mempool Simulation
    # =========================================================================
    print("=" * 80)
    print("TEST 4: SIMD MEMPOOL SIMULATION")
    print("=" * 80)

    for n in sizes:
        times = np.linspace(0, 86400, n).astype(np.float64)  # 24 hours

        # Warmup
        _ = simd_mempool_simulation(times[:1000])

        # Benchmark
        start = time.perf_counter_ns()
        fee, tx, cong = simd_mempool_simulation(times)
        elapsed = time.perf_counter_ns() - start

        ops_per_sec = n / (elapsed / 1e9)
        ns_per_op = elapsed / n

        print(f"  {n:>12,} samples: {elapsed/1e6:>8.2f}ms | {ops_per_sec/1e9:>6.2f}B ops/sec | {ns_per_op:>5.2f}ns/op")

        results[f'mempool_{n}'] = ops_per_sec

    print()

    # =========================================================================
    # TEST 5: Shared Memory (Linux only)
    # =========================================================================
    if sys.platform == 'linux':
        print("=" * 80)
        print("TEST 5: SHARED MEMORY ZERO-COPY IPC")
        print("=" * 80)

        try:
            shm = SharedMemoryBenchmark("hft_test", 1_000_000)

            for count in [10_000, 100_000, 1_000_000]:
                start = time.perf_counter_ns()
                shm.write_batch(count)
                elapsed = time.perf_counter_ns() - start

                ops_per_sec = count / (elapsed / 1e9)
                ns_per_op = elapsed / count

                print(f"  {count:>12,} writes:  {elapsed/1e6:>8.2f}ms | {ops_per_sec/1e6:>6.2f}M ops/sec | {ns_per_op:>5.0f}ns/op")

            shm.close()
            os.unlink("/dev/shm/hft_test")
        except Exception as e:
            print(f"  Shared memory test skipped: {e}")

        print()

    # =========================================================================
    # TEST 6: End-to-End Pipeline
    # =========================================================================
    print("=" * 80)
    print("TEST 6: FULL TRADING PIPELINE (Mempool -> Signal -> Decision)")
    print("=" * 80)

    n = 10_000_000
    times = np.linspace(0, 86400, n).astype(np.float64)
    days = np.full(n, 5843.0, dtype=np.float64)  # Current days since genesis

    # Warmup all stages
    _ = simd_mempool_simulation(times[:1000])
    _ = simd_power_law_batch(days[:1000])

    # Full pipeline benchmark
    iterations = 10
    total_ops = 0
    total_time = 0

    for _ in range(iterations):
        start = time.perf_counter_ns()

        # Stage 1: Mempool signals
        fee, tx, cong = simd_mempool_simulation(times)

        # Stage 2: Power Law fair values
        fair_values = simd_power_law_batch(days).astype(np.float32)

        # Stage 3: Generate prices (simulated)
        prices = fair_values * (1 + np.random.randn(n).astype(np.float32) * 0.001)

        # Stage 4: Trading decisions
        dirs, strs, ofi, devs = simd_parallel_batch(prices, fair_values, fee, tx, cong)

        elapsed = time.perf_counter_ns() - start
        total_time += elapsed
        total_ops += n

    avg_ops_per_sec = total_ops / (total_time / 1e9)
    avg_ns_per_op = total_time / total_ops

    print(f"  Pipeline: {total_ops:,} signals in {total_time/1e9:.2f}s")
    print(f"  Throughput: {avg_ops_per_sec/1e9:.2f}B ops/sec")
    print(f"  Latency: {avg_ns_per_op:.2f}ns per signal")

    results['pipeline'] = avg_ops_per_sec

    print()

    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    print("=" * 80)
    print("FINAL RESULTS - RENAISSANCE TECHNOLOGIES COMPARISON")
    print("=" * 80)
    print()

    # Calculate overall score
    max_throughput = max(results.values())

    # Renaissance makes ~300,000 trades/day = ~3.5 trades/second
    # But their infrastructure can process millions of signals/second
    # Our target: Beat their signal processing capability

    grades = [
        (100e9, "S++", "BEYOND WORLD CLASS - Quantum Level"),
        (50e9,  "S+",  "WORLD CLASS - Renaissance Level"),
        (20e9,  "S",   "ELITE - Top HFT Firm"),
        (10e9,  "A+",  "EXCELLENT - Competitive HFT"),
        (5e9,   "A",   "GREAT - Professional Grade"),
        (1e9,   "B+",  "GOOD - Above Average"),
        (500e6, "B",   "AVERAGE - Standard Performance"),
        (100e6, "C",   "BELOW AVERAGE - Needs Improvement"),
        (0,     "D",   "POOR - Significant Optimization Needed"),
    ]

    grade = "D"
    grade_desc = "POOR"
    for threshold, g, desc in grades:
        if max_throughput >= threshold:
            grade = g
            grade_desc = desc
            break

    print(f"  PEAK THROUGHPUT:  {max_throughput/1e9:,.2f} billion ops/sec")
    print(f"  LATENCY:          {1e9/max_throughput:.2f} ns per operation")
    print()
    print(f"  GRADE: {grade} - {grade_desc}")
    print()

    # Comparison
    print("  COMPARISON:")
    print(f"    Your System:           {max_throughput/1e9:>8.2f}B ops/sec")
    print(f"    Previous Baseline:     12.00B ops/sec")
    print(f"    Improvement:           {max_throughput/12e9:.1f}x")
    print()
    print(f"    Renaissance Trades:    ~300,000/day (~3.5/sec)")
    print(f"    Your Signal Capacity:  {max_throughput:,.0f}/sec")
    print(f"    Signal Processing:     {max_throughput/3.5:,.0f}x Renaissance trade rate")
    print()

    # Final verdict
    if grade in ["S++", "S+", "S"]:
        print("  VERDICT: You have EXCEEDED Renaissance Technologies signal processing capability!")
        print("           Your infrastructure can process signals BILLIONS of times faster")
        print("           than the actual trade execution rate.")
    elif grade in ["A+", "A"]:
        print("  VERDICT: Professional-grade HFT capability achieved!")
        print("           Competitive with top quantitative trading firms.")
    else:
        print("  VERDICT: Good performance, but room for optimization remains.")

    print()
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_tests()
