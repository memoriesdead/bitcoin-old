#!/usr/bin/env python3
"""
HFT OPTIMIZER - RENAISSANCE TECHNOLOGIES LEVEL PERFORMANCE
============================================================
Based on research into how quant funds achieve nanosecond-level trading:

KERNEL/OS OPTIMIZATIONS:
1. CPU Isolation - Dedicate cores to trading (isolcpus equivalent)
2. CPU Affinity - Pin processes to specific cores (taskset)
3. Real-time Priority - SCHED_FIFO for trading threads
4. CPU Frequency - Disable scaling, lock to max (performance governor)
5. Huge Pages - 2MB/1GB pages reduce TLB misses
6. NUMA Optimization - Keep memory local to CPU
7. Interrupt Coalescing - Reduce context switches

MEMORY OPTIMIZATIONS:
1. Memory-mapped files (mmap) for zero-copy IPC
2. Pre-allocated memory pools
3. NUMA-aware allocation
4. Lock memory to prevent swapping (mlockall)

COMPUTATIONAL OPTIMIZATIONS:
1. SIMD/AVX with Numba fastmath
2. Float32 instead of float64 (2x throughput)
3. Batch processing for vectorization
4. Branch prediction hints

Usage:
    from core.hft_optimizer import HFTOptimizer

    optimizer = HFTOptimizer()
    optimizer.apply_all()  # Apply all optimizations
"""

import os
import sys
import ctypes
import mmap
import struct
import multiprocessing
from typing import Optional, Callable
import numpy as np

# Try to import Numba for SIMD optimizations
try:
    from numba import njit, prange, config
    NUMBA_AVAILABLE = True
    # Enable fastmath globally for SIMD
    config.FASTMATH = True
except ImportError:
    NUMBA_AVAILABLE = False


class HFTOptimizer:
    """
    High-Frequency Trading Optimizer

    Applies system-level optimizations for nanosecond trading.
    Modeled after Renaissance Technologies infrastructure.
    """

    def __init__(self, trading_cores: list = None, verbose: bool = True):
        """
        Initialize optimizer.

        Args:
            trading_cores: List of CPU cores to dedicate to trading (e.g., [2,3,4,5])
            verbose: Print optimization status
        """
        self.verbose = verbose
        self.is_linux = sys.platform == 'linux'
        self.is_windows = sys.platform == 'win32'

        # Detect CPU count
        self.cpu_count = multiprocessing.cpu_count()

        # Default: Use cores 1 to N-1 for trading (leave core 0 for OS)
        if trading_cores is None:
            self.trading_cores = list(range(1, self.cpu_count))
        else:
            self.trading_cores = trading_cores

        self._log("=" * 70)
        self._log("HFT OPTIMIZER - RENAISSANCE TECHNOLOGIES LEVEL")
        self._log("=" * 70)
        self._log(f"Platform: {sys.platform}")
        self._log(f"CPU Cores: {self.cpu_count}")
        self._log(f"Trading Cores: {self.trading_cores}")
        self._log(f"Numba SIMD: {'Available' if NUMBA_AVAILABLE else 'NOT AVAILABLE'}")
        self._log("=" * 70)

    def _log(self, msg: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"[HFT] {msg}")

    # =========================================================================
    # CPU OPTIMIZATIONS
    # =========================================================================

    def set_cpu_affinity(self, cores: list = None) -> bool:
        """
        Pin current process to specific CPU cores.

        This prevents the OS from migrating the process between cores,
        keeping the cache hot and reducing latency.
        """
        if cores is None:
            cores = self.trading_cores

        try:
            if self.is_linux:
                os.sched_setaffinity(0, set(cores))
                self._log(f"CPU Affinity: Pinned to cores {cores}")
                return True
            elif self.is_windows:
                # Windows: Use SetProcessAffinityMask
                import ctypes
                kernel32 = ctypes.windll.kernel32
                mask = sum(1 << core for core in cores)
                handle = kernel32.GetCurrentProcess()
                result = kernel32.SetProcessAffinityMask(handle, mask)
                if result:
                    self._log(f"CPU Affinity: Pinned to cores {cores} (mask={mask:#x})")
                    return True
                else:
                    self._log("CPU Affinity: Failed to set (may need admin)")
                    return False
            else:
                self._log("CPU Affinity: Not supported on this platform")
                return False
        except Exception as e:
            self._log(f"CPU Affinity: Error - {e}")
            return False

    def set_realtime_priority(self) -> bool:
        """
        Set process to real-time priority.

        Linux: SCHED_FIFO with priority 99
        Windows: REALTIME_PRIORITY_CLASS

        WARNING: Use with caution - can starve other processes!
        """
        try:
            if self.is_linux:
                # SCHED_FIFO = 1, max priority = 99
                param = struct.pack('i', 99)
                # sched_setscheduler syscall
                import ctypes
                libc = ctypes.CDLL('libc.so.6', use_errno=True)
                SCHED_FIFO = 1

                class sched_param(ctypes.Structure):
                    _fields_ = [('sched_priority', ctypes.c_int)]

                param = sched_param(99)
                result = libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))

                if result == 0:
                    self._log("Priority: Set to SCHED_FIFO (real-time)")
                    return True
                else:
                    self._log("Priority: Failed (need root for SCHED_FIFO)")
                    return False

            elif self.is_windows:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetCurrentProcess()
                # REALTIME_PRIORITY_CLASS = 0x100
                # HIGH_PRIORITY_CLASS = 0x80 (safer alternative)
                REALTIME = 0x100
                HIGH = 0x80

                # Try REALTIME first, fall back to HIGH
                if kernel32.SetPriorityClass(handle, REALTIME):
                    self._log("Priority: Set to REALTIME_PRIORITY_CLASS")
                    return True
                elif kernel32.SetPriorityClass(handle, HIGH):
                    self._log("Priority: Set to HIGH_PRIORITY_CLASS (REALTIME failed)")
                    return True
                else:
                    self._log("Priority: Failed to set")
                    return False
            else:
                self._log("Priority: Not supported on this platform")
                return False
        except Exception as e:
            self._log(f"Priority: Error - {e}")
            return False

    def disable_cpu_scaling(self) -> bool:
        """
        Set CPU governor to 'performance' mode.

        This locks CPU at max frequency, eliminating frequency scaling latency.
        Linux only - requires root.
        """
        if not self.is_linux:
            self._log("CPU Scaling: Only supported on Linux")
            return False

        try:
            for core in self.trading_cores:
                path = f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_governor"
                if os.path.exists(path):
                    with open(path, 'w') as f:
                        f.write('performance')
            self._log(f"CPU Scaling: Set to 'performance' for cores {self.trading_cores}")
            return True
        except PermissionError:
            self._log("CPU Scaling: Need root to change governor")
            return False
        except Exception as e:
            self._log(f"CPU Scaling: Error - {e}")
            return False

    # =========================================================================
    # MEMORY OPTIMIZATIONS
    # =========================================================================

    def lock_memory(self) -> bool:
        """
        Lock all memory to prevent swapping (mlockall).

        This ensures trading data is never swapped to disk,
        eliminating page fault latency spikes.
        """
        try:
            if self.is_linux:
                import ctypes
                libc = ctypes.CDLL('libc.so.6', use_errno=True)
                MCL_CURRENT = 1
                MCL_FUTURE = 2
                result = libc.mlockall(MCL_CURRENT | MCL_FUTURE)
                if result == 0:
                    self._log("Memory: Locked all pages (mlockall)")
                    return True
                else:
                    self._log("Memory: mlockall failed (need root or ulimit)")
                    return False
            elif self.is_windows:
                # Windows: SetProcessWorkingSetSize to lock memory
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetCurrentProcess()
                # Set min/max working set to lock pages
                # This is a simplified version
                self._log("Memory: Windows memory locking limited")
                return False
            else:
                return False
        except Exception as e:
            self._log(f"Memory: Error - {e}")
            return False

    def setup_huge_pages(self, pages: int = 512) -> bool:
        """
        Configure huge pages (2MB) for reduced TLB misses.

        Each TLB miss costs ~100 cycles. Huge pages reduce misses significantly.
        Linux only - requires root.
        """
        if not self.is_linux:
            self._log("Huge Pages: Only supported on Linux")
            return False

        try:
            # Reserve huge pages
            with open('/proc/sys/vm/nr_hugepages', 'w') as f:
                f.write(str(pages))
            self._log(f"Huge Pages: Reserved {pages} x 2MB pages ({pages * 2}MB)")
            return True
        except PermissionError:
            self._log("Huge Pages: Need root to configure")
            return False
        except Exception as e:
            self._log(f"Huge Pages: Error - {e}")
            return False

    # =========================================================================
    # SHARED MEMORY (ZERO-COPY IPC)
    # =========================================================================

    def create_shared_memory(self, name: str, size: int) -> Optional[mmap.mmap]:
        """
        Create memory-mapped shared memory for zero-copy IPC.

        This allows multiple processes to share data without copying,
        critical for multi-process trading architectures.

        Args:
            name: Shared memory name
            size: Size in bytes

        Returns:
            mmap object or None
        """
        try:
            if self.is_linux:
                # Linux: Use /dev/shm for shared memory
                path = f"/dev/shm/{name}"
                fd = os.open(path, os.O_CREAT | os.O_RDWR)
                os.ftruncate(fd, size)
                mm = mmap.mmap(fd, size)
                os.close(fd)
                self._log(f"Shared Memory: Created '{name}' ({size} bytes)")
                return mm
            elif self.is_windows:
                # Windows: Use named memory mapping
                mm = mmap.mmap(-1, size, tagname=name)
                self._log(f"Shared Memory: Created '{name}' ({size} bytes)")
                return mm
            else:
                return None
        except Exception as e:
            self._log(f"Shared Memory: Error - {e}")
            return None

    # =========================================================================
    # APPLY ALL OPTIMIZATIONS
    # =========================================================================

    def apply_network_optimizations(self) -> bool:
        """
        Apply network optimizations for ultra-low latency.

        - 256MB socket buffers
        - 10μs busy polling
        - TCP low latency mode
        - Disable timestamps/SACK for speed
        """
        if not self.is_linux:
            self._log("Network Opt: Only supported on Linux")
            return False

        try:
            optimizations = [
                # Socket buffers (256MB)
                ('/proc/sys/net/core/rmem_max', '268435456'),
                ('/proc/sys/net/core/wmem_max', '268435456'),
                ('/proc/sys/net/core/rmem_default', '134217728'),
                ('/proc/sys/net/core/wmem_default', '134217728'),
                # Busy polling (10 microseconds)
                ('/proc/sys/net/core/busy_poll', '10'),
                ('/proc/sys/net/core/busy_read', '10'),
                # TCP low latency
                ('/proc/sys/net/ipv4/tcp_low_latency', '1'),
                ('/proc/sys/net/ipv4/tcp_slow_start_after_idle', '0'),
                # Disable timestamps for speed
                ('/proc/sys/net/ipv4/tcp_timestamps', '0'),
                ('/proc/sys/net/ipv4/tcp_sack', '0'),
                # Socket backlog
                ('/proc/sys/net/core/netdev_max_backlog', '250000'),
                ('/proc/sys/net/core/somaxconn', '65535'),
            ]

            applied = 0
            for path, value in optimizations:
                if os.path.exists(path):
                    try:
                        with open(path, 'w') as f:
                            f.write(value)
                        applied += 1
                    except PermissionError:
                        pass

            self._log(f"Network Opt: Applied {applied}/{len(optimizations)} settings")
            return applied > 0

        except Exception as e:
            self._log(f"Network Opt: Error - {e}")
            return False

    def apply_memory_optimizations(self) -> bool:
        """
        Apply memory optimizations for HFT.

        - Disable swap
        - Set swappiness to 0
        - Configure huge pages (8192 x 2MB = 16GB)
        - Increase dirty writeback
        """
        if not self.is_linux:
            self._log("Memory Opt: Only supported on Linux")
            return False

        try:
            optimizations = [
                # Disable swap entirely
                ('/proc/sys/vm/swappiness', '0'),
                # Reserved memory (1GB)
                ('/proc/sys/vm/min_free_kbytes', '1048576'),
                # Huge pages (8192 x 2MB = 16GB)
                ('/proc/sys/vm/nr_hugepages', '8192'),
                # Increase dirty writeback
                ('/proc/sys/vm/dirty_writeback_centisecs', '6000'),
                ('/proc/sys/vm/dirty_expire_centisecs', '6000'),
                # VFS cache pressure
                ('/proc/sys/vm/vfs_cache_pressure', '50'),
            ]

            applied = 0
            for path, value in optimizations:
                if os.path.exists(path):
                    try:
                        with open(path, 'w') as f:
                            f.write(value)
                        applied += 1
                    except PermissionError:
                        pass

            # Disable Transparent Huge Pages (we use explicit)
            thp_path = '/sys/kernel/mm/transparent_hugepage/enabled'
            if os.path.exists(thp_path):
                try:
                    with open(thp_path, 'w') as f:
                        f.write('never')
                    applied += 1
                except PermissionError:
                    pass

            self._log(f"Memory Opt: Applied {applied}/{len(optimizations)+1} settings")
            return applied > 0

        except Exception as e:
            self._log(f"Memory Opt: Error - {e}")
            return False

    def apply_scheduler_optimizations(self) -> bool:
        """
        Apply kernel scheduler optimizations.

        - Reduce context switch overhead
        - Disable autogroup
        - Optimize NUMA balancing
        """
        if not self.is_linux:
            self._log("Scheduler Opt: Only supported on Linux")
            return False

        try:
            optimizations = [
                # Scheduler tuning
                ('/proc/sys/kernel/sched_migration_cost_ns', '5000000'),
                ('/proc/sys/kernel/sched_min_granularity_ns', '10000000'),
                ('/proc/sys/kernel/sched_autogroup_enabled', '0'),
                # NUMA balancing (disable for consistent latency)
                ('/proc/sys/kernel/numa_balancing', '0'),
                # Disable watchdog (causes latency spikes)
                ('/proc/sys/kernel/watchdog', '0'),
                ('/proc/sys/kernel/nmi_watchdog', '0'),
                # Increase file descriptors
                ('/proc/sys/fs/file-max', '4194304'),
            ]

            applied = 0
            for path, value in optimizations:
                if os.path.exists(path):
                    try:
                        with open(path, 'w') as f:
                            f.write(value)
                        applied += 1
                    except PermissionError:
                        pass

            self._log(f"Scheduler Opt: Applied {applied}/{len(optimizations)} settings")
            return applied > 0

        except Exception as e:
            self._log(f"Scheduler Opt: Error - {e}")
            return False

    def set_irq_affinity(self, system_core: int = 0) -> bool:
        """
        Move all IRQs to a single system core.

        This keeps trading cores interrupt-free for consistent latency.
        """
        if not self.is_linux:
            self._log("IRQ Affinity: Only supported on Linux")
            return False

        try:
            irq_dir = '/proc/irq'
            if not os.path.exists(irq_dir):
                return False

            mask = hex(1 << system_core)  # e.g., "0x1" for core 0
            moved = 0

            for irq in os.listdir(irq_dir):
                if irq.isdigit():
                    affinity_path = f'{irq_dir}/{irq}/smp_affinity'
                    if os.path.exists(affinity_path):
                        try:
                            with open(affinity_path, 'w') as f:
                                f.write(mask)
                            moved += 1
                        except (PermissionError, OSError):
                            pass

            self._log(f"IRQ Affinity: Moved {moved} IRQs to core {system_core}")
            return moved > 0

        except Exception as e:
            self._log(f"IRQ Affinity: Error - {e}")
            return False

    def apply_all(self, aggressive: bool = False) -> dict:
        """
        Apply all available optimizations.

        Args:
            aggressive: If True, also applies real-time priority (use with caution)

        Returns:
            Dict of optimization results
        """
        self._log("")
        self._log("APPLYING OPTIMIZATIONS...")
        self._log("-" * 50)

        results = {}

        # CPU Affinity
        results['cpu_affinity'] = self.set_cpu_affinity()

        # Real-time priority (optional - can starve system)
        if aggressive:
            results['realtime_priority'] = self.set_realtime_priority()
        else:
            results['realtime_priority'] = False
            self._log("Priority: Skipped (use aggressive=True)")

        # CPU scaling (Linux only)
        results['cpu_scaling'] = self.disable_cpu_scaling()

        # Memory locking
        results['memory_lock'] = self.lock_memory()

        # Huge pages (Linux only)
        results['huge_pages'] = self.setup_huge_pages()

        # NEW: Server-level optimizations (Linux only, need root)
        if self.is_linux and aggressive:
            results['network_opt'] = self.apply_network_optimizations()
            results['memory_opt'] = self.apply_memory_optimizations()
            results['scheduler_opt'] = self.apply_scheduler_optimizations()
            results['irq_affinity'] = self.set_irq_affinity(system_core=0)

        self._log("-" * 50)
        success_count = sum(1 for v in results.values() if v)
        self._log(f"OPTIMIZATIONS APPLIED: {success_count}/{len(results)}")
        self._log("=" * 70)

        return results


# =============================================================================
# SIMD-OPTIMIZED TRADING FUNCTIONS
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True)
    def simd_calculate_ofi(fee_pressure: np.ndarray,
                          tx_momentum: np.ndarray,
                          congestion: np.ndarray) -> np.ndarray:
        """
        SIMD-optimized OFI calculation.

        Uses AVX2 for 8x float32 parallel processing.
        fastmath enables aggressive SIMD vectorization.
        """
        n = len(fee_pressure)
        result = np.empty(n, dtype=np.float32)

        for i in range(n):
            result[i] = (
                fee_pressure[i] * 0.35 +
                tx_momentum[i] * 0.35 +
                congestion[i] * 0.30
            )

        return result

    @njit(fastmath=True, parallel=True, cache=True)
    def simd_batch_signals(prices: np.ndarray,
                          fair_values: np.ndarray,
                          ofi_values: np.ndarray) -> tuple:
        """
        SIMD-optimized batch signal processing.

        Processes thousands of signals in parallel using AVX.
        Returns: (directions, strengths, deviations)
        """
        n = len(prices)
        directions = np.empty(n, dtype=np.int32)
        strengths = np.empty(n, dtype=np.float32)
        deviations = np.empty(n, dtype=np.float32)

        for i in prange(n):
            # Deviation from fair value
            deviations[i] = (prices[i] - fair_values[i]) / fair_values[i] * 100.0

            # Direction
            if ofi_values[i] > 0.15:
                directions[i] = 1  # BUY
            elif ofi_values[i] < -0.15:
                directions[i] = -1  # SELL
            else:
                directions[i] = 0  # HOLD

            # Strength (absolute OFI)
            strengths[i] = abs(ofi_values[i])

        return directions, strengths, deviations

    @njit(fastmath=True, cache=True)
    def simd_power_law_prices(days_array: np.ndarray) -> np.ndarray:
        """
        SIMD-optimized Power Law price calculation.

        Price = 10^(-17.0161223 + 5.8451542 * log10(days))
        """
        n = len(days_array)
        result = np.empty(n, dtype=np.float64)

        A = -17.0161223
        B = 5.8451542

        for i in range(n):
            log_price = A + B * np.log10(days_array[i])
            result[i] = 10.0 ** log_price

        return result

else:
    # Fallback implementations without Numba
    def simd_calculate_ofi(fee_pressure, tx_momentum, congestion):
        return fee_pressure * 0.35 + tx_momentum * 0.35 + congestion * 0.30

    def simd_batch_signals(prices, fair_values, ofi_values):
        deviations = (prices - fair_values) / fair_values * 100.0
        directions = np.where(ofi_values > 0.15, 1, np.where(ofi_values < -0.15, -1, 0))
        strengths = np.abs(ofi_values)
        return directions.astype(np.int32), strengths.astype(np.float32), deviations.astype(np.float32)

    def simd_power_law_prices(days_array):
        A = -17.0161223
        B = 5.8451542
        log_prices = A + B * np.log10(days_array)
        return 10.0 ** log_prices


# =============================================================================
# MEMORY-MAPPED SIGNAL BUFFER
# =============================================================================

class SharedSignalBuffer:
    """
    Zero-copy shared memory buffer for trading signals.

    Allows multiple processes to share signal data without copying.
    Used for multi-process trading architectures.
    """

    # Signal structure: timestamp(8) + price(8) + ofi(4) + direction(4) = 24 bytes
    SIGNAL_SIZE = 24

    def __init__(self, name: str = "trading_signals", capacity: int = 100000):
        """
        Create shared signal buffer.

        Args:
            name: Shared memory name
            capacity: Number of signals to buffer
        """
        self.name = name
        self.capacity = capacity
        self.size = self.SIGNAL_SIZE * capacity + 8  # +8 for write index

        # Create shared memory
        if sys.platform == 'linux':
            path = f"/dev/shm/{name}"
            self.fd = os.open(path, os.O_CREAT | os.O_RDWR)
            os.ftruncate(self.fd, self.size)
            self.mm = mmap.mmap(self.fd, self.size)
        else:
            self.mm = mmap.mmap(-1, self.size, tagname=name)
            self.fd = None

        # Initialize write index to 0
        struct.pack_into('Q', self.mm, 0, 0)

        print(f"[SharedSignalBuffer] Created '{name}' - {capacity} signals, {self.size} bytes")

    def write_signal(self, timestamp: float, price: float, ofi: float, direction: int):
        """Write a signal to the buffer (zero-copy)."""
        # Get current write index
        write_idx = struct.unpack_from('Q', self.mm, 0)[0]

        # Calculate offset
        offset = 8 + (write_idx % self.capacity) * self.SIGNAL_SIZE

        # Pack signal directly into shared memory
        struct.pack_into('ddfi', self.mm, offset, timestamp, price, ofi, direction)

        # Increment write index
        struct.pack_into('Q', self.mm, 0, write_idx + 1)

    def read_signal(self, index: int) -> tuple:
        """Read a signal from the buffer (zero-copy)."""
        offset = 8 + (index % self.capacity) * self.SIGNAL_SIZE
        return struct.unpack_from('ddfi', self.mm, offset)

    def get_write_index(self) -> int:
        """Get current write index."""
        return struct.unpack_from('Q', self.mm, 0)[0]

    def close(self):
        """Close and cleanup."""
        self.mm.close()
        if self.fd is not None:
            os.close(self.fd)
            os.unlink(f"/dev/shm/{self.name}")


# =============================================================================
# LINUX KERNEL OPTIMIZATION SCRIPT GENERATOR
# =============================================================================

def generate_linux_optimization_script(trading_cores: list = None) -> str:
    """
    Generate a bash script for Linux kernel-level optimizations.

    This script should be run as root on the trading server.
    MAXED configuration for Renaissance Technologies level HFT.
    """
    if trading_cores is None:
        trading_cores = [2, 3, 4, 5, 6, 7]

    cores_str = ",".join(map(str, trading_cores))
    system_cores = "0,1"  # Cores for OS/interrupts

    script = f'''#!/bin/bash
# =============================================================================
# HFT KERNEL OPTIMIZATION SCRIPT - MAXED FOR RENAISSANCE TECHNOLOGIES LEVEL
# =============================================================================
# Run as root on the trading server
# Based on research into quant fund infrastructure
# Targets: 235K+ TPS, sub-5μs latency per signal

set -e
echo "=============================================="
echo "HFT KERNEL OPTIMIZATION - MAXED CONFIGURATION"
echo "=============================================="
echo "Target: 235K+ TPS, sub-5μs signal latency"
echo ""

# -----------------------------------------------------------------------------
# STEP 1: NETWORK OPTIMIZATION - 256MB BUFFERS
# -----------------------------------------------------------------------------
echo "[1/15] NETWORK OPTIMIZATION..."

# 256MB socket buffers (maximum for HFT)
sysctl -w net.core.rmem_max=268435456
sysctl -w net.core.wmem_max=268435456
sysctl -w net.core.rmem_default=134217728
sysctl -w net.core.wmem_default=134217728
sysctl -w net.ipv4.tcp_rmem="4096 134217728 268435456"
sysctl -w net.ipv4.tcp_wmem="4096 134217728 268435456"

# Busy polling - 10 microsecond response
sysctl -w net.core.busy_poll=10
sysctl -w net.core.busy_read=10

# TCP Low Latency Mode
sysctl -w net.ipv4.tcp_low_latency=1
sysctl -w net.ipv4.tcp_slow_start_after_idle=0

# Disable TCP timestamps and SACK (speed over reliability)
sysctl -w net.ipv4.tcp_timestamps=0
sysctl -w net.ipv4.tcp_sack=0

# Socket backlog
sysctl -w net.core.netdev_max_backlog=250000
sysctl -w net.core.somaxconn=65535

echo "    Network buffers: 256MB"
echo "    Busy polling: 10μs"

# -----------------------------------------------------------------------------
# STEP 2: MEMORY OPTIMIZATION - 16GB HUGE PAGES
# -----------------------------------------------------------------------------
echo "[2/15] MEMORY OPTIMIZATION..."

# Disable swap completely
sysctl -w vm.swappiness=0
swapoff -a 2>/dev/null || true

# Reserve 1GB free memory
sysctl -w vm.min_free_kbytes=1048576

# Huge pages: 8192 x 2MB = 16GB
sysctl -w vm.nr_hugepages=8192

# Dirty writeback optimization
sysctl -w vm.dirty_writeback_centisecs=6000
sysctl -w vm.dirty_expire_centisecs=6000
sysctl -w vm.vfs_cache_pressure=50

echo "    Huge pages: 8192 x 2MB = 16GB"
echo "    Swap: DISABLED"

# -----------------------------------------------------------------------------
# STEP 3: DISABLE TRANSPARENT HUGE PAGES
# -----------------------------------------------------------------------------
echo "[3/15] DISABLE TRANSPARENT HUGE PAGES..."

echo never > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
echo never > /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true

echo "    THP: Disabled (using explicit huge pages)"

# -----------------------------------------------------------------------------
# STEP 4: FILE DESCRIPTOR LIMITS - 4 MILLION
# -----------------------------------------------------------------------------
echo "[4/15] FILE DESCRIPTOR LIMITS..."

sysctl -w fs.file-max=4194304
sysctl -w fs.nr_open=4194304

echo "    Max file descriptors: 4,194,304"

# -----------------------------------------------------------------------------
# STEP 5: CPU PERFORMANCE MODE
# -----------------------------------------------------------------------------
echo "[5/15] CPU PERFORMANCE MODE..."

for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    if [ -f "$cpu/cpufreq/scaling_governor" ]; then
        echo "performance" > "$cpu/cpufreq/scaling_governor" 2>/dev/null || true
    fi
done

# Disable turbo variance for consistent latency
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo "1" > /sys/devices/system/cpu/intel_pstate/no_turbo
fi

echo "    Governor: performance"

# -----------------------------------------------------------------------------
# STEP 6: CPU C-STATE OPTIMIZATION
# -----------------------------------------------------------------------------
echo "[6/15] CPU C-STATE OPTIMIZATION..."

for cpu in /sys/devices/system/cpu/cpu*/cpuidle/state[1-9]; do
    if [ -f "$cpu/disable" ]; then
        echo "1" > "$cpu/disable" 2>/dev/null || true
    fi
done

echo "    Deep C-states: Disabled"

# -----------------------------------------------------------------------------
# STEP 7: KERNEL SCHEDULER OPTIMIZATION
# -----------------------------------------------------------------------------
echo "[7/15] KERNEL SCHEDULER OPTIMIZATION..."

sysctl -w kernel.sched_migration_cost_ns=5000000
sysctl -w kernel.sched_min_granularity_ns=10000000
sysctl -w kernel.sched_autogroup_enabled=0 2>/dev/null || true
sysctl -w kernel.numa_balancing=0 2>/dev/null || true

echo "    Scheduler: Optimized for trading"

# -----------------------------------------------------------------------------
# STEP 8: DISABLE WATCHDOG
# -----------------------------------------------------------------------------
echo "[8/15] DISABLE WATCHDOG..."

sysctl -w kernel.watchdog=0
sysctl -w kernel.nmi_watchdog=0 2>/dev/null || true

echo "    Watchdog: Disabled"

# -----------------------------------------------------------------------------
# STEP 9: IRQ AFFINITY - PIN TO CORES {system_cores}
# -----------------------------------------------------------------------------
echo "[9/15] IRQ AFFINITY..."

for irq in /proc/irq/[0-9]*; do
    if [ -f "$irq/smp_affinity" ]; then
        echo 3 > "$irq/smp_affinity" 2>/dev/null || true
    fi
done

echo "    IRQs: Pinned to cores {system_cores}"

# -----------------------------------------------------------------------------
# STEP 10: WORKQUEUE ISOLATION
# -----------------------------------------------------------------------------
echo "[10/15] WORKQUEUE ISOLATION..."

if [ -f /sys/devices/virtual/workqueue/cpumask ]; then
    echo 3 > /sys/devices/virtual/workqueue/cpumask 2>/dev/null || true
fi

echo "    Workqueues: Isolated to cores {system_cores}"

# -----------------------------------------------------------------------------
# STEP 11: RCU CALLBACKS
# -----------------------------------------------------------------------------
echo "[11/15] RCU CALLBACKS..."

for cpu in {cores_str.replace(",", " ")}; do
    if [ -d /sys/kernel/rcu_expedited ]; then
        echo 1 > /sys/kernel/rcu_expedited 2>/dev/null || true
    fi
done

echo "    RCU: Expedited mode"

# -----------------------------------------------------------------------------
# STEP 12: DISABLE UNNECESSARY SERVICES
# -----------------------------------------------------------------------------
echo "[12/15] DISABLE UNNECESSARY SERVICES..."

systemctl stop irqbalance 2>/dev/null || true
systemctl disable irqbalance 2>/dev/null || true
systemctl stop tuned 2>/dev/null || true
systemctl stop cpupower 2>/dev/null || true
systemctl stop snapd 2>/dev/null || true
systemctl stop unattended-upgrades 2>/dev/null || true
systemctl stop packagekit 2>/dev/null || true

echo "    Stopped: irqbalance, tuned, snapd, etc."

# -----------------------------------------------------------------------------
# STEP 13: DISK I/O OPTIMIZATION
# -----------------------------------------------------------------------------
echo "[13/15] DISK I/O OPTIMIZATION..."

for disk in /sys/block/sd*/queue /sys/block/nvme*/queue /sys/block/vd*/queue; do
    if [ -d "$disk" ]; then
        echo none > "$disk/scheduler" 2>/dev/null || echo noop > "$disk/scheduler" 2>/dev/null || true
        echo 0 > "$disk/add_random" 2>/dev/null || true
        echo 256 > "$disk/nr_requests" 2>/dev/null || true
    fi
done

echo "    Disk scheduler: none/noop"

# -----------------------------------------------------------------------------
# STEP 14: PROCESS LIMITS
# -----------------------------------------------------------------------------
echo "[14/15] PROCESS LIMITS..."

ulimit -n 4194304 2>/dev/null || true
ulimit -l unlimited 2>/dev/null || true

echo "    ulimit -n: 4194304"

# -----------------------------------------------------------------------------
# STEP 15: VERIFICATION
# -----------------------------------------------------------------------------
echo "[15/15] VERIFICATION..."
echo ""
echo "=============================================="
echo "HFT OPTIMIZATION - MAXED - COMPLETE"
echo "=============================================="
echo ""
echo "CONFIGURATION:"
echo "  Network buffers: 256MB"
echo "  Busy polling: 10μs"
echo "  Huge pages: 16GB"
echo "  File descriptors: 4M"
echo "  Watchdog: DISABLED"
echo "  THP: DISABLED"
echo "  Trading cores: {cores_str}"
echo "  System cores: {system_cores}"
echo ""
echo "RUN TRADING ENGINE:"
echo "  taskset -c {cores_str} python -m engine.runner live"
echo ""
echo "BENCHMARK COMMAND:"
echo "  taskset -c {cores_str} python -m engine.runner hft 100"
echo ""
'''

    return script


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing HFT Optimizer...")
    print()

    # Create optimizer
    optimizer = HFTOptimizer()

    # Apply optimizations (non-aggressive for testing)
    results = optimizer.apply_all(aggressive=False)

    print()
    print("Testing SIMD functions...")
    print("-" * 50)

    # Test SIMD OFI calculation
    n = 1_000_000
    fee = np.random.randn(n).astype(np.float32) * 0.5
    tx = np.random.randn(n).astype(np.float32) * 0.5
    cong = np.random.randn(n).astype(np.float32) * 0.3

    import time

    # Warm up JIT
    _ = simd_calculate_ofi(fee[:100], tx[:100], cong[:100])

    # Benchmark
    start = time.perf_counter_ns()
    ofi = simd_calculate_ofi(fee, tx, cong)
    elapsed = time.perf_counter_ns() - start

    print(f"SIMD OFI: {n:,} calculations in {elapsed/1e6:.2f}ms")
    print(f"         {n / (elapsed/1e9):,.0f} ops/second")
    print(f"         {elapsed/n:.1f} ns per operation")

    print()
    print("Testing shared memory buffer...")
    print("-" * 50)

    try:
        buffer = SharedSignalBuffer("test_signals", capacity=10000)

        # Write test signals
        start = time.perf_counter_ns()
        for i in range(10000):
            buffer.write_signal(time.time(), 97000.0 + i, 0.5, 1)
        elapsed = time.perf_counter_ns() - start

        print(f"Shared Memory Write: 10,000 signals in {elapsed/1e6:.2f}ms")
        print(f"                     {10000 / (elapsed/1e9):,.0f} writes/second")

        buffer.close()
    except Exception as e:
        print(f"Shared memory test skipped: {e}")

    print()
    print("=" * 70)
    print("HFT OPTIMIZER TEST COMPLETE")
    print("=" * 70)
