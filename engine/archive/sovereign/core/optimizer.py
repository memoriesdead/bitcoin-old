"""
HFT Optimizer - CPU affinity and priority for trading.
"""
import os
import sys
import ctypes
import multiprocessing

class HFTOptimizer:
    """Apply CPU optimizations for low-latency trading."""

    def __init__(self, cores: list = None, verbose: bool = True):
        self.verbose = verbose
        self.is_linux = sys.platform == 'linux'
        self.is_windows = sys.platform == 'win32'
        self.cpu_count = multiprocessing.cpu_count()
        self.cores = cores or list(range(1, self.cpu_count))

    def _log(self, msg: str):
        if self.verbose:
            print(f"[HFT] {msg}")

    def set_cpu_affinity(self) -> bool:
        """Pin process to specific CPU cores."""
        try:
            if self.is_linux:
                os.sched_setaffinity(0, set(self.cores))
                self._log(f"CPU Affinity: cores {self.cores}")
                return True
            elif self.is_windows:
                kernel32 = ctypes.windll.kernel32
                mask = sum(1 << c for c in self.cores)
                if kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
                    self._log(f"CPU Affinity: cores {self.cores}")
                    return True
        except Exception as e:
            self._log(f"CPU Affinity failed: {e}")
        return False

    def set_high_priority(self) -> bool:
        """Set process to high priority."""
        try:
            if self.is_linux:
                os.nice(-10)
                self._log("Priority: HIGH (nice -10)")
                return True
            elif self.is_windows:
                kernel32 = ctypes.windll.kernel32
                if kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x80):
                    self._log("Priority: HIGH")
                    return True
        except Exception as e:
            self._log(f"Priority failed: {e}")
        return False

    def apply_all(self) -> dict:
        """Apply all optimizations."""
        return {
            'cpu_affinity': self.set_cpu_affinity(),
            'high_priority': self.set_high_priority(),
        }
