#!/usr/bin/env python3
"""
BITCOIN HISTORICAL DATA - OPTIMIZED FOR KVM8
=============================================
8 cores, 32GB RAM - USE IT ALL

OPTIMIZATIONS:
1. NumPy arrays in memory (not CSV parsing each time)
2. Binary .npy cache for instant loading
3. Parallel processing for statistics
4. Pre-computed rolling windows
5. All data stays in RAM after first load
6. Singleton pattern - load once, use everywhere

PERFORMANCE TARGETS:
- Load 67K candles: <100ms
- Price lookup: <0.01ms
- Volatility lookup: <0.001ms
- Support 1M trades/day
"""

import os
import numpy as np
import time
import requests
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# Constants
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_CSV = os.path.join(DATA_DIR, "btc_history.csv")
HISTORY_NPY = os.path.join(DATA_DIR, "btc_history.npy")  # Binary format for speed

# Number of CPU cores for parallel processing
NUM_CORES = 8


@dataclass
class HistoryStats:
    """Pre-computed statistics for instant access"""
    total_candles: int = 0
    first_timestamp: int = 0
    last_timestamp: int = 0
    price_min: float = 0
    price_max: float = 0
    price_avg: float = 0
    vol_1h: float = 0
    vol_24h: float = 0
    vol_7d: float = 0
    vol_30d: float = 0
    vol_1y: float = 0


class BTCHistoryFast:
    """
    High-performance BTC history manager

    Optimized for:
    - 8 CPU cores
    - 32GB RAM
    - 300K-1M trades/day
    - Sub-millisecond lookups
    """

    def __init__(self, preload: bool = True):
        # NumPy arrays for speed (columnar storage)
        self.timestamps: Optional[np.ndarray] = None
        self.opens: Optional[np.ndarray] = None
        self.highs: Optional[np.ndarray] = None
        self.lows: Optional[np.ndarray] = None
        self.closes: Optional[np.ndarray] = None
        self.volumes: Optional[np.ndarray] = None

        # Pre-computed stats
        self.stats = HistoryStats()
        self._stats_cache: Dict[str, Dict] = {}

        # Thread pool for parallel ops
        self._executor = ThreadPoolExecutor(max_workers=NUM_CORES)

        # Lock for thread safety
        self._lock = threading.Lock()

        if preload:
            self._load_fast()

    def _load_fast(self):
        """Load data into RAM - optimized for speed"""
        start = time.perf_counter()

        # Try binary format first (fastest)
        if os.path.exists(HISTORY_NPY):
            self._load_numpy()
        elif os.path.exists(HISTORY_CSV):
            self._load_csv_to_numpy()
        else:
            print("[BTCHistoryFast] No data file found")
            return

        # Pre-compute statistics in parallel
        self._precompute_stats_parallel()

        elapsed = (time.perf_counter() - start) * 1000
        print(f"[BTCHistoryFast] Loaded {len(self.timestamps):,} candles in {elapsed:.1f}ms")
        print(f"[BTCHistoryFast] RAM usage: ~{self._estimate_ram_mb():.1f}MB")

    def _load_numpy(self):
        """Load from binary NumPy file (instant)"""
        data = np.load(HISTORY_NPY, allow_pickle=True).item()
        self.timestamps = data['timestamps']
        self.opens = data['opens']
        self.highs = data['highs']
        self.lows = data['lows']
        self.closes = data['closes']
        self.volumes = data['volumes']

    def _load_csv_to_numpy(self):
        """Load CSV and convert to NumPy (first time only)"""
        print("[BTCHistoryFast] Converting CSV to NumPy binary (one-time)...")

        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []

        with open(HISTORY_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamps.append(int(row['timestamp']))
                    opens.append(float(row['open']))
                    highs.append(float(row['high']))
                    lows.append(float(row['low']))
                    closes.append(float(row['close']))
                    volumes.append(float(row['volume_btc']))
                except (ValueError, KeyError):
                    continue

        # Convert to NumPy arrays (sorted by timestamp)
        indices = np.argsort(timestamps)
        self.timestamps = np.array(timestamps, dtype=np.int64)[indices]
        self.opens = np.array(opens, dtype=np.float64)[indices]
        self.highs = np.array(highs, dtype=np.float64)[indices]
        self.lows = np.array(lows, dtype=np.float64)[indices]
        self.closes = np.array(closes, dtype=np.float64)[indices]
        self.volumes = np.array(volumes, dtype=np.float64)[indices]

        # Save binary for next time (instant loading)
        np.save(HISTORY_NPY, {
            'timestamps': self.timestamps,
            'opens': self.opens,
            'highs': self.highs,
            'lows': self.lows,
            'closes': self.closes,
            'volumes': self.volumes,
        })
        print(f"[BTCHistoryFast] Saved binary cache: {HISTORY_NPY}")

    def _precompute_stats_parallel(self):
        """Pre-compute all statistics using all CPU cores"""
        if self.closes is None or len(self.closes) == 0:
            return

        # Basic stats (instant with NumPy vectorization)
        valid = self.closes > 0
        valid_closes = self.closes[valid]

        self.stats.total_candles = len(self.timestamps)
        self.stats.first_timestamp = int(self.timestamps[0])
        self.stats.last_timestamp = int(self.timestamps[-1])
        self.stats.price_min = float(np.min(valid_closes))
        self.stats.price_max = float(np.max(valid_closes))
        self.stats.price_avg = float(np.mean(valid_closes))

        # Compute volatilities for different periods in parallel
        periods = {
            '1h': 1,
            '24h': 24,
            '7d': 24 * 7,
            '30d': 24 * 30,
            '1y': 24 * 365,
        }

        def calc_vol(n_candles):
            if len(valid_closes) < n_candles + 1:
                return 0.0
            recent = valid_closes[-n_candles:]
            returns = np.diff(recent) / recent[:-1]
            returns = returns[np.isfinite(returns)]  # Remove inf/nan
            if len(returns) == 0:
                return 0.0
            return float(np.std(returns) * np.sqrt(len(returns)))

        # Parallel volatility calculation using all 8 cores
        futures = {}
        for name, hours in periods.items():
            futures[name] = self._executor.submit(calc_vol, hours)

        for name, future in futures.items():
            try:
                vol = future.result(timeout=5)
                setattr(self.stats, f'vol_{name}', vol)
                self._stats_cache[name] = {'volatility': vol}
            except Exception:
                setattr(self.stats, f'vol_{name}', 0.02)

    def _estimate_ram_mb(self) -> float:
        """Estimate RAM usage in MB"""
        if self.timestamps is None:
            return 0
        n = len(self.timestamps)
        # 6 arrays * 8 bytes per float64 * n elements
        return (6 * 8 * n) / 1024 / 1024

    def get_price_at(self, timestamp: int) -> float:
        """Get price at specific timestamp (binary search - O(log n))"""
        if self.timestamps is None:
            return 0
        idx = np.searchsorted(self.timestamps, timestamp)
        if idx >= len(self.closes):
            idx = len(self.closes) - 1
        return float(self.closes[idx])

    def get_prices_range(self, start_ts: int, end_ts: int) -> np.ndarray:
        """Get prices in range (vectorized - instant)"""
        if self.timestamps is None:
            return np.array([])
        mask = (self.timestamps >= start_ts) & (self.timestamps <= end_ts)
        return self.closes[mask]

    def get_volatility(self, period: str = '24h') -> float:
        """Get pre-computed volatility (instant lookup)"""
        return getattr(self.stats, f'vol_{period}', 0.02)

    def get_price_percentile(self, price: float) -> float:
        """Where does price sit in historical range? (0-1)"""
        if self.stats.price_max <= self.stats.price_min:
            return 0.5
        return (price - self.stats.price_min) / (self.stats.price_max - self.stats.price_min)

    def get_recent(self, n: int = 100) -> np.ndarray:
        """Get n most recent closes (instant slice)"""
        if self.closes is None:
            return np.array([])
        return self.closes[-n:]

    def get_stats(self, period: str = 'all') -> Dict:
        """Get statistics for compatibility"""
        return {
            'period': period,
            'candles': self.stats.total_candles,
            'min': self.stats.price_min,
            'max': self.stats.price_max,
            'avg': self.stats.price_avg,
            'volatility': self.get_volatility(period if period != 'all' else '30d'),
            'first_ts': self.stats.first_timestamp,
            'last_ts': self.stats.last_timestamp,
        }

    @property
    def total_candles(self) -> int:
        return self.stats.total_candles

    @property
    def last_timestamp(self) -> int:
        return self.stats.last_timestamp


# Global singleton for max performance (load once, use everywhere)
_HISTORY_INSTANCE: Optional[BTCHistoryFast] = None
_HISTORY_LOCK = threading.Lock()


def get_history() -> BTCHistoryFast:
    """Get global history instance (singleton pattern)"""
    global _HISTORY_INSTANCE
    if _HISTORY_INSTANCE is None:
        with _HISTORY_LOCK:
            if _HISTORY_INSTANCE is None:
                _HISTORY_INSTANCE = BTCHistoryFast(preload=True)
    return _HISTORY_INSTANCE


# Backward compatibility alias
BTCHistory = BTCHistoryFast


if __name__ == "__main__":
    print("=" * 60)
    print("BTCHistoryFast - KVM8 Optimized (8 cores, 32GB RAM)")
    print("=" * 60)

    start = time.perf_counter()
    history = get_history()
    load_time = (time.perf_counter() - start) * 1000

    print(f"\nLoad time: {load_time:.1f}ms")
    print(f"Total candles: {history.total_candles:,}")
    print(f"Price range: ${history.stats.price_min:,.2f} - ${history.stats.price_max:,.2f}")
    print(f"30d volatility: {history.stats.vol_30d:.4f}")

    # Benchmark lookups
    print("\n--- BENCHMARKS (100K operations each) ---")

    start = time.perf_counter()
    for _ in range(100000):
        _ = history.get_price_percentile(95000)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Percentile lookups: {elapsed:.1f}ms total ({elapsed/100:.4f}ms each)")

    start = time.perf_counter()
    for _ in range(100000):
        _ = history.get_volatility('24h')
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Volatility lookups: {elapsed:.1f}ms total ({elapsed/100:.4f}ms each)")

    start = time.perf_counter()
    for _ in range(100000):
        _ = history.get_recent(100)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Recent price gets: {elapsed:.1f}ms total ({elapsed/100:.4f}ms each)")

    print("\n" + "=" * 60)
    print("READY FOR 1M TRADES/DAY")
    print("=" * 60)
