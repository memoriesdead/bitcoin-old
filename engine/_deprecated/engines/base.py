"""
BASE ENGINE - Abstract Trading Engine Interface
==============================================
All engine implementations inherit from this base class.

Import Hierarchy:
- Imports FROM: engine/core/ only
- Does NOT import from: engine/formulas/ (engines compose formulas)
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import numpy as np

from engine.core.interfaces import IEngine
from engine.core.dtypes import STATE_DTYPE, BUCKET_DTYPE, RESULT_DTYPE
from engine.core.constants.blockchain import GENESIS_TS, BLOCKS_PER_HALVING
from engine.core.constants.hft import (
    NUM_BUCKETS, CAPITAL_ALLOC_PER_TS,
    TP_BPS_PER_TS, SL_BPS_PER_TS, MAX_HOLD_TICKS,
    TICK_TIMESCALES
)


class BaseEngine(IEngine):
    """
    Abstract base class for all trading engines.

    Provides common functionality:
    - State management (prices, buckets, results)
    - Tick timing for performance monitoring
    - Stats calculation
    - Warmup handling

    Subclasses implement:
    - process_tick(): Main tick processing logic
    - _warmup(): JIT compilation warmup
    """

    __slots__ = ['state', 'buckets', 'prices', 'result', 'tick_times',
                 'tick_idx', 'start_time', 'initial_capital', '_running']

    def __init__(self, capital: float = 100.0):
        """
        Initialize base engine with capital.

        Args:
            capital: Starting capital in USD
        """
        self.initial_capital = capital
        self._running = False

        # Price buffer (1M ticks circular buffer)
        self.prices = np.ascontiguousarray(np.zeros(1000000, dtype=np.float64))

        # State array with formula values
        self.state = np.zeros(1, dtype=STATE_DTYPE)
        self.state[0]['total_capital'] = capital

        # Calculate halving cycle from current timestamp
        now = time.time()
        estimated_blocks = int((now - GENESIS_TS) / 600)
        halving_cycle = (estimated_blocks % BLOCKS_PER_HALVING) / BLOCKS_PER_HALVING
        self.state[0]['halving_cycle'] = halving_cycle

        # Bucket allocations per timescale
        self.buckets = np.zeros(NUM_BUCKETS, dtype=BUCKET_DTYPE)
        for i in range(NUM_BUCKETS):
            self.buckets[i]['capital'] = capital * CAPITAL_ALLOC_PER_TS[i]

        # Result buffer
        self.result = np.zeros(1, dtype=RESULT_DTYPE)

        # Tick timing for performance monitoring
        self.tick_times = np.zeros(10000, dtype=np.int64)
        self.tick_idx = 0
        self.start_time = time.time()

    @abstractmethod
    def process_tick(self) -> np.ndarray:
        """
        Process a single tick. Must be implemented by subclasses.

        Returns:
            Result array with tick data
        """
        pass

    @abstractmethod
    def _warmup(self):
        """
        Warmup JIT compilation. Must be implemented by subclasses.
        """
        pass

    def start(self):
        """Start the engine."""
        self._running = True
        self.start_time = time.time()

    def stop(self):
        """Stop the engine."""
        self._running = False

    def initialize(self, capital: float, **kwargs) -> None:
        """Initialize engine with capital - implements IEngine interface."""
        self.initial_capital = capital
        self.state[0]['total_capital'] = capital
        for i in range(NUM_BUCKETS):
            self.buckets[i]['capital'] = capital * CAPITAL_ALLOC_PER_TS[i]

    def get_state(self) -> Dict[str, Any]:
        """Get current engine state - implements IEngine interface."""
        return self.get_summary()

    def shutdown(self) -> None:
        """Gracefully shutdown the engine - implements IEngine interface."""
        self.stop()

    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

    def reset(self):
        """Reset engine state to initial values."""
        self.state[0]['total_capital'] = self.initial_capital
        self.state[0]['total_trades'] = 0
        self.state[0]['total_wins'] = 0
        self.state[0]['total_pnl'] = 0.0
        self.state[0]['tick_count'] = 0
        self.state[0]['last_price'] = 0.0

        for i in range(NUM_BUCKETS):
            self.buckets[i]['capital'] = self.initial_capital * CAPITAL_ALLOC_PER_TS[i]
            self.buckets[i]['position'] = 0
            self.buckets[i]['trades'] = 0
            self.buckets[i]['wins'] = 0
            self.buckets[i]['total_pnl'] = 0.0

        self.tick_idx = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine performance statistics.

        Returns:
            Dict with timing and trade statistics
        """
        n = min(self.tick_idx, 10000)
        if n == 0:
            return {}

        times = self.tick_times[:n]
        return {
            'avg_ns': np.mean(times),
            'p50_ns': np.percentile(times, 50),
            'p95_ns': np.percentile(times, 95),
            'p99_ns': np.percentile(times, 99),
            'min_ns': np.min(times),
            'max_ns': np.max(times),
            'total_ticks': self.tick_idx,
            'elapsed_s': time.time() - self.start_time,
            'tps': self.tick_idx / (time.time() - self.start_time) if self.tick_idx > 0 else 0,
        }

    def get_bucket_stats(self) -> list:
        """
        Get per-bucket performance statistics.

        Returns:
            List of dicts with bucket performance data
        """
        stats = []
        for i in range(NUM_BUCKETS):
            b = self.buckets[i]
            win_rate = b['wins'] / b['trades'] * 100 if b['trades'] > 0 else 0
            stats.append({
                'idx': i,
                'ticks': TICK_TIMESCALES[i],
                'capital': float(b['capital']),
                'position': int(b['position']),
                'trades': int(b['trades']),
                'wins': int(b['wins']),
                'win_rate': win_rate,
                'pnl': float(b['total_pnl']),
                'tp_bps': TP_BPS_PER_TS[i] * 10000,
                'sl_bps': SL_BPS_PER_TS[i] * 10000,
            })
        return stats

    def get_summary(self) -> Dict[str, Any]:
        """
        Get overall engine summary.

        Returns:
            Dict with capital, trades, win rate, PnL
        """
        trades = self.state[0]['total_trades']
        wins = self.state[0]['total_wins']
        win_rate = wins / trades * 100 if trades > 0 else 0

        return {
            'capital': float(self.state[0]['total_capital']),
            'initial_capital': self.initial_capital,
            'total_trades': int(trades),
            'total_wins': int(wins),
            'win_rate': win_rate,
            'total_pnl': float(self.state[0]['total_pnl']),
            'growth': self.state[0]['total_capital'] / self.initial_capital,
            'tick_count': int(self.state[0]['tick_count']),
        }
