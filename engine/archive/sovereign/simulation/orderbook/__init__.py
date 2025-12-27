"""
Order Book Simulation Module
============================

Ported from hftbacktest (https://github.com/nkaz001/hftbacktest)

Provides:
- Order book data structures (L2 depth)
- Realistic execution simulation
- Queue position tracking
- Latency modeling

This enables backtesting with realistic market microstructure.
"""

from .types import (
    Level,
    OrderBookSnapshot,
    Order,
    OrderStatus,
    Fill,
    Side,
)

from .loader import (
    OrderBookLoader,
    BinanceOrderBookLoader,
    load_orderbook_data,
)

from .execution import (
    ExecutionSimulator,
    QueuePositionTracker,
    DepthWalker,
)

from .latency import (
    LatencyModel,
    BinanceLatencyModel,
    BybitLatencyModel,
    ConstantLatencyModel,
)

from .hft_backtest import (
    HFTBacktester,
    BacktestConfig,
    BacktestState,
    BacktestResult,
)

__all__ = [
    # Types
    'Level',
    'OrderBookSnapshot',
    'Order',
    'OrderStatus',
    'Fill',
    'Side',

    # Loading
    'OrderBookLoader',
    'BinanceOrderBookLoader',
    'load_orderbook_data',

    # Execution
    'ExecutionSimulator',
    'QueuePositionTracker',
    'DepthWalker',

    # Latency
    'LatencyModel',
    'BinanceLatencyModel',
    'BybitLatencyModel',
    'ConstantLatencyModel',

    # HFT Backtesting
    'HFTBacktester',
    'BacktestConfig',
    'BacktestState',
    'BacktestResult',
]
