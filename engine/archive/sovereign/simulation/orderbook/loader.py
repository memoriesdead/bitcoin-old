"""
Order Book Data Loader
======================

Loads order book data from various sources.
Ported from hftbacktest data loading patterns.

Supports:
- Binance Vision historical data
- Local parquet/CSV files
- SQLite databases
"""

import os
import gzip
import json
from pathlib import Path
from typing import List, Iterator, Optional, Dict, Any, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from .types import OrderBookSnapshot, Level, Trade, Side


@dataclass
class DataConfig:
    """Configuration for data loading."""
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    data_dir: str = "data/orderbook"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class OrderBookLoader:
    """
    Base class for order book data loading.

    hftbacktest pattern: Lazy loading with iterators for memory efficiency.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)

    def iter_snapshots(self) -> Iterator[OrderBookSnapshot]:
        """
        Iterate over order book snapshots.

        Yields:
            OrderBookSnapshot instances
        """
        raise NotImplementedError

    def iter_trades(self) -> Iterator[Trade]:
        """
        Iterate over public trades.

        Yields:
            Trade instances
        """
        raise NotImplementedError

    def get_snapshot_at(self, timestamp: float) -> Optional[OrderBookSnapshot]:
        """
        Get snapshot closest to timestamp.

        Args:
            timestamp: Unix timestamp

        Returns:
            Closest snapshot before timestamp
        """
        raise NotImplementedError


class BinanceOrderBookLoader(OrderBookLoader):
    """
    Loader for Binance Vision order book data.

    Data source: https://data.binance.vision/
    Downloads: bookTicker files (100ms snapshots)
    """

    def __init__(self, config: DataConfig):
        super().__init__(config)

        # Binance-specific settings
        self.depth_levels = 20  # L2 depth

    def iter_snapshots(self) -> Iterator[OrderBookSnapshot]:
        """
        Iterate over Binance order book snapshots.

        Expected file format: depth20_<date>.json.gz or .csv
        """
        # Find data files
        pattern = f"{self.config.symbol.upper()}*depth*"
        files = sorted(self.data_dir.glob(pattern))

        if not files:
            # Try alternative patterns
            files = sorted(self.data_dir.glob("*.json.gz"))
            files += sorted(self.data_dir.glob("*.csv"))

        for file_path in files:
            yield from self._load_file(file_path)

    def _load_file(self, file_path: Path) -> Iterator[OrderBookSnapshot]:
        """Load snapshots from a single file."""
        suffix = file_path.suffix.lower()

        if suffix == '.gz':
            yield from self._load_gzip(file_path)
        elif suffix == '.json':
            yield from self._load_json(file_path)
        elif suffix == '.csv':
            yield from self._load_csv(file_path)
        else:
            print(f"Unknown file format: {file_path}")

    def _load_gzip(self, file_path: Path) -> Iterator[OrderBookSnapshot]:
        """Load from gzipped JSON."""
        with gzip.open(file_path, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    yield self._parse_binance_snapshot(data)
                except json.JSONDecodeError:
                    continue

    def _load_json(self, file_path: Path) -> Iterator[OrderBookSnapshot]:
        """Load from JSON file."""
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    yield self._parse_binance_snapshot(data)
                except json.JSONDecodeError:
                    continue

    def _load_csv(self, file_path: Path) -> Iterator[OrderBookSnapshot]:
        """Load from CSV file."""
        import csv

        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield self._parse_csv_row(row)

    def _parse_binance_snapshot(self, data: Dict) -> OrderBookSnapshot:
        """Parse Binance depth snapshot format."""
        # Binance format: {"lastUpdateId": ..., "bids": [[price, qty], ...], "asks": ...}
        timestamp = data.get('E', data.get('timestamp', 0)) / 1000  # ms to s

        bids = [
            Level(price=float(b[0]), quantity=float(b[1]))
            for b in data.get('bids', data.get('b', []))
        ]

        asks = [
            Level(price=float(a[0]), quantity=float(a[1]))
            for a in data.get('asks', data.get('a', []))
        ]

        return OrderBookSnapshot(
            timestamp=timestamp,
            symbol=self.config.symbol,
            bids=bids,
            asks=asks,
            exchange="binance",
            sequence=data.get('lastUpdateId', 0),
        )

    def _parse_csv_row(self, row: Dict) -> OrderBookSnapshot:
        """Parse CSV row format."""
        timestamp = float(row.get('timestamp', 0))

        # Parse bid/ask columns
        bids = []
        asks = []

        for i in range(self.depth_levels):
            bid_price_col = f'bid_price_{i}' if f'bid_price_{i}' in row else f'bp{i}'
            bid_qty_col = f'bid_qty_{i}' if f'bid_qty_{i}' in row else f'bq{i}'
            ask_price_col = f'ask_price_{i}' if f'ask_price_{i}' in row else f'ap{i}'
            ask_qty_col = f'ask_qty_{i}' if f'ask_qty_{i}' in row else f'aq{i}'

            if bid_price_col in row and row[bid_price_col]:
                bids.append(Level(
                    price=float(row[bid_price_col]),
                    quantity=float(row.get(bid_qty_col, 0))
                ))

            if ask_price_col in row and row[ask_price_col]:
                asks.append(Level(
                    price=float(row[ask_price_col]),
                    quantity=float(row.get(ask_qty_col, 0))
                ))

        return OrderBookSnapshot(
            timestamp=timestamp,
            symbol=self.config.symbol,
            bids=bids,
            asks=asks,
            exchange="binance",
        )

    def iter_trades(self) -> Iterator[Trade]:
        """
        Iterate over Binance trades.

        Expected file format: trades_<date>.json.gz
        """
        pattern = f"{self.config.symbol.upper()}*trades*"
        files = sorted(self.data_dir.glob(pattern))

        for file_path in files:
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            yield self._parse_trade(data)
                        except json.JSONDecodeError:
                            continue

    def _parse_trade(self, data: Dict) -> Trade:
        """Parse Binance trade format."""
        return Trade(
            timestamp=data.get('T', data.get('time', 0)) / 1000,
            price=float(data.get('p', data.get('price', 0))),
            quantity=float(data.get('q', data.get('qty', 0))),
            side=Side.SELL if data.get('m', False) else Side.BUY,
            trade_id=str(data.get('t', data.get('id', ''))),
        )


class InMemoryOrderBookLoader(OrderBookLoader):
    """
    In-memory order book loader for testing and small datasets.
    """

    def __init__(self, snapshots: List[OrderBookSnapshot]):
        self.snapshots = sorted(snapshots, key=lambda s: s.timestamp)
        self._index = 0

    def iter_snapshots(self) -> Iterator[OrderBookSnapshot]:
        for snapshot in self.snapshots:
            yield snapshot

    def get_snapshot_at(self, timestamp: float) -> Optional[OrderBookSnapshot]:
        """Binary search for closest snapshot."""
        if not self.snapshots:
            return None

        # Binary search
        left, right = 0, len(self.snapshots) - 1

        while left < right:
            mid = (left + right + 1) // 2
            if self.snapshots[mid].timestamp <= timestamp:
                left = mid
            else:
                right = mid - 1

        return self.snapshots[left] if self.snapshots[left].timestamp <= timestamp else None


def load_orderbook_data(
    data_dir: str,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    max_snapshots: Optional[int] = None
) -> List[OrderBookSnapshot]:
    """
    Convenience function to load order book data.

    Args:
        data_dir: Directory containing data files
        symbol: Trading symbol
        exchange: Exchange name
        start_date: Filter start date
        end_date: Filter end date
        max_snapshots: Maximum snapshots to load

    Returns:
        List of OrderBookSnapshot
    """
    config = DataConfig(
        symbol=symbol,
        exchange=exchange,
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
    )

    if exchange.lower() == "binance":
        loader = BinanceOrderBookLoader(config)
    else:
        loader = BinanceOrderBookLoader(config)  # Default to Binance format

    snapshots = []
    for snapshot in loader.iter_snapshots():
        # Filter by date if specified
        if start_date:
            snapshot_dt = datetime.fromtimestamp(snapshot.timestamp)
            if snapshot_dt < start_date:
                continue
            if end_date and snapshot_dt > end_date:
                break

        snapshots.append(snapshot)

        if max_snapshots and len(snapshots) >= max_snapshots:
            break

    return snapshots


def generate_synthetic_orderbook(
    n_snapshots: int = 1000,
    initial_price: float = 42000.0,
    volatility: float = 0.001,
    spread_bps: float = 1.0,
    depth_levels: int = 10,
    start_timestamp: float = None
) -> List[OrderBookSnapshot]:
    """
    Generate synthetic order book data for testing.

    Args:
        n_snapshots: Number of snapshots to generate
        initial_price: Starting mid price
        volatility: Price volatility (per tick)
        spread_bps: Spread in basis points
        depth_levels: Number of price levels
        start_timestamp: Starting timestamp (default: now)

    Returns:
        List of synthetic OrderBookSnapshot
    """
    np.random.seed(42)

    if start_timestamp is None:
        start_timestamp = datetime.now().timestamp()

    snapshots = []
    price = initial_price

    for i in range(n_snapshots):
        # Random walk price
        price *= (1 + np.random.randn() * volatility)

        # Calculate spread
        half_spread = price * (spread_bps / 10000) / 2

        # Generate levels
        bids = []
        asks = []

        bid_price = price - half_spread
        ask_price = price + half_spread

        for level in range(depth_levels):
            # Random quantity, decreasing with distance
            bid_qty = np.random.exponential(1.0) * (1 - level * 0.05)
            ask_qty = np.random.exponential(1.0) * (1 - level * 0.05)

            bids.append(Level(
                price=round(bid_price - level * 1.0, 2),
                quantity=round(max(0.01, bid_qty), 4)
            ))
            asks.append(Level(
                price=round(ask_price + level * 1.0, 2),
                quantity=round(max(0.01, ask_qty), 4)
            ))

        snapshot = OrderBookSnapshot(
            timestamp=start_timestamp + i * 0.1,  # 100ms intervals
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            exchange="synthetic",
            sequence=i,
        )
        snapshots.append(snapshot)

    return snapshots


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("Order Book Loader Demo")
    print("=" * 50)

    # Generate synthetic data
    snapshots = generate_synthetic_orderbook(
        n_snapshots=100,
        initial_price=42000.0,
        volatility=0.0005,
        spread_bps=2.0,
    )

    print(f"Generated {len(snapshots)} synthetic snapshots")
    print(f"\nFirst snapshot:")
    first = snapshots[0]
    print(f"  Timestamp: {first.timestamp}")
    print(f"  Mid Price: {first.mid_price:.2f}")
    print(f"  Spread: {first.spread_bps:.2f} bps")
    print(f"  Bid Depth: {first.get_bid_depth():.4f}")
    print(f"  Ask Depth: {first.get_ask_depth():.4f}")
    print(f"  Imbalance: {first.get_imbalance():.4f}")

    # Test in-memory loader
    loader = InMemoryOrderBookLoader(snapshots)

    # Find snapshot at specific time
    mid_ts = snapshots[50].timestamp
    found = loader.get_snapshot_at(mid_ts)
    print(f"\nSnapshot at {mid_ts}: mid_price={found.mid_price:.2f}")
