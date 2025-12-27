"""
Unified Data Pipeline - Sovereign Engine
==========================================

Consolidated data source for all modes:
- Blockchain (ZMQ from Bitcoin Core)
- Historical (SQLite replay)
- Live (WebSocket from exchanges)
- Simulated (for testing)

Replaces:
- data/pipeline.py
- backtest/rentech/data_loader.py
- simulation/historical.py
"""
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Dict, Any, List
from dataclasses import dataclass, field
import time
import sqlite3
import random
from pathlib import Path

from ..core.types import Tick, DataSource
from ..core.config import DataConfig


# =============================================================================
# BASE DATA SOURCE
# =============================================================================

class BaseDataSource(ABC):
    """Abstract base class for data sources."""

    def __init__(self, config: DataConfig):
        self.config = config
        self._running = False

    @abstractmethod
    def stream(self) -> Iterator[Tick]:
        """Stream ticks from the data source."""
        pass

    def stop(self):
        """Stop streaming."""
        self._running = False


# =============================================================================
# SIMULATED DATA SOURCE
# =============================================================================

class SimulatedDataSource(BaseDataSource):
    """
    Simulated data for testing.

    Generates random walk price data.
    """

    def __init__(
        self,
        config: DataConfig,
        initial_price: float = 100000.0,
        volatility: float = 0.0005,
        tick_interval: float = 0.1
    ):
        super().__init__(config)
        self.initial_price = initial_price
        self.volatility = volatility
        self.tick_interval = tick_interval

    def stream(self) -> Iterator[Tick]:
        """Generate simulated ticks."""
        self._running = True
        price = self.initial_price

        while self._running:
            # Random walk
            price *= (1 + random.gauss(0, self.volatility))

            for symbol in self.config.symbols:
                yield Tick(
                    timestamp=time.time(),
                    source="simulated",
                    symbol=symbol,
                    price=price,
                    bid=price * 0.9999,
                    ask=price * 1.0001,
                    volume=random.uniform(0.1, 10.0),
                )

            time.sleep(self.tick_interval)


# =============================================================================
# HISTORICAL DATA SOURCE
# =============================================================================

class HistoricalDataSource(BaseDataSource):
    """
    Historical data replay from SQLite database.

    Supports:
    - Time-based replay at configurable speed
    - Date range filtering
    - Multiple symbols
    """

    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.db_path = config.historical_db
        self.replay_speed = config.replay_speed
        self.start_date = config.start_date
        self.end_date = config.end_date

    def stream(self) -> Iterator[Tick]:
        """Replay historical ticks."""
        self._running = True

        if not Path(self.db_path).exists():
            print(f"[Historical] Database not found: {self.db_path}")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv"
        params = []

        if self.start_date or self.end_date:
            conditions = []
            if self.start_date:
                conditions.append("timestamp >= ?")
                params.append(self.start_date)
            if self.end_date:
                conditions.append("timestamp <= ?")
                params.append(self.end_date)
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp"

        try:
            cursor.execute(query, params)

            last_time = None
            for row in cursor:
                if not self._running:
                    break

                timestamp, open_p, high, low, close, volume = row

                # Calculate delay for replay speed
                if last_time is not None and self.replay_speed > 0:
                    delay = (timestamp - last_time) / self.replay_speed
                    if delay > 0:
                        time.sleep(min(delay, 1.0))  # Cap at 1 second

                last_time = timestamp

                for symbol in self.config.symbols:
                    yield Tick(
                        timestamp=timestamp,
                        source="historical",
                        symbol=symbol,
                        price=close,
                        bid=close * 0.9999,
                        ask=close * 1.0001,
                        volume=volume,
                        features={
                            "open": open_p,
                            "high": high,
                            "low": low,
                            "close": close,
                        }
                    )

        finally:
            conn.close()


# =============================================================================
# BLOCKCHAIN DATA SOURCE
# =============================================================================

class BlockchainDataSource(BaseDataSource):
    """
    Real-time blockchain data via ZMQ.

    Connects to Bitcoin Core for:
    - New transactions (mempool)
    - Block confirmations
    - Exchange flow detection
    """

    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.zmq_endpoint = config.zmq_endpoint
        self._socket = None
        self._context = None

    def _initialize_zmq(self):
        """Initialize ZMQ connection."""
        try:
            import zmq

            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.connect(self.zmq_endpoint)

            # Subscribe to raw transactions
            self._socket.subscribe(b"rawtx")

            print(f"[Blockchain] Connected to {self.zmq_endpoint}")

        except ImportError:
            print("[Blockchain] ZMQ not available. Install pyzmq.")
            raise
        except Exception as e:
            print(f"[Blockchain] Failed to connect: {e}")
            raise

    def stream(self) -> Iterator[Tick]:
        """Stream ticks from blockchain."""
        self._running = True
        self._initialize_zmq()

        # We need exchange wallet database for flow detection
        exchange_wallets = self._load_exchange_wallets()

        while self._running:
            try:
                # Non-blocking receive with timeout
                if self._socket.poll(timeout=1000):  # 1 second timeout
                    topic, body, _ = self._socket.recv_multipart()

                    if topic == b"rawtx":
                        tick = self._process_transaction(body, exchange_wallets)
                        if tick:
                            yield tick

            except Exception as e:
                print(f"[Blockchain] Error: {e}")
                time.sleep(1)

        self._cleanup()

    def _load_exchange_wallets(self) -> Dict[str, str]:
        """Load exchange wallet addresses."""
        import json

        wallets = {}
        exchanges_file = Path("data/exchanges.json")

        if exchanges_file.exists():
            with open(exchanges_file, 'r') as f:
                data = json.load(f)
                for exchange, addrs in data.items():
                    for addr in addrs:
                        wallets[addr] = exchange

        return wallets

    def _process_transaction(
        self,
        raw_tx: bytes,
        exchange_wallets: Dict[str, str]
    ) -> Optional[Tick]:
        """Process a raw transaction and detect exchange flows."""
        # This is a simplified implementation
        # Full implementation would parse the transaction and check addresses

        # For now, return a placeholder tick
        return Tick(
            timestamp=time.time(),
            source="blockchain",
            symbol="BTC/USDT",
            price=0.0,  # Price to be filled from market data
            tx_hash=raw_tx[:32].hex() if len(raw_tx) >= 32 else "",
        )

    def _cleanup(self):
        """Clean up ZMQ resources."""
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()


# =============================================================================
# LIVE DATA SOURCE
# =============================================================================

class LiveDataSource(BaseDataSource):
    """
    Live market data via WebSocket.

    Connects to exchanges for real-time prices.
    """

    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.exchanges = config.websocket_exchanges
        self._ccxt_exchanges = {}

    def stream(self) -> Iterator[Tick]:
        """Stream live ticks from exchanges."""
        self._running = True

        # For synchronous mode, use CCXT's fetch_ticker
        self._initialize_exchanges()

        while self._running:
            for exchange_id, exchange in self._ccxt_exchanges.items():
                for symbol in self.config.symbols:
                    try:
                        ticker = exchange.fetch_ticker(symbol)

                        yield Tick(
                            timestamp=ticker.get('timestamp', time.time() * 1000) / 1000,
                            source="live",
                            symbol=symbol,
                            price=ticker.get('last', 0),
                            bid=ticker.get('bid', 0),
                            ask=ticker.get('ask', 0),
                            volume=ticker.get('baseVolume', 0),
                            exchange_id=exchange_id,
                        )

                    except Exception as e:
                        print(f"[Live] Error fetching {symbol} from {exchange_id}: {e}")

            time.sleep(1)  # Poll every second

    def _initialize_exchanges(self):
        """Initialize CCXT exchanges."""
        try:
            import ccxt

            for exchange_id in self.exchanges:
                try:
                    exchange_class = getattr(ccxt, exchange_id)
                    self._ccxt_exchanges[exchange_id] = exchange_class({
                        'enableRateLimit': True,
                    })
                    print(f"[Live] Initialized {exchange_id}")

                except AttributeError:
                    print(f"[Live] Unknown exchange: {exchange_id}")

        except ImportError:
            print("[Live] CCXT not available. Install ccxt.")


# =============================================================================
# UNIFIED PIPELINE
# =============================================================================

class UnifiedDataPipeline:
    """
    Unified data pipeline that routes to appropriate source.

    Factory for data sources based on configuration.
    """

    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self._source: Optional[BaseDataSource] = None

    def _create_source(self) -> BaseDataSource:
        """Create appropriate data source based on config."""
        source_type = self.config.source

        if source_type == DataSource.SIMULATED:
            return SimulatedDataSource(self.config)
        elif source_type == DataSource.HISTORICAL:
            return HistoricalDataSource(self.config)
        elif source_type == DataSource.BLOCKCHAIN:
            return BlockchainDataSource(self.config)
        elif source_type == DataSource.LIVE:
            return LiveDataSource(self.config)
        else:
            # Default to simulated
            return SimulatedDataSource(self.config)

    def stream(self) -> Iterator[Tick]:
        """Stream ticks from configured source."""
        self._source = self._create_source()
        yield from self._source.stream()

    def stop(self):
        """Stop the data pipeline."""
        if self._source:
            self._source.stop()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_data_pipeline(
    source: DataSource = DataSource.SIMULATED,
    symbols: List[str] = None,
    **kwargs
) -> UnifiedDataPipeline:
    """
    Create a data pipeline.

    Args:
        source: Data source type
        symbols: Trading symbols
        **kwargs: Additional config options

    Returns:
        Configured UnifiedDataPipeline
    """
    config = DataConfig()
    config.source = source
    config.symbols = symbols or ["BTC/USDT"]

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return UnifiedDataPipeline(config)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Unified Data Pipeline Demo")
    print("="*50)

    # Test simulated source
    pipeline = create_data_pipeline(
        source=DataSource.SIMULATED,
        symbols=["BTC/USDT"],
    )

    print("\nSimulated data stream (10 ticks):")
    for i, tick in enumerate(pipeline.stream()):
        print(f"  {i+1}: {tick.symbol} @ ${tick.price:,.2f}")
        if i >= 9:
            pipeline.stop()
            break
