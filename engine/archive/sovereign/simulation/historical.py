"""
Historical data replayer for simulation.

Replays 2009-2025 Bitcoin data from unified_bitcoin.db.
Supports speed control (real-time to instant).
"""

import sqlite3
import time
from pathlib import Path
from typing import Generator, Optional, Dict, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class HistoricalTick:
    """Single historical data point."""
    timestamp: float
    date: str
    price: float
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Optional blockchain data
    block_height: Optional[int] = None
    hash_rate: Optional[float] = None
    difficulty: Optional[float] = None
    tx_count: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'date': self.date,
            'price': self.price,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'block_height': self.block_height,
            'hash_rate': self.hash_rate,
            'difficulty': self.difficulty,
            'tx_count': self.tx_count,
        }


class HistoricalReplayer:
    """
    Replay historical data for backtesting/simulation.

    Features:
    - Load data from unified_bitcoin.db
    - Date range filtering
    - Speed control (0=instant, 1=real-time, N=Nx speed)
    - Callback-based tick processing
    """

    def __init__(self, db_path: str = "data/unified_bitcoin.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        self.current_tick: Optional[HistoricalTick] = None
        self.tick_count = 0
        self.start_time: Optional[float] = None

    def get_date_range(self) -> Tuple[str, str]:
        """Get available date range in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT MIN(date), MAX(date) FROM prices
            ''')
            row = cursor.fetchone()
            return (row[0], row[1]) if row else (None, None)

    def get_total_days(self, start_date: str = None, end_date: str = None) -> int:
        """Get total number of days in range."""
        query = 'SELECT COUNT(*) FROM prices'
        params = []

        conditions = []
        if start_date:
            conditions.append('date >= ?')
            params.append(start_date)
        if end_date:
            conditions.append('date <= ?')
            params.append(end_date)

        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchone()[0]

    def iter_ticks(
        self,
        start_date: str = None,
        end_date: str = None,
        speed: float = 0
    ) -> Generator[HistoricalTick, None, None]:
        """
        Iterate through historical ticks.

        Args:
            start_date: Start date (YYYY-MM-DD), None for earliest
            end_date: End date (YYYY-MM-DD), None for latest
            speed: 0=instant, 1=real-time (1 day = 1 day), N=Nx speed

        Yields:
            HistoricalTick for each day
        """
        self.start_time = time.time()
        self.tick_count = 0

        # Main price query
        query = '''
            SELECT
                p.date, p.open, p.high, p.low, p.close, p.volume
            FROM prices p
        '''
        params = []
        conditions = []

        if start_date:
            conditions.append('p.date >= ?')
            params.append(start_date)
        if end_date:
            conditions.append('p.date <= ?')
            params.append(end_date)

        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        query += ' ORDER BY p.date ASC'

        last_tick_time = None

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)

            for row in cursor:
                # Parse date to timestamp
                date_str = row['date']
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                timestamp = dt.timestamp()

                tick = HistoricalTick(
                    timestamp=timestamp,
                    date=date_str,
                    price=row['close'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'] or 0,
                    block_height=None,
                    hash_rate=None,
                    difficulty=None,
                    tx_count=None,
                )

                # Speed control
                if speed > 0 and last_tick_time is not None:
                    # Calculate time delta between ticks
                    tick_delta = timestamp - last_tick_time
                    # Sleep for appropriate time based on speed
                    sleep_time = tick_delta / (speed * 86400)  # Normalize to days
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                last_tick_time = timestamp
                self.current_tick = tick
                self.tick_count += 1

                yield tick

    def replay(
        self,
        callback: Callable[[HistoricalTick], None],
        start_date: str = None,
        end_date: str = None,
        speed: float = 0,
        progress_interval: int = 500
    ) -> Dict:
        """
        Replay historical data with callback.

        Args:
            callback: Function called for each tick
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            speed: 0=instant, 1=real-time, N=Nx speed
            progress_interval: Print progress every N ticks

        Returns:
            Dict with replay statistics
        """
        total_days = self.get_total_days(start_date, end_date)
        print(f"[HISTORICAL] Replaying {total_days:,} days ({start_date or 'start'} to {end_date or 'end'})")
        print(f"[HISTORICAL] Speed: {'instant' if speed == 0 else f'{speed}x'}")

        start_time = time.time()
        first_price = None
        last_price = None

        for i, tick in enumerate(self.iter_ticks(start_date, end_date, speed)):
            if first_price is None:
                first_price = tick.price
            last_price = tick.price

            # Call user callback
            callback(tick)

            # Progress update
            if progress_interval > 0 and (i + 1) % progress_interval == 0:
                elapsed = time.time() - start_time
                pct = ((i + 1) / total_days) * 100
                tps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"[HISTORICAL] Progress: {i+1:,}/{total_days:,} ({pct:.1f}%) - {tps:.0f} days/sec")

        elapsed = time.time() - start_time

        stats = {
            'total_ticks': self.tick_count,
            'elapsed_seconds': elapsed,
            'ticks_per_second': self.tick_count / elapsed if elapsed > 0 else 0,
            'start_date': start_date,
            'end_date': end_date,
            'first_price': first_price,
            'last_price': last_price,
            'price_change_pct': ((last_price / first_price) - 1) * 100 if first_price else 0,
        }

        print(f"[HISTORICAL] Complete: {self.tick_count:,} ticks in {elapsed:.1f}s")

        return stats


class IntraHistoricalReplayer:
    """
    Intraday historical replayer for higher frequency simulation.

    Uses interpolation to generate intraday ticks from daily OHLCV.
    """

    def __init__(self, db_path: str = "data/unified_bitcoin.db"):
        self.daily_replayer = HistoricalReplayer(db_path)

    def iter_intraday_ticks(
        self,
        start_date: str = None,
        end_date: str = None,
        ticks_per_day: int = 24,  # Hourly by default
        speed: float = 0
    ) -> Generator[HistoricalTick, None, None]:
        """
        Generate intraday ticks from daily data.

        Uses simple interpolation between OHLCV to simulate intraday movement:
        - First tick: Open
        - Random path through High/Low
        - Last tick: Close

        Args:
            start_date: Start date
            end_date: End date
            ticks_per_day: Number of ticks per day (default 24 for hourly)
            speed: Speed multiplier

        Yields:
            HistoricalTick for each intraday tick
        """
        import numpy as np

        for daily_tick in self.daily_replayer.iter_ticks(start_date, end_date, 0):
            # Generate intraday path
            o, h, l, c = daily_tick.open, daily_tick.high, daily_tick.low, daily_tick.close

            # Simple path: O -> H/L -> C (with some randomness)
            prices = self._generate_intraday_path(o, h, l, c, ticks_per_day)

            base_timestamp = daily_tick.timestamp
            seconds_per_tick = 86400 / ticks_per_day

            for i, price in enumerate(prices):
                tick_timestamp = base_timestamp + (i * seconds_per_tick)

                yield HistoricalTick(
                    timestamp=tick_timestamp,
                    date=daily_tick.date,
                    price=price,
                    open=daily_tick.open,
                    high=daily_tick.high,
                    low=daily_tick.low,
                    close=daily_tick.close,
                    volume=daily_tick.volume / ticks_per_day,
                    block_height=daily_tick.block_height,
                    hash_rate=daily_tick.hash_rate,
                    difficulty=daily_tick.difficulty,
                    tx_count=daily_tick.tx_count,
                )

                if speed > 0:
                    time.sleep(seconds_per_tick / (speed * 86400))

    def _generate_intraday_path(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
        num_ticks: int
    ) -> list:
        """Generate realistic intraday price path."""
        import numpy as np

        prices = [open_price]

        # Determine if bullish or bearish day
        bullish = close > open_price

        # Generate path
        # First half: move toward high (bullish) or low (bearish)
        # Second half: move toward close

        mid_point = num_ticks // 2

        if bullish:
            # Bullish: Open -> High -> Close
            peak_idx = np.random.randint(mid_point - 2, mid_point + 2)
            peak_idx = max(1, min(peak_idx, num_ticks - 2))

            # Path to high
            for i in range(1, peak_idx + 1):
                progress = i / peak_idx
                price = open_price + (high - open_price) * progress
                # Add some noise
                noise = np.random.uniform(-0.001, 0.001) * price
                price = max(low, min(high, price + noise))
                prices.append(price)

            # Path from high to close
            for i in range(1, num_ticks - peak_idx):
                progress = i / (num_ticks - peak_idx - 1) if (num_ticks - peak_idx - 1) > 0 else 1
                price = high + (close - high) * progress
                noise = np.random.uniform(-0.001, 0.001) * price
                price = max(low, min(high, price + noise))
                prices.append(price)
        else:
            # Bearish: Open -> Low -> Close
            trough_idx = np.random.randint(mid_point - 2, mid_point + 2)
            trough_idx = max(1, min(trough_idx, num_ticks - 2))

            # Path to low
            for i in range(1, trough_idx + 1):
                progress = i / trough_idx
                price = open_price + (low - open_price) * progress
                noise = np.random.uniform(-0.001, 0.001) * price
                price = max(low, min(high, price + noise))
                prices.append(price)

            # Path from low to close
            for i in range(1, num_ticks - trough_idx):
                progress = i / (num_ticks - trough_idx - 1) if (num_ticks - trough_idx - 1) > 0 else 1
                price = low + (close - low) * progress
                noise = np.random.uniform(-0.001, 0.001) * price
                price = max(low, min(high, price + noise))
                prices.append(price)

        # Ensure we have exactly num_ticks and end at close
        while len(prices) < num_ticks:
            prices.append(close)
        prices = prices[:num_ticks]
        prices[-1] = close

        return prices
