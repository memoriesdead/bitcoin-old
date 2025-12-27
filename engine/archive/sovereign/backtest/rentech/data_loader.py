"""
RenTech Data Loader - Unified data from 2009-2025

Loads data from unified_bitcoin.db which combines:
- ORBITAAL (2009-2021): Transaction-level features
- Downloaded blocks (2021-2025): Block-level features
"""
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class DailyFeature:
    """Single day of features."""
    date: str
    timestamp: int
    source: str  # 'orbitaal' or 'downloaded'

    # Core metrics (available in both sources)
    tx_count: int

    # ORBITAAL-only (None for downloaded)
    total_value_btc: Optional[float]
    total_value_usd: Optional[float]
    unique_senders: Optional[int]
    unique_receivers: Optional[int]
    whale_tx_count: Optional[int]
    whale_value_btc: Optional[float]

    # Block-only (None for ORBITAAL)
    blocks: Optional[int]
    avg_block_size: Optional[float]
    avg_block_fullness: Optional[float]


class RentechDataLoader:
    """
    Load unified Bitcoin data for backtesting.

    Handles the transition between ORBITAAL (transaction-level) and
    downloaded (block-level) data seamlessly.
    """

    def __init__(self, db_path: str = "data/unified_bitcoin.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get_data_range(self) -> Tuple[str, str]:
        """Get the date range of available data."""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT MIN(date), MAX(date) FROM daily_features")
        row = c.fetchone()
        conn.close()
        return row[0], row[1]

    def load_daily_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load daily features as a pandas DataFrame.

        Args:
            start_date: Start date (YYYY-MM-DD), defaults to beginning
            end_date: End date (YYYY-MM-DD), defaults to end

        Returns:
            DataFrame with columns:
            - date, timestamp, source
            - tx_count (all rows)
            - total_value_btc, whale_tx_count, etc. (ORBITAAL only)
            - blocks, avg_block_fullness (downloaded only)
        """
        conn = self._get_conn()

        query = "SELECT * FROM daily_features WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Convert timestamp to datetime for easier manipulation
        df['datetime'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year

        return df

    def load_prices(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load BTC price data.

        Returns DataFrame with date, open, high, low, close, volume.
        """
        conn = self._get_conn()

        # Check if prices table exists
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prices'")
        if not c.fetchone():
            conn.close()
            return self._download_prices(start_date, end_date)

        query = "SELECT * FROM prices ORDER BY date"
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return self._download_prices(start_date, end_date)

        # Ensure date column exists (handle both old timestamp and new date formats)
        if 'date' not in df.columns and 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')

        return df

    def _download_prices(
        self,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Download price data from API if not in database."""
        import requests

        print("Downloading price data from Coinbase...")

        # Use Coinbase API (no geo-restrictions)
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

        all_prices = []

        # Coinbase returns max 300 candles per request
        # For daily data, we need multiple requests
        end_ts = int(datetime.now().timestamp())

        while True:
            params = {
                'granularity': 86400,  # Daily
                'end': end_ts,
            }

            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code != 200:
                    break

                data = resp.json()
                if not data:
                    break

                for candle in data:
                    # Coinbase format: [timestamp, low, high, open, close, volume]
                    all_prices.append({
                        'timestamp': candle[0],
                        'open': candle[3],
                        'high': candle[2],
                        'low': candle[1],
                        'close': candle[4],
                        'volume': candle[5],
                    })

                # Move end timestamp back
                end_ts = min(c[0] for c in data) - 86400

                if len(data) < 300:
                    break

            except Exception as e:
                print(f"Error downloading prices: {e}")
                break

        if not all_prices:
            return pd.DataFrame()

        df = pd.DataFrame(all_prices)
        df = df.sort_values('timestamp').drop_duplicates('timestamp')
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')

        # Save to database
        conn = self._get_conn()
        df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_sql(
            'prices', conn, if_exists='replace', index=False
        )
        conn.close()

        print(f"Downloaded {len(df)} days of price data")
        return df

    def load_merged_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load features merged with prices.

        Returns DataFrame ready for backtesting with:
        - All daily features
        - Price data (close price for each day)
        - Future returns for trade outcome calculation
        """
        features = self.load_daily_features(start_date, end_date)
        prices = self.load_prices(start_date, end_date)

        if prices.empty:
            raise ValueError("No price data available")

        # Merge on date
        df = features.merge(
            prices[['date', 'open', 'high', 'low', 'close', 'volume']],
            on='date',
            how='left'
        )

        # Calculate future returns for trade outcomes
        for days in [1, 3, 5, 7, 10]:
            df[f'return_{days}d'] = df['close'].shift(-days) / df['close'] - 1

        # Forward fill any missing prices (weekends in early data)
        df['close'] = df['close'].ffill()

        return df

    def get_summary(self) -> Dict:
        """Get summary statistics about the data."""
        conn = self._get_conn()
        c = conn.cursor()

        # Count by source
        c.execute("""
            SELECT source, MIN(date), MAX(date), COUNT(*),
                   AVG(tx_count), SUM(tx_count)
            FROM daily_features
            GROUP BY source
        """)

        sources = {}
        for row in c.fetchall():
            sources[row[0]] = {
                'start': row[1],
                'end': row[2],
                'days': row[3],
                'avg_tx_count': row[4],
                'total_tx_count': row[5],
            }

        # Total coverage
        c.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM daily_features")
        total = c.fetchone()

        conn.close()

        return {
            'total_start': total[0],
            'total_end': total[1],
            'total_days': total[2],
            'sources': sources,
        }


def quick_test():
    """Quick test of data loader."""
    loader = RentechDataLoader()

    print("Data Summary:")
    summary = loader.get_summary()
    print(f"  Range: {summary['total_start']} to {summary['total_end']}")
    print(f"  Total days: {summary['total_days']:,}")

    for source, info in summary['sources'].items():
        print(f"\n  {source}:")
        print(f"    Range: {info['start']} to {info['end']}")
        print(f"    Days: {info['days']:,}")
        print(f"    Avg TX/day: {info['avg_tx_count']:,.0f}")

    # Load sample data
    print("\nLoading merged data...")
    df = loader.load_merged_data()
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Show sample
    print("\nSample (2020):")
    sample = df[df['year'] == 2020].head()
    print(sample[['date', 'source', 'tx_count', 'close']].to_string())


if __name__ == "__main__":
    quick_test()
