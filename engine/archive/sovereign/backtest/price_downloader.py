"""
Price Data Downloader - Binance Historical Klines

Downloads 1-minute candles for correlation with blockchain flows.
"""
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen
from urllib.error import URLError
import gzip


class PriceDownloader:
    """
    Download historical price data from Binance.

    Uses public API - no authentication needed.
    """

    BINANCE_API = "https://api.binance.com/api/v3/klines"

    def __init__(self, db_path: str = "data/historical_flows.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize price table in database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                timestamp INTEGER PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        ''')

        c.execute('CREATE INDEX IF NOT EXISTS idx_price_time ON prices(timestamp)')

        conn.commit()
        conn.close()

    def _fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        start_time: int = None,
        end_time: int = None,
        limit: int = 1000
    ) -> List:
        """Fetch klines from Binance API."""
        url = f"{self.BINANCE_API}?symbol={symbol}&interval={interval}&limit={limit}"

        if start_time:
            url += f"&startTime={start_time}"
        if end_time:
            url += f"&endTime={end_time}"

        try:
            with urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except URLError as e:
            print(f"[!] API error: {e}")
            return []

    def _save_klines(self, klines: List):
        """Save klines to database."""
        if not klines:
            return 0

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        saved = 0
        for k in klines:
            try:
                # Binance kline format:
                # [open_time, open, high, low, close, volume, close_time, ...]
                timestamp = k[0] // 1000  # Convert ms to seconds
                c.execute('''
                    INSERT OR REPLACE INTO prices
                    (timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    float(k[1]),  # open
                    float(k[2]),  # high
                    float(k[3]),  # low
                    float(k[4]),  # close
                    float(k[5])   # volume
                ))
                saved += 1
            except Exception as e:
                pass

        conn.commit()
        conn.close()
        return saved

    def download(
        self,
        start_date: str = "2020-01-01",
        end_date: str = None,
        symbol: str = "BTCUSDT"
    ):
        """
        Download historical price data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), default today
            symbol: Trading pair
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        total_days = (end_dt - start_dt).days
        current_ms = start_ms
        total_saved = 0

        print(f"\n{'='*60}")
        print(f"PRICE DATA DOWNLOADER")
        print(f"{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"Period: {start_date} to {end_date or 'now'}")
        print(f"{'='*60}\n")

        while current_ms < end_ms:
            klines = self._fetch_klines(
                symbol=symbol,
                interval="1m",
                start_time=current_ms,
                limit=1000
            )

            if not klines:
                print("[!] No data returned, waiting...")
                time.sleep(5)
                continue

            saved = self._save_klines(klines)
            total_saved += saved

            # Move to next batch
            last_time = klines[-1][0]
            current_ms = last_time + 60000  # +1 minute

            # Progress
            current_dt = datetime.fromtimestamp(current_ms / 1000)
            progress = (current_ms - start_ms) / (end_ms - start_ms) * 100

            print(f"[{current_dt.strftime('%Y-%m-%d')}] "
                  f"Progress: {progress:.1f}% | "
                  f"Saved: {total_saved:,} candles")

            # Rate limiting
            time.sleep(0.2)

        print(f"\n{'='*60}")
        print(f"DOWNLOAD COMPLETE")
        print(f"{'='*60}")
        print(f"Total candles saved: {total_saved:,}")
        print(f"Database: {self.db_path}")
        print(f"{'='*60}\n")

    def get_price_at_time(self, timestamp: int) -> Optional[float]:
        """Get price at specific timestamp."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Get closest price within 60 seconds
        c.execute('''
            SELECT close FROM prices
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY ABS(timestamp - ?)
            LIMIT 1
        ''', (timestamp - 60, timestamp + 60, timestamp))

        row = c.fetchone()
        conn.close()

        return row[0] if row else None

    def get_price_range(self, start_time: int, end_time: int) -> List[tuple]:
        """Get all prices in time range."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            SELECT timestamp, open, high, low, close, volume
            FROM prices
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        ''', (start_time, end_time))

        rows = c.fetchall()
        conn.close()
        return rows

    def get_stats(self) -> dict:
        """Get price data statistics."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        stats = {}

        c.execute('SELECT COUNT(*) FROM prices')
        stats['total_candles'] = c.fetchone()[0]

        c.execute('SELECT MIN(timestamp), MAX(timestamp) FROM prices')
        row = c.fetchone()
        if row[0]:
            stats['start_date'] = datetime.fromtimestamp(row[0]).isoformat()
            stats['end_date'] = datetime.fromtimestamp(row[1]).isoformat()
            stats['days'] = (row[1] - row[0]) / 86400

        conn.close()
        return stats


def main():
    """Download price data."""
    import argparse

    parser = argparse.ArgumentParser(description='Download Binance price data')
    parser.add_argument('--start', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair')
    args = parser.parse_args()

    downloader = PriceDownloader()
    downloader.download(
        start_date=args.start,
        end_date=args.end,
        symbol=args.symbol
    )

    stats = downloader.get_stats()
    print("\nPRICE STATISTICS:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
