"""
Unified Data Pipeline - Combines all Bitcoin data sources from genesis to present.

Data Sources:
1. ORBITAAL (2009-01-09 to 2021-01-25): Transaction-level parquet files
2. Mempool.space download (2021-01-26 to present): Block-level features
3. Bitcoin Core RPC (optional): Live transaction-level scanning

This pipeline creates a unified SQLite database for backtesting and
live trading signals.
"""
import sqlite3
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, date, timedelta
import json

from .orbitaal_loader import OrbitaalLoader
from .btc_scanner import BitcoinCoreScanner, FastBlockScanner


class UnifiedDataPipeline:
    """
    Unified Bitcoin data pipeline from genesis to present.

    Combines:
    - ORBITAAL daily features (2009-2021)
    - Downloaded block features (2021-2025)
    - Optional live Bitcoin Core scanning

    Usage:
        pipeline = UnifiedDataPipeline()
        pipeline.build()

        # Query unified data
        stats = pipeline.get_daily_stats("2023-01-01", "2023-12-31")
    """

    def __init__(
        self,
        orbitaal_dir: str = "data/orbitaal",
        downloaded_db: str = "data/bitcoin_2021_2025.db",
        output_db: str = "data/unified_bitcoin.db"
    ):
        self.orbitaal_dir = Path(orbitaal_dir)
        self.downloaded_db = Path(downloaded_db)
        self.output_db = Path(output_db)

        self.orbitaal_loader = OrbitaalLoader(orbitaal_dir)

    def build(self, force_rebuild: bool = False) -> Dict:
        """
        Build the unified database from all sources.

        Steps:
        1. Process ORBITAAL daily features (2009-2021)
        2. Import downloaded block data (2021-2025)
        3. Aggregate into unified daily features

        Returns: Dict with stats about the build
        """
        self.output_db.parent.mkdir(parents=True, exist_ok=True)

        if force_rebuild and self.output_db.exists():
            self.output_db.unlink()

        conn = sqlite3.connect(self.output_db)
        self._create_schema(conn)

        stats = {
            'orbitaal_days': 0,
            'downloaded_blocks': 0,
            'total_days': 0,
            'date_range': None
        }

        # Step 1: Import ORBITAAL data
        print("=" * 60)
        print("STEP 1: Processing ORBITAAL data (2009-2021)")
        print("=" * 60)
        stats['orbitaal_days'] = self._import_orbitaal(conn)

        # Step 2: Import downloaded block data
        print("\n" + "=" * 60)
        print("STEP 2: Importing downloaded block data (2021-2025)")
        print("=" * 60)
        stats['downloaded_blocks'] = self._import_downloaded_blocks(conn)

        # Step 3: Aggregate daily features from blocks
        print("\n" + "=" * 60)
        print("STEP 3: Aggregating daily features")
        print("=" * 60)
        self._aggregate_daily_from_blocks(conn)

        # Get final stats
        c = conn.cursor()
        c.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM daily_features")
        row = c.fetchone()
        stats['date_range'] = (row[0], row[1])
        stats['total_days'] = row[2]

        conn.close()

        print("\n" + "=" * 60)
        print("BUILD COMPLETE")
        print("=" * 60)
        print(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print(f"Total days: {stats['total_days']:,}")
        print(f"Output: {self.output_db}")

        return stats

    def _create_schema(self, conn: sqlite3.Connection):
        """Create database schema."""
        c = conn.cursor()

        # Daily aggregated features (main table for trading signals)
        c.execute("""
            CREATE TABLE IF NOT EXISTS daily_features (
                date TEXT PRIMARY KEY,
                timestamp INTEGER,
                source TEXT,
                blocks INTEGER,
                tx_count INTEGER,
                total_value_btc REAL,
                total_value_usd REAL,
                unique_senders INTEGER,
                unique_receivers INTEGER,
                whale_tx_count INTEGER,
                whale_value_btc REAL,
                avg_block_size REAL,
                avg_block_fullness REAL
            )
        """)

        # Block-level features (for detailed analysis)
        c.execute("""
            CREATE TABLE IF NOT EXISTS block_features (
                height INTEGER PRIMARY KEY,
                timestamp INTEGER,
                hash TEXT,
                tx_count INTEGER,
                size INTEGER,
                weight INTEGER,
                fees_btc REAL,
                median_fee_rate REAL
            )
        """)

        # Metadata table
        c.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        c.execute("CREATE INDEX IF NOT EXISTS idx_daily_timestamp ON daily_features(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_block_timestamp ON block_features(timestamp)")

        conn.commit()

    def _import_orbitaal(self, conn: sqlite3.Connection) -> int:
        """Import ORBITAAL daily features."""
        c = conn.cursor()

        # Check if already imported
        c.execute("SELECT COUNT(*) FROM daily_features WHERE source = 'orbitaal'")
        existing = c.fetchone()[0]
        if existing > 0:
            print(f"ORBITAAL data already imported ({existing:,} days)")
            return existing

        imported = 0

        try:
            for file_date, file_path in self.orbitaal_loader.iter_daily_files():
                try:
                    edges = self.orbitaal_loader.read_daily_edges(file_path)
                    stats = self.orbitaal_loader.compute_daily_stats(edges)

                    c.execute("""
                        INSERT OR REPLACE INTO daily_features
                        (date, timestamp, source, blocks, tx_count, total_value_btc,
                         total_value_usd, unique_senders, unique_receivers,
                         whale_tx_count, whale_value_btc, avg_block_size, avg_block_fullness)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        str(file_date),
                        int(datetime.combine(file_date, datetime.min.time()).timestamp()),
                        'orbitaal',
                        None,  # blocks not tracked in ORBITAAL
                        stats['tx_count'],
                        stats['total_value_btc'],
                        stats['total_value_usd'],
                        stats['unique_senders'],
                        stats['unique_receivers'],
                        stats['whale_tx_count'],
                        stats['whale_value_btc'],
                        None,
                        None,
                    ))

                    imported += 1

                    if imported % 100 == 0:
                        conn.commit()
                        print(f"  Processed {imported:,} days... (current: {file_date})")

                except Exception as e:
                    print(f"  Error processing {file_date}: {e}")
                    continue

        except FileNotFoundError:
            print("  ORBITAAL data directory not found, skipping...")

        conn.commit()
        print(f"  Imported {imported:,} days from ORBITAAL")
        return imported

    def _import_downloaded_blocks(self, conn: sqlite3.Connection) -> int:
        """Import block features from downloaded database."""
        c = conn.cursor()

        if not self.downloaded_db.exists():
            print(f"  Downloaded database not found: {self.downloaded_db}")
            return 0

        # Check if already imported
        c.execute("SELECT COUNT(*) FROM block_features")
        existing = c.fetchone()[0]
        if existing > 0:
            print(f"  Block data already imported ({existing:,} blocks)")
            return existing

        # Connect to source database
        src_conn = sqlite3.connect(self.downloaded_db)
        src_c = src_conn.cursor()

        # Import blocks
        src_c.execute("SELECT * FROM blocks ORDER BY height")
        imported = 0

        for row in src_c:
            c.execute("""
                INSERT OR REPLACE INTO block_features
                (height, timestamp, hash, tx_count, size, weight, fees_btc, median_fee_rate)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                row[0],  # height
                row[1],  # timestamp
                row[2],  # hash
                row[3],  # tx_count
                row[4],  # size
                row[5],  # weight
                row[6] / 1e8 if row[6] else 0,  # fees (satoshi -> btc)
                row[7],  # median_fee
            ))
            imported += 1

            if imported % 10000 == 0:
                conn.commit()
                print(f"  Imported {imported:,} blocks...")

        src_conn.close()
        conn.commit()

        print(f"  Imported {imported:,} blocks")
        return imported

    def _aggregate_daily_from_blocks(self, conn: sqlite3.Connection):
        """Aggregate block features into daily features."""
        c = conn.cursor()

        # Find date range from blocks
        c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM block_features")
        row = c.fetchone()
        if not row[0]:
            return

        start_ts = row[0]
        end_ts = row[1]

        start_date = datetime.fromtimestamp(start_ts).date()
        end_date = datetime.fromtimestamp(end_ts).date()

        current = start_date
        aggregated = 0

        while current <= end_date:
            day_start = int(datetime.combine(current, datetime.min.time()).timestamp())
            day_end = day_start + 86400

            # Skip if already have ORBITAAL data for this day
            c.execute("SELECT 1 FROM daily_features WHERE date = ? AND source = 'orbitaal'", (str(current),))
            if c.fetchone():
                current += timedelta(days=1)
                continue

            # Aggregate from blocks
            c.execute("""
                SELECT
                    COUNT(*) as blocks,
                    SUM(tx_count) as tx_count,
                    AVG(size) as avg_size,
                    AVG(CAST(size AS FLOAT) / (weight / 4.0)) as avg_fullness,
                    SUM(fees_btc) as total_fees
                FROM block_features
                WHERE timestamp >= ? AND timestamp < ?
            """, (day_start, day_end))

            row = c.fetchone()
            if row[0] > 0:
                c.execute("""
                    INSERT OR REPLACE INTO daily_features
                    (date, timestamp, source, blocks, tx_count, total_value_btc,
                     total_value_usd, unique_senders, unique_receivers,
                     whale_tx_count, whale_value_btc, avg_block_size, avg_block_fullness)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    str(current),
                    day_start,
                    'downloaded',
                    row[0],  # blocks
                    row[1],  # tx_count
                    None,    # total_value_btc (not available from block-level)
                    None,    # total_value_usd
                    None,    # unique_senders
                    None,    # unique_receivers
                    None,    # whale_tx_count
                    None,    # whale_value_btc
                    row[2],  # avg_block_size
                    row[3],  # avg_block_fullness
                ))
                aggregated += 1

            current += timedelta(days=1)

        conn.commit()
        print(f"  Aggregated {aggregated:,} days from block data")

    def get_daily_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Query daily statistics from the unified database.

        Returns: List of daily stat dictionaries
        """
        conn = sqlite3.connect(self.output_db)
        c = conn.cursor()

        query = "SELECT * FROM daily_features WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"
        c.execute(query, params)

        columns = [d[0] for d in c.description]
        results = [dict(zip(columns, row)) for row in c.fetchall()]

        conn.close()
        return results

    def get_coverage(self) -> Dict:
        """Get data coverage information."""
        conn = sqlite3.connect(self.output_db)
        c = conn.cursor()

        result = {
            'daily_features': {},
            'block_features': {}
        }

        # Daily features by source
        c.execute("""
            SELECT source, MIN(date), MAX(date), COUNT(*)
            FROM daily_features
            GROUP BY source
        """)
        for row in c.fetchall():
            result['daily_features'][row[0]] = {
                'start': row[1],
                'end': row[2],
                'count': row[3]
            }

        # Block features
        c.execute("""
            SELECT MIN(height), MAX(height), COUNT(*),
                   MIN(timestamp), MAX(timestamp)
            FROM block_features
        """)
        row = c.fetchone()
        if row[0]:
            result['block_features'] = {
                'min_height': row[0],
                'max_height': row[1],
                'count': row[2],
                'start_date': datetime.fromtimestamp(row[3]).date().isoformat() if row[3] else None,
                'end_date': datetime.fromtimestamp(row[4]).date().isoformat() if row[4] else None,
            }

        conn.close()
        return result

    def export_for_backtest(
        self,
        output_file: str = "data/backtest_features.csv"
    ) -> int:
        """
        Export daily features to CSV for backtesting.

        Returns: Number of rows exported
        """
        import csv

        conn = sqlite3.connect(self.output_db)
        c = conn.cursor()

        c.execute("SELECT * FROM daily_features ORDER BY date")
        columns = [d[0] for d in c.description]

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            rows = c.fetchall()
            writer.writerows(rows)

        conn.close()
        print(f"Exported {len(rows):,} rows to {output_file}")
        return len(rows)


def build_pipeline():
    """Build the unified data pipeline."""
    pipeline = UnifiedDataPipeline()
    stats = pipeline.build()
    return stats


def quick_test():
    """Quick test of the pipeline."""
    print("Testing Unified Data Pipeline")
    print("=" * 60)

    pipeline = UnifiedDataPipeline()

    # Check if output exists
    if pipeline.output_db.exists():
        coverage = pipeline.get_coverage()
        print("\nCurrent coverage:")
        print(json.dumps(coverage, indent=2, default=str))

        # Sample query
        print("\nSample data (last 7 days):")
        stats = pipeline.get_daily_stats(start_date="2024-01-01", end_date="2024-01-07")
        for s in stats[:5]:
            print(f"  {s['date']}: {s['tx_count']:,} txs")
    else:
        print(f"No unified database found at {pipeline.output_db}")
        print("Run 'pipeline.build()' to create it.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_pipeline()
    else:
        quick_test()
