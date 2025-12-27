"""
ORBITAAL Dataset Loader - Transaction-level Bitcoin data from genesis to 2021.

The ORBITAAL dataset contains:
- NODE_TABLE: 364 million Bitcoin addresses with IDs
- SNAPSHOT/EDGES/day: Daily transaction edges (SRC_ID -> DST_ID, VALUE)
- Coverage: 2009-01-09 to 2021-01-25

This loader efficiently reads the parquet files and provides:
- Daily aggregations for trading signals
- Address-level flow analysis (with node table lookup)
- Exchange flow detection
"""
import sqlite3
from pathlib import Path
from typing import Optional, Iterator, Dict, List
from datetime import datetime, date
import pyarrow.parquet as pq
import pandas as pd
import numpy as np


class OrbitaalLoader:
    """
    Load and query the ORBITAAL Bitcoin transaction dataset.

    The dataset is too large to load entirely into memory (100+ GB),
    so we stream through daily files and aggregate as needed.
    """

    def __init__(self, data_dir: str = "data/orbitaal"):
        self.data_dir = Path(data_dir)
        self.edges_dir = self.data_dir / "SNAPSHOT" / "EDGES" / "day"
        self.node_table_path = self.data_dir / "NODE_TABLE" / "orbitaal-nodetable.snappy.parquet"

        # Cache for exchange address IDs (loaded lazily)
        self._exchange_ids: Optional[set] = None

    def get_coverage(self) -> tuple[date, date]:
        """Get the date range covered by ORBITAAL data."""
        files = sorted(self.edges_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {self.edges_dir}")

        # Parse dates from filenames
        first = files[0].stem.split("-date-")[1].split("-file")[0]
        last = files[-1].stem.split("-date-")[1].split("-file")[0]

        return (
            datetime.strptime(first, "%Y-%m-%d").date(),
            datetime.strptime(last, "%Y-%m-%d").date()
        )

    def iter_daily_files(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Iterator[tuple[date, Path]]:
        """
        Iterate through daily edge files within date range.

        Yields: (date, path) tuples
        """
        files = sorted(self.edges_dir.glob("*.parquet"))

        for f in files:
            # Extract date from filename: orbitaal-snapshot-date-YYYY-MM-DD-file-id-XXXX
            date_str = f.stem.split("-date-")[1].split("-file")[0]
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()

            if start_date and file_date < datetime.strptime(start_date, "%Y-%m-%d").date():
                continue
            if end_date and file_date > datetime.strptime(end_date, "%Y-%m-%d").date():
                break

            yield file_date, f

    def read_daily_edges(self, file_path: Path) -> pd.DataFrame:
        """
        Read a single day's transaction edges.

        Returns DataFrame with: SRC_ID, DST_ID, VALUE_SATOSHI, VALUE_USD
        """
        return pd.read_parquet(file_path)

    def compute_daily_stats(self, edges_df: pd.DataFrame) -> Dict:
        """
        Compute daily statistics from edge data.

        Returns dict with trading-relevant metrics.
        """
        return {
            'tx_count': len(edges_df),
            'total_value_btc': edges_df['VALUE_SATOSHI'].sum() / 1e8,
            'total_value_usd': edges_df['VALUE_USD'].sum(),
            'unique_senders': edges_df['SRC_ID'].nunique(),
            'unique_receivers': edges_df['DST_ID'].nunique(),
            'avg_tx_btc': edges_df['VALUE_SATOSHI'].mean() / 1e8,
            'median_tx_btc': edges_df['VALUE_SATOSHI'].median() / 1e8,
            'max_tx_btc': edges_df['VALUE_SATOSHI'].max() / 1e8,
            # Whale detection (>100 BTC)
            'whale_tx_count': (edges_df['VALUE_SATOSHI'] > 100e8).sum(),
            'whale_value_btc': edges_df[edges_df['VALUE_SATOSHI'] > 100e8]['VALUE_SATOSHI'].sum() / 1e8,
        }

    def build_daily_features_db(
        self,
        output_db: str = "data/orbitaal_daily.db",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback=None
    ) -> int:
        """
        Build SQLite database of daily features from ORBITAAL data.

        This processes all parquet files and creates an easily queryable
        database for backtesting.

        Returns: Number of days processed
        """
        output_path = Path(output_db)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(output_path)
        c = conn.cursor()

        # Create table
        c.execute("""
            CREATE TABLE IF NOT EXISTS daily_features (
                date TEXT PRIMARY KEY,
                timestamp INTEGER,
                tx_count INTEGER,
                total_value_btc REAL,
                total_value_usd REAL,
                unique_senders INTEGER,
                unique_receivers INTEGER,
                avg_tx_btc REAL,
                median_tx_btc REAL,
                max_tx_btc REAL,
                whale_tx_count INTEGER,
                whale_value_btc REAL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON daily_features(timestamp)")
        conn.commit()

        processed = 0
        for file_date, file_path in self.iter_daily_files(start_date, end_date):
            # Skip if already processed
            c.execute("SELECT 1 FROM daily_features WHERE date = ?", (str(file_date),))
            if c.fetchone():
                continue

            try:
                edges = self.read_daily_edges(file_path)
                stats = self.compute_daily_stats(edges)

                c.execute("""
                    INSERT OR REPLACE INTO daily_features VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    str(file_date),
                    int(datetime.combine(file_date, datetime.min.time()).timestamp()),
                    stats['tx_count'],
                    stats['total_value_btc'],
                    stats['total_value_usd'],
                    stats['unique_senders'],
                    stats['unique_receivers'],
                    stats['avg_tx_btc'],
                    stats['median_tx_btc'],
                    stats['max_tx_btc'],
                    stats['whale_tx_count'],
                    stats['whale_value_btc'],
                ))

                processed += 1

                if processed % 100 == 0:
                    conn.commit()
                    if progress_callback:
                        progress_callback(processed, file_date)

            except Exception as e:
                print(f"Error processing {file_date}: {e}")
                continue

        conn.commit()
        conn.close()
        return processed

    def load_exchange_addresses(self, exchange_file: str = "data/exchanges.json") -> Dict[str, List[str]]:
        """
        Load known exchange addresses from JSON file.

        Returns: Dict mapping exchange name -> list of addresses
        """
        import json
        exchange_path = Path(exchange_file)
        if not exchange_path.exists():
            return {}

        with open(exchange_path) as f:
            return json.load(f)

    def lookup_node_ids(self, addresses: List[str], batch_size: int = 10000) -> Dict[str, int]:
        """
        Look up ORBITAAL node IDs for given addresses.

        Uses streaming to handle the large node table efficiently.

        Returns: Dict mapping address -> node_id
        """
        if not self.node_table_path.exists():
            return {}

        address_set = set(addresses)
        result = {}

        # Stream through row groups
        parquet_file = pq.ParquetFile(self.node_table_path)

        for i in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(i, columns=['ID', 'NAME'])
            df = table.to_pandas()

            # Find matching addresses
            matches = df[df['NAME'].isin(address_set)]
            for _, row in matches.iterrows():
                result[row['NAME']] = row['ID']

            # Early exit if we found all
            if len(result) == len(addresses):
                break

        return result

    def compute_exchange_flows(
        self,
        file_path: Path,
        exchange_ids: set
    ) -> Dict[str, float]:
        """
        Compute exchange inflows/outflows from daily edges.

        Returns dict with:
        - exchange_inflow_btc: Total BTC sent TO exchanges
        - exchange_outflow_btc: Total BTC sent FROM exchanges
        - net_flow_btc: inflow - outflow (positive = accumulation on exchanges)
        """
        edges = self.read_daily_edges(file_path)

        # Inflows: destination is exchange
        inflows = edges[edges['DST_ID'].isin(exchange_ids)]
        inflow_btc = inflows['VALUE_SATOSHI'].sum() / 1e8

        # Outflows: source is exchange
        outflows = edges[edges['SRC_ID'].isin(exchange_ids)]
        outflow_btc = outflows['VALUE_SATOSHI'].sum() / 1e8

        return {
            'exchange_inflow_btc': inflow_btc,
            'exchange_outflow_btc': outflow_btc,
            'net_flow_btc': inflow_btc - outflow_btc
        }


def quick_test():
    """Quick test of the ORBITAAL loader."""
    loader = OrbitaalLoader()

    try:
        start, end = loader.get_coverage()
        print(f"ORBITAAL coverage: {start} to {end}")

        # Test reading one file
        for file_date, file_path in loader.iter_daily_files(start_date="2020-01-01", end_date="2020-01-02"):
            print(f"\nReading {file_date}...")
            edges = loader.read_daily_edges(file_path)
            stats = loader.compute_daily_stats(edges)
            print(f"Transactions: {stats['tx_count']:,}")
            print(f"Total value: {stats['total_value_btc']:,.2f} BTC")
            print(f"Whale txs: {stats['whale_tx_count']}")
            break

    except FileNotFoundError as e:
        print(f"ORBITAAL data not found: {e}")


if __name__ == "__main__":
    quick_test()
