# Sovereign Data Pipeline

**Complete Bitcoin blockchain data from genesis (2009) to present.**

This pipeline combines multiple data sources to provide a unified view of all Bitcoin transactions for Renaissance-style quantitative trading.

---

## Data Sources

### 1. ORBITAAL Dataset (2009-01-09 to 2021-01-25)

Transaction-level data from the ORBITAAL research project.

- **Location:** `data/orbitaal/`
- **Size:** ~104 GB (4,401 parquet files)
- **Coverage:** Genesis block to January 2021
- **Schema:**
  - `SNAPSHOT/EDGES/day/` - Daily transaction edges (SRC_ID, DST_ID, VALUE_SATOSHI, VALUE_USD)
  - `NODE_TABLE/` - Address lookup (364 million addresses)

**Key Metrics Available:**
- Transaction count and values
- Unique senders/receivers
- Whale transactions (>100 BTC)
- Full address-level flows

### 2. Mempool.space Download (2021-01-26 to 2025)

Block-level data downloaded from mempool.space API.

- **Location:** `data/bitcoin_2021_2025.db`
- **Size:** ~25 MB (263,791 blocks)
- **Coverage:** Block 664,000 to present
- **Schema:** height, timestamp, hash, tx_count, size, weight, fees, median_fee

**Key Metrics Available:**
- Block-level transaction counts
- Block size and fullness
- Fee statistics
- Block timing

### 3. Bitcoin Core RPC (Optional - Live)

Direct scanning of your local Bitcoin Core node for transaction-level data.

- **Requirement:** Fully synced Bitcoin Core with `txindex=1`
- **Speed:** ~10-50 blocks/second (depends on hardware)
- **Use Case:** Fill gaps or extend ORBITAAL-quality data beyond 2021

---

## File Structure

```
engine/sovereign/data/
├── __init__.py              # Module exports
├── pipeline.py              # UnifiedDataPipeline - combines all sources
├── orbitaal_loader.py       # ORBITAAL parquet file reader
├── btc_scanner.py           # Bitcoin Core RPC scanner
└── README.md                # This file

data/
├── orbitaal/                # ORBITAAL raw data (104 GB)
│   ├── SNAPSHOT/EDGES/day/  # 4,401 daily parquet files
│   └── NODE_TABLE/          # Address lookup table
├── bitcoin_2021_2025.db     # Downloaded block data (25 MB)
├── bitcoin_features.db      # Aggregated features (38 MB)
└── unified_bitcoin.db       # Combined pipeline output
```

---

## Quick Start

### Build the Unified Database

```python
from engine.sovereign.data import UnifiedDataPipeline

pipeline = UnifiedDataPipeline()
stats = pipeline.build()

print(f"Coverage: {stats['date_range']}")
print(f"Total days: {stats['total_days']}")
```

### Query Daily Features

```python
# Get daily stats for backtesting
stats = pipeline.get_daily_stats(
    start_date="2020-01-01",
    end_date="2024-12-31"
)

for day in stats:
    print(f"{day['date']}: {day['tx_count']:,} transactions")
```

### Export for Backtesting

```python
# Export to CSV
pipeline.export_for_backtest("data/backtest_features.csv")
```

---

## Data Quality Notes

### ORBITAAL Data (2009-2021)
- **Completeness:** Full transaction-level data with USD values
- **Addresses:** Can trace specific addresses using NODE_TABLE
- **Limitation:** Data ends January 2021

### Downloaded Data (2021-2025)
- **Completeness:** Block-level aggregates only
- **Limitation:** No individual transaction or address data
- **Use Case:** Trading signals based on tx_count, block fullness

### Filling the Gap

To get transaction-level data for 2021-2025 (matching ORBITAAL quality):

```python
from engine.sovereign.data import BitcoinCoreScanner

scanner = BitcoinCoreScanner()
scanner.build_daily_features_db(
    start_date="2021-01-26",
    end_date="2025-12-31",
    output_db="data/btc_scanner_daily.db"
)
```

**Note:** This requires a fully synced Bitcoin Core node and takes several hours.

---

## Trading Signals

The pipeline supports these signal types:

### From ORBITAAL (High Quality, 2009-2021)
- Exchange flow detection (using known exchange addresses)
- Whale accumulation/distribution
- Active address trends
- Transaction value distribution

### From Block Data (2021-Present)
- Transaction count z-scores
- Block fullness signals
- Fee market analysis
- Congestion indicators

### Example Backtest Results

From `block_features_backtest.py`:

| Strategy | Trades | Win Rate | Total PnL |
|----------|--------|----------|-----------|
| HIGH_TX_z>1.5_hold7d | 192 | 54.2% | +190.4% |
| LOW_TX_z<-1.5_hold5d | 191 | 55.0% | +152.5% |
| LOW_TX_z<-2.0_hold5d | 85 | 63.5% | +131.2% |

---

## Database Schema

### daily_features (Main Table)

```sql
CREATE TABLE daily_features (
    date TEXT PRIMARY KEY,
    timestamp INTEGER,
    source TEXT,                -- 'orbitaal' or 'downloaded'
    blocks INTEGER,
    tx_count INTEGER,
    total_value_btc REAL,       -- NULL for downloaded data
    total_value_usd REAL,       -- NULL for downloaded data
    unique_senders INTEGER,     -- NULL for downloaded data
    unique_receivers INTEGER,   -- NULL for downloaded data
    whale_tx_count INTEGER,
    whale_value_btc REAL,
    avg_block_size REAL,
    avg_block_fullness REAL
);
```

### block_features (Detail Table)

```sql
CREATE TABLE block_features (
    height INTEGER PRIMARY KEY,
    timestamp INTEGER,
    hash TEXT,
    tx_count INTEGER,
    size INTEGER,
    weight INTEGER,
    fees_btc REAL,
    median_fee_rate REAL
);
```

---

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Load ORBITAAL daily file | ~50ms | Single parquet file |
| Process all ORBITAAL | ~15 min | 4,401 files |
| Import downloaded blocks | ~30 sec | 263K blocks |
| Full pipeline build | ~20 min | First time only |
| Query daily stats | <100ms | SQLite indexed |

---

## Dependencies

```
pandas>=2.0
pyarrow>=14.0
numpy>=1.24
sqlite3 (builtin)
```

For Bitcoin Core scanning:
```
Bitcoin Core with server=1, rpcuser, rpcpassword
```

---

## Extending the Data

### Adding New Block Data

Use the Hostinger parallel downloader for fast block downloads:

```bash
# On Hostinger VPS (4 parallel sessions)
python3 hostinger_download.py 664000 730000 chunk1
python3 hostinger_download.py 730000 796000 chunk2
python3 hostinger_download.py 796000 862000 chunk3
python3 hostinger_download.py 862000 928000 chunk4

# Combine
python3 hostinger_download.py combine
```

### Adding Transaction-Level Data

Use BitcoinCoreScanner for detailed transaction data:

```python
scanner = BitcoinCoreScanner()
results = scanner.quick_scan(days=30)  # Test with 30 days
```

---

## Credits

- **ORBITAAL Dataset:** Academic research dataset for Bitcoin analysis
- **Mempool.space:** Block explorer API for recent data
- **Bitcoin Core:** Reference implementation for live data
