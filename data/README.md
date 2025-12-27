# Data Directory - RenTech-Style Bitcoin Analysis

## Overview
Complete Bitcoin blockchain dataset for Renaissance Technologies-style quantitative analysis.
Downloaded: 2025-12-13

## Data Sources

### 1. ORBITAAL Dataset (97.4 GB extracted)
**Source**: Zenodo - https://zenodo.org/records/12581515
**Coverage**: 2009-01-03 to 2021-12-31 (ALL Bitcoin transactions)
**Format**: Parquet files

Location: `data/orbitaal/`

| Directory | Size | Description |
|-----------|------|-------------|
| `NODE_TABLE/` | 33.29 GB | Entity-to-address mappings, clustering results |
| `SNAPSHOT/` | 37.70 GB | Daily balance snapshots for all addresses |
| `STREAM_GRAPH/` | 26.36 GB | Transaction flow graphs (sender -> receiver) |

**Key Files**:
- `NODE_TABLE/*.parquet` - Maps addresses to entity clusters
- `SNAPSHOT/*.parquet` - Daily UTXO snapshots with balances
- `STREAM_GRAPH/*.parquet` - Edge lists of BTC flows between entities

### 2. Exchange Addresses (7.6M addresses)
**Source**: EntityAddressBitcoin - https://drive.switch.ch/index.php/s/ag4OnNgwf7LhWFu
**Coverage**: 86 major exchanges

Location: `data/entity_addresses/` (extracted CSVs)
Processed: `data/exchanges.json` (address -> exchange mapping)

**Exchanges Included**:
- Binance, Coinbase, Kraken, Bitfinex, Bitstamp
- Huobi, OKEx, Bittrex, Poloniex, KuCoin
- Gemini, FTX (historical), Coincheck, Gate.io
- 72+ additional exchanges

### 3. Price Data (4,364 daily candles)
**Source**: CryptoCompare API (free tier)
**Coverage**: 2014-09-17 to 2025-12-13

Location: `data/historical_flows.db` (SQLite)

**Schema**:
```sql
CREATE TABLE prices (
    timestamp INTEGER PRIMARY KEY,  -- Unix timestamp
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL
);
```

## Data Gap: 2022-2025

ORBITAAL covers 2009-2021. For 2022-2025 data, options:
1. **Bitcoin Core Node** - Full blockchain sync (background validation in progress)
2. **mempool.space API** - Free, rate-limited (~1 block/sec)
3. **Blockchair API** - Free tier available

## Usage Examples

### Load Exchange Addresses
```python
import json
with open('data/exchanges.json') as f:
    exchanges = json.load(f)
# exchanges['binance'] = ['addr1', 'addr2', ...]
```

### Load Price Data
```python
import sqlite3
conn = sqlite3.connect('data/historical_flows.db')
prices = conn.execute('SELECT * FROM prices ORDER BY timestamp').fetchall()
```

### Load ORBITAAL Parquet Files
```python
import pandas as pd
# Entity mappings
nodes = pd.read_parquet('data/orbitaal/NODE_TABLE/')
# Daily snapshots
snapshots = pd.read_parquet('data/orbitaal/SNAPSHOT/')
# Transaction flows
flows = pd.read_parquet('data/orbitaal/STREAM_GRAPH/')
```

## RenTech Hypothesis Testing

Key signals to test with this data:

1. **Exchange Inflow** - Large deposits to exchanges often precede selling
2. **Exchange Outflow** - Withdrawals to cold storage = accumulation
3. **Whale Movement** - Track large entity balance changes
4. **Miner Flows** - Miner selling patterns
5. **Dormant Coins** - Old UTXO movements signal major events

## File Sizes Summary

| Component | Compressed | Extracted |
|-----------|------------|-----------|
| orbitaal-nodetable.tar.gz | 23.18 GB | 33.29 GB |
| orbitaal-snapshot-day.tar.gz | 23.13 GB | 37.70 GB |
| orbitaal-stream_graph.tar.gz | 22.30 GB | 26.36 GB |
| orbitaal-snapshot-all.tar.gz | 9.42 GB | (included above) |
| entity_addresses.zip | 1.0 GB | ~2 GB |
| historical_flows.db | - | ~1 MB |
| **TOTAL** | ~78 GB | ~97 GB |

## Credits

- ORBITAAL: "ORBITAAL: A Temporal Graph Dataset of Bitcoin Entity-Entity Transactions" (Zenodo)
- EntityAddressBitcoin: Maru92/EntityAddressBitcoin (GitHub)
- Price Data: CryptoCompare API
