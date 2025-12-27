# Sovereign HFT Pipeline - Technical Documentation

## Overview

This pipeline detects Bitcoin exchange flows in real-time and generates trading signals based on statistical correlation between flow events and price movements.

```
Bitcoin Core ZMQ (rawtx)
         │
         ▼
┌─────────────────────────────────────────┐
│  C++ Blockchain Runner                  │
│  - 8.6M addresses via mmap (56μs load)  │
│  - Nanosecond latency signal detection  │
│  - UTXO cache for outflow tracking      │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  cpp_master_pipeline.py                 │
│  - Parses C++ output                    │
│  - Feeds to correlation_formula.py      │
│  - Manages trading positions            │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  correlation_formula.py                 │
│  - Pattern matching by (exchange, dir,  │
│    bucket)                              │
│  - Statistical correlation tracking     │
│  - Signal generation when thresholds    │
│    met                                  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  deterministic_trader.py                │
│  - Position management                  │
│  - Time-based exits (5 min)             │
│  - P&L tracking per exchange            │
└─────────────────────────────────────────┘
```

---

## Configuration Thresholds (config.py)

### Pattern Enablement Criteria

A pattern is enabled for trading when ALL three conditions are met:

| Threshold | Current Value | Purpose |
|-----------|---------------|---------|
| `min_sample_size` | 10 | Minimum samples for statistical significance |
| `min_correlation` | 0.01 | Minimum abs(correlation) between flow and price |
| `min_win_rate` | 0.5 (50%) | Minimum win rate (price moved in expected direction) |

**Pattern Enablement Formula** (`correlation_formula.py:576`):
```python
enabled = (
    sample_count >= config.min_sample_size and
    abs(correlation) >= config.min_correlation and
    win_rate >= config.min_win_rate
)
```

### Flow Buckets

Flows are categorized into size buckets for separate correlation tracking:

| Bucket | BTC Range | Use Case |
|--------|-----------|----------|
| (0, 1) | 0-1 BTC | Small retail flows |
| (1, 5) | 1-5 BTC | Medium flows |
| (5, 10) | 5-10 BTC | Large retail/small institutional |
| (10, 50) | 10-50 BTC | Institutional flows |
| (50, 100) | 50-100 BTC | Large institutional |
| (100, 500) | 100-500 BTC | Whale activity |
| (500, inf) | 500+ BTC | Mega whale |

### Position Management

| Setting | Value | Description |
|---------|-------|-------------|
| `initial_capital` | $100 | Starting paper trading capital |
| `max_leverage` | 125x | Maximum leverage per position |
| `max_positions` | 4 | Maximum concurrent positions |
| `position_size_pct` | 25% | Capital allocation per trade |
| `exit_timeout_seconds` | 300s | Time-based exit (5 minutes) |
| `stop_loss_pct` | 1% | Stop loss trigger |
| `take_profit_pct` | 2% | Take profit trigger |

---

## Pattern Lookup Key Format

Patterns are stored and looked up using a composite key:

```python
key = (exchange.lower(), direction.upper(), bucket)
# Example: ('coinbase', 'INFLOW', (10, 50))
```

**Important**: The bucket tuple uses integers from the C++ side `(10, 50)` but floats from the database `(10.0, 50.0)`. Python handles equality correctly - they are equivalent.

---

## Database Schema (correlation.db)

### flows table
Stores every detected flow for correlation analysis:
```sql
CREATE TABLE flows (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    exchange TEXT,
    direction TEXT,      -- 'INFLOW' or 'OUTFLOW'
    flow_btc REAL,
    bucket_min REAL,
    bucket_max REAL,
    price_at_flow REAL,
    price_1min REAL,     -- Price 1 minute after flow
    price_5min REAL,     -- Price 5 minutes after flow
    price_10min REAL,    -- Price 10 minutes after flow
    verified INTEGER     -- 0=pending, 1=verified
)
```

### patterns table
Stores calculated pattern statistics:
```sql
CREATE TABLE patterns (
    id INTEGER PRIMARY KEY,
    exchange TEXT,
    direction TEXT,
    bucket_min REAL,
    bucket_max REAL,
    sample_count INTEGER,
    correlation REAL,    -- Pearson correlation
    win_rate REAL,       -- % of flows where price moved expected direction
    avg_price_change REAL,
    enabled INTEGER,     -- 0=disabled, 1=enabled for trading
    last_updated TEXT,
    UNIQUE(exchange, direction, bucket_min, bucket_max)
)
```

---

## Current Pattern Status

Query to check enabled patterns:
```bash
sqlite3 /root/sovereign/correlation.db \
  'SELECT exchange, direction, bucket_min, bucket_max, sample_count,
          printf("%.3f", correlation) as corr,
          printf("%.1f%%", win_rate*100) as win,
          enabled
   FROM patterns
   WHERE enabled=1
   ORDER BY sample_count DESC;'
```

As of 2025-12-25:
| Exchange | Direction | Bucket | Samples | Correlation | Win Rate | Status |
|----------|-----------|--------|---------|-------------|----------|--------|
| coinbase | INFLOW | 10-50 | 47 | -0.039 | 87.2% | ENABLED |
| coinbase | INFLOW | 0-1 | 46 | 0.110 | 84.8% | ENABLED |
| bitfinex | INFLOW | 1-5 | 18 | -0.064 | 94.4% | ENABLED |
| coinbase | INFLOW | 1-5 | 13 | -0.043 | 76.9% | ENABLED |
| binance | INFLOW | 1-5 | 10 | 0.092 | 90.0% | ENABLED |
| binance | INFLOW | 0-1 | 10 | 0.375 | 100% | ENABLED |

---

## Troubleshooting

### Signals Not Generating

1. **Check if pattern exists and is enabled**:
   ```bash
   sqlite3 /root/sovereign/correlation.db \
     'SELECT * FROM patterns WHERE exchange="coinbase" AND direction="INFLOW";'
   ```

2. **Check threshold values**:
   ```bash
   grep -E 'min_correlation|min_win_rate|min_sample' /root/sovereign/blockchain/config.py
   ```

3. **Manually enable a pattern** (if thresholds too strict):
   ```bash
   sqlite3 /root/sovereign/correlation.db \
     'UPDATE patterns SET enabled=1
      WHERE exchange="coinbase" AND direction="INFLOW" AND bucket_min=10;'
   ```

4. **Check debug output** - the pipeline prints debug lines for flows >= 10 BTC:
   ```
   [DEBUG] Flow: coinbase INFLOW 25.21 BTC -> bucket (10, 50)
   [DEBUG] Lookup key: ('coinbase', 'INFLOW', (10, 50))
   [DEBUG] Pattern found: True
   [DEBUG] Pattern enabled: False, samples: 47, win_rate: 0.87
   ```

### Pattern Getting Disabled After Enabling

The `verify_prices()` function recalculates pattern stats and may disable patterns that don't meet thresholds. To prevent this:

1. **Lower thresholds in config.py**:
   ```python
   min_correlation: float = 0.01  # Very low - almost any correlation
   min_win_rate: float = 0.5      # 50% - above random
   ```

2. **Enable all high win-rate patterns**:
   ```bash
   sqlite3 /root/sovereign/correlation.db \
     'UPDATE patterns SET enabled=1 WHERE win_rate >= 0.5 AND sample_count >= 10;'
   ```

### Price Feed Issues

Check if prices are being fetched:
```bash
ssh root@31.97.211.217 "cd /root/sovereign && python3 -c \"
from blockchain.multi_price_feed import MultiExchangePriceFeed
feed = MultiExchangePriceFeed()
print('coinbase:', feed.get_price('coinbase'))
print('binance:', feed.get_price('binance'))
\""
```

---

## Running the Pipeline

### Paper Trading Mode (Recommended for Testing)
```bash
ssh root@31.97.211.217 "cd /root/sovereign && python3 -u blockchain/cpp_master_pipeline.py --paper 2>&1 | tee pipeline.log"
```

### Data Collection Only (No Trading)
```bash
ssh root@31.97.211.217 "cd /root/sovereign && python3 -u blockchain/cpp_master_pipeline.py --collect-only 2>&1 | tee pipeline.log"
```

### Background Mode (tmux)
```bash
ssh root@31.97.211.217
tmux new -s pipeline
cd /root/sovereign && python3 -u blockchain/cpp_master_pipeline.py --paper 2>&1 | tee pipeline.log
# Ctrl+B, D to detach
```

### Monitor Running Pipeline
```bash
ssh root@31.97.211.217 "tail -f /root/sovereign/pipeline.log"
```

---

## File Locations

| File | Location | Purpose |
|------|----------|---------|
| `cpp_master_pipeline.py` | `/root/sovereign/blockchain/` | Main entry point |
| `correlation_formula.py` | `/root/sovereign/blockchain/` | Signal generation |
| `deterministic_trader.py` | `/root/sovereign/blockchain/` | Position management |
| `config.py` | `/root/sovereign/blockchain/` | All configuration |
| `multi_price_feed.py` | `/root/sovereign/blockchain/` | Exchange price feeds |
| `correlation.db` | `/root/sovereign/` | Flow/pattern database |
| `trades.db` | `/root/sovereign/` | Trade history |
| `walletexplorer_addresses.db` | `/root/sovereign/` | 8.6M exchange addresses |
| `exchange_utxos.db` | `/root/sovereign/` | UTXO cache |

---

## Signal Flow Explanation

1. **C++ Runner** detects BTC flow to/from exchange address
2. **cpp_master_pipeline.py** parses signal: `[SHORT] coinbase | In: 25.21 | Net: -25.21`
3. **correlation_formula.py** looks up pattern:
   - Key: `('coinbase', 'INFLOW', (10, 50))`
   - Checks if pattern enabled
   - If enabled, generates Signal object
4. **deterministic_trader.py** opens position if:
   - No existing position on this exchange
   - Under max_positions limit
   - Exchange is in tradeable_exchanges

---

## Key Insights Learned

1. **Correlation vs Win Rate**: High win rate (87%) doesn't mean high correlation (0.039). Both matter for different reasons:
   - **Win rate** = directional accuracy (did price move as expected?)
   - **Correlation** = magnitude relationship (bigger flows = bigger moves?)

2. **Small flows can be more predictive**: The 0-1 BTC coinbase bucket has better correlation (0.11) than 10-50 BTC bucket (0.039), likely because small flows happen more frequently and provide more samples.

3. **Pattern updates happen during price verification**: Every 10 seconds, `verify_prices()` checks pending flows and recalculates pattern stats. This can disable manually-enabled patterns if thresholds aren't met.

4. **The system is selective by design**: Not all detected flows become trades. Only patterns meeting statistical thresholds generate signals.

---

## Quick Reference Commands

```bash
# Check pattern status
sqlite3 /root/sovereign/correlation.db 'SELECT * FROM patterns ORDER BY sample_count DESC;'

# Enable all high win-rate patterns
sqlite3 /root/sovereign/correlation.db 'UPDATE patterns SET enabled=1 WHERE win_rate >= 0.5 AND sample_count >= 10;'

# Check config thresholds
grep -E 'min_correlation|min_win_rate|min_sample' /root/sovereign/blockchain/config.py

# Kill running pipeline
pkill -f cpp_master_pipeline

# Check if pipeline running
ps aux | grep cpp_master

# View recent trades
sqlite3 /root/sovereign/trades.db 'SELECT * FROM trades ORDER BY id DESC LIMIT 10;'
```

---

*Last Updated: 2025-12-25*
