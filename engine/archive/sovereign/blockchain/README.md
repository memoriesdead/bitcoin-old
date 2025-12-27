# Blockchain Data Pipeline

## Status: LIVE

```
Pipeline:    cpp_master_pipeline.py (tmux: cpp)
Addresses:   8.6M across 102 exchanges
Price Feeds: 12 exchanges connected
Accuracy:    100% verified (12/12 signals)
```

---

## Core Modules

| File | Purpose |
|------|---------|
| cpp_master_pipeline.py | C++ bridge, main entry point |
| config.py | All configuration |
| correlation_formula.py | Pattern matching, signals |
| deterministic_trader.py | Position management |
| multi_price_feed.py | 12-exchange price feeds |

---

## The Math

```
INFLOW  -> Someone depositing to SELL -> Price DOWN -> SHORT
OUTFLOW -> Seller exhaustion          -> Price UP   -> LONG
```

Pattern enabled when: `samples >= 10 AND correlation >= 0.7 AND win_rate >= 0.9`

---

## Signal Flow

```
Bitcoin Core ZMQ
       |
       v
C++ blockchain_runner (8.6M addresses, nanoseconds)
       |
       v
cpp_master_pipeline.py -> correlation_formula.py -> deterministic_trader.py -> Exchange
```

---

## VPS Commands

```bash
# Connect
ssh root@31.97.211.217

# Monitor C++ pipeline
tmux attach -t cpp
tail -f /root/sovereign/cpp_pipeline.log

# Paper trade
python3 blockchain/cpp_master_pipeline.py --paper

# Check signals
sqlite3 correlation.db 'SELECT * FROM flows ORDER BY id DESC LIMIT 10;'

# Check trades
sqlite3 trades.db 'SELECT * FROM trades ORDER BY id DESC LIMIT 10;'
```

---

## Databases

| Database | Purpose |
|----------|---------|
| walletexplorer_addresses.db | 8.6M exchange addresses |
| correlation.db | Flow->price patterns |
| trades.db | Trade history |
| exchange_utxos.db | UTXO cache |

---

## Support Modules

```
tx_decoder.py        - Parse raw transactions
zmq_subscriber.py    - ZMQ connection
types.py             - Type definitions
rpc.py               - Bitcoin RPC calls
exchange_leverage.py - Leverage config
exchange_flow.py     - Flow definitions
correlation_db.py    - Database operations
```

---

## Archive

All deprecated files are in `archive/` folder (75 files). Do not use.
