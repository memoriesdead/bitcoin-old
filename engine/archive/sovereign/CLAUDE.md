# Sovereign HFT Trading Engine

## Architecture (5 Core Modules Only)

```
engine/sovereign/blockchain/
├── cpp_master_pipeline.py   (601 lines) - Entry point
├── config.py                (138 lines) - Configuration
├── correlation_formula.py   (700 lines) - Signals
├── deterministic_trader.py  (508 lines) - Positions
└── multi_price_feed.py      (504 lines) - Prices
```

**Total**: 2,451 lines. Everything else is support/archived.

---

## Critical Rules

### 1. C++ for Speed
```
C++ (nanoseconds):           Python (only for):
- ZMQ blockchain             - Trading orchestration
- Address matching (8.6M)    - Exchange APIs (CCXT)
- TX decoding                - Database logging
- Signal generation          - Price feeds
```

### 2. Zero Mock Data
Real money trading with up to 125x leverage. Every component uses REAL data:
- Real Bitcoin Core ZMQ feed
- Real 8.6M WalletExplorer addresses
- Real exchange prices via CCXT

### 3. Let Data Speak
No arbitrary thresholds. Pattern is tradeable when:
```python
sample_count >= 10 AND correlation >= 0.7 AND win_rate >= 0.9
```

---

## The Math

```
INFLOW to exchange -> Deposit to SELL -> Price DOWN -> SHORT (100% accurate)
OUTFLOW + Seller exhaustion -> Price UP -> LONG (100% accurate)
```

---

## Signal Flow

```
Bitcoin Core ZMQ -> C++ Runner (8us) -> cpp_master_pipeline.py
    -> correlation_formula.py -> deterministic_trader.py -> Exchange
```

---

## Module Quick Reference

| Module | Purpose | Key Class |
|--------|---------|-----------|
| cpp_master_pipeline.py | C++ bridge | `CppMasterPipeline` |
| config.py | Settings | `TradingConfig` |
| correlation_formula.py | Signals | `CorrelationFormula`, `Signal` |
| deterministic_trader.py | Positions | `DeterministicTrader`, `Position` |
| multi_price_feed.py | Prices | `MultiExchangePriceFeed` |

---

## Support Modules (8 files)

```
__init__.py          - Package init
tx_decoder.py        - TX parsing
zmq_subscriber.py    - ZMQ connection
types.py             - Type definitions
rpc.py               - Bitcoin RPC calls
exchange_leverage.py - Leverage config
exchange_flow.py     - Flow definitions
correlation_db.py    - Database operations
```

---

## Extended Modules (9 files - optional)

```
deterministic_math.py      - Order book formula
deterministic_utxo.py      - UTXO tracking
formula_connector.py       - Formula integration
exchange_utxo_cache.py     - UTXO cache
address_cluster.py         - Address clustering
address_collector.py       - Address collection
continuous_wallet_sync.py  - Wallet sync
data_collector.py          - Data collection
utxo_lifecycle.py          - UTXO lifecycle
```

---

## VPS Commands

```bash
ssh root@31.97.211.217

# C++ pipeline
tmux attach -t cpp
tail -f /root/sovereign/cpp_pipeline.log

# Paper trade
python3 blockchain/cpp_master_pipeline.py --paper

# Check data
sqlite3 correlation.db 'SELECT * FROM patterns LIMIT 10;'
sqlite3 trades.db 'SELECT * FROM trades ORDER BY id DESC LIMIT 10;'
```

---

## Databases (VPS: /root/sovereign/)

| File | Purpose |
|------|---------|
| walletexplorer_addresses.db | 8.6M exchange addresses |
| correlation.db | Flow->price patterns |
| trades.db | Trade history |
| exchange_utxos.db | UTXO cache |

---

## Trading Config (config.py)

```python
initial_capital = 100.0      # USD
max_leverage = 125
max_positions = 4
position_size_pct = 0.25     # 25% per position
exit_timeout_seconds = 300   # 5 min
stop_loss_pct = 0.01         # 1%
take_profit_pct = 0.02       # 2%
```

---

## Exchange APIs

All exchanges via unified CCXT API. Keys in `~/.ccxt/config.json`.

**Tier 1 USA**: kraken, coinbase, gemini, bitstamp
**Max Leverage**: MEXC 500x, Binance/Bybit 125x, Kraken 50x

---

## Archived (Do Not Use)

Everything in `archive/` folder is deprecated.
