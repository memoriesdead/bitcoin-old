# Module Index - Sovereign HFT Engine

## Directory Tree (22 Active Files)

```
engine/sovereign/blockchain/
├── CORE (5 modules - 2,451 lines)
│   ├── cpp_master_pipeline.py   Entry point, C++ bridge
│   ├── config.py                Single source config
│   ├── correlation_formula.py   Pattern matching, signals
│   ├── deterministic_trader.py  Position management
│   └── multi_price_feed.py      12-exchange prices
│
├── SUPPORT (8 modules)
│   ├── __init__.py              Package init
│   ├── tx_decoder.py            TX parsing
│   ├── zmq_subscriber.py        ZMQ connection
│   ├── types.py                 Type definitions
│   ├── rpc.py                   Bitcoin RPC
│   ├── exchange_leverage.py     Leverage config
│   ├── exchange_flow.py         Flow definitions
│   └── correlation_db.py        DB operations
│
├── EXTENDED (9 modules - optional)
│   ├── deterministic_math.py    Order book formula
│   ├── deterministic_utxo.py    UTXO tracking
│   ├── formula_connector.py     Formula integration
│   ├── exchange_utxo_cache.py   UTXO cache
│   ├── address_cluster.py       Clustering
│   ├── address_collector.py     Collection
│   ├── continuous_wallet_sync.py Sync
│   ├── data_collector.py        Data
│   └── utxo_lifecycle.py        Lifecycle
│
└── archive/                     62+ deprecated files
```

---

## Core Module API

### cpp_master_pipeline.py
```python
class CppSignal:
    exchange: str
    direction: SignalType
    inflow_btc: float
    outflow_btc: float
    net_flow_btc: float
    latency_ns: int

class CppMasterPipeline:
    def run(): ...           # Main loop
    def _process_signal(): ...  # Parse & trade
```

### config.py
```python
class TradingConfig:
    min_correlation = 0.7
    min_win_rate = 0.9
    min_sample_size = 10
    max_leverage = 125
    exit_timeout_seconds = 300

get_config() -> TradingConfig
```

### correlation_formula.py
```python
class Signal:
    exchange: str
    direction: SignalType
    correlation: float
    win_rate: float
    sample_count: int
    is_tradeable: bool

class CorrelationFormula:
    def record_flow(exchange, direction, flow_btc, price): ...
    def generate_signal() -> Signal: ...
```

### deterministic_trader.py
```python
class Position:
    id, exchange, direction
    entry_price, exit_price
    stop_loss, take_profit
    pnl_usd, pnl_pct

class DeterministicTrader:
    def open_position(signal, price) -> Position: ...
    def check_exits(price, time) -> List[Position]: ...
    def get_stats() -> Dict: ...
```

### multi_price_feed.py
```python
class MultiExchangePriceFeed:
    def start(): ...           # Background thread
    def get_price(exchange): ...
    def get_all_prices(): ...
```

---

## Signal Flow

```
Bitcoin Core ZMQ
       |
       v
C++ blockchain_runner (8.6M addresses, nanoseconds)
       |
       v
cpp_master_pipeline.py (parse signals)
       |
   +---+---+
   |       |
   v       v
multi_   correlation_
price    formula.py
_feed    (patterns)
   |       |
   +---+---+
       |
       v
deterministic_trader.py (execute)
       |
       v
Exchange API (CCXT)
```

---

## Database Schema

**correlation.db**
```sql
patterns(exchange, direction, bucket, correlation, win_rate, samples)
flows(id, exchange, direction, flow_btc, price, timestamp)
```

**trades.db**
```sql
trades(id, exchange, direction, entry_price, exit_price,
       size_usd, leverage, pnl_usd, status, exit_reason)
equity_curve(timestamp, capital, open_positions)
```

---

## Key Settings

```python
# config.py - TradingConfig
initial_capital = 100.0
max_leverage = 125
max_positions = 4
position_size_pct = 0.25
stop_loss_pct = 0.01
take_profit_pct = 0.02
exit_timeout_seconds = 300
```

---

## VPS Paths

```
/root/sovereign/
├── walletexplorer_addresses.db  8.6M addresses
├── correlation.db               Pattern data
├── trades.db                    Trade history
├── exchange_utxos.db            UTXO cache
└── cpp_runner/build/blockchain_runner
```
