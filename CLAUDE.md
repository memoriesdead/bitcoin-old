# Sovereign HFT Trading Engine

## Architecture (5 Core Modules - 2,420 lines total)

```
engine/
├── __init__.py              (21)  - Package exports
├── config.py                (83)  - Configuration
├── multi_price_feed.py     (503)  - 12 Exchange Prices
├── correlation_formula.py  (700)  - Signal Generation
├── deterministic_trader.py (508)  - Position Management
├── cpp_master_pipeline.py  (605)  - C++ Bridge
└── archive/                       - Deprecated (352 files)
```

---

## The Edge

```
INFLOW  -> SHORT (sellers depositing to sell)
OUTFLOW -> LONG  (seller exhaustion)
```

**Pattern Tradeable When:**
```python
samples >= 10 AND correlation >= 0.7 AND win_rate >= 0.9
```

---

## Signal Flow

```
Bitcoin ZMQ -> C++ Runner (8us) -> cpp_master_pipeline.py -> Trade
```

---

## Usage

```python
from engine import (
    TradingConfig, get_config,
    Signal, SignalType, CorrelationFormula,
    Position, DeterministicTrader,
    MultiExchangePriceFeed
)

config = get_config()
price_feed = MultiExchangePriceFeed()
formula = CorrelationFormula(config)
trader = DeterministicTrader(config)
```

---

## VPS

```bash
ssh root@31.97.211.217
tmux attach -t cpp
```

---

## Config

```python
initial_capital = 100.0      # USD
max_leverage = 125
max_positions = 4
position_size_pct = 0.25     # 25%
exit_timeout_seconds = 300   # 5 min
stop_loss_pct = 0.01         # 1%
take_profit_pct = 0.02       # 2%
```

---

## Critical Rules

1. **C++ for speed** - nanosecond blockchain, Python for orchestration
2. **Zero mock data** - Real Bitcoin Core, real 8.6M addresses
3. **Let data speak** - No arbitrary thresholds

---

## Databases (VPS: /root/sovereign/)

| File | Purpose |
|------|---------|
| walletexplorer_addresses.db | 8.6M exchange addresses |
| correlation.db | Flow->price patterns |
| trades.db | Trade history |
