# Sovereign Trading Engine

## New Clean Structure

```
engine/sovereign/
├── signal_router.py      # Main entry - routes C++ signals
├── short/
│   ├── __init__.py
│   └── trader.py         # SHORT trader (INFLOW signals)
├── long/
│   ├── __init__.py
│   └── trader.py         # LONG trader (OUTFLOW signals)
├── shared/
│   ├── __init__.py
│   ├── config.py         # Configuration
│   └── price_feed.py     # Exchange prices
└── blockchain/           # C++ pipeline (archived, still runs on VPS)
```

---

## The Logic

```
INFLOW to exchange  -> Sellers depositing -> Price DOWN -> SHORT
OUTFLOW from exchange -> Seller exhaustion -> Price UP   -> LONG
```

---

## Quick Start

```bash
# Paper trading
python signal_router.py --paper

# Live trading
python signal_router.py --live
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
signal_router.py
       |
   +---+---+
   |       |
   v       v
short/  long/
trader  trader
   |       |
   v       v
Exchange API
```

---

## Configuration (shared/config.py)

```python
initial_capital = 100.0      # USD
max_leverage = 125
position_size_pct = 0.25     # 25% per trade
stop_loss_pct = 0.01         # 1%
take_profit_pct = 0.02       # 2%
exit_timeout_seconds = 300   # 5 min
min_flow_btc = 10.0          # Minimum to trade
```

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| signal_router.py | ~150 | Routes signals to traders |
| short/trader.py | ~170 | SHORT position management |
| long/trader.py | ~170 | LONG position management |
| shared/config.py | ~55 | Configuration |
| shared/price_feed.py | ~110 | Price fetching |

**Total**: ~655 lines (down from 2,451)

---

## VPS Commands

```bash
ssh root@31.97.211.217

# Run the router
cd /root/sovereign
python signal_router.py --paper

# Check C++ pipeline
tmux attach -t cpp

# Check trades
sqlite3 trades.db 'SELECT * FROM trades ORDER BY id DESC LIMIT 10;'
```

---

## See Also

- `blockchain/` - Archived C++ pipeline code (still runs on VPS)
- `CLAUDE.md` - Full documentation
- `MODULE_INDEX.md` - Detailed module reference
