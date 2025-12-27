# Bitcoin HFT Trading Engine

Minimal, high-performance Bitcoin trading engine using blockchain flow analysis.

## The Edge

```
INFLOW to exchange  → Deposit to SELL → Price DOWN → SHORT
OUTFLOW from exchange → Seller exhaustion → Price UP → LONG
```

## Trading Criteria

Pattern is tradeable when:
```python
sample_count >= 10 AND correlation >= 0.7 AND win_rate >= 0.9
```

## Architecture

```
Bitcoin Core ZMQ → C++ Runner (8μs) → run.py → Exchange
```

## Files

| File | Lines | Purpose |
|------|-------|---------|
| config.py | 84 | Trading configuration |
| signals.py | 700 | CorrelationFormula, Signal generation |
| trader.py | 509 | DeterministicTrader, Position management |
| price_feed.py | 285 | 12 exchange price feeds |
| run.py | 606 | C++ bridge, entry point |

**Total: 6 files, ~2,200 lines**

## Configuration

```python
initial_capital = 100.0      # USD
max_leverage = 125
max_positions = 4
position_size_pct = 0.25     # 25% per position
exit_timeout_seconds = 300   # 5 min
stop_loss_pct = 0.01         # 1%
take_profit_pct = 0.02       # 2%
```

## Usage

```bash
# Paper trading
python -m bitcoin.run --paper

# Live mode
python -m bitcoin.run

# Data collection only
python -m bitcoin.run --collect-only
```

## Exchanges

**Tradeable (Tier 1 USA):** coinbase, kraken, bitstamp, gemini, crypto.com

**Price Feeds (12 total):** binance, coinbase, kraken, bitfinex, okx, bybit, huobi, bitstamp, kucoin, gate.io, gemini, crypto.com

## Key Classes

```python
from bitcoin import TradingConfig, get_config
from bitcoin import Signal, SignalType, CorrelationFormula
from bitcoin import Position, DeterministicTrader
```

## VPS Deployment

```bash
# SSH to VPS
ssh root@31.97.211.217

# Run paper trading
cd /root/sovereign
python -m bitcoin.run --paper
```
