# RenTech-Style Blockchain Backtesting

## THE GOAL

Find **statistically proven edges** from blockchain data before trading live.

Like Renaissance Technologies:
- Test 100+ hypotheses
- Require 1000+ samples each
- Only trade patterns with p < 0.01 significance
- Win rate > 50.5% after costs

---

## DATA PIPELINE

```
BITCOIN NODE (full chain)
         |
         v
historical_scanner.py
(scan blocks 600000-840000+)
         |
         v
FLOW DATABASE (SQLite)
- timestamp, block, txid
- exchange, direction, amount
- address type (hot/cold)
         |
         v
PRICE DATA (Binance 1m)
- timestamp, open, high, low, close, volume
         |
         v
correlation_engine.py
(merge flows + prices)
         |
         v
hypothesis_tester.py
(test 50+ patterns)
         |
         v
PROVEN EDGES
(implement in live trading)
```

---

## HYPOTHESES TO TEST

### Category 1: Flow → Price Timing
| ID | Hypothesis | Test |
|----|------------|------|
| H01 | Inflow > 10 BTC → price drops within 5 min | Measure % correct |
| H02 | Inflow > 50 BTC → price drops within 10 min | Measure % correct |
| H03 | Inflow > 100 BTC → price drops within 30 min | Measure % correct |
| H04 | Outflow > 10 BTC → price rises within 5 min | Measure % correct |
| H05 | Outflow > 50 BTC → price rises within 10 min | Measure % correct |
| H06 | Outflow > 100 BTC → price rises within 30 min | Measure % correct |

### Category 2: Optimal Entry Delay
| ID | Hypothesis | Test |
|----|------------|------|
| H10 | Best entry is 0 seconds after signal | Compare returns |
| H11 | Best entry is 10 seconds after signal | Compare returns |
| H12 | Best entry is 30 seconds after signal | Compare returns |
| H13 | Best entry is 60 seconds after signal | Compare returns |

### Category 3: Exchange-Specific
| ID | Hypothesis | Test |
|----|------------|------|
| H20 | Binance flows are most predictive | Compare by exchange |
| H21 | Coinbase flows predict US session moves | Time-segmented test |
| H22 | Bitfinex whale flows = smart money | Large flow analysis |

### Category 4: Time Patterns
| ID | Hypothesis | Test |
|----|------------|------|
| H30 | Flows at market open more predictive | 9am UTC vs other |
| H31 | Weekend flows have less edge | Sat/Sun vs weekday |
| H32 | Night flows (low liquidity) = bigger moves | Time segmentation |

### Category 5: Sequence Patterns
| ID | Hypothesis | Test |
|----|------------|------|
| H40 | 3 consecutive inflows → strong short | Pattern matching |
| H41 | Large outflow after inflows = reversal | Sequence detection |
| H42 | Flow clustering (5+ in 1 min) = momentum | Cluster analysis |

### Category 6: Size Effects
| ID | Hypothesis | Test |
|----|------------|------|
| H50 | Flow size correlates with move size | Regression |
| H51 | Whale flows (>100 BTC) predict bigger moves | Size segmentation |
| H52 | Many small flows = less predictive | Aggregation test |

### Category 7: Mean Reversion vs Momentum
| ID | Hypothesis | Test |
|----|------------|------|
| H60 | After 1% move, price reverts | Mean reversion test |
| H61 | After flow spike, momentum continues | Momentum test |
| H62 | Volatility regime affects which works | Regime-conditional |

---

## STATISTICAL REQUIREMENTS

For each hypothesis:

```python
MIN_SAMPLES = 1000          # At least 1000 trades
MIN_WIN_RATE = 0.505        # Must beat 50% (after spread/fees)
MAX_P_VALUE = 0.01          # 99% confidence it's not random
MIN_SHARPE = 1.0            # Risk-adjusted returns
MAX_DRAWDOWN = 0.20         # Max 20% drawdown
```

---

## OUTPUT

After testing, we get a ranked list:

| Rank | Hypothesis | Win Rate | Edge | p-value | Samples | IMPLEMENT? |
|------|------------|----------|------|---------|---------|------------|
| 1 | H05 | 54.2% | 1.3x | 0.001 | 5,234 | ✅ YES |
| 2 | H42 | 52.8% | 1.2x | 0.003 | 3,892 | ✅ YES |
| 3 | H21 | 51.9% | 1.1x | 0.008 | 2,103 | ✅ YES |
| 4 | H30 | 51.2% | 1.05x | 0.042 | 1,892 | ❌ NO (p>0.01) |
| 5 | H01 | 50.3% | 0.98x | 0.210 | 8,234 | ❌ NO (no edge) |

Only implement patterns that pass ALL criteria.

---

## TIMELINE

1. **Historical Scanner** - Scan blocks, extract flows (runs on node)
2. **Price Data** - Download Binance 1m candles
3. **Correlation DB** - Merge flows + prices
4. **Hypothesis Tests** - Run all 50+ tests
5. **Validation** - Out-of-sample test on recent data
6. **Implementation** - Code proven edges into live engine

---

## FILES

| File | Purpose |
|------|---------|
| `historical_scanner.py` | Scan blockchain for exchange flows |
| `price_downloader.py` | Get Binance historical data |
| `correlation_engine.py` | Merge flows with prices |
| `hypothesis_tester.py` | Statistical tests |
| `proven_edges.py` | Implement winning strategies |
