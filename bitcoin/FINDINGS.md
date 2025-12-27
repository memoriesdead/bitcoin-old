# Trading Findings - December 26, 2025

## 10-Minute Live Test Results

**Session:** 2025-12-26, ~10 minutes paper trading on Hostinger VPS

### Raw Numbers

| Metric | Value |
|--------|-------|
| Duration | 598.7 seconds |
| Signals Detected | 51 |
| SHORT Signals | 50 |
| LONG Signals | 1 |
| Trades Executed | 58 |
| Win Rate | 63.8% |
| Total P&L | -$144.33 |
| Starting Capital | $100.00 |
| Ending Capital | -$44.33 |

### Per-Exchange Performance

| Exchange | Leverage | P&L |
|----------|----------|-----|
| Coinbase | 10x | -$68.62 |
| Kraken | 50x | -$71.22 |
| Gemini | 100x | -$4.48 |

### Flow Data

- Total Inflow: 916.58 BTC
- Total Outflow: 737.43 BTC
- Net: +179.15 BTC inflow
- Avg Latency: 1,137 microseconds

---

## The Original Thesis

```
INFLOW to exchange  → Deposit to SELL → Price DOWN → SHORT
OUTFLOW from exchange → Seller exhaustion → Price UP → LONG
```

**Expected:** 100% win rate (deterministic)

**Actual:** 63.8% win rate, net loss

---

## What Actually Happened

### Price Movement
- Entry range: $88,292 - $88,635
- Exit range: $88,598 - $88,635
- Direction: **UP** (+0.15%)

### Trade Outcomes
Most trades were SHORTs that timed out (5 min) with price higher than entry:

```
[CLOSE] SHORT COINBASE | Entry: $88,509.99 -> Exit: $88,604.30 | P&L: -2.3%
[CLOSE] SHORT GEMINI   | Entry: $88,520.18 -> Exit: $88,604.30 | P&L: -10.3%
[CLOSE] SHORT KRAKEN   | Entry: $88,292.15 -> Exit: $88,604.30 | P&L: -18.2%
```

### Leverage Impact
Higher leverage = larger losses when wrong:
- Coinbase 10x: -2.3% per trade
- Gemini 100x: -10.3% per trade
- Kraken 50x: -18.2% per trade

---

## Key Discoveries

### 1. Thesis Not Validated in Real-Time
The "INFLOW → SHORT = 100% win" assumption did not hold during this 10-minute window. Price moved opposite to prediction.

### 2. Timing Mismatch
- We detect deposit in ~1ms (nanosecond C++ runner)
- Position timeout: 5 minutes
- Actual price impact may occur over different timeframe (hours? days?)

### 3. Leverage Amplifies Losses
With 100% win rate, max leverage = max profit. With 63.8% win rate, max leverage = max loss.

### 4. Sample Size Issue
- 10 minutes is too short
- Need to observe over hours/days to validate thesis
- Historical correlation may not equal real-time causation

### 5. Market Context Matters
During this session, BTC was in a slight uptrend. The inflow signals may be correct on average but not in every market condition.

---

## Pattern Stats from Session

```
coinbase  INFLOW  bucket=(10-50)    samples=391 corr=0.02 win=62.1%
coinbase  INFLOW  bucket=(0-1)      samples=246 corr=0.13 win=62.2%
coinbase  INFLOW  bucket=(1-5)      samples=124 corr=0.08 win=54.0%
binance   INFLOW  bucket=(0-1)      samples=82  corr=0.08 win=52.4%
bitfinex  INFLOW  bucket=(1-5)      samples=72  corr=-0.19 win=66.7%
kraken    INFLOW  bucket=(0-1)      samples=13  corr=0.40 win=53.8%
```

**Observation:** Historical win rates are 52-67%, not 90%+. The "tradeable when win_rate >= 0.9" threshold was never met.

---

## Questions to Investigate

1. **Timeframe:** What is the optimal holding period? 5 min may be too short.
2. **Flow Size:** Do larger flows (>100 BTC) predict better than small flows?
3. **Exchange Specificity:** Does the pattern work better on certain exchanges?
4. **Market Regime:** Does it work in downtrends but fail in uptrends?
5. **Causation vs Correlation:** Is the flow actually causing price movement, or just correlated?

---

## Next Steps

1. Run for longer period (1 hour, 24 hours) to get more data
2. Analyze which trades won vs lost - what was different?
3. Consider adjusting timeout from 5 min to longer
4. Consider reducing leverage until pattern is validated
5. Track flow size vs outcome correlation

---

## Raw Trade Log (Sample)

```
[OPEN] SHORT COINBASE @ $88,509.99 | Size: $250 | SL: $89,395 | TP: $86,740
[OPEN] SHORT GEMINI @ $88,520.18 | Size: $2,500 | SL: $89,405 | TP: $86,750
[OPEN] SHORT KRAKEN @ $88,292.15 | Size: $1,250 | SL: $89,175 | TP: $86,526
...
[CLOSE] SHORT COINBASE | Entry: $88,509.99 -> Exit: $88,604.30 | P&L: -$0.58 | TIMEOUT
[CLOSE] SHORT GEMINI | Entry: $88,520.18 -> Exit: $88,604.30 | P&L: -$2.58 | TIMEOUT
[CLOSE] SHORT KRAKEN | Entry: $88,292.15 -> Exit: $88,604.30 | P&L: -$4.55 | TIMEOUT
```

---

## Conclusion

The deterministic thesis requires validation. 10 minutes of data shows 63.8% win rate with net loss. High leverage without high win rate = capital destruction.

**Data speaks. We listen.**

---

## Second Test: Data-Driven Adjustments

**Changes made based on first test:**
1. Exit timeout: 5 min → 1 hour
2. Min flow filter: 0 → 10 BTC (skip small flows)

### Results

| Metric | Test 1 | Test 2 |
|--------|--------|--------|
| Win Rate | 63.8% | 60% (3/5) |
| Total P&L | -$144.33 | -$157.47 |
| Trades | 58 | 5 |

### What Happened

**Stop-losses triggered:**
```
SHORT KRAKEN   | Entry: $87,007 -> Exit: $89,204 | P&L: -$84.01 | STOP_LOSS
SHORT COINBASE | Entry: $87,151 -> Exit: $89,204 | P&L: -$78.57 | STOP_LOSS
```

**Price moved +2.5% UP** while we were SHORT. The 1% stop-loss triggered.

### Flow Filter Worked

```
[SKIP] Flow 1.52 BTC < 10.0 BTC minimum  (bitfinex)
[SKIP] Flow 1.30 BTC < 10.0 BTC minimum  (gemini)
[SKIP] Flow 1.63 BTC < 10.0 BTC minimum  (crypto.com)
```

Fewer trades, but **same result** - thesis didn't hold.

---

## Hard Truth: The Thesis Is Not 100%

The data across both tests is consistent:

| What We Expected | What Data Shows |
|------------------|-----------------|
| INFLOW → Price DOWN | Price went UP (+2.5%) |
| 100% win rate | 60-64% win rate |
| Deterministic | Probabilistic at best |

### Possible Explanations

1. **Timing**: Deposit ≠ immediate sell. Depositors may wait.
2. **Market Force**: Overall buying pressure > inflow selling pressure
3. **Correlation ≠ Causation**: Inflows correlate with activity, not direction
4. **Sample Size**: Need more data to find the edge

---

## Current Data Summary

```
Total Tests:  2 sessions (~20 minutes)
Total Trades: 63
Win Rate:     ~62%
Total P&L:    -$301.80
```

The thesis as stated ("100% deterministic") is **falsified by data**.

---

## What Data Suggests

1. **The edge is weaker than expected** (62% not 100%)
2. **Leverage must match win rate** (62% win rate = low leverage)
3. **Timing window matters** (5 min or 1 hour both fail)
4. **Flow size filter helps** (fewer noisy trades, same outcome)

---

## Next: Listen More

Options:
1. Run 24-hour test to get more data
2. Analyze historical correlation data vs live results
3. Accept ~60% win rate and size accordingly
4. Find what differentiates winning vs losing trades

**Data doesn't lie. We adapt.**

---

## Noise Filter Implementation (Option 4)

### What We Analyzed

Queried correlation.db for patterns with high sample counts:

```
SELECT exchange, direction, bucket_min, bucket_max, sample_count, win_rate
FROM patterns
WHERE sample_count >= 30
ORDER BY win_rate DESC;
```

**Result:** Best patterns are 60-67% win rate. No patterns have 90%+ win rate.

### Patterns That Pass Filter (≥30 samples, ≥60% win rate)

| Exchange | Direction | Bucket | Win Rate | Samples |
|----------|-----------|--------|----------|---------|
| bitfinex | INFLOW | 1-5 BTC | 66.7% | 72 |
| coinbase | OUTFLOW | 10-50 BTC | 62.9% | 35 |
| coinbase | INFLOW | 0-1 BTC | 62.2% | 246 |
| coinbase | INFLOW | 10-50 BTC | 62.1% | 391 |

### Config Changes

```python
# Before (optimistic)          # After (data-driven)
min_win_rate = 0.90            min_win_rate = 0.60
min_sample_size = 10           min_sample_size = 30
max_leverage = 100             max_leverage = 5
position_size_pct = 0.25       position_size_pct = 0.10
max_positions = 4              max_positions = 2
```

### Code Changes

1. **signals.py**: Added `get_pattern_stats()` to query correlation.db
2. **run.py**: Added pattern validation before trading
3. **config.py**: Updated thresholds based on actual data

### Effective Tradeable Patterns

With `min_flow_btc = 10.0`:
- coinbase INFLOW 10-50 BTC: 62.1% win rate, 391 samples
- coinbase OUTFLOW 10-50 BTC: 62.9% win rate, 35 samples

### Position Sizing for 62% Edge

Kelly Criterion: `f = (0.62 * 2 - 0.38) / 2 = 0.43` (43% of bankroll)

Using fractional Kelly (1/4 Kelly) for safety: ~10% position size at 5x leverage.

Expected value per trade: `0.62 * (+2%) + 0.38 * (-1%) = +0.86%`

---

## Status: Noise Filter Deployed

The system now:
1. Skips flows < 10 BTC
2. Validates pattern stats from correlation.db before trading
3. Only trades patterns with ≥30 samples AND ≥60% win rate
4. Uses conservative position sizing (5x, 10% per trade)

**Next:** Monitor live performance to validate filtered patterns perform at expected 62% win rate.
