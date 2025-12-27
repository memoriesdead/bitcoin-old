# RenTech-Style Bitcoin Edge Detection

## The Insight

We don't need to know WHICH addresses are exchanges.
We need to find PATTERNS in blockchain data that PREDICT price movements.

RenTech's edge comes from finding statistical relationships that others miss.
We have a 10-60 second information advantage. Now we exploit it.

---

## Phase 1: Raw Data Collection (Week 1-2)

### What We Collect Every Second

From Bitcoin Core ZMQ (raw transactions):

| Metric | Description | Why It Might Matter |
|--------|-------------|---------------------|
| `tx_count` | Transactions per second | Activity spikes precede volatility |
| `tx_volume_btc` | Total BTC moved per second | Large volume = institutional activity |
| `avg_tx_size` | Average transaction size | Whale vs retail activity |
| `large_tx_count` | Transactions > 10 BTC | Whale movements |
| `whale_tx_count` | Transactions > 100 BTC | Major institutional moves |
| `median_fee_rate` | Median sat/vB | Urgency indicator |
| `fee_pressure` | 90th percentile fee rate | Competition for block space |
| `input_count` | Total inputs across all txs | Consolidation activity |
| `output_count` | Total outputs across all txs | Distribution activity |
| `consolidation_ratio` | inputs / outputs | >1 = consolidation, <1 = distribution |
| `segwit_ratio` | % SegWit transactions | Modern wallet activity |
| `taproot_ratio` | % Taproot transactions | Cutting-edge wallet activity |
| `mempool_size` | Total mempool size | Network congestion |
| `mempool_fees` | Total fees in mempool | Economic activity |

From Price Feed (Kraken WebSocket):

| Metric | Description |
|--------|-------------|
| `price` | Current BTC/USD |
| `bid` | Best bid |
| `ask` | Best ask |
| `spread` | ask - bid |
| `price_1m` | Price 1 minute ago |
| `price_5m` | Price 5 minutes ago |
| `return_1m` | 1-minute return |
| `return_5m` | 5-minute return |

---

## Phase 2: Feature Engineering (Week 2-3)

### Derived Features

```
# Momentum features
tx_count_zscore = (tx_count - rolling_mean) / rolling_std
volume_acceleration = volume_now - volume_1min_ago
fee_spike = current_fee / avg_fee_24h

# Pattern features
consolidation_streak = consecutive minutes of consolidation_ratio > 1
whale_cluster = whale_txs in last 5 min > threshold
volume_divergence = btc_volume_change - price_change (divergence signals)

# Timing features
minutes_since_block = time since last block
block_fullness = last block size / max block size
```

### Target Variables (What We Predict)

```
return_1m_future = price in 1 min / price now - 1
return_5m_future = price in 5 min / price now - 1
return_15m_future = price in 15 min / price now - 1
volatility_5m_future = std(returns) over next 5 min
```

---

## Phase 3: Statistical Analysis (Week 3-4)

### For Each Feature, Calculate:

1. **Correlation** with future returns (1m, 5m, 15m)
2. **Lead-lag relationship** - does feature lead price?
3. **Information coefficient** - predictive power
4. **Stability** - does relationship persist over time?

### Key Questions to Answer:

- Do whale transactions (>100 BTC) predict price moves?
- Does fee pressure predict volatility?
- Does consolidation ratio predict direction?
- Which features have the strongest signal?

---

## Phase 4: Model Building (Week 4-5)

### Start Simple

```python
# Linear model for interpretability
signal = w1 * whale_tx_zscore +
         w2 * fee_pressure_zscore +
         w3 * consolidation_ratio_zscore +
         ...

# If signal > threshold: LONG
# If signal < -threshold: SHORT
```

### Backtest Requirements

- Minimum 1000 signals for statistical significance
- Account for transaction costs (Hyperliquid fees)
- Test on out-of-sample data
- Calculate Sharpe ratio, win rate, max drawdown

---

## Phase 5: Paper Trading (Week 5-6)

### Execution Plan

1. Run model in real-time
2. Log signals WITHOUT executing
3. Track what WOULD have happened
4. Measure actual edge vs theoretical

### Success Criteria

- Sharpe > 1.5
- Win rate > 52%
- Edge per trade > 10 bps after fees

---

## Phase 6: Live Trading (Week 7+)

### Position Sizing (Kelly Criterion)

```
f* = (p * b - q) / b

Where:
p = win probability
q = 1 - p
b = avg win / avg loss
f* = fraction of capital to risk
```

### Risk Management

- Max position: 20% of capital
- Stop loss: 2% per trade
- Daily loss limit: 5%
- Correlation limit: don't stack same signals

---

## Implementation: Updated Collector

The new collector extracts ALL metrics, not just address matching.

```python
# Every transaction we process:
- Count it
- Sum the BTC value
- Track if it's a whale tx
- Calculate fee rate
- Track input/output counts
- Track SegWit/Taproot usage

# Every second we emit:
- Aggregated metrics
- Price at that moment
- Store for analysis
```

---

## The Edge

Everyone else using public APIs sees transactions 10-60 seconds AFTER us.

If whale activity at T=0 predicts price at T=60s, we can:
1. See the whale tx at T=0
2. Enter position at T=1
3. Exit when price moves at T=60

That's the edge. Not knowing WHO is trading, but knowing WHAT the blockchain activity predicts.

---

## Next Steps

1. Deploy the comprehensive metric collector
2. Run for 2 weeks minimum
3. Analyze correlations
4. Build predictive model
5. Paper trade
6. Go live

This is how RenTech would do it.
