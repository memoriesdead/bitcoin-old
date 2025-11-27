# Edge Mathematics - Expected Value Analysis

## Executive Summary

Your live trading data shows ALL strategies have negative edge. Based on comprehensive research, this document explains why and what mathematical requirements must be met for profitability.

## Core Formula: Expected Value (EV)

### Mathematical Definition

```
EV = (Win Rate × Average Win) - (Loss Rate × Average Loss)
```

Alternative expression:
```
EV = P(right) × R(right) - P(wrong) × R(wrong)
```

**Rule**: Positive EV = Profitable | Negative EV = Losing

### Your Live Data Analysis

| Strategy | Trades | WR% | Edge/Trade | Analysis |
|----------|--------|-----|------------|----------|
| V1 OFI | 108,487 | 0% | -$0.0001 | Catastrophic failure |
| V2 Hawkes | 154,679 | 1% | -$0.0000 | Essentially random |
| V3 VPIN | 9,744 | 27% | -$0.0000 | Best WR but still losing |
| V4 OU | 8,613 | 26% | -$0.0001 | Mean reversion failing |
| V5 Kalman | 1,940 | 0% | -$0.0003 | Filter not working |
| V6 HMM | 7,528 | 28% | -$0.0000 | Best WR, no edge |
| V7 Kyle | 38,873 | 5% | -$0.0001 | Microstructure mismatch |
| V8 Master | 0 | N/A | $0.0000 | Over-filtered |

**Total Trades**: 329,864
**Total Capital Lost**: ~$26 of $80 (32.5% drawdown)

## Break-Even Analysis

### Break-Even Win Rate Formula

```
Breakeven WR = Risk / (Risk + Reward)
```

### Risk:Reward Requirements

| R:R Ratio | Required Win Rate | Your V3/V4/V6 Reality (27%) |
|-----------|-------------------|----------------------------|
| 1:1 | 50% | BELOW breakeven |
| 1:1.5 | 40% | BELOW breakeven |
| 1:2 | 33% | BELOW breakeven |
| 1:2.7 | 27% | AT breakeven (no profit) |
| 1:3 | 25% | ABOVE breakeven |

**Critical Finding**: Your 27% win rate needs an R:R of at least 1:2.7 just to break even. With fees, you need 1:3 minimum for profit.

### Why High Win Rates Don't Guarantee Profit

From research: "Even with a 90% win rate you can lose money if the formula calculates 90% × $1 + 10% × -$10 = -$0.10"

**Your V6 Example**:
- 28% WR with negative edge means: `0.28 × AvgWin - 0.72 × AvgLoss < 0`
- Solving: `AvgLoss > 0.389 × AvgWin`
- Your losses are MORE than 2.5× your wins

## Profit Factor Analysis

### Formula

```
Profit Factor = (Win% × Avg Win) / (Loss% × Avg Loss)
```

**Target**: > 1.5 (minimum 1.0 to break even)

### Your Estimated Profit Factors

Based on negative edge and capital loss:

| Strategy | Est. PF | Status |
|----------|---------|--------|
| V1 | < 0.5 | Severe loss |
| V2 | < 0.5 | Severe loss |
| V3 | ~0.95 | Close to breakeven |
| V4 | ~0.93 | Slight loss |
| V5 | < 0.5 | Severe loss |
| V6 | ~0.97 | Slight loss |
| V7 | ~0.65 | Moderate loss |

## Minimum Edge Requirements

### Target Calculation

**Goal**: $10 → $300,000 = 30,000× return

### Compounding Requirements

```
Final Capital = Initial × (1 + Edge)^N
30,000 = 1 × (1 + Edge)^N
```

For 10,000 trades (realistic):
```
Edge per trade = (30,000)^(1/10000) - 1 = 0.102% = $0.01 per $10
```

**With $10 starting capital**: Minimum +$0.001 edge per trade
**Your current edge**: -$0.0001 (10× worse than needed)

### To Reach $300k in Different Timeframes

| Total Trades | Edge Needed | Annual % (if 250 days) |
|--------------|-------------|------------------------|
| 1,000 | 1.02% | 2,550% |
| 10,000 | 0.102% | 255% |
| 100,000 | 0.0102% | 25.5% |

**Renaissance Medallion**: 66% annual = 0.26% per trading day with ~100 trades/day = 0.0026% per trade

**You need**: 0.002% minimum edge per trade = $0.0002 on $10 capital

## Why Your Current Strategies Fail

### 1. High Frequency Death Spiral (V1, V2, V7)

```
V1: 108,487 trades × 0.04% taker fee = 43.4× capital in fees
V2: 154,679 trades × 0.04% taker fee = 61.9× capital in fees
```

**Problem**: You're paying MORE in fees than your starting capital

### 2. Win Rate Without Favorable R:R (V3, V4, V6)

27% WR requires:
- R:R of 2.7:1 to break even
- R:R of 3:1+ to be profitable after fees

**Current Reality**: Your strategies appear to have ~1:1 or worse R:R based on capital loss

### 3. Zero Signal Quality (V5, V8)

- V5: 0% win rate = random entries with poor exits
- V8: 0 trades = VPIN filter is too strict

## Mathematical Path to Positive Edge

### Required Changes

1. **Reduce Trade Frequency by 10×**
   - From 154,679 trades → 15,000 trades
   - Fee drag: 61.9× capital → 6× capital

2. **Improve R:R to 3:1 minimum**
   - Average win: $0.30
   - Average loss: $0.10
   - At 27% WR: EV = 0.27 × $0.30 - 0.73 × $0.10 = $0.008 (positive!)

3. **Increase Win Rate to 35%+ OR**
   - At 35% WR with 2:1 R:R: EV = 0.35 × $0.20 - 0.65 × $0.10 = $0.005 (positive!)

4. **Combine Both**
   - 35% WR with 3:1 R:R: EV = 0.35 × $0.30 - 0.65 × $0.10 = $0.040 (excellent!)

## Actionable Metrics

### Minimum Targets for Profitability

```python
# Minimum viable strategy
MIN_WIN_RATE = 0.30  # 30%
MIN_RISK_REWARD = 2.5  # 2.5:1
MAX_TRADES_PER_DAY = 50  # vs your 17,000+
MIN_EDGE_PER_TRADE = 0.0002  # $0.002 on $10

# Optimal strategy
TARGET_WIN_RATE = 0.40  # 40%
TARGET_RISK_REWARD = 3.0  # 3:1
TARGET_TRADES_PER_DAY = 10-20
TARGET_EDGE_PER_TRADE = 0.001  # $0.01 on $10
```

## Formula Verification

### Your V3 Strategy Example

**Current**:
- Trades: 9,744
- WR: 27%
- Capital: $8.78 (loss: $1.22)

**Implied Metrics**:
```
Loss = $1.22
Fee estimate = 9,744 × 0.0004 × $10 = $38.98 (using avg capital)
Total negative PnL from trades = $1.22 - fees
```

**What's needed**:
```
Target: 27% WR with 3:1 R:R
- Wins: 2,631 × $0.30 = $789
- Losses: 7,113 × $0.10 = -$711
- Net: $78
- After fees (9,744 × 0.04% × $10): $78 - $39 = $39 profit
- Final capital: $49 (490% return)
```

## Sources

- [Trading Expectancy Calculator](https://enlightenedstocktrading.com/trading-expectancy-calculator/)
- [Expected Value in Trading - DayTrading.com](https://www.daytrading.com/expected-value)
- [Understanding Edge - Rithmm](https://www.rithmm.com/articles/understanding-the-math-behind-edge-expected-value)
- [Breakeven Win Rate Calculator - MarketBulls](https://market-bulls.com/breakeven-win-rate-calculator/)
- [Risk-Reward vs Win Rate - LuxAlgo](https://www.luxalgo.com/blog/risk-reward-ratio-vs-win-rate-key-differences-2/)

## Next Steps

1. Calculate actual R:R ratios from trade logs
2. Measure actual average win vs average loss
3. Implement hard limits on trade frequency
4. Add pre-trade edge calculation
5. Only take trades with +0.0005 minimum expected edge
