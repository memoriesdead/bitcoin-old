# Plan: Simple 100% Win Rate

## THE PROBLEM

After 25 days of testing:
- Complex math formulas: **0% win rate** (210 losses, 0 wins)
- Simple flow trading: **50% win rate** (better, but not 100%)

We overcomplicated it. Time to go back to basics.

---

## THE EDGE (What's Real)

```
We see TX on blockchain → Exchange address matched → Trade BEFORE price moves
```

This timing advantage is REAL. But not every deposit leads to immediate price movement.

---

## DATA WE CAN EXTRACT FROM BLOCKCHAIN

### Per Transaction:
| Data Point | What It Tells Us | Can Extract? |
|------------|------------------|--------------|
| **Amount (BTC)** | Size of flow | YES |
| **Exchange address** | Which exchange | YES |
| **Direction** | Inflow vs Outflow | YES |
| **TX fee (sat/vB)** | Urgency of sender | YES |
| **TX size (bytes)** | Complexity | YES |
| **Input count** | Consolidation pattern | YES |
| **Output count** | Distribution pattern | YES |
| **UTXO age** | How long coins were held | YES |
| **RBF flag** | Replace-by-fee enabled | YES |
| **Locktime** | Time constraints | YES |
| **Witness data** | SegWit vs legacy | YES |
| **Mempool position** | Priority | YES (from mempool) |
| **Confirmation status** | 0-conf vs confirmed | YES |

### Per Address Cluster (Historical):
| Data Point | What It Tells Us | Can Extract? |
|------------|------------------|--------------|
| **Historical sell timing** | How fast they sell after deposit | YES (from our DB) |
| **Typical deposit size** | Normal vs abnormal | YES |
| **Frequency of deposits** | Regular trader vs one-time | YES |
| **Previous P&L on signals** | Did our trades win? | YES (from trade DB) |

---

## HYPOTHESIS: What Makes 100% Certain?

### Theory 1: TX Fee = Urgency
```
HIGH FEE deposit → They paid premium → URGENT → Immediate sell → SHORT
LOW FEE deposit  → No rush → Might hold → SKIP
```

**Test:** Compare win rate of high-fee vs low-fee deposits

### Theory 2: UTXO Age = Conviction
```
OLD COINS moved → Long-term holder finally selling → BIG move → SHORT
YOUNG COINS moved → Day trader, noise → SKIP
```

**Test:** Compare win rate by UTXO age

### Theory 3: Input Consolidation = Preparation to Sell
```
MANY INPUTS → Consolidating to sell everything → SHORT
ONE INPUT   → Might be partial, uncertain → SKIP
```

**Test:** Compare win rate by input count

### Theory 4: Historical Address Behavior
```
This cluster ALWAYS sells within 5 min of deposit → SHORT
This cluster often holds for hours → SKIP
```

**Test:** Track per-cluster behavior over time

### Theory 5: Confirmation Status
```
Wait for 1 confirmation → More certain it's real → Trade
0-conf → Could be RBF replaced → SKIP
```

**Test:** Compare 0-conf vs 1-conf signal accuracy

### Theory 6: Time-Based Patterns
```
Deposit during US market hours → Active trading → More likely to sell
Deposit at 3 AM UTC → Might be cold storage move → SKIP
```

**Test:** Compare win rate by hour of day

---

## WHAT TO BUILD

### Phase 1: Data Collection (No Trading)
```python
# For every exchange inflow, record:
{
    "txid": "...",
    "exchange": "coinbase",
    "amount_btc": 15.5,
    "fee_sat_vb": 45,
    "input_count": 3,
    "utxo_ages_blocks": [1000, 50000, 100],  # Age of each input
    "oldest_utxo_blocks": 50000,
    "rbf_enabled": False,
    "timestamp": "2025-12-26T04:00:00Z",
    "mempool_position": 15,  # If available

    # Track outcome
    "price_at_t0": 89000,
    "price_at_t1min": 88950,
    "price_at_t5min": 88800,
    "price_at_t10min": 88900,
    "price_moved_down": True,
    "max_down_move_pct": 0.22
}
```

### Phase 2: Analysis
After collecting 100+ data points:
- Which single factor has highest correlation with price drop?
- Is there a combination that's 100%?

### Phase 3: Simple Filter
```python
# Only trade when ALL conditions met:
if (
    fee_sat_vb > 50 and           # Urgent
    oldest_utxo_blocks > 10000 and # Not day trader
    input_count >= 2 and           # Consolidating
    amount_btc > 10                # Significant size
):
    TRADE_SHORT()
else:
    SKIP()  # Don't trade uncertain signals
```

---

## THE KEY INSIGHT

> "100% win rate comes from FILTERING, not from complex math"

We don't need formulas. We need to SKIP uncertain signals.

If we only trade the 10% of signals that are 100% certain,
we have 100% win rate (on fewer trades).

Better: 10 trades at 100% win rate
Than: 210 trades at 0% win rate

---

## IMMEDIATE NEXT STEPS

1. **Modify C++ runner** to output additional TX data:
   - Fee rate (sat/vB)
   - Input count
   - UTXO ages (need to look up inputs)

2. **Create data collection script** - record everything, trade nothing

3. **Run for 24-48 hours** - collect 100+ inflow data points

4. **Analyze** - find the 100% filter

5. **Test filter** - paper trade with filter

6. **Deploy** - only trade filtered signals

---

## QUESTIONS TO ANSWER

1. Do high-fee deposits predict immediate sells better than low-fee?
2. Do old-coin deposits predict bigger moves?
3. Do certain exchanges have more predictable patterns?
4. Is there a time-of-day effect?
5. Does confirmation status matter?

Let the DATA tell us what works, not complex math assumptions.
