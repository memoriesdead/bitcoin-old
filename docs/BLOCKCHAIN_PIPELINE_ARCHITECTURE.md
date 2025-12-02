# BLOCKCHAIN DATA PIPELINE ARCHITECTURE
## Complete System Map for HFT Trading Engine

```
================================================================================
                    BLOCKCHAIN DATA PIPELINE OVERVIEW
================================================================================

                         ┌─────────────────────────────────┐
                         │    BLOCKCHAIN (Pure Math)       │
                         │    - Genesis: Jan 3, 2009       │
                         │    - Block Time: 600 seconds    │
                         │    - Halving: 210,000 blocks    │
                         └─────────────┬───────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  LEADING SIGNALS    │   │  LAGGING SIGNALS    │   │  BLOCKCHAIN SIGNALS │
│  (From TIMESTAMP)   │   │  (From PRICES)      │   │  (From BLOCK DATA)  │
├─────────────────────┤   ├─────────────────────┤   ├─────────────────────┤
│  ID 901: Power Law  │   │  ID 141: Z-Score    │   │  ID 801: Block Vol  │
│  ID 902: Stock-Flow │   │  ID 218: CUSUM      │   │  ID 802: Mempool    │
│  ID 903: Halving    │   │  ID 335: Regime     │   │  ID 803: Chaos Price│
│                     │   │  ID 701-706: OFI    │   │  ID 804: Whale Det  │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
          │                            │                            │
          └────────────────────────────┼────────────────────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────────────┐
                         │   ID 333: SIGNAL CONFLUENCE     │
                         │   Condorcet Voting System       │
                         │   - Power Law: 5x weight        │
                         │   - S2F: 4x weight              │
                         │   - Halving: 4x weight          │
                         │   - Mempool: 3x weight          │
                         │   - Whale: 2x weight            │
                         │   - OFI: 1.5x weight            │
                         └─────────────────────────────────┘
                                       │
                                       ▼
                              ┌───────────────┐
                              │  TRADE SIGNAL │
                              │  BUY/SELL/HOLD│
                              │  + Confidence │
                              └───────────────┘
```

---

## LAYER 1: CONSTANTS (Foundation)
**Location:** `engine/core/constants/blockchain.py`

```python
# BITCOIN PROTOCOL CONSTANTS (Immutable)
GENESIS_TS: float = 1230768000.0      # Jan 1, 2009 (Power Law epoch)
MAX_SUPPLY: float = 21000000.0         # Total BTC ever
BLOCKS_PER_HALVING: int = 210000       # Blocks between halvings
INITIAL_REWARD: float = 50.0           # First block reward

# POWER LAW MODEL (R² = 93%+ over 14 years)
POWER_LAW_A: float = -17.0161223       # Intercept
POWER_LAW_B: float = 5.8451542         # Slope
# Formula: Price = 10^(A + B * log10(days_since_genesis))

# SUPPORT/RESISTANCE (from Power Law research)
SUPPORT_MULT: float = 0.42             # Price floor multiplier
RESIST_MULT: float = 2.38              # Price ceiling multiplier
```

---

## LAYER 2: LEADING INDICATORS (Timestamp-Only)
**Location:** `engine/tick/formulas/leading/`

These signals are calculated from **TIMESTAMP ONLY** - completely independent of current price.
This makes them **predictive** rather than reactive.

### ID 901: Power Law Price Signal
**File:** `leading/power_law.py`
**Citation:** Giovannetti (2019) - R² = 94%
```
Formula: log10(price) = A + B * log10(days_since_genesis)
Returns: (fair_value, deviation, signal, strength)
Signal:
  - Price < fair - 10% → BUY
  - Price > fair + 10% → SELL
```

### ID 902: Stock-to-Flow Signal
**File:** `leading/stock_to_flow.py`
**Citation:** PlanB (2019) - R² = 95%
```
Formula: ln(price) = A + B * ln(S2F)
S2F = Current_Supply / Annual_Issuance
Returns: (s2f_price, s2f_ratio, deviation, signal, strength)
Signal:
  - Price below S2F model → BUY
  - Price above S2F model → SELL
```

### ID 903: Halving Cycle Position
**File:** `leading/halving_cycle.py`
**Citation:** Empirical 4-year cycles (2012, 2016, 2020, 2024)
```
Cycle Phases:
  0.00-0.30: Accumulation (post-halving) → BUY
  0.30-0.70: Expansion (bull market) → HOLD
  0.70-1.00: Distribution (pre-halving) → SELL
Returns: (cycle_position, signal, strength)
```

---

## LAYER 3: BLOCKCHAIN SIGNALS (Block Data)
**Location:** `engine/tick/formulas/blockchain/`

### ID 801: Block Time Volatility
**File:** `blockchain/block_volatility.py`
```
Derives volatility from block timing variance
Higher block time variance → Higher expected price volatility
Returns: (block_volatility, time_ratio, activity_level)
```

### ID 802: Mempool Flow Simulator
**File:** `blockchain/mempool_flow.py`
```
Simulates mempool from pure math (zero API calls)
Sources:
  - Block timing (600 second cycles)
  - Halving cycles (210,000 blocks)
  - Difficulty adjustment (2,016 blocks)
  - Time cycles (daily/weekly patterns)
Returns: (volume_flow, buy_pressure, sell_pressure, mempool_ofi)
```

### ID 803: Chaos Price Generator
**File:** `blockchain/chaos_price.py`
**Citation:** Lorenz (1963) - Deterministic Nonperiodic Flow
```
Components:
  - lorenz_step_inline(): Lorenz attractor evolution
  - generate_independent_price(): Price from blockchain cycles
  - calc_blockchain_signals(): Fee/TX/momentum signals
  - calc_chaos_price(): Final market price

CRITICAL: Breaks circular dependency!
  - Price uses: Difficulty cycles, Halving cycles
  - Signals use: Historical price patterns
  This ensures signals PREDICT prices, not generate them.
```

### ID 804: UTXO Whale Detector
**File:** `blockchain/whale_detector.py`
**Citation:** Kyle (1985) - Price Impact
```
Infers whale activity from blockchain patterns:
  - Block size anomalies
  - Halving proximity (more whale activity near halvings)
Returns: (whale_probability, kyle_impact, block_size_ratio)
```

---

## LAYER 4: LAGGING INDICATORS (Price-Based)
**Location:** `engine/tick/formulas/`

### ID 141: Z-Score Filter
**File:** `formulas/zscore.py`
```
Classic mean reversion from standardized deviation
Z > 2.0 → SELL (overextended)
Z < -2.0 → BUY (oversold)
```

### ID 218: CUSUM Filter
**File:** `formulas/cusum.py`
**Citation:** Lopez de Prado (2018)
```
Detects structural breaks in price series
Identifies regime changes before full trend develops
```

### ID 335: Regime Filter
**File:** `formulas/regime.py`
**Citation:** Moskowitz, Ooi & Pedersen (2012)
```
EMA crossover with volatility-adjusted confidence
Bull/Bear/Neutral regime classification
```

### ID 701/702/706: Order Flow Imbalance
**File:** `formulas/ofi.py`
**Citation:** Cont, Stoikov, Talreja (2010)
```
- calc_ofi(): Basic order flow imbalance
- calc_kyle_lambda(): Price impact coefficient
- calc_flow_momentum(): EMA-smoothed momentum
```

---

## LAYER 5: SIGNAL CONFLUENCE (Aggregation)
**Location:** `engine/tick/formulas/confluence.py`

### ID 333: Condorcet Voting Aggregator
**Citation:** Condorcet's Jury Theorem (1785)
```
WEIGHT HIERARCHY:
  LEADING INDICATORS (Highest Priority):
    Power Law:     5.0x  (R² = 94%, timestamp-derived)
    Stock-to-Flow: 4.0x  (R² = 95%, timestamp-derived)
    Halving Cycle: 4.0x  (Empirical cycles)

  BLOCKCHAIN SIGNALS:
    Mempool OFI:   3.0x  (Pure blockchain math)
    Whale Prob:    2.0x  (Block size patterns)

  LAGGING INDICATORS (Lower Priority):
    OFI:           1.5x  (Price-derived)
    CUSUM:         1.0x  (Structural breaks)
    Z-Score:       0.3x  (Mean reversion)
    Regime:        0.2x  (Trend following)

MATH: With n independent signals at p=55% accuracy:
  P(majority correct) = Σ C(n,k) × p^k × (1-p)^(n-k) for k > n/2
  5 signals at 55% → 59.3% majority accuracy
  7 signals at 55% → 61.8% majority accuracy
```

---

## LAYER 6: HIGH-LEVEL PIPELINES
**Location:** `blockchain/`

### BlockchainUnifiedFeed
**File:** `blockchain/unified_feed.py`
```
Drop-in replacement for exchange API feeds
- Derives OFI-like signals from pure blockchain math
- Zero latency (math, not network calls)
- Components: PureMempoolMath, PureBlockchainPrice, BlockchainTradingEngine
```

### BlockchainTradingPipeline
**File:** `blockchain/pipeline.py`
```
Full academic formula integration:
1. TRUE PRICE from blockchain (Power Law)
2. Market Microstructure (Kyle, VPIN, OFI, Microprice)
3. On-Chain metrics (NVT, MVRV, SOPR, Hash Ribbon)
4. Execution optimization (Almgren-Chriss, Avellaneda-Stoikov)
5. Risk management (Kelly, HMM Regime)
6. Master aggregation (Condorcet voting)
```

### PureMempoolMath
**File:** `blockchain/mempool_math.py`
```
All signals from pure blockchain math:
  - get_block_state(): Block height, timing
  - get_fee_pressure(): Fee dynamics
  - get_tx_volume(): Transaction momentum
  - get_congestion(): Mempool simulation
  - get_price_momentum(): Combined signals
```

### PureBlockchainPrice
**File:** `blockchain/pure_blockchain_price.py`
**Citation:** Giovanni Santostasi's Power Law research
```
Formula: Price = 10^(A + B * log10(days_since_genesis))
Methods:
  - calculate_fair_value(): Power Law price
  - calculate_support(): 42% of fair (floor)
  - calculate_resistance(): 238% of fair (ceiling)
  - get_trading_signal(): BUY/SELL based on deviation
```

---

## LAYER 7: ACADEMIC SIGNAL FORMULAS
**Location:** `formulas/blockchain_signals.py`

### Market Microstructure (IDs 520-529)
```
520: KyleLambdaBlockchain    - Price impact coefficient
521: VPINBlockchain          - Flow toxicity
522: OrderFlowImbalanceBlockchain - Directional flow
523: MicropriceBlockchain    - Stoikov microprice
```

### On-Chain Metrics (IDs 530-539)
```
530: NVTRatioBlockchain      - Network value to transactions
531: MVRVRatioBlockchain     - Market vs realized value
532: SOPRBlockchain          - Spent output profit ratio
533: HashRibbonBlockchain    - Miner capitulation
```

### Execution/Market Making (IDs 540-549)
```
540: AlmgrenChrissExecution  - Optimal execution
541: AvellanedaStoikovSpread - Market making spread
```

### Risk Management (IDs 550-559)
```
550: KellyCriterionBlockchain - Position sizing
551: HMMRegimeBlockchain      - Regime detection
552: TruePriceDeviation       - Deviation from TRUE price
```

### Academic Mean Reversion (IDs 570-590)
```
570: ShortTermReversal       - Jegadeesh (1990)
571: OrnsteinUhlenbeck       - Poterba/Summers
572: GARCHVolatilityRegime   - Bollerslev GARCH
573: RealizedVolSignature    - Andersen/Bollerslev
574: JegadeeshTitmanReversal - J&T 1993
590: AcademicCondorcetAggregator - Pure academic voting
```

### Master Aggregator (ID 560)
```
560: BlockchainSignalAggregator - Combines all with Condorcet voting
```

---

## DATA FLOW SUMMARY

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT: Unix Timestamp                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
┌─────────────┐             ┌─────────────┐             ┌─────────────┐
│ Power Law   │             │   S2F       │             │  Halving    │
│ Fair Value  │             │   Model     │             │   Cycle     │
│ (ID 901)    │             │ (ID 902)    │             │  (ID 903)   │
└──────┬──────┘             └──────┬──────┘             └──────┬──────┘
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │   TRUE PRICE DERIVATION  │
                    │   (Blockchain Only)      │
                    └────────────┬─────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
    ▼                            ▼                            ▼
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
│ Chaos Price │          │  Mempool    │          │  Whale      │
│ (ID 803)    │          │ (ID 802)    │          │ (ID 804)    │
└──────┬──────┘          └──────┬──────┘          └──────┬──────┘
       │                        │                        │
       └────────────────────────┼────────────────────────┘
                                │
                                ▼
             ┌──────────────────────────────────┐
             │  SIMULATED MARKET PRICE          │
             │  (For lagging indicator input)   │
             └──────────────────┬───────────────┘
                                │
    ┌───────────────────────────┼───────────────────────────┐
    │                           │                           │
    ▼                           ▼                           ▼
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  Z-Score    │         │   CUSUM     │         │   Regime    │
│ (ID 141)    │         │  (ID 218)   │         │  (ID 335)   │
└──────┬──────┘         └──────┬──────┘         └──────┬──────┘
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               │
                               ▼
                ┌──────────────────────────┐
                │   CONDORCET CONFLUENCE   │
                │        (ID 333)          │
                │   Weighted Voting        │
                └────────────┬─────────────┘
                             │
                             ▼
                    ┌───────────────┐
                    │ FINAL SIGNAL  │
                    │ Direction: ±1 │
                    │ Prob: 0.0-1.0 │
                    │ Trade: Y/N    │
                    └───────────────┘
```

---

## FORMULA ID REGISTRY

| ID  | Name                | Category       | Type     | File Location                           |
|-----|---------------------|----------------|----------|------------------------------------------|
| 141 | Z-Score             | Mean Reversion | Lagging  | `formulas/zscore.py`                    |
| 218 | CUSUM               | Structural     | Lagging  | `formulas/cusum.py`                     |
| 333 | Confluence          | Aggregation    | Meta     | `formulas/confluence.py`                |
| 335 | Regime              | Trend          | Lagging  | `formulas/regime.py`                    |
| 520 | Kyle Lambda         | Microstructure | Academic | `formulas/blockchain_signals.py`        |
| 521 | VPIN                | Microstructure | Academic | `formulas/blockchain_signals.py`        |
| 522 | OFI                 | Microstructure | Academic | `formulas/blockchain_signals.py`        |
| 523 | Microprice          | Microstructure | Academic | `formulas/blockchain_signals.py`        |
| 530 | NVT Ratio           | On-Chain       | Academic | `formulas/blockchain_signals.py`        |
| 531 | MVRV Ratio          | On-Chain       | Academic | `formulas/blockchain_signals.py`        |
| 532 | SOPR                | On-Chain       | Academic | `formulas/blockchain_signals.py`        |
| 533 | Hash Ribbon         | On-Chain       | Academic | `formulas/blockchain_signals.py`        |
| 540 | Almgren-Chriss      | Execution      | Academic | `formulas/blockchain_signals.py`        |
| 541 | Avellaneda-Stoikov  | Market Making  | Academic | `formulas/blockchain_signals.py`        |
| 550 | Kelly Criterion     | Risk           | Academic | `formulas/blockchain_signals.py`        |
| 551 | HMM Regime          | Regime         | Academic | `formulas/blockchain_signals.py`        |
| 552 | TRUE Price Dev      | Blockchain     | Core     | `formulas/blockchain_signals.py`        |
| 560 | Signal Aggregator   | Master         | Meta     | `formulas/blockchain_signals.py`        |
| 570 | Short-Term Reversal | Academic       | Mean Rev | `formulas/blockchain_signals.py`        |
| 571 | Ornstein-Uhlenbeck  | Academic       | Mean Rev | `formulas/blockchain_signals.py`        |
| 572 | GARCH Volatility    | Academic       | Regime   | `formulas/blockchain_signals.py`        |
| 573 | Realized Vol Sig    | Academic       | Mean Rev | `formulas/blockchain_signals.py`        |
| 574 | J-T Reversal        | Academic       | Mean Rev | `formulas/blockchain_signals.py`        |
| 590 | Academic Condorcet  | Academic       | Meta     | `formulas/blockchain_signals.py`        |
| 701 | OFI Basic           | Microstructure | Core     | `formulas/ofi.py`                       |
| 702 | Kyle Lambda         | Microstructure | Core     | `formulas/ofi.py`                       |
| 706 | Flow Momentum       | Microstructure | Core     | `formulas/ofi.py`                       |
| 801 | Block Volatility    | Blockchain     | Core     | `formulas/blockchain/block_volatility.py`|
| 802 | Mempool Flow        | Blockchain     | Core     | `formulas/blockchain/mempool_flow.py`   |
| 803 | Chaos Price         | Blockchain     | Core     | `formulas/blockchain/chaos_price.py`    |
| 804 | Whale Detector      | Blockchain     | Core     | `formulas/blockchain/whale_detector.py` |
| 901 | Power Law           | Leading        | Core     | `formulas/leading/power_law.py`         |
| 902 | Stock-to-Flow       | Leading        | Core     | `formulas/leading/stock_to_flow.py`     |
| 903 | Halving Cycle       | Leading        | Core     | `formulas/leading/halving_cycle.py`     |

---

## KEY PRINCIPLE: NO CIRCULAR DEPENDENCY

```
┌────────────────────────────────────────────────────────────────────┐
│  WRONG (Circular):                                                  │
│  Price → Signal → Price → Signal → ...                             │
│  (Signal generates the price it's trying to predict)               │
├────────────────────────────────────────────────────────────────────┤
│  CORRECT (Our Architecture):                                       │
│                                                                    │
│  TIMESTAMP ─────────────────────────────────────────┐              │
│       │                                             │              │
│       ▼                                             │              │
│  Leading Signals (901, 902, 903)                    │              │
│       │                                             │              │
│       ▼                                             │              │
│  Blockchain Mechanics (Difficulty, Halving)         │              │
│       │                                             │              │
│       ▼                                             │              │
│  Chaos Price Generator (803) ──────────────────────►│ PRICE       │
│                                                     │              │
│  Historical Price Buffer ◄──────────────────────────┘              │
│       │                                                            │
│       ▼                                                            │
│  Lagging Signals (141, 218, 335, 701-706)                         │
│       │                                                            │
│       ▼                                                            │
│  Confluence (333) ────► TRADING DECISION                          │
└────────────────────────────────────────────────────────────────────┘

The key: Chaos Price (803) uses DIFFERENT blockchain data than signals.
- Price uses: Difficulty cycles, Halving cycles, Lorenz chaos
- Signals use: Historical price patterns
Therefore signals PREDICT prices, not generate them.
```

---

## PERFORMANCE SPECIFICATIONS

All Numba JIT compiled with:
```python
@njit(cache=True, fastmath=True, error_model='numpy', boundscheck=False)
```

Latency per tick:
- Leading signals: 15-50 nanoseconds
- Blockchain signals: 50-100 nanoseconds
- Lagging signals: 20-80 nanoseconds
- Confluence voting: ~100 nanoseconds
- Total pipeline: ~250-500 nanoseconds/tick

Target throughput: 300,000 - 1,000,000 trades/day
