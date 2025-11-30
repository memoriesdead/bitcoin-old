# Renaissance Trading System - Architecture
============================================

## Codebase Statistics
- **Total Files**: 62 Python files
- **Total Lines**: 41,849 lines of code
- **Formulas**: 411 academic trading formulas
- **TRUE PRICE**: Pure blockchain mathematical derivation (NO exchange APIs)

## TRUE BITCOIN PRICE FORMULA

See `FORMULAS.md` for complete mathematical derivation.

```
TRUE_PRICE = PRODUCTION_COST x (1 + SCARCITY + MATURITY x SUPPLY)

Where (ALL blockchain-derived, NO arbitrary constants):
    PRODUCTION_COST = Energy_kWh x $/kWh
    SCARCITY = ln(S2F) / (halvings + 1)^2
    MATURITY = ln(days) / (ln(days) + halvings^2)
    SUPPLY = 1 / (1 + ln(MAX_SUPPLY / current_supply))

Current TRUE PRICE: ~$96,972 (at $0.044/kWh implied market rate)
```

## Directory Structure

```
livetrading/
├── README.md                    # Project overview
├── ARCHITECTURE.md              # This file
├── run.py                       # Main entry point
├── __init__.py                  # Package initialization
│
├── blockchain/                  # PURE BLOCKCHAIN PRICE (NO APIS)
│   ├── __init__.py
│   ├── mathematical_price.py    # TRUE PRICE derivation (MAIN)
│   ├── first_principles_price.py # Landauer principle derivation
│   ├── pure_blockchain_value.py # Energy-unit valuation
│   ├── live_true_price.py       # Real-time TRUE PRICE streaming
│   ├── price_sensitivity.py     # Energy cost sensitivity analysis
│   └── price_variance_analysis.py # Exchange variance breakdown
│
├── blockchain_market_data_usd.py # USD-denominated blockchain signals
│
├── data/                        # Historical data & pipelines
│   ├── __init__.py
│   ├── btc_history.py           # Historical BTC data loader
│   ├── btc_history_ultra.py     # Optimized binary format loader
│   ├── pipeline.py              # Data processing pipeline
│   ├── pipeline_fast.py         # High-speed pipeline
│   ├── volume_aggregator.py     # Volume bar aggregation
│   ├── btc_history.csv          # Raw historical data (6.7MB)
│   ├── btc_history.npy          # NumPy format (3.2MB)
│   └── btc_ultra.bin            # Binary optimized (3.2MB)
│
├── engine/                      # Trading engines
│   ├── __init__.py
│   ├── live_engine_v1.py        # Main production engine (2,098 lines)
│   ├── blockchain_live_engine.py # Blockchain-only engine
│   ├── adaptive_thresholds.py   # Dynamic threshold adjustment
│   └── hft/                     # High-frequency trading
│       ├── __init__.py
│       ├── engine.py            # HFT engine core
│       ├── signals.py           # Signal generation
│       ├── models.py            # Data models
│       ├── tables.py            # Lookup tables
│       └── constants.py         # HFT constants
│
├── formulas/                    # 411 Academic Trading Formulas
│   ├── __init__.py              # Formula registry & imports
│   ├── base.py                  # BaseFormula class & registry
│   │
│   │── # STATISTICAL (IDs 1-30)
│   ├── statistical.py           # Bayesian, MLE, entropy
│   │
│   │── # TIME SERIES (IDs 31-60)
│   ├── timeseries.py            # ARIMA, GARCH
│   │
│   │── # MACHINE LEARNING (IDs 61-100)
│   ├── machine_learning.py      # Ensemble, neural
│   │
│   │── # MICROSTRUCTURE (IDs 101-130)
│   ├── microstructure.py        # Kyle, VPIN, OFI
│   │
│   │── # MEAN REVERSION (IDs 131-150)
│   ├── mean_reversion.py        # OU, Z-score
│   │
│   │── # VOLATILITY (IDs 151-170)
│   ├── volatility.py            # GARCH, rough vol
│   │
│   │── # REGIME DETECTION (IDs 171-190)
│   ├── regime.py                # HMM, CUSUM
│   ├── regime_filter.py         # Trend-aware filtering
│   │
│   │── # SIGNAL PROCESSING (IDs 191-210)
│   ├── signal_processing.py     # Kalman, wavelet
│   │
│   │── # RISK MANAGEMENT (IDs 211-222)
│   ├── risk.py                  # Kelly, VaR
│   │
│   │── # ADVANCED HFT (IDs 239-258)
│   ├── advanced_hft.py          # MicroPrice, tick bars
│   ├── hft_volume.py            # Volume-based HFT
│   │
│   │── # BITCOIN SPECIFIC (IDs 259-282)
│   ├── bitcoin_specific.py      # OBI, cross-exchange
│   ├── bitcoin_derivatives.py   # Funding rate
│   ├── bitcoin_timing.py        # Session filters
│   │
│   │── # MARKET MAKING (IDs 283-284)
│   ├── market_making.py         # Avellaneda-Stoikov
│   │
│   │── # EXECUTION (IDs 285-290)
│   ├── execution.py             # Almgren-Chriss
│   │
│   │── # VOLUME SCALING (IDs 295-299)
│   ├── volume_scaling.py
│   │
│   │── # ACADEMIC RESEARCH (IDs 300-310)
│   ├── academic_research.py
│   │
│   │── # TRANSACTION COSTS (IDs 311-319)
│   ├── transaction_costs.py
│   │
│   │── # EXIT STRATEGIES (IDs 320-322)
│   ├── exit_strategies.py       # Leung, Trailing Stop
│   │
│   │── # COMPOUNDING (IDs 323-330)
│   ├── compounding_strategies.py # Kelly, Optimal F
│   │
│   │── # PROFITABILITY FIXES (IDs 331-340)
│   ├── edge_measurement.py      # Real edge from outcomes
│   ├── optimal_frequency.py     # High freq + quality
│   ├── signal_confluence.py     # Condorcet voting
│   ├── drawdown_control.py      # Position sizing
│   │
│   │── # ADVANCED MICROSTRUCTURE (IDs 341-346)
│   ├── advanced_microstructure.py # Quant-level research
│   │
│   │── # TIME-SCALE INVARIANCE (IDs 347-360)
│   ├── timescale_invariance.py  # Multi-timeframe adaptation
│   │
│   │── # MULTI-SCALE ADVANCED (IDs 361-400)
│   ├── multiscale_advanced.py   # Structural breaks, DCCA
│   ├── multiscale_advanced_2.py # Price impact, ensemble
│   │
│   │── # VOLUME FREQUENCY (IDs 401-402)
│   ├── volume_frequency.py      # Dynamic from 24h volume
│   │
│   │── # BIDIRECTIONAL (IDs 403-411)
│   ├── bidirectional.py         # SHORT signals enabled
│   │
│   │── # SUPPORTING MODULES
│   ├── adaptive_online.py       # Online learning
│   ├── advanced_prediction.py   # Prediction models
│   ├── gap_analysis.py          # Gap detection
│   └── renaissance_strategies.py # Core strategies
│
├── utils/                       # Utility functions
│   └── (utility modules)
│
└── logs/                        # Trading logs (gitignored)
    ├── trading.log
    ├── trading_usd.log
    └── ...
```

## Formula ID Ranges

| Range     | Category              | Description                          |
|-----------|----------------------|--------------------------------------|
| 1-30      | Statistical          | Bayesian, MLE, entropy               |
| 31-60     | Time Series          | ARIMA, GARCH                         |
| 61-100    | Machine Learning     | Ensemble, neural                     |
| 101-130   | Microstructure       | Kyle, VPIN, OFI                      |
| 131-150   | Mean Reversion       | OU, Z-score                          |
| 151-170   | Volatility           | GARCH, rough vol                     |
| 171-190   | Regime Detection     | HMM, CUSUM                           |
| 191-210   | Signal Processing    | Kalman, wavelet                      |
| 211-222   | Risk Management      | Kelly, VaR                           |
| 239-258   | Advanced HFT         | MicroPrice, tick bars                |
| 259-268   | Bitcoin Specific     | OBI, cross-exchange                  |
| 269-276   | Bitcoin Derivatives  | Funding rate                         |
| 277-282   | Bitcoin Timing       | Session filters                      |
| 283-284   | Market Making        | Avellaneda-Stoikov                   |
| 285-290   | Execution            | Almgren-Chriss                       |
| 295-299   | Volume Scaling       | Volume-based sizing                  |
| 300-310   | Academic Research    | Research implementations             |
| 311-319   | Transaction Costs    | Spread, impact, fees                 |
| 320-322   | Exit Strategies      | Leung, Trailing Stop                 |
| 323-330   | Compounding          | Kelly, Optimal F                     |
| 331-340   | Profitability Fixes  | Edge, frequency, confluence          |
| 341-346   | Adv Microstructure   | Quant-level research                 |
| 347-360   | Time-Scale           | Multi-timeframe adaptation           |
| 361-380   | Multi-Scale Adv 1    | Structural breaks, DCCA              |
| 381-400   | Multi-Scale Adv 2    | Price impact, ensemble               |
| 401-402   | Volume Frequency     | Dynamic from volume                  |
| 403-411   | Bidirectional        | SHORT signals enabled                |

## Key Components

### 1. Blockchain Data Layer (PURE MATH - NO EXCHANGE APIs)
- Pure blockchain data from Bitcoin nodes
- Mathematical price derivation from:
  - Block height, difficulty, hash rate (blockchain)
  - Energy cost (only external input)
  - Protocol constants (halvings, supply schedule)
- TRUE PRICE = Production_Cost x Blockchain_Multiplier
- See `FORMULAS.md` for complete mathematical derivation
- Implementation: `blockchain/mathematical_price.py`

### 2. Trading Engine
- Paper/Live trading modes
- Kelly criterion position sizing
- Exit management (TP/SL/Trailing)
- Compounding strategies

### 3. Formula System
- 411 registered formulas
- Automatic registration via decorators
- Live data calibration (no hardcoding)
- Bidirectional (LONG/SHORT) signals

## Running the System

```bash
# Paper trading test (60 seconds)
python -c "from engine.live_engine_v1 import run_live_test; run_live_test(60)"

# Import and verify formulas
python -c "from formulas import FORMULA_REGISTRY; print(len(FORMULA_REGISTRY))"
```

## Data Flow

```
Bitcoin Blockchain (node data)
        ↓
        ├── block_height
        ├── difficulty
        └── hash_rate (derived: difficulty x 2^32 / 600)
        ↓
MathematicalPricer (pure blockchain math)
        ↓
        ├── Production Cost = Energy_kWh x $/kWh
        ├── Scarcity = ln(S2F) / (halvings+1)^2
        ├── Maturity = ln(days) / (ln(days) + halvings^2)
        └── Supply = 1 / (1 + ln(MAX/current))
        ↓
TRUE_PRICE = Production_Cost x (1 + Scarcity + Maturity x Supply)
        ↓
LiveEngineV1 (signal generation)
        ↓
Formula Evaluation (411 formulas)
        ↓
Signal Aggregation (Condorcet voting)
        ↓
Position Management (Kelly sizing)
        ↓
Trade Execution (blockchain direct)
```

**NO EXCHANGE APIs IN DATA FLOW. Pure blockchain mathematics.**
