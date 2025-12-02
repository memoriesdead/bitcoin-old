# FORMULA REGISTRY

## Overview

All trading formulas are assigned unique IDs and organized by category.
This document serves as the authoritative reference for all formula implementations.

## ID Ranges

| Range | Category | Description |
|-------|----------|-------------|
| 100-199 | Signals | Entry signals (Z-Score, etc.) |
| 200-299 | Filters | Signal filters (CUSUM) |
| 300-399 | Confluence | Signal combination & regime |
| 600-699 | Volume | Volume capture strategies |
| 700-799 | Flow | Order flow (PRIMARY EDGE) |
| 800-899 | Compounding | Renaissance growth formulas |

---

## SIGNALS (100-199)

### ID 141: Z-Score Mean Reversion
**Status:** LEGACY (ZERO EDGE - Use OFI Instead)

```
Formula: z = (price - mean) / std
Direction: z < -2.0 = BUY (oversold), z > 2.0 = SELL (overbought)
```

**Academic Finding:** Cont-Stoikov (2014) proved Z-score trades AGAINST order flow = ZERO EDGE.
Keep for confluence voting only; OFI is primary signal.

**File:** `engine/formulas/signals/f141_zscore.py`

---

## FILTERS (200-299)

### ID 218: CUSUM Filter
**Edge:** +8-12pp Win Rate

```
Formula: S⁺_t = max(0, S⁺_{t-1} + ΔP_t - h)  # Upside filter
         S⁻_t = max(0, S⁻_{t-1} - ΔP_t - h)  # Downside filter
Event:   S > threshold triggers signal
```

**Citation:** Lopez de Prado (2018) - Advances in Financial Machine Learning

**Purpose:** Eliminates false signals by requiring SUSTAINED price movement.

**File:** `engine/formulas/filters/f218_cusum.py`

---

## CONFLUENCE & REGIME (300-399)

### ID 333: Signal Confluence
**Edge:** Signal quality improvement

```
Theorem:  Condorcet's Jury Theorem
Formula:  Combined_Signal = sign(sum(individual_signals))
          Combined_Prob = Condorcet_Formula(individual_probs)
```

**Note:** With 3 independent 55% signals: P(majority correct) ≈ 59%

**File:** `engine/formulas/filters/f333_confluence.py`

### ID 335: Regime Filter
**Edge:** +3-5pp Win Rate

```
Formula: EMA_fast (20) vs EMA_slow (50)
Regimes:
  +2 = Strong Uptrend (BUY only)
  +1 = Weak Uptrend
   0 = Ranging (all signals)
  -1 = Weak Downtrend
  -2 = Strong Downtrend (SELL only)
```

**Citation:** Moskowitz, Ooi & Pedersen (2012) - Time Series Momentum

**File:** `engine/formulas/filters/f335_regime.py`

---

## ORDER FLOW (700-799) - PRIMARY EDGE

### ID 701: Order Flow Imbalance (OFI)
**Edge:** R² = 70% (PRIMARY SIGNAL)

```
Formula: OFI = Delta_Bid_Volume - Delta_Ask_Volume
Signal:  Trade WITH OFI direction (flow-following)
```

**Citation:** Cont, Kukanov & Stoikov (2014) - J. Financial Econometrics

**CRITICAL INSIGHT:** Trade WITH flow, NOT against it!
- Z-score mean reversion trades AGAINST flow = ZERO EDGE
- OFI flow-following trades WITH flow = POSITIVE EDGE

**File:** `engine/formulas/flow/f701_ofi.py`

### ID 702: Kyle Lambda
**Edge:** Position sizing (liquidity)

```
Formula: λ = Cov(ΔP, V) / Var(V)
Purpose: Price impact coefficient for position sizing
```

**Citation:** Kyle (1985) - Econometrica (10,000+ citations)

**File:** `engine/formulas/flow/f702_kyle.py`

### ID 706: Flow Momentum
**Edge:** +2-3pp Win Rate (signal confirmation)

```
Formula: Flow_Momentum = OFI(recent) - OFI(earlier)
Signal:  Positive = Flow accelerating (higher conviction)
```

**File:** `engine/formulas/flow/f706_momentum.py`

---

## RENAISSANCE COMPOUNDING (800-899)

### ID 801-810: Renaissance Growth Framework
**Target:** $100 → $10,000 in 46 days

```
Master Equation: Capital(t) = Capital(0) × (1 + f × edge)^n

Where:
  f = Kelly fraction (0.25 × full_kelly for safety)
  edge = net edge after costs
  n = number of trades
```

**Key Formulas:**
- 801: Master Growth Equation
- 802: Net Edge Calculator
- 803: Sharpe Threshold (2.0-3.0)
- 804: Win Rate Threshold (52-55%)
- 805: Quarter-Kelly Position Sizing
- 806: Trade Frequency Optimizer (100/day)
- 807: Time-to-Target Calculator
- 808: Drawdown-Constrained Growth
- 809: Compound Progress Tracker
- 810: Master Controller

**Citations:** Kelly (1956), Thorp (2007), Cont-Stoikov (2014)

---

## Directory Structure

```
engine/
├── core/                          # Layer 1: Foundation
│   ├── constants/
│   │   ├── blockchain.py          # Power Law, Genesis, Halving
│   │   ├── trading.py             # Formula parameters by ID
│   │   └── hft.py                 # HFT timescales
│   ├── dtypes/
│   │   ├── bucket.py              # BUCKET_DTYPE
│   │   ├── state.py               # STATE_DTYPE (formula states)
│   │   └── result.py              # RESULT_DTYPE
│   └── interfaces/
│       ├── formula.py             # IFormula base class
│       └── engine.py              # IEngine base class
│
├── formulas/                      # Layer 2: Formulas
│   ├── registry.py                # Auto-discovery registry
│   ├── signals/                   # ID 100-199
│   │   └── f141_zscore.py
│   ├── filters/                   # ID 200-399
│   │   ├── f218_cusum.py
│   │   ├── f333_confluence.py
│   │   └── f335_regime.py
│   ├── flow/                      # ID 700-799 (PRIMARY)
│   │   ├── f701_ofi.py
│   │   ├── f702_kyle.py
│   │   └── f706_momentum.py
│   └── compounding/               # ID 800-899
│       └── f8xx_renaissance.py
│
├── engines/                       # Layer 3: Engines
│   ├── base.py                    # BaseEngine
│   ├── hft.py                     # HFTEngine
│   └── renaissance.py             # RenaissanceEngine
│
├── tick/                          # Layer 3: Processing
│   └── processor.py               # process_tick_hft
│
└── runner.py                      # Layer 4: Entry point
```

---

## Usage

```python
from engine.formulas import FORMULA_REGISTRY, get_formula

# Get formula by ID
ofi = get_formula(701)
signal, confidence = ofi.compute(prices, tick)

# List all formulas
for fid, info in FORMULA_REGISTRY.items():
    print(f"{fid}: {info['name']} - {info['edge']}")

# Get formulas by category
flow_formulas = get_formulas_by_category('flow')
```

---

## Adding New Formulas

1. Create file in appropriate category folder: `f{ID}_{name}.py`
2. Implement `IFormula` interface
3. Use `@register_formula` decorator
4. Add entry to this document

```python
from engine.core.interfaces import IFormula
from engine.formulas.registry import register_formula

@register_formula
class NewFormula(IFormula):
    FORMULA_ID = 999
    FORMULA_NAME = "New Formula"
    EDGE_CONTRIBUTION = "+X pp Win Rate"
    CATEGORY = "flow"
    CITATION = "Author (Year) - Journal"

    def compute(self, prices, tick, **kwargs):
        # Implementation
        return signal, confidence
```
