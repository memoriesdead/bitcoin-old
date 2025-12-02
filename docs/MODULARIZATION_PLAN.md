# Engine Modularization Plan
## Codebase Cleanup & Organization

**Current State**: 76,175 lines across 136 files
**Target**: Well-organized, modular codebase with clear separation of concerns

---

## Phase 1: Extract Formulas from processor.py (1,550 LOC)

The `engine/tick/processor.py` file contains 15+ formula implementations. Each should be extracted to its own module while maintaining Numba JIT compatibility.

### Proposed Structure:

```
engine/tick/
├── __init__.py                    # Module exports
├── processor.py                   # Main orchestrator (reduced to ~400 LOC)
└── formulas/
    ├── __init__.py                # Formula exports
    ├── constants.py               # Blockchain constants (50 LOC)
    ├── zscore.py                  # ID 141: Z-Score (60 LOC)
    ├── ofi.py                     # ID 701: OFI + Kyle Lambda + Flow Momentum (100 LOC)
    ├── cusum.py                   # ID 218: CUSUM Filter (70 LOC)
    ├── regime.py                  # ID 335: Regime Filter (90 LOC)
    ├── confluence.py              # ID 333: Signal Confluence (150 LOC)
    ├── blockchain/
    │   ├── __init__.py
    │   ├── block_volatility.py    # ID 801: Block Time Volatility (50 LOC)
    │   ├── mempool_flow.py        # ID 802: Mempool Flow Simulator (80 LOC)
    │   ├── chaos_price.py         # ID 803: Lorenz Chaos Price (290 LOC)
    │   └── whale_detector.py      # ID 804: UTXO Whale Detector (60 LOC)
    └── leading/
        ├── __init__.py
        ├── power_law.py           # ID 901: Power Law Signal (60 LOC)
        ├── stock_to_flow.py       # ID 902: S2F Signal (90 LOC)
        └── halving_cycle.py       # ID 903: Halving Cycle (50 LOC)
```

### Numba Compatibility Notes:
- All `@njit` functions must be defined before being called
- Use lazy import pattern for cross-module JIT functions
- Keep constants as module-level variables for Numba caching

---

## Phase 2: Organize formulas/ Directory (51,788 LOC)

Current: 53 flat files in `formulas/`
Proposed: Category-based subdirectories

```
formulas/
├── __init__.py                    # Registry and exports
├── base.py                        # BaseFormula class
│
├── academic/                      # Research-based formulas
│   ├── academic_2024.py           # (1,825 LOC)
│   ├── timescale_invariance.py    # (1,666 LOC)
│   └── ...
│
├── hft/                           # High-frequency trading
│   ├── data_pipeline.py           # (3,014 LOC - SPLIT THIS)
│   ├── multiscale_advanced.py     # (2,065 LOC)
│   └── ...
│
├── blockchain/                    # Bitcoin-specific
│   ├── blockchain_signals.py      # (1,600 LOC)
│   ├── pure_math.py               # (1,483 LOC)
│   └── ...
│
├── statistical/                   # Statistical methods
│   ├── mean_reversion.py
│   ├── volatility.py
│   └── ...
│
└── ml/                            # Machine learning
    ├── advanced_ml.py
    └── ...
```

### Large File Splits:
1. `data_pipeline.py` (3,014 LOC) -> Split into:
   - `data_aggregation.py` (~750 LOC)
   - `feature_engineering.py` (~750 LOC)
   - `pipeline_utils.py` (~750 LOC)
   - `data_pipeline.py` (~750 LOC, main orchestrator)

2. `multiscale_advanced.py` (2,065 LOC) -> Consolidate with `multiscale_advanced_2.py` (1,550 LOC)

---

## Phase 3: Clean Up blockchain/ Directory (7,679 LOC)

Current: 22 files, some overlap
Proposed:

```
blockchain/
├── __init__.py                    # Main exports
│
├── feeds/                         # Data feeds
│   ├── unified_feed.py            # Main unified interface
│   ├── blockchain_feed.py         # (818 LOC)
│   └── market_data.py             # (536 LOC)
│
├── price_models/                  # Price derivation
│   ├── mathematical_price.py      # (638 LOC)
│   ├── true_price.py              # (528 LOC)
│   ├── first_principles.py        # (365 LOC)
│   └── power_law.py               # Power law model
│
├── signals/                       # Trading signals
│   ├── halving_signals.py         # (464 LOC) - Already clean
│   └── volume_flow.py             # Volume-based signals
│
└── simulation/                    # Pure simulation
    └── pure_simulation.py         # (627 LOC)
```

---

## Phase 4: Standardize Comments & Docstrings

### Docstring Standard:
```python
@njit(cache=True, fastmath=True)
def calc_ofi(prices: np.ndarray, tick: int, lookback: int) -> tuple:
    """
    ORDER FLOW IMBALANCE (Formula ID 701)

    Academic Citation: Cont, Kukanov & Stoikov (2014) - J. Financial Econometrics

    Calculates buy/sell pressure imbalance for price prediction.
    R² = 70% for short-term price movements (peer-reviewed).

    Mathematical Formula:
        OFI = (Buy_Pressure - Sell_Pressure) / Total_Pressure

    Args:
        prices: Circular price buffer (1M capacity)
        tick: Current tick index
        lookback: Number of ticks to analyze

    Returns:
        Tuple of (ofi_value, ofi_signal, ofi_strength, kyle_lambda, flow_momentum)
        - ofi_value: Raw OFI value [-1, 1]
        - ofi_signal: Direction signal (-1, 0, +1)
        - ofi_strength: Signal confidence [0, 1]
        - kyle_lambda: Price impact coefficient
        - flow_momentum: Rate of change in OFI

    Trade Direction:
        OFI > threshold -> BUY (trade WITH flow)
        OFI < -threshold -> SELL (trade WITH flow)

    References:
        - Kyle (1985) - Econometrica (kyle_lambda)
        - Academic microstructure literature
    """
```

### Comment Standard:
```python
# =============================================================================
# FORMULA ID 701: ORDER FLOW IMBALANCE
# Academic: Cont, Kukanov & Stoikov (2014) - R² = 70%
# =============================================================================

# STEP 1: Calculate buy/sell pressure from price changes
# Positive price change = buy pressure (aggressive buyers)
# Negative price change = sell pressure (aggressive sellers)
```

---

## Phase 5: Implementation Order

### Week 1: Core Extraction
1. Extract constants to `engine/tick/formulas/constants.py`
2. Extract each formula (ID 141, 701, 218, 335) to separate files
3. Update imports in processor.py
4. Test: Verify identical output

### Week 2: Blockchain Formulas
1. Extract ID 801-804 to `engine/tick/formulas/blockchain/`
2. Extract ID 901-903 to `engine/tick/formulas/leading/`
3. Extract confluence (ID 333)
4. Test: Full regression test

### Week 3: formulas/ Reorganization
1. Create subdirectory structure
2. Move files to categories
3. Update all imports throughout codebase
4. Test: Import verification

### Week 4: Comments & Documentation
1. Add standardized docstrings to all formulas
2. Add inline comments for complex logic
3. Create formula reference documentation
4. Final testing

---

## File Change Summary

| Before | After | Change |
|--------|-------|--------|
| `engine/tick/processor.py` (1,550 LOC) | ~400 LOC + 15 modules | -74% in main file |
| `formulas/` (53 flat files) | 5 subdirectories | Organized by category |
| `blockchain/` (22 files) | 4 subdirectories | Clear separation |

---

## Risk Mitigation

1. **Numba JIT Caching**: Pre-compile all modules on deployment
2. **Import Performance**: Use lazy imports where possible
3. **Testing**: Run 1M+ tick test after each phase
4. **Rollback**: Keep original files until fully validated

---

## Success Criteria

- [ ] All formulas in separate, documented files
- [ ] No single file > 500 LOC (except main orchestrators)
- [ ] Every function has docstring with:
  - Formula ID
  - Academic citation (if applicable)
  - Mathematical formula
  - Args/Returns documentation
- [ ] Directory structure reflects logical grouping
- [ ] All tests pass with identical output
- [ ] Engine performance unchanged (~200K+ TPS)
