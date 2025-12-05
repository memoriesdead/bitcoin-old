# USPTO Patent Formula Extractions
## IDs: 591-640

---

## SOURCE 1: US8140416B2 - Algorithmic Trading (Hidden Orders)
**URL:** https://patents.google.com/patent/US8140416B2/en
**Assignee:** Not Renaissance/Two Sigma (general trading patent)
**Status:** EXTRACTED

### Formula 591: Virtual Price Error
```
VPE = v_i - p_i
```
**Variables:**
- v_i = limit order book's perceived execution price
- p_i = actual execution price
**Edge:** Detects hidden liquidity not visible in LOB

### Formula 592: Price Impact of Market Order
```
PI_i = δ_i × (p_i_final - m_i)
```
**Variables:**
- δ_i = 1 for buy orders, -1 for sell orders
- p_i_final = last execution price
- m_i = mid-quote immediately prior
**Edge:** Measures true market impact

### Formula 593: Cost of Market Order
```
C_i = δ_i × (p̄_i - m_i)
```
**Variables:**
- p̄_i = share-weighted average execution price
- m_i = mid-quote prior to order
**Edge:** Quantifies execution cost

### Formula 594: Variable Standardization (Cross-Sectional)
```
X_standard = (x - x̄) / σ(x)
```
**Variables:**
- x̄ = mean of variable x
- σ(x) = standard deviation
**Edge:** Removes time-of-day effects, enables cross-asset analysis

### Formula 595: Hidden Order Book Regions
```
Region_k = (bid + (k-1)*0.2*(ask-bid), bid + k*0.2*(ask-bid)]
```
**Variables:**
- k = region number (1-6)
- bid, ask = best bid/ask prices
**Edge:** Identifies hidden order placement zones

---

## SOURCE 2: US8719146B2 - Micro Auction (HFT)
**URL:** https://patents.google.com/patent/US8719146B2/en
**Status:** EXTRACTED

### Formula 596: Adaptive Micro Auction Timing
```
Execute when: t_first_order + X_ms < Y_ms_randomized
```
**Variables:**
- X = 5-10 ms (call period)
- Y = 40-50 ms (outer boundary, randomized)
**Edge:** Reduces latency arbitrage exploitation

### Formula 597: Interval Length (Pseudorandom)
```
Interval_n = Item(n) from pseudorandom sequence
```
**Edge:** Unpredictable auction timing defeats HFT

### Formula 598: Interval Length (Random Distribution)
```
Interval = Fixed_component + Random(y, x)
```
**Variables:**
- y = negative bound
- x = positive bound
**Edge:** Randomization prevents pattern detection

---

## SOURCE 3: Kyle's Lambda (Academic - Kyle 1985)
**URL:** https://frds.io/measures/kyle_lambda/
**Citation:** Kyle, Albert (1985) "Continuous Auctions and Insider Trading" Econometrica 53:1315-35
**Status:** EXTRACTED

### Formula 599: Kyle Lambda Regression
```
r_{i,n} = λ_i × S_{i,n} + ε_{i,t}
```
**Variables:**
- r_{i,n} = percentage return for stock i in period n
- λ_i = Kyle's Lambda (price impact coefficient)
- S_{i,n} = signed square-root dollar volume
- ε_{i,t} = error term

### Formula 600: Signed Square-Root Dollar Volume
```
S_{i,n} = Σ_k sign(v_{k,n}) × √|v_{k,n}|
```
**Variables:**
- v_{k,n} = signed dollar volume of k-th trade
**Edge:** Measures liquidity cost, price impact per trade

### Formula 601: Kyle Equilibrium Price
```
P(y) = μ + λy
```
**Variables:**
- y = total order flow (informed + noise)
- μ = fundamental value
- λ = price impact coefficient
**Edge:** Core market microstructure model

---

## ADDITIONAL PATENTS IDENTIFIED

| Patent | Title | Key Formulas |
|--------|-------|--------------|
| US20130325684A1 | Latency Reduction HFT | Optical signal splitting |
| US10296973B2 | FIX Protocol Load Balancing | Weighting factors |
| US20160035027A1 | Synchronized Multi-Exchange | Timing coordination |
| US8571967B1 | Algorithmic Trading Strategies | Order structuring |

---

## PYTHON IMPLEMENTATIONS

```python
# ID: 591
@FormulaRegistry.register(591, "VirtualPriceError", "microstructure")
class VirtualPriceError(BaseFormula):
    """
    Source: US8140416B2 Patent
    URL: https://patents.google.com/patent/US8140416B2/en

    Formula: VPE = v_i - p_i

    Edge: Detect hidden liquidity beyond visible LOB
    """

    def _compute(self):
        # perceived_price - actual_price
        pass

# ID: 592
@FormulaRegistry.register(592, "MarketOrderPriceImpact", "microstructure")
class MarketOrderPriceImpact(BaseFormula):
    """
    Source: US8140416B2 Patent
    URL: https://patents.google.com/patent/US8140416B2/en

    Formula: PI_i = δ_i × (p_final - m_i)

    Edge: Measure true market impact for execution optimization
    """

    def _compute(self):
        # direction * (final_price - mid_quote)
        pass

# ID: 599
@FormulaRegistry.register(599, "KyleLambdaRegression", "microstructure")
class KyleLambdaRegression(BaseFormula):
    """
    Source: Kyle (1985) "Continuous Auctions and Insider Trading" Econometrica
    URL: https://frds.io/measures/kyle_lambda/

    Formula: r_{i,n} = λ_i × S_{i,n} + ε

    Edge: Estimate liquidity cost, price impact coefficient
    """

    def _compute(self):
        # OLS regression: returns ~ signed_sqrt_volume
        pass
```

---

## EXTRACTION STATUS

| ID | Name | Source | Status |
|----|------|--------|--------|
| 591 | VirtualPriceError | US8140416B2 | ✅ EXTRACTED |
| 592 | MarketOrderPriceImpact | US8140416B2 | ✅ EXTRACTED |
| 593 | MarketOrderCost | US8140416B2 | ✅ EXTRACTED |
| 594 | VariableStandardization | US8140416B2 | ✅ EXTRACTED |
| 595 | HiddenOrderRegions | US8140416B2 | ✅ EXTRACTED |
| 596 | MicroAuctionTiming | US8719146B2 | ✅ EXTRACTED |
| 597 | PseudorandomInterval | US8719146B2 | ✅ EXTRACTED |
| 598 | RandomDistInterval | US8719146B2 | ✅ EXTRACTED |
| 599 | KyleLambdaRegression | Kyle (1985) | ✅ EXTRACTED |
| 600 | SignedSqrtVolume | Kyle (1985) | ✅ EXTRACTED |
| 601 | KyleEquilibriumPrice | Kyle (1985) | ✅ EXTRACTED |
| 602-640 | - | - | PENDING |
