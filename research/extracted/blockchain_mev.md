# Blockchain MEV Formula Extractions
## IDs: 482-519

---

## SOURCE 1: Flashbots REV (Realized Extractable Value)
**URL:** https://writings.flashbots.net/quantifying-rev
**Status:** EXTRACTED

### Formula 482: Total REV Decomposition
```
REV = REV_S + REV_M
```
Total realized value splits between searcher and miner components.

### Formula 483: Searcher Revenue (REV_S)
```
REV_S = V_in - V_out - g_MEV × s_MEV
```
**Variables:**
- V_in = value flowing from blockchain to searcher
- V_out = value flowing from searcher to blockchain (excluding gas)
- g_MEV = gas price of extraction transactions (in ETH)
- s_MEV = total gas consumed (in gas units)

### Formula 484: Miner Revenue (REV_M)
```
REV_M = s_MEV × (g_MEV - g_eff)
```
**Variables:**
- g_eff = effective gas price of transactions that would have been included absent the opportunity

### Formula 485: REV_M Approximation
```
REV_M ≳ s_MEV × (g_MEV - g_tail)
```
**Variables:**
- g_tail = gas price of the final transaction in the block

### Formula 486: Total REV with Extraction Costs
```
REV ≳ V_in - V_out - s_MEV × g_tail
```
Extraction costs quantified as s_MEV × g_tail.

### Formula 487: Extractable Value Cost (EVC)
```
EVC = Σ(s_tx × g_tx), where X = {preflights, failures, cancellations}
```
Network costs from MEV activity not directly extracted.

### Formula 488: Flashbots Bundle Model
```
REV_M^Flashbots ≳ s_MEV × (g_MEV - g_tail) + V_direct
```
**Variables:**
- V_direct = direct coinbase transfer value in the bundle

---

## SOURCE 2: arXiv:2405.17944 - MEV Sandwich Attacks
**URL:** https://arxiv.org/html/2405.17944v1
**Status:** EXTRACTED

### Formula 489: Expected Loss (MEV Searcher Risk)
```
EL = gp × fg × (1 - sr)
```
**Variables:**
- gp = gas price
- fg = average gas of failed front-running arbitrages
- sr = success rate of mempool transactions

### Formula 490: Profit Ratio Calculation
```
ratio = (tkn_k.out - tkn_k.in) / tkn_k.in
```
**Variables:**
- tkn_k.out = output amount of token k
- tkn_k.in = input amount of token k
- Threshold: ratio < ε filters for minimal losses

---

## SOURCE 3: arXiv:2410.13624 - Optimal MEV Extraction
**URL:** https://arxiv.org/html/2410.13624v1
**Status:** NEEDS EXTRACTION

### Formulas to Extract:
- Absolute commitment attack formula
- AMM extraction optimization
- Sandwich attack profit leakage to AMM

---

## ADDITIONAL PAPERS IDENTIFIED

| arXiv ID | Title | Status |
|----------|-------|--------|
| 2411.03327 | MEV Taxonomy, Detection, Mitigation | PENDING |
| 2508.04003 | Marginal Effects of MEV Re-Ordering | PENDING |
| 2305.16468 | Time to Bribe: Block Construction | PENDING |

---

## PYTHON IMPLEMENTATIONS

```python
# ID: 482
@FormulaRegistry.register(482, "TotalREV", "mev")
class TotalREV(BaseFormula):
    """
    Source: Flashbots Research "Quantifying REV"
    URL: https://writings.flashbots.net/quantifying-rev

    Formula: REV = REV_S + REV_M

    Edge: Quantify total extractable value for MEV opportunity assessment
    """

    def _compute(self):
        # REV = searcher_revenue + miner_revenue
        pass

# ID: 483
@FormulaRegistry.register(483, "SearcherREV", "mev")
class SearcherREV(BaseFormula):
    """
    Source: Flashbots Research "Quantifying REV"
    URL: https://writings.flashbots.net/quantifying-rev

    Formula: REV_S = V_in - V_out - g_MEV × s_MEV

    Edge: Calculate searcher profit from MEV extraction
    """

    def _compute(self):
        # V_in - V_out - gas_price * gas_used
        pass

# ID: 484
@FormulaRegistry.register(484, "MinerREV", "mev")
class MinerREV(BaseFormula):
    """
    Source: Flashbots Research "Quantifying REV"
    URL: https://writings.flashbots.net/quantifying-rev

    Formula: REV_M = s_MEV × (g_MEV - g_eff)

    Edge: Calculate validator/miner profit from MEV
    """

    def _compute(self):
        # gas_used * (gas_price - effective_gas_price)
        pass

# ID: 489
@FormulaRegistry.register(489, "MEVExpectedLoss", "mev")
class MEVExpectedLoss(BaseFormula):
    """
    Source: arXiv:2405.17944 "Remeasuring MEV Attacks"
    URL: https://arxiv.org/html/2405.17944v1

    Formula: EL = gp × fg × (1 - sr)

    Edge: Risk assessment for MEV extraction attempts
    """

    def _compute(self):
        # gas_price * failed_gas * (1 - success_rate)
        pass
```

---

## EXTRACTION STATUS

| ID | Name | Source | Status |
|----|------|--------|--------|
| 482 | TotalREV | Flashbots | ✅ EXTRACTED |
| 483 | SearcherREV | Flashbots | ✅ EXTRACTED |
| 484 | MinerREV | Flashbots | ✅ EXTRACTED |
| 485 | REVApproximation | Flashbots | ✅ EXTRACTED |
| 486 | REVWithCosts | Flashbots | ✅ EXTRACTED |
| 487 | ExtractableValueCost | Flashbots | ✅ EXTRACTED |
| 488 | FlashbotsBundleREV | Flashbots | ✅ EXTRACTED |
| 489 | MEVExpectedLoss | arXiv | ✅ EXTRACTED |
| 490 | ProfitRatio | arXiv | ✅ EXTRACTED |
| 491-519 | - | - | PENDING |
