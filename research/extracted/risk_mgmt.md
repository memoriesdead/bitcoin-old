# Risk Management Formula Extractions
## IDs: 223-238

---

## SOURCE 1: Advanced VaR Decomposition
**Citation:** Jorion (2006) "Value at Risk" McGraw-Hill
**Status:** EXTRACTED

### Formula 223: Marginal VaR
```
MVaR_i = α × Cov(R_i, R_p) / σ_p = α × β_i × σ_p
```
**Variables:**
- α = z-score for confidence level (e.g., 1.645 for 95%)
- Cov(R_i, R_p) = covariance of asset i with portfolio
- σ_p = portfolio standard deviation
- β_i = beta of asset i to portfolio
**Edge:** Sensitivity of VaR to position change

### Formula 224: Component VaR
```
CVaR_i = w_i × MVaR_i = w_i × α × β_i × σ_p
```
**Variables:**
- w_i = weight of asset i
**Property:** Σ CVaR_i = Portfolio VaR
**Edge:** Risk contribution of each position

### Formula 225: Incremental VaR
```
IVaR = VaR(P + a) - VaR(P)
```
**Variables:**
- P = current portfolio
- a = proposed new position
**Edge:** Impact of adding new position

### Formula 226: VaR Decomposition Identity
```
VaR = Σ_i CVaR_i = Σ_i w_i × MVaR_i
```
**Edge:** Portfolio VaR equals sum of component VaRs

---

## SOURCE 2: Extreme Value Theory (EVT)
**URL:** https://faculty.washington.edu/ezivot/econ589/EVT_Mcneil_Frey_2000.pdf
**Citation:** McNeil & Frey (2000) "Estimation of tail-related risk measures"
**Status:** EXTRACTED

### Formula 227: Generalized Pareto Distribution (GPD) VaR
```
VaR_p = u + (β/ξ) × [(n/(N×(1-p)))^(-ξ) - 1]
```
**Variables:**
- u = threshold
- β = scale parameter
- ξ = shape parameter (tail index)
- n = number of exceedances
- N = total sample size
- p = confidence level
**Edge:** Heavy-tailed VaR estimation

### Formula 228: EVT Expected Shortfall
```
ES_p = VaR_p/(1-ξ) + (β - ξ×u)/(1-ξ),  for ξ < 1
```
**Edge:** Tail risk beyond VaR using EVT

### Formula 229: Hill Estimator (Tail Index)
```
ξ_Hill = (1/k) × Σ_{i=1}^k ln(X_{(n-i+1)} / X_{(n-k)})
```
**Variables:**
- X_{(i)} = order statistics
- k = number of upper order statistics used
**Edge:** Estimates tail heaviness

### Formula 230: Peaks Over Threshold (POT)
```
F_u(y) = P(X - u ≤ y | X > u) ≈ GPD(β, ξ)
```
**Edge:** Models exceedances over threshold

---

## SOURCE 3: Risk Parity / Equal Risk Contribution
**URL:** https://people.umass.edu/~kazemi/An%20Introduction%20to%20Risk%20Parity.pdf
**Citation:** Maillard et al. (2010) "On the Properties of Equally-Weighted Risk Contribution Portfolios"
**Status:** EXTRACTED

### Formula 231: Risk Contribution
```
RC_i = w_i × (Σw)_i / √(w'Σw) = w_i × ∂σ_p/∂w_i
```
**Variables:**
- Σ = covariance matrix
- (Σw)_i = i-th element of Σw
**Property:** Σ RC_i = σ_p
**Edge:** Each asset's contribution to portfolio volatility

### Formula 232: Equal Risk Contribution Condition
```
RC_i = RC_j = σ_p / N,  ∀ i,j
```
**Edge:** All assets contribute equally to risk

### Formula 233: Risk Parity Optimization
```
min_w Σ_i [RC_i - σ_p/N]²  s.t. Σw_i = 1, w_i ≥ 0
```
**Edge:** Finds ERC portfolio weights

### Formula 234: Naive Risk Parity
```
w_i = (1/σ_i) / Σ_j(1/σ_j)
```
**Edge:** Inverse volatility weighting (ignores correlation)

---

## SOURCE 4: Drawdown Risk Measures
**Status:** EXTRACTED

### Formula 235: Conditional Drawdown at Risk (CDaR)
```
CDaR_α = E[DD | DD > DD_α]
```
**Variables:**
- DD = drawdown
- DD_α = α-quantile of drawdown distribution
**Edge:** Average of worst drawdowns

### Formula 236: Ulcer Index
```
UI = √[(1/n) × Σ_i DD_i²]
```
**Variables:**
- DD_i = percentage drawdown at time i
**Edge:** RMS of drawdowns (penalizes depth and duration)

### Formula 237: Pain Index
```
Pain = (1/n) × Σ_i |DD_i|
```
**Edge:** Average absolute drawdown

### Formula 238: Recovery Factor
```
RF = Total_Return / Max_Drawdown
```
**Edge:** Return earned per unit of max drawdown risk

---

## EXTRACTION STATUS

| ID | Name | Source | Status |
|----|------|--------|--------|
| 223 | MarginalVaR | Jorion (2006) | ✅ EXTRACTED |
| 224 | ComponentVaR | Jorion (2006) | ✅ EXTRACTED |
| 225 | IncrementalVaR | Jorion (2006) | ✅ EXTRACTED |
| 226 | VaRDecomposition | Jorion (2006) | ✅ EXTRACTED |
| 227 | GPD_VaR | McNeil & Frey (2000) | ✅ EXTRACTED |
| 228 | EVT_ES | McNeil & Frey (2000) | ✅ EXTRACTED |
| 229 | HillEstimator | EVT | ✅ EXTRACTED |
| 230 | PeaksOverThreshold | EVT | ✅ EXTRACTED |
| 231 | RiskContribution | Maillard et al. (2010) | ✅ EXTRACTED |
| 232 | ERCCondition | Maillard et al. (2010) | ✅ EXTRACTED |
| 233 | RiskParityOptim | Maillard et al. (2010) | ✅ EXTRACTED |
| 234 | NaiveRiskParity | - | ✅ EXTRACTED |
| 235 | CDaR | - | ✅ EXTRACTED |
| 236 | UlcerIndex | - | ✅ EXTRACTED |
| 237 | PainIndex | - | ✅ EXTRACTED |
| 238 | RecoveryFactor | - | ✅ EXTRACTED |

---

## SUMMARY: IDs 223-238 COMPLETE

**Total Extracted in risk_mgmt.md: 16 formulas**

Categories Covered:
- VaR Decomposition (223-226)
- Extreme Value Theory (227-230)
- Risk Parity (231-234)
- Drawdown Measures (235-238)
