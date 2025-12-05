# Top Finance Journal Formula Extractions
## IDs: 741-760

---

## SOURCE 1: Capital Asset Pricing Model (CAPM)
**Citation:** Sharpe (1964) "Capital Asset Prices" Journal of Finance
**Citation:** Lintner (1965) "Security Prices, Risk, and Maximal Gains from Diversification"
**Status:** EXTRACTED

### Formula 741: CAPM Expected Return
```
E(R_i) = R_f + β_i × (E(R_m) - R_f)
```
**Variables:**
- R_f = risk-free rate
- β_i = systematic risk (beta)
- E(R_m) = expected market return
- E(R_m) - R_f = market risk premium
**Edge:** Foundation of asset pricing

### Formula 742: Beta (CAPM)
```
β_i = Cov(R_i, R_m) / Var(R_m)
```
**Edge:** Measures systematic (non-diversifiable) risk

### Formula 743: Security Market Line
```
E(R_i) = R_f + β_i × MRP
```
**Variables:**
- MRP = Market Risk Premium
**Edge:** Linear relationship between beta and expected return

### Formula 744: Jensen's Alpha
```
α = R_p - [R_f + β_p × (R_m - R_f)]
```
**Edge:** Risk-adjusted excess return

---

## SOURCE 2: Arbitrage Pricing Theory (APT)
**Citation:** Ross (1976) "The Arbitrage Theory of Capital Asset Pricing" J. Economic Theory
**Status:** EXTRACTED

### Formula 745: APT Multi-Factor Model
```
E(R_i) = λ_0 + Σ_{k=1}^K β_ik × λ_k
```
**Variables:**
- λ_0 = risk-free rate
- β_ik = sensitivity to factor k
- λ_k = risk premium for factor k
**Edge:** Multiple systematic risk factors

### Formula 746: APT Factor Model
```
R_i = a_i + Σ_k β_ik × F_k + ε_i
```
**Variables:**
- F_k = return on factor k
- ε_i = idiosyncratic risk
**Edge:** Decomposes returns into factor exposures

---

## SOURCE 3: Markowitz Mean-Variance Optimization
**Citation:** Markowitz (1952) "Portfolio Selection" Journal of Finance
**Status:** EXTRACTED

### Formula 747: Portfolio Expected Return
```
E(R_p) = Σ_i w_i × E(R_i) = w'μ
```
**Variables:**
- w = weight vector
- μ = expected return vector
**Edge:** Linear combination of asset returns

### Formula 748: Portfolio Variance
```
σ²_p = Σ_i Σ_j w_i × w_j × σ_ij = w'Σw
```
**Variables:**
- Σ = covariance matrix
**Edge:** Quadratic in weights (diversification effect)

### Formula 749: Minimum Variance Portfolio
```
w_mv = Σ^(-1) × 1 / (1'Σ^(-1)1)
```
**Variables:**
- 1 = vector of ones
- Σ^(-1) = inverse covariance matrix
**Edge:** Lowest risk portfolio on efficient frontier

### Formula 750: Mean-Variance Optimization (Lagrangian)
```
min_w ½w'Σw - λ(w'μ - μ_target)  s.t. w'1 = 1
```
**Edge:** Finds efficient frontier portfolios

### Formula 751: Efficient Frontier (Two-Fund Theorem)
```
w_eff = a × w_mv + (1-a) × w_tan
```
**Variables:**
- w_tan = tangency portfolio weights
- a = mixing parameter
**Edge:** All efficient portfolios are combinations of two portfolios

---

## SOURCE 4: Carhart Four-Factor Model
**Citation:** Carhart (1997) "On Persistence in Mutual Fund Performance" J. Finance
**Status:** EXTRACTED

### Formula 752: Carhart Four-Factor Model
```
R_i - R_f = α + β_MKT(R_m-R_f) + β_SMB×SMB + β_HML×HML + β_MOM×MOM + ε
```
**Variables:**
- MOM (or UMD) = momentum factor (winners minus losers)
**Edge:** Adds momentum to Fama-French 3-factor

### Formula 753: Momentum Factor (UMD)
```
MOM = ½(Small_Winners + Big_Winners) - ½(Small_Losers + Big_Losers)
```
**Variables:**
- Winners/Losers based on past 12-month returns (skip 1 month)
**Edge:** Captures momentum anomaly

### Formula 754: Momentum Signal
```
Momentum_i = R_i(t-12, t-1) = Π_{k=2}^{12}(1 + R_{i,t-k}) - 1
```
**Edge:** Prior 12-month return (skip most recent month)

---

## SOURCE 5: Market Efficiency Tests
**Citation:** Fama (1970) "Efficient Capital Markets: A Review" Journal of Finance
**Status:** EXTRACTED

### Formula 755: Autocorrelation Test
```
ρ_k = Cov(R_t, R_{t-k}) / Var(R_t)
```
**Variables:**
- k = lag
**Edge:** Tests return predictability (EMH implies ρ ≈ 0)

### Formula 756: Variance Ratio
```
VR(q) = Var(R_t(q)) / (q × Var(R_t))
```
**Variables:**
- R_t(q) = q-period return
**Edge:** Tests random walk (VR=1 under RW)

### Formula 757: Lo-MacKinlay Variance Ratio Statistic
```
z(q) = √(nq) × (VR(q) - 1) / √(2(2q-1)(q-1)/(3q))
```
**Edge:** Tests for mean reversion/momentum

---

## SOURCE 6: Liquidity Premium Models
**Citation:** Pastor & Stambaugh (2003) "Liquidity Risk and Expected Stock Returns" J. Political Economy
**Status:** EXTRACTED

### Formula 758: Liquidity Beta
```
β_LIQ = Cov(R_i, L_mkt) / Var(L_mkt)
```
**Variables:**
- L_mkt = market-wide liquidity factor
**Edge:** Sensitivity to liquidity shocks

### Formula 759: Pastor-Stambaugh Liquidity Measure
```
γ_i,t = coefficient from: R_{i,t+1}^e = θ + φ×R_i,t + γ×sign(R_i,t^e)×Vol_i,t + ε
```
**Variables:**
- R^e = excess return
- Vol = dollar volume
**Edge:** Price reversal induced by volume (illiquidity proxy)

### Formula 760: Liquidity-Adjusted CAPM
```
E(R_i) = R_f + β_MKT×MRP + β_LIQ×LRP
```
**Variables:**
- LRP = Liquidity Risk Premium
**Edge:** Compensation for liquidity risk exposure

---

## EXTRACTION STATUS

| ID | Name | Source | Status |
|----|------|--------|--------|
| 741 | CAPMReturn | Sharpe (1964) | ✅ EXTRACTED |
| 742 | BetaCAPM | Sharpe (1964) | ✅ EXTRACTED |
| 743 | SecurityMarketLine | Sharpe (1964) | ✅ EXTRACTED |
| 744 | JensensAlpha | Jensen (1968) | ✅ EXTRACTED |
| 745 | APTMultiFactor | Ross (1976) | ✅ EXTRACTED |
| 746 | APTFactorModel | Ross (1976) | ✅ EXTRACTED |
| 747 | PortfolioReturn | Markowitz (1952) | ✅ EXTRACTED |
| 748 | PortfolioVariance | Markowitz (1952) | ✅ EXTRACTED |
| 749 | MinVarPortfolio | Markowitz (1952) | ✅ EXTRACTED |
| 750 | MVOptimization | Markowitz (1952) | ✅ EXTRACTED |
| 751 | TwoFundTheorem | Markowitz (1952) | ✅ EXTRACTED |
| 752 | Carhart4Factor | Carhart (1997) | ✅ EXTRACTED |
| 753 | MomentumFactor | Carhart (1997) | ✅ EXTRACTED |
| 754 | MomentumSignal | Jegadeesh-Titman (1993) | ✅ EXTRACTED |
| 755 | AutocorrelationTest | Fama (1970) | ✅ EXTRACTED |
| 756 | VarianceRatio | Lo-MacKinlay (1988) | ✅ EXTRACTED |
| 757 | VRStatistic | Lo-MacKinlay (1988) | ✅ EXTRACTED |
| 758 | LiquidityBeta | Pastor-Stambaugh (2003) | ✅ EXTRACTED |
| 759 | PSLiquidityMeasure | Pastor-Stambaugh (2003) | ✅ EXTRACTED |
| 760 | LiquidityAdjCAPM | Pastor-Stambaugh (2003) | ✅ EXTRACTED |

---

## SUMMARY: IDs 741-760 COMPLETE

**Total Extracted in journals.md: 20 formulas**

Categories Covered:
- CAPM (741-744)
- APT (745-746)
- Markowitz Optimization (747-751)
- Carhart/Momentum (752-754)
- Market Efficiency (755-757)
- Liquidity Premium (758-760)
