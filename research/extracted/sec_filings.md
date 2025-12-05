# Hedge Fund Strategy Formula Extractions
## IDs: 761-800

---

## SOURCE 1: Long/Short Equity
**Status:** EXTRACTED

### Formula 761: Net Exposure
```
Net_Exposure = (Long_Value - |Short_Value|) / NAV
```
**Variables:**
- Long_Value = total long positions
- Short_Value = total short positions (absolute)
- NAV = Net Asset Value
**Edge:** Directional market bet indicator

### Formula 762: Gross Exposure
```
Gross_Exposure = (Long_Value + |Short_Value|) / NAV
```
**Edge:** Total market exposure (leverage indicator)

### Formula 763: Long/Short Ratio
```
L/S_Ratio = Long_Value / |Short_Value|
```
**Edge:** Bias indicator (>1 = long bias)

### Formula 764: 130/30 Portfolio
```
Long = 1.3 × NAV,  Short = 0.3 × NAV
Net = 100%,  Gross = 160%
```
**Edge:** Enhanced active management with shorting

### Formula 765: Long/Short Return Attribution
```
R_total = R_long × w_long + R_short × w_short + R_spread
```
**Variables:**
- R_spread = spread between longs and shorts
**Edge:** Decomposes L/S performance

---

## SOURCE 2: Market Neutral Strategies
**Status:** EXTRACTED

### Formula 766: Dollar Neutrality
```
Dollar_Neutral: Σ(Long_i) = Σ|Short_j|
```
**Edge:** Zero net dollar exposure

### Formula 767: Beta Neutrality
```
Beta_Neutral: Σ(β_i × Long_i) = Σ(β_j × |Short_j|)
```
**Edge:** Zero systematic risk exposure

### Formula 768: Sector Neutrality
```
Sector_Neutral: Long_sector_k = |Short_sector_k|,  ∀k
```
**Edge:** Balanced sector exposure

### Formula 769: Factor Neutrality
```
Factor_Neutral: Σ(f_i × w_i) = 0,  ∀ factors f
```
**Edge:** Zero factor exposure (size, value, momentum)

### Formula 770: Alpha Extraction (Market Neutral)
```
α = R_portfolio - β × R_market,  where β ≈ 0
```
**Edge:** Pure alpha from security selection

---

## SOURCE 3: Statistical Arbitrage
**Status:** EXTRACTED

### Formula 771: Pairs Trading Spread
```
S_t = log(P_A,t) - β × log(P_B,t) - c
```
**Variables:**
- β = hedge ratio from regression
- c = constant (drift adjustment)
**Edge:** Tracks relative value deviation

### Formula 772: Entry/Exit Signals
```
Entry: |z_t| > z_entry (e.g., 2.0)
Exit: |z_t| < z_exit (e.g., 0.5) or |z_t| > z_stop (e.g., 4.0)
```
**Edge:** Mean reversion trading rules

### Formula 773: Optimal Hedge Ratio
```
β* = Cov(ΔP_A, ΔP_B) / Var(ΔP_B)
```
**Edge:** Minimizes spread variance

### Formula 774: PCA Factor Model (StatArb)
```
R_i = Σ_{k=1}^K β_ik × F_k + ε_i
```
**Variables:**
- F_k = principal component factors
- ε_i = idiosyncratic return to trade
**Edge:** Residual return capture

### Formula 775: Mean Reversion Speed (OU Fit)
```
θ = -ln(ρ) / Δt
```
**Variables:**
- ρ = AR(1) coefficient of spread
- Δt = time step
**Edge:** Determines trading frequency

---

## SOURCE 4: Merger Arbitrage
**Status:** EXTRACTED

### Formula 776: Merger Spread (Cash Deal)
```
Spread = (Offer_Price - Current_Price) / Current_Price
```
**Edge:** Potential return if deal closes

### Formula 777: Merger Spread (Stock Deal)
```
Spread = (Exchange_Ratio × Acquirer_Price - Target_Price) / Target_Price
```
**Edge:** Stock-for-stock deal return

### Formula 778: Annualized Spread Return
```
Ann_Return = Spread × (365 / Expected_Days_to_Close)
```
**Edge:** Time-adjusted merger arb return

### Formula 779: Deal Probability Implied
```
P_deal = (Current_Price - Pre_Announce) / (Offer_Price - Pre_Announce)
```
**Edge:** Market-implied probability of deal completion

### Formula 780: Risk/Reward Ratio
```
R/R = Upside_if_Close / Downside_if_Fail
```
**Edge:** Asymmetric payoff assessment

---

## SOURCE 5: Convertible Arbitrage
**Status:** EXTRACTED

### Formula 781: Convertible Delta
```
Δ_CB = ∂CB_Price / ∂Stock_Price
```
**Edge:** Equity sensitivity of convertible

### Formula 782: Delta-Neutral Hedge Ratio
```
Shares_Short = Δ_CB × Conversion_Ratio × Bonds_Held
```
**Edge:** Number of shares to short per bond

### Formula 783: Convertible Arbitrage P&L
```
P&L = ΔCB - Δ_hedge × ΔStock + Carry - Borrow_Cost
```
**Variables:**
- Carry = coupon income
- Borrow_Cost = cost to short stock
**Edge:** Components of conv arb return

### Formula 784: Gamma Profit (Conv Arb)
```
Γ_profit = ½ × Γ × (ΔS)²
```
**Edge:** Profit from convexity on rebalancing

### Formula 785: Implied vs Realized Vol Trade
```
Edge = Vol_implied_CB - Vol_realized_stock
```
**Edge:** Cheap embedded optionality capture

---

## SOURCE 6: Global Macro / Carry Trade
**Status:** EXTRACTED

### Formula 786: FX Carry Return
```
R_carry = (1 + r_high) × S_t/S_{t+1} - (1 + r_low)
```
**Variables:**
- r_high = high yield currency rate
- r_low = low yield currency rate
- S = spot exchange rate
**Edge:** Interest differential + FX gain/loss

### Formula 787: Covered Interest Rate Parity
```
F/S = (1 + r_d) / (1 + r_f)
```
**Variables:**
- F = forward rate
- S = spot rate
- r_d, r_f = domestic/foreign rates
**Edge:** No-arbitrage forward pricing

### Formula 788: Uncovered Interest Rate Parity
```
E[S_{t+1}]/S_t = (1 + r_d) / (1 + r_f)
```
**Edge:** Expected depreciation equals rate differential

### Formula 789: Carry Trade Signal
```
Signal_i = r_i - r_USD
```
**Edge:** Long high yield, short low yield currencies

### Formula 790: Forward Rate Bias (Carry Profit)
```
Carry_Profit = (F - S) - E[S_{t+1} - S_t]
```
**Edge:** Forward premium puzzle exploitation

---

## SOURCE 7: Risk Premia Strategies
**Status:** EXTRACTED

### Formula 791: Value Factor (FX)
```
Value_FX = PPP_Rate / Spot_Rate - 1
```
**Variables:**
- PPP = Purchasing Power Parity rate
**Edge:** Currency undervaluation signal

### Formula 792: Momentum Factor (FX)
```
Mom_FX = R_currency(t-12, t-1)
```
**Edge:** Trend following in currencies

### Formula 793: Volatility Risk Premium
```
VRP = IV - RV_realized
```
**Variables:**
- IV = implied volatility
- RV = realized volatility
**Edge:** Sell volatility to capture premium

### Formula 794: Term Premium
```
Term_Premium = Y_long - Y_short - E[Rate_Changes]
```
**Edge:** Compensation for duration risk

### Formula 795: Credit Risk Premium
```
Credit_Spread = Y_corporate - Y_treasury - E[Default_Loss]
```
**Edge:** Compensation for credit risk

---

## SOURCE 8: CTA/Managed Futures
**Status:** EXTRACTED

### Formula 796: Trend Following Signal
```
Signal = SMA(P, fast) - SMA(P, slow)
```
**Variables:**
- fast = e.g., 20 days
- slow = e.g., 200 days
**Edge:** Momentum/trend detection

### Formula 797: Time Series Momentum
```
TSMOM_t = sign(R_{t-12,t}) × R_{t,t+1}
```
**Edge:** Trade own past return direction

### Formula 798: Position Sizing (Volatility Targeting)
```
w_i = σ_target / σ_i × sign(Signal_i)
```
**Variables:**
- σ_target = target volatility (e.g., 10%)
- σ_i = asset volatility
**Edge:** Risk-adjusted position sizing

### Formula 799: Breakout Signal
```
Long if: P > max(P_{t-n:t-1})
Short if: P < min(P_{t-n:t-1})
```
**Edge:** Donchian channel breakout

### Formula 800: Cross-Sectional Momentum (CTA)
```
XS_Mom = R_i - R̄_universe
```
**Edge:** Relative strength across assets

---

## EXTRACTION STATUS

| ID | Name | Source | Status |
|----|------|--------|--------|
| 761 | NetExposure | L/S Equity | ✅ EXTRACTED |
| 762 | GrossExposure | L/S Equity | ✅ EXTRACTED |
| 763 | LSRatio | L/S Equity | ✅ EXTRACTED |
| 764 | 130_30Portfolio | L/S Equity | ✅ EXTRACTED |
| 765 | LSReturnAttribution | L/S Equity | ✅ EXTRACTED |
| 766 | DollarNeutrality | Market Neutral | ✅ EXTRACTED |
| 767 | BetaNeutrality | Market Neutral | ✅ EXTRACTED |
| 768 | SectorNeutrality | Market Neutral | ✅ EXTRACTED |
| 769 | FactorNeutrality | Market Neutral | ✅ EXTRACTED |
| 770 | AlphaExtraction | Market Neutral | ✅ EXTRACTED |
| 771 | PairsTradingSpread | Stat Arb | ✅ EXTRACTED |
| 772 | EntryExitSignals | Stat Arb | ✅ EXTRACTED |
| 773 | OptimalHedgeRatio | Stat Arb | ✅ EXTRACTED |
| 774 | PCAFactorModel | Stat Arb | ✅ EXTRACTED |
| 775 | MeanReversionSpeed | Stat Arb | ✅ EXTRACTED |
| 776 | MergerSpreadCash | Merger Arb | ✅ EXTRACTED |
| 777 | MergerSpreadStock | Merger Arb | ✅ EXTRACTED |
| 778 | AnnualizedSpread | Merger Arb | ✅ EXTRACTED |
| 779 | DealProbability | Merger Arb | ✅ EXTRACTED |
| 780 | RiskRewardRatio | Merger Arb | ✅ EXTRACTED |
| 781 | ConvertibleDelta | Conv Arb | ✅ EXTRACTED |
| 782 | DeltaNeutralHedge | Conv Arb | ✅ EXTRACTED |
| 783 | ConvArbPnL | Conv Arb | ✅ EXTRACTED |
| 784 | GammaProfit | Conv Arb | ✅ EXTRACTED |
| 785 | ImpliedVsRealizedVol | Conv Arb | ✅ EXTRACTED |
| 786 | FXCarryReturn | Global Macro | ✅ EXTRACTED |
| 787 | CoveredIRP | Global Macro | ✅ EXTRACTED |
| 788 | UncoveredIRP | Global Macro | ✅ EXTRACTED |
| 789 | CarryTradeSignal | Global Macro | ✅ EXTRACTED |
| 790 | ForwardRateBias | Global Macro | ✅ EXTRACTED |
| 791 | ValueFactorFX | Risk Premia | ✅ EXTRACTED |
| 792 | MomentumFactorFX | Risk Premia | ✅ EXTRACTED |
| 793 | VolatilityRiskPremium | Risk Premia | ✅ EXTRACTED |
| 794 | TermPremium | Risk Premia | ✅ EXTRACTED |
| 795 | CreditRiskPremium | Risk Premia | ✅ EXTRACTED |
| 796 | TrendFollowingSignal | CTA | ✅ EXTRACTED |
| 797 | TimeSeriesMomentum | CTA | ✅ EXTRACTED |
| 798 | VolTargetPositionSize | CTA | ✅ EXTRACTED |
| 799 | BreakoutSignal | CTA | ✅ EXTRACTED |
| 800 | XSMomentum | CTA | ✅ EXTRACTED |

---

## SUMMARY: IDs 761-800 COMPLETE

**Total Extracted in sec_filings.md: 40 formulas**

Categories Covered:
- Long/Short Equity (761-765)
- Market Neutral (766-770)
- Statistical Arbitrage (771-775)
- Merger Arbitrage (776-780)
- Convertible Arbitrage (781-785)
- Global Macro / Carry (786-790)
- Risk Premia (791-795)
- CTA / Managed Futures (796-800)
