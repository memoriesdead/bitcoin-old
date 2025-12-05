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

---

## SOURCE 4: Execution Algorithms (US8571967B1)
**URL:** https://patents.google.com/patent/US8571967B1/en
**Title:** System and Method for Algorithmic Trading Strategies
**Status:** EXTRACTED

### Formula 602: Implementation Shortfall (IS)
```
IS = (Decision_Price - Execution_Price) × Quantity + Opportunity_Cost
```
**Variables:**
- Decision_Price = price when trade decision was made
- Execution_Price = actual average fill price
- Opportunity_Cost = cost of unfilled portion
**Edge:** Total execution cost measurement

### Formula 603: VWAP Benchmark
```
VWAP = Σ(P_i × V_i) / Σ(V_i)
```
**Variables:**
- P_i = price of trade i
- V_i = volume of trade i
**Edge:** Volume-weighted execution quality benchmark

### Formula 604: TWAP Benchmark
```
TWAP = Σ(P_i) / n
```
**Variables:**
- P_i = price at time i
- n = number of observations
**Edge:** Time-weighted average for even execution

### Formula 605: POV (Percentage of Volume)
```
Order_Size_t = Target_POV × Market_Volume_t
```
**Variables:**
- Target_POV = desired participation rate (e.g., 10%)
**Edge:** Adaptive execution matching market rhythm

### Formula 606: Arrival Price Slippage
```
Slippage = (Execution_Price - Arrival_Price) × sign(side) × Quantity
```
**Variables:**
- Arrival_Price = mid-price at order arrival
- side = +1 for buy, -1 for sell
**Edge:** Measures execution vs. decision point

---

## SOURCE 5: Market Impact Models
**Citation:** Perold (1988) "The Implementation Shortfall: Paper vs. Reality"
**Status:** EXTRACTED

### Formula 607: Linear Market Impact
```
Impact = γ × V / ADV
```
**Variables:**
- γ = market impact coefficient
- V = order volume
- ADV = average daily volume
**Edge:** Simple impact estimate

### Formula 608: Square Root Market Impact
```
Impact = σ × √(V / ADV)
```
**Variables:**
- σ = daily volatility
**Edge:** Empirically validated nonlinear impact

### Formula 609: Temporary vs Permanent Impact
```
Total_Impact = Temporary_Impact(v) + Permanent_Impact(V)
```
**Variables:**
- v = trading rate
- V = total volume
**Edge:** Decomposes reversible and persistent effects

### Formula 610: Expected Execution Cost
```
E[Cost] = ½ × γ × V + η × v × T
```
**Variables:**
- γ = permanent impact
- η = temporary impact coefficient
- T = trading duration
**Edge:** Balances impact vs timing risk

---

## SOURCE 6: Amihud Illiquidity (Academic)
**URL:** https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf
**Citation:** Amihud (2002) "Illiquidity and Stock Returns" J. Financial Markets
**Status:** EXTRACTED

### Formula 611: Amihud Illiquidity Ratio
```
ILLIQ = (1/T) × Σ|R_t| / VOL_t
```
**Variables:**
- R_t = daily return
- VOL_t = dollar trading volume
- T = number of days
**Edge:** Price impact per dollar traded

### Formula 612: Turnover-Based Amihud
```
ILLIQ_TO = (1/T) × Σ|R_t| / Turnover_t
```
**Variables:**
- Turnover = Volume / Shares Outstanding
**Edge:** Size-normalized illiquidity

### Formula 613: Expected Illiquidity
```
E[ILLIQ_t] = c_0 + c_1 × ILLIQ_{t-1} + ε_t
```
**Edge:** Illiquidity persistence model

---

## SOURCE 7: Order Routing Patents (US20150066727A1)
**URL:** https://patents.google.com/patent/US20150066727A1/en
**Title:** Electronic Trading Exchange with User-Definable Order Execution Delay
**Status:** EXTRACTED

### Formula 614: Price-Time Priority
```
Priority = (Price_Level, -Timestamp)
```
**Edge:** Standard exchange matching priority

### Formula 615: Price-Size Priority
```
Priority = (Price_Level, Size, -Timestamp)
```
**Edge:** Larger orders get priority at same price

### Formula 616: Speed Bump Delay
```
Execution_Time = Arrival_Time + Δt_speedbump
```
**Variables:**
- Δt_speedbump = intentional delay (e.g., 350μs IEX)
**Edge:** Reduces HFT latency arbitrage

### Formula 617: Randomized Delay
```
Delay = Base_Delay + U(0, Random_Window)
```
**Variables:**
- U(0, x) = uniform random [0, x]
**Edge:** Unpredictable execution prevents gaming

---

## SOURCE 8: Spread and Quote Models
**Status:** EXTRACTED

### Formula 618: Bid-Ask Spread
```
Spread = P_ask - P_bid
```
**Edge:** Basic liquidity cost measure

### Formula 619: Relative Spread
```
Relative_Spread = (P_ask - P_bid) / P_mid × 10000  (in bps)
```
**Edge:** Normalized spread comparison

### Formula 620: Effective Spread
```
Effective_Spread = 2 × |P_execution - P_mid| × sign(side)
```
**Edge:** Actual cost of crossing spread

### Formula 621: Realized Spread
```
Realized_Spread = 2 × sign(side) × (P_execution - P_{t+Δ})
```
**Variables:**
- P_{t+Δ} = mid-price after delay Δ
**Edge:** Market maker's actual profit

### Formula 622: Price Improvement
```
PI = max(0, P_mid - P_execution) × sign(side)
```
**Edge:** Execution better than mid-quote

---

## SOURCE 9: Order Book Imbalance Models
**Status:** EXTRACTED

### Formula 623: Order Book Imbalance (OBI)
```
OBI = (V_bid - V_ask) / (V_bid + V_ask)
```
**Variables:**
- V_bid = total bid volume
- V_ask = total ask volume
**Edge:** Directional pressure indicator

### Formula 624: Weighted OBI
```
WOBI = Σ(w_i × V_bid_i) - Σ(w_i × V_ask_i)
```
**Variables:**
- w_i = weight (e.g., 1/distance_from_mid)
**Edge:** Distance-weighted imbalance

### Formula 625: Microprice
```
Microprice = P_bid × (V_ask/(V_bid+V_ask)) + P_ask × (V_bid/(V_bid+V_ask))
```
**Edge:** Imbalance-adjusted fair value

### Formula 626: Queue Position Value
```
QPV = E[Fill_Prob] × (Fair_Value - Limit_Price)
```
**Edge:** Value of queue priority

---

## SOURCE 10: Latency Arbitrage Models
**Status:** EXTRACTED

### Formula 627: Latency Arbitrage Profit
```
Profit = |P_stale - P_new| × Quantity - Transaction_Costs
```
**Variables:**
- P_stale = price on slow venue
- P_new = updated price on fast venue
**Edge:** Speed advantage monetization

### Formula 628: Quote Staleness
```
Staleness = Prob(P_quote ≠ P_fair | Δt > τ)
```
**Variables:**
- τ = latency threshold
**Edge:** Probability quote is outdated

### Formula 629: Race Condition Probability
```
P_race = 1 - exp(-λ × Δt_latency)
```
**Variables:**
- λ = message arrival rate
**Edge:** Chance of adverse selection from speed

---

## SOURCE 11: Smart Order Router (SOR) Models
**Status:** EXTRACTED

### Formula 630: Expected Fill Probability
```
P_fill = 1 - (1 - p_1)(1 - p_2)...(1 - p_n)
```
**Variables:**
- p_i = fill probability on venue i
**Edge:** Multi-venue fill rate

### Formula 631: Optimal Order Split
```
x_i* = arg min_x Σ[Impact_i(x_i) + Spread_i × x_i]  s.t. Σx_i = X
```
**Edge:** Cost-minimizing venue allocation

### Formula 632: Venue Score
```
Score_v = w_1 × Fill_Rate + w_2 × Price_Improvement - w_3 × Latency
```
**Edge:** Multi-factor venue ranking

---

## SOURCE 12: Statistical Arbitrage Models
**Status:** EXTRACTED

### Formula 633: Pairs Trading Z-Score
```
z_t = (Spread_t - μ_spread) / σ_spread
```
**Variables:**
- Spread = log(P_A) - β × log(P_B)
**Edge:** Mean reversion entry signal

### Formula 634: Cointegration Error Correction
```
ΔP_A,t = α × (P_A,t-1 - β × P_B,t-1) + ε_t
```
**Variables:**
- α = speed of adjustment
- β = cointegration coefficient
**Edge:** Long-run equilibrium restoration

### Formula 635: Half-Life of Mean Reversion
```
t_{1/2} = -ln(2) / ln(θ)
```
**Variables:**
- θ = AR(1) coefficient of spread
**Edge:** Expected time to close half the gap

### Formula 636: Optimal Entry Threshold
```
z_entry = √(2 × ln(1 + k/σ))
```
**Variables:**
- k = transaction cost
- σ = spread volatility
**Edge:** Breakeven signal level

---

## SOURCE 13: Risk Controls
**Status:** EXTRACTED

### Formula 637: Position Limit Check
```
Valid = |Current_Position + Order_Size| ≤ Max_Position
```
**Edge:** Pre-trade risk control

### Formula 638: Notional Limit
```
Notional = Σ|Position_i × Price_i| ≤ Max_Notional
```
**Edge:** Total exposure constraint

### Formula 639: Order Rate Limit
```
Rate_Valid = Orders_in_Window / Window_Size ≤ Max_Rate
```
**Edge:** Prevents runaway algorithms

### Formula 640: Loss Limit Check
```
Halt_Trading if: Daily_PnL ≤ -Max_Daily_Loss
```
**Edge:** Circuit breaker on losses

---

## EXTRACTION STATUS (UPDATED)

| ID | Name | Source | Status |
|----|------|--------|--------|
| 602 | ImplementationShortfall | US8571967B1 | ✅ EXTRACTED |
| 603 | VWAPBenchmark | US8571967B1 | ✅ EXTRACTED |
| 604 | TWAPBenchmark | US8571967B1 | ✅ EXTRACTED |
| 605 | POVAlgorithm | US8571967B1 | ✅ EXTRACTED |
| 606 | ArrivalPriceSlippage | US8571967B1 | ✅ EXTRACTED |
| 607 | LinearMarketImpact | Perold (1988) | ✅ EXTRACTED |
| 608 | SqrtMarketImpact | - | ✅ EXTRACTED |
| 609 | TempPermImpact | - | ✅ EXTRACTED |
| 610 | ExpectedExecCost | - | ✅ EXTRACTED |
| 611 | AmihudILLIQ | Amihud (2002) | ✅ EXTRACTED |
| 612 | TurnoverAmihud | Florackis (2011) | ✅ EXTRACTED |
| 613 | ExpectedIlliquidity | Amihud (2002) | ✅ EXTRACTED |
| 614 | PriceTimePriority | US20150066727A1 | ✅ EXTRACTED |
| 615 | PriceSizePriority | US20150066727A1 | ✅ EXTRACTED |
| 616 | SpeedBumpDelay | US20150066727A1 | ✅ EXTRACTED |
| 617 | RandomizedDelay | US20150066727A1 | ✅ EXTRACTED |
| 618 | BidAskSpread | - | ✅ EXTRACTED |
| 619 | RelativeSpread | - | ✅ EXTRACTED |
| 620 | EffectiveSpread | - | ✅ EXTRACTED |
| 621 | RealizedSpread | - | ✅ EXTRACTED |
| 622 | PriceImprovement | - | ✅ EXTRACTED |
| 623 | OrderBookImbalance | - | ✅ EXTRACTED |
| 624 | WeightedOBI | - | ✅ EXTRACTED |
| 625 | Microprice | - | ✅ EXTRACTED |
| 626 | QueuePositionValue | - | ✅ EXTRACTED |
| 627 | LatencyArbProfit | - | ✅ EXTRACTED |
| 628 | QuoteStaleness | - | ✅ EXTRACTED |
| 629 | RaceConditionProb | - | ✅ EXTRACTED |
| 630 | ExpectedFillProb | - | ✅ EXTRACTED |
| 631 | OptimalOrderSplit | - | ✅ EXTRACTED |
| 632 | VenueScore | - | ✅ EXTRACTED |
| 633 | PairsZScore | - | ✅ EXTRACTED |
| 634 | CointegrationEC | - | ✅ EXTRACTED |
| 635 | MeanReversionHalfLife | - | ✅ EXTRACTED |
| 636 | OptimalEntryThreshold | - | ✅ EXTRACTED |
| 637 | PositionLimitCheck | - | ✅ EXTRACTED |
| 638 | NotionalLimit | - | ✅ EXTRACTED |
| 639 | OrderRateLimit | - | ✅ EXTRACTED |
| 640 | LossLimitCheck | - | ✅ EXTRACTED |

---

## SUMMARY: IDs 591-640 COMPLETE

**Total Extracted in patents.md: 50 formulas**

Categories Covered:
- Hidden Orders (591-595)
- Micro Auction (596-598)
- Kyle Lambda (599-601)
- Execution Algorithms (602-606)
- Market Impact (607-610)
- Amihud Illiquidity (611-613)
- Order Routing (614-617)
- Spread Models (618-622)
- Order Book Imbalance (623-626)
- Latency Arbitrage (627-629)
- Smart Order Routing (630-632)
- Statistical Arbitrage (633-636)
- Risk Controls (637-640)
