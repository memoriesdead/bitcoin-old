# Academic Core Formula Extractions
## IDs: 641-719

---

## SOURCE 1: Hawkes Processes (Order Flow Modeling)
**URL:** https://arxiv.org/html/2408.03594v1
**Citation:** "Forecasting high frequency order flow imbalance using Hawkes processes" (2024)
**Status:** EXTRACTED

### Formula 641: Hawkes Intensity Function (General)
```
λ_t^i = μ_i + Σ_{j=1}^D ∫ dN_{t'}^j × φ_{ij}(t - t')
```
**Variables:**
- μ_i = exogenous (background) intensity
- φ_{ij}(t) = kernel function for cross-excitation
- N_t = counting process
**Edge:** Models self-exciting order arrivals

### Formula 642: Exponential Hawkes Kernel
```
φ(t) = α × e^(-βt)
```
**Variables:**
- α = excitation magnitude
- β = decay rate
**Edge:** Each arrival increases intensity by α, decays at rate β

### Formula 643: Sum of Exponentials Kernel
```
φ_{i,j}(t) = Σ_{u=1}^U α_{ij}^u × e^(-β^u × t)
```
**Edge:** Multi-timescale excitation effects

### Formula 644: Power Law Kernel
```
φ(t) = α / (δ + t)^β
```
**Edge:** Long memory effects in order flow

### Formula 645: Two-Dimensional Buy/Sell Hawkes
```
[λ_t^s]   [φ_ss  φ_sb]
[λ_t^b] = [φ_bs  φ_bb] ∗ ΔN_t
```
**Variables:**
- s = sell, b = buy
- ∗ = convolution
**Edge:** Models cross-excitation between buy/sell flows

### Formula 646: Order Flow Imbalance (Hawkes)
```
OFI(T,h) = [ΔN_{t-h,t}^s - ΔN_{t-h,t}^b] / [ΔN_{t-h,t}^s + ΔN_{t-h,t}^b]
```
**Variables:**
- h = lookback window
- ΔN = incremental trade counts
**Edge:** Normalized order imbalance for prediction

### Formula 647: Hawkes Log-Likelihood
```
ln L(λ(t,w)) = ∫_0^T ln(λ(t,w)) × dN_t - ∫_0^T λ(t,w) × dt
```
**Edge:** MLE estimation of Hawkes parameters

---

## SOURCE 2: Almgren-Chriss Optimal Execution
**URL:** https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf
**Citation:** Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions" J. Risk
**Status:** EXTRACTED

### Formula 648: Optimal Trajectory (Discrete)
```
x_j = [sinh(κ(T - t_j)) / sinh(κT)] × X,  for j = 0,...,N
```
**Variables:**
- x_j = holdings at time t_j
- X = initial position
- T = execution horizon
- κ = urgency parameter
**Edge:** Front-loaded execution minimizes risk

### Formula 649: Optimal Trading List
```
n_j = [2 sinh(½κτ) / sinh(κT)] × cosh(κ(T - t_{j-½})) × X
```
**Variables:**
- n_j = shares traded in period j
- τ = time step
**Edge:** Determines trade sizes per period

### Formula 650: Optimal Trajectory (Continuous)
```
x(t) = X × sinh(κ(T-t)) / sinh(κT)
```
**Edge:** Continuous-time limit of discrete solution

### Formula 651: Urgency Parameter κ
```
κ² = λσ² / η
```
**Variables:**
- λ = risk aversion
- σ = volatility
- η = temporary impact coefficient
**Edge:** Higher risk aversion → faster execution

### Formula 652: κ Discrete Approximation
```
κ = (1/τ) × cosh⁻¹((τ²/2)κ̃² + 1)
```
**Edge:** Converts continuous to discrete

### Formula 653: Euler-Lagrange Equation
```
ẍ - κ²x = 0
```
**Edge:** Differential equation for optimal path

### Formula 654: Optimal Trajectory with Drift
```
x_j = [sinh(κ(T-t_j))/sinh(κT)] × X + [1 - (sinh(κ(T-t_j)) + sinh(κt_j))/sinh(κT)] × x̄
```
**Variables:**
- x̄ = drift-adjusted target
**Edge:** Accounts for expected price drift

### Formula 655: Temporary Impact Function
```
h(v) = η × |v|
```
**Variables:**
- v = trading rate
- η = temporary impact coefficient
**Edge:** Linear temporary price impact

### Formula 656: Permanent Impact Function
```
g(v) = γ × v
```
**Variables:**
- γ = permanent impact coefficient
**Edge:** Linear permanent price impact

### Formula 657: Expected Shortfall (Cost)
```
E[C] = ½γX² + ε × Σ_{k=1}^N |n_k|
```
**Edge:** Mean execution cost

### Formula 658: Variance of Shortfall
```
V[C] = σ² × Σ_{k=1}^N τ × x_k²
```
**Edge:** Risk of execution

### Formula 659: Mean-Variance Objective
```
U(x) = E[C] + λ × V[C]
```
**Edge:** Risk-adjusted cost minimization

---

## SOURCE 3: Kyle Lambda (Market Microstructure)
**Citation:** Kyle (1985) "Continuous Auctions and Insider Trading" Econometrica
**Status:** EXTRACTED (see patents.md)

### Formula 660: Kyle Equilibrium
```
P(y) = μ + λy
```
Referenced in patents.md as IDs 599-601

---

## PYTHON IMPLEMENTATIONS

```python
# ID: 641
@FormulaRegistry.register(641, "HawkesIntensity", "point_process")
class HawkesIntensity(BaseFormula):
    """
    Source: arXiv:2408.03594 "Forecasting OFI using Hawkes"
    URL: https://arxiv.org/html/2408.03594v1

    Formula: λ_t^i = μ_i + Σ ∫ dN × φ(t-t')

    Edge: Model self-exciting order arrivals for flow prediction
    """

    def _compute(self):
        # background_intensity + sum(kernel * event_history)
        pass

# ID: 648
@FormulaRegistry.register(648, "AlmgrenChrissTrajectory", "execution")
class AlmgrenChrissTrajectory(BaseFormula):
    """
    Source: Almgren & Chriss (2000) J. Risk
    URL: https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf

    Formula: x_j = sinh(κ(T-t_j))/sinh(κT) × X

    Edge: Optimal execution path minimizing risk-adjusted cost
    """

    def _compute(self):
        # sinh(kappa*(T-t)) / sinh(kappa*T) * X
        pass

# ID: 651
@FormulaRegistry.register(651, "UrgencyParameter", "execution")
class UrgencyParameter(BaseFormula):
    """
    Source: Almgren & Chriss (2000)

    Formula: κ² = λσ²/η

    Edge: Determines execution urgency from risk aversion
    """

    def _compute(self):
        # sqrt(risk_aversion * volatility^2 / temp_impact)
        pass
```

---

## EXTRACTION STATUS

| ID | Name | Source | Status |
|----|------|--------|--------|
| 641 | HawkesIntensity | arXiv:2408.03594 | ✅ EXTRACTED |
| 642 | ExponentialKernel | arXiv:2408.03594 | ✅ EXTRACTED |
| 643 | SumExpKernel | arXiv:2408.03594 | ✅ EXTRACTED |
| 644 | PowerLawKernel | arXiv:2408.03594 | ✅ EXTRACTED |
| 645 | BuySellHawkes | arXiv:2408.03594 | ✅ EXTRACTED |
| 646 | HawkesOFI | arXiv:2408.03594 | ✅ EXTRACTED |
| 647 | HawkesLogLikelihood | arXiv:2408.03594 | ✅ EXTRACTED |
| 648 | AlmgrenChrissTrajectory | Almgren-Chriss (2000) | ✅ EXTRACTED |
| 649 | OptimalTradingList | Almgren-Chriss (2000) | ✅ EXTRACTED |
| 650 | ContinuousTrajectory | Almgren-Chriss (2000) | ✅ EXTRACTED |
| 651 | UrgencyParameter | Almgren-Chriss (2000) | ✅ EXTRACTED |
| 652 | KappaDiscrete | Almgren-Chriss (2000) | ✅ EXTRACTED |
| 653 | EulerLagrange | Almgren-Chriss (2000) | ✅ EXTRACTED |
| 654 | TrajectoryWithDrift | Almgren-Chriss (2000) | ✅ EXTRACTED |
| 655 | TemporaryImpact | Almgren-Chriss (2000) | ✅ EXTRACTED |
| 656 | PermanentImpact | Almgren-Chriss (2000) | ✅ EXTRACTED |
| 657 | ExpectedShortfall | Almgren-Chriss (2000) | ✅ EXTRACTED |
| 658 | VarianceShortfall | Almgren-Chriss (2000) | ✅ EXTRACTED |
| 659 | MeanVarianceObjective | Almgren-Chriss (2000) | ✅ EXTRACTED |
| 660 | KyleEquilibrium | Kyle (1985) | ✅ EXTRACTED |

---

## SOURCE 4: VPIN (Volume-Synchronized Probability)
**Citation:** Easley, Lopez de Prado, O'Hara (2012) "Flow Toxicity and Liquidity" RFS
**Status:** EXTRACTED

### Formula 661: VPIN (Order Toxicity)
```
VPIN = (1/n) × Σ |V^B_τ - V^S_τ| / VBS
```
**Variables:**
- n = number of volume buckets in sample
- V^B_τ = buy volume in bucket τ
- V^S_τ = sell volume in bucket τ
- VBS = volume bucket size
**Edge:** Predicts flash crashes, detects informed trading

### Formula 662: Volume Bucket Size
```
VBS = Average_Daily_Volume / 50
```
**Edge:** Normalizes information content per bucket

### Formula 663: Order Imbalance per Bucket
```
OI_τ = |V^B_τ - V^S_τ|
```
**Edge:** Measures directional pressure

---

## SOURCE 5: Heston Stochastic Volatility
**Citation:** Heston (1993) "A Closed-Form Solution for Options with Stochastic Volatility" RFS
**Status:** EXTRACTED

### Formula 664: Heston Variance SDE
```
dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v
```
**Variables:**
- v_t = instantaneous variance
- κ = mean reversion rate
- θ = long-term variance
- σ = vol of vol
- dW_t^v = Wiener process
**Edge:** Models volatility clustering, mean reversion

### Formula 665: Heston Asset SDE
```
dS_t = μS_t dt + √v_t S_t dW_t^S
```
**Variables:**
- Correlation: dW_t^S × dW_t^v = ρdt
**Edge:** Captures leverage effect via ρ

### Formula 666: Heston Characteristic Function (Core)
```
Ψ(u) = exp(C(u,τ) + D(u,τ)v_0 + iu ln(S_0))
```
**Edge:** Enables semi-analytical option pricing

---

## SOURCE 6: Rough Volatility
**Citation:** Gatheral, Jaisson, Rosenbaum (2018) "Volatility is Rough" Quant Finance
**Status:** EXTRACTED

### Formula 667: Fractional Brownian Motion
```
W_t^H = ∫_0^t K_H(t,s) dW_s
```
**Variables:**
- H = Hurst exponent (H < 0.5 for rough)
- K_H = kernel function
**Edge:** Models rough paths in volatility

### Formula 668: Hurst Exponent from Realized Variance
```
E[|log(RV_{t+Δ}) - log(RV_t)|^q] ∝ Δ^(qH)
```
**Variables:**
- H ≈ 0.1 empirically
**Edge:** Volatility is rougher than Brownian motion

### Formula 669: RFSV Model (Rough FSV)
```
log(v_t) = m + ν × fOU_t^H
```
**Variables:**
- fOU = fractional Ornstein-Uhlenbeck
- H < 0.5
**Edge:** Predicts volatility better than GARCH/HAR

---

## EXTRACTION STATUS (CONTINUED)

| ID | Name | Source | Status |
|----|------|--------|--------|
| 661 | VPIN | Easley et al. (2012) | ✅ EXTRACTED |
| 662 | VolumeBucketSize | Easley et al. (2012) | ✅ EXTRACTED |
| 663 | OrderImbalance | Easley et al. (2012) | ✅ EXTRACTED |
| 664 | HestonVarianceSDE | Heston (1993) | ✅ EXTRACTED |
| 665 | HestonAssetSDE | Heston (1993) | ✅ EXTRACTED |
| 666 | HestonCharFunc | Heston (1993) | ✅ EXTRACTED |
| 667 | FractionalBM | Gatheral et al. (2018) | ✅ EXTRACTED |
| 668 | HurstFromRV | Gatheral et al. (2018) | ✅ EXTRACTED |
| 669 | RFSVModel | Gatheral et al. (2018) | ✅ EXTRACTED |
| 670-719 | - | - | PENDING |

---

## SOURCE 7: Avellaneda-Stoikov Market Making
**Citation:** Avellaneda & Stoikov (2008) "High-frequency trading in a limit order book" Quantitative Finance
**URL:** https://www.math.nyu.edu/~avellane/HighFrequencyTrading.pdf
**Status:** EXTRACTED

### Formula 670: Reservation Price
```
r(s,q,t) = s - q × γ × σ² × (T - t)
```
**Variables:**
- s = mid-price
- q = inventory position
- γ = risk aversion coefficient
- σ = volatility
- T-t = time to horizon
**Edge:** Adjusts fair value based on inventory risk

### Formula 671: Optimal Bid-Ask Spread
```
δ(t) = γσ²(T-t) + (2/γ) × ln(1 + γ/k)
```
**Variables:**
- k = order arrival intensity parameter
**Edge:** Widens spread when inventory risk is high

### Formula 672: Indifference Bid Price
```
p_b = r - δ/2
```
**Edge:** Where to place bid quote

### Formula 673: Indifference Ask Price
```
p_a = r + δ/2
```
**Edge:** Where to place ask quote

### Formula 674: Inventory Penalty Term
```
Penalty = q × γ × σ² × (T - t)
```
**Edge:** Risk from holding inventory

---

## SOURCE 8: GARCH Volatility Models
**Citation:** Bollerslev (1986) "Generalized Autoregressive Conditional Heteroskedasticity" J. Econometrics
**Status:** EXTRACTED

### Formula 675: GARCH(1,1) Variance
```
σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
```
**Variables:**
- ω = long-run variance weight
- α = ARCH coefficient (shock impact)
- β = GARCH coefficient (persistence)
- ε_{t-1} = previous return shock
**Constraint:** α + β < 1 for stationarity
**Edge:** Models volatility clustering

### Formula 676: GARCH Long-Run Variance
```
σ²_∞ = ω / (1 - α - β)
```
**Edge:** Unconditional variance target

### Formula 677: GARCH(p,q) General Form
```
σ²_t = ω + Σ_{i=1}^q α_i × ε²_{t-i} + Σ_{j=1}^p β_j × σ²_{t-j}
```
**Edge:** Extended lag structure

### Formula 678: EGARCH (Exponential)
```
ln(σ²_t) = ω + α × (|z_{t-1}| - E[|z|]) + γ × z_{t-1} + β × ln(σ²_{t-1})
```
**Variables:**
- z_t = standardized residual
- γ = leverage parameter
**Edge:** Captures asymmetric volatility response

---

## SOURCE 9: AMM/DeFi Pricing
**Citation:** Uniswap V2 Whitepaper (2020)
**URL:** https://uniswap.org/whitepaper.pdf
**Status:** EXTRACTED

### Formula 679: Constant Product AMM
```
x × y = k
```
**Variables:**
- x = reserve of token X
- y = reserve of token Y
- k = invariant constant
**Edge:** Core DEX pricing mechanism

### Formula 680: AMM Price
```
P = y / x
```
**Edge:** Spot price from reserves

### Formula 681: AMM Swap Output
```
Δy = y × Δx / (x + Δx)
```
**Variables:**
- Δx = input amount
- Δy = output amount
**Edge:** Calculates trade execution

### Formula 682: AMM Price Impact
```
Impact = 2 × Δx / (x + Δx)
```
**Edge:** Slippage from trade size

### Formula 683: AMM Impermanent Loss
```
IL = 2 × √(P_ratio) / (1 + P_ratio) - 1
```
**Variables:**
- P_ratio = P_new / P_initial
**Edge:** LP value loss from price divergence

---

## SOURCE 10: Risk Measures (VaR/CVaR)
**Citation:** Artzner et al. (1999) "Coherent Measures of Risk" Mathematical Finance
**Status:** EXTRACTED

### Formula 684: Value at Risk (VaR)
```
VaR_α = -inf{x : P(L ≤ x) ≥ α}
```
**Variables:**
- α = confidence level (e.g., 0.95)
- L = loss distribution
**Edge:** Maximum loss at confidence level

### Formula 685: Parametric VaR (Normal)
```
VaR_α = μ + σ × Φ^{-1}(α)
```
**Variables:**
- Φ^{-1} = inverse normal CDF
**Edge:** Assumes normal returns

### Formula 686: Conditional VaR (CVaR/ES)
```
CVaR_α = E[L | L > VaR_α]
```
**Edge:** Expected loss beyond VaR (coherent risk measure)

### Formula 687: CVaR Normal Formula
```
CVaR_α = μ + σ × φ(Φ^{-1}(α)) / (1-α)
```
**Variables:**
- φ = normal PDF
**Edge:** Closed-form for normal distribution

---

## EXTRACTION STATUS (CONTINUED 2)

| ID | Name | Source | Status |
|----|------|--------|--------|
| 670 | ReservationPrice | Avellaneda-Stoikov (2008) | ✅ EXTRACTED |
| 671 | OptimalSpread | Avellaneda-Stoikov (2008) | ✅ EXTRACTED |
| 672 | IndifferenceBid | Avellaneda-Stoikov (2008) | ✅ EXTRACTED |
| 673 | IndifferenceAsk | Avellaneda-Stoikov (2008) | ✅ EXTRACTED |
| 674 | InventoryPenalty | Avellaneda-Stoikov (2008) | ✅ EXTRACTED |
| 675 | GARCH11Variance | Bollerslev (1986) | ✅ EXTRACTED |
| 676 | GARCHLongRunVar | Bollerslev (1986) | ✅ EXTRACTED |
| 677 | GARCHpq | Bollerslev (1986) | ✅ EXTRACTED |
| 678 | EGARCH | Nelson (1991) | ✅ EXTRACTED |
| 679 | ConstantProductAMM | Uniswap V2 (2020) | ✅ EXTRACTED |
| 680 | AMMPrice | Uniswap V2 (2020) | ✅ EXTRACTED |
| 681 | AMMSwapOutput | Uniswap V2 (2020) | ✅ EXTRACTED |
| 682 | AMMPriceImpact | Uniswap V2 (2020) | ✅ EXTRACTED |
| 683 | ImpermanentLoss | Uniswap V2 (2020) | ✅ EXTRACTED |
| 684 | VaR | Artzner et al. (1999) | ✅ EXTRACTED |
| 685 | ParametricVaR | Artzner et al. (1999) | ✅ EXTRACTED |
| 686 | CVaR | Artzner et al. (1999) | ✅ EXTRACTED |
| 687 | CVaRNormal | Artzner et al. (1999) | ✅ EXTRACTED |
| 688-719 | - | - | PENDING |

---

## SOURCE 11: Black-Scholes Option Pricing
**Citation:** Black & Scholes (1973) "The Pricing of Options and Corporate Liabilities" J. Political Economy
**Citation:** Merton (1973) "Theory of Rational Option Pricing" Bell Journal of Economics
**URL:** https://www.macroption.com/black-scholes-formula/
**Status:** EXTRACTED

### Formula 688: d1 Parameter
```
d1 = [ln(S/K) + (r + ½σ²)T] / (σ√T)
```
**Variables:**
- S = current stock price
- K = strike price
- r = risk-free rate
- σ = volatility
- T = time to expiration
**Edge:** Core input for option price calculation

### Formula 689: d2 Parameter
```
d2 = d1 - σ√T = [ln(S/K) + (r - ½σ²)T] / (σ√T)
```
**Edge:** Probability that option finishes in-the-money

### Formula 690: Call Option Price
```
C = S × N(d1) - K × e^(-rT) × N(d2)
```
**Variables:**
- N(x) = standard normal CDF
**Edge:** Theoretical fair value of European call

### Formula 691: Put Option Price
```
P = K × e^(-rT) × N(-d2) - S × N(-d1)
```
**Edge:** Theoretical fair value of European put

### Formula 692: Put-Call Parity
```
C - P = S - K × e^(-rT)
```
**Edge:** Arbitrage relationship between calls and puts

---

## SOURCE 12: Fama-French Factor Models
**Citation:** Fama & French (1993) "Common Risk Factors in Returns" J. Financial Economics
**URL:** https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
**Status:** EXTRACTED

### Formula 693: Three-Factor Model
```
R_i - R_f = α_i + β_1 × (R_m - R_f) + β_2 × SMB + β_3 × HML + ε_i
```
**Variables:**
- R_i = portfolio return
- R_f = risk-free rate
- R_m = market return
- SMB = Small Minus Big (size factor)
- HML = High Minus Low (value factor)
**Edge:** Explains cross-section of stock returns better than CAPM

### Formula 694: SMB Factor
```
SMB = ⅓(Small Value + Small Neutral + Small Growth) - ⅓(Big Value + Big Neutral + Big Growth)
```
**Edge:** Size premium capture

### Formula 695: HML Factor
```
HML = ½(Small Value + Big Value) - ½(Small Growth + Big Growth)
```
**Edge:** Value premium capture

### Formula 696: Five-Factor Model
```
R_i - R_f = α + β_MKT(R_m-R_f) + β_SMB×SMB + β_HML×HML + β_RMW×RMW + β_CMA×CMA + ε
```
**Variables:**
- RMW = Robust Minus Weak (profitability)
- CMA = Conservative Minus Aggressive (investment)
**Edge:** Extended factor model with profitability and investment factors

---

## SOURCE 13: Merton Jump Diffusion
**Citation:** Merton (1976) "Option Pricing When Underlying Stock Returns Are Discontinuous" J. Financial Economics
**URL:** https://quant-next.com/the-merton-jump-diffusion-model/
**Status:** EXTRACTED

### Formula 697: Jump Diffusion SDE
```
dS/S = μ dt + σ dW + dJ
```
**Variables:**
- μ = drift
- σ = diffusion volatility
- W = Brownian motion
- J = compound Poisson process (jump component)
**Edge:** Models rare large price movements (crashes/rallies)

### Formula 698: Jump Component
```
dJ = Σ(Y_i - 1) dN_t, where Y_i ~ LogNormal(μ_J, σ_J²)
```
**Variables:**
- N_t = Poisson process with intensity λ
- Y_i = i.i.d. jump size multipliers
**Edge:** Lognormal jump sizes driven by Poisson arrivals

### Formula 699: Merton Option Price (Infinite Series)
```
C = Σ_{n=0}^∞ [e^(-λτ)(λτ)^n / n!] × BS(S_n, K, r_n, σ_n, τ)
```
**Variables:**
- S_n = S × exp(n × (μ_J + ½σ_J²))
- σ_n² = σ² + n × σ_J² / τ
- r_n = r - λ × k + n × log(1+k) / τ
**Edge:** Weighted sum of Black-Scholes prices conditioned on jump count

---

## SOURCE 14: Ornstein-Uhlenbeck Process
**Citation:** Uhlenbeck & Ornstein (1930) Physical Review
**URL:** https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process
**Status:** EXTRACTED

### Formula 700: OU Process SDE
```
dX_t = θ(μ - X_t)dt + σ dW_t
```
**Variables:**
- θ = mean reversion speed
- μ = long-term mean
- σ = volatility
- W_t = Brownian motion
**Edge:** Models mean-reverting processes (spreads, rates)

### Formula 701: OU Analytical Solution
```
X_t = X_0 × e^(-θt) + μ(1 - e^(-θt)) + σ ∫_0^t e^(-θ(t-s)) dW_s
```
**Edge:** Explicit solution for simulation

### Formula 702: OU Half-Life
```
t_{1/2} = ln(2) / θ
```
**Edge:** Time to revert halfway to mean

---

## SOURCE 15: Kelly Criterion
**Citation:** Kelly (1956) "A New Interpretation of Information Rate" Bell System Technical Journal
**URL:** https://corporatefinanceinstitute.com/resources/data-science/kelly-criterion/
**Status:** EXTRACTED

### Formula 703: Kelly Fraction
```
f* = (bp - q) / b = (p(b+1) - 1) / b
```
**Variables:**
- f* = optimal fraction to bet
- b = odds received on bet (decimal - 1)
- p = probability of winning
- q = probability of losing (1-p)
**Edge:** Maximizes long-term geometric growth rate

### Formula 704: Kelly for Continuous Outcomes
```
f* = (μ - r) / σ²
```
**Variables:**
- μ = expected return
- r = risk-free rate
- σ² = variance of returns
**Edge:** Optimal leverage for continuous distributions

### Formula 705: Fractional Kelly
```
f_fractional = κ × f*,  where κ ∈ (0, 1)
```
**Edge:** Reduces volatility at cost of growth rate

---

## SOURCE 16: Performance Ratios
**Citation:** Sharpe (1966) "Mutual Fund Performance" Journal of Business
**Citation:** Sortino & Price (1994) "Performance Measurement in a Downside Risk Framework" Journal of Investing
**URL:** https://www.cmegroup.com/education/files/rr-sortino-a-sharper-ratio.pdf
**Status:** EXTRACTED

### Formula 706: Sharpe Ratio
```
SR = (R_p - R_f) / σ_p
```
**Variables:**
- R_p = portfolio return
- R_f = risk-free rate
- σ_p = portfolio standard deviation
**Edge:** Risk-adjusted performance measure

### Formula 707: Sortino Ratio
```
Sortino = (R_p - MAR) / σ_downside
```
**Variables:**
- MAR = minimum acceptable return
- σ_downside = downside deviation (only negative returns)
**Edge:** Penalizes only downside volatility

### Formula 708: Information Ratio
```
IR = (R_p - R_b) / TE
```
**Variables:**
- R_b = benchmark return
- TE = tracking error (std dev of excess returns)
**Edge:** Measures alpha generation consistency

### Formula 709: Treynor Ratio
```
TR = (R_p - R_f) / β_p
```
**Variables:**
- β_p = portfolio beta
**Edge:** Return per unit of systematic risk

### Formula 710: Calmar Ratio
```
Calmar = CAGR / Max Drawdown
```
**Edge:** Return relative to worst loss

### Formula 711: Maximum Drawdown
```
MDD = max_{t∈[0,T]} [(max_{s∈[0,t]} P_s) - P_t] / max_{s∈[0,t]} P_s
```
**Edge:** Largest peak-to-trough decline

---

## EXTRACTION STATUS (CONTINUED 3)

| ID | Name | Source | Status |
|----|------|--------|--------|
| 688 | BS_d1 | Black-Scholes (1973) | ✅ EXTRACTED |
| 689 | BS_d2 | Black-Scholes (1973) | ✅ EXTRACTED |
| 690 | BSCallPrice | Black-Scholes (1973) | ✅ EXTRACTED |
| 691 | BSPutPrice | Black-Scholes (1973) | ✅ EXTRACTED |
| 692 | PutCallParity | Black-Scholes (1973) | ✅ EXTRACTED |
| 693 | FF3Factor | Fama-French (1993) | ✅ EXTRACTED |
| 694 | SMBFactor | Fama-French (1993) | ✅ EXTRACTED |
| 695 | HMLFactor | Fama-French (1993) | ✅ EXTRACTED |
| 696 | FF5Factor | Fama-French (2015) | ✅ EXTRACTED |
| 697 | JumpDiffusionSDE | Merton (1976) | ✅ EXTRACTED |
| 698 | JumpComponent | Merton (1976) | ✅ EXTRACTED |
| 699 | MertonOptionPrice | Merton (1976) | ✅ EXTRACTED |
| 700 | OUProcessSDE | Uhlenbeck-Ornstein (1930) | ✅ EXTRACTED |
| 701 | OUSolution | Uhlenbeck-Ornstein (1930) | ✅ EXTRACTED |
| 702 | OUHalfLife | Uhlenbeck-Ornstein (1930) | ✅ EXTRACTED |
| 703 | KellyFraction | Kelly (1956) | ✅ EXTRACTED |
| 704 | KellyContinuous | Kelly (1956) | ✅ EXTRACTED |
| 705 | FractionalKelly | Kelly (1956) | ✅ EXTRACTED |
| 706 | SharpeRatio | Sharpe (1966) | ✅ EXTRACTED |
| 707 | SortinoRatio | Sortino (1994) | ✅ EXTRACTED |
| 708 | InformationRatio | - | ✅ EXTRACTED |
| 709 | TreynorRatio | Treynor (1965) | ✅ EXTRACTED |
| 710 | CalmarRatio | - | ✅ EXTRACTED |
| 711 | MaxDrawdown | - | ✅ EXTRACTED |
| 712-719 | - | - | PENDING |

---

## SOURCE 17: Option Greeks
**Citation:** Black & Scholes (1973), Merton (1973)
**URL:** https://www.macroption.com/black-scholes-formula/
**Status:** EXTRACTED

### Formula 712: Delta (Call)
```
Δ_call = N(d1)
```
**Edge:** Hedge ratio, probability proxy for ITM

### Formula 713: Delta (Put)
```
Δ_put = N(d1) - 1 = -N(-d1)
```
**Edge:** Negative for puts (inverse relationship)

### Formula 714: Gamma (Both)
```
Γ = φ(d1) / (S × σ × √T)
```
**Variables:**
- φ(x) = standard normal PDF
**Edge:** Convexity of option price, delta sensitivity

### Formula 715: Theta (Call)
```
Θ_call = -[S × φ(d1) × σ / (2√T)] - r × K × e^(-rT) × N(d2)
```
**Edge:** Time decay (usually negative)

### Formula 716: Vega (Both)
```
ν = S × √T × φ(d1)
```
**Edge:** Volatility sensitivity (same for calls/puts)

### Formula 717: Rho (Call)
```
ρ_call = K × T × e^(-rT) × N(d2)
```
**Edge:** Interest rate sensitivity

---

## SOURCE 18: Interest Rate Models
**Citation:** Vasicek (1977) "An Equilibrium Characterization of the Term Structure"
**Citation:** Cox, Ingersoll, Ross (1985) "A Theory of the Term Structure of Interest Rates"
**URL:** https://en.wikipedia.org/wiki/Vasicek_model
**Status:** EXTRACTED

### Formula 718: Vasicek Model
```
dr_t = θ(μ - r_t)dt + σ dW_t
```
**Variables:**
- θ = mean reversion speed
- μ = long-term mean rate
- σ = volatility
**Edge:** Mean-reverting rates (allows negative rates)

### Formula 719: Cox-Ingersoll-Ross (CIR) Model
```
dr_t = κ(θ - r_t)dt + σ√r_t dW_t
```
**Variables:**
- Feller condition: 2κθ > σ² (ensures positivity)
**Edge:** Non-negative rates via square-root diffusion

---

## EXTRACTION STATUS (FINAL)

| ID | Name | Source | Status |
|----|------|--------|--------|
| 712 | DeltaCall | Black-Scholes Greeks | ✅ EXTRACTED |
| 713 | DeltaPut | Black-Scholes Greeks | ✅ EXTRACTED |
| 714 | Gamma | Black-Scholes Greeks | ✅ EXTRACTED |
| 715 | ThetaCall | Black-Scholes Greeks | ✅ EXTRACTED |
| 716 | Vega | Black-Scholes Greeks | ✅ EXTRACTED |
| 717 | RhoCall | Black-Scholes Greeks | ✅ EXTRACTED |
| 718 | VasicekModel | Vasicek (1977) | ✅ EXTRACTED |
| 719 | CIRModel | Cox-Ingersoll-Ross (1985) | ✅ EXTRACTED |

---

## SUMMARY: IDs 641-719 COMPLETE

**Total Extracted in academic_core.md: 79 formulas**

Categories Covered:
- Hawkes Processes (641-647)
- Almgren-Chriss Execution (648-659)
- Kyle Lambda (660)
- VPIN (661-663)
- Heston Volatility (664-666)
- Rough Volatility (667-669)
- Avellaneda-Stoikov (670-674)
- GARCH (675-678)
- AMM/DeFi (679-683)
- Risk Measures (684-687)
- Black-Scholes (688-692)
- Fama-French (693-696)
- Merton Jump (697-699)
- Ornstein-Uhlenbeck (700-702)
- Kelly Criterion (703-705)
- Performance Ratios (706-711)
- Option Greeks (712-717)
- Interest Rate Models (718-719)
