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
