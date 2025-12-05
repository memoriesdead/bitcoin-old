# QuantLib Formula Extractions
## IDs: 811-900

---

## SOURCE 1: Short Rate Models
**Citation:** Vasicek (1977), CIR (1985), Hull-White (1990)
**Status:** EXTRACTED

### Formula 811: Vasicek Model (QuantLib)
```
dr_t = κ(θ - r_t)dt + σ dW_t
```
**Variables:**
- κ = mean reversion speed
- θ = long-term mean rate
- σ = volatility
- dW_t = Wiener process
**Edge:** Analytical bond prices, allows negative rates

### Formula 812: Vasicek Bond Price
```
P(t,T) = A(t,T) × exp(-B(t,T) × r_t)
B(t,T) = (1 - exp(-κ(T-t))) / κ
A(t,T) = exp[(B(t,T) - T + t)(κ²θ - σ²/2)/κ² - σ²B(t,T)²/(4κ)]
```
**Edge:** Closed-form zero-coupon bond pricing

### Formula 813: CIR Model (Cox-Ingersoll-Ross)
```
dr_t = κ(θ - r_t)dt + σ√r_t dW_t
```
**Constraint:** 2κθ > σ² (Feller condition for positivity)
**Edge:** Rate stays positive, mean-reverting

### Formula 814: CIR Bond Price
```
P(t,T) = A(t,T) × exp(-B(t,T) × r_t)
B(t,T) = 2(exp(γτ) - 1) / [(γ + κ)(exp(γτ) - 1) + 2γ]
γ = √(κ² + 2σ²)
```
**Edge:** Closed-form positive rate model

### Formula 815: Hull-White One-Factor Model
```
dr_t = (θ(t) - κ × r_t)dt + σ(t) dW_t
```
**Variables:**
- θ(t) = time-varying drift (fitted to term structure)
**Edge:** Exact fit to initial yield curve

### Formula 816: Hull-White Bond Price
```
P(t,T) = P(0,T)/P(0,t) × exp[-B(t,T)(r_t - f(0,t)) - ½σ²(1-e^(-2κt))B(t,T)²/(2κ)]
B(t,T) = (1 - exp(-κ(T-t))) / κ
```
**Edge:** Arbitrage-free term structure fitting

### Formula 817: Hull-White Calibration (Mean Reversion)
```
θ(t) = ∂f(0,t)/∂t + κf(0,t) + σ²(1 - e^(-2κt))/(2κ)
```
**Variables:**
- f(0,t) = instantaneous forward rate at time 0 for time t
**Edge:** Forces model to fit observed forward curve

### Formula 818: Black-Karasinski Model
```
d(ln r_t) = (θ(t) - κ ln r_t)dt + σ dW_t
```
**Edge:** Lognormal rate distribution, always positive

### Formula 819: G2++ Two-Factor Model
```
r_t = x_t + y_t + φ(t)
dx_t = -a × x_t dt + σ dW₁
dy_t = -b × y_t dt + η dW₂
⟨dW₁, dW₂⟩ = ρ dt
```
**Edge:** Two-factor affine model with correlation

### Formula 820: Affine Term Structure Model (General)
```
P(t,T) = exp(A(t,T) - B(t,T)ᵀx_t)
```
**Variables:**
- x_t = vector of state variables
- A, B satisfy Riccati ODEs
**Edge:** General framework for multi-factor models

---

## SOURCE 2: Market Models (LMM/BGM)
**Citation:** Brace, Gatarek, Musiela (1997) "The Market Model of Interest Rate Dynamics"
**Status:** EXTRACTED

### Formula 821: LIBOR Forward Rate Dynamics
```
dF_i(t)/F_i(t) = σ_i(t) dW_i^{T_i}(t)
```
**Variables:**
- F_i(t) = forward LIBOR for period [T_i, T_{i+1}]
- W^{T_i} = Brownian motion under T_i-forward measure
**Edge:** Each forward rate is lognormal under its own measure

### Formula 822: LMM Drift Under Terminal Measure
```
dF_i(t)/F_i(t) = -Σ_{j=i+1}^N [ρ_{ij}σ_i(t)σ_j(t)τ_jF_j(t)/(1+τ_jF_j(t))] dt + σ_i(t)dW_i^N
```
**Variables:**
- τ_j = day count fraction for period j
- ρ_{ij} = correlation between forward rates
**Edge:** Drift correction for terminal measure simulation

### Formula 823: LMM Caplet Price (Black)
```
Caplet = τ × P(0,T_{i+1}) × [F_i(0)N(d₁) - K×N(d₂)]
d₁ = [ln(F_i/K) + ½σ_i²T_i] / (σ_i√T_i)
d₂ = d₁ - σ_i√T_i
```
**Edge:** Direct application of Black formula to forward rate

### Formula 824: Swap Rate Dynamics (Market Model)
```
S_α,β(t) = [P(t,T_α) - P(t,T_β)] / A_α,β(t)
A_α,β(t) = Σ_{i=α+1}^β τ_i P(t,T_i)
```
**Variables:**
- S_α,β = swap rate from T_α to T_β
- A_α,β = annuity (PV01)
**Edge:** Swap rate formula in terms of bond prices

### Formula 825: Swaption Black Formula
```
Swaption = A_α,β(0) × [S(0)N(d₁) - K×N(d₂)] × (1 for payer, -1 for receiver)
d₁ = [ln(S/K) + ½σ²T_α] / (σ√T_α)
```
**Edge:** Market standard for swaption pricing

---

## SOURCE 3: Yield Curve Mathematics
**Citation:** QuantLib Documentation, "Implementing QuantLib" (Ballabio)
**Status:** EXTRACTED

### Formula 826: Discount Factor to Zero Rate
```
z(t) = -ln(D(t)) / t
D(t) = exp(-z(t) × t)
```
**Variables:**
- D(t) = discount factor for time t
- z(t) = continuously compounded zero rate
**Edge:** Fundamental relationship for curve representation

### Formula 827: Forward Rate from Discount Factors
```
f(t₁,t₂) = [ln(D(t₁)) - ln(D(t₂))] / (t₂ - t₁)
f(t₁,t₂) = -[D(t₂)/D(t₁) - 1] / (t₂ - t₁)  [simple compounding]
```
**Edge:** Extract forward rate from discount curve

### Formula 828: Instantaneous Forward Rate
```
f(t) = -∂ln(D(t))/∂t = z(t) + t × ∂z(t)/∂t
```
**Edge:** Limit of forward rate as tenor → 0

### Formula 829: Bootstrapping Iteration
```
D(T_n) = [Quote_n - Σ_{i<n} c_i × D(T_i)] / (1 + c_n × τ_n)
```
**Variables:**
- Quote_n = market quote (swap rate, deposit rate)
- c_i = fixed coupon rate
- τ_n = day count fraction
**Edge:** Sequential extraction of discount factors

### Formula 830: Log-Linear Interpolation (Discount)
```
ln(D(t)) = ln(D(t₁)) + (t - t₁)/(t₂ - t₁) × [ln(D(t₂)) - ln(D(t₁))]
```
**Edge:** Continuous forward rates between nodes

### Formula 831: Cubic Spline Zero Rate Interpolation
```
z(t) = a_i + b_i(t-t_i) + c_i(t-t_i)² + d_i(t-t_i)³,  t ∈ [t_i, t_{i+1}]
```
**Constraint:** Continuity of z, z', z'' at nodes
**Edge:** Smooth zero curve with continuous forwards

### Formula 832: Nelson-Siegel Yield Curve
```
z(t) = β₀ + β₁[(1-e^(-t/τ))/(t/τ)] + β₂[(1-e^(-t/τ))/(t/τ) - e^(-t/τ)]
```
**Variables:**
- β₀ = long-term level
- β₁ = slope
- β₂ = curvature
- τ = decay parameter
**Edge:** Parsimonious yield curve fitting

### Formula 833: Svensson Extended Nelson-Siegel
```
z(t) = β₀ + β₁[(1-e^(-t/τ₁))/(t/τ₁)] + β₂[(1-e^(-t/τ₁))/(t/τ₁) - e^(-t/τ₁)]
        + β₃[(1-e^(-t/τ₂))/(t/τ₂) - e^(-t/τ₂)]
```
**Edge:** Better fit with two humps

### Formula 834: OIS Discounting
```
D_OIS(t) = E^Q[exp(-∫₀ᵗ r_s ds)]
Forward_OIS = [D_OIS(t₁)/D_OIS(t₂) - 1] / τ
```
**Edge:** Post-2008 multi-curve framework

### Formula 835: Forward-OIS Spread
```
Spread = Forward_LIBOR - Forward_OIS
```
**Edge:** Credit/liquidity risk in interbank rates

---

## SOURCE 4: Credit Derivatives (CDS)
**Citation:** ISDA Standard Model, QuantLib Credit Module
**Status:** EXTRACTED

### Formula 836: Survival Probability
```
S(t) = P(τ > t) = exp(-∫₀ᵗ λ(s)ds)
```
**Variables:**
- τ = default time
- λ(t) = hazard rate (intensity)
**Edge:** Probability of no default by time t

### Formula 837: Hazard Rate (Flat)
```
λ = -ln(S(t)) / t
S(t) = exp(-λ × t)
```
**Edge:** Simple hazard rate from survival probability

### Formula 838: CDS Premium Leg PV
```
PV_premium = s × Σᵢ Δtᵢ × D(tᵢ) × S(tᵢ)
```
**Variables:**
- s = CDS spread (running premium)
- Δtᵢ = accrual period
- D(tᵢ) = risk-free discount factor
- S(tᵢ) = survival probability
**Edge:** Expected present value of premium payments

### Formula 839: CDS Protection Leg PV
```
PV_protection = (1 - R) × ∫₀ᵀ D(t) × (-dS(t))
             ≈ (1 - R) × Σᵢ D(tᵢ) × [S(tᵢ₋₁) - S(tᵢ)]
```
**Variables:**
- R = recovery rate
- (1-R) = loss given default
**Edge:** Expected present value of protection payment

### Formula 840: CDS Fair Spread
```
s_fair = [(1 - R) × Σᵢ D(tᵢ)(S(tᵢ₋₁) - S(tᵢ))] / [Σᵢ Δtᵢ × D(tᵢ) × S(tᵢ)]
```
**Edge:** Spread that makes CDS NPV = 0

### Formula 841: CDS Mark-to-Market
```
MTM = (s_market - s_contract) × Risky_PV01
Risky_PV01 = Σᵢ Δtᵢ × D(tᵢ) × S(tᵢ)
```
**Edge:** MTM value of existing CDS position

### Formula 842: Implied Hazard Rate from CDS
```
λ_implied ≈ s / (1 - R)
```
**Variables:**
- s = CDS spread
- R = recovery rate
**Edge:** Quick approximation for flat hazard rate

### Formula 843: Hazard Rate Bootstrapping
```
S(T_n) = [PV_protection_n - (1-R)Σᵢ₌₁^{n-1} D(tᵢ)(S(tᵢ₋₁)-S(tᵢ))] / [(1-R)D(T_n)S(T_{n-1})]
```
**Edge:** Sequential extraction of survival curve from CDS quotes

### Formula 844: Upfront CDS Convention (Post-Big Bang)
```
Upfront = (s_par - s_running) × Risky_PV01
```
**Variables:**
- s_par = par spread
- s_running = standardized running spread (100bp or 500bp)
**Edge:** ISDA standard quoting convention

### Formula 845: JPMorgan CDS Index (CDX/iTraxx)
```
Index_Spread = Σᵢ wᵢ × sᵢ / Σᵢ wᵢ
```
**Variables:**
- wᵢ = constituent weight (usually equal)
- sᵢ = individual CDS spread
**Edge:** Credit index fair spread

---

## SOURCE 5: Exotic Options
**Citation:** QuantLib Exotic Pricing Engines
**Status:** EXTRACTED

### Formula 846: Barrier Option - Down-and-Out Call
```
C_do = C_BS - C_di
C_di = (S/H)^(2λ) × C_BS(H²/S, K, σ, r, T)
λ = (r - q + σ²/2) / σ²
```
**Variables:**
- H = barrier level (H < S for down)
- C_di = rebate from in-barrier
**Edge:** Analytical knock-out option price

### Formula 847: Barrier Option - Up-and-Out Put
```
P_uo = P_BS - P_ui
P_ui = (H/S)^(2λ) × P_BS(H²/S, K, σ, r, T)
```
**Edge:** Analytical up-and-out put

### Formula 848: Double Barrier Option
```
V = Σₙ₌₋∞^∞ Aₙ × [V_call(S×e^{2n×d}) - V_call(S×e^{2n×d+2a})]
d = ln(U/L),  a = ln(U/S)
```
**Variables:**
- U = upper barrier
- L = lower barrier
**Edge:** Both knock-out levels

### Formula 849: Asian Option - Geometric Average (Analytical)
```
Price = BSM(S, K, σ_adj, r, T)
σ_adj = σ / √3
drift_adj = (r - q - σ²/2)/2 + σ_adj²/2
```
**Edge:** Closed-form geometric Asian

### Formula 850: Asian Option - Arithmetic Average (Turnbull-Wakeman)
```
Price ≈ BSM(S, K, σ_A, μ_A, T)
M₁ = [exp((r-q)T) - 1] / [(r-q)T]
M₂ = [2exp((2(r-q)+σ²)T)] / [(r-q+σ²)(2(r-q)+σ²)T²] + ...
σ_A² = (1/T) × ln(M₂/M₁²)
```
**Edge:** Moment-matching for arithmetic Asian

### Formula 851: Lookback Call (Floating Strike)
```
C_lookback = S×e^(-qT)×N(a₁) - S_min×e^(-rT)×N(a₂)
             + S×e^(-rT)×(σ²/2r)×[-(S/S_min)^(-2r/σ²)×N(-a₁+2r√T/σ) + e^(rT)×N(-a₁)]
```
**Variables:**
- S_min = minimum price observed
**Edge:** Option on realized minimum

### Formula 852: Lookback Put (Floating Strike)
```
P_lookback = -S×e^(-qT)×N(-a₁) + S_max×e^(-rT)×N(-a₂)
             + S×e^(-rT)×(σ²/2r)×[(S/S_max)^(-2r/σ²)×N(a₁-2r√T/σ) - e^(rT)×N(a₁)]
```
**Variables:**
- S_max = maximum price observed
**Edge:** Option on realized maximum

### Formula 853: Cliquet Option (Forward Starting)
```
Cliquet_Payoff = Σᵢ max(0, Sᵢ/Sᵢ₋₁ - 1 - local_floor)
                 subject to global_cap, global_floor
```
**Edge:** Sum of capped/floored periodic returns

### Formula 854: Chooser Option
```
Chooser = C(S,K,T) + P(S,K×e^(-r(T-t)),t)×e^(-q(T-t)}
```
**Variables:**
- t = choice date
- T = expiry
**Edge:** Option to choose call or put at future date

### Formula 855: Compound Option (Call on Call)
```
CoC = S×N₂(a,b;√(t/T)) - K₂×e^(-rT)×N₂(a-σ√t, b-σ√T; √(t/T)) - K₁×e^(-rt)×N(a-σ√t)
```
**Variables:**
- K₁ = strike for outer option
- K₂ = strike for inner option
- N₂ = bivariate normal CDF
**Edge:** Option on option

---

## SOURCE 6: Swaption and Cap/Floor Pricing
**Citation:** QuantLib Interest Rate Derivatives
**Status:** EXTRACTED

### Formula 856: Black76 Swaption (Lognormal)
```
Swaption = A × ω × [S×N(ω×d₁) - K×N(ω×d₂)]
d₁ = [ln(S/K) + ½σ²T] / (σ√T)
d₂ = d₁ - σ√T
ω = +1 (payer), -1 (receiver)
```
**Variables:**
- A = annuity (swap PV01)
- S = forward swap rate
- σ = Black (lognormal) volatility
**Edge:** Market standard for normal rate environments

### Formula 857: Bachelier Swaption (Normal)
```
Swaption = A × [(S-K)×N(d) + σ√T×n(d)]  (payer)
d = (S - K) / (σ√T)
n(d) = (1/√2π)×exp(-d²/2)
```
**Variables:**
- σ = normal (Bachelier) volatility (in rate units, e.g., 100bp)
**Edge:** Handles negative rates, post-2012 standard

### Formula 858: Normal to Lognormal Vol Conversion
```
σ_Black ≈ σ_Bachelier / S  (at-the-money)
σ_Bachelier = σ_Black × S
```
**Edge:** Approximate conversion between conventions

### Formula 859: Caplet Black Formula
```
Caplet = τ × D(T₂) × [F×N(d₁) - K×N(d₂)]
d₁ = [ln(F/K) + ½σ²T₁] / (σ√T₁)
```
**Variables:**
- F = forward rate for period [T₁, T₂]
- τ = day count fraction
**Edge:** Individual caplet pricing

### Formula 860: Cap Price (Sum of Caplets)
```
Cap = Σᵢ Caplet_i = Σᵢ τᵢ × D(Tᵢ) × [Fᵢ×N(d₁ᵢ) - K×N(d₂ᵢ)]
```
**Edge:** Cap = portfolio of caplets

### Formula 861: Cap/Floor Parity
```
Cap - Floor = Swap_payer
```
**Edge:** Arbitrage relationship

### Formula 862: Swaption Cash Settlement
```
Cash_Annuity = Σᵢ τᵢ / (1 + S×τᵢ)^{i×τᵢ}
```
**Edge:** Cash-settled swaption annuity convention

### Formula 863: SABR Model for Swaption Vol
```
σ_SABR(K,F) = α/[(FK)^((1-β)/2)] × [z/x(z)] × {...}
z = (ν/α)×(FK)^((1-β)/2)×ln(F/K)
x(z) = ln[(√(1-2ρz+z²) + z - ρ)/(1-ρ)]
```
**Variables:**
- α = vol of vol level
- β = backbone parameter
- ρ = correlation
- ν = vol of vol
**Edge:** Smile dynamics for interest rate options

### Formula 864: Displaced Diffusion
```
dS = σ(S + d) dW
```
**Variables:**
- d = displacement parameter
**Edge:** Shifted lognormal for low/negative rates

### Formula 865: Shifted SABR
```
Apply SABR to (F + shift) with strike (K + shift)
```
**Edge:** SABR for negative rate environment

---

## SOURCE 7: Monte Carlo Methods
**Citation:** Glasserman (2003) "Monte Carlo Methods in Financial Engineering"
**Status:** EXTRACTED

### Formula 866: Monte Carlo Estimator
```
V̂ = e^(-rT) × (1/N) × Σᵢ f(Sᵢ_T)
SE = σ̂ / √N
```
**Variables:**
- N = number of paths
- SE = standard error
- σ̂ = sample standard deviation of payoffs
**Edge:** Basic Monte Carlo pricing

### Formula 867: Antithetic Variates
```
V̂_AV = e^(-rT) × (1/N) × Σᵢ [f(Sᵢ⁺) + f(Sᵢ⁻)] / 2
Sᵢ⁺ uses Zᵢ, Sᵢ⁻ uses -Zᵢ
```
**Edge:** Variance reduction via negative correlation

### Formula 868: Control Variate
```
V̂_CV = V̂_MC - β × (Ĉ_MC - C_analytical)
β* = Cov(V, C) / Var(C)
```
**Variables:**
- C = control variate with known analytical value
**Edge:** Reduce variance using correlated instrument

### Formula 869: Importance Sampling
```
V = E^P[f(X)] = E^Q[f(X) × (dP/dQ)]
```
**Variables:**
- dP/dQ = likelihood ratio (Radon-Nikodym derivative)
**Edge:** Sample more from important regions

### Formula 870: Stratified Sampling
```
V̂_SS = Σⱼ pⱼ × V̂ⱼ
```
**Variables:**
- pⱼ = probability of stratum j
- V̂ⱼ = estimate within stratum j
**Edge:** Reduce variance by dividing sample space

### Formula 871: Quasi-Monte Carlo (Low Discrepancy)
```
D*_N = sup |#(points in [0,x])/N - x|
```
**Variables:**
- D*_N = star discrepancy
**Edge:** Sobol, Halton sequences fill space more uniformly

### Formula 872: Longstaff-Schwartz (LSM) for American Options
```
Continuation Value_t = E[V_{t+1} | S_t]
                     ≈ Σₖ βₖ × Lₖ(S_t)
```
**Variables:**
- Lₖ = basis functions (e.g., Laguerre polynomials)
**Edge:** Regression-based optimal stopping

### Formula 873: Euler-Maruyama Discretization
```
S_{t+Δt} = S_t + μ(S_t)Δt + σ(S_t)√Δt × Z
```
**Edge:** Simple SDE discretization

### Formula 874: Milstein Scheme
```
S_{t+Δt} = S_t + μΔt + σ√Δt×Z + ½σ(∂σ/∂S)(Z² - 1)Δt
```
**Edge:** Higher order accuracy than Euler

### Formula 875: Pathwise Greeks (Delta)
```
Δ = e^(-rT) × E[f'(S_T) × ∂S_T/∂S_0]
∂S_T/∂S_0 = S_T/S_0  (for GBM)
```
**Edge:** Unbiased delta via automatic differentiation

---

## SOURCE 8: Finite Difference Methods
**Citation:** Wilmott (2006) "Paul Wilmott on Quantitative Finance"
**Status:** EXTRACTED

### Formula 876: Black-Scholes PDE
```
∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
```
**Edge:** Fundamental PDE for option pricing

### Formula 877: Explicit Finite Difference
```
V(i,j-1) = a×V(i-1,j) + b×V(i,j) + c×V(i+1,j)
a = ½Δt(σ²i² - ri)
b = 1 - σ²i²Δt - rΔt
c = ½Δt(σ²i² + ri)
```
**Edge:** Simple but stability constrained

### Formula 878: Implicit Finite Difference
```
-a×V(i-1,j-1) + (1+b)×V(i,j-1) - c×V(i+1,j-1) = V(i,j)
```
**Edge:** Unconditionally stable, requires tridiagonal solve

### Formula 879: Crank-Nicolson Scheme
```
V_{j-1} = ½[Explicit(V_j) + Implicit(V_{j-1})]
```
**Edge:** Second-order accurate in time and space

### Formula 880: CFL Condition (Explicit Stability)
```
Δt ≤ 1 / (σ²i²_max + r)
Δt/Δx² ≤ ½  (heat equation)
```
**Edge:** Maximum stable time step

### Formula 881: Log-Transform (Change of Variables)
```
x = ln(S),  τ = T - t
∂V/∂τ = ½σ²∂²V/∂x² + (r - ½σ²)∂V/∂x - rV
```
**Edge:** Constant coefficients, easier numerics

### Formula 882: American Option Boundary Condition
```
V(S,t) ≥ max(S - K, 0)  [call]
V(S,t) ≥ max(K - S, 0)  [put]
```
**Edge:** Free boundary problem for early exercise

### Formula 883: PSOR (Projected SOR) for American
```
V^{n+1}_{k+1,i} = max(payoff_i, V^{n+1}_{k,i} + ω(V̄ - V^{n+1}_{k,i}))
```
**Variables:**
- ω = relaxation parameter
- V̄ = Gauss-Seidel update
**Edge:** Iterative American option solver

### Formula 884: ADI (Alternating Direction Implicit) for 2D
```
(I - ½ΔtA_x)V* = (I + ½ΔtA_y)V^n
(I - ½ΔtA_y)V^{n+1} = (I + ½ΔtA_x)V*
```
**Variables:**
- A_x, A_y = differential operators in x, y directions
**Edge:** Efficient multi-dimensional PDE solver

### Formula 885: Douglas-Rachford Splitting
```
V* = V^n + θΔt(A_x + A_y)V^n + (1-θ)Δt×A_x(V* - V^n)
V^{n+1} = V* + (1-θ)Δt×A_y(V^{n+1} - V^n)
```
**Edge:** Alternative ADI scheme for 2D options

---

## SOURCE 9: Risk Neutral Pricing Theory
**Citation:** Harrison & Kreps (1979), Harrison & Pliska (1981)
**Status:** EXTRACTED

### Formula 886: Risk-Neutral Pricing Fundamental
```
V_t = E^Q[e^{-∫_t^T r_s ds} × Payoff_T | F_t]
```
**Variables:**
- Q = risk-neutral (equivalent martingale) measure
- F_t = filtration at time t
**Edge:** Foundation of derivative pricing

### Formula 887: Girsanov Theorem (Measure Change)
```
dW^Q_t = dW^P_t + λ_t dt
dP/dQ|_{F_T} = exp(-∫_0^T λ_t dW^Q_t - ½∫_0^T λ_t² dt)
```
**Variables:**
- λ_t = market price of risk
**Edge:** Change from physical to risk-neutral measure

### Formula 888: Numeraire Change
```
V_t/N_t = E^{Q_N}[V_T/N_T | F_t]
```
**Variables:**
- N_t = numeraire (e.g., money market, bond, annuity)
- Q_N = measure associated with numeraire N
**Edge:** Simplifies pricing with appropriate numeraire

### Formula 889: Forward Measure
```
Under Q^T: dS_t/S_t = σ_t dW^T_t  (forward = martingale)
F(t,T) = E^{Q^T}[S_T | F_t]
```
**Edge:** Forward price is martingale under T-forward measure

### Formula 890: Annuity Measure (Swaptions)
```
Under Q^A: S_{α,β}(t) = E^{Q^A}[S_{α,β}(T_α) | F_t]
```
**Variables:**
- Q^A = annuity measure with numeraire A_α,β
**Edge:** Swap rate is martingale under annuity measure

---

## SOURCE 10: Calibration Methods
**Citation:** QuantLib Calibration Helpers
**Status:** EXTRACTED

### Formula 891: Calibration Objective (Least Squares)
```
min_θ Σᵢ wᵢ × [V_model(θ) - V_market]² / V_market²
```
**Variables:**
- θ = model parameters
- wᵢ = calibration weights
**Edge:** Relative pricing error minimization

### Formula 892: Implied Vol Calibration Objective
```
min_θ Σᵢ wᵢ × [σ_model(θ) - σ_market]²
```
**Edge:** Fit to implied vol surface

### Formula 893: Vega-Weighted Calibration
```
min_θ Σᵢ (Vega_i)² × [σ_model - σ_market]²
```
**Edge:** More weight on liquid options

### Formula 894: Regularization (Tikhonov)
```
min_θ [Σᵢ error_i² + λ × ||θ - θ_prior||²]
```
**Variables:**
- λ = regularization parameter
- θ_prior = prior parameter values
**Edge:** Prevent overfitting, smooth parameters

### Formula 895: Local Volatility Dupire
```
σ_local²(K,T) = [∂C/∂T + rK×∂C/∂K + qC] / [½K²×∂²C/∂K²]
```
**Edge:** Extract local vol from option prices

### Formula 896: Dupire Forward PDE
```
∂C/∂T = ½σ²(K,T)K²∂²C/∂K² - (r-q)K∂C/∂K - qC
```
**Edge:** Price all strikes/maturities in one sweep

### Formula 897: Implied Vol Smile Interpolation (SVI)
```
w(k) = a + b(ρ(k-m) + √((k-m)² + σ²))
```
**Variables:**
- w = total implied variance
- k = log-moneyness
- a, b, ρ, m, σ = SVI parameters
**Edge:** Arbitrage-free smile parameterization

### Formula 898: Vol Surface Calendar Spread Arbitrage
```
∂w/∂T ≥ 0  (total variance increasing in maturity)
```
**Edge:** No-arbitrage constraint on vol surface

### Formula 899: Vol Surface Butterfly Arbitrage
```
∂²C/∂K² ≥ 0  (convexity in strike)
```
**Edge:** No-arbitrage constraint on smile

### Formula 900: Heston Calibration (Fourier Method)
```
C = S×P₁ - K×e^{-rT}×P₂
P_j = ½ + (1/π)∫_0^∞ Re[e^{-iu×ln(K)}×f_j(u)/(iu)] du
```
**Variables:**
- f_j = characteristic function of Heston model
**Edge:** Fast calibration via Fourier inversion

---

## EXTRACTION STATUS

| ID | Name | Source | Status |
|----|------|--------|--------|
| 811 | VasicekModel | Vasicek (1977) | EXTRACTED |
| 812 | VasicekBondPrice | Vasicek (1977) | EXTRACTED |
| 813 | CIRModel | CIR (1985) | EXTRACTED |
| 814 | CIRBondPrice | CIR (1985) | EXTRACTED |
| 815 | HullWhite1F | Hull-White (1990) | EXTRACTED |
| 816 | HWBondPrice | Hull-White (1990) | EXTRACTED |
| 817 | HWCalibration | Hull-White (1990) | EXTRACTED |
| 818 | BlackKarasinski | Black-Karasinski (1991) | EXTRACTED |
| 819 | G2ppModel | Brigo-Mercurio (2006) | EXTRACTED |
| 820 | AffineTSModel | Duffie-Kan (1996) | EXTRACTED |
| 821 | LIBORForwardDynamics | BGM (1997) | EXTRACTED |
| 822 | LMMTerminalDrift | BGM (1997) | EXTRACTED |
| 823 | LMMCapletBlack | BGM (1997) | EXTRACTED |
| 824 | SwapRateFormula | Market Standard | EXTRACTED |
| 825 | SwaptionBlack | Market Standard | EXTRACTED |
| 826 | DiscountToZero | QuantLib | EXTRACTED |
| 827 | ForwardFromDiscount | QuantLib | EXTRACTED |
| 828 | InstantaneousForward | QuantLib | EXTRACTED |
| 829 | BootstrappingIteration | QuantLib | EXTRACTED |
| 830 | LogLinearInterp | QuantLib | EXTRACTED |
| 831 | CubicSplineZero | QuantLib | EXTRACTED |
| 832 | NelsonSiegel | Nelson-Siegel (1987) | EXTRACTED |
| 833 | Svensson | Svensson (1994) | EXTRACTED |
| 834 | OISDiscounting | Post-2008 | EXTRACTED |
| 835 | ForwardOISSpread | Post-2008 | EXTRACTED |
| 836 | SurvivalProbability | Credit | EXTRACTED |
| 837 | FlatHazardRate | Credit | EXTRACTED |
| 838 | CDSPremiumLeg | ISDA | EXTRACTED |
| 839 | CDSProtectionLeg | ISDA | EXTRACTED |
| 840 | CDSFairSpread | ISDA | EXTRACTED |
| 841 | CDSMTM | ISDA | EXTRACTED |
| 842 | ImpliedHazardRate | Credit | EXTRACTED |
| 843 | HazardBootstrap | Credit | EXTRACTED |
| 844 | UpfrontCDS | ISDA Big Bang | EXTRACTED |
| 845 | CDXIndex | JPMorgan | EXTRACTED |
| 846 | BarrierDownOutCall | Merton (1973) | EXTRACTED |
| 847 | BarrierUpOutPut | Merton (1973) | EXTRACTED |
| 848 | DoubleBarrier | Ikeda-Kunitomo (1992) | EXTRACTED |
| 849 | GeometricAsian | Kemna-Vorst (1990) | EXTRACTED |
| 850 | ArithmeticAsian | Turnbull-Wakeman (1991) | EXTRACTED |
| 851 | LookbackCallFloat | Goldman et al. (1979) | EXTRACTED |
| 852 | LookbackPutFloat | Goldman et al. (1979) | EXTRACTED |
| 853 | CliquetOption | Exotic | EXTRACTED |
| 854 | ChooserOption | Rubinstein (1991) | EXTRACTED |
| 855 | CompoundOption | Geske (1979) | EXTRACTED |
| 856 | Black76Swaption | Black (1976) | EXTRACTED |
| 857 | BachelierSwaption | Bachelier (1900) | EXTRACTED |
| 858 | NormalToLogVolConvert | Hagan (2002) | EXTRACTED |
| 859 | CapletBlack | Black (1976) | EXTRACTED |
| 860 | CapPrice | Market Standard | EXTRACTED |
| 861 | CapFloorParity | Arbitrage | EXTRACTED |
| 862 | SwaptionCashSettle | Market Convention | EXTRACTED |
| 863 | SABRVol | Hagan et al. (2002) | EXTRACTED |
| 864 | DisplacedDiffusion | Rubinstein (1983) | EXTRACTED |
| 865 | ShiftedSABR | Post-2012 | EXTRACTED |
| 866 | MonteCarloEstimator | Glasserman (2003) | EXTRACTED |
| 867 | AntitheticVariates | Glasserman (2003) | EXTRACTED |
| 868 | ControlVariate | Glasserman (2003) | EXTRACTED |
| 869 | ImportanceSampling | Glasserman (2003) | EXTRACTED |
| 870 | StratifiedSampling | Glasserman (2003) | EXTRACTED |
| 871 | QuasiMonteCarlo | Niederreiter (1992) | EXTRACTED |
| 872 | LongstaffSchwartz | LSM (2001) | EXTRACTED |
| 873 | EulerMaruyama | Kloeden-Platen (1992) | EXTRACTED |
| 874 | MilsteinScheme | Milstein (1974) | EXTRACTED |
| 875 | PathwiseGreeks | Glasserman (2003) | EXTRACTED |
| 876 | BlackScholesPDE | BS (1973) | EXTRACTED |
| 877 | ExplicitFD | Wilmott (2006) | EXTRACTED |
| 878 | ImplicitFD | Wilmott (2006) | EXTRACTED |
| 879 | CrankNicolson | Crank-Nicolson (1947) | EXTRACTED |
| 880 | CFLCondition | Stability | EXTRACTED |
| 881 | LogTransformPDE | Wilmott (2006) | EXTRACTED |
| 882 | AmericanBoundary | Free Boundary | EXTRACTED |
| 883 | PSORMethod | PSOR | EXTRACTED |
| 884 | ADIScheme | Peaceman-Rachford (1955) | EXTRACTED |
| 885 | DouglasRachford | Douglas-Rachford (1956) | EXTRACTED |
| 886 | RiskNeutralPricing | Harrison-Kreps (1979) | EXTRACTED |
| 887 | GirsanovTheorem | Girsanov (1960) | EXTRACTED |
| 888 | NumeraireChange | Geman et al. (1995) | EXTRACTED |
| 889 | ForwardMeasure | Musiela-Rutkowski (1997) | EXTRACTED |
| 890 | AnnuityMeasure | Interest Rates | EXTRACTED |
| 891 | CalibrationLSQ | QuantLib | EXTRACTED |
| 892 | ImpliedVolCalib | QuantLib | EXTRACTED |
| 893 | VegaWeightedCalib | QuantLib | EXTRACTED |
| 894 | TikhonovRegularization | Calibration | EXTRACTED |
| 895 | DupireLocalVol | Dupire (1994) | EXTRACTED |
| 896 | DupireForwardPDE | Dupire (1994) | EXTRACTED |
| 897 | SVISmile | Gatheral (2004) | EXTRACTED |
| 898 | CalendarArbitrage | Vol Surface | EXTRACTED |
| 899 | ButterflyArbitrage | Vol Surface | EXTRACTED |
| 900 | HestonFourierCalib | Heston (1993) | EXTRACTED |

---

## SUMMARY: IDs 811-900 COMPLETE

**Total Extracted in quantlib.md: 90 formulas**

Categories Covered:
- Short Rate Models (811-820)
- Market Models LMM/BGM (821-825)
- Yield Curve Mathematics (826-835)
- Credit Derivatives CDS (836-845)
- Exotic Options (846-855)
- Swaption/Cap Pricing (856-865)
- Monte Carlo Methods (866-875)
- Finite Difference PDE (876-885)
- Risk Neutral Theory (886-890)
- Calibration Methods (891-900)
