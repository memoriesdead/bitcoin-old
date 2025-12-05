# Gap Formula Extractions
## IDs: 308-310, 427-430, 436-445, 454-460, 468-475

---

## SOURCE 1: Machine Learning Signals (308-310)
**Status:** EXTRACTED

### Formula 308: LSTM Hidden State (Trading)
```
h_t = o_t ⊙ tanh(c_t)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
```
**Edge:** Sequential price pattern memory

### Formula 309: Attention Score (Transformer)
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```
**Edge:** Focus on relevant market features

### Formula 310: Gradient Boosting Prediction
```
F_m(x) = F_{m-1}(x) + η × h_m(x)
h_m = argmin_h Σ L(y_i, F_{m-1}(x_i) + h(x_i))
```
**Edge:** Ensemble return prediction

---

## SOURCE 2: Signal Processing (427-430)
**Status:** EXTRACTED

### Formula 427: Kalman Filter State Update
```
x̂_t|t = x̂_t|t-1 + K_t(y_t - H_t x̂_t|t-1)
K_t = P_t|t-1 H_t^T (H_t P_t|t-1 H_t^T + R_t)^{-1}
```
**Edge:** Optimal price/volatility estimation

### Formula 428: Kalman Prediction Step
```
x̂_t|t-1 = F_t x̂_{t-1|t-1}
P_t|t-1 = F_t P_{t-1|t-1} F_t^T + Q_t
```
**Edge:** State propagation forward

### Formula 429: Particle Filter Weight Update
```
w_t^i ∝ w_{t-1}^i × p(y_t|x_t^i)
```
**Edge:** Non-linear/non-Gaussian filtering

### Formula 430: Wavelet Decomposition (Trading)
```
W_f(a,b) = (1/√a) ∫ f(t)ψ*((t-b)/a)dt
```
**Edge:** Multi-scale trend/noise separation

---

## SOURCE 3: HFT Microstructure (436-445)
**Status:** EXTRACTED

### Formula 436: Queue Position Value
```
V_queue = P(fill) × E[profit|fill] × position_size
P(fill) = f(queue_pos, order_flow, cancellation_rate)
```
**Edge:** Value of queue priority

### Formula 437: Adverse Selection Cost
```
AS_cost = E[|ΔP| × size | trade] = λ × σ × √(size/ADV)
```
**Edge:** Information leakage cost

### Formula 438: Maker-Taker Rebate Optimization
```
Net_Cost = spread/2 - rebate_maker (passive)
Net_Cost = spread/2 + fee_taker (aggressive)
```
**Edge:** Execution venue selection

### Formula 439: Toxic Flow Indicator
```
Toxicity = (Trades_against_move) / (Total_trades)
```
**Edge:** Informed trader detection

### Formula 440: Fill Probability Model
```
P(fill|queue_pos, time) = 1 - exp(-λ_fill × time × f(queue_pos))
```
**Edge:** Expected execution probability

### Formula 441: Latency Arbitrage Profit
```
π_latency = P(stale_quote) × E[price_move] × size - costs
```
**Edge:** Speed advantage monetization

### Formula 442: Co-location Value
```
V_colo = Σ_t [π_t(fast) - π_t(slow)] × P(opportunity_t)
```
**Edge:** Infrastructure investment ROI

### Formula 443: Message Rate Optimization
```
Optimal_rate = argmax[P(fill)×profit - message_cost×rate]
```
**Edge:** Order update frequency

### Formula 444: Inventory Skew (MM)
```
Skew = -γ × inventory × σ²
Mid_adjusted = Mid + Skew
```
**Edge:** Inventory-based quote adjustment

### Formula 445: Last Look Window
```
Reject_if: |ΔP| > threshold in window_ms
P(reject) = P(|ΔP| > threshold | trade_request)
```
**Edge:** LP protection mechanism

---

## SOURCE 4: Alternative Data (454-460)
**Status:** EXTRACTED

### Formula 454: Sentiment Score Aggregation
```
Sentiment = Σ_i w_i × s_i / Σ_i w_i
w_i = f(source_reliability, recency, volume)
```
**Edge:** Multi-source sentiment

### Formula 455: News Impact Decay
```
Impact_t = Impact_0 × exp(-λ × t)
Cumulative = Impact_0 × (1 - exp(-λ × T)) / λ
```
**Edge:** News alpha decay

### Formula 456: Satellite Imagery Signal
```
Signal = (Pixel_count_t - Pixel_count_{t-1}) / Pixel_count_{t-1}
```
**Edge:** Real activity proxy (parking lots, ships)

### Formula 457: Web Traffic Alpha
```
α = β × (Traffic_growth - E[Traffic_growth])
```
**Edge:** Consumer interest signal

### Formula 458: Credit Card Spend Signal
```
Revenue_estimate = Σ_merchant (transactions × avg_basket)
Surprise = (Estimate - Consensus) / σ_consensus
```
**Edge:** Real-time revenue proxy

### Formula 459: Job Posting Signal
```
Growth_signal = (Postings_t / Postings_{t-12}) - 1
```
**Edge:** Company expansion indicator

### Formula 460: Patent Filing Signal
```
Innovation_score = Σ (citations × novelty_factor) / R&D_spend
```
**Edge:** R&D productivity measure

---

## SOURCE 5: Order Flow Extended (468-475)
**Status:** EXTRACTED

### Formula 468: Permanent vs Temporary Impact
```
ΔP_permanent = γ × sign(Q) × |Q|^δ
ΔP_temporary = η × sign(Q) × |Q|^β
```
**Edge:** Impact decomposition

### Formula 469: Order Flow Imbalance Decay
```
OFI_decay_t = Σ_{k=0}^{t} λ^{t-k} × OFI_k
```
**Edge:** Weighted historical flow

### Formula 470: Trade Arrival Rate (Hawkes)
```
λ(t) = μ + Σ_{t_i<t} α × exp(-β(t-t_i))
```
**Edge:** Self-exciting trade intensity

### Formula 471: Branching Ratio
```
n* = α/β
```
**Edge:** Hawkes process criticality (n*<1 stable)

### Formula 472: Realized Kernel Variance
```
RK = Σ_{|h|≤H} k(h/H) × Σ_j r_{j} r_{j+h}
```
**Edge:** Noise-robust variance estimator

### Formula 473: Signature Plot
```
RV(Δ) = Σ_i r_i² where r_i = P_{iΔ} - P_{(i-1)Δ}
```
**Edge:** Microstructure noise detection

### Formula 474: Roll Spread Estimator
```
Spread = 2√(-Cov(ΔP_t, ΔP_{t-1}))
```
**Edge:** Implicit spread from serial covariance

### Formula 475: Hasbrouck Information Share
```
IS_i = σ_i² × θ_i² / Σ_j(σ_j² × θ_j²)
```
**Edge:** Price discovery contribution

---

## SUMMARY: 32 formulas extracted

| Range | Category | Count |
|-------|----------|-------|
| 308-310 | ML Signals | 3 |
| 427-430 | Signal Processing | 4 |
| 436-445 | HFT Microstructure | 10 |
| 454-460 | Alternative Data | 7 |
| 468-475 | Order Flow Extended | 8 |
