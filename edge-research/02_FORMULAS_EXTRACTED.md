# Formulas Extracted - Complete Library

## F001: Break-Even Win Rate with Fees

### Formula
```
WR_breakeven = (1 + F/AvgLoss) / (1 + AvgWin/AvgLoss + F/AvgLoss)
```

### Alternative Formula (Simpler)
```
WR_breakeven = Cost_per_Trade / (AvgWin - AvgLoss + Cost_per_Trade)
```

### Alternative Formula (Risk/Reward Based)
```
WR_breakeven = Risk / (Risk + Reward)
```

### Variables
- `WR_breakeven`: Minimum win rate to break even
- `F`: Fee per trade (0.0004 for us)
- `AvgWin`: Average winning trade size
- `AvgLoss`: Average losing trade size
- `Cost_per_Trade`: Total transaction cost per trade

### Sources
1. **MarketBulls Calculator** (Score: 5, Relevance: 10)
2. **The Balance Money** (Score: 5, Relevance: 10)
3. **Crypto Trading Fees Calculator** (Score: 5, Relevance: 10)

### Verification Count: 3
### Authority × Verification × Relevance: 5 × 3 × 10 = 150

### Implementation
```python
def breakeven_win_rate(avg_win, avg_loss, fee_per_trade):
    """
    Calculate minimum win rate needed to break even.

    Args:
        avg_win: Average winning trade profit (as decimal, e.g., 0.045 for 4.5%)
        avg_loss: Average losing trade loss (as decimal, e.g., 0.025 for 2.5%)
        fee_per_trade: Fee per trade (0.0004 for 0.04%)

    Returns:
        Minimum win rate to break even (as decimal)
    """
    cost_per_trade = 2 * fee_per_trade  # Entry + exit
    return cost_per_trade / (avg_win - avg_loss + cost_per_trade)
```

### Expected Impact
- **V1-V8**: This tells us IF we should even trade
- For TP=4.5%, SL=2.5%, Fee=0.04%: WR_breakeven = 0.08% / (0.045 - 0.025 + 0.08%) = 3.8%
- **Current Reality**: V1,V2,V5,V7,V8 have WR < 5%, so they're BELOW breakeven before fees

### Application to Our Problem
- **Critical**: We need WR > 3.8% just to not lose money
- V3 (26%), V4 (28%), V6 (27%) are ABOVE breakeven theoretically
- But they still died → something else is wrong (likely AvgWin < AvgLoss after fees)

---

## F002: Profit Factor

### Formula
```
Profit_Factor = (WR × AvgWin) / ((1 - WR) × AvgLoss)
```

### Alternative Formula (Gross Profit/Loss)
```
Profit_Factor = Total_Gross_Profit / Total_Gross_Loss
```

### Variables
- `Profit_Factor`: Ratio of total wins to total losses (must be > 1.0 to be profitable)
- `WR`: Win rate (as decimal)
- `AvgWin`: Average winning trade size
- `AvgLoss`: Average losing trade size

### Sources
1. **Medium - 10 Numbers Every Trader Should Know** (Score: 5, Relevance: 10)
2. **Electronic Trading Hub** (Score: 5, Relevance: 10)
3. **BacktestBase** (Score: 5, Relevance: 10)

### Verification Count: 3
### Authority × Verification × Relevance: 5 × 3 × 10 = 150

### Target Value
- **Minimum**: 1.0 (breakeven)
- **Target**: 2.0+ (profitable)
- **Excellent**: 3.0+

### Implementation
```python
def profit_factor(win_rate, avg_win, avg_loss):
    """
    Calculate profit factor.

    Returns:
        Profit factor (>1.0 is profitable, <1.0 is losing)
    """
    if win_rate == 0 or win_rate == 1:
        return 0.0
    return (win_rate * avg_win) / ((1 - win_rate) * avg_loss)
```

### Expected Impact
- **V3**: PF = (0.26 × AvgWin) / (0.74 × AvgLoss)
- For V3 to be profitable: AvgWin/AvgLoss > 2.85
- **This is our KEY METRIC**: We need to measure actual AvgWin and AvgLoss

---

## F003: Expectancy (Edge per Trade)

### Formula
```
Expectancy = (WR × AvgWin) - ((1 - WR) × AvgLoss) - Fees
```

### Alternative Name
- **Edge**
- **Expected Value per Trade**

### Variables
- `Expectancy`: Average profit/loss per trade
- `WR`: Win rate
- `AvgWin`: Average winning trade
- `AvgLoss`: Average losing trade
- `Fees`: Total fees per trade (entry + exit)

### Sources
1. **Medium - 10 Numbers Every Trader Should Know** (Score: 5, Relevance: 10)
2. **Electronic Trading Hub** (Score: 5, Relevance: 10)
3. **New Trader U** (Score: 5, Relevance: 10)

### Verification Count: 3
### Authority × Verification × Relevance: 5 × 3 × 10 = 150

### Target Value
- **Minimum**: > $0
- **Target**: > $0.02 per trade (per CLAUDE.md)

### Implementation
```python
def expectancy(win_rate, avg_win, avg_loss, fee_per_trade):
    """
    Calculate expected value per trade (edge).

    Returns:
        Edge in dollars per trade
    """
    fees = 2 * fee_per_trade  # Entry + exit
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss) - fees
```

### Expected Impact
- **THIS IS THE EDGE EQUATION FROM METHODOLOGY**
- Must be > 0 for profitability
- Must be > $0.02 for target performance

---

## F004: Kelly Criterion

### Formula
```
f* = (p × b - q) / b
```

### Where
- `f*`: Fraction of capital to bet (Kelly fraction)
- `p`: Probability of winning (win rate)
- `q`: Probability of losing (1 - win rate)
- `b`: Ratio of win to loss (AvgWin / AvgLoss)

### Alternative Formula (Trading Version)
```
f* = (WR × (AvgWin/AvgLoss) - (1 - WR)) / (AvgWin/AvgLoss)
```

### Sources
1. **Alpha Theory** (Score: 6, Relevance: 7)
2. **Medium - Nicolae Filip** (Score: 5, Relevance: 7)
3. **Wikipedia** (Score: 5, Relevance: 6)
4. **QuantPedia** (Score: 6, Relevance: 7)

### Verification Count: 4
### Authority × Verification × Relevance: 5.5 × 4 × 6.75 = 148.5

### Practical Considerations
- **Fractional Kelly**: Most traders use f*/2 or f*/4 to reduce volatility
- **Never use full Kelly**: Too aggressive, leads to massive drawdowns
- **Current Usage**: V1-V4,V6,V8 use Kelly=0.80 (reasonable)

### Implementation
```python
def kelly_criterion(win_rate, avg_win, avg_loss):
    """
    Calculate optimal Kelly fraction.

    Returns:
        Kelly fraction (fraction of capital to risk)
    """
    if avg_loss == 0:
        return 0.0

    b = avg_win / avg_loss
    p = win_rate
    q = 1 - win_rate

    f_star = (p * b - q) / b

    # Ensure non-negative
    return max(0.0, f_star)
```

### Expected Impact
- **V3**: f* = (0.26 × b - 0.74) / b
- For f* > 0 (positive edge): b > 2.85
- **This confirms F002**: We need AvgWin/AvgLoss > 2.85 for V3

---

## F005: Optimal F (Ralph Vince)

### Formula
```
Optimal_f = -1 / Largest_Loss
```

### Then Position Size
```
Position_Size = (Optimal_f × Capital) / Expected_Loss
```

### Sources
1. **QuantPedia** (Score: 6, Relevance: 7)
2. **QuantifiedStrategies** (Score: 5, Relevance: 7)
3. **Medium - Nicolae Filip** (Score: 5, Relevance: 6)

### Verification Count: 3
### Authority × Verification × Relevance: 5.33 × 3 × 6.67 = 106.6

### Difference from Kelly
- **Kelly**: Assumes fixed win/loss sizes
- **Optimal F**: Accounts for variable win/loss sizes
- **Better for our use**: Yes, because our wins/losses vary

### Expected Impact
- **Lower priority**: Kelly is simpler and works well
- **Use if**: We see highly variable win/loss sizes

---

## F006: Z-Score Entry/Exit Thresholds

### Formula
```
Z = (Price - Mean) / StdDev
```

### Entry Thresholds
- **Conservative**: ±2.5 to ±3.0 (fewer trades, higher probability)
- **Moderate**: ±2.0 (standard)
- **Aggressive**: ±1.5 to ±2.0 (more trades, lower probability)

### Exit Thresholds
- **Standard**: Z → 0 (mean reversion complete)
- **Our Current**: Z → 0.03 (almost at mean)

### Stop Loss
- **Standard**: ±3.0 (if entry at ±2.0)
- **Conservative**: ±3.5 to ±4.0

### Sources
1. **QuantStock** (Score: 5, Relevance: 10)
2. **FasterCapital** (Score: 5, Relevance: 9)
3. **StatOasis** (Score: 5, Relevance: 10)
4. **QuantInsti** (Score: 5, Relevance: 9)

### Verification Count: 4
### Authority × Verification × Relevance: 5 × 4 × 9.5 = 190

### Current vs Recommended
- **Current Entry**: z = 2.0 (reasonable)
- **Current Exit**: z = 0.03 (reasonable, close to mean)
- **Problem**: No stop-loss threshold defined!

### Implementation
```python
def z_score_thresholds(entry_conservative=False, stop_loss=True):
    """
    Return recommended z-score thresholds.

    Returns:
        dict with 'entry', 'exit', 'stop_loss'
    """
    if entry_conservative:
        entry = 2.5
        stop = 3.5
    else:
        entry = 2.0
        stop = 3.0

    return {
        'entry_long': -entry,
        'entry_short': entry,
        'exit': 0.0,
        'stop_loss_long': -stop if stop_loss else None,
        'stop_loss_short': stop if stop_loss else None
    }
```

### Expected Impact
- **HIGH PRIORITY**: Add stop-loss at z = ±3.0
- **V3,V4,V6**: Currently NO stop-loss on z-score
- **Expected**: Reduce AvgLoss, improve AvgWin/AvgLoss ratio

---

## F007: Ornstein-Uhlenbeck Process

### SDE (Stochastic Differential Equation)
```
dX_t = θ(μ - X_t)dt + σdB_t
```

### Where
- `θ` (theta): Mean reversion speed
- `μ` (mu): Long-term mean
- `σ` (sigma): Volatility
- `X_t`: Current price/spread
- `B_t`: Brownian motion

### Alternative Notation
```
dX_t = λ(m - X_t)dt + σdB_t
```
(Some sources use λ instead of θ, m instead of μ)

### Half-Life of Mean Reversion
```
Half_Life = ln(2) / θ
```

### Sources
1. **Hudson & Thames - OU Model** (Score: 6, Relevance: 10)
2. **Wikipedia** (Score: 5, Relevance: 8)
3. **QuantStart** (Score: 6, Relevance: 9)
4. **Lipton & López de Prado** (Score: 8, Relevance: 10)

### Verification Count: 4
### Authority × Verification × Relevance: 6.25 × 4 × 9.25 = 231.25

### Optimal Threshold Formula (Lipton & López de Prado)
- **Complex**: Involves heat potentials, value functions
- **Result**: Analytical formulas for optimal entry/exit
- **Maximizes**: Expected return and Sharpe ratio

### Implementation Priority
- **Medium**: Our current z-score approach is simpler
- **Use if**: We want to optimize thresholds mathematically
- **Benefit**: Closed-form solution for optimal levels

---

## F008: Half-Life of Mean Reversion

### Formula
```
Half_Life = ln(2) / λ
```

### Where
- `λ` (lambda): Mean reversion speed from AR(1) regression
- `ln(2)`: Natural log of 2 (≈ 0.693)

### AR(1) Regression to Get λ
```
y(t) - y(t-1) = α + β × y(t-1) + ε
```

Then:
```
λ = -ln(1 + β)
```

### Sources
1. **Hudson & Thames ArbitrageLab** (Score: 6, Relevance: 9)
2. **Flare9x Blog** (Score: 5, Relevance: 9)
3. **LetianzJ** (Score: 5, Relevance: 9)
4. **QuantInsti** (Score: 5, Relevance: 9)

### Verification Count: 4
### Authority × Verification × Relevance: 5.25 × 4 × 9 = 189

### Practical Use
- **Lookback Period**: Set equal to half-life
- **Our Current**: 500 lookback for mean/std
- **Recommendation**: Calculate half-life, adjust lookback

### Implementation
```python
import numpy as np
from statsmodels.regression.linear_model import OLS

def calculate_half_life(price_series):
    """
    Calculate half-life of mean reversion.

    Args:
        price_series: Pandas Series of prices

    Returns:
        Half-life in same units as price_series index (e.g., hours)
    """
    # AR(1) regression: y(t) - y(t-1) vs y(t-1)
    y = price_series.values
    y_lag = y[:-1]
    y_diff = y[1:] - y[:-1]

    # Regression
    model = OLS(y_diff, y_lag).fit()
    beta = model.params[0]

    # Calculate lambda and half-life
    lambda_mr = -np.log(1 + beta)
    half_life = np.log(2) / lambda_mr

    return half_life
```

### Expected Impact
- **Medium Priority**: Optimize lookback period
- **Expected**: Better mean/std estimation → better z-scores

---

## F009: Hurst Exponent

### Interpretation
- **H < 0.5**: Mean-reverting (anti-persistent)
- **H = 0.5**: Random walk (no memory)
- **H > 0.5**: Trending (persistent)

### Calculation (R/S Analysis)
```
1. Calculate differenced series
2. For each lag, calculate std dev
3. Slope of log(lags) vs log(std) = Hurst exponent
```

### Sources
1. **Macrosynergy** (Score: 5, Relevance: 8)
2. **Robot Wealth Part 1** (Score: 5, Relevance: 8)
3. **Robot Wealth Part 2** (Score: 5, Relevance: 8)
4. **QuantInsti** (Score: 5, Relevance: 8)
5. **Wikipedia** (Score: 5, Relevance: 7)

### Verification Count: 5
### Authority × Verification × Relevance: 5 × 5 × 7.8 = 195

### Use Case
- **Regime Detection**: Is market mean-reverting right now?
- **Filter**: Only trade when H < 0.5
- **Dynamic**: Adjust strategy based on H

### Implementation
```python
import numpy as np

def hurst_exponent(price_series, lags=range(2, 20)):
    """
    Calculate Hurst exponent using R/S analysis.

    Args:
        price_series: Price series
        lags: Range of lags to use

    Returns:
        Hurst exponent
    """
    tau = []
    lagvec = []

    # Step through different lags
    for lag in lags:
        # Calculate std of differenced series
        pp = np.array([price_series[i:i+lag] for i in range(len(price_series)-lag)])
        tau.append(np.std(np.sum(pp, axis=1)))
        lagvec.append(lag)

    # Fit slope
    m = np.polyfit(np.log(lagvec), np.log(tau), 1)
    hurst = m[0]

    return hurst
```

### Expected Impact
- **HIGH PRIORITY**: Filter trades by regime
- **V1,V2,V5,V7,V8 (0% WR)**: May be trading in trending regime (H > 0.5)
- **Expected**: Massive WR improvement by only trading when H < 0.5

---

## F010: ADF Test (Augmented Dickey-Fuller)

### Hypothesis
- **Null**: Unit root exists, series is non-stationary
- **Alternative**: No unit root, series is stationary (mean-reverting)

### P-Value Threshold
- **0.05**: Standard (95% confidence)
- **0.01**: Conservative (99% confidence)
- **0.10**: Liberal (90% confidence)

### Decision Rule
```
if p_value < 0.05:
    print("Series is mean-reverting (reject null)")
else:
    print("Series is NOT mean-reverting (fail to reject null)")
```

### Sources
1. **QuantInsti** (Score: 5, Relevance: 9)
2. **Machine Learning Plus** (Score: 5, Relevance: 9)
3. **QuantStart** (Score: 6, Relevance: 9)
4. **Statistics How To** (Score: 5, Relevance: 8)

### Verification Count: 4
### Authority × Verification × Relevance: 5.25 × 4 × 8.75 = 183.75

### Implementation
```python
from statsmodels.tsa.stattools import adfuller

def is_mean_reverting(price_series, significance=0.05):
    """
    Test if series is mean-reverting using ADF test.

    Args:
        price_series: Price series
        significance: P-value threshold (default 0.05)

    Returns:
        True if mean-reverting, False otherwise
    """
    result = adfuller(price_series)
    p_value = result[1]

    return p_value < significance
```

### Expected Impact
- **CRITICAL FILTER**: Don't trade if ADF p > 0.05
- **V1,V2,V5,V7,V8**: May be trading non-stationary series
- **Expected**: Eliminate losing trades in non-mean-reverting periods

---

## F011: Volatility-Adjusted Position Sizing (ATR)

### Formula
```
Position_Size = Account_Risk / (ATR × Multiple)
```

### Alternative
```
Position_Size = Risk_Amount / (ATR × Asset_Price)
```

### Where
- `Account_Risk`: Amount willing to risk per trade (e.g., 1% of capital)
- `ATR`: Average True Range
- `Multiple`: ATR multiplier for stop-loss (typically 2-3)

### Sources
1. **QuantifiedStrategies** (Score: 5, Relevance: 8)
2. **The Robust Trader** (Score: 5, Relevance: 8)
3. **LuxAlgo** (Score: 5, Relevance: 8)
4. **ThetaTrend** (Score: 5, Relevance: 8)

### Verification Count: 4
### Authority × Verification × Relevance: 5 × 4 × 8 = 160

### Implementation
```python
def position_size_atr(capital, risk_pct, atr, atr_multiple=2.5):
    """
    Calculate position size using ATR.

    Args:
        capital: Total capital
        risk_pct: Risk per trade (e.g., 0.01 for 1%)
        atr: Current ATR value
        atr_multiple: Stop-loss distance in ATR units

    Returns:
        Position size in dollars
    """
    account_risk = capital * risk_pct
    position_size = account_risk / (atr * atr_multiple)

    return position_size
```

### Expected Impact
- **Medium Priority**: Better position sizing
- **Benefit**: Normalize risk across different volatility regimes

---

## F012: Sharpe Ratio Optimization

### Formula
```
Sharpe = (Mean_Return - Risk_Free_Rate) / Std_Return
```

### For Mean Reversion
- **Optimize parameters** to maximize Sharpe
- **Parameters**: Entry threshold, exit threshold, stop-loss, lookback period

### Sources
1. **ResearchGate - Optimal Mean Reversion** (Score: 8, Relevance: 9)
2. **Teddy Koker** (Score: 5, Relevance: 8)
3. **Stanford - Sharpe** (Score: 8, Relevance: 6)

### Verification Count: 3
### Authority × Verification × Relevance: 7 × 3 × 7.67 = 161.07

### Expected Impact
- **Use for**: Parameter optimization
- **Method**: Grid search over parameters, select max Sharpe

---

## Summary Statistics

### Total Formulas Extracted: 12

### By Priority (Based on Score)
1. **F007**: OU Process (231.25)
2. **F009**: Hurst Exponent (195)
3. **F006**: Z-Score Thresholds (190)
4. **F008**: Half-Life (189)
5. **F010**: ADF Test (183.75)
6. **F012**: Sharpe Optimization (161.07)
7. **F011**: ATR Position Sizing (160)
8. **F001**: Break-Even WR (150)
9. **F002**: Profit Factor (150)
10. **F003**: Expectancy (150)
11. **F004**: Kelly Criterion (148.5)
12. **F005**: Optimal F (106.6)

### By Implementation Priority
1. **F009** (Hurst): Filter non-mean-reverting regimes
2. **F010** (ADF): Confirm stationarity before trading
3. **F006** (Z-Score): Add stop-loss thresholds
4. **F008** (Half-Life): Optimize lookback period
5. **F001** (Break-Even): Validate if we should trade
6. **F003** (Expectancy): Measure actual edge
7. **F002** (Profit Factor): Track performance metric
8. **F012** (Sharpe): Optimize parameters
9. **F011** (ATR): Volatility-adjusted sizing
10. **F007** (OU): Advanced threshold optimization
