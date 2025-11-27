# Implementation Order - Priority Sequence

## Executive Summary

**Problem**: All V1-V8 strategies have NEGATIVE edge
- V1,V2,V5,V7,V8: 0-5% win rate (died immediately)
- V3,V4,V6: 26-28% win rate but still unprofitable

**Root Cause**: Trading in non-mean-reverting regimes + no stop-loss protection

**Solution**: Implement regime filters first, then optimize parameters

---

## Phase 1: CRITICAL FIXES (Stop the Bleeding)

### Priority 1A: Regime Detection - Hurst Exponent (F009)

**Why First**: V1,V2,V5,V7,V8 have 0-5% WR → likely trading in trending regime

**Implementation**:
```python
# Add to base_strategy.py
def is_mean_reverting_regime(self, prices, window=100):
    """Filter: Only trade if Hurst < 0.5"""
    if len(prices) < window:
        return False

    recent_prices = prices[-window:]
    hurst = self.calculate_hurst(recent_prices)

    # Only trade if mean-reverting (H < 0.5)
    # Use 0.45 threshold for safety margin
    return hurst < 0.45
```

**Expected Impact**:
- V1,V2,V5,V7,V8: WR 0% → 40%+ (eliminate trending regime trades)
- V3,V4,V6: WR 26-28% → 35-45% (filter marginal trades)

**Files to Modify**:
- `officialtesting/strategies/base_strategy.py`
- `officialtesting/formulas/signal_processing.py` (add Hurst calculation)

### Priority 1B: Stationarity Test - ADF (F010)

**Why Second**: Confirm mean reversion exists before trading

**Implementation**:
```python
from statsmodels.tsa.stattools import adfuller

def is_stationary(self, prices, significance=0.05):
    """Confirm stationarity with ADF test"""
    if len(prices) < 50:
        return False

    result = adfuller(prices)
    p_value = result[1]

    # Reject null (non-stationary) if p < 0.05
    return p_value < significance
```

**Expected Impact**:
- Combined with Hurst: WR boost 15-25%
- Sharpe ratio: +0.5 to +1.0

**Files to Modify**:
- `officialtesting/strategies/base_strategy.py`
- Add `statsmodels` to requirements

### Priority 1C: Stop-Loss at Z=3.0 (F006)

**Why Third**: V3,V4,V6 die slowly because no stop-loss

**Implementation**:
```python
# Current: Only exit on TP or opposite signal
# Add: Exit on stop-loss

def check_stop_loss(self, position, current_z):
    """Exit if z-score exceeds stop-loss threshold"""
    if position == 'long' and current_z < -3.0:
        return True  # Stop out long
    elif position == 'short' and current_z > 3.0:
        return True  # Stop out short
    return False
```

**Expected Impact**:
- V3,V4,V6: Reduce AvgLoss by 30-50%
- Profit Factor: 0.5 → 1.5+

**Files to Modify**:
- `officialtesting/strategies/base_strategy.py`
- `officialtesting/core/config.py` (add STOP_LOSS_Z config)

---

## Phase 2: OPTIMIZATION (Improve Edge)

### Priority 2A: Half-Life Lookback (F008)

**Why**: Current 500-bar lookback may be suboptimal

**Implementation**:
```python
def calculate_optimal_lookback(self, prices):
    """Set lookback = half-life of mean reversion"""
    half_life = self.calculate_half_life(prices[-1000:])  # Use 1000 bars to estimate

    # Bound between 100 and 1000
    lookback = int(np.clip(half_life, 100, 1000))

    return lookback
```

**Expected Impact**:
- Better mean/std estimation → better z-scores
- WR improvement: +2-5%

**Files to Modify**:
- `officialtesting/formulas/mean_reversion.py`

### Priority 2B: Break-Even Analysis (F001)

**Why**: Validate if parameters can ever be profitable

**Implementation**:
```python
def validate_parameters(self, take_profit, stop_loss, fee=0.0004):
    """Check if parameters have positive expectancy"""
    # Calculate break-even win rate
    cost_per_trade = 2 * fee  # Entry + exit
    avg_win = take_profit
    avg_loss = stop_loss

    breakeven_wr = cost_per_trade / (avg_win - avg_loss + cost_per_trade)

    print(f"Break-even WR: {breakeven_wr:.1%}")
    print(f"TP={take_profit:.1%}, SL={stop_loss:.1%}, Fee={fee:.2%}")

    # For 55% target WR, what R:R do we need?
    # At WR=0.55: AvgWin/AvgLoss > (1-0.55)/(0.55) = 0.82
    required_rr = (1 - 0.55) / 0.55
    print(f"Required R:R at 55% WR: {required_rr:.2f}")

    # Current R:R
    current_rr = avg_win / avg_loss
    print(f"Current R:R: {current_rr:.2f}")

    return current_rr > required_rr
```

**Expected Impact**:
- Identify unprofitable parameter combinations
- Guide V1-V8 parameter selection

**Files to Modify**:
- `officialtesting/utils/helpers.py`

### Priority 2C: Dynamic Z-Score Thresholds (F006)

**Why**: Fixed thresholds may be suboptimal

**Implementation**:
```python
def get_dynamic_thresholds(self, recent_returns, volatility_regime):
    """Adjust thresholds based on market conditions"""

    # Base thresholds
    entry_z = 2.0
    exit_z = 0.0
    stop_z = 3.0

    # Increase in high volatility (wider bands)
    if volatility_regime == 'high':
        entry_z = 2.5
        stop_z = 3.5

    # Decrease in low volatility (tighter bands)
    elif volatility_regime == 'low':
        entry_z = 1.5
        stop_z = 2.5

    return entry_z, exit_z, stop_z
```

**Expected Impact**:
- WR improvement: +3-7%
- Sharpe improvement: +0.2 to +0.5

**Files to Modify**:
- `officialtesting/strategies/base_strategy.py`
- `officialtesting/formulas/volatility.py`

---

## Phase 3: ADVANCED (Maximize Performance)

### Priority 3A: OU Process Optimal Thresholds (F007)

**Why**: Mathematical optimization of entry/exit levels

**Implementation**:
```python
# Complex - requires solving optimal stopping problem
# Use simplified version from Lipton & López de Prado

def optimal_ou_thresholds(self, theta, mu, sigma, transaction_cost):
    """
    Calculate optimal entry/exit for OU process

    Args:
        theta: Mean reversion speed
        mu: Long-term mean
        sigma: Volatility
        transaction_cost: Cost per trade

    Returns:
        (entry_upper, entry_lower, exit_level)
    """
    # Simplified approximation
    # Full solution requires numerical optimization

    # Scale thresholds by volatility and cost
    cost_scaled = transaction_cost / sigma

    entry_distance = sigma * np.sqrt(2 * np.log(1 / cost_scaled))
    exit_level = mu

    return (mu + entry_distance, mu - entry_distance, exit_level)
```

**Expected Impact**:
- Sharpe improvement: +0.3 to +0.7
- Edge improvement: +10-20%

**Files to Modify**:
- `officialtesting/formulas/mean_reversion.py`

### Priority 3B: Sharpe Ratio Optimization (F012)

**Why**: Systematically find best parameters

**Implementation**:
```python
from scipy.optimize import differential_evolution

def optimize_parameters(self, price_data):
    """
    Find parameters that maximize Sharpe ratio

    Parameters to optimize:
    - entry_z
    - exit_z
    - stop_z
    - lookback
    - hurst_threshold
    """

    def objective(params):
        entry_z, exit_z, stop_z, lookback, hurst_thresh = params

        # Run backtest with these parameters
        results = self.backtest(
            price_data,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            lookback=int(lookback),
            hurst_threshold=hurst_thresh
        )

        # Return negative Sharpe (minimize negative = maximize positive)
        return -results['sharpe_ratio']

    # Bounds
    bounds = [
        (1.5, 3.0),   # entry_z
        (-0.5, 0.5),  # exit_z
        (2.5, 4.0),   # stop_z
        (100, 1000),  # lookback
        (0.40, 0.50)  # hurst_threshold
    ]

    # Optimize
    result = differential_evolution(objective, bounds, maxiter=100)

    return result.x
```

**Expected Impact**:
- Find globally optimal parameters
- Sharpe: +0.5 to +1.5
- WR: +5-15%

**Files to Modify**:
- `officialtesting/utils/optimizer.py` (new file)

### Priority 3C: ATR Position Sizing (F011)

**Why**: Normalize risk across volatility regimes

**Implementation**:
```python
def calculate_position_size_atr(self, capital, atr, risk_pct=0.02):
    """
    Volatility-adjusted position sizing

    Args:
        capital: Current capital
        atr: Current ATR
        risk_pct: Risk per trade (2% default)

    Returns:
        Position size multiplier
    """
    # Risk amount in dollars
    risk_amount = capital * risk_pct

    # Position size based on ATR
    # Assume stop-loss at 2.5× ATR
    stop_distance = 2.5 * atr

    position_size = risk_amount / stop_distance

    return position_size
```

**Expected Impact**:
- Reduce drawdowns in high volatility
- Sharpe improvement: +0.2 to +0.4

**Files to Modify**:
- `officialtesting/formulas/position_sizing.py`
- `officialtesting/strategies/base_strategy.py`

---

## Implementation Roadmap

### Week 1: Phase 1 (CRITICAL)
- ✅ Day 1-2: Implement Hurst filter (F009)
- ✅ Day 3-4: Implement ADF test (F010)
- ✅ Day 5-7: Add stop-loss at z=3.0 (F006)
- **Test**: Run all V1-V8 with Phase 1 changes
- **Expected**: V1,V2,V5,V7,V8 now have WR > 30%

### Week 2: Phase 2 (OPTIMIZATION)
- ✅ Day 1-2: Half-life lookback (F008)
- ✅ Day 3-4: Break-even validation (F001)
- ✅ Day 5-7: Dynamic z-thresholds (F006)
- **Test**: Measure improvement in edge/trade
- **Expected**: All versions have positive edge

### Week 3: Phase 3 (ADVANCED)
- ✅ Day 1-3: OU thresholds (F007)
- ✅ Day 4-6: Sharpe optimization (F012)
- ✅ Day 7: ATR sizing (F011)
- **Test**: Final parameter sweep
- **Expected**: Sharpe > 1.5, WR > 55%

---

## Success Metrics by Phase

### Phase 1 Success (After Week 1)
- [ ] All versions WR > 25%
- [ ] V1,V2,V5,V7,V8 WR > 30% (up from 0-5%)
- [ ] V3,V4,V6 WR > 35% (up from 26-28%)
- [ ] No versions die completely ($0 capital)
- [ ] Edge > 0 for at least 3 versions

### Phase 2 Success (After Week 2)
- [ ] All versions WR > 40%
- [ ] Edge > $0.01 for all versions
- [ ] Profit Factor > 1.0 for all versions
- [ ] Sharpe > 0.5 for all versions

### Phase 3 Success (After Week 3)
- [ ] At least one version WR > 55%
- [ ] At least one version Edge > $0.02
- [ ] At least one version Profit Factor > 2.0
- [ ] At least one version Sharpe > 1.5
- [ ] Best version: $10 → $300,000+ target

---

## Files to Create/Modify Summary

### New Files
```
officialtesting/formulas/regime_detection.py  # Hurst, ADF
officialtesting/utils/optimizer.py            # Sharpe optimization
```

### Modified Files
```
officialtesting/strategies/base_strategy.py
officialtesting/formulas/mean_reversion.py
officialtesting/formulas/position_sizing.py
officialtesting/formulas/volatility.py
officialtesting/formulas/signal_processing.py
officialtesting/core/config.py
officialtesting/utils/helpers.py
```

### Dependencies to Add
```
statsmodels  # For ADF test
scipy        # For optimization
```

---

## Risk Mitigation

### Phase 1 Risks
- **Risk**: Hurst filter too restrictive, no trades
- **Mitigation**: Use threshold 0.45 instead of 0.5

- **Risk**: ADF test fails in short windows
- **Mitigation**: Minimum 50 bars before testing

### Phase 2 Risks
- **Risk**: Half-life calculation unstable
- **Mitigation**: Bound lookback between 100-1000

- **Risk**: Dynamic thresholds over-optimize
- **Mitigation**: Use only 2 regimes (high/low vol)

### Phase 3 Risks
- **Risk**: OU optimization too complex
- **Mitigation**: Use simplified approximation first

- **Risk**: Sharpe optimization overfits
- **Mitigation**: Use walk-forward validation

---

## Next Steps

1. ✅ Review this implementation order
2. ✅ Approve Phase 1 priorities
3. ✅ Begin implementation starting with Hurst filter
4. ✅ Test after each phase
5. ✅ Iterate based on results

**Target**: Complete all 3 phases, achieve Edge > $0.02, WR > 55%, $10 → $300,000
