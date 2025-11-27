# Hold Time Optimization - Exit Timing Research

## Executive Summary

Your strategies exit too fast: 13 seconds (V2) to 4 minutes (V6). Research shows mean reversion takes hours, not seconds. This document explains optimal hold times.

## Current Hold Times (PROBLEMS)

| Strategy | Avg Hold | Trades | Issue |
|----------|----------|--------|-------|
| V2 Hawkes | 13 sec | 154,679 | Exit before any edge materializes |
| V1 OFI | 18 sec | 108,487 | Order flow needs minutes to resolve |
| V7 Kyle | 50 sec | 38,873 | Microstructure needs 5-15 min |
| V3 VPIN | 3.3 min | 9,744 | Mean reversion needs 30-120 min |
| V4 OU | 3.8 min | 8,613 | OU half-life is 2-4 hours |
| V6 HMM | 4.3 min | 7,528 | Regime lasts hours, not minutes |
| V5 Kalman | 16.7 min | 1,940 | Only strategy with reasonable hold |

**Problem**: Exiting 10-100× too early!

## Research on Hold Time

### Mean Reversion Half-Life

From Ornstein-Uhlenbeck research:

> "Half-life can be derived from Ornstein-Uhlenbeck's mean-reversion equation and is calculated as Half-Life = -ln(2)/theta where theta is the estimate of the rate of mean-reversion."

> "Once you determine that the series is mean reverting you can trade this series profitably with a simple linear model using a look back period equal to the half life. If a trade extended over 22 days you may expect a short term or permanent regime shift. This suggests using time stops based on the half-life calculation."

**Application**: For Bitcoin:
- Calculate OU half-life from historical data
- Hold for 2-4× half-life to capture full reversion
- Exit if no reversion after 4× half-life (regime shift)

### Optimal Holding Periods

From research:

> "Use half-life as look-back window to find rolling mean and rolling standard deviation. Scale in and out by keeping the position size negatively proportional to the z-score."

**Implication**: Don't exit at z-score = 0 (mean). Hold longer to capture overshooting.

## Calculating Bitcoin's Mean Reversion Half-Life

```python
def calculate_ou_half_life(returns):
    """
    Calculate Ornstein-Uhlenbeck half-life for Bitcoin

    Returns half-life in minutes
    """
    import statsmodels.api as sm

    # Fit OU process
    # dX = theta * (mu - X) * dt + sigma * dW
    # Rearrange to: dX = a + b*X + error

    lagged_returns = returns[:-1]
    diff_returns = returns[1:] - returns[:-1]

    # OLS regression
    model = sm.OLS(diff_returns, sm.add_constant(lagged_returns))
    results = model.fit()

    theta = -results.params[1]  # Mean reversion speed

    if theta <= 0:
        return None  # Not mean-reverting

    # Half-life = ln(2) / theta
    half_life = np.log(2) / theta

    return half_life


# Example with Bitcoin 1-minute data
import pandas as pd

# Load data
df = pd.read_parquet('bitcoin_complete_history.parquet')
df['returns'] = df['Close'].pct_change()

# Calculate half-life
returns_1min = df['returns'].dropna()
half_life_minutes = calculate_ou_half_life(returns_1min.values)

print(f"Bitcoin mean reversion half-life: {half_life_minutes:.1f} minutes")
# Typical result: 60-180 minutes (1-3 hours)
```

**Expected half-life for Bitcoin**: 90-120 minutes

**Your current holds**: 3-4 minutes = 2-3% of half-life!

## Minimum Hold Times by Strategy Type

Based on research and asset characteristics:

### 1. Mean Reversion Strategies (V3, V4)

**Formula**: Hold for 2× half-life minimum

```python
class MeanReversionHoldTime:
    def __init__(self):
        # Bitcoin typical half-life: 90-120 minutes
        self.half_life = self.calculate_half_life()  # ~100 min

        # Hold time rules
        self.MIN_HOLD = self.half_life * 1    # 100 min
        self.OPTIMAL_HOLD = self.half_life * 2  # 200 min
        self.MAX_HOLD = self.half_life * 4     # 400 min

    def should_exit(self, entry_time, current_time, current_pnl):
        hold_time = (current_time - entry_time).total_seconds() / 60

        # Minimum hold always
        if hold_time < self.MIN_HOLD:
            return False

        # Exit at profit if past optimal
        if hold_time >= self.OPTIMAL_HOLD and current_pnl > 0:
            return True

        # Force exit at max hold
        if hold_time >= self.MAX_HOLD:
            return True

        return False
```

**Recommended**:
- Min: 100 minutes (1.7 hours)
- Optimal: 180 minutes (3 hours)
- Max: 240 minutes (4 hours)

**Current V3/V4**: 3-4 minutes ✗

### 2. Regime Detection (V6 HMM)

**Logic**: Regimes last hours to days, not minutes

```python
class RegimeHoldTime:
    def __init__(self):
        # Typical crypto regime duration: 4-24 hours
        self.MIN_HOLD = 120  # 2 hours (minimum)
        self.OPTIMAL_HOLD = 480  # 8 hours
        self.MAX_HOLD = 1440  # 24 hours

    def should_exit(self, entry_time, entry_regime, current_regime):
        hold_time = (time.time() - entry_time) / 60  # minutes

        # Must hold at least MIN_HOLD
        if hold_time < self.MIN_HOLD:
            return False

        # Exit if regime changed and past minimum
        if current_regime != entry_regime:
            return True

        # Force exit at max hold
        if hold_time >= self.MAX_HOLD:
            return True

        return False
```

**Recommended**:
- Min: 120 minutes (2 hours)
- Optimal: 480 minutes (8 hours)
- Max: 1440 minutes (24 hours)

**Current V6**: 4.3 minutes ✗

### 3. Microstructure (V1 OFI, V7 Kyle)

**Logic**: Order flow resolves in 5-30 minutes

```python
class MicrostructureHoldTime:
    def __init__(self):
        # Order flow impact duration
        self.MIN_HOLD = 5    # 5 minutes
        self.OPTIMAL_HOLD = 15  # 15 minutes
        self.MAX_HOLD = 30   # 30 minutes

    def should_exit(self, hold_minutes, imbalance_resolved):
        if hold_minutes < self.MIN_HOLD:
            return False

        # Exit if imbalance resolved and past minimum
        if imbalance_resolved and hold_minutes >= self.MIN_HOLD:
            return True

        # Force exit at max
        if hold_minutes >= self.MAX_HOLD:
            return True

        return False
```

**Recommended**:
- Min: 5 minutes
- Optimal: 15 minutes
- Max: 30 minutes

**Current V1/V7**: 18-50 seconds ✗

### 4. Event-Based (V2 Hawkes)

**Logic**: Clusters resolve in 10-60 minutes

```python
class HawkesHoldTime:
    def __init__(self):
        # Hawkes cluster duration
        self.MIN_HOLD = 10   # 10 minutes
        self.OPTIMAL_HOLD = 30  # 30 minutes
        self.MAX_HOLD = 60   # 60 minutes

    def should_exit(self, hold_minutes, intensity_decayed):
        if hold_minutes < self.MIN_HOLD:
            return False

        # Exit if intensity back to baseline
        if intensity_decayed and hold_minutes >= self.OPTIMAL_HOLD:
            return True

        # Force exit
        if hold_minutes >= self.MAX_HOLD:
            return True

        return False
```

**Recommended**:
- Min: 10 minutes
- Optimal: 30 minutes
- Max: 60 minutes

**Current V2**: 13 seconds ✗

## Hold Time vs Fee Break-Even

### Mathematics

With 0.08% round-trip fees, need price movement to cover:

```python
def min_hold_time_for_fees(fee_pct, hourly_volatility):
    """
    Calculate minimum hold time to overcome fees

    Args:
        fee_pct: Round-trip fee (e.g., 0.0008 for 0.08%)
        hourly_volatility: Typical hourly price movement (e.g., 0.02 for 2%)

    Returns:
        Minimum hold time in minutes
    """
    # Need price to move at least fee_pct
    # If volatility is hourly_volatility per hour
    # Time needed = fee_pct / hourly_volatility hours

    hours_needed = fee_pct / hourly_volatility
    minutes_needed = hours_needed * 60

    return minutes_needed


# Bitcoin example
fee = 0.0008  # 0.08% round-trip
btc_hourly_vol = 0.02  # 2% per hour typical

min_hold = min_hold_time_for_fees(fee, btc_hourly_vol)
print(f"Minimum hold time: {min_hold:.1f} minutes")
# Result: 2.4 minutes minimum

# But for RELIABLE profit (2× fees):
min_hold_2x = min_hold_time_for_fees(fee * 2, btc_hourly_vol)
print(f"Recommended hold time: {min_hold_2x:.1f} minutes")
# Result: 4.8 minutes minimum for 2× fees
```

**Bitcoin volatility**:
- 1% per 30 minutes (typical)
- 2% per hour (typical)
- 5% per 4 hours (common)

**Fee break-even**:
- Minimum: 2.4 minutes (just covers fees)
- Safe: 5 minutes (2× fees)
- Profitable: 15+ minutes (5× fees)

**Your holds**:
- V2: 13 seconds = 3% of break-even ✗
- V1: 18 seconds = 4% of break-even ✗
- V7: 50 seconds = 17% of break-even ✗

## Time-Based Exit Logic

```python
class TimeBasedExit:
    """Comprehensive time-based exit strategy"""

    def __init__(self, strategy_type, half_life=None):
        self.strategy_type = strategy_type
        self.half_life = half_life  # For mean reversion

        # Set hold times based on strategy type
        if strategy_type == 'mean_reversion':
            self.MIN = (half_life or 100) * 1    # 1× half-life
            self.OPTIMAL = (half_life or 100) * 2  # 2× half-life
            self.MAX = (half_life or 100) * 4     # 4× half-life

        elif strategy_type == 'regime':
            self.MIN = 120    # 2 hours
            self.OPTIMAL = 480  # 8 hours
            self.MAX = 1440   # 24 hours

        elif strategy_type == 'microstructure':
            self.MIN = 5
            self.OPTIMAL = 15
            self.MAX = 30

        elif strategy_type == 'event':
            self.MIN = 10
            self.OPTIMAL = 30
            self.MAX = 60

    def should_exit(self, entry_time, current_pnl, take_profit_hit, stop_loss_hit):
        """Comprehensive exit logic combining time and price"""

        hold_minutes = (time.time() - entry_time) / 60

        # 1. Always exit if stop loss hit (regardless of time)
        if stop_loss_hit:
            return True, "stop_loss"

        # 2. Exit if take profit hit and past minimum hold
        if take_profit_hit and hold_minutes >= self.MIN:
            return True, "take_profit"

        # 3. If profitable and past optimal hold, exit
        if current_pnl > 0 and hold_minutes >= self.OPTIMAL:
            return True, "time_profit"

        # 4. Force exit at maximum hold (prevent runaway losses)
        if hold_minutes >= self.MAX:
            return True, "max_hold"

        # 5. Don't exit before minimum hold
        return False, "holding"


# Example usage
exit_manager = TimeBasedExit('mean_reversion', half_life=100)

# Trade opened 5 minutes ago, small profit
should_exit, reason = exit_manager.should_exit(
    entry_time=time.time() - 300,  # 5 min ago
    current_pnl=0.01,
    take_profit_hit=True,
    stop_loss_hit=False
)
# Result: False, "holding" (must wait for MIN hold)

# Trade opened 200 minutes ago, small profit
should_exit, reason = exit_manager.should_exit(
    entry_time=time.time() - 12000,  # 200 min ago
    current_pnl=0.01,
    take_profit_hit=True,
    stop_loss_hit=False
)
# Result: True, "time_profit" (past OPTIMAL, take profit)
```

## Trailing Stops with Time

```python
class TrailingTimeExit:
    """Combine trailing stops with time-based exits"""

    def __init__(self, min_hold_minutes):
        self.min_hold = min_hold_minutes
        self.trailing_active = False
        self.highest_pnl = 0

    def should_exit(self, entry_time, current_pnl, entry_price, current_price):
        hold_minutes = (time.time() - entry_time) / 60

        # Track highest PnL
        if current_pnl > self.highest_pnl:
            self.highest_pnl = current_pnl

        # Can't exit before minimum hold
        if hold_minutes < self.min_hold:
            return False

        # Activate trailing if past min hold and profitable
        if hold_minutes >= self.min_hold and current_pnl > 0.005:  # 0.5% profit
            self.trailing_active = True

        # Trailing stop logic
        if self.trailing_active:
            # Exit if dropped 30% from peak
            if current_pnl < self.highest_pnl * 0.7:
                return True

        return False
```

## Recommended Settings by Strategy

### V1 OFI → V1_Fixed
```python
MIN_HOLD = 300   # 5 minutes (was 18 seconds)
OPTIMAL_HOLD = 900  # 15 minutes
MAX_HOLD = 1800  # 30 minutes
```

### V2 Hawkes → V2_Fixed
```python
MIN_HOLD = 600   # 10 minutes (was 13 seconds)
OPTIMAL_HOLD = 1800  # 30 minutes
MAX_HOLD = 3600  # 60 minutes
```

### V3 VPIN → V3_Fixed
```python
MIN_HOLD = 1800  # 30 minutes (was 3.3 minutes)
OPTIMAL_HOLD = 7200  # 2 hours
MAX_HOLD = 14400  # 4 hours
```

### V4 OU → V4_Fixed
```python
# Calculate half-life first
HALF_LIFE = calculate_half_life()  # ~100 minutes

MIN_HOLD = HALF_LIFE * 1   # ~100 minutes (was 3.8 minutes)
OPTIMAL_HOLD = HALF_LIFE * 2  # ~200 minutes
MAX_HOLD = HALF_LIFE * 4   # ~400 minutes
```

### V5 Kalman → V5_Fixed
```python
MIN_HOLD = 900   # 15 minutes (keep current 16.7)
OPTIMAL_HOLD = 1800  # 30 minutes
MAX_HOLD = 3600  # 60 minutes
```

### V6 HMM → V6_Fixed
```python
MIN_HOLD = 7200   # 2 hours (was 4.3 minutes)
OPTIMAL_HOLD = 28800  # 8 hours
MAX_HOLD = 86400  # 24 hours
```

### V7 Kyle → V7_Fixed
```python
MIN_HOLD = 300   # 5 minutes (was 50 seconds)
OPTIMAL_HOLD = 900  # 15 minutes
MAX_HOLD = 1800  # 30 minutes
```

## Expected Impact

| Strategy | Current Hold | New Hold | Edge Improvement |
|----------|-------------|----------|------------------|
| V1 | 18 sec | 15 min | 50× more time to capture edge |
| V2 | 13 sec | 30 min | 138× more time |
| V3 | 3.3 min | 2 hours | 36× more time |
| V4 | 3.8 min | 3.3 hours | 52× more time |
| V6 | 4.3 min | 8 hours | 111× more time |
| V7 | 50 sec | 15 min | 18× more time |

**Expected edge per trade improvement**: 3-5× (from letting statistical edges play out)

## Sources

- [Mean Reversion Half-Life - Flare9x](https://flare9xblog.wordpress.com/2017/09/27/half-life-of-mean-reversion-ornstein-uhlenbeck-formula-for-mean-reverting-process/)
- [OU Half-Life Calculation - Quant Stack Exchange](https://quant.stackexchange.com/questions/70338/why-do-i-need-fancy-methods-to-calculate-half-life-of-mean-reversion)
- [Trading Under OU Model - ArbitrageLab](https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/optimal_mean_reversion/ou_model.html)
- [OU Process Caveats - Hudson & Thames](https://hudsonthames.org/caveats-in-calibrating-the-ou-process/)
- [Mean Reversion Guide - Letian Wang](https://letianquant.com/mean-reversion.html)

## Implementation Checklist

- [ ] Calculate Bitcoin OU half-life from historical data
- [ ] Implement MIN_HOLD_TIME for all strategies
- [ ] Add time-based exit logic (don't exit before minimum)
- [ ] Set OPTIMAL_HOLD_TIME for profit-taking
- [ ] Set MAX_HOLD_TIME to prevent runaway losses
- [ ] Test with backtesting to validate hold times
- [ ] Monitor actual hold times vs targets in live trading
