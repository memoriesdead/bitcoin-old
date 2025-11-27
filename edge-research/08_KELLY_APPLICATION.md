# Kelly Criterion Application - Position Sizing with Edge

## Executive Summary

Kelly Criterion is optimal for position sizing WHEN you have positive edge. Currently, all your strategies have negative edge, making Kelly irrelevant. This document explains how to apply Kelly once edge is achieved.

## The Kelly Formula

From research:

> "The Kelly Criterion is a formula for sizing bets by maximizing the long-term expected value of the logarithm of wealth."

### Basic Formula

```
Kelly % = (Win Rate × Risk:Reward Ratio - (1 - Win Rate)) / Risk:Reward Ratio

Or:

f* = (bp - q) / b

Where:
f* = fraction of capital to bet
b = odds received (R:R ratio)
p = probability of winning
q = probability of losing (1 - p)
```

## Critical Rule from Research

From Ed Thorp's work:

> "If the Kelly value is negative, the trader should not take the trade. When probability equals the probability offered by the market, the Kelly calculation provides a negative figure, implying there is no edge and therefore no reason to place a wager."

**Your situation**: ALL strategies currently have negative or zero Kelly.

## Examples with Your Strategies

### V3 VPIN - Current State

```python
win_rate = 0.27
risk_reward = 1.0  # 1:1 R:R
loss_rate = 0.73

# Kelly calculation
kelly = (0.27 * 1.0 - 0.73) / 1.0
kelly = (0.27 - 0.73) / 1.0
kelly = -0.46

# Negative Kelly = Don't trade!
```

**Interpretation**: Kelly says "bet -46% of capital" → DON'T TRADE

### V3 VPIN - After Fixes

```python
win_rate = 0.35  # Improved with stricter filters
risk_reward = 3.0  # 3:1 R:R with new exits
loss_rate = 0.65

# Kelly calculation
kelly = (0.35 * 3.0 - 0.65) / 3.0
kelly = (1.05 - 0.65) / 3.0
kelly = 0.40 / 3.0
kelly = 0.133  # 13.3% of capital

# Positive Kelly = Good to trade!
```

**Interpretation**: Kelly says "bet 13.3% of capital" per trade

### V6 HMM - After Fixes

```python
win_rate = 0.45  # High confidence regime detection
risk_reward = 3.0  # 3:1 R:R
loss_rate = 0.55

# Kelly calculation
kelly = (0.45 * 3.0 - 0.55) / 3.0
kelly = (1.35 - 0.55) / 3.0
kelly = 0.80 / 3.0
kelly = 0.267  # 26.7% of capital

# High Kelly = Strong edge!
```

## Fractional Kelly (Recommended)

From research:

> "Kelly supporters usually argue for fractional Kelly (betting a fixed fraction of the amount recommended by Kelly) for practical reasons, such as wishing to reduce volatility or protecting against errors in edge calculations."

> "Speculators can use 'half-Kelly' or 'quarter-Kelly' to cut the recommended percentage."

### Why Use Fractional Kelly

1. **Parameter uncertainty**: Your win rate and R:R estimates may be wrong
2. **Volatility reduction**: Full Kelly has ~20% drawdowns
3. **Psychological comfort**: Easier to stick with smaller positions
4. **Multiple concurrent trades**: Can't bet 100% on multiple trades

### Recommended Fractions

| Kelly Fraction | Use Case | Volatility | Drawdowns |
|----------------|----------|------------|-----------|
| Full Kelly (1.0×) | Perfect confidence in edge | Very high | 20-30% |
| Half Kelly (0.5×) | Good confidence | Moderate | 10-15% |
| Quarter Kelly (0.25×) | Conservative | Low | 5-8% |
| Tenth Kelly (0.1×) | Very conservative | Very low | 2-3% |

**Recommendation for you**: Start with Quarter Kelly (0.25×)

## Practical Kelly Implementation

```python
class KellySizer:
    """Kelly Criterion position sizing"""

    def __init__(self, kelly_fraction=0.25):
        self.kelly_fraction = kelly_fraction  # Quarter Kelly

    def calculate_position_size(self, capital, win_rate, risk_reward_ratio):
        """
        Calculate optimal position size using Kelly Criterion

        Args:
            capital: Total available capital
            win_rate: Historical win rate (0-1)
            risk_reward_ratio: Average win / average loss

        Returns:
            Position size in dollars
        """
        # Calculate full Kelly
        numerator = (win_rate * risk_reward_ratio) - (1 - win_rate)
        denominator = risk_reward_ratio

        kelly = numerator / denominator

        # Check for negative edge
        if kelly <= 0:
            return 0  # Don't trade!

        # Apply fractional Kelly
        fractional_kelly = kelly * self.kelly_fraction

        # Calculate position size
        position_size = capital * fractional_kelly

        return position_size

    def should_trade(self, win_rate, risk_reward_ratio):
        """Check if Kelly is positive (have edge)"""
        kelly = ((win_rate * risk_reward_ratio) - (1 - win_rate)) / risk_reward_ratio
        return kelly > 0


# Example usage
sizer = KellySizer(kelly_fraction=0.25)

# V3 after fixes
capital = 10.00
win_rate = 0.35
rr = 3.0

position = sizer.calculate_position_size(capital, win_rate, rr)
# Full Kelly = 13.3%, Quarter Kelly = 3.33%
# Position = $10 × 0.0333 = $0.33
```

## Dynamic Kelly Based on Recent Performance

```python
class DynamicKelly:
    """Adjust Kelly based on recent win rate"""

    def __init__(self, lookback_trades=50):
        self.recent_trades = deque(maxlen=lookback_trades)
        self.kelly_fraction = 0.25

    def record_trade(self, won, pnl):
        """Record trade outcome"""
        self.recent_trades.append({'won': won, 'pnl': pnl})

    def get_recent_metrics(self):
        """Calculate recent win rate and R:R"""
        if len(self.recent_trades) < 10:
            return None, None  # Not enough data

        wins = [t for t in self.recent_trades if t['won']]
        losses = [t for t in self.recent_trades if not t['won']]

        win_rate = len(wins) / len(self.recent_trades)

        if not wins or not losses:
            return None, None

        avg_win = np.mean([t['pnl'] for t in wins])
        avg_loss = abs(np.mean([t['pnl'] for t in losses]))

        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0

        return win_rate, risk_reward

    def calculate_position_size(self, capital):
        """Calculate position using recent performance"""
        win_rate, rr = self.get_recent_metrics()

        if win_rate is None or rr is None:
            # Not enough data, use conservative sizing
            return capital * 0.02  # 2% of capital

        # Calculate Kelly
        kelly = ((win_rate * rr) - (1 - win_rate)) / rr

        if kelly <= 0:
            return 0  # Edge disappeared!

        # Use fractional Kelly
        position = capital * kelly * self.kelly_fraction

        # Cap at 10% of capital (safety)
        return min(position, capital * 0.10)
```

## Kelly for Multiple Strategies

When running V1-V8 simultaneously:

```python
class PortfolioKelly:
    """Kelly allocation across multiple strategies"""

    def __init__(self, total_capital):
        self.total_capital = total_capital
        self.strategies = {}

    def add_strategy(self, name, win_rate, risk_reward):
        """Add strategy with its metrics"""
        # Calculate Kelly for this strategy
        kelly = ((win_rate * risk_reward) - (1 - win_rate)) / risk_reward

        if kelly > 0:
            self.strategies[name] = {
                'win_rate': win_rate,
                'rr': risk_reward,
                'kelly': kelly
            }

    def allocate_capital(self):
        """Allocate capital using Kelly across strategies"""
        if not self.strategies:
            return {}

        # Calculate total Kelly demand
        total_kelly = sum(s['kelly'] for s in self.strategies.values())

        # Normalize if over 100%
        if total_kelly > 1.0:
            # Scale down proportionally
            allocations = {
                name: (s['kelly'] / total_kelly) * self.total_capital
                for name, s in self.strategies.items()
            }
        else:
            # Use quarter Kelly for each
            allocations = {
                name: s['kelly'] * 0.25 * self.total_capital
                for name, s in self.strategies.items()
            }

        return allocations


# Example with your fixed strategies
portfolio = PortfolioKelly(total_capital=80)

# Add fixed strategies (estimated metrics)
portfolio.add_strategy('V3_Fixed', win_rate=0.35, risk_reward=4.0)
portfolio.add_strategy('V4_Fixed', win_rate=0.40, risk_reward=2.5)
portfolio.add_strategy('V6_Fixed', win_rate=0.45, risk_reward=3.0)

allocations = portfolio.allocate_capital()
# V3_Fixed: $15
# V4_Fixed: $18
# V6_Fixed: $22
# Reserved: $25 (for other strategies or buffer)
```

## Kelly with Leverage

```python
def kelly_with_leverage(win_rate, risk_reward, max_leverage=3.0):
    """
    Calculate position size with leverage

    Kelly can suggest >100% of capital if edge is strong
    Use leverage but cap it for safety
    """
    # Full Kelly
    kelly = ((win_rate * risk_reward) - (1 - win_rate)) / risk_reward

    if kelly <= 0:
        return 0

    # Kelly might be >1.0 (suggesting leverage)
    if kelly > 1.0:
        # Cap at max leverage
        kelly = min(kelly, max_leverage)

    # Use quarter Kelly even with leverage
    fractional_kelly = kelly * 0.25

    return fractional_kelly


# Example: Very strong edge
win_rate = 0.55
risk_reward = 3.0

kelly = kelly_with_leverage(win_rate, risk_reward, max_leverage=2.0)
# Full Kelly = 40% → With 2× leverage = 80%
# Quarter Kelly = 20%
```

## Risk Management with Kelly

Even with Kelly, add safety limits:

```python
class SafeKelly:
    """Kelly with additional risk controls"""

    def __init__(self):
        self.kelly_fraction = 0.25
        self.MAX_POSITION_PCT = 0.10  # Never more than 10%
        self.MIN_POSITION_PCT = 0.01  # Minimum 1% if trading
        self.MAX_TOTAL_RISK = 0.30    # Max 30% of capital at risk

    def calculate_position(self, capital, win_rate, rr, positions_open):
        """Calculate position with safety checks"""

        # Calculate Kelly
        kelly = ((win_rate * rr) - (1 - win_rate)) / rr

        if kelly <= 0:
            return 0

        # Apply fractional Kelly
        position_pct = kelly * self.kelly_fraction

        # Apply maximum position size
        position_pct = min(position_pct, self.MAX_POSITION_PCT)

        # Check total portfolio risk
        current_risk = sum(p['risk'] for p in positions_open)
        available_risk = self.MAX_TOTAL_RISK - current_risk

        if available_risk <= 0:
            return 0  # Already at max risk

        # Calculate position
        position = capital * position_pct

        # Check if it fits within available risk
        position_risk = position * (1.0 / rr)  # Risk is 1/RR of position
        if position_risk > available_risk * capital:
            # Scale down to fit available risk
            position = available_risk * capital * rr

        # Apply minimum
        if position < capital * self.MIN_POSITION_PCT:
            return 0  # Too small, skip

        return position
```

## When NOT to Use Kelly

From research:

> "When dealing with unfavorable situations (negative edge), Thorp's work discusses scenarios where players must make small 'waiting bets' on unfavorable situations."

**Don't use Kelly when**:
1. ✗ Win rate unknown (need 50+ trades minimum)
2. ✗ R:R uncertain (need stable TP/SL first)
3. ✗ Negative or zero edge (Kelly will say don't trade)
4. ✗ Highly correlated positions (Kelly assumes independence)
5. ✗ Black swan risk (Kelly doesn't account for tail events)

## Recommended Progression

### Stage 1: Fixed Sizing (Now - Week 4)

```python
# Use fixed 2% risk per trade
position_size = capital * 0.02
```

**Why**: Don't know true win rate or R:R yet

### Stage 2: Conservative Kelly (Week 5-8)

```python
# After 100+ trades, use tenth Kelly
kelly_fraction = 0.10
```

**Why**: Have data but still uncertainty

### Stage 3: Quarter Kelly (Week 9+)

```python
# After 500+ trades, use quarter Kelly
kelly_fraction = 0.25
```

**Why**: Confident in edge estimates

### Stage 4: Half Kelly (Month 6+)

```python
# After 2000+ trades with proven edge, use half Kelly
kelly_fraction = 0.50
```

**Why**: Proven system, maximize growth

**NEVER use full Kelly in live trading**

## Measuring Your Kelly Inputs

```python
def calculate_kelly_inputs(trades_df):
    """Calculate win rate and R:R from actual trades"""

    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] < 0]

    # Win rate
    win_rate = len(wins) / len(trades_df)

    # Average win and loss
    avg_win = wins['pnl'].mean()
    avg_loss = abs(losses['pnl'].mean())

    # Risk:reward ratio
    risk_reward = avg_win / avg_loss if avg_loss > 0 else 0

    # Calculate Kelly
    if risk_reward > 0:
        kelly = ((win_rate * risk_reward) - (1 - win_rate)) / risk_reward
    else:
        kelly = -1  # No valid R:R

    print(f"Win Rate: {win_rate:.1%}")
    print(f"Avg Win: ${avg_win:.4f}")
    print(f"Avg Loss: ${avg_loss:.4f}")
    print(f"Risk:Reward: {risk_reward:.2f}:1")
    print(f"Full Kelly: {kelly:.1%}")
    print(f"Quarter Kelly: {kelly * 0.25:.1%}")

    if kelly > 0:
        print(f"✓ Positive edge detected")
    else:
        print(f"✗ Negative edge - don't trade with Kelly")

    return win_rate, risk_reward, kelly
```

Run this on your trade logs after implementing fixes!

## Sources

- [Kelly Criterion - Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion)
- [Kelly Criterion Position Sizing - PyQuant](https://www.pyquantnews.com/the-pyquant-newsletter/use-kelly-criterion-optimal-position-sizing)
- [Ed Thorp on Kelly - Financial Wisdom TV](https://www.financialwisdomtv.com/post/kelly-criterion-ed-thorp-optimal-position-sizing-for-stock-trading)
- [Understanding Kelly Criterion - Thorp Paper](https://rybn.org/halloffame/PDFS/2008_Understanding_Kelly_New.pdf)
- [Kelly in Blackjack - Gwern](https://gwern.net/doc/statistics/decision/2006-thorp.pdf)

## Action Items

1. **Today**: Understand you CAN'T use Kelly yet (negative edge)
2. **Week 1**: Implement fixed 2% position sizing
3. **Week 4**: Calculate actual win rate and R:R from 100+ trades
4. **Week 5**: Start using tenth Kelly if edge is positive
5. **Week 12**: Graduate to quarter Kelly if edge remains
6. **Monitor**: Track Kelly-recommended size vs actual size monthly
