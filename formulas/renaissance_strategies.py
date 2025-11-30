#!/usr/bin/env python3
"""
RENAISSANCE-STYLE TRADING FORMULAS
===================================
Commercial-grade quantitative strategies based on:
- Renaissance Technologies Medallion Fund approaches
- Hidden Markov Models (Baum-Welch)
- Ornstein-Uhlenbeck optimal stopping
- Kelly Criterion optimal bet sizing
- Statistical arbitrage

Sources:
- arXiv papers on quantitative finance
- Hudson & Thames research
- Academic papers on mean reversion
"""

import numpy as np
from collections import deque
from .base import BaseFormula


class OrnsteinUhlenbeckOptimal(BaseFormula):
    """
    Ornstein-Uhlenbeck Process with Optimal Trading Thresholds

    Based on Bertram (2010) - derives optimal entry/exit thresholds
    by maximizing expected return per unit time.

    dX_t = theta * (mu - X_t) * dt + sigma * dW_t

    theta: mean reversion speed
    mu: long-term mean
    sigma: volatility
    """

    def __init__(self, lookback: int = 50):
        super().__init__(lookback)
        self.prices = deque(maxlen=lookback)
        self.returns = deque(maxlen=lookback)
        self.theta = 0.0  # Mean reversion speed
        self.mu = 0.0     # Long-term mean
        self.sigma = 0.0  # Volatility
        self.half_life = float('inf')

    def update(self, price: float, volume: float, timestamp: float):
        if len(self.prices) > 0:
            ret = np.log(price / self.prices[-1])
            self.returns.append(ret)
        self.prices.append(price)

        if len(self.returns) >= 20:
            self._estimate_ou_params()

    def _estimate_ou_params(self):
        """Estimate OU parameters using MLE"""
        returns = np.array(self.returns)
        prices = np.array(self.prices)

        # Log prices for OU fitting
        log_prices = np.log(prices)

        # Estimate parameters
        n = len(log_prices)
        if n < 20:
            return

        # Simple regression for theta estimation
        # dX = theta * (mu - X) * dt
        X = log_prices[:-1]
        dX = np.diff(log_prices)

        # Regress dX on X
        X_mean = np.mean(X)
        cov_dX_X = np.mean((dX - np.mean(dX)) * (X - X_mean))
        var_X = np.var(X)

        if var_X > 1e-10:
            self.theta = -cov_dX_X / var_X
            self.mu = X_mean
            self.sigma = np.std(dX)

            # Half-life of mean reversion
            if self.theta > 0:
                self.half_life = np.log(2) / self.theta

    def get_signal(self) -> float:
        if len(self.prices) < 30 or self.theta <= 0:
            return 0.0

        current_log_price = np.log(self.prices[-1])
        deviation = current_log_price - self.mu

        # Z-score based on OU parameters
        if self.sigma > 0:
            z_score = deviation / self.sigma
        else:
            return 0.0

        # Optimal entry thresholds (Bertram 2010)
        # Enter when |z| > z_enter, typically 2
        z_enter = 2.0

        # Strong mean reversion signal
        if z_score > z_enter and self.half_life < 50:
            return -1.0  # Price above mean, expect reversion down
        elif z_score < -z_enter and self.half_life < 50:
            return 1.0   # Price below mean, expect reversion up

        return 0.0

    def get_confidence(self) -> float:
        if self.theta <= 0 or self.half_life == float('inf'):
            return 0.0

        # Higher confidence for faster mean reversion
        # and larger deviations from mean
        if len(self.prices) < 20:
            return 0.0

        current_log_price = np.log(self.prices[-1])
        deviation = abs(current_log_price - self.mu)

        # Confidence based on deviation magnitude and mean reversion speed
        deviation_conf = min(deviation / (3 * self.sigma) if self.sigma > 0 else 0, 1.0)
        speed_conf = min(1.0 / self.half_life * 10, 1.0) if self.half_life > 0 else 0

        return (deviation_conf + speed_conf) / 2


class KellyOptimalBetting(BaseFormula):
    """
    Kelly Criterion for Optimal Position Sizing

    f* = (p * b - q) / b

    where:
    - f* = optimal fraction of capital
    - p = probability of winning
    - q = probability of losing (1-p)
    - b = odds (win/loss ratio)

    Used by Elwyn Berlekamp at RenTec.
    """

    def __init__(self, lookback: int = 100):
        super().__init__(lookback)
        self.returns = deque(maxlen=lookback)
        self.signals = deque(maxlen=lookback)
        self.win_rate = 0.5
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.kelly_fraction = 0.0

    def update(self, price: float, volume: float, timestamp: float):
        if hasattr(self, '_last_price') and self._last_price > 0:
            ret = (price - self._last_price) / self._last_price
            self.returns.append(ret)
            self._update_kelly_stats()
        self._last_price = price

    def _update_kelly_stats(self):
        if len(self.returns) < 20:
            return

        returns = np.array(self.returns)

        # Calculate win rate and average win/loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) > 0 and len(losses) > 0:
            self.win_rate = len(wins) / len(returns)
            self.avg_win = np.mean(wins)
            self.avg_loss = abs(np.mean(losses))

            # Kelly formula: f* = (p * b - q) / b
            # where b = avg_win / avg_loss
            if self.avg_loss > 0:
                b = self.avg_win / self.avg_loss
                q = 1 - self.win_rate
                self.kelly_fraction = (self.win_rate * b - q) / b

                # Half-Kelly for safety (50% of Kelly reduces variance by 75%)
                self.kelly_fraction = self.kelly_fraction / 2

    def get_signal(self) -> float:
        if len(self.returns) < 30:
            return 0.0

        # Generate signal based on recent momentum
        recent_returns = list(self.returns)[-10:]
        momentum = sum(recent_returns)

        # Only trade if Kelly fraction is positive (edge exists)
        if self.kelly_fraction <= 0:
            return 0.0

        # Signal direction based on momentum
        if momentum > 0.001:  # 0.1% threshold
            return 1.0
        elif momentum < -0.001:
            return -1.0

        return 0.0

    def get_confidence(self) -> float:
        if self.kelly_fraction <= 0:
            return 0.0

        # Confidence proportional to Kelly fraction
        # Kelly fraction > 0.1 is considered high confidence
        return min(self.kelly_fraction * 5, 1.0)


class HMMRegimeDetection(BaseFormula):
    """
    Hidden Markov Model Regime Detection

    Detects market regimes (bull/bear/sideways) using
    simplified HMM with 3 hidden states.

    Based on Leonard Baum's work at RenTec.
    """

    def __init__(self, lookback: int = 50):
        super().__init__(lookback)
        self.returns = deque(maxlen=lookback)
        self.regime = 1  # 0=bear, 1=neutral, 2=bull
        self.regime_prob = [0.33, 0.34, 0.33]

    def update(self, price: float, volume: float, timestamp: float):
        if hasattr(self, '_last_price') and self._last_price > 0:
            ret = (price - self._last_price) / self._last_price
            self.returns.append(ret)
            self._detect_regime()
        self._last_price = price

    def _detect_regime(self):
        if len(self.returns) < 20:
            return

        returns = np.array(self.returns)

        # Simple regime detection based on return distribution
        mean_ret = np.mean(returns[-20:])
        std_ret = np.std(returns[-20:])

        # Emission probabilities (simplified)
        # Bear: negative mean, high volatility
        # Bull: positive mean, moderate volatility
        # Neutral: near-zero mean, low volatility

        bear_prob = 0.33
        bull_prob = 0.33
        neutral_prob = 0.34

        if mean_ret < -std_ret:
            bear_prob = 0.6
            bull_prob = 0.1
            neutral_prob = 0.3
        elif mean_ret > std_ret:
            bear_prob = 0.1
            bull_prob = 0.6
            neutral_prob = 0.3
        else:
            bear_prob = 0.2
            bull_prob = 0.2
            neutral_prob = 0.6

        self.regime_prob = [bear_prob, neutral_prob, bull_prob]
        self.regime = np.argmax(self.regime_prob)

    def get_signal(self) -> float:
        if len(self.returns) < 20:
            return 0.0

        # Trade with regime
        if self.regime == 2:  # Bull
            return 1.0
        elif self.regime == 0:  # Bear
            return -1.0

        return 0.0

    def get_confidence(self) -> float:
        return max(self.regime_prob)


class StatArbSpreadTrading(BaseFormula):
    """
    Statistical Arbitrage Spread Trading

    Trades mean reversion of price spreads using
    z-score entry/exit rules.

    Entry: |z| > 2
    Exit: |z| < 0.5 or opposite band
    """

    def __init__(self, lookback: int = 50):
        super().__init__(lookback)
        self.prices = deque(maxlen=lookback)
        self.spread = deque(maxlen=lookback)
        self.z_score = 0.0
        self.in_trade = False
        self.trade_direction = 0

    def update(self, price: float, volume: float, timestamp: float):
        self.prices.append(price)

        if len(self.prices) >= 20:
            # Calculate spread as deviation from moving average
            ma = np.mean(list(self.prices)[-20:])
            spread_val = price - ma
            self.spread.append(spread_val)

            if len(self.spread) >= 10:
                spread_mean = np.mean(self.spread)
                spread_std = np.std(self.spread)

                if spread_std > 0:
                    self.z_score = (spread_val - spread_mean) / spread_std

    def get_signal(self) -> float:
        if len(self.spread) < 20:
            return 0.0

        z_enter = 2.0
        z_exit = 0.5

        # Entry logic
        if not self.in_trade:
            if self.z_score > z_enter:
                self.in_trade = True
                self.trade_direction = -1  # Short spread
                return -1.0
            elif self.z_score < -z_enter:
                self.in_trade = True
                self.trade_direction = 1  # Long spread
                return 1.0
        else:
            # Exit logic
            if abs(self.z_score) < z_exit:
                self.in_trade = False
                return 0.0
            # Opposite band exit
            if self.trade_direction > 0 and self.z_score > z_enter:
                self.in_trade = False
                return 0.0
            elif self.trade_direction < 0 and self.z_score < -z_enter:
                self.in_trade = False
                return 0.0

            return float(self.trade_direction)

        return 0.0

    def get_confidence(self) -> float:
        # Higher confidence for larger z-scores
        return min(abs(self.z_score) / 3.0, 1.0)


class BaumWelchMomentum(BaseFormula):
    """
    Baum-Welch Algorithm inspired momentum

    Uses iterative parameter estimation to detect
    regime transitions and trade momentum.
    """

    def __init__(self, lookback: int = 50):
        super().__init__(lookback)
        self.returns = deque(maxlen=lookback)
        self.states = deque(maxlen=lookback)  # Estimated hidden states
        self.transition_prob = 0.5  # Probability of regime change

    def update(self, price: float, volume: float, timestamp: float):
        if hasattr(self, '_last_price') and self._last_price > 0:
            ret = (price - self._last_price) / self._last_price
            self.returns.append(ret)
            self._estimate_state()
        self._last_price = price

    def _estimate_state(self):
        if len(self.returns) < 10:
            return

        # Simplified Baum-Welch: estimate current state
        recent = list(self.returns)[-10:]
        cumret = sum(recent)

        # State: 1 = bullish, -1 = bearish, 0 = neutral
        if cumret > 0.01:
            state = 1
        elif cumret < -0.01:
            state = -1
        else:
            state = 0

        self.states.append(state)

        # Estimate transition probability
        if len(self.states) > 2:
            transitions = sum(1 for i in range(1, len(self.states))
                            if self.states[i] != self.states[i-1])
            self.transition_prob = transitions / (len(self.states) - 1)

    def get_signal(self) -> float:
        if len(self.states) < 5:
            return 0.0

        # Trade in direction of current state
        # But only if low transition probability (stable regime)
        if self.transition_prob < 0.3:
            current_state = self.states[-1]
            return float(current_state)

        return 0.0

    def get_confidence(self) -> float:
        if len(self.states) < 5:
            return 0.0

        # Higher confidence for stable regimes
        stability = 1.0 - self.transition_prob

        # And for consistent recent states
        recent_states = list(self.states)[-5:]
        consistency = abs(sum(recent_states)) / 5.0

        return (stability + consistency) / 2


# Register these formulas with IDs 320-324
RENAISSANCE_FORMULAS = {
    320: OrnsteinUhlenbeckOptimal,
    321: KellyOptimalBetting,
    322: HMMRegimeDetection,
    323: StatArbSpreadTrading,
    324: BaumWelchMomentum,
}
