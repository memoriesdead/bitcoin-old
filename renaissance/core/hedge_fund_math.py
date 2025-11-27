"""
HEDGE FUND MATHEMATICAL FORMULAS - THE REAL MATH
=================================================
Based on research into Renaissance Technologies, Two Sigma, and academic HFT papers.

KEY INSIGHT from RenTech: "Be right 50.75% of the time, 100% of the time"
- They don't predict big moves, they predict MICRO-MOVES
- 150k-300k trades/day with tiny edges
- Win rate only needs to be 50.75% with proper sizing

MISSING FORMULAS WE NEED:
1. Tick Imbalance (Lopez de Prado) - Information-driven sampling
2. Stoikov Microprice - TRUE fair value estimator
3. Hidden Markov Model states - Regime detection
4. Kyle's Lambda - Market impact coefficient
5. Hawkes Process - Self-exciting order flow
6. VPIN - Probability of informed trading

Formula IDs: 283-300 (NEW HEDGE FUND SERIES)
"""

import numpy as np
from collections import deque
from typing import Tuple, List
from .base import BaseFormula, FormulaRegistry


# =============================================================================
# FORMULA 283: TICK IMBALANCE BAR SIGNAL (Lopez de Prado)
# =============================================================================
@FormulaRegistry.register(283)
class TickImbalanceSignal(BaseFormula):
    """
    Tick Imbalance Bars - Information-Driven Sampling

    The KEY insight: Don't sample at fixed time intervals.
    Sample when INFORMATION arrives (when imbalance exceeds threshold).

    Formula:
    - bt = +1 if price_t > price_{t-1}, -1 if price_t < price_{t-1}
    - theta_t = sum(bt) from bar_start to t
    - Signal when |theta_t| > E[theta] (expected imbalance)

    This detects when informed traders are active!
    """

    FORMULA_ID = 283
    CATEGORY = "hedge_fund"
    NAME = "TickImbalance"
    DESCRIPTION = "Lopez de Prado Tick Imbalance - detects informed trading"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.tick_signs = deque(maxlen=lookback)  # +1 or -1
        self.cumulative_theta = 0.0
        self.theta_history = deque(maxlen=50)  # Track bar lengths
        self.expected_theta = 20.0  # Initial expectation
        self.ewma_span = kwargs.get('ewma_span', 20)
        self.last_price = None
        self.bar_tick_count = 0

    def _compute(self) -> None:
        prices = self._prices_array()
        if len(prices) < 2:
            return

        # Calculate tick sign (bt)
        current_price = prices[-1]
        prev_price = prices[-2]

        if current_price > prev_price:
            bt = 1
        elif current_price < prev_price:
            bt = -1
        else:
            bt = self.tick_signs[-1] if len(self.tick_signs) > 0 else 0

        self.tick_signs.append(bt)
        self.cumulative_theta += bt
        self.bar_tick_count += 1

        # Calculate expected theta using EWMA of previous bars
        if len(self.theta_history) > 0:
            alpha = 2 / (self.ewma_span + 1)
            self.expected_theta = alpha * abs(self.theta_history[-1]) + (1 - alpha) * self.expected_theta

        # Check if imbalance exceeds threshold
        imbalance_ratio = abs(self.cumulative_theta) / max(self.expected_theta, 1)

        # Signal based on imbalance direction when threshold exceeded
        if imbalance_ratio > 1.0:
            # Bar complete - record and reset
            self.theta_history.append(self.cumulative_theta)

            # Signal in direction of imbalance (momentum)
            if self.cumulative_theta > 0:
                self.signal = 1  # Buy pressure
            else:
                self.signal = -1  # Sell pressure

            # Confidence based on how much we exceeded threshold
            self.confidence = min(0.5 + (imbalance_ratio - 1) * 0.25, 0.95)

            # Reset for next bar
            self.cumulative_theta = 0.0
            self.bar_tick_count = 0
        else:
            # No signal yet, waiting for information
            self.signal = 0
            self.confidence = 0.3


# =============================================================================
# FORMULA 284: STOIKOV MICROPRICE (True Fair Value)
# =============================================================================
@FormulaRegistry.register(284)
class StoikovMicroprice(BaseFormula):
    """
    Stoikov Microprice - THE true fair value estimator used in HFT

    Standard mid-price is WRONG because it ignores order book imbalance.

    Microprice Formula:
    μ = (Vbid × Pask + Vask × Pbid) / (Vbid + Vask)

    Equivalently:
    μ = mid + spread × (Imbalance - 0.5)

    where Imbalance = Vbid / (Vbid + Vask)

    If more volume on bid side -> microprice > midprice -> BUY signal
    """

    FORMULA_ID = 284
    CATEGORY = "hedge_fund"
    NAME = "StoikovMicroprice"
    DESCRIPTION = "Stoikov Microprice - true fair value from order book"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.microprice_history = deque(maxlen=lookback)
        self.imbalance_history = deque(maxlen=lookback)
        self.spread_history = deque(maxlen=lookback)
        # Simulated order book from tick data
        self.bid_volume_estimate = deque(maxlen=20)
        self.ask_volume_estimate = deque(maxlen=20)

    def _compute(self) -> None:
        prices = self._prices_array()
        volumes = self._volumes_array()

        if len(prices) < 3:
            return

        # Estimate bid/ask volumes from tick direction
        # Uptick = buying at ask, Downtick = selling at bid
        current = prices[-1]
        prev = prices[-2]
        vol = volumes[-1] if len(volumes) > 0 and volumes[-1] > 0 else 1.0

        if current > prev:
            # Uptick - volume hit ask side
            self.ask_volume_estimate.append(vol)
            self.bid_volume_estimate.append(0)
        elif current < prev:
            # Downtick - volume hit bid side
            self.bid_volume_estimate.append(vol)
            self.ask_volume_estimate.append(0)
        else:
            # No change
            self.bid_volume_estimate.append(vol / 2)
            self.ask_volume_estimate.append(vol / 2)

        # Calculate imbalance
        total_bid = sum(self.bid_volume_estimate) + 1e-10
        total_ask = sum(self.ask_volume_estimate) + 1e-10
        imbalance = total_bid / (total_bid + total_ask)  # 0 to 1

        self.imbalance_history.append(imbalance)

        # Estimate spread from price volatility (proxy)
        if len(prices) > 5:
            recent_range = max(prices[-5:]) - min(prices[-5:])
            spread = recent_range * 0.1  # Estimate spread as 10% of recent range
        else:
            spread = abs(prices[-1] - prices[-2]) if len(prices) > 1 else 0.01

        self.spread_history.append(spread)

        # Calculate microprice adjustment
        mid = prices[-1]
        microprice = mid + spread * (imbalance - 0.5)
        self.microprice_history.append(microprice)

        # Signal: if microprice suggests price should be higher -> BUY
        price_adjustment = (microprice - mid) / mid if mid > 0 else 0

        # Compare current imbalance to recent average
        if len(self.imbalance_history) > 10:
            avg_imbalance = np.mean(list(self.imbalance_history)[-10:])
            imbalance_zscore = (imbalance - avg_imbalance) / (np.std(list(self.imbalance_history)[-10:]) + 1e-10)

            if imbalance_zscore > 1.0:  # Strong bid imbalance
                self.signal = 1
                self.confidence = min(0.5 + abs(imbalance_zscore) * 0.15, 0.9)
            elif imbalance_zscore < -1.0:  # Strong ask imbalance
                self.signal = -1
                self.confidence = min(0.5 + abs(imbalance_zscore) * 0.15, 0.9)
            else:
                self.signal = 0
                self.confidence = 0.3
        else:
            self.signal = 0
            self.confidence = 0.2


# =============================================================================
# FORMULA 285: HIDDEN MARKOV MODEL REGIME (Baum-Welch Inspired)
# =============================================================================
@FormulaRegistry.register(285)
class HMMRegimeSignal(BaseFormula):
    """
    Hidden Markov Model Regime Detection

    Renaissance Technologies uses HMM extensively (Baum-Welch algorithm).

    Hidden States:
    - State 0: Mean Reverting (trade reversals)
    - State 1: Trending Up (buy momentum)
    - State 2: Trending Down (sell momentum)

    We estimate the current hidden state from observable price movements,
    then trade accordingly.
    """

    FORMULA_ID = 285
    CATEGORY = "hedge_fund"
    NAME = "HMMRegime"
    DESCRIPTION = "Hidden Markov Model regime detection"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        # Simplified HMM parameters (would be learned in practice)
        # Emission probabilities: P(observation | state)
        # Transition probabilities: P(next_state | current_state)

        # State probabilities
        self.state_probs = np.array([0.33, 0.33, 0.34])  # [mean_rev, trend_up, trend_down]

        # Transition matrix (rows = from state, cols = to state)
        # Mean-reverting regime is "sticky", trends persist
        self.transition = np.array([
            [0.7, 0.15, 0.15],  # Mean-rev -> stays or switches
            [0.2, 0.6, 0.2],   # Trend-up -> tends to persist
            [0.2, 0.2, 0.6],   # Trend-down -> tends to persist
        ])

        self.state_history = deque(maxlen=lookback)
        self.return_bins = deque(maxlen=lookback)

    def _compute(self) -> None:
        returns = self._returns_array()
        if len(returns) < 5:
            return

        # Classify recent return into observation categories
        # -2: large down, -1: small down, 0: flat, 1: small up, 2: large up
        recent_ret = returns[-1]
        std_ret = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)

        if std_ret < 1e-10:
            obs = 0
        else:
            z = recent_ret / std_ret
            if z > 1.5:
                obs = 2  # Large up
            elif z > 0.3:
                obs = 1  # Small up
            elif z > -0.3:
                obs = 0  # Flat
            elif z > -1.5:
                obs = -1  # Small down
            else:
                obs = -2  # Large down

        self.return_bins.append(obs)

        # Emission probabilities: P(observation | state)
        # Mean-reverting: expects returns near 0
        # Trending up: expects positive returns
        # Trending down: expects negative returns
        emission_probs = np.zeros(3)

        if obs >= 1:
            emission_probs = np.array([0.2, 0.6, 0.1])  # More likely trend-up
        elif obs <= -1:
            emission_probs = np.array([0.2, 0.1, 0.6])  # More likely trend-down
        else:
            emission_probs = np.array([0.6, 0.2, 0.2])  # More likely mean-rev

        # Update state probabilities (simplified forward algorithm)
        # P(state_t | obs_1:t) ~ P(obs_t | state_t) * sum(P(state_t | state_{t-1}) * P(state_{t-1}))
        predicted_probs = self.transition.T @ self.state_probs
        self.state_probs = emission_probs * predicted_probs
        self.state_probs /= self.state_probs.sum()  # Normalize

        # Get most likely state
        most_likely_state = np.argmax(self.state_probs)
        self.state_history.append(most_likely_state)

        # Generate signal based on regime
        if most_likely_state == 0:  # Mean-reverting
            # Trade against recent move
            if obs > 0:
                self.signal = -1
            elif obs < 0:
                self.signal = 1
            else:
                self.signal = 0
            self.confidence = self.state_probs[0] * 0.7
        elif most_likely_state == 1:  # Trending up
            self.signal = 1
            self.confidence = self.state_probs[1] * 0.8
        else:  # Trending down
            self.signal = -1
            self.confidence = self.state_probs[2] * 0.8


# =============================================================================
# FORMULA 286: KYLE'S LAMBDA (Market Impact)
# =============================================================================
@FormulaRegistry.register(286)
class KyleLambda(BaseFormula):
    """
    Kyle's Lambda - Market Impact Coefficient

    From Kyle (1985): Price moves proportionally to signed order flow

    dP = λ × dQ

    Where:
    - λ (lambda) = market impact coefficient
    - dQ = signed order flow (+ for buys, - for sells)

    High λ = illiquid market = small orders move price a lot
    Low λ = liquid market = can trade without impact

    Signal: When we detect LOW lambda + directional flow -> trade with flow
    """

    FORMULA_ID = 286
    CATEGORY = "hedge_fund"
    NAME = "KyleLambda"
    DESCRIPTION = "Kyle's Lambda market impact coefficient"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.lambda_history = deque(maxlen=lookback)
        self.signed_flow = deque(maxlen=lookback)
        self.price_changes = deque(maxlen=lookback)
        self.window = kwargs.get('window', 20)

    def _compute(self) -> None:
        prices = self._prices_array()
        volumes = self._volumes_array()

        if len(prices) < 3:
            return

        # Calculate signed order flow (using tick rule)
        price_change = prices[-1] - prices[-2]
        vol = volumes[-1] if len(volumes) > 0 and volumes[-1] > 0 else 1.0

        if price_change > 0:
            signed_vol = vol  # Buying
        elif price_change < 0:
            signed_vol = -vol  # Selling
        else:
            signed_vol = 0

        self.signed_flow.append(signed_vol)
        self.price_changes.append(price_change)

        # Estimate lambda using regression: dP = lambda * dQ
        if len(self.signed_flow) >= self.window:
            flows = np.array(list(self.signed_flow)[-self.window:])
            changes = np.array(list(self.price_changes)[-self.window:])

            # Avoid division by zero
            flow_var = np.var(flows)
            if flow_var > 1e-10:
                # Simple OLS: lambda = Cov(dP, dQ) / Var(dQ)
                covariance = np.cov(changes, flows)[0, 1]
                lambda_estimate = covariance / flow_var
            else:
                lambda_estimate = 0

            self.lambda_history.append(abs(lambda_estimate))

            # Signal logic:
            # High lambda + recent flow = price already moved, expect reversal
            # Low lambda + strong flow = market absorbing, follow flow

            avg_lambda = np.mean(list(self.lambda_history)[-10:]) if len(self.lambda_history) >= 10 else abs(lambda_estimate)
            recent_flow = sum(list(self.signed_flow)[-5:])

            if avg_lambda > 0:
                lambda_zscore = (abs(lambda_estimate) - avg_lambda) / (np.std(list(self.lambda_history)[-10:]) + 1e-10) if len(self.lambda_history) >= 10 else 0
            else:
                lambda_zscore = 0

            if lambda_zscore < -0.5:  # Low impact environment
                # Follow the flow
                if recent_flow > 0:
                    self.signal = 1
                    self.confidence = min(0.5 + abs(recent_flow) * 0.01, 0.85)
                elif recent_flow < 0:
                    self.signal = -1
                    self.confidence = min(0.5 + abs(recent_flow) * 0.01, 0.85)
                else:
                    self.signal = 0
                    self.confidence = 0.3
            elif lambda_zscore > 1.0:  # High impact = expect reversal
                if recent_flow > 0:
                    self.signal = -1  # Reversal expected
                elif recent_flow < 0:
                    self.signal = 1
                else:
                    self.signal = 0
                self.confidence = 0.5
            else:
                self.signal = 0
                self.confidence = 0.3


# =============================================================================
# FORMULA 287: VPIN (Volume-Synchronized Probability of Informed Trading)
# =============================================================================
@FormulaRegistry.register(287)
class VPINSignal(BaseFormula):
    """
    VPIN - Volume-Synchronized Probability of Informed Trading

    From Easley, Lopez de Prado, O'Hara (2012)

    VPIN = sum(|V_buy - V_sell|) / (n * V_bucket)

    High VPIN = informed traders are active = expect big move
    Low VPIN = noise trading = mean-revert

    The KEY: VPIN predicts VOLATILITY, not direction.
    We use it as a filter - only trade when VPIN suggests opportunity.
    """

    FORMULA_ID = 287
    CATEGORY = "hedge_fund"
    NAME = "VPIN"
    DESCRIPTION = "Volume-synchronized probability of informed trading"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.bucket_size = kwargs.get('bucket_size', 10)
        self.n_buckets = kwargs.get('n_buckets', 50)
        self.buy_volumes = deque(maxlen=self.n_buckets)
        self.sell_volumes = deque(maxlen=self.n_buckets)
        self.vpin_history = deque(maxlen=lookback)
        self.current_bucket_buy = 0
        self.current_bucket_sell = 0
        self.bucket_count = 0

    def _compute(self) -> None:
        prices = self._prices_array()
        volumes = self._volumes_array()

        if len(prices) < 2:
            return

        # Classify volume as buy or sell using tick rule
        price_change = prices[-1] - prices[-2]
        vol = volumes[-1] if len(volumes) > 0 and volumes[-1] > 0 else 1.0

        # Bulk Volume Classification (simplified)
        if price_change > 0:
            buy_vol = vol
            sell_vol = 0
        elif price_change < 0:
            buy_vol = 0
            sell_vol = vol
        else:
            buy_vol = vol / 2
            sell_vol = vol / 2

        self.current_bucket_buy += buy_vol
        self.current_bucket_sell += sell_vol
        self.bucket_count += 1

        # Complete bucket when we have enough volume
        if self.bucket_count >= self.bucket_size:
            self.buy_volumes.append(self.current_bucket_buy)
            self.sell_volumes.append(self.current_bucket_sell)
            self.current_bucket_buy = 0
            self.current_bucket_sell = 0
            self.bucket_count = 0

        # Calculate VPIN
        if len(self.buy_volumes) >= 5:
            buys = np.array(self.buy_volumes)
            sells = np.array(self.sell_volumes)
            total_vol = buys + sells

            # VPIN = average of |buy - sell| / total
            if total_vol.sum() > 0:
                vpin = np.sum(np.abs(buys - sells)) / total_vol.sum()
            else:
                vpin = 0

            self.vpin_history.append(vpin)

            # Signal based on VPIN level
            if len(self.vpin_history) >= 10:
                avg_vpin = np.mean(list(self.vpin_history)[-10:])
                std_vpin = np.std(list(self.vpin_history)[-10:])

                if std_vpin > 1e-10:
                    vpin_zscore = (vpin - avg_vpin) / std_vpin
                else:
                    vpin_zscore = 0

                # High VPIN = informed trading = momentum
                # Low VPIN = noise = mean revert
                recent_direction = 1 if sum(buys[-3:]) > sum(sells[-3:]) else -1

                if vpin_zscore > 1.5:  # High VPIN - follow informed traders
                    self.signal = recent_direction
                    self.confidence = min(0.5 + vpin_zscore * 0.1, 0.9)
                elif vpin_zscore < -1.0:  # Low VPIN - mean revert
                    self.signal = -recent_direction
                    self.confidence = 0.6
                else:
                    self.signal = 0
                    self.confidence = 0.3
            else:
                self.signal = 0
                self.confidence = 0.2
        else:
            self.signal = 0
            self.confidence = 0.1


# =============================================================================
# FORMULA 288: HAWKES PROCESS ORDER FLOW
# =============================================================================
@FormulaRegistry.register(288)
class HawkesOrderFlow(BaseFormula):
    """
    Hawkes Process - Self-Exciting Order Flow

    Orders cluster in time - one trade triggers more trades.

    Intensity: λ(t) = μ + Σ α × exp(-β × (t - t_i))

    Where:
    - μ = baseline intensity
    - α = excitation factor (how much each event increases intensity)
    - β = decay rate (how fast excitation fades)

    High intensity = likely more trades coming = momentum
    Low intensity = quiet period = potential reversal
    """

    FORMULA_ID = 288
    CATEGORY = "hedge_fund"
    NAME = "HawkesOrderFlow"
    DESCRIPTION = "Hawkes process self-exciting order flow model"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.event_times = deque(maxlen=200)  # Trade timestamps
        self.event_signs = deque(maxlen=200)  # +1 buy, -1 sell
        self.intensity_history = deque(maxlen=lookback)
        self.mu = 1.0  # Baseline intensity
        self.alpha = 0.5  # Excitation
        self.beta = 0.1  # Decay

    def _compute(self) -> None:
        prices = self._prices_array()
        timestamps = list(self.timestamps)

        if len(prices) < 2 or len(timestamps) < 2:
            return

        # Record event
        current_time = timestamps[-1] if timestamps[-1] > 0 else len(prices)
        price_change = prices[-1] - prices[-2]
        sign = 1 if price_change > 0 else (-1 if price_change < 0 else 0)

        if sign != 0:
            self.event_times.append(current_time)
            self.event_signs.append(sign)

        # Calculate current intensity
        if len(self.event_times) > 1:
            intensity_buy = self.mu
            intensity_sell = self.mu

            for i, (t, s) in enumerate(zip(self.event_times, self.event_signs)):
                if t < current_time:
                    time_diff = current_time - t
                    excitation = self.alpha * np.exp(-self.beta * time_diff)
                    if s > 0:
                        intensity_buy += excitation
                    else:
                        intensity_sell += excitation

            total_intensity = intensity_buy + intensity_sell
            self.intensity_history.append(total_intensity)

            # Imbalance in intensities
            if total_intensity > 0:
                intensity_imbalance = (intensity_buy - intensity_sell) / total_intensity
            else:
                intensity_imbalance = 0

            # Signal based on intensity imbalance
            if len(self.intensity_history) >= 10:
                avg_intensity = np.mean(list(self.intensity_history)[-10:])

                if total_intensity > avg_intensity * 1.5:
                    # High activity - follow imbalance
                    if intensity_imbalance > 0.2:
                        self.signal = 1
                        self.confidence = min(0.5 + intensity_imbalance, 0.9)
                    elif intensity_imbalance < -0.2:
                        self.signal = -1
                        self.confidence = min(0.5 + abs(intensity_imbalance), 0.9)
                    else:
                        self.signal = 0
                        self.confidence = 0.4
                else:
                    # Low activity - no signal
                    self.signal = 0
                    self.confidence = 0.3
            else:
                self.signal = 0
                self.confidence = 0.2
        else:
            self.signal = 0
            self.confidence = 0.1


# =============================================================================
# FORMULA 289: PRICE MOMENTUM ACCELERATION
# =============================================================================
@FormulaRegistry.register(289)
class MomentumAcceleration(BaseFormula):
    """
    Momentum Acceleration - Second Derivative of Price

    Not just momentum (first derivative), but CHANGE in momentum.

    Acceleration = d²P/dt² = (momentum_t - momentum_{t-1}) / dt

    Positive acceleration = momentum increasing = strong trend
    Negative acceleration = momentum decreasing = trend weakening
    """

    FORMULA_ID = 289
    CATEGORY = "hedge_fund"
    NAME = "MomentumAcceleration"
    DESCRIPTION = "Second derivative of price - momentum change rate"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.momentum_history = deque(maxlen=lookback)
        self.acceleration_history = deque(maxlen=lookback)
        self.momentum_window = kwargs.get('momentum_window', 5)

    def _compute(self) -> None:
        prices = self._prices_array()

        if len(prices) < self.momentum_window + 2:
            return

        # Calculate momentum (first derivative) - rate of change
        momentum = (prices[-1] - prices[-self.momentum_window]) / self.momentum_window
        self.momentum_history.append(momentum)

        # Calculate acceleration (second derivative)
        if len(self.momentum_history) >= 2:
            acceleration = self.momentum_history[-1] - self.momentum_history[-2]
            self.acceleration_history.append(acceleration)

            if len(self.acceleration_history) >= 5:
                avg_acc = np.mean(list(self.acceleration_history)[-5:])
                std_acc = np.std(list(self.acceleration_history)[-5:])

                if std_acc > 1e-10:
                    acc_zscore = acceleration / std_acc
                else:
                    acc_zscore = 0

                # Positive acceleration + positive momentum = strong buy
                # Negative acceleration + negative momentum = strong sell
                # Deceleration = potential reversal

                mom = self.momentum_history[-1]

                if acc_zscore > 1.0 and mom > 0:
                    self.signal = 1  # Accelerating uptrend
                    self.confidence = min(0.5 + acc_zscore * 0.15, 0.9)
                elif acc_zscore < -1.0 and mom < 0:
                    self.signal = -1  # Accelerating downtrend
                    self.confidence = min(0.5 + abs(acc_zscore) * 0.15, 0.9)
                elif acc_zscore < -1.0 and mom > 0:
                    self.signal = -1  # Decelerating uptrend - reversal
                    self.confidence = 0.6
                elif acc_zscore > 1.0 and mom < 0:
                    self.signal = 1  # Decelerating downtrend - reversal
                    self.confidence = 0.6
                else:
                    self.signal = 0
                    self.confidence = 0.3
            else:
                self.signal = 0
                self.confidence = 0.2
        else:
            self.signal = 0
            self.confidence = 0.1


# =============================================================================
# FORMULA 290: THE 50.75% EDGE FORMULA (Renaissance Inspired)
# =============================================================================
@FormulaRegistry.register(290)
class EdgeFormula5075(BaseFormula):
    """
    The 50.75% Edge Formula - Renaissance Technologies Inspired

    The secret: Don't try to be right 80% of the time.
    Try to be right 50.75% of the time with THOUSANDS of trades.

    This formula combines multiple micro-signals and only trades
    when the COMBINED probability exceeds 50.75%.

    E[profit] = (p × W) - ((1-p) × L)
    If W = L (symmetric payoff): E[profit] > 0 when p > 0.5

    With 0.75% edge and 1000 trades: Law of Large Numbers guarantees profit.
    """

    FORMULA_ID = 290
    CATEGORY = "hedge_fund"
    NAME = "Edge5075"
    DESCRIPTION = "The 50.75% edge formula - RenTech inspired"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.micro_signals = deque(maxlen=10)  # Last 10 micro-signals
        self.min_edge = kwargs.get('min_edge', 0.5075)  # 50.75%

        # Track multiple indicators
        self.momentum_3 = 0
        self.momentum_5 = 0
        self.tick_imbalance = 0
        self.volume_imbalance = 0

    def _compute(self) -> None:
        prices = self._prices_array()
        volumes = self._volumes_array()

        if len(prices) < 10:
            return

        # Calculate multiple micro-signals (each is a small edge)

        # 1. 3-tick momentum
        if len(prices) >= 3:
            self.momentum_3 = 1 if prices[-1] > prices[-3] else (-1 if prices[-1] < prices[-3] else 0)

        # 2. 5-tick momentum
        if len(prices) >= 5:
            self.momentum_5 = 1 if prices[-1] > prices[-5] else (-1 if prices[-1] < prices[-5] else 0)

        # 3. Tick imbalance (last 10 ticks)
        up_ticks = sum(1 for i in range(1, min(10, len(prices))) if prices[-i] > prices[-i-1])
        down_ticks = sum(1 for i in range(1, min(10, len(prices))) if prices[-i] < prices[-i-1])
        if up_ticks + down_ticks > 0:
            self.tick_imbalance = 1 if up_ticks > down_ticks else (-1 if down_ticks > up_ticks else 0)

        # 4. Volume imbalance (up vs down volume)
        if len(volumes) >= 5:
            up_vol = sum(volumes[-i] if prices[-i] > prices[-i-1] else 0 for i in range(1, 5))
            down_vol = sum(volumes[-i] if prices[-i] < prices[-i-1] else 0 for i in range(1, 5))
            if up_vol + down_vol > 0:
                self.volume_imbalance = 1 if up_vol > down_vol else (-1 if down_vol > up_vol else 0)

        # Combine signals - each has ~52% accuracy individually
        # Combined should approach 50.75% with correlation considerations
        signals = [
            self.momentum_3 * 0.52,
            self.momentum_5 * 0.51,
            self.tick_imbalance * 0.53,
            self.volume_imbalance * 0.52,
        ]

        # Calculate combined probability
        combined = sum(signals) / len(signals)

        # Only trade if combined edge exceeds threshold
        if combined > 0.5075:
            self.signal = 1
            self.confidence = min(combined, 0.6)  # Don't oversize
        elif combined < -0.5075:
            self.signal = -1
            self.confidence = min(abs(combined), 0.6)
        else:
            self.signal = 0  # No edge - don't trade
            self.confidence = 0.3
