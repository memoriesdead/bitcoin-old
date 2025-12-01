#!/usr/bin/env python3
"""
PEER-REVIEWED ACADEMIC FORMULAS (IDs 701-710)
==============================================
Gold-standard formulas from top finance journals.

SOURCES:
    701: Cont, Kukanov & Stoikov (2014) - J. Financial Econometrics (R²=70%)
    702: Kyle (1985) - Econometrica (10,000+ citations)
    703: Bacry et al. (2012) - J. Banking & Finance (Hawkes Process)
    704: Easley, Lopez de Prado & O'Hara (2012) - Review of Financial Studies
    705: Almgren & Chriss (2000) - Journal of Risk
    706: Academic Consensus - Flow Momentum
    707: Cont et al. (2024) - Cross-Asset OFI
    708: Kolm, Turiel & Westray (2021) - Deep OFI
    709: Multi-Level OFI - Xu, Gould, Howison (2019)
    710: Unified Academic Controller - Combines All

Key insight: Trade WITH flow, not against. Z-score mean reversion = zero edge.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from numba import njit

from .base import BaseFormula, FORMULA_REGISTRY


# =============================================================================
# ID 701: CONT-STOIKOV ORDER FLOW IMBALANCE (R² = 70%)
# Source: J. Financial Econometrics 12(1), 47-88 (2014)
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_cont_stoikov_ofi(
    bid_changes: np.ndarray,
    ask_changes: np.ndarray,
    price_directions: np.ndarray,
    n: int
) -> Tuple[float, float]:
    """
    Cont, Kukanov & Stoikov (2014) Order Flow Imbalance

    OFI = Σ(ΔBid_qty × I[price_up]) - Σ(ΔAsk_qty × I[price_down])

    Achieves R² = 65-70% for price prediction.
    This is the gold standard for short-term price prediction.

    Returns: (ofi_value, normalized_signal)
    """
    if n <= 0:
        return (0.0, 0.0)

    ofi = 0.0
    for i in range(n):
        if price_directions[i] > 0:  # Price moved up
            ofi += bid_changes[i]    # Bid queue increase = buying pressure
        elif price_directions[i] < 0:  # Price moved down
            ofi -= ask_changes[i]    # Ask queue increase = selling pressure

    # Normalize to -1 to 1 range
    abs_ofi = abs(ofi)
    if abs_ofi > 0:
        normalized = ofi / (abs_ofi + 1.0)  # Soft normalization
    else:
        normalized = 0.0

    return (ofi, normalized)


class ContStoikovOFI(BaseFormula):
    """
    ID 701: Cont-Stoikov Order Flow Imbalance

    Source: "The Price Impact of Order Book Events"
            Journal of Financial Econometrics, 12(1), 47-88 (2014)

    Key finding: Linear relation between OFI and price changes with R²=70%
    Trading strategy: Trade WITH the OFI direction, not against it

    Formula: ΔP = β × OFI / depth
    """
    formula_id = 701
    name = "ContStoikovOFI"
    category = "peer_reviewed"
    source = "J. Financial Econometrics 12(1), 47-88 (2014)"
    r_squared = 0.70

    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.bid_changes = np.zeros(lookback, dtype=np.float64)
        self.ask_changes = np.zeros(lookback, dtype=np.float64)
        self.price_directions = np.zeros(lookback, dtype=np.float64)
        self.idx = 0
        self.last_bid = 0.0
        self.last_ask = 0.0
        self.last_price = 0.0

    def calculate(self, price: float, bid_volume: float, ask_volume: float, **kwargs) -> dict:
        # Calculate changes
        bid_delta = bid_volume - self.last_bid if self.last_bid > 0 else 0.0
        ask_delta = ask_volume - self.last_ask if self.last_ask > 0 else 0.0
        price_dir = np.sign(price - self.last_price) if self.last_price > 0 else 0.0

        # Store in circular buffer
        pos = self.idx % self.lookback
        self.bid_changes[pos] = bid_delta
        self.ask_changes[pos] = ask_delta
        self.price_directions[pos] = price_dir
        self.idx += 1

        # Update last values
        self.last_bid = bid_volume
        self.last_ask = ask_volume
        self.last_price = price

        # Calculate OFI
        n = min(self.idx, self.lookback)
        ofi, signal = calc_cont_stoikov_ofi(
            self.bid_changes, self.ask_changes, self.price_directions, n
        )

        return {
            'ofi': ofi,
            'signal': signal,  # Trade WITH this direction
            'direction': 'BUY' if signal > 0.3 else 'SELL' if signal < -0.3 else 'HOLD',
            'r_squared': self.r_squared,
            'source': self.source
        }


# =============================================================================
# ID 702: KYLE LAMBDA (Price Impact Coefficient)
# Source: Econometrica 53(6), 1315-1335 (1985) - 10,000+ citations
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_kyle_lambda(
    price_changes: np.ndarray,
    volume_imbalances: np.ndarray,
    n: int
) -> Tuple[float, float]:
    """
    Kyle (1985) Lambda - Price Impact Coefficient

    ΔP = λ × (Buy Volume - Sell Volume)
    λ = Cov(ΔP, V) / Var(V)

    High λ = market is sensitive to informed trading
    Low λ = market can absorb large orders

    Returns: (lambda_value, information_signal)
    """
    if n < 5:
        return (0.0, 0.0)

    # Calculate means
    mean_dp = 0.0
    mean_vi = 0.0
    for i in range(n):
        mean_dp += price_changes[i]
        mean_vi += volume_imbalances[i]
    mean_dp /= n
    mean_vi /= n

    # Calculate covariance and variance
    cov = 0.0
    var_vi = 0.0
    for i in range(n):
        dp_dev = price_changes[i] - mean_dp
        vi_dev = volume_imbalances[i] - mean_vi
        cov += dp_dev * vi_dev
        var_vi += vi_dev * vi_dev

    cov /= n
    var_vi /= n

    # Kyle's lambda
    if var_vi > 1e-10:
        kyle_lambda = cov / var_vi
    else:
        kyle_lambda = 0.0

    # Information signal: recent volume imbalance * lambda = expected price move
    recent_vi = volume_imbalances[n-1] if n > 0 else 0.0
    expected_move = kyle_lambda * recent_vi

    # Normalize signal
    signal = np.tanh(expected_move * 100)  # Scale appropriately

    return (kyle_lambda, signal)


class KyleLambda(BaseFormula):
    """
    ID 702: Kyle's Lambda - Price Impact Coefficient

    Source: "Continuous Auctions and Insider Trading"
            Econometrica 53(6), 1315-1335 (1985)
            10,000+ citations - foundational market microstructure paper

    Key insight: Price impact is proportional to order flow imbalance
    Trading strategy:
        - High lambda + positive imbalance = BUY (informed buying detected)
        - High lambda + negative imbalance = SELL (informed selling detected)
    """
    formula_id = 702
    name = "KyleLambda"
    category = "peer_reviewed"
    source = "Econometrica 53(6), 1315-1335 (1985)"
    citations = 10000

    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.price_changes = np.zeros(lookback, dtype=np.float64)
        self.volume_imbalances = np.zeros(lookback, dtype=np.float64)
        self.idx = 0
        self.last_price = 0.0

    def calculate(self, price: float, buy_volume: float, sell_volume: float, **kwargs) -> dict:
        # Calculate price change and volume imbalance
        dp = (price - self.last_price) / self.last_price if self.last_price > 0 else 0.0
        vi = buy_volume - sell_volume

        # Store in circular buffer
        pos = self.idx % self.lookback
        self.price_changes[pos] = dp
        self.volume_imbalances[pos] = vi
        self.idx += 1
        self.last_price = price

        # Calculate Kyle's lambda
        n = min(self.idx, self.lookback)
        kyle_lambda, signal = calc_kyle_lambda(self.price_changes, self.volume_imbalances, n)

        return {
            'kyle_lambda': kyle_lambda,
            'signal': signal,
            'direction': 'BUY' if signal > 0.3 else 'SELL' if signal < -0.3 else 'HOLD',
            'volume_imbalance': vi,
            'information_detected': abs(kyle_lambda) > 1e-6,
            'source': self.source
        }


# =============================================================================
# ID 703: HAWKES SELF-EXCITATION PREDICTOR
# Source: J. Banking & Finance (2012)
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_hawkes_prediction(
    event_times: np.ndarray,
    event_signs: np.ndarray,
    n_events: int,
    current_time: float,
    mu: float = 0.1,
    alpha: float = 0.5,
    beta: float = 1.0
) -> Tuple[float, float, float]:
    """
    Hawkes Process for Order Arrival Prediction

    λ(t) = μ + Σ α × exp(-β × (t - t_i))

    Self-excitation: each event increases probability of more events

    Returns: (intensity, predicted_direction, confidence)
    """
    intensity = mu
    weighted_direction = 0.0
    weight_sum = 0.0

    for i in range(n_events):
        dt = current_time - event_times[i]
        if dt > 0 and dt < 100:  # Only recent events matter
            decay = alpha * math.exp(-beta * dt)
            intensity += decay
            weighted_direction += event_signs[i] * decay
            weight_sum += decay

    # Predicted direction based on recent flow
    if weight_sum > 1e-6:
        direction = weighted_direction / weight_sum
    else:
        direction = 0.0

    # Confidence based on intensity
    confidence = min(1.0, intensity / (mu + 1.0))

    return (intensity, direction, confidence)


class HawkesPredictor(BaseFormula):
    """
    ID 703: Hawkes Self-Excitation Predictor

    Source: "High-frequency financial data modeling using Hawkes processes"
            Journal of Banking & Finance (2012)

    Key insight: Order arrivals cluster - one order predicts more orders
    Trading strategy: Position BEFORE the predicted order burst arrives

    Self-excitation parameter α > 0 means:
    - Buy order → more buy orders likely
    - Sell order → more sell orders likely
    """
    formula_id = 703
    name = "HawkesPredictor"
    category = "peer_reviewed"
    source = "J. Banking & Finance (2012)"

    def __init__(self, max_events: int = 200, mu: float = 0.1, alpha: float = 0.5, beta: float = 1.0):
        self.max_events = max_events
        self.event_times = np.zeros(max_events, dtype=np.float64)
        self.event_signs = np.zeros(max_events, dtype=np.float64)
        self.n_events = 0
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def calculate(self, timestamp: float, order_sign: float, **kwargs) -> dict:
        # Record event
        if abs(order_sign) > 0.1:  # Significant order
            pos = self.n_events % self.max_events
            self.event_times[pos] = timestamp
            self.event_signs[pos] = np.sign(order_sign)
            self.n_events += 1

        # Predict
        n = min(self.n_events, self.max_events)
        intensity, direction, confidence = calc_hawkes_prediction(
            self.event_times, self.event_signs, n, timestamp,
            self.mu, self.alpha, self.beta
        )

        # Signal: direction * confidence
        signal = direction * confidence

        return {
            'intensity': intensity,
            'predicted_direction': direction,
            'confidence': confidence,
            'signal': signal,
            'direction': 'BUY' if signal > 0.3 else 'SELL' if signal < -0.3 else 'HOLD',
            'burst_imminent': intensity > self.mu * 2,
            'source': self.source
        }


# =============================================================================
# ID 704: VPIN VOLUME-CLOCK (Flash Crash Predictor)
# Source: Review of Financial Studies 25(5), 1457-1493 (2012)
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_vpin_academic(
    buy_buckets: np.ndarray,
    sell_buckets: np.ndarray,
    n_buckets: int
) -> Tuple[float, float]:
    """
    VPIN - Volume-Synchronized Probability of Informed Trading

    VPIN = Σ|V_buy - V_sell| / (n × V_bucket)

    KEY: Uses volume buckets, not time buckets.
    Volume = information arrival.

    High VPIN (>0.5) = toxic flow, informed traders present
    Low VPIN (<0.3) = noise trading, safe to provide liquidity

    Famous for predicting Flash Crash 2 hours before it happened.

    Returns: (vpin_value, toxicity_signal)
    """
    if n_buckets <= 0:
        return (0.0, 0.0)

    total_imbalance = 0.0
    total_volume = 0.0

    for i in range(n_buckets):
        total_imbalance += abs(buy_buckets[i] - sell_buckets[i])
        total_volume += buy_buckets[i] + sell_buckets[i]

    if total_volume > 0:
        vpin = total_imbalance / total_volume
    else:
        vpin = 0.0

    # Toxicity signal: high VPIN = DON'T trade (or trade carefully)
    # We invert this for signal: low toxicity = safe to trade WITH flow
    toxicity_signal = 1.0 - vpin

    return (vpin, toxicity_signal)


class VPINAcademic(BaseFormula):
    """
    ID 704: VPIN - Volume-Synchronized Probability of Informed Trading

    Source: "Flow Toxicity and Liquidity in a High-Frequency World"
            Review of Financial Studies 25(5), 1457-1493 (2012)
            Easley, Lopez de Prado & O'Hara

    Key insight: Volume-time better than clock-time for information arrival
    Trading strategy:
        - High VPIN (>0.5): Reduce position, toxic informed flow
        - Low VPIN (<0.3): Safe to trade, noise trading dominates

    Historical note: Predicted Flash Crash of May 6, 2010 two hours early
    """
    formula_id = 704
    name = "VPINAcademic"
    category = "peer_reviewed"
    source = "Review of Financial Studies 25(5), 1457-1493 (2012)"

    def __init__(self, n_buckets: int = 50, bucket_size: float = 10000.0):
        self.n_buckets = n_buckets
        self.bucket_size = bucket_size
        self.buy_buckets = np.zeros(n_buckets, dtype=np.float64)
        self.sell_buckets = np.zeros(n_buckets, dtype=np.float64)
        self.current_bucket_buy = 0.0
        self.current_bucket_sell = 0.0
        self.current_bucket_volume = 0.0
        self.bucket_idx = 0

    def calculate(self, buy_volume: float, sell_volume: float, **kwargs) -> dict:
        # Add to current bucket
        self.current_bucket_buy += buy_volume
        self.current_bucket_sell += sell_volume
        self.current_bucket_volume += buy_volume + sell_volume

        # Check if bucket is full
        bucket_complete = False
        if self.current_bucket_volume >= self.bucket_size:
            # Store bucket
            pos = self.bucket_idx % self.n_buckets
            self.buy_buckets[pos] = self.current_bucket_buy
            self.sell_buckets[pos] = self.current_bucket_sell
            self.bucket_idx += 1

            # Reset current bucket
            self.current_bucket_buy = 0.0
            self.current_bucket_sell = 0.0
            self.current_bucket_volume = 0.0
            bucket_complete = True

        # Calculate VPIN
        n = min(self.bucket_idx, self.n_buckets)
        vpin, safe_signal = calc_vpin_academic(self.buy_buckets, self.sell_buckets, n)

        return {
            'vpin': vpin,
            'signal': safe_signal,  # Higher = safer to trade
            'toxicity_level': 'HIGH' if vpin > 0.5 else 'MEDIUM' if vpin > 0.3 else 'LOW',
            'safe_to_trade': vpin < 0.3,
            'bucket_complete': bucket_complete,
            'buckets_filled': self.bucket_idx,
            'source': self.source
        }


# =============================================================================
# ID 705: ALMGREN-CHRISS OPTIMAL EXECUTION
# Source: Journal of Risk 3(5) (2000)
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_almgren_chriss_optimal_rate(
    total_shares: float,
    time_remaining: float,
    total_time: float,
    volatility: float,
    eta: float,  # Temporary impact
    gamma: float,  # Permanent impact
    lambda_risk: float  # Risk aversion
) -> Tuple[float, float]:
    """
    Almgren-Chriss Optimal Execution Trajectory

    Optimal rate of trading to minimize:
    E[Cost] + λ × Var[Cost]

    Where cost = permanent impact + temporary impact + timing risk

    Returns: (optimal_trade_rate, remaining_shares)
    """
    if time_remaining <= 0 or total_time <= 0:
        return (total_shares, 0.0)

    # Calculate kappa (urgency parameter)
    if eta > 0:
        kappa_sq = lambda_risk * volatility * volatility / eta
        kappa = math.sqrt(kappa_sq) if kappa_sq > 0 else 0.01
    else:
        kappa = 0.01

    # Optimal holdings at current time
    tau = time_remaining / total_time

    if kappa * total_time > 0.01:
        # sinh formula for optimal trajectory
        sinh_ratio = math.sinh(kappa * time_remaining) / math.sinh(kappa * total_time)
        optimal_holdings = total_shares * sinh_ratio
    else:
        # Linear approximation for small kappa
        optimal_holdings = total_shares * tau

    # Optimal trade rate (negative = selling)
    trade_rate = (total_shares - optimal_holdings) / (total_time - time_remaining + 0.001)

    return (trade_rate, optimal_holdings)


class AlmgrenChrissExecution(BaseFormula):
    """
    ID 705: Almgren-Chriss Optimal Execution

    Source: "Optimal Execution of Portfolio Transactions"
            Journal of Risk 3(5) (2000)

    Key insight: Balance urgency (timing risk) vs patience (market impact)

    Parameters:
        - eta: temporary impact coefficient
        - gamma: permanent impact coefficient
        - lambda: risk aversion (higher = more urgent execution)

    Trading strategy:
        - High volatility → execute faster
        - High lambda → more aggressive execution
        - Low impact → more patient execution
    """
    formula_id = 705
    name = "AlmgrenChrissExecution"
    category = "peer_reviewed"
    source = "Journal of Risk 3(5) (2000)"

    def __init__(self, eta: float = 0.001, gamma: float = 0.0001, lambda_risk: float = 1.0):
        self.eta = eta
        self.gamma = gamma
        self.lambda_risk = lambda_risk

    def calculate(self, total_shares: float, time_remaining: float, total_time: float,
                  volatility: float = 0.01, **kwargs) -> dict:

        trade_rate, remaining = calc_almgren_chriss_optimal_rate(
            total_shares, time_remaining, total_time,
            volatility, self.eta, self.gamma, self.lambda_risk
        )

        # Signal: positive rate = buy, negative = sell
        signal = np.sign(trade_rate) if abs(trade_rate) > 0.01 else 0.0

        # Urgency metric
        urgency = 1.0 - (time_remaining / total_time) if total_time > 0 else 1.0

        return {
            'optimal_trade_rate': trade_rate,
            'remaining_shares': remaining,
            'signal': signal,
            'urgency': urgency,
            'execution_style': 'AGGRESSIVE' if urgency > 0.7 else 'PATIENT',
            'source': self.source
        }


# =============================================================================
# ID 706: FLOW MOMENTUM (Academic Consensus)
# Source: Multiple papers consensus
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_flow_momentum_academic(
    ofi_history: np.ndarray,
    n: int,
    decay: float = 0.95
) -> Tuple[float, float, float]:
    """
    Flow Momentum - Academic Consensus Strategy

    Trade WITH the flow, not against it.

    Multiple papers show OFI direction predicts price direction.
    Mean reversion (trading against flow) has ZERO edge.

    Returns: (momentum, signal, strength)
    """
    if n <= 0:
        return (0.0, 0.0, 0.0)

    # Exponentially weighted momentum
    momentum = 0.0
    weight_sum = 0.0

    for i in range(n):
        weight = decay ** (n - 1 - i)
        momentum += ofi_history[i] * weight
        weight_sum += weight

    if weight_sum > 0:
        momentum /= weight_sum

    # Recent acceleration
    if n >= 5:
        recent_avg = 0.0
        old_avg = 0.0
        for i in range(n - 5, n):
            recent_avg += ofi_history[i]
        for i in range(max(0, n - 10), n - 5):
            old_avg += ofi_history[i]
        recent_avg /= 5
        old_avg /= 5 if n >= 10 else max(1, n - 5)
        acceleration = recent_avg - old_avg
    else:
        acceleration = 0.0

    # Signal: momentum direction × strength
    strength = min(1.0, abs(momentum) + abs(acceleration) * 0.5)
    signal = np.sign(momentum) * strength if strength > 0.1 else 0.0

    return (momentum, signal, strength)


class FlowMomentumAcademic(BaseFormula):
    """
    ID 706: Flow Momentum - Academic Consensus

    Source: Multiple peer-reviewed papers
        - Cont-Stoikov (2014): OFI predicts price
        - Kyle (1985): Price impact proportional to flow
        - All MEV research: Trade WITH flow, not against

    Key insight: Z-score mean reversion = ZERO edge
                 Flow following = positive edge

    Why it works:
        - OFI explains 70% of price variance (Cont-Stoikov)
        - Trading against OFI = trading against the trend
        - Trading WITH OFI = surfing the information wave
    """
    formula_id = 706
    name = "FlowMomentumAcademic"
    category = "peer_reviewed"
    source = "Academic Consensus (Multiple Papers)"

    def __init__(self, lookback: int = 50, decay: float = 0.95):
        self.lookback = lookback
        self.decay = decay
        self.ofi_history = np.zeros(lookback, dtype=np.float64)
        self.idx = 0

    def calculate(self, ofi: float, **kwargs) -> dict:
        # Store OFI
        pos = self.idx % self.lookback
        self.ofi_history[pos] = ofi
        self.idx += 1

        # Calculate momentum
        n = min(self.idx, self.lookback)
        momentum, signal, strength = calc_flow_momentum_academic(
            self.ofi_history, n, self.decay
        )

        return {
            'momentum': momentum,
            'signal': signal,  # Trade WITH this direction
            'strength': strength,
            'direction': 'BUY' if signal > 0.3 else 'SELL' if signal < -0.3 else 'HOLD',
            'confidence': strength,
            'source': self.source,
            'key_insight': "Trade WITH flow, not against"
        }


# =============================================================================
# ID 707: CROSS-ASSET OFI
# Source: Cont et al. (2024) - Quantitative Finance
# =============================================================================
class CrossAssetOFI(BaseFormula):
    """
    ID 707: Cross-Asset Order Flow Imbalance

    Source: "Cross-Impact of Order Flow Imbalance in Equity Markets"
            Quantitative Finance (2024)
            Cont, Cucuringu & Zhang

    Key insight: OFI in correlated assets predicts price in target asset
    Formula: ΔP_i = Σ_j β_ij × OFI_j
    """
    formula_id = 707
    name = "CrossAssetOFI"
    category = "peer_reviewed"
    source = "Quantitative Finance (2024)"

    def __init__(self):
        self.eth_ofi = 0.0
        self.btc_ofi = 0.0

    def calculate(self, btc_ofi: float, eth_ofi: float = 0.0,
                  correlation: float = 0.8, **kwargs) -> dict:
        """Cross-asset signal for BTC using ETH OFI"""
        self.btc_ofi = btc_ofi
        self.eth_ofi = eth_ofi

        # Cross-impact: ETH OFI can predict BTC
        cross_signal = correlation * eth_ofi + btc_ofi
        signal = np.tanh(cross_signal)

        return {
            'btc_ofi': btc_ofi,
            'eth_ofi': eth_ofi,
            'cross_signal': cross_signal,
            'signal': signal,
            'direction': 'BUY' if signal > 0.3 else 'SELL' if signal < -0.3 else 'HOLD',
            'source': self.source
        }


# =============================================================================
# ID 708: DEEP OFI (Neural Network Enhanced)
# Source: Kolm, Turiel & Westray (2021)
# =============================================================================
class DeepOFI(BaseFormula):
    """
    ID 708: Deep Order Flow Imbalance

    Source: "Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons"
            Kolm, Turiel & Westray (2021)

    Uses deep learning to extract alpha from order book at multiple horizons.
    Simplified version uses multi-scale OFI aggregation.
    """
    formula_id = 708
    name = "DeepOFI"
    category = "peer_reviewed"
    source = "SSRN:3900141 (2021)"

    def __init__(self):
        self.ofi_1 = 0.0   # 1-tick horizon
        self.ofi_5 = 0.0   # 5-tick horizon
        self.ofi_10 = 0.0  # 10-tick horizon
        self.history = []

    def calculate(self, ofi: float, **kwargs) -> dict:
        # Track history
        self.history.append(ofi)
        if len(self.history) > 100:
            self.history = self.history[-100:]

        # Multi-scale OFI
        n = len(self.history)
        self.ofi_1 = self.history[-1] if n >= 1 else 0
        self.ofi_5 = np.mean(self.history[-5:]) if n >= 5 else self.ofi_1
        self.ofi_10 = np.mean(self.history[-10:]) if n >= 10 else self.ofi_5

        # Weighted combination (mimics neural network output)
        deep_signal = 0.5 * self.ofi_1 + 0.3 * self.ofi_5 + 0.2 * self.ofi_10
        signal = np.tanh(deep_signal)

        return {
            'ofi_1tick': self.ofi_1,
            'ofi_5tick': self.ofi_5,
            'ofi_10tick': self.ofi_10,
            'deep_signal': deep_signal,
            'signal': signal,
            'direction': 'BUY' if signal > 0.3 else 'SELL' if signal < -0.3 else 'HOLD',
            'source': self.source
        }


# =============================================================================
# ID 709: MULTI-LEVEL OFI
# Source: Xu, Gould & Howison (2019)
# =============================================================================
class MultiLevelOFI(BaseFormula):
    """
    ID 709: Multi-Level Order Flow Imbalance

    Source: "Multi-Level Order-Flow Imbalance in a Limit Order Book"
            Xu, Gould & Howison (2019) - SSRN:3479741

    Key insight: OFI at multiple price levels provides better signal
    Formula: OFI_integrated = Σ_k w_k × OFI_level_k
    """
    formula_id = 709
    name = "MultiLevelOFI"
    category = "peer_reviewed"
    source = "SSRN:3479741 (2019)"

    def calculate(self, level1_ofi: float, level2_ofi: float = 0.0,
                  level3_ofi: float = 0.0, **kwargs) -> dict:
        # Depth-weighted OFI
        # Level 1 (best bid/ask) has highest weight
        w1, w2, w3 = 0.6, 0.25, 0.15

        integrated_ofi = w1 * level1_ofi + w2 * level2_ofi + w3 * level3_ofi
        signal = np.tanh(integrated_ofi)

        return {
            'level1_ofi': level1_ofi,
            'level2_ofi': level2_ofi,
            'level3_ofi': level3_ofi,
            'integrated_ofi': integrated_ofi,
            'signal': signal,
            'direction': 'BUY' if signal > 0.3 else 'SELL' if signal < -0.3 else 'HOLD',
            'source': self.source
        }


# =============================================================================
# ID 710: UNIFIED ACADEMIC CONTROLLER
# Combines all peer-reviewed formulas
# =============================================================================
class UnifiedAcademicController(BaseFormula):
    """
    ID 710: Unified Academic Controller

    Master controller combining ALL peer-reviewed formulas:
        - 701: Cont-Stoikov OFI (R²=70%)
        - 702: Kyle Lambda (price impact)
        - 703: Hawkes Predictor (timing)
        - 704: VPIN (toxicity filter)
        - 705: Almgren-Chriss (execution)
        - 706: Flow Momentum (direction)

    Key insight: Trade WITH flow when VPIN is low.
    """
    formula_id = 710
    name = "UnifiedAcademicController"
    category = "peer_reviewed"
    source = "Combined Academic Research"

    def __init__(self):
        self.cont_stoikov = ContStoikovOFI()
        self.kyle = KyleLambda()
        self.hawkes = HawkesPredictor()
        self.vpin = VPINAcademic()
        self.flow_momentum = FlowMomentumAcademic()
        self.total_signals = 0
        self.profitable_signals = 0

    def calculate(self, price: float, buy_volume: float, sell_volume: float,
                  timestamp: float = 0.0, **kwargs) -> dict:

        # Calculate OFI
        ofi = buy_volume - sell_volume

        # Get all sub-signals
        cs_result = self.cont_stoikov.calculate(price, buy_volume, sell_volume)
        kyle_result = self.kyle.calculate(price, buy_volume, sell_volume)
        hawkes_result = self.hawkes.calculate(timestamp, ofi)
        vpin_result = self.vpin.calculate(buy_volume, sell_volume)
        momentum_result = self.flow_momentum.calculate(ofi)

        # Combine signals with weights based on R² / reliability
        combined = (
            0.30 * cs_result['signal'] +      # Highest R² (70%)
            0.25 * kyle_result['signal'] +    # Foundational
            0.15 * hawkes_result['signal'] +  # Timing
            0.30 * momentum_result['signal']  # Consensus
        )

        # Apply VPIN toxicity filter
        if vpin_result['vpin'] > 0.5:
            combined *= 0.3  # High toxicity = reduce signal
        elif vpin_result['vpin'] < 0.2:
            combined *= 1.2  # Low toxicity = amplify signal

        # Clip to [-1, 1]
        final_signal = max(-1.0, min(1.0, combined))

        self.total_signals += 1

        return {
            'signal': final_signal,
            'direction': 'BUY' if final_signal > 0.3 else 'SELL' if final_signal < -0.3 else 'HOLD',
            'cont_stoikov_signal': cs_result['signal'],
            'kyle_signal': kyle_result['signal'],
            'hawkes_signal': hawkes_result['signal'],
            'momentum_signal': momentum_result['signal'],
            'vpin': vpin_result['vpin'],
            'toxicity': vpin_result['toxicity_level'],
            'confidence': abs(final_signal),
            'key_insight': "Trade WITH flow when toxicity is LOW",
            'total_signals': self.total_signals
        }


# =============================================================================
# REGISTER ALL FORMULAS
# =============================================================================
FORMULA_REGISTRY[701] = ContStoikovOFI
FORMULA_REGISTRY[702] = KyleLambda
FORMULA_REGISTRY[703] = HawkesPredictor
FORMULA_REGISTRY[704] = VPINAcademic
FORMULA_REGISTRY[705] = AlmgrenChrissExecution
FORMULA_REGISTRY[706] = FlowMomentumAcademic
FORMULA_REGISTRY[707] = CrossAssetOFI
FORMULA_REGISTRY[708] = DeepOFI
FORMULA_REGISTRY[709] = MultiLevelOFI
FORMULA_REGISTRY[710] = UnifiedAcademicController

print(f"[PeerReviewed] Registered 10 peer-reviewed academic formulas (701-710)")


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PEER-REVIEWED ACADEMIC FORMULAS TEST (IDs 701-710)")
    print("=" * 70)

    # Test Unified Controller
    controller = UnifiedAcademicController()

    print("\nSimulating order flow with BUY pressure...")
    print("-" * 70)

    for i in range(20):
        # Simulate buy pressure
        buy_vol = 100 + i * 10  # Increasing buy volume
        sell_vol = 80
        price = 97000 + i * 5  # Price rising with buys

        result = controller.calculate(
            price=price,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            timestamp=float(i)
        )

        print(f"[{i:2d}] Signal: {result['signal']:+.3f} | "
              f"Dir: {result['direction']:4s} | "
              f"CS: {result['cont_stoikov_signal']:+.2f} | "
              f"Kyle: {result['kyle_signal']:+.2f} | "
              f"VPIN: {result['vpin']:.2f} ({result['toxicity']})")

    print("-" * 70)
    print("KEY INSIGHT:", result['key_insight'])
    print("=" * 70)
