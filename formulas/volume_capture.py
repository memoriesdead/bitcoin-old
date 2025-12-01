#!/usr/bin/env python3
"""
VOLUME CAPTURE FORMULAS (IDs 601-610)
=====================================
Mathematical formulas to capture percentage of blockchain volume flow.

Based on academic research:
- Kyle (1985) - Continuous auctions and insider trading
- Easley, Lopez de Prado, O'Hara - VPIN and Flow Toxicity
- Almgren-Chriss - Optimal execution
- Cont et al. - Order Flow Imbalance

Target: Capture X% of $722,435/second blockchain volume
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from numba import njit

from .base import BaseFormula, FORMULA_REGISTRY

# =============================================================================
# BLOCKCHAIN VOLUME CONSTANTS
# =============================================================================
DAILY_BTC_VOLUME = 450_000.0       # BTC/day on-chain
SECONDS_PER_DAY = 86_400.0
BTC_PER_SECOND = DAILY_BTC_VOLUME / SECONDS_PER_DAY  # ~5.208 BTC/sec

# Volume bucket sizes for volume-clock trading
VOLUME_BUCKET_BTC = 0.1            # Trade signal per 0.1 BTC flow
VOLUME_BUCKET_USD = 10_000.0       # Or per $10K flow


# =============================================================================
# ID 601: POV PARTICIPATION (Percentage of Volume)
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_pov_participation(
    capital: float,
    volume_per_second: float,
    target_participation: float = 0.0001,  # 0.01% of volume
    max_participation: float = 0.001       # Cap at 0.1%
) -> Tuple[float, float, float]:
    """
    POV (Percentage of Volume) Algorithm

    Execute as fixed percentage of actual blockchain volume.

    Formula: position_size = min(capital, volume_per_second * participation_rate)

    Args:
        capital: Available capital in USD
        volume_per_second: Current blockchain volume in USD/second
        target_participation: Target % of volume to capture
        max_participation: Maximum allowed participation rate

    Returns:
        (position_size, actual_participation, capture_potential)
    """
    if volume_per_second <= 0 or capital <= 0:
        return (0.0, 0.0, 0.0)

    # Calculate position size as % of volume
    target_size = volume_per_second * target_participation

    # Cap at capital and max participation
    position_size = min(capital, target_size)
    actual_participation = position_size / volume_per_second
    actual_participation = min(actual_participation, max_participation)

    # Recalculate position with capped participation
    position_size = volume_per_second * actual_participation

    # Capture potential (theoretical max capture per second)
    capture_potential = position_size * 0.001  # Assume 0.1% edge per trade

    return (position_size, actual_participation, capture_potential)


class POVParticipation(BaseFormula):
    """
    ID 601: Percentage of Volume Participation

    Execute trades as a fixed percentage of blockchain volume flow.
    Adjusts position size based on actual volume, not time.
    """
    formula_id = 601
    name = "POVParticipation"
    category = "volume_capture"

    def __init__(self, target_rate: float = 0.0001):
        self.target_rate = target_rate
        self.cumulative_volume = 0.0
        self.cumulative_capture = 0.0

    def calculate(self, capital: float, volume_usd: float, **kwargs) -> dict:
        pos_size, actual_rate, capture = calc_pov_participation(
            capital, volume_usd, self.target_rate
        )
        self.cumulative_volume += volume_usd
        self.cumulative_capture += capture

        return {
            'position_size': pos_size,
            'participation_rate': actual_rate,
            'capture_per_second': capture,
            'signal': 1.0 if pos_size > 0 else 0.0
        }


# =============================================================================
# ID 602: VOLUME CLOCK TRADING
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_volume_clock(
    cumulative_volume: float,
    bucket_size: float,
    last_bucket: int
) -> Tuple[int, bool, float]:
    """
    Volume Clock - Trade on volume buckets, not time.

    Signal fires every time X BTC/USD of volume flows through.
    This syncs trading with information arrival (volume = information).

    Args:
        cumulative_volume: Total volume seen so far
        bucket_size: Volume per bucket (e.g., $10,000)
        last_bucket: Last bucket number processed

    Returns:
        (current_bucket, signal_fire, bucket_progress)
    """
    current_bucket = int(cumulative_volume / bucket_size)
    signal_fire = current_bucket > last_bucket
    bucket_progress = (cumulative_volume % bucket_size) / bucket_size

    return (current_bucket, signal_fire, bucket_progress)


class VolumeClockTrading(BaseFormula):
    """
    ID 602: Volume Clock Trading

    Updates signals in volume-time, not clock-time.
    Each bucket represents equal information content.
    Based on VPIN methodology from Easley, Lopez de Prado, O'Hara.
    """
    formula_id = 602
    name = "VolumeClockTrading"
    category = "volume_capture"

    def __init__(self, bucket_size_usd: float = 10000.0):
        self.bucket_size = bucket_size_usd
        self.cumulative_volume = 0.0
        self.last_bucket = 0
        self.signals_fired = 0

    def calculate(self, volume_usd: float, **kwargs) -> dict:
        self.cumulative_volume += volume_usd
        bucket, fire, progress = calc_volume_clock(
            self.cumulative_volume, self.bucket_size, self.last_bucket
        )

        if fire:
            self.last_bucket = bucket
            self.signals_fired += 1

        return {
            'current_bucket': bucket,
            'signal': 1.0 if fire else 0.0,
            'bucket_progress': progress,
            'total_signals': self.signals_fired
        }


# =============================================================================
# ID 603: VWAP PARTICIPATION
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_vwap(
    prices: np.ndarray,
    volumes: np.ndarray,
    n: int
) -> float:
    """
    Volume-Weighted Average Price

    VWAP = Sum(Price_i * Volume_i) / Sum(Volume_i)
    """
    if n <= 0:
        return 0.0

    total_pv = 0.0
    total_v = 0.0

    for i in range(n):
        total_pv += prices[i] * volumes[i]
        total_v += volumes[i]

    if total_v <= 0:
        return prices[n-1] if n > 0 else 0.0

    return total_pv / total_v


@njit(cache=True, fastmath=True)
def calc_vwap_signal(
    current_price: float,
    vwap: float,
    threshold: float = 0.001
) -> Tuple[float, float]:
    """
    Generate signal based on price vs VWAP.

    Buy below VWAP, Sell above VWAP.
    """
    if vwap <= 0:
        return (0.0, 0.0)

    deviation = (current_price - vwap) / vwap

    if deviation < -threshold:
        signal = 1.0   # Price below VWAP = BUY
    elif deviation > threshold:
        signal = -1.0  # Price above VWAP = SELL
    else:
        signal = 0.0

    return (signal, deviation)


class VWAPParticipation(BaseFormula):
    """
    ID 603: VWAP Participation Rate

    Track volume-weighted average price and trade to beat it.
    Buy below VWAP, sell above VWAP.
    """
    formula_id = 603
    name = "VWAPParticipation"
    category = "volume_capture"

    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.prices = np.zeros(lookback, dtype=np.float64)
        self.volumes = np.zeros(lookback, dtype=np.float64)
        self.idx = 0

    def calculate(self, price: float, volume: float, **kwargs) -> dict:
        # Update rolling buffers
        pos = self.idx % self.lookback
        self.prices[pos] = price
        self.volumes[pos] = volume
        self.idx += 1

        n = min(self.idx, self.lookback)
        vwap = calc_vwap(self.prices, self.volumes, n)
        signal, deviation = calc_vwap_signal(price, vwap)

        return {
            'vwap': vwap,
            'signal': signal,
            'deviation': deviation,
            'beat_vwap': price < vwap
        }


# =============================================================================
# ID 604: FLOW MOMENTUM SCALPER
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_ofi(
    bid_volume_delta: float,
    ask_volume_delta: float
) -> float:
    """
    Order Flow Imbalance (OFI)

    OFI = Delta_Bid_Volume - Delta_Ask_Volume

    Positive OFI = More buying pressure
    Negative OFI = More selling pressure
    """
    return bid_volume_delta - ask_volume_delta


@njit(cache=True, fastmath=True)
def calc_flow_momentum(
    ofi_history: np.ndarray,
    n: int,
    decay: float = 0.94
) -> Tuple[float, float]:
    """
    Flow Momentum from OFI history.

    Exponentially weighted OFI momentum.
    Trade in direction of flow.
    """
    if n <= 0:
        return (0.0, 0.0)

    momentum = 0.0
    weight_sum = 0.0

    for i in range(n):
        w = decay ** (n - 1 - i)
        momentum += ofi_history[i] * w
        weight_sum += w

    if weight_sum > 0:
        momentum /= weight_sum

    # Normalize to -1 to 1
    strength = min(abs(momentum), 1.0)
    direction = 1.0 if momentum > 0 else -1.0 if momentum < 0 else 0.0

    return (direction * strength, momentum)


class FlowMomentumScalper(BaseFormula):
    """
    ID 604: Flow Momentum Scalper

    Trade in the direction of Order Flow Imbalance.
    OFI = Bid_Volume_Change - Ask_Volume_Change

    Based on Cont et al. market microstructure research.
    """
    formula_id = 604
    name = "FlowMomentumScalper"
    category = "volume_capture"

    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.ofi_history = np.zeros(lookback, dtype=np.float64)
        self.idx = 0
        self.last_bid_vol = 0.0
        self.last_ask_vol = 0.0

    def calculate(self, bid_volume: float, ask_volume: float, **kwargs) -> dict:
        # Calculate OFI
        bid_delta = bid_volume - self.last_bid_vol
        ask_delta = ask_volume - self.last_ask_vol
        ofi = calc_ofi(bid_delta, ask_delta)

        # Update history
        pos = self.idx % self.lookback
        self.ofi_history[pos] = ofi
        self.idx += 1

        self.last_bid_vol = bid_volume
        self.last_ask_vol = ask_volume

        # Calculate momentum
        n = min(self.idx, self.lookback)
        signal, raw_momentum = calc_flow_momentum(self.ofi_history, n)

        return {
            'ofi': ofi,
            'signal': signal,
            'momentum': raw_momentum,
            'direction': 'BUY' if signal > 0 else 'SELL' if signal < 0 else 'HOLD'
        }


# =============================================================================
# ID 605: VOLUME IMBALANCE PREDICTOR (Hawkes Process)
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_hawkes_intensity(
    event_times: np.ndarray,
    n_events: int,
    current_time: float,
    mu: float = 0.1,      # Base intensity
    alpha: float = 0.5,   # Excitation
    beta: float = 1.0     # Decay rate
) -> float:
    """
    Hawkes Process Intensity for predicting volume bursts.

    lambda(t) = mu + sum(alpha * exp(-beta * (t - t_i)))

    Higher intensity = expect more volume soon.
    """
    intensity = mu

    for i in range(n_events):
        dt = current_time - event_times[i]
        if dt > 0:
            intensity += alpha * math.exp(-beta * dt)

    return intensity


class VolumeImbalancePredictor(BaseFormula):
    """
    ID 605: Volume Imbalance Predictor

    Uses Hawkes process to forecast OFI spikes.
    Pre-position before large flow arrives.

    Based on: "Forecasting high frequency order flow imbalance using Hawkes processes"
    """
    formula_id = 605
    name = "VolumeImbalancePredictor"
    category = "volume_capture"

    def __init__(self, max_events: int = 100):
        self.max_events = max_events
        self.event_times = np.zeros(max_events, dtype=np.float64)
        self.event_signs = np.zeros(max_events, dtype=np.float64)  # +1 buy, -1 sell
        self.n_events = 0

    def calculate(self, timestamp: float, ofi: float, threshold: float = 0.5, **kwargs) -> dict:
        # Record significant OFI events
        if abs(ofi) > threshold:
            pos = self.n_events % self.max_events
            self.event_times[pos] = timestamp
            self.event_signs[pos] = 1.0 if ofi > 0 else -1.0
            self.n_events += 1

        # Calculate intensity
        n = min(self.n_events, self.max_events)
        intensity = calc_hawkes_intensity(self.event_times, n, timestamp)

        # Predict direction based on recent event signs
        recent_direction = 0.0
        if n > 0:
            recent_direction = np.mean(self.event_signs[:n])

        # Signal: high intensity + clear direction = strong signal
        signal = intensity * recent_direction

        return {
            'intensity': intensity,
            'predicted_direction': recent_direction,
            'signal': max(-1.0, min(1.0, signal)),
            'event_count': self.n_events
        }


# =============================================================================
# ID 606: SHAPLEY VALUE CALCULATOR
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_shapley_value(
    total_value: float,
    n_participants: int,
    your_contribution: float,
    total_contribution: float
) -> float:
    """
    Shapley Value for fair allocation of captured volume.

    Your share = (Your Contribution / Total Contribution) * Total Value

    From cooperative game theory.
    """
    if total_contribution <= 0 or n_participants <= 0:
        return 0.0

    # Simple proportional Shapley approximation
    your_share = (your_contribution / total_contribution) * total_value

    # Adjust for number of participants (competition factor)
    competition_factor = 1.0 / (1.0 + math.log(n_participants + 1))

    return your_share * competition_factor


class ShapleyVolumeValue(BaseFormula):
    """
    ID 606: Shapley Value Calculator

    Game theory fair value allocation from volume capture.
    Determines your fair share of the MEV/volume pie.
    """
    formula_id = 606
    name = "ShapleyVolumeValue"
    category = "volume_capture"

    def __init__(self):
        self.cumulative_value = 0.0

    def calculate(self, total_volume: float, your_capital: float,
                  total_capital: float = 1_000_000_000.0,  # Assume $1B total market
                  n_traders: int = 10000, **kwargs) -> dict:

        fair_value = calc_shapley_value(
            total_volume, n_traders, your_capital, total_capital
        )
        self.cumulative_value += fair_value

        return {
            'fair_value': fair_value,
            'your_share_pct': (your_capital / total_capital) * 100,
            'cumulative': self.cumulative_value,
            'signal': 1.0 if fair_value > 0 else 0.0
        }


# =============================================================================
# ID 607: BLOCK SPACE OPTIMIZER
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_block_position_value(
    position_in_block: int,
    block_size: int,
    total_mev: float
) -> float:
    """
    Value of position in block ordering.

    Earlier positions capture more MEV.
    Linear decay: first position gets most value.
    """
    if block_size <= 0 or position_in_block >= block_size:
        return 0.0

    # Position value decays linearly
    position_factor = 1.0 - (position_in_block / block_size)

    return total_mev * position_factor * position_factor  # Quadratic decay


class BlockSpaceOptimizer(BaseFormula):
    """
    ID 607: Block Space Optimizer

    Optimize position within block for MEV capture.
    Earlier = better for front-running.
    Later = better for back-running.
    """
    formula_id = 607
    name = "BlockSpaceOptimizer"
    category = "volume_capture"

    def calculate(self, block_progress: float, estimated_mev: float = 1000.0,
                  block_size: int = 1000, **kwargs) -> dict:

        position = int(block_progress * block_size)
        value = calc_block_position_value(position, block_size, estimated_mev)

        # Signal: trade when position value is high
        signal = value / estimated_mev if estimated_mev > 0 else 0.0

        return {
            'position': position,
            'position_value': value,
            'signal': signal,
            'optimal_position': 'EARLY' if block_progress < 0.3 else 'LATE' if block_progress > 0.7 else 'MID'
        }


# =============================================================================
# ID 608: VPIN VOLUME SYNC
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_vpin(
    buy_volume: float,
    sell_volume: float,
    total_volume: float
) -> float:
    """
    Volume-Synchronized Probability of Informed Trading (VPIN)

    VPIN = |Buy Volume - Sell Volume| / Total Volume

    High VPIN = toxic flow, informed traders active
    Low VPIN = noise trading, safe to provide liquidity
    """
    if total_volume <= 0:
        return 0.0

    return abs(buy_volume - sell_volume) / total_volume


class VPINVolumeSync(BaseFormula):
    """
    ID 608: VPIN Volume Synchronized

    Real-time toxicity signal updated per volume bucket.
    High VPIN = avoid (toxic flow)
    Low VPIN = capture (safe flow)

    Based on Easley, Lopez de Prado, O'Hara (2012).
    """
    formula_id = 608
    name = "VPINVolumeSync"
    category = "volume_capture"

    def __init__(self, n_buckets: int = 50):
        self.n_buckets = n_buckets
        self.buy_volumes = np.zeros(n_buckets, dtype=np.float64)
        self.sell_volumes = np.zeros(n_buckets, dtype=np.float64)
        self.idx = 0

    def calculate(self, buy_volume: float, sell_volume: float, **kwargs) -> dict:
        pos = self.idx % self.n_buckets
        self.buy_volumes[pos] = buy_volume
        self.sell_volumes[pos] = sell_volume
        self.idx += 1

        n = min(self.idx, self.n_buckets)
        total_buy = np.sum(self.buy_volumes[:n])
        total_sell = np.sum(self.sell_volumes[:n])
        total = total_buy + total_sell

        vpin = calc_vpin(total_buy, total_sell, total)

        # Signal: inverse of VPIN (trade when toxicity is LOW)
        signal = 1.0 - vpin

        return {
            'vpin': vpin,
            'toxicity': 'HIGH' if vpin > 0.5 else 'LOW',
            'signal': signal,
            'safe_to_trade': vpin < 0.3
        }


# =============================================================================
# ID 609: ADAPTIVE PARTICIPATION RATE
# =============================================================================
@njit(cache=True, fastmath=True)
def calc_adaptive_participation(
    base_rate: float,
    volatility: float,
    spread: float,
    volume_ratio: float  # actual/expected volume
) -> float:
    """
    Adaptive participation rate based on market conditions.

    Increase participation when:
    - Low volatility (stable prices)
    - Tight spreads (low cost)
    - High volume (more opportunities)

    Decrease when opposite.
    """
    # Volatility adjustment (lower vol = higher participation)
    vol_factor = 1.0 / (1.0 + volatility * 10)

    # Spread adjustment (tighter spread = higher participation)
    spread_factor = 1.0 / (1.0 + spread * 1000)

    # Volume adjustment (higher volume = higher participation)
    vol_factor_adj = min(volume_ratio, 2.0)  # Cap at 2x

    adjusted_rate = base_rate * vol_factor * spread_factor * vol_factor_adj

    # Cap at reasonable limits
    return max(0.00001, min(0.01, adjusted_rate))  # 0.001% to 1%


class AdaptiveParticipation(BaseFormula):
    """
    ID 609: Adaptive Participation Rate

    Dynamically adjust capture rate based on:
    - Volatility (reduce in high vol)
    - Spread (increase in tight spreads)
    - Volume (increase in high volume)
    """
    formula_id = 609
    name = "AdaptiveParticipation"
    category = "volume_capture"

    def __init__(self, base_rate: float = 0.0001):
        self.base_rate = base_rate

    def calculate(self, volatility: float, spread: float,
                  actual_volume: float, expected_volume: float, **kwargs) -> dict:

        vol_ratio = actual_volume / expected_volume if expected_volume > 0 else 1.0

        rate = calc_adaptive_participation(
            self.base_rate, volatility, spread, vol_ratio
        )

        return {
            'participation_rate': rate,
            'rate_multiplier': rate / self.base_rate,
            'signal': 1.0,
            'conditions': {
                'volatility': 'LOW' if volatility < 0.01 else 'HIGH',
                'spread': 'TIGHT' if spread < 0.001 else 'WIDE',
                'volume': 'HIGH' if vol_ratio > 1.0 else 'LOW'
            }
        }


# =============================================================================
# ID 610: VOLUME CAPTURE MASTER CONTROLLER
# =============================================================================
class VolumeCaptureController(BaseFormula):
    """
    ID 610: Volume Capture Master Controller

    Unified controller combining all volume capture formulas.
    Target: Capture X% of $722,435/second blockchain volume.

    Combines:
    - POV participation rate
    - Volume clock signals
    - VWAP positioning
    - Flow momentum
    - VPIN toxicity filter
    - Adaptive rate scaling
    """
    formula_id = 610
    name = "VolumeCaptureController"
    category = "volume_capture"

    def __init__(self, target_capture_pct: float = 0.0001):
        self.target_pct = target_capture_pct
        self.pov = POVParticipation(target_capture_pct)
        self.volume_clock = VolumeClockTrading()
        self.vwap = VWAPParticipation()
        self.flow = FlowMomentumScalper()
        self.vpin = VPINVolumeSync()
        self.adaptive = AdaptiveParticipation(target_capture_pct)

        self.total_captured = 0.0
        self.trades = 0

    def calculate(self, price: float, capital: float, volume_usd: float,
                  bid_volume: float = 0.0, ask_volume: float = 0.0,
                  volatility: float = 0.01, spread: float = 0.0005,
                  expected_volume: float = 722435.0, **kwargs) -> dict:

        # Get all sub-signals
        pov_result = self.pov.calculate(capital, volume_usd)
        clock_result = self.volume_clock.calculate(volume_usd)
        vwap_result = self.vwap.calculate(price, volume_usd)
        flow_result = self.flow.calculate(bid_volume, ask_volume)
        vpin_result = self.vpin.calculate(bid_volume, ask_volume)
        adaptive_result = self.adaptive.calculate(
            volatility, spread, volume_usd, expected_volume
        )

        # Combine signals with weights
        combined_signal = (
            0.20 * pov_result['signal'] +
            0.15 * clock_result['signal'] +
            0.20 * vwap_result['signal'] +
            0.25 * flow_result['signal'] +
            0.20 * vpin_result['signal']
        )

        # Apply VPIN toxicity filter
        if vpin_result['vpin'] > 0.6:  # High toxicity
            combined_signal *= 0.3  # Reduce participation

        # Calculate final position and capture
        final_rate = adaptive_result['participation_rate']
        position_size = min(capital, volume_usd * final_rate)

        # Estimate capture (assume 0.05% edge per trade)
        capture = position_size * 0.0005 * abs(combined_signal)
        self.total_captured += capture

        if abs(combined_signal) > 0.5:
            self.trades += 1

        return {
            'signal': combined_signal,
            'direction': 'BUY' if combined_signal > 0.3 else 'SELL' if combined_signal < -0.3 else 'HOLD',
            'position_size': position_size,
            'participation_rate': final_rate,
            'estimated_capture': capture,
            'total_captured': self.total_captured,
            'trade_count': self.trades,
            'vpin_toxicity': vpin_result['vpin'],
            'flow_momentum': flow_result['momentum'],
            'vwap_deviation': vwap_result['deviation'],
            'volume_bucket': clock_result['current_bucket']
        }


# =============================================================================
# REGISTER ALL FORMULAS
# =============================================================================
# Use direct dict assignment - FORMULA_REGISTRY is a dict
FORMULA_REGISTRY[601] = POVParticipation
FORMULA_REGISTRY[602] = VolumeClockTrading
FORMULA_REGISTRY[603] = VWAPParticipation
FORMULA_REGISTRY[604] = FlowMomentumScalper
FORMULA_REGISTRY[605] = VolumeImbalancePredictor
FORMULA_REGISTRY[606] = ShapleyVolumeValue
FORMULA_REGISTRY[607] = BlockSpaceOptimizer
FORMULA_REGISTRY[608] = VPINVolumeSync
FORMULA_REGISTRY[609] = AdaptiveParticipation
FORMULA_REGISTRY[610] = VolumeCaptureController

print(f"[VolumeCapture] Registered 10 volume capture formulas (601-610)")


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("VOLUME CAPTURE FORMULAS TEST (IDs 601-610)")
    print("=" * 70)

    # Test Volume Capture Controller
    controller = VolumeCaptureController(target_capture_pct=0.0001)

    # Simulate with $100 capital, $722K/sec volume
    capital = 100.0
    volume_per_sec = 722435.0
    price = 97000.0

    print(f"\nCapital: ${capital}")
    print(f"Volume/sec: ${volume_per_sec:,.0f}")
    print(f"Target participation: 0.01%")
    print("-" * 70)

    for i in range(10):
        result = controller.calculate(
            price=price + i * 10,
            capital=capital,
            volume_usd=volume_per_sec,
            bid_volume=volume_per_sec * 0.52,
            ask_volume=volume_per_sec * 0.48,
            volatility=0.01,
            spread=0.0005
        )

        print(f"[{i}] Signal: {result['signal']:+.3f} | "
              f"Dir: {result['direction']:4s} | "
              f"Pos: ${result['position_size']:.2f} | "
              f"Capture: ${result['estimated_capture']:.4f} | "
              f"Total: ${result['total_captured']:.4f}")

    print("-" * 70)
    print(f"Total Trades: {result['trade_count']}")
    print(f"Total Captured: ${result['total_captured']:.4f}")
    print("=" * 70)
