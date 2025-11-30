"""
Renaissance Formula Library - HFT Volume Formulas (F019-F024)
==============================================================
IDs 311-316: Critical missing volume formulas for HFT systems

These formulas were MISSING from our HFT system and explain why we weren't
getting enough trade signals despite having large order detection.

Research Sources:
- Kyle & Obizhaeva (2016): Market Microstructure Invariance
- Avellaneda & Stoikov (2008): Order arrival rate modeling
- Rosu (2009): Dynamic Model of the Limit Order Book
- Easley et al: Flow Toxicity in HFT

ID Mapping:
- 311: Volume Per Second (VPS) - F019
- 312: Order Arrival Rate (Poisson) - F020
- 313: Participation Rate - F021
- 314: Trading Activity (Kyle's W) - F022
- 315: Dynamic Flow Threshold - F023
- 316: Expected Trade Frequency - F024
"""

import numpy as np
from typing import Dict, Any, Optional
from collections import deque

from .base import BaseFormula, FormulaRegistry


@FormulaRegistry.register(311)
class VolumePerSecondFormula(BaseFormula):
    """
    ID 311: Volume Per Second (VPS) - F019

    CRITICAL FOUNDATION: The base for all volume normalization!

    Formula:
        VPS = ADV / (24 * 3600)

    Where:
        - VPS: Volume per second in BTC
        - ADV: Average Daily Volume (24h volume)
        - 86,400: Seconds in a day

    For BTC (typical values):
        ADV ~ 30,000 BTC/day
        VPS = 30,000 / 86,400 = 0.347 BTC/second

    WHY THIS WAS MISSING:
    We used a STATIC flow_threshold (0.25-1.0 BTC) but didn't normalize it to
    volume per second. Large orders are relative to current market activity!

    Authority x Verification x Relevance: Foundation (required for all others)
    """

    FORMULA_ID = 311
    CATEGORY = "hft_volume"
    NAME = "Volume Per Second (VPS)"
    DESCRIPTION = "Calculate volume per second for threshold normalization"

    SECONDS_PER_DAY = 86400

    def __init__(self, lookback: int = 100,
                 initial_daily_volume: float = 30000.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.daily_volume = initial_daily_volume  # In BTC
        self.current_vps = initial_daily_volume / self.SECONDS_PER_DAY

        # Track volume over time for dynamic estimation
        self.volume_samples = deque(maxlen=3600)  # 1 hour of 1-second samples
        self.sample_timestamps = deque(maxlen=3600)

        # Rolling averages
        self.vps_1min = 0.0
        self.vps_5min = 0.0
        self.vps_1hour = 0.0

    def _compute(self) -> None:
        """Compute VPS from recent volume data"""
        volumes = self._volumes_array()

        if len(volumes) < 2:
            return

        # Current VPS from most recent samples
        recent_volume = np.sum(volumes[-60:]) if len(volumes) >= 60 else np.sum(volumes)
        time_span = min(60, len(volumes))

        if time_span > 0:
            self.vps_1min = recent_volume / time_span

        # 5-minute VPS
        if len(volumes) >= 300:
            self.vps_5min = np.sum(volumes[-300:]) / 300
        else:
            self.vps_5min = self.vps_1min

        # Estimate current VPS
        self.current_vps = self.vps_1min if self.vps_1min > 0 else self.daily_volume / self.SECONDS_PER_DAY

        # Signal: High volume = more opportunities
        avg_vps = self.daily_volume / self.SECONDS_PER_DAY
        volume_ratio = self.current_vps / avg_vps if avg_vps > 0 else 1.0

        if volume_ratio > 2.0:  # Volume spike
            self.signal = 1
            self.confidence = min(0.9, 0.5 + 0.2 * (volume_ratio - 1))
        elif volume_ratio < 0.5:  # Low volume
            self.signal = -1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.5

    def set_daily_volume(self, volume_btc: float):
        """Update daily volume estimate"""
        self.daily_volume = volume_btc

    def get_vps(self) -> float:
        """Get current volume per second"""
        return self.current_vps

    def get_normalized_threshold(self, base_threshold: float) -> float:
        """Normalize a threshold to current volume conditions"""
        avg_vps = self.daily_volume / self.SECONDS_PER_DAY
        if avg_vps <= 0:
            return base_threshold
        return base_threshold * (self.current_vps / avg_vps)

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'daily_volume_btc': self.daily_volume,
            'current_vps': self.current_vps,
            'vps_1min': self.vps_1min,
            'vps_5min': self.vps_5min,
            'expected_vps': self.daily_volume / self.SECONDS_PER_DAY,
        })
        return state


@FormulaRegistry.register(312)
class OrderArrivalRateFormula(BaseFormula):
    """
    ID 312: Order Arrival Rate (Poisson Process) - F020

    CRITICAL FOR SIGNAL GENERATION!

    Formula:
        lambda(delta) = A * exp(-k * delta)

    Where:
        - lambda(delta): Arrival rate of orders at distance delta from mid
        - A: Base arrival rate (calibrated from data, ~10 orders/sec typical)
        - k: Decay parameter (how fast arrival drops with distance, ~0.5 typical)
        - delta: Distance from mid-price in basis points

    Sources:
        1. Avellaneda & Stoikov (2008) - Score: 10, Relevance: 10
        2. Rosu (2009): Dynamic LOB Model - Score: 9, Relevance: 10

    Authority x Verification x Relevance: 9.5 x 2 x 10 = 190

    WHY THIS WAS MISSING:
    Orders don't arrive uniformly - they follow an exponential decay from mid price.
    We need to weight orders by their probability of filling!
    """

    FORMULA_ID = 312
    CATEGORY = "hft_volume"
    NAME = "Order Arrival Rate (Poisson)"
    DESCRIPTION = "Model order arrival using Poisson process with exponential decay"

    def __init__(self, lookback: int = 100,
                 A_base: float = 10.0,
                 k_decay: float = 0.5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.A = A_base  # Base arrival rate at mid
        self.k = k_decay  # Decay constant

        # Calibration tracking
        self.observed_arrivals = deque(maxlen=1000)
        self.arrival_distances = deque(maxlen=1000)

        # Current estimates
        self.current_arrival_rate = A_base
        self.calibrated_A = A_base
        self.calibrated_k = k_decay

    def _compute(self) -> None:
        """Compute current arrival rate estimates"""
        # Calculate arrival rate at different distances
        # Near mid (0-2 bps): highest rate
        self.rate_at_mid = self.A
        self.rate_at_5bps = self.A * np.exp(-self.k * 5)
        self.rate_at_10bps = self.A * np.exp(-self.k * 10)

        # Current rate based on recent activity
        if len(self.observed_arrivals) > 10:
            recent = list(self.observed_arrivals)[-100:]
            self.current_arrival_rate = np.mean(recent)
        else:
            self.current_arrival_rate = self.A

        # Signal based on arrival intensity
        intensity_ratio = self.current_arrival_rate / self.A if self.A > 0 else 1.0

        if intensity_ratio > 1.5:  # High activity
            self.signal = 1
            self.confidence = 0.8
        elif intensity_ratio < 0.5:  # Low activity
            self.signal = -1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.5

    def compute_arrival_rate(self, distance_bps: float) -> float:
        """
        Compute order arrival rate at given distance from mid.

        Args:
            distance_bps: Distance from mid-price in basis points

        Returns:
            Expected arrival rate (orders per second)
        """
        return self.calibrated_A * np.exp(-self.calibrated_k * distance_bps)

    def record_arrival(self, rate: float, distance_bps: float):
        """Record observed arrival for calibration"""
        self.observed_arrivals.append(rate)
        self.arrival_distances.append(distance_bps)

    def calibrate(self):
        """Calibrate A and k from observed data"""
        if len(self.observed_arrivals) < 50:
            return

        arrivals = np.array(list(self.observed_arrivals))
        distances = np.array(list(self.arrival_distances))

        # Simple calibration: estimate A from near-mid arrivals
        near_mid = arrivals[distances < 5]
        if len(near_mid) > 10:
            self.calibrated_A = np.mean(near_mid)

        # Estimate k from decay rate
        far = arrivals[distances > 10]
        if len(far) > 10 and self.calibrated_A > 0:
            avg_far_rate = np.mean(far)
            avg_far_dist = np.mean(distances[distances > 10])
            if avg_far_rate > 0:
                self.calibrated_k = -np.log(avg_far_rate / self.calibrated_A) / avg_far_dist

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'A_base': self.A,
            'k_decay': self.k,
            'calibrated_A': self.calibrated_A,
            'calibrated_k': self.calibrated_k,
            'current_arrival_rate': self.current_arrival_rate,
            'rate_at_5bps': getattr(self, 'rate_at_5bps', 0),
            'rate_at_10bps': getattr(self, 'rate_at_10bps', 0),
        })
        return state


@FormulaRegistry.register(313)
class ParticipationRateFormula(BaseFormula):
    """
    ID 313: Participation Rate (PR) - F021

    CRITICAL FOR MARKET IMPACT!

    Formula:
        PR = Our_Order_Size / (VPS * Time_Window)

    Where:
        - PR: Participation rate (0 to 1)
        - Our_Order_Size: Size of our order in BTC
        - VPS: Volume per second
        - Time_Window: Time window in seconds

    Optimal Range: 1-5% participation rate
    Above 5% = too aggressive, market impact increases significantly

    Sources:
        1. Kyle-Obizhaeva (2016): Market Microstructure Invariance - Score: 10, Relevance: 10

    Authority x Verification x Relevance: 10 x 1 x 10 = 100

    WHY THIS WAS MISSING:
    We weren't sizing our trades relative to market volume!
    Trading 0.001 BTC in a 0.347 BTC/sec market is 0.3% participation - very safe.
    """

    FORMULA_ID = 313
    CATEGORY = "hft_volume"
    NAME = "Participation Rate"
    DESCRIPTION = "Calculate our participation as % of market volume"

    def __init__(self, lookback: int = 100,
                 order_size: float = 0.001,
                 target_participation: float = 0.01, **kwargs):
        super().__init__(lookback, **kwargs)
        self.order_size = order_size  # Our order size in BTC
        self.target_participation = target_participation  # Target 1%

        # VPS tracking (should be connected to ID 311)
        self.current_vps = 0.347  # Default BTC/sec

        # Computed values
        self.current_participation = 0.0
        self.is_safe = True
        self.recommended_size = order_size

    def _compute(self) -> None:
        """Compute participation rate"""
        volumes = self._volumes_array()

        if len(volumes) < 10:
            return

        # Estimate current VPS from volume data
        recent_vol = np.sum(volumes[-60:]) if len(volumes) >= 60 else np.sum(volumes)
        time_span = min(60, len(volumes))
        self.current_vps = recent_vol / time_span if time_span > 0 else 0.347

        # Calculate participation rate
        market_volume_1sec = self.current_vps
        if market_volume_1sec > 0:
            self.current_participation = self.order_size / market_volume_1sec
        else:
            self.current_participation = 1.0

        # Safety check
        self.is_safe = self.current_participation < 0.05  # Under 5%

        # Recommended size to hit target participation
        self.recommended_size = market_volume_1sec * self.target_participation

        # Signal: Safe to trade more?
        if self.current_participation < 0.01:  # Under 1%
            self.signal = 1  # Can increase size
            self.confidence = 0.9
        elif self.current_participation < 0.05:  # 1-5%
            self.signal = 0  # Good range
            self.confidence = 0.7
        else:  # Over 5%
            self.signal = -1  # Reduce size!
            self.confidence = 0.95

    def compute_participation(self, order_size: float, vps: float,
                             time_window: float = 1.0) -> float:
        """
        Calculate participation rate.

        Args:
            order_size: Our order size in BTC
            vps: Market volume per second
            time_window: Time window in seconds

        Returns:
            Participation rate (0 to 1)
        """
        market_volume = vps * time_window
        if market_volume == 0:
            return 1.0
        return min(order_size / market_volume, 1.0)

    def get_safe_order_size(self, vps: float, max_participation: float = 0.01) -> float:
        """Get maximum safe order size for given participation target"""
        return vps * max_participation

    def set_vps(self, vps: float):
        """Update VPS from external source"""
        self.current_vps = vps

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'order_size': self.order_size,
            'current_vps': self.current_vps,
            'current_participation': self.current_participation,
            'target_participation': self.target_participation,
            'is_safe': self.is_safe,
            'recommended_size': self.recommended_size,
        })
        return state


@FormulaRegistry.register(314)
class TradingActivityFormula(BaseFormula):
    """
    ID 314: Trading Activity (Kyle's W) - F022

    CRITICAL FOR MARKET REGIME DETECTION!

    Formula:
        W = V * sigma

    Where:
        - W: Trading activity (dollars)
        - V: Dollar volume per unit time
        - sigma: Returns volatility (standard deviation)

    Key Relationships (Kyle-Obizhaeva Invariance):
        Transaction_Cost proportional to W^(1/3)
        Order_Size proportional to W^(-2/3)
        Spread proportional to W^(-1/3)

    Sources:
        1. Kyle & Obizhaeva (2016): Market Microstructure Invariance - Score: 10, Relevance: 10

    Authority x Verification x Relevance: 10 x 1 x 10 = 100

    WHY THIS WAS MISSING:
    This ties volume AND volatility together. High volume + low vol = different
    market than low volume + high vol, even if price is the same!
    """

    FORMULA_ID = 314
    CATEGORY = "hft_volume"
    NAME = "Trading Activity (Kyle's W)"
    DESCRIPTION = "Calculate trading activity = Volume * Volatility"

    def __init__(self, lookback: int = 100,
                 reference_W: float = 1e9, **kwargs):
        super().__init__(lookback, **kwargs)
        self.W_reference = reference_W  # Reference for scaling

        # Current values
        self.current_W = 0.0
        self.current_volatility = 0.002  # 0.2% default
        self.dollar_volume_per_sec = 0.0

        # Derived scaling factors
        self.transaction_cost_factor = 1.0
        self.order_size_factor = 1.0
        self.spread_factor = 1.0

    def _compute(self) -> None:
        """Compute Kyle's Trading Activity"""
        prices = self._prices_array()
        volumes = self._volumes_array()
        returns = self._returns_array()

        if len(prices) < 20 or len(returns) < 10:
            return

        current_price = prices[-1]

        # Calculate current volatility
        self.current_volatility = np.std(returns) if len(returns) > 0 else 0.002

        # Calculate dollar volume per second
        recent_btc_volume = np.sum(volumes[-60:]) / 60 if len(volumes) >= 60 else np.mean(volumes)
        self.dollar_volume_per_sec = recent_btc_volume * current_price

        # Kyle's W
        self.current_W = self.dollar_volume_per_sec * self.current_volatility

        # Scaling factors from invariance relationships
        if self.W_reference > 0:
            W_ratio = self.current_W / self.W_reference
            self.transaction_cost_factor = W_ratio ** (1/3) if W_ratio > 0 else 1.0
            self.order_size_factor = W_ratio ** (-2/3) if W_ratio > 0 else 1.0
            self.spread_factor = W_ratio ** (-1/3) if W_ratio > 0 else 1.0

        # Signal based on activity regime
        if self.current_W > self.W_reference * 2:  # High activity
            self.signal = 1
            self.confidence = 0.8
        elif self.current_W < self.W_reference * 0.5:  # Low activity
            self.signal = -1
            self.confidence = 0.7
        else:
            self.signal = 0
            self.confidence = 0.5

    def compute_trading_activity(self, dollar_volume: float, volatility: float) -> float:
        """
        Calculate Kyle's Trading Activity (W).

        Args:
            dollar_volume: Dollar volume per time unit
            volatility: Returns volatility (decimal)

        Returns:
            Trading activity W
        """
        return dollar_volume * volatility

    def get_optimal_order_size_factor(self) -> float:
        """Get scaling factor for optimal order size"""
        return self.order_size_factor

    def get_expected_spread_factor(self) -> float:
        """Get expected spread scaling factor"""
        return self.spread_factor

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'current_W': self.current_W,
            'W_reference': self.W_reference,
            'dollar_volume_per_sec': self.dollar_volume_per_sec,
            'current_volatility': self.current_volatility,
            'transaction_cost_factor': self.transaction_cost_factor,
            'order_size_factor': self.order_size_factor,
            'spread_factor': self.spread_factor,
        })
        return state


@FormulaRegistry.register(315)
class DynamicFlowThresholdFormula(BaseFormula):
    """
    ID 315: Relative Flow Threshold (Dynamic) - F023

    CRITICAL FOR SIGNAL GENERATION!

    Formula:
        Dynamic_Threshold = base_threshold * (VPS_current / VPS_average) * sqrt(volatility_multiplier)

    Where:
        - Dynamic_Threshold: Adjusted threshold for current conditions
        - base_threshold: Static base (e.g., 0.25 BTC)
        - VPS_current: Current volume per second
        - VPS_average: Average VPS (24h average)
        - volatility_multiplier: Higher vol = higher threshold

    Sources:
        1. Flow Toxicity Model (Easley et al)
        2. Kyle Lambda Model

    Authority x Verification x Relevance: 8 x 2 x 10 = 160

    WHY THIS WAS MISSING:
    Our threshold was STATIC at 0.25-1.0 BTC regardless of market conditions!
    In low volume periods, 0.25 BTC is HUGE. In high volume, it's tiny.
    """

    FORMULA_ID = 315
    CATEGORY = "hft_volume"
    NAME = "Dynamic Flow Threshold"
    DESCRIPTION = "Calculate volume-adjusted flow threshold"

    def __init__(self, lookback: int = 100,
                 base_threshold: float = 0.25,
                 average_volatility: float = 0.002, **kwargs):
        super().__init__(lookback, **kwargs)
        self.base_threshold = base_threshold
        self.average_volatility = average_volatility

        # VPS tracking
        self.average_vps = 0.347  # Default: 30k BTC/day
        self.current_vps = 0.347
        self.vps_history = deque(maxlen=86400)  # 24h

        # Current threshold
        self.dynamic_threshold = base_threshold
        self.volume_multiplier = 1.0
        self.volatility_multiplier = 1.0

    def _compute(self) -> None:
        """Compute dynamic flow threshold"""
        volumes = self._volumes_array()
        returns = self._returns_array()

        if len(volumes) < 10:
            return

        # Current VPS
        self.current_vps = np.mean(volumes[-60:]) if len(volumes) >= 60 else np.mean(volumes)

        # Update VPS history
        if len(volumes) > 0:
            self.vps_history.append(volumes[-1])

        # Average VPS from history
        if len(self.vps_history) > 0:
            self.average_vps = np.mean(list(self.vps_history))

        # Current volatility
        current_volatility = np.std(returns) if len(returns) > 0 else self.average_volatility

        # Calculate multipliers
        self.volume_multiplier = self.current_vps / self.average_vps if self.average_vps > 0 else 1.0
        self.volatility_multiplier = current_volatility / self.average_volatility if self.average_volatility > 0 else 1.0

        # Dynamic threshold
        self.dynamic_threshold = (
            self.base_threshold *
            self.volume_multiplier *
            np.sqrt(self.volatility_multiplier)
        )

        # Floor: never go below 10% of base
        self.dynamic_threshold = max(self.dynamic_threshold, self.base_threshold * 0.1)

        # Signal based on threshold vs base
        threshold_ratio = self.dynamic_threshold / self.base_threshold

        if threshold_ratio > 1.5:  # High threshold = harder to trigger
            self.signal = -1  # Be more selective
            self.confidence = 0.7
        elif threshold_ratio < 0.5:  # Low threshold = easier to trigger
            self.signal = 1  # More opportunities
            self.confidence = 0.8
        else:
            self.signal = 0
            self.confidence = 0.5

    def compute_dynamic_threshold(self, base_threshold: float,
                                  current_vps: float, average_vps: float,
                                  current_volatility: float,
                                  average_volatility: float = 0.002) -> float:
        """
        Calculate dynamic flow threshold.

        Args:
            base_threshold: Base threshold in BTC
            current_vps: Current volume per second
            average_vps: Average VPS (24h rolling)
            current_volatility: Current volatility
            average_volatility: Average volatility (typically 0.2%)

        Returns:
            Adjusted threshold in BTC
        """
        volume_mult = current_vps / average_vps if average_vps > 0 else 1.0
        vol_mult = current_volatility / average_volatility if average_volatility > 0 else 1.0

        dynamic_threshold = base_threshold * volume_mult * np.sqrt(vol_mult)

        # Floor
        return max(dynamic_threshold, base_threshold * 0.1)

    def get_threshold(self) -> float:
        """Get current dynamic threshold"""
        return self.dynamic_threshold

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'base_threshold': self.base_threshold,
            'dynamic_threshold': self.dynamic_threshold,
            'current_vps': self.current_vps,
            'average_vps': self.average_vps,
            'volume_multiplier': self.volume_multiplier,
            'volatility_multiplier': self.volatility_multiplier,
        })
        return state


@FormulaRegistry.register(316)
class ExpectedTradeFrequencyFormula(BaseFormula):
    """
    ID 316: Expected Trade Frequency (ETF) - F024

    CRITICAL FOR STRATEGY TUNING!

    Formula:
        ETF = lambda_total * P(signal) * P(fill)

    Where:
        - ETF: Expected trades per time unit
        - lambda_total: Total order arrival rate
        - P(signal): Probability our signal triggers
        - P(fill): Probability our order fills

    For Limit Orders:
        P(fill) = 1 - exp(-lambda_opp * T)

    Where:
        - lambda_opp = opposite side order arrival rate
        - T = time in queue

    Sources:
        1. Avellaneda & Stoikov (2008)
        2. Rosu (2009): LOB Model

    Authority x Verification x Relevance: 9 x 2 x 10 = 180

    WHY THIS WAS MISSING:
    We had NO estimate of how many trades we SHOULD be making!
    This helps us tune thresholds to hit target trade frequency.
    """

    FORMULA_ID = 316
    CATEGORY = "hft_volume"
    NAME = "Expected Trade Frequency"
    DESCRIPTION = "Calculate expected trades per time unit"

    def __init__(self, lookback: int = 100,
                 target_trades_per_minute: float = 1.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.target_tpm = target_trades_per_minute

        # Arrival rate (from ID 312)
        self.order_arrival_rate = 10.0  # Orders/sec at mid

        # Signal probability (how often our signal fires)
        self.signal_probability = 0.1  # 10% default

        # Fill probability
        self.fill_probability = 0.9
        self.time_in_queue = 0.5  # seconds

        # Computed values
        self.expected_tpm = 0.0
        self.expected_tps = 0.0

    def _compute(self) -> None:
        """Compute expected trade frequency"""
        # ETF = arrival_rate * P(signal) * P(fill)
        self.expected_tps = (
            self.order_arrival_rate *
            self.signal_probability *
            self.fill_probability
        )

        self.expected_tpm = self.expected_tps * 60

        # Signal based on expected vs target
        frequency_ratio = self.expected_tpm / self.target_tpm if self.target_tpm > 0 else 1.0

        if frequency_ratio < 0.5:  # Below target
            self.signal = 1  # Need to lower thresholds
            self.confidence = 0.8
        elif frequency_ratio > 2.0:  # Above target
            self.signal = -1  # Need to raise thresholds
            self.confidence = 0.7
        else:
            self.signal = 0
            self.confidence = 0.6

    def compute_expected_frequency(self, order_arrival_rate: float,
                                   signal_probability: float,
                                   fill_probability: float) -> float:
        """
        Calculate expected trade frequency.

        Args:
            order_arrival_rate: Orders arriving per second
            signal_probability: P(our signal fires)
            fill_probability: P(order fills given signal)

        Returns:
            Expected trades per second
        """
        return order_arrival_rate * signal_probability * fill_probability

    def compute_fill_probability(self, opposite_arrival_rate: float,
                                 time_in_queue: float) -> float:
        """
        Calculate fill probability for limit order.

        Args:
            opposite_arrival_rate: Arrival rate of opposite side orders
            time_in_queue: Expected time in queue (seconds)

        Returns:
            Probability of fill
        """
        return 1 - np.exp(-opposite_arrival_rate * time_in_queue)

    def set_arrival_rate(self, rate: float):
        """Set order arrival rate (from ID 312)"""
        self.order_arrival_rate = rate

    def set_signal_probability(self, prob: float):
        """Set signal trigger probability"""
        self.signal_probability = prob

    def tune_for_target(self) -> Dict[str, float]:
        """Calculate parameters needed to hit target frequency"""
        if self.expected_tpm == 0:
            return {}

        ratio = self.target_tpm / self.expected_tpm

        return {
            'required_signal_prob': min(1.0, self.signal_probability * ratio),
            'required_fill_prob': min(1.0, self.fill_probability * ratio),
            'adjustment_factor': ratio,
        }

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'expected_tps': self.expected_tps,
            'expected_tpm': self.expected_tpm,
            'target_tpm': self.target_tpm,
            'order_arrival_rate': self.order_arrival_rate,
            'signal_probability': self.signal_probability,
            'fill_probability': self.fill_probability,
            'time_in_queue': self.time_in_queue,
        })
        return state


# =============================================================================
# HFT Volume Aggregator - Combines all F019-F024
# =============================================================================

class HFTVolumeAggregator:
    """
    Combines all HFT volume formulas (F019-F024) for unified access.

    This aggregator manages:
    - VPS (311): Volume per second for normalization
    - Order Arrival Rate (312): Poisson process modeling
    - Participation Rate (313): Market impact estimation
    - Trading Activity (314): Kyle's W for regime detection
    - Dynamic Threshold (315): Volume-adjusted signals
    - Expected Frequency (316): Trade frequency estimation

    Usage:
        agg = HFTVolumeAggregator()
        agg.update(price, volume, timestamp)
        config = agg.get_optimal_config()
    """

    def __init__(self, daily_volume_btc: float = 30000.0,
                 base_threshold: float = 0.25,
                 order_size: float = 0.001):
        # Initialize all formulas
        self.vps = VolumePerSecondFormula(initial_daily_volume=daily_volume_btc)
        self.arrival = OrderArrivalRateFormula()
        self.participation = ParticipationRateFormula(order_size=order_size)
        self.trading_activity = TradingActivityFormula()
        self.dynamic_threshold = DynamicFlowThresholdFormula(base_threshold=base_threshold)
        self.expected_frequency = ExpectedTradeFrequencyFormula()

        self.daily_volume = daily_volume_btc
        self.base_threshold = base_threshold

    def update(self, price: float, volume: float = 0.0, timestamp: float = 0.0):
        """Update all formulas with new data"""
        self.vps.update(price, volume, timestamp)
        self.arrival.update(price, volume, timestamp)
        self.participation.update(price, volume, timestamp)
        self.trading_activity.update(price, volume, timestamp)
        self.dynamic_threshold.update(price, volume, timestamp)
        self.expected_frequency.update(price, volume, timestamp)

        # Cross-update dependencies
        current_vps = self.vps.get_vps()
        self.participation.set_vps(current_vps)
        self.expected_frequency.set_arrival_rate(self.arrival.current_arrival_rate)

    def get_optimal_config(self) -> Dict[str, Any]:
        """Get optimal trading configuration based on all volume formulas"""
        return {
            # VPS
            'current_vps': self.vps.current_vps,
            'vps_ratio': self.vps.current_vps / (self.daily_volume / 86400) if self.daily_volume > 0 else 1.0,

            # Arrival Rate
            'order_arrival_rate': self.arrival.current_arrival_rate,
            'rate_at_5bps': self.arrival.compute_arrival_rate(5),

            # Participation
            'participation_rate': self.participation.current_participation,
            'is_safe_participation': self.participation.is_safe,
            'recommended_order_size': self.participation.recommended_size,

            # Trading Activity
            'kyle_W': self.trading_activity.current_W,
            'order_size_factor': self.trading_activity.order_size_factor,
            'spread_factor': self.trading_activity.spread_factor,

            # Dynamic Threshold
            'dynamic_threshold': self.dynamic_threshold.dynamic_threshold,
            'threshold_multiplier': self.dynamic_threshold.dynamic_threshold / self.base_threshold,

            # Expected Frequency
            'expected_trades_per_minute': self.expected_frequency.expected_tpm,
            'expected_trades_per_second': self.expected_frequency.expected_tps,
        }

    def should_increase_trading(self) -> bool:
        """Determine if conditions favor more trading"""
        signals = [
            self.vps.signal,
            self.arrival.signal,
            self.participation.signal,
            self.trading_activity.signal,
        ]
        return sum(signals) > 1

    def get_all_states(self) -> Dict[str, Dict]:
        """Get state from all formulas"""
        return {
            'vps_311': self.vps.get_state(),
            'arrival_312': self.arrival.get_state(),
            'participation_313': self.participation.get_state(),
            'trading_activity_314': self.trading_activity.get_state(),
            'dynamic_threshold_315': self.dynamic_threshold.get_state(),
            'expected_frequency_316': self.expected_frequency.get_state(),
        }
