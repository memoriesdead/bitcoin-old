"""
Renaissance Formula Library - Volume Scaling & Edge Amplification
==================================================================
IDs 295-299: The MISSING VARIABLES for capturing market volume

With $66.89 BILLION daily BTC volume, these formulas scale edge capture.

CRITICAL MISSING VARIABLES IDENTIFIED:
1. Market Depth Coefficient (MDC) - How much liquidity we can tap
2. Velocity of Capital (VOC) - How fast we turn over capital
3. Volume Capture Rate (VCR) - What % of market volume we capture
4. Edge Amplification Factor (EAF) - Grinold-Kahn √BR scaling
5. Liquidity-Adjusted Leverage (LAL) - Safe leverage based on depth

Mathematical Foundation:
- Grinold-Kahn: IR = IC * sqrt(BR) - More trades = exponential edge
- Kelly: f* = (p*b - q) / b - Optimal sizing
- Kyle Lambda: price_impact = lambda * order_flow - Market impact model
"""

import numpy as np
from typing import Dict, Any, Optional, List
from collections import deque

from .base import BaseFormula, FormulaRegistry


@FormulaRegistry.register(295)
class MarketDepthCoefficientFormula(BaseFormula):
    """
    ID 295: Market Depth Coefficient (MDC)

    MISSING VARIABLE #1: We need to know how much liquidity is available!

    MDC = Available_Depth_at_Target_Slippage / Our_Trade_Size

    If MDC > 100: Market can absorb 100x our order with minimal impact
    If MDC > 10:  Safe to increase position size 10x
    If MDC < 1:   Our order will cause slippage - reduce size!

    Example with $66B daily volume:
    - BTC average depth at 0.1%: ~$10M on major exchanges
    - Our capital: $10
    - MDC = $10M / $10 = 1,000,000x headroom!
    - We COULD safely trade $100,000 (1% of depth)
    """

    FORMULA_ID = 295
    CATEGORY = "volume_scaling"
    NAME = "Market Depth Coefficient"
    DESCRIPTION = "Calculate safe position size based on available liquidity"

    def __init__(self, lookback: int = 100, target_slippage_bps: float = 10,
                 daily_volume_usd: float = 66_888_130_238, **kwargs):
        super().__init__(lookback, **kwargs)
        self.target_slippage = target_slippage_bps / 10000  # 0.1% = 10 bps
        self.daily_volume = daily_volume_usd

        # Depth estimation from volume (empirical: depth ~ 0.5-2% of daily volume at 0.1%)
        self.depth_ratio = kwargs.get('depth_ratio', 0.01)  # 1% of daily volume at 0.1%

        # Current estimates
        self.estimated_depth_usd = 0.0
        self.mdc = 1.0
        self.max_safe_trade_usd = 0.0
        self.depth_utilization_pct = 0.0

        # Our position
        self.our_trade_size = kwargs.get('trade_size', 10.0)

    def _compute(self) -> None:
        """Compute Market Depth Coefficient"""
        prices = self._prices_array()
        volumes = self._volumes_array()

        if len(prices) < 2:
            return

        current_price = prices[-1]

        # Estimate available depth from daily volume
        # Empirical: On major exchanges, depth at 0.1% ~ 0.5-2% of daily volume
        self.estimated_depth_usd = self.daily_volume * self.depth_ratio

        # Calculate MDC
        if self.our_trade_size > 0:
            self.mdc = self.estimated_depth_usd / self.our_trade_size
        else:
            self.mdc = float('inf')

        # Safe trade size = 1% of depth (to minimize impact)
        self.max_safe_trade_usd = self.estimated_depth_usd * 0.01

        # How much of available depth are we using?
        self.depth_utilization_pct = (self.our_trade_size / self.estimated_depth_usd) * 100 if self.estimated_depth_usd > 0 else 0

        # Signal: If we're vastly underutilizing depth, signal to scale up
        if self.mdc > 10000:  # Huge headroom
            self.signal = 1  # SCALE UP!
            self.confidence = 1.0
        elif self.mdc > 100:  # Good headroom
            self.signal = 1
            self.confidence = 0.8
        elif self.mdc > 10:   # Moderate headroom
            self.signal = 1
            self.confidence = 0.5
        elif self.mdc < 1:    # Danger - we'll cause impact
            self.signal = -1  # SCALE DOWN!
            self.confidence = 1.0
        else:
            self.signal = 0
            self.confidence = 0.3

    def get_recommended_size(self, current_size: float) -> float:
        """Get recommended trade size based on MDC"""
        if self.mdc > 10000:
            return min(current_size * 1000, self.max_safe_trade_usd)
        elif self.mdc > 100:
            return min(current_size * 100, self.max_safe_trade_usd)
        elif self.mdc > 10:
            return min(current_size * 10, self.max_safe_trade_usd)
        elif self.mdc < 1:
            return current_size * 0.1  # Reduce!
        return current_size

    def set_daily_volume(self, volume_usd: float):
        """Update daily volume (call with fresh data)"""
        self.daily_volume = volume_usd

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'mdc': self.mdc,
            'estimated_depth_usd': self.estimated_depth_usd,
            'max_safe_trade_usd': self.max_safe_trade_usd,
            'depth_utilization_pct': self.depth_utilization_pct,
            'daily_volume': self.daily_volume
        })
        return state


@FormulaRegistry.register(296)
class VelocityOfCapitalFormula(BaseFormula):
    """
    ID 296: Velocity of Capital (VOC)

    MISSING VARIABLE #2: How fast can we turn over our capital?

    VOC = (Trade_Frequency_per_Hour * Avg_Position_Size) / Capital

    Current System:
    - 6 trades/min = 360 trades/hour
    - $1 per trade / $10 capital = 0.1x per trade
    - VOC = 360 * 0.1 = 36x/hour

    Target System (HFT):
    - 1200 trades/min = 72,000 trades/hour
    - $10 per trade / $10 capital = 1x per trade
    - VOC = 72,000 * 1.0 = 72,000x/hour

    Edge Scaling: Edge_per_Hour = Base_Edge * VOC
    - At 36x VOC: 0.019% * 36 = 0.68%/hour
    - At 72,000x VOC: 0.019% * 72000 = 1368%/hour (theoretical max)
    """

    FORMULA_ID = 296
    CATEGORY = "volume_scaling"
    NAME = "Velocity of Capital"
    DESCRIPTION = "Calculate capital turnover rate for edge amplification"

    def __init__(self, lookback: int = 100, capital: float = 10.0,
                 target_voc: float = 1000.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.capital = capital
        self.target_voc = target_voc  # Target turnover per hour

        # Trade tracking
        self.trade_timestamps = deque(maxlen=10000)
        self.trade_sizes = deque(maxlen=10000)

        # Computed values
        self.trades_per_hour = 0.0
        self.avg_trade_size = 0.0
        self.current_voc = 0.0
        self.voc_ratio = 0.0  # current/target
        self.hourly_edge_multiplier = 1.0

    def _compute(self) -> None:
        """Compute Velocity of Capital"""
        if len(self.trade_timestamps) < 2:
            return

        # Calculate trades per hour
        timestamps = list(self.trade_timestamps)
        if len(timestamps) >= 2:
            time_span_hours = (timestamps[-1] - timestamps[0]) / 3600 if timestamps[-1] != timestamps[0] else 1/3600
            self.trades_per_hour = len(timestamps) / time_span_hours if time_span_hours > 0 else 0

        # Average trade size
        if len(self.trade_sizes) > 0:
            self.avg_trade_size = np.mean(list(self.trade_sizes))

        # Calculate VOC
        if self.capital > 0:
            position_turnover = self.avg_trade_size / self.capital
            self.current_voc = self.trades_per_hour * position_turnover
        else:
            self.current_voc = 0

        # VOC ratio (how close to target)
        self.voc_ratio = self.current_voc / self.target_voc if self.target_voc > 0 else 0

        # Hourly edge multiplier
        self.hourly_edge_multiplier = self.current_voc

        # Signal based on VOC
        if self.voc_ratio < 0.1:  # Way below target
            self.signal = 1  # TRADE MORE!
            self.confidence = 1.0
        elif self.voc_ratio < 0.5:
            self.signal = 1
            self.confidence = 0.7
        elif self.voc_ratio > 2.0:  # Above target - watch for overtrading
            self.signal = -1
            self.confidence = 0.5
        else:
            self.signal = 0
            self.confidence = 0.3

    def record_trade(self, timestamp: float, size: float):
        """Record a trade for VOC calculation"""
        self.trade_timestamps.append(timestamp)
        self.trade_sizes.append(size)

    def get_required_trades_per_minute(self) -> float:
        """How many trades/min needed to hit target VOC"""
        if self.capital > 0 and self.avg_trade_size > 0:
            position_turnover = self.avg_trade_size / self.capital
            required_per_hour = self.target_voc / position_turnover if position_turnover > 0 else 0
            return required_per_hour / 60
        return 0

    def get_edge_per_hour(self, base_edge_pct: float) -> float:
        """Calculate expected edge per hour given base edge"""
        return base_edge_pct * self.hourly_edge_multiplier

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'current_voc': self.current_voc,
            'target_voc': self.target_voc,
            'voc_ratio': self.voc_ratio,
            'trades_per_hour': self.trades_per_hour,
            'avg_trade_size': self.avg_trade_size,
            'hourly_edge_multiplier': self.hourly_edge_multiplier
        })
        return state


@FormulaRegistry.register(297)
class VolumeCaptureRateFormula(BaseFormula):
    """
    ID 297: Volume Capture Rate (VCR)

    MISSING VARIABLE #3: What % of market volume are we capturing?

    VCR = Our_Trading_Volume / Market_Volume

    With $66.89B daily volume:
    - Per minute: $66.89B / 1440 = $46.45M/minute
    - Per second: $774,189/second

    Current system:
    - 6 trades/min * $1 = $6/minute
    - VCR = $6 / $46.45M = 0.000013%

    Target (still tiny but profitable):
    - Capture 0.0001% = $46.45/minute = $66,888/day
    - At 0.019% edge = $12.71/day profit on $46.45 volume

    Scalable target:
    - Capture 0.001% = $464.5/minute = $668,881/day volume
    - At 0.019% edge = $127.09/day profit
    """

    FORMULA_ID = 297
    CATEGORY = "volume_scaling"
    NAME = "Volume Capture Rate"
    DESCRIPTION = "Calculate what percentage of market volume we capture"

    def __init__(self, lookback: int = 100,
                 daily_market_volume: float = 66_888_130_238,
                 target_vcr_pct: float = 0.0001, **kwargs):
        super().__init__(lookback, **kwargs)
        self.daily_market_volume = daily_market_volume
        self.target_vcr = target_vcr_pct / 100  # Convert to decimal

        # Our volume tracking
        self.our_volume_per_minute = deque(maxlen=1440)  # Track by minute
        self.minute_volumes = {}

        # Computed values
        self.market_volume_per_minute = daily_market_volume / 1440
        self.market_volume_per_second = daily_market_volume / 86400
        self.current_vcr = 0.0
        self.vcr_ratio = 0.0  # current/target
        self.volume_gap_usd = 0.0  # How much more volume we need

    def _compute(self) -> None:
        """Compute Volume Capture Rate"""
        if len(self.our_volume_per_minute) < 1:
            return

        # Average our volume per minute
        our_avg_per_min = np.mean(list(self.our_volume_per_minute)) if len(self.our_volume_per_minute) > 0 else 0

        # Calculate VCR
        self.current_vcr = our_avg_per_min / self.market_volume_per_minute if self.market_volume_per_minute > 0 else 0

        # VCR ratio
        self.vcr_ratio = self.current_vcr / self.target_vcr if self.target_vcr > 0 else 0

        # Volume gap
        target_volume_per_min = self.market_volume_per_minute * self.target_vcr
        self.volume_gap_usd = target_volume_per_min - our_avg_per_min

        # Signal based on VCR
        if self.vcr_ratio < 0.01:  # Capturing almost nothing
            self.signal = 1  # SCALE UP MASSIVELY
            self.confidence = 1.0
        elif self.vcr_ratio < 0.1:
            self.signal = 1
            self.confidence = 0.9
        elif self.vcr_ratio < 0.5:
            self.signal = 1
            self.confidence = 0.6
        elif self.vcr_ratio > 1.0:  # At or above target
            self.signal = 0  # Good!
            self.confidence = 0.8
        else:
            self.signal = 0
            self.confidence = 0.5

    def record_volume(self, minute_key: int, volume_usd: float):
        """Record our trading volume for a minute"""
        if minute_key in self.minute_volumes:
            self.minute_volumes[minute_key] += volume_usd
        else:
            self.minute_volumes[minute_key] = volume_usd

        # Update deque with latest minute
        self.our_volume_per_minute.append(volume_usd)

    def get_required_volume_per_minute(self) -> float:
        """Volume per minute needed to hit target VCR"""
        return self.market_volume_per_minute * self.target_vcr

    def get_daily_profit_at_target(self, edge_pct: float) -> float:
        """Calculate daily profit if we hit target VCR"""
        daily_target_volume = self.daily_market_volume * self.target_vcr
        return daily_target_volume * (edge_pct / 100)

    def set_market_volume(self, daily_volume_usd: float):
        """Update market volume"""
        self.daily_market_volume = daily_volume_usd
        self.market_volume_per_minute = daily_volume_usd / 1440
        self.market_volume_per_second = daily_volume_usd / 86400

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'current_vcr': self.current_vcr * 100,  # As percentage
            'target_vcr_pct': self.target_vcr * 100,
            'vcr_ratio': self.vcr_ratio,
            'volume_gap_usd': self.volume_gap_usd,
            'market_volume_per_minute': self.market_volume_per_minute,
            'required_volume_per_minute': self.get_required_volume_per_minute()
        })
        return state


@FormulaRegistry.register(298)
class EdgeAmplificationFormula(BaseFormula):
    """
    ID 298: Edge Amplification Factor (EAF) - GRINOLD-KAHN LAW

    MISSING VARIABLE #4: The most important one!

    FUNDAMENTAL LAW OF ACTIVE MANAGEMENT:
    IR = IC * sqrt(BR)

    Where:
    - IR = Information Ratio (risk-adjusted return)
    - IC = Information Coefficient (forecast accuracy, ~0.05-0.1 for us)
    - BR = Breadth (number of independent bets/trades per year)

    The MAGIC: Edge scales with SQUARE ROOT of trade count!

    Current system (example):
    - IC = 0.05 (5% forecast accuracy above random)
    - BR = 6 trades/min * 60 * 24 * 365 = 3,153,600 trades/year
    - IR = 0.05 * sqrt(3,153,600) = 0.05 * 1776 = 88.8!

    Target HFT system:
    - IC = 0.05
    - BR = 1200 trades/min * 60 * 24 * 365 = 630,720,000 trades/year
    - IR = 0.05 * sqrt(630,720,000) = 0.05 * 25,114 = 1,256!

    EDGE AMPLIFICATION:
    - More trades = exponentially better risk-adjusted returns
    - At 1M trades/year with 51% WR: IR = 0.02 * 1000 = 20 (exceptional)
    """

    FORMULA_ID = 298
    CATEGORY = "volume_scaling"
    NAME = "Edge Amplification (Grinold-Kahn)"
    DESCRIPTION = "Apply Fundamental Law: IR = IC * sqrt(BR)"

    def __init__(self, lookback: int = 100, base_ic: float = 0.05,
                 target_ir: float = 2.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.base_ic = base_ic  # Information coefficient (forecast accuracy)
        self.target_ir = target_ir  # Target information ratio

        # Trade tracking
        self.trade_count = 0
        self.winning_trades = 0
        self.trades_per_year = 0

        # Computed values
        self.realized_ic = 0.0
        self.breadth = 0
        self.expected_ir = 0.0
        self.amplification_factor = 1.0
        self.required_trades_for_target = 0

    def _compute(self) -> None:
        """Compute Edge Amplification using Grinold-Kahn Law"""
        if self.trade_count < 10:
            return

        # Calculate realized IC from win rate
        win_rate = self.winning_trades / self.trade_count if self.trade_count > 0 else 0.5
        # IC approximation: 2 * (WR - 0.5) for binary outcomes
        self.realized_ic = 2 * (win_rate - 0.5)

        # Use better of realized or assumed IC
        effective_ic = max(self.realized_ic, self.base_ic)

        # Breadth = annual trade count
        self.breadth = self.trades_per_year

        # FUNDAMENTAL LAW: IR = IC * sqrt(BR)
        if self.breadth > 0:
            self.expected_ir = effective_ic * np.sqrt(self.breadth)
        else:
            self.expected_ir = 0

        # Amplification factor (compared to single trade)
        self.amplification_factor = np.sqrt(self.breadth) if self.breadth > 0 else 1

        # Required trades to hit target IR
        if effective_ic > 0:
            self.required_trades_for_target = int((self.target_ir / effective_ic) ** 2)
        else:
            self.required_trades_for_target = float('inf')

        # Signal based on IR
        if self.expected_ir < self.target_ir * 0.1:  # Way below target
            self.signal = 1  # NEED MORE TRADES!
            self.confidence = 1.0
        elif self.expected_ir < self.target_ir * 0.5:
            self.signal = 1
            self.confidence = 0.7
        elif self.expected_ir >= self.target_ir:  # At target
            self.signal = 0  # Good!
            self.confidence = 0.9
        else:
            self.signal = 0
            self.confidence = 0.5

    def record_trade(self, won: bool):
        """Record a trade outcome"""
        self.trade_count += 1
        if won:
            self.winning_trades += 1

    def set_annual_trades(self, trades: int):
        """Set expected annual trade count"""
        self.trades_per_year = trades

    def get_required_trades_per_minute(self) -> float:
        """Trades per minute needed for target IR"""
        return self.required_trades_for_target / (365 * 24 * 60)

    def get_edge_multiplier(self) -> float:
        """How much does breadth multiply our edge?"""
        return self.amplification_factor

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'realized_ic': self.realized_ic,
            'base_ic': self.base_ic,
            'breadth': self.breadth,
            'expected_ir': self.expected_ir,
            'target_ir': self.target_ir,
            'amplification_factor': self.amplification_factor,
            'required_trades_for_target': self.required_trades_for_target,
            'win_rate': self.winning_trades / self.trade_count if self.trade_count > 0 else 0.5
        })
        return state


@FormulaRegistry.register(299)
class LiquidityAdjustedLeverageFormula(BaseFormula):
    """
    ID 299: Liquidity-Adjusted Leverage (LAL)

    MISSING VARIABLE #5: Safe leverage based on market depth!

    LAL = Market_Depth / (Base_Position * Max_Slippage)

    With $66.89B daily volume and ~$668M depth at 0.1%:
    - For $10 capital: LAL = $668M / ($10 * 0.001) = 66,800,000x max!
    - Safe leverage: min(LAL/1000, exchange_max) = min(66800, 125) = 125x

    Position Scaling:
    - Unleveraged $10 at 0.019% edge = $0.0019/trade
    - 10x leverage: $100 effective, = $0.019/trade
    - 100x leverage: $1000 effective = $0.19/trade
    - 125x leverage (max): $1250 effective = $0.2375/trade

    Daily with 1000 trades:
    - 1x: $1.90/day
    - 10x: $19/day
    - 100x: $190/day
    - 125x: $237.50/day (from $10 capital!)
    """

    FORMULA_ID = 299
    CATEGORY = "volume_scaling"
    NAME = "Liquidity-Adjusted Leverage"
    DESCRIPTION = "Calculate safe leverage based on market liquidity"

    def __init__(self, lookback: int = 100,
                 daily_volume_usd: float = 66_888_130_238,
                 max_exchange_leverage: float = 125.0,
                 target_slippage_pct: float = 0.1, **kwargs):
        super().__init__(lookback, **kwargs)
        self.daily_volume = daily_volume_usd
        self.max_exchange_leverage = max_exchange_leverage
        self.target_slippage = target_slippage_pct / 100  # 0.1% = 0.001

        # Depth estimation (1% of daily volume at 0.1%)
        self.depth_ratio = kwargs.get('depth_ratio', 0.01)

        # Current values
        self.estimated_depth = 0.0
        self.theoretical_max_lal = 0.0
        self.safe_leverage = 1.0
        self.recommended_leverage = 1.0
        self.effective_position_size = 0.0

        # Capital
        self.base_capital = kwargs.get('capital', 10.0)

    def _compute(self) -> None:
        """Compute Liquidity-Adjusted Leverage"""
        prices = self._prices_array()

        if len(prices) < 2:
            return

        current_price = prices[-1]

        # Estimate market depth
        self.estimated_depth = self.daily_volume * self.depth_ratio

        # Calculate theoretical max LAL
        min_order_impact = self.base_capital * self.target_slippage
        if min_order_impact > 0:
            self.theoretical_max_lal = self.estimated_depth / min_order_impact
        else:
            self.theoretical_max_lal = float('inf')

        # Safe leverage (1/1000th of theoretical, capped by exchange)
        safe_from_depth = self.theoretical_max_lal / 1000
        self.safe_leverage = min(safe_from_depth, self.max_exchange_leverage)

        # Recommended leverage (more conservative: 1/10000th)
        recommended_from_depth = self.theoretical_max_lal / 10000
        self.recommended_leverage = min(recommended_from_depth, self.max_exchange_leverage * 0.5)
        self.recommended_leverage = max(1.0, self.recommended_leverage)  # At least 1x

        # Effective position size with recommended leverage
        self.effective_position_size = self.base_capital * self.recommended_leverage

        # Signal based on available leverage headroom
        leverage_utilization = self.recommended_leverage / self.max_exchange_leverage

        if leverage_utilization < 0.1:  # Using less than 10% of available
            self.signal = 1  # CAN USE MORE LEVERAGE
            self.confidence = 0.9
        elif leverage_utilization < 0.5:
            self.signal = 1
            self.confidence = 0.6
        elif leverage_utilization > 0.9:  # Near max
            self.signal = -1  # CAUTION
            self.confidence = 0.8
        else:
            self.signal = 0
            self.confidence = 0.5

    def get_effective_capital(self) -> float:
        """Get effective capital with recommended leverage"""
        return self.base_capital * self.recommended_leverage

    def get_edge_per_trade(self, base_edge_pct: float) -> float:
        """Get edge per trade with leverage applied"""
        return self.effective_position_size * (base_edge_pct / 100)

    def get_daily_profit(self, base_edge_pct: float, trades_per_day: int) -> float:
        """Calculate daily profit with leverage"""
        edge_per_trade = self.get_edge_per_trade(base_edge_pct)
        return edge_per_trade * trades_per_day

    def set_capital(self, capital: float):
        """Update base capital"""
        self.base_capital = capital

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'estimated_depth_usd': self.estimated_depth,
            'theoretical_max_lal': self.theoretical_max_lal,
            'safe_leverage': self.safe_leverage,
            'recommended_leverage': self.recommended_leverage,
            'max_exchange_leverage': self.max_exchange_leverage,
            'effective_position_size': self.effective_position_size,
            'base_capital': self.base_capital
        })
        return state


# ============================================================================
# MASTER SCALER: Combines all 5 missing variables
# ============================================================================

class VolumeScalingAggregator:
    """
    MASTER FORMULA: Combines all 5 missing variables to "infinity" the edge.

    Input: $10 capital, 0.019% base edge, $66.89B daily BTC volume

    Process:
    1. MDC: Determine we can safely trade much larger
    2. VOC: Calculate required trade frequency for target
    3. VCR: Target specific market volume capture %
    4. EAF: Apply Grinold-Kahn amplification
    5. LAL: Apply safe leverage

    Output: Optimal position size, trade frequency, expected daily profit
    """

    def __init__(self, capital: float = 10.0,
                 daily_market_volume: float = 66_888_130_238,
                 base_edge_pct: float = 0.019):
        self.capital = capital
        self.daily_volume = daily_market_volume
        self.base_edge = base_edge_pct

        # Initialize all 5 formulas
        self.mdc = MarketDepthCoefficientFormula(
            daily_volume_usd=daily_market_volume,
            trade_size=capital
        )
        self.voc = VelocityOfCapitalFormula(capital=capital)
        self.vcr = VolumeCaptureRateFormula(daily_market_volume=daily_market_volume)
        self.eaf = EdgeAmplificationFormula()
        self.lal = LiquidityAdjustedLeverageFormula(
            daily_volume_usd=daily_market_volume,
            capital=capital
        )

    def update(self, price: float, volume: float = 0.0, timestamp: float = 0.0):
        """Update all formulas"""
        self.mdc.update(price, volume, timestamp)
        self.voc.update(price, volume, timestamp)
        self.vcr.update(price, volume, timestamp)
        self.eaf.update(price, volume, timestamp)
        self.lal.update(price, volume, timestamp)

    def get_optimal_config(self) -> Dict[str, Any]:
        """Get optimal trading configuration based on all 5 variables"""

        # 1. MDC: Max safe trade size
        max_safe_size = self.mdc.max_safe_trade_usd

        # 2. LAL: Recommended leverage and effective capital
        recommended_leverage = self.lal.recommended_leverage
        effective_capital = self.capital * recommended_leverage

        # 3. Optimal position size (min of MDC safe and leveraged capital)
        optimal_position = min(max_safe_size, effective_capital)

        # 4. VCR: Required volume per minute for target capture
        required_volume_per_min = self.vcr.get_required_volume_per_minute()

        # 5. Calculate required trade frequency
        if optimal_position > 0:
            trades_per_minute = required_volume_per_min / optimal_position
        else:
            trades_per_minute = 0

        # 6. EAF: Calculate amplified edge
        annual_trades = trades_per_minute * 60 * 24 * 365
        amplification = np.sqrt(annual_trades) if annual_trades > 0 else 1
        amplified_edge = self.base_edge * amplification

        # 7. Calculate expected daily profit
        daily_trades = trades_per_minute * 60 * 24
        edge_per_trade = optimal_position * (self.base_edge / 100)
        daily_profit = edge_per_trade * daily_trades

        # 8. Calculate path to $300k target
        if daily_profit > 0:
            days_to_target = (300000 - self.capital) / daily_profit
        else:
            days_to_target = float('inf')

        return {
            'optimal_position_size': optimal_position,
            'recommended_leverage': recommended_leverage,
            'effective_capital': effective_capital,
            'trades_per_minute': trades_per_minute,
            'trades_per_day': daily_trades,
            'base_edge_pct': self.base_edge,
            'amplified_ir': self.eaf.expected_ir,
            'amplification_factor': amplification,
            'edge_per_trade_usd': edge_per_trade,
            'daily_profit_usd': daily_profit,
            'days_to_300k': days_to_target,
            'mdc': self.mdc.mdc,
            'vcr_ratio': self.vcr.vcr_ratio,
            'depth_utilization_pct': self.mdc.depth_utilization_pct
        }

    def print_scaling_analysis(self):
        """Print detailed scaling analysis"""
        config = self.get_optimal_config()

        print("\n" + "="*70)
        print("VOLUME SCALING ANALYSIS - THE 5 MISSING VARIABLES")
        print("="*70)
        print(f"\nStarting Capital: ${self.capital:.2f}")
        print(f"Daily BTC Volume: ${self.daily_volume:,.0f}")
        print(f"Base Edge: {self.base_edge:.3f}%")

        print("\n--- MISSING VARIABLE ANALYSIS ---")
        print(f"1. MDC (Depth Coefficient): {config['mdc']:,.0f}x headroom")
        print(f"2. VOC (Velocity): {self.voc.current_voc:.2f}x/hour")
        print(f"3. VCR (Volume Capture): {config['vcr_ratio']*100:.6f}% of target")
        print(f"4. EAF (Amplification): {config['amplification_factor']:.2f}x")
        print(f"5. LAL (Safe Leverage): {config['recommended_leverage']:.1f}x")

        print("\n--- OPTIMAL CONFIGURATION ---")
        print(f"Position Size: ${config['optimal_position_size']:.2f}")
        print(f"Effective Capital: ${config['effective_capital']:.2f}")
        print(f"Trades/Minute: {config['trades_per_minute']:.1f}")
        print(f"Trades/Day: {config['trades_per_day']:,.0f}")

        print("\n--- EXPECTED RESULTS ---")
        print(f"Edge/Trade: ${config['edge_per_trade_usd']:.4f}")
        print(f"Daily Profit: ${config['daily_profit_usd']:.2f}")
        print(f"Days to $300k: {config['days_to_300k']:.1f}")
        print("="*70)
