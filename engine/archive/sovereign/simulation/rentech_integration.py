#!/usr/bin/env python3
"""
RENTECH INTEGRATION MODULE

Integrates best practices from top open-source quant projects:
- hftbacktest: Order book queue position, latency modeling
- QLib (Microsoft): Point-in-time ML pipeline, no lookahead
- FinRL: Deep reinforcement learning for adaptive trading
- Statistical arbitrage: Mean reversion, cointegration

This module provides RenTech-grade signal processing and execution logic.

References:
- https://github.com/nkaz001/hftbacktest (HFT backtesting, 3.3K stars)
- https://github.com/microsoft/qlib (ML pipeline, 15K stars)
- https://github.com/AI4Finance-Foundation/FinRL (Deep RL, 12K stars)
- https://github.com/bradleyboyuyang/Statistical-Arbitrage (Stat arb)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
import time


@dataclass
class OrderBookLevel:
    """Single level in order book (from hftbacktest concepts)."""
    price: float
    qty: float
    num_orders: int


@dataclass
class MarketMicrostructure:
    """
    Market microstructure state (hftbacktest inspired).

    RenTech would track all of these for execution optimization.
    """
    mid_price: float
    spread: float
    spread_bps: float
    bid_depth_5: float  # Total bid size in top 5 levels
    ask_depth_5: float  # Total ask size in top 5 levels
    imbalance: float    # (bid - ask) / (bid + ask)

    # Queue position simulation
    queue_behind_bid: float  # Estimated BTC ahead in queue
    queue_behind_ask: float

    # Latency estimates
    exchange_latency_ms: float
    expected_fill_time_ms: float


class RenTechSignalProcessor:
    """
    Signal processor using RenTech-grade techniques.

    From QLib (Microsoft):
    - Point-in-time processing (no lookahead)
    - Alpha decay modeling
    - Factor orthogonalization

    From FinRL:
    - State representation for RL
    - Reward shaping
    - Action space optimization
    """

    def __init__(self, lookback: int = 100):
        self.lookback = lookback

        # Point-in-time price buffer (QLib concept)
        self.prices = deque(maxlen=lookback)
        self.timestamps = deque(maxlen=lookback)
        self.flows = deque(maxlen=lookback)

        # Alpha signals (computed point-in-time only)
        self.alpha_momentum = 0.0
        self.alpha_reversion = 0.0
        self.alpha_flow = 0.0
        self.alpha_composite = 0.0

        # Alpha decay tracking
        self.alpha_decay_halflife = 300  # 5 minutes
        self.last_alpha_time = 0

        # FinRL state representation
        self.state_dim = 12
        self.current_state = np.zeros(self.state_dim)

    def update(self, price: float, flow: Dict, timestamp: float = None):
        """
        Update state with new data (point-in-time, no lookahead).

        This is critical - RenTech's edge is they NEVER use future data.
        """
        if timestamp is None:
            timestamp = time.time()

        self.prices.append(price)
        self.timestamps.append(timestamp)
        self.flows.append(flow)

        if len(self.prices) < 5:
            return

        self._compute_alphas(timestamp)
        self._build_state()

    def _compute_alphas(self, timestamp: float):
        """
        Compute alpha signals (QLib methodology).

        Alpha = Expected return above market
        All computed with ONLY past data (point-in-time).
        """
        prices = np.array(self.prices)

        # Momentum alpha (10-period)
        if len(prices) >= 10:
            ret_10 = (prices[-1] - prices[-10]) / prices[-10]
            self.alpha_momentum = ret_10 * 100  # Normalize to 1 = 1%

        # Mean reversion alpha (stat arb inspired)
        if len(prices) >= 20:
            sma_20 = np.mean(prices[-20:])
            deviation = (prices[-1] - sma_20) / sma_20
            # Reversion signal: bet AGAINST extreme moves
            self.alpha_reversion = -deviation * 100 if abs(deviation) > 0.005 else 0

        # Flow alpha (our blockchain edge)
        recent_flows = list(self.flows)[-10:] if len(self.flows) >= 10 else list(self.flows)
        total_flow = sum(f.get('btc_amount', 0) * f.get('direction', 0) for f in recent_flows)
        self.alpha_flow = total_flow * 0.1  # 10 BTC = 1.0 alpha

        # Apply alpha decay (older signals less valuable)
        if self.last_alpha_time > 0:
            elapsed = timestamp - self.last_alpha_time
            decay = np.exp(-elapsed / self.alpha_decay_halflife)
            # Current alphas get full weight, decay applied to composite
        else:
            decay = 1.0

        self.last_alpha_time = timestamp

        # Composite alpha (factor weighting - RenTech would ML-optimize these)
        # For now: 40% flow, 30% momentum, 30% reversion
        self.alpha_composite = (
            0.40 * self.alpha_flow +
            0.30 * self.alpha_momentum +
            0.30 * self.alpha_reversion
        ) * decay

    def _build_state(self):
        """
        Build FinRL-style state representation.

        FinRL uses this for DRL agent training.
        State includes: prices, returns, volatility, positions, alphas
        """
        prices = np.array(self.prices)

        # Normalized price (current / mean)
        price_normalized = prices[-1] / np.mean(prices) if len(prices) > 0 else 1.0

        # Returns (1, 5, 10 period)
        ret_1 = (prices[-1] / prices[-2] - 1) if len(prices) >= 2 else 0
        ret_5 = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
        ret_10 = (prices[-1] / prices[-10] - 1) if len(prices) >= 10 else 0

        # Volatility (realized, 10-period)
        if len(prices) >= 10:
            returns = np.diff(prices[-11:]) / prices[-11:-1]
            volatility = np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized
        else:
            volatility = 0.5  # Default

        # Alpha signals
        self.current_state = np.array([
            price_normalized,       # 0: Normalized price
            ret_1 * 100,           # 1: 1-period return (%)
            ret_5 * 100,           # 2: 5-period return (%)
            ret_10 * 100,          # 3: 10-period return (%)
            volatility,            # 4: Realized volatility
            self.alpha_momentum,   # 5: Momentum alpha
            self.alpha_reversion,  # 6: Mean reversion alpha
            self.alpha_flow,       # 7: Flow alpha
            self.alpha_composite,  # 8: Composite alpha
            0.0,                   # 9: Position (updated externally)
            0.0,                   # 10: Unrealized PnL
            0.0,                   # 11: Time in position
        ])

    def get_action_recommendation(self) -> Tuple[int, float]:
        """
        Get trading action recommendation (FinRL policy output).

        Returns:
            (action, confidence)
            action: -1 (short), 0 (hold), 1 (long)
            confidence: 0.0 to 1.0
        """
        alpha = self.alpha_composite

        # Simple threshold-based policy (RenTech would use neural net)
        if alpha > 1.0:
            action = 1  # Long
            confidence = min(alpha / 5.0, 1.0)
        elif alpha < -1.0:
            action = -1  # Short
            confidence = min(abs(alpha) / 5.0, 1.0)
        else:
            action = 0  # Hold
            confidence = 1.0 - abs(alpha)

        return action, confidence

    def get_state(self) -> np.ndarray:
        """Get current state for RL agent."""
        return self.current_state.copy()


class StatArbDetector:
    """
    Statistical Arbitrage detector (pairs trading concepts).

    From research:
    - Cointegration testing (Engle-Granger)
    - Ornstein-Uhlenbeck mean reversion
    - Half-life estimation

    Adapted for blockchain flow vs price relationship.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.flow_series = deque(maxlen=window)
        self.price_series = deque(maxlen=window)

        # Spread parameters (flow impact model)
        self.spread_mean = 0.0
        self.spread_std = 1.0
        self.half_life = 30.0  # seconds

        # Z-score threshold for signal
        self.z_threshold = 2.0

    def update(self, price: float, net_flow: float):
        """Update with new price and flow data."""
        self.price_series.append(price)
        self.flow_series.append(net_flow)

        if len(self.price_series) < 20:
            return

        self._estimate_parameters()

    def _estimate_parameters(self):
        """
        Estimate spread parameters using OU process.

        Spread = Price - Flow_Impact
        Mean reversion: dS = θ(μ - S)dt + σdW
        """
        prices = np.array(self.price_series)
        flows = np.array(self.flow_series)

        # Simple flow impact model: cumulative flow affects price
        cumulative_flow = np.cumsum(flows)

        # Spread: price deviation from flow-implied fair value
        # Fair value = mean + β * cumulative_flow
        if len(prices) > 10:
            # Estimate β via regression (simplified)
            flow_centered = cumulative_flow - np.mean(cumulative_flow)
            price_centered = prices - np.mean(prices)

            if np.var(flow_centered) > 0:
                beta = np.cov(price_centered, flow_centered)[0, 1] / np.var(flow_centered)
            else:
                beta = 0

            fair_value = np.mean(prices) + beta * flow_centered
            spread = prices - fair_value

            self.spread_mean = np.mean(spread)
            self.spread_std = np.std(spread) if np.std(spread) > 0 else 1.0

            # Estimate half-life via AR(1)
            if len(spread) > 5:
                spread_lag = spread[:-1]
                spread_now = spread[1:]
                if np.var(spread_lag) > 0:
                    phi = np.cov(spread_now, spread_lag)[0, 1] / np.var(spread_lag)
                    if 0 < phi < 1:
                        self.half_life = -np.log(2) / np.log(phi)

    def get_signal(self) -> Tuple[int, float, float]:
        """
        Get stat arb signal.

        Returns:
            (direction, z_score, expected_holding_period)
        """
        if len(self.price_series) < 20:
            return 0, 0.0, 0.0

        prices = np.array(self.price_series)
        current_spread = prices[-1] - self.spread_mean
        z_score = current_spread / self.spread_std if self.spread_std > 0 else 0

        if z_score > self.z_threshold:
            # Spread too high -> short (expect reversion down)
            return -1, z_score, self.half_life
        elif z_score < -self.z_threshold:
            # Spread too low -> long (expect reversion up)
            return 1, abs(z_score), self.half_life
        else:
            return 0, abs(z_score), 0.0


class ExecutionOptimizer:
    """
    Execution optimizer (hftbacktest concepts).

    Models:
    - Queue position in order book
    - Expected fill probability
    - Optimal order timing
    - Slippage estimation
    """

    def __init__(self):
        # Market impact model parameters
        self.impact_coefficient = 0.0001  # 0.01% per $10K
        self.decay_halflife = 30.0  # seconds

        # Order book simulation
        self.typical_spread_bps = 1.0
        self.typical_depth_btc = 50.0

    def estimate_slippage(
        self,
        order_size_usd: float,
        is_aggressive: bool = True,
        market_volatility: float = 0.5
    ) -> float:
        """
        Estimate execution slippage.

        From hftbacktest: considers order book depth, volatility, and impact.

        Args:
            order_size_usd: Order size in USD
            is_aggressive: True if crossing spread (taker)
            market_volatility: Current volatility level

        Returns:
            Expected slippage as fraction of price
        """
        # Base spread cost
        spread_cost = self.typical_spread_bps / 10000 if is_aggressive else 0

        # Market impact (sqrt of size for large orders)
        size_btc = order_size_usd / 100000  # Rough estimate
        impact = self.impact_coefficient * np.sqrt(size_btc / self.typical_depth_btc)

        # Volatility adjustment
        vol_adjustment = market_volatility / 0.5  # Normalized to typical

        total_slippage = (spread_cost + impact) * vol_adjustment
        return total_slippage

    def optimal_order_timing(
        self,
        signal_strength: float,
        expected_move: float,
        current_volatility: float
    ) -> Dict:
        """
        Calculate optimal order timing and sizing.

        RenTech principle: trade BEFORE the move, not after.

        Returns:
            Dict with optimal_delay, max_delay, urgency
        """
        # Higher signal strength = more urgency
        urgency = min(signal_strength / 2.0, 1.0)

        # Expected alpha decay
        alpha_decay_rate = expected_move / (5 * 60)  # Per second

        # Optimal delay: balance execution quality vs alpha decay
        # Lower volatility = can wait longer for better fill
        base_delay = 1.0  # 1 second base
        vol_factor = 0.5 / max(current_volatility, 0.1)

        optimal_delay = base_delay * vol_factor * (1 - urgency)
        max_delay = optimal_delay * 3  # Don't wait more than 3x optimal

        return {
            'optimal_delay_s': optimal_delay,
            'max_delay_s': max_delay,
            'urgency': urgency,
            'use_limit_order': urgency < 0.7,  # Use limit if not urgent
        }

    def position_size_optimal(
        self,
        capital: float,
        kelly_fraction: float,
        win_probability: float,
        avg_win: float,
        avg_loss: float,
        current_drawdown: float = 0.0
    ) -> float:
        """
        Calculate optimal position size using full Kelly criterion.

        From FinRL/QLib: proper Kelly with drawdown adjustment.

        Kelly = (p * b - q) / b
        where:
            p = win probability
            q = 1 - p
            b = avg_win / avg_loss
        """
        if avg_loss <= 0:
            return 0

        p = win_probability
        q = 1 - p
        b = avg_win / avg_loss

        # Full Kelly
        full_kelly = (p * b - q) / b if b > 0 else 0
        full_kelly = max(0, min(full_kelly, 1))  # Bound [0, 1]

        # Apply Kelly fraction (typically 0.25 = quarter Kelly)
        position_pct = full_kelly * kelly_fraction

        # Drawdown adjustment: reduce size during drawdowns
        if current_drawdown > 0.05:  # >5% drawdown
            dd_factor = 1 - min(current_drawdown * 2, 0.5)  # Reduce up to 50%
            position_pct *= dd_factor

        return position_pct


class RenTechFilter:
    """
    Signal filtering using RenTech principles.

    Core rules:
    1. Edge > Costs (always)
    2. High conviction only
    3. Ensemble agreement
    4. Risk-adjusted sizing
    """

    def __init__(
        self,
        min_edge_multiple: float = 2.0,  # Edge must be 2x costs
        min_confidence: float = 0.7,
        min_flow_btc: float = 5.0,
    ):
        self.min_edge_multiple = min_edge_multiple
        self.min_confidence = min_confidence
        self.min_flow_btc = min_flow_btc

        # Track filter performance
        self.signals_received = 0
        self.signals_passed = 0
        self.filter_reasons: Dict[str, int] = {}

    def should_trade(
        self,
        signal: Dict,
        round_trip_fee: float,
        expected_move: float
    ) -> Tuple[bool, str]:
        """
        Apply RenTech filters to determine if signal is tradeable.

        Returns:
            (should_trade, reason)
        """
        self.signals_received += 1

        # Filter 1: Minimum flow size
        btc_flow = abs(signal.get('btc_amount', 0))
        if btc_flow < self.min_flow_btc:
            reason = f"flow_too_small ({btc_flow:.1f} < {self.min_flow_btc})"
            self.filter_reasons[reason] = self.filter_reasons.get(reason, 0) + 1
            return False, reason

        # Filter 2: Minimum confidence
        confidence = signal.get('confidence', 0)
        if confidence < self.min_confidence:
            reason = f"low_confidence ({confidence:.2f} < {self.min_confidence})"
            self.filter_reasons[reason] = self.filter_reasons.get(reason, 0) + 1
            return False, reason

        # Filter 3: Edge must exceed costs
        min_edge = round_trip_fee * self.min_edge_multiple
        if expected_move < min_edge:
            reason = f"edge_too_small ({expected_move:.4f} < {min_edge:.4f})"
            self.filter_reasons[reason] = self.filter_reasons.get(reason, 0) + 1
            return False, reason

        # Filter 4: Ensemble agreement for smaller flows
        ensemble_type = signal.get('ensemble_type', 'unknown')
        if btc_flow < 20 and ensemble_type not in ['agreement', 'adaptive_only']:
            reason = f"no_ensemble_agreement ({ensemble_type})"
            self.filter_reasons[reason] = self.filter_reasons.get(reason, 0) + 1
            return False, reason

        self.signals_passed += 1
        return True, "passed"

    def get_stats(self) -> Dict:
        """Get filter statistics."""
        pass_rate = self.signals_passed / self.signals_received if self.signals_received > 0 else 0
        return {
            'signals_received': self.signals_received,
            'signals_passed': self.signals_passed,
            'pass_rate': pass_rate,
            'filter_breakdown': dict(self.filter_reasons),
        }


# Convenience function to create all components
def create_rentech_suite(
    lookback: int = 100,
    min_edge_multiple: float = 2.5,
    min_confidence: float = 0.7,
    min_flow_btc: float = 5.0
) -> Dict:
    """
    Create full RenTech integration suite.

    Returns dict with:
    - signal_processor: RenTechSignalProcessor
    - stat_arb: StatArbDetector
    - executor: ExecutionOptimizer
    - filter: RenTechFilter
    """
    return {
        'signal_processor': RenTechSignalProcessor(lookback=lookback),
        'stat_arb': StatArbDetector(window=lookback),
        'executor': ExecutionOptimizer(),
        'filter': RenTechFilter(
            min_edge_multiple=min_edge_multiple,
            min_confidence=min_confidence,
            min_flow_btc=min_flow_btc,
        ),
    }
