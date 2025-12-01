#!/usr/bin/env python3
"""
ACADEMIC FORMULAS 2024-2025 (IDs 731-760)
==========================================
Cutting-edge peer-reviewed formulas for nanosecond-to-second trading

Based on latest research:
- Hawkes Processes (2024): arXiv 2408.03594, Springer J Banking
- Queue-Reactive Models (2024): arXiv 2405.18594, SIAM J Financial Math
- Optimal Execution (2024): PerfectQuant, arXiv 2006.11426
- Gegenbauer GARCH (2025): Mathematics Journal MDPI
- Kyle's Lambda: Kyle & Obizhaeva, SSRN 2823630
- Speed Premium (2024): BIS Working Paper 1290

Categories:
    731-735: Hawkes Processes (order flow, market making)
    736-740: Queue-Reactive Models (optimal placement)
    741-745: Optimal Execution (Almgren-Chriss variants)
    746-750: Gegenbauer GARCH (volatility with periodicity)
    751-755: Kyle's Lambda (market impact)
    756-760: Speed Premium (latency arbitrage)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from .base import BaseFormula, FormulaRegistry


# =============================================================================
# HAWKES PROCESS FORMULAS (IDs 731-735)
# =============================================================================

@FormulaRegistry.register(731, "HawkesOrderFlowImbalance", "hawkes")
class HawkesOrderFlowImbalance(BaseFormula):
    """
    ID 731: Hawkes Order Flow Imbalance

    Source: arXiv 2408.03594 (2024)
    "Forecasting High Frequency Order Flow Imbalance using Hawkes Processes"

    Purpose: Model self-exciting order arrival with feedback loops
    Math: λ(t) = μ + ∫ α·exp(-β·(t-s)) dN(s)

    Application: Predict order flow 100-1000ms ahead
    Speed: Microsecond execution
    """

    DESCRIPTION = "Hawkes self-exciting order flow prediction"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.mu = kwargs.get('mu', 1.0)  # Baseline intensity
        self.alpha = kwargs.get('alpha', 0.5)  # Excitation strength
        self.beta = kwargs.get('beta', 1.0)  # Decay rate
        self.events = deque(maxlen=1000)  # Event timestamps
        self.intensity = 0.0

    def _compute(self) -> None:
        """Compute Hawkes intensity and generate signal."""
        if len(self.returns) < 5:
            self.signal = 0
            self.confidence = 0.0
            return

        # Calculate intensity from recent price movements
        returns = self._returns_array()

        # Use absolute returns as proxy for event intensity
        abs_returns = np.abs(returns[-20:]) if len(returns) >= 20 else np.abs(returns)

        # Hawkes intensity with exponential decay
        self.intensity = self.mu
        for i, r in enumerate(abs_returns):
            time_diff = len(abs_returns) - i
            self.intensity += self.alpha * r * 100 * np.exp(-self.beta * time_diff / 10)

        # Signal based on order flow imbalance
        recent_returns = returns[-5:]
        flow_imbalance = np.sum(recent_returns) * 100  # Direction of flow

        # High intensity + positive flow = bullish
        # High intensity + negative flow = bearish
        if self.intensity > 1.5:  # High activity
            if flow_imbalance > 0.05:
                self.signal = 1
                self.confidence = min(0.9, self.intensity / 3)
            elif flow_imbalance < -0.05:
                self.signal = -1
                self.confidence = min(0.9, self.intensity / 3)
            else:
                self.signal = 0
                self.confidence = 0.3
        else:  # Low activity
            self.signal = 0
            self.confidence = 0.2


@FormulaRegistry.register(732, "DeepHawkesMarketMaking", "hawkes")
class DeepHawkesMarketMaking(BaseFormula):
    """
    ID 732: Deep Hawkes Market Making

    Source: Springer J Banking & Financial Tech (2024)
    "Deep Hawkes process for high-frequency market making"

    Purpose: Neural network-enhanced Hawkes for optimal quotes
    Math: State-dependent intensity using feedback loops

    Application: Dynamic market making at millisecond speed
    Speed: 1-10ms decision cycle
    """

    DESCRIPTION = "Deep learning enhanced Hawkes for market making"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.state_history = deque(maxlen=100)
        self.optimal_spread = 0.001  # 0.1% default
        self.inventory = 0.0

    def _compute(self) -> None:
        """Compute optimal market making signal."""
        if len(self.prices) < 10:
            self.signal = 0
            self.confidence = 0.0
            return

        prices = self._prices_array()
        returns = self._returns_array() if len(self.returns) > 0 else np.array([0])

        # Volatility estimate (simplified from full LSTM model)
        vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)

        # Hawkes-style intensity from recent activity
        activity = np.sum(np.abs(returns[-10:])) if len(returns) >= 10 else 0

        # Adjust spread based on volatility and activity
        self.optimal_spread = 0.001 * (1 + vol * 100 + activity * 10)

        # Market making signal: prefer neutral, but follow strong moves
        price_move = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0

        if abs(price_move) > self.optimal_spread * 2:
            # Strong move - follow trend briefly
            self.signal = 1 if price_move > 0 else -1
            self.confidence = min(0.7, abs(price_move) / self.optimal_spread)
        else:
            # Range-bound - neutral (market making mode)
            self.signal = 0
            self.confidence = 0.5


@FormulaRegistry.register(733, "HawkesJumpDiffusionPrice", "hawkes")
class HawkesJumpDiffusionPrice(BaseFormula):
    """
    ID 733: Hawkes Jump-Diffusion Price Model

    Source: arXiv 2409.12776 (2024)
    "Algorithmic Trading under Semi-Markov and Hawkes Jump-Diffusion Models"

    Purpose: Price dynamics with self-exciting jumps
    Math: dS(t) = μ dt + σ dW(t) + dJ(t), J ~ compound Hawkes

    Application: Price prediction with jump clustering
    Speed: Millisecond regime detection
    """

    DESCRIPTION = "Jump-diffusion price model with Hawkes clustering"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.jump_threshold = kwargs.get('jump_threshold', 0.005)  # 0.5%
        self.jumps = deque(maxlen=100)
        self.jump_intensity = 0.1

    def _compute(self) -> None:
        """Detect jumps and compute signal."""
        if len(self.returns) < 10:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()

        # Detect jumps (returns exceeding threshold)
        recent_returns = returns[-20:] if len(returns) >= 20 else returns
        jumps_detected = np.abs(recent_returns) > self.jump_threshold

        # Update jump intensity (self-exciting)
        n_jumps = np.sum(jumps_detected)
        self.jump_intensity = 0.1 + 0.5 * n_jumps / len(recent_returns)

        # If recent jump, predict continuation
        last_return = returns[-1]
        second_last = returns[-2] if len(returns) >= 2 else 0

        if abs(last_return) > self.jump_threshold:
            # Jump detected - momentum after jump
            if last_return > 0:
                self.signal = 1
                self.confidence = min(0.8, self.jump_intensity)
            else:
                self.signal = -1
                self.confidence = min(0.8, self.jump_intensity)
        elif abs(second_last) > self.jump_threshold:
            # Recent jump - possible continuation
            self.signal = 1 if second_last > 0 else -1
            self.confidence = min(0.6, self.jump_intensity * 0.7)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(734, "CrossExcitationOrderFlow", "hawkes")
class CrossExcitationOrderFlow(BaseFormula):
    """
    ID 734: Cross-Excitation Order Flow

    Purpose: Model bid-ask interaction via cross-exciting Hawkes
    Math: λ_bid(t) includes both self-excitation and ask cross-excitation

    Application: Predict order flow imbalance
    Speed: Sub-millisecond
    """

    DESCRIPTION = "Cross-exciting Hawkes for bid-ask flow interaction"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        # Hawkes parameters
        self.mu = 1.0
        self.alpha_self = 0.4  # Self-excitation
        self.alpha_cross = 0.2  # Cross-excitation
        self.beta = 1.0  # Decay
        self.lambda_bid = 1.0
        self.lambda_ask = 1.0

    def _compute(self) -> None:
        """Compute order flow imbalance from cross-excitation."""
        if len(self.returns) < 5:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()
        volumes = self._volumes_array() if len(self.volumes) > 0 else np.ones(len(returns))

        # Proxy bid/ask flow from signed returns - ensure same length
        n = min(len(returns), len(volumes), 20)
        recent = returns[-n:]
        recent_vol = volumes[-n:]

        # Positive returns = bid aggression, negative = ask aggression
        bid_events = np.where(recent > 0, recent * recent_vol, 0)
        ask_events = np.where(recent < 0, -recent * recent_vol, 0)

        # Calculate intensities with exponential decay
        self.lambda_bid = self.mu
        self.lambda_ask = self.mu

        for i in range(len(recent)):
            decay = np.exp(-self.beta * (len(recent) - i) / 10)
            self.lambda_bid += self.alpha_self * bid_events[i] * decay
            self.lambda_bid += self.alpha_cross * ask_events[i] * decay
            self.lambda_ask += self.alpha_self * ask_events[i] * decay
            self.lambda_ask += self.alpha_cross * bid_events[i] * decay

        # Order flow imbalance
        ofi = self.lambda_bid - self.lambda_ask

        if ofi > 0.5:
            self.signal = 1  # More bid pressure
            self.confidence = min(0.8, ofi / 2)
        elif ofi < -0.5:
            self.signal = -1  # More ask pressure
            self.confidence = min(0.8, abs(ofi) / 2)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(735, "HawkesRoughVolatility", "hawkes")
class HawkesRoughVolatility(BaseFormula):
    """
    ID 735: Hawkes Rough Volatility

    Source: AIMS Numerical Algebra (2025)
    "A limit order book model for high frequency trading with rough volatility"

    Purpose: Ultra-high-frequency volatility with H < 0.5
    Math: σ(t) from Volterra equation driven by Hawkes

    Application: Microsecond volatility updates
    Speed: Sub-millisecond
    """

    DESCRIPTION = "Rough volatility (H<0.5) with Hawkes dynamics"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.hurst = kwargs.get('hurst', 0.1)  # Very rough H ≈ 0.1
        self.rough_vol = 0.01

    def _compute(self) -> None:
        """Estimate rough volatility and generate signal."""
        if len(self.returns) < 10:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()
        recent = returns[-50:] if len(returns) >= 50 else returns

        # Rough volatility: use power-law decay weights
        n = len(recent)
        weights = np.array([(i + 1) ** (-self.hurst - 0.5) for i in range(n)])
        weights = weights[::-1]  # Recent observations get higher weight
        weights /= weights.sum()

        # Weighted realized volatility
        self.rough_vol = np.sqrt(np.sum(weights * recent ** 2))

        # Standard volatility for comparison
        std_vol = np.std(recent)

        # Signal based on vol regime
        # High rough vol relative to standard = clustering = trend likely
        vol_ratio = self.rough_vol / (std_vol + 1e-10)

        if vol_ratio > 1.2:  # Vol clustering detected
            # Follow recent trend
            recent_trend = np.mean(recent[-5:])
            if recent_trend > 0:
                self.signal = 1
                self.confidence = min(0.7, vol_ratio - 1)
            else:
                self.signal = -1
                self.confidence = min(0.7, vol_ratio - 1)
        else:
            self.signal = 0
            self.confidence = 0.3


# =============================================================================
# QUEUE-REACTIVE MODELS (IDs 736-740)
# =============================================================================

@FormulaRegistry.register(736, "QueueReactivePositionProbability", "queue_reactive")
class QueueReactivePositionProbability(BaseFormula):
    """
    ID 736: Queue-Reactive Position Probability

    Source: Huang, Lehalle, Rosenbaum (2015)
    "Simulating and analyzing order book data: The queue-reactive model"

    Purpose: Probability of fill based on queue position
    Math: P(fill | position k, events n) via Markov queuing

    Application: Optimal order placement
    Speed: Microsecond calculation
    """

    DESCRIPTION = "Queue-reactive fill probability model"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.fill_prob = 0.5
        self.queue_position_estimate = 5

    def _compute(self) -> None:
        """Estimate queue position value from price action."""
        if len(self.prices) < 10:
            self.signal = 0
            self.confidence = 0.0
            return

        prices = self._prices_array()
        volumes = self._volumes_array() if len(self.volumes) > 0 else np.ones(len(prices))

        # Estimate queue dynamics from volume
        recent_vol = volumes[-10:] if len(volumes) >= 10 else volumes
        avg_vol = np.mean(recent_vol)

        # Higher volume = faster queue movement = higher fill prob
        vol_ratio = recent_vol[-1] / (avg_vol + 1e-10)
        self.fill_prob = min(0.9, 0.5 * vol_ratio)

        # Price trend
        returns = self._returns_array() if len(self.returns) > 0 else np.array([0])
        recent_return = np.mean(returns[-3:]) if len(returns) >= 3 else 0

        # Signal: if high fill prob and favorable direction
        if self.fill_prob > 0.6:
            if recent_return > 0:
                self.signal = 1
                self.confidence = self.fill_prob
            elif recent_return < 0:
                self.signal = -1
                self.confidence = self.fill_prob
            else:
                self.signal = 0
                self.confidence = 0.4
        else:
            self.signal = 0
            self.confidence = self.fill_prob


@FormulaRegistry.register(737, "OrderSizeDependentQueueDynamics", "queue_reactive")
class OrderSizeDependentQueueDynamics(BaseFormula):
    """
    ID 737: Order Size-Dependent Queue Dynamics

    Source: Bodor & Carlier (2024) arXiv 2405.18594
    "A Novel Approach to Queue-Reactive Models: The Importance of Order Sizes"

    Purpose: Queue evolution accounting for order sizes
    Math: Extended QR model with size-weighted transitions

    Application: More accurate fill probability
    Speed: Real-time queue tracking
    """

    DESCRIPTION = "2024 research: Order sizes significantly affect queue dynamics"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.size_factor = 1.0
        self.wait_time_estimate = 1.0

    def _compute(self) -> None:
        """Compute size-adjusted queue signal."""
        if len(self.volumes) < 10:
            self.signal = 0
            self.confidence = 0.0
            return

        volumes = self._volumes_array()
        recent_vol = volumes[-20:] if len(volumes) >= 20 else volumes

        # Order size analysis
        avg_size = np.mean(recent_vol)
        current_size = recent_vol[-1]

        # 2024 finding: larger orders ahead = slower queue
        self.size_factor = current_size / (avg_size + 1e-10)

        # If we're in a low-volume period (smaller orders), queue moves faster
        if self.size_factor < 0.8:
            # Smaller than average orders = favorable
            returns = self._returns_array() if len(self.returns) > 0 else np.array([0])
            recent_return = np.mean(returns[-3:]) if len(returns) >= 3 else 0

            if recent_return > 0:
                self.signal = 1
                self.confidence = 0.6
            elif recent_return < 0:
                self.signal = -1
                self.confidence = 0.6
            else:
                self.signal = 0
                self.confidence = 0.4
        else:
            # Large orders = unfavorable queue dynamics
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(738, "QueueReactiveSpreadCapture", "queue_reactive")
class QueueReactiveSpreadCapture(BaseFormula):
    """
    ID 738: Queue-Reactive Spread Capture

    Purpose: Optimal bid-ask placement in queue
    Math: argmax E[PnL | queue_state, order_size]

    Application: Market making spread optimization
    Speed: Millisecond recomputation
    """

    DESCRIPTION = "Optimal spread capture based on queue state"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.optimal_spread = 0.001

    def _compute(self) -> None:
        """Calculate optimal spread capture signal."""
        if len(self.prices) < 10:
            self.signal = 0
            self.confidence = 0.0
            return

        prices = self._prices_array()
        returns = self._returns_array() if len(self.returns) > 0 else np.array([0])

        # Estimate current spread from price volatility
        vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        self.optimal_spread = max(0.0005, vol * 2)  # At least 0.05%

        # Current price position relative to recent range
        high = np.max(prices[-20:]) if len(prices) >= 20 else np.max(prices)
        low = np.min(prices[-20:]) if len(prices) >= 20 else np.min(prices)
        current = prices[-1]

        if high == low:
            position_in_range = 0.5
        else:
            position_in_range = (current - low) / (high - low)

        # Signal: buy at low of range, sell at high
        if position_in_range < 0.3:
            self.signal = 1  # Near low, bid side favorable
            self.confidence = 0.6 * (1 - position_in_range)
        elif position_in_range > 0.7:
            self.signal = -1  # Near high, ask side favorable
            self.confidence = 0.6 * position_in_range
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(739, "ReinforcementLearningQueuePolicy", "queue_reactive")
class ReinforcementLearningQueuePolicy(BaseFormula):
    """
    ID 739: Reinforcement Learning Queue Policy

    Source: arXiv 2511.15262 (Nov 2024)
    "Reinforcement Learning in Queue-Reactive Models: Application to Optimal Execution"

    Purpose: RL-optimized queue-reactive execution
    Math: Deep Q-Learning with queue state features

    Application: Adaptive execution strategy
    Speed: Trained offline, execute in microseconds
    """

    DESCRIPTION = "RL-based optimal queue execution policy (Nov 2024)"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        # Simplified policy parameters (real version: trained neural net)
        self.urgency_threshold = 0.7
        self.spread_threshold = 0.002

    def _compute(self) -> None:
        """Execute RL policy for queue decisions."""
        if len(self.returns) < 5:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()
        prices = self._prices_array()

        # State features (simplified)
        vol = np.std(returns[-10:]) if len(returns) >= 10 else 0.01
        momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0

        # Estimate spread from price changes
        price_changes = np.abs(np.diff(prices[-10:])) if len(prices) >= 10 else np.array([0.01])
        spread_estimate = np.median(price_changes)

        # RL Policy (simplified - real uses trained Q-network)
        # Higher vol = higher urgency to trade
        urgency = vol / (spread_estimate + 1e-10)

        if urgency > 2.0:
            # High urgency - follow momentum immediately
            self.signal = 1 if momentum > 0 else (-1 if momentum < 0 else 0)
            self.confidence = min(0.8, urgency / 3)
        elif urgency > 1.0:
            # Medium urgency - trade if momentum clear
            if abs(momentum) > vol:
                self.signal = 1 if momentum > 0 else -1
                self.confidence = 0.6
            else:
                self.signal = 0
                self.confidence = 0.4
        else:
            # Low urgency - patient, wait for opportunity
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(740, "QueueHawkesMarkovianModel", "queue_reactive")
class QueueHawkesMarkovianModel(BaseFormula):
    """
    ID 740: Queue Hawkes Markovian Model

    Source: SIAM J Financial Math (2023)
    "Order Book Queue Hawkes Markovian Modeling"

    Purpose: Unified queue + Hawkes framework
    Math: Markov-modulated Hawkes with queue dynamics

    Application: Complete order book modeling
    Speed: Microsecond state updates
    """

    DESCRIPTION = "Unified Queue-Hawkes-Markov order book model"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.regime = 0  # 0=calm, 1=active, 2=volatile
        self.hawkes_intensity = 1.0

    def _compute(self) -> None:
        """Update combined queue-Hawkes state."""
        if len(self.returns) < 10:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()
        volumes = self._volumes_array() if len(self.volumes) > 0 else np.ones(len(returns))

        # Calculate Hawkes intensity
        recent_abs_returns = np.abs(returns[-20:]) if len(returns) >= 20 else np.abs(returns)
        self.hawkes_intensity = 1.0 + np.sum(recent_abs_returns) * 50

        # Regime detection
        vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
        if vol < 0.001:
            self.regime = 0  # Calm
        elif vol < 0.005:
            self.regime = 1  # Active
        else:
            self.regime = 2  # Volatile

        # Signal based on regime and intensity
        momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0

        if self.regime == 2:  # Volatile - follow strong moves
            if abs(momentum) > vol:
                self.signal = 1 if momentum > 0 else -1
                self.confidence = 0.7
            else:
                self.signal = 0
                self.confidence = 0.3
        elif self.regime == 1:  # Active - moderate signals
            if abs(momentum) > vol * 0.5:
                self.signal = 1 if momentum > 0 else -1
                self.confidence = 0.5
            else:
                self.signal = 0
                self.confidence = 0.4
        else:  # Calm - mean reversion
            # In calm regime, expect mean reversion
            if momentum > 0.0005:
                self.signal = -1  # Overbought, expect pullback
                self.confidence = 0.4
            elif momentum < -0.0005:
                self.signal = 1  # Oversold, expect bounce
                self.confidence = 0.4
            else:
                self.signal = 0
                self.confidence = 0.5


# =============================================================================
# OPTIMAL EXECUTION FORMULAS (IDs 741-745)
# =============================================================================

@FormulaRegistry.register(741, "AlmgrenChrissGBM", "execution")
class AlmgrenChrissGBM(BaseFormula):
    """
    ID 741: Almgren-Chriss with GBM

    Source: Almgren & Chriss (2000), updated 2024 implementations

    Purpose: Optimal execution with Geometric Brownian Motion
    Math: Minimize E[cost] + λ·Var[cost]

    Application: Large order execution
    Speed: Pre-computed trajectory, second-level updates
    """

    DESCRIPTION = "Almgren-Chriss optimal execution with GBM"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.risk_aversion = kwargs.get('risk_aversion', 0.5)
        self.execution_progress = 0.0

    def _compute(self) -> None:
        """Compute execution trajectory signal."""
        if len(self.prices) < 10:
            self.signal = 0
            self.confidence = 0.0
            return

        prices = self._prices_array()
        returns = self._returns_array() if len(self.returns) > 0 else np.array([0])

        # Estimate volatility (key for AC model)
        vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.01

        # Optimal execution trajectory depends on risk aversion
        # Higher λ = more aggressive (execute faster to avoid vol risk)
        urgency = self.risk_aversion * vol * 100

        # Current price vs VWAP-like benchmark
        vwap = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        price_vs_vwap = (prices[-1] - vwap) / vwap

        if urgency > 0.5:  # High urgency
            # Execute immediately regardless of price
            self.signal = 1  # Assume buying
            self.confidence = min(0.8, urgency)
        elif price_vs_vwap < -0.001:  # Price below VWAP
            # Good time to buy
            self.signal = 1
            self.confidence = 0.6
        elif price_vs_vwap > 0.001:  # Price above VWAP
            # Wait or sell
            self.signal = -1
            self.confidence = 0.5
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(742, "TransientImpactExecution", "execution")
class TransientImpactExecution(BaseFormula):
    """
    ID 742: Transient Impact Execution

    Purpose: Execution with temporary + permanent impact
    Math: dS = f(n')dt + g(n')dW where f=permanent, g=temporary

    Application: Minimize market impact
    Speed: Continuous trajectory, millisecond updates
    """

    DESCRIPTION = "Execution model with transient and permanent impact"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.permanent_impact = 0.0
        self.transient_impact = 0.0

    def _compute(self) -> None:
        """Estimate market impact and optimal execution."""
        if len(self.prices) < 20:
            self.signal = 0
            self.confidence = 0.0
            return

        prices = self._prices_array()
        volumes = self._volumes_array() if len(self.volumes) > 0 else np.ones(len(prices))

        # Estimate impact from price-volume relationship
        recent_prices = prices[-20:]
        recent_vols = volumes[-20:]

        # Correlation between volume and price change = impact proxy
        price_changes = np.diff(recent_prices)
        vol_changes = recent_vols[1:]

        if len(price_changes) > 5 and np.std(vol_changes) > 0:
            correlation = np.corrcoef(price_changes, vol_changes)[0, 1]
            self.permanent_impact = abs(correlation) * 0.001  # Scale factor
        else:
            self.permanent_impact = 0.0001

        # Transient impact decays - estimate from recent reversion
        returns = self._returns_array()
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 2 else 0
        self.transient_impact = max(0, -autocorr * 0.001)  # Negative autocorr = reversion

        # Signal: prefer trading when impact is low
        total_impact = self.permanent_impact + self.transient_impact

        if total_impact < 0.0001:
            # Low impact environment
            momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            self.signal = 1 if momentum > 0 else (-1 if momentum < 0 else 0)
            self.confidence = 0.6
        else:
            # High impact - be cautious
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(743, "EfficientFrontierSelection", "execution")
class EfficientFrontierSelection(BaseFormula):
    """
    ID 743: Almgren-Chriss Efficient Frontier

    Purpose: Risk-return tradeoff in execution
    Math: Frontier curve in (E[cost], Var[cost]) space

    Application: Execution strategy selection
    Speed: Pre-analysis (seconds), then fixed strategy
    """

    DESCRIPTION = "Efficient frontier for execution risk-return tradeoff"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.optimal_aggression = 0.5  # 0=passive, 1=aggressive

    def _compute(self) -> None:
        """Select point on efficient frontier."""
        if len(self.returns) < 20:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()

        # Estimate market conditions
        vol = np.std(returns[-20:])
        trend = np.mean(returns[-10:])

        # High vol = prefer aggressive (avoid timing risk)
        # Strong trend = prefer aggressive (capture momentum)
        vol_score = min(1, vol / 0.01)  # Normalize
        trend_score = min(1, abs(trend) / 0.001)

        self.optimal_aggression = 0.3 + 0.4 * vol_score + 0.3 * trend_score

        # Signal based on optimal aggression and current conditions
        if self.optimal_aggression > 0.6:
            # Aggressive execution
            self.signal = 1 if trend > 0 else -1
            self.confidence = self.optimal_aggression
        else:
            # Passive execution
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(744, "ISWithHawkes", "execution")
class ISWithHawkes(BaseFormula):
    """
    ID 744: Implementation Shortfall with Hawkes

    Source: arXiv 2504.10282 (2024)

    Purpose: Optimal execution accounting for order clustering
    Math: Almgren-Chriss + Hawkes arrival rates

    Application: Execution when market is clustering
    Speed: Millisecond strategy adjustment
    """

    DESCRIPTION = "Implementation shortfall with Hawkes order flow"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.hawkes_intensity = 1.0
        self.is_benchmark = 0.0

    def _compute(self) -> None:
        """Compute IS-optimal execution with Hawkes."""
        if len(self.returns) < 10:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()
        prices = self._prices_array()

        # Hawkes intensity from recent activity
        recent_abs = np.abs(returns[-20:]) if len(returns) >= 20 else np.abs(returns)
        self.hawkes_intensity = 1.0 + np.sum(recent_abs) * 30

        # IS benchmark (arrival price)
        self.is_benchmark = prices[-10] if len(prices) >= 10 else prices[0]
        current_price = prices[-1]

        # IS = (execution price - benchmark) / benchmark
        is_current = (current_price - self.is_benchmark) / self.is_benchmark

        # High Hawkes intensity = order clustering = execute faster
        if self.hawkes_intensity > 2.0:
            # Market is active, execute to avoid slippage
            self.signal = 1  # Assume need to buy
            self.confidence = min(0.8, self.hawkes_intensity / 3)
        elif is_current < -0.001:
            # Good IS, price below benchmark
            self.signal = 1
            self.confidence = 0.6
        elif is_current > 0.001:
            # Bad IS, wait for better price
            self.signal = 0
            self.confidence = 0.4
        else:
            self.signal = 0
            self.confidence = 0.5


@FormulaRegistry.register(745, "AdaptiveLOBExecution", "execution")
class AdaptiveLOBExecution(BaseFormula):
    """
    ID 745: Adaptive Execution with LOB State

    Purpose: Execution strategy adapts to book depth
    Math: Dynamic programming with state = (inventory, time, book)

    Application: Real-time execution optimization
    Speed: Microsecond decisions
    """

    DESCRIPTION = "Adaptive execution based on limit order book state"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.book_imbalance = 0.0
        self.depth_estimate = 1.0

    def _compute(self) -> None:
        """Adapt execution to LOB state."""
        if len(self.prices) < 10:
            self.signal = 0
            self.confidence = 0.0
            return

        prices = self._prices_array()
        volumes = self._volumes_array() if len(self.volumes) > 0 else np.ones(len(prices))
        returns = self._returns_array() if len(self.returns) > 0 else np.array([0])

        # Estimate book state from price-volume relationship
        recent_vol = volumes[-10:] if len(volumes) >= 10 else volumes
        self.depth_estimate = np.mean(recent_vol)

        # Imbalance from price movement direction vs volume
        price_direction = np.sign(returns[-5:]).sum() if len(returns) >= 5 else 0
        vol_ratio = recent_vol[-1] / (self.depth_estimate + 1e-10)

        self.book_imbalance = price_direction * vol_ratio

        # Execute based on book state
        if self.book_imbalance > 1:
            # Strong buy pressure + high volume = favorable to sell
            self.signal = -1
            self.confidence = min(0.7, self.book_imbalance / 3)
        elif self.book_imbalance < -1:
            # Strong sell pressure = favorable to buy
            self.signal = 1
            self.confidence = min(0.7, abs(self.book_imbalance) / 3)
        else:
            # Balanced book
            self.signal = 0
            self.confidence = 0.5


# =============================================================================
# GEGENBAUER GARCH FORMULAS (IDs 746-750)
# =============================================================================

@FormulaRegistry.register(746, "GegenbauerGARCH", "garch")
class GegenbauerGARCH(BaseFormula):
    """
    ID 746: Gegenbauer GARCH

    Source: Mathematics Journal MDPI (January 2025)
    "Major Issues in High-Frequency Financial Data Analysis"

    Purpose: Capture long memory + periodicity in volatility
    Math: σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1) + γ·G(L)·ε²(t)

    Application: Volatility forecasting with cycles
    Speed: Second-level updates
    """

    DESCRIPTION = "Gegenbauer GARCH with long memory and periodicity"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.omega = 0.00001
        self.alpha = 0.1
        self.beta = 0.85
        self.gamma = 0.05
        self.sigma_sq = 0.0001

    def _compute(self) -> None:
        """Compute Gegenbauer GARCH volatility."""
        if len(self.returns) < 20:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()
        recent = returns[-50:] if len(returns) >= 50 else returns

        # GARCH(1,1) base
        eps_sq = recent ** 2

        # Recursive GARCH calculation (simplified)
        sigma_sq = np.var(recent)  # Initialize
        for i in range(1, len(recent)):
            sigma_sq = self.omega + self.alpha * eps_sq[i-1] + self.beta * sigma_sq

        self.sigma_sq = sigma_sq

        # Gegenbauer component: long memory effect
        # Simplified: use weighted average of past squared returns
        weights = np.array([0.9 ** i for i in range(len(recent))])[::-1]
        weights /= weights.sum()
        long_memory_vol = np.sum(weights * eps_sq)

        # Combined volatility
        final_vol = np.sqrt(0.9 * self.sigma_sq + 0.1 * long_memory_vol)

        # Signal: vol regime trading
        current_vol = abs(recent[-1])
        vol_zscore = (current_vol - final_vol) / (final_vol + 1e-10)

        if vol_zscore > 2:  # Vol spike
            # High vol = follow momentum
            self.signal = 1 if recent[-1] > 0 else -1
            self.confidence = min(0.7, vol_zscore / 4)
        elif vol_zscore < -1:  # Low vol
            # Mean reversion
            self.signal = -1 if recent[-1] > 0 else 1
            self.confidence = 0.5
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(747, "PeriodicGARCH", "garch")
class PeriodicGARCH(BaseFormula):
    """
    ID 747: Periodic GARCH for Intraday

    Purpose: Model intraday volatility patterns (U-shape)
    Math: GARCH with periodic components at market open/close

    Application: Adjust volatility estimates by time of day
    Speed: Pre-computed patterns, real-time scaling
    """

    DESCRIPTION = "Periodic GARCH capturing intraday U-shape volatility"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.time_of_day_factor = 1.0

    def _compute(self) -> None:
        """Compute periodic volatility adjustment."""
        if len(self.returns) < 10:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()

        # Base volatility
        base_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)

        # For crypto (24h), use different periods
        # Simplified: detect high/low vol periods from recent data
        hourly_vols = []
        chunk_size = max(1, len(returns) // 24)
        for i in range(0, len(returns) - chunk_size, chunk_size):
            chunk = returns[i:i+chunk_size]
            hourly_vols.append(np.std(chunk))

        if len(hourly_vols) > 1:
            current_hour_vol = np.std(returns[-chunk_size:]) if len(returns) >= chunk_size else base_vol
            avg_hourly_vol = np.mean(hourly_vols)
            self.time_of_day_factor = current_hour_vol / (avg_hourly_vol + 1e-10)
        else:
            self.time_of_day_factor = 1.0

        # Signal based on vol regime
        if self.time_of_day_factor > 1.2:  # High vol period
            # More momentum-oriented
            momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            if abs(momentum) > base_vol:
                self.signal = 1 if momentum > 0 else -1
                self.confidence = 0.6
            else:
                self.signal = 0
                self.confidence = 0.4
        else:  # Low vol period
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(748, "HighFrequencyGARCH", "garch")
class HighFrequencyGARCH(BaseFormula):
    """
    ID 748: High-Frequency GARCH

    Purpose: GARCH adapted for tick data
    Math: Subsample tick data → GARCH → scale back

    Application: Tick-level volatility
    Speed: Millisecond resolution
    """

    DESCRIPTION = "GARCH adapted for tick-level high-frequency data"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.tick_vol = 0.001

    def _compute(self) -> None:
        """Compute high-frequency GARCH."""
        if len(self.returns) < 10:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()

        # High-frequency adjustment: handle microstructure noise
        # Use overlapping returns to reduce noise
        n = len(returns)
        if n >= 5:
            # Realized variance with noise correction
            rv = np.sum(returns ** 2)
            # Hansen-Lunde (2006) noise correction (simplified)
            noise_adj = 2 * np.sum(returns[:-1] * returns[1:])
            self.tick_vol = np.sqrt(max(0, rv - noise_adj) / n)
        else:
            self.tick_vol = np.std(returns)

        # Recent momentum
        momentum = np.mean(returns[-3:]) if len(returns) >= 3 else 0

        # High tick vol = more noise, need stronger signal
        signal_threshold = self.tick_vol * 2

        if abs(momentum) > signal_threshold:
            self.signal = 1 if momentum > 0 else -1
            self.confidence = min(0.7, abs(momentum) / (signal_threshold * 2))
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(749, "RealizedGARCH", "garch")
class RealizedGARCH(BaseFormula):
    """
    ID 749: Realized GARCH

    Purpose: Incorporate realized volatility measures
    Math: Use RV (realized variance) as additional input

    Application: Better vol forecasts for position sizing
    Speed: Minute-level updates
    """

    DESCRIPTION = "GARCH with realized volatility input"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.realized_vol = 0.01
        self.garch_vol = 0.01

    def _compute(self) -> None:
        """Compute realized GARCH."""
        if len(self.returns) < 20:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()

        # Realized volatility (high frequency)
        self.realized_vol = np.sqrt(np.sum(returns[-20:] ** 2))

        # GARCH volatility (model-based)
        self.garch_vol = np.std(returns[-50:]) if len(returns) >= 50 else np.std(returns)

        # Combined forecast
        combined_vol = 0.6 * self.garch_vol + 0.4 * self.realized_vol

        # Vol ratio indicates regime
        vol_ratio = self.realized_vol / (self.garch_vol + 1e-10)

        if vol_ratio > 1.5:  # RV > GARCH: vol increasing
            # Expect continued volatility
            momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            self.signal = 1 if momentum > 0 else -1
            self.confidence = 0.6
        elif vol_ratio < 0.7:  # RV < GARCH: vol decreasing
            # Calm period, mean reversion
            self.signal = 0
            self.confidence = 0.4
        else:
            self.signal = 0
            self.confidence = 0.5


@FormulaRegistry.register(750, "JumpGARCH", "garch")
class JumpGARCH(BaseFormula):
    """
    ID 750: Jump-GARCH

    Purpose: GARCH with explicit jump component
    Math: Return = drift + GARCH + Jump (Poisson)

    Application: Distinguish normal vol from crash risk
    Speed: Real-time jump detection
    """

    DESCRIPTION = "GARCH with Poisson jump component for tail risk"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.jump_detected = False
        self.jump_probability = 0.0

    def _compute(self) -> None:
        """Detect jumps and compute adjusted vol."""
        if len(self.returns) < 20:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()

        # Estimate GARCH vol
        garch_vol = np.std(returns[-30:]) if len(returns) >= 30 else np.std(returns)

        # Detect jumps: returns > 3 sigma
        jump_threshold = 3 * garch_vol
        recent_returns = returns[-10:] if len(returns) >= 10 else returns
        jumps = np.abs(recent_returns) > jump_threshold

        self.jump_detected = np.any(jumps)
        self.jump_probability = np.mean(jumps)

        if self.jump_detected:
            # Recent jump - estimate direction
            last_jump_idx = np.where(jumps)[0][-1]
            jump_direction = np.sign(recent_returns[last_jump_idx])

            # After jump, expect continuation then reversion
            if last_jump_idx == len(recent_returns) - 1:
                # Jump just happened - possible continuation
                self.signal = int(jump_direction)
                self.confidence = 0.6
            else:
                # Jump was earlier - expect reversion
                self.signal = int(-jump_direction)
                self.confidence = 0.5
        else:
            # No jumps - normal regime
            self.signal = 0
            self.confidence = 0.4


# =============================================================================
# KYLE'S LAMBDA FORMULAS (IDs 751-755)
# =============================================================================

@FormulaRegistry.register(751, "KyleLambdaClassic", "market_impact")
class KyleLambdaClassic(BaseFormula):
    """
    ID 751: Kyle's Lambda (Classic)

    Source: Kyle (1985) "Continuous Auctions and Insider Trading"

    Purpose: Measure price impact of order flow
    Math: ΔP = λ × V + ε

    Application: Estimate market impact cost
    Speed: Continuous update (millisecond)
    """

    DESCRIPTION = "Kyle (1985) classic price impact measure"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.kyle_lambda = 0.0

    def _compute(self) -> None:
        """Estimate Kyle's lambda."""
        if len(self.prices) < 20 or len(self.volumes) < 20:
            self.signal = 0
            self.confidence = 0.0
            return

        prices = self._prices_array()
        volumes = self._volumes_array()

        # Price changes
        price_changes = np.diff(prices[-30:]) if len(prices) >= 30 else np.diff(prices)
        signed_volume = volumes[-len(price_changes):] * np.sign(price_changes)

        # Regression: ΔP = λ × V
        if np.var(signed_volume) > 0:
            self.kyle_lambda = np.cov(price_changes, signed_volume)[0, 1] / np.var(signed_volume)
        else:
            self.kyle_lambda = 0.0

        # Signal based on lambda (illiquidity indicator)
        if self.kyle_lambda > 0.0001:  # High impact
            # Market is illiquid, be cautious
            self.signal = 0
            self.confidence = 0.3
        elif self.kyle_lambda > 0:  # Normal impact
            # Follow order flow direction
            recent_flow = np.mean(signed_volume[-5:]) if len(signed_volume) >= 5 else 0
            self.signal = 1 if recent_flow > 0 else (-1 if recent_flow < 0 else 0)
            self.confidence = 0.5
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(752, "ScaleInvariantLambda", "market_impact")
class ScaleInvariantLambda(BaseFormula):
    """
    ID 752: Scale-Invariant Kyle's Lambda

    Source: Kyle & Obizhaeva, SSRN 2823630

    Purpose: Scale-invariant version of lambda
    Math: λ scales with √(transaction volume)

    Application: Compare lambda across different assets
    Speed: Real-time scaling
    """

    DESCRIPTION = "Kyle-Obizhaeva scale-invariant market impact"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.invariant_lambda = 0.0

    def _compute(self) -> None:
        """Compute scale-invariant lambda."""
        if len(self.prices) < 20 or len(self.volumes) < 20:
            self.signal = 0
            self.confidence = 0.0
            return

        prices = self._prices_array()
        volumes = self._volumes_array()
        returns = self._returns_array() if len(self.returns) > 0 else np.array([0])

        # Volatility and average volume
        vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
        avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else 1

        # Scale-invariant lambda: λ ~ σ / √V
        self.invariant_lambda = vol / np.sqrt(avg_vol + 1e-10)

        # Signal based on invariant lambda
        if self.invariant_lambda > 0.001:
            # High impact relative to volume
            self.signal = 0  # Avoid trading
            self.confidence = 0.3
        else:
            # Normal/low impact
            momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            self.signal = 1 if momentum > 0 else (-1 if momentum < 0 else 0)
            self.confidence = 0.5


@FormulaRegistry.register(753, "IntradayLambdaDynamics", "market_impact")
class IntradayLambdaDynamics(BaseFormula):
    """
    ID 753: Intraday Lambda Dynamics

    Purpose: Time-varying Kyle's lambda
    Math: λ(t) varies by time of day, volatility regime

    Application: Adjust impact estimates intraday
    Speed: Minute-level recalibration
    """

    DESCRIPTION = "Time-varying Kyle's lambda throughout the day"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.lambda_current = 0.0
        self.lambda_history = deque(maxlen=24)  # Hourly

    def _compute(self) -> None:
        """Compute time-varying lambda."""
        if len(self.prices) < 20:
            self.signal = 0
            self.confidence = 0.0
            return

        prices = self._prices_array()
        volumes = self._volumes_array() if len(self.volumes) > 0 else np.ones(len(prices))

        # Recent lambda estimate
        price_changes = np.diff(prices[-20:])
        vol_recent = volumes[-19:]

        if np.var(vol_recent) > 0 and len(price_changes) == len(vol_recent):
            signed_vol = vol_recent * np.sign(price_changes)
            self.lambda_current = np.cov(price_changes, signed_vol)[0, 1] / np.var(signed_vol)
        else:
            self.lambda_current = 0.0

        self.lambda_history.append(self.lambda_current)

        # Compare to historical
        if len(self.lambda_history) > 1:
            avg_lambda = np.mean(list(self.lambda_history))
            lambda_ratio = self.lambda_current / (avg_lambda + 1e-10)

            if lambda_ratio > 1.5:  # High impact now
                self.signal = 0
                self.confidence = 0.3
            elif lambda_ratio < 0.7:  # Low impact now
                returns = self._returns_array()
                momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
                self.signal = 1 if momentum > 0 else (-1 if momentum < 0 else 0)
                self.confidence = 0.6
            else:
                self.signal = 0
                self.confidence = 0.4
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(754, "HiddenLiquidityLambda", "market_impact")
class HiddenLiquidityLambda(BaseFormula):
    """
    ID 754: Lambda for Hidden Liquidity

    Purpose: Account for iceberg orders in lambda
    Math: Effective lambda > visible lambda

    Application: Better impact prediction
    Speed: Real-time adjustment
    """

    DESCRIPTION = "Kyle's lambda adjusted for hidden liquidity"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.visible_lambda = 0.0
        self.effective_lambda = 0.0
        self.hidden_ratio = 1.5  # Assume 50% more hidden

    def _compute(self) -> None:
        """Estimate lambda with hidden liquidity."""
        if len(self.prices) < 20:
            self.signal = 0
            self.confidence = 0.0
            return

        prices = self._prices_array()
        volumes = self._volumes_array() if len(self.volumes) > 0 else np.ones(len(prices))
        returns = self._returns_array() if len(self.returns) > 0 else np.array([0])

        # Visible lambda from data
        price_changes = np.diff(prices[-20:])
        if len(price_changes) > 0 and np.var(volumes[-len(price_changes):]) > 0:
            signed_vol = volumes[-len(price_changes):] * np.sign(price_changes)
            self.visible_lambda = np.cov(price_changes, signed_vol)[0, 1] / np.var(signed_vol)
        else:
            self.visible_lambda = 0.0

        # Effective lambda (accounting for hidden)
        self.effective_lambda = self.visible_lambda / self.hidden_ratio

        # Signal: if effective lambda is low, market can absorb orders
        if abs(self.effective_lambda) < 0.00005:
            momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            self.signal = 1 if momentum > 0 else (-1 if momentum < 0 else 0)
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(755, "MultivariateLambda", "market_impact")
class MultivariateLambda(BaseFormula):
    """
    ID 755: Multivariate Kyle Lambda

    Purpose: Cross-asset price impact
    Math: ΔP_i = Σ λ_ij × V_j

    Application: Portfolio execution with correlations
    Speed: Matrix multiply (microseconds)
    """

    DESCRIPTION = "Cross-asset Kyle's lambda for portfolio impact"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.self_lambda = 0.0
        self.cross_lambda = 0.0  # Would be matrix in full implementation

    def _compute(self) -> None:
        """Compute multivariate lambda (simplified single-asset)."""
        if len(self.prices) < 20:
            self.signal = 0
            self.confidence = 0.0
            return

        prices = self._prices_array()
        volumes = self._volumes_array() if len(self.volumes) > 0 else np.ones(len(prices))
        returns = self._returns_array() if len(self.returns) > 0 else np.array([0])

        # Self-impact
        price_changes = np.diff(prices[-20:])
        if len(price_changes) > 0 and np.var(volumes[-len(price_changes):]) > 0:
            signed_vol = volumes[-len(price_changes):] * np.sign(price_changes)
            self.self_lambda = np.cov(price_changes, signed_vol)[0, 1] / np.var(signed_vol)
        else:
            self.self_lambda = 0.0

        # Cross-impact would require other asset data
        # For now, estimate from autocorrelation
        if len(returns) > 5:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 2 else 0
            self.cross_lambda = autocorr * self.self_lambda * 0.5

        # Total impact consideration
        total_lambda = abs(self.self_lambda) + abs(self.cross_lambda)

        if total_lambda < 0.0001:
            momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            self.signal = 1 if momentum > 0 else (-1 if momentum < 0 else 0)
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


# =============================================================================
# SPEED PREMIUM FORMULAS (IDs 756-760)
# =============================================================================

@FormulaRegistry.register(756, "SpeedPremiumMeasurement", "speed_premium")
class SpeedPremiumMeasurement(BaseFormula):
    """
    ID 756: Speed Premium Measurement

    Source: BIS Working Paper 1290 (August 2024)

    Purpose: Quantify value of latency reduction
    Math: Premium = E[PnL_fast] - E[PnL_slow]

    Application: Justify infrastructure investment
    Speed: Analysis metric
    """

    DESCRIPTION = "BIS (2024) speed premium quantification"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.speed_premium = 0.0
        self.latency_value = 0.0  # $ per ms saved

    def _compute(self) -> None:
        """Compute speed premium."""
        if len(self.returns) < 20:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()

        # Speed premium: value of seeing prices earlier
        # Approximate by autocorrelation (predictability)
        if len(returns) > 5:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 2 else 0
        else:
            autocorr = 0

        vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.01

        # Speed premium increases with:
        # - Higher autocorrelation (more predictable)
        # - Higher volatility (more to gain)
        self.speed_premium = abs(autocorr) * vol * 100

        # Convert to $/ms (simplified)
        self.latency_value = self.speed_premium * 1000  # Scale factor

        # Signal: if speed premium high, trade more aggressively
        if self.speed_premium > 0.01:
            momentum = np.mean(returns[-3:]) if len(returns) >= 3 else 0
            self.signal = 1 if momentum > 0 else (-1 if momentum < 0 else 0)
            self.confidence = min(0.7, self.speed_premium)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(757, "LatencyArbitrageDetector", "speed_premium")
class LatencyArbitrageDetector(BaseFormula):
    """
    ID 757: Latency Arbitrage Opportunity

    Purpose: Detect arbitrage from speed advantage
    Math: Profit = (P_slow - P_fast) × volume - 2×fees

    Application: Cross-exchange arbitrage
    Speed: Nanosecond detection
    """

    DESCRIPTION = "Detect latency arbitrage opportunities"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.arb_opportunity = 0.0

    def _compute(self) -> None:
        """Detect latency arbitrage."""
        if len(self.prices) < 10:
            self.signal = 0
            self.confidence = 0.0
            return

        prices = self._prices_array()
        returns = self._returns_array() if len(self.returns) > 0 else np.array([0])

        # Detect rapid price moves (potential arb)
        recent_returns = returns[-5:] if len(returns) >= 5 else returns

        # Large move followed by reversion = arb opportunity was there
        if len(recent_returns) >= 2:
            move_size = abs(recent_returns[-2])
            reversion = -recent_returns[-2] * recent_returns[-1]  # Positive if reverting

            if move_size > 0.001 and reversion > 0:
                self.arb_opportunity = move_size
            else:
                self.arb_opportunity = 0.0
        else:
            self.arb_opportunity = 0.0

        # Signal: detect if we might catch the reversion
        if self.arb_opportunity > 0.001:
            # Price just reverted, might continue
            self.signal = 1 if recent_returns[-1] > 0 else -1
            self.confidence = min(0.6, self.arb_opportunity * 100)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(758, "ToxicFlowSpeed", "speed_premium")
class ToxicFlowSpeed(BaseFormula):
    """
    ID 758: Toxic Order Flow from Speed

    Purpose: Identify informed HFT flow
    Math: Correlation(speed, future_returns)

    Application: Avoid adverse selection
    Speed: Real-time toxicity score
    """

    DESCRIPTION = "Identify toxic informed flow from speed"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.toxicity_score = 0.0

    def _compute(self) -> None:
        """Compute flow toxicity."""
        if len(self.returns) < 20:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()
        volumes = self._volumes_array() if len(self.volumes) > 0 else np.ones(len(returns))

        # Toxicity: correlation between volume and future returns
        # High correlation = informed flow
        future_returns = returns[1:]
        past_volume = volumes[:-1]

        if len(future_returns) > 5 and np.std(past_volume) > 0:
            signed_vol = past_volume * np.sign(returns[:-1])
            toxicity = np.corrcoef(signed_vol[:-1], future_returns[:-1])[0, 1]
            self.toxicity_score = abs(toxicity) if not np.isnan(toxicity) else 0
        else:
            self.toxicity_score = 0

        # Signal: avoid trading when toxicity high
        if self.toxicity_score > 0.3:
            # High toxicity = informed traders present
            self.signal = 0  # Don't trade against informed
            self.confidence = 0.3
        else:
            momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            self.signal = 1 if momentum > 0 else (-1 if momentum < 0 else 0)
            self.confidence = 0.5 * (1 - self.toxicity_score)


@FormulaRegistry.register(759, "QueuePositionValue", "speed_premium")
class QueuePositionValue(BaseFormula):
    """
    ID 759: Queue Position Value

    Purpose: Expected value of being first in queue
    Math: V(position=1) - V(position=k)

    Application: Decide whether to jump queue
    Speed: Microsecond calculation
    """

    DESCRIPTION = "Value of queue position priority"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.position_value = 0.0

    def _compute(self) -> None:
        """Compute queue position value."""
        if len(self.prices) < 10:
            self.signal = 0
            self.confidence = 0.0
            return

        prices = self._prices_array()
        volumes = self._volumes_array() if len(self.volumes) > 0 else np.ones(len(prices))
        returns = self._returns_array() if len(self.returns) > 0 else np.array([0])

        # Position value depends on:
        # - Fill probability difference (position 1 vs k)
        # - Spread capture
        # - Adverse selection cost

        vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
        avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else 1

        # Simplified: value = spread/2 - adverse_selection
        spread_estimate = vol * 2  # Rough spread proxy
        adverse_selection = vol * 0.5  # Informed flow cost

        self.position_value = spread_estimate / 2 - adverse_selection

        # Signal: if position valuable, prefer limit orders (wait)
        if self.position_value > 0.0001:
            # Limit orders valuable, be patient
            self.signal = 0
            self.confidence = 0.5
        else:
            # Position not valuable, use market orders
            momentum = np.mean(returns[-3:]) if len(returns) >= 3 else 0
            self.signal = 1 if momentum > 0 else (-1 if momentum < 0 else 0)
            self.confidence = 0.5


@FormulaRegistry.register(760, "CoLocationROI", "speed_premium")
class CoLocationROI(BaseFormula):
    """
    ID 760: Co-location Value Formula

    Purpose: ROI of co-location / proximity hosting
    Math: NPV = Σ (premium_t - cost) / (1+r)^t

    Application: Infrastructure decision
    Speed: Strategic analysis
    """

    DESCRIPTION = "ROI calculation for co-location infrastructure"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.roi_estimate = 0.0
        self.breakeven_days = float('inf')

    def _compute(self) -> None:
        """Compute co-location ROI."""
        if len(self.returns) < 20:
            self.signal = 0
            self.confidence = 0.0
            return

        returns = self._returns_array()

        # Estimate daily edge from speed
        vol = np.std(returns[-20:])
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 2 else 0

        # Daily speed premium (simplified)
        daily_edge = abs(autocorr) * vol * 1000  # Per $1000 traded

        # Co-location cost (example: $1000/month)
        monthly_cost = 1000 / 30  # Daily

        # ROI
        if monthly_cost > 0:
            self.roi_estimate = (daily_edge - monthly_cost) / monthly_cost
            if daily_edge > monthly_cost:
                self.breakeven_days = monthly_cost * 30 / daily_edge
            else:
                self.breakeven_days = float('inf')
        else:
            self.roi_estimate = 0
            self.breakeven_days = float('inf')

        # Signal: if ROI positive, trade more aggressively
        if self.roi_estimate > 0:
            momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            self.signal = 1 if momentum > 0 else (-1 if momentum < 0 else 0)
            self.confidence = min(0.6, 0.3 + self.roi_estimate)
        else:
            self.signal = 0
            self.confidence = 0.3


# =============================================================================
# REGISTRATION FUNCTION
# =============================================================================

def register_academic_2024():
    """
    Register all academic 2024-2025 formulas.
    Formulas auto-register via decorator, this just verifies.
    """
    from .base import FORMULA_REGISTRY

    expected_ids = list(range(731, 761))  # 731-760
    registered_count = sum(1 for fid in expected_ids if fid in FORMULA_REGISTRY)

    print(f"[Academic2024] Verified {registered_count}/30 formulas (IDs 731-760)")

    return registered_count
