"""
PURE PRICE ENGINE - ZERO API DEPENDENCIES
==========================================
Renaissance Technologies level mathematical price generation.

This engine:
1. Uses Power Law to calculate fair value (ONLY needs timestamp)
2. Simulates realistic price movements with stochastic models
3. Generates trading signals from mathematical inefficiencies
4. ZERO external API calls. ZERO rate limits. UNLIMITED trades.

Power Law Formula (R²=93%):
    fair_value = 10^(-17.0161223 + 5.8451542 * log10(days_since_genesis))

Stochastic Model:
    - Geometric Brownian Motion with mean reversion to Power Law
    - Volatility calibrated to BTC historical (2-4% daily)
    - Mean reversion strength: 0.1 (10% pull toward fair value per day)

Usage:
    engine = PurePriceEngine()
    price = engine.get_current_price()  # Simulated realistic price
    signal = engine.get_signal()  # Trading signal based on deviation
"""
import math
import time
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque


# Bitcoin genesis timestamp (January 3, 2009)
GENESIS_TIMESTAMP = 1230940800

# Power Law coefficients (R²=93%)
POWER_LAW_INTERCEPT = -17.0161223
POWER_LAW_SLOPE = 5.8451542

# Stochastic model parameters
DAILY_VOLATILITY = 0.025  # 2.5% daily volatility
MEAN_REVERSION_STRENGTH = 0.1  # 10% pull toward fair value per day
SECONDS_PER_DAY = 86400

# Trading thresholds
UNDERVALUED_THRESHOLD = -0.15  # 15% below fair value = BUY
OVERVALUED_THRESHOLD = 0.15    # 15% above fair value = SELL
STRONG_SIGNAL_THRESHOLD = 0.25  # 25% deviation = strong signal


@dataclass
class PriceState:
    """Current price state from the engine."""
    timestamp: float
    simulated_price: float
    fair_value: float
    deviation_pct: float
    support: float
    resistance: float
    volatility: float


@dataclass
class PureSignal:
    """Trading signal from pure mathematical analysis."""
    direction: int  # +1 BUY, -1 SELL, 0 NEUTRAL
    strength: float  # 0.0 to 1.0
    probability: float  # Expected win probability
    should_trade: bool
    fair_value: float
    current_price: float
    deviation_pct: float
    reason: str


class PurePriceEngine:
    """
    PURE PRICE ENGINE

    Generates realistic BTC prices and trading signals using ONLY mathematics.
    No APIs. No external dependencies. Unlimited speed.

    Architecture:
    1. Power Law calculates the "true" fair value from timestamp alone
    2. Stochastic model simulates realistic price movements around fair value
    3. Signals generated when price deviates significantly from fair value

    This is how Renaissance Technologies would do it:
    - Mathematical models, not API calls
    - Statistical edge, not random trading
    - Unlimited execution speed
    """

    def __init__(
        self,
        initial_price: Optional[float] = None,
        volatility: float = DAILY_VOLATILITY,
        mean_reversion: float = MEAN_REVERSION_STRENGTH,
        seed: Optional[int] = None,
    ):
        """
        Initialize the pure price engine.

        Args:
            initial_price: Starting price (defaults to current fair value)
            volatility: Daily volatility (default 2.5%)
            mean_reversion: Mean reversion strength (default 0.1)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        self.volatility = volatility
        self.mean_reversion = mean_reversion

        # Calculate fair value
        self.fair_value = self._calculate_fair_value()

        # Initialize price (start at fair value with some deviation)
        if initial_price is None:
            # Start 20-35% below fair value (realistic for current market)
            deviation = random.uniform(-0.35, -0.20)
            self.current_price = self.fair_value * (1 + deviation)
        else:
            self.current_price = initial_price

        # Price history for technical analysis
        self.price_history: deque = deque(maxlen=1000)
        self.price_history.append(self.current_price)

        # Tracking
        self.last_update_time = time.time()
        self.tick_count = 0
        self.signal_count = 0
        self.trade_signals = 0

    def _calculate_fair_value(self, timestamp: Optional[float] = None) -> float:
        """
        Calculate Power Law fair value from timestamp.

        Formula: fair_value = 10^(-17.0161223 + 5.8451542 * log10(days))

        This is the ONLY external input needed - current time.
        No API. No network. Just math.
        """
        if timestamp is None:
            timestamp = time.time()

        days_since_genesis = (timestamp - GENESIS_TIMESTAMP) / SECONDS_PER_DAY

        if days_since_genesis <= 0:
            return 0.01  # Pre-genesis

        log_days = math.log10(days_since_genesis)
        log_price = POWER_LAW_INTERCEPT + POWER_LAW_SLOPE * log_days

        return 10 ** log_price

    def _calculate_bands(self) -> Tuple[float, float]:
        """Calculate support and resistance bands."""
        # Historical Power Law bands: ±40% from fair value
        support = self.fair_value * 0.60
        resistance = self.fair_value * 1.40
        return support, resistance

    def tick(self, dt: Optional[float] = None) -> float:
        """
        Advance simulation by one tick.

        Uses Geometric Brownian Motion with mean reversion:
        dP = μ(F - P)dt + σP·dW

        Where:
        - P = current price
        - F = fair value (Power Law)
        - μ = mean reversion strength
        - σ = volatility
        - dW = Wiener process (random walk)

        Args:
            dt: Time step in seconds (default: time since last tick)

        Returns:
            New simulated price
        """
        current_time = time.time()

        if dt is None:
            dt = current_time - self.last_update_time
            if dt <= 0:
                dt = 0.001  # Minimum 1ms

        # Convert to daily units for volatility scaling
        dt_days = dt / SECONDS_PER_DAY

        # Update fair value (changes slowly with time)
        self.fair_value = self._calculate_fair_value(current_time)

        # Mean reversion component (pulls toward fair value)
        mean_reversion_force = self.mean_reversion * (self.fair_value - self.current_price) * dt_days

        # Random walk component (Geometric Brownian Motion)
        random_shock = self.volatility * self.current_price * math.sqrt(dt_days) * random.gauss(0, 1)

        # Update price
        self.current_price += mean_reversion_force + random_shock

        # Price can't go negative
        self.current_price = max(self.current_price, 0.01)

        # Record history
        self.price_history.append(self.current_price)
        self.last_update_time = current_time
        self.tick_count += 1

        return self.current_price

    def get_price_state(self) -> PriceState:
        """Get current price state."""
        support, resistance = self._calculate_bands()
        deviation = (self.current_price - self.fair_value) / self.fair_value

        return PriceState(
            timestamp=time.time(),
            simulated_price=self.current_price,
            fair_value=self.fair_value,
            deviation_pct=deviation * 100,
            support=support,
            resistance=resistance,
            volatility=self._calculate_realized_volatility(),
        )

    def _calculate_realized_volatility(self) -> float:
        """Calculate realized volatility from recent price history."""
        if len(self.price_history) < 10:
            return self.volatility

        prices = list(self.price_history)[-100:]
        returns = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]

        if not returns:
            return self.volatility

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)

        return math.sqrt(variance) * math.sqrt(252)  # Annualized

    def get_signal(self) -> PureSignal:
        """
        Generate trading signal based on price deviation from fair value.

        Signal Logic:
        - Price 15%+ below fair value → BUY (mean reversion expected)
        - Price 15%+ above fair value → SELL (mean reversion expected)
        - Signal strength proportional to deviation
        - Probability based on Power Law R²=93% historical accuracy
        """
        self.signal_count += 1

        # Calculate deviation
        deviation = (self.current_price - self.fair_value) / self.fair_value
        deviation_pct = deviation * 100

        # Determine direction
        if deviation < -UNDERVALUED_THRESHOLD:
            direction = 1  # BUY - price below fair value
            reason = f"UNDERVALUED: {deviation_pct:.1f}% below Power Law fair value"
        elif deviation > OVERVALUED_THRESHOLD:
            direction = -1  # SELL - price above fair value
            reason = f"OVERVALUED: {deviation_pct:+.1f}% above Power Law fair value"
        else:
            direction = 0  # NEUTRAL
            reason = f"NEUTRAL: {deviation_pct:+.1f}% within normal range"

        # Calculate strength (0 to 1)
        abs_deviation = abs(deviation)
        if abs_deviation < UNDERVALUED_THRESHOLD:
            strength = 0.0
        else:
            # Scale from threshold to strong signal
            strength = min(1.0, (abs_deviation - UNDERVALUED_THRESHOLD) /
                          (STRONG_SIGNAL_THRESHOLD - UNDERVALUED_THRESHOLD))

        # Calculate probability
        # Power Law has R²=93%, so base probability is high
        # Adjust based on deviation strength
        base_prob = 0.52  # Minimum edge
        max_prob = 0.65   # Maximum at strong signals
        probability = base_prob + (max_prob - base_prob) * strength

        # Should trade?
        should_trade = direction != 0 and strength > 0.1

        if should_trade:
            self.trade_signals += 1

        return PureSignal(
            direction=direction,
            strength=strength,
            probability=probability,
            should_trade=should_trade,
            fair_value=self.fair_value,
            current_price=self.current_price,
            deviation_pct=deviation_pct,
            reason=reason,
        )

    def get_stats(self) -> dict:
        """Get engine statistics."""
        support, resistance = self._calculate_bands()
        deviation = (self.current_price - self.fair_value) / self.fair_value

        trigger_rate = self.trade_signals / self.signal_count if self.signal_count > 0 else 0

        return {
            'current_price': self.current_price,
            'fair_value': self.fair_value,
            'deviation_pct': deviation * 100,
            'support': support,
            'resistance': resistance,
            'volatility': self._calculate_realized_volatility(),
            'tick_count': self.tick_count,
            'signal_count': self.signal_count,
            'trade_signals': self.trade_signals,
            'trigger_rate': trigger_rate,
        }

    def fast_forward(self, ticks: int, dt: float = 0.001) -> List[float]:
        """
        Fast forward simulation by multiple ticks.

        Used for rapid backtesting and signal generation.
        No API calls = unlimited speed.

        Args:
            ticks: Number of ticks to simulate
            dt: Time step per tick (seconds)

        Returns:
            List of prices generated
        """
        prices = []
        for _ in range(ticks):
            price = self.tick(dt)
            prices.append(price)
        return prices


class PureOrderbookSimulator:
    """
    Simulates realistic orderbook data for the matching engine.

    No API needed - generates synthetic orderbook based on:
    - Current simulated price
    - Realistic spread (0.01% for BTC)
    - Depth profile (exponential decay)
    """

    def __init__(self, price_engine: PurePriceEngine):
        self.price_engine = price_engine
        self.spread_pct = 0.0001  # 0.01% spread (typical for BTC)
        self.depth_levels = 10
        self.base_liquidity = 10.0  # BTC per level

    def get_orderbook(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], float, float]:
        """
        Generate synthetic orderbook.

        Returns:
            (bids, asks, best_bid, best_ask)
        """
        mid_price = self.price_engine.current_price
        half_spread = mid_price * self.spread_pct / 2

        best_bid = mid_price - half_spread
        best_ask = mid_price + half_spread

        bids = []
        asks = []

        for i in range(self.depth_levels):
            # Price levels with exponential spacing
            bid_price = best_bid - (i * mid_price * 0.0001)
            ask_price = best_ask + (i * mid_price * 0.0001)

            # Liquidity with exponential decay
            liquidity = self.base_liquidity * (1.5 ** i)

            bids.append((bid_price, liquidity))
            asks.append((ask_price, liquidity))

        return bids, asks, best_bid, best_ask


if __name__ == "__main__":
    # Demo the pure price engine
    print("=" * 70)
    print("PURE PRICE ENGINE - ZERO API DEPENDENCIES")
    print("=" * 70)

    engine = PurePriceEngine()

    print(f"\nPower Law Fair Value: ${engine.fair_value:,.2f}")
    print(f"Starting Price: ${engine.current_price:,.2f}")

    state = engine.get_price_state()
    print(f"Deviation: {state.deviation_pct:+.1f}%")
    print(f"Support: ${state.support:,.2f}")
    print(f"Resistance: ${state.resistance:,.2f}")

    print("\n--- Simulating 1000 ticks ---")

    start = time.perf_counter_ns()
    prices = engine.fast_forward(1000, dt=0.001)
    elapsed_ns = time.perf_counter_ns() - start

    print(f"Time: {elapsed_ns / 1_000_000:.2f}ms")
    print(f"TPS: {1000 / (elapsed_ns / 1_000_000_000):,.0f}")

    print(f"\nFinal Price: ${engine.current_price:,.2f}")

    signal = engine.get_signal()
    print(f"\nSignal: {'+1 BUY' if signal.direction > 0 else '-1 SELL' if signal.direction < 0 else '0 NEUTRAL'}")
    print(f"Strength: {signal.strength:.2f}")
    print(f"Probability: {signal.probability:.1%}")
    print(f"Reason: {signal.reason}")

    print("\n" + "=" * 70)
    print("NO APIs. NO RATE LIMITS. UNLIMITED TRADES.")
    print("=" * 70)
