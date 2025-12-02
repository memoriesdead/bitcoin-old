"""
PURE BLOCKCHAIN SIMULATION - Renaissance Level Trading
=======================================================
Formula IDs 801-804: Pure blockchain data simulation

ALL data derived from blockchain constants and timestamps.
ZERO API calls. ZERO external dependencies.

Formulas:
- 801: BlockTimeVolatility - Volatility from block interval variance
- 802: MempoolFlowSimulator - TX flow from block patterns
- 803: DeterministicChaosPrice - Lorenz attractor price dynamics
- 804: UTXOWhaleDetector - Whale detection from block size patterns
"""

import math
import time
import numpy as np
from typing import Tuple, Dict, Any
from collections import deque
from numba import njit, float64
from dataclasses import dataclass

# =============================================================================
# BLOCKCHAIN CONSTANTS (Pure Protocol - No APIs)
# =============================================================================
GENESIS_TS = 1230768000.0           # Jan 1, 2009
BLOCKS_PER_HALVING = 210000
INITIAL_REWARD = 50.0
CURRENT_REWARD = 3.125              # After 4th halving
SECONDS_PER_BLOCK = 600.0           # Target block time
BLOCKS_PER_DAY = 144.0
MAX_SUPPLY = 21_000_000.0

# Power Law constants (93%+ correlation over 14 years)
POWER_LAW_A = -17.0161223
POWER_LAW_B = 5.8451542

# Volatility constants
BASE_DAILY_VOLATILITY = 0.03        # ~3% daily volatility base
BLOCK_TIME_VOL_MULTIPLIER = 2.0     # Sensitivity to block time variance

# Volume constants
ONCHAIN_VOLUME_BASE = 750.0         # Base multiplier for miner output
ONCHAIN_VOLUME_AMPLITUDE = 250.0    # Cyclical amplitude

# Chaos constants (Lorenz attractor)
LORENZ_SIGMA = 10.0
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0

# =============================================================================
# JIT-COMPILED CORE FUNCTIONS
# =============================================================================

@njit(cache=True, fastmath=True)
def calculate_power_law_price(timestamp: float) -> float:
    """
    Calculate fair value price from Power Law.

    Formula: Price = 10^(a + b * log10(days))
    Source: Giovanni Santostasi research (93%+ correlation)
    """
    days = (timestamp - GENESIS_TS) / 86400.0
    if days <= 0:
        return 0.0
    log10_days = math.log10(days)
    return 10.0 ** (POWER_LAW_A + POWER_LAW_B * log10_days)


@njit(cache=True, fastmath=True)
def calculate_halving_cycle(timestamp: float) -> Tuple[int, float]:
    """
    Calculate halving position from timestamp.

    Returns:
        (halving_number, cycle_progress)
    """
    seconds_since_genesis = timestamp - GENESIS_TS
    estimated_blocks = seconds_since_genesis / SECONDS_PER_BLOCK
    halving_number = int(estimated_blocks // BLOCKS_PER_HALVING)
    blocks_in_cycle = estimated_blocks % BLOCKS_PER_HALVING
    cycle_progress = blocks_in_cycle / BLOCKS_PER_HALVING
    return halving_number, cycle_progress


@njit(cache=True, fastmath=True)
def calculate_block_time_volatility(timestamp: float) -> float:
    """
    ID 801: BlockTimeVolatility

    Derive volatility from expected block time variance.
    In reality, block times deviate from 600s target.
    We simulate this using a sinusoidal pattern based on difficulty adjustment.

    Difficulty adjusts every 2016 blocks (~2 weeks).
    """
    seconds_since_genesis = timestamp - GENESIS_TS
    blocks = seconds_since_genesis / SECONDS_PER_BLOCK

    # Difficulty adjustment cycle (2016 blocks)
    difficulty_cycle = (blocks % 2016) / 2016.0

    # Block time variance follows sinusoidal pattern
    # Before adjustment: blocks slow down or speed up
    # After adjustment: corrects toward target
    block_time_deviation = 0.1 * math.sin(difficulty_cycle * 2 * math.pi)

    # Volatility increases when blocks deviate from target
    volatility = BASE_DAILY_VOLATILITY * (1.0 + abs(block_time_deviation) * BLOCK_TIME_VOL_MULTIPLIER)

    return volatility


@njit(cache=True, fastmath=True)
def calculate_mempool_flow(timestamp: float, halving_cycle: float) -> Tuple[float, float]:
    """
    ID 802: MempoolFlowSimulator

    Simulate mempool transaction flow from:
    1. Miner reward (3.125 BTC/block)
    2. Halving cycle position
    3. Cyclical on-chain activity patterns

    Returns:
        (daily_volume_btc, ofi_signal)
    """
    # Miner daily output
    miner_daily = CURRENT_REWARD * BLOCKS_PER_DAY

    # On-chain multiplier varies with halving cycle
    # Early cycle = less activity, late cycle = more activity
    onchain_mult = ONCHAIN_VOLUME_BASE + ONCHAIN_VOLUME_AMPLITUDE * math.sin(halving_cycle * 2 * math.pi)

    # Daily volume estimate
    daily_volume = miner_daily * onchain_mult

    # OFI signal from halving cycle
    # Early cycle tends bullish, late cycle mixed
    if halving_cycle < 0.25:
        ofi_signal = 0.7  # Early post-halving: bullish
    elif halving_cycle < 0.5:
        ofi_signal = 0.5  # Mid cycle: neutral to bullish
    elif halving_cycle < 0.75:
        ofi_signal = 0.3  # Late mid: caution
    else:
        ofi_signal = 0.6  # Pre-halving: accumulation

    return daily_volume, ofi_signal


@njit(cache=True, fastmath=True)
def lorenz_step(x: float, y: float, z: float, dt: float = 0.001) -> Tuple[float, float, float]:
    """
    Single step of Lorenz attractor for deterministic chaos.

    Properties:
    - Deterministic: same inputs = same outputs
    - Chaotic: small changes cause large divergence
    - Bounded: oscillates within finite range
    """
    dx = LORENZ_SIGMA * (y - x)
    dy = x * (LORENZ_RHO - z) - y
    dz = x * y - LORENZ_BETA * z

    return x + dx * dt, y + dy * dt, z + dz * dt


@njit(cache=True, fastmath=True)
def calculate_chaos_price(timestamp: float, true_price: float, volatility: float) -> float:
    """
    ID 803: DeterministicChaosPrice

    Generate market price using Lorenz attractor for realistic dynamics.

    Key insight: Use timestamp as seed for deterministic chaos.
    Same timestamp always produces same price path.
    """
    # Initialize from timestamp (deterministic)
    seed_val = (timestamp * 1e6) % 1000
    x = 1.0 + (seed_val % 10) / 10
    y = 1.0 + ((seed_val // 10) % 10) / 10
    z = 1.0 + ((seed_val // 100) % 10) / 10

    # Run Lorenz for a few steps based on timestamp fraction
    steps = int((timestamp % 1.0) * 100) + 10
    for _ in range(steps):
        x, y, z = lorenz_step(x, y, z)

    # Normalize x to noise factor (-1 to +1 range, then scale by volatility)
    # Lorenz x typically ranges from -20 to +20
    noise_factor = (x / 40.0) * volatility

    # Apply noise to true price
    market_price = true_price * (1.0 + noise_factor)

    return market_price


@njit(cache=True, fastmath=True)
def calculate_whale_probability(timestamp: float, halving_cycle: float) -> Tuple[float, float]:
    """
    ID 804: UTXOWhaleDetector

    Detect whale activity probability from:
    1. Halving cycle position (whales accumulate pre-halving)
    2. Block patterns (large blocks = whale movements)

    Returns:
        (whale_probability, price_impact)
    """
    # Whale activity follows patterns:
    # - Increases before halvings
    # - Increases during high volatility periods
    # - Follows accumulation/distribution cycles

    # Simulated block size ratio (1.0 = normal, >1.5 = whale activity)
    # Use timestamp for deterministic pattern
    block_pattern = 1.0 + 0.5 * math.sin(timestamp / 3600 * 0.1)  # Hourly cycles

    # Halving effect on whale activity
    if halving_cycle > 0.8:
        halving_effect = 1.5  # Pre-halving accumulation
    elif halving_cycle < 0.2:
        halving_effect = 1.3  # Post-halving distribution
    else:
        halving_effect = 1.0

    # Combined whale probability
    whale_prob = min(0.9, (block_pattern * halving_effect - 1.0) * 0.6)
    whale_prob = max(0.0, whale_prob)

    # Price impact (Kyle Lambda style)
    # Higher whale probability = larger price impact per trade
    if whale_prob > 0.5:
        price_impact = 0.001 * whale_prob  # Up to 0.1% impact
    else:
        price_impact = 0.0001  # Minimal impact

    return whale_prob, price_impact


@njit(cache=True, fastmath=True)
def generate_tick_data(timestamp: float) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Master function: Generate complete tick data from pure blockchain math.

    Returns:
        (true_price, market_price, volatility, volume, ofi, whale_prob, impact, halving_progress)
    """
    # Power Law fair value
    true_price = calculate_power_law_price(timestamp)

    # Halving cycle
    halving_num, halving_cycle = calculate_halving_cycle(timestamp)

    # Volatility from block patterns
    volatility = calculate_block_time_volatility(timestamp)

    # Mempool flow simulation
    daily_volume, ofi = calculate_mempool_flow(timestamp, halving_cycle)

    # Market price with chaos dynamics
    market_price = calculate_chaos_price(timestamp, true_price, volatility)

    # Whale detection
    whale_prob, price_impact = calculate_whale_probability(timestamp, halving_cycle)

    # Per-tick volume (daily / 86400 seconds / 1000 ticks per second)
    tick_volume = daily_volume / 86400.0 / 1000.0

    return (
        true_price,
        market_price,
        volatility,
        tick_volume,
        ofi,
        whale_prob,
        price_impact,
        halving_cycle
    )


# =============================================================================
# PYTHON CLASS WRAPPERS (For integration with existing formula system)
# =============================================================================

@dataclass
class PureBlockchainTick:
    """Single tick of pure blockchain data."""
    timestamp: float
    true_price: float
    market_price: float
    volatility: float
    volume: float
    ofi: float
    whale_probability: float
    price_impact: float
    halving_cycle: float

    @property
    def edge_pct(self) -> float:
        """Edge = (true - market) / market"""
        if self.market_price > 0:
            return (self.true_price - self.market_price) / self.market_price
        return 0.0

    @property
    def signal(self) -> int:
        """Trading signal: 1=long, -1=short, 0=neutral"""
        edge = self.edge_pct
        if edge > 0.002:  # True price 0.2% above market = BUY
            return 1
        elif edge < -0.002:  # True price 0.2% below = SELL
            return -1
        return 0


class PureBlockchainSimulator:
    """
    Pure blockchain price and volume simulator.

    ZERO APIs. ZERO external data. ALL derived from blockchain math.

    Usage:
        sim = PureBlockchainSimulator()
        tick = sim.get_tick()  # Uses current time
        tick = sim.get_tick(timestamp=1700000000.0)  # Specific time
    """

    def __init__(self):
        self.tick_history = deque(maxlen=10000)
        self.last_timestamp = 0.0

    def get_tick(self, timestamp: float = None) -> PureBlockchainTick:
        """
        Generate tick data for given timestamp.

        If timestamp not provided, uses current time.
        """
        if timestamp is None:
            timestamp = time.time()

        # Generate all tick data from pure blockchain math
        data = generate_tick_data(timestamp)

        tick = PureBlockchainTick(
            timestamp=timestamp,
            true_price=data[0],
            market_price=data[1],
            volatility=data[2],
            volume=data[3],
            ofi=data[4],
            whale_probability=data[5],
            price_impact=data[6],
            halving_cycle=data[7]
        )

        self.tick_history.append(tick)
        self.last_timestamp = timestamp

        return tick

    def get_tick_batch(self, start_ts: float, count: int, interval_ns: int = 1000) -> list:
        """
        Generate batch of ticks for backtesting.

        Args:
            start_ts: Starting timestamp
            count: Number of ticks
            interval_ns: Nanoseconds between ticks

        Returns:
            List of PureBlockchainTick objects
        """
        interval_s = interval_ns / 1e9
        ticks = []

        for i in range(count):
            ts = start_ts + i * interval_s
            ticks.append(self.get_tick(ts))

        return ticks

    def get_current_state(self) -> Dict[str, Any]:
        """Get current blockchain state."""
        tick = self.get_tick()
        halving_num, _ = calculate_halving_cycle(tick.timestamp)

        return {
            'timestamp': tick.timestamp,
            'block_height_est': int((tick.timestamp - GENESIS_TS) / SECONDS_PER_BLOCK),
            'halving_number': halving_num,
            'halving_progress': tick.halving_cycle,
            'true_price': tick.true_price,
            'market_price': tick.market_price,
            'edge_pct': tick.edge_pct * 100,
            'volatility': tick.volatility,
            'ofi': tick.ofi,
            'whale_probability': tick.whale_probability,
            'signal': tick.signal,
        }


# =============================================================================
# FORMULA INTEGRATIONS (For BlockchainSignalAggregator)
# =============================================================================

class BlockTimeVolatility:
    """ID 801: Block time volatility signal."""

    FORMULA_ID = 801

    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.vol_history = deque(maxlen=lookback)
        self.signal = 0
        self.confidence = 0.0
        self.is_ready = False

    def update(self, price: float, volume: float, timestamp: float):
        vol = calculate_block_time_volatility(timestamp)
        self.vol_history.append(vol)

        if len(self.vol_history) >= 20:
            self.is_ready = True

            # High volatility = mean reversion expected
            current_vol = vol
            avg_vol = np.mean(list(self.vol_history))

            if current_vol > avg_vol * 1.5:
                self.signal = -1  # High vol = reversal likely
                self.confidence = min(0.7, (current_vol / avg_vol - 1) * 0.5)
            elif current_vol < avg_vol * 0.7:
                self.signal = 1  # Low vol = can hold positions
                self.confidence = min(0.6, (1 - current_vol / avg_vol) * 0.5)
            else:
                self.signal = 0
                self.confidence = 0.3

    def get_signal(self) -> int:
        return self.signal

    def get_confidence(self) -> float:
        return self.confidence


class MempoolFlowSignal:
    """ID 802: Mempool flow signal."""

    FORMULA_ID = 802

    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.ofi_history = deque(maxlen=lookback)
        self.signal = 0
        self.confidence = 0.0
        self.is_ready = False

    def update(self, price: float, volume: float, timestamp: float):
        halving_num, halving_cycle = calculate_halving_cycle(timestamp)
        daily_vol, ofi = calculate_mempool_flow(timestamp, halving_cycle)

        self.ofi_history.append(ofi)

        if len(self.ofi_history) >= 10:
            self.is_ready = True

            avg_ofi = np.mean(list(self.ofi_history))

            if avg_ofi > 0.6:
                self.signal = 1  # Bullish flow
                self.confidence = min(0.75, avg_ofi)
            elif avg_ofi < 0.4:
                self.signal = -1  # Bearish flow
                self.confidence = min(0.75, 1 - avg_ofi)
            else:
                self.signal = 0
                self.confidence = 0.3

    def get_signal(self) -> int:
        return self.signal

    def get_confidence(self) -> float:
        return self.confidence


class DeterministicPriceSignal:
    """ID 803: Chaos-based price signal."""

    FORMULA_ID = 803

    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.edge_history = deque(maxlen=lookback)
        self.signal = 0
        self.confidence = 0.0
        self.is_ready = False

    def update(self, price: float, volume: float, timestamp: float):
        true_price = calculate_power_law_price(timestamp)
        volatility = calculate_block_time_volatility(timestamp)
        market_price = calculate_chaos_price(timestamp, true_price, volatility)

        edge = (true_price - market_price) / market_price if market_price > 0 else 0
        self.edge_history.append(edge)

        if len(self.edge_history) >= 10:
            self.is_ready = True

            # Trade based on edge from true price
            if edge > 0.002:  # Market undervalued
                self.signal = 1
                self.confidence = min(0.8, abs(edge) * 100)
            elif edge < -0.002:  # Market overvalued
                self.signal = -1
                self.confidence = min(0.8, abs(edge) * 100)
            else:
                self.signal = 0
                self.confidence = 0.3

    def get_signal(self) -> int:
        return self.signal

    def get_confidence(self) -> float:
        return self.confidence


class WhaleDetectionSignal:
    """ID 804: Whale detection signal."""

    FORMULA_ID = 804

    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.whale_history = deque(maxlen=lookback)
        self.signal = 0
        self.confidence = 0.0
        self.is_ready = False

    def update(self, price: float, volume: float, timestamp: float):
        halving_num, halving_cycle = calculate_halving_cycle(timestamp)
        whale_prob, impact = calculate_whale_probability(timestamp, halving_cycle)

        self.whale_history.append(whale_prob)

        if len(self.whale_history) >= 10:
            self.is_ready = True

            avg_whale = np.mean(list(self.whale_history))

            # High whale activity in pre-halving = bullish
            # High whale activity otherwise = cautious
            if whale_prob > 0.5:
                if halving_cycle > 0.7:  # Pre-halving
                    self.signal = 1  # Whales accumulating
                    self.confidence = min(0.7, whale_prob)
                else:
                    self.signal = 0  # Caution
                    self.confidence = 0.4
            else:
                self.signal = 0
                self.confidence = 0.3

    def get_signal(self) -> int:
        return self.signal

    def get_confidence(self) -> float:
        return self.confidence


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PURE BLOCKCHAIN SIMULATION - RENAISSANCE LEVEL")
    print("=" * 70)
    print()

    sim = PureBlockchainSimulator()
    state = sim.get_current_state()

    print("CURRENT BLOCKCHAIN STATE:")
    print(f"  Timestamp:        {state['timestamp']:.0f}")
    print(f"  Block Height Est: {state['block_height_est']:,}")
    print(f"  Halving #:        {state['halving_number']}")
    print(f"  Halving Progress: {state['halving_progress']:.2%}")
    print()
    print("DERIVED PRICES:")
    print(f"  TRUE Price:       ${state['true_price']:,.2f}")
    print(f"  Market Price:     ${state['market_price']:,.2f}")
    print(f"  Edge:             {state['edge_pct']:+.4f}%")
    print()
    print("MARKET SIGNALS:")
    print(f"  Volatility:       {state['volatility']:.4f}")
    print(f"  OFI:              {state['ofi']:.2f}")
    print(f"  Whale Probability:{state['whale_probability']:.2f}")
    print(f"  Signal:           {state['signal']} ({'LONG' if state['signal'] > 0 else 'SHORT' if state['signal'] < 0 else 'NEUTRAL'})")
    print()
    print("=" * 70)
    print("TICK GENERATION TEST:")
    print("=" * 70)

    import time as t
    start = t.perf_counter()
    count = 100000

    for i in range(count):
        tick = sim.get_tick()

    elapsed = t.perf_counter() - start
    tps = count / elapsed

    print(f"  Generated {count:,} ticks in {elapsed:.3f}s")
    print(f"  TPS: {tps:,.0f}")
    print()
    print("FORMULA SUMMARY:")
    print("  801: BlockTimeVolatility - Derives volatility from block intervals")
    print("  802: MempoolFlowSimulator - Simulates TX flow from halving cycle")
    print("  803: DeterministicChaosPrice - Lorenz attractor price dynamics")
    print("  804: UTXOWhaleDetector - Whale detection from block patterns")
    print()
    print("ALL DATA DERIVED FROM BLOCKCHAIN MATH - ZERO APIs")
    print("=" * 70)
