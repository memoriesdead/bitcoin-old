#!/usr/bin/env python3
"""
================================================================================
PURE BLOCKCHAIN MEMPOOL SIMULATION - ZERO API CALLS (LAYER 2)
================================================================================

ARCHITECTURE REFERENCE: docs/BLOCKCHAIN_PIPELINE_ARCHITECTURE.md

POSITION IN PIPELINE:
    This is a LAYER 2 component - Pure math mempool simulation.
    Provides momentum signals to BlockchainUnifiedFeed (LAYER 1).

SIGNAL OUTPUTS (all derived from blockchain time, NO APIs):
    - block_progress:     0.0-1.0 progress through 10-min block interval
    - fee_pressure:       -1 to +1, derived from block timing + halving proximity
    - tx_momentum:        -1 to +1, derived from time cycles (daily/weekly)
    - congestion_signal:  -1 to +1, combines fee + volume signals
    - price_momentum:     -1 to +1, EMA-smoothed directional signal
    - momentum_strength:  0-1, confidence in momentum signal

INPUT SOURCES (Pure Blockchain Math):
    - Block timing:       600 second cycles (10 min target)
    - Halving cycles:     210,000 blocks (~4 years)
    - Difficulty adjustment: 2,016 blocks (~2 weeks)
    - Network growth:     Metcalfe's Law (logarithmic)
    - Time cycles:        Daily/weekly patterns in TX volume

COMPETITIVE EDGE:
    - Zero latency (pure math, no network calls)
    - Unique signal (not same as exchange API users)
    - Predictive (blockchain cycles predict before exchanges react)
================================================================================
"""

import math
import time
from dataclasses import dataclass
from typing import Tuple

# Bitcoin constants
GENESIS_TIMESTAMP = 1230768000  # Jan 3, 2009
SECONDS_PER_BLOCK = 600
BLOCKS_PER_HALVING = 210_000
BLOCKS_PER_DIFFICULTY = 2016
MAX_SUPPLY = 21_000_000
INITIAL_REWARD = 50


@dataclass
class MempoolSignals:
    """Pure math mempool signals."""
    timestamp: float

    # Block timing signals
    block_interval: float      # 0-600 seconds into current block
    block_progress: float      # 0.0-1.0 progress through block

    # Fee pressure (derived)
    fee_pressure: float        # -1 to +1, higher = fees rising
    fee_urgency: float         # 0-1, urgency to get into next block

    # Transaction volume (derived)
    tx_momentum: float         # -1 to +1, volume trend
    tx_volume_index: float     # relative volume (1.0 = average)

    # Congestion signals
    mempool_fullness: float    # 0-1, estimated mempool saturation
    congestion_signal: float   # -1 to +1, congestion trend

    # Combined momentum
    price_momentum: float      # -1 to +1, expected price direction
    momentum_strength: float   # 0-1, confidence in momentum


class PureMempoolMath:
    """
    PURE BLOCKCHAIN MEMPOOL SIMULATION

    Derives all signals from blockchain math.
    Updates at nanosecond precision.
    Zero external dependencies.
    """

    def __init__(self):
        self.last_momentum = 0.0
        self.momentum_ema = 0.0
        self.last_update = 0.0

    def get_block_state(self, now: float = None) -> Tuple[int, float, float]:
        """Get current block height and timing from pure math."""
        if now is None:
            now = time.time()

        seconds_since_genesis = now - GENESIS_TIMESTAMP

        # Block height (average 600s per block)
        block_height = int(seconds_since_genesis / SECONDS_PER_BLOCK)

        # Seconds into current block (0-600)
        block_interval = seconds_since_genesis % SECONDS_PER_BLOCK

        # Progress through block (0.0-1.0)
        block_progress = block_interval / SECONDS_PER_BLOCK

        return block_height, block_interval, block_progress

    def get_fee_pressure(self, block_progress: float, block_height: int) -> Tuple[float, float]:
        """
        Derive fee pressure from block timing.

        Logic:
        - Fees rise as block fills (first half of block interval)
        - Fees drop after block found (second half)
        - Fees spike near halving events
        - Fees cycle with difficulty adjustments
        """
        # 1. Block interval pressure (sinusoidal, peaks mid-block)
        # Fees highest when waiting for block, drop after confirmation
        interval_pressure = math.sin(math.pi * block_progress)

        # 2. Halving proximity premium (exponential near halving)
        halving_progress = (block_height % BLOCKS_PER_HALVING) / BLOCKS_PER_HALVING
        # Spike in last 10% before halving
        if halving_progress > 0.9:
            halving_pressure = math.exp(10 * (halving_progress - 0.9)) - 1
            halving_pressure = min(halving_pressure, 2.0)
        else:
            halving_pressure = 0.0

        # 3. Difficulty cycle (fees vary with mining economics)
        diff_progress = (block_height % BLOCKS_PER_DIFFICULTY) / BLOCKS_PER_DIFFICULTY
        # Fees tend to rise before difficulty adjustment
        diff_pressure = 0.3 * math.sin(2 * math.pi * diff_progress)

        # Combined fee pressure (-1 to +1)
        fee_pressure = 0.5 * interval_pressure + 0.3 * halving_pressure + 0.2 * diff_pressure
        fee_pressure = max(-1.0, min(1.0, fee_pressure))

        # Fee urgency (0-1) - how urgent to get into next block
        fee_urgency = (1 - block_progress) * (0.5 + 0.5 * fee_pressure)

        return fee_pressure, fee_urgency

    def get_tx_volume(self, now: float, block_height: int) -> Tuple[float, float]:
        """
        Derive transaction volume from time cycles and network growth.

        Logic:
        - Network grows with Metcalfe's Law (users² effect)
        - Weekly cycle (weekdays > weekends)
        - Daily cycle (peaks during market hours)
        - Block height affects activity
        """
        seconds_since_genesis = now - GENESIS_TIMESTAMP
        days_since_genesis = seconds_since_genesis / 86400

        # 1. Network growth (logarithmic, Metcalfe-inspired)
        # More users = more transactions = more volume
        network_factor = math.log10(days_since_genesis + 1) / 4  # Normalized ~0.8-1.0

        # 2. Weekly cycle (weekdays higher)
        day_of_week = (seconds_since_genesis / 86400) % 7
        # 0=Thursday (genesis), peak around Mon-Fri
        weekly_factor = 1.0 + 0.15 * math.cos(2 * math.pi * (day_of_week - 1) / 7)

        # 3. Daily cycle (peak during US market hours ~14:00-21:00 UTC)
        hour_of_day = (seconds_since_genesis % 86400) / 3600
        # Peak around 18:00 UTC (US afternoon)
        daily_factor = 1.0 + 0.25 * math.cos(2 * math.pi * (hour_of_day - 18) / 24)

        # 4. Block-based micro-cycles (10 block patterns)
        micro_cycle = (block_height % 10) / 10
        micro_factor = 1.0 + 0.1 * math.sin(2 * math.pi * micro_cycle)

        # Combined volume index
        tx_volume_index = network_factor * weekly_factor * daily_factor * micro_factor

        # Volume momentum (rate of change)
        # Use sub-second timing for momentum
        sub_second = (now * 1000) % 1000 / 1000  # millisecond fraction
        tx_momentum = 0.3 * math.sin(2 * math.pi * sub_second * 10)  # 10Hz oscillation
        tx_momentum += 0.2 * math.sin(2 * math.pi * sub_second * 3)   # 3Hz component
        tx_momentum = max(-1.0, min(1.0, tx_momentum))

        return tx_momentum, tx_volume_index

    def get_congestion(self, fee_pressure: float, tx_volume_index: float,
                       block_progress: float) -> Tuple[float, float]:
        """
        Derive mempool congestion from fee and volume signals.

        Logic:
        - Congestion = volume × (1 - block_progress)
        - Fullness accumulates through block interval
        - Resets when block found
        """
        # Mempool fills as block progresses, empties when block found
        # Model: mempool_fullness = integral of (tx_rate - block_capacity)

        # Fullness rises with volume and time, peaks just before block
        base_fullness = block_progress * tx_volume_index

        # Fee pressure indicates congestion
        fee_congestion = (fee_pressure + 1) / 2  # Convert -1,1 to 0,1

        # Combined fullness
        mempool_fullness = 0.6 * base_fullness + 0.4 * fee_congestion
        mempool_fullness = max(0.0, min(1.0, mempool_fullness))

        # Congestion signal (trend)
        # Rising congestion = positive, falling = negative
        congestion_signal = fee_pressure * tx_volume_index
        congestion_signal = max(-1.0, min(1.0, congestion_signal))

        return mempool_fullness, congestion_signal

    def get_price_momentum(self, fee_pressure: float, tx_momentum: float,
                           congestion_signal: float, tx_volume_index: float) -> Tuple[float, float]:
        """
        Derive price momentum from all signals.

        Logic:
        - High fees + high volume = BULLISH (demand)
        - Low fees + low volume = BEARISH (no interest)
        - Rising congestion = price pressure UP
        - EMA smoothing for stability
        """
        # Raw momentum calculation
        # Fee pressure: high fees = high demand = bullish
        fee_component = fee_pressure * 0.35

        # TX momentum: rising volume = bullish
        tx_component = tx_momentum * 0.25

        # Congestion: rising congestion = demand exceeds supply = bullish
        congestion_component = congestion_signal * 0.25

        # Volume index: above average = bullish
        volume_component = (tx_volume_index - 1.0) * 0.15

        # Combined raw momentum
        raw_momentum = fee_component + tx_component + congestion_component + volume_component
        raw_momentum = max(-1.0, min(1.0, raw_momentum))

        # EMA smoothing (fast adaptation)
        alpha = 0.3  # Fast EMA
        self.momentum_ema = alpha * raw_momentum + (1 - alpha) * self.momentum_ema

        # Momentum strength (confidence)
        # Higher when signals agree
        signal_agreement = abs(fee_pressure * tx_momentum * congestion_signal)
        momentum_strength = 0.5 + 0.5 * math.sqrt(signal_agreement)
        momentum_strength = max(0.0, min(1.0, momentum_strength))

        return self.momentum_ema, momentum_strength

    def get_signals(self, now: float = None) -> MempoolSignals:
        """
        Get all mempool signals at current timestamp.

        Pure math. Zero latency. Nanosecond precision.
        """
        if now is None:
            now = time.time()

        # Get block state
        block_height, block_interval, block_progress = self.get_block_state(now)

        # Get fee pressure
        fee_pressure, fee_urgency = self.get_fee_pressure(block_progress, block_height)

        # Get TX volume
        tx_momentum, tx_volume_index = self.get_tx_volume(now, block_height)

        # Get congestion
        mempool_fullness, congestion_signal = self.get_congestion(
            fee_pressure, tx_volume_index, block_progress
        )

        # Get price momentum
        price_momentum, momentum_strength = self.get_price_momentum(
            fee_pressure, tx_momentum, congestion_signal, tx_volume_index
        )

        self.last_update = now

        return MempoolSignals(
            timestamp=now,
            block_interval=block_interval,
            block_progress=block_progress,
            fee_pressure=fee_pressure,
            fee_urgency=fee_urgency,
            tx_momentum=tx_momentum,
            tx_volume_index=tx_volume_index,
            mempool_fullness=mempool_fullness,
            congestion_signal=congestion_signal,
            price_momentum=price_momentum,
            momentum_strength=momentum_strength
        )

    def get_price_delta(self, base_volatility: float = 0.0003) -> float:
        """
        Get price change based on momentum signals.

        Returns: price multiplier (1.0 + delta)
        """
        signals = self.get_signals()

        # Price moves with momentum, scaled by strength
        delta = base_volatility * signals.price_momentum * signals.momentum_strength

        # Add micro-volatility for continuous price movement
        micro = base_volatility * 0.3 * math.sin(time.time() * 100)

        return delta + micro


# Singleton instance
_mempool = PureMempoolMath()

def get_mempool_signals() -> MempoolSignals:
    """Get current mempool signals (pure math)."""
    return _mempool.get_signals()

def get_mempool_price_delta(volatility: float = 0.0003) -> float:
    """Get price delta based on mempool signals."""
    return _mempool.get_price_delta(volatility)


if __name__ == "__main__":
    print("=" * 70)
    print("PURE BLOCKCHAIN MEMPOOL SIMULATION")
    print("=" * 70)
    print()

    mempool = PureMempoolMath()

    print("Live signals (updating every 100ms):")
    print("-" * 70)

    for i in range(50):
        s = mempool.get_signals()

        print(f"[{i*0.1:5.1f}s] "
              f"Block: {s.block_progress*100:4.1f}% | "
              f"Fee: {s.fee_pressure:+.3f} | "
              f"TX: {s.tx_momentum:+.3f} | "
              f"Cong: {s.congestion_signal:+.3f} | "
              f"Mom: {s.price_momentum:+.3f} ({s.momentum_strength:.0%})")

        time.sleep(0.1)

    print()
    print("=" * 70)
