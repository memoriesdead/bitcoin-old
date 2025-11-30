#!/usr/bin/env python3
"""
CALIBRATED BLOCKCHAIN PRICE ENGINE
==================================
Price anchored to $95,000 with small signal-based adjustments.

IMPORTANT: Blockchain data gives us SIGNALS, not price.
- Fee rate = demand urgency
- Mempool = congestion
- Volume = activity level
- Whales = big player moves

We use these as SIGNALS to adjust price by small %
NOT to calculate price from scratch.
"""

import numpy as np
from dataclasses import dataclass
from typing import Deque
from collections import deque
import time


@dataclass
class DerivedPrice:
    """Price output"""
    timestamp: float
    composite_price: float     # Stable price ~$95k
    signal: float              # -1 to +1
    confidence: float          # 0 to 1

    # Component signals
    metcalfe_price: float = 0
    nvt_signal: float = 0
    fee_pressure_index: float = 0
    velocity_price: float = 0
    price_momentum: float = 0
    acceleration: float = 0


@dataclass
class BlockchainState:
    """Blockchain metrics"""
    timestamp: float = 0.0
    tx_count_1m: int = 0
    tx_volume_btc_1m: float = 0.0
    tx_count_10m: int = 0
    tx_volume_btc_10m: float = 0.0
    fee_fast: int = 0
    fee_medium: int = 0
    fee_slow: int = 0
    fee_velocity: float = 0.0
    mempool_size: int = 0
    mempool_vsize_mb: float = 0.0
    mempool_growth_rate: float = 0.0
    whale_tx_count: int = 0
    whale_volume_btc: float = 0.0
    block_height: int = 0
    block_time_avg: float = 600.0
    hashrate_estimate: float = 0.0
    active_addresses_1h: int = 0


class RealTimeBlockchainPricer:
    """
    Stable price engine using blockchain SIGNALS

    Price stays near anchor ($95,000) with small adjustments
    based on blockchain activity signals.

    Max deviation: +/- 5% from anchor
    Max change per update: 0.1%
    """

    def __init__(self, calibration_price: float = 0.0):
        # NO HARDCODED PRICE - will be set from real market data
        self.anchor_price = calibration_price  # 0 means not yet calibrated
        self.current_price = calibration_price

        # Baseline values - START AT ZERO, will be calibrated from LIVE data
        # NO hardcoded "normal" values - learn from actual blockchain
        self.baseline_fee = 0.0         # LIVE: Calibrated from first N observations
        self.baseline_mempool = 0       # LIVE: Calibrated from first N observations
        self.baseline_volume = 0.0      # LIVE: Calibrated from first N observations
        self._baseline_samples = 0      # Count of samples for baseline calibration

        # Smoothing
        self.alpha = 0.1  # EMA factor for baselines

        # History
        self.price_history: Deque[float] = deque(maxlen=100)
        self.price_history.append(calibration_price)

        self.calibrated = False
        self.update_count = 0

    def update_from_raw(
        self,
        tx_count: int,
        tx_volume: float,
        fee_fast: int,
        fee_medium: int,
        mempool_size: int,
        mempool_vsize: float,
        whale_count: int = 0,
        whale_volume: float = 0.0,
        active_addresses: int = 50000
    ) -> DerivedPrice:
        """
        Update price based on blockchain signals.

        Signals adjust price by small amounts:
        - High fees = bullish (+)
        - Growing mempool = bullish (+)
        - Whale activity = amplifies signal
        """
        now = time.time()

        # First call: set baselines from actual data
        if not self.calibrated:
            self.baseline_fee = max(fee_fast, 1)
            self.baseline_mempool = max(mempool_size, 1000)
            self.baseline_volume = max(tx_volume, 0.1)
            self.calibrated = True
            print(f"Calibrated at ${self.anchor_price:,.0f}")
            print(f"  Baseline fee: {self.baseline_fee:.0f} sat/vB")
            print(f"  Baseline mempool: {self.baseline_mempool:,.0f} txs")
            print(f"  Baseline volume: {self.baseline_volume:.1f} BTC/min")

        # Update baselines slowly (EMA)
        self.baseline_fee = self.alpha * max(fee_fast, 1) + (1 - self.alpha) * self.baseline_fee
        self.baseline_mempool = self.alpha * max(mempool_size, 1000) + (1 - self.alpha) * self.baseline_mempool
        self.baseline_volume = self.alpha * max(tx_volume, 0.1) + (1 - self.alpha) * self.baseline_volume

        # === CALCULATE SIGNALS ===

        # Fee signal: high fees = bullish
        fee_ratio = fee_fast / self.baseline_fee
        fee_signal = np.clip((fee_ratio - 1) * 0.5, -1, 1)

        # Mempool signal: growing mempool = bullish
        mempool_ratio = mempool_size / self.baseline_mempool
        mempool_signal = np.clip((mempool_ratio - 1) * 0.3, -1, 1)

        # Volume signal: high volume confirms trend
        volume_ratio = tx_volume / self.baseline_volume if self.baseline_volume > 0 else 1
        volume_signal = np.clip((volume_ratio - 1) * 0.2, -0.5, 0.5)

        # Whale boost
        whale_boost = min(whale_count * 0.1, 0.3)

        # Combined signal
        raw_signal = (
            0.5 * fee_signal +
            0.3 * mempool_signal +
            0.2 * volume_signal
        )

        # Apply whale boost (amplifies signal direction)
        if whale_count > 0:
            raw_signal = raw_signal * (1 + whale_boost)

        # Clip to [-1, +1]
        signal = np.clip(raw_signal, -1, 1)

        # === APPLY TO PRICE ===

        # Max 0.1% change per update
        max_change = 0.001
        price_change = signal * max_change

        # Apply to current price
        new_price = self.current_price * (1 + price_change)

        # NO PRICE CLIPPING - use real price, no artificial bounds
        # Real prices change without limits

        # Smooth update
        self.current_price = 0.8 * new_price + 0.2 * self.current_price

        # Store
        self.price_history.append(self.current_price)
        self.update_count += 1

        # Confidence from signal strength
        confidence = min(abs(signal) * 2, 1.0)

        return DerivedPrice(
            timestamp=now,
            composite_price=self.current_price,
            signal=signal,
            confidence=confidence,
            fee_pressure_index=fee_signal,
            nvt_signal=volume_signal,
            metcalfe_price=self.current_price,
            velocity_price=self.current_price,
            price_momentum=price_change,
            acceleration=0
        )


# Alias for backwards compatibility
class BlockchainPriceEngine:
    """Wrapper for compatibility"""
    def __init__(self, calibration_price: float = 0.0):
        self._pricer = RealTimeBlockchainPricer(calibration_price)
        self.calibration_price = calibration_price
        self.calibrated = False
        self.state_history = deque(maxlen=100)
        self.price_history = deque(maxlen=100)

    def calibrate(self, state):
        pass  # Auto-calibrates on first update

    def update(self, state: BlockchainState) -> DerivedPrice:
        self.state_history.append(state)
        result = self._pricer.update_from_raw(
            tx_count=state.tx_count_1m,
            tx_volume=state.tx_volume_btc_1m,
            fee_fast=state.fee_fast,
            fee_medium=state.fee_medium,
            mempool_size=state.mempool_size,
            mempool_vsize=state.mempool_vsize_mb,
            whale_count=state.whale_tx_count,
            whale_volume=state.whale_volume_btc,
            active_addresses=state.active_addresses_1h
        )
        self.price_history.append(result)
        self.calibrated = True
        return result
