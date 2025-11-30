#!/usr/bin/env python3
"""
VOLUME AGGREGATOR - On-Chain Transaction Volume Tracking
=========================================================
Aggregates real transaction volume from blockchain data.

MATHEMATICAL FOUNDATION:
------------------------
Transaction Volume Metrics:
1. Raw BTC Volume: Sum of all transaction outputs
2. Economic Volume: Filters out change outputs and dust
3. Rolling Windows: 1h, 4h, 24h, 7d moving averages
4. Volume Velocity: dV/dt (rate of change)

On-Chain Volume Estimation:
- Each transaction has inputs and outputs
- Economic volume ≈ min(sum(inputs), sum(outputs) - change)
- Whale detection: Transactions > 100 BTC
- Retail detection: Transactions < 0.1 BTC

Volume-Price Relationships:
- High volume + rising price = Confirmation (bullish)
- High volume + falling price = Distribution (bearish)
- Low volume + rising price = Weak rally (caution)
- Low volume + falling price = Accumulation (bullish potential)

Author: Renaissance Trading System
"""

import time
import math
from dataclasses import dataclass, field
from typing import Optional, Deque, List, Tuple
from collections import deque


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Volume thresholds (in BTC)
WHALE_THRESHOLD = 100.0      # > 100 BTC = whale
LARGE_TX_THRESHOLD = 10.0    # > 10 BTC = large
RETAIL_THRESHOLD = 0.1       # < 0.1 BTC = retail
DUST_THRESHOLD = 0.00001     # < 0.00001 BTC = dust (filter out)

# Rolling windows (in seconds)
WINDOW_1H = 3600
WINDOW_4H = 14400
WINDOW_24H = 86400
WINDOW_7D = 604800

# Average block time
AVG_BLOCK_TIME = 600  # 10 minutes


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class TransactionRecord:
    """Single transaction record from blockchain."""
    timestamp: float
    txid: str = ""
    value_btc: float = 0.0
    fee_sats: int = 0
    input_count: int = 0
    output_count: int = 0
    is_whale: bool = False
    is_coinbase: bool = False


@dataclass
class VolumeMetrics:
    """Aggregated volume metrics."""
    timestamp: float = 0.0

    # Raw volume (all transactions)
    volume_1h_btc: float = 0.0
    volume_4h_btc: float = 0.0
    volume_24h_btc: float = 0.0
    volume_7d_btc: float = 0.0

    # Transaction counts
    tx_count_1h: int = 0
    tx_count_4h: int = 0
    tx_count_24h: int = 0

    # Whale metrics
    whale_volume_1h_btc: float = 0.0
    whale_count_1h: int = 0
    whale_percentage: float = 0.0  # % of volume from whales

    # Retail metrics
    retail_volume_1h_btc: float = 0.0
    retail_count_1h: int = 0

    # Fee metrics
    total_fees_1h_sats: int = 0
    avg_fee_per_tx: float = 0.0

    # Velocity metrics
    volume_velocity: float = 0.0   # BTC per second
    volume_acceleration: float = 0.0  # d(velocity)/dt

    # Average transaction size
    avg_tx_size_btc: float = 0.0
    median_tx_size_btc: float = 0.0

    # Volume trend signals
    volume_trend: str = "NEUTRAL"  # INCREASING, DECREASING, NEUTRAL
    volume_zscore: float = 0.0     # Standard deviations from mean


@dataclass
class VolumeSnapshot:
    """Point-in-time volume snapshot for history."""
    timestamp: float
    total_volume: float
    tx_count: int
    whale_volume: float
    whale_count: int


# ==============================================================================
# VOLUME AGGREGATOR CLASS
# ==============================================================================

class VolumeAggregator:
    """
    Aggregates and analyzes on-chain transaction volume.

    Tracks rolling windows of volume data to compute
    24h volume and volume-based trading signals.

    Usage:
        aggregator = VolumeAggregator()

        # Add transactions as they come in
        aggregator.add_transaction(value_btc=2.5, fee_sats=5000)
        aggregator.add_transaction(value_btc=150.0, fee_sats=15000)  # whale

        # Get metrics
        metrics = aggregator.get_metrics()
        print(f"24h Volume: {metrics.volume_24h_btc:,.2f} BTC")
    """

    def __init__(self, max_history: int = 100_000):
        """
        Initialize volume aggregator.

        Args:
            max_history: Maximum transactions to keep in memory
        """
        self.max_history = max_history

        # Transaction history (ring buffer)
        self.transactions: Deque[TransactionRecord] = deque(maxlen=max_history)

        # Volume snapshots for trend analysis
        self.snapshots: Deque[VolumeSnapshot] = deque(maxlen=1000)

        # Current metrics
        self.current_metrics = VolumeMetrics()

        # Statistics for z-score calculation
        self.volume_samples: Deque[float] = deque(maxlen=1000)

    def add_transaction(
        self,
        value_btc: float,
        fee_sats: int = 0,
        txid: str = "",
        input_count: int = 1,
        output_count: int = 2,
        is_coinbase: bool = False,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Add a transaction to the aggregator.

        Args:
            value_btc: Transaction value in BTC
            fee_sats: Transaction fee in satoshis
            txid: Transaction ID (optional)
            input_count: Number of inputs
            output_count: Number of outputs
            is_coinbase: Whether this is a coinbase transaction
            timestamp: Transaction timestamp (default: now)
        """
        if value_btc < DUST_THRESHOLD:
            return  # Filter dust

        ts = timestamp or time.time()

        record = TransactionRecord(
            timestamp=ts,
            txid=txid,
            value_btc=value_btc,
            fee_sats=fee_sats,
            input_count=input_count,
            output_count=output_count,
            is_whale=value_btc >= WHALE_THRESHOLD,
            is_coinbase=is_coinbase,
        )

        self.transactions.append(record)

    def add_block_transactions(
        self,
        transactions: List[dict],
        block_timestamp: Optional[float] = None
    ) -> None:
        """
        Add all transactions from a block.

        Args:
            transactions: List of transaction dicts with 'value', 'fee', etc.
            block_timestamp: Block timestamp
        """
        ts = block_timestamp or time.time()

        for tx in transactions:
            self.add_transaction(
                value_btc=tx.get('value', 0) or tx.get('value_btc', 0),
                fee_sats=int(tx.get('fee', 0) or tx.get('fee_sats', 0)),
                txid=tx.get('txid', ''),
                input_count=tx.get('input_count', 1),
                output_count=tx.get('output_count', 2),
                is_coinbase=tx.get('is_coinbase', False),
                timestamp=ts,
            )

    def _prune_old_transactions(self, cutoff_time: float) -> None:
        """Remove transactions older than cutoff time."""
        while self.transactions and self.transactions[0].timestamp < cutoff_time:
            self.transactions.popleft()

    def _calculate_window_metrics(
        self,
        window_seconds: int
    ) -> Tuple[float, int, float, int, float, int, int]:
        """
        Calculate metrics for a specific time window.

        Returns:
            (total_volume, tx_count, whale_volume, whale_count,
             retail_volume, retail_count, total_fees)
        """
        now = time.time()
        cutoff = now - window_seconds

        total_volume = 0.0
        tx_count = 0
        whale_volume = 0.0
        whale_count = 0
        retail_volume = 0.0
        retail_count = 0
        total_fees = 0

        for tx in reversed(self.transactions):
            if tx.timestamp < cutoff:
                break

            total_volume += tx.value_btc
            tx_count += 1
            total_fees += tx.fee_sats

            if tx.is_whale:
                whale_volume += tx.value_btc
                whale_count += 1
            elif tx.value_btc < RETAIL_THRESHOLD:
                retail_volume += tx.value_btc
                retail_count += 1

        return (total_volume, tx_count, whale_volume, whale_count,
                retail_volume, retail_count, total_fees)

    def _calculate_median_tx_size(self, window_seconds: int) -> float:
        """Calculate median transaction size in window."""
        now = time.time()
        cutoff = now - window_seconds

        sizes = []
        for tx in reversed(self.transactions):
            if tx.timestamp < cutoff:
                break
            sizes.append(tx.value_btc)

        if not sizes:
            return 0.0

        sizes.sort()
        n = len(sizes)
        if n % 2 == 0:
            return (sizes[n // 2 - 1] + sizes[n // 2]) / 2
        return sizes[n // 2]

    def _calculate_velocity(self) -> Tuple[float, float]:
        """
        Calculate volume velocity and acceleration.

        Returns:
            (velocity_btc_per_sec, acceleration)
        """
        if len(self.snapshots) < 2:
            return 0.0, 0.0

        # Get recent snapshots
        recent = list(self.snapshots)[-10:]

        if len(recent) < 2:
            return 0.0, 0.0

        # Calculate velocity (dV/dt)
        time_delta = recent[-1].timestamp - recent[-2].timestamp
        if time_delta <= 0:
            return 0.0, 0.0

        volume_delta = recent[-1].total_volume - recent[-2].total_volume
        velocity = volume_delta / time_delta

        # Calculate acceleration (d²V/dt²)
        acceleration = 0.0
        if len(recent) >= 3:
            prev_time_delta = recent[-2].timestamp - recent[-3].timestamp
            if prev_time_delta > 0:
                prev_velocity = (recent[-2].total_volume - recent[-3].total_volume) / prev_time_delta
                acceleration = (velocity - prev_velocity) / time_delta

        return velocity, acceleration

    def _calculate_volume_zscore(self, current_volume: float) -> float:
        """Calculate z-score of current volume vs historical."""
        if len(self.volume_samples) < 20:
            return 0.0

        samples = list(self.volume_samples)
        mean_vol = sum(samples) / len(samples)
        variance = sum((v - mean_vol) ** 2 for v in samples) / len(samples)
        std_dev = math.sqrt(variance) if variance > 0 else 1.0

        return (current_volume - mean_vol) / std_dev

    def _determine_trend(self) -> str:
        """Determine volume trend from recent history."""
        if len(self.snapshots) < 5:
            return "NEUTRAL"

        recent = list(self.snapshots)[-10:]
        if len(recent) < 5:
            return "NEUTRAL"

        # Compare first half to second half
        mid = len(recent) // 2
        first_half = sum(s.total_volume for s in recent[:mid])
        second_half = sum(s.total_volume for s in recent[mid:])

        if second_half > first_half * 1.2:
            return "INCREASING"
        elif second_half < first_half * 0.8:
            return "DECREASING"
        return "NEUTRAL"

    def update(self) -> VolumeMetrics:
        """
        Update volume metrics.

        Should be called periodically to refresh all metrics.

        Returns:
            Updated VolumeMetrics
        """
        now = time.time()

        # Prune old transactions (keep 7 days + buffer)
        self._prune_old_transactions(now - WINDOW_7D - 3600)

        # Calculate 1h metrics
        (vol_1h, tx_1h, whale_vol_1h, whale_cnt_1h,
         retail_vol_1h, retail_cnt_1h, fees_1h) = self._calculate_window_metrics(WINDOW_1H)

        # Calculate 4h metrics
        vol_4h, tx_4h, _, _, _, _, _ = self._calculate_window_metrics(WINDOW_4H)

        # Calculate 24h metrics
        vol_24h, tx_24h, _, _, _, _, _ = self._calculate_window_metrics(WINDOW_24H)

        # Calculate 7d metrics
        vol_7d, _, _, _, _, _, _ = self._calculate_window_metrics(WINDOW_7D)

        # Velocity and acceleration
        velocity, acceleration = self._calculate_velocity()

        # Average and median tx size
        avg_tx_size = vol_1h / tx_1h if tx_1h > 0 else 0.0
        median_tx_size = self._calculate_median_tx_size(WINDOW_1H)

        # Whale percentage
        whale_pct = (whale_vol_1h / vol_1h * 100) if vol_1h > 0 else 0.0

        # Z-score
        zscore = self._calculate_volume_zscore(vol_1h)

        # Trend
        trend = self._determine_trend()

        # Store snapshot for history
        snapshot = VolumeSnapshot(
            timestamp=now,
            total_volume=vol_1h,
            tx_count=tx_1h,
            whale_volume=whale_vol_1h,
            whale_count=whale_cnt_1h,
        )
        self.snapshots.append(snapshot)
        self.volume_samples.append(vol_1h)

        # Build metrics
        self.current_metrics = VolumeMetrics(
            timestamp=now,
            volume_1h_btc=vol_1h,
            volume_4h_btc=vol_4h,
            volume_24h_btc=vol_24h,
            volume_7d_btc=vol_7d,
            tx_count_1h=tx_1h,
            tx_count_4h=tx_4h,
            tx_count_24h=tx_24h,
            whale_volume_1h_btc=whale_vol_1h,
            whale_count_1h=whale_cnt_1h,
            whale_percentage=whale_pct,
            retail_volume_1h_btc=retail_vol_1h,
            retail_count_1h=retail_cnt_1h,
            total_fees_1h_sats=fees_1h,
            avg_fee_per_tx=fees_1h / tx_1h if tx_1h > 0 else 0,
            volume_velocity=velocity,
            volume_acceleration=acceleration,
            avg_tx_size_btc=avg_tx_size,
            median_tx_size_btc=median_tx_size,
            volume_trend=trend,
            volume_zscore=zscore,
        )

        return self.current_metrics

    def get_metrics(self) -> VolumeMetrics:
        """Get current volume metrics."""
        return self.current_metrics

    def get_24h_volume(self) -> float:
        """Get 24-hour volume in BTC."""
        return self.current_metrics.volume_24h_btc

    def get_volume_signal(self) -> Tuple[str, float]:
        """
        Get volume-based trading signal.

        Returns:
            (signal_type, strength)
            signal_type: 'BULLISH', 'BEARISH', 'NEUTRAL'
            strength: 0.0 to 1.0
        """
        m = self.current_metrics

        # High volume + increasing = bullish
        # High volume + decreasing = bearish
        # Low volume = neutral

        if m.volume_zscore > 1.5 and m.volume_trend == "INCREASING":
            return ("BULLISH", min(1.0, m.volume_zscore / 3.0))

        if m.volume_zscore > 1.5 and m.volume_trend == "DECREASING":
            return ("BEARISH", min(1.0, m.volume_zscore / 3.0))

        if m.whale_percentage > 60:
            # Heavy whale activity - follow the whales
            if m.volume_trend == "INCREASING":
                return ("BULLISH", 0.7)
            return ("BEARISH", 0.7)

        return ("NEUTRAL", 0.0)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def format_volume_summary(m: VolumeMetrics) -> str:
    """Format volume metrics for display."""
    return f"""
================================================================================
                        VOLUME AGGREGATOR OUTPUT
================================================================================
  Timestamp:           {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(m.timestamp))}
--------------------------------------------------------------------------------
  ROLLING WINDOWS:
    1h Volume:         {m.volume_1h_btc:>15,.2f} BTC  ({m.tx_count_1h:,} txs)
    4h Volume:         {m.volume_4h_btc:>15,.2f} BTC  ({m.tx_count_4h:,} txs)
    24h Volume:        {m.volume_24h_btc:>15,.2f} BTC  ({m.tx_count_24h:,} txs)
    7d Volume:         {m.volume_7d_btc:>15,.2f} BTC
--------------------------------------------------------------------------------
  WHALE ACTIVITY:
    Whale Volume (1h): {m.whale_volume_1h_btc:>15,.2f} BTC
    Whale Count:       {m.whale_count_1h:>15,}
    Whale Percentage:  {m.whale_percentage:>14.1f}%
--------------------------------------------------------------------------------
  RETAIL ACTIVITY:
    Retail Volume:     {m.retail_volume_1h_btc:>15,.2f} BTC
    Retail Count:      {m.retail_count_1h:>15,}
--------------------------------------------------------------------------------
  VELOCITY:
    Volume/sec:        {m.volume_velocity:>15.4f} BTC/s
    Acceleration:      {m.volume_acceleration:>15.6f}
    Trend:             {m.volume_trend:>15}
    Z-Score:           {m.volume_zscore:>15.2f}
--------------------------------------------------------------------------------
  TRANSACTION SIZE:
    Average:           {m.avg_tx_size_btc:>15.4f} BTC
    Median:            {m.median_tx_size_btc:>15.4f} BTC
--------------------------------------------------------------------------------
  FEES:
    Total (1h):        {m.total_fees_1h_sats:>15,} sats
    Avg per TX:        {m.avg_fee_per_tx:>15,.0f} sats
================================================================================
"""


# ==============================================================================
# TEST / DEMO
# ==============================================================================

if __name__ == "__main__":
    import random

    print("=" * 70)
    print("VOLUME AGGREGATOR - On-Chain Transaction Volume Tracking")
    print("=" * 70)
    print()

    aggregator = VolumeAggregator()

    # Simulate adding transactions from blocks
    print("Simulating 1000 transactions...")

    for i in range(1000):
        # Random transaction sizes with power law distribution
        if random.random() < 0.02:  # 2% are whales
            value = random.uniform(100, 1000)
        elif random.random() < 0.3:  # 30% are retail
            value = random.uniform(0.001, 0.1)
        else:  # 68% are normal
            value = random.uniform(0.1, 10)

        fee = int(random.uniform(1000, 50000))  # 1000-50000 sats

        aggregator.add_transaction(
            value_btc=value,
            fee_sats=fee,
            timestamp=time.time() - random.uniform(0, 3600)  # Last hour
        )

    # Update and get metrics
    metrics = aggregator.update()
    print(format_volume_summary(metrics))

    # Get signal
    signal, strength = aggregator.get_volume_signal()
    print(f"Volume Signal: {signal} (strength: {strength:.2f})")

    print("\n" + "=" * 70)
    print("PURE BLOCKCHAIN DATA - NO EXTERNAL APIS")
    print("=" * 70)
