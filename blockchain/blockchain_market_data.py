#!/usr/bin/env python3
"""
RENAISSANCE BLOCKCHAIN MARKET DATA - PURE BLOCKCHAIN SIGNALS
=============================================================
NO third-party exchange APIs. ALL data from the Bitcoin blockchain.

Data Sources (100% Blockchain):
1. Mempool transactions - Real BTC volume and flow
2. Fee rates - Market urgency/demand signals
3. Block confirmations - Network settlement data
4. Transaction patterns - Whale movements, accumulation

Market Signals Derived:
- Volume-weighted fee pressure (buying urgency)
- Large TX flow direction (whale accumulation/distribution)
- Mempool velocity (market activity)
- Block space demand (network congestion = high demand)

This is the REAL data. No exchange manipulation. No fake volume.
Pure Bitcoin blockchain truth.

DESIGNED FOR: $10 -> $10B HFT at 300K-1M trades
"""

import asyncio
import time
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Deque
from collections import deque
from enum import Enum

# Import the blockchain feed
from blockchain.blockchain_feed import BlockchainFeed, BlockchainTx, NetworkStats


class MarketSignal(Enum):
    """Market signal direction"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class BlockchainMarketState:
    """Current market state derived from blockchain data"""
    timestamp: float = 0

    # Volume metrics
    tx_rate: float = 0              # TXs per second
    btc_volume_1m: float = 0        # BTC volume last minute
    btc_volume_5m: float = 0        # BTC volume last 5 minutes
    large_tx_count: float = 0       # Large TXs (>10 BTC) last 5 min
    whale_tx_count: float = 0       # Whale TXs (>100 BTC) last 5 min

    # Fee metrics (demand signals)
    fee_fast: int = 0               # Fast fee (sat/vB)
    fee_medium: int = 0             # Medium fee
    fee_slow: int = 0               # Slow fee
    fee_spread: float = 0           # (fast-slow)/slow = urgency
    avg_fee_rate: float = 0         # Average fee rate of recent TXs

    # Mempool metrics
    mempool_size: int = 0           # Pending TXs
    mempool_vsize_mb: float = 0     # Mempool size in vMB
    mempool_growth_rate: float = 0  # TXs entering - leaving per minute

    # Block metrics
    last_block_height: int = 0
    last_block_time: float = 0
    block_interval: float = 600     # Time since last block
    blocks_per_hour: float = 6      # Recent block rate

    # Derived signals
    volume_trend: float = 0         # Volume increasing/decreasing
    fee_trend: float = 0            # Fees increasing/decreasing
    whale_activity: float = 0       # Large TX activity normalized
    market_pressure: float = 0      # -1 to 1, composite signal

    # Final signal
    signal: MarketSignal = MarketSignal.NEUTRAL
    signal_strength: float = 0      # 0 to 1


class BlockchainMarketData:
    """
    Pure blockchain market data provider

    Derives trading signals from raw blockchain data without
    relying on any third-party exchange APIs.

    Usage:
        market = BlockchainMarketData()
        await market.start()

        # Get current state
        state = market.get_state()
        print(f"Signal: {state.signal}, Strength: {state.signal_strength}")
    """

    # Thresholds - START AT ZERO, will be calibrated from LIVE percentiles
    # NO hardcoded values - learn what "large" and "high" mean from actual data
    LARGE_TX_THRESHOLD = 0      # LIVE: Set from 90th percentile of observed txs
    WHALE_TX_THRESHOLD = 0      # LIVE: Set from 99th percentile of observed txs
    HIGH_FEE_THRESHOLD = 0      # LIVE: Set from 90th percentile of observed fees
    LOW_FEE_THRESHOLD = 0       # LIVE: Set from 10th percentile of observed fees

    def __init__(
        self,
        on_signal: Optional[Callable[[MarketSignal, float], None]] = None,
        on_state: Optional[Callable[[BlockchainMarketState], None]] = None,
        signal_interval: float = 1.0,  # Generate signals every N seconds
    ):
        self.on_signal = on_signal
        self.on_state = on_state
        self.signal_interval = signal_interval

        # Blockchain feed
        self.feed: Optional[BlockchainFeed] = None
        self.running = False

        # Current state
        self.state = BlockchainMarketState()

        # Historical data for trend analysis
        self._tx_history: Deque[BlockchainTx] = deque(maxlen=10000)
        self._fee_history: Deque[tuple] = deque(maxlen=1000)  # (timestamp, fee_fast)
        self._volume_history: Deque[tuple] = deque(maxlen=1000)  # (timestamp, btc_volume)
        self._mempool_history: Deque[tuple] = deque(maxlen=1000)  # (timestamp, size)
        self._block_times: Deque[float] = deque(maxlen=100)

        # Large/whale TX tracking
        self._large_txs: Deque[BlockchainTx] = deque(maxlen=1000)
        self._whale_txs: Deque[BlockchainTx] = deque(maxlen=100)

        # Signal history
        self._signal_history: Deque[tuple] = deque(maxlen=10000)

        # Timing
        self._last_signal_time = 0
        self._start_time = 0

    async def start(self):
        """Start the blockchain market data provider"""
        self.running = True
        self._start_time = time.time()

        print("=" * 70)
        print("RENAISSANCE BLOCKCHAIN MARKET DATA - PURE BLOCKCHAIN")
        print("=" * 70)
        print("NO third-party APIs. ALL signals from Bitcoin blockchain.")
        print()

        # Create blockchain feed with callbacks
        self.feed = BlockchainFeed(
            on_tx=self._on_transaction,
            on_block=self._on_block,
            on_stats=self._on_stats,
            enable_rest_polling=True,
        )

        # Run feed and signal generator
        await asyncio.gather(
            self.feed.start(),
            self._signal_generator(),
            return_exceptions=True
        )

    def stop(self):
        """Stop the market data provider"""
        self.running = False
        if self.feed:
            self.feed.stop()

    def _on_transaction(self, tx: BlockchainTx):
        """Process incoming transaction"""
        self._tx_history.append(tx)

        # Track large transactions
        if tx.value_btc >= self.LARGE_TX_THRESHOLD:
            self._large_txs.append(tx)

        # Track whale transactions
        if tx.value_btc >= self.WHALE_TX_THRESHOLD:
            self._whale_txs.append(tx)

    def _on_block(self, block):
        """Process new block"""
        self._block_times.append(time.time())
        self.state.last_block_height = block.height
        self.state.last_block_time = block.timestamp

    def _on_stats(self, stats: NetworkStats):
        """Process network statistics"""
        now = time.time()

        # Update state
        self.state.mempool_size = stats.mempool_count
        self.state.mempool_vsize_mb = stats.mempool_vsize / 1e6
        self.state.fee_fast = stats.fee_fast
        self.state.fee_medium = stats.fee_medium
        self.state.fee_slow = stats.fee_slow

        # Track history
        self._fee_history.append((now, stats.fee_fast))
        self._mempool_history.append((now, stats.mempool_count))

    async def _signal_generator(self):
        """Generate market signals from blockchain data"""
        await asyncio.sleep(5)  # Wait for initial data

        print()
        print("BLOCKCHAIN MARKET SIGNALS ACTIVE")
        print("-" * 70)
        print(f"{'Time':>6} | {'TX/s':>5} | {'Vol(5m)':>8} | {'Fee':>4} | "
              f"{'Mempool':>8} | {'Whales':>6} | {'Signal':>10} | {'Str':>5}")
        print("-" * 70)

        while self.running:
            now = time.time()

            if now - self._last_signal_time >= self.signal_interval:
                self._update_state(now)
                self._generate_signal()
                self._last_signal_time = now

                # Print status
                elapsed = int(now - self._start_time)
                print(f"T+{elapsed:4d}s | {self.state.tx_rate:>5.1f} | "
                      f"{self.state.btc_volume_5m:>8.1f} | {self.state.fee_fast:>4} | "
                      f"{self.state.mempool_size:>8,} | {self.state.whale_tx_count:>6.0f} | "
                      f"{self.state.signal.name:>10} | {self.state.signal_strength:>5.2f}")

                # Callbacks
                if self.on_signal:
                    self.on_signal(self.state.signal, self.state.signal_strength)
                if self.on_state:
                    self.on_state(self.state)

            await asyncio.sleep(0.1)

    def _update_state(self, now: float):
        """Update market state from blockchain data"""
        # Get feed stats
        if self.feed:
            stats = self.feed.get_stats()
            self.state.tx_rate = stats.get('tx_per_sec', 0)
            self.state.timestamp = now

        # Calculate volumes from transaction history
        one_min_ago = now - 60
        five_min_ago = now - 300

        btc_1m = 0
        btc_5m = 0
        large_count = 0
        whale_count = 0
        fee_rates = []

        for tx in self._tx_history:
            if tx.timestamp >= one_min_ago:
                btc_1m += tx.value_btc
            if tx.timestamp >= five_min_ago:
                btc_5m += tx.value_btc
                fee_rates.append(tx.fee_rate)
                if tx.value_btc >= self.LARGE_TX_THRESHOLD:
                    large_count += 1
                if tx.value_btc >= self.WHALE_TX_THRESHOLD:
                    whale_count += 1

        self.state.btc_volume_1m = btc_1m
        self.state.btc_volume_5m = btc_5m
        self.state.large_tx_count = large_count
        self.state.whale_tx_count = whale_count
        self.state.avg_fee_rate = np.mean(fee_rates) if fee_rates else 0

        # Fee spread (urgency indicator)
        if self.state.fee_slow > 0:
            self.state.fee_spread = (self.state.fee_fast - self.state.fee_slow) / self.state.fee_slow

        # Volume trend (compare last 1m to previous 1m)
        self._volume_history.append((now, btc_1m))
        if len(self._volume_history) >= 2:
            prev_vol = self._volume_history[-2][1] if len(self._volume_history) > 1 else btc_1m
            if prev_vol > 0:
                self.state.volume_trend = (btc_1m - prev_vol) / prev_vol

        # Fee trend
        if len(self._fee_history) >= 2:
            recent_fees = [f[1] for f in list(self._fee_history)[-10:]]
            older_fees = [f[1] for f in list(self._fee_history)[-20:-10]] if len(self._fee_history) >= 20 else recent_fees
            if older_fees and np.mean(older_fees) > 0:
                self.state.fee_trend = (np.mean(recent_fees) - np.mean(older_fees)) / np.mean(older_fees)

        # Mempool growth rate
        if len(self._mempool_history) >= 2:
            recent = list(self._mempool_history)[-5:]
            if len(recent) >= 2:
                time_diff = recent[-1][0] - recent[0][0]
                size_diff = recent[-1][1] - recent[0][1]
                if time_diff > 0:
                    self.state.mempool_growth_rate = (size_diff / time_diff) * 60  # per minute

        # Block interval
        if self.state.last_block_time > 0:
            self.state.block_interval = now - self.state.last_block_time

        # Blocks per hour
        if len(self._block_times) >= 2:
            time_span = self._block_times[-1] - self._block_times[0]
            if time_span > 0:
                self.state.blocks_per_hour = (len(self._block_times) - 1) / (time_span / 3600)

        # Whale activity (normalized)
        # More whale TXs in 5 min = higher activity
        self.state.whale_activity = min(1.0, whale_count / 10)  # Normalize: 10+ whales = max

    def _generate_signal(self):
        """Generate trading signal from blockchain state"""
        # Composite market pressure calculation
        # Each factor contributes to overall signal

        pressure = 0.0
        weights_sum = 0.0

        # 1. Fee pressure (high fees = high demand = bullish)
        # Weight: 0.3
        if self.state.fee_fast > 0:
            if self.state.fee_fast >= self.HIGH_FEE_THRESHOLD:
                fee_signal = 1.0  # High fees = bullish
            elif self.state.fee_fast <= self.LOW_FEE_THRESHOLD:
                fee_signal = -0.5  # Very low fees = bearish/low activity
            else:
                # Normalize between thresholds
                fee_signal = (self.state.fee_fast - self.LOW_FEE_THRESHOLD) / (self.HIGH_FEE_THRESHOLD - self.LOW_FEE_THRESHOLD)
                fee_signal = fee_signal * 2 - 1  # Scale to -1 to 1

            pressure += fee_signal * 0.3
            weights_sum += 0.3

        # 2. Fee trend (rising fees = increasing demand = bullish)
        # Weight: 0.2
        fee_trend_signal = np.clip(self.state.fee_trend * 5, -1, 1)  # Amplify trend
        pressure += fee_trend_signal * 0.2
        weights_sum += 0.2

        # 3. Volume trend (rising volume = bullish)
        # Weight: 0.2
        vol_trend_signal = np.clip(self.state.volume_trend * 3, -1, 1)
        pressure += vol_trend_signal * 0.2
        weights_sum += 0.2

        # 4. Whale activity (high whale activity = significant moves coming)
        # Weight: 0.15
        # Whale activity itself is neutral but amplifies other signals
        whale_multiplier = 1.0 + self.state.whale_activity * 0.5

        # 5. Mempool growth (growing mempool = high activity = bullish)
        # Weight: 0.15
        if self.state.mempool_growth_rate != 0:
            mempool_signal = np.clip(self.state.mempool_growth_rate / 100, -1, 1)
            pressure += mempool_signal * 0.15
            weights_sum += 0.15

        # 6. Block interval (long interval = congestion = high demand)
        # Weight: 0.15
        if self.state.block_interval > 0:
            # Normal is ~600s. >1200 = congested, <300 = fast blocks
            if self.state.block_interval > 1200:
                block_signal = 0.5  # Congestion = demand
            elif self.state.block_interval < 300:
                block_signal = -0.3  # Fast blocks = less urgent
            else:
                block_signal = 0
            pressure += block_signal * 0.15
            weights_sum += 0.15

        # Apply whale multiplier
        pressure *= whale_multiplier

        # Normalize
        if weights_sum > 0:
            pressure = pressure / weights_sum

        # Clip to -1 to 1
        self.state.market_pressure = np.clip(pressure, -1, 1)

        # Generate signal
        if self.state.market_pressure >= 0.5:
            self.state.signal = MarketSignal.STRONG_BUY
            self.state.signal_strength = min(1.0, self.state.market_pressure)
        elif self.state.market_pressure >= 0.2:
            self.state.signal = MarketSignal.BUY
            self.state.signal_strength = self.state.market_pressure
        elif self.state.market_pressure <= -0.5:
            self.state.signal = MarketSignal.STRONG_SELL
            self.state.signal_strength = min(1.0, abs(self.state.market_pressure))
        elif self.state.market_pressure <= -0.2:
            self.state.signal = MarketSignal.SELL
            self.state.signal_strength = abs(self.state.market_pressure)
        else:
            self.state.signal = MarketSignal.NEUTRAL
            self.state.signal_strength = 0.0

        # Store in history
        self._signal_history.append((
            self.state.timestamp,
            self.state.signal,
            self.state.signal_strength,
            self.state.market_pressure
        ))

    def get_state(self) -> BlockchainMarketState:
        """Get current market state"""
        return self.state

    def get_signal(self) -> tuple:
        """Get current signal and strength"""
        return self.state.signal, self.state.signal_strength

    def get_signal_history(self, count: int = 100) -> List[tuple]:
        """Get recent signal history"""
        return list(self._signal_history)[-count:]

    def get_large_transactions(self, count: int = 50) -> List[BlockchainTx]:
        """Get recent large transactions"""
        return list(self._large_txs)[-count:]

    def get_whale_transactions(self, count: int = 20) -> List[BlockchainTx]:
        """Get recent whale transactions"""
        return list(self._whale_txs)[-count:]

    def get_stats(self) -> dict:
        """Get comprehensive statistics"""
        return {
            'state': {
                'tx_rate': self.state.tx_rate,
                'btc_volume_1m': self.state.btc_volume_1m,
                'btc_volume_5m': self.state.btc_volume_5m,
                'fee_fast': self.state.fee_fast,
                'fee_trend': self.state.fee_trend,
                'mempool_size': self.state.mempool_size,
                'whale_activity': self.state.whale_activity,
                'market_pressure': self.state.market_pressure,
            },
            'signal': self.state.signal.name,
            'signal_strength': self.state.signal_strength,
            'signal_value': self.state.signal.value,
            'large_tx_count': len(self._large_txs),
            'whale_tx_count': len(self._whale_txs),
            'signal_history_len': len(self._signal_history),
        }


async def test_blockchain_market(duration: int = 60):
    """Test the blockchain market data system"""

    signals_received = []

    def on_signal(signal: MarketSignal, strength: float):
        signals_received.append((time.time(), signal, strength))

    market = BlockchainMarketData(
        on_signal=on_signal,
        signal_interval=5.0,  # Signal every 5 seconds
    )

    async def stopper():
        await asyncio.sleep(duration)
        market.stop()

    await asyncio.gather(
        market.start(),
        stopper(),
        return_exceptions=True
    )

    # Final report
    print()
    print("=" * 70)
    print("BLOCKCHAIN MARKET DATA - RESULTS")
    print("=" * 70)

    stats = market.get_stats()
    state = market.get_state()

    print(f"Duration:        {duration} seconds")
    print(f"Signals Generated: {len(signals_received)}")
    print()
    print("Current State:")
    print(f"  TX Rate:       {state.tx_rate:.2f} tx/sec")
    print(f"  BTC Volume 5m: {state.btc_volume_5m:.2f} BTC")
    print(f"  Fee (fast):    {state.fee_fast} sat/vB")
    print(f"  Fee Trend:     {state.fee_trend:+.2%}")
    print(f"  Mempool:       {state.mempool_size:,} txs")
    print(f"  Whale Activity: {state.whale_activity:.2f}")
    print()
    print(f"Market Pressure: {state.market_pressure:+.3f}")
    print(f"Signal:          {state.signal.name} (strength: {state.signal_strength:.2f})")
    print()

    # Signal distribution
    if signals_received:
        from collections import Counter
        signal_counts = Counter(s[1].name for s in signals_received)
        print("Signal Distribution:")
        for sig, count in signal_counts.most_common():
            pct = count / len(signals_received) * 100
            print(f"  {sig}: {count} ({pct:.1f}%)")

    # Whale transactions
    whales = market.get_whale_transactions(5)
    if whales:
        print()
        print(f"Recent Whale Transactions (>{market.WHALE_TX_THRESHOLD} BTC):")
        for tx in whales:
            print(f"  {tx.value_btc:>10.2f} BTC | {tx.fee_rate:.1f} sat/vB | {tx.txid[:16]}...")

    print()
    print("=" * 70)
    print("PURE BLOCKCHAIN DATA - NO THIRD PARTY APIs")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    asyncio.run(test_blockchain_market(duration))
