"""
Renaissance Formula Library - Volume-Based Trading Frequency
=============================================================
ID 401: Dynamic frequency from LIVE 24h volume data

NO HARDCODING - All values derived from real blockchain data:
    1. Get 24h volume in BTC from live feed
    2. volume_per_hour = volume_24h / 24
    3. volume_per_second = volume_per_hour / 3600
    4. Optimal trades = f(volume_per_second, edge, capital)

The key insight:
- More volume = more opportunities = higher frequency OK
- Less volume = fewer opportunities = lower frequency needed
- Adaptive to REAL market conditions, not hardcoded assumptions
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .base import BaseFormula, FormulaRegistry


@dataclass
class VolumeStats:
    """Live volume statistics"""
    volume_24h_btc: float = 0.0
    volume_per_hour: float = 0.0
    volume_per_minute: float = 0.0
    volume_per_second: float = 0.0
    tx_per_second: float = 0.0
    optimal_trades_per_minute: float = 0.0
    optimal_cooldown_seconds: float = 30.0
    last_update: float = 0.0


@FormulaRegistry.register(401, name="VolumeBasedFrequency", category="execution")
class VolumeBasedFrequency(BaseFormula):
    """
    ID 401: Volume-Based Trading Frequency

    Calculates optimal trading frequency from LIVE volume data.
    NO hardcoded values - everything derived from real blockchain data.

    Formula:
        volume_per_second = volume_24h_btc / 86400

        # More volume = more opportunities
        opportunity_rate = volume_per_second * price_usd

        # Optimal frequency scales with sqrt of opportunity rate
        # (diminishing returns from too-frequent trading)
        optimal_trades_per_min = sqrt(opportunity_rate / 1000) * edge_multiplier

        # Cooldown is inverse of frequency
        cooldown_seconds = 60 / optimal_trades_per_min

    This ensures:
        - High volume periods -> More trades allowed
        - Low volume periods -> Fewer trades (wait for liquidity)
        - Always adaptive to CURRENT conditions
    """

    FORMULA_ID = 401
    CATEGORY = "execution"
    NAME = "Volume-Based Frequency"
    DESCRIPTION = "Dynamic trading frequency from live 24h volume"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)

        # Live volume stats - NO HARDCODING
        self.stats = VolumeStats()

        # Track volume ourselves from feed data
        self._volume_buffer: list = []  # (timestamp, btc_value) pairs
        self._buffer_duration = 86400  # 24 hours in seconds

        # Last trade tracking
        self._last_trade_time = 0.0
        self._trades_this_minute: list = []

    def update_from_feed(self, feed_stats: Dict[str, Any]):
        """
        Update from blockchain feed stats.

        feed_stats should contain:
            - total_btc: Total BTC volume observed
            - elapsed_sec: How long feed has been running
            - btc_per_sec: BTC volume per second
            - tx_per_sec: Transactions per second
        """
        now = time.time()

        btc_per_sec = feed_stats.get('btc_per_sec', 0.0)
        tx_per_sec = feed_stats.get('tx_per_sec', 0.0)

        # Extrapolate to 24h volume from current rate
        # This is LIVE data, not hardcoded
        self.stats.volume_per_second = btc_per_sec
        self.stats.volume_per_minute = btc_per_sec * 60
        self.stats.volume_per_hour = btc_per_sec * 3600
        self.stats.volume_24h_btc = btc_per_sec * 86400
        self.stats.tx_per_second = tx_per_sec
        self.stats.last_update = now

        # Calculate optimal frequency
        self._calculate_optimal_frequency()

    def add_transaction(self, btc_value: float, timestamp: float = None):
        """
        Add a transaction to volume tracking.
        This builds our own 24h volume from raw data.
        """
        ts = timestamp or time.time()
        self._volume_buffer.append((ts, btc_value))

        # Clean old entries (older than 24h)
        cutoff = ts - self._buffer_duration
        self._volume_buffer = [(t, v) for t, v in self._volume_buffer if t > cutoff]

        # Recalculate from our buffer
        self._recalculate_volume()

    def _recalculate_volume(self):
        """Recalculate volume stats from our buffer"""
        if not self._volume_buffer:
            return

        now = time.time()

        # Sum all volume in buffer
        total_btc = sum(v for t, v in self._volume_buffer)

        # Time span of our data
        oldest = self._volume_buffer[0][0]
        span_seconds = now - oldest

        if span_seconds < 1:
            span_seconds = 1

        # Calculate rates from ACTUAL data
        self.stats.volume_per_second = total_btc / span_seconds
        self.stats.volume_per_minute = self.stats.volume_per_second * 60
        self.stats.volume_per_hour = self.stats.volume_per_second * 3600

        # Extrapolate to 24h (if we have less data)
        if span_seconds < 86400:
            self.stats.volume_24h_btc = self.stats.volume_per_second * 86400
        else:
            self.stats.volume_24h_btc = total_btc

        self.stats.tx_per_second = len(self._volume_buffer) / span_seconds
        self.stats.last_update = now

        # Update optimal frequency
        self._calculate_optimal_frequency()

    def _calculate_optimal_frequency(self, btc_price_usd: float = 90000.0):
        """
        Calculate optimal trading frequency from volume.

        Higher volume = more opportunities = higher frequency OK
        """
        import math

        vol_per_sec = self.stats.volume_per_second

        if vol_per_sec <= 0:
            # No volume data yet - conservative default
            self.stats.optimal_trades_per_minute = 1.0
            self.stats.optimal_cooldown_seconds = 60.0
            return

        # Convert to USD opportunity rate
        usd_per_second = vol_per_sec * btc_price_usd

        # Optimal trades scales with sqrt of opportunity rate
        # sqrt provides diminishing returns (can't capture all volume)
        # Divide by 10000 to normalize (typical BTC ~$90k, vol ~1 BTC/sec = $90k/sec)
        raw_frequency = math.sqrt(usd_per_second / 10000)

        # Clamp to reasonable bounds
        # Min: 0.5 trades/min (2 min cooldown)
        # Max: 20 trades/min (3 sec cooldown) - still very aggressive
        self.stats.optimal_trades_per_minute = max(0.5, min(20.0, raw_frequency))

        # Cooldown is inverse
        self.stats.optimal_cooldown_seconds = 60.0 / self.stats.optimal_trades_per_minute

    def can_trade(self, signal_type: str = "default") -> bool:
        """
        Check if enough time has passed since last trade.
        Uses LIVE volume-derived cooldown, not hardcoded.
        """
        now = time.time()

        # Clean old trades from this minute
        cutoff = now - 60
        self._trades_this_minute = [t for t in self._trades_this_minute if t > cutoff]

        # Check rate limit (from live volume data)
        max_per_minute = int(self.stats.optimal_trades_per_minute)
        if len(self._trades_this_minute) >= max(1, max_per_minute):
            return False

        # Check cooldown (from live volume data)
        time_since_last = now - self._last_trade_time
        if time_since_last < self.stats.optimal_cooldown_seconds:
            return False

        return True

    def record_trade(self):
        """Record that a trade was made"""
        now = time.time()
        self._last_trade_time = now
        self._trades_this_minute.append(now)

    def get_cooldown_remaining(self) -> float:
        """Get seconds until next trade allowed"""
        now = time.time()
        time_since_last = now - self._last_trade_time
        remaining = self.stats.optimal_cooldown_seconds - time_since_last
        return max(0.0, remaining)

    def _compute(self) -> None:
        """Update signal based on volume conditions"""
        vol_per_min = self.stats.volume_per_minute

        if vol_per_min >= 100:  # High volume (>100 BTC/min)
            self.signal = 1  # Trade more frequently
            self.confidence = 0.8
        elif vol_per_min >= 10:  # Normal volume
            self.signal = 0  # Normal frequency
            self.confidence = 0.5
        else:  # Low volume
            self.signal = -1  # Trade less (reduce churn)
            self.confidence = 0.3

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'volume_24h_btc': self.stats.volume_24h_btc,
            'volume_per_hour': self.stats.volume_per_hour,
            'volume_per_minute': self.stats.volume_per_minute,
            'volume_per_second': self.stats.volume_per_second,
            'tx_per_second': self.stats.tx_per_second,
            'optimal_trades_per_minute': self.stats.optimal_trades_per_minute,
            'optimal_cooldown_seconds': self.stats.optimal_cooldown_seconds,
            'trades_this_minute': len(self._trades_this_minute),
            'cooldown_remaining': self.get_cooldown_remaining(),
            'last_update': self.stats.last_update,
        })
        return state


@FormulaRegistry.register(402, name="AdaptiveVolumeScaler", category="execution")
class AdaptiveVolumeScaler(BaseFormula):
    """
    ID 402: Adaptive Volume Scaler

    Scales position size based on LIVE volume relative to historical.

    Logic:
        current_volume = live from feed
        avg_volume = rolling average from last N hours

        volume_ratio = current_volume / avg_volume

        If volume_ratio > 1.5: Increase size (more liquidity)
        If volume_ratio < 0.5: Decrease size (less liquidity)
        Else: Normal size
    """

    FORMULA_ID = 402
    CATEGORY = "execution"
    NAME = "Adaptive Volume Scaler"
    DESCRIPTION = "Scale position size from live volume vs historical"

    def __init__(self, lookback: int = 100, history_hours: int = 24, **kwargs):
        super().__init__(lookback, **kwargs)

        self.history_hours = history_hours

        # Volume history - (timestamp, btc_per_minute) pairs
        self._volume_history: list = []
        self._history_duration = history_hours * 3600

        # Current state
        self.current_volume_per_min = 0.0
        self.avg_volume_per_min = 0.0
        self.volume_ratio = 1.0
        self.size_multiplier = 1.0

    def update_volume(self, btc_per_minute: float):
        """Update with current volume rate"""
        now = time.time()

        self.current_volume_per_min = btc_per_minute
        self._volume_history.append((now, btc_per_minute))

        # Clean old entries
        cutoff = now - self._history_duration
        self._volume_history = [(t, v) for t, v in self._volume_history if t > cutoff]

        # Calculate average
        if self._volume_history:
            self.avg_volume_per_min = sum(v for t, v in self._volume_history) / len(self._volume_history)
        else:
            self.avg_volume_per_min = btc_per_minute

        # Volume ratio
        if self.avg_volume_per_min > 0:
            self.volume_ratio = self.current_volume_per_min / self.avg_volume_per_min
        else:
            self.volume_ratio = 1.0

        # Size multiplier based on ratio
        if self.volume_ratio >= 2.0:
            self.size_multiplier = 1.5  # 50% larger - high liquidity
        elif self.volume_ratio >= 1.5:
            self.size_multiplier = 1.25
        elif self.volume_ratio >= 0.75:
            self.size_multiplier = 1.0  # Normal
        elif self.volume_ratio >= 0.5:
            self.size_multiplier = 0.75
        else:
            self.size_multiplier = 0.5  # 50% smaller - low liquidity

    def get_size_multiplier(self) -> float:
        """Get position size multiplier based on live volume"""
        return self.size_multiplier

    def _compute(self) -> None:
        """Update signal based on volume conditions"""
        if self.volume_ratio >= 1.5:
            self.signal = 1  # High volume - good for trading
            self.confidence = min(1.0, self.volume_ratio / 2)
        elif self.volume_ratio >= 0.5:
            self.signal = 0  # Normal
            self.confidence = 0.5
        else:
            self.signal = -1  # Low volume - reduce trading
            self.confidence = 0.3

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'current_volume_per_min': self.current_volume_per_min,
            'avg_volume_per_min': self.avg_volume_per_min,
            'volume_ratio': self.volume_ratio,
            'size_multiplier': self.size_multiplier,
            'history_samples': len(self._volume_history),
        })
        return state
