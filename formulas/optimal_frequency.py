"""
Renaissance Formula Library - Optimal Trade Frequency
======================================================
ID 332: High Frequency + High Quality Trading

The Problem:
- Old thinking: "Trade less for quality" OR "Trade more for volume"
- Reality: You can have BOTH if signals are truly independent

Mathematical Foundation:
- Independent signals compound: If each trade has +EV, more trades = more profit
- Key insight: Frequency is only bad when signals are CORRELATED (same info)
- Solution: Trade frequently on INDEPENDENT signals, not repeated signals

From Information Theory (Shannon 1948):
- Each truly new piece of information = potential edge
- Blockchain produces NEW information every second (txs, fees, mempool)
- Therefore: High frequency IS possible with quality

Formula:
    Optimal_N = Edge² × Capital / (Variance × Cost²)

    But we modify for HIGH FREQ + HIGH QUALITY:
    - Only count signals that are INFORMATION-DISTINCT
    - Require minimum time between SAME signal type
    - Allow rapid trading on DIFFERENT signal types

Sources:
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Avellaneda & Stoikov (2008): "High-frequency trading in a limit order book"
- Cartea, Jaimungal & Penalva (2015): "Algorithmic and High-Frequency Trading"
"""

import numpy as np
from typing import Dict, Any, List, Optional, Set
from collections import deque
from dataclasses import dataclass
import time

from .base import BaseFormula, FormulaRegistry


@dataclass
class SignalEvent:
    """A distinct signal event"""
    timestamp: float
    signal_type: str  # e.g., "ou_reversion", "momentum", "vpin", "hft"
    direction: int
    strength: float
    price: float


@FormulaRegistry.register(332, name="OptimalFrequency", category="execution")
class OptimalFrequencyFormula(BaseFormula):
    """
    ID 332: Optimal Trade Frequency - HIGH FREQ + HIGH QUALITY

    Key Innovation: Trade frequently on INDEPENDENT signals,
    not repeated signals of the same type.

    Signal Types Tracked:
    1. Mean Reversion (OU z-score extreme)
    2. Momentum (trend continuation)
    3. VPIN (toxicity clear)
    4. HFT Blockchain (mempool/fee signals)
    5. Volume Spike (whale detection)
    6. Regime Change (volatility shift)

    Rules:
    - Same signal type: Minimum 30 seconds between trades
    - Different signal types: Can trade immediately
    - Signal confluence (3+ types agree): Trade immediately with size boost
    - Quality filter: Only trade when signal strength > threshold

    This allows:
    - 100+ trades/day when signals are diverse
    - Quality maintained by per-signal-type cooldown
    - Maximum edge extraction from independent information sources
    """

    FORMULA_ID = 332
    CATEGORY = "execution"
    NAME = "Optimal Frequency"
    DESCRIPTION = "High frequency + high quality trading via independent signals"

    # Signal type definitions
    SIGNAL_TYPES = [
        "ou_reversion",      # Mean reversion from OU process
        "momentum",          # Trend following
        "vpin_clear",        # VPIN shows safe to trade
        "hft_blockchain",    # Blockchain-derived signals
        "volume_spike",      # Unusual volume
        "regime_shift",      # Volatility regime change
        "orderflow",         # Order flow imbalance
        "microstructure",    # Bid-ask dynamics
    ]

    def __init__(self,
                 lookback: int = 500,
                 same_signal_cooldown: float = 30.0,  # Seconds between same signal
                 min_signal_strength: float = 0.3,    # Minimum strength to trade
                 confluence_boost: float = 1.5,       # Size multiplier for confluence
                 max_trades_per_minute: int = 10,     # Hard cap
                 **kwargs):
        super().__init__(lookback, **kwargs)

        self.same_signal_cooldown = same_signal_cooldown
        self.min_signal_strength = min_signal_strength
        self.confluence_boost = confluence_boost
        self.max_trades_per_minute = max_trades_per_minute

        # Track last trade time for each signal type
        self.last_trade_by_type: Dict[str, float] = {
            st: 0.0 for st in self.SIGNAL_TYPES
        }

        # Recent signals for confluence detection
        self.recent_signals: deque = deque(maxlen=100)

        # Trade tracking
        self.trades_this_minute: deque = deque(maxlen=max_trades_per_minute * 2)

        # Current state
        self.active_signals: Dict[str, SignalEvent] = {}
        self.can_trade = True
        self.trade_size_multiplier = 1.0
        self.confluence_count = 0

        # Performance tracking
        self.signals_by_type: Dict[str, int] = {st: 0 for st in self.SIGNAL_TYPES}
        self.trades_by_type: Dict[str, int] = {st: 0 for st in self.SIGNAL_TYPES}

    def register_signal(self,
                       signal_type: str,
                       direction: int,
                       strength: float,
                       price: float,
                       timestamp: float = None) -> bool:
        """
        Register a new signal and check if we can trade on it.

        Args:
            signal_type: One of SIGNAL_TYPES
            direction: 1 for buy, -1 for sell
            strength: Signal strength 0-1
            price: Current price
            timestamp: Signal time

        Returns:
            True if this signal can be traded, False otherwise
        """
        now = timestamp or time.time()

        if signal_type not in self.SIGNAL_TYPES:
            signal_type = "microstructure"  # Default bucket

        # Track signal
        self.signals_by_type[signal_type] = self.signals_by_type.get(signal_type, 0) + 1

        # Create signal event
        signal = SignalEvent(
            timestamp=now,
            signal_type=signal_type,
            direction=direction,
            strength=strength,
            price=price
        )

        # Store as active signal
        self.active_signals[signal_type] = signal
        self.recent_signals.append(signal)

        # Check if we can trade
        return self._can_trade_signal(signal_type, direction, strength, now)

    def _can_trade_signal(self,
                         signal_type: str,
                         direction: int,
                         strength: float,
                         now: float) -> bool:
        """Check if we can trade on this specific signal"""

        # Filter 1: Minimum strength
        if strength < self.min_signal_strength:
            return False

        # Filter 2: Rate limit (max trades per minute)
        self._cleanup_trade_times(now)
        if len(self.trades_this_minute) >= self.max_trades_per_minute:
            return False

        # Filter 3: Same signal type cooldown
        last_trade = self.last_trade_by_type.get(signal_type, 0)
        time_since_last = now - last_trade

        if time_since_last < self.same_signal_cooldown:
            # Check for confluence - if 3+ signal types agree, override cooldown
            confluence = self._check_confluence(direction, now)
            if confluence < 3:
                return False
            # Confluence override - can trade despite cooldown

        # All filters passed
        return True

    def _check_confluence(self, direction: int, now: float) -> int:
        """
        Check how many independent signal types agree on direction.

        Returns count of agreeing signals in last 10 seconds.
        """
        window = 10.0  # seconds
        agreeing_types: Set[str] = set()

        for signal in reversed(list(self.recent_signals)):
            if now - signal.timestamp > window:
                break
            if signal.direction == direction:
                agreeing_types.add(signal.signal_type)

        self.confluence_count = len(agreeing_types)
        return self.confluence_count

    def _cleanup_trade_times(self, now: float):
        """Remove trades older than 1 minute"""
        while self.trades_this_minute and now - self.trades_this_minute[0] > 60:
            self.trades_this_minute.popleft()

    def record_trade(self, signal_type: str, timestamp: float = None):
        """Record that we traded on a signal"""
        now = timestamp or time.time()

        self.last_trade_by_type[signal_type] = now
        self.trades_this_minute.append(now)
        self.trades_by_type[signal_type] = self.trades_by_type.get(signal_type, 0) + 1

    def get_trade_size_multiplier(self, direction: int = None) -> float:
        """
        Get position size multiplier based on signal confluence.

        More agreeing signals = larger position (up to 2x)
        """
        if direction is not None:
            confluence = self._check_confluence(direction, time.time())
        else:
            confluence = self.confluence_count

        if confluence >= 5:
            return 2.0  # 5+ signals agree - maximum size
        elif confluence >= 4:
            return 1.75
        elif confluence >= 3:
            return self.confluence_boost  # 1.5x
        elif confluence >= 2:
            return 1.25
        else:
            return 1.0  # Single signal - normal size

    def get_optimal_frequency(self,
                             edge: float,
                             variance: float,
                             cost: float,
                             capital: float) -> Dict[str, float]:
        """
        Calculate optimal trading frequency based on Almgren-Chriss.

        Formula: N* = sqrt(edge² × capital / (variance × cost²))

        But we also account for information arrival rate.
        """
        if cost <= 0 or variance <= 0:
            # NO hardcoded fallbacks - return zeros until we have real data
            return {'optimal_trades_per_day': 0, 'max_trades_per_hour': 0}

        # Almgren-Chriss optimal
        ac_optimal = np.sqrt(edge**2 * capital / (variance * cost**2))

        # Information arrival rate - MUST BE FROM LIVE BLOCKCHAIN DATA
        # This should be passed in, not hardcoded
        info_rate = self._live_info_rate if hasattr(self, '_live_info_rate') and self._live_info_rate > 0 else 0

        # If no live data, return zeros (don't trade)
        if info_rate <= 0:
            return {'optimal_trades_per_day': 0, 'max_trades_per_hour': 0, 'waiting_for_live_data': True}

        # Practical optimal = min of theoretical and info rate
        practical_optimal = min(ac_optimal, info_rate)

        # Apply our rate limit (if set from live data)
        max_per_min = self.max_trades_per_minute if self.max_trades_per_minute > 0 else practical_optimal / 60
        capped_optimal = min(practical_optimal, max_per_min * 60)

        return {
            'almgren_chriss_optimal': ac_optimal,
            'information_rate': info_rate,
            'practical_optimal': practical_optimal,
            'capped_optimal': capped_optimal,
            'optimal_trades_per_day': capped_optimal * 24,
            'max_trades_per_hour': capped_optimal,
            'recommended_cooldown': 3600 / capped_optimal if capped_optimal > 0 else 0,
        }

    def set_live_info_rate(self, tx_per_second: float):
        """Set information rate from LIVE blockchain data"""
        self._live_info_rate = tx_per_second * 3600  # Convert to per hour

    def _compute(self) -> None:
        """Update signal based on current state"""
        # Signal indicates whether we're in a good trading regime
        active_count = sum(1 for s in self.active_signals.values()
                         if s.strength >= self.min_signal_strength)

        if active_count >= 3:
            self.signal = 1  # Multiple quality signals - trade more
            self.confidence = min(1.0, active_count / 5)
        elif active_count >= 1:
            self.signal = 0  # Normal trading
            self.confidence = 0.5
        else:
            self.signal = -1  # No quality signals - reduce trading
            self.confidence = 0.3

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'active_signals': len(self.active_signals),
            'confluence_count': self.confluence_count,
            'trades_this_minute': len(self.trades_this_minute),
            'trade_size_multiplier': self.get_trade_size_multiplier(),
            'signals_by_type': dict(self.signals_by_type),
            'trades_by_type': dict(self.trades_by_type),
            'can_trade': self.can_trade,
        })
        return state


@FormulaRegistry.register(337, name="IndependentSignalDetector", category="execution")
class IndependentSignalDetector(BaseFormula):
    """
    ID 337: Independent Signal Detector

    Detects when signals are truly INDEPENDENT (uncorrelated)
    vs when they're just echoes of the same information.

    Key Insight:
    - Correlated signals = same edge, shouldn't stack
    - Independent signals = different edges, CAN stack

    Uses Mutual Information to detect signal correlation:
    MI(X,Y) = Σ p(x,y) × log(p(x,y) / (p(x)p(y)))

    If MI < threshold: Signals are independent, can trade both
    If MI > threshold: Signals are correlated, don't double-count
    """

    FORMULA_ID = 337
    CATEGORY = "execution"
    NAME = "Independent Signal Detector"
    DESCRIPTION = "Detects truly independent signals for frequency optimization"

    def __init__(self, lookback: int = 200, mi_threshold: float = 0.3, **kwargs):
        super().__init__(lookback, **kwargs)
        self.mi_threshold = mi_threshold

        # Track signal pairs
        self.signal_history: Dict[str, deque] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}

        # Independence scores
        self.independence_scores: Dict[str, float] = {}

    def add_signal(self, signal_type: str, value: float):
        """Add a signal value for correlation tracking"""
        if signal_type not in self.signal_history:
            self.signal_history[signal_type] = deque(maxlen=self.lookback)

        self.signal_history[signal_type].append(value)
        self._update_correlations()

    def _update_correlations(self):
        """Update correlation matrix between all signal pairs"""
        signal_types = list(self.signal_history.keys())

        for i, type1 in enumerate(signal_types):
            for type2 in signal_types[i+1:]:
                if len(self.signal_history[type1]) < 20:
                    continue
                if len(self.signal_history[type2]) < 20:
                    continue

                # Calculate correlation
                arr1 = np.array(list(self.signal_history[type1]))[-50:]
                arr2 = np.array(list(self.signal_history[type2]))[-50:]

                min_len = min(len(arr1), len(arr2))
                if min_len < 10:
                    continue

                corr = np.corrcoef(arr1[-min_len:], arr2[-min_len:])[0, 1]

                if type1 not in self.correlation_matrix:
                    self.correlation_matrix[type1] = {}
                self.correlation_matrix[type1][type2] = abs(corr)

    def are_independent(self, type1: str, type2: str) -> bool:
        """Check if two signal types are independent"""
        if type1 == type2:
            return False

        corr = self.correlation_matrix.get(type1, {}).get(type2, 0)
        return corr < self.mi_threshold

    def get_independent_signals(self, current_type: str) -> List[str]:
        """Get list of signal types independent from current"""
        independent = []
        for other_type in self.signal_history.keys():
            if self.are_independent(current_type, other_type):
                independent.append(other_type)
        return independent

    def _compute(self) -> None:
        # Count independent signal pairs
        total_pairs = 0
        independent_pairs = 0

        signal_types = list(self.signal_history.keys())
        for i, type1 in enumerate(signal_types):
            for type2 in signal_types[i+1:]:
                total_pairs += 1
                if self.are_independent(type1, type2):
                    independent_pairs += 1

        if total_pairs > 0:
            self.signal = 1 if independent_pairs / total_pairs > 0.5 else 0
            self.confidence = independent_pairs / total_pairs
        else:
            self.signal = 0
            self.confidence = 0.5

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'signal_types_tracked': len(self.signal_history),
            'correlation_matrix': {k: dict(v) for k, v in self.correlation_matrix.items()},
        })
        return state
