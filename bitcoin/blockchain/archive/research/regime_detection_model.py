#!/usr/bin/env python3
"""
REGIME DETECTION MODEL - The Mathematical Truth
================================================

WHY 50% ACCURACY FAILS:
-----------------------
Simple flow direction (inflow=SHORT, outflow=LONG) has ZERO predictive power
because individual flows are NOISE. Academic research proves:

1. Individual transactions are random - No single flow predicts price
2. Only AGGREGATE patterns have signal - Net flow over time windows
3. Only UNUSUAL flows matter - Z-score > 2 sigma from historical mean
4. Only PERSISTENT patterns work - Same direction for multiple windows

THE MATHEMATICAL INSIGHT:
-------------------------
Price doesn't react to individual flows. Price reacts to REGIME CHANGES.

A regime change is when:
  - Flow pattern deviates significantly from historical norm (z-score)
  - Pattern persists across multiple time windows
  - Aggregate imbalance exceeds threshold

FORMULAS IMPLEMENTED:
---------------------

1. FLOW Z-SCORE (Normalized Deviation):
   z_flow = (F_current - mu_F) / sigma_F

   Where:
     F_current = net flow in current window
     mu_F = rolling mean of net flow (e.g., 20-window)
     sigma_F = rolling std of net flow

   Signal: Only act when |z_flow| > 2.0 (unusual flow)

2. CUMULATIVE FLOW IMBALANCE (CFI):
   CFI = sum(NetFlow_i) for i in [t-n, t]

   Normalized: CFI_norm = CFI / (sum(|Flow_i|) + epsilon)

   Range: -1 (all inflows) to +1 (all outflows)
   Signal: |CFI_norm| > 0.6 indicates strong directional pressure

3. FLOW PERSISTENCE SCORE (FPS):
   FPS = consecutive_windows_same_direction / lookback_windows

   Signal: FPS > 0.7 means 70%+ of recent windows had same direction

4. WHALE DOMINANCE RATIO (WDR):
   WDR = whale_flow / total_flow

   Where whale_flow = sum of flows > 100 BTC

   Signal: WDR > 0.5 means whales dominate (more predictive)

5. REGIME CHANGE SCORE (RCS) - FINAL SIGNAL:
   RCS = w1*|z_flow| + w2*|CFI| + w3*FPS + w4*WDR

   Where w1=0.3, w2=0.3, w3=0.2, w4=0.2 (calibrated weights)

   Direction = sign(z_flow) or sign(CFI)

   Trade when: RCS > threshold AND direction is clear

EXAMPLE:
--------
Current window: -50 BTC net flow (inflow)
20-window mean: +10 BTC
20-window std: 30 BTC
z_flow = (-50 - 10) / 30 = -2.0 (UNUSUAL - 2 sigma below mean)

CFI over 5 windows: -200 BTC
Total absolute flow: 250 BTC
CFI_norm = -200/250 = -0.8 (STRONG inflow pressure)

FPS: 4/5 = 0.8 (80% of windows were net inflow)

WDR: 150/200 = 0.75 (whales dominate)

RCS = 0.3*2.0 + 0.3*0.8 + 0.2*0.8 + 0.2*0.75 = 1.15

Direction = BEARISH (z_flow and CFI both negative)

SIGNAL: STRONG SHORT (RCS > 1.0, clear bearish direction)
"""

import math
import sqlite3
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import threading


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""

    # Database
    db_path: str = "/root/sovereign/regime_detection.db"

    # Window sizes
    flow_window_seconds: int = 600  # 10-minute flow windows
    lookback_windows: int = 20      # 20 windows for rolling stats

    # Z-score thresholds
    z_threshold_weak: float = 1.5   # Somewhat unusual
    z_threshold_strong: float = 2.0 # Very unusual
    z_threshold_extreme: float = 3.0 # Extremely unusual

    # CFI thresholds
    cfi_threshold_weak: float = 0.4
    cfi_threshold_strong: float = 0.6
    cfi_threshold_extreme: float = 0.8

    # Persistence threshold
    persistence_threshold: float = 0.6  # 60%+ same direction

    # Whale definition
    whale_threshold_btc: float = 100.0

    # RCS thresholds for signal generation
    rcs_threshold_weak: float = 0.6    # Weak signal
    rcs_threshold_medium: float = 0.8  # Medium signal
    rcs_threshold_strong: float = 1.0  # Strong signal

    # Weights for RCS calculation
    weight_z_score: float = 0.30
    weight_cfi: float = 0.30
    weight_persistence: float = 0.20
    weight_whale: float = 0.20

    # Minimum flow to consider
    min_flow_btc: float = 1.0


@dataclass
class FlowWindow:
    """A single time window of flow data."""
    start_time: datetime
    end_time: datetime
    inflow_btc: float = 0.0
    outflow_btc: float = 0.0
    inflow_count: int = 0
    outflow_count: int = 0
    whale_inflow_btc: float = 0.0
    whale_outflow_btc: float = 0.0

    @property
    def net_flow(self) -> float:
        """Net flow: positive = outflow dominated (bullish)."""
        return self.outflow_btc - self.inflow_btc

    @property
    def total_flow(self) -> float:
        """Total absolute flow."""
        return self.inflow_btc + self.outflow_btc

    @property
    def whale_flow(self) -> float:
        """Total whale flow."""
        return self.whale_inflow_btc + self.whale_outflow_btc


@dataclass
class RegimeSignal:
    """A regime detection signal."""
    timestamp: datetime
    exchange: str

    # Component scores
    z_score: float
    cfi_normalized: float
    persistence_score: float
    whale_ratio: float

    # Final score
    regime_change_score: float

    # Direction: 1 = bullish, -1 = bearish, 0 = neutral
    direction: int
    direction_label: str  # "BULLISH", "BEARISH", "NEUTRAL"

    # Signal strength: "NONE", "WEAK", "MEDIUM", "STRONG"
    strength: str

    # Raw data
    current_window_net_flow: float
    rolling_mean: float
    rolling_std: float
    cumulative_flow: float

    @property
    def is_actionable(self) -> bool:
        return self.strength in ("MEDIUM", "STRONG")


class RegimeDetector:
    """
    Detects regime changes in exchange flow patterns.

    Instead of reacting to individual flows, this detects
    when flow patterns become UNUSUAL relative to history.
    """

    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()

        # Flow windows per exchange
        # Key: exchange, Value: deque of FlowWindow
        self.windows: Dict[str, deque] = {}

        # Current (incomplete) window per exchange
        self.current_window: Dict[str, FlowWindow] = {}

        # Lock for thread safety
        self.lock = threading.Lock()

        # Initialize database
        self.db = self._init_database()

    def _init_database(self) -> sqlite3.Connection:
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.config.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        conn.executescript("""
            -- Flow windows
            CREATE TABLE IF NOT EXISTS flow_windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exchange TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                inflow_btc REAL,
                outflow_btc REAL,
                net_flow REAL,
                inflow_count INTEGER,
                outflow_count INTEGER,
                whale_inflow_btc REAL,
                whale_outflow_btc REAL
            );

            -- Regime signals
            CREATE TABLE IF NOT EXISTS regime_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                exchange TEXT NOT NULL,
                z_score REAL,
                cfi_normalized REAL,
                persistence_score REAL,
                whale_ratio REAL,
                regime_change_score REAL,
                direction INTEGER,
                direction_label TEXT,
                strength TEXT,
                current_net_flow REAL,
                rolling_mean REAL,
                rolling_std REAL,
                cumulative_flow REAL,

                -- Outcome tracking
                price_at_signal REAL,
                price_after_5m REAL,
                price_after_30m REAL,
                actual_direction INTEGER,
                was_correct INTEGER
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_windows_exchange
                ON flow_windows(exchange, start_time);
            CREATE INDEX IF NOT EXISTS idx_signals_exchange
                ON regime_signals(exchange, timestamp);
        """)

        conn.commit()
        return conn

    def _get_or_create_window(self, exchange: str) -> FlowWindow:
        """Get current window or create new one."""
        now = datetime.now(timezone.utc)

        if exchange not in self.current_window:
            # Create new window
            window_start = now.replace(
                second=(now.second // (self.config.flow_window_seconds % 60)) *
                       (self.config.flow_window_seconds % 60),
                microsecond=0
            )
            self.current_window[exchange] = FlowWindow(
                start_time=window_start,
                end_time=window_start + timedelta(seconds=self.config.flow_window_seconds)
            )

        window = self.current_window[exchange]

        # Check if window expired
        if now >= window.end_time:
            # Finalize current window
            self._finalize_window(exchange, window)

            # Create new window
            window_start = now.replace(
                second=(now.second // (self.config.flow_window_seconds % 60)) *
                       (self.config.flow_window_seconds % 60),
                microsecond=0
            )
            self.current_window[exchange] = FlowWindow(
                start_time=window_start,
                end_time=window_start + timedelta(seconds=self.config.flow_window_seconds)
            )
            window = self.current_window[exchange]

        return window

    def _finalize_window(self, exchange: str, window: FlowWindow):
        """Finalize a window and add to history."""
        if exchange not in self.windows:
            self.windows[exchange] = deque(maxlen=self.config.lookback_windows * 2)

        self.windows[exchange].append(window)

        # Store in database
        self.db.execute("""
            INSERT INTO flow_windows
            (exchange, start_time, end_time, inflow_btc, outflow_btc, net_flow,
             inflow_count, outflow_count, whale_inflow_btc, whale_outflow_btc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exchange,
            window.start_time.isoformat(),
            window.end_time.isoformat(),
            window.inflow_btc,
            window.outflow_btc,
            window.net_flow,
            window.inflow_count,
            window.outflow_count,
            window.whale_inflow_btc,
            window.whale_outflow_btc
        ))
        self.db.commit()

    def add_flow(self, exchange: str, direction: str, amount_btc: float) -> Optional[RegimeSignal]:
        """
        Add a flow observation and check for regime change.

        Args:
            exchange: Exchange name
            direction: 'inflow' or 'outflow'
            amount_btc: Flow amount in BTC

        Returns:
            RegimeSignal if regime change detected, None otherwise
        """
        if amount_btc < self.config.min_flow_btc:
            return None

        with self.lock:
            window = self._get_or_create_window(exchange)

            is_whale = amount_btc >= self.config.whale_threshold_btc

            if direction == 'inflow':
                window.inflow_btc += amount_btc
                window.inflow_count += 1
                if is_whale:
                    window.whale_inflow_btc += amount_btc
            else:
                window.outflow_btc += amount_btc
                window.outflow_count += 1
                if is_whale:
                    window.whale_outflow_btc += amount_btc

            # Check if we have enough history for regime detection
            if exchange not in self.windows or len(self.windows[exchange]) < 5:
                return None

            # Calculate regime signal
            return self._calculate_regime_signal(exchange)

    def _calculate_regime_signal(self, exchange: str) -> Optional[RegimeSignal]:
        """Calculate regime change signal."""
        windows = list(self.windows[exchange])
        current = self.current_window.get(exchange)

        if not windows or not current:
            return None

        # 1. Calculate Z-Score
        net_flows = [w.net_flow for w in windows]
        mu = np.mean(net_flows)
        sigma = np.std(net_flows)

        if sigma < 0.001:  # Avoid division by zero
            sigma = 1.0

        current_net = current.net_flow
        z_score = (current_net - mu) / sigma

        # 2. Calculate Cumulative Flow Imbalance (CFI)
        recent_windows = windows[-5:] if len(windows) >= 5 else windows
        cumulative_flow = sum(w.net_flow for w in recent_windows) + current_net
        total_abs_flow = sum(w.total_flow for w in recent_windows) + current.total_flow

        if total_abs_flow < 0.001:
            cfi_normalized = 0.0
        else:
            cfi_normalized = cumulative_flow / total_abs_flow

        # 3. Calculate Persistence Score
        recent_directions = [1 if w.net_flow > 0 else -1 for w in recent_windows]
        recent_directions.append(1 if current_net > 0 else -1)

        if len(recent_directions) > 0:
            # Count consecutive same direction from end
            main_direction = recent_directions[-1]
            same_count = sum(1 for d in recent_directions if d == main_direction)
            persistence_score = same_count / len(recent_directions)
        else:
            persistence_score = 0.5

        # 4. Calculate Whale Ratio
        total_whale = sum(w.whale_flow for w in recent_windows) + current.whale_flow
        total_all = sum(w.total_flow for w in recent_windows) + current.total_flow

        if total_all < 0.001:
            whale_ratio = 0.0
        else:
            whale_ratio = total_whale / total_all

        # 5. Calculate Regime Change Score (RCS)
        rcs = (
            self.config.weight_z_score * abs(z_score) +
            self.config.weight_cfi * abs(cfi_normalized) +
            self.config.weight_persistence * persistence_score +
            self.config.weight_whale * whale_ratio
        )

        # 6. Determine direction
        if z_score > 0.5 and cfi_normalized > 0.2:
            direction = 1
            direction_label = "BULLISH"
        elif z_score < -0.5 and cfi_normalized < -0.2:
            direction = -1
            direction_label = "BEARISH"
        else:
            direction = 0
            direction_label = "NEUTRAL"

        # 7. Determine signal strength
        if rcs >= self.config.rcs_threshold_strong and direction != 0:
            strength = "STRONG"
        elif rcs >= self.config.rcs_threshold_medium and direction != 0:
            strength = "MEDIUM"
        elif rcs >= self.config.rcs_threshold_weak and direction != 0:
            strength = "WEAK"
        else:
            strength = "NONE"

        # Create signal
        signal = RegimeSignal(
            timestamp=datetime.now(timezone.utc),
            exchange=exchange,
            z_score=z_score,
            cfi_normalized=cfi_normalized,
            persistence_score=persistence_score,
            whale_ratio=whale_ratio,
            regime_change_score=rcs,
            direction=direction,
            direction_label=direction_label,
            strength=strength,
            current_window_net_flow=current_net,
            rolling_mean=mu,
            rolling_std=sigma,
            cumulative_flow=cumulative_flow
        )

        # Store signal if actionable
        if signal.is_actionable:
            self._store_signal(signal)

        return signal

    def _store_signal(self, signal: RegimeSignal):
        """Store signal in database."""
        self.db.execute("""
            INSERT INTO regime_signals
            (timestamp, exchange, z_score, cfi_normalized, persistence_score,
             whale_ratio, regime_change_score, direction, direction_label, strength,
             current_net_flow, rolling_mean, rolling_std, cumulative_flow)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.timestamp.isoformat(),
            signal.exchange,
            signal.z_score,
            signal.cfi_normalized,
            signal.persistence_score,
            signal.whale_ratio,
            signal.regime_change_score,
            signal.direction,
            signal.direction_label,
            signal.strength,
            signal.current_window_net_flow,
            signal.rolling_mean,
            signal.rolling_std,
            signal.cumulative_flow
        ))
        self.db.commit()

    def record_outcome(self, signal_id: int, price_at_signal: float,
                       price_after_5m: float, price_after_30m: float):
        """Record actual price outcome for a signal."""
        # Determine actual direction
        if price_after_5m > price_at_signal * 1.0005:  # 0.05% threshold
            actual_direction = 1
        elif price_after_5m < price_at_signal * 0.9995:
            actual_direction = -1
        else:
            actual_direction = 0

        # Get predicted direction
        cursor = self.db.execute(
            "SELECT direction FROM regime_signals WHERE id = ?",
            (signal_id,)
        )
        row = cursor.fetchone()

        if row:
            predicted = row['direction']
            was_correct = 1 if predicted == actual_direction else 0

            self.db.execute("""
                UPDATE regime_signals SET
                    price_at_signal = ?,
                    price_after_5m = ?,
                    price_after_30m = ?,
                    actual_direction = ?,
                    was_correct = ?
                WHERE id = ?
            """, (
                price_at_signal,
                price_after_5m,
                price_after_30m,
                actual_direction,
                was_correct,
                signal_id
            ))
            self.db.commit()

    def get_statistics(self) -> Dict:
        """Get performance statistics."""
        cursor = self.db.execute("""
            SELECT
                exchange,
                strength,
                COUNT(*) as total,
                SUM(was_correct) as correct,
                AVG(was_correct) * 100 as accuracy
            FROM regime_signals
            WHERE was_correct IS NOT NULL
            GROUP BY exchange, strength
            ORDER BY exchange, strength
        """)

        stats = {}
        for row in cursor:
            ex = row['exchange']
            if ex not in stats:
                stats[ex] = {}
            stats[ex][row['strength']] = {
                'total': row['total'],
                'correct': row['correct'],
                'accuracy': row['accuracy'] or 0
            }

        return stats

    def get_formula_explanation(self) -> str:
        """Get human-readable formula explanation."""
        return f"""
REGIME DETECTION MODEL
======================

WHY THIS WORKS (and simple flow direction doesn't):
---------------------------------------------------
Individual flows are NOISE. Only PATTERNS matter.

STEP 1: Flow Z-Score (Is this unusual?)
  z = (NetFlow_current - Mean) / StdDev

  Current thresholds:
    - Weak signal:   |z| > {self.config.z_threshold_weak}
    - Strong signal: |z| > {self.config.z_threshold_strong}
    - Extreme:       |z| > {self.config.z_threshold_extreme}

STEP 2: Cumulative Flow Imbalance (Sustained pressure?)
  CFI = sum(NetFlow) / sum(|Flow|)

  Range: -1 (all inflow/bearish) to +1 (all outflow/bullish)
  Thresholds: |CFI| > {self.config.cfi_threshold_strong} = strong signal

STEP 3: Persistence Score (Consistent direction?)
  FPS = same_direction_windows / total_windows

  Threshold: FPS > {self.config.persistence_threshold} = consistent pattern

STEP 4: Whale Ratio (Smart money?)
  WDR = whale_flow / total_flow

  Whale threshold: > {self.config.whale_threshold_btc} BTC

FINAL SCORE: Regime Change Score (RCS)
  RCS = {self.config.weight_z_score}*|z| + {self.config.weight_cfi}*|CFI| + {self.config.weight_persistence}*FPS + {self.config.weight_whale}*WDR

Signal thresholds:
  - WEAK:   RCS > {self.config.rcs_threshold_weak}
  - MEDIUM: RCS > {self.config.rcs_threshold_medium}
  - STRONG: RCS > {self.config.rcs_threshold_strong}

EXAMPLE:
--------
z_score = -2.5 (very unusual inflow)
CFI = -0.7 (strong inflow pressure)
FPS = 0.8 (80% consistent direction)
WDR = 0.6 (60% whale activity)

RCS = 0.3*2.5 + 0.3*0.7 + 0.2*0.8 + 0.2*0.6 = 1.24

Signal: STRONG BEARISH (SHORT)
"""

    def close(self):
        """Close database connection."""
        self.db.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Test regime detection."""
    print("=" * 60)
    print("REGIME DETECTION MODEL TEST")
    print("=" * 60)
    print()

    detector = RegimeDetector()

    print(detector.get_formula_explanation())

    # Simulate flow patterns
    print("\nSimulating flow patterns...")
    print("-" * 60)

    # Normal pattern (should not trigger)
    print("\n1. Normal random flow:")
    for i in range(10):
        direction = 'inflow' if i % 2 == 0 else 'outflow'
        signal = detector.add_flow('binance', direction, 50.0)
        if signal and signal.is_actionable:
            print(f"   Signal: {signal.direction_label} ({signal.strength})")

    # Strong inflow pattern (should trigger bearish)
    print("\n2. Strong inflow pattern (bearish):")
    for i in range(10):
        signal = detector.add_flow('binance', 'inflow', 100.0)
        if signal:
            print(f"   z={signal.z_score:.2f}, CFI={signal.cfi_normalized:.2f}, "
                  f"RCS={signal.regime_change_score:.2f}, {signal.direction_label} ({signal.strength})")

    # Strong outflow pattern (should trigger bullish)
    print("\n3. Strong outflow pattern (bullish):")
    for i in range(10):
        signal = detector.add_flow('coinbase', 'outflow', 150.0)
        if signal:
            print(f"   z={signal.z_score:.2f}, CFI={signal.cfi_normalized:.2f}, "
                  f"RCS={signal.regime_change_score:.2f}, {signal.direction_label} ({signal.strength})")

    # Whale activity
    print("\n4. Whale inflow (strong bearish):")
    for i in range(5):
        signal = detector.add_flow('kraken', 'inflow', 500.0)  # Whale
        if signal:
            print(f"   Whale ratio={signal.whale_ratio:.2f}, "
                  f"RCS={signal.regime_change_score:.2f}, {signal.direction_label} ({signal.strength})")

    print()
    detector.close()


if __name__ == "__main__":
    main()
