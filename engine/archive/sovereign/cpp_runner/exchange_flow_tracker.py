#!/usr/bin/env python3
"""
PURE MATHEMATICAL LONG SIGNALS

Track per-exchange flows over time windows.
LONG when: Net outflow sustained = supply leaving = price UP

This is DETERMINISTIC:
- We measure ACTUAL BTC leaving exchanges
- Less supply = less selling pressure = price UP
- Pure math, no predictions
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import threading
import time


@dataclass
class ExchangeFlowState:
    """Track flow state per exchange."""
    exchange: str

    # Rolling windows (in BTC)
    inflow_1m: float = 0.0
    inflow_5m: float = 0.0
    inflow_15m: float = 0.0
    inflow_1h: float = 0.0

    outflow_1m: float = 0.0
    outflow_5m: float = 0.0
    outflow_15m: float = 0.0
    outflow_1h: float = 0.0

    # Timestamps of flows for rolling calculation
    inflows: List[tuple] = field(default_factory=list)   # [(timestamp, btc), ...]
    outflows: List[tuple] = field(default_factory=list)  # [(timestamp, btc), ...]

    # Current balance delta
    balance_delta_1h: float = 0.0  # Positive = accumulating, Negative = draining

    @property
    def net_flow_1m(self) -> float:
        return self.inflow_1m - self.outflow_1m

    @property
    def net_flow_5m(self) -> float:
        return self.inflow_5m - self.outflow_5m

    @property
    def net_flow_15m(self) -> float:
        return self.inflow_15m - self.outflow_15m

    @property
    def net_flow_1h(self) -> float:
        return self.inflow_1h - self.outflow_1h


@dataclass
class LongSignal:
    """Mathematical LONG signal."""
    timestamp: datetime
    exchange: str
    signal_type: str  # 'NET_OUTFLOW', 'SUPPLY_DRAIN', 'SELLER_EXHAUSTION'

    # The math
    net_outflow_1h: float      # Total BTC left exchange in 1h
    outflow_velocity: float    # BTC/minute leaving
    inflow_ratio: float        # Current inflow vs average (< 1 = exhaustion)

    # Confidence based on magnitude
    confidence: float

    reason: str


class ExchangeFlowTracker:
    """
    Pure mathematical flow tracking per exchange.

    LONG SIGNALS (deterministic):
    1. NET_OUTFLOW: Sustained net outflow > threshold = supply leaving
    2. SUPPLY_DRAIN: Balance dropping consistently = sellers gone
    3. SELLER_EXHAUSTION: Inflows << average = no new sellers
    """

    # Thresholds (pure math, will tune based on data)
    MIN_NET_OUTFLOW_1H = 50.0      # 50 BTC net outflow in 1h = significant
    MIN_OUTFLOW_VELOCITY = 1.0     # 1 BTC/min sustained
    EXHAUSTION_RATIO = 0.3         # Inflows < 30% of average = exhaustion

    # Windows
    WINDOW_1M = timedelta(minutes=1)
    WINDOW_5M = timedelta(minutes=5)
    WINDOW_15M = timedelta(minutes=15)
    WINDOW_1H = timedelta(hours=1)

    def __init__(self, db_path: str = "/root/sovereign/exchange_flows.db"):
        self.db_path = db_path
        self.states: Dict[str, ExchangeFlowState] = {}
        self.signals: List[LongSignal] = []

        # Historical averages per exchange (for exhaustion detection)
        self.avg_inflow_1h: Dict[str, float] = defaultdict(float)
        self.flow_count: Dict[str, int] = defaultdict(int)

        self._init_db()
        self._lock = threading.Lock()

    def _init_db(self):
        """Initialize flow tracking database."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS exchange_flows (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                exchange TEXT,
                direction TEXT,
                btc REAL,
                txid TEXT
            );

            CREATE TABLE IF NOT EXISTS long_signals (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                exchange TEXT,
                signal_type TEXT,
                net_outflow_1h REAL,
                outflow_velocity REAL,
                inflow_ratio REAL,
                confidence REAL,
                reason TEXT,
                price_at_signal REAL,
                price_after_5m REAL,
                pnl_pct REAL
            );

            CREATE TABLE IF NOT EXISTS exchange_snapshots (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                exchange TEXT,
                inflow_1h REAL,
                outflow_1h REAL,
                net_flow_1h REAL,
                balance_estimate REAL
            );

            CREATE INDEX IF NOT EXISTS idx_flows_exchange ON exchange_flows(exchange);
            CREATE INDEX IF NOT EXISTS idx_flows_ts ON exchange_flows(timestamp);
        """)
        conn.commit()
        conn.close()

    def record_flow(self, exchange: str, direction: str, btc: float, txid: str = ""):
        """Record a flow and update state."""
        now = datetime.now(timezone.utc)

        with self._lock:
            # Get or create state
            if exchange not in self.states:
                self.states[exchange] = ExchangeFlowState(exchange=exchange)

            state = self.states[exchange]

            # Record flow
            if direction == "INFLOW":
                state.inflows.append((now, btc))
            else:
                state.outflows.append((now, btc))

            # Update rolling windows
            self._update_windows(state, now)

            # Update historical average
            if direction == "INFLOW":
                self.flow_count[exchange] += 1
                n = self.flow_count[exchange]
                old_avg = self.avg_inflow_1h[exchange]
                self.avg_inflow_1h[exchange] = old_avg + (btc - old_avg) / n

            # Save to DB
            self._save_flow(now, exchange, direction, btc, txid)

            # Check for LONG signals
            signal = self._check_long_signal(state, now)
            if signal:
                self.signals.append(signal)
                self._save_signal(signal)
                return signal

        return None

    def _update_windows(self, state: ExchangeFlowState, now: datetime):
        """Update rolling window calculations."""
        # Clean old flows and recalculate
        cutoff_1m = now - self.WINDOW_1M
        cutoff_5m = now - self.WINDOW_5M
        cutoff_15m = now - self.WINDOW_15M
        cutoff_1h = now - self.WINDOW_1H

        # Filter and sum inflows
        state.inflows = [(t, v) for t, v in state.inflows if t > cutoff_1h]
        state.inflow_1m = sum(v for t, v in state.inflows if t > cutoff_1m)
        state.inflow_5m = sum(v for t, v in state.inflows if t > cutoff_5m)
        state.inflow_15m = sum(v for t, v in state.inflows if t > cutoff_15m)
        state.inflow_1h = sum(v for t, v in state.inflows if t > cutoff_1h)

        # Filter and sum outflows
        state.outflows = [(t, v) for t, v in state.outflows if t > cutoff_1h]
        state.outflow_1m = sum(v for t, v in state.outflows if t > cutoff_1m)
        state.outflow_5m = sum(v for t, v in state.outflows if t > cutoff_5m)
        state.outflow_15m = sum(v for t, v in state.outflows if t > cutoff_15m)
        state.outflow_1h = sum(v for t, v in state.outflows if t > cutoff_1h)

        # Balance delta
        state.balance_delta_1h = state.inflow_1h - state.outflow_1h

    def _check_long_signal(self, state: ExchangeFlowState, now: datetime) -> Optional[LongSignal]:
        """Check if current state triggers a LONG signal."""

        # Calculate metrics
        net_outflow_1h = -state.net_flow_1h  # Positive when outflow > inflow
        outflow_velocity = state.outflow_5m / 5.0  # BTC per minute

        # Inflow ratio (current vs average)
        avg = self.avg_inflow_1h.get(state.exchange, 0)
        inflow_ratio = state.inflow_1h / avg if avg > 0 else 1.0

        signal_type = None
        reason = None
        confidence = 0.0

        # Signal 1: NET_OUTFLOW - Sustained net outflow
        if net_outflow_1h >= self.MIN_NET_OUTFLOW_1H:
            signal_type = "NET_OUTFLOW"
            reason = f"{net_outflow_1h:.1f} BTC net outflow in 1h"
            confidence = min(1.0, net_outflow_1h / 100.0)  # Scale to 100 BTC

        # Signal 2: SUPPLY_DRAIN - High velocity outflow
        elif outflow_velocity >= self.MIN_OUTFLOW_VELOCITY and net_outflow_1h > 10:
            signal_type = "SUPPLY_DRAIN"
            reason = f"{outflow_velocity:.2f} BTC/min leaving, {net_outflow_1h:.1f} BTC net"
            confidence = min(1.0, outflow_velocity / 2.0)

        # Signal 3: SELLER_EXHAUSTION - Inflows dried up
        elif inflow_ratio <= self.EXHAUSTION_RATIO and state.outflow_1h > 5:
            signal_type = "SELLER_EXHAUSTION"
            reason = f"Inflows at {inflow_ratio:.0%} of average, {state.outflow_1h:.1f} BTC leaving"
            confidence = min(1.0, (1 - inflow_ratio) * 2)

        if signal_type:
            return LongSignal(
                timestamp=now,
                exchange=state.exchange,
                signal_type=signal_type,
                net_outflow_1h=net_outflow_1h,
                outflow_velocity=outflow_velocity,
                inflow_ratio=inflow_ratio,
                confidence=confidence,
                reason=reason
            )

        return None

    def _save_flow(self, ts: datetime, exchange: str, direction: str, btc: float, txid: str):
        """Save flow to database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO exchange_flows (timestamp, exchange, direction, btc, txid) VALUES (?, ?, ?, ?, ?)",
            (ts.isoformat(), exchange, direction, btc, txid)
        )
        conn.commit()
        conn.close()

    def _save_signal(self, signal: LongSignal):
        """Save signal to database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO long_signals
            (timestamp, exchange, signal_type, net_outflow_1h, outflow_velocity, inflow_ratio, confidence, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.timestamp.isoformat(),
            signal.exchange,
            signal.signal_type,
            signal.net_outflow_1h,
            signal.outflow_velocity,
            signal.inflow_ratio,
            signal.confidence,
            signal.reason
        ))
        conn.commit()
        conn.close()

    def get_state(self, exchange: str) -> Optional[ExchangeFlowState]:
        """Get current state for an exchange."""
        return self.states.get(exchange)

    def get_all_states(self) -> Dict[str, ExchangeFlowState]:
        """Get all exchange states."""
        return self.states.copy()

    def print_status(self):
        """Print current status of all exchanges."""
        print("\n" + "=" * 70)
        print("EXCHANGE FLOW STATUS (Mathematical LONG Detection)")
        print("=" * 70)
        print(f"{'Exchange':<15} {'In/1h':>10} {'Out/1h':>10} {'Net':>10} {'Signal?':<20}")
        print("-" * 70)

        for exchange, state in sorted(self.states.items(), key=lambda x: -x[1].outflow_1h):
            net = state.net_flow_1h
            net_str = f"{net:+.2f}" if net != 0 else "0.00"

            signal = ""
            if -net >= self.MIN_NET_OUTFLOW_1H:
                signal = "NET_OUTFLOW"
            elif state.outflow_5m / 5.0 >= self.MIN_OUTFLOW_VELOCITY:
                signal = "SUPPLY_DRAIN"

            print(f"{exchange:<15} {state.inflow_1h:>10.2f} {state.outflow_1h:>10.2f} {net_str:>10} {signal:<20}")

        print("=" * 70)


# Test
if __name__ == "__main__":
    tracker = ExchangeFlowTracker(db_path="test_flows.db")

    # Simulate some flows
    print("Simulating exchange flows...")

    # Coinbase: Heavy outflows (should trigger LONG)
    for i in range(10):
        tracker.record_flow("coinbase", "OUTFLOW", 8.5)
        tracker.record_flow("coinbase", "INFLOW", 2.0)

    # Binance: Heavy inflows (no signal)
    for i in range(10):
        tracker.record_flow("binance", "INFLOW", 15.0)
        tracker.record_flow("binance", "OUTFLOW", 3.0)

    tracker.print_status()

    print("\nLONG Signals generated:")
    for sig in tracker.signals:
        print(f"  {sig.exchange}: {sig.signal_type} - {sig.reason}")
