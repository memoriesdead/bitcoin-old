"""
Whale Transaction Sequence Analysis
===================================

Formula IDs: 72076-72080

Analyzes patterns in large Bitcoin transactions (whale movements).
What happens after whales buy or sell?

RenTech insight: Large players leave footprints. Track them.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import deque


@dataclass
class FlowSequence:
    """Sequence of flow events."""
    events: List[str]  # 'inflow_large', 'outflow_large', etc.
    timestamps: List[float]
    magnitudes: List[float]


@dataclass
class LargeTransactionPattern:
    """Pattern in large transactions."""
    pattern_type: str
    avg_return: float
    win_rate: float
    sample_count: int
    significance: float


@dataclass
class WhaleSignal:
    """Signal from whale analysis."""
    direction: int
    confidence: float
    whale_activity: str  # 'accumulating', 'distributing', 'neutral'
    flow_imbalance: float
    recent_events: List[str]


class WhaleTracker:
    """
    Tracks large transactions and derives signals.
    """

    def __init__(self, large_threshold: float = 100.0):  # BTC
        self.large_threshold = large_threshold
        self.event_history: deque = deque(maxlen=100)
        self.pattern_stats: Dict[str, LargeTransactionPattern] = {}

    def classify_flow(self, flow_value: float, is_inflow: bool) -> str:
        """Classify a flow event."""
        abs_value = abs(flow_value)

        if abs_value < self.large_threshold:
            size = 'small'
        elif abs_value < self.large_threshold * 10:
            size = 'medium'
        else:
            size = 'large'

        direction = 'inflow' if is_inflow else 'outflow'
        return f'{direction}_{size}'

    def add_event(self, flow_value: float, is_inflow: bool, timestamp: float):
        """Add a flow event to history."""
        event_type = self.classify_flow(flow_value, is_inflow)
        self.event_history.append({
            'type': event_type,
            'value': flow_value,
            'timestamp': timestamp,
        })

    def build_statistics(self, flows: np.ndarray, inflows: np.ndarray,
                        outflows: np.ndarray, returns: np.ndarray,
                        min_samples: int = 20):
        """Build pattern statistics from historical data."""
        n = len(flows)

        # Track sequences of large flows
        sequences: Dict[str, List[float]] = {}

        for i in range(1, n - 1):
            # Check for large flow events
            if abs(inflows[i]) > self.large_threshold:
                event = 'large_inflow'
            elif abs(outflows[i]) > self.large_threshold:
                event = 'large_outflow'
            else:
                continue

            # Look at surrounding context
            prev_flow = flows[i - 1]
            next_return = returns[i + 1] if i + 1 < n else 0

            # Pattern: what happened before -> what happened -> what follows
            if prev_flow > 0:
                context = f'after_net_inflow_{event}'
            elif prev_flow < 0:
                context = f'after_net_outflow_{event}'
            else:
                context = f'neutral_{event}'

            if context not in sequences:
                sequences[context] = []
            sequences[context].append(next_return)

        # Compute statistics
        for pattern, rets in sequences.items():
            if len(rets) >= min_samples:
                avg = np.mean(rets)
                std = np.std(rets)
                t_stat = avg / (std / np.sqrt(len(rets)) + 1e-10)

                self.pattern_stats[pattern] = LargeTransactionPattern(
                    pattern_type=pattern,
                    avg_return=avg,
                    win_rate=sum(1 for r in rets if r > 0) / len(rets),
                    sample_count=len(rets),
                    significance=abs(t_stat),
                )

    def get_current_activity(self, lookback: int = 10) -> str:
        """Determine current whale activity from recent events."""
        if len(self.event_history) < lookback:
            return 'neutral'

        recent = list(self.event_history)[-lookback:]

        inflow_count = sum(1 for e in recent if 'inflow' in e['type'] and 'large' in e['type'])
        outflow_count = sum(1 for e in recent if 'outflow' in e['type'] and 'large' in e['type'])

        if inflow_count > outflow_count + 2:
            return 'accumulating'
        elif outflow_count > inflow_count + 2:
            return 'distributing'
        else:
            return 'neutral'

    def get_flow_imbalance(self, lookback: int = 10) -> float:
        """Get flow imbalance from recent events."""
        if len(self.event_history) < lookback:
            return 0.0

        recent = list(self.event_history)[-lookback:]

        total_inflow = sum(e['value'] for e in recent if 'inflow' in e['type'])
        total_outflow = sum(e['value'] for e in recent if 'outflow' in e['type'])

        total = total_inflow + total_outflow
        if total == 0:
            return 0.0

        return (total_inflow - total_outflow) / total


# =============================================================================
# FORMULA IMPLEMENTATIONS (72076-72080)
# =============================================================================

class WhaleAccumSignal:
    """
    Formula 72076: Whale Accumulation Signal

    Signals when large holders are accumulating (outflow from exchanges).
    Exchange outflow = buyers taking custody = bullish.
    """

    FORMULA_ID = 72076

    def __init__(self, threshold: float = 100.0):
        self.tracker = WhaleTracker(large_threshold=threshold)
        self.is_fitted = False

    def fit(self, flows: np.ndarray, inflows: np.ndarray, outflows: np.ndarray, returns: np.ndarray):
        self.tracker.build_statistics(flows, inflows, outflows, returns)
        self.is_fitted = True

    def update(self, net_flow: float, is_inflow: bool, timestamp: float):
        """Update with new flow data."""
        self.tracker.add_event(net_flow, is_inflow, timestamp)

    def generate_signal(self, recent_outflow: float) -> WhaleSignal:
        activity = self.tracker.get_current_activity()
        imbalance = self.tracker.get_flow_imbalance()

        if activity == 'accumulating' or recent_outflow > self.tracker.large_threshold:
            # Whales accumulating - bullish
            direction = 1
            confidence = min(1.0, abs(imbalance))
        else:
            direction = 0
            confidence = 0.0

        recent_events = [e['type'] for e in list(self.tracker.event_history)[-5:]]

        return WhaleSignal(
            direction=direction,
            confidence=confidence,
            whale_activity=activity,
            flow_imbalance=imbalance,
            recent_events=recent_events,
        )


class WhaleDistribSignal:
    """
    Formula 72077: Whale Distribution Signal

    Signals when large holders are distributing (inflow to exchanges).
    Exchange inflow = sellers preparing to dump = bearish.
    """

    FORMULA_ID = 72077

    def __init__(self, threshold: float = 100.0):
        self.tracker = WhaleTracker(large_threshold=threshold)

    def fit(self, flows: np.ndarray, inflows: np.ndarray, outflows: np.ndarray, returns: np.ndarray):
        self.tracker.build_statistics(flows, inflows, outflows, returns)

    def update(self, net_flow: float, is_inflow: bool, timestamp: float):
        self.tracker.add_event(net_flow, is_inflow, timestamp)

    def generate_signal(self, recent_inflow: float) -> WhaleSignal:
        activity = self.tracker.get_current_activity()
        imbalance = self.tracker.get_flow_imbalance()

        if activity == 'distributing' or recent_inflow > self.tracker.large_threshold:
            # Whales distributing - bearish
            direction = -1
            confidence = min(1.0, abs(imbalance))
        else:
            direction = 0
            confidence = 0.0

        recent_events = [e['type'] for e in list(self.tracker.event_history)[-5:]]

        return WhaleSignal(
            direction=direction,
            confidence=confidence,
            whale_activity=activity,
            flow_imbalance=imbalance,
            recent_events=recent_events,
        )


class WhaleSequenceSignal:
    """
    Formula 72078: Whale Sequence Signal

    Analyzes sequences of whale transactions.
    E.g., large outflow -> large outflow -> ? (likely continuation)
    """

    FORMULA_ID = 72078

    def __init__(self, sequence_length: int = 3):
        self.sequence_length = sequence_length
        self.tracker = WhaleTracker()
        self.sequence_stats: Dict[str, LargeTransactionPattern] = {}

    def fit(self, flows: np.ndarray, inflows: np.ndarray, outflows: np.ndarray, returns: np.ndarray):
        """Build sequence statistics."""
        n = len(flows)
        sequences: Dict[str, List[float]] = {}

        for i in range(self.sequence_length, n - 1):
            # Build sequence string
            seq_parts = []
            for j in range(self.sequence_length):
                idx = i - self.sequence_length + j
                if abs(flows[idx]) > self.tracker.large_threshold:
                    seq_parts.append('L' if flows[idx] > 0 else 'S')  # Large in/out
                else:
                    seq_parts.append('N')  # Normal

            seq_key = ''.join(seq_parts)

            if seq_key not in sequences:
                sequences[seq_key] = []
            sequences[seq_key].append(returns[i + 1] if i + 1 < n else 0)

        # Compute statistics
        for seq_key, rets in sequences.items():
            if len(rets) >= 15:
                avg = np.mean(rets)
                std = np.std(rets)
                t_stat = avg / (std / np.sqrt(len(rets)) + 1e-10)

                self.sequence_stats[seq_key] = LargeTransactionPattern(
                    pattern_type=seq_key,
                    avg_return=avg,
                    win_rate=sum(1 for r in rets if r > 0) / len(rets),
                    sample_count=len(rets),
                    significance=abs(t_stat),
                )

    def generate_signal(self, recent_flows: np.ndarray) -> WhaleSignal:
        if len(recent_flows) < self.sequence_length:
            return WhaleSignal(0, 0.0, 'insufficient_data', 0.0, [])

        # Build current sequence
        seq_parts = []
        for f in recent_flows[-self.sequence_length:]:
            if abs(f) > self.tracker.large_threshold:
                seq_parts.append('L' if f > 0 else 'S')
            else:
                seq_parts.append('N')

        seq_key = ''.join(seq_parts)
        pattern = self.sequence_stats.get(seq_key)

        if pattern is None or pattern.significance < 2.0:
            return WhaleSignal(0, 0.0, seq_key, 0.0, seq_parts)

        direction = 1 if pattern.avg_return > 0 else -1
        confidence = min(1.0, (pattern.significance - 2.0) / 2.0)

        return WhaleSignal(
            direction=direction,
            confidence=confidence,
            whale_activity=seq_key,
            flow_imbalance=0.0,
            recent_events=seq_parts,
        )


class FlowMomentumSignal:
    """
    Formula 72079: Flow Momentum Signal

    Trades momentum in exchange flows.
    Increasing outflows = accelerating accumulation.
    """

    FORMULA_ID = 72079

    def __init__(self, lookback: int = 7):
        self.lookback = lookback
        self.flow_history: List[float] = []

    def generate_signal(self, current_flow: float) -> WhaleSignal:
        self.flow_history.append(current_flow)

        if len(self.flow_history) > 100:
            self.flow_history = self.flow_history[-100:]

        if len(self.flow_history) < self.lookback * 2:
            return WhaleSignal(0, 0.0, 'building_history', 0.0, [])

        # Flow momentum = recent avg vs older avg
        recent_avg = np.mean(self.flow_history[-self.lookback:])
        older_avg = np.mean(self.flow_history[-self.lookback * 2:-self.lookback])

        momentum = recent_avg - older_avg

        if momentum < -50:  # Increasing outflows (bullish)
            direction = 1
            confidence = min(1.0, abs(momentum) / 100)
            activity = 'accelerating_accumulation'
        elif momentum > 50:  # Increasing inflows (bearish)
            direction = -1
            confidence = min(1.0, abs(momentum) / 100)
            activity = 'accelerating_distribution'
        else:
            direction = 0
            confidence = 0.0
            activity = 'neutral_momentum'

        return WhaleSignal(
            direction=direction,
            confidence=confidence,
            whale_activity=activity,
            flow_imbalance=momentum,
            recent_events=[],
        )


class WhaleEnsembleSignal:
    """
    Formula 72080: Whale Ensemble Signal

    Combines all whale-based signals.
    """

    FORMULA_ID = 72080

    def __init__(self):
        self.accum_signal = WhaleAccumSignal()
        self.distrib_signal = WhaleDistribSignal()
        self.flow_momentum = FlowMomentumSignal()

    def fit(self, flows: np.ndarray, inflows: np.ndarray, outflows: np.ndarray, returns: np.ndarray):
        self.accum_signal.fit(flows, inflows, outflows, returns)
        self.distrib_signal.fit(flows, inflows, outflows, returns)

    def generate_signal(self, current_flow: float, inflow: float, outflow: float) -> WhaleSignal:
        results = []

        # Accumulation signal
        self.accum_signal.update(abs(outflow), False, 0)
        results.append(self.accum_signal.generate_signal(outflow))

        # Distribution signal
        self.distrib_signal.update(abs(inflow), True, 0)
        results.append(self.distrib_signal.generate_signal(inflow))

        # Flow momentum
        results.append(self.flow_momentum.generate_signal(current_flow))

        # Combine
        total_dir = sum(r.direction * r.confidence for r in results)
        total_conf = sum(r.confidence for r in results)

        if total_conf > 0:
            avg_dir = total_dir / total_conf
            direction = 1 if avg_dir > 0.3 else (-1 if avg_dir < -0.3 else 0)
            confidence = total_conf / len(results)
        else:
            direction = 0
            confidence = 0.0

        # Aggregate activity
        activities = [r.whale_activity for r in results]
        if 'accumulating' in activities or 'accelerating_accumulation' in activities:
            activity = 'net_accumulating'
        elif 'distributing' in activities or 'accelerating_distribution' in activities:
            activity = 'net_distributing'
        else:
            activity = 'neutral'

        return WhaleSignal(
            direction=direction,
            confidence=confidence,
            whale_activity=activity,
            flow_imbalance=np.mean([r.flow_imbalance for r in results]),
            recent_events=[],
        )
