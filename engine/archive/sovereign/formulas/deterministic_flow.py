"""
Deterministic Bitcoin Flow Tracking
===================================

FORMULA IDs: 90001-90099 (Deterministic UTXO Tracking)

MATHEMATICAL FOUNDATION:
------------------------
Bitcoin uses the UTXO (Unspent Transaction Output) model.
Every UTXO is deterministic - it either exists or it doesn't.
Flow classification is BINARY, not probabilistic:

    address ∈ Exchange_Addresses → 100% certainty
    address ∉ Exchange_Addresses → Unknown

With 7.6M known exchange addresses, we achieve high coverage.
The signal derivation is purely mathematical:

    F_net = Σ(Outflow) - Σ(Inflow)
    Signal = sign(F_net)

    F_net > 0 → LONG  (accumulation - BTC leaving exchanges)
    F_net < 0 → SHORT (distribution - BTC entering exchanges)

REFERENCES:
-----------
1. Nakamoto (2008) - Bitcoin whitepaper (UTXO model)
2. ArXiv 2411.10325 - Transaction Graph Dataset
3. Meiklejohn et al. (2013) - A Fistful of Bitcoins
4. HAL-03896866 - Pattern Analysis of Money Flow
"""

import json
import os
import time
from typing import Dict, Set, Optional, Tuple, List
from dataclasses import dataclass, field
from collections import deque
from enum import IntEnum


class FlowDirection(IntEnum):
    """Deterministic flow direction."""
    SHORT = -1   # Inflow to exchange → selling pressure
    NEUTRAL = 0  # No exchange interaction or balanced
    LONG = 1     # Outflow from exchange → accumulation


@dataclass
class DeterministicFlow:
    """
    Deterministic flow measurement from a single transaction.

    This is NOT a prediction - these are exact values derived
    from the UTXO model with known exchange addresses.
    """
    txid: str
    timestamp: float

    # Exact BTC amounts (from UTXO values)
    inflow_btc: float      # BTC going TO exchange addresses
    outflow_btc: float     # BTC coming FROM exchange addresses
    net_flow_btc: float    # outflow - inflow

    # Deterministic classification
    direction: FlowDirection

    # Exchange breakdown (if identified)
    exchanges_involved: List[str] = field(default_factory=list)

    # Confidence = 1.0 when addresses are known
    # < 1.0 only when some addresses are unidentified
    confidence: float = 1.0


@dataclass
class AggregateSignal:
    """
    Aggregated flow signal over time window.

    Mathematical formula:
        F_agg = Σ(F_out) - Σ(F_in)
        Signal = sign(F_agg)
    """
    window_seconds: float
    timestamp: float

    # Aggregated flows
    total_inflow_btc: float
    total_outflow_btc: float
    net_flow_btc: float

    # Signal
    direction: FlowDirection
    signal: str  # "LONG", "SHORT", "NEUTRAL"

    # Statistics
    tx_count: int
    exchanges: Dict[str, float]  # exchange -> net flow

    # Confidence (weighted by known vs unknown addresses)
    confidence: float


class DeterministicFlowTracker:
    """
    Deterministic Bitcoin Flow Tracking Engine.

    Uses the UTXO model and known exchange addresses to provide
    mathematically certain flow classification.

    FORMULA ID: 90001
    """

    FORMULA_ID = 90001
    NAME = "Deterministic UTXO Flow (Mathematical)"

    def __init__(self,
                 exchange_addresses: Set[str] = None,
                 address_to_exchange: Dict[str, str] = None,
                 min_flow_btc: float = 1.0):
        """
        Initialize deterministic tracker.

        Args:
            exchange_addresses: Set of known exchange addresses (7.6M+)
            address_to_exchange: Mapping of address -> exchange name
            min_flow_btc: Minimum flow to generate signal
        """
        self.exchange_addresses = exchange_addresses or set()
        self.address_to_exchange = address_to_exchange or {}
        self.min_flow_btc = min_flow_btc

        # Flow history for aggregation
        self.flow_history: deque = deque(maxlen=10000)

        # Statistics
        self.total_txs = 0
        self.total_inflow = 0.0
        self.total_outflow = 0.0
        self.exchange_flows: Dict[str, Dict[str, float]] = {}

    def load_addresses(self, path: str = None):
        """
        Load exchange addresses from JSON file.

        Expected format: {"exchange_name": ["addr1", "addr2", ...], ...}
        """
        if path is None:
            # Default path
            base = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(__file__)))))
            path = os.path.join(base, "data", "exchanges.json")

        if not os.path.exists(path):
            print(f"[DETERMINISTIC] Address file not found: {path}")
            return

        print(f"[DETERMINISTIC] Loading addresses from {path}...")
        with open(path, 'r') as f:
            data = json.load(f)

        total = 0
        for exchange, addresses in data.items():
            for addr in addresses:
                self.exchange_addresses.add(addr)
                self.address_to_exchange[addr] = exchange
                total += 1

        print(f"[DETERMINISTIC] Loaded {total:,} addresses from {len(data)} exchanges")

    def load_complete_database(self):
        """
        Load from the Complete Address Database.

        This includes:
        1. Known cold wallets (verified from exchange proof-of-reserves)
        2. Existing exchanges.json (7.6M addresses)
        3. Scraped addresses from WalletExplorer
        4. Arkham Intelligence data (if API key available)

        GOAL: 100% address coverage for DETERMINISTIC flow tracking.
        """
        try:
            from engine.sovereign.blockchain.address_collector import CompleteAddressDatabase
        except ImportError:
            print("[DETERMINISTIC] CompleteAddressDatabase not available, using basic load")
            return self.load_addresses()

        print("[DETERMINISTIC] Loading Complete Address Database...")
        db = CompleteAddressDatabase()

        # Load all sources
        db.load_cold_wallets()
        db.load_existing()
        db.load_scraped()

        # Copy to our internal structures
        self.exchange_addresses = set(db.address_to_exchange.keys())
        self.address_to_exchange = db.address_to_exchange.copy()

        # Report coverage
        report = db.get_coverage_report()
        print(f"[DETERMINISTIC] Total addresses: {report['total_addresses']:,}")
        print(f"[DETERMINISTIC] Exchanges covered: {report['total_exchanges']}")

        # Show major exchange status
        print("[DETERMINISTIC] Major exchange coverage:")
        for ex, data in report['major_exchanges'].items():
            status = "OK" if data['status'] == 'COVERED' else "MISSING"
            print(f"  [{status}] {ex}: {data['count']:,} addresses")

    def classify_transaction(self, tx: Dict) -> Optional[DeterministicFlow]:
        """
        Classify a transaction using deterministic UTXO tracking.

        Mathematical formula:
            F_in = Σ {o.value : o ∈ outputs, o.address ∈ E}
            F_out = Σ {i.value : i ∈ inputs, i.address ∈ E}
            F_net = F_out - F_in
            Direction = sign(F_net)

        Args:
            tx: Transaction dict with 'inputs' and 'outputs'
                Each input/output has 'address' and 'btc' fields

        Returns:
            DeterministicFlow with exact measurements
        """
        if not self.exchange_addresses:
            return None

        inflow = 0.0
        outflow = 0.0
        exchanges = set()
        known_addresses = 0
        total_addresses = 0

        # Check outputs (potential inflows TO exchanges)
        for output in tx.get('outputs', []):
            addr = output.get('address')
            btc = output.get('btc', 0)
            total_addresses += 1

            if addr and addr in self.exchange_addresses:
                inflow += btc
                known_addresses += 1
                if addr in self.address_to_exchange:
                    exchanges.add(self.address_to_exchange[addr])

        # Check inputs (potential outflows FROM exchanges)
        for inp in tx.get('inputs', []):
            addr = inp.get('address')
            btc = inp.get('btc', 0)
            total_addresses += 1

            if addr and addr in self.exchange_addresses:
                outflow += btc
                known_addresses += 1
                if addr in self.address_to_exchange:
                    exchanges.add(self.address_to_exchange[addr])

        # No exchange interaction
        if inflow == 0 and outflow == 0:
            return None

        # Calculate net flow
        net_flow = outflow - inflow

        # Determine direction (DETERMINISTIC, not probabilistic)
        if net_flow > self.min_flow_btc:
            direction = FlowDirection.LONG
        elif net_flow < -self.min_flow_btc:
            direction = FlowDirection.SHORT
        else:
            direction = FlowDirection.NEUTRAL

        # Confidence based on address coverage
        # 1.0 = all addresses identified, < 1.0 = some unknown
        confidence = known_addresses / total_addresses if total_addresses > 0 else 0.0

        flow = DeterministicFlow(
            txid=tx.get('txid', ''),
            timestamp=time.time(),
            inflow_btc=inflow,
            outflow_btc=outflow,
            net_flow_btc=net_flow,
            direction=direction,
            exchanges_involved=list(exchanges),
            confidence=confidence,
        )

        # Track history
        self.flow_history.append(flow)
        self.total_txs += 1
        self.total_inflow += inflow
        self.total_outflow += outflow

        # Track per-exchange
        for ex in exchanges:
            if ex not in self.exchange_flows:
                self.exchange_flows[ex] = {'inflow': 0, 'outflow': 0}
            self.exchange_flows[ex]['inflow'] += inflow / max(1, len(exchanges))
            self.exchange_flows[ex]['outflow'] += outflow / max(1, len(exchanges))

        return flow

    def get_aggregate_signal(self, window_seconds: float = 300) -> Optional[AggregateSignal]:
        """
        Get aggregated signal over time window.

        Mathematical formula:
            F_agg = Σ(F_out_i) - Σ(F_in_i) for all i in window

        Args:
            window_seconds: Time window in seconds

        Returns:
            AggregateSignal with net flow direction
        """
        now = time.time()
        cutoff = now - window_seconds

        # Filter to window
        window_flows = [f for f in self.flow_history if f.timestamp >= cutoff]

        if not window_flows:
            return None

        # Aggregate
        total_in = sum(f.inflow_btc for f in window_flows)
        total_out = sum(f.outflow_btc for f in window_flows)
        net = total_out - total_in

        # Direction
        if net > self.min_flow_btc:
            direction = FlowDirection.LONG
            signal = "LONG"
        elif net < -self.min_flow_btc:
            direction = FlowDirection.SHORT
            signal = "SHORT"
        else:
            direction = FlowDirection.NEUTRAL
            signal = "NEUTRAL"

        # Per-exchange breakdown
        ex_flows = {}
        for f in window_flows:
            for ex in f.exchanges_involved:
                if ex not in ex_flows:
                    ex_flows[ex] = 0
                ex_flows[ex] += f.net_flow_btc / max(1, len(f.exchanges_involved))

        # Weighted confidence
        total_conf = sum(f.confidence * abs(f.net_flow_btc) for f in window_flows)
        total_flow = sum(abs(f.net_flow_btc) for f in window_flows)
        confidence = total_conf / total_flow if total_flow > 0 else 0.0

        return AggregateSignal(
            window_seconds=window_seconds,
            timestamp=now,
            total_inflow_btc=total_in,
            total_outflow_btc=total_out,
            net_flow_btc=net,
            direction=direction,
            signal=signal,
            tx_count=len(window_flows),
            exchanges=ex_flows,
            confidence=confidence,
        )

    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        return {
            'address_count': len(self.exchange_addresses),
            'total_txs': self.total_txs,
            'total_inflow_btc': self.total_inflow,
            'total_outflow_btc': self.total_outflow,
            'net_flow_btc': self.total_outflow - self.total_inflow,
            'exchange_count': len(self.exchange_flows),
            'recent_flows': len(self.flow_history),
        }


class NetValueCalculator:
    """
    Net Value (v_delta) Calculator from ArXiv 2411.10325.

    Formula:
        v_Δ(a) = Σ(outputs to a) - Σ(inputs from a)

        v_Δ > 0 → address receives value (accumulating)
        v_Δ < 0 → address sends value (distributing)

    FORMULA ID: 90002
    """

    FORMULA_ID = 90002
    NAME = "Net Value Delta (ArXiv 2411.10325)"

    def __init__(self):
        self.address_balances: Dict[str, float] = {}

    def process_transaction(self, tx: Dict) -> Dict[str, float]:
        """
        Calculate net value change for all addresses in transaction.

        Returns: {address: v_delta, ...}
        """
        deltas = {}

        # Subtract inputs (value leaving)
        for inp in tx.get('inputs', []):
            addr = inp.get('address')
            btc = inp.get('btc', 0)
            if addr:
                deltas[addr] = deltas.get(addr, 0) - btc

        # Add outputs (value arriving)
        for out in tx.get('outputs', []):
            addr = out.get('address')
            btc = out.get('btc', 0)
            if addr:
                deltas[addr] = deltas.get(addr, 0) + btc

        # Update running balances
        for addr, delta in deltas.items():
            self.address_balances[addr] = self.address_balances.get(addr, 0) + delta

        return deltas


class TaintFlowTracker:
    """
    Taint Flow Analysis from HAL-03896866.

    Tracks the flow of specific BTC through the transaction graph.
    Uses purity measure to determine dissolution.

    Formula:
        Purity(addr) = tainted_value / total_value
        Dissolved when Purity < threshold (e.g., 0.001)

    FORMULA ID: 90003
    """

    FORMULA_ID = 90003
    NAME = "Taint Flow (HAL-03896866)"

    def __init__(self, purity_threshold: float = 0.001):
        """
        Initialize taint tracker.

        Args:
            purity_threshold: Below this, taint is considered dissolved
        """
        self.purity_threshold = purity_threshold

        # Track tainted UTXOs: {(txid, vout): (taint_value, total_value, source)}
        self.tainted_utxos: Dict[Tuple[str, int], Tuple[float, float, str]] = {}

    def taint_utxo(self, txid: str, vout: int, value: float, source: str):
        """Mark a UTXO as tainted from a source."""
        self.tainted_utxos[(txid, vout)] = (value, value, source)

    def process_transaction(self, tx: Dict) -> List[Dict]:
        """
        Process transaction and propagate taint.

        Returns list of taint events with purity measures.
        """
        events = []

        # Check if any inputs are tainted
        tainted_input_value = 0.0
        total_input_value = 0.0
        sources = set()

        for inp in tx.get('inputs', []):
            utxo_key = (inp.get('prev_txid'), inp.get('prev_vout', 0))
            if utxo_key in self.tainted_utxos:
                taint_val, total_val, source = self.tainted_utxos[utxo_key]
                tainted_input_value += taint_val
                sources.add(source)
                del self.tainted_utxos[utxo_key]
            total_input_value += inp.get('btc', 0)

        if tainted_input_value == 0:
            return events

        # Propagate taint to outputs proportionally
        total_output_value = sum(o.get('btc', 0) for o in tx.get('outputs', []))

        for i, out in enumerate(tx.get('outputs', [])):
            out_value = out.get('btc', 0)
            if out_value == 0:
                continue

            # Proportional taint
            proportion = out_value / total_output_value if total_output_value > 0 else 0
            out_taint = tainted_input_value * proportion

            # Purity = taint / total
            purity = out_taint / out_value if out_value > 0 else 0

            if purity >= self.purity_threshold:
                # Still tainted
                self.tainted_utxos[(tx.get('txid'), i)] = (
                    out_taint, out_value, ','.join(sources)
                )

                events.append({
                    'txid': tx.get('txid'),
                    'vout': i,
                    'address': out.get('address'),
                    'taint_value': out_taint,
                    'total_value': out_value,
                    'purity': purity,
                    'sources': list(sources),
                    'dissolved': False,
                })
            else:
                # Dissolved
                events.append({
                    'txid': tx.get('txid'),
                    'vout': i,
                    'address': out.get('address'),
                    'taint_value': out_taint,
                    'total_value': out_value,
                    'purity': purity,
                    'sources': list(sources),
                    'dissolved': True,
                })

        return events


# Factory function
def create_deterministic_tracker(load_addresses: bool = True) -> DeterministicFlowTracker:
    """
    Create a deterministic flow tracker with exchange addresses loaded.

    Args:
        load_addresses: Whether to load the 7.6M address database

    Returns:
        DeterministicFlowTracker ready for use
    """
    tracker = DeterministicFlowTracker()

    if load_addresses:
        tracker.load_addresses()

    return tracker


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("DETERMINISTIC FLOW TRACKING TEST")
    print("=" * 60)

    # Create tracker (without loading full database for test)
    tracker = DeterministicFlowTracker(min_flow_btc=0.1)

    # Add some test addresses
    test_exchanges = {
        "binance": ["1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s", "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h"],
        "coinbase": ["1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ", "bc1q7cyrfmck2ffu2ud3rn5l5a8yv6f0chkp0zpemf"],
    }

    for exchange, addrs in test_exchanges.items():
        for addr in addrs:
            tracker.exchange_addresses.add(addr)
            tracker.address_to_exchange[addr] = exchange

    print(f"Loaded {len(tracker.exchange_addresses)} test addresses")

    # Simulate transactions
    test_txs = [
        {
            'txid': 'test1',
            'inputs': [{'address': 'unknown1', 'btc': 10.0}],
            'outputs': [
                {'address': '1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s', 'btc': 9.9},  # Binance
                {'address': 'change1', 'btc': 0.1}
            ]
        },
        {
            'txid': 'test2',
            'inputs': [{'address': '1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ', 'btc': 50.0}],  # Coinbase
            'outputs': [
                {'address': 'unknown2', 'btc': 49.9},
                {'address': 'change2', 'btc': 0.1}
            ]
        },
    ]

    for tx in test_txs:
        flow = tracker.classify_transaction(tx)
        if flow:
            print(f"\nTx: {flow.txid}")
            print(f"  Inflow:  {flow.inflow_btc:.2f} BTC")
            print(f"  Outflow: {flow.outflow_btc:.2f} BTC")
            print(f"  Net:     {flow.net_flow_btc:+.2f} BTC")
            print(f"  Signal:  {flow.direction.name}")
            print(f"  Exchanges: {flow.exchanges_involved}")

    # Get aggregate signal
    signal = tracker.get_aggregate_signal(window_seconds=3600)
    if signal:
        print(f"\n{'=' * 60}")
        print(f"AGGREGATE SIGNAL ({signal.window_seconds}s window)")
        print(f"{'=' * 60}")
        print(f"  Total Inflow:  {signal.total_inflow_btc:.2f} BTC")
        print(f"  Total Outflow: {signal.total_outflow_btc:.2f} BTC")
        print(f"  Net Flow:      {signal.net_flow_btc:+.2f} BTC")
        print(f"  Direction:     {signal.signal}")
        print(f"  Confidence:    {signal.confidence:.2%}")
        print(f"  Tx Count:      {signal.tx_count}")

    print(f"\n{'=' * 60}")
    print("MATHEMATICAL FOUNDATION")
    print("=" * 60)
    print("  F_in  = Sum of outputs TO exchange addresses")
    print("  F_out = Sum of inputs FROM exchange addresses")
    print("  F_net = F_out - F_in")
    print("  Signal = sign(F_net)")
    print("")
    print("  This is DETERMINISTIC, not probabilistic.")
    print("  Address membership is BINARY (in set or not).")
    print("  UTXO values are EXACT (satoshi precision).")
