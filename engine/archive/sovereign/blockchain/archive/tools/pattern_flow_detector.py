"""
Pattern-Based Exchange Flow Detector.
Works WITHOUT address database by detecting exchange BEHAVIOR patterns.

PATTERNS:
1. CONSOLIDATION: 10+ inputs → 1-3 outputs = Exchange gathering deposits → LONG
2. FAN-OUT: 1-3 inputs → 20+ outputs = Exchange distributing withdrawals → LONG
3. MEGA_DEPOSIT: 100+ BTC to few outputs = Whale deposit → SHORT
4. PEEL_CHAIN: Exactly 2 outputs, one small change = Potential exchange deposit

Self-learns addresses from detected patterns for future matching.
"""
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class FlowPattern(Enum):
    """Detected flow patterns."""
    CONSOLIDATION = "consolidation"      # Many→few: exchange gathering
    FAN_OUT = "fan_out"                  # Few→many: exchange distributing
    MEGA_DEPOSIT = "mega_deposit"        # Large value to few outputs
    PEEL_CHAIN = "peel_chain"            # 2 outputs, one small (change)
    HIGH_FREQ_OUT = "high_freq_out"      # Same address frequent sender
    ADDRESS_MATCH = "address_match"       # Known address detected
    UNKNOWN = "unknown"


@dataclass
class FlowSignal:
    """Detected flow signal."""
    timestamp: float
    txid: str
    pattern: FlowPattern
    direction: int              # +1 = LONG (outflow), -1 = SHORT (inflow)
    confidence: float           # 0.0 - 1.0
    btc_amount: float
    num_inputs: int
    num_outputs: int
    addresses_involved: List[str] = field(default_factory=list)
    exchange_id: Optional[str] = None
    reasons: List[str] = field(default_factory=list)


class PatternFlowDetector:
    """
    Detect exchange flows by transaction PATTERNS.
    No address database required - learns as it goes.

    CORE INSIGHT: Exchanges have distinctive tx patterns:
    - Consolidation (gathering deposits into hot wallet)
    - Fan-out (batch withdrawals to users)
    - High frequency (24/7 activity)
    """

    # Pattern thresholds
    CONSOLIDATION_MIN_INPUTS = 10       # 10+ inputs = definitely consolidation
    CONSOLIDATION_MAX_OUTPUTS = 3       # → few outputs
    FAN_OUT_MIN_OUTPUTS = 20            # 20+ outputs = batch withdrawal
    FAN_OUT_MAX_INPUTS = 3              # From few inputs
    MEGA_DEPOSIT_MIN_BTC = 100          # 100+ BTC = mega whale
    PEEL_CHANGE_MAX_RATIO = 0.1         # Change output < 10% of total

    # Learning thresholds
    HIGH_FREQ_THRESHOLD = 5             # 5+ txs = high frequency address
    LEARNING_DECAY_HOURS = 24           # Reset counts after 24h

    def __init__(self,
                 known_addresses: Dict[str, str] = None,
                 learn_addresses: bool = True,
                 learned_file: str = None):
        """
        Initialize detector.

        Args:
            known_addresses: {address: exchange_id} initial known addresses
            learn_addresses: Whether to learn new addresses from patterns
            learned_file: Path to persist learned addresses
        """
        # Known addresses (seed + learned)
        self.known_addresses: Dict[str, str] = known_addresses or {}
        self.learn_addresses = learn_addresses
        self.learned_file = learned_file

        # Address activity tracking for frequency detection
        self.address_tx_count: Dict[str, int] = defaultdict(int)
        self.address_last_seen: Dict[str, float] = {}
        self.address_total_btc: Dict[str, float] = defaultdict(float)

        # Learned addresses from pattern detection
        self.learned_addresses: Dict[str, str] = {}  # addr → pattern that learned it

        # Consolidation output tracking (these become hot wallet candidates)
        self.consolidation_outputs: Set[str] = set()

        # Statistics
        self.stats = {
            'txs_processed': 0,
            'patterns_detected': defaultdict(int),
            'signals_long': 0,
            'signals_short': 0,
            'addresses_learned': 0,
            'total_btc_flow': 0.0,
        }

        # Load learned addresses if file exists
        self._load_learned()

        print(f"[PATTERN] Initialized with {len(self.known_addresses)} known addresses")
        print(f"[PATTERN] Learning enabled: {learn_addresses}")

    def _load_learned(self):
        """Load previously learned addresses."""
        if not self.learned_file or not os.path.exists(self.learned_file):
            return

        try:
            with open(self.learned_file) as f:
                data = json.load(f)
                self.learned_addresses = data.get('addresses', {})
                self.consolidation_outputs = set(data.get('consolidation_outputs', []))
                print(f"[PATTERN] Loaded {len(self.learned_addresses)} learned addresses")
        except Exception as e:
            print(f"[PATTERN] Failed to load learned: {e}")

    def _save_learned(self):
        """Persist learned addresses."""
        if not self.learned_file:
            return

        try:
            data = {
                'addresses': self.learned_addresses,
                'consolidation_outputs': list(self.consolidation_outputs),
                'updated': time.time()
            }
            with open(self.learned_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[PATTERN] Failed to save learned: {e}")

    def process_transaction(self, tx: Dict) -> Optional[FlowSignal]:
        """
        Process a transaction and detect flow patterns.

        Args:
            tx: Decoded transaction dict with inputs/outputs

        Returns:
            FlowSignal if pattern detected, None otherwise
        """
        self.stats['txs_processed'] += 1

        txid = tx.get('txid', '')
        inputs = tx.get('inputs', [])
        outputs = tx.get('outputs', [])
        total_btc = tx.get('total_btc', 0)

        num_inputs = len(inputs)
        num_outputs = len(outputs)

        # Extract addresses
        input_addrs = [i.get('address') for i in inputs if i.get('address')]
        output_addrs = [o.get('address') for o in outputs if o.get('address')]
        all_addrs = input_addrs + output_addrs

        # Update address activity tracking
        self._update_address_tracking(input_addrs, output_addrs, total_btc)

        # Try each pattern detector in priority order
        signal = None

        # 1. Check known addresses first (highest confidence)
        signal = self._check_known_addresses(txid, inputs, outputs, total_btc)
        if signal:
            return signal

        # 2. Check consolidation pattern
        signal = self._check_consolidation(txid, inputs, outputs, num_inputs,
                                           num_outputs, total_btc, output_addrs)
        if signal:
            return signal

        # 3. Check fan-out pattern
        signal = self._check_fan_out(txid, inputs, outputs, num_inputs,
                                     num_outputs, total_btc, input_addrs)
        if signal:
            return signal

        # 4. Check mega deposit
        signal = self._check_mega_deposit(txid, inputs, outputs, num_inputs,
                                          num_outputs, total_btc, output_addrs)
        if signal:
            return signal

        # 5. Check high-frequency senders (learned hot wallets)
        signal = self._check_high_frequency(txid, inputs, outputs, total_btc, input_addrs)
        if signal:
            return signal

        return None

    def _update_address_tracking(self, input_addrs: List[str],
                                  output_addrs: List[str], btc: float):
        """Update address activity tracking for frequency detection."""
        now = time.time()

        for addr in input_addrs:
            if addr:
                # Decay old counts
                if addr in self.address_last_seen:
                    age_hours = (now - self.address_last_seen[addr]) / 3600
                    if age_hours > self.LEARNING_DECAY_HOURS:
                        self.address_tx_count[addr] = 0

                self.address_tx_count[addr] += 1
                self.address_last_seen[addr] = now
                self.address_total_btc[addr] += btc

    def _check_known_addresses(self, txid: str, inputs: List, outputs: List,
                                total_btc: float) -> Optional[FlowSignal]:
        """Check if transaction involves known exchange addresses."""
        inflow_exchanges = []
        outflow_exchanges = []

        # Check outputs for inflows (TO exchange = SHORT)
        for out in outputs:
            addr = out.get('address')
            if addr in self.known_addresses:
                inflow_exchanges.append(self.known_addresses[addr])
            elif addr in self.learned_addresses:
                inflow_exchanges.append(f"learned:{self.learned_addresses[addr]}")
            elif addr in self.consolidation_outputs:
                inflow_exchanges.append("consolidation_output")

        # Check inputs for outflows (FROM exchange = LONG)
        for inp in inputs:
            addr = inp.get('address')
            if addr in self.known_addresses:
                outflow_exchanges.append(self.known_addresses[addr])
            elif addr in self.learned_addresses:
                outflow_exchanges.append(f"learned:{self.learned_addresses[addr]}")
            elif addr in self.consolidation_outputs:
                outflow_exchanges.append("consolidation_output")

        if outflow_exchanges:
            self.stats['patterns_detected']['address_match'] += 1
            self.stats['signals_long'] += 1
            self.stats['total_btc_flow'] += total_btc

            return FlowSignal(
                timestamp=time.time(),
                txid=txid,
                pattern=FlowPattern.ADDRESS_MATCH,
                direction=+1,  # LONG
                confidence=0.9,
                btc_amount=total_btc,
                num_inputs=len(inputs),
                num_outputs=len(outputs),
                exchange_id=outflow_exchanges[0],
                reasons=[f"Outflow from {outflow_exchanges}"]
            )

        if inflow_exchanges:
            self.stats['patterns_detected']['address_match'] += 1
            self.stats['signals_short'] += 1
            self.stats['total_btc_flow'] += total_btc

            return FlowSignal(
                timestamp=time.time(),
                txid=txid,
                pattern=FlowPattern.ADDRESS_MATCH,
                direction=-1,  # SHORT
                confidence=0.9,
                btc_amount=total_btc,
                num_inputs=len(inputs),
                num_outputs=len(outputs),
                exchange_id=inflow_exchanges[0],
                reasons=[f"Inflow to {inflow_exchanges}"]
            )

        return None

    def _check_consolidation(self, txid: str, inputs: List, outputs: List,
                              num_inputs: int, num_outputs: int, total_btc: float,
                              output_addrs: List[str]) -> Optional[FlowSignal]:
        """
        Detect consolidation pattern.

        CONSOLIDATION: 10+ inputs → 1-3 outputs
        = Exchange gathering user deposits into hot wallet
        = Outflow signal (gathering before distribution) → LONG

        High confidence: This pattern is VERY distinctive of exchanges.
        Regular users don't consolidate 10+ UTXOs.
        """
        if num_inputs >= self.CONSOLIDATION_MIN_INPUTS and \
           num_outputs <= self.CONSOLIDATION_MAX_OUTPUTS:

            # Confidence scales with number of inputs
            confidence = min(0.85, 0.5 + (num_inputs - 10) * 0.02)

            # Learn the output addresses (these are hot wallet candidates)
            if self.learn_addresses:
                for addr in output_addrs:
                    if addr and addr not in self.known_addresses:
                        self.consolidation_outputs.add(addr)
                        self.learned_addresses[addr] = 'consolidation'
                        self.stats['addresses_learned'] += 1

                # Save periodically
                if self.stats['addresses_learned'] % 10 == 0:
                    self._save_learned()

            self.stats['patterns_detected']['consolidation'] += 1
            self.stats['signals_long'] += 1
            self.stats['total_btc_flow'] += total_btc

            return FlowSignal(
                timestamp=time.time(),
                txid=txid,
                pattern=FlowPattern.CONSOLIDATION,
                direction=+1,  # LONG - exchange gathering = bullish
                confidence=confidence,
                btc_amount=total_btc,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                addresses_involved=output_addrs[:3],
                reasons=[f"{num_inputs} inputs consolidated to {num_outputs} outputs"]
            )

        return None

    def _check_fan_out(self, txid: str, inputs: List, outputs: List,
                       num_inputs: int, num_outputs: int, total_btc: float,
                       input_addrs: List[str]) -> Optional[FlowSignal]:
        """
        Detect fan-out pattern.

        FAN-OUT: 1-3 inputs → 20+ outputs
        = Exchange batch withdrawal to users
        = Outflow (leaving exchange) → LONG

        Very high confidence: Regular users don't send to 20+ addresses.
        """
        if num_inputs <= self.FAN_OUT_MAX_INPUTS and \
           num_outputs >= self.FAN_OUT_MIN_OUTPUTS:

            # Confidence scales with number of outputs
            confidence = min(0.90, 0.6 + (num_outputs - 20) * 0.01)

            # Learn the input addresses (these are hot wallets)
            if self.learn_addresses:
                for addr in input_addrs:
                    if addr and addr not in self.known_addresses:
                        self.learned_addresses[addr] = 'fan_out'
                        self.stats['addresses_learned'] += 1

            self.stats['patterns_detected']['fan_out'] += 1
            self.stats['signals_long'] += 1
            self.stats['total_btc_flow'] += total_btc

            return FlowSignal(
                timestamp=time.time(),
                txid=txid,
                pattern=FlowPattern.FAN_OUT,
                direction=+1,  # LONG - mass withdrawal = bullish
                confidence=confidence,
                btc_amount=total_btc,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                addresses_involved=input_addrs,
                reasons=[f"Fan-out: {num_inputs} inputs to {num_outputs} outputs"]
            )

        return None

    def _check_mega_deposit(self, txid: str, inputs: List, outputs: List,
                            num_inputs: int, num_outputs: int, total_btc: float,
                            output_addrs: List[str]) -> Optional[FlowSignal]:
        """
        Detect mega deposit pattern.

        MEGA_DEPOSIT: 100+ BTC to 1-3 outputs
        = Whale moving to exchange (potential sell pressure)
        = Inflow → SHORT

        This is what Whale Alert tweets about!
        """
        if total_btc >= self.MEGA_DEPOSIT_MIN_BTC and num_outputs <= 3:
            # Higher confidence for larger amounts
            confidence = min(0.80, 0.5 + (total_btc / 1000) * 0.1)

            self.stats['patterns_detected']['mega_deposit'] += 1
            self.stats['signals_short'] += 1
            self.stats['total_btc_flow'] += total_btc

            return FlowSignal(
                timestamp=time.time(),
                txid=txid,
                pattern=FlowPattern.MEGA_DEPOSIT,
                direction=-1,  # SHORT - whale deposit = bearish
                confidence=confidence,
                btc_amount=total_btc,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                addresses_involved=output_addrs,
                reasons=[f"Mega deposit: {total_btc:.2f} BTC to {num_outputs} outputs"]
            )

        return None

    def _check_high_frequency(self, txid: str, inputs: List, outputs: List,
                               total_btc: float, input_addrs: List[str]) -> Optional[FlowSignal]:
        """
        Detect high-frequency sender (learned hot wallet).

        If an address sends 5+ transactions, it's likely an exchange hot wallet.
        """
        for addr in input_addrs:
            if addr and self.address_tx_count.get(addr, 0) >= self.HIGH_FREQ_THRESHOLD:
                # This address sends frequently - likely exchange
                if addr not in self.known_addresses and self.learn_addresses:
                    self.learned_addresses[addr] = 'high_freq'
                    self.stats['addresses_learned'] += 1

                self.stats['patterns_detected']['high_freq'] += 1
                self.stats['signals_long'] += 1
                self.stats['total_btc_flow'] += total_btc

                return FlowSignal(
                    timestamp=time.time(),
                    txid=txid,
                    pattern=FlowPattern.HIGH_FREQ_OUT,
                    direction=+1,  # LONG - exchange sending out
                    confidence=0.65,  # Lower confidence for frequency-based
                    btc_amount=total_btc,
                    num_inputs=len(inputs),
                    num_outputs=len(outputs),
                    addresses_involved=[addr],
                    reasons=[f"High-freq sender: {self.address_tx_count[addr]} txs from {addr[:16]}..."]
                )

        return None

    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            'txs_processed': self.stats['txs_processed'],
            'patterns': dict(self.stats['patterns_detected']),
            'signals_long': self.stats['signals_long'],
            'signals_short': self.stats['signals_short'],
            'total_btc_flow': self.stats['total_btc_flow'],
            'known_addresses': len(self.known_addresses),
            'learned_addresses': len(self.learned_addresses),
            'consolidation_outputs': len(self.consolidation_outputs),
            'high_freq_addresses': sum(1 for c in self.address_tx_count.values()
                                       if c >= self.HIGH_FREQ_THRESHOLD)
        }

    def get_all_known_addresses(self) -> Dict[str, str]:
        """Get all known addresses (seed + learned)."""
        all_addrs = dict(self.known_addresses)
        all_addrs.update(self.learned_addresses)
        return all_addrs


class HybridFlowDetector:
    """
    Combined detector: Pattern detection + address matching.

    Uses patterns for immediate detection, learns addresses over time
    to improve accuracy.
    """

    def __init__(self,
                 known_addresses: Dict[str, str] = None,
                 learned_file: str = None,
                 min_btc_for_signal: float = 10.0):
        """
        Initialize hybrid detector.

        Args:
            known_addresses: Initial known exchange addresses
            learned_file: Path to persist learned addresses
            min_btc_for_signal: Minimum BTC for signal generation
        """
        self.pattern_detector = PatternFlowDetector(
            known_addresses=known_addresses,
            learn_addresses=True,
            learned_file=learned_file
        )
        self.min_btc = min_btc_for_signal

        # Signal aggregation
        self.recent_signals: List[FlowSignal] = []
        self.signal_window_sec = 60  # Aggregate signals within 60s

        print(f"[HYBRID] Initialized, min_btc={min_btc_for_signal}")

    def process_transaction(self, tx: Dict) -> Optional[FlowSignal]:
        """Process transaction through hybrid detection."""
        signal = self.pattern_detector.process_transaction(tx)

        if signal and signal.btc_amount >= self.min_btc:
            # Clean old signals
            now = time.time()
            self.recent_signals = [s for s in self.recent_signals
                                   if now - s.timestamp < self.signal_window_sec]
            self.recent_signals.append(signal)

            return signal

        return None

    def get_net_flow(self) -> Tuple[float, float, float]:
        """Get net flow from recent signals."""
        now = time.time()
        recent = [s for s in self.recent_signals
                  if now - s.timestamp < self.signal_window_sec]

        long_btc = sum(s.btc_amount for s in recent if s.direction > 0)
        short_btc = sum(s.btc_amount for s in recent if s.direction < 0)
        net = long_btc - short_btc

        return long_btc, short_btc, net

    def get_stats(self) -> Dict:
        """Get combined statistics."""
        stats = self.pattern_detector.get_stats()
        long_btc, short_btc, net = self.get_net_flow()
        stats['recent_long_btc'] = long_btc
        stats['recent_short_btc'] = short_btc
        stats['recent_net_flow'] = net
        return stats


# Quick test
if __name__ == '__main__':
    print("Pattern Flow Detector Test")
    print("=" * 50)

    detector = PatternFlowDetector()

    # Simulate consolidation tx
    consolidation_tx = {
        'txid': 'test_consolidation',
        'inputs': [{'address': f'1Input{i}'} for i in range(15)],
        'outputs': [{'address': '1HotWallet', 'btc': 10.5}],
        'total_btc': 10.5
    }

    signal = detector.process_transaction(consolidation_tx)
    if signal:
        print(f"Pattern: {signal.pattern.value}")
        print(f"Direction: {'LONG' if signal.direction > 0 else 'SHORT'}")
        print(f"Confidence: {signal.confidence:.2f}")
        print(f"BTC: {signal.btc_amount:.2f}")
        print(f"Reasons: {signal.reasons}")

    print()

    # Simulate fan-out tx
    fanout_tx = {
        'txid': 'test_fanout',
        'inputs': [{'address': '1ExchangeHot'}],
        'outputs': [{'address': f'1User{i}', 'btc': 0.5} for i in range(30)],
        'total_btc': 15.0
    }

    signal = detector.process_transaction(fanout_tx)
    if signal:
        print(f"Pattern: {signal.pattern.value}")
        print(f"Direction: {'LONG' if signal.direction > 0 else 'SHORT'}")
        print(f"Confidence: {signal.confidence:.2f}")
        print(f"BTC: {signal.btc_amount:.2f}")

    print()
    print("Stats:", detector.get_stats())
