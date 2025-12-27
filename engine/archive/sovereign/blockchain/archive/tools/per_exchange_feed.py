"""
Per-Exchange Blockchain Feed - Detects BTC flow TO/FROM exchanges via ZMQ.
INFLOW = SHORT (depositing to sell), OUTFLOW = LONG (withdrawing to hold).

DETECTION MODES:
1. Pattern-based detection - NO addresses needed! Uses tx structure:
   - CONSOLIDATION: 10+ inputs -> 1-3 outputs = exchange gathering
   - FAN_OUT: 1-3 inputs -> 20+ outputs = exchange distributing
   - HIGH_FREQ: Address sending 5+ txs = hot wallet
2. Address matching - Uses exchanges.json for known exchange addresses
3. Self-learning - Learns new addresses from detected patterns

Both pattern detection and address matching emit trading signals.
"""
import time
import threading
import os
from typing import Dict, Callable, Set, Optional

from .zmq_subscriber import BlockchainZMQ
from .tx_decoder import TransactionDecoder
from .exchange_wallets import ExchangeWalletTracker
from .pattern_flow_detector import PatternFlowDetector, FlowSignal, FlowPattern
from .types import FlowType, ExchangeTick


class PerExchangeBlockchainFeed:
    """
    Per-exchange blockchain feed via Bitcoin Core ZMQ.

    TRIPLE MODE DETECTION:
    1. Pattern-based detection (NO addresses needed - ALWAYS works)
    2. Address matching from exchanges.json (may be outdated)
    3. Self-learning from detected patterns (builds address database over time)

    The pattern detector can work with ZERO known addresses.
    """

    TRADING_EXCHANGES = ['coinbase', 'kraken', 'bitstamp', 'gemini', 'binance']

    # Detection thresholds
    MIN_CONSOLIDATION_INPUTS = 10   # 10+ inputs = likely exchange hot wallet
    MIN_FANOUT_OUTPUTS = 20         # 20+ outputs = exchange distributing
    MIN_SIGNAL_BTC = 0.5            # Minimum BTC to generate signal

    # INFLOW detection (deposits)
    MIN_DEPOSIT_BTC = 1.0           # High-value single output = potential deposit
    MIN_BATCH_DEPOSIT_OUTPUTS = 5   # Many small outputs to same pattern = batch deposits

    def __init__(self, on_tick: Callable[[ExchangeTick], None] = None,
                 zmq_endpoint: str = "tcp://127.0.0.1:28332",
                 json_path: str = None,
                 enable_pattern_detector: bool = True,
                 learned_addresses_path: str = None):
        """
        Initialize feed.

        Args:
            on_tick: Callback for each tick
            zmq_endpoint: Bitcoin Core ZMQ endpoint
            json_path: Path to exchanges.json (default: data/exchanges.json)
            enable_pattern_detector: Enable pattern-based detection (default: True)
            learned_addresses_path: Path to persist learned addresses
        """
        self.on_tick = on_tick
        self.zmq = BlockchainZMQ(rawtx_endpoint=zmq_endpoint, on_transaction=self._on_raw_tx)
        self.decoder = TransactionDecoder()

        # Load exchange addresses from JSON
        self.wallet_tracker = ExchangeWalletTracker(json_path=json_path)
        self.address_to_exchange = self.wallet_tracker.address_to_exchange
        self.exchange_addresses_set = self.wallet_tracker.exchange_addresses_set

        # === PATTERN-BASED DETECTOR (NO ADDRESSES NEEDED) ===
        # This is the key innovation - works even with 0 known addresses
        self.enable_pattern_detector = enable_pattern_detector
        if enable_pattern_detector:
            # Default learned path
            if learned_addresses_path is None:
                data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                learned_addresses_path = os.path.join(data_dir, "data", "learned_addresses.json")

            self.pattern_detector = PatternFlowDetector(
                known_addresses=dict(self.address_to_exchange),
                learn_addresses=True,
                learned_file=learned_addresses_path
            )
        else:
            self.pattern_detector = None

        self.reference_price = 0.0
        self.txs_processed = 0
        self.ticks_emitted = 0
        self.inflows = 0
        self.outflows = 0
        self.total_inflow_btc = 0.0
        self.total_outflow_btc = 0.0
        self.start_time = 0.0
        self.lock = threading.Lock()
        self.running = False

        # Stats for tracking coverage
        self.consolidation_detected = 0
        self.fanout_detected = 0
        self.consolidation_signals = 0
        self.address_match_signals = 0
        self.potential_deposits = 0
        self.confirmed_deposits = 0
        self.pattern_signals = 0  # NEW: pattern detector signals

        print(f"[FEED] Tracking {len(self.address_to_exchange):,} exchange addresses "
              f"(json={self.wallet_tracker.json_loaded})")
        print(f"[FEED] Pattern detector: {'ENABLED' if enable_pattern_detector else 'disabled'}")
        print(f"[FEED] Consolidation: {self.MIN_CONSOLIDATION_INPUTS}+ inputs, "
              f"Fan-out: {self.MIN_FANOUT_OUTPUTS}+ outputs")

    def start(self) -> bool:
        print("[FEED] Connecting to Bitcoin Core ZMQ...")
        self.running = True
        self.start_time = time.time()
        success = self.zmq.start()
        print("[FEED] LIVE" if success else "[FEED] Failed")
        return success

    def stop(self):
        self.running = False
        self.zmq.stop()

    def set_reference_price(self, price: float):
        self.reference_price = price

    def _on_raw_tx(self, raw_tx: bytes):
        if not self.running:
            return
        tx = self.decoder.decode(raw_tx)
        if not tx:
            return

        with self.lock:
            self.txs_processed += 1

        ts = time.time()
        txid = tx.get('txid', '')
        inputs = tx.get('inputs', [])
        outputs = tx.get('outputs', [])
        num_inputs = len(inputs)
        num_outputs = len(outputs)
        total_btc = sum(o.get('btc', 0) for o in outputs)

        # === PRIMARY: PATTERN-BASED DETECTION (NO ADDRESSES NEEDED) ===
        # This is the key innovation - works even with ZERO known addresses
        # Patterns like consolidation and fan-out are UNMISTAKABLE exchange behavior
        if self.pattern_detector:
            signal = self.pattern_detector.process_transaction(tx)
            if signal and signal.btc_amount >= self.MIN_SIGNAL_BTC:
                self._emit_pattern_signal(signal, txid, ts)
                return  # Pattern detector handled it, skip legacy detection

        # === FALLBACK: LEGACY ADDRESS MATCHING ===
        # Only runs if pattern detector is disabled or didn't match
        self._check_address_matching(inputs, outputs, txid, ts)

        # === LEGACY CONSOLIDATION DETECTION (10+ inputs) ===
        # Exchange gathering UTXOs into hot wallet
        # This is BULLISH - preparing for withdrawals
        if num_inputs >= self.MIN_CONSOLIDATION_INPUTS:
            self._handle_consolidation(txid, inputs, outputs, ts, total_btc)

        # === LEGACY FAN-OUT DETECTION (20+ outputs) ===
        # Exchange distributing to many addresses (withdrawals)
        # This is BULLISH - coins leaving exchange
        elif num_outputs >= self.MIN_FANOUT_OUTPUTS:
            self._handle_fanout(txid, inputs, outputs, ts, total_btc)

        # === LEGACY HIGH-VALUE DEPOSIT DETECTION ===
        # Large amount to 1-2 outputs = potential deposit to exchange
        # This is BEARISH - coins entering exchange to sell
        elif num_outputs <= 2 and total_btc >= self.MIN_DEPOSIT_BTC:
            self._handle_potential_deposit(txid, inputs, outputs, ts, total_btc)

    def _emit_pattern_signal(self, signal: FlowSignal, txid: str, ts: float):
        """Convert pattern detector signal to tick and emit."""
        flow_type = FlowType.OUTFLOW if signal.direction > 0 else FlowType.INFLOW
        exchange_id = signal.exchange_id or f"pattern_{signal.pattern.value}"

        self._emit_tick(exchange_id, signal.btc_amount, flow_type, txid, ts)

        with self.lock:
            self.pattern_signals += 1

        # Log significant pattern signals
        if signal.btc_amount >= 1.0:
            dir_str = "LONG" if signal.direction > 0 else "SHORT"
            print(f"[PATTERN:{signal.pattern.value.upper()}] {signal.btc_amount:.2f} BTC | "
                  f"{dir_str} | conf={signal.confidence:.2f} | {signal.reasons[0] if signal.reasons else ''}")

    def _check_address_matching(self, inputs: list, outputs: list, txid: str, ts: float):
        """Check for matches with known/discovered exchange addresses."""
        # INPUTS = OUTFLOW (money leaving exchange = LONG)
        for inp in inputs:
            addr, btc = inp.get('address', ''), inp.get('btc', 0)
            if addr and btc > 0 and addr in self.exchange_addresses_set:
                ex_id = self.address_to_exchange.get(addr, 'discovered_hot_wallet')
                self._emit_tick(ex_id, btc, FlowType.OUTFLOW, txid, ts)
                self.address_match_signals += 1

        # OUTPUTS = INFLOW (money entering exchange = SHORT)
        for out in outputs:
            addr, btc = out.get('address', ''), out.get('btc', 0)
            if addr and btc > 0 and addr in self.exchange_addresses_set:
                ex_id = self.address_to_exchange.get(addr, 'discovered_hot_wallet')
                self._emit_tick(ex_id, btc, FlowType.INFLOW, txid, ts)
                self.address_match_signals += 1

    def _handle_potential_deposit(self, txid: str, inputs: list, outputs: list, ts: float, total_btc: float):
        """
        Handle high-value transaction to few outputs.
        Could be deposit to exchange = BEARISH (they'll sell).

        NOTE: Without address matching, we emit with lower confidence.
        The formula engine should weight these appropriately.
        """
        # Get output addresses
        output_addrs = [o.get('address') for o in outputs if o.get('address')]

        # If any output goes to known exchange, it's definitely INFLOW
        for out in outputs:
            addr, btc = out.get('address', ''), out.get('btc', 0)
            if addr and addr in self.exchange_addresses_set:
                # Confirmed deposit - emit INFLOW
                ex_id = self.address_to_exchange.get(addr, 'discovered_hot_wallet')
                self._emit_tick(ex_id, btc, FlowType.INFLOW, txid, ts)
                self.confirmed_deposits += 1
                print(f"[DEPOSIT] {btc:.2f} BTC to {ex_id} | SHORT")
                return

        # No known address match - still track as potential deposit
        # High-value to 1-2 addresses is often exchange deposit
        if len(output_addrs) <= 2 and total_btc >= 2.0:
            # Large single/dual output - might be deposit, emit INFLOW
            # Use 50% weighting since we can't confirm exchange
            self._emit_tick('potential_deposit', total_btc * 0.5, FlowType.INFLOW, txid, ts)
            self.potential_deposits += 1
            if total_btc >= 5.0:
                print(f"[POTENTIAL_DEPOSIT] {total_btc:.2f} BTC | SHORT")

    def _handle_consolidation(self, txid: str, inputs: list, outputs: list, ts: float, total_btc: float):
        """
        Handle consolidation transaction.
        Consolidation = exchange gathering UTXOs = preparing for withdrawals = BULLISH
        """
        self.consolidation_detected += 1
        num_inputs = len(inputs)
        num_outputs = len(outputs)

        # Learn new addresses in memory
        input_addrs = [i.get('address') for i in inputs if i.get('address')]
        output_addrs = [o.get('address') for o in outputs if o.get('address')]
        new_addrs = 0
        for addr in input_addrs + output_addrs:
            if addr and addr not in self.exchange_addresses_set:
                self.wallet_tracker.add_address('discovered_hot_wallet', addr)
                self.exchange_addresses_set.add(addr)
                self.address_to_exchange[addr] = 'discovered_hot_wallet'
                new_addrs += 1

        # Generate trading signal if significant
        if total_btc >= self.MIN_SIGNAL_BTC:
            self.consolidation_signals += 1
            # Consolidation = OUTFLOW signal (bullish - preparing withdrawals)
            self._emit_tick('consolidation', total_btc, FlowType.OUTFLOW, txid, ts)
            print(f"[CONSOLIDATION] {num_inputs} in -> {num_outputs} out | "
                  f"{total_btc:.2f} BTC | LONG | +{new_addrs} addrs")

    def _handle_fanout(self, txid: str, inputs: list, outputs: list, ts: float, total_btc: float):
        """
        Handle fan-out transaction.
        Fan-out = exchange distributing to many addresses = withdrawals = BULLISH
        """
        self.fanout_detected += 1
        num_inputs = len(inputs)
        num_outputs = len(outputs)

        # Generate trading signal if significant
        if total_btc >= self.MIN_SIGNAL_BTC:
            self.consolidation_signals += 1
            # Fan-out = OUTFLOW signal (bullish - withdrawals happening)
            self._emit_tick('fanout', total_btc, FlowType.OUTFLOW, txid, ts)
            print(f"[FAN-OUT] {num_inputs} in -> {num_outputs} out | "
                  f"{total_btc:.2f} BTC | LONG")

    def _emit_tick(self, ex_id: str, btc: float, flow_type: FlowType, txid: str, ts: float):
        direction = 1 if flow_type == FlowType.OUTFLOW else -1
        price = self.reference_price if self.reference_price > 0 else 95000.0

        tick = ExchangeTick(
            exchange=ex_id, timestamp=ts, price=price,
            bid=price * 0.9999, ask=price * 1.0001, spread=price * 0.0002,
            volume=btc, volume_1m=btc, volume_5m=btc, volume_1h=btc,
            buy_volume=btc if direction == 1 else 0,
            sell_volume=btc if direction == -1 else 0,
            direction=direction, pressure=min(1.0, btc / 100),
            tx_count=1, source='blockchain', txid=txid, flow_type=flow_type.value,
        )

        with self.lock:
            self.ticks_emitted += 1
            if direction == 1:
                self.outflows += 1
                self.total_outflow_btc += btc
            else:
                self.inflows += 1
                self.total_inflow_btc += btc

        if btc >= 1.0:
            print(f"[{ex_id.upper()}] {btc:.2f} BTC {'OUTFLOW->LONG' if direction == 1 else 'INFLOW->SHORT'}")

        if self.on_tick:
            self.on_tick(tick)

    def get_aggregated_signal(self) -> Dict:
        with self.lock:
            net = self.total_outflow_btc - self.total_inflow_btc
            if net > 1.0:
                direction, strength = 1, min(1.0, net / 50)
            elif net < -1.0:
                direction, strength = -1, min(1.0, abs(net) / 50)
            else:
                direction, strength = 0, 0.0
            return {
                'direction': direction, 'strength': strength, 'net_flow': net,
                'total_inflow': self.total_inflow_btc, 'total_outflow': self.total_outflow_btc,
                'should_trade': direction != 0,
            }

    def get_stats(self) -> Dict:
        with self.lock:
            elapsed = time.time() - self.start_time if self.start_time > 0 else 1

            # Base stats
            stats = {
                'running': self.running,
                'txs_processed': self.txs_processed,
                'ticks_emitted': self.ticks_emitted,
                'ticks_per_min': (self.ticks_emitted / elapsed) * 60,
                'inflows': self.inflows,
                'outflows': self.outflows,
                'total_inflow_btc': self.total_inflow_btc,
                'total_outflow_btc': self.total_outflow_btc,
                'net_flow_btc': self.total_outflow_btc - self.total_inflow_btc,
                'addresses_tracked': len(self.address_to_exchange),
                'json_loaded': self.wallet_tracker.json_loaded,
                'consolidations_detected': self.consolidation_detected,
                'fanouts_detected': self.fanout_detected,
                'consolidation_signals': self.consolidation_signals,
                'address_match_signals': self.address_match_signals,
                'potential_deposits': self.potential_deposits,
                'confirmed_deposits': self.confirmed_deposits,
                'pattern_signals': self.pattern_signals,
            }

            # Add pattern detector stats if enabled
            if self.pattern_detector:
                pattern_stats = self.pattern_detector.get_stats()
                stats['pattern_detector'] = pattern_stats
                stats['addresses_learned'] = pattern_stats.get('learned_addresses', 0)
                stats['high_freq_addresses'] = pattern_stats.get('high_freq_addresses', 0)

            return stats

    def reload_addresses(self):
        """Reload addresses from JSON (call after scan completes)."""
        old_count = len(self.address_to_exchange)
        self.wallet_tracker.reload()
        self.address_to_exchange = self.wallet_tracker.address_to_exchange
        self.exchange_addresses_set = self.wallet_tracker.exchange_addresses_set
        new_count = len(self.address_to_exchange)
        print(f"[FEED] Reloaded: {old_count:,} -> {new_count:,} addresses")

        # Sync learned addresses from pattern detector
        if self.pattern_detector:
            learned = self.pattern_detector.get_all_known_addresses()
            for addr, ex_id in learned.items():
                if addr not in self.address_to_exchange:
                    self.address_to_exchange[addr] = ex_id
                    self.exchange_addresses_set.add(addr)
            print(f"[FEED] Added {len(learned)} addresses from pattern learning")

    def get_learned_addresses(self) -> Dict[str, str]:
        """Get addresses learned by pattern detector."""
        if self.pattern_detector:
            return self.pattern_detector.learned_addresses.copy()
        return {}

    def get_pattern_stats(self) -> Dict:
        """Get detailed pattern detector statistics."""
        if self.pattern_detector:
            return self.pattern_detector.get_stats()
        return {'enabled': False}
