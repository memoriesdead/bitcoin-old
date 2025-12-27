"""
FORMULA CONNECTOR - Connects live blockchain data to trading formulas.

This is the MISSING LINK between:
    - PerExchangeBlockchainFeed (raw blockchain ticks)
    - AdaptiveTradingEngine (trading formulas)
    - PatternRecognitionEngine (HMM, stat arb, ML pattern detection)
    - RenTechPatternEngine (advanced patterns 72001-72099)

FLOW:
    Bitcoin Core ZMQ --> PerExchangeBlockchainFeed --> FormulaConnector
                                                             |
                              +------------------------------+------------------------------+
                              |                              |                              |
                  AdaptiveTradingEngine      PatternRecognitionEngine      RenTechPatternEngine
                  (IDs 10001-10005)          (IDs 20001-20012)             (IDs 72001-72099)
                              |                              |                              |
                              +------------------------------+------------------------------+
                                                             |
                                                    ENSEMBLE SIGNAL
                                                    (LONG/SHORT/NEUTRAL)

USAGE:
    from engine.sovereign.blockchain import FormulaConnector

    connector = FormulaConnector()
    connector.start()

    # Check for signals periodically
    signal = connector.get_signal()
    if signal and signal['direction'] != 0:
        execute_trade(signal)
"""
import time
import threading
from typing import Dict, Optional, Callable, List, TYPE_CHECKING

from .per_exchange_feed import PerExchangeBlockchainFeed
from .types import ExchangeTick, FlowType
from ..formulas.adaptive import AdaptiveTradingEngine
from ..formulas.pattern_recognition import PatternRecognitionEngine
from ..formulas.rentech_engine import RenTechPatternEngine, RenTechSignal, SignalDirection, create_rentech_engine

if TYPE_CHECKING:
    from ..ai.claude_adapter import ClaudeAdapter


class FormulaConnector:
    """
    Connects blockchain feed to ALL trading formulas.

    FORMULA ENGINES:
    1. AdaptiveTradingEngine (IDs 10001-10005) - Adaptive flow-based formulas
    2. PatternRecognitionEngine (IDs 20001-20012) - HMM, stat arb, ML patterns
    3. RenTechPatternEngine (IDs 72001-72099) - RenTech advanced patterns

    Responsibilities:
    1. Start blockchain feed
    2. Process each tick through ALL THREE engines
    3. ENSEMBLE: Combine signals from all engines (3-way voting)
    4. Expose trading signals to engine
    5. Track performance for formula learning
    """

    def __init__(self,
                 zmq_endpoint: str = "tcp://127.0.0.1:28332",
                 json_path: str = None,
                 on_signal: Callable[[Dict], None] = None,
                 enable_pattern_recognition: bool = True,
                 enable_rentech: bool = True,
                 rentech_mode: str = "full",
                 claude: "ClaudeAdapter" = None):
        """
        Initialize connector.

        Args:
            zmq_endpoint: Bitcoin Core ZMQ endpoint
            json_path: Path to exchanges.json
            on_signal: Callback for trading signals
            enable_pattern_recognition: Enable HMM/ML pattern detection (20001-20012)
            enable_rentech: Enable RenTech advanced patterns (72001-72099)
            rentech_mode: RenTech engine mode - "full", "best", "hmm", "ensemble"
            claude: Optional Claude AI adapter for signal validation
        """
        self.on_signal = on_signal
        self.reference_price = 95000.0  # Updated from external source
        self.enable_pattern = enable_pattern_recognition
        self.enable_rentech = enable_rentech
        self.claude = claude

        # Initialize components
        self.feed = PerExchangeBlockchainFeed(
            on_tick=self._on_tick,
            zmq_endpoint=zmq_endpoint,
            json_path=json_path
        )

        # FORMULA ENGINE 1: Adaptive formulas (10001-10005)
        self.engine = AdaptiveTradingEngine()

        # FORMULA ENGINE 2: Pattern recognition (20001-20012)
        self.pattern_engine = PatternRecognitionEngine() if enable_pattern_recognition else None

        # FORMULA ENGINE 3: RenTech advanced patterns (72001-72099)
        self.rentech_engine = create_rentech_engine(rentech_mode) if enable_rentech else None

        # Track signals and pending trades
        self.signals: List[Dict] = []
        self.active_trades: List[Dict] = []
        self.completed_trades: List[Dict] = []

        # Stats
        self.ticks_processed = 0
        self.signals_generated = 0
        self.pattern_signals = 0
        self.adaptive_signals = 0
        self.rentech_signals = 0
        self.lock = threading.Lock()

    def start(self) -> bool:
        """Start the blockchain feed."""
        print("[CONNECTOR] Starting blockchain -> formula connection")
        self.feed.set_reference_price(self.reference_price)
        return self.feed.start()

    def stop(self):
        """Stop the blockchain feed."""
        self.feed.stop()

    def set_reference_price(self, price: float):
        """Update reference price (should be called periodically from exchange)."""
        self.reference_price = price
        self.feed.set_reference_price(price)
        self.engine.regime.add_price(price, time.time())

    def _on_tick(self, tick: ExchangeTick):
        """Process each blockchain tick through ALL formula engines."""
        with self.lock:
            self.ticks_processed += 1

        # ENGINE 1: Adaptive Trading Engine (IDs 10001-10005)
        adaptive_signal = self.engine.on_flow(
            exchange=tick.exchange,
            direction=tick.direction,
            btc=tick.volume,
            ts=tick.timestamp,
            price=tick.price
        )

        # ENGINE 2: Pattern Recognition Engine (IDs 20001-20012)
        pattern_signal = None
        if self.pattern_engine:
            pattern_signal = self.pattern_engine.on_flow(
                exchange=tick.exchange,
                direction=tick.direction,
                btc=tick.volume,
                timestamp=tick.timestamp,
                price=tick.price
            )

        # ENGINE 3: RenTech Advanced Patterns (IDs 72001-72099)
        rentech_signal = None
        if self.rentech_engine:
            # Build features dict from tick data
            features = {
                'exchange': tick.exchange,
                'direction': tick.direction,
                'volume': tick.volume,
                'timestamp': tick.timestamp,
            }
            rentech_signal = self.rentech_engine.on_tick(tick.price, features)

        # ENSEMBLE: Combine signals from ALL THREE engines
        final_signal = self._ensemble_signals(adaptive_signal, pattern_signal, rentech_signal, tick)

        if final_signal:
            # CLAUDE AI VALIDATION (if enabled)
            if self.claude:
                validation = self.claude.validate_signal(final_signal, {
                    'win_rate': self._get_win_rate(),
                    'recent_trades': len(self.completed_trades),
                })
                if validation.success:
                    # Apply confidence adjustment
                    final_signal['confidence'] *= validation.confidence_adjustment
                    final_signal['claude_validation'] = {
                        'action': validation.action,
                        'adjustment': validation.confidence_adjustment,
                        'reasoning': validation.reasoning,
                        'warnings': validation.warnings,
                        'latency_ms': validation.latency_ms,
                    }
                    # Reject if Claude recommends
                    if validation.action == "REJECT":
                        print(f"[CLAUDE] Signal REJECTED: {validation.reasoning}")
                        return  # Don't process this signal
                    elif validation.action == "ADJUST":
                        final_signal['position_size'] *= validation.size_adjustment
                        print(f"[CLAUDE] Signal ADJUSTED: size *{validation.size_adjustment:.2f}")

            with self.lock:
                self.signals.append(final_signal)
                self.signals_generated += 1

            # Log significant signals
            dir_str = "LONG" if final_signal['direction'] == 1 else "SHORT"
            sources = []
            if final_signal.get('adaptive_signal'):
                sources.append('adaptive')
            if final_signal.get('pattern_signal'):
                sources.append('pattern')
            if final_signal.get('rentech_signal'):
                sources.append('rentech')
            claude_str = " | claude=OK" if final_signal.get('claude_validation') else ""
            print(f"[ENSEMBLE] {dir_str} | {tick.volume:.2f} BTC | "
                  f"conf={final_signal['confidence']:.2f} | sources={sources}{claude_str}")

            if self.on_signal:
                self.on_signal(final_signal)

    def _ensemble_signals(self, adaptive: Optional[Dict], pattern: Optional[Dict],
                          rentech: Optional[RenTechSignal], tick: ExchangeTick) -> Optional[Dict]:
        """
        Combine signals from ALL THREE engines using weighted voting.

        ENSEMBLE RULES (3-way voting):
        1. All 3 agree -> HIGHEST confidence (boost 1.5x)
        2. 2 of 3 agree (majority) -> HIGH confidence (boost 1.3x)
        3. 1 signal with high confidence -> use it
        4. Conflicting signals -> use highest confidence if > 0.7, else wait
        """
        # Extract direction and confidence from each engine
        adaptive_dir = adaptive['direction'] if adaptive else 0
        adaptive_conf = adaptive['confidence'] if adaptive else 0

        pattern_dir = pattern['direction'] if pattern and pattern.get('should_trade') else 0
        pattern_conf = pattern['confidence'] if pattern and pattern.get('should_trade') else 0

        rentech_dir = rentech.direction.value if rentech and rentech.direction != SignalDirection.NEUTRAL else 0
        rentech_conf = rentech.confidence if rentech and rentech.direction != SignalDirection.NEUTRAL else 0

        # Track individual signals
        if adaptive_dir != 0:
            with self.lock:
                self.adaptive_signals += 1
        if pattern_dir != 0:
            with self.lock:
                self.pattern_signals += 1
        if rentech_dir != 0:
            with self.lock:
                self.rentech_signals += 1

        # Collect all active signals
        signals = []
        if adaptive_dir != 0:
            signals.append(('adaptive', adaptive_dir, adaptive_conf, adaptive))
        if pattern_dir != 0:
            signals.append(('pattern', pattern_dir, pattern_conf, pattern))
        if rentech_dir != 0:
            signals.append(('rentech', rentech_dir, rentech_conf, rentech))

        # No signals from any engine
        if not signals:
            return None

        # Count votes for each direction
        long_votes = [(s[0], s[2], s[3]) for s in signals if s[1] == 1]
        short_votes = [(s[0], s[2], s[3]) for s in signals if s[1] == -1]

        long_count = len(long_votes)
        short_count = len(short_votes)

        # Determine winning direction and confidence
        if long_count > short_count:
            direction = 1
            winning_votes = long_votes
        elif short_count > long_count:
            direction = -1
            winning_votes = short_votes
        else:
            # Tie - use highest individual confidence
            all_by_conf = sorted(signals, key=lambda x: x[2], reverse=True)
            if all_by_conf[0][2] > 0.7:
                direction = all_by_conf[0][1]
                winning_votes = [(all_by_conf[0][0], all_by_conf[0][2], all_by_conf[0][3])]
            else:
                return None  # Conflicting signals with low confidence, wait

        # Calculate ensemble confidence based on agreement level
        avg_conf = sum(v[1] for v in winning_votes) / len(winning_votes)
        vote_count = len(winning_votes)

        if vote_count == 3:
            # All 3 agree - HIGHEST confidence
            ensemble_conf = min(0.98, avg_conf * 1.5)
            ensemble_type = 'unanimous'
            position_mult = 1.5
        elif vote_count == 2:
            # 2 of 3 agree - HIGH confidence
            ensemble_conf = min(0.95, avg_conf * 1.3)
            ensemble_type = 'majority'
            position_mult = 1.2
        else:
            # Only 1 signal
            ensemble_conf = avg_conf
            ensemble_type = f'{winning_votes[0][0]}_only'
            position_mult = 1.0

        # Build regime from best available source
        regime = 'UNKNOWN'
        if rentech and hasattr(rentech, 'regime'):
            regime = rentech.regime
        elif pattern and pattern.get('regime'):
            regime = pattern['regime']
        elif adaptive and adaptive.get('regime'):
            regime = adaptive.get('regime')

        # Get Kelly fraction from RenTech if available
        kelly = 0.05
        if rentech and hasattr(rentech, 'kelly_fraction'):
            kelly = rentech.kelly_fraction

        # Determine entry parameters (prefer adaptive, then rentech, then defaults)
        entry_delay = 10
        hold_time = 30
        stop_loss = 0.005
        take_profit = 0.008
        base_position = 0.05

        if adaptive:
            entry_delay = adaptive.get('entry_delay', 10)
            hold_time = adaptive.get('hold_time', 30)
            stop_loss = adaptive.get('stop_loss', 0.005)
            take_profit = adaptive.get('take_profit', 0.008)
            base_position = adaptive.get('position_size', 0.05)

        return {
            'direction': direction,
            'confidence': ensemble_conf,
            'btc_amount': tick.volume,
            'exchange': tick.exchange,
            'price': tick.price,
            'entry_time': tick.timestamp + entry_delay,
            'entry_delay': entry_delay,
            'hold_time': hold_time,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'regime': regime,
            'position_size': min(0.15, base_position * position_mult),  # Cap at 15%
            'kelly_fraction': kelly,
            'adaptive_signal': adaptive if 'adaptive' in [v[0] for v in winning_votes] else None,
            'pattern_signal': pattern if 'pattern' in [v[0] for v in winning_votes] else None,
            'rentech_signal': rentech if 'rentech' in [v[0] for v in winning_votes] else None,
            'ensemble_type': ensemble_type,
            'vote_count': vote_count,
            'total_engines': len(signals),
        }

    def get_signal(self) -> Optional[Dict]:
        """Get latest pending signal (if any)."""
        with self.lock:
            if self.signals:
                return self.signals[-1]
        return None

    def get_pending_signals(self) -> List[Dict]:
        """Get all pending signals (without clearing - use consume_pending_signals instead)."""
        with self.lock:
            return list(self.signals)

    def consume_pending_signals(self) -> List[Dict]:
        """Get all pending signals AND CLEAR them. Use this to avoid signal flooding."""
        with self.lock:
            signals = list(self.signals)
            self.signals = []
            return signals

    def check_entries(self, current_price: float) -> List[Dict]:
        """
        Check if any pending signals should enter.

        Call this periodically from the engine loop.
        Returns list of signals ready to enter.
        """
        now = time.time()
        entries = self.engine.check_entries(now, current_price)

        with self.lock:
            for entry in entries:
                self.active_trades.append(entry)
                # Remove from pending signals
                self.signals = [s for s in self.signals
                               if s.get('entry_time') != entry.get('entry_time')]

        return entries

    def record_trade_result(self, trade: Dict, exit_price: float) -> float:
        """
        Record trade result for formula learning.

        CRITICAL: This is how formulas LEARN from outcomes.
        Call this after every trade closes.

        LEARNING HAPPENS IN:
        1. AdaptiveTradingEngine (10001-10005) - updates timing, impact estimates
        2. PatternRecognitionEngine (20001-20012) - updates HMM, pattern success rates
        3. RenTechPatternEngine (72001-72099) - updates pattern success rates

        Returns:
            PnL of the trade
        """
        exit_time = time.time()
        pnl = self.engine.record_result(trade, exit_price, exit_time)

        # Calculate if trade was profitable
        was_profitable = pnl > 0

        # Update pattern recognition engine
        if self.pattern_engine:
            self.pattern_engine.record_outcome(was_profitable)

        # Update RenTech pattern engine
        if self.rentech_engine:
            self.rentech_engine.record_outcome(was_profitable, pnl)

        with self.lock:
            # Move from active to completed
            self.active_trades = [t for t in self.active_trades
                                  if t.get('entry_time') != trade.get('entry_time')]
            self.completed_trades.append({**trade, 'exit_price': exit_price, 'pnl': pnl})

        return pnl

    def get_aggregated_signal(self) -> Dict:
        """Get aggregated blockchain signal (from raw feed)."""
        return self.feed.get_aggregated_signal()

    def get_formula_signal(self) -> Dict:
        """Get signal from multi-timescale aggregator."""
        return self.engine.multiscale.get()

    def get_regime(self) -> Dict:
        """Get current market regime."""
        return self.engine.regime.get()

    def get_pattern_signal(self) -> Optional[Dict]:
        """Get signal from pattern recognition engine."""
        if self.pattern_engine:
            return self.pattern_engine.last_signal
        return None

    def get_pattern_stats(self) -> Dict:
        """Get pattern recognition statistics."""
        if self.pattern_engine:
            return self.pattern_engine.get_stats()
        return {}

    def get_rentech_signal(self) -> Optional[RenTechSignal]:
        """Get latest signal from RenTech pattern engine."""
        if self.rentech_engine:
            return self.rentech_engine.last_signal
        return None

    def get_rentech_stats(self) -> Dict:
        """Get RenTech pattern engine statistics."""
        if self.rentech_engine:
            if hasattr(self.rentech_engine, 'get_all_stats'):
                return self.rentech_engine.get_all_stats()
            elif hasattr(self.rentech_engine, 'get_stats'):
                return self.rentech_engine.get_stats()
        return {}

    def _get_win_rate(self) -> float:
        """Get current win rate from completed trades."""
        with self.lock:
            if not self.completed_trades:
                return 50.0
            wins = sum(1 for t in self.completed_trades if t.get('pnl', 0) > 0)
            return (wins / len(self.completed_trades)) * 100

    def get_stats(self) -> Dict:
        """Get comprehensive stats."""
        feed_stats = self.feed.get_stats()
        engine_stats = self.engine.get_stats()
        pattern_stats = self.get_pattern_stats()
        rentech_stats = self.get_rentech_stats()

        with self.lock:
            return {
                # Feed stats
                'feed': feed_stats,
                # Adaptive engine stats (10001-10005)
                'adaptive_engine': engine_stats,
                # Pattern recognition stats (20001-20012)
                'pattern_engine': pattern_stats,
                # RenTech advanced patterns (72001-72099)
                'rentech_engine': rentech_stats,
                # Connector stats
                'ticks_processed': self.ticks_processed,
                'signals_generated': self.signals_generated,
                'adaptive_signals': self.adaptive_signals,
                'pattern_signals': self.pattern_signals,
                'rentech_signals': self.rentech_signals,
                'pending_signals': len(self.signals),
                'active_trades': len(self.active_trades),
                'completed_trades': len(self.completed_trades),
                # Claude AI stats
                'claude': self.claude.get_stats() if self.claude else None,
            }


# Convenience function for quick testing
def create_connector(
    zmq_endpoint: str = "tcp://127.0.0.1:28332",
    on_signal: Callable[[Dict], None] = None,
    claude: "ClaudeAdapter" = None
) -> FormulaConnector:
    """Create and return a FormulaConnector."""
    return FormulaConnector(zmq_endpoint=zmq_endpoint, on_signal=on_signal, claude=claude)
