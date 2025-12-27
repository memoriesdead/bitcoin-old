"""
Hyperliquid Signal Bridge
=========================

Connects the Sovereign signal engine to Hyperliquid execution.

This module bridges:
- ExchangeSignalEngine (generates signals from blockchain data)
- HyperliquidExecutor (executes trades on Hyperliquid DEX)

Usage:
  bridge = HyperliquidBridge(signal_engine, hyperliquid_executor)
  bridge.start()  # Starts listening for signals

For Hostinger VPS:
  python -m engine.sovereign.execution.hyperliquid_bridge
"""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Bridge configuration."""
    # Signal file for cross-process communication
    signal_file: str = "/root/sovereign/signal.json"

    # Minimum interval between trades (seconds)
    min_trade_interval: float = 60.0

    # Signal validation
    require_blockchain_signal: bool = True
    min_confidence: float = 0.55
    strong_signal_confidence: float = 0.65

    # Position management
    close_on_reversal: bool = True
    close_on_neutral: bool = False

    # Logging
    log_all_signals: bool = True
    log_file: str = "/root/sovereign/bridge.log"


class HyperliquidBridge:
    """
    Bridge between Sovereign signal engine and Hyperliquid executor.

    Two modes of operation:
    1. Direct: Pass signals programmatically via process_signal()
    2. File-based: Write signals to signal.json, bridge reads and executes
    """

    def __init__(self,
                 executor,
                 config: Optional[BridgeConfig] = None):
        """
        Initialize bridge.

        Args:
            executor: HyperliquidExecutor instance
            config: Bridge configuration
        """
        self.executor = executor
        self.config = config or BridgeConfig()

        # State
        self.running = False
        self.last_trade_time = 0.0
        self.last_direction = 0
        self.trade_count = 0

        # File monitoring
        self._last_signal_mtime = 0.0
        self._monitor_thread = None

        # Stats
        self.stats = {
            'signals_received': 0,
            'signals_executed': 0,
            'signals_skipped': 0,
            'total_pnl': 0.0,
        }

        logger.info("HyperliquidBridge initialized")

    def process_signal(self, signal: Dict) -> Optional[Dict]:
        """
        Process a signal from the engine.

        Args:
            signal: Signal dictionary with:
                - direction: 1 (LONG), -1 (SHORT), 0 (NEUTRAL)
                - confidence: float [0, 1]
                - should_trade: bool (optional)
                - position_size: float (optional)

        Returns:
            Trade result or None
        """
        self.stats['signals_received'] += 1

        if self.config.log_all_signals:
            logger.info(f"Signal received: {signal}")

        # Validate signal
        if not self._validate_signal(signal):
            self.stats['signals_skipped'] += 1
            return None

        direction = signal.get('direction', 0)
        confidence = signal.get('confidence', 0.5)

        # Check trade interval
        now = time.time()
        if now - self.last_trade_time < self.config.min_trade_interval:
            logger.debug(f"Skipping - cooldown ({now - self.last_trade_time:.0f}s)")
            self.stats['signals_skipped'] += 1
            return None

        # Handle direction
        if direction == 0:
            if self.config.close_on_neutral:
                return self._close_position("neutral_signal")
            return None

        # Check for reversal
        if self.config.close_on_reversal and self.last_direction != 0:
            if direction != self.last_direction:
                logger.info(f"Reversal detected: {self.last_direction} -> {direction}")
                self._close_position("reversal")

        # Execute trade
        try:
            result = self.executor.execute_with_brackets(
                direction=direction,
                confidence=confidence
            )

            entry_result = result.get('entry')
            if entry_result and entry_result.success:
                self.last_trade_time = now
                self.last_direction = direction
                self.trade_count += 1
                self.stats['signals_executed'] += 1

                logger.info(
                    f"Trade #{self.trade_count}: "
                    f"{'LONG' if direction == 1 else 'SHORT'} "
                    f"@ {entry_result.price:.2f} "
                    f"(conf: {confidence:.2f})"
                )

                return {
                    'success': True,
                    'direction': direction,
                    'price': entry_result.price,
                    'size': entry_result.size,
                    'order_id': entry_result.order_id,
                    'sl_set': result.get('sl', {}).success if result.get('sl') else False,
                    'tp_set': result.get('tp', {}).success if result.get('tp') else False,
                }
            else:
                self.stats['signals_skipped'] += 1
                logger.warning(f"Trade failed: {entry_result.error if entry_result else 'No result'}")
                return None

        except Exception as e:
            logger.error(f"Execution error: {e}")
            self.stats['signals_skipped'] += 1
            return None

    def _validate_signal(self, signal: Dict) -> bool:
        """Validate signal before execution."""
        # Check should_trade flag (from blockchain signal)
        if self.config.require_blockchain_signal:
            if not signal.get('should_trade', True):
                logger.debug("Signal has should_trade=False")
                return False

        # Check confidence
        confidence = signal.get('confidence', 0)
        if confidence < self.config.min_confidence:
            logger.debug(f"Confidence {confidence:.2f} below threshold {self.config.min_confidence}")
            return False

        # Check direction
        direction = signal.get('direction', 0)
        if direction not in [1, -1, 0]:
            logger.warning(f"Invalid direction: {direction}")
            return False

        return True

    def _close_position(self, reason: str) -> Optional[Dict]:
        """Close current position."""
        try:
            result = self.executor.execute_signal(direction=0, confidence=1.0)
            if result.success:
                logger.info(f"Position closed: {reason}")
                self.last_direction = 0
            return {'closed': True, 'reason': reason}
        except Exception as e:
            logger.error(f"Close error: {e}")
            return None

    # =========================================================================
    # FILE-BASED SIGNAL MONITORING (for cross-process communication)
    # =========================================================================

    def start_file_monitor(self):
        """Start monitoring signal file for new signals."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("File monitor already running")
            return

        self.running = True
        self._monitor_thread = threading.Thread(target=self._file_monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"File monitor started: {self.config.signal_file}")

    def stop_file_monitor(self):
        """Stop file monitoring."""
        self.running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("File monitor stopped")

    def _file_monitor_loop(self):
        """Main loop for file monitoring."""
        signal_path = Path(self.config.signal_file)

        while self.running:
            try:
                if signal_path.exists():
                    mtime = signal_path.stat().st_mtime
                    if mtime > self._last_signal_mtime:
                        # New signal detected
                        self._last_signal_mtime = mtime
                        signal_data = json.loads(signal_path.read_text())
                        self.process_signal(signal_data)

                time.sleep(0.5)  # Check every 500ms

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid signal JSON: {e}")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(5)

    # =========================================================================
    # STATS AND STATUS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        executor_stats = self.executor.get_stats() if self.executor else {}

        return {
            'bridge': {
                'running': self.running,
                'trade_count': self.trade_count,
                'last_trade_time': self.last_trade_time,
                'last_direction': self.last_direction,
                **self.stats
            },
            'executor': executor_stats
        }

    def write_signal(self, direction: int, confidence: float, **kwargs) -> bool:
        """
        Write a signal to the signal file.

        Use this from your signal engine to communicate with the bridge.
        """
        signal_path = Path(self.config.signal_file)
        signal_path.parent.mkdir(parents=True, exist_ok=True)

        signal_data = {
            'direction': direction,
            'confidence': confidence,
            'timestamp': time.time(),
            'should_trade': True,
            **kwargs
        }

        try:
            signal_path.write_text(json.dumps(signal_data))
            return True
        except Exception as e:
            logger.error(f"Failed to write signal: {e}")
            return False


# =============================================================================
# RENTECH FORMULA INTEGRATION
# =============================================================================

class RenTechIntegration:
    """
    Integration of RenTech formulas with Hyperliquid execution.

    This class:
    1. Calculates real-time features from price and blockchain data
    2. Evaluates all 9 RenTech formulas
    3. Validates trades with SPRT/CUSUM
    4. Sizes positions with uncertainty-adjusted Kelly
    5. Executes via HyperliquidExecutor
    6. Monitors performance in real-time
    """

    def __init__(
        self,
        capital: float = 100.0,
        private_key: Optional[str] = None,
        mode: str = "paper",
        base_leverage: float = 10.0,
        db_path: str = "data/unified_bitcoin.db"
    ):
        """
        Initialize RenTech integration.

        Args:
            capital: Starting capital (USD)
            private_key: Hyperliquid wallet private key
            mode: Execution mode (paper, testnet, live)
            base_leverage: Base leverage to use (10x recommended)
            db_path: Path to historical data for warm-up
        """
        # Import RenTech components
        from ..formulas.rentech_features import RenTechFeatures
        from ..formulas.rentech_evaluator import RenTechEvaluator
        from ..formulas.rentech_validator import RenTechValidator
        from ..formulas.rentech_monitor import RenTechMonitor
        from ..strategy.kelly import RenTechSizer

        # Import executor
        from .hyperliquid_executor import (
            HyperliquidExecutor,
            HyperliquidConfig,
            ExecutionMode
        )

        # Create RenTech components
        self.features = RenTechFeatures(db_path=db_path)
        self.evaluator = RenTechEvaluator()
        self.validator = RenTechValidator()
        self.sizer = RenTechSizer(base_leverage=base_leverage)

        # Map mode
        mode_map = {
            'paper': ExecutionMode.PAPER,
            'testnet': ExecutionMode.TESTNET,
            'live': ExecutionMode.LIVE
        }

        # Create executor config
        exec_config = HyperliquidConfig(
            private_key=private_key or os.getenv('HYPERLIQUID_PRIVATE_KEY', '0x' + '0' * 64),
            mode=mode_map.get(mode, ExecutionMode.PAPER),
            leverage=int(base_leverage),
            max_position_usd=capital * 0.9,
        )

        # Create executor
        self.executor = HyperliquidExecutor(exec_config)

        # Create bridge
        self.bridge = HyperliquidBridge(self.executor)

        # Create monitor
        self.monitor = RenTechMonitor(
            validator=self.validator,
            initial_capital=capital,
            output_file="data/rentech_monitor.json"
        )

        # State
        self.capital = capital
        self.mode = mode
        self.base_leverage = base_leverage
        self.running = False

        # Initialize formulas in validator
        self._initialize_formulas()

        logger.info(f"RenTechIntegration initialized: {mode} mode, ${capital}, {base_leverage}x")

    def _initialize_formulas(self):
        """Initialize all formulas in validator with baseline stats."""
        from ..formulas.rentech_evaluator import RENTECH_FORMULA_CONDITIONS

        for formula_id, formula in RENTECH_FORMULA_CONDITIONS.items():
            stats = formula["stats"]
            self.validator.initialize_formula(
                formula_id=formula_id,
                baseline_win_rate=stats["win_rate"],
                baseline_avg_return=stats["avg_return"],
                baseline_trades=stats["safe_50x_trades"]
            )

    def on_price_update(
        self,
        price: float,
        tx_count: Optional[int] = None,
        whale_count: Optional[int] = None,
        total_value: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Process a price update and potentially generate trades.

        Args:
            price: Current BTC price
            tx_count: Daily transaction count
            whale_count: Daily whale transaction count
            total_value: Daily total value in BTC

        Returns:
            Trade result if executed, None otherwise
        """
        # Update features
        feature_snapshot = self.features.update(
            price=price,
            tx_count=tx_count,
            whale_count=whale_count,
            total_value=total_value
        )

        if not self.features.is_ready():
            logger.debug("Features not ready yet")
            return None

        # Get feature dictionary for evaluation
        feature_dict = self.features.get_feature_dict()

        # Evaluate all formulas
        signals = self.evaluator.evaluate(feature_dict, price)

        if not signals:
            return None

        # Process each signal
        results = []
        for signal in signals:
            result = self._process_signal(signal)
            if result:
                results.append(result)

        # Update monitor
        self.monitor.update_prices({"current_price": price})
        self.monitor.save_snapshot()

        return results[0] if results else None

    def _process_signal(self, signal) -> Optional[Dict]:
        """Process a single RenTech signal."""
        from ..formulas.rentech_evaluator import SignalDirection

        formula_id = signal.formula_id

        # Check validator
        should_trade, position_mult = self.validator.should_trade(formula_id)
        if not should_trade:
            logger.info(f"Validator blocked trade for {formula_id}")
            return None

        # Get validator state for live trade count
        validator_state = self.validator.formula_states.get(formula_id)
        n_live_trades = validator_state.live_trades if validator_state else 0

        # Calculate position size
        size_info = self.sizer.calculate_position_size(
            formula_id=formula_id,
            capital=self.capital,
            win_rate=signal.historical_win_rate,
            avg_return=signal.historical_avg_return,
            n_live_trades=n_live_trades,
            validator_multiplier=position_mult,
            current_drawdown=self.evaluator.current_drawdown
        )

        logger.info(
            f"RenTech signal: {formula_id} ({signal.formula_name})\n"
            f"  Direction: {signal.direction.name}\n"
            f"  Confidence: {signal.confidence:.2%}\n"
            f"  Position: ${size_info['position_size']:.2f} ({size_info['position_fraction']:.1%})\n"
            f"  Kelly: {size_info['kelly_fraction']:.2%}"
        )

        # Record signal in monitor
        self.monitor.on_signal(signal)

        # Convert direction
        direction = 1 if signal.direction == SignalDirection.LONG else -1

        # Execute via bridge
        bridge_signal = {
            'direction': direction,
            'confidence': signal.confidence,
            'should_trade': True,
            'position_size': size_info['position_size'],
            'formula_id': formula_id,
            'source': 'rentech',
        }

        result = self.bridge.process_signal(bridge_signal)

        if result and result.get('success'):
            # Record position in monitor
            self.monitor.on_position_opened(
                signal=signal,
                position_size=size_info['position_size'],
                leverage=size_info['effective_leverage']
            )

            # Record in evaluator
            self.evaluator.on_position_opened(signal)

            return {
                **result,
                'formula_id': formula_id,
                'formula_name': signal.formula_name,
                'hold_days': signal.hold_days,
                'size_info': size_info
            }

        return None

    def on_position_closed(
        self,
        formula_id: str,
        exit_price: float,
        entry_price: float,
        direction: int
    ):
        """
        Record a position being closed.

        Args:
            formula_id: Formula that generated the trade
            exit_price: Exit price
            entry_price: Entry price
            direction: 1 for LONG, -1 for SHORT
        """
        pnl_pct = direction * (exit_price / entry_price - 1) * 100

        # Get position size for PnL calculation
        active = self.evaluator.active_positions.get(formula_id)
        if active:
            position_size = 5.0  # Would need actual size from position
        else:
            position_size = 5.0

        pnl = position_size * (pnl_pct / 100) * self.base_leverage

        # Update validator (triggers SPRT, EMA, CUSUM updates)
        self.validator.record_trade(
            formula_id=formula_id,
            entry_time=time.time() - 86400,  # Would need actual time
            exit_time=time.time(),
            entry_price=entry_price,
            exit_price=exit_price,
            direction=direction,
            pnl_pct=pnl_pct
        )

        # Update evaluator
        self.evaluator.on_position_closed(formula_id, exit_price, pnl, pnl_pct)

        # Update monitor
        self.monitor.on_position_closed(formula_id, exit_price, pnl, pnl_pct)

        # Update capital
        self.capital += pnl

        logger.info(
            f"Position closed: {formula_id}\n"
            f"  P&L: {pnl_pct:+.2f}% (${pnl:+.2f})\n"
            f"  Capital: ${self.capital:.2f}"
        )

    def get_status(self) -> Dict:
        """Get current system status."""
        snapshot = self.monitor.get_snapshot()
        return {
            "mode": self.mode,
            "capital": self.capital,
            "equity": snapshot.current_equity,
            "drawdown": snapshot.current_drawdown,
            "active_positions": snapshot.num_active_positions,
            "system_status": snapshot.system_status,
            "formulas": {
                fid: self.validator.get_diagnostics(fid)
                for fid in self.validator.formula_states
            }
        }

    def print_status(self):
        """Print current status."""
        self.monitor.print_status()


# =============================================================================
# INTEGRATION WITH SOVEREIGN ENGINE
# =============================================================================

class SovereignIntegration:
    """
    Full integration with Sovereign signal engine.

    This class:
    1. Listens to blockchain ZMQ for exchange flows
    2. Generates signals via ExchangeSignalEngine
    3. Executes via HyperliquidExecutor
    """

    def __init__(self,
                 exchange_id: str = "hyperliquid",
                 capital: float = 100.0,
                 private_key: Optional[str] = None,
                 mode: str = "paper"):
        """
        Initialize full integration.

        Args:
            exchange_id: Exchange ID for signal engine
            capital: Starting capital (USD)
            private_key: Hyperliquid wallet private key
            mode: Execution mode (paper, testnet, live)
        """
        from ..strategy.signal_engine import ExchangeSignalEngine

        # Import executor
        from .hyperliquid_executor import (
            HyperliquidExecutor,
            HyperliquidConfig,
            ExecutionMode
        )

        # Create signal engine
        self.signal_engine = ExchangeSignalEngine(
            exchange_id=exchange_id,
            capital=capital
        )

        # Map mode
        mode_map = {
            'paper': ExecutionMode.PAPER,
            'testnet': ExecutionMode.TESTNET,
            'live': ExecutionMode.LIVE
        }

        # Create executor config
        exec_config = HyperliquidConfig(
            private_key=private_key or os.getenv('HYPERLIQUID_PRIVATE_KEY', '0x' + '0' * 64),
            mode=mode_map.get(mode, ExecutionMode.PAPER),
            leverage=10,
            max_position_usd=capital * 0.9,  # Use 90% max
        )

        # Create executor
        self.executor = HyperliquidExecutor(exec_config)

        # Create bridge
        self.bridge = HyperliquidBridge(self.executor)

        logger.info(f"SovereignIntegration initialized: {mode} mode, ${capital}")

    def process_blockchain_signal(self, blockchain_signal: Dict) -> Optional[Dict]:
        """
        Process a blockchain signal through the full pipeline.

        Args:
            blockchain_signal: Signal from per_exchange_feed.get_aggregated_signal()

        Returns:
            Trade result or None
        """
        # Generate trade signal via signal engine
        trade_signal = self.signal_engine.generate_signal(
            blockchain_signal=blockchain_signal
        )

        if trade_signal is None:
            return None

        # Convert to bridge format
        bridge_signal = {
            'direction': trade_signal.direction,
            'confidence': trade_signal.signal_confidence,
            'should_trade': trade_signal.gate_passed,
            'position_size': trade_signal.position_size,
            'source': 'blockchain',
            'exchange_id': trade_signal.exchange_id,
        }

        # Execute via bridge
        return self.bridge.process_signal(bridge_signal)

    def update_price(self, price: float):
        """Update price in signal engine."""
        self.signal_engine.update_price(price)

    def get_stats(self) -> Dict:
        """Get combined statistics."""
        return {
            'signal_engine': self.signal_engine.get_stats(),
            'bridge': self.bridge.get_stats(),
        }


# =============================================================================
# STANDALONE RUNNER
# =============================================================================

def run_standalone():
    """
    Run bridge standalone (file-based signal monitoring).

    For Hostinger VPS deployment.
    """
    from dotenv import load_dotenv

    # Load environment
    env_file = Path('/root/sovereign/.env')
    if env_file.exists():
        load_dotenv(env_file)

    # Import executor
    try:
        from .hyperliquid_executor import (
            HyperliquidExecutor,
            HyperliquidConfig,
            ExecutionMode
        )
    except ImportError:
        # Try absolute import
        from hyperliquid_executor import (
            HyperliquidExecutor,
            HyperliquidConfig,
            ExecutionMode
        )

    # Get mode from env
    mode_str = os.getenv('EXECUTION_MODE', 'paper').lower()
    mode_map = {
        'paper': ExecutionMode.PAPER,
        'testnet': ExecutionMode.TESTNET,
        'live': ExecutionMode.LIVE
    }

    # Create executor
    config = HyperliquidConfig(
        private_key=os.getenv('HYPERLIQUID_PRIVATE_KEY', '0x' + '0' * 64),
        mode=mode_map.get(mode_str, ExecutionMode.PAPER),
        leverage=int(os.getenv('LEVERAGE', 10)),
        max_position_usd=float(os.getenv('MAX_POSITION_USD', 100)),
    )

    executor = HyperliquidExecutor(config)

    # Create bridge
    bridge_config = BridgeConfig(
        signal_file=os.getenv('SIGNAL_FILE', '/root/sovereign/signal.json'),
        min_trade_interval=float(os.getenv('MIN_TRADE_INTERVAL', 60)),
    )

    bridge = HyperliquidBridge(executor, bridge_config)

    # Start monitoring
    logger.info("=" * 60)
    logger.info("HYPERLIQUID BRIDGE - STANDALONE MODE")
    logger.info("=" * 60)
    logger.info(f"Mode: {mode_str}")
    logger.info(f"Signal file: {bridge_config.signal_file}")
    logger.info("")

    bridge.start_file_monitor()

    # Keep running
    try:
        while True:
            time.sleep(60)
            stats = bridge.get_stats()
            logger.info(f"Stats: {stats['bridge']}")
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        bridge.stop_file_monitor()


if __name__ == "__main__":
    run_standalone()
