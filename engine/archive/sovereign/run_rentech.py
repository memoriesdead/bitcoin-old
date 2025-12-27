"""
RenTech Live Trading Runner
============================

Dedicated runner for live trading using RenTech advanced patterns (72001-72099).

Modes:
    paper   - Simulation with fake fills (no real money)
    dry_run - Real prices, simulated fills
    live    - Real trading (REAL MONEY)

Usage:
    # Paper trading (simulation)
    python -m engine.sovereign.run_rentech --mode paper --duration 3600

    # Dry run (real prices, simulated fills)
    python -m engine.sovereign.run_rentech --mode dry_run --duration 3600

    # LIVE (real money - be careful!)
    python -m engine.sovereign.run_rentech --mode live --capital 10000

Safety:
    - Start with paper mode for testing
    - Graduate to dry_run to verify execution
    - Only use live mode after extensive testing
    - Kill switch available via Telegram /stop
"""

import argparse
import time
import signal
import sys
import threading
from typing import Optional, Dict, Any
from datetime import datetime

# RenTech components
from .formulas.rentech_engine import (
    RenTechPatternEngine,
    RenTechSignal,
    SignalDirection,
    create_rentech_engine,
)

# Blockchain feed
from .blockchain.formula_connector import FormulaConnector

# Integrated trading system
from .integration import (
    IntegratedTradingSystem,
    TradingMode,
    IntegratedSignal,
    create_trading_system,
)

# Execution
from .execution import (
    ExecutionResult,
    load_config,
)


class RenTechLiveRunner:
    """
    Live trading runner using RenTech patterns (72001-72099).

    FLOW:
        Bitcoin ZMQ → FormulaConnector → RenTechEngine → IntegratedSystem → Execution
                                              ↓
                                     [99 pattern formulas]
                                     [HMM, Signal, NonLinear, Micro, Ensemble]
                                              ↓
                                     [ML Enhancement (QLib)]
                                              ↓
                                     [RL Position Sizing (FinRL)]
                                              ↓
                                     [Safety Checks]
                                              ↓
                                     [CCXT Execution]
    """

    def __init__(self,
                 mode: str = "paper",
                 capital: float = 10000.0,
                 config_path: Optional[str] = None,
                 rentech_mode: str = "full",
                 enable_claude: bool = False):
        """
        Initialize the RenTech live runner.

        Args:
            mode: Trading mode ("paper", "dry_run", "live")
            capital: Initial capital in USD
            config_path: Path to trading config file
            rentech_mode: RenTech engine mode ("full", "best", "ensemble")
            enable_claude: Enable Claude AI for signal validation
        """
        self.mode = mode
        self.capital = capital
        self.config_path = config_path
        self.rentech_mode = rentech_mode
        self.enable_claude = enable_claude
        self.claude = None

        # State
        self._running = False
        self._shutdown = threading.Event()
        self._lock = threading.Lock()

        # Stats
        self.start_time = 0.0
        self.signals_received = 0
        self.signals_processed = 0
        self.trades_executed = 0
        self.errors = []

        # Components (initialized in start())
        self.connector: Optional[FormulaConnector] = None
        self.system: Optional[IntegratedTradingSystem] = None
        self.current_price = 0.0

        # Callbacks
        self.on_signal = None
        self.on_trade = None
        self.on_error = None

        print(f"[RENTECH] Initializing in {mode} mode with ${capital:,.0f} capital")
        print(f"[RENTECH] RenTech engine mode: {rentech_mode}")
        if enable_claude:
            print(f"[RENTECH] Claude AI: ENABLED (Sonnet)")

    def _init_components(self):
        """Initialize all components."""
        # 0. Initialize Claude AI if enabled
        if self.enable_claude:
            try:
                from .ai.claude_adapter import ClaudeAdapter, ClaudeConfig
                config = ClaudeConfig(
                    enabled=True,
                    model="sonnet",
                    validate_signals=True,
                    confirm_trades=True,
                    risk_assessment=True,
                    timeout=30,  # Longer timeout for real trading
                    fallback_on_timeout=True,
                )
                self.claude = ClaudeAdapter(config)
                if self.claude.client:
                    print("[RENTECH] Claude AI initialized (Sonnet)")
                else:
                    print("[RENTECH] Claude AI failed to initialize")
                    self.claude = None
            except Exception as e:
                print(f"[RENTECH] Claude AI error: {e}")
                self.claude = None

        # 1. Create trading system
        self.system = create_trading_system(
            mode=self.mode,
            config_path=self.config_path,
        )
        self.system.state.capital = self.capital

        # Set callbacks
        self.system.on_trade = self._on_trade_result
        self.system.on_signal = self._on_integrated_signal

        # 2. Create formula connector with RenTech enabled
        self.connector = FormulaConnector(
            enable_pattern_recognition=True,
            enable_rentech=True,
            rentech_mode=self.rentech_mode,
            on_signal=self._on_connector_signal,
        )

        print("[RENTECH] Components initialized")

    def start(self, duration: int = 0):
        """
        Start the live trading runner.

        Args:
            duration: Run duration in seconds (0 = infinite)
        """
        if self._running:
            print("[RENTECH] Already running")
            return

        print(f"[RENTECH] Starting live trading...")
        print(f"[RENTECH] Mode: {self.mode}")
        print(f"[RENTECH] Duration: {'infinite' if duration == 0 else f'{duration}s'}")
        print("=" * 60)

        # Initialize components
        self._init_components()

        # Set running state
        self._running = True
        self.start_time = time.time()
        self._shutdown.clear()

        # Start integrated system
        self.system.start()

        # Start blockchain connector
        if not self.connector.start():
            print("[RENTECH] ERROR: Failed to start blockchain connector")
            self.stop()
            return

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("[RENTECH] Running... (Ctrl+C to stop)")
        print("-" * 60)

        # Main loop
        end_time = time.time() + duration if duration > 0 else float('inf')

        try:
            while self._running and time.time() < end_time:
                # Periodic status update
                if int(time.time()) % 60 == 0:
                    self._print_status()

                # Wait for shutdown or timeout
                if self._shutdown.wait(timeout=1.0):
                    break

        except KeyboardInterrupt:
            print("\n[RENTECH] Keyboard interrupt received")

        finally:
            self.stop()

    def stop(self):
        """Stop the live trading runner."""
        if not self._running:
            return

        print("\n[RENTECH] Stopping...")
        self._running = False
        self._shutdown.set()

        # Stop components
        if self.connector:
            self.connector.stop()

        if self.system:
            self.system.stop()

        # Print final stats
        self._print_final_stats()

    def _signal_handler(self, signum, frame):
        """Handle system signals."""
        print(f"\n[RENTECH] Signal {signum} received")
        self._shutdown.set()

    def _on_connector_signal(self, signal: Dict[str, Any]):
        """Handle signal from FormulaConnector."""
        with self._lock:
            self.signals_received += 1

        # Check if this is a RenTech signal
        rentech_signal = signal.get('rentech_signal')
        if rentech_signal:
            self._process_rentech_signal(rentech_signal)
        else:
            # Process as regular flow signal
            self._process_flow_signal(signal)

    def _process_rentech_signal(self, signal: RenTechSignal):
        """Process a RenTech signal through the integrated system."""
        if not self._running:
            return

        if signal.direction == SignalDirection.NEUTRAL:
            return

        with self._lock:
            self.signals_processed += 1

        # Get current price from connector
        price = self._get_current_price()
        if price <= 0:
            print("[RENTECH] Warning: No valid price available")
            return

        # CLAUDE AI SIGNAL VALIDATION
        if self.claude:
            validation = self.claude.validate_signal(
                {
                    'direction': 1 if signal.direction == SignalDirection.LONG else -1,
                    'confidence': signal.confidence,
                    'vote_count': 1,
                    'total_engines': 1,
                    'ensemble_type': 'rentech',
                    'btc_amount': 0,
                    'exchange': 'blockchain',
                    'regime': signal.regime,
                },
                {'win_rate': 50.0}
            )
            if validation.success:
                if validation.action == "REJECT":
                    print(f"[CLAUDE] Signal REJECTED: {validation.reasoning}")
                    return
                elif validation.action == "ADJUST":
                    # Adjust confidence
                    signal.confidence *= validation.confidence_adjustment
                    print(f"[CLAUDE] Signal ADJUSTED: conf={validation.confidence_adjustment:.2f}x")

        # Process through integrated system
        try:
            result = self.system.on_rentech_signal(signal, price)

            if result and result.final_direction != 0:
                self._log_signal(signal, result)

                # Fire callback
                if self.on_signal:
                    self.on_signal(signal, result)

        except Exception as e:
            error_msg = f"Error processing RenTech signal: {e}"
            self.errors.append(error_msg)
            print(f"[RENTECH] {error_msg}")

            if self.on_error:
                self.on_error(error_msg)

    def _process_flow_signal(self, signal: Dict):
        """Process a regular flow signal through the integrated system."""
        if not self._running:
            return

        if signal.get('direction', 0) == 0:
            return

        with self._lock:
            self.signals_processed += 1

        # Get current price
        price = signal.get('price', self._get_current_price())
        if price <= 0:
            return

        # Process through integrated system
        try:
            result = self.system.on_flow_signal(signal, price)

            if result and result.final_direction != 0:
                dir_str = "LONG" if result.final_direction == 1 else "SHORT"
                print(f"[FLOW] {dir_str} | conf={result.final_confidence:.2f} | "
                      f"size={result.final_size:.6f}")

        except Exception as e:
            self.errors.append(f"Error processing flow signal: {e}")

    def _on_trade_result(self, result: ExecutionResult):
        """Handle trade execution result."""
        with self._lock:
            if result.success:
                self.trades_executed += 1

        # Fire callback
        if self.on_trade:
            self.on_trade(result)

    def _on_integrated_signal(self, signal: IntegratedSignal):
        """Handle integrated signal."""
        # Log for monitoring
        if signal.source == "rentech":
            print(f"[INTEGRATED] RenTech formula {signal.rentech_formula_id} | "
                  f"dir={signal.final_direction} | conf={signal.final_confidence:.2f}")

    def _get_current_price(self) -> float:
        """Get current BTC price."""
        # Try from connector aggregated signal
        if self.connector:
            agg = self.connector.get_aggregated_signal()
            if agg and agg.get('price', 0) > 0:
                self.current_price = agg['price']
                return self.current_price

        # Fallback to stored price
        return self.current_price if self.current_price > 0 else 100000.0

    def _log_signal(self, signal: RenTechSignal, result: IntegratedSignal):
        """Log a processed signal."""
        dir_str = "LONG" if signal.direction == SignalDirection.LONG else "SHORT"
        timestamp = datetime.now().strftime("%H:%M:%S")

        print(f"[{timestamp}] RENTECH {dir_str} | "
              f"formula={signal.formula_id} | "
              f"regime={signal.regime} | "
              f"conf={signal.confidence:.2f} | "
              f"kelly={signal.kelly_fraction:.3f} | "
              f"final_size={result.final_size:.6f}")

    def _print_status(self):
        """Print periodic status update."""
        uptime = time.time() - self.start_time
        hours = int(uptime // 3600)
        mins = int((uptime % 3600) // 60)

        with self._lock:
            print(f"[STATUS] uptime={hours}h{mins}m | "
                  f"signals={self.signals_received} | "
                  f"processed={self.signals_processed} | "
                  f"trades={self.trades_executed} | "
                  f"errors={len(self.errors)}")

    def _print_final_stats(self):
        """Print final statistics."""
        print("\n" + "=" * 60)
        print("[RENTECH] FINAL STATISTICS")
        print("=" * 60)

        uptime = time.time() - self.start_time
        hours = int(uptime // 3600)
        mins = int((uptime % 3600) // 60)
        secs = int(uptime % 60)

        print(f"  Mode: {self.mode}")
        print(f"  Uptime: {hours}h {mins}m {secs}s")
        print(f"  Signals received: {self.signals_received}")
        print(f"  Signals processed: {self.signals_processed}")
        print(f"  Trades executed: {self.trades_executed}")
        print(f"  Errors: {len(self.errors)}")

        if self.system:
            stats = self.system.get_stats()
            print(f"  Daily PnL: ${stats.get('daily_pnl', 0):,.2f}")
            print(f"  Total PnL: ${stats.get('total_pnl', 0):,.2f}")

        if self.claude:
            c_stats = self.claude.get_stats()
            print(f"\n  --- Claude AI ---")
            print(f"  Calls Made: {c_stats['calls_made']}")
            print(f"  Success Rate: {c_stats['success_rate']:.1f}%")
            print(f"  Avg Latency: {c_stats['avg_latency_ms']:.0f}ms")

        print("=" * 60)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self._lock:
            stats = {
                'mode': self.mode,
                'running': self._running,
                'uptime': time.time() - self.start_time if self.start_time else 0,
                'signals_received': self.signals_received,
                'signals_processed': self.signals_processed,
                'trades_executed': self.trades_executed,
                'errors': len(self.errors),
            }

        if self.system:
            stats['system'] = self.system.get_stats()

        if self.connector:
            stats['connector'] = self.connector.get_stats()

        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RenTech Live Trading Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trading (simulation)
  python -m engine.sovereign.run_rentech --mode paper

  # Dry run (real prices, simulated fills)
  python -m engine.sovereign.run_rentech --mode dry_run --duration 3600

  # LIVE trading (real money)
  python -m engine.sovereign.run_rentech --mode live --capital 10000

Safety Notes:
  - Always start with paper mode
  - Graduate to dry_run to verify execution
  - Only use live mode after extensive testing
  - Kill switch available via Telegram /stop
        """
    )

    parser.add_argument(
        "--mode",
        choices=["paper", "dry_run", "live"],
        default="paper",
        help="Trading mode (default: paper)"
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital in USD (default: 10000)"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Run duration in seconds (0 = infinite, default: 0)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to trading config file"
    )

    parser.add_argument(
        "--rentech-mode",
        choices=["full", "best", "hmm", "signal", "nonlinear", "micro", "ensemble"],
        default="full",
        help="RenTech engine mode (default: full)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--claude",
        action="store_true",
        help="Enable Claude AI for signal validation and trade confirmation"
    )

    args = parser.parse_args()

    # Safety warning for live mode
    if args.mode == "live":
        print("=" * 60)
        print(" WARNING: LIVE TRADING MODE")
        print(" This will trade with REAL MONEY!")
        print("=" * 60)
        confirm = input("Type 'I UNDERSTAND' to continue: ")
        if confirm != "I UNDERSTAND":
            print("Aborted.")
            sys.exit(1)

    # Create and run
    runner = RenTechLiveRunner(
        mode=args.mode,
        capital=args.capital,
        config_path=args.config,
        rentech_mode=args.rentech_mode,
        enable_claude=args.claude,
    )

    runner.start(duration=args.duration)


if __name__ == "__main__":
    main()
