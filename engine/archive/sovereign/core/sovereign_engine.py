"""
Sovereign Engine - Unified Controller
======================================

The main engine that orchestrates:
- Data pipeline
- Formula engines
- Execution router
- Position sizing
- Auto-learning feedback

This is the single entry point for all trading modes.
"""
from typing import Dict, Any, Optional, Iterator, List, Callable
from dataclasses import dataclass, field
import time
import signal
import sys
from datetime import datetime

from .types import (
    Tick, Signal, Order, SizedOrder, ExecutionResult,
    TradeOutcome, ExecutionMode, OrderSide, OrderType, Position
)
from .config import SovereignConfig, load_config, ExecutionConfig, SafetyConfig
from ..formulas.registry import FormulaRegistry, create_registry
from ..formulas.base import BaseEngine
from ..execution.router import ExecutionRouter, create_router, SafetyManager


@dataclass
class EngineStats:
    """Engine performance statistics."""
    start_time: float = 0.0
    ticks_processed: int = 0
    signals_generated: int = 0
    trades_executed: int = 0
    trades_closed: int = 0
    wins: int = 0
    losses: int = 0

    # PnL
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    peak_pnl: float = 0.0
    max_drawdown: float = 0.0

    # Capital
    initial_capital: float = 0.0
    current_capital: float = 0.0

    @property
    def uptime_hours(self) -> float:
        if self.start_time == 0:
            return 0.0
        return (time.time() - self.start_time) / 3600

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.current_capital - self.initial_capital) / self.initial_capital * 100


@dataclass
class OpenPosition:
    """Tracked open position."""
    symbol: str
    direction: int              # +1 long, -1 short
    size: float
    entry_price: float
    entry_time: float
    signal: Signal
    stop_loss: float = 0.0
    take_profit: float = 0.0
    hold_until: float = 0.0     # Time-based exit


class SovereignEngine:
    """
    Unified controller for all trading modes.

    Main loop: tick -> formulas -> execute -> learn
    """

    def __init__(self, config: SovereignConfig = None):
        """
        Initialize the Sovereign Engine.

        Args:
            config: Complete configuration. If None, loads from default location.
        """
        self.config = config or load_config()
        self.stats = EngineStats(
            initial_capital=self.config.initial_capital,
            current_capital=self.config.initial_capital,
        )

        # Components (lazy initialized)
        self._registry: Optional[FormulaRegistry] = None
        self._router: Optional[ExecutionRouter] = None
        self._data_pipeline = None
        self._position_sizer = None
        self._telegram = None
        self._claude = None

        # State
        self._running = False
        self._positions: Dict[str, OpenPosition] = {}
        self._current_price: float = 0.0

        # Callbacks
        self.on_tick: Optional[Callable[[Tick], None]] = None
        self.on_signal: Optional[Callable[[Signal], None]] = None
        self.on_trade: Optional[Callable[[ExecutionResult], None]] = None
        self.on_close: Optional[Callable[[TradeOutcome], None]] = None

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def initialize(self):
        """Initialize all components."""
        print(f"[Sovereign] Initializing...")
        print(f"[Sovereign] Mode: {self.config.execution.mode.value}")
        print(f"[Sovereign] Capital: ${self.config.initial_capital:,.2f}")

        # Formula registry
        self._registry = create_registry(
            engines_config=self.config.engines,
            ensemble_config=self.config.ensemble,
        )
        self._registry.initialize()

        # Execution router
        self._router = create_router(
            mode=self.config.execution.mode,
            config=self.config.execution,
            safety=self.config.safety,
        )

        # Set initial equity in safety manager
        self._router.safety.set_equity(self.config.initial_capital)

        # Initialize Claude AI if enabled
        if self.config.claude.enabled:
            self._init_claude()

        # Initialize telegram if enabled
        if self.config.telegram.enabled:
            self._init_telegram()

        self.stats.start_time = time.time()
        print(f"[Sovereign] Initialized successfully")

    def _init_claude(self):
        """Initialize Claude AI adapter if configured."""
        try:
            from ..ai.claude_adapter import ClaudeAdapter, ClaudeConfig as AdapterConfig

            # Convert core config to adapter config
            adapter_config = AdapterConfig(
                enabled=True,
                model=self.config.claude.model,
                validate_signals=self.config.claude.validate_signals,
                confirm_trades=self.config.claude.confirm_trades,
                risk_assessment=self.config.claude.risk_assessment,
                market_context=self.config.claude.market_context,
                timeout=self.config.claude.timeout,
                fallback_on_timeout=self.config.claude.fallback_on_timeout,
                log_responses=self.config.claude.log_responses,
            )

            self._claude = ClaudeAdapter(adapter_config)

            if self._claude.client is not None:
                print(f"[Sovereign] Claude AI ready (model: {self.config.claude.model})")
            else:
                print(f"[Sovereign] Claude AI disabled (client not available)")
                self._claude = None

        except Exception as e:
            print(f"[Sovereign] Claude AI init failed: {e}")
            self._claude = None

    def _init_telegram(self):
        """Initialize Telegram bot if configured."""
        try:
            from ..execution.telegram_bot import create_telegram_bot

            self._telegram = create_telegram_bot(
                token=self.config.telegram.token,
                chat_id=self.config.telegram.chat_id,
                mock=(not self.config.telegram.enabled),
            )

            # Set callbacks
            self._telegram.on_stop = self.stop
            self._telegram.on_start = lambda: setattr(self, '_running', True)
            self._telegram.get_status = self._get_telegram_status

            self._telegram.start()

        except Exception as e:
            print(f"[Sovereign] Telegram init failed: {e}")

    def _get_telegram_status(self):
        """Get status for Telegram bot."""
        from ..execution.telegram_bot import TradingStatus

        return TradingStatus(
            mode=self.config.execution.mode.value,
            is_running=self._running,
            kill_switch=self._router.safety.kill_switch_active if self._router else False,
            daily_pnl=self.stats.daily_pnl,
            total_pnl=self.stats.total_pnl,
            win_rate=self.stats.win_rate,
            trades_today=self.stats.trades_executed,
            total_trades=self.stats.trades_executed,
            open_positions=len(self._positions),
            current_drawdown=self.stats.max_drawdown,
            consecutive_losses=self._router.safety.consecutive_losses if self._router else 0,
            uptime_hours=self.stats.uptime_hours,
        )

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self, data_iterator: Iterator[Tick] = None, duration_seconds: int = None):
        """
        Run the main trading loop.

        Args:
            data_iterator: Iterator of Tick objects. If None, creates from config.
            duration_seconds: Max runtime in seconds. If None, uses config.
        """
        self.initialize()

        duration = duration_seconds or self.config.duration_seconds
        end_time = time.time() + duration if duration > 0 else float('inf')

        # Get data iterator
        if data_iterator is None:
            data_iterator = self._create_data_iterator()

        self._running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

        print(f"[Sovereign] Starting main loop (duration: {duration}s)")

        try:
            for tick in data_iterator:
                if not self._running:
                    break

                if time.time() > end_time:
                    print(f"[Sovereign] Duration limit reached")
                    break

                self._process_tick(tick)

        except KeyboardInterrupt:
            print(f"\n[Sovereign] Interrupted by user")

        finally:
            self._shutdown()

    def _create_data_iterator(self) -> Iterator[Tick]:
        """Create data iterator from config."""
        # This is a placeholder - actual implementation depends on data source
        # For now, yield simulated ticks

        print(f"[Sovereign] Using simulated data (no data pipeline configured)")

        import random

        base_price = 100000.0
        tick_count = 0

        while self._running:
            # Random walk
            base_price *= (1 + random.uniform(-0.001, 0.001))

            tick = Tick(
                timestamp=time.time(),
                source="simulated",
                symbol="BTC/USDT",
                price=base_price,
                bid=base_price * 0.9999,
                ask=base_price * 1.0001,
                volume=random.uniform(0.1, 10.0),
            )

            yield tick

            tick_count += 1
            time.sleep(0.1)  # 10 ticks/second

            if tick_count % 100 == 0:
                print(f"[Sovereign] Processed {tick_count} ticks, "
                      f"PnL: ${self.stats.total_pnl:+.2f}")

    def _process_tick(self, tick: Tick):
        """Process a single tick through the pipeline."""
        self.stats.ticks_processed += 1
        self._current_price = tick.price

        # Update router with current price
        if self._router:
            self._router.set_current_price(tick.symbol, tick.price)

        # Check existing positions
        self._check_positions(tick)

        # Callback
        if self.on_tick:
            self.on_tick(tick)

        # Generate signal from formulas
        if self._registry:
            signal = self._registry.process(tick)

            if signal.tradeable:
                self.stats.signals_generated += 1

                # CLAUDE AI SIGNAL VALIDATION
                if self._claude and self.config.claude.validate_signals:
                    validation = self._claude.validate_signal(
                        {
                            'direction': signal.direction,
                            'confidence': signal.confidence,
                            'vote_count': len([v for v in (signal.votes or {}).values() if v == signal.direction]),
                            'total_engines': len(signal.votes or {}),
                            'ensemble_type': signal.source_engine,
                            'btc_amount': tick.volume,
                            'exchange': tick.source,
                            'regime': getattr(signal, 'regime', 'UNKNOWN'),
                        },
                        {'win_rate': self.stats.win_rate * 100}
                    )
                    if validation.success:
                        # Apply confidence adjustment
                        signal.confidence *= validation.confidence_adjustment
                        if validation.action == "REJECT":
                            print(f"[CLAUDE] Signal REJECTED: {validation.reasoning}")
                            return
                        elif validation.action == "ADJUST":
                            signal.suggested_size *= validation.size_adjustment

                # Callback
                if self.on_signal:
                    self.on_signal(signal)

                # Execute if we should
                if self._should_trade(signal, tick):
                    self._execute_signal(signal, tick)

    def _should_trade(self, signal: Signal, tick: Tick) -> bool:
        """Check if we should trade on this signal."""
        # Check if already in position
        if tick.symbol in self._positions:
            existing = self._positions[tick.symbol]
            # Only trade if signal is opposite direction
            if existing.direction == signal.direction:
                return False

        # Check confidence threshold
        if signal.confidence < self.config.ensemble.confidence_threshold:
            return False

        return True

    def _execute_signal(self, signal: Signal, tick: Tick):
        """Execute a trading signal."""
        # Close opposite position first
        if tick.symbol in self._positions:
            existing = self._positions[tick.symbol]
            if existing.direction != signal.direction:
                self._close_position(tick.symbol, tick.price, "signal_reversal")

        # Determine side
        side = OrderSide.BUY if signal.direction > 0 else OrderSide.SELL

        # Position sizing
        position_size_pct = signal.suggested_size if signal.suggested_size > 0 else 0.02
        position_size_usd = self.stats.current_capital * position_size_pct

        # Apply config limits
        position_size_usd = min(
            position_size_usd,
            self.config.safety.max_position_usd
        )

        # CLAUDE AI TRADE CONFIRMATION
        if self._claude and self.config.claude.confirm_trades:
            # Get recent trade PnLs
            recent_pnl = []
            if self.stats.trades_closed > 0:
                recent_pnl = []  # Would need to track actual PnL history

            confirmation = self._claude.confirm_trade(
                {
                    'direction': signal.direction,
                    'position_size': position_size_pct,
                    'price': tick.price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                },
                {
                    'capital': self.stats.current_capital,
                    'drawdown': self.stats.max_drawdown,
                    'win_rate': self.stats.win_rate * 100,
                    'recent_trades': recent_pnl,
                }
            )
            if confirmation.action == "REJECT":
                print(f"[CLAUDE] Trade REJECTED: {confirmation.reasoning}")
                return
            elif confirmation.action == "ADJUST":
                position_size_pct *= confirmation.size_adjustment
                position_size_usd = self.stats.current_capital * position_size_pct
                print(f"[CLAUDE] Trade ADJUSTED: size {confirmation.size_adjustment:.2f}x")

        # Calculate amount
        amount = position_size_usd / tick.price if tick.price > 0 else 0.0

        # Create order
        order = Order(
            symbol=tick.symbol,
            side=side,
            order_type=OrderType.MARKET,
            amount=amount,
            signal=signal,
        )

        sized_order = SizedOrder(
            order=order,
            position_size_pct=position_size_pct,
            position_size_usd=position_size_usd,
        )

        # Execute
        result = self._router.execute(sized_order)

        if result.success:
            self.stats.trades_executed += 1

            # Track position
            self._positions[tick.symbol] = OpenPosition(
                symbol=tick.symbol,
                direction=signal.direction,
                size=result.fill_amount,
                entry_price=result.fill_price,
                entry_time=result.fill_time,
                signal=signal,
                stop_loss=result.fill_price * (1 - signal.stop_loss * signal.direction),
                take_profit=result.fill_price * (1 + signal.take_profit * signal.direction),
                hold_until=time.time() + signal.hold_seconds,
            )

            # Callback
            if self.on_trade:
                self.on_trade(result)

            # Telegram
            if self._telegram:
                self._telegram.notify_trade(
                    symbol=tick.symbol,
                    side=side.value,
                    amount=result.fill_amount,
                    price=result.fill_price,
                )

        else:
            print(f"[Sovereign] Order rejected: {result.reject_reason}")

    def _check_positions(self, tick: Tick):
        """Check open positions for SL/TP/time exits."""
        if tick.symbol not in self._positions:
            return

        pos = self._positions[tick.symbol]
        price = tick.price

        # Check stop loss
        if pos.direction > 0 and price <= pos.stop_loss:
            self._close_position(tick.symbol, price, "stop_loss")
            return

        if pos.direction < 0 and price >= pos.stop_loss:
            self._close_position(tick.symbol, price, "stop_loss")
            return

        # Check take profit
        if pos.direction > 0 and price >= pos.take_profit:
            self._close_position(tick.symbol, price, "take_profit")
            return

        if pos.direction < 0 and price <= pos.take_profit:
            self._close_position(tick.symbol, price, "take_profit")
            return

        # Check time exit
        if pos.hold_until > 0 and time.time() >= pos.hold_until:
            self._close_position(tick.symbol, price, "time_exit")
            return

    def _close_position(self, symbol: str, exit_price: float, reason: str):
        """Close a position and record outcome."""
        if symbol not in self._positions:
            return

        pos = self._positions[symbol]

        # Calculate PnL
        if pos.direction > 0:
            pnl = pos.size * (exit_price - pos.entry_price)
        else:
            pnl = pos.size * (pos.entry_price - exit_price)

        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * pos.direction

        # Update stats
        self.stats.trades_closed += 1
        self.stats.total_pnl += pnl
        self.stats.daily_pnl += pnl
        self.stats.current_capital += pnl

        if pnl > 0:
            self.stats.wins += 1
        else:
            self.stats.losses += 1

        # Update peak/drawdown
        if self.stats.total_pnl > self.stats.peak_pnl:
            self.stats.peak_pnl = self.stats.total_pnl
        else:
            current_dd = (self.stats.peak_pnl - self.stats.total_pnl) / (self.stats.peak_pnl + self.config.initial_capital)
            if current_dd > self.stats.max_drawdown:
                self.stats.max_drawdown = current_dd

        # Record in safety manager
        if self._router:
            self._router.safety.record_close(pnl)

        # Create outcome for learning
        outcome = TradeOutcome(
            signal=pos.signal,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=time.time(),
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_duration=time.time() - pos.entry_time,
            exit_reason=reason,
        )

        # Feedback to formulas (AUTO-LEARNING)
        if self._registry:
            self._registry.learn(outcome)

        # Callback
        if self.on_close:
            self.on_close(outcome)

        # Telegram
        if self._telegram:
            self._telegram.notify_trade(
                symbol=symbol,
                side="close",
                amount=pos.size,
                price=exit_price,
                pnl=pnl,
            )

        # Remove position
        del self._positions[symbol]

        print(f"[Sovereign] Closed {symbol} | {reason} | PnL: ${pnl:+.2f}")

    # =========================================================================
    # CONTROL
    # =========================================================================

    def stop(self):
        """Stop the engine."""
        print(f"[Sovereign] Stopping...")
        self._running = False

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals."""
        print(f"\n[Sovereign] Received signal {signum}")
        self.stop()

    def _shutdown(self):
        """Clean shutdown."""
        print(f"[Sovereign] Shutting down...")

        # Close all positions
        for symbol in list(self._positions.keys()):
            self._close_position(symbol, self._current_price, "shutdown")

        # Stop telegram
        if self._telegram:
            self._telegram.stop()

        # Stop Claude
        if self._claude:
            self._claude.shutdown()

        # Print final stats
        self._print_stats()

    def _print_stats(self):
        """Print final statistics."""
        print(f"\n{'='*60}")
        print(f"SOVEREIGN ENGINE - FINAL STATISTICS")
        print(f"{'='*60}")
        print(f"Runtime:        {self.stats.uptime_hours:.2f} hours")
        print(f"Ticks:          {self.stats.ticks_processed:,}")
        print(f"Signals:        {self.stats.signals_generated}")
        print(f"Trades:         {self.stats.trades_executed}")
        print(f"Wins/Losses:    {self.stats.wins}/{self.stats.losses}")
        print(f"Win Rate:       {self.stats.win_rate*100:.1f}%")
        print(f"Total PnL:      ${self.stats.total_pnl:+,.2f}")
        print(f"Return:         {self.stats.return_pct:+.2f}%")
        print(f"Max Drawdown:   {self.stats.max_drawdown*100:.2f}%")

        # Claude AI stats
        if self._claude:
            c_stats = self._claude.get_stats()
            print(f"\n--- Claude AI ---")
            print(f"Calls Made:     {c_stats['calls_made']}")
            print(f"Success Rate:   {c_stats['success_rate']:.1f}%")
            print(f"Avg Latency:    {c_stats['avg_latency_ms']:.0f}ms")

        print(f"{'='*60}\n")

    # =========================================================================
    # ACCESSORS
    # =========================================================================

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def registry(self) -> Optional[FormulaRegistry]:
        return self._registry

    @property
    def router(self) -> Optional[ExecutionRouter]:
        return self._router

    def get_stats(self) -> EngineStats:
        return self.stats

    def get_positions(self) -> Dict[str, OpenPosition]:
        return self._positions.copy()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        return {
            "stats": {
                "uptime_hours": self.stats.uptime_hours,
                "ticks": self.stats.ticks_processed,
                "signals": self.stats.signals_generated,
                "trades": self.stats.trades_executed,
                "win_rate": self.stats.win_rate,
                "pnl": self.stats.total_pnl,
            },
            "registry": self._registry.get_diagnostics() if self._registry else {},
            "router": self._router.get_stats() if self._router else {},
            "positions": len(self._positions),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_engine(
    mode: ExecutionMode = ExecutionMode.PAPER,
    capital: float = 10000.0,
    config_path: str = None,
) -> SovereignEngine:
    """
    Create a Sovereign Engine.

    Args:
        mode: Execution mode
        capital: Initial capital
        config_path: Path to config file

    Returns:
        Configured SovereignEngine
    """
    if config_path:
        config = load_config(config_path)
    else:
        config = SovereignConfig()

    config.execution.mode = mode
    config.initial_capital = capital

    return SovereignEngine(config)


def run_paper(capital: float = 10000.0, duration: int = 3600):
    """Quick start paper trading."""
    engine = create_engine(ExecutionMode.PAPER, capital)
    engine.run(duration_seconds=duration)


def run_dry(capital: float = 10000.0, duration: int = 3600):
    """Quick start dry run trading."""
    engine = create_engine(ExecutionMode.DRY_RUN, capital)
    engine.run(duration_seconds=duration)
