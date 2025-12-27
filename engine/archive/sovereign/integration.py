"""
RenTech Integration Module
==========================

Unified integration of all ported components:
- QLib: Point-in-time data, alpha expressions, LightGBM
- FinRL: RL position sizing (SAC/PPO)
- CCXT: Exchange connectivity
- Freqtrade: Order management, safety, Telegram
- hftbacktest: Order book simulation

This module provides the glue to connect everything.
"""

import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import threading

# QLib components
from .formulas.qlib_alpha import (
    PointInTimeHandler,
    FlowMomentum, FlowZScore, FlowRegimeDetector,
)
from .formulas.qlib_alpha.lightgbm_flow import LightGBMFlowClassifier, OnlineLightGBM
from .formulas.qlib_alpha.online_learner import OnlineLearner, EnsembleOnlineLearner

# FinRL components
from .formulas.finrl_rl import (
    TradingEnvironment, TradingState,
    SACPositionSizer, PPOPositionSizer,
)

# ML Enhancer
from .formulas.ml_enhancer import MLEnhancer

# Execution components
from .execution import (
    ExecutionEngine, ExecutionMode, ExecutionResult,
    OrderManager, OrderState, ManagedOrder,
    SafetyManager, SafetyConfig,
    ConfigManager, TradingConfig, load_config,
    TelegramBot, TradingStatus, create_telegram_bot,
)

# Order book simulation
from .simulation.orderbook import (
    HFTBacktester, BacktestConfig, BacktestResult,
    OrderBookSnapshot, ExecutionSimulator,
)

# RenTech patterns
from .formulas.rentech_engine import RenTechPatternEngine, RenTechSignal, SignalDirection


class TradingMode(Enum):
    """Trading modes for the integrated system."""
    BACKTEST = "backtest"        # Historical simulation with order book
    PAPER = "paper"              # Real-time simulation (no real orders)
    DRY_RUN = "dry_run"          # Real prices, simulated fills
    SANDBOX = "sandbox"          # Exchange testnet
    LIVE = "live"                # Real trading


@dataclass
class IntegratedSignal:
    """Signal with all enhancements."""
    # Base signal from blockchain
    base_direction: int = 0      # -1, 0, 1
    base_confidence: float = 0.0
    base_ensemble_type: str = ""

    # RenTech signal (72001-72099)
    rentech_direction: int = 0
    rentech_confidence: float = 0.0
    rentech_formula_id: int = 0
    rentech_regime: str = ""
    rentech_kelly: float = 0.0

    # ML enhancement
    ml_direction: int = 0
    ml_confidence: float = 0.0
    ml_features: Dict[str, float] = field(default_factory=dict)

    # RL position sizing
    position_size: float = 0.0
    rl_confidence: float = 0.0

    # Final decision
    final_direction: int = 0
    final_size: float = 0.0
    final_confidence: float = 0.0

    # Metadata
    timestamp: float = 0.0
    source: str = ""  # "flow", "rentech", or "combined"
    components_used: List[str] = field(default_factory=list)


@dataclass
class IntegratedState:
    """Current state of the integrated system."""
    mode: TradingMode = TradingMode.PAPER
    is_running: bool = False

    # Performance
    capital: float = 10000.0
    position: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    trades_today: int = 0

    # Risk
    current_drawdown: float = 0.0
    consecutive_losses: int = 0

    # Last signal
    last_signal: Optional[IntegratedSignal] = None
    last_signal_time: float = 0.0

    # System
    uptime_start: float = 0.0
    errors: List[str] = field(default_factory=list)


class IntegratedTradingSystem:
    """
    Unified trading system integrating all components.

    Usage:
        system = IntegratedTradingSystem(mode=TradingMode.DRY_RUN)
        system.start()

        # Feed blockchain data
        system.on_flow_signal(signal)

        system.stop()
    """

    def __init__(self, mode: TradingMode = TradingMode.PAPER,
                 config: Optional[TradingConfig] = None):
        """
        Initialize integrated system.

        Args:
            mode: Trading mode
            config: Trading configuration
        """
        self.mode = mode
        self.config = config or load_config()
        self.state = IntegratedState(mode=mode)

        # Components
        self._init_pit_handler()
        self._init_ml_components()
        self._init_rl_components()
        self._init_execution()
        self._init_telegram()

        # State
        self._running = False
        self._lock = threading.Lock()

        # Callbacks
        self.on_trade: Optional[Callable[[ExecutionResult], None]] = None
        self.on_signal: Optional[Callable[[IntegratedSignal], None]] = None

    def _init_pit_handler(self):
        """Initialize point-in-time data handler."""
        self.pit_handler = PointInTimeHandler(
            min_delay_seconds=0,  # No delay for live trading (0 = immediate)
            live_mode=True,  # Skip strict PIT validation for live blockchain data
        )

    def _init_ml_components(self):
        """Initialize ML components."""
        # ML Enhancer (unified component)
        self.ml_enhancer = MLEnhancer()

        # Online learner for continuous adaptation
        self.online_learner = OnlineLearner(
            self.ml_enhancer.lgbm,
            drift_window=100,
            drift_threshold=0.15,
        )

    def _init_rl_components(self):
        """Initialize RL components."""
        # Trading environment
        self.trading_env = TradingEnvironment(
            initial_capital=self.config.safety.max_total_exposure_usd,
            max_position=1.0,
        )

        # Position sizer (SAC primary, PPO fallback)
        self.position_sizer = SACPositionSizer(
            state_dim=11,
            action_dim=1,
        )

        # PPO backup
        self.position_sizer_backup = PPOPositionSizer(
            state_dim=11,
            action_dim=1,
        )

    def _init_execution(self):
        """Initialize execution components."""
        # Map mode to execution mode
        execution_mode_map = {
            TradingMode.BACKTEST: ExecutionMode.PAPER,
            TradingMode.PAPER: ExecutionMode.PAPER,
            TradingMode.DRY_RUN: ExecutionMode.DRY_RUN,
            TradingMode.SANDBOX: ExecutionMode.LIVE,
            TradingMode.LIVE: ExecutionMode.LIVE,
        }

        exec_mode = execution_mode_map.get(self.mode, ExecutionMode.PAPER)

        # Safety config from trading config
        safety_config = SafetyConfig(
            max_position_usd=self.config.safety.max_position_usd,
            max_total_exposure_usd=self.config.safety.max_total_exposure_usd,
            max_daily_trades=self.config.safety.max_daily_trades,
            max_daily_loss_usd=self.config.safety.max_daily_loss_usd,
            max_daily_loss_pct=self.config.safety.max_daily_loss_pct,
            consecutive_loss_limit=self.config.safety.consecutive_loss_limit,
            max_drawdown_pct=self.config.safety.max_drawdown_pct,
        )

        # Execution engine
        self.execution = ExecutionEngine(
            mode=exec_mode,
            safety_config=safety_config,
        )

        # Execution callback
        self.execution.on_execution = self._on_execution

    def _init_telegram(self):
        """Initialize Telegram bot."""
        if self.config.telegram and self.config.telegram.enabled:
            self.telegram = create_telegram_bot(
                token=self.config.telegram.token,
                chat_id=self.config.telegram.chat_id,
            )

            # Set callbacks
            self.telegram.on_stop = self.stop
            self.telegram.on_start = self.start
            self.telegram.get_status = self._get_telegram_status
        else:
            self.telegram = create_telegram_bot("", "", mock=True)

    def start(self):
        """Start the trading system."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self.state.is_running = True
            self.state.uptime_start = time.time()

            # Start Telegram
            if hasattr(self.telegram, 'start'):
                self.telegram.start()

            # Set capital in safety manager
            self.execution.safety.set_capital(self.state.capital)

            print(f"[INTEGRATED] System started in {self.mode.value} mode")

    def stop(self):
        """Stop the trading system."""
        with self._lock:
            self._running = False
            self.state.is_running = False

            # Activate kill switch
            self.execution.safety.activate_kill_switch("Manual stop")

            # Stop Telegram
            if hasattr(self.telegram, 'stop'):
                self.telegram.stop()

            print("[INTEGRATED] System stopped")

    def on_flow_signal(self, signal: Dict[str, Any], price: float) -> Optional[IntegratedSignal]:
        """
        Process incoming flow signal.

        Args:
            signal: Raw signal from blockchain analysis
            price: Current market price

        Returns:
            IntegratedSignal if action needed
        """
        if not self._running:
            return None

        timestamp = time.time()

        # 1. Create integrated signal
        integrated = IntegratedSignal(
            timestamp=timestamp,
            base_direction=signal.get('direction', 0),
            base_confidence=signal.get('confidence', 0.0),
            base_ensemble_type=signal.get('ensemble_type', 'unknown'),
        )

        # 2. PIT validation
        if not self.pit_handler.is_valid(signal.get('data_timestamp', timestamp), timestamp):
            print("[INTEGRATED] Signal rejected: PIT validation failed")
            return None

        # 3. ML enhancement
        try:
            ml_signal = self.ml_enhancer.enhance_signal(signal, price)
            integrated.ml_direction = ml_signal.direction
            integrated.ml_confidence = ml_signal.confidence
            integrated.ml_features = ml_signal.features
            integrated.components_used.append("ml_enhancer")
        except Exception as e:
            print(f"[INTEGRATED] ML enhancement failed: {e}")
            integrated.ml_direction = integrated.base_direction
            integrated.ml_confidence = integrated.base_confidence

        # 4. RL position sizing
        try:
            rl_state = self._build_rl_state(signal, price)
            position_size, rl_conf = self.position_sizer.get_position_size(rl_state)
            integrated.position_size = position_size
            integrated.rl_confidence = rl_conf
            integrated.components_used.append("sac_sizer")
        except Exception as e:
            print(f"[INTEGRATED] RL sizing failed: {e}")
            # Fallback to base position
            integrated.position_size = self.config.base_position_size

        # 5. Compute final decision
        integrated = self._compute_final_decision(integrated)

        # 6. Store state
        self.state.last_signal = integrated
        self.state.last_signal_time = timestamp

        # 7. Fire callback
        if self.on_signal:
            self.on_signal(integrated)

        # 8. Execute if needed
        if integrated.final_direction != 0 and integrated.final_size > 0:
            self._execute_signal(integrated, price)

        return integrated

    def on_rentech_signal(self, signal: RenTechSignal, price: float) -> Optional[IntegratedSignal]:
        """
        Process incoming RenTech pattern signal through ML/RL pipeline.

        This method is for signals from RenTechPatternEngine (72001-72099).
        The signal is enhanced with ML and sized with RL before execution.

        Args:
            signal: RenTechSignal from RenTechPatternEngine
            price: Current market price

        Returns:
            IntegratedSignal if action needed
        """
        if not self._running:
            return None

        # Skip neutral signals
        if signal.direction == SignalDirection.NEUTRAL:
            return None

        timestamp = time.time()

        # 1. Create integrated signal from RenTech signal
        integrated = IntegratedSignal(
            timestamp=timestamp,
            source="rentech",
            # Store RenTech signal info
            rentech_direction=signal.direction.value,
            rentech_confidence=signal.confidence,
            rentech_formula_id=signal.formula_id,
            rentech_regime=signal.regime,
            rentech_kelly=signal.kelly_fraction,
            # Use RenTech as base
            base_direction=signal.direction.value,
            base_confidence=signal.confidence,
            base_ensemble_type=f"rentech_{signal.formula_id}",
        )
        integrated.components_used.append("rentech")

        # 2. PIT validation
        if not self.pit_handler.is_valid(signal.timestamp, timestamp):
            print("[INTEGRATED] RenTech signal rejected: PIT validation failed")
            return None

        # 3. ML enhancement
        try:
            # Convert RenTech signal to dict for ML enhancer
            signal_dict = {
                'direction': signal.direction.value,
                'confidence': signal.confidence,
                'regime': signal.regime,
                'kelly_fraction': signal.kelly_fraction,
                'contributing_signals': signal.contributing_signals,
            }
            ml_signal = self.ml_enhancer.enhance_signal(signal_dict, price)
            integrated.ml_direction = ml_signal.direction
            integrated.ml_confidence = ml_signal.confidence
            integrated.ml_features = ml_signal.features
            integrated.components_used.append("ml_enhancer")
        except Exception as e:
            print(f"[INTEGRATED] ML enhancement failed for RenTech: {e}")
            integrated.ml_direction = signal.direction.value
            integrated.ml_confidence = signal.confidence

        # 4. RL position sizing
        try:
            rl_state = self._build_rl_state_from_rentech(signal, price)
            position_size, rl_conf = self.position_sizer.get_position_size(rl_state)
            integrated.position_size = position_size
            integrated.rl_confidence = rl_conf
            integrated.components_used.append("sac_sizer")
        except Exception as e:
            print(f"[INTEGRATED] RL sizing failed for RenTech: {e}")
            # Use Kelly fraction from RenTech signal as fallback
            integrated.position_size = min(signal.kelly_fraction, self.config.base_position_size)

        # 5. Compute final decision
        integrated = self._compute_final_decision_rentech(integrated)

        # 6. Store state
        self.state.last_signal = integrated
        self.state.last_signal_time = timestamp

        # 7. Fire callback
        if self.on_signal:
            self.on_signal(integrated)

        # 8. Execute if needed
        if integrated.final_direction != 0 and integrated.final_size > 0:
            self._execute_signal(integrated, price)

        return integrated

    def _build_rl_state_from_rentech(self, signal: RenTechSignal, price: float) -> TradingState:
        """Build RL state from RenTech signal and system state."""
        # Map regime to volatility estimate
        regime_volatility = {
            'HIGH_VOLATILITY': 0.04,
            'TRENDING_UP': 0.02,
            'TRENDING_DOWN': 0.02,
            'LOW_VOLATILITY': 0.01,
            'CHOPPY': 0.03,
            'UNKNOWN': 0.02,
        }
        volatility = regime_volatility.get(signal.regime, 0.02)

        return TradingState(
            price_change=0.0,  # Not available in RenTech signal
            volatility=volatility,
            volume_ratio=1.0,  # Not available
            flow_imbalance=signal.confidence if signal.direction == SignalDirection.LONG else -signal.confidence,
            position_ratio=self.state.position / self.config.safety.max_position_usd if self.config.safety.max_position_usd > 0 else 0,
            unrealized_pnl=0.0,
            drawdown=self.state.current_drawdown,
            time_in_position=0.0,
            rsi=50.0,  # Default
            macd=0.0,  # Default
            signal_strength=signal.confidence,
        )

    def _compute_final_decision_rentech(self, signal: IntegratedSignal) -> IntegratedSignal:
        """Compute final trading decision from RenTech and ML components."""
        votes = []

        # RenTech signal (primary)
        if signal.rentech_direction != 0:
            # Weight RenTech higher since it's the primary signal source
            votes.append((signal.rentech_direction, signal.rentech_confidence * 1.2))

        # ML enhancement (secondary)
        if signal.ml_direction != 0:
            votes.append((signal.ml_direction, signal.ml_confidence))

        if not votes:
            signal.final_direction = 0
            signal.final_size = 0
            signal.final_confidence = 0
            return signal

        # Check agreement
        if len(votes) == 2 and votes[0][0] != votes[1][0]:
            # Disagreement between RenTech and ML
            # Use RenTech if confidence difference > 0.1, else wait
            if abs(votes[0][1] - votes[1][1]) > 0.1:
                winner = max(votes, key=lambda x: x[1])
                signal.final_direction = winner[0]
                signal.final_confidence = winner[1] * 0.8  # Reduce due to disagreement
            else:
                signal.final_direction = 0
                signal.final_size = 0
                signal.final_confidence = 0
                return signal
        else:
            # Agreement or single signal
            total_weight = sum(conf for _, conf in votes)
            weighted_direction = sum(d * c for d, c in votes) / total_weight
            signal.final_direction = 1 if weighted_direction > 0.1 else (-1 if weighted_direction < -0.1 else 0)
            signal.final_confidence = total_weight / len(votes)

        # Apply RL position sizing with Kelly constraint
        if signal.rentech_kelly > 0:
            # Use Kelly fraction as a cap
            signal.final_size = min(abs(signal.position_size), signal.rentech_kelly * 2)
        else:
            signal.final_size = abs(signal.position_size)

        return signal

    def _build_rl_state(self, signal: Dict, price: float) -> TradingState:
        """Build RL state from signal and system state."""
        return TradingState(
            price_change=signal.get('price_change', 0.0),
            volatility=signal.get('volatility', 0.01),
            volume_ratio=signal.get('volume_ratio', 1.0),
            flow_imbalance=signal.get('flow_imbalance', 0.0),
            position_ratio=self.state.position / self.config.safety.max_position_usd if self.config.safety.max_position_usd > 0 else 0,
            unrealized_pnl=0.0,
            drawdown=self.state.current_drawdown,
            time_in_position=0.0,
            rsi=signal.get('rsi', 50.0),
            macd=signal.get('macd', 0.0),
            signal_strength=signal.get('confidence', 0.5),
        )

    def _compute_final_decision(self, signal: IntegratedSignal) -> IntegratedSignal:
        """Compute final trading decision from all components."""
        # Voting system
        votes = []

        if signal.base_direction != 0:
            votes.append((signal.base_direction, signal.base_confidence))

        if signal.ml_direction != 0:
            votes.append((signal.ml_direction, signal.ml_confidence))

        if not votes:
            signal.final_direction = 0
            signal.final_size = 0
            signal.final_confidence = 0
            return signal

        # Weighted vote
        total_weight = sum(conf for _, conf in votes)
        if total_weight > 0:
            weighted_direction = sum(d * c for d, c in votes) / total_weight
            signal.final_direction = 1 if weighted_direction > 0.1 else (-1 if weighted_direction < -0.1 else 0)
            signal.final_confidence = total_weight / len(votes)
        else:
            signal.final_direction = 0
            signal.final_confidence = 0

        # Apply RL position sizing
        signal.final_size = abs(signal.position_size)

        return signal

    def _execute_signal(self, signal: IntegratedSignal, price: float):
        """Execute trading signal."""
        side = "buy" if signal.final_direction > 0 else "sell"

        # Set price for execution
        self.execution.set_price("BTC/USDT", price)

        # Execute
        result = self.execution.execute(
            symbol="BTC/USDT",
            side=side,
            amount=signal.final_size,
            order_type="market",
        )

        if result.success:
            print(f"[INTEGRATED] Executed {side} {signal.final_size:.6f} @ ${result.executed_price:.2f}")

            # Notify via Telegram
            self.telegram.notify_trade(
                symbol="BTC/USDT",
                side=side,
                amount=signal.final_size,
                price=result.executed_price,
            )
        else:
            print(f"[INTEGRATED] Execution failed: {result.error}")
            self.telegram.notify_error(f"Execution failed: {result.error}")

    def _on_execution(self, result: ExecutionResult):
        """Handle execution result."""
        if result.success:
            self.state.trades_today += 1

            # Update position
            if "buy" in str(result.details):
                self.state.position += result.executed_amount
            else:
                self.state.position -= result.executed_amount

        # Fire callback
        if self.on_trade:
            self.on_trade(result)

    def _get_telegram_status(self) -> TradingStatus:
        """Build Telegram status from current state."""
        uptime = time.time() - self.state.uptime_start if self.state.uptime_start else 0

        return TradingStatus(
            mode=self.mode.value,
            is_running=self.state.is_running,
            kill_switch=self.execution.safety.kill_switch,
            daily_pnl=self.state.daily_pnl,
            total_pnl=self.state.total_pnl,
            trades_today=self.state.trades_today,
            current_drawdown=self.state.current_drawdown,
            consecutive_losses=self.state.consecutive_losses,
            open_positions=1 if self.state.position != 0 else 0,
            total_exposure=abs(self.state.position * 42000),  # Approximate
            uptime_hours=uptime / 3600,
            last_signal_time=self.state.last_signal_time,
            errors_today=len(self.state.errors),
        )

    def run_backtest(self, loader, strategy_callback) -> BacktestResult:
        """
        Run HFT backtest with integrated components.

        Args:
            loader: Order book data loader
            strategy_callback: Strategy function

        Returns:
            BacktestResult
        """
        if self.mode != TradingMode.BACKTEST:
            print("[INTEGRATED] Warning: run_backtest called in non-backtest mode")

        config = BacktestConfig(
            initial_capital=self.state.capital,
            exchange="binance",
            fee_rate=0.001,
            max_position=self.config.safety.max_position_usd / 42000,  # Convert to BTC
            use_latency=True,
        )

        backtester = HFTBacktester(config, loader)

        # Wrap strategy with ML enhancement
        def enhanced_strategy(book: OrderBookSnapshot, state) -> List:
            # Get base orders from strategy
            base_orders = strategy_callback(book, state)

            # Enhance with ML (if applicable)
            # For now, pass through
            return base_orders

        backtester.set_strategy(enhanced_strategy)

        return backtester.run()

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        # Get last RenTech signal info if available
        rentech_info = {}
        if self.state.last_signal and self.state.last_signal.source == "rentech":
            rentech_info = {
                'last_formula_id': self.state.last_signal.rentech_formula_id,
                'last_confidence': self.state.last_signal.rentech_confidence,
                'last_regime': self.state.last_signal.rentech_regime,
                'last_kelly': self.state.last_signal.rentech_kelly,
            }

        return {
            'mode': self.mode.value,
            'is_running': self.state.is_running,
            'capital': self.state.capital,
            'position': self.state.position,
            'daily_pnl': self.state.daily_pnl,
            'total_pnl': self.state.total_pnl,
            'trades_today': self.state.trades_today,
            'execution_stats': self.execution.get_stats(),
            'safety_stats': self.execution.safety.get_stats(),
            'components': {
                'ml_enhancer': 'active',
                'position_sizer': 'sac',
                'pit_handler': 'active',
                'rentech_engine': 'supported',  # via on_rentech_signal()
            },
            'rentech': rentech_info,
        }


def create_trading_system(mode: str = "paper",
                          config_path: Optional[str] = None) -> IntegratedTradingSystem:
    """
    Factory function to create trading system.

    Args:
        mode: Trading mode ("backtest", "paper", "dry_run", "sandbox", "live")
        config_path: Path to config file

    Returns:
        IntegratedTradingSystem
    """
    mode_map = {
        'backtest': TradingMode.BACKTEST,
        'paper': TradingMode.PAPER,
        'dry_run': TradingMode.DRY_RUN,
        'sandbox': TradingMode.SANDBOX,
        'live': TradingMode.LIVE,
    }

    trading_mode = mode_map.get(mode.lower(), TradingMode.PAPER)
    config = load_config(config_path) if config_path else None

    return IntegratedTradingSystem(mode=trading_mode, config=config)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("Integrated Trading System Demo")
    print("=" * 50)

    # Create system in paper mode
    system = create_trading_system(mode="paper")

    # Start system
    system.start()

    # Simulate some signals
    print("\nSimulating flow signals...")

    test_signals = [
        {'direction': 1, 'confidence': 0.75, 'flow_imbalance': 0.3, 'volatility': 0.02},
        {'direction': -1, 'confidence': 0.60, 'flow_imbalance': -0.2, 'volatility': 0.015},
        {'direction': 1, 'confidence': 0.80, 'flow_imbalance': 0.5, 'volatility': 0.025},
    ]

    price = 42000.0
    for i, signal in enumerate(test_signals):
        print(f"\nSignal {i+1}:")
        result = system.on_flow_signal(signal, price)
        if result:
            print(f"  Direction: {result.final_direction}")
            print(f"  Size: {result.final_size:.6f}")
            print(f"  Confidence: {result.final_confidence:.2f}")
            print(f"  Components: {result.components_used}")

        # Simulate price movement
        price += 100 if signal['direction'] > 0 else -100

    # Get stats
    print(f"\nSystem Stats:")
    stats = system.get_stats()
    print(f"  Mode: {stats['mode']}")
    print(f"  Running: {stats['is_running']}")
    print(f"  Trades today: {stats['trades_today']}")

    # Stop system
    system.stop()
