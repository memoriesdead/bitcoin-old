"""
Main Simulation Engine - Controller.

Integrates:
- Formula Engine (31001-31199)
- Trade Logger (SQLite audit trail)
- Exchange Verifier (cross-reference)
- Historical Replayer / Live Feed

Pure math execution - no second guessing.
"""

import time
from typing import Optional, Dict, List
from dataclasses import dataclass
import numpy as np

from .types import FormulaSignal, Position, TradeResult, ExitReason
from .formula_engine import ProductionFormulaEngine, PRODUCTION_FORMULA_IDS
from .trade_logger import TradeLogger
from .verifier import ExchangeVerifier, HistoricalVerifier
from .historical import HistoricalReplayer, HistoricalTick
from .live import LivePaperTrader, LiveTick


@dataclass
class EngineConfig:
    """Engine configuration."""
    initial_capital: float = 10000.0
    kelly_fraction: float = 0.25
    max_positions: int = 5
    max_position_pct: float = 0.20
    formula_ids: List[int] = None
    db_path: str = "data/simulation_trades.db"

    def __post_init__(self):
        if self.formula_ids is None:
            self.formula_ids = PRODUCTION_FORMULA_IDS


class SimulationEngine:
    """
    Main simulation engine controller.

    Features:
    - Pure math execution (no second guessing)
    - Quarter Kelly position sizing
    - Full audit trail
    - Historical replay or live paper trading
    - Exchange cross-reference verification
    """

    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()

        # Core components
        self.formula_engine = ProductionFormulaEngine(
            formula_ids=self.config.formula_ids,
            kelly_fraction=self.config.kelly_fraction,
        )
        self.logger = TradeLogger(self.config.db_path)

        # Verifier (set based on mode)
        self.verifier = None

        # State
        self.capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.returns: List[float] = []

        # Tracking
        self.current_price = 0.0
        self.tick_count = 0

    def run_historical(
        self,
        start_date: str = None,
        end_date: str = None,
        speed: float = 0,
        db_path: str = "data/unified_bitcoin.db"
    ) -> Dict:
        """
        Run historical simulation.

        Args:
            start_date: Start date (YYYY-MM-DD), None for earliest
            end_date: End date (YYYY-MM-DD), None for latest
            speed: 0=instant, N=Nx speed
            db_path: Path to historical data

        Returns:
            Dict with simulation results
        """
        print("=" * 70)
        print("RENTECH 1:1 SIMULATION - HISTORICAL MODE")
        print("=" * 70)

        # Initialize
        self.verifier = HistoricalVerifier()
        replayer = HistoricalReplayer(db_path)

        # Create session
        session = self.logger.create_session(
            mode="historical",
            initial_capital=self.config.initial_capital,
            kelly_fraction=self.config.kelly_fraction,
            formula_ids=self.config.formula_ids,
        )

        # Track initial equity
        self.logger.log_equity(self.capital)

        # Run replay
        def on_tick(tick: HistoricalTick):
            self._process_tick(tick.timestamp, tick.price, tick.to_dict())

        stats = replayer.replay(
            callback=on_tick,
            start_date=start_date,
            end_date=end_date,
            speed=speed,
            progress_interval=500,
        )

        # Close all positions at end
        self._close_all_positions(stats.get('last_price', self.current_price), time.time())

        # Calculate final stats
        results = self._calculate_results(stats)

        # Close session
        self.logger.close_session(
            final_capital=self.capital,
            max_drawdown=results['max_drawdown_pct'],
            sharpe_ratio=results.get('sharpe_ratio'),
        )

        self._print_results(results)

        return results

    def run_live(
        self,
        duration_seconds: float = 3600,
        poll_interval: float = 1.0
    ) -> Dict:
        """
        Run live paper trading.

        Args:
            duration_seconds: Duration in seconds (None = indefinite)
            poll_interval: Price poll interval in seconds

        Returns:
            Dict with simulation results
        """
        print("=" * 70)
        print("RENTECH 1:1 SIMULATION - LIVE PAPER TRADING")
        print("=" * 70)

        # Initialize
        self.verifier = ExchangeVerifier()
        paper_trader = LivePaperTrader(poll_interval=poll_interval)

        # Create session
        session = self.logger.create_session(
            mode="live",
            initial_capital=self.config.initial_capital,
            kelly_fraction=self.config.kelly_fraction,
            formula_ids=self.config.formula_ids,
        )

        # Track initial equity
        self.logger.log_equity(self.capital)

        start_time = time.time()

        # Run live
        def on_tick(tick: LiveTick):
            self._process_tick(tick.timestamp, tick.price, tick.to_dict())

        stats = paper_trader.run_blocking(
            callback=on_tick,
            duration_seconds=duration_seconds,
            progress_interval=60,
        )

        # Close all positions at end
        self._close_all_positions(stats.get('last_price', self.current_price), time.time())

        # Calculate final stats
        results = self._calculate_results(stats)

        # Close session
        self.logger.close_session(
            final_capital=self.capital,
            max_drawdown=results['max_drawdown_pct'],
            sharpe_ratio=results.get('sharpe_ratio'),
        )

        self._print_results(results)

        return results

    def _process_tick(self, timestamp: float, price: float, data: Dict = None):
        """
        Process single price tick.

        1. Update price tracking
        2. Check existing positions for exit
        3. Run formulas for new signals
        4. Execute signals (no second guessing)
        """
        self.current_price = price
        self.tick_count += 1

        # Update historical verifier
        if isinstance(self.verifier, HistoricalVerifier):
            self.verifier.set_price(price)

        # Check exits on existing positions
        self._check_exits(price, timestamp)

        # Run formulas for new signals
        signals = self.formula_engine.update(price, data)

        # Execute signals - PURE MATH, NO SECOND GUESSING
        for signal in signals:
            self._execute_signal(signal, price, timestamp)

        # Track returns for Sharpe calculation
        if len(self.returns) > 0:
            prev_capital = self.capital / (1 + self.returns[-1]) if self.returns[-1] != 0 else self.capital
        else:
            prev_capital = self.config.initial_capital

        if prev_capital > 0:
            ret = (self.capital / prev_capital) - 1
            self.returns.append(ret)

        # Log equity periodically
        if self.tick_count % 100 == 0:
            drawdown = ((self.peak_capital - self.capital) / self.peak_capital) * 100 if self.peak_capital > 0 else 0
            self.logger.log_equity(self.capital, drawdown)

    def _check_exits(self, price: float, timestamp: float):
        """Check all positions for exit conditions."""
        positions_to_close = []

        for trade_id, position in self.positions.items():
            exit_reason = position.check_exit(price, timestamp)

            if exit_reason:
                positions_to_close.append((trade_id, position, exit_reason))

        # Close positions
        for trade_id, position, exit_reason in positions_to_close:
            self._close_position(trade_id, position, price, timestamp, exit_reason.value)

    def _execute_signal(self, signal: FormulaSignal, price: float, timestamp: float):
        """
        Execute trading signal.

        NO SECOND GUESSING - when formula fires, trade executes.
        """
        # Check position limits
        if len(self.positions) >= self.config.max_positions:
            return

        # Check if already have position from this formula
        for pos in self.positions.values():
            if pos.formula_id == signal.formula_id:
                return  # Already positioned

        # Calculate position size
        position_size_pct = min(signal.position_size_pct, self.config.max_position_pct)
        position_usd = self.capital * position_size_pct

        if position_usd < 10:  # Minimum position
            return

        # Get execution price (with slippage estimation)
        entry_price = self.verifier.estimate_execution_price(
            signal_price=price,
            direction=signal.direction,
            size_btc=position_usd / price,
        )

        # Create position
        position = Position(
            trade_id="",  # Will be assigned by logger
            formula_id=signal.formula_id,
            formula_name=signal.formula_name,
            direction=signal.direction,
            entry_price=entry_price,
            entry_timestamp=timestamp,
            position_size_pct=position_size_pct,
            position_btc=position_usd / entry_price,
            position_usd=position_usd,
            stop_loss_price=entry_price * (1 - signal.stop_loss_pct) if signal.direction == 1 else entry_price * (1 + signal.stop_loss_pct),
            take_profit_price=entry_price * (1 + signal.take_profit_pct) if signal.direction == 1 else entry_price * (1 - signal.take_profit_pct),
            max_exit_time=timestamp + signal.max_hold_seconds,
            signal_strength=signal.confidence,
        )

        # Log signal
        trade_id = self.logger.log_signal(signal, price)
        position.trade_id = trade_id

        # Log entry
        self.logger.log_entry(
            trade_id=trade_id,
            entry_price=entry_price,
            position_btc=position.position_btc,
            position_usd=position.position_usd,
            entry_timestamp=timestamp,
        )

        # Log verification
        if isinstance(self.verifier, ExchangeVerifier):
            exchange_price = self.verifier.get_current_price()
            self.logger.log_verification(trade_id, exchange_price, entry_price)

        # Add to active positions
        self.positions[trade_id] = position

    def _close_position(
        self,
        trade_id: str,
        position: Position,
        price: float,
        timestamp: float,
        exit_reason: str
    ):
        """Close a position and log results."""
        # Get exit price with slippage
        exit_price = self.verifier.estimate_execution_price(
            signal_price=price,
            direction=-position.direction,  # Opposite direction for exit
            size_btc=position.position_btc,
        )

        # Calculate PnL
        pnl_usd, pnl_pct = position.calculate_pnl(exit_price)

        # Update capital
        self.capital += pnl_usd

        # Update peak
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        # Log exit
        self.logger.log_exit(
            trade_id=trade_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_timestamp=timestamp,
        )

        # Update prediction verification
        prediction_correct = self.verifier.verify_prediction(
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
        )
        self.logger.update_prediction_result(trade_id, prediction_correct)

        # Remove from active positions
        del self.positions[trade_id]

    def _close_all_positions(self, price: float, timestamp: float):
        """Close all remaining positions at end of simulation."""
        for trade_id, position in list(self.positions.items()):
            self._close_position(trade_id, position, price, timestamp, "session_end")

    def _calculate_results(self, stats: Dict) -> Dict:
        """Calculate comprehensive simulation results."""
        session = self.logger.get_session_summary()
        formula_perf = self.logger.get_formula_performance()

        # Calculate Sharpe ratio
        sharpe = 0.0
        if len(self.returns) > 1:
            returns = np.array(self.returns)
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized

        # Max drawdown
        max_drawdown = ((self.peak_capital - self.capital) / self.peak_capital) * 100 if self.peak_capital > 0 else 0

        # Get session stats from logger
        session_obj = self.logger.session
        total_trades = session_obj.total_trades if session_obj else 0
        total_wins = session_obj.total_wins if session_obj else 0
        total_losses = session_obj.total_losses if session_obj else 0
        win_rate = session_obj.win_rate if session_obj else 0

        return {
            'initial_capital': self.config.initial_capital,
            'final_capital': self.capital,
            'total_pnl_usd': self.capital - self.config.initial_capital,
            'total_pnl_pct': ((self.capital / self.config.initial_capital) - 1) * 100,
            'total_trades': total_trades,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': win_rate or 0,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'formula_performance': formula_perf,
            'ticks_processed': self.tick_count,
            'replay_stats': stats,
        }

    def _print_results(self, results: Dict):
        """Print formatted results."""
        print("\n" + "=" * 70)
        print("SIMULATION RESULTS")
        print("=" * 70)

        print(f"\nCAPITAL:")
        print(f"  Initial:    ${results['initial_capital']:,.2f}")
        print(f"  Final:      ${results['final_capital']:,.2f}")
        print(f"  Total PnL:  ${results['total_pnl_usd']:+,.2f} ({results['total_pnl_pct']:+.2f}%)")

        print(f"\nTRADES:")
        print(f"  Total:      {results['total_trades']}")
        print(f"  Wins:       {results['total_wins']} ({results['win_rate']*100:.1f}%)" if results['total_trades'] > 0 else "  Wins: N/A")
        print(f"  Losses:     {results['total_losses']}")

        print(f"\nRISK METRICS:")
        print(f"  Sharpe Ratio:   {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:   {results['max_drawdown_pct']:.2f}%")

        print(f"\nFORMULA BREAKDOWN:")
        for perf in results.get('formula_performance', []):
            wr = perf.get('win_rate', 0) * 100 if perf.get('win_rate') else 0
            print(f"  {perf['formula_id']} {perf['formula_name']}: "
                  f"{perf['trades']} trades, {wr:.1f}% WR, ${perf['total_pnl']:+,.2f}")

        print("\n" + "=" * 70)

    def reset(self):
        """Reset engine state for new simulation."""
        self.capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
        self.positions = {}
        self.returns = []
        self.current_price = 0.0
        self.tick_count = 0
        self.formula_engine.reset()
