"""
RenTech Performance Monitor
============================

Real-time monitoring dashboard for RenTech trading formulas.

Tracks:
- Win rate per formula (10/50/100 trade EMA)
- SPRT status (confirming/rejecting/neutral)
- Current drawdown vs limit
- Edge decay CUSUM
- Active positions and P&L
- Equity curve

Outputs to JSON for web dashboard integration.

Created: 2025-12-16
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import json
import time
import logging
from pathlib import Path
from datetime import datetime

from .rentech_validator import RenTechValidator, FormulaStatus
from .rentech_evaluator import RenTechSignal, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class PositionSnapshot:
    """Current state of an open position."""
    formula_id: str
    formula_name: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    entry_time: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    hold_days: int
    days_held: float
    target_exit_time: float


@dataclass
class FormulaPerformance:
    """Performance metrics for a single formula."""
    formula_id: str
    formula_name: str
    status: str

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Win rates
    overall_win_rate: float = 0.0
    ema_10_win_rate: float = 0.5
    ema_20_win_rate: float = 0.5

    # Returns
    total_return: float = 0.0
    avg_return: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    # Validation metrics
    sprt_status: str = "CONTINUE"
    sprt_llr: float = 0.0
    cusum_value: float = 0.0
    position_multiplier: float = 1.0

    # Historical baseline
    baseline_win_rate: float = 1.0
    baseline_avg_return: float = 20.0


@dataclass
class EquityPoint:
    """Point on the equity curve."""
    timestamp: float
    equity: float
    drawdown: float
    high_water_mark: float


@dataclass
class MonitorSnapshot:
    """Complete monitoring snapshot."""
    timestamp: float
    timestamp_str: str

    # Overall stats
    total_capital: float
    current_equity: float
    unrealized_pnl: float
    realized_pnl: float
    current_drawdown: float
    max_drawdown: float
    high_water_mark: float

    # Active positions
    num_active_positions: int
    positions: List[PositionSnapshot]

    # Per-formula performance
    formula_performance: Dict[str, FormulaPerformance]

    # System status
    system_status: str  # "ACTIVE", "PAUSED", "STOPPED"
    last_signal_time: float
    signals_today: int
    trades_today: int


class RenTechMonitor:
    """
    Real-time performance monitor for RenTech formulas.

    Integrates with validator and tracks all trading activity.
    """

    def __init__(
        self,
        validator: RenTechValidator,
        initial_capital: float = 100.0,
        output_file: str = "data/rentech_monitor.json"
    ):
        """
        Initialize monitor.

        Args:
            validator: RenTech validator instance
            initial_capital: Starting capital
            output_file: Path for JSON output
        """
        self.validator = validator
        self.initial_capital = initial_capital
        self.output_file = Path(output_file)

        # Equity tracking
        self.current_equity = initial_capital
        self.high_water_mark = initial_capital
        self.max_drawdown = 0.0
        self.realized_pnl = 0.0

        # Active positions
        self.active_positions: Dict[str, dict] = {}

        # Trade history
        self.trade_returns: Dict[str, List[float]] = {}  # formula_id -> returns

        # Equity curve
        self.equity_curve: List[EquityPoint] = [
            EquityPoint(
                timestamp=time.time(),
                equity=initial_capital,
                drawdown=0.0,
                high_water_mark=initial_capital
            )
        ]

        # Daily tracking
        self.today_start = self._get_day_start()
        self.signals_today = 0
        self.trades_today = 0

        # System status
        self.system_status = "ACTIVE"
        self.last_signal_time = 0.0
        self.circuit_breaker_triggered = False
        self.consecutive_losses = 0

    def _get_day_start(self) -> float:
        """Get timestamp for start of today."""
        now = datetime.now()
        return datetime(now.year, now.month, now.day).timestamp()

    def on_signal(self, signal: RenTechSignal):
        """Record a new signal."""
        self.last_signal_time = signal.timestamp

        # Reset daily counters if new day
        day_start = self._get_day_start()
        if day_start > self.today_start:
            self.today_start = day_start
            self.signals_today = 0
            self.trades_today = 0

        self.signals_today += 1

    def on_position_opened(
        self,
        signal: RenTechSignal,
        position_size: float,
        leverage: float
    ):
        """Record a new position being opened."""
        self.active_positions[signal.formula_id] = {
            "signal": signal,
            "position_size": position_size,
            "leverage": leverage,
            "entry_time": time.time(),
        }
        self.trades_today += 1

        # Initialize trade returns list if needed
        if signal.formula_id not in self.trade_returns:
            self.trade_returns[signal.formula_id] = []

    def on_position_closed(
        self,
        formula_id: str,
        exit_price: float,
        pnl: float,
        pnl_pct: float
    ):
        """Record a position being closed."""
        if formula_id in self.active_positions:
            del self.active_positions[formula_id]

        # Track return
        if formula_id in self.trade_returns:
            self.trade_returns[formula_id].append(pnl_pct)
        else:
            self.trade_returns[formula_id] = [pnl_pct]

        # Update equity
        self.realized_pnl += pnl
        self.current_equity = self.initial_capital + self.realized_pnl

        # Update high water mark and drawdown
        if self.current_equity > self.high_water_mark:
            self.high_water_mark = self.current_equity

        current_dd = (self.high_water_mark - self.current_equity) / self.high_water_mark
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd

        # Track consecutive losses
        if pnl_pct < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 5:
                self.circuit_breaker_triggered = True
                self.system_status = "PAUSED"
                logger.warning("Circuit breaker triggered: 5 consecutive losses")
        else:
            self.consecutive_losses = 0
            if self.circuit_breaker_triggered:
                # Reset after a win
                self.circuit_breaker_triggered = False
                self.system_status = "ACTIVE"

        # Add equity point
        self.equity_curve.append(EquityPoint(
            timestamp=time.time(),
            equity=self.current_equity,
            drawdown=current_dd,
            high_water_mark=self.high_water_mark
        ))

    def update_prices(self, current_prices: Dict[str, float]):
        """
        Update current prices for unrealized P&L calculation.

        Args:
            current_prices: Dict mapping formula_id to current BTC price
        """
        for formula_id, pos in self.active_positions.items():
            if "current_price" not in current_prices:
                continue

            price = current_prices.get("current_price", pos["signal"].entry_price)
            pos["current_price"] = price

            # Calculate unrealized P&L
            entry_price = pos["signal"].entry_price
            direction = 1 if pos["signal"].direction == SignalDirection.LONG else -1
            pnl_pct = direction * (price / entry_price - 1) * 100

            pos["unrealized_pnl_pct"] = pnl_pct
            pos["unrealized_pnl"] = pos["position_size"] * (pnl_pct / 100) * pos["leverage"]

    def get_snapshot(self) -> MonitorSnapshot:
        """Get current monitoring snapshot."""
        now = time.time()

        # Calculate unrealized P&L
        unrealized_pnl = sum(
            pos.get("unrealized_pnl", 0)
            for pos in self.active_positions.values()
        )

        # Calculate current drawdown
        total_equity = self.current_equity + unrealized_pnl
        if total_equity > self.high_water_mark:
            current_dd = 0.0
        else:
            current_dd = (self.high_water_mark - total_equity) / self.high_water_mark

        # Build position snapshots
        positions = []
        for formula_id, pos in self.active_positions.items():
            signal = pos["signal"]
            entry_time = pos.get("entry_time", signal.timestamp)
            days_held = (now - entry_time) / (24 * 60 * 60)

            positions.append(PositionSnapshot(
                formula_id=formula_id,
                formula_name=signal.formula_name,
                direction=signal.direction.name,
                entry_price=signal.entry_price,
                entry_time=entry_time,
                current_price=pos.get("current_price", signal.entry_price),
                unrealized_pnl=pos.get("unrealized_pnl", 0),
                unrealized_pnl_pct=pos.get("unrealized_pnl_pct", 0),
                hold_days=signal.hold_days,
                days_held=days_held,
                target_exit_time=entry_time + signal.hold_days * 24 * 60 * 60
            ))

        # Build formula performance
        formula_performance = {}
        for formula_id in self.validator.formula_states:
            state = self.validator.formula_states[formula_id]
            returns = self.trade_returns.get(formula_id, [])

            perf = FormulaPerformance(
                formula_id=formula_id,
                formula_name=formula_id,  # Would need lookup for actual name
                status=state.status.value,
                total_trades=state.live_trades,
                winning_trades=state.live_wins,
                losing_trades=state.live_losses,
                overall_win_rate=state.live_win_rate,
                ema_10_win_rate=state.ema_win_rate_10,
                ema_20_win_rate=state.ema_win_rate_20,
                total_return=state.live_total_return,
                avg_return=state.live_total_return / state.live_trades if state.live_trades > 0 else 0,
                best_trade=max(returns) if returns else 0,
                worst_trade=min(returns) if returns else 0,
                sprt_status=state.sprt_decision,
                sprt_llr=state.sprt_log_likelihood,
                cusum_value=state.cusum_value,
                position_multiplier=state.position_multiplier,
                baseline_win_rate=state.baseline_win_rate,
                baseline_avg_return=state.baseline_avg_return
            )
            formula_performance[formula_id] = perf

        return MonitorSnapshot(
            timestamp=now,
            timestamp_str=datetime.fromtimestamp(now).isoformat(),
            total_capital=self.initial_capital,
            current_equity=total_equity,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.realized_pnl,
            current_drawdown=current_dd,
            max_drawdown=self.max_drawdown,
            high_water_mark=self.high_water_mark,
            num_active_positions=len(self.active_positions),
            positions=positions,
            formula_performance=formula_performance,
            system_status=self.system_status,
            last_signal_time=self.last_signal_time,
            signals_today=self.signals_today,
            trades_today=self.trades_today
        )

    def save_snapshot(self):
        """Save current snapshot to JSON file."""
        try:
            snapshot = self.get_snapshot()

            # Convert to dict for JSON serialization
            data = {
                "timestamp": snapshot.timestamp,
                "timestamp_str": snapshot.timestamp_str,
                "total_capital": snapshot.total_capital,
                "current_equity": snapshot.current_equity,
                "unrealized_pnl": snapshot.unrealized_pnl,
                "realized_pnl": snapshot.realized_pnl,
                "current_drawdown": snapshot.current_drawdown,
                "max_drawdown": snapshot.max_drawdown,
                "high_water_mark": snapshot.high_water_mark,
                "num_active_positions": snapshot.num_active_positions,
                "positions": [asdict(p) for p in snapshot.positions],
                "formula_performance": {
                    k: asdict(v) for k, v in snapshot.formula_performance.items()
                },
                "system_status": snapshot.system_status,
                "last_signal_time": snapshot.last_signal_time,
                "signals_today": snapshot.signals_today,
                "trades_today": snapshot.trades_today,
                "equity_curve_length": len(self.equity_curve),
            }

            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved monitor snapshot to {self.output_file}")

        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")

    def print_status(self):
        """Print current status to console."""
        snapshot = self.get_snapshot()

        print("\n" + "=" * 70)
        print("  RENTECH MONITOR STATUS")
        print("=" * 70)

        print(f"\nSystem: {snapshot.system_status}")
        print(f"Time: {snapshot.timestamp_str}")

        print(f"\nEquity:")
        print(f"  Initial:    ${snapshot.total_capital:,.2f}")
        print(f"  Current:    ${snapshot.current_equity:,.2f}")
        print(f"  Realized:   ${snapshot.realized_pnl:+,.2f}")
        print(f"  Unrealized: ${snapshot.unrealized_pnl:+,.2f}")
        print(f"  Drawdown:   {snapshot.current_drawdown:.1%} (max: {snapshot.max_drawdown:.1%})")

        print(f"\nPositions: {snapshot.num_active_positions}")
        for pos in snapshot.positions:
            print(f"  {pos.formula_id}: {pos.direction} @ ${pos.entry_price:,.0f}")
            print(f"    P&L: {pos.unrealized_pnl_pct:+.1f}% (${pos.unrealized_pnl:+,.2f})")
            print(f"    Days: {pos.days_held:.1f} / {pos.hold_days}")

        print(f"\nFormula Performance:")
        for fid, perf in snapshot.formula_performance.items():
            print(f"\n  {fid}: {perf.status}")
            print(f"    Trades: {perf.total_trades} ({perf.winning_trades}W / {perf.losing_trades}L)")
            print(f"    Win Rate: {perf.overall_win_rate:.0%} (EMA-10: {perf.ema_10_win_rate:.0%})")
            print(f"    Total Return: {perf.total_return:+.1f}%")
            print(f"    SPRT: {perf.sprt_status} (LLR: {perf.sprt_llr:.2f})")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    # Test the monitor
    print("Testing RenTech Monitor")
    print("=" * 60)

    from .rentech_validator import RenTechValidator
    from .rentech_evaluator import RenTechSignal, SignalDirection

    # Create validator and monitor
    validator = RenTechValidator(state_file="data/test_validator_state.json")
    monitor = RenTechMonitor(
        validator=validator,
        initial_capital=100.0,
        output_file="data/test_monitor.json"
    )

    # Initialize a formula
    validator.initialize_formula("RENTECH_001", 1.0, 26.37, 18)

    # Simulate a signal
    signal = RenTechSignal(
        formula_id="RENTECH_001",
        formula_name="EXTREME_ANOMALY_LONG",
        direction=SignalDirection.LONG,
        confidence=0.85,
        hold_days=30,
        entry_price=50000,
        timestamp=time.time()
    )

    monitor.on_signal(signal)
    monitor.on_position_opened(signal, position_size=5.0, leverage=10.0)

    # Update price
    monitor.update_prices({"current_price": 52500})

    # Print status
    monitor.print_status()

    # Save snapshot
    monitor.save_snapshot()
    print(f"\nSnapshot saved to: {monitor.output_file}")
