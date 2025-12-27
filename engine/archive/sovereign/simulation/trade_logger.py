"""
Trade logger with full audit trail.

Logs every trade with:
- Timestamps (signal, entry, exit)
- Prices (signal, entry, exit)
- Formula details (ID, name, confidence)
- PnL calculation
- Exchange cross-reference
"""

import time
from typing import Optional, List, Dict
from .database import SimulationDatabase
from .types import (
    SimulationSession,
    FormulaSignal,
    Position,
    TradeResult,
)


class TradeLogger:
    """
    Full audit trail trade logging.

    Every trade is logged to SQLite with complete details.
    """

    def __init__(self, db_path: str = "data/simulation_trades.db"):
        self.db = SimulationDatabase(db_path)
        self.session: Optional[SimulationSession] = None
        self.active_trades: Dict[str, Dict] = {}

    def create_session(
        self,
        mode: str,
        initial_capital: float,
        kelly_fraction: float = 0.25,
        formula_ids: List[int] = None
    ) -> SimulationSession:
        """Create new simulation session."""
        session = SimulationSession.create(
            mode=mode,
            initial_capital=initial_capital,
            kelly_fraction=kelly_fraction,
            formula_ids=formula_ids or [],
        )

        self.db.create_session(
            session_id=session.session_id,
            mode=session.mode,
            start_timestamp=session.start_timestamp,
            initial_capital=session.initial_capital,
            kelly_fraction=session.kelly_fraction,
            formula_ids=session.formula_ids,
        )

        self.session = session
        print(f"[SESSION] Created session {session.session_id} ({mode} mode)")
        return session

    def log_signal(
        self,
        signal: FormulaSignal,
        price: float
    ) -> str:
        """Log signal when formula fires.

        Returns:
            trade_id for tracking
        """
        if not self.session:
            raise RuntimeError("No active session. Call create_session first.")

        import uuid
        trade_id = str(uuid.uuid4())[:12]

        self.db.insert_trade(
            trade_id=trade_id,
            session_id=self.session.session_id,
            mode=self.session.mode,
            signal_timestamp=signal.timestamp,
            formula_id=signal.formula_id,
            formula_name=signal.formula_name,
            direction=signal.direction,
            signal_strength=signal.confidence,
            signal_price=price,
            position_size_pct=signal.position_size_pct,
            stop_loss_pct=signal.stop_loss_pct,
            take_profit_pct=signal.take_profit_pct,
        )

        # Track active trade
        self.active_trades[trade_id] = {
            'signal': signal,
            'signal_price': price,
        }

        direction_str = "LONG" if signal.direction == 1 else "SHORT"
        print(f"[SIGNAL] {trade_id}: {signal.formula_name} -> {direction_str} "
              f"@ ${price:,.2f} (conf: {signal.confidence:.2%})")

        return trade_id

    def log_entry(
        self,
        trade_id: str,
        entry_price: float,
        position_btc: float,
        position_usd: float,
        entry_timestamp: float = None
    ):
        """Log trade entry execution."""
        entry_timestamp = entry_timestamp or time.time()

        self.db.update_trade_entry(
            trade_id=trade_id,
            entry_timestamp=entry_timestamp,
            entry_price=entry_price,
            position_btc=position_btc,
            position_usd=position_usd,
        )

        if trade_id in self.active_trades:
            self.active_trades[trade_id]['entry_price'] = entry_price
            self.active_trades[trade_id]['entry_timestamp'] = entry_timestamp
            self.active_trades[trade_id]['position_btc'] = position_btc
            self.active_trades[trade_id]['position_usd'] = position_usd

        print(f"[ENTRY] {trade_id}: Entered @ ${entry_price:,.2f} "
              f"({position_btc:.6f} BTC = ${position_usd:,.2f})")

    def log_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        pnl_usd: float,
        pnl_pct: float,
        exit_timestamp: float = None
    ) -> TradeResult:
        """Log trade exit and return result."""
        exit_timestamp = exit_timestamp or time.time()

        self.db.update_trade_exit(
            trade_id=trade_id,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
        )

        # Get full trade data
        trade_data = self.db.get_trade(trade_id)

        # Update formula performance
        self.db.update_formula_performance(
            session_id=self.session.session_id,
            formula_id=trade_data['formula_id'],
            formula_name=trade_data['formula_name'],
            is_win=pnl_usd > 0,
            pnl=pnl_usd,
        )

        # Update session stats
        self.session.update_stats(TradeResult(
            trade_id=trade_id,
            session_id=self.session.session_id,
            formula_id=trade_data['formula_id'],
            formula_name=trade_data['formula_name'],
            direction=trade_data['direction'],
            signal_strength=trade_data['signal_strength'],
            signal_timestamp=trade_data['signal_timestamp'],
            entry_timestamp=trade_data['entry_timestamp'],
            exit_timestamp=exit_timestamp,
            signal_price=trade_data['signal_price'],
            entry_price=trade_data['entry_price'],
            exit_price=exit_price,
            position_size_pct=trade_data['position_size_pct'],
            position_btc=trade_data['position_btc'] or 0,
            position_usd=trade_data['position_usd'] or 0,
            stop_loss_pct=trade_data['stop_loss_pct'],
            take_profit_pct=trade_data['take_profit_pct'],
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
        ))

        # Remove from active trades
        if trade_id in self.active_trades:
            del self.active_trades[trade_id]

        # Print result
        pnl_sign = "+" if pnl_usd >= 0 else ""
        result_emoji = "WIN" if pnl_usd > 0 else "LOSS"
        print(f"[EXIT] {trade_id}: {exit_reason.upper()} @ ${exit_price:,.2f} "
              f"-> {result_emoji} {pnl_sign}${pnl_usd:,.2f} ({pnl_sign}{pnl_pct:.2f}%)")

        return TradeResult(
            trade_id=trade_id,
            session_id=self.session.session_id,
            formula_id=trade_data['formula_id'],
            formula_name=trade_data['formula_name'],
            direction=trade_data['direction'],
            signal_strength=trade_data['signal_strength'],
            signal_timestamp=trade_data['signal_timestamp'],
            entry_timestamp=trade_data['entry_timestamp'],
            exit_timestamp=exit_timestamp,
            signal_price=trade_data['signal_price'],
            entry_price=trade_data['entry_price'],
            exit_price=exit_price,
            position_size_pct=trade_data['position_size_pct'],
            position_btc=trade_data['position_btc'] or 0,
            position_usd=trade_data['position_usd'] or 0,
            stop_loss_pct=trade_data['stop_loss_pct'],
            take_profit_pct=trade_data['take_profit_pct'],
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
        )

    def log_verification(
        self,
        trade_id: str,
        exchange_price: float,
        entry_price: float
    ):
        """Log exchange cross-reference verification."""
        slippage = ((entry_price - exchange_price) / exchange_price) * 100

        # Determine if prediction was correct (need to check later)
        trade_data = self.db.get_trade(trade_id)
        direction = trade_data['direction'] if trade_data else 0

        self.db.update_trade_verification(
            trade_id=trade_id,
            exchange_price=exchange_price,
            slippage=slippage,
            prediction_correct=False,  # Updated on exit
        )

        print(f"[VERIFY] {trade_id}: Exchange price ${exchange_price:,.2f}, "
              f"slippage {slippage:+.3f}%")

    def update_prediction_result(
        self,
        trade_id: str,
        prediction_correct: bool
    ):
        """Update whether prediction was correct after exit."""
        trade_data = self.db.get_trade(trade_id)
        if trade_data and trade_data.get('exchange_price_at_signal'):
            self.db.update_trade_verification(
                trade_id=trade_id,
                exchange_price=trade_data['exchange_price_at_signal'],
                slippage=trade_data.get('slippage_estimated', 0),
                prediction_correct=prediction_correct,
            )

    def log_equity(self, capital: float, drawdown_pct: float = 0.0):
        """Log equity curve data point."""
        if self.session:
            self.db.add_equity_point(
                session_id=self.session.session_id,
                timestamp=time.time(),
                capital=capital,
                drawdown_pct=drawdown_pct,
            )

    def close_session(
        self,
        final_capital: float,
        max_drawdown: float = 0.0,
        sharpe_ratio: float = None
    ):
        """Close session and update final stats."""
        if not self.session:
            return

        self.session.end_timestamp = time.time()
        self.session.final_capital = final_capital
        self.session.max_drawdown_pct = max_drawdown

        self.db.update_session(
            session_id=self.session.session_id,
            end_timestamp=self.session.end_timestamp,
            final_capital=final_capital,
            total_trades=self.session.total_trades,
            total_wins=self.session.total_wins,
            total_losses=self.session.total_losses,
            total_pnl_usd=self.session.total_pnl_usd,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=self.session.win_rate,
        )

        print(f"\n[SESSION CLOSED] {self.session.session_id}")
        print(f"  Trades: {self.session.total_trades}")
        print(f"  Win Rate: {self.session.win_rate*100:.1f}%" if self.session.win_rate else "  Win Rate: N/A")
        print(f"  Total PnL: ${self.session.total_pnl_usd:+,.2f}")
        print(f"  Final Capital: ${final_capital:,.2f}")

    def get_session_summary(self) -> Dict:
        """Get current session summary."""
        if not self.session:
            return {}
        return self.db.get_session_summary(self.session.session_id)

    def get_formula_performance(self) -> List[Dict]:
        """Get formula performance for current session."""
        if not self.session:
            return []
        return self.db.get_formula_performance(self.session.session_id)
