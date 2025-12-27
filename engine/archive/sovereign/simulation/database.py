"""
SQLite database for simulation trade logging.
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import json


SCHEMA_SQL = """
-- Core trade log
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    mode TEXT NOT NULL,

    -- Timestamps
    signal_timestamp REAL NOT NULL,
    entry_timestamp REAL,
    exit_timestamp REAL,

    -- Trade Details
    formula_id INTEGER NOT NULL,
    formula_name TEXT NOT NULL,
    direction INTEGER NOT NULL,
    signal_strength REAL NOT NULL,

    -- Prices
    signal_price REAL NOT NULL,
    entry_price REAL,
    exit_price REAL,

    -- Position
    position_size_pct REAL NOT NULL,
    position_btc REAL,
    position_usd REAL,

    -- Risk
    stop_loss_pct REAL,
    take_profit_pct REAL,

    -- Result
    pnl_usd REAL,
    pnl_pct REAL,
    exit_reason TEXT,

    -- Verification
    exchange_price_at_signal REAL,
    slippage_estimated REAL,
    prediction_correct INTEGER,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Session tracking
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    start_timestamp REAL NOT NULL,
    end_timestamp REAL,

    initial_capital REAL NOT NULL,
    final_capital REAL,
    kelly_fraction REAL,

    formula_ids TEXT,

    total_trades INTEGER DEFAULT 0,
    total_wins INTEGER DEFAULT 0,
    total_losses INTEGER DEFAULT 0,
    total_pnl_usd REAL DEFAULT 0,

    max_drawdown_pct REAL,
    sharpe_ratio REAL,
    win_rate REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Formula performance tracking
CREATE TABLE IF NOT EXISTS formula_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    formula_id INTEGER NOT NULL,
    formula_name TEXT,

    trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    total_pnl REAL DEFAULT 0,
    win_rate REAL,
    avg_pnl REAL,

    UNIQUE(session_id, formula_id)
);

-- Price snapshots for verification
CREATE TABLE IF NOT EXISTS price_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    trade_id TEXT,

    coinbase_price REAL,
    kraken_price REAL,
    bitstamp_price REAL,
    binance_price REAL,

    avg_price REAL,
    spread_pct REAL
);

-- Equity curve tracking
CREATE TABLE IF NOT EXISTS equity_curve (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    capital REAL NOT NULL,
    drawdown_pct REAL
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(session_id);
CREATE INDEX IF NOT EXISTS idx_trades_formula ON trades(formula_id);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(signal_timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_mode ON sessions(mode);
CREATE INDEX IF NOT EXISTS idx_equity_session ON equity_curve(session_id);
"""


class SimulationDatabase:
    """SQLite database for simulation data."""

    def __init__(self, db_path: str = "data/simulation_trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database with schema."""
        with self.connection() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def connection(self):
        """Context manager for database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # Session operations
    def create_session(
        self,
        session_id: str,
        mode: str,
        start_timestamp: float,
        initial_capital: float,
        kelly_fraction: float,
        formula_ids: List[int]
    ):
        """Create new simulation session."""
        with self.connection() as conn:
            conn.execute('''
                INSERT INTO sessions (
                    session_id, mode, start_timestamp, initial_capital,
                    kelly_fraction, formula_ids
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session_id, mode, start_timestamp, initial_capital,
                kelly_fraction, json.dumps(formula_ids)
            ))

    def update_session(
        self,
        session_id: str,
        end_timestamp: float = None,
        final_capital: float = None,
        total_trades: int = None,
        total_wins: int = None,
        total_losses: int = None,
        total_pnl_usd: float = None,
        max_drawdown_pct: float = None,
        sharpe_ratio: float = None,
        win_rate: float = None
    ):
        """Update session statistics."""
        updates = []
        values = []

        if end_timestamp is not None:
            updates.append("end_timestamp = ?")
            values.append(end_timestamp)
        if final_capital is not None:
            updates.append("final_capital = ?")
            values.append(final_capital)
        if total_trades is not None:
            updates.append("total_trades = ?")
            values.append(total_trades)
        if total_wins is not None:
            updates.append("total_wins = ?")
            values.append(total_wins)
        if total_losses is not None:
            updates.append("total_losses = ?")
            values.append(total_losses)
        if total_pnl_usd is not None:
            updates.append("total_pnl_usd = ?")
            values.append(total_pnl_usd)
        if max_drawdown_pct is not None:
            updates.append("max_drawdown_pct = ?")
            values.append(max_drawdown_pct)
        if sharpe_ratio is not None:
            updates.append("sharpe_ratio = ?")
            values.append(sharpe_ratio)
        if win_rate is not None:
            updates.append("win_rate = ?")
            values.append(win_rate)

        if updates:
            values.append(session_id)
            with self.connection() as conn:
                conn.execute(f'''
                    UPDATE sessions SET {", ".join(updates)}
                    WHERE session_id = ?
                ''', values)

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID."""
        with self.connection() as conn:
            row = conn.execute(
                'SELECT * FROM sessions WHERE session_id = ?',
                (session_id,)
            ).fetchone()
            return dict(row) if row else None

    # Trade operations
    def insert_trade(
        self,
        trade_id: str,
        session_id: str,
        mode: str,
        signal_timestamp: float,
        formula_id: int,
        formula_name: str,
        direction: int,
        signal_strength: float,
        signal_price: float,
        position_size_pct: float,
        stop_loss_pct: float,
        take_profit_pct: float
    ):
        """Insert new trade entry."""
        with self.connection() as conn:
            conn.execute('''
                INSERT INTO trades (
                    trade_id, session_id, mode, signal_timestamp,
                    formula_id, formula_name, direction, signal_strength,
                    signal_price, position_size_pct, stop_loss_pct, take_profit_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id, session_id, mode, signal_timestamp,
                formula_id, formula_name, direction, signal_strength,
                signal_price, position_size_pct, stop_loss_pct, take_profit_pct
            ))

    def update_trade_entry(
        self,
        trade_id: str,
        entry_timestamp: float,
        entry_price: float,
        position_btc: float,
        position_usd: float
    ):
        """Update trade with entry details."""
        with self.connection() as conn:
            conn.execute('''
                UPDATE trades SET
                    entry_timestamp = ?,
                    entry_price = ?,
                    position_btc = ?,
                    position_usd = ?
                WHERE trade_id = ?
            ''', (entry_timestamp, entry_price, position_btc, position_usd, trade_id))

    def update_trade_exit(
        self,
        trade_id: str,
        exit_timestamp: float,
        exit_price: float,
        exit_reason: str,
        pnl_usd: float,
        pnl_pct: float
    ):
        """Update trade with exit details."""
        with self.connection() as conn:
            conn.execute('''
                UPDATE trades SET
                    exit_timestamp = ?,
                    exit_price = ?,
                    exit_reason = ?,
                    pnl_usd = ?,
                    pnl_pct = ?
                WHERE trade_id = ?
            ''', (exit_timestamp, exit_price, exit_reason, pnl_usd, pnl_pct, trade_id))

    def update_trade_verification(
        self,
        trade_id: str,
        exchange_price: float,
        slippage: float,
        prediction_correct: bool
    ):
        """Update trade with verification details."""
        with self.connection() as conn:
            conn.execute('''
                UPDATE trades SET
                    exchange_price_at_signal = ?,
                    slippage_estimated = ?,
                    prediction_correct = ?
                WHERE trade_id = ?
            ''', (exchange_price, slippage, int(prediction_correct), trade_id))

    def get_trade(self, trade_id: str) -> Optional[Dict]:
        """Get trade by ID."""
        with self.connection() as conn:
            row = conn.execute(
                'SELECT * FROM trades WHERE trade_id = ?',
                (trade_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_session_trades(self, session_id: str) -> List[Dict]:
        """Get all trades for a session."""
        with self.connection() as conn:
            rows = conn.execute(
                'SELECT * FROM trades WHERE session_id = ? ORDER BY signal_timestamp',
                (session_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    # Formula performance
    def update_formula_performance(
        self,
        session_id: str,
        formula_id: int,
        formula_name: str,
        is_win: bool,
        pnl: float
    ):
        """Update formula performance stats."""
        with self.connection() as conn:
            # Try to update existing
            result = conn.execute('''
                UPDATE formula_performance SET
                    trades = trades + 1,
                    wins = wins + ?,
                    losses = losses + ?,
                    total_pnl = total_pnl + ?
                WHERE session_id = ? AND formula_id = ?
            ''', (int(is_win), int(not is_win), pnl, session_id, formula_id))

            # Insert if doesn't exist
            if result.rowcount == 0:
                conn.execute('''
                    INSERT INTO formula_performance (
                        session_id, formula_id, formula_name,
                        trades, wins, losses, total_pnl
                    ) VALUES (?, ?, ?, 1, ?, ?, ?)
                ''', (
                    session_id, formula_id, formula_name,
                    int(is_win), int(not is_win), pnl
                ))

            # Update win_rate and avg_pnl
            conn.execute('''
                UPDATE formula_performance SET
                    win_rate = CAST(wins AS REAL) / trades,
                    avg_pnl = total_pnl / trades
                WHERE session_id = ? AND formula_id = ?
            ''', (session_id, formula_id))

    def get_formula_performance(self, session_id: str) -> List[Dict]:
        """Get formula performance for session."""
        with self.connection() as conn:
            rows = conn.execute('''
                SELECT * FROM formula_performance
                WHERE session_id = ?
                ORDER BY total_pnl DESC
            ''', (session_id,)).fetchall()
            return [dict(row) for row in rows]

    # Equity curve
    def add_equity_point(
        self,
        session_id: str,
        timestamp: float,
        capital: float,
        drawdown_pct: float
    ):
        """Add equity curve data point."""
        with self.connection() as conn:
            conn.execute('''
                INSERT INTO equity_curve (session_id, timestamp, capital, drawdown_pct)
                VALUES (?, ?, ?, ?)
            ''', (session_id, timestamp, capital, drawdown_pct))

    def get_equity_curve(self, session_id: str) -> List[Dict]:
        """Get equity curve for session."""
        with self.connection() as conn:
            rows = conn.execute('''
                SELECT timestamp, capital, drawdown_pct
                FROM equity_curve
                WHERE session_id = ?
                ORDER BY timestamp
            ''', (session_id,)).fetchall()
            return [dict(row) for row in rows]

    # Price snapshots
    def add_price_snapshot(
        self,
        timestamp: float,
        trade_id: str = None,
        prices: Dict[str, float] = None
    ):
        """Add price snapshot for verification."""
        prices = prices or {}
        avg_price = sum(prices.values()) / len(prices) if prices else 0

        with self.connection() as conn:
            conn.execute('''
                INSERT INTO price_snapshots (
                    timestamp, trade_id, coinbase_price, kraken_price,
                    bitstamp_price, binance_price, avg_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, trade_id,
                prices.get('coinbase'),
                prices.get('kraken'),
                prices.get('bitstamp'),
                prices.get('binance'),
                avg_price
            ))

    # Summary queries
    def get_session_summary(self, session_id: str) -> Dict:
        """Get comprehensive session summary."""
        session = self.get_session(session_id)
        if not session:
            return {}

        trades = self.get_session_trades(session_id)
        formula_perf = self.get_formula_performance(session_id)

        return {
            'session': session,
            'trades_count': len(trades),
            'formula_performance': formula_perf,
            'trades': trades,
        }
