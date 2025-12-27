#!/usr/bin/env python3
"""
RENAISSANCE-STYLE MATHEMATICAL TRADER
======================================
Combines multiple signals using Kelly Criterion position sizing.
Trades both direction AND volatility.

Mathematical Framework:
1. Signal Combination: S = Σ(wi * si) where wi = edge_i / variance_i
2. Kelly Criterion: f* = (p*b - q) / b = edge / variance
3. Volatility Targeting: position_size = target_vol / realized_vol
4. Mean Reversion: Counter-trend on short timeframes

Key Insight: With 50x leverage and proper sizing, even 1.4 bps edge
compounds to massive returns if variance is managed.
"""

import sqlite3
import json
import time
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Paths
DATA_DIR = Path("/root/validation/data")
METRICS_DB = DATA_DIR / "metrics.db"
TRADES_DB = DATA_DIR / "rentech_trades.db"
LOG_FILE = DATA_DIR / "rentech_trader.log"
STATE_FILE = DATA_DIR / "rentech_state.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Individual signal with its statistics."""
    name: str
    value: float
    edge_bps: float      # Historical edge in bps
    variance: float      # Historical variance
    kelly_weight: float  # Optimal Kelly fraction


@dataclass
class Position:
    """Current position."""
    direction: int       # 1=LONG, -1=SHORT, 0=FLAT
    entry_price: float
    entry_time: float
    size_btc: float      # Position size in BTC
    leverage: float      # Effective leverage used
    signal_score: float  # Combined signal at entry
    stop_loss: float     # Stop loss price
    take_profit: float   # Take profit price


class RentechTrader:
    """
    Renaissance-style systematic trader.

    Key principles:
    1. Combine multiple weak signals into one strong signal
    2. Size positions using Kelly Criterion (but fractional Kelly for safety)
    3. Target constant volatility, not constant leverage
    4. Trade mean reversion on short timeframes
    5. Strict risk management with stops
    """

    def __init__(
        self,
        capital_usd: float = 100.0,
        max_leverage: float = 50.0,
        kelly_fraction: float = 0.25,    # Use 1/4 Kelly for safety
        target_vol_daily: float = 0.50,  # Target 50% daily vol
        maker_fee_bps: float = 4.0,
        taker_fee_bps: float = 10.0,
        max_position_pct: float = 0.80,  # Max 80% of capital at risk
        stop_loss_pct: float = 0.02,     # 2% stop loss
        take_profit_pct: float = 0.01,   # 1% take profit (2:1 risk/reward with 50x)
    ):
        self.capital_usd = capital_usd
        self.max_leverage = max_leverage
        self.kelly_fraction = kelly_fraction
        self.target_vol_daily = target_vol_daily
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.position: Optional[Position] = None
        self.trade_history: List[Dict] = []

        # Signal statistics from backtesting
        # Format: {signal_name: (edge_bps, variance, optimal_direction)}
        self.signal_stats = {
            # Whale signals (SHORT when whales active - they dump)
            'whale_3': {'edge': 1.4, 'var': 25.0, 'direction': -1},
            'whale_5': {'edge': 0.4, 'var': 20.0, 'direction': -1},

            # Mean reversion signals (counter-trend)
            'price_zscore_high': {'edge': 2.0, 'var': 30.0, 'direction': -1},  # SHORT when price spikes
            'price_zscore_low': {'edge': 2.0, 'var': 30.0, 'direction': 1},    # LONG when price drops

            # Volume signals (fade high volume)
            'vol_zscore_high': {'edge': 0.8, 'var': 28.0, 'direction': -1},

            # Consolidation (breakout fade)
            'consol_high': {'edge': 0.5, 'var': 22.0, 'direction': -1},
        }

        # Rolling statistics for real-time calculation
        self.price_history: List[Tuple[float, float]] = []  # (timestamp, price)
        self.vol_history: List[Tuple[float, float]] = []
        self.lookback_minutes = 60

        self._init_db()
        self._load_state()

    def _init_db(self):
        """Initialize trades database."""
        conn = sqlite3.connect(TRADES_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                entry_time REAL,
                exit_time REAL,
                direction INTEGER,
                entry_price REAL,
                exit_price REAL,
                size_btc REAL,
                leverage REAL,
                signal_score REAL,
                pnl_bps REAL,
                pnl_usd REAL,
                exit_reason TEXT,
                signals_json TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_curve (
                timestamp REAL PRIMARY KEY,
                equity_usd REAL,
                drawdown_pct REAL
            )
        """)
        conn.commit()
        conn.close()

    def _load_state(self):
        """Load state from file."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                self.capital_usd = state.get('capital_usd', self.capital_usd)
                self.trade_history = state.get('trade_history', [])

                if state.get('position'):
                    p = state['position']
                    self.position = Position(
                        direction=p['direction'],
                        entry_price=p['entry_price'],
                        entry_time=p['entry_time'],
                        size_btc=p['size_btc'],
                        leverage=p['leverage'],
                        signal_score=p['signal_score'],
                        stop_loss=p['stop_loss'],
                        take_profit=p['take_profit']
                    )
                    logger.info(f"Restored position: {'LONG' if self.position.direction == 1 else 'SHORT'} @ ${self.position.entry_price:,.0f}")
            except Exception as e:
                logger.error(f"Error loading state: {e}")

    def _save_state(self):
        """Save state to file."""
        state = {
            'capital_usd': self.capital_usd,
            'trade_history': self.trade_history[-100:],  # Keep last 100
            'position': None
        }

        if self.position:
            state['position'] = {
                'direction': self.position.direction,
                'entry_price': self.position.entry_price,
                'entry_time': self.position.entry_time,
                'size_btc': self.position.size_btc,
                'leverage': self.position.leverage,
                'signal_score': self.position.signal_score,
                'stop_loss': self.position.stop_loss,
                'take_profit': self.position.take_profit
            }

        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    def calculate_realized_volatility(self, prices: List[float], period_minutes: int = 60) -> float:
        """
        Calculate realized volatility from recent prices.
        Returns annualized volatility.
        """
        if len(prices) < 10:
            return 0.5  # Default 50% annual vol

        returns = np.diff(np.log(prices))
        if len(returns) == 0:
            return 0.5

        # Annualize: sqrt(minutes_per_year / period_minutes) * std
        minutes_per_year = 365 * 24 * 60
        annualized_vol = np.std(returns) * np.sqrt(minutes_per_year / period_minutes)

        return max(0.1, min(2.0, annualized_vol))  # Clamp between 10% and 200%

    def calculate_signals(self, data: Dict) -> List[Signal]:
        """
        Calculate all signals from current data.
        Returns list of Signal objects with their current values and statistics.
        """
        signals = []

        # Whale signals
        whale_count = data.get('whale_count', 0)
        if whale_count >= 3:
            stats = self.signal_stats['whale_3']
            signals.append(Signal(
                name='whale_3',
                value=whale_count,
                edge_bps=stats['edge'],
                variance=stats['var'],
                kelly_weight=stats['edge'] / stats['var'] if stats['var'] > 0 else 0
            ))

        if whale_count >= 5:
            stats = self.signal_stats['whale_5']
            signals.append(Signal(
                name='whale_5',
                value=whale_count,
                edge_bps=stats['edge'],
                variance=stats['var'],
                kelly_weight=stats['edge'] / stats['var'] if stats['var'] > 0 else 0
            ))

        # Price z-score signals (mean reversion)
        price_zscore = data.get('price_zscore', 0)
        if price_zscore > 2.0:
            stats = self.signal_stats['price_zscore_high']
            signals.append(Signal(
                name='price_zscore_high',
                value=price_zscore,
                edge_bps=stats['edge'],
                variance=stats['var'],
                kelly_weight=stats['edge'] / stats['var'] if stats['var'] > 0 else 0
            ))
        elif price_zscore < -2.0:
            stats = self.signal_stats['price_zscore_low']
            signals.append(Signal(
                name='price_zscore_low',
                value=price_zscore,
                edge_bps=stats['edge'],
                variance=stats['var'],
                kelly_weight=stats['edge'] / stats['var'] if stats['var'] > 0 else 0
            ))

        # Volume z-score (fade high volume)
        vol_zscore = data.get('vol_zscore', 0)
        if vol_zscore > 2.0:
            stats = self.signal_stats['vol_zscore_high']
            signals.append(Signal(
                name='vol_zscore_high',
                value=vol_zscore,
                edge_bps=stats['edge'],
                variance=stats['var'],
                kelly_weight=stats['edge'] / stats['var'] if stats['var'] > 0 else 0
            ))

        return signals

    def calculate_combined_score(self, signals: List[Signal]) -> Tuple[float, int]:
        """
        Combine signals using Kelly-weighted average.
        Returns (score, direction).

        Score = Σ(kelly_weight_i * signal_value_i) / Σ(kelly_weight_i)
        Direction = sign of weighted sum considering each signal's optimal direction
        """
        if not signals:
            return 0.0, 0

        weighted_direction_sum = 0.0
        total_weight = 0.0
        total_edge = 0.0

        for sig in signals:
            optimal_dir = self.signal_stats.get(sig.name, {}).get('direction', 1)
            weight = sig.kelly_weight

            weighted_direction_sum += weight * optimal_dir
            total_weight += weight
            total_edge += sig.edge_bps * weight

        if total_weight == 0:
            return 0.0, 0

        # Score is weighted average edge
        score = total_edge / total_weight

        # Direction is sign of weighted direction sum
        direction = 1 if weighted_direction_sum > 0 else -1

        return score, direction

    def calculate_position_size(
        self,
        score: float,
        realized_vol: float,
        current_price: float
    ) -> Tuple[float, float]:
        """
        Calculate optimal position size using:
        1. Kelly Criterion (edge/variance)
        2. Volatility targeting
        3. Maximum position limits

        Returns (size_btc, effective_leverage).
        """
        if score <= 0:
            return 0.0, 0.0

        # Kelly fraction: f* = edge / variance
        # We use fractional Kelly for safety
        avg_variance = 25.0  # Average variance from our signals
        kelly_optimal = score / avg_variance
        kelly_safe = kelly_optimal * self.kelly_fraction

        # Volatility targeting: scale position to target volatility
        # position_vol = position_size * realized_vol
        # We want position_vol = target_vol * capital
        vol_target_size = (self.target_vol_daily * self.capital_usd) / (realized_vol * current_price)

        # Kelly sizing (fraction of capital)
        kelly_size_usd = self.capital_usd * kelly_safe
        kelly_size_btc = kelly_size_usd / current_price

        # Take minimum of Kelly and vol-target
        size_btc = min(kelly_size_btc, vol_target_size)

        # Apply maximum position limit
        max_size_usd = self.capital_usd * self.max_position_pct * self.max_leverage
        max_size_btc = max_size_usd / current_price
        size_btc = min(size_btc, max_size_btc)

        # Calculate effective leverage
        position_value = size_btc * current_price
        leverage = position_value / self.capital_usd
        leverage = min(leverage, self.max_leverage)

        # Recalculate size with capped leverage
        size_btc = (self.capital_usd * leverage) / current_price

        return size_btc, leverage

    def should_enter(self, signals: List[Signal], score: float) -> bool:
        """
        Determine if we should enter a new position.

        Entry criteria:
        1. At least one signal active
        2. Combined score > minimum threshold
        3. Not already in a position
        """
        if self.position is not None:
            return False

        if len(signals) == 0:
            return False

        # Require minimum score (edge > fees)
        min_score = self.taker_fee_bps  # Need edge > taker fees
        return score > min_score

    def check_exit_conditions(self, current_price: float, current_time: float) -> Optional[str]:
        """
        Check if we should exit current position.

        Exit conditions:
        1. Stop loss hit
        2. Take profit hit
        3. Time-based exit (10 minutes)
        4. Signal reversal
        """
        if self.position is None:
            return None

        p = self.position

        # Stop loss
        if p.direction == 1:  # LONG
            if current_price <= p.stop_loss:
                return 'stop_loss'
            if current_price >= p.take_profit:
                return 'take_profit'
        else:  # SHORT
            if current_price >= p.stop_loss:
                return 'stop_loss'
            if current_price <= p.take_profit:
                return 'take_profit'

        # Time-based exit (10 minutes)
        hold_time = current_time - p.entry_time
        if hold_time >= 600:  # 10 minutes
            return 'time_exit'

        return None

    def enter_position(
        self,
        direction: int,
        price: float,
        size_btc: float,
        leverage: float,
        score: float,
        signals: List[Signal]
    ):
        """Enter a new position."""

        # Calculate stop loss and take profit
        if direction == 1:  # LONG
            stop_loss = price * (1 - self.stop_loss_pct / leverage)
            take_profit = price * (1 + self.take_profit_pct / leverage)
        else:  # SHORT
            stop_loss = price * (1 + self.stop_loss_pct / leverage)
            take_profit = price * (1 - self.take_profit_pct / leverage)

        self.position = Position(
            direction=direction,
            entry_price=price,
            entry_time=time.time(),
            size_btc=size_btc,
            leverage=leverage,
            signal_score=score,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        dir_str = 'LONG' if direction == 1 else 'SHORT'
        signal_names = [s.name for s in signals]
        logger.info(f"ENTER {dir_str} @ ${price:,.0f} | Size: {size_btc:.6f} BTC | Leverage: {leverage:.1f}x | Score: {score:.2f}")
        logger.info(f"  Signals: {signal_names}")
        logger.info(f"  Stop: ${stop_loss:,.0f} | TP: ${take_profit:,.0f}")

        self._save_state()

    def exit_position(self, price: float, reason: str) -> Dict:
        """Exit current position and record trade."""
        if self.position is None:
            return {}

        p = self.position

        # Calculate P&L
        if p.direction == 1:  # LONG
            gross_return_pct = (price / p.entry_price - 1) * 100
        else:  # SHORT
            gross_return_pct = (p.entry_price / price - 1) * 100

        # Apply leverage
        leveraged_return_pct = gross_return_pct * p.leverage

        # Subtract fees (taker for stop/TP, maker for time exit)
        if reason == 'time_exit':
            fee_pct = self.maker_fee_bps / 100 * 2  # Entry + exit
        else:
            fee_pct = self.taker_fee_bps / 100 * 2

        net_return_pct = leveraged_return_pct - fee_pct
        pnl_bps = net_return_pct * 100

        # Calculate USD P&L
        position_value = p.size_btc * p.entry_price
        pnl_usd = position_value * (net_return_pct / 100)

        # Update capital
        old_capital = self.capital_usd
        self.capital_usd += pnl_usd

        dir_str = 'LONG' if p.direction == 1 else 'SHORT'
        logger.info(f"EXIT {dir_str} @ ${price:,.0f} ({reason}) | P&L: {pnl_bps:+.1f} bps (${pnl_usd:+.2f})")
        logger.info(f"  Capital: ${old_capital:.2f} -> ${self.capital_usd:.2f}")

        # Record trade
        trade = {
            'entry_time': p.entry_time,
            'exit_time': time.time(),
            'direction': p.direction,
            'entry_price': p.entry_price,
            'exit_price': price,
            'size_btc': p.size_btc,
            'leverage': p.leverage,
            'signal_score': p.signal_score,
            'pnl_bps': pnl_bps,
            'pnl_usd': pnl_usd,
            'exit_reason': reason
        }

        self.trade_history.append(trade)
        self._record_trade(trade)

        # Clear position
        self.position = None
        self._save_state()

        return trade

    def _record_trade(self, trade: Dict):
        """Record trade to database."""
        conn = sqlite3.connect(TRADES_DB)
        conn.execute("""
            INSERT INTO trades (
                entry_time, exit_time, direction, entry_price, exit_price,
                size_btc, leverage, signal_score, pnl_bps, pnl_usd, exit_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade['entry_time'], trade['exit_time'], trade['direction'],
            trade['entry_price'], trade['exit_price'], trade['size_btc'],
            trade['leverage'], trade['signal_score'], trade['pnl_bps'],
            trade['pnl_usd'], trade['exit_reason']
        ))

        # Record equity curve
        conn.execute("""
            INSERT OR REPLACE INTO equity_curve (timestamp, equity_usd, drawdown_pct)
            VALUES (?, ?, ?)
        """, (time.time(), self.capital_usd, self._calculate_drawdown()))

        conn.commit()
        conn.close()

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if not self.trade_history:
            return 0.0

        peak = 100.0  # Starting capital
        for trade in self.trade_history:
            peak = max(peak, peak + trade['pnl_usd'])

        current = self.capital_usd
        drawdown = (peak - current) / peak * 100
        return drawdown

    def process_tick(self, data: Dict):
        """
        Process one tick of data.
        This is the main trading loop.
        """
        price = data.get('price', 0)
        if price <= 0:
            return

        current_time = time.time()

        # Update price history for volatility calculation
        self.price_history.append((current_time, price))

        # Keep only last hour of prices
        cutoff = current_time - 3600
        self.price_history = [(t, p) for t, p in self.price_history if t > cutoff]

        # Check exit conditions first
        if self.position:
            exit_reason = self.check_exit_conditions(price, current_time)
            if exit_reason:
                self.exit_position(price, exit_reason)
                return

        # Calculate signals
        signals = self.calculate_signals(data)

        if not signals:
            return

        # Calculate combined score and direction
        score, direction = self.calculate_combined_score(signals)

        # Check entry conditions
        if self.should_enter(signals, score):
            # Calculate realized volatility
            prices = [p for _, p in self.price_history]
            realized_vol = self.calculate_realized_volatility(prices)

            # Calculate position size
            size_btc, leverage = self.calculate_position_size(score, realized_vol, price)

            if size_btc > 0 and leverage > 0:
                self.enter_position(direction, price, size_btc, leverage, score, signals)

    def get_latest_data(self) -> Optional[Dict]:
        """Get latest data from metrics database."""
        conn = sqlite3.connect(METRICS_DB)

        # Get latest row
        cursor = conn.execute("""
            SELECT timestamp, tx_count, total_volume_btc, tx_whale,
                   tx_mega, consolidation_ratio, price
            FROM metrics
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        ts, tx_count, volume, whale, mega, consol, price = row

        # Get last 3600 rows for rolling stats (calculate in Python since SQLite lacks STDDEV)
        cursor = conn.execute("""
            SELECT tx_count, total_volume_btc, price
            FROM metrics
            ORDER BY timestamp DESC
            LIMIT 3600
        """)
        history = cursor.fetchall()
        conn.close()

        if len(history) < 10:
            return {
                'timestamp': ts,
                'price': price,
                'whale_count': whale,
                'mega_count': mega,
                'tx_zscore': 0,
                'vol_zscore': 0,
                'price_zscore': 0
            }

        # Calculate stats in Python
        tx_vals = np.array([r[0] for r in history])
        vol_vals = np.array([r[1] for r in history])
        price_vals = np.array([r[2] for r in history])

        tx_mean, tx_std = np.mean(tx_vals), np.std(tx_vals)
        vol_mean, vol_std = np.mean(vol_vals), np.std(vol_vals)
        price_mean, price_std = np.mean(price_vals), np.std(price_vals)

        tx_zscore = (tx_count - tx_mean) / tx_std if tx_std > 0 else 0
        vol_zscore = (volume - vol_mean) / vol_std if vol_std > 0 else 0
        price_zscore = (price - price_mean) / price_std if price_std > 0 else 0

        return {
            'timestamp': ts,
            'price': price,
            'whale_count': whale,
            'mega_count': mega,
            'tx_zscore': tx_zscore,
            'vol_zscore': vol_zscore,
            'price_zscore': price_zscore
        }

    def print_status(self):
        """Print current status."""
        logger.info("=" * 60)
        logger.info(f"  RENTECH TRADER STATUS")
        logger.info("=" * 60)
        logger.info(f"Capital: ${self.capital_usd:.2f}")
        logger.info(f"Max Leverage: {self.max_leverage}x")
        logger.info(f"Kelly Fraction: {self.kelly_fraction}")
        logger.info(f"Target Vol: {self.target_vol_daily * 100:.0f}%")

        if self.position:
            p = self.position
            dir_str = 'LONG' if p.direction == 1 else 'SHORT'
            logger.info(f"Position: {dir_str} @ ${p.entry_price:,.0f}")
            logger.info(f"  Size: {p.size_btc:.6f} BTC | Leverage: {p.leverage:.1f}x")
            logger.info(f"  Stop: ${p.stop_loss:,.0f} | TP: ${p.take_profit:,.0f}")
        else:
            logger.info("Position: FLAT")

        if self.trade_history:
            recent = self.trade_history[-10:]
            wins = sum(1 for t in recent if t['pnl_bps'] > 0)
            total_pnl = sum(t['pnl_bps'] for t in recent)
            logger.info(f"Last 10 trades: {wins}/10 wins | {total_pnl:+.1f} bps")

        logger.info("=" * 60)

    def run(self):
        """Main trading loop."""
        logger.info("=" * 60)
        logger.info("  RENTECH TRADER STARTING")
        logger.info("=" * 60)

        self.print_status()

        last_status = time.time()

        while True:
            try:
                # Get latest data
                data = self.get_latest_data()

                if data:
                    self.process_tick(data)

                # Print status every 5 minutes
                if time.time() - last_status >= 300:
                    self.print_status()
                    last_status = time.time()

                # Sleep 1 second
                time.sleep(1)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)


def main():
    trader = RentechTrader(
        capital_usd=100.0,
        max_leverage=50.0,
        kelly_fraction=0.25,
        target_vol_daily=0.50
    )
    trader.run()


if __name__ == "__main__":
    main()
