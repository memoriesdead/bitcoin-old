#!/usr/bin/env python3
"""
VOLATILITY TRADER
=================
Instead of predicting price direction, predict volatility.
Bitcoin volatility is more predictable than direction.

Key Insight: When whale activity spikes, volatility expands.
We don't know which direction, but we know SOMETHING will happen.

Strategy:
1. Buy straddle-equivalent when volatility signal fires
2. Long AND Short at same time (market neutral)
3. Exit whichever side is winning after N minutes
4. Net effect: We profit from large moves in EITHER direction

Mathematical Framework:
Expected profit = P(big_move) * avg_move_size - P(no_move) * fees
If P(big_move|signal) > P(big_move|no_signal), we have edge.
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
TRADES_DB = DATA_DIR / "vol_trades.db"
LOG_FILE = DATA_DIR / "vol_trader.log"
STATE_FILE = DATA_DIR / "vol_state.json"

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
class StraddlePosition:
    """
    Straddle-equivalent position.
    Long AND Short at same time.
    """
    entry_price: float
    entry_time: float
    size_btc: float
    leverage: float
    long_active: bool
    short_active: bool
    long_pnl: float
    short_pnl: float


class VolatilityTrader:
    """
    Trade volatility expansion, not direction.

    When whale signal fires:
    1. Open LONG and SHORT simultaneously (straddle)
    2. After 2 minutes, close the losing side
    3. Let the winning side run to take profit
    4. Net effect: We profit from volatility, direction-neutral
    """

    def __init__(
        self,
        capital_usd: float = 100.0,
        max_leverage: float = 25.0,  # Per side, so 50x total
        fee_bps: float = 4.0,        # Maker fees
        vol_threshold: float = 2.0,  # Z-score for entry
        close_loser_mins: float = 2.0,  # When to close losing side
        max_hold_mins: float = 10.0,    # Maximum hold time
        profit_target_bps: float = 30.0,  # Take profit
    ):
        self.capital_usd = capital_usd
        self.max_leverage = max_leverage
        self.fee_bps = fee_bps
        self.vol_threshold = vol_threshold
        self.close_loser_mins = close_loser_mins
        self.max_hold_mins = max_hold_mins
        self.profit_target_bps = profit_target_bps

        self.position: Optional[StraddlePosition] = None
        self.trade_history: List[Dict] = []
        self.cooldown_until = 0

        self._init_db()
        self._load_state()

    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(TRADES_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS straddles (
                id INTEGER PRIMARY KEY,
                entry_time REAL,
                exit_time REAL,
                entry_price REAL,
                size_btc REAL,
                leverage REAL,
                long_exit_price REAL,
                short_exit_price REAL,
                long_pnl_bps REAL,
                short_pnl_bps REAL,
                net_pnl_bps REAL,
                net_pnl_usd REAL,
                trigger_signal TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _load_state(self):
        """Load state."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                self.capital_usd = state.get('capital_usd', self.capital_usd)
                if state.get('position'):
                    p = state['position']
                    self.position = StraddlePosition(
                        entry_price=p['entry_price'],
                        entry_time=p['entry_time'],
                        size_btc=p['size_btc'],
                        leverage=p['leverage'],
                        long_active=p['long_active'],
                        short_active=p['short_active'],
                        long_pnl=p['long_pnl'],
                        short_pnl=p['short_pnl']
                    )
            except Exception as e:
                logger.error(f"Error loading state: {e}")

    def _save_state(self):
        """Save state."""
        state = {
            'capital_usd': self.capital_usd,
            'position': None
        }
        if self.position:
            state['position'] = {
                'entry_price': self.position.entry_price,
                'entry_time': self.position.entry_time,
                'size_btc': self.position.size_btc,
                'leverage': self.position.leverage,
                'long_active': self.position.long_active,
                'short_active': self.position.short_active,
                'long_pnl': self.position.long_pnl,
                'short_pnl': self.position.short_pnl
            }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    def detect_volatility_signal(self, data: Dict) -> bool:
        """
        Detect volatility expansion signal.

        Triggers:
        1. Whale count >= 3
        2. Volume z-score > threshold
        3. Transaction z-score > threshold
        """
        if time.time() < self.cooldown_until:
            return False

        whale_count = data.get('whale_count', 0)
        vol_zscore = abs(data.get('vol_zscore', 0))
        tx_zscore = abs(data.get('tx_zscore', 0))

        # Any of these signals volatility expansion
        if whale_count >= 3:
            return True
        if vol_zscore > self.vol_threshold:
            return True
        if tx_zscore > self.vol_threshold:
            return True

        return False

    def enter_straddle(self, price: float, signal: str):
        """
        Enter straddle position.
        Long AND Short simultaneously.
        """
        # Calculate position size per side
        per_side_usd = self.capital_usd * 0.4  # 40% per side
        size_btc = (per_side_usd * self.max_leverage) / price

        self.position = StraddlePosition(
            entry_price=price,
            entry_time=time.time(),
            size_btc=size_btc,
            leverage=self.max_leverage,
            long_active=True,
            short_active=True,
            long_pnl=0,
            short_pnl=0
        )

        logger.info(f"ENTER STRADDLE @ ${price:,.0f} | Size: {size_btc:.6f} BTC/side | {self.max_leverage}x")
        logger.info(f"  Trigger: {signal}")
        logger.info(f"  Close loser in {self.close_loser_mins} min")

        self._save_state()

    def update_straddle_pnl(self, current_price: float):
        """Update P&L for both legs."""
        if not self.position:
            return

        p = self.position

        if p.long_active:
            p.long_pnl = (current_price / p.entry_price - 1) * 100 * p.leverage
        if p.short_active:
            p.short_pnl = (p.entry_price / current_price - 1) * 100 * p.leverage

    def should_close_loser(self) -> bool:
        """Check if it's time to close the losing leg."""
        if not self.position:
            return False

        hold_time_mins = (time.time() - self.position.entry_time) / 60
        return hold_time_mins >= self.close_loser_mins

    def close_losing_leg(self, current_price: float):
        """Close the losing leg of the straddle."""
        if not self.position:
            return

        p = self.position

        if not (p.long_active and p.short_active):
            return  # Already closed one side

        self.update_straddle_pnl(current_price)

        if p.long_pnl < p.short_pnl:
            # Close long (it's losing)
            pnl_bps = p.long_pnl * 100 - self.fee_bps * 2
            p.long_active = False
            logger.info(f"CLOSE LONG LEG @ ${current_price:,.0f} | P&L: {pnl_bps:+.1f} bps")
        else:
            # Close short (it's losing)
            pnl_bps = p.short_pnl * 100 - self.fee_bps * 2
            p.short_active = False
            logger.info(f"CLOSE SHORT LEG @ ${current_price:,.0f} | P&L: {pnl_bps:+.1f} bps")

        self._save_state()

    def should_exit_winner(self, current_price: float) -> bool:
        """Check if winning leg should be closed."""
        if not self.position:
            return False

        p = self.position
        self.update_straddle_pnl(current_price)

        # Check time limit
        hold_time_mins = (time.time() - p.entry_time) / 60
        if hold_time_mins >= self.max_hold_mins:
            return True

        # Check profit target
        active_pnl = 0
        if p.long_active:
            active_pnl = p.long_pnl * 100
        elif p.short_active:
            active_pnl = p.short_pnl * 100

        if active_pnl >= self.profit_target_bps:
            return True

        return False

    def exit_straddle(self, current_price: float, reason: str) -> Dict:
        """Exit remaining position and calculate total P&L."""
        if not self.position:
            return {}

        p = self.position
        self.update_straddle_pnl(current_price)

        # Calculate total P&L
        total_fees = self.fee_bps * 4  # Entry and exit for both legs

        if p.long_active and p.short_active:
            # Both still active (should not happen normally)
            gross_pnl = p.long_pnl * 100 + p.short_pnl * 100
        elif p.long_active:
            # Only long active
            gross_pnl = p.long_pnl * 100 + p.short_pnl * 100  # short was closed earlier
        elif p.short_active:
            # Only short active
            gross_pnl = p.long_pnl * 100 + p.short_pnl * 100  # long was closed earlier
        else:
            gross_pnl = 0

        net_pnl_bps = gross_pnl - total_fees

        # Calculate USD P&L
        position_value = p.size_btc * p.entry_price * 2  # Both legs
        net_pnl_usd = position_value * (net_pnl_bps / 10000)

        # Update capital
        old_capital = self.capital_usd
        self.capital_usd += net_pnl_usd

        logger.info(f"EXIT STRADDLE ({reason})")
        logger.info(f"  Long P&L: {p.long_pnl*100:+.1f} bps | Short P&L: {p.short_pnl*100:+.1f} bps")
        logger.info(f"  Net P&L: {net_pnl_bps:+.1f} bps (${net_pnl_usd:+.2f})")
        logger.info(f"  Capital: ${old_capital:.2f} -> ${self.capital_usd:.2f}")

        # Record trade
        trade = {
            'entry_time': p.entry_time,
            'exit_time': time.time(),
            'entry_price': p.entry_price,
            'exit_price': current_price,
            'size_btc': p.size_btc,
            'leverage': p.leverage,
            'long_pnl_bps': p.long_pnl * 100,
            'short_pnl_bps': p.short_pnl * 100,
            'net_pnl_bps': net_pnl_bps,
            'net_pnl_usd': net_pnl_usd,
            'exit_reason': reason
        }

        self._record_trade(trade)

        # Set cooldown
        self.cooldown_until = time.time() + 60  # 1 minute cooldown

        # Clear position
        self.position = None
        self._save_state()

        return trade

    def _record_trade(self, trade: Dict):
        """Record trade to database."""
        conn = sqlite3.connect(TRADES_DB)
        conn.execute("""
            INSERT INTO straddles (
                entry_time, exit_time, entry_price, size_btc, leverage,
                long_pnl_bps, short_pnl_bps, net_pnl_bps, net_pnl_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade['entry_time'], trade['exit_time'], trade['entry_price'],
            trade['size_btc'], trade['leverage'], trade['long_pnl_bps'],
            trade['short_pnl_bps'], trade['net_pnl_bps'], trade['net_pnl_usd']
        ))
        conn.commit()
        conn.close()

    def get_latest_data(self) -> Optional[Dict]:
        """Get latest data from metrics database."""
        conn = sqlite3.connect(METRICS_DB)

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

        # Get rolling stats (calculate in Python since SQLite lacks STDDEV)
        cursor = conn.execute("""
            SELECT tx_count, total_volume_btc
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
                'vol_zscore': 0
            }

        # Calculate stats in Python
        import numpy as np
        tx_vals = np.array([r[0] for r in history])
        vol_vals = np.array([r[1] for r in history])

        tx_mean, tx_std = np.mean(tx_vals), np.std(tx_vals)
        vol_mean, vol_std = np.mean(vol_vals), np.std(vol_vals)

        tx_std = tx_std if tx_std > 0 else 1
        vol_std = vol_std if vol_std > 0 else 1

        return {
            'timestamp': ts,
            'price': price,
            'whale_count': whale,
            'mega_count': mega,
            'tx_zscore': (tx_count - tx_mean) / tx_std,
            'vol_zscore': (volume - vol_mean) / vol_std
        }

    def process_tick(self, data: Dict):
        """Process one tick."""
        price = data.get('price', 0)
        if price <= 0:
            return

        # If in position, manage it
        if self.position:
            # Check if time to close loser
            if self.position.long_active and self.position.short_active:
                if self.should_close_loser():
                    self.close_losing_leg(price)
            else:
                # Check if time to exit winner
                if self.should_exit_winner(price):
                    reason = 'take_profit' if self.position.long_pnl * 100 >= self.profit_target_bps or \
                                              self.position.short_pnl * 100 >= self.profit_target_bps else 'time_exit'
                    self.exit_straddle(price, reason)
            return

        # Check for entry signal
        if self.detect_volatility_signal(data):
            signal = []
            if data.get('whale_count', 0) >= 3:
                signal.append(f"whale={data['whale_count']}")
            if abs(data.get('vol_zscore', 0)) > self.vol_threshold:
                signal.append(f"vol_z={data['vol_zscore']:.1f}")
            if abs(data.get('tx_zscore', 0)) > self.vol_threshold:
                signal.append(f"tx_z={data['tx_zscore']:.1f}")

            self.enter_straddle(price, ', '.join(signal))

    def print_status(self):
        """Print status."""
        logger.info("=" * 60)
        logger.info("  VOLATILITY TRADER STATUS")
        logger.info("=" * 60)
        logger.info(f"Capital: ${self.capital_usd:.2f}")
        logger.info(f"Strategy: Straddle on volatility signals")

        if self.position:
            p = self.position
            hold_mins = (time.time() - p.entry_time) / 60
            logger.info(f"Position: STRADDLE @ ${p.entry_price:,.0f}")
            logger.info(f"  Hold time: {hold_mins:.1f} min")
            logger.info(f"  Long active: {p.long_active} | Short active: {p.short_active}")
            logger.info(f"  Long P&L: {p.long_pnl*100:+.1f} bps | Short P&L: {p.short_pnl*100:+.1f} bps")
        else:
            logger.info("Position: FLAT")

        logger.info("=" * 60)

    def run(self):
        """Main loop."""
        logger.info("=" * 60)
        logger.info("  VOLATILITY TRADER STARTING")
        logger.info("=" * 60)

        self.print_status()
        last_status = time.time()

        while True:
            try:
                data = self.get_latest_data()
                if data:
                    self.process_tick(data)

                if time.time() - last_status >= 300:
                    self.print_status()
                    last_status = time.time()

                time.sleep(1)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(5)


def main():
    trader = VolatilityTrader(
        capital_usd=100.0,
        max_leverage=25.0,
        vol_threshold=2.0,
        close_loser_mins=2.0,
        max_hold_mins=10.0,
        profit_target_bps=30.0
    )
    trader.run()


if __name__ == "__main__":
    main()
