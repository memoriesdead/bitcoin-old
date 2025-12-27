#!/usr/bin/env python3
"""
VPS PAPER TRADER
================
Runs on Hostinger VPS alongside the metric collector.
Reads blockchain data, generates signals, simulates trades.
Logs everything for morning review.
"""

import os
import sys
import time
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List
import numpy as np

# Paths
DATA_DIR = Path("/root/validation/data")
METRICS_DB = DATA_DIR / "metrics.db"
TRADES_DB = DATA_DIR / "paper_trades.db"
LOG_FILE = DATA_DIR / "paper_trader.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Signal:
    timestamp: float
    direction: int  # 1=LONG, -1=SHORT
    confidence: float
    reason: str
    price: float
    tx_zscore: float
    vol_zscore: float


@dataclass
class Trade:
    id: int
    entry_time: float
    entry_price: float
    direction: int
    size_usd: float
    exit_time: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_bps: Optional[float] = None
    pnl_usd: Optional[float] = None


class PaperTrader:
    """Paper trading system that runs on VPS."""

    def __init__(
        self,
        capital_usd: float = 100.0,
        leverage: int = 10,
        signal_threshold: float = 2.0,
        hold_time_seconds: int = 300,  # 5 minutes
        stop_loss_pct: float = 0.003,  # 0.3%
        take_profit_pct: float = 0.006,  # 0.6%
        cooldown_seconds: int = 60,
        lookback_seconds: int = 60
    ):
        self.capital_usd = capital_usd
        self.leverage = leverage
        self.signal_threshold = signal_threshold
        self.hold_time_seconds = hold_time_seconds
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.cooldown_seconds = cooldown_seconds
        self.lookback_seconds = lookback_seconds

        self.position: Optional[Trade] = None
        self.last_trade_time = 0
        self.trade_count = 0

        # Stats
        self.trades: List[Trade] = []
        self.wins = 0
        self.losses = 0
        self.total_pnl_bps = 0
        self.total_pnl_usd = 0

        # Initialize trades database
        self._init_db()

    def _init_db(self):
        """Initialize trades database."""
        conn = sqlite3.connect(TRADES_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                entry_time REAL,
                entry_price REAL,
                direction INTEGER,
                size_usd REAL,
                exit_time REAL,
                exit_price REAL,
                exit_reason TEXT,
                pnl_bps REAL,
                pnl_usd REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                direction INTEGER,
                confidence REAL,
                reason TEXT,
                price REAL,
                tx_zscore REAL,
                vol_zscore REAL,
                acted_on INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                timestamp REAL PRIMARY KEY,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                total_pnl_bps REAL,
                total_pnl_usd REAL,
                current_price REAL
            )
        """)
        conn.commit()
        conn.close()

    def get_recent_metrics(self) -> Optional[dict]:
        """Get most recent metrics from collector database."""
        try:
            conn = sqlite3.connect(METRICS_DB)
            cursor = conn.execute("""
                SELECT timestamp, tx_count, total_volume_btc, tx_whale,
                       tx_mega, price
                FROM metrics
                WHERE price > 0
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            row = cursor.fetchone()

            if not row:
                return None

            # Get lookback data for z-scores
            cursor = conn.execute(f"""
                SELECT tx_count, total_volume_btc
                FROM metrics
                WHERE timestamp > ? AND price > 0
                ORDER BY timestamp DESC
                LIMIT {self.lookback_seconds}
            """, (row[0] - self.lookback_seconds,))

            history = cursor.fetchall()
            conn.close()

            if len(history) < 30:
                return None

            tx_counts = [h[0] for h in history]
            volumes = [h[1] for h in history]

            # Calculate z-scores
            tx_mean = np.mean(tx_counts[1:])
            tx_std = np.std(tx_counts[1:])
            vol_mean = np.mean(volumes[1:])
            vol_std = np.std(volumes[1:])

            tx_zscore = (row[1] - tx_mean) / tx_std if tx_std > 0 else 0
            vol_zscore = (row[2] - vol_mean) / vol_std if vol_std > 0 else 0

            return {
                'timestamp': row[0],
                'tx_count': row[1],
                'total_volume_btc': row[2],
                'tx_whale': row[3],
                'tx_mega': row[4],
                'price': row[5],
                'tx_zscore': tx_zscore,
                'vol_zscore': vol_zscore
            }

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return None

    def generate_signal(self, metrics: dict) -> Optional[Signal]:
        """Generate trading signal from metrics."""
        tx_z = metrics['tx_zscore']
        vol_z = metrics['vol_zscore']
        price = metrics['price']

        # Strategy: SHORT on blockchain spikes (based on our analysis)
        # High tx/volume activity often precedes price drops

        if tx_z > self.signal_threshold or vol_z > self.signal_threshold:
            confidence = min(max(tx_z, vol_z) / 4, 1.0)
            return Signal(
                timestamp=metrics['timestamp'],
                direction=-1,  # SHORT
                confidence=confidence,
                reason=f"Spike: tx_z={tx_z:.2f}, vol_z={vol_z:.2f}",
                price=price,
                tx_zscore=tx_z,
                vol_zscore=vol_z
            )

        # Also test LONG on low activity (accumulation phase)
        if tx_z < -self.signal_threshold and vol_z < -self.signal_threshold:
            confidence = min(abs(min(tx_z, vol_z)) / 4, 1.0)
            return Signal(
                timestamp=metrics['timestamp'],
                direction=1,  # LONG
                confidence=confidence,
                reason=f"Low activity: tx_z={tx_z:.2f}, vol_z={vol_z:.2f}",
                price=price,
                tx_zscore=tx_z,
                vol_zscore=vol_z
            )

        return None

    def open_position(self, signal: Signal):
        """Open a paper position."""
        self.trade_count += 1

        self.position = Trade(
            id=self.trade_count,
            entry_time=signal.timestamp,
            entry_price=signal.price,
            direction=signal.direction,
            size_usd=self.capital_usd
        )

        self.last_trade_time = time.time()

        direction_str = "LONG" if signal.direction == 1 else "SHORT"
        logger.info(f"OPEN {direction_str} #{self.trade_count} @ ${signal.price:,.2f}")
        logger.info(f"  Reason: {signal.reason}")

        # Log signal
        self._log_signal(signal, acted_on=True)

    def check_exit(self, current_price: float, current_time: float) -> Optional[str]:
        """Check if position should be closed."""
        if not self.position:
            return None

        pos = self.position

        # Calculate current P&L
        if pos.direction == 1:  # LONG
            pnl_pct = (current_price / pos.entry_price - 1)
        else:  # SHORT
            pnl_pct = (pos.entry_price / current_price - 1)

        # Check stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return "STOP_LOSS"

        # Check take profit
        if pnl_pct >= self.take_profit_pct:
            return "TAKE_PROFIT"

        # Check hold time
        if current_time - pos.entry_time >= self.hold_time_seconds:
            return "TIME_EXIT"

        return None

    def close_position(self, exit_price: float, exit_reason: str):
        """Close the current position."""
        if not self.position:
            return

        pos = self.position

        # Calculate P&L
        if pos.direction == 1:  # LONG
            pnl_pct = (exit_price / pos.entry_price - 1)
        else:  # SHORT
            pnl_pct = (pos.entry_price / exit_price - 1)

        # Account for fees (10 bps round-trip taker)
        pnl_bps = pnl_pct * 10000 - 10  # Subtract fees
        pnl_usd = pos.size_usd * self.leverage * pnl_pct - (pos.size_usd * self.leverage * 0.001)

        # Update position
        pos.exit_time = time.time()
        pos.exit_price = exit_price
        pos.exit_reason = exit_reason
        pos.pnl_bps = pnl_bps
        pos.pnl_usd = pnl_usd

        # Update stats
        if pnl_bps > 0:
            self.wins += 1
        else:
            self.losses += 1

        self.total_pnl_bps += pnl_bps
        self.total_pnl_usd += pnl_usd

        self.trades.append(pos)
        self._save_trade(pos)

        direction_str = "LONG" if pos.direction == 1 else "SHORT"
        logger.info(f"CLOSE {direction_str} #{pos.id} @ ${exit_price:,.2f} ({exit_reason})")
        logger.info(f"  P&L: {pnl_bps:+.1f} bps (${pnl_usd:+.2f}) [after fees]")

        win_rate = self.wins / len(self.trades) * 100 if self.trades else 0
        logger.info(f"  Stats: {self.wins}W/{self.losses}L ({win_rate:.0f}%), Total: {self.total_pnl_bps:+.1f} bps")

        self.position = None

    def _save_trade(self, trade: Trade):
        """Save trade to database."""
        conn = sqlite3.connect(TRADES_DB)
        conn.execute("""
            INSERT INTO trades (id, entry_time, entry_price, direction, size_usd,
                               exit_time, exit_price, exit_reason, pnl_bps, pnl_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (trade.id, trade.entry_time, trade.entry_price, trade.direction,
              trade.size_usd, trade.exit_time, trade.exit_price, trade.exit_reason,
              trade.pnl_bps, trade.pnl_usd))
        conn.commit()
        conn.close()

    def _log_signal(self, signal: Signal, acted_on: bool):
        """Log signal to database."""
        conn = sqlite3.connect(TRADES_DB)
        conn.execute("""
            INSERT INTO signals (timestamp, direction, confidence, reason, price,
                                tx_zscore, vol_zscore, acted_on)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (signal.timestamp, signal.direction, signal.confidence, signal.reason,
              signal.price, signal.tx_zscore, signal.vol_zscore, int(acted_on)))
        conn.commit()
        conn.close()

    def _log_stats(self, price: float):
        """Log current stats to database."""
        win_rate = self.wins / len(self.trades) * 100 if self.trades else 0
        conn = sqlite3.connect(TRADES_DB)
        conn.execute("""
            INSERT OR REPLACE INTO stats
            (timestamp, total_trades, wins, losses, win_rate, total_pnl_bps, total_pnl_usd, current_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (time.time(), len(self.trades), self.wins, self.losses, win_rate,
              self.total_pnl_bps, self.total_pnl_usd, price))
        conn.commit()
        conn.close()

    def run(self):
        """Main trading loop."""
        logger.info("="*60)
        logger.info("  VPS PAPER TRADER STARTING")
        logger.info("="*60)
        logger.info(f"Capital: ${self.capital_usd}")
        logger.info(f"Leverage: {self.leverage}x")
        logger.info(f"Signal threshold: z > {self.signal_threshold}")
        logger.info(f"Hold time: {self.hold_time_seconds}s")
        logger.info(f"Stop loss: {self.stop_loss_pct*100:.1f}%")
        logger.info(f"Take profit: {self.take_profit_pct*100:.1f}%")
        logger.info("="*60)
        logger.info("Strategy: SHORT on blockchain spikes, LONG on low activity")
        logger.info("="*60)

        last_status_time = 0
        status_interval = 300  # Log status every 5 minutes

        while True:
            try:
                # Get latest metrics
                metrics = self.get_recent_metrics()

                if not metrics:
                    time.sleep(1)
                    continue

                current_time = time.time()
                current_price = metrics['price']

                # Check for exit if in position
                if self.position:
                    exit_reason = self.check_exit(current_price, metrics['timestamp'])
                    if exit_reason:
                        self.close_position(current_price, exit_reason)

                # Generate signal if not in position
                if not self.position:
                    # Check cooldown
                    if current_time - self.last_trade_time >= self.cooldown_seconds:
                        signal = self.generate_signal(metrics)
                        if signal:
                            self.open_position(signal)
                        elif metrics['tx_zscore'] > 1.5 or metrics['vol_zscore'] > 1.5:
                            # Log interesting signals we didn't act on
                            pass

                # Log status periodically
                if current_time - last_status_time >= status_interval:
                    self._log_stats(current_price)
                    logger.info("-"*40)
                    logger.info(f"STATUS | Price: ${current_price:,.2f}")
                    logger.info(f"STATUS | Trades: {len(self.trades)} | Win rate: {self.wins}/{len(self.trades) if self.trades else 0}")
                    logger.info(f"STATUS | Total P&L: {self.total_pnl_bps:+.1f} bps (${self.total_pnl_usd:+.2f})")
                    if self.position:
                        logger.info(f"STATUS | IN POSITION: {'LONG' if self.position.direction == 1 else 'SHORT'}")
                    logger.info("-"*40)
                    last_status_time = current_time

                time.sleep(1)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                if self.position:
                    metrics = self.get_recent_metrics()
                    if metrics:
                        self.close_position(metrics['price'], "SHUTDOWN")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)

        # Final summary
        logger.info("="*60)
        logger.info("  FINAL SUMMARY")
        logger.info("="*60)
        logger.info(f"Total trades: {len(self.trades)}")
        logger.info(f"Wins: {self.wins} | Losses: {self.losses}")
        if self.trades:
            logger.info(f"Win rate: {self.wins/len(self.trades)*100:.1f}%")
        logger.info(f"Total P&L: {self.total_pnl_bps:+.1f} bps (${self.total_pnl_usd:+.2f})")
        logger.info("="*60)


def main():
    trader = PaperTrader(
        capital_usd=100.0,
        leverage=10,
        signal_threshold=2.0,
        hold_time_seconds=300,
        stop_loss_pct=0.003,
        take_profit_pct=0.006,
        cooldown_seconds=60
    )
    trader.run()


if __name__ == "__main__":
    main()
