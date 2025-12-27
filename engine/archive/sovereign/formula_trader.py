#!/usr/bin/env python3
"""
FORMULA TRADER - Full Mathematical Edge Trading System
======================================================

Uses FormulaConnector to route blockchain data through 3 formula engines:
1. Adaptive Trading Engine (IDs 10001-10005) - Flow-based formulas
2. Pattern Recognition Engine (IDs 20001-20012) - HMM, stat arb, ML
3. RenTech Pattern Engine (IDs 72001-72099) - Advanced RenTech patterns

ENSEMBLE VOTING:
- All 3 agree: 1.5x confidence boost (UNANIMOUS)
- 2 of 3 agree: 1.3x confidence boost (MAJORITY)
- Single high-confidence: Use if > 0.7
- Conflicting: WAIT

This provides 10-60 second information edge from blockchain before exchange APIs.
"""

import os
import sys
import json
import time
import logging
import asyncio
import urllib.request
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, List
from pathlib import Path
from collections import deque
import threading

# Add sovereign to path for imports
sys.path.insert(0, '/root')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Trading parameters
LEVERAGE = 35.0
BASE_TAKE_PROFIT_PCT = 0.008    # 0.8% base, adjusted by formula
BASE_STOP_LOSS_PCT = 0.005     # 0.5% base, adjusted by formula
MAX_POSITIONS = 4
MIN_CONFIDENCE = 0.4           # Minimum ensemble confidence to trade
COOLDOWN_SECONDS = 30

# Capital
STARTING_CAPITAL = 100.0

# Bitcoin Core ZMQ endpoint
ZMQ_ENDPOINT = "tcp://127.0.0.1:28332"

# Exchange addresses (gzipped JSON)
EXCHANGES_JSON = "/root/exchanges.json.gz"

# Paths
LOG_FILE = Path("/root/formula.log")
TRADES_FILE = Path("/root/formula_trades.json")
STATS_FILE = Path("/root/formula_stats.json")

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE) if LOG_FILE.parent.exists() else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Position:
    id: str
    direction: int              # +1 LONG, -1 SHORT
    size_usd: float
    entry_price: float
    entry_time: float
    stop_loss: float
    take_profit: float
    ensemble_type: str          # 'unanimous', 'majority', or engine name
    vote_count: int
    kelly_fraction: float
    signal_data: Dict


@dataclass
class Trade:
    id: str
    direction: int
    size_usd: float
    entry_price: float
    entry_time: float
    exit_price: float
    exit_time: float
    exit_reason: str
    pnl_usd: float
    pnl_pct: float
    ensemble_type: str
    vote_count: int


# =============================================================================
# PAPER EXECUTOR (for testing)
# =============================================================================

class PaperExecutor:
    """Paper trading executor for testing formula signals."""

    def __init__(self, capital: float = 100.0, leverage: float = 35.0):
        self.capital = capital
        self.leverage = leverage
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []

    def open_position(self, direction: int, size_usd: float, entry_price: float,
                      stop_loss: float, take_profit: float, signal: Dict) -> bool:
        """Open a position."""
        if self.position:
            logger.warning("[PAPER] Already have open position")
            return False

        self.position = Position(
            id=f"paper_{int(time.time())}",
            direction=direction,
            size_usd=size_usd,
            entry_price=entry_price,
            entry_time=time.time(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            ensemble_type=signal.get('ensemble_type', 'unknown'),
            vote_count=signal.get('vote_count', 0),
            kelly_fraction=signal.get('kelly_fraction', 0.05),
            signal_data=signal
        )

        dir_str = "LONG" if direction == 1 else "SHORT"
        logger.info(f"[PAPER] OPENED {dir_str} ${size_usd:.2f} @ ${entry_price:.2f} | "
                    f"SL=${stop_loss:.2f} TP=${take_profit:.2f} | {signal.get('ensemble_type')}")
        return True

    def close_position(self, exit_price: float, reason: str) -> Optional[Trade]:
        """Close current position."""
        if not self.position:
            return None

        # Calculate PnL
        if self.position.direction == 1:  # LONG
            pnl_pct = (exit_price - self.position.entry_price) / self.position.entry_price
        else:  # SHORT
            pnl_pct = (self.position.entry_price - exit_price) / self.position.entry_price

        pnl_usd = self.position.size_usd * pnl_pct

        trade = Trade(
            id=self.position.id,
            direction=self.position.direction,
            size_usd=self.position.size_usd,
            entry_price=self.position.entry_price,
            entry_time=self.position.entry_time,
            exit_price=exit_price,
            exit_time=time.time(),
            exit_reason=reason,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct * 100,
            ensemble_type=self.position.ensemble_type,
            vote_count=self.position.vote_count
        )

        self.trades.append(trade)
        self.capital += pnl_usd

        dir_str = "LONG" if trade.direction == 1 else "SHORT"
        pnl_sign = "+" if pnl_usd >= 0 else ""
        logger.info(f"[PAPER] CLOSED {dir_str} @ ${exit_price:.2f} | "
                    f"PnL: {pnl_sign}${pnl_usd:.2f} ({pnl_sign}{trade.pnl_pct:.2f}%) | {reason}")

        self.position = None
        return trade

    def check_stops(self, current_price: float) -> Optional[Trade]:
        """Check if current price hits stop loss or take profit."""
        if not self.position:
            return None

        if self.position.direction == 1:  # LONG
            if current_price <= self.position.stop_loss:
                return self.close_position(current_price, "STOP_LOSS")
            elif current_price >= self.position.take_profit:
                return self.close_position(current_price, "TAKE_PROFIT")
        else:  # SHORT
            if current_price >= self.position.stop_loss:
                return self.close_position(current_price, "STOP_LOSS")
            elif current_price <= self.position.take_profit:
                return self.close_position(current_price, "TAKE_PROFIT")

        return None

    def get_stats(self) -> Dict:
        """Get trading statistics."""
        if not self.trades:
            return {
                'trades': 0, 'wins': 0, 'losses': 0,
                'win_rate': 0, 'total_pnl': 0, 'capital': self.capital,
                'positions': 1 if self.position else 0
            }

        wins = [t for t in self.trades if t.pnl_usd > 0]
        losses = [t for t in self.trades if t.pnl_usd <= 0]

        return {
            'trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades) * 100,
            'total_pnl': sum(t.pnl_usd for t in self.trades),
            'avg_win': sum(t.pnl_usd for t in wins) / len(wins) if wins else 0,
            'avg_loss': sum(t.pnl_usd for t in losses) / len(losses) if losses else 0,
            'capital': self.capital,
            'positions': 1 if self.position else 0,
            'by_ensemble': self._stats_by_ensemble()
        }

    def _stats_by_ensemble(self) -> Dict:
        """Get stats broken down by ensemble type."""
        stats = {}
        for trade in self.trades:
            etype = trade.ensemble_type
            if etype not in stats:
                stats[etype] = {'trades': 0, 'wins': 0, 'pnl': 0}
            stats[etype]['trades'] += 1
            if trade.pnl_usd > 0:
                stats[etype]['wins'] += 1
            stats[etype]['pnl'] += trade.pnl_usd
        return stats


# =============================================================================
# FORMULA TRADER
# =============================================================================

class FormulaTrader:
    """
    Main trading system using FormulaConnector for mathematical edge.
    """

    def __init__(self, mode: str = 'paper'):
        self.mode = mode
        self.capital = STARTING_CAPITAL
        self.leverage = LEVERAGE
        self.running = False
        self.last_trade_time = 0
        self.current_price = 95000.0  # Updated from exchange
        self.price_lock = threading.Lock()

        # Initialize executor
        if mode == 'paper':
            self.executor = PaperExecutor(capital=self.capital, leverage=self.leverage)
        else:
            # Live executor would go here
            logger.warning("Live mode not yet implemented, using paper")
            self.executor = PaperExecutor(capital=self.capital, leverage=self.leverage)

        # FormulaConnector initialized in start()
        self.connector = None

    def on_signal(self, signal: Dict) -> None:
        """
        Callback when FormulaConnector generates a trading signal.

        This is called by FormulaConnector when the 3-engine ensemble
        produces an actionable signal.
        """
        # Check cooldown
        if time.time() - self.last_trade_time < COOLDOWN_SECONDS:
            remaining = COOLDOWN_SECONDS - (time.time() - self.last_trade_time)
            logger.debug(f"[SIGNAL] Skipping - cooldown {remaining:.0f}s remaining")
            return

        # Check confidence threshold
        confidence = signal.get('confidence', 0)
        if confidence < MIN_CONFIDENCE:
            logger.debug(f"[SIGNAL] Skipping - confidence {confidence:.2f} < {MIN_CONFIDENCE}")
            return

        # Check if we already have a position
        if self.executor.position:
            logger.debug("[SIGNAL] Skipping - already have position")
            return

        # Extract signal parameters
        direction = signal.get('direction', 0)
        if direction == 0:
            return

        # Use formula-derived parameters or defaults
        stop_loss_pct = signal.get('stop_loss', BASE_STOP_LOSS_PCT)
        take_profit_pct = signal.get('take_profit', BASE_TAKE_PROFIT_PCT)
        kelly = signal.get('kelly_fraction', 0.05)
        position_size_pct = signal.get('position_size', kelly)

        # Get current price
        with self.price_lock:
            entry_price = signal.get('price', self.current_price)

        # Calculate position size (Kelly-based with leverage)
        size_usd = self.capital * position_size_pct * self.leverage

        # Calculate stop loss and take profit prices
        if direction == 1:  # LONG
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:  # SHORT
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)

        # Log signal details
        dir_str = "LONG" if direction == 1 else "SHORT"
        ensemble = signal.get('ensemble_type', 'unknown')
        votes = signal.get('vote_count', 0)
        logger.info(f"[SIGNAL] {dir_str} | conf={confidence:.2f} | {ensemble} ({votes}/3) | "
                    f"kelly={kelly:.3f} | BTC={signal.get('btc_amount', 0):.2f}")

        # Execute trade
        success = self.executor.open_position(
            direction=direction,
            size_usd=size_usd,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal=signal
        )

        if success:
            self.last_trade_time = time.time()

            # Record trade with FormulaConnector for learning
            if self.connector:
                # The connector will learn from this trade result
                pass

    def update_price(self) -> None:
        """Fetch current BTC price from exchange."""
        try:
            url = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                if 'result' in data and 'XXBTZUSD' in data['result']:
                    price = float(data['result']['XXBTZUSD']['c'][0])
                    with self.price_lock:
                        self.current_price = price
                    if self.connector:
                        self.connector.set_reference_price(price)
        except Exception as e:
            logger.debug(f"Price fetch error: {e}")

    def check_positions(self) -> None:
        """Check and manage open positions."""
        with self.price_lock:
            price = self.current_price

        # Check stops
        trade = self.executor.check_stops(price)
        if trade and self.connector:
            # Record result for formula learning
            signal_data = self.executor.trades[-1].signal_data if self.executor.trades else {}
            self.connector.record_trade_result(signal_data, trade.exit_price)

    def print_stats(self) -> None:
        """Print current statistics."""
        stats = self.executor.get_stats()

        logger.info("=" * 60)
        logger.info(f"[STATS] Capital: ${stats['capital']:.2f} | "
                    f"Trades: {stats['trades']} | "
                    f"Win Rate: {stats['win_rate']:.1f}%")
        logger.info(f"[STATS] Total PnL: ${stats['total_pnl']:.2f} | "
                    f"Position: {'OPEN' if stats['positions'] else 'NONE'}")

        if self.connector:
            conn_stats = self.connector.get_stats()
            logger.info(f"[CONNECTOR] Ticks: {conn_stats.get('ticks_processed', 0)} | "
                        f"Signals: {conn_stats.get('signals_generated', 0)}")
            logger.info(f"[ENGINES] Adaptive: {conn_stats.get('adaptive_signals', 0)} | "
                        f"Pattern: {conn_stats.get('pattern_signals', 0)} | "
                        f"RenTech: {conn_stats.get('rentech_signals', 0)}")

        # Stats by ensemble type
        by_ensemble = stats.get('by_ensemble', {})
        for etype, estats in by_ensemble.items():
            wr = (estats['wins'] / estats['trades'] * 100) if estats['trades'] else 0
            logger.info(f"[{etype.upper()}] Trades: {estats['trades']} | "
                        f"Wins: {estats['wins']} ({wr:.0f}%) | PnL: ${estats['pnl']:.2f}")

        logger.info("=" * 60)

    def save_state(self) -> None:
        """Save current state to files."""
        try:
            # Save trades
            trades_data = [asdict(t) for t in self.executor.trades]
            with open(TRADES_FILE, 'w') as f:
                json.dump(trades_data, f, indent=2)

            # Save stats
            stats = self.executor.get_stats()
            if self.connector:
                stats['connector'] = self.connector.get_stats()
            with open(STATS_FILE, 'w') as f:
                json.dump(stats, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def start(self) -> None:
        """Start the formula trading system."""
        logger.info("=" * 60)
        logger.info("FORMULA TRADER - Mathematical Edge Trading System")
        logger.info("=" * 60)
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Capital: ${self.capital:.2f}")
        logger.info(f"Leverage: {self.leverage}x")
        logger.info(f"ZMQ Endpoint: {ZMQ_ENDPOINT}")
        logger.info("=" * 60)
        logger.info("Engines: Adaptive (10001-10005) + Pattern (20001-20012) + RenTech (72001-72099)")
        logger.info("Ensemble: 3-way voting with confidence boosting")
        logger.info("=" * 60)

        try:
            # Import FormulaConnector
            from sovereign.blockchain.formula_connector import FormulaConnector

            # Initialize connector with all 3 engines
            self.connector = FormulaConnector(
                zmq_endpoint=ZMQ_ENDPOINT,
                json_path=EXCHANGES_JSON if os.path.exists(EXCHANGES_JSON) else None,
                on_signal=self.on_signal,
                enable_pattern_recognition=True,
                enable_rentech=True,
                rentech_mode="full"
            )

            # Start connector (starts blockchain feed)
            if not self.connector.start():
                logger.error("Failed to start FormulaConnector")
                return

            logger.info("[STARTUP] FormulaConnector started successfully")
            logger.info("[STARTUP] Listening for blockchain transactions...")

        except ImportError as e:
            logger.error(f"Import error: {e}")
            logger.error("Make sure formula engines are in /root/sovereign/")
            return
        except Exception as e:
            logger.error(f"Startup error: {e}")
            return

        self.running = True
        last_price_update = 0
        last_stats_print = 0

        try:
            while self.running:
                now = time.time()

                # Update price every 5 seconds
                if now - last_price_update >= 5:
                    self.update_price()
                    last_price_update = now

                # Check positions every second
                self.check_positions()

                # Print stats every 60 seconds
                if now - last_stats_print >= 60:
                    self.print_stats()
                    self.save_state()
                    last_stats_print = now

                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.running = False
            if self.connector:
                self.connector.stop()
            self.save_state()
            self.print_stats()
            logger.info("Formula Trader stopped")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Formula Trader - Mathematical Edge Trading')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                        help='Trading mode (default: paper)')
    args = parser.parse_args()

    trader = FormulaTrader(mode=args.mode)
    trader.start()


if __name__ == '__main__':
    main()
