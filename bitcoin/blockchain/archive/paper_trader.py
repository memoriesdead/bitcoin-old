#!/usr/bin/env python3
"""
BLOCKCHAIN PAPER TRADER
=======================
Paper trading for deterministic blockchain signals.

Integrates with master_exchange_pipeline to:
1. Receive real-time signals (SHORT on inflow, LONG on exhaustion)
2. Simulate trades with realistic fees/slippage
3. Track P&L per exchange
4. Log everything to SQLite

Usage:
    python3 paper_trader.py                    # Paper trade (default)
    python3 paper_trader.py --capital 10000    # Set starting capital
    python3 paper_trader.py --leverage 2       # Use 2x leverage
"""

import os
import sys
import time
import json
import sqlite3
import threading
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict

# Add paths
sys.path.insert(0, '/root/sovereign')
sys.path.insert(0, '/root/sovereign/blockchain')

# Also try local paths for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PaperTraderConfig:
    """Paper trader configuration."""
    # Capital
    initial_capital: float = 10000.0    # Starting capital in USD
    leverage: float = 1.0               # Leverage multiplier
    max_position_pct: float = 0.25      # Max 25% of capital per trade

    # Risk management
    stop_loss_pct: float = 0.02         # 2% stop loss
    take_profit_pct: float = 0.04       # 4% take profit
    max_positions: int = 5              # Max concurrent positions

    # Exchange fees (realistic)
    fees: Dict[str, float] = field(default_factory=lambda: {
        'coinbase': 0.006,    # 0.6% taker
        'kraken': 0.0026,     # 0.26% taker
        'bitstamp': 0.005,    # 0.5% taker
        'gemini': 0.004,      # 0.4% taker
        'crypto.com': 0.004,  # 0.4% taker
        'default': 0.001,     # 0.1% default
    })

    # Slippage (realistic)
    slippage: Dict[str, float] = field(default_factory=lambda: {
        'coinbase': 0.0003,   # 0.03% base slippage
        'kraken': 0.0002,
        'bitstamp': 0.0003,
        'gemini': 0.0002,
        'crypto.com': 0.0003,
        'default': 0.0001,
    })

    # USA exchanges only
    usa_exchanges: Set[str] = field(default_factory=lambda: {
        'coinbase', 'kraken', 'bitstamp', 'gemini', 'crypto.com'
    })

    # Database
    db_path: str = "/root/sovereign/paper_trades.db"

    # Signal settings
    min_confidence: float = 0.1         # Minimum signal confidence
    position_hold_seconds: int = 300    # Default hold time (5 min)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Position:
    """Active trading position."""
    id: int
    exchange: str
    direction: str          # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: float
    size_usd: float         # Position size in USD
    size_btc: float         # Position size in BTC
    stop_loss: float
    take_profit: float
    signal_amount_btc: float
    signal_confidence: float
    signal_reason: str

    # Tracking
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: str = 'OPEN'    # OPEN, CLOSED, STOPPED, TP_HIT


@dataclass
class Trade:
    """Completed trade record."""
    id: int
    exchange: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    size_usd: float
    size_btc: float
    pnl_usd: float
    pnl_pct: float
    fees_paid: float
    slippage_cost: float
    exit_reason: str        # 'TP', 'SL', 'TIMEOUT', 'MANUAL'
    signal_amount_btc: float
    signal_confidence: float


# =============================================================================
# PAPER TRADER ENGINE
# =============================================================================

class BlockchainPaperTrader:
    """
    Paper trading engine for blockchain signals.

    Features:
    - Real-time signal processing from master_exchange_pipeline
    - Per-exchange position tracking
    - Realistic fee and slippage simulation
    - SQLite trade logging
    - P&L tracking and reporting
    """

    def __init__(self, config: PaperTraderConfig = None):
        self.config = config or PaperTraderConfig()

        # State
        self.capital = self.config.initial_capital
        self.positions: Dict[int, Position] = {}
        self.trades: List[Trade] = []
        self.next_position_id = 1
        self.next_trade_id = 1

        # Per-exchange stats
        self.exchange_stats: Dict[str, Dict] = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
        })

        # Threading
        self.running = False
        self.lock = threading.Lock()

        # Initialize database
        self._init_db()

        print("=" * 70)
        print("BLOCKCHAIN PAPER TRADER")
        print("=" * 70)
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        print(f"Leverage: {self.config.leverage}x")
        print(f"Max Position: {self.config.max_position_pct * 100:.0f}%")
        print(f"Exchanges: {', '.join(sorted(self.config.usa_exchanges))}")
        print("=" * 70)
        print()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()

        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY,
                exchange TEXT,
                direction TEXT,
                entry_price REAL,
                entry_time REAL,
                size_usd REAL,
                size_btc REAL,
                stop_loss REAL,
                take_profit REAL,
                signal_amount_btc REAL,
                signal_confidence REAL,
                signal_reason TEXT,
                status TEXT,
                exit_price REAL,
                exit_time REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                fees_paid REAL,
                exit_reason TEXT
            )
        """)

        # Trades table (completed)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                exchange TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                entry_time REAL,
                exit_time REAL,
                size_usd REAL,
                size_btc REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                fees_paid REAL,
                slippage_cost REAL,
                exit_reason TEXT,
                signal_amount_btc REAL,
                signal_confidence REAL
            )
        """)

        # Equity curve
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_curve (
                timestamp REAL PRIMARY KEY,
                capital REAL,
                open_positions INTEGER,
                total_pnl REAL
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_exchange ON trades(exchange)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(entry_time)")

        conn.commit()
        conn.close()

        print(f"[DB] Initialized at {self.config.db_path}")

    def _get_fee(self, exchange: str) -> float:
        """Get fee rate for exchange."""
        return self.config.fees.get(exchange, self.config.fees['default'])

    def _get_slippage(self, exchange: str) -> float:
        """Get slippage rate for exchange."""
        return self.config.slippage.get(exchange, self.config.slippage['default'])

    def _calculate_position_size(self, signal_confidence: float) -> float:
        """Calculate position size based on confidence."""
        base_size = self.capital * self.config.max_position_pct * self.config.leverage
        # Scale by confidence
        return base_size * min(signal_confidence + 0.5, 1.0)

    def on_signal(self, exchange: str, direction: str, amount_btc: float,
                  confidence: float, reason: str, price: float, pattern: str):
        """
        Handle incoming signal from blockchain pipeline.

        Args:
            exchange: Exchange name
            direction: 'LONG' or 'SHORT'
            amount_btc: Signal flow amount
            confidence: Signal confidence (0-1)
            reason: Signal reason text
            price: Current BTC price
            pattern: Signal pattern type
        """
        with self.lock:
            # Skip if below confidence threshold
            if confidence < self.config.min_confidence:
                return

            # Skip non-USA exchanges
            if exchange not in self.config.usa_exchanges:
                return

            # Skip if max positions reached
            open_positions = sum(1 for p in self.positions.values() if p.status == 'OPEN')
            if open_positions >= self.config.max_positions:
                print(f"[SKIP] Max positions ({self.config.max_positions}) reached")
                return

            # Skip if already have position on this exchange
            for pos in self.positions.values():
                if pos.exchange == exchange and pos.status == 'OPEN':
                    print(f"[SKIP] Already have position on {exchange}")
                    return

            # Calculate entry price with slippage
            slippage = self._get_slippage(exchange)
            if direction == 'LONG':
                entry_price = price * (1 + slippage)  # Buy higher
            else:
                entry_price = price * (1 - slippage)  # Sell lower

            # Calculate position size
            size_usd = self._calculate_position_size(confidence)
            size_btc = size_usd / entry_price

            # Calculate stop loss and take profit
            if direction == 'LONG':
                stop_loss = entry_price * (1 - self.config.stop_loss_pct)
                take_profit = entry_price * (1 + self.config.take_profit_pct)
            else:
                stop_loss = entry_price * (1 + self.config.stop_loss_pct)
                take_profit = entry_price * (1 - self.config.take_profit_pct)

            # Calculate entry fee
            fee = size_usd * self._get_fee(exchange)

            # Create position
            position = Position(
                id=self.next_position_id,
                exchange=exchange,
                direction=direction,
                entry_price=entry_price,
                entry_time=time.time(),
                size_usd=size_usd,
                size_btc=size_btc,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal_amount_btc=amount_btc,
                signal_confidence=confidence,
                signal_reason=reason,
                current_price=entry_price,
            )

            self.positions[position.id] = position
            self.next_position_id += 1

            # Deduct fee from capital
            self.capital -= fee

            # Log to database
            self._save_position(position)

            # Print
            print(f"\n[OPEN] {direction} {exchange.upper()}")
            print(f"       Entry: ${entry_price:,.2f} | Size: ${size_usd:,.2f} ({size_btc:.4f} BTC)")
            print(f"       SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}")
            print(f"       Signal: {amount_btc:.1f} BTC ({pattern}) | Conf: {confidence:.0%}")
            print(f"       Fee: ${fee:.2f}")

    def update_prices(self, prices: Dict[str, float]):
        """
        Update positions with current prices.

        Args:
            prices: Dict of exchange -> current price
        """
        with self.lock:
            now = time.time()

            for pos_id, pos in list(self.positions.items()):
                if pos.status != 'OPEN':
                    continue

                # Get price for this exchange
                price = prices.get(pos.exchange)
                if not price:
                    continue

                pos.current_price = price

                # Calculate unrealized P&L
                if pos.direction == 'LONG':
                    pos.unrealized_pnl = (price - pos.entry_price) / pos.entry_price * pos.size_usd
                else:  # SHORT
                    pos.unrealized_pnl = (pos.entry_price - price) / pos.entry_price * pos.size_usd

                # Check stop loss
                if pos.direction == 'LONG' and price <= pos.stop_loss:
                    self._close_position(pos, price, 'SL')
                elif pos.direction == 'SHORT' and price >= pos.stop_loss:
                    self._close_position(pos, price, 'SL')

                # Check take profit
                elif pos.direction == 'LONG' and price >= pos.take_profit:
                    self._close_position(pos, price, 'TP')
                elif pos.direction == 'SHORT' and price <= pos.take_profit:
                    self._close_position(pos, price, 'TP')

                # Check timeout
                elif now - pos.entry_time >= self.config.position_hold_seconds:
                    self._close_position(pos, price, 'TIMEOUT')

    def _close_position(self, pos: Position, exit_price: float, reason: str):
        """Close a position."""
        # Apply exit slippage
        slippage = self._get_slippage(pos.exchange)
        if pos.direction == 'LONG':
            exit_price = exit_price * (1 - slippage)  # Sell lower
        else:
            exit_price = exit_price * (1 + slippage)  # Buy higher

        # Calculate P&L
        if pos.direction == 'LONG':
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl_usd = pnl_pct * pos.size_usd

        # Calculate exit fee
        exit_fee = pos.size_usd * self._get_fee(pos.exchange)
        total_fees = pos.size_usd * self._get_fee(pos.exchange) * 2  # Entry + exit

        # Net P&L after fees
        net_pnl = pnl_usd - exit_fee

        # Update capital
        self.capital += pos.size_usd + net_pnl

        # Update position
        pos.status = reason

        # Create trade record
        trade = Trade(
            id=self.next_trade_id,
            exchange=pos.exchange,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=time.time(),
            size_usd=pos.size_usd,
            size_btc=pos.size_btc,
            pnl_usd=net_pnl,
            pnl_pct=pnl_pct * 100,
            fees_paid=total_fees,
            slippage_cost=slippage * pos.size_usd * 2,
            exit_reason=reason,
            signal_amount_btc=pos.signal_amount_btc,
            signal_confidence=pos.signal_confidence,
        )

        self.trades.append(trade)
        self.next_trade_id += 1

        # Update exchange stats
        stats = self.exchange_stats[pos.exchange]
        stats['trades'] += 1
        stats['total_pnl'] += net_pnl
        stats['total_fees'] += total_fees
        if net_pnl > 0:
            stats['wins'] += 1
        else:
            stats['losses'] += 1

        # Save to database
        self._save_trade(trade)

        # Print
        color = '\033[92m' if net_pnl > 0 else '\033[91m'
        reset = '\033[0m'

        print(f"\n{color}[CLOSE]{reset} {pos.direction} {pos.exchange.upper()} - {reason}")
        print(f"        Entry: ${pos.entry_price:,.2f} -> Exit: ${exit_price:,.2f}")
        print(f"        P&L: {color}${net_pnl:+,.2f}{reset} ({pnl_pct*100:+.2f}%)")
        print(f"        Fees: ${total_fees:.2f} | Capital: ${self.capital:,.2f}")

    def _save_position(self, pos: Position):
        """Save position to database."""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO positions (id, exchange, direction, entry_price, entry_time,
                                   size_usd, size_btc, stop_loss, take_profit,
                                   signal_amount_btc, signal_confidence, signal_reason, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (pos.id, pos.exchange, pos.direction, pos.entry_price, pos.entry_time,
              pos.size_usd, pos.size_btc, pos.stop_loss, pos.take_profit,
              pos.signal_amount_btc, pos.signal_confidence, pos.signal_reason, pos.status))

        conn.commit()
        conn.close()

    def _save_trade(self, trade: Trade):
        """Save trade to database."""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO trades (id, exchange, direction, entry_price, exit_price,
                               entry_time, exit_time, size_usd, size_btc,
                               pnl_usd, pnl_pct, fees_paid, slippage_cost,
                               exit_reason, signal_amount_btc, signal_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (trade.id, trade.exchange, trade.direction, trade.entry_price,
              trade.exit_price, trade.entry_time, trade.exit_time, trade.size_usd,
              trade.size_btc, trade.pnl_usd, trade.pnl_pct, trade.fees_paid,
              trade.slippage_cost, trade.exit_reason, trade.signal_amount_btc,
              trade.signal_confidence))

        # Update equity curve
        cursor.execute("""
            INSERT OR REPLACE INTO equity_curve (timestamp, capital, open_positions, total_pnl)
            VALUES (?, ?, ?, ?)
        """, (time.time(), self.capital,
              sum(1 for p in self.positions.values() if p.status == 'OPEN'),
              sum(t.pnl_usd for t in self.trades)))

        conn.commit()
        conn.close()

    def print_stats(self):
        """Print trading statistics."""
        print()
        print("=" * 70)
        print("PAPER TRADING STATISTICS")
        print("=" * 70)

        # Overall stats
        total_pnl = sum(t.pnl_usd for t in self.trades)
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.pnl_usd > 0)
        losses = total_trades - wins
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        print(f"\nCapital: ${self.capital:,.2f} (started: ${self.config.initial_capital:,.2f})")
        print(f"Total P&L: ${total_pnl:+,.2f} ({total_pnl/self.config.initial_capital*100:+.2f}%)")
        print(f"Trades: {total_trades} | Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%")

        # Open positions
        open_positions = [p for p in self.positions.values() if p.status == 'OPEN']
        if open_positions:
            print(f"\nOpen Positions: {len(open_positions)}")
            for pos in open_positions:
                print(f"  {pos.direction} {pos.exchange}: ${pos.size_usd:.2f} @ ${pos.entry_price:,.2f}")

        # Per-exchange stats
        print(f"\n{'Exchange':<12} {'Trades':>8} {'Wins':>6} {'Losses':>6} {'Win%':>8} {'P&L':>12}")
        print("-" * 60)

        for exchange in sorted(self.config.usa_exchanges):
            stats = self.exchange_stats[exchange]
            if stats['trades'] == 0:
                continue
            win_rate = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
            print(f"{exchange:<12} {stats['trades']:>8} {stats['wins']:>6} {stats['losses']:>6} "
                  f"{win_rate:>7.1f}% ${stats['total_pnl']:>+11,.2f}")

        print()

    def get_stats_dict(self) -> Dict:
        """Get stats as dictionary."""
        total_pnl = sum(t.pnl_usd for t in self.trades)
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.pnl_usd > 0)

        return {
            'capital': self.capital,
            'initial_capital': self.config.initial_capital,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl / self.config.initial_capital * 100,
            'total_trades': total_trades,
            'wins': wins,
            'losses': total_trades - wins,
            'win_rate': wins / total_trades * 100 if total_trades > 0 else 0,
            'open_positions': sum(1 for p in self.positions.values() if p.status == 'OPEN'),
            'exchange_stats': dict(self.exchange_stats),
        }


# =============================================================================
# INTEGRATED PIPELINE
# =============================================================================

class IntegratedPaperPipeline:
    """
    Integrated pipeline that runs paper trading on blockchain signals.

    Combines:
    - MasterExchangePipeline for signal generation
    - BlockchainPaperTrader for trade simulation
    - Real-time price updates
    """

    def __init__(self, capital: float = 10000.0, leverage: float = 1.0):
        # Import pipeline
        try:
            from master_exchange_pipeline import MasterExchangePipeline, PipelineConfig
        except ImportError:
            from blockchain.master_exchange_pipeline import MasterExchangePipeline, PipelineConfig

        # Create pipeline config
        pipeline_config = PipelineConfig(
            test_mode=True,
            short_only=False,
            usa_only=True,
        )

        # Create paper trader config
        trader_config = PaperTraderConfig(
            initial_capital=capital,
            leverage=leverage,
        )

        # Initialize components
        self.pipeline = MasterExchangePipeline(pipeline_config)
        self.trader = BlockchainPaperTrader(trader_config)

        # Override signal handler
        self._original_print_signal = self.pipeline._print_signal
        self.pipeline._print_signal = self._on_pipeline_signal

        self.running = False

    def _on_pipeline_signal(self, signal):
        """Handle signal from pipeline."""
        # Call original printer
        self._original_print_signal(signal)

        # Send to paper trader
        self.trader.on_signal(
            exchange=signal.exchange,
            direction=signal.direction,
            amount_btc=signal.amount_btc,
            confidence=signal.confidence,
            reason=signal.reason,
            price=signal.price,
            pattern=signal.pattern,
        )

    def _price_update_loop(self):
        """Update prices for open positions."""
        while self.running:
            try:
                # Get prices from pipeline's price feed
                prices = {}
                for exchange in self.trader.config.usa_exchanges:
                    price = self.pipeline.price_feed.get_price(exchange)
                    if price:
                        prices[exchange] = price

                # Update trader
                self.trader.update_prices(prices)

            except Exception as e:
                print(f"[PRICE] Error: {e}")

            time.sleep(1)

    def start(self):
        """Start integrated pipeline."""
        self.running = True

        # Start pipeline
        self.pipeline.start()

        # Start price update thread
        self.price_thread = threading.Thread(
            target=self._price_update_loop,
            daemon=True
        )
        self.price_thread.start()

    def stop(self):
        """Stop integrated pipeline."""
        self.running = False
        self.pipeline.stop()
        self.trader.print_stats()

    def run(self):
        """Run the integrated pipeline."""
        self.start()

        try:
            while self.running:
                time.sleep(60)
                self.pipeline.print_stats()
                self.trader.print_stats()
        except KeyboardInterrupt:
            print("\n[Ctrl+C] Stopping...")
        finally:
            self.stop()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Blockchain Paper Trader')
    parser.add_argument('--capital', type=float, default=10000.0, help='Starting capital')
    parser.add_argument('--leverage', type=float, default=1.0, help='Leverage multiplier')
    args = parser.parse_args()

    print()
    print("Starting Blockchain Paper Trader...")
    print()

    # Run integrated pipeline
    pipeline = IntegratedPaperPipeline(
        capital=args.capital,
        leverage=args.leverage,
    )
    pipeline.run()


if __name__ == '__main__':
    main()
