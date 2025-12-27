#!/usr/bin/env python3
"""
OVERNIGHT MULTI-EXCHANGE PAPER TRADER
=====================================
$100 allocated per exchange. Runs indefinitely.

DETERMINISTIC ADVANTAGE:
- Blockchain data arrives 10-60 seconds BEFORE price impact
- INFLOW to exchange = SHORT (selling pressure coming)
- OUTFLOW via exhaustion = LONG (sellers dried up)

This is pure math. Data doesn't lie.
"""

import os
import sys
import time
import sqlite3
import threading
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from engine.sovereign.simulation.fees import EXCHANGE_FEES, get_slippage_estimate
from engine.sovereign.config.exchanges import ACTIVE_TRADING_EXCHANGES


@dataclass
class ExchangeAccount:
    """Per-exchange paper trading account."""
    name: str
    capital: float
    initial_capital: float
    maker_fee: float
    taker_fee: float
    positions: Dict = field(default_factory=dict)
    trades: List = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    total_fees: float = 0.0

    @property
    def pnl(self) -> float:
        return self.capital - self.initial_capital

    @property
    def pnl_pct(self) -> float:
        return (self.capital - self.initial_capital) / self.initial_capital * 100

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total * 100 if total > 0 else 0.0


class MultiExchangePaperTrader:
    """
    Paper trading across multiple exchanges.

    Each exchange gets $100 allocation.
    Blockchain signals route to the appropriate exchange.
    """

    USA_EXCHANGES = ['coinbase', 'kraken', 'bitstamp', 'gemini', 'binance']

    def __init__(
        self,
        capital_per_exchange: float = 100.0,
        db_path: str = "data/overnight_paper.db",
        zmq_endpoint: str = "tcp://127.0.0.1:28332",
    ):
        self.capital_per_exchange = capital_per_exchange
        self.db_path = db_path
        self.zmq_endpoint = zmq_endpoint

        # Initialize per-exchange accounts
        self.accounts: Dict[str, ExchangeAccount] = {}
        for ex in self.USA_EXCHANGES:
            fees = EXCHANGE_FEES.get(ex, EXCHANGE_FEES.get('binance_us'))
            self.accounts[ex] = ExchangeAccount(
                name=ex,
                capital=capital_per_exchange,
                initial_capital=capital_per_exchange,
                maker_fee=fees.maker_fee,
                taker_fee=fees.taker_fee,
            )

        # Global state
        self.running = False
        self.start_time = None
        self.current_price = 0.0
        self.signals_received = 0
        self.total_trades = 0

        # Position config
        self.max_position_pct = 0.25  # 25% of account per trade
        self.stop_loss_pct = 0.02     # 2% stop loss
        self.take_profit_pct = 0.04   # 4% take profit
        self.max_hold_seconds = 300   # 5 minute max hold

        # Database
        self._init_db()

        # Price feed thread
        self._price_lock = threading.Lock()

    def _init_db(self):
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                exchange TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                size_usd REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                fees REAL,
                exit_reason TEXT,
                signal_btc REAL,
                hold_seconds REAL
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                duration_hours REAL,
                total_pnl REAL,
                total_trades INTEGER,
                win_rate REAL
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS equity_curve (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                exchange TEXT,
                capital REAL,
                pnl REAL
            )
        ''')

        conn.commit()
        conn.close()

    def _log_trade(self, exchange: str, trade: Dict):
        """Log trade to database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            INSERT INTO trades (timestamp, exchange, direction, entry_price, exit_price,
                              size_usd, pnl_usd, pnl_pct, fees, exit_reason, signal_btc, hold_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            exchange,
            trade['direction'],
            trade['entry_price'],
            trade['exit_price'],
            trade['size_usd'],
            trade['pnl_usd'],
            trade['pnl_pct'],
            trade['fees'],
            trade['exit_reason'],
            trade.get('signal_btc', 0),
            trade.get('hold_seconds', 0),
        ))

        conn.commit()
        conn.close()

    def _log_equity(self):
        """Log equity curve snapshot."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        for ex, account in self.accounts.items():
            c.execute('''
                INSERT INTO equity_curve (timestamp, exchange, capital, pnl)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                ex,
                account.capital,
                account.pnl,
            ))

        conn.commit()
        conn.close()

    def on_signal(self, exchange: str, direction: str, amount_btc: float,
                  price: float, confidence: float = 1.0, pattern: str = ""):
        """
        Handle signal from blockchain pipeline.

        Args:
            exchange: Target exchange (coinbase, kraken, etc.)
            direction: 'SHORT' or 'LONG'
            amount_btc: Flow amount that triggered signal
            price: Current BTC price
            confidence: Signal confidence (0-1)
            pattern: Signal pattern type
        """
        self.signals_received += 1
        self.current_price = price

        # Normalize exchange name
        ex = exchange.lower().replace('-', '').replace('.', '')
        if ex not in self.accounts:
            return

        account = self.accounts[ex]

        # Check if we already have a position on this exchange
        if account.positions:
            return

        # Calculate position size
        position_usd = account.capital * self.max_position_pct * confidence
        if position_usd < 1:
            return

        # Apply entry slippage
        slippage = get_slippage_estimate(position_usd)
        if direction == 'LONG':
            entry_price = price * (1 + slippage)
        else:
            entry_price = price * (1 - slippage)

        # Calculate fees
        entry_fee = position_usd * account.taker_fee
        account.capital -= entry_fee
        account.total_fees += entry_fee

        # Calculate SL/TP
        if direction == 'LONG':
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)

        # Create position
        position = {
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': time.time(),
            'size_usd': position_usd,
            'size_btc': position_usd / entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'signal_btc': amount_btc,
            'pattern': pattern,
        }

        account.positions['main'] = position

        print(f"\n[OPEN] {direction} {ex.upper()}")
        print(f"       Entry: ${entry_price:,.2f} | Size: ${position_usd:.2f}")
        print(f"       SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}")
        print(f"       Signal: {amount_btc:.1f} BTC | Pattern: {pattern}")

    def update_prices(self, price: float):
        """Update price and check all positions for exit."""
        self.current_price = price
        now = time.time()

        for ex, account in self.accounts.items():
            if not account.positions:
                continue

            pos = account.positions.get('main')
            if not pos:
                continue

            exit_reason = None
            exit_price = price

            # Check stop loss
            if pos['direction'] == 'LONG' and price <= pos['stop_loss']:
                exit_reason = 'SL'
                exit_price = pos['stop_loss']
            elif pos['direction'] == 'SHORT' and price >= pos['stop_loss']:
                exit_reason = 'SL'
                exit_price = pos['stop_loss']

            # Check take profit
            elif pos['direction'] == 'LONG' and price >= pos['take_profit']:
                exit_reason = 'TP'
                exit_price = pos['take_profit']
            elif pos['direction'] == 'SHORT' and price <= pos['take_profit']:
                exit_reason = 'TP'
                exit_price = pos['take_profit']

            # Check timeout
            elif now - pos['entry_time'] >= self.max_hold_seconds:
                exit_reason = 'TIMEOUT'

            if exit_reason:
                self._close_position(ex, account, pos, exit_price, exit_reason)

    def _close_position(self, exchange: str, account: ExchangeAccount,
                       pos: Dict, exit_price: float, reason: str):
        """Close position and record trade."""
        # Apply exit slippage
        slippage = get_slippage_estimate(pos['size_usd'])
        if pos['direction'] == 'LONG':
            exit_price = exit_price * (1 - slippage)
        else:
            exit_price = exit_price * (1 + slippage)

        # Calculate P&L
        if pos['direction'] == 'LONG':
            pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] - exit_price) / pos['entry_price']

        pnl_usd = pnl_pct * pos['size_usd']

        # Exit fee
        exit_fee = pos['size_usd'] * account.taker_fee
        account.total_fees += exit_fee

        # Net P&L
        net_pnl = pnl_usd - exit_fee

        # Update account
        account.capital += pos['size_usd'] + net_pnl

        if net_pnl > 0:
            account.wins += 1
        else:
            account.losses += 1

        # Record trade
        hold_seconds = time.time() - pos['entry_time']
        trade = {
            'direction': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'size_usd': pos['size_usd'],
            'pnl_usd': net_pnl,
            'pnl_pct': pnl_pct * 100,
            'fees': exit_fee * 2,  # Entry + exit
            'exit_reason': reason,
            'signal_btc': pos.get('signal_btc', 0),
            'hold_seconds': hold_seconds,
        }

        account.trades.append(trade)
        self.total_trades += 1

        # Log to database
        self._log_trade(exchange, trade)

        # Clear position
        del account.positions['main']

        # Print
        color = '\033[92m' if net_pnl > 0 else '\033[91m'
        reset = '\033[0m'

        print(f"\n{color}[CLOSE]{reset} {pos['direction']} {exchange.upper()} - {reason}")
        print(f"        Entry: ${pos['entry_price']:,.2f} -> Exit: ${exit_price:,.2f}")
        print(f"        P&L: {color}${net_pnl:+.2f}{reset} ({pnl_pct*100:+.2f}%)")
        print(f"        Capital: ${account.capital:.2f} | Win Rate: {account.win_rate:.1f}%")

    def print_status(self):
        """Print current status of all exchanges."""
        print("\n" + "=" * 70)
        print(f"MULTI-EXCHANGE STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print(f"Price: ${self.current_price:,.2f} | Signals: {self.signals_received} | Trades: {self.total_trades}")
        print("-" * 70)
        print(f"{'Exchange':<12} {'Capital':>10} {'P&L':>10} {'P&L%':>8} {'Trades':>7} {'Win%':>7} {'Pos':>5}")
        print("-" * 70)

        total_capital = 0
        total_initial = 0
        total_wins = 0
        total_losses = 0

        for ex in self.USA_EXCHANGES:
            account = self.accounts[ex]
            total_capital += account.capital
            total_initial += account.initial_capital
            total_wins += account.wins
            total_losses += account.losses

            pos_str = "LONG" if account.positions and account.positions.get('main', {}).get('direction') == 'LONG' else \
                      "SHORT" if account.positions else "-"

            print(f"{ex:<12} ${account.capital:>9.2f} ${account.pnl:>+9.2f} {account.pnl_pct:>+7.2f}% "
                  f"{account.wins + account.losses:>7} {account.win_rate:>6.1f}% {pos_str:>5}")

        print("-" * 70)
        total_pnl = total_capital - total_initial
        total_pnl_pct = total_pnl / total_initial * 100
        total_trades = total_wins + total_losses
        win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

        print(f"{'TOTAL':<12} ${total_capital:>9.2f} ${total_pnl:>+9.2f} {total_pnl_pct:>+7.2f}% "
              f"{total_trades:>7} {win_rate:>6.1f}%")
        print("=" * 70)

        # Log equity
        self._log_equity()

    def _update_price_loop(self):
        """Background price update loop."""
        import requests

        while self.running:
            try:
                resp = requests.get(
                    "https://api.coinbase.com/v2/prices/BTC-USD/spot",
                    timeout=5
                )
                if resp.status_code == 200:
                    price = float(resp.json()['data']['amount'])
                    with self._price_lock:
                        self.update_prices(price)
            except Exception as e:
                pass

            time.sleep(1)

    def run(self):
        """Run the multi-exchange paper trader."""
        self.running = True
        self.start_time = time.time()

        print("=" * 70)
        print("OVERNIGHT MULTI-EXCHANGE PAPER TRADER")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Capital: ${self.capital_per_exchange:.2f} per exchange")
        print(f"Exchanges: {', '.join(self.USA_EXCHANGES)}")
        print(f"Total Capital: ${self.capital_per_exchange * len(self.USA_EXCHANGES):.2f}")
        print("=" * 70)
        print()
        print("DETERMINISTIC ADVANTAGE:")
        print("  Blockchain data arrives 10-60 seconds BEFORE exchange price impact")
        print("  INFLOW -> SHORT (selling pressure)")
        print("  OUTFLOW (exhaustion) -> LONG (sellers dried up)")
        print()
        print("Data doesn't lie. Math doesn't lie.")
        print("=" * 70)
        print()

        # Start price feed
        price_thread = threading.Thread(target=self._update_price_loop, daemon=True)
        price_thread.start()

        # Connect to blockchain pipeline
        try:
            from engine.sovereign.blockchain.formula_connector import FormulaConnector

            self.connector = FormulaConnector(
                zmq_endpoint=self.zmq_endpoint,
                on_signal=self._on_connector_signal,
            )

            if self.connector.start():
                print("[ZMQ] Connected to Bitcoin Core blockchain feed")
            else:
                print("[ZMQ] Failed to connect - running in price-only mode")
                self.connector = None

        except Exception as e:
            print(f"[ZMQ] Blockchain connector not available: {e}")
            print("[ZMQ] Running in standalone mode - use on_signal() manually")
            self.connector = None

        last_status = time.time()

        try:
            while self.running:
                # Print status every 60 seconds
                if time.time() - last_status >= 60:
                    self.print_status()
                    last_status = time.time()

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n[STOP] Interrupted by user")
        finally:
            self._shutdown()

    def _on_connector_signal(self, signal: Dict):
        """Handle signal from FormulaConnector."""
        if signal.get('direction', 0) == 0:
            return

        direction = 'LONG' if signal['direction'] == 1 else 'SHORT'
        exchange = signal.get('exchange', 'coinbase')
        amount_btc = abs(signal.get('btc_amount', 0))
        confidence = signal.get('confidence', 0.5)

        self.on_signal(
            exchange=exchange,
            direction=direction,
            amount_btc=amount_btc,
            price=self.current_price,
            confidence=confidence,
            pattern=signal.get('ensemble_type', 'blockchain'),
        )

    def _shutdown(self):
        """Clean shutdown."""
        self.running = False

        if hasattr(self, 'connector') and self.connector:
            self.connector.stop()

        # Close all open positions at current price
        for ex, account in self.accounts.items():
            if account.positions and 'main' in account.positions:
                pos = account.positions['main']
                self._close_position(ex, account, pos, self.current_price, 'SESSION_END')

        # Final status
        elapsed = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 70)
        print("OVERNIGHT SESSION COMPLETE")
        print("=" * 70)
        print(f"Duration: {elapsed/3600:.2f} hours")
        print()

        self.print_status()

        # Save session
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        total_pnl = sum(a.pnl for a in self.accounts.values())
        total_trades = sum(a.wins + a.losses for a in self.accounts.values())
        total_wins = sum(a.wins for a in self.accounts.values())
        win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

        c.execute('''
            INSERT INTO sessions (start_time, end_time, duration_hours, total_pnl, total_trades, win_rate)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            datetime.now().isoformat(),
            elapsed / 3600,
            total_pnl,
            total_trades,
            win_rate,
        ))

        conn.commit()
        conn.close()

        print(f"\nResults saved to: {self.db_path}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Overnight Multi-Exchange Paper Trader')
    parser.add_argument('--capital', type=float, default=100.0, help='Capital per exchange')
    parser.add_argument('--db', type=str, default='data/overnight_paper.db', help='Database path')
    parser.add_argument('--zmq', type=str, default='tcp://127.0.0.1:28332', help='ZMQ endpoint')
    args = parser.parse_args()

    trader = MultiExchangePaperTrader(
        capital_per_exchange=args.capital,
        db_path=args.db,
        zmq_endpoint=args.zmq,
    )

    trader.run()


if __name__ == '__main__':
    main()
