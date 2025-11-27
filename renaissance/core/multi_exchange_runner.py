"""
Renaissance Trading System - Multi-Exchange WebSocket Runner
High-frequency trading with aggregated data from Kraken + Coinbase

This provides 10-50x more ticks than single-exchange mode!

Usage:
    python run_multi_ws.py                     # Run V5 with multi-exchange
    python run_multi_ws.py --duration 300      # Run for 5 minutes
    python run_multi_ws.py --version V3        # Use specific version
"""
import sys
import os
import time
import argparse
import signal
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import BaseStrategy, get_config, STARTING_CAPITAL
from core.data import MultiExchangeFeed, Tick

try:
    from colorama import init, Fore, Style
    init()
except ImportError:
    class Fore:
        GREEN = RED = YELLOW = CYAN = WHITE = RESET = MAGENTA = ''
    class Style:
        BRIGHT = RESET_ALL = ''


class MultiExchangeTrader:
    """
    High-frequency trading engine using multi-exchange WebSocket feeds

    10-50x more ticks than single exchange!
    """

    def __init__(self, version: str = 'V5', paper_mode: bool = True):
        self.version = version
        self.paper_mode = paper_mode

        # Initialize strategy
        self.config = get_config(version)
        self.config['use_probability_mode'] = True
        self.strategy = BaseStrategy(self.config)

        # Initialize multi-exchange feed
        self.feed = MultiExchangeFeed(symbols=['BTCUSD'], buffer_size=50000)

        # Trading state
        self.position = None
        self.capital = STARTING_CAPITAL
        self.peak_capital = STARTING_CAPITAL

        # Trade history
        self.trades = []
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

        # Stats
        self.tick_count = 0
        self.signal_count = 0
        self.start_time = None
        self.last_print_time = 0
        self.last_price = None

        # Running flag
        self.running = False

    def on_tick(self, symbol: str, tick: Tick):
        """Process each tick"""
        self.tick_count += 1
        price = tick.price
        self.last_price = price

        # Update strategy
        self.strategy.update(price)

        # Check exit first if in position
        if self.position:
            self._check_exit(price, tick.timestamp)

        # Get signal for new entry
        if not self.position:
            direction, size = self.strategy.get_signal(price)

            if direction != 0 and size > 0:
                self.signal_count += 1
                self._enter_trade(direction, size, price, tick.timestamp)

        # Print status periodically
        now = time.time()
        if now - self.last_print_time >= 1.0:
            self._print_status()
            self.last_print_time = now

    def _enter_trade(self, direction: int, size: float, price: float, timestamp: float):
        """Enter a new trade"""
        position_value = self.capital * size
        btc_size = position_value / price

        self.position = {
            'direction': direction,
            'entry_price': price,
            'size': btc_size,
            'value': position_value,
            'entry_time': timestamp,
            'barriers': self.strategy.exit_manager.current_barriers
        }

        dir_str = f"{Fore.GREEN}LONG{Style.RESET_ALL}" if direction > 0 else f"{Fore.RED}SHORT{Style.RESET_ALL}"
        print(f"\n{'='*60}")
        print(f"{Fore.CYAN}[ENTRY]{Style.RESET_ALL} {dir_str} @ ${price:,.2f}")
        print(f"  Size: {btc_size:.6f} BTC (${position_value:.2f})")
        print(f"  Reason: {self.strategy.last_reason}")
        if self.position['barriers']:
            b = self.position['barriers']
            print(f"  TP: ${b['tp_price']:,.2f} | SL: ${b['sl_price']:,.2f}")
        print(f"{'='*60}\n")

    def _check_exit(self, price: float, timestamp: float):
        """Check if position should be exited"""
        if not self.position:
            return

        entry_time = self.position['entry_time']
        should_exit, reason, pnl_pct = self.strategy.exit_manager.check_exit(
            price, timestamp, entry_time
        )

        if should_exit:
            self._exit_trade(price, reason, pnl_pct)

    def _exit_trade(self, price: float, reason: str, pnl_pct: float):
        """Exit current position"""
        if not self.position:
            return

        position_value = self.position['value']
        pnl_dollars = position_value * pnl_pct

        self.capital += pnl_dollars
        self.peak_capital = max(self.peak_capital, self.capital)
        self.total_pnl += pnl_dollars

        won = pnl_dollars > 0
        if won:
            self.wins += 1
            color = Fore.GREEN
        else:
            self.losses += 1
            color = Fore.RED

        self.trades.append({
            'entry_price': self.position['entry_price'],
            'exit_price': price,
            'direction': self.position['direction'],
            'pnl_pct': pnl_pct,
            'pnl_dollars': pnl_dollars,
            'reason': reason,
            'duration': time.time() - self.position['entry_time']
        })

        self.strategy.exit_manager.record_trade(won, pnl_dollars, self.capital)

        dir_str = "LONG" if self.position['direction'] > 0 else "SHORT"
        print(f"\n{'='*60}")
        print(f"{color}[EXIT]{Style.RESET_ALL} {dir_str} @ ${price:,.2f} ({reason})")
        print(f"  P&L: {color}${pnl_dollars:+.4f}{Style.RESET_ALL} ({pnl_pct*100:+.2f}%)")
        print(f"  Capital: ${self.capital:.4f}")
        print(f"{'='*60}\n")

        self.position = None

    def _print_status(self):
        """Print current status"""
        if not self.last_price:
            return

        elapsed = time.time() - self.start_time if self.start_time else 0
        tps = self.tick_count / elapsed if elapsed > 0 else 0

        unrealized = 0
        if self.position:
            if self.position['direction'] > 0:
                unrealized = (self.last_price - self.position['entry_price']) / self.position['entry_price']
            else:
                unrealized = (self.position['entry_price'] - self.last_price) / self.position['entry_price']
            unrealized *= self.position['value']

        total_trades = self.wins + self.losses
        win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0

        # Get exchange-specific stats
        stats = self.feed.get_stats()
        ex_info = " | ".join([
            f"{n[:2].upper()}:{s['ticks']}"
            for n, s in stats.get('exchanges', {}).items()
        ])

        pos_str = ""
        if self.position:
            dir_char = "L" if self.position['direction'] > 0 else "S"
            pos_color = Fore.GREEN if unrealized >= 0 else Fore.RED
            pos_str = f" | Pos: {dir_char} {pos_color}${unrealized:+.4f}{Style.RESET_ALL}"

        status = (
            f"\r{Fore.CYAN}[{datetime.now().strftime('%H:%M:%S')}]{Style.RESET_ALL} "
            f"BTC: ${self.last_price:,.2f} | "
            f"{Fore.MAGENTA}Ticks: {self.tick_count} ({tps:.1f}/s){Style.RESET_ALL} | "
            f"[{ex_info}] | "
            f"Trades: {total_trades} (WR: {win_rate:.0f}%) | "
            f"P&L: ${self.total_pnl:+.4f}"
            f"{pos_str}    "
        )

        print(status, end='', flush=True)

    def _print_summary(self):
        """Print final trading summary"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        total_trades = self.wins + self.losses
        win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0

        stats = self.feed.get_stats()

        print(f"\n\n{'='*70}")
        print(f"{Fore.CYAN}MULTI-EXCHANGE TRADING SESSION SUMMARY{Style.RESET_ALL}")
        print(f"{'='*70}")
        print(f"Duration: {elapsed/60:.1f} minutes")
        print(f"Total Ticks: {Fore.MAGENTA}{self.tick_count:,}{Style.RESET_ALL} ({self.tick_count/elapsed:.1f}/sec)")

        print(f"\n{Fore.YELLOW}EXCHANGE BREAKDOWN:{Style.RESET_ALL}")
        for name, ex_stats in stats.get('exchanges', {}).items():
            connected = f"{Fore.GREEN}CONNECTED{Style.RESET_ALL}" if ex_stats['connected'] else f"{Fore.RED}DISCONNECTED{Style.RESET_ALL}"
            print(f"  {name.upper()}: {ex_stats['ticks']:,} ticks ({ex_stats['rate']:.1f}/sec) - {connected}")

        print(f"\n{Fore.YELLOW}TRADES:{Style.RESET_ALL}")
        print(f"  Signals: {self.signal_count}")
        print(f"  Total Trades: {total_trades}")
        print(f"  Wins: {Fore.GREEN}{self.wins}{Style.RESET_ALL}")
        print(f"  Losses: {Fore.RED}{self.losses}{Style.RESET_ALL}")
        print(f"  Win Rate: {win_rate:.1f}%")

        print(f"\n{Fore.YELLOW}P&L:{Style.RESET_ALL}")
        print(f"  Total P&L: ${self.total_pnl:+.4f}")
        print(f"  Starting Capital: ${STARTING_CAPITAL:.2f}")
        print(f"  Ending Capital: ${self.capital:.4f}")
        print(f"  Return: {(self.capital/STARTING_CAPITAL - 1)*100:+.2f}%")

        if total_trades > 0:
            avg_pnl = self.total_pnl / total_trades
            print(f"  Avg P&L/Trade: ${avg_pnl:.6f}")

        if self.trades:
            print(f"\n{Fore.YELLOW}LAST 10 TRADES:{Style.RESET_ALL}")
            for i, t in enumerate(self.trades[-10:], 1):
                dir_str = "L" if t['direction'] > 0 else "S"
                color = Fore.GREEN if t['pnl_dollars'] > 0 else Fore.RED
                print(f"  {i}. {dir_str} ${t['entry_price']:,.2f} -> ${t['exit_price']:,.2f} "
                      f"{color}{t['pnl_pct']*100:+.2f}%{Style.RESET_ALL} ({t['reason']}) "
                      f"[{t['duration']:.1f}s]")

        print(f"{'='*70}\n")

    def start(self, duration: float = None):
        """Start trading"""
        self.running = True
        self.start_time = time.time()

        def signal_handler(sig, frame):
            print(f"\n{Fore.YELLOW}Shutting down...{Style.RESET_ALL}")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)

        print(f"\n{'='*70}")
        print(f"{Fore.CYAN}RENAISSANCE TRADING SYSTEM - Multi-Exchange Mode{Style.RESET_ALL}")
        print(f"{'='*70}")
        print(f"Version: {self.version}")
        print(f"Exchanges: Coinbase + Kraken (10-50x tick rate)")
        print(f"Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        print(f"Starting Capital: ${STARTING_CAPITAL:.2f}")
        print(f"Duration: {duration/60:.1f} minutes" if duration else "Duration: Unlimited")
        print(f"{'='*70}\n")

        # Register tick callback
        self.feed.on_tick(self.on_tick)

        # Start feed
        self.feed.start()

        print("Waiting for market data from exchanges...")

        try:
            while self.running:
                time.sleep(0.1)

                if duration and (time.time() - self.start_time) >= duration:
                    print(f"\n{Fore.YELLOW}Duration limit reached.{Style.RESET_ALL}")
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.feed.stop()
            self._print_summary()

    def stop(self):
        """Stop trading"""
        self.running = False


def main():
    parser = argparse.ArgumentParser(description='Renaissance Multi-Exchange Trader')
    parser.add_argument('--version', '-v', default='V5',
                        help='Strategy version (default: V5)')
    parser.add_argument('--duration', '-d', type=int, default=None,
                        help='Trading duration in seconds')
    parser.add_argument('--paper', '-p', action='store_true', default=True,
                        help='Paper trading mode')

    args = parser.parse_args()

    try:
        import websockets
    except ImportError:
        print(f"{Fore.RED}Error: websockets not installed{Style.RESET_ALL}")
        print("Install with: pip install websockets")
        sys.exit(1)

    trader = MultiExchangeTrader(
        version=args.version,
        paper_mode=args.paper
    )

    trader.start(duration=args.duration)


if __name__ == '__main__':
    main()
