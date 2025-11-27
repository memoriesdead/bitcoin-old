"""
Renaissance Trading System - Enterprise Resilient Trading Runner
Commercial-grade system with 4 USA-friendly exchanges and automatic failover

Usage:
    python run_resilient.py                     # Run with all 4 providers
    python run_resilient.py --duration 300      # Run for 5 minutes
    python run_resilient.py --version V5        # Use specific strategy version

Providers (all USA-friendly):
1. Coinbase - Highest US volume, SEC-compliant
2. Kraken - Institutional grade, 99.9% uptime
3. Gemini - NY Trust Company, SOC 2 certified
4. Bitstamp - 13+ years operation, NY BitLicense
"""
import sys
import os
import time
import argparse
import signal
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import BaseStrategy, get_config, STARTING_CAPITAL
from core.data import ResilientFeed, Tick, ProviderStatus
from core.utils.test_results_updater import resilient_auto_update_hook

try:
    from colorama import init, Fore, Style
    init()
except ImportError:
    class Fore:
        GREEN = RED = YELLOW = CYAN = WHITE = RESET = MAGENTA = BLUE = ''
    class Style:
        BRIGHT = RESET_ALL = ''


class ResilientTrader:
    """
    Enterprise-grade trading engine with resilient multi-provider feed

    Features:
    - 4 USA-friendly exchanges
    - Automatic failover
    - Health monitoring
    - Arbitrage detection
    """

    def __init__(self, version: str = 'V5', paper_mode: bool = True):
        self.version = version
        self.paper_mode = paper_mode

        # Initialize strategy
        self.config = get_config(version)
        self.config['use_probability_mode'] = True
        self.strategy = BaseStrategy(self.config)

        # Initialize resilient feed
        self.feed = ResilientFeed(symbols=['BTCUSD'], buffer_size=100000)
        self.feed.on_provider_status_change = self._on_provider_status_change

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
        self.last_status_time = 0

        # Running flag
        self.running = False

    def _on_provider_status_change(self, provider: str, status: ProviderStatus):
        """Handle provider status changes"""
        color = {
            ProviderStatus.HEALTHY: Fore.GREEN,
            ProviderStatus.DEGRADED: Fore.YELLOW,
            ProviderStatus.FAILED: Fore.RED,
            ProviderStatus.RECOVERING: Fore.BLUE,
        }.get(status, Fore.WHITE)

        print(f"\n{color}[PROVIDER]{Style.RESET_ALL} {provider.upper()}: {status.value.upper()}")

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

        # Print detailed provider status every 30 seconds
        if now - self.last_status_time >= 30.0:
            self._print_provider_status()
            self.last_status_time = now

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

        # Get provider status summary
        stats = self.feed.get_stats()
        healthy = stats['healthy_providers']
        total = stats['total_providers']
        provider_color = Fore.GREEN if healthy >= 3 else (Fore.YELLOW if healthy >= 2 else Fore.RED)

        # Arbitrage info
        arb = stats.get('arbitrage')
        arb_str = ""
        if arb and arb['spread_pct'] > 0.05:  # > 0.05% spread
            arb_str = f" | {Fore.YELLOW}ARB: {arb['spread_pct']:.2f}%{Style.RESET_ALL}"

        pos_str = ""
        if self.position:
            dir_char = "L" if self.position['direction'] > 0 else "S"
            pos_color = Fore.GREEN if unrealized >= 0 else Fore.RED
            pos_str = f" | Pos: {dir_char} {pos_color}${unrealized:+.4f}{Style.RESET_ALL}"

        status = (
            f"\r{Fore.CYAN}[{datetime.now().strftime('%H:%M:%S')}]{Style.RESET_ALL} "
            f"BTC: ${self.last_price:,.2f} | "
            f"{Fore.MAGENTA}Ticks: {self.tick_count} ({tps:.1f}/s){Style.RESET_ALL} | "
            f"{provider_color}Providers: {healthy}/{total}{Style.RESET_ALL} | "
            f"Trades: {total_trades} (WR: {win_rate:.0f}%) | "
            f"P&L: ${self.total_pnl:+.4f}"
            f"{arb_str}{pos_str}    "
        )

        print(status, end='', flush=True)

    def _print_provider_status(self):
        """Print detailed provider status"""
        stats = self.feed.get_stats()

        print(f"\n{Fore.BLUE}--- Provider Status ---{Style.RESET_ALL}")
        for name, prov_stats in stats['providers'].items():
            status = prov_stats['status']
            color = {
                'healthy': Fore.GREEN,
                'degraded': Fore.YELLOW,
                'failed': Fore.RED,
                'recovering': Fore.BLUE,
            }.get(status, Fore.WHITE)

            connected = f"{Fore.GREEN}CONN{Style.RESET_ALL}" if prov_stats['connected'] else f"{Fore.RED}DISC{Style.RESET_ALL}"
            print(f"  {name.upper():10} {color}{status:10}{Style.RESET_ALL} [{connected}] Ticks: {prov_stats['tick_count']:,}")

        if stats.get('arbitrage'):
            arb = stats['arbitrage']
            print(f"  {Fore.YELLOW}Arbitrage: {arb['spread_pct']:.3f}% ({arb['high_exchange']} > {arb['low_exchange']}){Style.RESET_ALL}")
        print()

    def _print_summary(self):
        """Print final trading summary"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        total_trades = self.wins + self.losses
        win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0

        stats = self.feed.get_stats()

        print(f"\n\n{'='*70}")
        print(f"{Fore.CYAN}ENTERPRISE RESILIENT TRADING SESSION SUMMARY{Style.RESET_ALL}")
        print(f"{'='*70}")
        print(f"Duration: {elapsed/60:.1f} minutes")
        print(f"Total Ticks: {Fore.MAGENTA}{self.tick_count:,}{Style.RESET_ALL} ({self.tick_count/elapsed:.1f}/sec)")
        print(f"Failovers: {stats['failover_count']}")

        print(f"\n{Fore.YELLOW}PROVIDER BREAKDOWN:{Style.RESET_ALL}")
        for name, prov_stats in stats['providers'].items():
            status = prov_stats['status']
            color = Fore.GREEN if status == 'healthy' else (Fore.YELLOW if status == 'degraded' else Fore.RED)
            print(f"  {name.upper():10} {color}{status:10}{Style.RESET_ALL} Ticks: {prov_stats['tick_count']:,} Errors: {prov_stats['error_count']}")

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
        print(f"{Fore.CYAN}RENAISSANCE TRADING SYSTEM - Enterprise Resilient Mode{Style.RESET_ALL}")
        print(f"{'='*70}")
        print(f"Version: {self.version}")
        print(f"Providers: Coinbase + Kraken + Gemini + Bitstamp (4 USA exchanges)")
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
            # AUTO-UPDATE: Update comments.md file with test results
            resilient_auto_update_hook(self, duration if duration else 0)

    def stop(self):
        """Stop trading"""
        self.running = False


def main():
    parser = argparse.ArgumentParser(description='Renaissance Enterprise Resilient Trader')
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

    trader = ResilientTrader(
        version=args.version,
        paper_mode=args.paper
    )

    trader.start(duration=args.duration)


if __name__ == '__main__':
    main()
