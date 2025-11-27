"""
Renaissance Trading System - PARALLEL V1-V14 HFT EXPLOSIVE GROWTH
=================================================================
Runs all 14 HFT strategies simultaneously on the same market data feed
to find the OPTIMAL configuration for explosive growth $10 -> $300,000.

All 306 formulas distributed across 14 unique HFT strategies:
- V1:  MARKET_MAKER     - Avellaneda-Stoikov spread capture (ID 283)
- V2:  MICROSTRUCTURE   - Kyle Lambda + OFI flow detection (IDs 304, 101-130)
- V3:  VPIN_FILTER      - Toxicity avoidance (IDs 303, 286)
- V4:  DOLLAR_BARS      - Information-driven sampling (ID 285)
- V5:  REGIME_HMM       - HMM + CUSUM regime detection (IDs 176, 171-190)
- V6:  ROUGH_VOL        - Hurst + fractional volatility (IDs 168-170, 243)
- V7:  MEAN_REVERT      - OU process mean reversion (IDs 131-150, 166-167)
- V8:  WAVELET_FFT      - Cycle detection (IDs 191-210, 186-189)
- V9:  GRINOLD_IR       - IR maximizer (IDs 300, 298)
- V10: KELLY_THORP      - Optimal position sizing (ID 302)
- V11: ALMGREN_EXEC     - Optimal execution (ID 301)
- V12: FUNDING_ARB      - BIS crypto carry (ID 305)
- V13: TRIPLE_META      - Meta-labeling exits (IDs 151-152)
- V14: MASTER_QUANT     - ALL 306 formulas combined

ALL VOLUME SCALING DATA FROM LIVE APIs - NO HARDCODING!
Sources: CoinGecko, Kraken, Coinbase, Bitstamp, Gemini

Usage:
    python run_parallel.py                    # Run all V1-V14 for 5 minutes
    python run_parallel.py --duration 600    # Run for 10 minutes
    python run_parallel.py -v V1 V9 V14      # Run specific strategies
"""
import sys
import os
import time
import argparse
import signal
import threading
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import BaseStrategy, get_config, STARTING_CAPITAL, print_config_summary
from core.data import ResilientFeed, Tick, ProviderStatus
from core.data import MultiAPIFeed, create_enhanced_btc_feed
from core.utils.test_results_updater import auto_update_hook

try:
    from colorama import init, Fore, Style
    init()
except ImportError:
    class Fore:
        GREEN = RED = YELLOW = CYAN = WHITE = RESET = MAGENTA = BLUE = ''
    class Style:
        BRIGHT = RESET_ALL = ''


class ParallelTrader:
    """
    Run ALL V1-V14 HFT strategies in parallel on the same data feed.
    Find the BEST performer for explosive growth $10 -> $300,000.

    Uses all 306 formulas across 14 unique strategies:
    - V1: Market Maker (Avellaneda-Stoikov ID 283)
    - V2: Microstructure (Kyle Lambda + OFI IDs 304, 101-130)
    - V3: VPIN Toxicity Filter (IDs 303, 286)
    - V4: Dollar Bars (ID 285)
    - V5: Regime HMM + CUSUM (IDs 176, 171-190)
    - V6: Rough Volatility + Hurst (IDs 168-170, 243)
    - V7: Mean Reversion OU (IDs 131-150, 166-167)
    - V8: Wavelet FFT Cycles (IDs 191-210, 186-189)
    - V9: Grinold-Kahn IR Maximizer (IDs 300, 298)
    - V10: Kelly-Thorp Optimal (ID 302)
    - V11: Almgren-Chriss Execution (ID 301)
    - V12: BIS Funding Arbitrage (ID 305)
    - V13: Triple Barrier Meta-Label (IDs 151-152)
    - V14: Master Quant (ALL 306 formulas)
    """

    def __init__(self, versions=None, paper_mode=True, use_multi_api=False):
        self.versions = versions or ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14']
        self.paper_mode = paper_mode
        self.use_multi_api = use_multi_api

        # Initialize all strategies
        self.strategies = {}
        self.positions = {}
        self.capitals = {}
        self.trades = {}
        self.wins = {}
        self.losses = {}
        self.total_pnls = {}

        for v in self.versions:
            config = get_config(v)
            config['use_probability_mode'] = True
            self.strategies[v] = BaseStrategy(config)
            self.positions[v] = None
            self.capitals[v] = STARTING_CAPITAL
            self.trades[v] = []
            self.wins[v] = 0
            self.losses[v] = 0
            self.total_pnls[v] = 0.0

        # Initialize feed (shared by all strategies)
        # Multi-API: Exchange feeds + OKX (no auth needed for public data)
        # Standard: ResilientFeed (Coinbase, Kraken, Gemini, Bitstamp)
        if use_multi_api:
            self.feed = create_enhanced_btc_feed(buffer_size=100000)
            print(f"{Fore.GREEN}[MULTI-API] Using enhanced feed (4 exchanges + OKX){Style.RESET_ALL}")
        else:
            self.feed = ResilientFeed(symbols=['BTCUSD'], buffer_size=100000)

        # Stats
        self.tick_count = 0
        self.start_time = None
        self.last_print_time = 0
        self.last_price = None
        self.running = False

    def on_tick(self, symbol: str, tick: Tick):
        """Process each tick for ALL strategies"""
        self.tick_count += 1
        price = tick.price
        self.last_price = price
        ts = tick.timestamp

        # Update and check each strategy
        for v in self.versions:
            strategy = self.strategies[v]

            # Update strategy
            strategy.update(price)

            # Check exit first if in position
            if self.positions[v]:
                self._check_exit(v, price, ts)

            # Get signal for new entry
            if not self.positions[v]:
                direction, size = strategy.get_signal(price)

                if direction != 0 and size > 0:
                    self._enter_trade(v, direction, size, price, ts)

        # Print status periodically
        now = time.time()
        if now - self.last_print_time >= 2.0:  # Every 2 seconds
            self._print_status()
            self.last_print_time = now

    def _enter_trade(self, version: str, direction: int, size: float, price: float, ts: float):
        """Enter a trade for a specific strategy version"""
        strategy = self.strategies[version]
        capital = self.capitals[version]

        position_value = capital * min(size, 0.80)  # Max 80% of capital
        btc_size = position_value / price

        self.positions[version] = {
            'direction': direction,
            'entry_price': price,
            'size': btc_size,
            'value': position_value,
            'entry_time': ts,
            'barriers': strategy.exit_manager.current_barriers
        }

    def _check_exit(self, version: str, price: float, ts: float):
        """Check if position should be exited"""
        if not self.positions[version]:
            return

        strategy = self.strategies[version]
        position = self.positions[version]
        entry_time = position['entry_time']

        should_exit, reason, pnl_pct = strategy.exit_manager.check_exit(
            price, ts, entry_time
        )

        if should_exit:
            self._exit_trade(version, price, reason, pnl_pct)

    def _exit_trade(self, version: str, price: float, reason: str, pnl_pct: float):
        """Exit trade for a specific strategy version"""
        if not self.positions[version]:
            return

        position = self.positions[version]
        position_value = position['value']
        pnl_dollars = position_value * pnl_pct

        self.capitals[version] += pnl_dollars
        self.total_pnls[version] += pnl_dollars

        won = pnl_dollars > 0
        if won:
            self.wins[version] += 1
        else:
            self.losses[version] += 1

        self.trades[version].append({
            'entry_price': position['entry_price'],
            'exit_price': price,
            'direction': position['direction'],
            'pnl_pct': pnl_pct,
            'pnl_dollars': pnl_dollars,
            'reason': reason,
            'duration': time.time() - position['entry_time']
        })

        self.strategies[version].exit_manager.record_trade(won, pnl_dollars, self.capitals[version])
        self.positions[version] = None

    def _print_status(self):
        """Print live status for all strategies"""
        if not self.last_price:
            return

        elapsed = time.time() - self.start_time if self.start_time else 0
        tps = self.tick_count / elapsed if elapsed > 0 else 0

        # Clear screen and print header
        print(f"\n{Fore.CYAN}{'='*90}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[{datetime.now().strftime('%H:%M:%S')}] BTC: ${self.last_price:,.2f} | Ticks: {self.tick_count:,} ({tps:.0f}/s) | Elapsed: {elapsed:.0f}s{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*90}{Style.RESET_ALL}")
        print(f"{'VER':<4} {'NAME':<12} {'CAPITAL':>10} {'P&L':>10} {'TRADES':>7} {'WR':>6} {'POS':>5}")
        print(f"{'-'*4} {'-'*12} {'-'*10} {'-'*10} {'-'*7} {'-'*6} {'-'*5}")

        # Sort by capital (best performer first)
        sorted_versions = sorted(self.versions, key=lambda v: self.capitals[v], reverse=True)

        for v in sorted_versions:
            config = get_config(v)
            name = config.get('name', v)[:12]
            capital = self.capitals[v]
            pnl = self.total_pnls[v]
            total_trades = self.wins[v] + self.losses[v]
            win_rate = (self.wins[v] / total_trades * 100) if total_trades > 0 else 0

            # Color based on P&L
            if pnl > 0:
                pnl_color = Fore.GREEN
            elif pnl < 0:
                pnl_color = Fore.RED
            else:
                pnl_color = Fore.WHITE

            # Position indicator
            pos_str = ""
            if self.positions[v]:
                pos = self.positions[v]
                pos_dir = "L" if pos['direction'] > 0 else "S"
                unrealized = (self.last_price - pos['entry_price']) / pos['entry_price']
                if pos['direction'] < 0:
                    unrealized = -unrealized
                unrealized *= pos['value']
                pos_color = Fore.GREEN if unrealized >= 0 else Fore.RED
                pos_str = f"{pos_dir} {pos_color}${unrealized:+.2f}{Style.RESET_ALL}"

            print(f"{v:<4} {name:<12} ${capital:>9.2f} {pnl_color}${pnl:>+9.4f}{Style.RESET_ALL} {total_trades:>7} {win_rate:>5.0f}% {pos_str}")

        # Best performer highlight
        best_v = sorted_versions[0]
        best_capital = self.capitals[best_v]
        best_name = get_config(best_v).get('name', best_v)
        print(f"\n{Fore.YELLOW}LEADER: {best_v} ({best_name}) - ${best_capital:.4f}{Style.RESET_ALL}")

    def _print_final_summary(self):
        """Print final summary with rankings"""
        elapsed = time.time() - self.start_time if self.start_time else 0

        print(f"\n\n{'='*90}")
        print(f"{Fore.CYAN}HFT EXPLOSIVE V1-V14 - FINAL RESULTS ($10 -> $300k){Style.RESET_ALL}")
        print(f"{'='*90}")
        print(f"Duration: {elapsed/60:.1f} minutes | Total Ticks: {self.tick_count:,}")

        # Sort by capital (best first)
        sorted_versions = sorted(self.versions, key=lambda v: self.capitals[v], reverse=True)

        print(f"\n{Fore.YELLOW}RANKINGS (by Final Capital):{Style.RESET_ALL}")
        print(f"{'RANK':<5} {'VER':<4} {'NAME':<12} {'CAPITAL':>12} {'RETURN':>10} {'TRADES':>8} {'WR':>7} {'PF':>7}")
        print(f"{'-'*5} {'-'*4} {'-'*12} {'-'*12} {'-'*10} {'-'*8} {'-'*7} {'-'*7}")

        for rank, v in enumerate(sorted_versions, 1):
            config = get_config(v)
            name = config.get('name', v)[:12]
            capital = self.capitals[v]
            pnl = self.total_pnls[v]
            ret = (capital / STARTING_CAPITAL - 1) * 100
            total_trades = self.wins[v] + self.losses[v]
            win_rate = (self.wins[v] / total_trades * 100) if total_trades > 0 else 0

            # Calculate profit factor
            wins_total = sum(t['pnl_dollars'] for t in self.trades[v] if t['pnl_dollars'] > 0)
            losses_total = abs(sum(t['pnl_dollars'] for t in self.trades[v] if t['pnl_dollars'] < 0))
            pf = wins_total / losses_total if losses_total > 0 else float('inf')

            # Color based on return
            if ret > 0:
                ret_color = Fore.GREEN
            elif ret < 0:
                ret_color = Fore.RED
            else:
                ret_color = Fore.WHITE

            medal = ""
            if rank == 1:
                medal = f"{Fore.YELLOW}[1st]{Style.RESET_ALL} "
            elif rank == 2:
                medal = f"{Fore.WHITE}[2nd]{Style.RESET_ALL} "
            elif rank == 3:
                medal = f"{Fore.RED}[3rd]{Style.RESET_ALL} "

            print(f"{medal}{rank:<5} {v:<4} {name:<12} ${capital:>11.4f} {ret_color}{ret:>+9.2f}%{Style.RESET_ALL} {total_trades:>8} {win_rate:>6.1f}% {pf:>6.2f}x")

        # Winner details
        winner = sorted_versions[0]
        winner_config = get_config(winner)
        print(f"\n{Fore.GREEN}{'='*90}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}WINNER: {winner} - {winner_config.get('name', winner)}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}  {winner_config.get('description', '')}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}  Final Capital: ${self.capitals[winner]:.4f} (Return: {(self.capitals[winner]/STARTING_CAPITAL-1)*100:+.2f}%){Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*90}{Style.RESET_ALL}")

        # Trade breakdown for winner
        if self.trades[winner]:
            print(f"\n{Fore.YELLOW}WINNER LAST 10 TRADES:{Style.RESET_ALL}")
            for i, t in enumerate(self.trades[winner][-10:], 1):
                dir_str = "LONG " if t['direction'] > 0 else "SHORT"
                color = Fore.GREEN if t['pnl_dollars'] > 0 else Fore.RED
                print(f"  {i}. {dir_str} ${t['entry_price']:,.2f} -> ${t['exit_price']:,.2f} "
                      f"{color}{t['pnl_pct']*100:+.2f}%{Style.RESET_ALL} ({t['reason']}) [{t['duration']:.1f}s]")

    def start(self, duration: float = 300):
        """Start parallel trading test"""
        self.running = True
        self.start_time = time.time()

        def signal_handler(sig, frame):
            print(f"\n{Fore.YELLOW}Shutting down...{Style.RESET_ALL}")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)

        # Print config summary
        print_config_summary()

        print(f"\n{'='*90}")
        print(f"{Fore.CYAN}RENAISSANCE TRADING SYSTEM - PARALLEL V1-V14 HFT EXPLOSIVE GROWTH{Style.RESET_ALL}")
        print(f"{'='*90}")
        print(f"Versions: {', '.join(self.versions)}")
        print(f"Starting Capital: ${STARTING_CAPITAL:.2f} each")
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Target: Find BEST strategy for $10 -> $300,000")
        print(f"Formula Count: 306 (including HFT + Academic + Volume Scaling)")
        print(f"Data Source: LIVE APIs (CoinGecko, Kraken, Coinbase, Bitstamp, Gemini)")
        print(f"{'='*90}\n")

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
            self._print_final_summary()
            # AUTO-UPDATE: Update all comments.md files with test results
            auto_update_hook(self, duration)


def main():
    parser = argparse.ArgumentParser(description='Renaissance Parallel V1-V14 HFT Explosive Growth')
    parser.add_argument('--duration', '-d', type=int, default=300,
                        help='Test duration in seconds (default: 300 = 5 min)')
    parser.add_argument('--versions', '-v', nargs='+', default=None,
                        help='Specific versions to test (default: all V1-V14). Example: -v V1 V9 V14')

    args = parser.parse_args()

    try:
        import websockets
    except ImportError:
        print(f"{Fore.RED}Error: websockets not installed{Style.RESET_ALL}")
        print("Install with: pip install websockets")
        sys.exit(1)

    trader = ParallelTrader(versions=args.versions)
    trader.start(duration=args.duration)


if __name__ == '__main__':
    main()
