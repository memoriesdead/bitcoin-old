#!/usr/bin/env python3
"""
RENAISSANCE V1-V25 PARALLEL TRADER WITH MEGA USA FEED
======================================================
Runs all 25 strategy versions on the MEGA 9-exchange aggregated feed
for maximum tick coverage and trading opportunities.

Exchanges: Coinbase, Kraken, Gemini, Bitstamp, OKCoin, Blockchain, CEX.IO, Bitfinex, Gate.io
Target: 5-20+ ticks/second combined

Author: Renaissance Trading System
"""
import sys
import os
import time
import signal
import threading
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import BaseStrategy, get_config, STARTING_CAPITAL
from mega_usa_feed import MegaUSAFeed, Tick

# Colors
class C:
    G = '\033[92m'
    R = '\033[91m'
    Y = '\033[93m'
    B = '\033[94m'
    M = '\033[95m'
    W = '\033[0m'
    BOLD = '\033[1m'


class MegaParallelTrader:
    """
    Run ALL V1-V25 strategies in parallel on the MEGA 9-exchange feed.
    """

    def __init__(self, versions=None, duration=600):
        self.versions = versions or [f'V{i}' for i in range(1, 26)]
        self.duration = duration

        # Initialize all strategies
        self.strategies = {}
        self.positions = {}
        self.capitals = {}
        self.trades = {}
        self.wins = {}
        self.losses = {}
        self.total_pnls = {}
        self.signals_generated = {}

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
            self.signals_generated[v] = 0

        # Initialize MEGA feed
        self.feed = MegaUSAFeed(buffer_size=500000)

        # Stats
        self.tick_count = 0
        self.start_time = None
        self.last_print_time = 0
        self.last_price = None
        self.running = False
        self.price_history = deque(maxlen=10000)

    def on_tick(self, tick: Tick):
        """Process each tick for ALL strategies"""
        self.tick_count += 1
        price = tick.price
        self.last_price = price
        self.price_history.append(price)
        ts = time.time()

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
                    self.signals_generated[v] += 1
                    self._enter_trade(v, direction, size, price, ts)

        # Print status periodically
        now = time.time()
        if now - self.last_print_time >= 3.0:
            self._print_status()
            self.last_print_time = now

    def _enter_trade(self, version: str, direction: int, size: float, price: float, ts: float):
        """Enter a trade for a specific strategy version"""
        config = get_config(version)
        capital = self.capitals[version]

        kelly_frac = config.get('kelly_frac', 0.5)
        position_value = capital * min(kelly_frac, 0.80)
        btc_size = position_value / price

        self.positions[version] = {
            'direction': direction,
            'entry_price': price,
            'size': btc_size,
            'value': position_value,
            'entry_time': ts
        }

    def _check_exit(self, version: str, price: float, ts: float):
        """Check if position should be exited"""
        if not self.positions[version]:
            return

        config = get_config(version)
        position = self.positions[version]
        entry_price = position['entry_price']
        direction = position['direction']
        entry_time = position['entry_time']

        # Calculate P&L
        if direction > 0:  # Long
            pnl_pct = (price - entry_price) / entry_price
        else:  # Short
            pnl_pct = (entry_price - price) / entry_price

        profit_target = config.get('profit_target', 0.01)
        stop_loss = config.get('stop_loss', 0.005)
        max_hold = config.get('max_hold_sec', 5)

        should_exit = False
        reason = ""

        if pnl_pct >= profit_target:
            should_exit = True
            reason = "PROFIT"
        elif pnl_pct <= -stop_loss:
            should_exit = True
            reason = "STOP"
        elif (ts - entry_time) >= max_hold:
            should_exit = True
            reason = "TIME"

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

        self.positions[version] = None

    def _print_status(self):
        """Print live status for all strategies"""
        if not self.last_price:
            return

        elapsed = time.time() - self.start_time if self.start_time else 0
        tps = self.tick_count / elapsed if elapsed > 0 else 0
        feed_stats = self.feed.get_stats()

        # Price movement
        if len(self.price_history) > 1:
            price_range = max(self.price_history) - min(self.price_history)
            price_pct = price_range / min(self.price_history) * 100
        else:
            price_range = 0
            price_pct = 0

        print(f"\n{C.B}{'='*100}{C.W}")
        print(f"{C.B}[{datetime.now().strftime('%H:%M:%S')}] BTC: ${self.last_price:,.2f} | "
              f"Ticks: {self.tick_count:,} ({tps:.1f}/s) | "
              f"Range: ${price_range:.2f} ({price_pct:.4f}%) | "
              f"Feeds: {feed_stats['active_feeds']}/9{C.W}")

        # Exchange breakdown
        ex_str = " | ".join([f"{k}:{v}" for k, v in sorted(feed_stats['by_exchange'].items(), key=lambda x: -x[1])[:5]])
        print(f"{C.B}  {ex_str}{C.W}")

        print(f"{C.B}{'='*100}{C.W}")
        print(f"{'VER':<4} {'NAME':<14} {'CAPITAL':>12} {'P&L':>10} {'TRADES':>7} {'WR':>6} {'SIGNALS':>8} {'POS'}")
        print(f"{'-'*4} {'-'*14} {'-'*12} {'-'*10} {'-'*7} {'-'*6} {'-'*8} {'-'*15}")

        # Sort by capital (best performer first)
        sorted_versions = sorted(self.versions, key=lambda v: self.capitals[v], reverse=True)

        for i, v in enumerate(sorted_versions[:10]):  # Top 10 only
            config = get_config(v)
            name = config.get('name', v)[:14]
            capital = self.capitals[v]
            pnl = self.total_pnls[v]
            total_trades = self.wins[v] + self.losses[v]
            win_rate = (self.wins[v] / total_trades * 100) if total_trades > 0 else 0
            signals = self.signals_generated[v]

            # Color based on P&L
            if pnl > 0:
                pnl_color = C.G
            elif pnl < 0:
                pnl_color = C.R
            else:
                pnl_color = C.W

            # Position indicator
            pos_str = ""
            if self.positions[v]:
                pos = self.positions[v]
                pos_dir = "LONG" if pos['direction'] > 0 else "SHORT"
                unrealized = (self.last_price - pos['entry_price']) / pos['entry_price']
                if pos['direction'] < 0:
                    unrealized = -unrealized
                pos_color = C.G if unrealized >= 0 else C.R
                pos_str = f"{pos_dir} {pos_color}{unrealized*100:+.3f}%{C.W}"

            rank_str = f"{C.Y}[{i+1}]{C.W}" if i < 3 else f"[{i+1}]"
            print(f"{rank_str} {v:<4} {name:<14} ${capital:>11.4f} {pnl_color}${pnl:>+9.4f}{C.W} {total_trades:>7} {win_rate:>5.0f}% {signals:>8} {pos_str}")

        # Best performer highlight
        if sorted_versions:
            best_v = sorted_versions[0]
            best_capital = self.capitals[best_v]
            best_ret = (best_capital / STARTING_CAPITAL - 1) * 100
            print(f"\n{C.Y}LEADER: {best_v} - ${best_capital:.4f} ({best_ret:+.2f}%){C.W}")

    def _print_final_summary(self):
        """Print final summary with rankings"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        feed_stats = self.feed.get_stats()

        print(f"\n\n{'='*100}")
        print(f"{C.Y}{C.BOLD}RENAISSANCE V1-V25 MEGA FEED TEST - FINAL RESULTS{C.W}")
        print(f"{'='*100}")
        print(f"Duration: {elapsed/60:.1f} minutes | Total Ticks: {self.tick_count:,} ({feed_stats['ticks_per_second']:.1f}/s)")
        print(f"Active Feeds: {feed_stats['active_feeds']}/9 exchanges")

        # Sort by capital (best first)
        sorted_versions = sorted(self.versions, key=lambda v: self.capitals[v], reverse=True)

        print(f"\n{C.Y}FINAL RANKINGS (by Capital):{C.W}")
        print(f"{'RANK':<5} {'VER':<4} {'NAME':<16} {'CAPITAL':>14} {'RETURN':>10} {'TRADES':>8} {'WR':>7} {'PF':>7}")
        print(f"{'-'*5} {'-'*4} {'-'*16} {'-'*14} {'-'*10} {'-'*8} {'-'*7} {'-'*7}")

        for rank, v in enumerate(sorted_versions, 1):
            config = get_config(v)
            name = config.get('name', v)[:16]
            capital = self.capitals[v]
            ret = (capital / STARTING_CAPITAL - 1) * 100
            total_trades = self.wins[v] + self.losses[v]
            win_rate = (self.wins[v] / total_trades * 100) if total_trades > 0 else 0

            # Calculate profit factor
            wins_total = sum(t['pnl_dollars'] for t in self.trades[v] if t['pnl_dollars'] > 0)
            losses_total = abs(sum(t['pnl_dollars'] for t in self.trades[v] if t['pnl_dollars'] < 0))
            pf = wins_total / losses_total if losses_total > 0 else float('inf')

            # Color based on return
            ret_color = C.G if ret > 0 else (C.R if ret < 0 else C.W)

            medal = ""
            if rank == 1:
                medal = f"{C.Y}[1st]{C.W}"
            elif rank == 2:
                medal = f"{C.W}[2nd]{C.W}"
            elif rank == 3:
                medal = f"{C.M}[3rd]{C.W}"

            pf_str = f"{pf:.2f}x" if pf < 100 else "INF"

            print(f"{medal}{rank:<5} {v:<4} {name:<16} ${capital:>13.4f} {ret_color}{ret:>+9.2f}%{C.W} {total_trades:>8} {win_rate:>6.1f}% {pf_str:>7}")

        # Winner details
        if sorted_versions:
            winner = sorted_versions[0]
            winner_config = get_config(winner)
            winner_ret = (self.capitals[winner] / STARTING_CAPITAL - 1) * 100

            print(f"\n{C.G}{'='*100}{C.W}")
            print(f"{C.G}WINNER: {winner} - {winner_config.get('name', winner)}{C.W}")
            print(f"{C.G}  Final Capital: ${self.capitals[winner]:.4f} (Return: {winner_ret:+.2f}%){C.W}")
            print(f"{C.G}  Trades: {self.wins[winner] + self.losses[winner]} | "
                  f"Wins: {self.wins[winner]} | Losses: {self.losses[winner]}{C.W}")
            print(f"{C.G}{'='*100}{C.W}")

            # Recent trades for winner
            if self.trades[winner]:
                print(f"\n{C.Y}WINNER RECENT TRADES:{C.W}")
                for t in self.trades[winner][-10:]:
                    dir_str = "LONG " if t['direction'] > 0 else "SHORT"
                    color = C.G if t['pnl_dollars'] > 0 else C.R
                    print(f"  {dir_str} ${t['entry_price']:,.2f} -> ${t['exit_price']:,.2f} "
                          f"{color}{t['pnl_pct']*100:+.3f}%{C.W} ({t['reason']}) [{t['duration']:.1f}s]")

    def start(self):
        """Start the mega parallel test"""
        self.running = True
        self.start_time = time.time()

        def signal_handler(sig, frame):
            print(f"\n{C.Y}Shutting down...{C.W}")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)

        print(f"\n{'='*100}")
        print(f"{C.Y}{C.BOLD}RENAISSANCE V1-V25 PARALLEL TEST - MEGA 9-EXCHANGE FEED{C.W}")
        print(f"{'='*100}")
        print(f"""
EXCHANGES CONNECTED:
  TIER 1: Coinbase, Kraken, Gemini, Bitstamp
  TIER 2: OKCoin, Blockchain.com, CEX.IO
  TIER 3: Bitfinex, Gate.io

TARGET: 5-20+ ticks/second combined
""")
        print(f"Versions: V1-V25 ({len(self.versions)} strategies)")
        print(f"Starting Capital: ${STARTING_CAPITAL:.2f} each")
        print(f"Duration: {self.duration/60:.1f} minutes")
        print(f"{'='*100}\n")

        # Register tick callback
        self.feed.on_tick(self.on_tick)

        # Start mega feed
        self.feed.start()

        print("Connecting to 9 exchanges... please wait...")
        time.sleep(5)  # Wait for connections

        try:
            while self.running:
                time.sleep(0.1)

                if self.duration and (time.time() - self.start_time) >= self.duration:
                    print(f"\n{C.Y}Duration limit reached.{C.W}")
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.feed.stop()
            self._print_final_summary()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Renaissance V1-V25 Mega Feed Test')
    parser.add_argument('--duration', '-d', type=int, default=600,
                        help='Test duration in seconds (default: 600 = 10 min)')
    args = parser.parse_args()

    trader = MegaParallelTrader(duration=args.duration)
    trader.start()


if __name__ == '__main__':
    main()
