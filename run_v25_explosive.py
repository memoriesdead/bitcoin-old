#!/usr/bin/env python3
"""
RENAISSANCE V1-V25 EXPLOSIVE TRADER
===================================
Optimized for MAXIMUM volume capture using USA-only CCXT feed.

KEY OPTIMIZATIONS:
1. Profit targets ABOVE spread costs (minimum 0.05% = $50 spread cost @ $100K BTC)
2. Proper Kelly sizing based on actual win rate
3. Volume-weighted signal generation
4. ALL 25 versions displayed
5. Edge calculation after costs

MATHEMATICAL FOUNDATION:
- Expected Edge = WinRate * ProfitTarget - (1-WinRate) * StopLoss - SpreadCost
- For positive edge: WinRate > (StopLoss + Spread) / (ProfitTarget + StopLoss)
- With 0.03% spread, 0.1% target, 0.05% stop:
  WinRate > (0.05 + 0.03) / (0.1 + 0.05) = 53.3% minimum needed
"""
import sys
import os
import time
import signal
import threading
from datetime import datetime
from collections import deque
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ccxt_usa_optimized import CCXTUSAOptimized, Tick

# Colors
class C:
    G = '\033[92m'
    R = '\033[91m'
    Y = '\033[93m'
    B = '\033[94m'
    M = '\033[95m'
    C = '\033[96m'
    W = '\033[0m'
    BOLD = '\033[1m'


STARTING_CAPITAL = 10.0

# OPTIMIZED V1-V25 CONFIGS
# Formula: Edge = WinRate * TP - (1-WinRate) * SL - SpreadCost
# Minimum WinRate needed = (SL + Spread) / (TP + SL)
EXPLOSIVE_CONFIGS = {
    # CONSERVATIVE TIER (V1-V5): Higher win rate required, safer stops
    # Target 55-60% win rate, profit target > 3x spread
    "V1": {"name": "SAFE_EDGE", "profit_target": 0.001, "stop_loss": 0.0005, "kelly_frac": 0.10, "max_hold_sec": 10, "min_wr": 0.55},
    "V2": {"name": "SAFE_MOMENTUM", "profit_target": 0.0015, "stop_loss": 0.0008, "kelly_frac": 0.12, "max_hold_sec": 8, "min_wr": 0.55},
    "V3": {"name": "SAFE_TREND", "profit_target": 0.002, "stop_loss": 0.001, "kelly_frac": 0.15, "max_hold_sec": 6, "min_wr": 0.54},
    "V4": {"name": "SAFE_REVERSION", "profit_target": 0.0025, "stop_loss": 0.0012, "kelly_frac": 0.18, "max_hold_sec": 5, "min_wr": 0.53},
    "V5": {"name": "SAFE_COMPOUND", "profit_target": 0.003, "stop_loss": 0.0015, "kelly_frac": 0.20, "max_hold_sec": 4, "min_wr": 0.52},

    # MODERATE TIER (V6-V10): Balanced risk/reward
    "V6": {"name": "BALANCED_EDGE", "profit_target": 0.004, "stop_loss": 0.002, "kelly_frac": 0.25, "max_hold_sec": 3, "min_wr": 0.52},
    "V7": {"name": "BALANCED_FLOW", "profit_target": 0.005, "stop_loss": 0.0025, "kelly_frac": 0.28, "max_hold_sec": 3, "min_wr": 0.51},
    "V8": {"name": "BALANCED_VWAP", "profit_target": 0.006, "stop_loss": 0.003, "kelly_frac": 0.30, "max_hold_sec": 2, "min_wr": 0.51},
    "V9": {"name": "BALANCED_MICRO", "profit_target": 0.008, "stop_loss": 0.004, "kelly_frac": 0.32, "max_hold_sec": 2, "min_wr": 0.51},
    "V10": {"name": "BALANCED_MAX", "profit_target": 0.01, "stop_loss": 0.005, "kelly_frac": 0.35, "max_hold_sec": 2, "min_wr": 0.50},

    # AGGRESSIVE TIER (V11-V15): Higher frequency, tighter stops
    "V11": {"name": "AGGRO_SCALP", "profit_target": 0.008, "stop_loss": 0.004, "kelly_frac": 0.40, "max_hold_sec": 1.5, "min_wr": 0.51},
    "V12": {"name": "AGGRO_MOMENTUM", "profit_target": 0.01, "stop_loss": 0.005, "kelly_frac": 0.45, "max_hold_sec": 1.5, "min_wr": 0.50},
    "V13": {"name": "AGGRO_BREAKOUT", "profit_target": 0.012, "stop_loss": 0.006, "kelly_frac": 0.50, "max_hold_sec": 1, "min_wr": 0.50},
    "V14": {"name": "AGGRO_FLOW", "profit_target": 0.015, "stop_loss": 0.008, "kelly_frac": 0.55, "max_hold_sec": 1, "min_wr": 0.50},
    "V15": {"name": "AGGRO_MAX", "profit_target": 0.02, "stop_loss": 0.01, "kelly_frac": 0.60, "max_hold_sec": 1, "min_wr": 0.49},

    # EXPLOSIVE TIER (V16-V20): Maximum capture
    "V16": {"name": "EXPLOSIVE_BASE", "profit_target": 0.015, "stop_loss": 0.008, "kelly_frac": 0.65, "max_hold_sec": 0.8, "min_wr": 0.50},
    "V17": {"name": "EXPLOSIVE_COMPOUND", "profit_target": 0.02, "stop_loss": 0.01, "kelly_frac": 0.70, "max_hold_sec": 0.6, "min_wr": 0.49},
    "V18": {"name": "EXPLOSIVE_VINCE", "profit_target": 0.025, "stop_loss": 0.012, "kelly_frac": 0.75, "max_hold_sec": 0.5, "min_wr": 0.49},
    "V19": {"name": "EXPLOSIVE_LEVERAGE", "profit_target": 0.03, "stop_loss": 0.015, "kelly_frac": 0.80, "max_hold_sec": 0.4, "min_wr": 0.48},
    "V20": {"name": "EXPLOSIVE_MAX", "profit_target": 0.04, "stop_loss": 0.02, "kelly_frac": 0.85, "max_hold_sec": 0.3, "min_wr": 0.48},

    # MAXIMUM TIER (V21-V25): Full throttle
    "V21": {"name": "MAX_GROWTH", "profit_target": 0.03, "stop_loss": 0.015, "kelly_frac": 0.90, "max_hold_sec": 0.3, "min_wr": 0.48},
    "V22": {"name": "MAX_COMPOUND", "profit_target": 0.04, "stop_loss": 0.02, "kelly_frac": 0.95, "max_hold_sec": 0.2, "min_wr": 0.48},
    "V23": {"name": "MAX_KELLY", "profit_target": 0.05, "stop_loss": 0.025, "kelly_frac": 1.0, "max_hold_sec": 0.2, "min_wr": 0.47},
    "V24": {"name": "MAX_VINCE", "profit_target": 0.06, "stop_loss": 0.03, "kelly_frac": 1.0, "max_hold_sec": 0.1, "min_wr": 0.47},
    "V25": {"name": "MAX_EXPLOSIVE", "profit_target": 0.08, "stop_loss": 0.04, "kelly_frac": 1.0, "max_hold_sec": 0.1, "min_wr": 0.46},
}


class ExplosiveStrategy:
    """
    Volume-capture strategy with edge calculation.
    """

    def __init__(self, config: dict):
        self.config = config
        self.prices = deque(maxlen=100)
        self.volumes = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)

        # Derived metrics
        self.momentum = 0.0
        self.volatility = 0.0
        self.volume_imbalance = 0.0  # Buy vs sell volume
        self.spread_cost = 0.0003  # Default 0.03% spread cost (round trip)

    def update(self, tick: Tick):
        """Update strategy with new tick"""
        self.prices.append(tick.price)
        self.volumes.append(tick.usd_volume)
        self.timestamps.append(tick.timestamp_ms)

        if len(self.prices) >= 5:
            # Calculate momentum (last 5 ticks)
            price_change = (self.prices[-1] - self.prices[-5]) / self.prices[-5]
            self.momentum = price_change

            # Calculate realized volatility
            returns = [(self.prices[i] - self.prices[i-1]) / self.prices[i-1]
                      for i in range(-5, 0)]
            self.volatility = math.sqrt(sum(r**2 for r in returns) / len(returns)) if returns else 0

            # Volume imbalance
            recent_volumes = list(self.volumes)[-10:]
            total_vol = sum(recent_volumes)
            self.volume_imbalance = sum(recent_volumes[-5:]) / total_vol if total_vol > 0 else 0.5

    def update_spread(self, spread_pct: float):
        """Update spread cost from live data"""
        self.spread_cost = spread_pct / 100 * 2  # Round trip cost

    def get_signal(self, price: float) -> tuple:
        """
        Generate trading signal based on volume-weighted momentum.

        Returns: (direction, edge, size_fraction)
        direction: 1 = long, -1 = short, 0 = no signal
        edge: expected edge per trade (after costs)
        size_fraction: Kelly-optimal position size
        """
        if len(self.prices) < 10:
            return 0, 0, 0

        # Calculate signal components
        momentum_signal = self.momentum
        volatility_signal = self.volatility
        volume_signal = self.volume_imbalance - 0.5  # Centered around 0

        # Combined signal
        signal_strength = (
            momentum_signal * 0.4 +      # Trend following
            volume_signal * 0.3 +        # Volume imbalance
            (volatility_signal > 0.0005) * 0.3  # Volatility filter
        )

        # Direction
        if abs(signal_strength) < 0.00001:  # Minimum signal threshold
            return 0, 0, 0

        direction = 1 if signal_strength > 0 else -1

        # Estimate win rate based on signal strength
        base_win_rate = 0.50
        signal_boost = min(abs(signal_strength) * 100, 0.10)  # Max 10% boost
        estimated_win_rate = base_win_rate + signal_boost

        # Calculate expected edge
        tp = self.config["profit_target"]
        sl = self.config["stop_loss"]
        spread = self.spread_cost

        # Edge = WinRate * TP - (1-WinRate) * SL - Spread
        edge = estimated_win_rate * tp - (1 - estimated_win_rate) * sl - spread

        # Only trade if positive edge AND above minimum win rate
        min_wr = self.config.get("min_wr", 0.50)
        if edge <= 0 or estimated_win_rate < min_wr:
            return 0, 0, 0

        # Kelly fraction
        # f* = (p*b - q) / b where b = tp/sl, p = win_rate, q = 1-p
        b = tp / sl if sl > 0 else 2
        kelly_f = (estimated_win_rate * b - (1 - estimated_win_rate)) / b
        kelly_f = max(0, min(kelly_f, self.config["kelly_frac"]))  # Capped

        return direction, edge, kelly_f

    def get_expected_profit_per_sec(self, capital: float, trades_per_sec: float = 1.0) -> float:
        """
        Calculate expected profit per second.

        Formula: Capital * Kelly * Edge * TradesPerSec
        """
        _, edge, kelly = self.get_signal(self.prices[-1] if self.prices else 0)
        if edge <= 0:
            return 0
        return capital * kelly * edge * trades_per_sec


class ExplosiveTrader:
    """
    Run ALL V1-V25 strategies on USA-optimized feed.
    """

    def __init__(self, duration=600):
        self.duration = duration
        self.versions = [f'V{i}' for i in range(1, 26)]

        # Initialize strategies
        self.strategies = {}
        self.positions = {}
        self.capitals = {}
        self.trades = {}
        self.wins = {}
        self.losses = {}
        self.total_pnls = {}
        self.edges_captured = {}

        for v in self.versions:
            config = EXPLOSIVE_CONFIGS[v].copy()
            config["version"] = v
            self.strategies[v] = ExplosiveStrategy(config)
            self.positions[v] = None
            self.capitals[v] = STARTING_CAPITAL
            self.trades[v] = []
            self.wins[v] = 0
            self.losses[v] = 0
            self.total_pnls[v] = 0.0
            self.edges_captured[v] = 0.0

        # Initialize USA-optimized feed
        self.feed = CCXTUSAOptimized(buffer_size=500000)

        # Stats
        self.tick_count = 0
        self.start_time = None
        self.last_print_time = 0
        self.last_price = None
        self.running = False
        self.price_history = deque(maxlen=10000)
        self.total_volume_captured = 0.0

    def on_tick(self, tick: Tick):
        """Process each tick for ALL strategies"""
        self.tick_count += 1
        price = tick.price
        self.last_price = price
        self.price_history.append(price)
        self.total_volume_captured += tick.usd_volume
        ts = time.time()

        # Get current spread from feed
        feed_stats = self.feed.get_stats()
        avg_spread = feed_stats.get("avg_spread_pct", 0.03)

        # Update and check each strategy
        for v in self.versions:
            strategy = self.strategies[v]
            config = EXPLOSIVE_CONFIGS[v]

            # Update strategy with tick and spread
            strategy.update(tick)
            strategy.update_spread(avg_spread)

            # Check exit first if in position
            if self.positions[v]:
                self._check_exit(v, price, ts)

            # Get signal for new entry
            if not self.positions[v]:
                direction, edge, kelly_frac = strategy.get_signal(price)

                if direction != 0 and edge > 0:
                    self._enter_trade(v, direction, edge, kelly_frac, price, ts)

        # Print status periodically
        now = time.time()
        if now - self.last_print_time >= 3.0:
            self._print_status()
            self.last_print_time = now

    def _enter_trade(self, version: str, direction: int, edge: float, kelly_frac: float, price: float, ts: float):
        """Enter a trade"""
        capital = self.capitals[version]

        # Position size based on Kelly
        position_value = capital * min(kelly_frac, 0.80)
        btc_size = position_value / price

        self.positions[version] = {
            'direction': direction,
            'entry_price': price,
            'size': btc_size,
            'value': position_value,
            'entry_time': ts,
            'expected_edge': edge
        }

    def _check_exit(self, version: str, price: float, ts: float):
        """Check if position should be exited"""
        if not self.positions[version]:
            return

        config = EXPLOSIVE_CONFIGS[version]
        position = self.positions[version]
        entry_price = position['entry_price']
        direction = position['direction']
        entry_time = position['entry_time']

        # Calculate P&L
        if direction > 0:  # Long
            pnl_pct = (price - entry_price) / entry_price
        else:  # Short
            pnl_pct = (entry_price - price) / entry_price

        profit_target = config["profit_target"]
        stop_loss = config["stop_loss"]
        max_hold = config["max_hold_sec"]

        should_exit = False
        reason = ""

        if pnl_pct >= profit_target:
            should_exit = True
            reason = "TP"
        elif pnl_pct <= -stop_loss:
            should_exit = True
            reason = "SL"
        elif (ts - entry_time) >= max_hold:
            should_exit = True
            reason = "TIME"

        if should_exit:
            self._exit_trade(version, price, reason, pnl_pct)

    def _exit_trade(self, version: str, price: float, reason: str, pnl_pct: float):
        """Exit trade"""
        if not self.positions[version]:
            return

        position = self.positions[version]
        position_value = position['value']
        pnl_dollars = position_value * pnl_pct
        expected_edge = position.get('expected_edge', 0)

        self.capitals[version] += pnl_dollars
        self.total_pnls[version] += pnl_dollars

        if pnl_dollars > 0:
            self.wins[version] += 1
            self.edges_captured[version] += pnl_pct
        else:
            self.losses[version] += 1

        self.trades[version].append({
            'entry_price': position['entry_price'],
            'exit_price': price,
            'direction': position['direction'],
            'pnl_pct': pnl_pct,
            'pnl_dollars': pnl_dollars,
            'reason': reason,
            'duration': time.time() - position['entry_time'],
            'expected_edge': expected_edge
        })

        self.positions[version] = None

    def _print_status(self):
        """Print live status for ALL 25 strategies"""
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

        print(f"\n{C.B}{'='*120}{C.W}")
        print(f"{C.B}[{datetime.now().strftime('%H:%M:%S')}] BTC: ${self.last_price:,.2f} | "
              f"Ticks: {self.tick_count:,} ({tps:.1f}/s) | "
              f"Range: ${price_range:.2f} ({price_pct:.4f}%) | "
              f"Exchanges: {feed_stats['active_exchanges']}{C.W}")
        print(f"{C.B}  USD Volume: ${self.total_volume_captured:,.0f} | "
              f"Market Capture: {feed_stats.get('market_capture_pct', 0):.2f}% | "
              f"Avg Spread: {feed_stats.get('avg_spread_pct', 0):.4f}%{C.W}")
        print(f"{C.B}{'='*120}{C.W}")

        # Header for ALL 25 versions
        print(f"{'VER':<4} {'NAME':<16} {'CAPITAL':>12} {'P&L':>10} {'TR':>4} {'WR':>5} {'EDGE':>7} {'POS':<12}")
        print(f"{'-'*4} {'-'*16} {'-'*12} {'-'*10} {'-'*4} {'-'*5} {'-'*7} {'-'*12}")

        # Sort by capital (best first)
        sorted_versions = sorted(self.versions, key=lambda v: self.capitals[v], reverse=True)

        # Show ALL 25 versions
        for rank, v in enumerate(sorted_versions, 1):
            config = EXPLOSIVE_CONFIGS[v]
            name = config["name"][:16]
            capital = self.capitals[v]
            pnl = self.total_pnls[v]
            total_trades = self.wins[v] + self.losses[v]
            win_rate = (self.wins[v] / total_trades * 100) if total_trades > 0 else 0

            # Calculate realized edge
            if total_trades > 0:
                avg_edge = self.edges_captured[v] / self.wins[v] * 100 if self.wins[v] > 0 else 0
            else:
                avg_edge = 0

            # Color based on P&L
            if pnl > 0:
                pnl_color = C.G
            elif pnl < 0:
                pnl_color = C.R
            else:
                pnl_color = C.W

            # Rank color
            if rank <= 5:
                rank_color = C.Y
            elif rank <= 10:
                rank_color = C.C
            else:
                rank_color = C.W

            # Position indicator
            pos_str = ""
            if self.positions[v]:
                pos = self.positions[v]
                pos_dir = "L" if pos['direction'] > 0 else "S"
                unrealized = (self.last_price - pos['entry_price']) / pos['entry_price']
                if pos['direction'] < 0:
                    unrealized = -unrealized
                pos_color = C.G if unrealized >= 0 else C.R
                pos_str = f"{pos_dir} {pos_color}{unrealized*100:+.2f}%{C.W}"

            print(f"{rank_color}[{rank:>2}]{C.W} {v:<4} {name:<16} ${capital:>11.4f} {pnl_color}${pnl:>+9.4f}{C.W} "
                  f"{total_trades:>4} {win_rate:>4.0f}% {avg_edge:>6.2f}% {pos_str}")

        # Summary stats
        total_capital = sum(self.capitals.values())
        total_pnl = sum(self.total_pnls.values())
        total_trades_all = sum(self.wins[v] + self.losses[v] for v in self.versions)
        total_wins = sum(self.wins.values())
        overall_wr = total_wins / total_trades_all * 100 if total_trades_all > 0 else 0

        print(f"\n{C.Y}AGGREGATE: Capital ${total_capital:,.2f} | P&L ${total_pnl:+.4f} | "
              f"Trades {total_trades_all} | WR {overall_wr:.1f}%{C.W}")

    def _print_final_summary(self):
        """Print final summary"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        feed_stats = self.feed.get_stats()

        print(f"\n\n{'='*120}")
        print(f"{C.Y}{C.BOLD}RENAISSANCE V1-V25 EXPLOSIVE TRADER - FINAL RESULTS{C.W}")
        print(f"{'='*120}")
        print(f"Duration: {elapsed/60:.1f} min | Ticks: {self.tick_count:,} ({feed_stats['ticks_per_second']:.1f}/s)")
        print(f"USD Volume Captured: ${self.total_volume_captured:,.0f}")
        print(f"Market Capture: {feed_stats.get('market_capture_pct', 0):.2f}%")

        # Sort by capital
        sorted_versions = sorted(self.versions, key=lambda v: self.capitals[v], reverse=True)

        print(f"\n{C.Y}FINAL RANKINGS (ALL 25 VERSIONS):{C.W}")
        print(f"{'RANK':<5} {'VER':<5} {'NAME':<18} {'CAPITAL':>14} {'RETURN':>10} {'TRADES':>8} {'WR':>7} {'EDGE':>8}")
        print(f"{'-'*5} {'-'*5} {'-'*18} {'-'*14} {'-'*10} {'-'*8} {'-'*7} {'-'*8}")

        for rank, v in enumerate(sorted_versions, 1):
            config = EXPLOSIVE_CONFIGS[v]
            name = config["name"][:18]
            capital = self.capitals[v]
            ret = (capital / STARTING_CAPITAL - 1) * 100
            total_trades = self.wins[v] + self.losses[v]
            win_rate = (self.wins[v] / total_trades * 100) if total_trades > 0 else 0
            avg_edge = self.edges_captured[v] / self.wins[v] * 100 if self.wins[v] > 0 else 0

            ret_color = C.G if ret > 0 else (C.R if ret < 0 else C.W)

            medal = ""
            if rank == 1:
                medal = f"{C.Y}[1st]{C.W}"
            elif rank == 2:
                medal = "[2nd]"
            elif rank == 3:
                medal = f"{C.M}[3rd]{C.W}"
            else:
                medal = f"[{rank}]"

            print(f"{medal:<7} {v:<5} {name:<18} ${capital:>13.4f} {ret_color}{ret:>+9.2f}%{C.W} "
                  f"{total_trades:>8} {win_rate:>6.1f}% {avg_edge:>7.2f}%")

        # Winner details
        winner = sorted_versions[0]
        winner_config = EXPLOSIVE_CONFIGS[winner]
        winner_ret = (self.capitals[winner] / STARTING_CAPITAL - 1) * 100

        print(f"\n{C.G}{'='*120}{C.W}")
        print(f"{C.G}{C.BOLD}WINNER: {winner} - {winner_config['name']}{C.W}")
        print(f"{C.G}  Final Capital: ${self.capitals[winner]:.4f} (Return: {winner_ret:+.2f}%)")
        print(f"{C.G}  Trades: {self.wins[winner] + self.losses[winner]} | Wins: {self.wins[winner]} | Losses: {self.losses[winner]}")
        print(f"{C.G}  Config: TP {winner_config['profit_target']*100:.2f}% | SL {winner_config['stop_loss']*100:.2f}% | "
              f"Kelly {winner_config['kelly_frac']*100:.0f}%{C.W}")
        print(f"{C.G}{'='*120}{C.W}")

    def start(self):
        """Start the explosive trader"""
        self.running = True
        self.start_time = time.time()

        def signal_handler(sig, frame):
            print(f"\n{C.Y}Shutting down...{C.W}")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)

        print(f"\n{'='*120}")
        print(f"{C.Y}{C.BOLD}RENAISSANCE V1-V25 EXPLOSIVE TRADER - USA-ONLY FEED{C.W}")
        print(f"{'='*120}")
        print(f"""
EXCHANGES: Coinbase, Kraken, Gemini, Bitstamp, Crypto.com, Bitfinex (USA-LEGAL ONLY)

OPTIMIZATION FEATURES:
  - Millisecond tick precision
  - Volume-weighted signals
  - Dynamic spread cost adjustment
  - Edge calculation before entry
  - Kelly-optimal position sizing

ALL 25 VERSIONS RUNNING:
  V1-V5:   Conservative (0.1-0.3% targets)
  V6-V10:  Moderate (0.4-1.0% targets)
  V11-V15: Aggressive (0.8-2.0% targets)
  V16-V20: Explosive (1.5-4.0% targets)
  V21-V25: Maximum (3.0-8.0% targets)
""")
        print(f"Starting Capital: ${STARTING_CAPITAL:.2f} each")
        print(f"Duration: {self.duration/60:.1f} minutes")
        print(f"{'='*120}\n")

        # Register tick callback
        self.feed.on_tick(self.on_tick)

        # Start feed
        self.feed.start()

        print("Connecting to USA-legal exchanges...")
        time.sleep(5)

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
    parser = argparse.ArgumentParser(description='Renaissance V1-V25 Explosive Trader')
    parser.add_argument('--duration', '-d', type=int, default=600,
                        help='Test duration in seconds (default: 600 = 10 min)')
    args = parser.parse_args()

    trader = ExplosiveTrader(duration=args.duration)
    trader.start()


if __name__ == '__main__':
    main()
