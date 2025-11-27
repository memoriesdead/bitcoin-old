#!/usr/bin/env python3
"""
Renaissance Trading System - V1-V25 Historical Backtest
======================================================
Runs all 25 strategy versions on Bitcoin historical data (2014-2024)
to demonstrate explosive growth potential from $10 starting capital.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import BaseStrategy, get_config, STARTING_CAPITAL

# Colors
class C:
    G = '\033[92m'  # Green
    R = '\033[91m'  # Red
    Y = '\033[93m'  # Yellow
    B = '\033[94m'  # Blue
    W = '\033[0m'   # White/Reset

def load_historical_data():
    """Load Bitcoin historical price data"""
    csv_path = '/root/livetrading/kvm8/bitcoin_complete_history.csv'
    parquet_path = '/root/livetrading/kvm8/bitcoin_complete_history.parquet'

    try:
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
        else:
            df = pd.read_csv(csv_path)

        # Ensure we have the right columns
        if 'close' in df.columns:
            return df['close'].values
        elif 'price' in df.columns:
            return df['price'].values
        else:
            return df.iloc[:, 4].values  # Assume 5th column is close
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

class BacktestEngine:
    """Backtest all V1-V25 strategies on historical data"""

    def __init__(self, versions=None):
        self.versions = versions or [f'V{i}' for i in range(1, 26)]
        self.results = {}

    def run_strategy(self, version: str, prices: np.ndarray) -> dict:
        """Run a single strategy through all price data"""
        config = get_config(version)
        config['use_probability_mode'] = True

        strategy = BaseStrategy(config)

        capital = STARTING_CAPITAL
        peak_capital = capital
        position = None
        trades = []
        wins = 0
        losses = 0
        max_drawdown = 0

        profit_target = config.get('profit_target', 0.01)
        stop_loss = config.get('stop_loss', 0.005)
        kelly_frac = config.get('kelly_frac', 0.5)

        for i, price in enumerate(prices):
            if price <= 0:
                continue

            # Update strategy with new price
            strategy.update(price)

            # Check exit if in position
            if position:
                entry_price = position['entry_price']
                direction = position['direction']

                if direction > 0:  # Long
                    pnl_pct = (price - entry_price) / entry_price
                else:  # Short
                    pnl_pct = (entry_price - price) / entry_price

                # Check exit conditions
                should_exit = False
                reason = ""

                if pnl_pct >= profit_target:
                    should_exit = True
                    reason = "PROFIT"
                elif pnl_pct <= -stop_loss:
                    should_exit = True
                    reason = "STOP"

                if should_exit:
                    pnl_dollars = position['value'] * pnl_pct
                    capital += pnl_dollars

                    if pnl_dollars > 0:
                        wins += 1
                    else:
                        losses += 1

                    trades.append({
                        'entry': entry_price,
                        'exit': price,
                        'direction': direction,
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'reason': reason,
                        'capital_after': capital
                    })

                    position = None

                    # Track drawdown
                    peak_capital = max(peak_capital, capital)
                    drawdown = (peak_capital - capital) / peak_capital
                    max_drawdown = max(max_drawdown, drawdown)

            # Try to enter new position if not in one
            if not position and i > 20:  # Wait for warmup
                direction, size = strategy.get_signal(price)

                if direction != 0 and size > 0 and capital > 1:
                    # Use Kelly fraction for position sizing
                    position_size = min(kelly_frac, 0.8)  # Max 80%
                    position_value = capital * position_size

                    position = {
                        'entry_price': price,
                        'direction': direction,
                        'value': position_value
                    }

        # Close any open position at end
        if position and len(prices) > 0:
            final_price = prices[-1]
            entry_price = position['entry_price']
            direction = position['direction']

            if direction > 0:
                pnl_pct = (final_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - final_price) / entry_price

            pnl_dollars = position['value'] * pnl_pct
            capital += pnl_dollars

            if pnl_dollars > 0:
                wins += 1
            else:
                losses += 1

        total_trades = wins + losses
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        # Calculate profit factor
        gross_profit = sum(t['pnl_dollars'] for t in trades if t['pnl_dollars'] > 0)
        gross_loss = abs(sum(t['pnl_dollars'] for t in trades if t['pnl_dollars'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            'version': version,
            'name': config.get('name', version),
            'final_capital': capital,
            'return_pct': (capital / STARTING_CAPITAL - 1) * 100,
            'trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown * 100,
            'profit_target': profit_target * 100,
            'stop_loss': stop_loss * 100,
            'kelly_frac': kelly_frac * 100,
            'trade_history': trades
        }

    def run_all(self, prices: np.ndarray):
        """Run all strategies and collect results"""
        print(f"\n{'='*90}")
        print(f"{C.Y}RENAISSANCE V1-V25 HISTORICAL BACKTEST{C.W}")
        print(f"{'='*90}")
        print(f"Data points: {len(prices):,}")
        print(f"Price range: ${prices.min():,.2f} - ${prices.max():,.2f}")
        print(f"Starting capital: ${STARTING_CAPITAL:.2f}")
        print(f"{'='*90}\n")

        print(f"Running {len(self.versions)} strategies...")

        for i, version in enumerate(self.versions):
            result = self.run_strategy(version, prices)
            self.results[version] = result

            # Progress indicator
            color = C.G if result['return_pct'] > 0 else C.R
            print(f"  [{i+1:2d}/25] {version}: {color}${result['final_capital']:>12.2f}{C.W} ({result['return_pct']:+.1f}%) - {result['trades']} trades")

        return self.results

    def print_results(self):
        """Print final ranked results"""
        print(f"\n\n{'='*90}")
        print(f"{C.Y}FINAL RANKINGS - SORTED BY FINAL CAPITAL{C.W}")
        print(f"{'='*90}")

        # Sort by final capital
        sorted_results = sorted(self.results.values(), key=lambda x: x['final_capital'], reverse=True)

        print(f"{'RANK':<5} {'VER':<4} {'NAME':<16} {'CAPITAL':>14} {'RETURN':>10} {'TRADES':>7} {'WR':>6} {'PF':>6} {'MDD':>7}")
        print(f"{'-'*5} {'-'*4} {'-'*16} {'-'*14} {'-'*10} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")

        for rank, r in enumerate(sorted_results, 1):
            color = C.G if r['return_pct'] > 0 else C.R

            medal = ""
            if rank == 1:
                medal = f"{C.Y}[1st]{C.W}"
            elif rank == 2:
                medal = f"{C.W}[2nd]{C.W}"
            elif rank == 3:
                medal = f"{C.R}[3rd]{C.W}"

            cap_str = f"${r['final_capital']:,.2f}"
            if r['final_capital'] >= 1000000:
                cap_str = f"${r['final_capital']/1000000:.2f}M"
            elif r['final_capital'] >= 1000:
                cap_str = f"${r['final_capital']/1000:.2f}K"

            pf_str = f"{r['profit_factor']:.2f}x" if r['profit_factor'] < 100 else "INF"

            print(f"{medal}{rank:<5} {r['version']:<4} {r['name'][:16]:<16} {cap_str:>14} {color}{r['return_pct']:>+9.1f}%{C.W} {r['trades']:>7} {r['win_rate']:>5.1f}% {pf_str:>6} {r['max_drawdown']:>6.1f}%")

        # Winner summary
        winner = sorted_results[0]
        print(f"\n{'='*90}")
        print(f"{C.G}WINNER: {winner['version']} - {winner['name']}{C.W}")
        print(f"{C.G}  Final Capital: ${winner['final_capital']:,.2f}{C.W}")
        print(f"{C.G}  Return: {winner['return_pct']:+,.1f}%{C.W}")
        print(f"{C.G}  Win Rate: {winner['win_rate']:.1f}% | Profit Factor: {winner['profit_factor']:.2f}x{C.W}")
        print(f"{C.G}  Settings: TP {winner['profit_target']:.1f}% | SL {winner['stop_loss']:.2f}% | Kelly {winner['kelly_frac']:.0f}%{C.W}")
        print(f"{'='*90}")

        # Show top 3 trade summaries
        print(f"\n{C.Y}TOP 3 STRATEGIES - RECENT TRADES:{C.W}")
        for r in sorted_results[:3]:
            print(f"\n{r['version']} ({r['name']}):")
            for t in r['trade_history'][-5:]:  # Last 5 trades
                dir_str = "LONG " if t['direction'] > 0 else "SHORT"
                color = C.G if t['pnl_dollars'] > 0 else C.R
                print(f"  {dir_str} ${t['entry']:,.2f} -> ${t['exit']:,.2f} {color}{t['pnl_pct']*100:+.2f}%{C.W} ({t['reason']}) -> ${t['capital_after']:,.2f}")


def main():
    print(f"{C.B}Loading Bitcoin historical data...{C.W}")
    prices = load_historical_data()

    if prices is None or len(prices) == 0:
        print(f"{C.R}Failed to load data!{C.W}")
        return

    print(f"{C.G}Loaded {len(prices):,} price points{C.W}")

    # Run backtest
    engine = BacktestEngine()
    engine.run_all(prices)
    engine.print_results()


if __name__ == '__main__':
    main()
