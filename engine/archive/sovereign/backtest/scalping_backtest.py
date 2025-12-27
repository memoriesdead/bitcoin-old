#!/usr/bin/env python3
"""
Scalping Strategy Backtest
==========================
Test HFT scalping parameters on historical price data.

Parameters to optimize:
- Take Profit: 0.3%, 0.5%, 0.7%, 1.0%
- Stop Loss: 0.2%, 0.3%, 0.5%
- Max Hold: 5s, 10s, 30s, 60s

Since we don't have live blockchain signal data, we simulate
signals based on price momentum (proxy for what blockchain
flows would indicate).
"""
import sqlite3
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json


@dataclass
class Trade:
    entry_time: int
    entry_price: float
    exit_time: int
    exit_price: float
    direction: int  # 1=LONG, -1=SHORT
    exit_reason: str  # 'tp', 'sl', 'time'
    pnl_pct: float
    pnl_usd: float


@dataclass
class BacktestResult:
    params: Dict
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_pct: float
    total_pnl_usd: float
    avg_trade_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    exits_by_tp: int
    exits_by_sl: int
    exits_by_time: int
    daily_trades_avg: float


class ScalpingBacktester:
    """
    Backtester for scalping strategy.

    Uses minute-level price data to simulate sub-minute scalping.
    """

    # Fee structure (Hyperliquid maker orders)
    FEE_RATE = 0.0001  # 0.01% per side = 0.02% round trip

    def __init__(self, db_path: str = "data/historical_flows.db"):
        self.db_path = Path(db_path)
        self.prices: List[Tuple[int, float, float, float, float]] = []  # timestamp, open, high, low, close
        self._load_prices()

    def _load_prices(self):
        """Load price data from database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT timestamp, open, high, low, close FROM prices ORDER BY timestamp')
        self.prices = c.fetchall()
        conn.close()
        print(f"[BACKTEST] Loaded {len(self.prices):,} price candles")

    def simulate_signal(self, idx: int, lookback: int = 5) -> Optional[int]:
        """
        Simulate blockchain signal based on price momentum.

        In production, this would be replaced by actual blockchain
        signal detection (FIS, MPI, WM, CR, CRS).

        Returns: 1 (LONG), -1 (SHORT), or None (no signal)
        """
        if idx < lookback:
            return None

        # Calculate short-term momentum
        current_close = self.prices[idx][4]
        prev_close = self.prices[idx - lookback][4]

        momentum = (current_close - prev_close) / prev_close

        # Signal threshold (tune this based on blockchain edge)
        # This simulates ~50-100 trades per day
        threshold = 0.002  # 0.2% momentum

        if momentum > threshold:
            return 1  # LONG - momentum up (simulates outflow detection)
        elif momentum < -threshold:
            return -1  # SHORT - momentum down (simulates inflow detection)

        return None

    def run_backtest(
        self,
        take_profit_pct: float = 0.005,  # 0.5%
        stop_loss_pct: float = 0.003,     # 0.3%
        max_hold_candles: int = 10,       # ~10 minutes (1-min candles)
        capital: float = 100.0,
        leverage: float = 5.0,
        win_rate_boost: float = 0.05,     # Blockchain edge estimate (+5%)
        signal_frequency: float = 0.1     # Probability of signal per candle
    ) -> BacktestResult:
        """
        Run backtest with given parameters.

        Args:
            take_profit_pct: Take profit threshold (0.005 = 0.5%)
            stop_loss_pct: Stop loss threshold (0.003 = 0.3%)
            max_hold_candles: Maximum hold time in candles
            capital: Starting capital
            leverage: Leverage multiplier
            win_rate_boost: Estimated blockchain edge boost
            signal_frequency: How often to generate signals

        Returns:
            BacktestResult with full statistics
        """
        trades: List[Trade] = []
        position_size = capital * leverage

        current_capital = capital
        peak_capital = capital
        max_drawdown = 0.0

        i = 0
        while i < len(self.prices) - max_hold_candles - 1:
            # Generate signal (simulated blockchain signal)
            if random.random() > signal_frequency:
                i += 1
                continue

            signal = self.simulate_signal(i)
            if signal is None:
                i += 1
                continue

            # Apply blockchain edge boost to direction decision
            # This simulates having better than random direction selection
            if random.random() < win_rate_boost:
                # Blockchain edge: correct the signal if wrong
                future_price = self.prices[min(i + max_hold_candles, len(self.prices)-1)][4]
                actual_direction = 1 if future_price > self.prices[i][4] else -1
                signal = actual_direction

            entry_time = self.prices[i][0]
            entry_price = self.prices[i][4]  # Close price

            # Simulate trade execution through future candles
            exit_price = None
            exit_time = None
            exit_reason = None

            for j in range(1, max_hold_candles + 1):
                candle_idx = i + j
                if candle_idx >= len(self.prices):
                    break

                _, candle_open, candle_high, candle_low, candle_close = self.prices[candle_idx]

                # Check exit conditions
                if signal == 1:  # LONG
                    # Check TP
                    if candle_high >= entry_price * (1 + take_profit_pct):
                        exit_price = entry_price * (1 + take_profit_pct)
                        exit_reason = 'tp'
                        break
                    # Check SL
                    if candle_low <= entry_price * (1 - stop_loss_pct):
                        exit_price = entry_price * (1 - stop_loss_pct)
                        exit_reason = 'sl'
                        break
                else:  # SHORT
                    # Check TP
                    if candle_low <= entry_price * (1 - take_profit_pct):
                        exit_price = entry_price * (1 - take_profit_pct)
                        exit_reason = 'tp'
                        break
                    # Check SL
                    if candle_high >= entry_price * (1 + stop_loss_pct):
                        exit_price = entry_price * (1 + stop_loss_pct)
                        exit_reason = 'sl'
                        break

            # Time exit if no TP/SL hit
            if exit_price is None:
                candle_idx = min(i + max_hold_candles, len(self.prices) - 1)
                exit_price = self.prices[candle_idx][4]
                exit_reason = 'time'

            exit_time = self.prices[min(i + max_hold_candles, len(self.prices) - 1)][0]

            # Calculate PnL
            if signal == 1:  # LONG
                pnl_pct = (exit_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - exit_price) / entry_price

            # Subtract fees (round trip)
            pnl_pct -= (self.FEE_RATE * 2)

            pnl_usd = position_size * pnl_pct

            # Record trade
            trades.append(Trade(
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_price,
                direction=signal,
                exit_reason=exit_reason,
                pnl_pct=pnl_pct * 100,  # Convert to percentage
                pnl_usd=pnl_usd
            ))

            # Update capital
            current_capital += pnl_usd
            if current_capital > peak_capital:
                peak_capital = current_capital

            dd = (peak_capital - current_capital) / peak_capital if peak_capital > 0 else 0
            max_drawdown = max(max_drawdown, dd)

            # Skip ahead (cooldown period)
            i += max(1, max_hold_candles // 2)

        # Calculate statistics
        if not trades:
            return BacktestResult(
                params={'tp': take_profit_pct, 'sl': stop_loss_pct, 'hold': max_hold_candles},
                total_trades=0, wins=0, losses=0, win_rate=0,
                total_pnl_pct=0, total_pnl_usd=0, avg_trade_pnl=0,
                max_drawdown=0, sharpe_ratio=0,
                exits_by_tp=0, exits_by_sl=0, exits_by_time=0,
                daily_trades_avg=0
            )

        wins = len([t for t in trades if t.pnl_usd > 0])
        losses = len([t for t in trades if t.pnl_usd <= 0])
        win_rate = wins / len(trades) if trades else 0

        total_pnl_pct = sum(t.pnl_pct for t in trades)
        total_pnl_usd = sum(t.pnl_usd for t in trades)
        avg_trade_pnl = total_pnl_usd / len(trades)

        # Exit breakdown
        exits_tp = len([t for t in trades if t.exit_reason == 'tp'])
        exits_sl = len([t for t in trades if t.exit_reason == 'sl'])
        exits_time = len([t for t in trades if t.exit_reason == 'time'])

        # Sharpe ratio (simplified)
        returns = [t.pnl_pct for t in trades]
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5 if variance > 0 else 0.001
        sharpe = (mean_return / std_dev) * (252 ** 0.5)  # Annualized

        # Daily trades
        if len(self.prices) > 1:
            days = (self.prices[-1][0] - self.prices[0][0]) / 86400
            daily_trades = len(trades) / days if days > 0 else 0
        else:
            daily_trades = 0

        return BacktestResult(
            params={'tp': take_profit_pct * 100, 'sl': stop_loss_pct * 100, 'hold': max_hold_candles},
            total_trades=len(trades),
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_pnl_pct=total_pnl_pct,
            total_pnl_usd=total_pnl_usd,
            avg_trade_pnl=avg_trade_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            exits_by_tp=exits_tp,
            exits_by_sl=exits_sl,
            exits_by_time=exits_time,
            daily_trades_avg=daily_trades
        )

    def grid_search(
        self,
        tp_range: List[float] = [0.003, 0.005, 0.007, 0.01],
        sl_range: List[float] = [0.002, 0.003, 0.005],
        hold_range: List[int] = [5, 10, 30, 60],
        edge_boost: float = 0.05
    ) -> List[BacktestResult]:
        """
        Run grid search over parameter combinations.
        """
        results = []
        total_combos = len(tp_range) * len(sl_range) * len(hold_range)

        print(f"\n[GRID SEARCH] Testing {total_combos} parameter combinations...")
        print("=" * 80)

        combo = 0
        for tp in tp_range:
            for sl in sl_range:
                for hold in hold_range:
                    combo += 1
                    result = self.run_backtest(
                        take_profit_pct=tp,
                        stop_loss_pct=sl,
                        max_hold_candles=hold,
                        win_rate_boost=edge_boost
                    )
                    results.append(result)

                    print(f"[{combo}/{total_combos}] TP={tp*100:.1f}% SL={sl*100:.1f}% Hold={hold}m | "
                          f"Trades={result.total_trades} WR={result.win_rate:.1%} "
                          f"PnL=${result.total_pnl_usd:.2f}")

        # Rank by PnL
        results.sort(key=lambda r: r.total_pnl_usd, reverse=True)

        print("\n" + "=" * 80)
        print("TOP 5 PARAMETER COMBINATIONS")
        print("=" * 80)

        for i, r in enumerate(results[:5], 1):
            print(f"\n#{i}: TP={r.params['tp']:.1f}% SL={r.params['sl']:.1f}% Hold={r.params['hold']}min")
            print(f"    Trades: {r.total_trades}, Win Rate: {r.win_rate:.1%}")
            print(f"    Total PnL: ${r.total_pnl_usd:.2f} ({r.total_pnl_pct:.1f}%)")
            print(f"    Avg Trade: ${r.avg_trade_pnl:.2f}")
            print(f"    Max DD: {r.max_drawdown:.1%}")
            print(f"    Exits: TP={r.exits_by_tp}, SL={r.exits_by_sl}, Time={r.exits_by_time}")
            print(f"    Daily Trades: {r.daily_trades_avg:.1f}")

        return results

    def save_results(self, results: List[BacktestResult], filepath: str = "data/scalping_backtest_results.json"):
        """Save results to JSON."""
        output = []
        for r in results:
            output.append({
                "params": r.params,
                "total_trades": r.total_trades,
                "wins": r.wins,
                "losses": r.losses,
                "win_rate": r.win_rate,
                "total_pnl_pct": r.total_pnl_pct,
                "total_pnl_usd": r.total_pnl_usd,
                "avg_trade_pnl": r.avg_trade_pnl,
                "max_drawdown": r.max_drawdown,
                "sharpe_ratio": r.sharpe_ratio,
                "exits_by_tp": r.exits_by_tp,
                "exits_by_sl": r.exits_by_sl,
                "exits_by_time": r.exits_by_time,
                "daily_trades_avg": r.daily_trades_avg
            })

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n[+] Results saved to {filepath}")


def main():
    """Run scalping backtest."""
    print("\n" + "=" * 80)
    print("SCALPING STRATEGY BACKTEST")
    print("=" * 80)
    print("\nParameters being tested:")
    print("  Take Profit: 0.3%, 0.5%, 0.7%, 1.0%")
    print("  Stop Loss: 0.2%, 0.3%, 0.5%")
    print("  Max Hold: 5, 10, 30, 60 minutes")
    print("  Leverage: 5x")
    print("  Capital: $100")
    print("  Blockchain Edge Boost: +5% win rate")
    print("  Hyperliquid Fees: 0.02% round trip (maker)")

    backtester = ScalpingBacktester()

    if len(backtester.prices) < 100:
        print("\n[!] Need more price data for meaningful backtest.")
        return

    # Run grid search
    results = backtester.grid_search(
        tp_range=[0.003, 0.005, 0.007, 0.01],
        sl_range=[0.002, 0.003, 0.005],
        hold_range=[5, 10, 30, 60],
        edge_boost=0.05  # +5% from blockchain signals
    )

    # Save results
    backtester.save_results(results)

    # Summary
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    best = results[0]
    print(f"\nBest Parameters:")
    print(f"  Take Profit: {best.params['tp']:.1f}%")
    print(f"  Stop Loss: {best.params['sl']:.1f}%")
    print(f"  Max Hold: {best.params['hold']} minutes")
    print(f"\nExpected Performance:")
    print(f"  Win Rate: {best.win_rate:.1%}")
    print(f"  Daily Trades: {best.daily_trades_avg:.0f}")
    print(f"  Total PnL: ${best.total_pnl_usd:.2f}")
    print(f"  Max Drawdown: {best.max_drawdown:.1%}")

    print("\n[!] NOTE: These are simulated results.")
    print("[!] Real performance depends on actual blockchain signal quality.")


if __name__ == '__main__':
    main()
