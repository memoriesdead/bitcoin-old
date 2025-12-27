#!/usr/bin/env python3
"""
RENTECH BACKTESTER

Backtests trading signals against historical price data.

Features:
- Walk-forward optimization
- Transaction cost modeling
- Performance metrics (Sharpe, Sortino, Max DD)
- Statistical significance testing
"""
import numpy as np
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json


@dataclass
class Trade:
    """Single trade record."""
    entry_time: int
    exit_time: int
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float


@dataclass
class BacktestResult:
    """Backtest performance summary."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_pnl: float
    avg_holding_period: float  # in blocks
    trades: List[Trade]


class Backtester:
    """
    Backtest trading signals against historical data.
    """

    def __init__(self,
                 features_db: str = "data/bitcoin_features.db",
                 prices_db: str = "data/historical_flows.db",
                 initial_capital: float = 100000,
                 position_size: float = 0.1,  # 10% per trade
                 slippage: float = 0.001,  # 0.1%
                 commission: float = 0.0004):  # 0.04%
        self.features_db = features_db
        self.prices_db = prices_db
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.slippage = slippage
        self.commission = commission

    def load_features(self, start_height: int, end_height: int) -> List[dict]:
        """Load block features from database."""
        conn = sqlite3.connect(self.features_db)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        rows = c.execute("""
            SELECT * FROM block_features
            WHERE height BETWEEN ? AND ?
            ORDER BY height
        """, (start_height, end_height)).fetchall()

        conn.close()
        return [dict(row) for row in rows]

    def load_prices(self, start_ts: int, end_ts: int) -> Dict[int, float]:
        """Load price data indexed by timestamp."""
        conn = sqlite3.connect(self.prices_db)
        c = conn.cursor()

        rows = c.execute("""
            SELECT timestamp, close FROM prices
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """, (start_ts, end_ts)).fetchall()

        conn.close()
        return {ts: price for ts, price in rows}

    def get_price_at_time(self, prices: Dict[int, float], timestamp: int) -> float:
        """Get price closest to timestamp."""
        if not prices:
            return 0.0

        # Find closest timestamp
        closest_ts = min(prices.keys(), key=lambda x: abs(x - timestamp))
        return prices[closest_ts]

    def run_backtest(self,
                     signals: List[dict],  # [{'timestamp': ts, 'direction': dir, 'confidence': conf}]
                     prices: Dict[int, float],
                     min_confidence: float = 0.3,
                     holding_period: int = 6 * 6,  # 6 hours in blocks
                     ) -> BacktestResult:
        """
        Run backtest on signals.

        Args:
            signals: List of signal dicts with timestamp, direction, confidence
            prices: Dict mapping timestamp to price
            min_confidence: Minimum confidence to take trade
            holding_period: How long to hold position (in blocks, ~10 min each)
        """
        capital = self.initial_capital
        peak_capital = capital
        max_drawdown = 0.0

        trades: List[Trade] = []
        equity_curve = [capital]

        position = None  # Current position: {'direction', 'entry_time', 'entry_price', 'size'}

        for signal in signals:
            ts = signal['timestamp']
            direction = signal['direction']
            confidence = signal['confidence']

            current_price = self.get_price_at_time(prices, ts)
            if current_price == 0:
                continue

            # Check if we need to exit current position
            if position:
                # Exit after holding period or on opposite signal
                hold_time = ts - position['entry_time']
                should_exit = (
                    hold_time >= holding_period * 600 or  # 600 sec per block
                    (direction != 'NEUTRAL' and direction != position['direction'])
                )

                if should_exit:
                    # Calculate exit
                    exit_price = current_price * (1 - self.slippage)
                    if position['direction'] == 'LONG':
                        pnl = (exit_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl = (position['entry_price'] - exit_price) / position['entry_price']

                    pnl -= self.commission * 2  # Entry and exit

                    trade_pnl = position['size'] * pnl
                    capital += trade_pnl

                    trades.append(Trade(
                        entry_time=position['entry_time'],
                        exit_time=ts,
                        direction=position['direction'],
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        size=position['size'],
                        pnl=trade_pnl,
                        pnl_pct=pnl,
                    ))

                    position = None

            # Enter new position if signal is strong enough
            if position is None and direction != 'NEUTRAL' and confidence >= min_confidence:
                entry_price = current_price * (1 + self.slippage)
                size = capital * self.position_size * confidence  # Scale by confidence

                position = {
                    'direction': direction,
                    'entry_time': ts,
                    'entry_price': entry_price,
                    'size': size,
                }

            # Update equity curve and drawdown
            equity_curve.append(capital)
            if capital > peak_capital:
                peak_capital = capital
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)

        # Close any remaining position
        if position and prices:
            last_ts = max(prices.keys())
            exit_price = prices[last_ts]
            if position['direction'] == 'LONG':
                pnl = (exit_price - position['entry_price']) / position['entry_price']
            else:
                pnl = (position['entry_price'] - exit_price) / position['entry_price']
            pnl -= self.commission * 2
            trade_pnl = position['size'] * pnl
            capital += trade_pnl
            trades.append(Trade(
                entry_time=position['entry_time'],
                exit_time=last_ts,
                direction=position['direction'],
                entry_price=position['entry_price'],
                exit_price=exit_price,
                size=position['size'],
                pnl=trade_pnl,
                pnl_pct=pnl,
            ))

        # Calculate metrics
        if not trades:
            return BacktestResult(
                total_return=0, sharpe_ratio=0, sortino_ratio=0,
                max_drawdown=0, win_rate=0, profit_factor=0,
                num_trades=0, avg_trade_pnl=0, avg_holding_period=0,
                trades=[]
            )

        returns = [t.pnl_pct for t in trades]
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        total_return = (capital - self.initial_capital) / self.initial_capital

        # Sharpe ratio (annualized, assuming 365*24*6 blocks per year)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24 * 6 / len(trades))
        else:
            sharpe = 0

        # Sortino ratio (downside deviation only)
        downside_returns = [r for r in returns if r < 0]
        if downside_returns and np.std(downside_returns) > 0:
            sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(365 * 24 * 6 / len(trades))
        else:
            sortino = sharpe

        win_rate = len(winning_trades) / len(trades) if trades else 0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_holding = np.mean([t.exit_time - t.entry_time for t in trades]) / 600  # Convert to blocks

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(trades),
            avg_trade_pnl=np.mean([t.pnl for t in trades]),
            avg_holding_period=avg_holding,
            trades=trades,
        )

    def statistical_significance(self, result: BacktestResult) -> dict:
        """
        Test statistical significance of backtest results.
        """
        if result.num_trades < 30:
            return {
                'sufficient_trades': False,
                't_statistic': 0,
                'p_value': 1.0,
                'confidence_interval': (0, 0),
            }

        returns = [t.pnl_pct for t in result.trades]

        # T-test: is mean return significantly different from zero?
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        n = len(returns)

        t_stat = mean / (std / np.sqrt(n)) if std > 0 else 0

        # Approximate p-value (two-tailed)
        # Using rough approximation for t-distribution
        p_value = 2 * (1 - min(0.9999, 0.5 + 0.5 * np.tanh(t_stat / 1.5)))

        # 95% confidence interval
        margin = 1.96 * std / np.sqrt(n)
        ci = (mean - margin, mean + margin)

        return {
            'sufficient_trades': True,
            't_statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': ci,
            'mean_return': mean,
            'std_return': std,
        }


def run_full_backtest():
    """Run complete backtest with pattern recognition."""
    from pattern_recognition import SignalCombiner

    print("=" * 60)
    print("RENTECH BACKTEST")
    print("=" * 60)

    # Initialize
    backtester = Backtester()
    combiner = SignalCombiner()

    # Load data
    print("[1] Loading data...")

    # Check if we have features
    features_path = Path("data/bitcoin_features.db")
    if not features_path.exists():
        print("    No features database found. Run bitcoin_pipeline.py first.")
        return

    features = backtester.load_features(700000, 927000)  # ~2021-2025
    print(f"    Loaded {len(features):,} blocks")

    # Load prices
    prices_db = Path("data/historical_flows.db")
    if prices_db.exists():
        conn = sqlite3.connect(prices_db)
        price_rows = conn.execute("SELECT timestamp, close FROM prices").fetchall()
        prices = {ts: price for ts, price in price_rows}
        conn.close()
        print(f"    Loaded {len(prices):,} price points")
    else:
        print("    No price database found!")
        return

    # Generate signals
    print("[2] Generating signals...")
    signals = []

    for feat in features:
        signal = combiner.generate_signal(feat, feat['timestamp'])
        signals.append({
            'timestamp': feat['timestamp'],
            'direction': signal.direction,
            'confidence': signal.confidence,
        })

    print(f"    Generated {len(signals):,} signals")
    long_count = sum(1 for s in signals if s['direction'] == 'LONG')
    short_count = sum(1 for s in signals if s['direction'] == 'SHORT')
    print(f"    LONG: {long_count}, SHORT: {short_count}, NEUTRAL: {len(signals) - long_count - short_count}")

    # Run backtest
    print("[3] Running backtest...")
    result = backtester.run_backtest(signals, prices)

    # Statistical significance
    stats = backtester.statistical_significance(result)

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total Return:     {result.total_return*100:,.2f}%")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:    {result.sortino_ratio:.2f}")
    print(f"Max Drawdown:     {result.max_drawdown*100:.2f}%")
    print(f"Win Rate:         {result.win_rate*100:.1f}%")
    print(f"Profit Factor:    {result.profit_factor:.2f}")
    print(f"Num Trades:       {result.num_trades:,}")
    print(f"Avg Trade PnL:    ${result.avg_trade_pnl:,.2f}")
    print(f"Avg Hold Period:  {result.avg_holding_period:.1f} blocks")
    print()
    print("Statistical Significance:")
    print(f"  T-statistic:    {stats.get('t_statistic', 0):.2f}")
    print(f"  P-value:        {stats.get('p_value', 1):.4f}")
    print(f"  95% CI:         {stats.get('confidence_interval', (0,0))}")
    print("=" * 60)


if __name__ == "__main__":
    run_full_backtest()
