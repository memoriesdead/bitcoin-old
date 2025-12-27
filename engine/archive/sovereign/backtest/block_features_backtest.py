#!/usr/bin/env python3
"""
Block Features Backtester

Uses block-level features (tx_count, fees, fullness) to generate trading signals.
This works with the data we've downloaded from mempool.space API.

Features available:
- tx_count: Number of transactions in block
- total_fees_btc: Total fees in BTC
- avg_fee_rate: Average fee rate (sat/vB)
- block_fullness: How full the block is (0-1)
- block_size: Block size in bytes
- block_weight: Block weight units
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: int  # 1=long, -1=short
    signal_value: float
    pnl_pct: float


@dataclass
class BacktestResult:
    strategy_name: str
    total_trades: int
    win_rate: float
    total_pnl_pct: float
    avg_pnl_pct: float
    sharpe_ratio: float
    max_drawdown: float
    winning_trades: int
    losing_trades: int


class BlockFeaturesBacktester:
    """Backtest trading strategies using block-level features."""

    def __init__(self, db_path: str = "data/bitcoin_features.db"):
        self.db_path = Path(db_path)
        self.price_data = {}  # timestamp -> price
        self.block_data = []  # list of block dicts

    def load_data(self, start_date: str = "2021-01-01", end_date: str = None):
        """Load block features and price data."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Load block features
        query = '''
            SELECT height, timestamp, tx_count, total_fees_btc, avg_fee_rate,
                   block_fullness, block_size
            FROM block_features
            WHERE timestamp IS NOT NULL
            ORDER BY height
        '''
        c.execute(query)
        rows = c.fetchall()

        self.block_data = []
        for row in rows:
            height, ts, tx_count, fees, fee_rate, fullness, size = row
            if ts:
                self.block_data.append({
                    'height': height,
                    'timestamp': ts,
                    'datetime': datetime.fromtimestamp(ts),
                    'tx_count': tx_count or 0,
                    'fees_btc': fees or 0,
                    'fee_rate': fee_rate or 0,
                    'fullness': fullness or 0,
                    'size': size or 0
                })

        print(f"Loaded {len(self.block_data):,} blocks")

        # Load price data from external source or generate from timestamps
        # For now, use simple price simulation based on block data availability
        conn.close()
        return len(self.block_data)

    def load_prices_from_csv(self, csv_path: str = "data/btc_prices.csv"):
        """Load minute price data from CSV."""
        csv_file = Path(csv_path)
        if not csv_file.exists():
            print(f"[!] Price file not found: {csv_path}")
            return False

        with open(csv_file) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    try:
                        ts = int(parts[0])
                        close = float(parts[4])
                        self.price_data[ts] = close
                    except:
                        pass

        print(f"Loaded {len(self.price_data):,} price candles")
        return True

    def download_prices(self, start_date: str = "2021-01-01"):
        """Load prices from database (already downloaded)."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Check if prices table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prices'")
        if not c.fetchone():
            print("[!] No prices table found. Run price download first.")
            conn.close()
            return 0

        c.execute("SELECT timestamp, close FROM prices ORDER BY timestamp")
        rows = c.fetchall()
        conn.close()

        # Store prices with expanded coverage for block lookups
        for ts, close in rows:
            self.price_data[ts] = close
            # Also store for neighboring hours
            for offset in range(-12, 13):
                self.price_data[ts + offset * 3600] = close

        print(f"Loaded {len(rows):,} daily price candles")
        return len(rows)

    def get_price_at_time(self, timestamp: int) -> Optional[float]:
        """Get price closest to timestamp."""
        if timestamp in self.price_data:
            return self.price_data[timestamp]

        # Find closest
        closest = None
        min_diff = float('inf')
        for ts, price in self.price_data.items():
            diff = abs(ts - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest = price

        return closest if min_diff < 86400 else None  # Within 24 hours

    def calculate_rolling_stats(self, window: int = 144):
        """Calculate rolling statistics for each block (144 blocks = ~24 hours)."""
        for i, block in enumerate(self.block_data):
            start_idx = max(0, i - window)
            window_blocks = self.block_data[start_idx:i+1]

            if len(window_blocks) >= window // 2:
                block['avg_tx_24h'] = sum(b['tx_count'] for b in window_blocks) / len(window_blocks)
                block['avg_fullness_24h'] = sum(b['fullness'] for b in window_blocks) / len(window_blocks)

                # Z-scores for transaction count and fullness
                tx_std = self._std([b['tx_count'] for b in window_blocks])
                block['tx_zscore'] = (block['tx_count'] - block['avg_tx_24h']) / tx_std if tx_std > 0 else 0

                fullness_std = self._std([b['fullness'] for b in window_blocks])
                block['fullness_zscore'] = (block['fullness'] - block['avg_fullness_24h']) / fullness_std if fullness_std > 0 else 0
            else:
                block['avg_tx_24h'] = block['tx_count']
                block['avg_fullness_24h'] = block['fullness']
                block['tx_zscore'] = 0
                block['fullness_zscore'] = 0

    def _std(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def backtest_high_tx_strategy(self, tx_threshold: float = 2.0, hold_hours: int = 24) -> BacktestResult:
        """
        Strategy: High tx count = high network activity = potential selling = SHORT
        When tx z-score > threshold, go short for hold_hours
        """
        trades = []

        for i, block in enumerate(self.block_data):
            if 'tx_zscore' not in block:
                continue

            if block['tx_zscore'] > tx_threshold:
                entry_time = block['datetime']
                exit_time = entry_time + timedelta(hours=hold_hours)

                entry_price = self.get_price_at_time(block['timestamp'])
                exit_price = self.get_price_at_time(block['timestamp'] + hold_hours * 3600)

                if entry_price and exit_price:
                    # SHORT trade (high activity = selling)
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100 - 0.1

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=-1,
                        signal_value=block['tx_zscore'],
                        pnl_pct=pnl_pct
                    ))

        return self._calculate_result(f"HIGH_TX_z{tx_threshold}_hold{hold_hours}h_SHORT", trades)

    def backtest_high_tx_long_strategy(self, tx_threshold: float = 2.0, hold_hours: int = 24) -> BacktestResult:
        """
        Strategy: High tx count = high adoption = bullish = LONG
        Alternative hypothesis: high activity is bullish
        """
        trades = []

        for i, block in enumerate(self.block_data):
            if 'tx_zscore' not in block:
                continue

            if block['tx_zscore'] > tx_threshold:
                entry_time = block['datetime']
                exit_time = entry_time + timedelta(hours=hold_hours)

                entry_price = self.get_price_at_time(block['timestamp'])
                exit_price = self.get_price_at_time(block['timestamp'] + hold_hours * 3600)

                if entry_price and exit_price:
                    # LONG trade (high activity = bullish)
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100 - 0.1

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=1,
                        signal_value=block['tx_zscore'],
                        pnl_pct=pnl_pct
                    ))

        return self._calculate_result(f"HIGH_TX_z{tx_threshold}_hold{hold_hours}h_LONG", trades)

    def backtest_low_activity_strategy(self, tx_threshold: float = -2.0, hold_hours: int = 24) -> BacktestResult:
        """
        Strategy: Low network activity = accumulation phase = LONG
        When tx z-score < threshold (very low activity), go long
        """
        trades = []

        for i, block in enumerate(self.block_data):
            if 'tx_zscore' not in block:
                continue

            if block['tx_zscore'] < tx_threshold:
                entry_time = block['datetime']
                exit_time = entry_time + timedelta(hours=hold_hours)

                entry_price = self.get_price_at_time(block['timestamp'])
                exit_price = self.get_price_at_time(block['timestamp'] + hold_hours * 3600)

                if entry_price and exit_price:
                    # LONG trade
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100 - 0.1

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=1,
                        signal_value=block['tx_zscore'],
                        pnl_pct=pnl_pct
                    ))

        return self._calculate_result(f"LOW_TX_z{tx_threshold}_hold{hold_hours}h", trades)

    def backtest_congestion_strategy(self, fullness_threshold: float = 0.95, hold_hours: int = 12) -> BacktestResult:
        """
        Strategy: Full blocks = high demand = SHORT (people rushing to sell)
        When block fullness > threshold, go short
        """
        trades = []

        for i, block in enumerate(self.block_data):
            if block['fullness'] > fullness_threshold:
                entry_time = block['datetime']
                exit_time = entry_time + timedelta(hours=hold_hours)

                entry_price = self.get_price_at_time(block['timestamp'])
                exit_price = self.get_price_at_time(block['timestamp'] + hold_hours * 3600)

                if entry_price and exit_price:
                    # SHORT trade
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100 - 0.1

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=-1,
                        signal_value=block['fullness'],
                        pnl_pct=pnl_pct
                    ))

        return self._calculate_result(f"CONGESTION_{fullness_threshold}_hold{hold_hours}h", trades)

    def backtest_tx_spike_strategy(self, spike_multiplier: float = 3.0, hold_hours: int = 6) -> BacktestResult:
        """
        Strategy: TX spike = panic selling = temporary bottom = LONG after cooldown
        When tx count is 2x+ average, wait 1 hour then go long
        """
        trades = []

        for i, block in enumerate(self.block_data):
            if 'avg_tx_24h' not in block or block['avg_tx_24h'] == 0:
                continue

            tx_ratio = block['tx_count'] / block['avg_tx_24h']

            if tx_ratio > spike_multiplier:
                # Wait 1 hour cooldown
                entry_time = block['datetime'] + timedelta(hours=1)
                exit_time = entry_time + timedelta(hours=hold_hours)

                entry_price = self.get_price_at_time(block['timestamp'] + 3600)
                exit_price = self.get_price_at_time(block['timestamp'] + 3600 + hold_hours * 3600)

                if entry_price and exit_price:
                    # LONG trade (betting on recovery)
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100 - 0.1

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=1,
                        signal_value=tx_ratio,
                        pnl_pct=pnl_pct
                    ))

        return self._calculate_result(f"TX_SPIKE_{spike_multiplier}x_hold{hold_hours}h", trades)

    def _calculate_result(self, strategy_name: str, trades: List[Trade]) -> BacktestResult:
        """Calculate backtest statistics."""
        if not trades:
            return BacktestResult(
                strategy_name=strategy_name,
                total_trades=0,
                win_rate=0,
                total_pnl_pct=0,
                avg_pnl_pct=0,
                sharpe_ratio=0,
                max_drawdown=0,
                winning_trades=0,
                losing_trades=0
            )

        winners = [t for t in trades if t.pnl_pct > 0]
        losers = [t for t in trades if t.pnl_pct <= 0]

        total_pnl = sum(t.pnl_pct for t in trades)
        avg_pnl = total_pnl / len(trades)

        # Sharpe ratio (simplified)
        returns = [t.pnl_pct for t in trades]
        std = self._std(returns)
        sharpe = (avg_pnl / std) * math.sqrt(252) if std > 0 else 0

        # Max drawdown
        cumulative = 0
        peak = 0
        max_dd = 0
        for t in trades:
            cumulative += t.pnl_pct
            peak = max(peak, cumulative)
            dd = (peak - cumulative) / (peak + 100) if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return BacktestResult(
            strategy_name=strategy_name,
            total_trades=len(trades),
            win_rate=len(winners) / len(trades),
            total_pnl_pct=total_pnl,
            avg_pnl_pct=avg_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            winning_trades=len(winners),
            losing_trades=len(losers)
        )

    def run_all_strategies(self) -> List[BacktestResult]:
        """Run all strategies and return ranked results."""
        print("\n" + "=" * 80)
        print("BLOCK FEATURES BACKTESTING")
        print("=" * 80)

        # Calculate rolling stats
        print("\nCalculating rolling statistics...")
        self.calculate_rolling_stats()

        results = []

        # Test high tx activity strategies (both long and short)
        print("\n[1/5] Testing high TX activity strategies (SHORT)...")
        for threshold in [1.5, 2.0, 2.5, 3.0]:
            for hold in [6, 12, 24, 48]:
                result = self.backtest_high_tx_strategy(threshold, hold)
                results.append(result)
                if result.total_trades >= 100:
                    print(f"  z>{threshold}, {hold}h SHORT: {result.win_rate:.1%} ({result.total_trades} trades, {result.total_pnl_pct:+.1f}% PnL)")

        print("\n[2/5] Testing high TX activity strategies (LONG)...")
        for threshold in [1.5, 2.0, 2.5, 3.0]:
            for hold in [6, 12, 24, 48]:
                result = self.backtest_high_tx_long_strategy(threshold, hold)
                results.append(result)
                if result.total_trades >= 100:
                    print(f"  z>{threshold}, {hold}h LONG: {result.win_rate:.1%} ({result.total_trades} trades, {result.total_pnl_pct:+.1f}% PnL)")

        print("\n[3/5] Testing low activity strategies (accumulation = LONG)...")
        for threshold in [-1.5, -2.0, -2.5]:
            for hold in [12, 24, 48]:
                result = self.backtest_low_activity_strategy(threshold, hold)
                results.append(result)
                if result.total_trades >= 100:
                    print(f"  z<{threshold}, {hold}h: {result.win_rate:.1%} ({result.total_trades} trades, {result.total_pnl_pct:+.1f}% PnL)")

        print("\n[4/5] Testing congestion strategies (full blocks = selling = SHORT)...")
        for fullness in [0.90, 0.95, 0.98]:
            for hold in [6, 12, 24]:
                result = self.backtest_congestion_strategy(fullness, hold)
                results.append(result)
                if result.total_trades >= 100:
                    print(f"  fullness>{fullness}, {hold}h: {result.win_rate:.1%} ({result.total_trades} trades, {result.total_pnl_pct:+.1f}% PnL)")

        print("\n[5/5] Testing tx spike recovery strategies (panic selling = LONG after cooldown)...")
        for multiplier in [2.0, 3.0, 5.0]:
            for hold in [6, 12, 24]:
                result = self.backtest_tx_spike_strategy(multiplier, hold)
                results.append(result)
                if result.total_trades >= 50:
                    print(f"  {multiplier}x spike, {hold}h: {result.win_rate:.1%} ({result.total_trades} trades, {result.total_pnl_pct:+.1f}% PnL)")

        # Rank by total PnL
        results = sorted(results, key=lambda r: r.total_pnl_pct, reverse=True)

        print("\n" + "=" * 80)
        print("TOP 10 STRATEGIES BY TOTAL PNL")
        print("=" * 80)
        print(f"\n{'Strategy':<45} {'Trades':<8} {'Win%':<8} {'PnL%':<10} {'Sharpe':<8} {'MaxDD':<8}")
        print("-" * 95)

        for result in results[:10]:
            if result.total_trades > 0:
                print(f"{result.strategy_name:<45} {result.total_trades:<8} {result.win_rate:.1%}    {result.total_pnl_pct:>+8.1f}%  {result.sharpe_ratio:>6.2f}   {result.max_drawdown:>6.1%}")

        return results


def main():
    """Run block features backtesting."""
    backtester = BlockFeaturesBacktester()

    # Load data
    blocks = backtester.load_data()
    if blocks == 0:
        print("[!] No block data found")
        return

    # Download prices
    backtester.download_prices("2021-01-01")

    if len(backtester.price_data) == 0:
        print("[!] No price data available")
        return

    # Run all strategies
    results = backtester.run_all_strategies()

    # Save results
    output = []
    for r in results:
        output.append({
            "strategy": r.strategy_name,
            "total_trades": r.total_trades,
            "win_rate": r.win_rate,
            "total_pnl_pct": r.total_pnl_pct,
            "sharpe_ratio": r.sharpe_ratio,
            "max_drawdown": r.max_drawdown
        })

    with open("data/block_backtest_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[+] Results saved to data/block_backtest_results.json")


if __name__ == "__main__":
    main()
