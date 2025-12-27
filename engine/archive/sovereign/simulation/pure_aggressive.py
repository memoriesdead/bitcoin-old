#!/usr/bin/env python3
"""
PURE AGGRESSIVE TRADING ENGINE
===============================
RenTech-style explosive trading: Many longs AND shorts, pure compounding.
No extraction rules. Maximum theoretical daily growth.

Goal: $100 to $1000 as fast as mathematically possible with proven edge.
"""

import os
import sys
import json
import sqlite3
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "simulation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class Direction(Enum):
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    direction: Direction
    size_usd: float
    leverage: float
    tp_price: float
    sl_price: float
    signal_type: str
    confidence: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    result: Optional[str] = None


@dataclass
class EngineConfig:
    """Pure aggression config - no safety rails."""
    kelly_fraction: float = 2.0      # Super-Kelly for max growth
    leverage: float = 25.0           # Maximum leverage
    tp_pct: float = 0.015            # 1.5% take profit
    sl_pct: float = 0.005            # 0.5% stop loss (3:1 R:R)
    min_confidence: float = 0.52     # Trade anything with edge
    max_trades_per_day: int = 50     # Many trades
    taker_fee: float = 0.00035       # Hyperliquid fee
    slippage_bps: float = 2          # Tight slippage
    compound_all: bool = True        # Pure compounding, no extraction


class PureAggressiveEngine:
    """
    Maximum aggression trading engine.

    Key differences from 100% rule:
    - NO extraction ever - compound everything
    - Trade BOTH directions aggressively
    - Many trades per day (not waiting for perfect setups)
    - Super-Kelly sizing for explosive growth
    """

    def __init__(self, config: EngineConfig = None, initial_capital: float = 100.0):
        self.config = config or EngineConfig()
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_trades: Dict[str, int] = {}

    def calculate_position_size(self, confidence: float) -> float:
        """Super-Kelly position sizing."""
        # Kelly formula: f = (p * b - q) / b
        # Where p = win prob, q = 1-p, b = win/loss ratio
        p = confidence
        q = 1 - p
        b = self.config.tp_pct / self.config.sl_pct

        kelly = max(0, (p * b - q) / b)

        # Apply super-Kelly multiplier
        kelly *= self.config.kelly_fraction

        # Cap at 50% of equity per trade
        kelly = min(kelly, 0.5)

        return self.equity * kelly

    def generate_signals(self, data: List[Dict], idx: int) -> List[Dict]:
        """
        Generate MANY signals - both long and short.
        We want explosive trading, not conservative waiting.
        """
        if idx < 10:
            return []

        signals = []
        current = data[idx]
        window_5 = data[idx-5:idx]
        window_10 = data[idx-10:idx]

        price = current['close']

        # === LONG SIGNALS ===

        # 1. Whale accumulation (receivers > senders)
        if current.get('unique_receivers', 0) > current.get('unique_senders', 0) * 1.2:
            ratio = current['unique_receivers'] / max(current['unique_senders'], 1)
            conf = min(0.65, 0.52 + (ratio - 1.2) * 0.1)
            signals.append({
                'direction': Direction.LONG,
                'confidence': conf,
                'type': 'WHALE_ACCUM',
                'price': price
            })

        # 2. TX surge (bullish momentum)
        avg_tx = sum(d.get('tx_count', 0) for d in window_5) / 5
        if avg_tx > 0 and current.get('tx_count', 0) > avg_tx * 1.3:
            ratio = current['tx_count'] / avg_tx
            conf = min(0.62, 0.53 + (ratio - 1.3) * 0.08)
            signals.append({
                'direction': Direction.LONG,
                'confidence': conf,
                'type': 'TX_SURGE_LONG',
                'price': price
            })

        # 3. Value spike (big money moving in)
        avg_val = sum(d.get('total_value_btc', 0) for d in window_5) / 5
        if avg_val > 0 and current.get('total_value_btc', 0) > avg_val * 1.5:
            ratio = current['total_value_btc'] / avg_val
            conf = min(0.68, 0.55 + (ratio - 1.5) * 0.1)
            signals.append({
                'direction': Direction.LONG,
                'confidence': conf,
                'type': 'VALUE_SPIKE_LONG',
                'price': price
            })

        # 4. Price momentum up
        closes_5 = [d['close'] for d in window_5]
        ma_5 = sum(closes_5) / 5
        closes_10 = [d['close'] for d in window_10]
        ma_10 = sum(closes_10) / 10

        if price > ma_5 * 1.01 and ma_5 > ma_10:
            strength = (price / ma_5 - 1) * 100
            conf = min(0.60, 0.53 + strength * 0.02)
            signals.append({
                'direction': Direction.LONG,
                'confidence': conf,
                'type': 'MOM_LONG',
                'price': price
            })

        # 5. Breakout (new high in window)
        highs = [d['high'] for d in window_10]
        if price > max(highs) * 0.998:
            conf = 0.58
            signals.append({
                'direction': Direction.LONG,
                'confidence': conf,
                'type': 'BREAKOUT_LONG',
                'price': price
            })

        # === SHORT SIGNALS ===

        # 1. Whale distribution (senders > receivers)
        if current.get('unique_senders', 0) > current.get('unique_receivers', 0) * 1.2:
            ratio = current['unique_senders'] / max(current['unique_receivers'], 1)
            conf = min(0.63, 0.52 + (ratio - 1.2) * 0.08)
            signals.append({
                'direction': Direction.SHORT,
                'confidence': conf,
                'type': 'WHALE_DISTRIB',
                'price': price
            })

        # 2. TX drop (bearish momentum)
        if avg_tx > 0 and current.get('tx_count', 0) < avg_tx * 0.7:
            ratio = avg_tx / max(current['tx_count'], 1)
            conf = min(0.58, 0.52 + (ratio - 1.4) * 0.05)
            signals.append({
                'direction': Direction.SHORT,
                'confidence': conf,
                'type': 'TX_DROP_SHORT',
                'price': price
            })

        # 3. Value drain (money leaving)
        if avg_val > 0 and current.get('total_value_btc', 0) < avg_val * 0.6:
            conf = 0.57
            signals.append({
                'direction': Direction.SHORT,
                'confidence': conf,
                'type': 'VALUE_DRAIN_SHORT',
                'price': price
            })

        # 4. Price momentum down
        if price < ma_5 * 0.99 and ma_5 < ma_10:
            strength = (1 - price / ma_5) * 100
            conf = min(0.58, 0.52 + strength * 0.02)
            signals.append({
                'direction': Direction.SHORT,
                'confidence': conf,
                'type': 'MOM_SHORT',
                'price': price
            })

        # 5. Breakdown (new low in window)
        lows = [d['low'] for d in window_10]
        if price < min(lows) * 1.002:
            conf = 0.56
            signals.append({
                'direction': Direction.SHORT,
                'confidence': conf,
                'type': 'BREAKDOWN_SHORT',
                'price': price
            })

        # 6. Exhaustion top (high whale + value but price stalling)
        if (current.get('whale_tx_count', 0) > sum(d.get('whale_tx_count', 0) for d in window_5) / 5 * 1.5
            and price < max(d['high'] for d in window_5)):
            conf = 0.55
            signals.append({
                'direction': Direction.SHORT,
                'confidence': conf,
                'type': 'EXHAUSTION_SHORT',
                'price': price
            })

        return signals

    def execute_trade(self, signal: Dict, current_time: datetime) -> Optional[Trade]:
        """Execute a trade based on signal."""
        if signal['confidence'] < self.config.min_confidence:
            return None

        # Check daily trade limit
        date_key = current_time.strftime('%Y-%m-%d')
        if self.daily_trades.get(date_key, 0) >= self.config.max_trades_per_day:
            return None

        # Calculate position size
        size = self.calculate_position_size(signal['confidence'])
        if size < 1:  # Min $1 trade
            return None

        # Entry price with slippage
        entry = signal['price']
        if signal['direction'] == Direction.LONG:
            entry *= (1 + self.config.slippage_bps / 10000)
        else:
            entry *= (1 - self.config.slippage_bps / 10000)

        # TP/SL prices
        if signal['direction'] == Direction.LONG:
            tp = entry * (1 + self.config.tp_pct)
            sl = entry * (1 - self.config.sl_pct)
        else:
            tp = entry * (1 - self.config.tp_pct)
            sl = entry * (1 + self.config.sl_pct)

        trade = Trade(
            entry_time=current_time,
            entry_price=entry,
            direction=signal['direction'],
            size_usd=size,
            leverage=self.config.leverage,
            tp_price=tp,
            sl_price=sl,
            signal_type=signal['type'],
            confidence=signal['confidence']
        )

        self.daily_trades[date_key] = self.daily_trades.get(date_key, 0) + 1
        return trade

    def check_exit(self, trade: Trade, candle: Dict, current_time: datetime) -> bool:
        """Check if trade should exit."""
        high = candle['high']
        low = candle['low']

        if trade.direction == Direction.LONG:
            if high >= trade.tp_price:
                trade.exit_price = trade.tp_price
                trade.result = 'WIN'
            elif low <= trade.sl_price:
                trade.exit_price = trade.sl_price
                trade.result = 'LOSS'
        else:
            if low <= trade.tp_price:
                trade.exit_price = trade.tp_price
                trade.result = 'WIN'
            elif high >= trade.sl_price:
                trade.exit_price = trade.sl_price
                trade.result = 'LOSS'

        if trade.exit_price:
            trade.exit_time = current_time

            # Calculate PnL
            if trade.direction == Direction.LONG:
                pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
            else:
                pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price

            trade.pnl = trade.size_usd * pnl_pct * trade.leverage

            # Subtract fees
            fees = trade.size_usd * self.config.taker_fee * 2
            trade.pnl -= fees

            # Update equity
            self.equity += trade.pnl

            if self.equity > self.peak_equity:
                self.peak_equity = self.equity

            return True

        return False

    def run_backtest(self, data: List[Dict]) -> Dict:
        """Run full backtest on historical data."""
        print(f"\n{'='*70}")
        print("PURE AGGRESSIVE ENGINE - MAXIMUM THEORETICAL GROWTH")
        print(f"{'='*70}")
        print(f"\nConfig:")
        print(f"  Kelly Fraction: {self.config.kelly_fraction}x (Super-Kelly)")
        print(f"  Leverage: {self.config.leverage}x")
        print(f"  TP/SL: {self.config.tp_pct*100:.1f}% / {self.config.sl_pct*100:.1f}%")
        print(f"  R:R Ratio: {self.config.tp_pct/self.config.sl_pct:.1f}:1")
        print(f"  Max Trades/Day: {self.config.max_trades_per_day}")
        print(f"  Initial Capital: ${self.initial_capital}")

        active_trades: List[Trade] = []
        wins = 0
        losses = 0

        for i in range(10, len(data) - 1):
            current = data[i]
            next_candle = data[i + 1]

            ts = current.get('timestamp', 0)
            current_time = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)

            # Check exits
            for trade in active_trades[:]:
                if self.check_exit(trade, current, current_time):
                    active_trades.remove(trade)
                    self.trades.append(trade)
                    if trade.result == 'WIN':
                        wins += 1
                    else:
                        losses += 1

            # Record equity
            self.equity_curve.append((current_time, self.equity))

            # Check for bust
            if self.equity <= 0:
                print(f"\n*** BUSTED at index {i} ***")
                break

            # Generate new signals
            signals = self.generate_signals(data, i)

            # Execute trades on best signals (can have multiple open)
            for signal in sorted(signals, key=lambda s: s['confidence'], reverse=True)[:3]:
                if len(active_trades) < 5:  # Max 5 concurrent trades
                    trade = self.execute_trade(signal, current_time)
                    if trade:
                        active_trades.append(trade)

            # Progress
            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(data)} | Equity: ${self.equity:.2f} | Trades: {len(self.trades)}")

        # Close any remaining trades at last price
        last_candle = data[-1]
        for trade in active_trades:
            trade.exit_price = last_candle['close']
            trade.exit_time = datetime.now(timezone.utc)
            if trade.direction == Direction.LONG:
                pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
            else:
                pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price
            trade.pnl = trade.size_usd * pnl_pct * trade.leverage
            trade.pnl -= trade.size_usd * self.config.taker_fee * 2
            trade.result = 'WIN' if trade.pnl > 0 else 'LOSS'
            self.equity += trade.pnl
            self.trades.append(trade)
            if trade.result == 'WIN':
                wins += 1
            else:
                losses += 1

        # Calculate metrics
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0

        total_return = (self.equity / self.initial_capital) - 1

        # Calculate days
        if self.equity_curve:
            start_time = self.equity_curve[0][0]
            end_time = self.equity_curve[-1][0]
            days = max(1, (end_time - start_time).days)
        else:
            days = 1

        daily_roi = (self.equity / self.initial_capital) ** (1 / days) - 1 if days > 0 else 0

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0
        for _, eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        # Days to 10x
        if daily_roi > 0:
            days_to_10x = math.log(10) / math.log(1 + daily_roi)
        else:
            days_to_10x = float('inf')

        # Trades per day
        trades_per_day = total_trades / days if days > 0 else 0

        results = {
            'initial_capital': self.initial_capital,
            'final_equity': self.equity,
            'total_return_pct': total_return * 100,
            'daily_roi_pct': daily_roi * 100,
            'days_to_10x': days_to_10x,
            'total_trades': total_trades,
            'trades_per_day': trades_per_day,
            'wins': wins,
            'losses': losses,
            'win_rate_pct': win_rate * 100,
            'max_drawdown_pct': max_dd * 100,
            'days_tested': days,
            'config': {
                'kelly_fraction': self.config.kelly_fraction,
                'leverage': self.config.leverage,
                'tp_pct': self.config.tp_pct,
                'sl_pct': self.config.sl_pct,
                'min_confidence': self.config.min_confidence,
                'max_trades_per_day': self.config.max_trades_per_day,
            }
        }

        # Print results
        print(f"\n{'='*70}")
        print("RESULTS - PURE AGGRESSIVE TRADING")
        print(f"{'='*70}")
        print(f"\nPerformance:")
        print(f"  Final Equity:    ${self.equity:,.2f}")
        print(f"  Total Return:    {total_return*100:+.1f}%")
        print(f"  Daily ROI:       {daily_roi*100:.2f}%")
        print(f"  Days to 10x:     {days_to_10x:.1f} days")
        print(f"\nTrading Stats:")
        print(f"  Total Trades:    {total_trades}")
        print(f"  Trades/Day:      {trades_per_day:.1f}")
        print(f"  Win Rate:        {win_rate*100:.1f}%")
        print(f"  Max Drawdown:    {max_dd*100:.1f}%")
        print(f"  Test Period:     {days} days")

        # Projection
        print(f"\n{'='*70}")
        print("PROJECTION: $100 to $1000")
        print(f"{'='*70}")

        if daily_roi > 0:
            proj_equity = 100
            print(f"\n{'Day':<8} {'Equity':<15} {'Trades Est.'}")
            print("-" * 40)
            for day in range(1, 366):
                proj_equity *= (1 + daily_roi)
                trades_est = int(day * trades_per_day)
                if day in [1, 3, 7, 14, 30, 60] or proj_equity >= 1000:
                    print(f"{day:<8} ${proj_equity:>12,.2f}   ~{trades_est} trades")
                    if proj_equity >= 1000:
                        print(f"\n>>> $100 to $1000 in {day} DAYS with {trades_est} trades <<<")
                        break

        return results


def load_historical_data() -> List[Dict]:
    """Load all available historical data."""
    features = {}
    prices = {}

    # Load features
    for path in [DATA_DIR / "bitcoin_features.db", DATA_DIR / "historical_flows.db"]:
        if path.exists():
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cursor.fetchall()]

            if 'daily_features' in tables:
                cursor.execute("""
                    SELECT timestamp, tx_count, total_value_btc, whale_tx_count,
                           unique_senders, unique_receivers
                    FROM daily_features WHERE tx_count IS NOT NULL
                """)
                for row in cursor.fetchall():
                    ts = row[0]
                    date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
                    features[date_str] = {
                        'timestamp': ts,
                        'tx_count': row[1] or 0,
                        'total_value_btc': row[2] or 0,
                        'whale_tx_count': row[3] or 0,
                        'unique_senders': row[4] or 0,
                        'unique_receivers': row[5] or 0,
                    }
            conn.close()
            if features:
                break

    # Load prices
    for path in [DATA_DIR / "bitcoin_2021_2025.db", DATA_DIR / "historical_flows.db"]:
        if path.exists():
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cursor.fetchall()]

            for table in ['prices', 'ohlcv']:
                if table in tables:
                    cursor.execute(f"SELECT timestamp, open, high, low, close FROM {table}")
                    for row in cursor.fetchall():
                        ts = row[0]
                        date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
                        prices[date_str] = {
                            'timestamp': ts,
                            'open': row[1],
                            'high': row[2],
                            'low': row[3],
                            'close': row[4],
                        }
                    break
            conn.close()
            if prices:
                break

    # Merge data
    data = []
    for date_str in sorted(features.keys()):
        if date_str in prices:
            data.append({'date': date_str, **features[date_str], **prices[date_str]})

    print(f"Loaded {len(data)} data points with features and prices")
    return data


def run_parameter_sweep():
    """Find the absolute maximum growth configuration."""
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP - FINDING MAXIMUM THEORETICAL GROWTH")
    print("=" * 70)

    data = load_historical_data()
    if len(data) < 30:
        print("Not enough data for meaningful test")
        return

    # Test grid - aggressive configurations only
    configs = []
    for kelly in [1.5, 2.0, 2.5, 3.0]:
        for leverage in [20, 25, 30]:
            for tp in [0.01, 0.015, 0.02, 0.025]:
                for sl in [0.003, 0.005, 0.007]:
                    if tp / sl >= 2:  # Minimum 2:1 R:R
                        configs.append(EngineConfig(
                            kelly_fraction=kelly,
                            leverage=leverage,
                            tp_pct=tp,
                            sl_pct=sl,
                            min_confidence=0.52,
                            max_trades_per_day=50,
                        ))

    print(f"Testing {len(configs)} configurations...")

    results = []
    for i, config in enumerate(configs):
        engine = PureAggressiveEngine(config=config, initial_capital=100)
        # Use subset for speed
        test_data = data[-120:]  # Last 120 days

        # Quiet run
        active_trades = []
        wins = 0
        losses = 0

        for j in range(10, len(test_data) - 1):
            current = test_data[j]
            ts = current.get('timestamp', 0)
            current_time = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)

            for trade in active_trades[:]:
                if engine.check_exit(trade, current, current_time):
                    active_trades.remove(trade)
                    if trade.result == 'WIN':
                        wins += 1
                    else:
                        losses += 1

            if engine.equity <= 0:
                break

            signals = engine.generate_signals(test_data, j)
            for signal in sorted(signals, key=lambda s: s['confidence'], reverse=True)[:3]:
                if len(active_trades) < 5:
                    trade = engine.execute_trade(signal, current_time)
                    if trade:
                        active_trades.append(trade)

        total_trades = wins + losses
        if total_trades > 0 and engine.equity > 0:
            days = 120
            daily_roi = (engine.equity / 100) ** (1 / days) - 1
            days_to_10x = math.log(10) / math.log(1 + daily_roi) if daily_roi > 0 else 9999

            results.append({
                'config': config,
                'final_equity': engine.equity,
                'daily_roi': daily_roi * 100,
                'days_to_10x': days_to_10x,
                'total_trades': total_trades,
                'win_rate': wins / total_trades * 100,
            })

        if (i + 1) % 50 == 0:
            print(f"  Tested {i+1}/{len(configs)}")

    # Sort by daily ROI
    results.sort(key=lambda x: x['daily_roi'], reverse=True)

    print(f"\n{'='*70}")
    print("TOP 10 CONFIGURATIONS BY DAILY ROI")
    print(f"{'='*70}")

    print(f"\n{'Rank':<5} {'Daily ROI':<12} {'Days-10x':<10} {'WinRate':<10} {'Trades':<8} {'Config'}")
    print("-" * 100)

    for i, r in enumerate(results[:10]):
        c = r['config']
        config_str = f"K={c.kelly_fraction} L={c.leverage}x TP={c.tp_pct*100:.1f}% SL={c.sl_pct*100:.1f}%"
        print(f"{i+1:<5} {r['daily_roi']:>8.2f}%    {r['days_to_10x']:>8.1f}d   {r['win_rate']:>7.1f}%   {r['total_trades']:<8} {config_str}")

    if results:
        best = results[0]
        print(f"\n{'='*70}")
        print("OPTIMAL CONFIGURATION FOR MAXIMUM GROWTH")
        print(f"{'='*70}")
        print(f"\n  Kelly Fraction:  {best['config'].kelly_fraction}x")
        print(f"  Leverage:        {best['config'].leverage}x")
        print(f"  Take Profit:     {best['config'].tp_pct*100:.1f}%")
        print(f"  Stop Loss:       {best['config'].sl_pct*100:.1f}%")
        print(f"  R:R Ratio:       {best['config'].tp_pct/best['config'].sl_pct:.1f}:1")
        print(f"\n  Daily ROI:       {best['daily_roi']:.2f}%")
        print(f"  Days to 10x:     {best['days_to_10x']:.1f}")
        print(f"  Win Rate:        {best['win_rate']:.1f}%")

        return best['config']

    return None


def main():
    """Run full analysis."""
    # First find optimal config
    print("\n[1] Running parameter sweep...")
    optimal_config = run_parameter_sweep()

    if not optimal_config:
        optimal_config = EngineConfig()  # Use defaults

    # Run full backtest with optimal config
    print("\n\n[2] Running full backtest with optimal config...")
    data = load_historical_data()

    engine = PureAggressiveEngine(config=optimal_config, initial_capital=100)
    results = engine.run_backtest(data)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f"pure_aggressive_{timestamp}.json"

    # Add trade details
    results['trades'] = [
        {
            'entry_time': t.entry_time.isoformat() if t.entry_time else None,
            'entry_price': t.entry_price,
            'direction': t.direction.name,
            'size_usd': t.size_usd,
            'tp_price': t.tp_price,
            'sl_price': t.sl_price,
            'exit_price': t.exit_price,
            'pnl': t.pnl,
            'result': t.result,
            'signal_type': t.signal_type,
        }
        for t in engine.trades[:100]  # First 100 trades
    ]

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[SAVED] {results_file}")

    return results


if __name__ == '__main__':
    main()
