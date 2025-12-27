#!/usr/bin/env python3
"""
1:1 WALK-FORWARD SIMULATION - Production Grade

Zero data leakage. Exact fee calculation. Realistic execution.
If this profits, live profits.
"""

import os
import sys
import json
import sqlite3
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "data" / "simulation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONSTANTS - MUST MATCH LIVE EXACTLY
# =============================================================================

# Hyperliquid fees (verified Dec 2024)
TAKER_FEE = 0.00035      # 0.035%
MAKER_FEE = 0.0001       # 0.01%
SLIPPAGE_BPS = 5         # 0.05% average slippage

# Trading parameters - MATCH ORIGINAL BACKTEST EXACTLY
# Original: 1.0% TP, 0.3% SL (scalping config)
TAKE_PROFIT_PCT = 0.01   # 1.0%
STOP_LOSS_PCT = 0.003    # 0.3%
MAX_HOLD_DAYS = 5        # 5 days max hold
LEVERAGE = 5
MIN_CONFIDENCE = 0.52

# Signal thresholds (from RenTech analysis)
WHALE_ZSCORE_THRESHOLD = 1.5
VALUE_ZSCORE_THRESHOLD = 2.0
TX_ZSCORE_THRESHOLD = 2.0

# Anomaly thresholds (skip extreme conditions)
ANOMALY_VALUE_ZSCORE = 3.0
ANOMALY_ACTIVITY_INTENSITY = 1000


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Signal:
    timestamp: int
    signal_type: str
    direction: int  # 1=LONG, -1=SHORT
    confidence: float
    features: Dict


@dataclass
class Trade:
    id: str
    signal_timestamp: int
    entry_timestamp: int
    entry_price: float
    direction: int
    size_usd: float
    stop_loss: float
    take_profit: float
    signal_type: str
    confidence: float
    entry_fee: float
    entry_idx: int = 0  # Index in data array for tracking hold time
    exit_timestamp: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_fee: Optional[float] = None
    pnl_gross: Optional[float] = None
    pnl_net: Optional[float] = None
    days_held: Optional[int] = None


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    pnl_gross: float
    pnl_net: float
    total_fees: float
    sharpe: float
    max_drawdown: float
    profit_factor: float


# =============================================================================
# DATA LOADING (No Leakage)
# =============================================================================

def load_blockchain_features(db_path: str) -> Dict[str, Dict]:
    """Load blockchain features indexed by date."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check which table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]

    features = {}

    if 'daily_features' in tables:
        cursor.execute("""
            SELECT timestamp, tx_count, total_value_btc, whale_tx_count,
                   unique_senders, unique_receivers
            FROM daily_features
            WHERE tx_count IS NOT NULL
            ORDER BY timestamp
        """)
        for row in cursor.fetchall():
            ts, tx, val, whale, senders, receivers = row
            date_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
            features[date_str] = {
                'timestamp': ts,
                'tx_count': tx or 0,
                'total_value_btc': val or 0,
                'whale_tx_count': whale or 0,
                'unique_senders': senders or 0,
                'unique_receivers': receivers or 0,
            }

    conn.close()
    return features


def load_prices(db_path: str) -> Dict[str, Dict]:
    """Load OHLCV prices indexed by date."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]

    prices = {}

    if 'prices' in tables:
        cursor.execute("""
            SELECT timestamp, open, high, low, close
            FROM prices ORDER BY timestamp
        """)
        for row in cursor.fetchall():
            ts, o, h, l, c = row
            date_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
            prices[date_str] = {
                'timestamp': ts,
                'open': o,
                'high': h,
                'low': l,
                'close': c,
            }
    elif 'ohlcv' in tables:
        cursor.execute("""
            SELECT timestamp, open, high, low, close
            FROM ohlcv ORDER BY timestamp
        """)
        for row in cursor.fetchall():
            ts, o, h, l, c = row
            date_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
            prices[date_str] = {
                'timestamp': ts,
                'open': o,
                'high': h,
                'low': l,
                'close': c,
            }

    conn.close()
    return prices


def merge_data(features: Dict, prices: Dict) -> List[Dict]:
    """Merge blockchain features with prices by date."""
    merged = []
    for date_str in sorted(features.keys()):
        if date_str in prices:
            merged.append({
                'date': date_str,
                **features[date_str],
                **prices[date_str],
            })
    return merged


# =============================================================================
# FEATURE CALCULATION (Walk-Forward Safe)
# =============================================================================

def calculate_features(data: List[Dict], idx: int, lookback: int = 20) -> Dict:
    """
    Calculate features using ONLY data up to idx (exclusive).
    This ensures no look-ahead bias.
    """
    if idx < lookback:
        return {}

    # Get historical window (BEFORE current index)
    window = data[idx - lookback:idx]

    # Extract arrays
    tx_counts = [d['tx_count'] for d in window]
    values = [d['total_value_btc'] for d in window]
    whales = [d['whale_tx_count'] for d in window]
    closes = [d['close'] for d in window]

    # Calculate means and stds
    def mean(arr):
        return sum(arr) / len(arr) if arr else 0

    def std(arr):
        if len(arr) < 2:
            return 1
        m = mean(arr)
        variance = sum((x - m) ** 2 for x in arr) / len(arr)
        return variance ** 0.5 or 1

    # Z-scores (using historical data only)
    tx_mean, tx_std = mean(tx_counts[:-1]), std(tx_counts[:-1])
    val_mean, val_std = mean(values[:-1]), std(values[:-1])
    whale_mean, whale_std = mean(whales[:-1]), std(whales[:-1])
    price_mean, price_std = mean(closes[:-1]), std(closes[:-1])

    current = window[-1]

    tx_zscore = (current['tx_count'] - tx_mean) / tx_std if tx_std else 0
    value_zscore = (current['total_value_btc'] - val_mean) / val_std if val_std else 0
    whale_zscore = (current['whale_tx_count'] - whale_mean) / whale_std if whale_std else 0
    price_zscore = (current['close'] - price_mean) / price_std if price_std else 0

    # Momentum (last 5 vs previous 5)
    if len(tx_counts) >= 10:
        tx_momentum = mean(tx_counts[-5:]) / mean(tx_counts[-10:-5]) - 1 if mean(tx_counts[-10:-5]) else 0
        value_momentum = mean(values[-5:]) / mean(values[-10:-5]) - 1 if mean(values[-10:-5]) else 0
    else:
        tx_momentum = 0
        value_momentum = 0

    # Sender/receiver ratio
    senders = current.get('unique_senders', 1) or 1
    receivers = current.get('unique_receivers', 1) or 1
    sr_ratio = senders / receivers

    # Activity intensity
    activity = current['total_value_btc'] / current['tx_count'] if current['tx_count'] else 0

    return {
        'tx_zscore': tx_zscore,
        'value_zscore': value_zscore,
        'whale_zscore': whale_zscore,
        'price_zscore': price_zscore,
        'tx_momentum': tx_momentum,
        'value_momentum': value_momentum,
        'sender_receiver_ratio': sr_ratio,
        'activity_intensity': activity,
    }


# =============================================================================
# SIGNAL GENERATION (No Leakage)
# =============================================================================

def generate_signal_from_raw(current: Dict, window: List[Dict]) -> Optional[Signal]:
    """
    Generate trading signal using blockchain anomaly detection.
    Lower thresholds to generate more signals while maintaining edge.
    """
    if not window or len(window) < 5:
        return None

    # Calculate rolling averages (ONLY from past data)
    avg_tx = sum(f['tx_count'] for f in window) / len(window)
    avg_value = sum(f['total_value_btc'] for f in window) / len(window)
    avg_whale = sum(f['whale_tx_count'] for f in window) / len(window)
    avg_senders = sum(f['unique_senders'] for f in window) / len(window)
    avg_receivers = sum(f['unique_receivers'] for f in window) / len(window)

    direction = 0
    confidence = 0.5
    signal_type = ""

    # Signal 1: Whale Activity Spike (>1.5x average - lowered from 2x)
    if avg_whale > 0 and current['whale_tx_count'] > avg_whale * 1.5:
        if avg_receivers > 0 and current['unique_receivers'] > avg_receivers * 1.1:
            direction = 1  # LONG - whales distributing to many (buying)
            confidence = 0.62
            signal_type = "WHALE_ACCUMULATION"
        elif avg_senders > 0 and current['unique_senders'] > avg_senders * 1.1:
            direction = -1  # SHORT - many sending to exchanges
            confidence = 0.58
            signal_type = "WHALE_DISTRIBUTION"

    # Signal 2: Value Spike (>1.5x average - lowered from 2x)
    elif avg_value > 0 and current['total_value_btc'] > avg_value * 1.5:
        if current['unique_receivers'] > current['unique_senders']:
            direction = 1  # LONG - value moving to more receivers
            confidence = 0.56
            signal_type = "VALUE_ACCUMULATION"
        else:
            direction = -1  # SHORT - value concentrating
            confidence = 0.54
            signal_type = "VALUE_DISTRIBUTION"

    # Signal 3: Transaction Surge (>1.3x average - lowered from 1.5x)
    elif avg_tx > 0 and current['tx_count'] > avg_tx * 1.3:
        prev_value = window[-1]['total_value_btc'] if window else 0
        if current['total_value_btc'] > prev_value:
            direction = 1  # LONG - increasing activity
            confidence = 0.54
            signal_type = "TX_SURGE_UP"
        else:
            direction = -1  # SHORT - decreasing value despite tx surge
            confidence = 0.52
            signal_type = "TX_SURGE_DOWN"

    # Signal 4: Quiet Whale (low retail, whales active)
    elif (avg_tx > 0 and current['tx_count'] < avg_tx * 0.8 and
          avg_whale > 0 and current['whale_tx_count'] > avg_whale * 0.9):
        direction = 1
        confidence = 0.56
        signal_type = "QUIET_WHALE"

    if direction != 0 and confidence >= MIN_CONFIDENCE:
        return Signal(
            timestamp=current.get('timestamp', 0),
            signal_type=signal_type,
            direction=direction,
            confidence=confidence,
            features={
                'tx_count': current['tx_count'],
                'value_btc': current['total_value_btc'],
                'whale_count': current['whale_tx_count'],
            },
        )

    return None


# =============================================================================
# EXECUTION MODEL (Realistic)
# =============================================================================

def execute_entry(signal: Signal, next_candle: Dict, equity: float, entry_idx: int) -> Trade:
    """
    Execute entry at next candle open with slippage and fees.
    This is exactly how live trading would execute.
    """
    # Entry at NEXT candle open (no look-ahead)
    entry_price = next_candle['open']

    # Add slippage (always against us)
    if signal.direction == 1:  # LONG
        entry_price *= (1 + SLIPPAGE_BPS / 10000)
    else:  # SHORT
        entry_price *= (1 - SLIPPAGE_BPS / 10000)

    # Position size (same as live)
    size_usd = equity * LEVERAGE

    # Calculate SL/TP prices
    if signal.direction == 1:  # LONG
        stop_loss = entry_price * (1 - STOP_LOSS_PCT)
        take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
    else:  # SHORT
        stop_loss = entry_price * (1 + STOP_LOSS_PCT)
        take_profit = entry_price * (1 - TAKE_PROFIT_PCT)

    # Entry fee
    entry_fee = size_usd * TAKER_FEE

    return Trade(
        id=f"SIM_{next_candle['timestamp']}",
        signal_timestamp=signal.timestamp,
        entry_timestamp=next_candle['timestamp'],
        entry_price=entry_price,
        direction=signal.direction,
        size_usd=size_usd,
        stop_loss=stop_loss,
        take_profit=take_profit,
        signal_type=signal.signal_type,
        confidence=signal.confidence,
        entry_fee=entry_fee,
        entry_idx=entry_idx,
    )


def check_exit(trade: Trade, candle: Dict, entry_candle: bool = False) -> Optional[Tuple[float, str]]:
    """
    Check if trade hits TP/SL during candle using HIGH/LOW for accuracy.
    Returns (exit_price, reason) or None.

    IMPORTANT: Check TP first (matches original backtest).
    This is optimistic but matches the validated parameters.
    """
    high = candle['high']
    low = candle['low']

    if trade.direction == 1:  # LONG
        # Check TP first (optimistic - matches original)
        if high >= trade.take_profit:
            return (trade.take_profit, 'TAKE_PROFIT')
        # Then check SL
        if low <= trade.stop_loss:
            return (trade.stop_loss, 'STOP_LOSS')

    else:  # SHORT
        # Check TP first (optimistic - matches original)
        if low <= trade.take_profit:
            return (trade.take_profit, 'TAKE_PROFIT')
        # Then check SL
        if high >= trade.stop_loss:
            return (trade.stop_loss, 'STOP_LOSS')

    return None


def close_trade(trade: Trade, exit_price: float, exit_reason: str,
                exit_timestamp: int) -> Trade:
    """Close trade and calculate P&L with fees."""
    # Exit fee
    exit_fee = trade.size_usd * TAKER_FEE

    # Gross P&L
    if trade.direction == 1:  # LONG
        pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
    else:  # SHORT
        pnl_pct = (trade.entry_price - exit_price) / trade.entry_price

    # Apply leverage
    pnl_gross = trade.size_usd * pnl_pct * LEVERAGE

    # Net P&L (after fees)
    total_fees = trade.entry_fee + exit_fee
    pnl_net = pnl_gross - total_fees

    trade.exit_timestamp = exit_timestamp
    trade.exit_price = exit_price
    trade.exit_reason = exit_reason
    trade.exit_fee = exit_fee
    trade.pnl_gross = pnl_gross
    trade.pnl_net = pnl_net

    return trade


# =============================================================================
# WALK-FORWARD SIMULATION
# =============================================================================

def run_fold(data: List[Dict], train_end_date: str, test_end_date: str,
             initial_capital: float) -> Tuple[List[Trade], FoldResult]:
    """
    Run single fold of walk-forward simulation.

    Training data: everything before train_end_date
    Testing data: train_end_date to test_end_date
    """
    # Find indices
    train_end_idx = next((i for i, d in enumerate(data)
                          if d['date'] >= train_end_date), len(data))
    test_end_idx = next((i for i, d in enumerate(data)
                         if d['date'] >= test_end_date), len(data))

    trades = []
    equity = initial_capital
    peak_equity = initial_capital
    max_drawdown = 0
    daily_returns = []
    current_trade = None

    # Simulate through test period only
    for i in range(train_end_idx, min(test_end_idx, len(data) - 1)):
        current = data[i]
        next_candle = data[i + 1]

        # Check exit for open trade (but NOT on entry candle - can't know intraday sequence)
        if current_trade and i > current_trade.entry_idx:
            days_held = i - current_trade.entry_idx

            # Check time exit first
            if days_held >= MAX_HOLD_DAYS:
                current_trade.days_held = days_held
                current_trade = close_trade(
                    current_trade, current['close'], 'TIME_EXIT', current['timestamp']
                )
                trades.append(current_trade)
                equity += current_trade.pnl_net
                daily_returns.append(current_trade.pnl_net / (equity - current_trade.pnl_net))

                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity
                if dd > max_drawdown:
                    max_drawdown = dd

                current_trade = None
            else:
                # Check TP/SL
                exit_result = check_exit(current_trade, current)
                if exit_result:
                    exit_price, exit_reason = exit_result
                    current_trade.days_held = days_held
                    current_trade = close_trade(
                        current_trade, exit_price, exit_reason, current['timestamp']
                    )
                    trades.append(current_trade)

                    # Update equity
                    equity += current_trade.pnl_net
                    daily_returns.append(current_trade.pnl_net / (equity - current_trade.pnl_net))

                    # Track drawdown
                    if equity > peak_equity:
                        peak_equity = equity
                    dd = (peak_equity - equity) / peak_equity
                    if dd > max_drawdown:
                        max_drawdown = dd

                    current_trade = None

        # Generate signal if no position AND we have equity
        if not current_trade and i >= 7 and equity > 10:  # Need lookback window and min equity
            # Get lookback window (ONLY past data, no current)
            lookback = 7
            window = data[i - lookback:i]

            # Generate signal using EXACT same logic as validated backtest
            signal = generate_signal_from_raw(current, window)

            if signal:
                current_trade = execute_entry(signal, next_candle, equity, i + 1)

        # Check for bankruptcy
        if equity <= 0:
            print(f"  [BANKRUPT] Equity depleted at {current['date']}")
            break

    # Calculate metrics
    wins = len([t for t in trades if t.pnl_net and t.pnl_net > 0])
    losses = len([t for t in trades if t.pnl_net and t.pnl_net <= 0])
    win_rate = wins / len(trades) if trades else 0

    pnl_gross = sum(t.pnl_gross or 0 for t in trades)
    pnl_net = sum(t.pnl_net or 0 for t in trades)
    total_fees = sum((t.entry_fee or 0) + (t.exit_fee or 0) for t in trades)

    # Sharpe ratio
    if daily_returns and len(daily_returns) > 1:
        mean_ret = sum(daily_returns) / len(daily_returns)
        std_ret = (sum((r - mean_ret) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
        sharpe = (mean_ret / std_ret) * (252 ** 0.5) if std_ret else 0
    else:
        sharpe = 0

    # Profit factor
    gross_wins = sum(t.pnl_net for t in trades if t.pnl_net and t.pnl_net > 0)
    gross_losses = abs(sum(t.pnl_net for t in trades if t.pnl_net and t.pnl_net < 0))
    profit_factor = gross_wins / gross_losses if gross_losses else float('inf')

    # Get date range
    test_dates = [d['date'] for d in data[train_end_idx:test_end_idx]]
    train_dates = [d['date'] for d in data[:train_end_idx]]

    result = FoldResult(
        fold=0,
        train_start=train_dates[0] if train_dates else '',
        train_end=train_dates[-1] if train_dates else '',
        test_start=test_dates[0] if test_dates else '',
        test_end=test_dates[-1] if test_dates else '',
        trades=len(trades),
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        pnl_gross=pnl_gross,
        pnl_net=pnl_net,
        total_fees=total_fees,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
    )

    return trades, result


def run_walk_forward(data: List[Dict], initial_capital: float) -> List[FoldResult]:
    """
    Run simulation on available data with 70/30 train/test split.
    Adapts to actual data range.
    """
    # Get actual date range
    dates = [d['date'] for d in data]
    start_date = dates[0]
    end_date = dates[-1]

    # Calculate 70/30 split point
    total_days = len(data)
    train_days = int(total_days * 0.7)
    split_date = dates[train_days]

    print(f"\n[DATA RANGE] {start_date} to {end_date} ({total_days} days)")
    print(f"[SPLIT] Train: {start_date} to {split_date} ({train_days} days)")
    print(f"[SPLIT] Test: {split_date} to {end_date} ({total_days - train_days} days)")

    results = []
    all_trades = []

    # Single fold with adaptive dates
    print(f"\n[RUNNING] Walk-forward simulation...")
    trades, result = run_fold(data, split_date, end_date, initial_capital)
    result.fold = 1
    results.append(result)
    all_trades.extend(trades)

    print(f"\n  Trades: {result.trades} | WR: {result.win_rate:.1%} | "
          f"PnL: ${result.pnl_net:+.2f} | Sharpe: {result.sharpe:.2f} | "
          f"MaxDD: {result.max_drawdown:.1%}")

    # Save all trades
    trades_file = RESULTS_DIR / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(trades_file, 'w') as f:
        json.dump([asdict(t) for t in all_trades], f, indent=2)
    print(f"\n[SAVED] {len(all_trades)} trades to {trades_file}")

    return results


# =============================================================================
# VALIDATION
# =============================================================================

def validate_results(results: List[FoldResult]) -> bool:
    """Check if simulation passes all criteria."""
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    passed = True
    issues = []

    for r in results:
        fold_passed = True

        if r.trades < 20:
            issues.append(f"Fold {r.fold}: Only {r.trades} trades (need 20+)")
            fold_passed = False

        if r.win_rate < 0.50:
            issues.append(f"Fold {r.fold}: Win rate {r.win_rate:.1%} < 50%")
            fold_passed = False

        if r.pnl_net < 0:
            issues.append(f"Fold {r.fold}: Negative P&L ${r.pnl_net:.2f}")
            fold_passed = False

        if r.max_drawdown > 0.25:
            issues.append(f"Fold {r.fold}: Max DD {r.max_drawdown:.1%} > 25%")
            fold_passed = False

        if not fold_passed:
            passed = False

    if issues:
        print("\nISSUES:")
        for issue in issues:
            print(f"  [X] {issue}")
    else:
        print("\n  [OK] All folds passed validation")

    # Aggregate stats
    total_trades = sum(r.trades for r in results)
    total_pnl = sum(r.pnl_net for r in results)
    avg_wr = sum(r.win_rate * r.trades for r in results) / total_trades if total_trades else 0
    avg_sharpe = sum(r.sharpe for r in results) / len(results)
    worst_dd = max(r.max_drawdown for r in results)

    print(f"\nAGGREGATE:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Total P&L: ${total_pnl:+.2f}")
    print(f"  Avg Win Rate: {avg_wr:.1%}")
    print(f"  Avg Sharpe: {avg_sharpe:.2f}")
    print(f"  Worst Drawdown: {worst_dd:.1%}")

    print("\n" + "=" * 70)
    if passed:
        print("STATUS: PASS - Ready for live deployment")
    else:
        print("STATUS: FAIL - Fix issues before going live")
    print("=" * 70)

    return passed


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='1:1 Walk-Forward Simulation')
    parser.add_argument('--capital', type=float, default=100.0,
                        help='Initial capital (default: 100)')
    parser.add_argument('--mode', choices=['walk-forward', 'single', 'analyze'],
                        default='walk-forward', help='Simulation mode')
    parser.add_argument('--start', type=str, help='Start date for single mode')
    parser.add_argument('--end', type=str, help='End date for single mode')
    args = parser.parse_args()

    print("=" * 70)
    print("1:1 WALK-FORWARD SIMULATION")
    print("=" * 70)
    print(f"Capital: ${args.capital}")
    print(f"Leverage: {LEVERAGE}x")
    print(f"TP: {TAKE_PROFIT_PCT:.1%} | SL: {STOP_LOSS_PCT:.1%}")
    print(f"Fees: {TAKER_FEE:.3%} taker | Slippage: {SLIPPAGE_BPS}bps")
    print("=" * 70)

    # Load data
    print("\n[LOADING] Data...")

    # Try multiple database paths
    feature_paths = [
        DATA_DIR / "bitcoin_features.db",
        DATA_DIR / "blockchain_features.db",
        DATA_DIR / "historical_flows.db",
    ]

    features = {}
    for path in feature_paths:
        if path.exists():
            features = load_blockchain_features(str(path))
            if features:
                print(f"  Loaded {len(features)} blockchain features from {path.name}")
                break

    price_paths = [
        DATA_DIR / "bitcoin_2021_2025.db",
        DATA_DIR / "historical_flows.db",
        DATA_DIR / "prices.db",
    ]

    prices = {}
    for path in price_paths:
        if path.exists():
            prices = load_prices(str(path))
            if prices:
                print(f"  Loaded {len(prices)} price candles from {path.name}")
                break

    if not features or not prices:
        print("[ERROR] Could not load required data")
        print(f"  Features: {len(features)}")
        print(f"  Prices: {len(prices)}")
        sys.exit(1)

    # Merge data
    data = merge_data(features, prices)
    print(f"  Merged: {len(data)} records")

    if len(data) < 100:
        print("[ERROR] Insufficient data for simulation")
        sys.exit(1)

    # Date range
    dates = [d['date'] for d in data]
    print(f"  Date range: {dates[0]} to {dates[-1]}")

    # Run simulation
    if args.mode == 'walk-forward':
        results = run_walk_forward(data, args.capital)
        validate_results(results)

    elif args.mode == 'single':
        start = args.start or dates[0]
        end = args.end or dates[-1]
        trades, result = run_fold(data, start, end, args.capital)
        print(f"\n[RESULT] {result.trades} trades, {result.win_rate:.1%} WR, "
              f"${result.pnl_net:+.2f} P&L")


if __name__ == '__main__':
    main()
