#!/usr/bin/env python3
"""
HFT MAX ROI - $100 to Maximum Returns

AGGRESSIVE configuration:
- Full Kelly sizing (not quarter)
- 20x leverage on high-confidence signals
- Trade EVERY signal (no filtering)
- Compound after every win
- Tight stops, let winners run

Target: 1-5% daily returns = 30-150% monthly
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "simulation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# AGGRESSIVE CONSTANTS - MAX ROI MODE
# =============================================================================

# Hyperliquid fees
TAKER_FEE = 0.00035      # 0.035%
MAKER_FEE = 0.0001       # 0.01%
SLIPPAGE_BPS = 3         # Reduced - we're fast

# AGGRESSIVE Trading Parameters
TAKE_PROFIT_PCT = 0.008  # 0.8% TP (tighter, faster exits)
STOP_LOSS_PCT = 0.004    # 0.4% SL (tighter stops)
MAX_HOLD_HOURS = 24      # Exit within 24 hours
BASE_LEVERAGE = 10       # Base leverage
MAX_LEVERAGE = 20        # Max on high confidence
MIN_CONFIDENCE = 0.50    # Trade ALL signals with any edge

# Kelly sizing - FULL KELLY for max growth
KELLY_FRACTION = 1.0     # Full Kelly (aggressive)

# Signal thresholds - LOWER to catch more signals
WHALE_THRESHOLD = 1.2    # 1.2x average (was 1.5x)
VALUE_THRESHOLD = 1.2    # 1.2x average (was 1.5x)
TX_THRESHOLD = 1.1       # 1.1x average (was 1.3x)


@dataclass
class Trade:
    id: str
    entry_timestamp: int
    entry_price: float
    direction: int
    size_usd: float
    leverage: float
    stop_loss: float
    take_profit: float
    signal_type: str
    confidence: float
    entry_fee: float
    exit_timestamp: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_fee: Optional[float] = None
    pnl_gross: Optional[float] = None
    pnl_net: Optional[float] = None


def load_data() -> List[Dict]:
    """Load and merge all available data."""
    # Try hourly data first (more granular = more trades)
    hourly_path = DATA_DIR / "bitcoin_hourly.db"
    daily_path = DATA_DIR / "bitcoin_features.db"
    price_path = DATA_DIR / "bitcoin_2021_2025.db"

    data = []

    # Load features
    features = {}
    for path in [daily_path, DATA_DIR / "historical_flows.db"]:
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
                    date_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
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
    prices = {}
    for path in [price_path, DATA_DIR / "historical_flows.db"]:
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
                        date_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
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

    # Merge
    for date_str in sorted(features.keys()):
        if date_str in prices:
            data.append({
                'date': date_str,
                **features[date_str],
                **prices[date_str],
            })

    return data


def generate_aggressive_signals(data: List[Dict], idx: int, lookback: int = 5) -> List[Dict]:
    """
    Generate ALL possible signals - no filtering.
    Returns list of signals with confidence scores.
    """
    if idx < lookback:
        return []

    signals = []
    window = data[idx - lookback:idx]
    current = data[idx]

    # Calculate averages
    avg_tx = sum(f['tx_count'] for f in window) / len(window)
    avg_value = sum(f['total_value_btc'] for f in window) / len(window)
    avg_whale = sum(f['whale_tx_count'] for f in window) / len(window)

    # Signal 1: Whale Activity (ANY deviation)
    if avg_whale > 0:
        whale_ratio = current['whale_tx_count'] / avg_whale
        if whale_ratio > WHALE_THRESHOLD:
            receivers = current.get('unique_receivers', 1) or 1
            senders = current.get('unique_senders', 1) or 1

            if receivers > senders:
                signals.append({
                    'type': 'WHALE_ACCUMULATION',
                    'direction': 1,
                    'confidence': min(0.70, 0.55 + (whale_ratio - 1) * 0.1),
                    'leverage': min(MAX_LEVERAGE, BASE_LEVERAGE + int((whale_ratio - 1) * 5)),
                })
            else:
                signals.append({
                    'type': 'WHALE_DISTRIBUTION',
                    'direction': -1,
                    'confidence': min(0.65, 0.52 + (whale_ratio - 1) * 0.08),
                    'leverage': min(MAX_LEVERAGE, BASE_LEVERAGE + int((whale_ratio - 1) * 3)),
                })

    # Signal 2: Value Spike
    if avg_value > 0:
        value_ratio = current['total_value_btc'] / avg_value
        if value_ratio > VALUE_THRESHOLD:
            receivers = current.get('unique_receivers', 1) or 1
            senders = current.get('unique_senders', 1) or 1

            if receivers > senders:
                signals.append({
                    'type': 'VALUE_INFLOW',
                    'direction': 1,
                    'confidence': min(0.65, 0.52 + (value_ratio - 1) * 0.08),
                    'leverage': min(MAX_LEVERAGE, BASE_LEVERAGE + int((value_ratio - 1) * 4)),
                })
            else:
                signals.append({
                    'type': 'VALUE_OUTFLOW',
                    'direction': -1,
                    'confidence': min(0.60, 0.51 + (value_ratio - 1) * 0.06),
                    'leverage': min(15, BASE_LEVERAGE + int((value_ratio - 1) * 3)),
                })

    # Signal 3: TX Surge
    if avg_tx > 0:
        tx_ratio = current['tx_count'] / avg_tx
        if tx_ratio > TX_THRESHOLD:
            # Check momentum
            prev_value = window[-1]['total_value_btc'] if window else 0
            if current['total_value_btc'] > prev_value:
                signals.append({
                    'type': 'TX_SURGE_BULLISH',
                    'direction': 1,
                    'confidence': min(0.58, 0.51 + (tx_ratio - 1) * 0.05),
                    'leverage': min(12, BASE_LEVERAGE),
                })
            else:
                signals.append({
                    'type': 'TX_SURGE_BEARISH',
                    'direction': -1,
                    'confidence': min(0.55, 0.50 + (tx_ratio - 1) * 0.04),
                    'leverage': min(10, BASE_LEVERAGE - 2),
                })

    # Signal 4: Quiet Accumulation (whales active, retail quiet)
    if avg_tx > 0 and avg_whale > 0:
        tx_ratio = current['tx_count'] / avg_tx
        whale_ratio = current['whale_tx_count'] / avg_whale
        if tx_ratio < 0.9 and whale_ratio > 0.95:
            signals.append({
                'type': 'QUIET_WHALE',
                'direction': 1,
                'confidence': 0.60,
                'leverage': 15,
            })

    # Signal 5: Momentum - price vs moving average
    closes = [d['close'] for d in window]
    ma = sum(closes) / len(closes)
    current_price = current['close']

    if current_price > ma * 1.02:  # 2% above MA
        signals.append({
            'type': 'MOMENTUM_UP',
            'direction': 1,
            'confidence': 0.54,
            'leverage': 10,
        })
    elif current_price < ma * 0.98:  # 2% below MA
        signals.append({
            'type': 'MOMENTUM_DOWN',
            'direction': -1,
            'confidence': 0.52,
            'leverage': 8,
        })

    return signals


def calculate_kelly_size(confidence: float, win_loss_ratio: float = 2.5) -> float:
    """
    Full Kelly sizing for maximum growth.

    Kelly = W - (1-W)/R
    Where W = win probability, R = win/loss ratio
    """
    w = confidence
    r = win_loss_ratio
    kelly = w - (1 - w) / r
    return max(0, min(kelly * KELLY_FRACTION, 0.5))  # Cap at 50% per trade


def run_aggressive_simulation(initial_capital: float = 100.0) -> Dict:
    """Run aggressive HFT simulation for maximum ROI."""

    print("=" * 70)
    print("HFT MAX ROI SIMULATION")
    print("=" * 70)
    print(f"Initial Capital: ${initial_capital}")
    print(f"Kelly Fraction: {KELLY_FRACTION} (FULL)")
    print(f"Leverage Range: {BASE_LEVERAGE}x - {MAX_LEVERAGE}x")
    print(f"TP: {TAKE_PROFIT_PCT:.2%} | SL: {STOP_LOSS_PCT:.2%}")
    print("=" * 70)

    # Load data
    print("\n[LOADING DATA]...")
    data = load_data()
    print(f"  Loaded {len(data)} data points")

    if len(data) < 20:
        print("[ERROR] Insufficient data")
        return {}

    # Split 60/40 for more test data
    split_idx = int(len(data) * 0.6)
    test_data = data[split_idx:]

    print(f"  Test period: {test_data[0]['date']} to {test_data[-1]['date']}")
    print(f"  Test days: {len(test_data)}")

    # Simulation state
    equity = initial_capital
    peak_equity = initial_capital
    max_drawdown = 0
    trades = []
    daily_returns = []
    current_trade = None

    # Run simulation
    print("\n[RUNNING SIMULATION]...")

    for i in range(5, len(test_data) - 1):
        current = test_data[i]
        next_candle = test_data[i + 1]

        # Check exit for open trade
        if current_trade:
            high = current['high']
            low = current['low']

            exit_price = None
            exit_reason = None

            if current_trade['direction'] == 1:  # LONG
                if high >= current_trade['take_profit']:
                    exit_price = current_trade['take_profit']
                    exit_reason = 'TAKE_PROFIT'
                elif low <= current_trade['stop_loss']:
                    exit_price = current_trade['stop_loss']
                    exit_reason = 'STOP_LOSS'
            else:  # SHORT
                if low <= current_trade['take_profit']:
                    exit_price = current_trade['take_profit']
                    exit_reason = 'TAKE_PROFIT'
                elif high >= current_trade['stop_loss']:
                    exit_price = current_trade['stop_loss']
                    exit_reason = 'STOP_LOSS'

            if exit_price:
                # Calculate P&L
                if current_trade['direction'] == 1:
                    pnl_pct = (exit_price - current_trade['entry_price']) / current_trade['entry_price']
                else:
                    pnl_pct = (current_trade['entry_price'] - exit_price) / current_trade['entry_price']

                pnl_gross = current_trade['size_usd'] * pnl_pct * current_trade['leverage']
                exit_fee = current_trade['size_usd'] * TAKER_FEE
                pnl_net = pnl_gross - current_trade['entry_fee'] - exit_fee

                current_trade['exit_price'] = exit_price
                current_trade['exit_reason'] = exit_reason
                current_trade['pnl_net'] = pnl_net
                trades.append(current_trade)

                # Update equity
                old_equity = equity
                equity += pnl_net
                daily_returns.append(pnl_net / old_equity)

                # Track drawdown
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity
                if dd > max_drawdown:
                    max_drawdown = dd

                current_trade = None

        # Generate new signals if no position
        if not current_trade and equity > 1:
            signals = generate_aggressive_signals(test_data[:split_idx + i], i, lookback=5)

            if signals:
                # Take the highest confidence signal
                best_signal = max(signals, key=lambda s: s['confidence'])

                if best_signal['confidence'] >= MIN_CONFIDENCE:
                    entry_price = next_candle['open']

                    # Apply slippage
                    if best_signal['direction'] == 1:
                        entry_price *= (1 + SLIPPAGE_BPS / 10000)
                    else:
                        entry_price *= (1 - SLIPPAGE_BPS / 10000)

                    # Kelly sizing
                    kelly_pct = calculate_kelly_size(best_signal['confidence'])
                    size_usd = equity * kelly_pct
                    leverage = best_signal['leverage']

                    # TP/SL
                    if best_signal['direction'] == 1:
                        stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                        take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
                    else:
                        stop_loss = entry_price * (1 + STOP_LOSS_PCT)
                        take_profit = entry_price * (1 - TAKE_PROFIT_PCT)

                    entry_fee = size_usd * TAKER_FEE

                    current_trade = {
                        'id': f"HFT_{next_candle['timestamp']}",
                        'entry_price': entry_price,
                        'direction': best_signal['direction'],
                        'size_usd': size_usd,
                        'leverage': leverage,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'signal_type': best_signal['type'],
                        'confidence': best_signal['confidence'],
                        'entry_fee': entry_fee,
                    }

        # Bankruptcy check
        if equity <= 0:
            print(f"  [BANKRUPT] at {current['date']}")
            break

    # Calculate results
    wins = [t for t in trades if t.get('pnl_net', 0) > 0]
    losses = [t for t in trades if t.get('pnl_net', 0) <= 0]

    total_pnl = sum(t.get('pnl_net', 0) for t in trades)
    win_rate = len(wins) / len(trades) if trades else 0

    gross_wins = sum(t['pnl_net'] for t in wins)
    gross_losses = abs(sum(t['pnl_net'] for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Sharpe
    if daily_returns and len(daily_returns) > 1:
        mean_ret = sum(daily_returns) / len(daily_returns)
        std_ret = (sum((r - mean_ret) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
        sharpe = (mean_ret / std_ret) * (252 ** 0.5) if std_ret else 0
    else:
        sharpe = 0

    # Daily ROI
    test_days = len(test_data)
    daily_roi = (equity / initial_capital) ** (1 / test_days) - 1 if test_days > 0 else 0
    monthly_roi = (1 + daily_roi) ** 30 - 1
    yearly_roi = (1 + daily_roi) ** 365 - 1

    results = {
        'initial_capital': initial_capital,
        'final_equity': equity,
        'total_pnl': total_pnl,
        'return_pct': (equity / initial_capital - 1) * 100,
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'daily_roi': daily_roi,
        'monthly_roi': monthly_roi,
        'yearly_roi': yearly_roi,
        'test_days': test_days,
        'trades_per_day': len(trades) / test_days if test_days > 0 else 0,
    }

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Initial Capital:  ${initial_capital:.2f}")
    print(f"Final Equity:     ${equity:.2f}")
    print(f"Total P&L:        ${total_pnl:+.2f}")
    print(f"Return:           {results['return_pct']:+.2f}%")
    print()
    print(f"Total Trades:     {len(trades)}")
    print(f"Wins/Losses:      {len(wins)}/{len(losses)}")
    print(f"Win Rate:         {win_rate:.1%}")
    print(f"Profit Factor:    {profit_factor:.2f}")
    print(f"Max Drawdown:     {max_drawdown:.2%}")
    print(f"Sharpe Ratio:     {sharpe:.2f}")
    print()
    print(f"Trades/Day:       {results['trades_per_day']:.2f}")
    print(f"Daily ROI:        {daily_roi:.4%}")
    print(f"Monthly ROI:      {monthly_roi:.2%}")
    print(f"Yearly ROI:       {yearly_roi:.2%}")
    print()
    print("PROJECTIONS at current rate:")
    print(f"  $100 x 30 days:  ${100 * (1 + monthly_roi):.2f}")
    print(f"  $100 x 90 days:  ${100 * ((1 + daily_roi) ** 90):.2f}")
    print(f"  $100 x 365 days: ${100 * (1 + yearly_roi):.2f}")
    print("=" * 70)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f"hft_max_roi_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'kelly_fraction': KELLY_FRACTION,
                'base_leverage': BASE_LEVERAGE,
                'max_leverage': MAX_LEVERAGE,
                'take_profit': TAKE_PROFIT_PCT,
                'stop_loss': STOP_LOSS_PCT,
            },
            'results': results,
            'trades': trades[-50:],  # Last 50 trades
        }, f, indent=2, default=str)
    print(f"\n[SAVED] {results_file}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--capital', type=float, default=100.0)
    args = parser.parse_args()

    run_aggressive_simulation(args.capital)
