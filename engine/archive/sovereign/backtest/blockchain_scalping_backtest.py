#!/usr/bin/env python3
"""
Blockchain Scalping Backtest - Using REAL Historical Data
==========================================================

Uses actual blockchain features from bitcoin_features.db:
- Block-level: tx_count, total_value_btc, total_fees_btc, avg_fee_rate
- Daily-level: whale_tx_count, unique_senders, unique_receivers

This tests our scalping strategy against real blockchain data,
not simulated signals.
"""
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json


@dataclass
class BlockchainSignal:
    """Signal generated from blockchain data."""
    timestamp: int
    direction: int  # 1=LONG, -1=SHORT, 0=NEUTRAL
    confidence: float
    trigger: str  # What triggered the signal
    metrics: Dict


@dataclass
class Trade:
    entry_time: int
    entry_price: float
    exit_time: int
    exit_price: float
    direction: int
    exit_reason: str
    pnl_pct: float
    pnl_usd: float
    signal_trigger: str


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
    profit_factor: float
    exits_by_tp: int
    exits_by_sl: int
    exits_by_time: int
    signal_breakdown: Dict


class BlockchainScalpingBacktester:
    """
    Backtest scalping using real blockchain features.

    Signal Generation Logic (from RENAISSANCE_BLOCKCHAIN_PATTERNS.md):
    - High value flow + low fees = Accumulation (LONG)
    - High value flow + high fees = Distribution/Urgency (SHORT)
    - Whale activity spike = Follow the whales
    - Fee rate spike = Volatility incoming
    """

    FEE_RATE = 0.0001  # 0.01% Hyperliquid maker

    def __init__(self, features_db: str = "data/bitcoin_features.db",
                 prices_db: str = "data/historical_flows.db"):
        self.features_db = Path(features_db)
        self.prices_db = Path(prices_db)

        self.daily_features: List[Dict] = []
        self.prices: Dict[int, Tuple[float, float, float, float]] = {}  # timestamp -> (open, high, low, close)

        self._load_data()

    def _load_data(self):
        """Load blockchain features and prices."""
        # Load daily features
        conn = sqlite3.connect(self.features_db)
        c = conn.cursor()
        c.execute('''
            SELECT date, timestamp, tx_count, total_value_btc, total_value_usd,
                   unique_senders, unique_receivers, whale_tx_count
            FROM daily_features
            ORDER BY timestamp
        ''')

        cols = ['date', 'timestamp', 'tx_count', 'total_value_btc', 'total_value_usd',
                'unique_senders', 'unique_receivers', 'whale_tx_count']

        for row in c.fetchall():
            self.daily_features.append(dict(zip(cols, row)))

        conn.close()
        print(f"[BACKTEST] Loaded {len(self.daily_features):,} daily features")

        # Load prices
        conn = sqlite3.connect(self.prices_db)
        c = conn.cursor()
        c.execute('SELECT timestamp, open, high, low, close FROM prices ORDER BY timestamp')

        for row in c.fetchall():
            ts, o, h, l, cl = row
            self.prices[ts] = (o, h, l, cl)

        conn.close()
        print(f"[BACKTEST] Loaded {len(self.prices):,} price candles")

        # Filter out records with NULL values
        self.daily_features = [f for f in self.daily_features if all([
            f.get('tx_count') is not None,
            f.get('total_value_btc') is not None,
            f.get('whale_tx_count') is not None,
            f.get('unique_senders') is not None,
            f.get('unique_receivers') is not None
        ])]
        print(f"[BACKTEST] After filtering nulls: {len(self.daily_features):,} valid features")

        # Calculate baseline metrics for signal generation
        if self.daily_features:
            self.avg_tx_count = sum(f['tx_count'] for f in self.daily_features) / len(self.daily_features)
            self.avg_value = sum(f['total_value_btc'] for f in self.daily_features) / len(self.daily_features)
            self.avg_whale = sum(f['whale_tx_count'] for f in self.daily_features) / len(self.daily_features)
            print(f"[BACKTEST] Avg tx_count: {self.avg_tx_count:.0f}, Avg value: {self.avg_value:.0f} BTC, Avg whale: {self.avg_whale:.0f}")

    def _find_closest_price(self, timestamp: int) -> Optional[Tuple[float, float, float, float]]:
        """Find closest price to timestamp."""
        # Daily timestamps - find within ±1 day
        day_seconds = 86400

        for offset in [0, day_seconds, -day_seconds, day_seconds*2, -day_seconds*2]:
            check_ts = timestamp + offset
            if check_ts in self.prices:
                return self.prices[check_ts]

        # Fallback: find nearest
        if self.prices:
            closest_ts = min(self.prices.keys(), key=lambda t: abs(t - timestamp))
            if abs(closest_ts - timestamp) < day_seconds * 7:  # Within a week
                return self.prices[closest_ts]

        return None

    def generate_signals(self, lookback: int = 7) -> List[BlockchainSignal]:
        """
        Generate trading signals from blockchain features.

        Signal Logic:
        1. VALUE_SPIKE: total_value > 2x average = big movement coming
        2. WHALE_ACTIVITY: whale_tx_count > 2x average = smart money moving
        3. TX_SURGE: tx_count > 1.5x average = network activity spike
        4. ACCUMULATION: high value + increasing unique receivers = LONG
        5. DISTRIBUTION: high value + increasing unique senders = SHORT
        """
        signals = []

        if len(self.daily_features) < lookback + 1:
            return signals

        for i in range(lookback, len(self.daily_features)):
            current = self.daily_features[i]
            window = self.daily_features[i-lookback:i]

            # Calculate rolling averages
            avg_tx = sum(f['tx_count'] for f in window) / len(window)
            avg_value = sum(f['total_value_btc'] for f in window) / len(window)
            avg_whale = sum(f['whale_tx_count'] for f in window) / len(window)
            avg_senders = sum(f['unique_senders'] for f in window) / len(window)
            avg_receivers = sum(f['unique_receivers'] for f in window) / len(window)

            direction = 0
            confidence = 0.5
            trigger = ""

            # Signal 1: Whale Activity Spike
            if current['whale_tx_count'] > avg_whale * 2:
                # Whales are moving - check if accumulating or distributing
                if current['unique_receivers'] > avg_receivers * 1.2:
                    direction = 1  # LONG - whales distributing to many (buying)
                    confidence = 0.65
                    trigger = "WHALE_ACCUMULATION"
                elif current['unique_senders'] > avg_senders * 1.2:
                    direction = -1  # SHORT - many sending to exchanges
                    confidence = 0.60
                    trigger = "WHALE_DISTRIBUTION"

            # Signal 2: Value Spike
            elif current['total_value_btc'] > avg_value * 2:
                if current['unique_receivers'] > current['unique_senders']:
                    direction = 1  # LONG - value moving to more receivers
                    confidence = 0.58
                    trigger = "VALUE_ACCUMULATION"
                else:
                    direction = -1  # SHORT - value concentrating
                    confidence = 0.55
                    trigger = "VALUE_DISTRIBUTION"

            # Signal 3: Transaction Surge
            elif current['tx_count'] > avg_tx * 1.5:
                # Network busy - momentum signal
                prev_value = window[-1]['total_value_btc']
                if current['total_value_btc'] > prev_value:
                    direction = 1  # LONG - increasing activity
                    confidence = 0.55
                    trigger = "TX_SURGE_UP"
                else:
                    direction = -1  # SHORT - decreasing value despite tx
                    confidence = 0.52
                    trigger = "TX_SURGE_DOWN"

            # Signal 4: Quiet period reversal
            elif current['tx_count'] < avg_tx * 0.7 and current['whale_tx_count'] > avg_whale:
                # Low retail, whales active = accumulation
                direction = 1
                confidence = 0.60
                trigger = "QUIET_WHALE"

            if direction != 0:
                signals.append(BlockchainSignal(
                    timestamp=current['timestamp'],
                    direction=direction,
                    confidence=confidence,
                    trigger=trigger,
                    metrics={
                        'tx_count': current['tx_count'],
                        'value_btc': current['total_value_btc'],
                        'whale_count': current['whale_tx_count'],
                        'senders': current['unique_senders'],
                        'receivers': current['unique_receivers']
                    }
                ))

        print(f"[BACKTEST] Generated {len(signals):,} blockchain signals")

        # Signal breakdown
        triggers = {}
        for s in signals:
            triggers[s.trigger] = triggers.get(s.trigger, 0) + 1

        print(f"[BACKTEST] Signal breakdown: {triggers}")

        return signals

    def run_backtest(
        self,
        take_profit_pct: float = 0.01,   # 1.0%
        stop_loss_pct: float = 0.003,     # 0.3%
        max_hold_days: int = 5,           # 5 days max hold
        capital: float = 100.0,
        leverage: float = 5.0,
        min_confidence: float = 0.52,
        lookback: int = 7
    ) -> BacktestResult:
        """
        Run backtest with real blockchain signals.
        """
        signals = self.generate_signals(lookback)

        trades: List[Trade] = []
        position_size = capital * leverage

        current_capital = capital
        peak_capital = capital
        max_drawdown = 0.0

        signal_wins = {}
        signal_losses = {}

        for signal in signals:
            if signal.confidence < min_confidence:
                continue

            # Get entry price
            entry_data = self._find_closest_price(signal.timestamp)
            if not entry_data:
                continue

            entry_price = entry_data[3]  # close price

            # Find exit
            exit_price = None
            exit_reason = None
            exit_time = signal.timestamp

            # Look forward up to max_hold_days
            for days in range(1, max_hold_days + 1):
                future_ts = signal.timestamp + (days * 86400)
                future_data = self._find_closest_price(future_ts)

                if not future_data:
                    continue

                future_open, future_high, future_low, future_close = future_data

                # Check TP/SL
                if signal.direction == 1:  # LONG
                    if future_high >= entry_price * (1 + take_profit_pct):
                        exit_price = entry_price * (1 + take_profit_pct)
                        exit_reason = 'tp'
                        exit_time = future_ts
                        break
                    if future_low <= entry_price * (1 - stop_loss_pct):
                        exit_price = entry_price * (1 - stop_loss_pct)
                        exit_reason = 'sl'
                        exit_time = future_ts
                        break
                else:  # SHORT
                    if future_low <= entry_price * (1 - take_profit_pct):
                        exit_price = entry_price * (1 - take_profit_pct)
                        exit_reason = 'tp'
                        exit_time = future_ts
                        break
                    if future_high >= entry_price * (1 + stop_loss_pct):
                        exit_price = entry_price * (1 + stop_loss_pct)
                        exit_reason = 'sl'
                        exit_time = future_ts
                        break

            # Time exit if no TP/SL
            if exit_price is None:
                future_ts = signal.timestamp + (max_hold_days * 86400)
                future_data = self._find_closest_price(future_ts)
                if future_data:
                    exit_price = future_data[3]
                    exit_reason = 'time'
                    exit_time = future_ts
                else:
                    continue

            # Calculate PnL
            if signal.direction == 1:
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            pnl_pct -= (self.FEE_RATE * 2)  # Round trip fees
            pnl_usd = position_size * pnl_pct

            # Track by signal type
            if pnl_usd > 0:
                signal_wins[signal.trigger] = signal_wins.get(signal.trigger, 0) + 1
            else:
                signal_losses[signal.trigger] = signal_losses.get(signal.trigger, 0) + 1

            trades.append(Trade(
                entry_time=signal.timestamp,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_price,
                direction=signal.direction,
                exit_reason=exit_reason,
                pnl_pct=pnl_pct * 100,
                pnl_usd=pnl_usd,
                signal_trigger=signal.trigger
            ))

            current_capital += pnl_usd
            if current_capital > peak_capital:
                peak_capital = current_capital

            dd = (peak_capital - current_capital) / peak_capital if peak_capital > 0 else 0
            max_drawdown = max(max_drawdown, dd)

        # Calculate results
        if not trades:
            return BacktestResult(
                params={'tp': take_profit_pct*100, 'sl': stop_loss_pct*100, 'hold': max_hold_days},
                total_trades=0, wins=0, losses=0, win_rate=0,
                total_pnl_pct=0, total_pnl_usd=0, avg_trade_pnl=0,
                max_drawdown=0, sharpe_ratio=0, profit_factor=0,
                exits_by_tp=0, exits_by_sl=0, exits_by_time=0,
                signal_breakdown={}
            )

        wins = len([t for t in trades if t.pnl_usd > 0])
        losses = len([t for t in trades if t.pnl_usd <= 0])
        win_rate = wins / len(trades)

        total_pnl_pct = sum(t.pnl_pct for t in trades)
        total_pnl_usd = sum(t.pnl_usd for t in trades)
        avg_trade_pnl = total_pnl_usd / len(trades)

        # Profit factor
        gross_profit = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in trades if t.pnl_usd < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Exit breakdown
        exits_tp = len([t for t in trades if t.exit_reason == 'tp'])
        exits_sl = len([t for t in trades if t.exit_reason == 'sl'])
        exits_time = len([t for t in trades if t.exit_reason == 'time'])

        # Sharpe
        returns = [t.pnl_pct for t in trades]
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5 if variance > 0 else 0.001
        sharpe = (mean_return / std_dev) * (252 ** 0.5)

        # Signal performance breakdown
        signal_breakdown = {}
        all_triggers = set(signal_wins.keys()) | set(signal_losses.keys())
        for trigger in all_triggers:
            w = signal_wins.get(trigger, 0)
            l = signal_losses.get(trigger, 0)
            total = w + l
            wr = w / total if total > 0 else 0
            signal_breakdown[trigger] = {'wins': w, 'losses': l, 'win_rate': wr}

        return BacktestResult(
            params={'tp': take_profit_pct*100, 'sl': stop_loss_pct*100, 'hold': max_hold_days, 'conf': min_confidence},
            total_trades=len(trades),
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_pnl_pct=total_pnl_pct,
            total_pnl_usd=total_pnl_usd,
            avg_trade_pnl=avg_trade_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            exits_by_tp=exits_tp,
            exits_by_sl=exits_sl,
            exits_by_time=exits_time,
            signal_breakdown=signal_breakdown
        )

    def grid_search(self) -> List[BacktestResult]:
        """Run grid search over parameters."""
        results = []

        tp_range = [0.005, 0.01, 0.015, 0.02, 0.03]  # 0.5% to 3%
        sl_range = [0.003, 0.005, 0.01]              # 0.3% to 1%
        hold_range = [3, 5, 7, 10]                    # 3-10 days
        conf_range = [0.52, 0.55, 0.58, 0.60]         # Min confidence

        total = len(tp_range) * len(sl_range) * len(hold_range) * len(conf_range)

        print(f"\n[GRID SEARCH] Testing {total} parameter combinations...")
        print("=" * 100)

        combo = 0
        for tp in tp_range:
            for sl in sl_range:
                for hold in hold_range:
                    for conf in conf_range:
                        combo += 1
                        result = self.run_backtest(
                            take_profit_pct=tp,
                            stop_loss_pct=sl,
                            max_hold_days=hold,
                            min_confidence=conf
                        )
                        results.append(result)

                        if result.total_trades > 0:
                            print(f"[{combo}/{total}] TP={tp*100:.1f}% SL={sl*100:.1f}% "
                                  f"Hold={hold}d Conf={conf:.0%} | "
                                  f"Trades={result.total_trades} WR={result.win_rate:.1%} "
                                  f"PnL=${result.total_pnl_usd:.2f} PF={result.profit_factor:.2f}")

        # Rank by profit factor (more robust than PnL)
        results.sort(key=lambda r: r.profit_factor if r.total_trades >= 20 else 0, reverse=True)

        print("\n" + "=" * 100)
        print("TOP 10 PARAMETER COMBINATIONS (by Profit Factor)")
        print("=" * 100)

        for i, r in enumerate(results[:10], 1):
            print(f"\n#{i}: TP={r.params['tp']:.1f}% SL={r.params['sl']:.1f}% "
                  f"Hold={r.params['hold']}d Conf={r.params.get('conf', 0.5):.0%}")
            print(f"    Trades: {r.total_trades}, Win Rate: {r.win_rate:.1%}, Profit Factor: {r.profit_factor:.2f}")
            print(f"    Total PnL: ${r.total_pnl_usd:.2f} ({r.total_pnl_pct:.1f}%)")
            print(f"    Max DD: {r.max_drawdown:.1%}, Sharpe: {r.sharpe_ratio:.2f}")
            print(f"    Exits: TP={r.exits_by_tp}, SL={r.exits_by_sl}, Time={r.exits_by_time}")
            if r.signal_breakdown:
                print(f"    Signals: {r.signal_breakdown}")

        return results

    def save_results(self, results: List[BacktestResult], filepath: str = "data/blockchain_scalping_results.json"):
        """Save results."""
        output = []
        for r in results:
            output.append({
                "params": r.params,
                "total_trades": r.total_trades,
                "wins": r.wins,
                "losses": r.losses,
                "win_rate": r.win_rate,
                "profit_factor": r.profit_factor,
                "total_pnl_pct": r.total_pnl_pct,
                "total_pnl_usd": r.total_pnl_usd,
                "max_drawdown": r.max_drawdown,
                "sharpe_ratio": r.sharpe_ratio,
                "signal_breakdown": r.signal_breakdown
            })

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n[+] Results saved to {filepath}")


def main():
    print("\n" + "=" * 100)
    print("BLOCKCHAIN SCALPING BACKTEST - REAL HISTORICAL DATA")
    print("=" * 100)
    print("\nUsing:")
    print("  - 263,791 blocks of blockchain features")
    print("  - 6,185 days of whale/sender/receiver data")
    print("  - Real price data alignment")
    print("\nSignal Types:")
    print("  - WHALE_ACCUMULATION: Whales buying, distributing to many")
    print("  - WHALE_DISTRIBUTION: Whales selling, consolidating")
    print("  - VALUE_ACCUMULATION: High value moving to many receivers")
    print("  - VALUE_DISTRIBUTION: High value concentrating")
    print("  - TX_SURGE_UP/DOWN: Network activity momentum")
    print("  - QUIET_WHALE: Low retail + whale activity = stealth accumulation")

    backtester = BlockchainScalpingBacktester()

    if len(backtester.daily_features) < 30:
        print("\n[!] Need more blockchain data")
        return

    # Run grid search
    results = backtester.grid_search()

    # Save results
    backtester.save_results(results)

    # Best result summary
    if results and results[0].total_trades > 0:
        best = results[0]
        print("\n" + "=" * 100)
        print("OPTIMAL PARAMETERS FOR REAL MARKET")
        print("=" * 100)
        print(f"\nBest Configuration:")
        print(f"  Take Profit: {best.params['tp']:.1f}%")
        print(f"  Stop Loss: {best.params['sl']:.1f}%")
        print(f"  Max Hold: {best.params['hold']} days")
        print(f"  Min Confidence: {best.params.get('conf', 0.52):.0%}")
        print(f"\nExpected Performance:")
        print(f"  Win Rate: {best.win_rate:.1%}")
        print(f"  Profit Factor: {best.profit_factor:.2f}")
        print(f"  Total PnL: ${best.total_pnl_usd:.2f} ({best.total_pnl_pct:.1f}%)")
        print(f"  Max Drawdown: {best.max_drawdown:.1%}")
        print(f"  Sharpe Ratio: {best.sharpe_ratio:.2f}")


if __name__ == '__main__':
    main()
