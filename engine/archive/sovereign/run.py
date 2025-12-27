#!/usr/bin/env python3
"""
Sovereign Engine - RENAISSANCE PATTERN RECOGNITION
==================================================
Run with: python -m engine.sovereign.run --capital 100 --duration 300

THE EDGE:
- INFLOW to exchange = They will SELL = SHORT
- OUTFLOW from exchange = They are accumulating = LONG

TWO FORMULA ENGINES:
1. ADAPTIVE FORMULAS (IDs 10001-10005):
   - Flow impact estimator, timing optimizer, regime detector
   - All parameters self-calibrate from actual trade outcomes

2. PATTERN RECOGNITION (IDs 20001-20012) - RENAISSANCE STYLE:
   - Hidden Markov Models (HMM) for regime detection
   - Statistical Arbitrage on flow imbalances
   - Changepoint detection (CUSUM + Bayesian)
   - Ensemble voting (Condorcet)

ENSEMBLE: Both engines vote on every signal. Agreement = higher confidence.

"We adapt to ALL POSSIBILITIES mathematically."
"""

import sys
import os
import time
import argparse
import threading
import math
from typing import Dict, Optional, List

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from engine.sovereign.blockchain import FormulaConnector, ExchangeTick
from engine.sovereign.strategy.signal_engine import ExchangeSignalEngine
from engine.sovereign.strategy.renaissance import RenaissanceTracker


class BlockchainFlowRunner:
    """
    Renaissance-style blockchain flow trading.

    TWO FORMULA ENGINES:
    1. Adaptive (10001-10005) - flow impact, timing, regime
    2. Pattern Recognition (20001-20012) - HMM, stat arb, changepoint

    INFLOW to exchange = SHORT
    OUTFLOW from exchange = LONG

    All parameters learn from actual trade outcomes.
    """

    TRADING_EXCHANGES = ['gemini', 'coinbase', 'kraken', 'bitstamp', 'binance']

    def __init__(self, initial_capital: float = 100.0):
        self.initial_capital = initial_capital
        self.reference_price = self._fetch_live_price()

        # FORMULA CONNECTOR - connects blockchain feed to BOTH engines
        # Engine 1: Adaptive (10001-10005)
        # Engine 2: Pattern Recognition (20001-20012)
        self.connector = FormulaConnector(
            on_signal=self._on_signal,
            enable_pattern_recognition=True
        )
        self.connector.set_reference_price(self.reference_price)

        # Per-exchange engines
        self.engines: Dict[str, ExchangeSignalEngine] = {}
        per_exchange = initial_capital / len(self.TRADING_EXCHANGES)
        for ex in self.TRADING_EXCHANGES:
            self.engines[ex] = ExchangeSignalEngine(ex, per_exchange)

        # Renaissance tracker
        self.renaissance = RenaissanceTracker(
            initial_capital=initial_capital,
            target_multiplier=100.0,
            kelly_fraction=0.25,
            max_drawdown=0.20
        )

        # Active trades for delayed entry
        self.pending_entries: List[Dict] = []
        self.active_trades: List[Dict] = []

        # Stats
        self.ticks = 0
        self.signals_received = 0
        self.agreement_signals = 0
        self.running = False
        self.start_time = 0.0
        self.lock = threading.Lock()

        self._print_banner()

    def _fetch_live_price(self) -> float:
        """Fetch live BTC price from Coinbase."""
        import json
        from urllib.request import urlopen, Request
        try:
            url = 'https://api.exchange.coinbase.com/products/BTC-USD/ticker'
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                price = float(data['price'])
                print(f"[PRICE] Live: ${price:,.2f}")
                return price
        except Exception as e:
            print(f"[PRICE] Error: {e}, using default")
            return 95000.0

    def _print_banner(self):
        """Print startup banner."""
        per_ex = self.initial_capital / len(self.TRADING_EXCHANGES)
        params = self.connector.engine.bayesian.get()
        print(f"""
================================================================================
SOVEREIGN ENGINE - RENAISSANCE PATTERN RECOGNITION (v10.0)
================================================================================
THE EDGE:
- INFLOW to exchange = SHORT (they will sell)
- OUTFLOW from exchange = LONG (they are accumulating)

ENGINE 1 - ADAPTIVE FORMULAS (10001-10005):
- 10001: Flow Impact Estimator (learns btc->price relationship)
- 10002: Timing Optimizer (learns optimal entry delay & hold time)
- 10003: Regime Detector (adapts to market conditions)
- 10004: Bayesian Updater (tracks parameter uncertainty)
- 10005: Multi-Timescale (aggregates signals across timeframes)

ENGINE 2 - PATTERN RECOGNITION (20001-20012):
- 20001: BlockchainHMM (5-state Hidden Markov Model)
- 20003: StatArbFlowDetector (statistical arbitrage)
- 20004: ChangePointDetector (CUSUM + Bayesian)
- 20011: EnsemblePatternVoter (Condorcet voting)

ENSEMBLE: Both engines vote. Agreement = boosted confidence.

Capital: ${self.initial_capital:.2f} ({len(self.TRADING_EXCHANGES)} exchanges @ ${per_ex:.2f} each)
Price: ${self.reference_price:,.2f}
Entry Delay: {params['entry_delay']:.1f}s (learns)
Hold Time: {params['hold_time']:.1f}s (learns)

"We adapt to ALL POSSIBILITIES mathematically."
================================================================================
""")

    def _on_signal(self, signal: Dict) -> None:
        """
        Callback when FormulaConnector generates an ensemble signal.
        Called for every signal that passes gates from BOTH engines.
        """
        with self.lock:
            self.signals_received += 1
            if signal.get('ensemble_type') == 'agreement':
                self.agreement_signals += 1

        # Log signal details
        dir_str = "LONG" if signal['direction'] == 1 else "SHORT"
        ensemble = signal.get('ensemble_type', 'unknown')
        regime = signal.get('regime', 'UNKNOWN')

        sources = []
        if signal.get('adaptive_signal'):
            sources.append('adaptive')
        if signal.get('pattern_signal'):
            sources.append('pattern')

        print(f"[SIGNAL] {signal['exchange'].upper()} {dir_str} | "
              f"BTC: {signal['btc_amount']:.1f} | Conf: {signal['confidence']:.1%} | "
              f"Type: {ensemble} | Sources: {sources} | Regime: {regime}")

        # Add to pending for delayed entry
        self.pending_entries.append(signal)

    def _main_loop_tick(self) -> None:
        """Called periodically in main loop to check entries and exits."""
        now = time.time()

        # Update price periodically
        new_price = self._fetch_price_fast()
        if new_price > 0:
            self.reference_price = new_price
            self.connector.set_reference_price(new_price)

        # Check pending entries via connector
        entries = self.connector.check_entries(self.reference_price)
        for entry in entries:
            self._execute_delayed_entry(entry)

        # Check active trades for exits
        self._check_exits(now)

        # Check exits on exchange engines too
        for ex, engine in self.engines.items():
            exit_info = engine.check_exit_conditions()
            if exit_info:
                result = engine.force_exit(exit_info['reason'])
                if result:
                    self._record_result(result)

    def _execute_delayed_entry(self, signal: Dict) -> None:
        """Execute trade after optimal delay."""
        engine = self.engines.get(signal['exchange'])
        if not engine:
            return

        dir_str = "LONG" if signal['direction'] == 1 else "SHORT"
        print(f"[ENTRY] {signal['exchange'].upper()} {dir_str} @ ${signal['entry_price']:.2f} | "
              f"Size: {signal['position_size']:.1%} | Delay was {signal['entry_delay']:.0f}s")

        # Add to active trades
        signal['entry_time_actual'] = time.time()
        signal['exit_time'] = signal['entry_time_actual'] + signal['hold_time']
        self.active_trades.append(signal)

        # Execute via engine
        blockchain_signal = {
            'should_trade': True,
            'direction': signal['direction'],
            'confidence': signal['confidence'],
            'strength': signal['confidence'],
            'position_size': signal['position_size'],
        }

        sig = engine.generate_signal(blockchain_signal=blockchain_signal)
        if sig and sig.gate_passed:
            result = engine.execute_trade(sig)
            if result:
                self._record_result(result)
                progress = self.renaissance.get_progress()
                print(f"[TRADE] {signal['exchange'].upper()} {result['type']} @ ${result['price']:.2f} | "
                      f"Size: {result['size']:.6f} BTC | {progress['compound']:.4f}x")

    def _check_exits(self, now: float) -> None:
        """Check active trades for exit conditions."""
        remaining = []
        for trade in self.active_trades:
            should_exit = False
            exit_reason = None

            # Time-based exit
            if now >= trade.get('exit_time', now + 9999):
                should_exit = True
                exit_reason = 'hold_complete'

            # Price-based exits
            if self.reference_price > 0 and trade.get('entry_price', 0) > 0:
                price_change = (self.reference_price - trade['entry_price']) / trade['entry_price']
                pnl = price_change * trade['direction']

                if pnl >= trade.get('take_profit', 0.008):
                    should_exit = True
                    exit_reason = 'take_profit'
                elif pnl <= -trade.get('stop_loss', 0.005):
                    should_exit = True
                    exit_reason = 'stop_loss'

            if should_exit:
                # Record result in BOTH engines via connector
                pnl = self.connector.record_trade_result(trade, self.reference_price)
                dir_str = "LONG" if trade['direction'] == 1 else "SHORT"
                pnl_str = f"+${pnl:.4f}" if pnl >= 0 else f"-${abs(pnl):.4f}"
                ensemble = trade.get('ensemble_type', 'unknown')
                print(f"[EXIT] {trade['exchange'].upper()} {dir_str} | {exit_reason} | "
                      f"PnL: {pnl_str} | Type: {ensemble}")
            else:
                remaining.append(trade)

        self.active_trades = remaining

    def _fetch_price_fast(self) -> float:
        """Fetch price quickly (no logging)."""
        import json
        from urllib.request import urlopen, Request
        try:
            url = 'https://api.exchange.coinbase.com/products/BTC-USD/ticker'
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=5) as response:
                return float(json.loads(response.read().decode())['price'])
        except:
            return 0.0

    def _record_result(self, result: Dict) -> None:
        """Record trade result."""
        pnl = result.get('pnl', 0)
        if pnl != 0:
            total_capital = sum(e.capital for e in self.engines.values())
            self.renaissance.record_trade(pnl, total_capital)

    def run(self, duration: int = 300) -> None:
        """Run the engine."""
        print(f"\n[START] Duration: {duration}s")
        print("[START] Connecting to Bitcoin Core ZMQ...")

        if not self.connector.start():
            print("[ERROR] Failed to connect to ZMQ")
            print("[ERROR] Ensure Bitcoin Core is running with: zmqpubrawtx=tcp://127.0.0.1:28332")
            return

        print("[LIVE] Connected - trading on blockchain flow")
        print("[LIVE] Two engines running: Adaptive (10001-10005) + Pattern Recognition (20001-20012)")
        print()

        self.running = True
        self.start_time = time.time()
        last_status = self.start_time
        last_tick = self.start_time

        try:
            while time.time() - self.start_time < duration and self.running:
                now = time.time()

                # Main loop tick (check entries/exits)
                if now - last_tick >= 1.0:
                    self._main_loop_tick()
                    last_tick = now

                # Status update
                if now - last_status >= 30:
                    self._print_status(duration)
                    last_status = now

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n[STOP] Interrupted")

        finally:
            self.running = False
            self.connector.stop()
            self._print_report()

    def _print_status(self, duration: int) -> None:
        """Print status with both engine stats."""
        elapsed = int(time.time() - self.start_time)
        total_capital = sum(e.capital for e in self.engines.values())
        total_closed = sum(e.closed_trades for e in self.engines.values())
        total_wins = sum(e.wins for e in self.engines.values())
        win_rate = (total_wins / max(1, total_closed)) * 100

        # Get stats from connector
        stats = self.connector.get_stats()
        regime = self.connector.engine.regime.regime
        params = self.connector.engine.bayesian.get()
        pending = stats.get('pending_signals', 0)
        active = len(self.active_trades)

        with self.lock:
            signals = self.signals_received
            agreements = self.agreement_signals

        print(f"[{elapsed:3d}s/{duration}s] ${total_capital:.4f} | "
              f"WR: {win_rate:.1f}% | Regime: {regime} | "
              f"Signals: {signals} (agree: {agreements}) | Active: {active}")

    def _print_report(self) -> None:
        """Print final report with both engine stats."""
        elapsed = time.time() - self.start_time
        total_capital = sum(e.capital for e in self.engines.values())
        total_closed = sum(e.closed_trades for e in self.engines.values())
        total_wins = sum(e.wins for e in self.engines.values())
        total_losses = sum(e.losses for e in self.engines.values())
        total_pnl = sum(e.total_pnl for e in self.engines.values())
        win_rate = (total_wins / max(1, total_closed)) * 100

        # Get comprehensive stats from connector
        stats = self.connector.get_stats()
        feed_stats = stats.get('feed', {})
        progress = self.renaissance.get_progress()

        # Adaptive engine stats
        params = self.connector.engine.bayesian.get()
        params_unc = self.connector.engine.bayesian.get_with_uncertainty()
        regime = self.connector.engine.regime.regime
        adaptive_trades = len(self.connector.engine.trades)
        adaptive_pnl = self.connector.engine.pnl

        # Pattern recognition stats
        pattern_stats = stats.get('pattern_engine', {})

        with self.lock:
            signals = self.signals_received
            agreements = self.agreement_signals

        print("\n" + "="*70)
        print("FINAL REPORT - RENAISSANCE PATTERN RECOGNITION (v10.0)")
        print("="*70)
        print(f"Runtime: {elapsed:.1f}s")
        print(f"Transactions: {feed_stats.get('txs_processed', 0):,}")
        print(f"Ticks: {stats.get('ticks_processed', 0):,}")
        print()
        print(f"Initial: ${self.initial_capital:.2f}")
        print(f"Final: ${total_capital:.4f}")
        print(f"PnL: ${total_pnl:.4f}")
        print(f"Trades: {total_closed}")
        print(f"Wins: {total_wins} | Losses: {total_losses}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Compound: {progress['compound']:.6f}x")
        print()
        print("SIGNAL GENERATION:")
        print(f"  Total Signals: {signals}")
        print(f"  Agreement (both engines): {agreements}")
        print(f"  Adaptive Only: {stats.get('adaptive_signals', 0)}")
        print(f"  Pattern Only: {stats.get('pattern_signals', 0)}")
        print()
        print("ENGINE 1 - ADAPTIVE (Formulas 10001-10005):")
        print(f"  Regime: {regime}")
        print(f"  Trades: {adaptive_trades}")
        print(f"  PnL: ${adaptive_pnl:.4f}")
        print()
        print("ENGINE 2 - PATTERN RECOGNITION (Formulas 20001-20012):")
        print(f"  Signals Generated: {pattern_stats.get('signals_generated', 0)}")
        print(f"  Trades Recorded: {pattern_stats.get('trades_recorded', 0)}")
        voter_stats = pattern_stats.get('voter_stats', {})
        if voter_stats:
            print(f"  Voter Weights: {voter_stats.get('weights', {})}")
        print()
        print("LEARNED PARAMETERS (Bayesian estimates):")
        print(f"  Entry Delay: {params['entry_delay']:.1f}s (n={params_unc['entry_delay']['n']})")
        print(f"  Hold Time: {params['hold_time']:.1f}s (n={params_unc['hold_time']['n']})")
        print(f"  Min BTC: {params['min_btc']:.1f}")
        print(f"  Stop Loss: {params['stop_loss']:.2%}")
        print(f"  Take Profit: {params['take_profit']:.2%}")
        print()
        print("TIMESCALE WEIGHTS (learned):")
        for name, ts in self.connector.engine.multiscale.scales.items():
            acc = self.connector.engine.multiscale.accuracy.get(name, 0.5)
            print(f"  {name:8}: {ts['weight']:.1%} (accuracy: {acc:.1%})")
        print()
        print("Per-Exchange:")
        for ex, engine in self.engines.items():
            s = engine.get_stats()
            print(f"  {ex.upper():12}: ${s['capital']:.4f} | Trades: {s['trades_closed']} | WR: {s['win_rate']:.1f}%")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Sovereign Blockchain Flow Trading')
    parser.add_argument('--capital', type=float, default=100.0, help='Initial capital')
    parser.add_argument('--duration', type=int, default=300, help='Duration in seconds')
    args = parser.parse_args()

    runner = BlockchainFlowRunner(initial_capital=args.capital)
    runner.run(duration=args.duration)


if __name__ == "__main__":
    main()
