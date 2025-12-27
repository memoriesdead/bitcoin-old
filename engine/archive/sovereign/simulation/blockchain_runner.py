#!/usr/bin/env python3
"""
BLOCKCHAIN LIVE TRADING RUNNER - The Real Edge.

Uses ZMQ to get blockchain flow data 10-60 seconds BEFORE the market sees it.
- INFLOW to exchange = Selling pressure = SHORT
- OUTFLOW from exchange = Accumulation = LONG

Dual Formula Engines:
- Adaptive (10001-10005): Flow impact, timing, regime, Bayesian, multi-timescale
- Pattern Recognition (20001-20012): HMM, patterns, stat arb, changepoint, ML

Target: 50.75% win rate after fees.
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, Optional, List
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from engine.sovereign.blockchain.formula_connector import FormulaConnector
from engine.sovereign.simulation.fees import EXCHANGE_FEES, get_slippage_estimate
from engine.sovereign.simulation.trade_logger import TradeLogger
from engine.sovereign.simulation.rentech_integration import (
    create_rentech_suite,
    RenTechSignalProcessor,
    StatArbDetector,
    ExecutionOptimizer,
    RenTechFilter,
)


class BlockchainLiveRunner:
    """
    Production live trading using blockchain flow signals.

    THE EDGE: We see Bitcoin transactions 10-60 seconds before they
    affect price on exchanges.

    - ZMQ from Bitcoin Core gets raw transactions
    - Exchange flow detector identifies inflows/outflows
    - Dual formula engines generate trading signals
    - Fee-aware execution with real exchange costs

    Target: 50.75% win rate after ALL costs.
    """

    def __init__(
        self,
        initial_capital: float = 100.0,
        kelly_fraction: float = 0.25,
        exchange: str = "binance_us",
        zmq_endpoint: str = "tcp://127.0.0.1:28332",
        db_path: str = "data/blockchain_trades.db",
        enable_pattern_recognition: bool = True,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.kelly_fraction = kelly_fraction
        self.exchange = exchange
        self.fees = EXCHANGE_FEES.get(exchange, EXCHANGE_FEES['binance_us'])
        self.db_path = db_path

        # Initialize FormulaConnector (connects blockchain to formulas)
        self.connector = FormulaConnector(
            zmq_endpoint=zmq_endpoint,
            on_signal=self._on_signal,
            enable_pattern_recognition=enable_pattern_recognition,
        )

        # Trade logging
        self.logger = TradeLogger(db_path)

        # State
        self.running = False
        self.start_time = None
        self.current_price = 95000.0  # Will be updated from exchange

        # Position tracking
        self.positions: Dict[str, Dict] = {}
        self.max_positions = 5
        self.max_position_pct = 0.20

        # Fee tracking
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0

        # Stats
        self.signals_received = 0
        self.trades_executed = 0
        self.wins = 0
        self.losses = 0

        # Price feed (from exchange API)
        self._price_lock = threading.Lock()
        self._price_thread = None

        # === RENTECH INTEGRATION (from GitHub open-source research) ===
        # hftbacktest + QLib + FinRL concepts combined
        self.rentech = create_rentech_suite(
            lookback=100,
            min_edge_multiple=2.5,  # Edge must be 2.5x costs
            min_confidence=0.7,
            min_flow_btc=5.0,
        )

        # Track learning performance
        self.cumulative_alpha = 0.0
        self.drawdown = 0.0
        self.peak_capital = initial_capital

    def _on_signal(self, signal: Dict):
        """
        Called when FormulaConnector generates a trading signal.

        RENTECH PRINCIPLES (integrated from GitHub open-source):
        1. Edge > Costs: Only trade when expected profit > 2.5x fees
        2. Flow magnitude: Minimum 5 BTC flow to trigger trade
        3. Ensemble agreement: Both engines must agree for high conviction
        4. Dynamic hold time: Bigger flow = longer hold for price impact
        5. QLib point-in-time: No lookahead, alpha decay
        6. FinRL state: Optimal position sizing with drawdown adjustment
        7. Stat arb: Mean reversion detection
        """
        self.signals_received += 1

        # Check if we should trade
        if len(self.positions) >= self.max_positions:
            return

        btc_flow = abs(signal.get('btc_amount', 0))
        round_trip_fee = self.fees.round_trip_taker

        # === RENTECH INTEGRATION: Update signal processor (QLib point-in-time) ===
        self.rentech['signal_processor'].update(
            price=self.current_price,
            flow=signal,
            timestamp=time.time()
        )

        # === RENTECH INTEGRATION: Update stat arb detector ===
        net_flow = signal.get('btc_amount', 0) * signal.get('direction', 0)
        self.rentech['stat_arb'].update(self.current_price, net_flow)

        # === RENTECH FILTER: Apply all filters at once ===
        expected_move = btc_flow * 0.001  # 0.1% per 100 BTC
        should_trade, reason = self.rentech['filter'].should_trade(
            signal=signal,
            round_trip_fee=round_trip_fee,
            expected_move=expected_move
        )

        if not should_trade:
            if btc_flow >= 2.0:  # Only log significant rejections
                print(f"[RENTECH FILTER] {reason}")
            return

        # === CHECK STAT ARB SIGNAL (extra confirmation) ===
        stat_arb_dir, z_score, half_life = self.rentech['stat_arb'].get_signal()
        if stat_arb_dir != 0 and stat_arb_dir != signal['direction']:
            # Stat arb conflicts with flow signal - reduce confidence
            if z_score > 2.5:  # Strong stat arb signal
                print(f"[STAT ARB CONFLICT] z={z_score:.2f}, reducing size")
                btc_flow *= 0.5  # Reduce effective flow

        # === RENTECH POSITION SIZING (FinRL optimal Kelly) ===
        # Calculate optimal size with drawdown adjustment
        self.drawdown = max(0, (self.peak_capital - self.capital) / self.peak_capital)

        optimal_size = self.rentech['executor'].position_size_optimal(
            capital=self.capital,
            kelly_fraction=self.kelly_fraction,
            win_probability=0.55,  # Our target
            avg_win=0.005,         # 0.5% avg win
            avg_loss=0.003,        # 0.3% avg loss
            current_drawdown=self.drawdown
        )

        # Scale with flow magnitude and confidence
        flow_multiplier = min(btc_flow / 10.0, 3.0)
        confidence_multiplier = signal['confidence']

        position_size_pct = min(
            optimal_size * flow_multiplier * confidence_multiplier,
            self.max_position_pct
        )
        position_usd = self.capital * position_size_pct

        if position_usd < 1:
            return

        # Calculate entry costs
        fee_usd = position_usd * self.fees.taker_fee
        slippage_usd = position_usd * get_slippage_estimate(position_usd)
        total_cost = fee_usd + slippage_usd

        # Deduct entry fee
        self.capital -= fee_usd
        self.total_fees_paid += fee_usd
        self.total_slippage_cost += slippage_usd

        # === RENTECH STOP/PROFIT: Based on fees, not fixed ===
        # Stop loss: 2x round-trip fees (room to breathe)
        # Take profit: 4x round-trip fees (2:1 reward/risk)
        stop_loss_pct = max(round_trip_fee * 2, 0.003)  # Min 0.3%
        take_profit_pct = max(round_trip_fee * 4, 0.006)  # Min 0.6%

        # === RENTECH HOLD TIME: Based on flow size ===
        # Bigger flows take longer to impact price
        # 5 BTC = 60s, 20 BTC = 120s, 50+ BTC = 300s
        base_hold = 60
        hold_time = min(base_hold + (btc_flow * 3), 300)  # 60-300 seconds

        # Create position
        trade_id = f"BC_{int(time.time()*1000)}"
        entry_price = self.current_price * (1 + get_slippage_estimate(position_usd) * signal['direction'])
        ensemble_type = signal.get('ensemble_type', 'unknown')

        position = {
            'trade_id': trade_id,
            'direction': signal['direction'],
            'entry_price': entry_price,
            'entry_time': time.time(),
            'position_usd': position_usd,
            'position_size': position_usd,  # Alias for adaptive.py compatibility
            'position_btc': position_usd / entry_price,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'max_hold_time': hold_time,
            'entry_delay': signal.get('entry_delay', 10),
            'confidence': signal['confidence'],
            'regime': signal.get('regime', 'UNKNOWN'),
            'ensemble_type': ensemble_type,
            'btc_flow': btc_flow,
        }

        self.positions[trade_id] = position
        self.trades_executed += 1

        # Log to database
        try:
            from engine.sovereign.simulation.types import FormulaSignal
            sig = FormulaSignal(
                formula_id=signal.get('formula_id', 10001),
                formula_name=signal.get('formula_name', 'blockchain_flow'),
                direction=signal['direction'],
                confidence=signal['confidence'],
                position_size_pct=position_size_pct,
                stop_loss_pct=position['stop_loss_pct'],
                take_profit_pct=position['take_profit_pct'],
                max_hold_seconds=signal.get('hold_time', 30),
            )
            db_trade_id = self.logger.log_signal(sig, entry_price)
            position['db_trade_id'] = db_trade_id
            self.logger.log_entry(db_trade_id, entry_price, position['position_btc'], position_usd)
        except Exception as e:
            print(f"[DB ERROR] {e}")

        dir_str = "LONG" if signal['direction'] == 1 else "SHORT"
        print(f"[TRADE] {dir_str} @ ${entry_price:,.2f} | "
              f"${position_usd:.2f} ({position_size_pct*100:.1f}%) | "
              f"flow={btc_flow:.1f}BTC | hold={hold_time:.0f}s | "
              f"SL={stop_loss_pct*100:.2f}%/TP={take_profit_pct*100:.2f}%")

    def _check_exits(self):
        """Check all positions for exit conditions."""
        now = time.time()
        price = self.current_price

        if self.positions and now - getattr(self, '_last_exit_check_log', 0) > 10:
            pos = list(self.positions.values())[0]
            age = now - pos['entry_time']
            hold = pos['max_hold_time']
            print(f"[EXIT_CHECK] {len(self.positions)} positions, price=${price:,.2f}, age={age:.1f}s, hold={hold}s")
            self._last_exit_check_log = now

        positions_to_close = []

        for trade_id, pos in list(self.positions.items()):
            exit_reason = None

            # Calculate PnL
            if pos['direction'] == 1:  # LONG
                pnl_pct = (price - pos['entry_price']) / pos['entry_price']
            else:  # SHORT
                pnl_pct = (pos['entry_price'] - price) / pos['entry_price']

            age = now - pos['entry_time']

            # Check stop loss
            if pnl_pct <= -pos['stop_loss_pct']:
                exit_reason = 'stop_loss'
            # Check take profit
            elif pnl_pct >= pos['take_profit_pct']:
                exit_reason = 'take_profit'
            # Check time exit
            elif age > pos['max_hold_time']:
                exit_reason = 'time_exit'

            if exit_reason:
                print(f"[EXIT_TRIGGER] {trade_id[:8]} | {exit_reason} | age={age:.1f}s | pnl={pnl_pct*100:.2f}%")
                positions_to_close.append((trade_id, pos, exit_reason, pnl_pct))

        for trade_id, pos, exit_reason, pnl_pct in positions_to_close:
            self._close_position(trade_id, pos, exit_reason, pnl_pct)

    def _close_position(self, trade_id: str, pos: Dict, exit_reason: str, pnl_pct: float):
        """Close position with fee deduction."""
        exit_price = self.current_price
        exit_value = pos['position_btc'] * exit_price

        # Calculate exit costs
        fee_usd = exit_value * self.fees.taker_fee
        slippage_usd = exit_value * get_slippage_estimate(exit_value)

        # Deduct exit fee
        self.capital -= fee_usd
        self.total_fees_paid += fee_usd
        self.total_slippage_cost += slippage_usd

        # Calculate PnL
        pnl_usd = pos['position_usd'] * pnl_pct
        self.capital += pnl_usd

        # Track win/loss
        if pnl_usd > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Update peak capital for drawdown tracking
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        # Record outcome for formula learning
        was_profitable = pnl_usd > 0
        try:
            self.connector.record_trade_result(pos, exit_price)
        except Exception as e:
            print(f"[LEARNING ERROR] {e}")

        # Log exit to database
        try:
            if 'db_trade_id' in pos:
                self.logger.log_exit(pos['db_trade_id'], exit_price, exit_reason, pnl_usd, pnl_pct)
        except Exception as e:
            print(f"[DB ERROR] {e}")

        # Remove position
        del self.positions[trade_id]

        dir_str = "LONG" if pos['direction'] == 1 else "SHORT"
        print(f"[EXIT] {dir_str} | {exit_reason} | PnL: ${pnl_usd:+.4f} ({pnl_pct*100:+.2f}%) | "
              f"fee=${fee_usd:.4f}")

    def _update_price(self):
        """Update price from exchange (background thread)."""
        import requests

        while self.running:
            try:
                # Try Coinbase first
                resp = requests.get(
                    "https://api.coinbase.com/v2/prices/BTC-USD/spot",
                    timeout=5
                )
                if resp.status_code == 200:
                    data = resp.json()
                    with self._price_lock:
                        self.current_price = float(data['data']['amount'])
                    self.connector.set_reference_price(self.current_price)
            except Exception:
                pass

            time.sleep(5)  # Update every 5 seconds

    def run(self, duration_seconds: float = None):
        """Run blockchain live trading."""
        self.running = True
        self.start_time = time.time()

        end_time = time.time() + duration_seconds if duration_seconds else None

        # Print header
        print("=" * 70)
        print("BLOCKCHAIN LIVE TRADING - THE REAL EDGE")
        print("=" * 70)
        print(f"Started:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Exchange:        {self.fees.name}")
        print(f"Fee Rate:        {self.fees.taker_fee*100:.2f}% (round-trip: {self.fees.round_trip_taker*100:.2f}%)")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Kelly Fraction:  {self.kelly_fraction}")
        print(f"Duration:        {'Indefinite' if duration_seconds is None else f'{duration_seconds}s'}")
        print(f"Target Win Rate: 50.75% (after fees)")
        print("=" * 70)
        print()
        print("EDGE: Blockchain flow data 10-60 seconds before market")
        print("  INFLOW to exchange  = SHORT (selling pressure)")
        print("  OUTFLOW from exchange = LONG (accumulation)")
        print()
        print("Formula Engines:")
        print("  Adaptive (10001-10005): Flow impact, timing, regime, Bayesian")
        print("  Pattern (20001-20012):  HMM, patterns, stat arb, changepoint")
        print("=" * 70)

        # Create session
        session = self.logger.create_session(
            mode="blockchain_live",
            initial_capital=self.initial_capital,
            kelly_fraction=self.kelly_fraction,
            formula_ids=list(range(10001, 10006)) + list(range(20001, 20013)),
        )

        # Start price feed
        self._price_thread = threading.Thread(target=self._update_price, daemon=True)
        self._price_thread.start()

        # Start FormulaConnector (connects to ZMQ)
        if not self.connector.start():
            print("[ERROR] Failed to start FormulaConnector. Is Bitcoin Core running with ZMQ?")
            print("        Check: bitcoind -zmqpubrawtx=tcp://127.0.0.1:28332")
            return

        print("[ZMQ] Connected to Bitcoin Core")
        print("[LIVE] Waiting for blockchain flow signals...")
        print()

        last_status = time.time()

        try:
            while self.running:
                if end_time and time.time() >= end_time:
                    print("\n[LIVE] Duration complete")
                    break

                # Check exits
                self._check_exits()

                # Check pending entries
                entries = self.connector.check_entries(self.current_price)

                # Print status every 60 seconds
                now = time.time()
                if now - last_status >= 60:
                    self._print_status()
                    last_status = now

                time.sleep(0.1)  # 100ms loop

        except KeyboardInterrupt:
            print("\n[LIVE] Interrupted by user")
        finally:
            self._shutdown()

    def _print_status(self):
        """Print current status."""
        elapsed = time.time() - self.start_time
        stats = self.connector.get_stats()

        win_rate = self.wins / (self.wins + self.losses) if (self.wins + self.losses) > 0 else 0
        net_pnl = self.capital - self.initial_capital

        print(f"\n[STATUS] {datetime.now().strftime('%H:%M:%S')} | "
              f"Price: ${self.current_price:,.2f} | "
              f"Capital: ${self.capital:.2f} | "
              f"Net PnL: ${net_pnl:+.4f}")
        print(f"         Signals: {self.signals_received} | "
              f"Trades: {self.trades_executed} | "
              f"Win Rate: {win_rate*100:.1f}% ({self.wins}W/{self.losses}L) | "
              f"Fees: ${self.total_fees_paid:.4f}")
        print(f"         ZMQ TXs: {stats.get('feed', {}).get('tx_count', 0)} | "
              f"Flows: {stats.get('feed', {}).get('flows_detected', 0)} | "
              f"Regime: {stats.get('adaptive_engine', {}).get('regime', 'N/A')}")

        # RenTech filter stats
        filter_stats = self.rentech['filter'].get_stats()
        print(f"[RENTECH] Filter Pass Rate: {filter_stats['pass_rate']*100:.1f}% | "
              f"Alpha: {self.rentech['signal_processor'].alpha_composite:.2f} | "
              f"Drawdown: {self.drawdown*100:.2f}%")

        if self.positions:
            print(f"[POSITIONS] {len(self.positions)} active:")
            for trade_id, pos in self.positions.items():
                if pos['direction'] == 1:
                    pnl_pct = (self.current_price - pos['entry_price']) / pos['entry_price']
                else:
                    pnl_pct = (pos['entry_price'] - self.current_price) / pos['entry_price']
                dir_str = "LONG" if pos['direction'] == 1 else "SHORT"
                print(f"  {trade_id[:12]}: {dir_str} @ ${pos['entry_price']:,.2f} ({pnl_pct*100:+.2f}%)")

    def _shutdown(self):
        """Clean shutdown."""
        self.running = False

        # Stop connector
        self.connector.stop()

        # Close all positions
        for trade_id, pos in list(self.positions.items()):
            if pos['direction'] == 1:
                pnl_pct = (self.current_price - pos['entry_price']) / pos['entry_price']
            else:
                pnl_pct = (pos['entry_price'] - self.current_price) / pos['entry_price']
            self._close_position(trade_id, pos, "session_end", pnl_pct)

        elapsed = time.time() - self.start_time if self.start_time else 0

        # Close session
        self.logger.close_session(
            final_capital=self.capital,
            max_drawdown=0,
            sharpe_ratio=None,
        )

        # Final results
        gross_pnl = self.capital - self.initial_capital + self.total_fees_paid + self.total_slippage_cost
        net_pnl = self.capital - self.initial_capital
        win_rate = self.wins / (self.wins + self.losses) if (self.wins + self.losses) > 0 else 0

        print("\n" + "=" * 70)
        print("BLOCKCHAIN LIVE TRADING COMPLETE")
        print("=" * 70)
        print(f"Duration:        {elapsed/3600:.2f} hours")
        print(f"Exchange:        {self.exchange}")
        print(f"\nCapital:")
        print(f"  Initial:       ${self.initial_capital:.2f}")
        print(f"  Final:         ${self.capital:.2f}")
        print(f"\nPnL Breakdown:")
        print(f"  Gross PnL:     ${gross_pnl:+.4f}")
        print(f"  Fees Paid:     -${self.total_fees_paid:.4f}")
        print(f"  Slippage:      -${self.total_slippage_cost:.4f}")
        print(f"  NET PnL:       ${net_pnl:+.4f}")
        print(f"\nTrades:")
        print(f"  Total:         {self.trades_executed}")
        print(f"  Wins:          {self.wins}")
        print(f"  Losses:        {self.losses}")
        print(f"  Win Rate:      {win_rate*100:.2f}%")
        print(f"\nTarget: 50.75% | Actual: {win_rate*100:.2f}%")
        if win_rate > 0.5075:
            print(f"STATUS: BEATING TARGET BY {(win_rate - 0.5075)*100:.2f}%")

        # RenTech integration stats
        filter_stats = self.rentech['filter'].get_stats()
        print(f"\nRenTech Integration Stats:")
        print(f"  Filter Pass Rate: {filter_stats['pass_rate']*100:.1f}%")
        print(f"  Final Alpha:      {self.rentech['signal_processor'].alpha_composite:.2f}")
        print(f"  Peak Drawdown:    {self.drawdown*100:.2f}%")
        if filter_stats['filter_breakdown']:
            print(f"  Filter Breakdown:")
            for reason, count in filter_stats['filter_breakdown'].items():
                print(f"    {reason}: {count}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Blockchain Live Trading")

    parser.add_argument('--capital', type=float, default=100.0,
                        help='Initial capital (default: 100)')
    parser.add_argument('--kelly', type=float, default=0.25,
                        help='Kelly fraction (default: 0.25)')
    parser.add_argument('--exchange', type=str, default='binance_us',
                        choices=list(EXCHANGE_FEES.keys()),
                        help='Exchange for fee calculation (default: binance_us)')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration in seconds (default: indefinite)')
    parser.add_argument('--zmq', type=str, default='tcp://127.0.0.1:28332',
                        help='Bitcoin Core ZMQ endpoint')
    parser.add_argument('--db-path', type=str, default='data/blockchain_trades.db',
                        help='Trade database path')
    parser.add_argument('--no-pattern', action='store_true',
                        help='Disable pattern recognition engine')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("BLOCKCHAIN LIVE TRADING - THE REAL EDGE")
    print("=" * 70)
    print(f"\nAvailable exchanges and fees:")
    for name, fees in EXCHANGE_FEES.items():
        print(f"  {name}: {fees.taker_fee*100:.2f}% taker, {fees.round_trip_taker*100:.2f}% round-trip")
    print(f"\nUsing: {args.exchange}")
    print(f"ZMQ Endpoint: {args.zmq}")
    print()

    runner = BlockchainLiveRunner(
        initial_capital=args.capital,
        kelly_fraction=args.kelly,
        exchange=args.exchange,
        zmq_endpoint=args.zmq,
        db_path=args.db_path,
        enable_pattern_recognition=not args.no_pattern,
    )

    runner.run(duration_seconds=args.duration)


if __name__ == '__main__':
    main()
