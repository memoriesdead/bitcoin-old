#!/usr/bin/env python3
"""
LIVE TRADING RUNNER WITH FEES - True 1:1 Production.

Deducts real exchange fees and slippage from every trade.
Target: 50.75% win rate after all costs.
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from engine.sovereign.simulation.engine_with_fees import SimulationEngineWithFees, EngineConfigWithFees
from engine.sovereign.simulation.live import LivePaperTrader, ExchangeFeed, LiveTick
from engine.sovereign.simulation.formula_engine import PRODUCTION_FORMULA_IDS
from engine.sovereign.simulation.fees import EXCHANGE_FEES


class BlockchainDataFeed:
    """Optional blockchain data from Bitcoin node."""

    def __init__(self, rpc_host: str = "127.0.0.1", rpc_port: int = 8332,
                 rpc_user: str = "bitcoin", rpc_pass: str = "bitcoin"):
        try:
            from engine.sovereign.blockchain.rpc import BitcoinRPC
            self.rpc = BitcoinRPC(rpc_host, rpc_port, rpc_user, rpc_pass)
            self.enabled = self.rpc.test_connection()
            if self.enabled:
                print(f"[BLOCKCHAIN] Connected to Bitcoin node")
        except Exception as e:
            print(f"[BLOCKCHAIN] Not available: {e}")
            self.rpc = None
            self.enabled = False

    def get_data(self) -> Dict:
        if not self.enabled:
            return {}
        try:
            mempool = self.rpc.getmempoolinfo()
            blockchain = self.rpc.getblockchaininfo()
            return {
                'block_height': blockchain.get('blocks', 0),
                'mempool_size': mempool.get('size', 0),
                'mempool_bytes': mempool.get('bytes', 0),
                'mempool_fee': mempool.get('mempoolminfee', 0),
                'difficulty': blockchain.get('difficulty', 0),
            }
        except Exception as e:
            print(f"[BLOCKCHAIN] Error: {e}")
            return {}


class LiveTradingRunnerWithFees:
    """
    Production live trading with real fee accounting.

    Every trade deducts:
    - Entry fee (maker or taker)
    - Exit fee (maker or taker)
    - Slippage estimate

    Target: 50.75% win rate after ALL costs.
    """

    def __init__(
        self,
        initial_capital: float = 100.0,
        kelly_fraction: float = 0.25,
        poll_interval: float = 1.0,
        exchange: str = "binance_us",
        rpc_host: str = "127.0.0.1",
        rpc_port: int = 8332,
        rpc_user: str = "bitcoin",
        rpc_pass: str = "bitcoin",
        db_path: str = "data/live_trades_fees.db",
    ):
        self.poll_interval = poll_interval

        # Initialize fee-aware engine
        self.config = EngineConfigWithFees(
            initial_capital=initial_capital,
            kelly_fraction=kelly_fraction,
            exchange=exchange,
            use_taker_fees=True,
            include_slippage=True,
            db_path=db_path,
        )
        self.engine = SimulationEngineWithFees(self.config)

        # Initialize feeds
        self.price_feed = ExchangeFeed(primary_exchange='coinbase')
        self.blockchain_feed = BlockchainDataFeed(rpc_host, rpc_port, rpc_user, rpc_pass)

        # State
        self.running = False
        self.tick_count = 0
        self.last_price = 0.0
        self.start_time = None

    def run(self, duration_seconds: float = None):
        """Run live trading with fee accounting."""
        self.running = True
        self.start_time = time.time()
        self.tick_count = 0

        end_time = time.time() + duration_seconds if duration_seconds else None
        fees = self.config.fees

        # Print header
        print("=" * 70)
        print("TRUE 1:1 LIVE TRADING - WITH REAL FEES")
        print("=" * 70)
        print(f"Started:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Exchange:        {fees.name}")
        print(f"Fee Rate:        {fees.taker_fee*100:.2f}% taker (round-trip: {fees.round_trip_taker*100:.2f}%)")
        print(f"Initial Capital: ${self.config.initial_capital:.2f}")
        print(f"Kelly Fraction:  {self.config.kelly_fraction}")
        print(f"Formulas:        {len(PRODUCTION_FORMULA_IDS)}")
        print(f"Blockchain:      {'Connected' if self.blockchain_feed.enabled else 'Disabled'}")
        print(f"Duration:        {'Indefinite' if duration_seconds is None else f'{duration_seconds}s'}")
        print(f"Target Win Rate: 50.75% (after fees)")
        print("=" * 70)

        # Create session
        from engine.sovereign.simulation.trade_logger import TradeLogger
        self.engine.logger = TradeLogger(self.config.db_path)
        session = self.engine.logger.create_session(
            mode="live_with_fees",
            initial_capital=self.config.initial_capital,
            kelly_fraction=self.config.kelly_fraction,
            formula_ids=PRODUCTION_FORMULA_IDS,
        )

        # Initialize verifier
        from engine.sovereign.simulation.verifier import ExchangeVerifier
        self.engine.verifier = ExchangeVerifier()

        last_status = time.time()

        try:
            while self.running:
                if end_time and time.time() >= end_time:
                    print("\n[LIVE] Duration complete")
                    break

                tick = self.price_feed.get_price()
                if not tick:
                    print("[LIVE] No price data, retrying...")
                    time.sleep(self.poll_interval)
                    continue

                blockchain_data = self.blockchain_feed.get_data()

                data = tick.to_dict()
                data.update(blockchain_data)

                self.engine._process_tick(tick.timestamp, tick.price, data)

                self.last_price = tick.price
                self.tick_count += 1

                now = time.time()
                if now - last_status >= 60:
                    self._print_status()
                    last_status = now

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            print("\n[LIVE] Interrupted by user")
        finally:
            self._shutdown()

    def _print_status(self):
        """Print status with fee tracking."""
        elapsed = time.time() - self.start_time
        session = self.engine.logger.session

        net_pnl = self.engine.capital - self.config.initial_capital

        print(f"\n[STATUS] {datetime.now().strftime('%H:%M:%S')} | "
              f"Price: ${self.last_price:,.2f} | "
              f"Capital: ${self.engine.capital:.2f} | "
              f"Fees Paid: ${self.engine.total_fees_paid:.4f} | "
              f"Net PnL: ${net_pnl:+.4f}")

        if self.engine.positions:
            print(f"[POSITIONS] {len(self.engine.positions)} active:")
            for trade_id, pos in self.engine.positions.items():
                pnl_pct = ((self.last_price / pos.entry_price) - 1) * 100 if pos.direction == 1 else ((pos.entry_price / self.last_price) - 1) * 100
                print(f"  {trade_id[:8]}: {pos.formula_name} {'LONG' if pos.direction == 1 else 'SHORT'} "
                      f"@ ${pos.entry_price:,.2f} ({pnl_pct:+.2f}%)")

    def _shutdown(self):
        """Clean shutdown with fee summary."""
        self.running = False

        if self.last_price > 0:
            self.engine._close_all_positions(self.last_price, time.time())

        elapsed = time.time() - self.start_time if self.start_time else 0
        session = self.engine.logger.session

        self.engine.logger.close_session(
            final_capital=self.engine.capital,
            max_drawdown=0,
            sharpe_ratio=None,
        )

        # Final results with fee breakdown
        gross_pnl = self.engine.capital - self.config.initial_capital + self.engine.total_fees_paid + self.engine.total_slippage_cost
        net_pnl = self.engine.capital - self.config.initial_capital

        print("\n" + "=" * 70)
        print("TRUE 1:1 LIVE TRADING COMPLETE")
        print("=" * 70)
        print(f"Duration:        {elapsed/3600:.2f} hours")
        print(f"Exchange:        {self.config.exchange}")
        print(f"\nCapital:")
        print(f"  Initial:       ${self.config.initial_capital:.2f}")
        print(f"  Final:         ${self.engine.capital:.2f}")
        print(f"\nPnL Breakdown:")
        print(f"  Gross PnL:     ${gross_pnl:+.4f}")
        print(f"  Fees Paid:     -${self.engine.total_fees_paid:.4f}")
        print(f"  Slippage:      -${self.engine.total_slippage_cost:.4f}")
        print(f"  NET PnL:       ${net_pnl:+.4f}")
        if session:
            print(f"\nTrades:")
            print(f"  Total:         {session.total_trades}")
            print(f"  Win Rate:      {session.win_rate*100:.2f}%" if session.win_rate else "  Win Rate: N/A")
            print(f"\nTarget: 50.75% | Actual: {session.win_rate*100:.2f}%" if session.win_rate else "")
            if session.win_rate and session.win_rate > 0.5075:
                print(f"STATUS: BEATING TARGET BY {(session.win_rate - 0.5075)*100:.2f}%")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="True 1:1 Live Trading with Fees")

    parser.add_argument('--capital', type=float, default=100.0,
                        help='Initial capital (default: 100)')
    parser.add_argument('--kelly', type=float, default=0.25,
                        help='Kelly fraction (default: 0.25)')
    parser.add_argument('--exchange', type=str, default='binance_us',
                        choices=list(EXCHANGE_FEES.keys()),
                        help='Exchange for fee calculation (default: binance_us)')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration in seconds (default: indefinite)')
    parser.add_argument('--poll-interval', type=float, default=1.0,
                        help='Price poll interval (default: 1.0)')
    parser.add_argument('--rpc-host', type=str, default='127.0.0.1',
                        help='Bitcoin RPC host')
    parser.add_argument('--rpc-port', type=int, default=8332,
                        help='Bitcoin RPC port')
    parser.add_argument('--rpc-user', type=str, default='bitcoin',
                        help='Bitcoin RPC user')
    parser.add_argument('--rpc-pass', type=str, default='bitcoin',
                        help='Bitcoin RPC password')
    parser.add_argument('--db-path', type=str, default='data/live_trades_fees.db',
                        help='Trade database path')

    args = parser.parse_args()

    print(f"\nAvailable exchanges and fees:")
    for name, fees in EXCHANGE_FEES.items():
        print(f"  {name}: {fees.taker_fee*100:.2f}% taker, {fees.round_trip_taker*100:.2f}% round-trip")
    print(f"\nUsing: {args.exchange}\n")

    runner = LiveTradingRunnerWithFees(
        initial_capital=args.capital,
        kelly_fraction=args.kelly,
        poll_interval=args.poll_interval,
        exchange=args.exchange,
        rpc_host=args.rpc_host,
        rpc_port=args.rpc_port,
        rpc_user=args.rpc_user,
        rpc_pass=args.rpc_pass,
        db_path=args.db_path,
    )

    runner.run(duration_seconds=args.duration)


if __name__ == '__main__':
    main()
