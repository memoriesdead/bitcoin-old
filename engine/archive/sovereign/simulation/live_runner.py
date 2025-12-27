#!/usr/bin/env python3
"""
LIVE TRADING RUNNER - Production deployment.

Connects to:
- Exchange APIs for real-time price data
- Bitcoin node RPC for blockchain data (optional)

Runs production formulas 31001-31050 with full audit trail.
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Optional, Dict

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from engine.sovereign.simulation.engine import SimulationEngine, EngineConfig
from engine.sovereign.simulation.live import LivePaperTrader, ExchangeFeed, LiveTick
from engine.sovereign.simulation.formula_engine import PRODUCTION_FORMULA_IDS


class BlockchainDataFeed:
    """
    Optional blockchain data from Bitcoin node.

    Provides additional signals:
    - Mempool size/fees
    - Block intervals
    - Hash rate changes
    """

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
        """Get current blockchain data."""
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


class LiveTradingRunner:
    """
    Production live trading runner.

    Features:
    - Real-time exchange prices
    - Optional blockchain data
    - Full trade audit trail
    - Automatic reconnection
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        kelly_fraction: float = 0.25,
        poll_interval: float = 1.0,
        rpc_host: str = "127.0.0.1",
        rpc_port: int = 8332,
        rpc_user: str = "bitcoin",
        rpc_pass: str = "bitcoin",
        db_path: str = "data/live_trades.db",
    ):
        self.poll_interval = poll_interval

        # Initialize engine
        self.config = EngineConfig(
            initial_capital=initial_capital,
            kelly_fraction=kelly_fraction,
            db_path=db_path,
        )
        self.engine = SimulationEngine(self.config)

        # Initialize feeds
        self.price_feed = ExchangeFeed(primary_exchange='coinbase')
        self.blockchain_feed = BlockchainDataFeed(rpc_host, rpc_port, rpc_user, rpc_pass)

        # State
        self.running = False
        self.tick_count = 0
        self.last_price = 0.0
        self.start_time = None

    def run(self, duration_seconds: float = None):
        """
        Run live trading.

        Args:
            duration_seconds: Run duration (None = indefinite)
        """
        self.running = True
        self.start_time = time.time()
        self.tick_count = 0

        end_time = time.time() + duration_seconds if duration_seconds else None

        # Print header
        print("=" * 70)
        print("RENTECH LIVE TRADING - PRODUCTION")
        print("=" * 70)
        print(f"Started:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        print(f"Kelly Fraction:  {self.config.kelly_fraction}")
        print(f"Formulas:        {len(PRODUCTION_FORMULA_IDS)}")
        print(f"Blockchain:      {'Connected' if self.blockchain_feed.enabled else 'Disabled'}")
        print(f"Duration:        {'Indefinite' if duration_seconds is None else f'{duration_seconds}s'}")
        print("=" * 70)

        # Create session
        from engine.sovereign.simulation.trade_logger import TradeLogger
        self.engine.logger = TradeLogger(self.config.db_path)
        session = self.engine.logger.create_session(
            mode="live",
            initial_capital=self.config.initial_capital,
            kelly_fraction=self.config.kelly_fraction,
            formula_ids=PRODUCTION_FORMULA_IDS,
        )

        # Initialize verifier for live mode
        from engine.sovereign.simulation.verifier import ExchangeVerifier
        self.engine.verifier = ExchangeVerifier()

        last_status = time.time()

        try:
            while self.running:
                # Check duration
                if end_time and time.time() >= end_time:
                    print("\n[LIVE] Duration complete")
                    break

                # Get price
                tick = self.price_feed.get_price()
                if not tick:
                    print("[LIVE] No price data, retrying...")
                    time.sleep(self.poll_interval)
                    continue

                # Get blockchain data (optional)
                blockchain_data = self.blockchain_feed.get_data()

                # Combine data
                data = tick.to_dict()
                data.update(blockchain_data)

                # Process tick
                self.engine._process_tick(tick.timestamp, tick.price, data)

                self.last_price = tick.price
                self.tick_count += 1

                # Status update every 60 seconds
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
        """Print current status."""
        elapsed = time.time() - self.start_time
        session = self.engine.logger.session

        print(f"\n[STATUS] {datetime.now().strftime('%H:%M:%S')} | "
              f"Price: ${self.last_price:,.2f} | "
              f"Capital: ${self.engine.capital:,.2f} | "
              f"Trades: {session.total_trades if session else 0} | "
              f"PnL: ${self.engine.capital - self.config.initial_capital:+,.2f}")

        # Print active positions
        if self.engine.positions:
            print(f"[POSITIONS] {len(self.engine.positions)} active:")
            for trade_id, pos in self.engine.positions.items():
                pnl_pct = ((self.last_price / pos.entry_price) - 1) * 100 if pos.direction == 1 else ((pos.entry_price / self.last_price) - 1) * 100
                print(f"  {trade_id[:8]}: {pos.formula_name} {'LONG' if pos.direction == 1 else 'SHORT'} "
                      f"@ ${pos.entry_price:,.2f} ({pnl_pct:+.2f}%)")

    def _shutdown(self):
        """Clean shutdown."""
        self.running = False

        # Close all positions
        if self.last_price > 0:
            self.engine._close_all_positions(self.last_price, time.time())

        # Calculate results
        elapsed = time.time() - self.start_time if self.start_time else 0
        session = self.engine.logger.session

        # Close session
        self.engine.logger.close_session(
            final_capital=self.engine.capital,
            max_drawdown=0,  # TODO: track properly
            sharpe_ratio=None,
        )

        # Print final results
        print("\n" + "=" * 70)
        print("LIVE TRADING COMPLETE")
        print("=" * 70)
        print(f"Duration:        {elapsed/3600:.2f} hours")
        print(f"Ticks:           {self.tick_count:,}")
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        print(f"Final Capital:   ${self.engine.capital:,.2f}")
        print(f"Total PnL:       ${self.engine.capital - self.config.initial_capital:+,.2f}")
        if session:
            print(f"Total Trades:    {session.total_trades}")
            print(f"Win Rate:        {session.win_rate*100:.1f}%" if session.win_rate else "Win Rate: N/A")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="RenTech Live Trading Runner")

    parser.add_argument('--capital', type=float, default=10000.0,
                        help='Initial capital (default: 10000)')
    parser.add_argument('--kelly', type=float, default=0.25,
                        help='Kelly fraction (default: 0.25)')
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
    parser.add_argument('--db-path', type=str, default='data/live_trades.db',
                        help='Trade database path')

    args = parser.parse_args()

    runner = LiveTradingRunner(
        initial_capital=args.capital,
        kelly_fraction=args.kelly,
        poll_interval=args.poll_interval,
        rpc_host=args.rpc_host,
        rpc_port=args.rpc_port,
        rpc_user=args.rpc_user,
        rpc_pass=args.rpc_pass,
        db_path=args.db_path,
    )

    runner.run(duration_seconds=args.duration)


if __name__ == '__main__':
    main()
