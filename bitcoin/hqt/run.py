#!/usr/bin/env python3
"""
HQT Runner - Deterministic Arbitrage Trading

Main entry point for the HQT (High Quality Trades) system.
Only trades when profit is mathematically guaranteed.

Usage:
    python -m bitcoin.hqt.run --paper     # Paper trading mode
    python -m bitcoin.hqt.run --monitor   # Monitor only (no trades)
    python -m bitcoin.hqt.run --live      # Live trading (requires API keys)
"""

import argparse
import time
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bitcoin.hqt.config import HQTConfig, get_config
from bitcoin.hqt.arbitrage import ArbitrageDetector
from bitcoin.hqt.executor import ArbitrageExecutor, ExecutionMode
from bitcoin.hqt.ccxt_client import CCXTClient


class HQTRunner:
    """
    Main runner for HQT arbitrage system.

    Monitors exchange prices and executes when
    spread > fees + slippage (guaranteed profit).
    """

    def __init__(self, mode: str = 'paper'):
        """
        Initialize HQT runner.

        Args:
            mode: 'paper', 'monitor', or 'live'
        """
        self.config = get_config()
        self.detector = ArbitrageDetector(self.config)
        self.mode = mode

        if mode == 'live':
            self.executor = ArbitrageExecutor(self.config, ExecutionMode.LIVE)
        elif mode == 'paper':
            self.executor = ArbitrageExecutor(self.config, ExecutionMode.PAPER)
        else:
            self.executor = None  # Monitor only

        # Price clients (public API only for monitoring)
        self.price_clients: dict = {}

        self._init_price_clients()

    def _init_price_clients(self):
        """Initialize public price clients for all exchanges."""
        for exchange in self.config.exchanges:
            try:
                self.price_clients[exchange] = CCXTClient(
                    exchange_id=exchange,
                    api_key="",  # Public API
                    secret=""
                )
                print(f"  Connected to {exchange}")
            except Exception as e:
                print(f"  Failed to connect to {exchange}: {e}")

    def fetch_prices(self):
        """Fetch current prices from all exchanges."""
        for exchange, client in self.price_clients.items():
            try:
                ticker = client.get_ticker(self.config.symbol)

                if 'error' not in ticker:
                    self.detector.update_price(
                        exchange=exchange,
                        bid=ticker['bid'],
                        ask=ticker['ask'],
                        timestamp=ticker.get('timestamp', time.time())
                    )
            except Exception as e:
                print(f"Error fetching {exchange}: {e}")

    def run_once(self) -> bool:
        """
        Run one iteration of arbitrage detection.

        Returns:
            True if trade was executed
        """
        # Fetch latest prices
        self.fetch_prices()

        # Find opportunity
        opp = self.detector.find_opportunity()

        if opp is None:
            return False

        print(f"\n{'='*60}")
        print("ARBITRAGE OPPORTUNITY FOUND!")
        print(f"{'='*60}")
        print(f"Buy on {opp.buy_exchange} @ ${opp.buy_price:,.2f}")
        print(f"Sell on {opp.sell_exchange} @ ${opp.sell_price:,.2f}")
        print(f"Spread: {opp.spread_pct*100:.3f}%")
        print(f"Costs: {opp.total_cost_pct*100:.3f}%")
        print(f"Net Profit: {opp.profit_pct*100:.3f}% (${opp.profit_usd:.2f})")
        print(f"Win Rate: {opp.win_rate*100:.0f}% (GUARANTEED)")

        # Execute if not monitor-only
        if self.executor is not None:
            print(f"\nExecuting in {self.mode.upper()} mode...")
            result = self.executor.execute(opp)

            if result.success:
                print(f"SUCCESS: Realized ${result.realized_profit_usd:.2f}")
            else:
                print(f"FAILED: {result.error}")

            return result.success

        return False

    def run(self, duration_seconds: int = 0):
        """
        Run arbitrage loop.

        Args:
            duration_seconds: How long to run (0 = forever)
        """
        print("\n" + "=" * 60)
        print("HQT - HIGH QUALITY TRADES")
        print("Deterministic Arbitrage System")
        print("=" * 60)
        print(f"\nMode: {self.mode.upper()}")
        print(f"Exchanges: {', '.join(self.config.exchanges)}")
        print(f"Min Spread: {self.config.min_spread_pct*100:.2f}%")
        print(f"Min Profit: ${self.config.min_profit_usd:.2f}")
        print(f"Position Size: {self.config.position_size_btc} BTC")
        print("\n" + "=" * 60)
        print("\nStarting arbitrage detection...")
        print("Only trades when profit is GUARANTEED (100% win rate)")
        print("Press Ctrl+C to stop\n")

        start_time = time.time()
        iterations = 0
        trades = 0

        try:
            while True:
                iterations += 1

                # Check duration
                if duration_seconds > 0:
                    elapsed = time.time() - start_time
                    if elapsed >= duration_seconds:
                        break

                # Run detection
                if self.run_once():
                    trades += 1

                # Status every 10 iterations
                if iterations % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:.0f}s] Iterations: {iterations} | "
                          f"Trades: {trades} | "
                          f"Found: {self.detector.opportunities_found} | "
                          f"Skipped: {self.detector.opportunities_skipped}")

                # Rate limit
                time.sleep(self.config.poll_interval_ms / 1000)

        except KeyboardInterrupt:
            print("\n\nStopping...")

        # Print final stats
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Runtime: {elapsed:.0f} seconds")
        print(f"Iterations: {iterations}")
        print(f"Trades: {trades}")

        self.detector.print_status()

        if self.executor:
            self.executor.print_stats()

    def cleanup(self):
        """Clean up resources."""
        for client in self.price_clients.values():
            try:
                client.close()
            except:
                pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='HQT - Deterministic Arbitrage Trading'
    )
    parser.add_argument(
        '--paper', action='store_true',
        help='Paper trading mode (simulated execution)'
    )
    parser.add_argument(
        '--monitor', action='store_true',
        help='Monitor only (no execution)'
    )
    parser.add_argument(
        '--live', action='store_true',
        help='Live trading mode (requires API keys)'
    )
    parser.add_argument(
        '--duration', type=int, default=0,
        help='Run duration in seconds (0 = forever)'
    )

    args = parser.parse_args()

    # Determine mode
    if args.live:
        mode = 'live'
    elif args.monitor:
        mode = 'monitor'
    else:
        mode = 'paper'

    # Safety check for live mode
    if mode == 'live':
        confirm = input("WARNING: Live trading mode. Type 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            return

    # Run
    runner = HQTRunner(mode=mode)

    try:
        runner.run(duration_seconds=args.duration)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
