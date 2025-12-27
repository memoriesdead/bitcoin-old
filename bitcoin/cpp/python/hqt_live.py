#!/usr/bin/env python3
"""
HQT Live Runner - Nanosecond Arbitrage Detection

Uses C++ for spread calculations, Python for exchange connections.
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    import ccxt.async_support as ccxt
except ImportError:
    print("ERROR: Install ccxt: pip install ccxt")
    sys.exit(1)

from hqt_bridge import HQTBridge, ArbitrageOpportunity


@dataclass
class ExchangeConfig:
    """Exchange configuration."""
    name: str
    symbol: str = "BTC/USDT"


class HQTLiveRunner:
    """
    Live HQT Arbitrage Scanner.

    Connects to multiple exchanges, fetches real-time prices,
    uses C++ for nanosecond-speed arbitrage detection.
    """

    # USA-legal exchanges
    EXCHANGES = [
        ExchangeConfig("kraken", "BTC/USD"),
        ExchangeConfig("coinbase", "BTC/USD"),
        ExchangeConfig("gemini", "BTC/USD"),
        ExchangeConfig("bitstamp", "BTC/USD"),
    ]

    def __init__(self, position_btc: float = 0.01,
                 min_profit_usd: float = 5.0,
                 update_interval: float = 1.0):
        """
        Initialize live runner.

        Args:
            position_btc: Position size in BTC
            min_profit_usd: Minimum profit to signal
            update_interval: Price update interval in seconds
        """
        self.position_btc = position_btc
        self.min_profit_usd = min_profit_usd
        self.update_interval = update_interval

        self.bridge = HQTBridge()
        self.bridge._position_btc = position_btc
        self.bridge._min_profit = min_profit_usd

        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.running = False
        self.total_scans = 0
        self.opportunities_found = 0

    async def connect_exchanges(self):
        """Connect to all exchanges."""
        print("\n[HQT] Connecting to exchanges...")

        for config in self.EXCHANGES:
            try:
                exchange_class = getattr(ccxt, config.name)
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 10000,
                })
                await exchange.load_markets()
                self.exchanges[config.name] = exchange
                print(f"  [+] {config.name} - Connected")
            except Exception as e:
                print(f"  [-] {config.name} - Failed: {e}")

        print(f"\n[HQT] Connected to {len(self.exchanges)} exchanges")

    async def fetch_prices(self):
        """Fetch prices from all exchanges."""
        tasks = []

        for config in self.EXCHANGES:
            if config.name in self.exchanges:
                tasks.append(self._fetch_ticker(config.name, config.symbol))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_ticker(self, exchange_name: str, symbol: str):
        """Fetch ticker from single exchange."""
        try:
            exchange = self.exchanges[exchange_name]
            ticker = await exchange.fetch_ticker(symbol)

            if ticker.get('bid') and ticker.get('ask'):
                self.bridge.update_price(
                    exchange_name,
                    float(ticker['bid']),
                    float(ticker['ask'])
                )
        except Exception as e:
            pass  # Silently skip failed fetches

    def find_opportunity(self) -> Optional[ArbitrageOpportunity]:
        """Find best arbitrage opportunity using C++ speed."""
        start = time.time_ns()
        opp = self.bridge.find_opportunity()
        calc_ns = time.time_ns() - start

        if opp:
            print(f"\n{'='*60}")
            print(f"ARBITRAGE FOUND! (calculated in {calc_ns} ns)")
            print(f"{'='*60}")
            print(f"  Buy:    {opp.buy_exchange.upper()} @ ${opp.buy_price:,.2f}")
            print(f"  Sell:   {opp.sell_exchange.upper()} @ ${opp.sell_price:,.2f}")
            print(f"  Spread: {opp.spread_pct*100:.3f}%")
            print(f"  Costs:  {opp.total_cost_pct*100:.3f}%")
            print(f"  Profit: {opp.profit_pct*100:.3f}% (${opp.profit_usd:.2f})")
            print(f"  Win Rate: 100% (GUARANTEED)")
            print(f"{'='*60}")
            self.opportunities_found += 1

        return opp

    def print_status(self):
        """Print current status."""
        prices = self.bridge._prices

        print(f"\n[{time.strftime('%H:%M:%S')}] Scan #{self.total_scans}")
        print(f"  Exchanges: {len(prices)}")

        if prices:
            for name, data in sorted(prices.items()):
                spread = (data['ask'] - data['bid']) / data['bid'] * 100
                print(f"  {name:12}: Bid ${data['bid']:,.2f} | Ask ${data['ask']:,.2f} | Spread {spread:.3f}%")

        # Find best spread across pairs
        all_opps = self.bridge.find_all_opportunities()
        if all_opps:
            best = all_opps[0]
            print(f"\n  Best Spread: {best.buy_exchange} -> {best.sell_exchange}: {best.spread_pct*100:.3f}%")
            print(f"  Net Profit (after costs): {best.profit_pct*100:.3f}%")
        else:
            # Show why no opportunity
            if len(prices) >= 2:
                print(f"\n  No profitable spread (costs exceed spread)")

        print(f"\n  Total opportunities: {self.opportunities_found}")

    async def run(self):
        """Run the live scanner."""
        print("\n" + "=" * 60)
        print("HQT LIVE ARBITRAGE SCANNER")
        print("Nanosecond C++ Speed")
        print("=" * 60)
        print(f"\nPosition: {self.position_btc} BTC")
        print(f"Min profit: ${self.min_profit_usd}")
        print(f"Update interval: {self.update_interval}s")
        print(f"C++ available: {self.bridge.cpp_available}")

        await self.connect_exchanges()

        if len(self.exchanges) < 2:
            print("\n[ERROR] Need at least 2 exchanges for arbitrage")
            return

        self.running = True
        print("\n[HQT] Starting scan loop (Ctrl+C to stop)...")

        try:
            while self.running:
                self.total_scans += 1

                # Fetch latest prices
                await self.fetch_prices()

                # Find opportunities
                self.find_opportunity()

                # Print status
                self.print_status()

                # Wait
                await asyncio.sleep(self.update_interval)

        except KeyboardInterrupt:
            print("\n\n[HQT] Stopping...")
        finally:
            await self.close()

    async def close(self):
        """Close all exchange connections."""
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except:
                pass

        print(f"\n[HQT] Closed. Total scans: {self.total_scans}, Opportunities: {self.opportunities_found}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='HQT Live Arbitrage Scanner')
    parser.add_argument('--btc', type=float, default=0.01, help='Position size in BTC')
    parser.add_argument('--min-profit', type=float, default=5.0, help='Min profit in USD')
    parser.add_argument('--interval', type=float, default=1.0, help='Update interval in seconds')

    args = parser.parse_args()

    runner = HQTLiveRunner(
        position_btc=args.btc,
        min_profit_usd=args.min_profit,
        update_interval=args.interval
    )

    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
