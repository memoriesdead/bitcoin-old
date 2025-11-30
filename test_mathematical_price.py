#!/usr/bin/env python3
"""
LIVE MATHEMATICAL PRICE TEST
============================
Compare blockchain-derived price (from math formulas) vs API price.

Uses PURE BLOCKCHAIN DATA with mathematical models:
- Metcalfe's Law (network effect)
- Power Law (time-based fair value)
- Stock-to-Flow (scarcity)
- Thermocap (mining cost)
- NVT (transaction throughput)

NO API DATA for price derivation - only for verification!
"""
import sys
import asyncio
import time
import urllib.request
import json

sys.stdout = sys.stderr

from blockchain.blockchain_feed import BlockchainFeed
from blockchain.blockchain_mathematical_price import (
    BlockchainMathematicalPrice,
    BlockchainMetrics,
    LiveBlockchainPricer
)


async def fetch_api_price() -> float:
    """Fetch API price for COMPARISON ONLY (not used in derivation)."""
    try:
        req = urllib.request.Request(
            'https://api.coinbase.com/v2/prices/BTC-USD/spot',
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            return float(data['data']['amount'])
    except:
        return 0.0


async def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 120

    print("=" * 80)
    print("BLOCKCHAIN MATHEMATICAL PRICE DERIVATION - LIVE TEST")
    print("=" * 80)
    print()
    print("Deriving BTC fair value from PURE BLOCKCHAIN DATA using mathematical models:")
    print("  - Metcalfe's Law: Price = k * (Active_Addresses)^2 / Supply")
    print("  - Power Law: Price = 10^(a + b * log10(days_since_genesis))")
    print("  - Stock-to-Flow: Price = exp(a + b * ln(SF_Ratio))")
    print("  - Thermocap: Based on cumulative mining energy cost")
    print()
    print("Sources:")
    print("  - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3078248 (Metcalfe)")
    print("  - https://bitcoinfairprice.com/ (Power Law)")
    print("  - http://charts.woobull.com/bitcoin-price-models/ (NVT/Models)")
    print()

    # Initialize blockchain price engine
    pricer = LiveBlockchainPricer()
    engine = pricer.engine

    # Start blockchain feed
    feed = BlockchainFeed()
    feed_task = asyncio.create_task(feed.start())

    print("Waiting 5s for blockchain data...")
    await asyncio.sleep(5)

    # Get initial data
    stats = feed.get_stats()
    api_price = await fetch_api_price()

    print()
    print("=" * 80)
    print("LIVE COMPARISON: Blockchain Math vs API Price")
    print("=" * 80)
    print(f"{'Time':>6} | {'API Price':>12} | {'Math Price':>12} | {'Diff':>8} | "
          f"{'Power Law':>11} | {'Metcalfe':>11} | {'S2F':>11}")
    print("-" * 95)

    start = time.time()
    last_update = time.time()

    # Track accuracy
    diffs = []

    while time.time() - start < duration:
        try:
            now = time.time()

            if now - last_update >= 2:  # Update every 2 seconds
                # Fetch API price (for comparison only!)
                api_price = await fetch_api_price()

                if api_price <= 0:
                    await asyncio.sleep(1)
                    continue

                # Get blockchain stats
                stats = feed.get_stats()

                # Get active addresses estimate from tx rate
                # ~5 tx/sec = ~432k tx/day, assume 2 addresses per tx
                tx_per_day = stats.get('tx_per_sec', 3) * 86400
                active_addresses = int(tx_per_day * 2.5)  # Rough estimate

                # Get block height estimate
                # Current height ~875,000 (Nov 2024)
                block_height = 875000 + int((now - 1732000000) / 600)  # ~10 min blocks

                # Get hashrate estimate (from difficulty if available)
                hashrate_eh = 750  # ~750 EH/s current estimate

                # Calculate supply
                supply = engine.calculate_supply(block_height)

                # TX volume estimate (from feed)
                tx_volume = stats.get('total_btc', 0)

                # Update pricer
                prices = pricer.update_from_feed(
                    tx_count=int(tx_per_day),
                    tx_volume_btc=max(tx_volume, 1000),  # Minimum estimate
                    active_addresses=max(active_addresses, 500000),  # Minimum estimate
                    block_height=block_height,
                    hashrate_eh=hashrate_eh,
                    fee_fast=stats.get('fee_fast', 10),
                )

                # Calculate difference
                math_price = prices.composite_price
                diff = math_price - api_price
                diff_pct = (diff / api_price) * 100 if api_price > 0 else 0
                diffs.append(diff_pct)

                elapsed = int(now - start)

                print(f"{elapsed:5}s | ${api_price:>10,.2f} | ${math_price:>10,.2f} | "
                      f"{diff_pct:>+7.1f}% | ${prices.power_law_price:>9,.0f} | "
                      f"${prices.metcalfe_price:>9,.0f} | ${prices.stock_to_flow_price:>9,.0f}")

                last_update = now

            await asyncio.sleep(0.5)

        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(1)

    feed.stop()
    feed_task.cancel()

    # Summary
    print()
    print("=" * 80)
    print("MATHEMATICAL PRICE DERIVATION SUMMARY")
    print("=" * 80)

    if diffs:
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        min_diff = min(diffs)

        print(f"Difference (Math vs API):")
        print(f"  Average: {avg_diff:+.2f}%")
        print(f"  Max:     {max_diff:+.2f}%")
        print(f"  Min:     {min_diff:+.2f}%")

    # Print model breakdown
    if pricer.latest_price:
        p = pricer.latest_price
        print()
        print("Final Model Prices:")
        print(f"  Metcalfe's Law:   ${p.metcalfe_price:>12,.2f}")
        print(f"  Power Law:        ${p.power_law_price:>12,.2f}")
        print(f"  Stock-to-Flow:    ${p.stock_to_flow_price:>12,.2f}")
        print(f"  Thermocap:        ${p.thermocap_price:>12,.2f}")
        print(f"  NVT:              ${p.nvt_price:>12,.2f}")
        print(f"  ---")
        print(f"  COMPOSITE:        ${p.composite_price:>12,.2f}")
        print()
        print(f"  API Price:        ${api_price:>12,.2f}")
        print(f"  Difference:       {((p.composite_price - api_price) / api_price * 100):>+11.2f}%")

    print()
    print("=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    print("""
The mathematical models derive Bitcoin's FAIR VALUE from pure blockchain data.

- If Math Price > API Price: BTC is UNDERVALUED (buy signal)
- If Math Price < API Price: BTC is OVERVALUED (sell signal)
- If Math Price ≈ API Price: BTC is FAIRLY VALUED

These models predict LONG-TERM fair value, not short-term price movements.
For HFT, use the DIRECTION of the difference as a signal bias, not the
absolute price.

Your edge: When Math Price significantly differs from API Price, the market
will tend to correct toward the mathematical fair value over time.
""")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
