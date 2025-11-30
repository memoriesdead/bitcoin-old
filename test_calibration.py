#!/usr/bin/env python3
"""
BLOCKCHAIN vs API PRICE CALIBRATION
====================================
Cross-reference our blockchain-derived signals vs real API price.
Find the math relationship to calibrate formulas.
"""
import sys
import asyncio
import time
import urllib.request
import json

sys.stdout = sys.stderr

from blockchain.blockchain_feed import BlockchainFeed
from blockchain.blockchain_price_engine import BlockchainPriceEngine, BlockchainState
from freqtrade_bridge.explosive_trader import ExplosiveTrader


async def fetch_real_price() -> float:
    """Fetch REAL BTC price from Coinbase."""
    try:
        req = urllib.request.Request(
            'https://api.coinbase.com/v2/prices/BTC-USD/spot',
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            return float(data['data']['amount'])
    except Exception as e:
        print(f"Price fetch error: {e}")
        return 0.0


async def main():
    print("=" * 70)
    print("BLOCKCHAIN vs API PRICE CALIBRATION")
    print("=" * 70)

    # Get initial real price
    real_price = await fetch_real_price()
    print(f"REAL API PRICE (Coinbase): ${real_price:,.2f}")

    # Initialize blockchain price engine
    price_engine = BlockchainPriceEngine(calibration_price=real_price)
    print(f"BLOCKCHAIN ENGINE initialized at: ${real_price:,.2f}")

    # Initialize trader for formula signals
    trader = ExplosiveTrader(initial_capital=10000)

    # Start blockchain feed
    feed = BlockchainFeed()
    feed_task = asyncio.create_task(feed.start())

    print("\nWaiting 5s for blockchain data...")
    await asyncio.sleep(5)

    print("\n" + "=" * 70)
    print("CALIBRATION DATA - Comparing Blockchain vs Real Price")
    print("=" * 70)
    print(f"{'Time':>6} | {'Real API':>12} | {'Blockchain':>12} | {'Diff':>8} | {'Diff%':>8} | {'Signal':>8} | {'TX/s':>6} | {'Fee':>6}")
    print("-" * 90)

    # Track correlation data
    diffs = []
    signals_vs_moves = []
    last_real_price = real_price

    start = time.time()
    last_update = time.time()

    for i in range(120):  # 2 minutes of calibration data
        try:
            # Fetch real price
            new_real_price = await fetch_real_price()
            if new_real_price <= 0:
                await asyncio.sleep(0.5)
                continue

            # Get blockchain stats
            stats = feed.get_stats()
            tx_per_sec = stats.get('tx_per_sec', 0)
            total_btc = stats.get('total_btc', 0)
            fee_fast = stats.get('fee_fast', 1)
            fee_medium = stats.get('fee_medium', 1)
            mempool = stats.get('mempool_count', 0)

            # Create blockchain state
            bs = BlockchainState(
                timestamp=time.time(),
                tx_count_1m=int(tx_per_sec * 60),
                tx_volume_btc_1m=total_btc / max(stats.get('elapsed_sec', 1), 1) * 60,
                fee_fast=max(int(fee_fast), 1),
                fee_medium=max(int(fee_medium), 1),
                mempool_size=max(int(mempool), 1000),
                whale_tx_count=len(feed.get_large_transactions(min_btc=100, limit=10)),
            )

            # Update blockchain price engine
            derived = price_engine.update(bs)
            blockchain_price = derived.composite_price  # The derived price from blockchain data
            blockchain_signal = derived.signal  # -1 to +1 directional signal

            # Get formula signal
            trader.entry_module.update(new_real_price, total_btc, time.time())
            signal, confidence = trader.entry_module.aggregate()

            # Calculate differences
            diff = blockchain_price - new_real_price
            diff_pct = (diff / new_real_price) * 100 if new_real_price > 0 else 0
            diffs.append(diff_pct)

            # Track if signal predicted price direction
            price_move = new_real_price - last_real_price
            if abs(signal) > 0.01 and abs(price_move) > 0.01:
                correct = (signal > 0 and price_move > 0) or (signal < 0 and price_move < 0)
                signals_vs_moves.append(1 if correct else 0)

            # Print every second
            elapsed = time.time() - start
            print(f"{elapsed:5.0f}s | ${new_real_price:>10,.2f} | ${blockchain_price:>10,.2f} | "
                  f"${diff:>+7.2f} | {diff_pct:>+7.3f}% | {signal:>+7.2f} | {tx_per_sec:>5.1f} | {fee_fast:>5.0f}")

            last_real_price = new_real_price
            await asyncio.sleep(1)

        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(1)

    # Summary statistics
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)

    if diffs:
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        min_diff = min(diffs)
        print(f"Price Difference (Blockchain vs API):")
        print(f"  Average: {avg_diff:+.4f}%")
        print(f"  Max:     {max_diff:+.4f}%")
        print(f"  Min:     {min_diff:+.4f}%")

    if signals_vs_moves:
        accuracy = sum(signals_vs_moves) / len(signals_vs_moves) * 100
        print(f"\nSignal Accuracy (predicted price direction):")
        print(f"  Accuracy: {accuracy:.1f}% ({sum(signals_vs_moves)}/{len(signals_vs_moves)})")
        print(f"  Edge over random: {accuracy - 50:+.1f}%")

    print("\n" + "=" * 70)
    print("BLOCKCHAIN METRICS CORRELATION")
    print("=" * 70)
    print("Based on this data, calibrate formulas to weight signals that correlate")
    print("with real price movements more heavily.")

    feed.stop()
    feed_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
