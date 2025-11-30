#!/usr/bin/env python3
"""
BLOCKCHAIN SIGNAL ACCURACY TEST
================================
Test the BlockchainTruePrice engine against real price movements.

Goal: Achieve >50% accuracy = trading edge.

This test:
1. Captures blockchain signals (fees, mempool, velocity, whales)
2. Generates direction predictions
3. Compares predictions to actual price movements
4. Tracks accuracy by signal type
5. Reports edge over random
"""
import sys
import asyncio
import time
import urllib.request
import json

sys.stdout = sys.stderr

from blockchain.blockchain_feed import BlockchainFeed
from blockchain.blockchain_true_price import BlockchainPricePredictor


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
        return 0.0


async def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 120  # Default 2 minutes

    print("=" * 70)
    print("BLOCKCHAIN SIGNAL ACCURACY TEST")
    print("=" * 70)
    print(f"Duration: {duration} seconds")
    print()
    print("Testing BlockchainTruePrice engine against real price movements")
    print("Goal: >50% accuracy = trading edge")
    print()

    # Initialize blockchain predictor
    predictor = BlockchainPricePredictor()

    # Start blockchain feed
    feed = BlockchainFeed()
    feed_task = asyncio.create_task(feed.start())

    print("Waiting 5s for blockchain data...")
    await asyncio.sleep(5)

    # Get initial price
    real_price = await fetch_real_price()
    print(f"Initial BTC Price: ${real_price:,.2f}")
    print()

    print("=" * 70)
    print("LIVE SIGNALS")
    print("=" * 70)
    print(f"{'Time':>6} | {'Price':>10} | {'Dir':>6} | {'Mom':>5} | {'Conf':>5} | "
          f"{'Fee':>6} | {'Memp':>6} | {'Vel':>6} | {'Whale':>5} | {'Acc':>6}")
    print("-" * 90)

    start = time.time()
    last_price_fetch = time.time()
    last_verify = time.time()

    while time.time() - start < duration:
        try:
            now = time.time()

            # Fetch real price every 500ms
            if now - last_price_fetch >= 0.5:
                new_price = await fetch_real_price()
                if new_price <= 0:
                    await asyncio.sleep(0.1)
                    continue

                real_price = new_price
                last_price_fetch = now

                # Get blockchain stats
                stats = feed.get_stats()

                # Generate prediction
                signal = predictor.predict(
                    fee_fast=stats.get('fee_fast', 1),
                    fee_medium=stats.get('fee_medium', 1),
                    mempool_size=stats.get('mempool_count', 10000),
                    tx_per_sec=stats.get('tx_per_sec', 3.0),
                    whale_count=len(feed.get_large_transactions(min_btc=100, limit=10)),
                    current_price=real_price,
                )

                # Verify predictions from 10 seconds ago
                if now - last_verify >= 2:
                    predictor.verify_predictions(real_price, lookback_sec=10)
                    last_verify = now

                # Print every second
                elapsed = int(now - start)
                acc = predictor.get_accuracy() * 100
                edge = predictor.get_edge() * 100

                direction_str = f"{signal.direction:+.2f}"
                if signal.direction > 0.1:
                    direction_str = f"+{signal.direction:.2f}"
                elif signal.direction < -0.1:
                    direction_str = f"{signal.direction:.2f}"

                print(f"{elapsed:5}s | ${real_price:>9,.2f} | {direction_str:>6} | "
                      f"{signal.momentum:.2f} | {signal.confidence:.2f} | "
                      f"{signal.fee_signal:+.2f} | {signal.mempool_signal:+.2f} | "
                      f"{signal.velocity_signal:+.2f} | {signal.whale_count:>5} | "
                      f"{acc:5.1f}%")

            await asyncio.sleep(0.25)

        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(1)

    # Final verification
    predictor.verify_predictions(real_price, lookback_sec=5)

    # Stop feed
    feed.stop()
    feed_task.cancel()

    # Print report
    print()
    predictor.print_report()

    # Additional analysis
    print()
    print("=" * 70)
    print("SIGNAL CONTRIBUTION ANALYSIS")
    print("=" * 70)

    acc_by_type = predictor.get_accuracy_by_type()
    for signal_type, accuracy in sorted(acc_by_type.items(), key=lambda x: x[1], reverse=True):
        edge = (accuracy - 0.5) * 100
        status = "USEFUL" if edge > 0 else "NOISE" if edge < 0 else "NEUTRAL"
        data = predictor.accuracy_by_type[signal_type]
        print(f"  {signal_type:12} {accuracy*100:5.1f}% | Edge: {edge:+5.1f}% | {status:8} ({data['total']} samples)")

    overall_edge = predictor.get_edge() * 100

    print()
    print("=" * 70)
    if overall_edge > 2:
        print(f"EXCELLENT: {overall_edge:+.1f}% edge - SIGNIFICANT TRADING EDGE!")
    elif overall_edge > 0:
        print(f"POSITIVE: {overall_edge:+.1f}% edge - small but usable")
    elif overall_edge > -2:
        print(f"NEUTRAL: {overall_edge:+.1f}% - no clear edge yet")
    else:
        print(f"NEGATIVE: {overall_edge:+.1f}% - signals need recalibration")
    print("=" * 70)

    # Engine stats
    engine_stats = predictor.engine.get_stats()
    print()
    print("Engine Statistics:")
    print(f"  Updates: {engine_stats['updates']}")
    print(f"  Signals: {engine_stats['signals']}")
    print(f"  Baseline Fee: {engine_stats['baseline_fee']:.1f} sat/vB")
    print(f"  Baseline Mempool: {engine_stats['baseline_mempool']:,.0f} txs")
    print(f"  Baseline Velocity: {engine_stats['baseline_velocity']:.2f} tx/s")


if __name__ == "__main__":
    asyncio.run(main())
