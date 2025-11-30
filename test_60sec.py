#!/usr/bin/env python3
"""
QUICK 60-SECOND LIVE TEST - $10 MILLION
Real Coinbase BTC price + Real blockchain signals
"""
import sys
import asyncio
import time
import urllib.request
import json

# Force unbuffered output
sys.stdout = sys.stderr

from freqtrade_bridge.explosive_trader import ExplosiveTrader
from blockchain.blockchain_price_engine import BlockchainPriceEngine, BlockchainState
from blockchain.blockchain_feed import BlockchainFeed

async def main():
    # 1. Get REAL BTC price from Coinbase
    print("=" * 60, flush=True)
    print("FETCHING REAL BTC PRICE...", flush=True)
    req = urllib.request.Request('https://api.coinbase.com/v2/prices/BTC-USD/spot')
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
        btc_price = float(data['data']['amount'])
    print(f"REAL BTC: ${btc_price:,.2f}", flush=True)
    print("=" * 60, flush=True)

    # 2. Initialize trader - get capital from command line or default $10M
    capital = float(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000.0
    trader = ExplosiveTrader(initial_capital=capital)
    print(f"CAPITAL: ${capital:,.2f}", flush=True)
    trader.price_engine = BlockchainPriceEngine(calibration_price=btc_price)

    # 3. Start blockchain feed as background task (doesn't block)
    feed = BlockchainFeed()
    feed_task = asyncio.create_task(feed.start())

    # Wait for data
    print("Waiting 5s for blockchain data...", flush=True)
    await asyncio.sleep(5)

    print(f"\n>>> LIVE TEST: 60 SECONDS | $10,000,000 <<<", flush=True)
    print(f">>> SL: {trader.stop_loss_pct*100}% | TP: {trader.take_profit_pct*100}%", flush=True)
    print("-" * 60, flush=True)

    start = time.time()
    updates = 0

    while time.time() - start < 60:
        # Get stats from feed
        stats = feed.get_stats()

        # Convert to BlockchainState
        bs = BlockchainState(
            timestamp=time.time(),
            tx_count_1m=int(stats.get('tx_per_sec', 0) * 60),
            tx_volume_btc_1m=stats.get('total_btc', 0) / max(stats.get('elapsed_sec', 1), 1) * 60,
            fee_fast=max(int(stats.get('fee_fast', 1)), 1),
            fee_medium=max(int(stats.get('fee_medium', 1)), 1),
            mempool_size=max(int(stats.get('mempool_count', 1000)), 1000),
            whale_tx_count=len(feed.get_large_transactions(min_btc=100, limit=10)),
        )

        result = trader.update(bs)
        updates += 1

        # Print actions
        if result['action'] not in ['HOLD', 'HOLD_POSITION', 'WAIT']:
            elapsed = time.time() - start
            print(f"[{elapsed:.0f}s] {result['action']} @ ${trader._last_price:,.2f} | "
                  f"Capital: ${trader.capital:,.2f}", flush=True)

        # Status every 10s
        if updates % 100 == 0:
            trader_stats = trader.get_stats()
            elapsed = time.time() - start
            print(f"[{elapsed:.0f}s] Price: ${trader._last_price:,.2f} | "
                  f"Capital: ${trader_stats['current_capital']:,.2f} | "
                  f"Trades: {trader_stats['total_trades']} | "
                  f"TX/s: {stats.get('tx_per_sec', 0):.1f}", flush=True)

        await asyncio.sleep(0.1)

    feed.stop()
    feed_task.cancel()

    # Results
    final_stats = trader.get_stats()
    print("\n" + "=" * 60, flush=True)
    print(">>> 60-SECOND LIVE RESULTS <<<", flush=True)
    print("=" * 60, flush=True)
    print(f"Initial:  ${final_stats['initial_capital']:,.2f}", flush=True)
    print(f"Final:    ${final_stats['current_capital']:,.2f}", flush=True)
    print(f"Trades:   {final_stats['total_trades']}", flush=True)
    print(f"Win Rate: {final_stats['win_rate']:.1f}%", flush=True)
    profit = final_stats['current_capital'] - final_stats['initial_capital']
    print(f"PROFIT:   ${profit:+,.2f}", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    asyncio.run(main())
