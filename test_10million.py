#!/usr/bin/env python3
"""$10 MILLION LIVE TEST - REAL BTC PRICE + REAL BLOCKCHAIN DATA"""
import sys
import asyncio
import time
import urllib.request
import json

# Force unbuffered to stderr
sys.stdout = sys.stderr

from freqtrade_bridge.explosive_trader import ExplosiveTrader
from blockchain.blockchain_price_engine import BlockchainPriceEngine, BlockchainState
from blockchain_market_data_usd import BlockchainMarketData

def convert_state(market_state) -> BlockchainState:
    """Convert BlockchainMarketState to BlockchainState for price engine"""
    return BlockchainState(
        timestamp=time.time(),  # Use current time
        tx_count_1m=int(market_state.tx_rate * 60),
        tx_volume_btc_1m=market_state.btc_volume_5m / 5,  # Divide 5min by 5
        tx_count_10m=int(market_state.tx_rate * 600),
        tx_volume_btc_10m=market_state.btc_volume_5m * 2,
        fee_fast=max(market_state.fee_fast, 1),  # Min 1
        fee_medium=max(market_state.fee_medium, 1),
        fee_slow=max(market_state.fee_slow, 1),
        mempool_size=max(market_state.mempool_size, 1000),
        mempool_vsize_mb=market_state.mempool_vsize_mb,
        mempool_growth_rate=market_state.mempool_growth_rate,
        whale_tx_count=int(market_state.whale_tx_count),
        whale_volume_btc=market_state.btc_volume_5m * 0.1,
        block_height=market_state.last_block_height,
        block_time_avg=market_state.block_interval,
        active_addresses_1h=50000,
    )

async def main():
    # Get REAL BTC price from Coinbase
    print("\n" + "="*60, flush=True)
    print("FETCHING REAL BTC PRICE FROM COINBASE...", flush=True)
    req = urllib.request.Request('https://api.coinbase.com/v2/prices/BTC-USD/spot')
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
        real_price = float(data['data']['amount'])

    print(f"REAL BTC PRICE: ${real_price:,.2f}", flush=True)
    print("="*60 + "\n", flush=True)

    # Initialize trader with $10 MILLION
    trader = ExplosiveTrader(initial_capital=10_000_000.0)
    market_data = BlockchainMarketData()
    trader.price_engine = BlockchainPriceEngine(calibration_price=real_price)

    # Start blockchain feed
    await market_data.start()

    # Wait for data to start flowing
    print("Waiting for blockchain data...", flush=True)
    await asyncio.sleep(5)

    print(f"\n>>> STARTING 60-SECOND LIVE TEST <<<", flush=True)
    print(f">>> Capital: $10,000,000", flush=True)
    print(f">>> SL: {trader.stop_loss_pct*100}% | TP: {trader.take_profit_pct*100}%", flush=True)

    start_time = time.time()
    update_count = 0

    try:
        while time.time() - start_time < 60:
            market_state = market_data.get_state()

            # Always try to update
            state = convert_state(market_state)
            result = trader.update(state)
            update_count += 1

            # Print all non-hold actions
            if result['action'] not in ['HOLD', 'HOLD_POSITION', 'WAIT']:
                elapsed = time.time() - start_time
                print(f"\n>>> [{elapsed:.0f}s] {result['action']} @ ${trader._last_price:,.2f} | "
                      f"Capital: ${trader.capital:,.2f}", flush=True)

            # Print status every 10 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and update_count > 0:
                stats = trader.get_stats()
                if update_count % 100 == 1:  # Once per 10sec roughly
                    print(f"\n>>> [{elapsed:.0f}s] Price: ${trader._last_price:,.2f} | "
                          f"Trades: {stats['total_trades']} | "
                          f"Capital: ${stats['current_capital']:,.2f}", flush=True)

            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped by user", flush=True)

    finally:
        await market_data.stop()

    # Final results
    stats = trader.get_stats()
    print("\n" + "="*60, flush=True)
    print(">>> 60-SECOND LIVE TEST COMPLETE <<<", flush=True)
    print("="*60, flush=True)
    print(f"Initial:     ${stats['initial_capital']:,.2f}", flush=True)
    print(f"Final:       ${stats['current_capital']:,.2f}", flush=True)
    print(f"Trades:      {stats['total_trades']}", flush=True)
    print(f"Win Rate:    {stats['win_rate']:.1f}%", flush=True)
    print(f"NET PROFIT:  ${stats['current_capital'] - stats['initial_capital']:+,.2f}", flush=True)
    print("="*60 + "\n", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
