#!/usr/bin/env python3
"""
ULTRA-AGGRESSIVE REAL PRICE TRADING
====================================
Enter on ANY edge. Even 50.75% = profit over time.
Smart about direction, aggressive on execution.
"""
import sys
import asyncio
import time
import urllib.request
import json

# Force unbuffered output
sys.stdout = sys.stderr

from freqtrade_bridge.explosive_trader import ExplosiveTrader
from blockchain.blockchain_price_engine import BlockchainState
from blockchain.blockchain_feed import BlockchainFeed


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
    except:
        return 0.0


class RealPriceTracker:
    """Tracks REAL prices from Coinbase."""

    def __init__(self):
        self.current_price = 0.0
        self.last_update = 0.0
        self.price_history = []

    async def update(self) -> float:
        """Fetch latest real price."""
        price = await fetch_real_price()
        if price > 0:
            self.current_price = price
            self.last_update = time.time()
            self.price_history.append((self.last_update, price))
            if len(self.price_history) > 1000:
                self.price_history = self.price_history[-1000:]
        return self.current_price


async def main():
    capital = float(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000.0

    print("=" * 70, flush=True)
    print("ULTRA-AGGRESSIVE REAL PRICE TRADING", flush=True)
    print("=" * 70, flush=True)

    # Initialize REAL price tracker
    price_tracker = RealPriceTracker()
    btc_price = await price_tracker.update()
    print(f"REAL BTC PRICE: ${btc_price:,.2f}", flush=True)

    # Initialize trader with ULTRA-AGGRESSIVE settings
    trader = ExplosiveTrader(initial_capital=capital)

    # ULTRA-AGGRESSIVE CONFIG
    STOP_LOSS_PCT = 0.0003      # 0.03% SL = $27 on $90k
    TAKE_PROFIT_PCT = 0.0005   # 0.05% TP = $45 on $90k
    MIN_SIGNAL = 0.01          # Enter on 1% majority (50.5% vs 49.5%)
    MIN_CONFIDENCE = 0.01      # Almost any confidence
    POSITION_SIZE_PCT = 0.25   # 25% of capital per trade (Kelly-adjusted)

    # BLOCKCHAIN → PRICE CORRELATION TRACKING
    # This is the EDGE: blockchain sees things BEFORE price moves
    signal_history = []  # Track (signal, price_before, price_after) to learn correlations

    print(f"CAPITAL: ${capital:,.2f}", flush=True)
    print(f"ULTRA-AGGRESSIVE: SL={STOP_LOSS_PCT*100:.3f}% TP={TAKE_PROFIT_PCT*100:.3f}%", flush=True)
    print(f"ENTRY: Signal>{MIN_SIGNAL*100:.1f}% Conf>{MIN_CONFIDENCE*100:.1f}%", flush=True)

    # Start blockchain feed for SIGNALS
    feed = BlockchainFeed()
    feed_task = asyncio.create_task(feed.start())

    print("Waiting 3s for blockchain signals...", flush=True)
    await asyncio.sleep(3)

    print(f"\n>>> ULTRA-AGGRESSIVE MODE ACTIVE <<<", flush=True)
    print(f">>> Capital: ${capital:,.2f} | SL: {STOP_LOSS_PCT*100:.3f}% | TP: {TAKE_PROFIT_PCT*100:.3f}%", flush=True)
    print("-" * 70, flush=True)

    start = time.time()
    last_print = time.time()
    last_price_fetch = time.time()

    # Position tracking
    in_position = False
    position_type = None
    entry_price = 0.0
    position_size = 0.0
    trades = 0
    wins = 0
    total_pnl = 0.0
    current_capital = capital

    # Cooldown to avoid whipsaw
    last_exit_time = 0.0
    COOLDOWN_SEC = 2.0  # 2 second cooldown between trades

    # Track consecutive losses for smart sizing
    consecutive_losses = 0

    while True:
        try:
            now = time.time()

            # Fetch REAL price every 250ms (faster!)
            if now - last_price_fetch >= 0.25:
                real_price = await price_tracker.update()
                last_price_fetch = now

                if real_price <= 0:
                    await asyncio.sleep(0.1)
                    continue

                # Check SL/TP if in position
                if in_position:
                    if position_type == 'LONG':
                        pnl_pct = (real_price - entry_price) / entry_price
                    else:  # SHORT
                        pnl_pct = (entry_price - real_price) / entry_price

                    current_pnl = position_size * pnl_pct

                    # Check TP
                    if pnl_pct >= TAKE_PROFIT_PCT:
                        total_pnl += current_pnl
                        current_capital += current_pnl
                        trades += 1
                        wins += 1
                        consecutive_losses = 0
                        elapsed = now - start
                        mins, secs = int(elapsed // 60), int(elapsed % 60)
                        print(f"[{mins}m{secs}s] ✓ TP HIT! {position_type} +${current_pnl:,.2f} | "
                              f"Capital: ${current_capital:,.2f} | W:{wins}/{trades}", flush=True)
                        in_position = False
                        last_exit_time = now

                    # Check SL
                    elif pnl_pct <= -STOP_LOSS_PCT:
                        total_pnl += current_pnl
                        current_capital += current_pnl
                        trades += 1
                        consecutive_losses += 1
                        elapsed = now - start
                        mins, secs = int(elapsed // 60), int(elapsed % 60)
                        print(f"[{mins}m{secs}s] ✗ SL HIT! {position_type} ${current_pnl:,.2f} | "
                              f"Capital: ${current_capital:,.2f} | W:{wins}/{trades}", flush=True)
                        in_position = False
                        last_exit_time = now

                # If not in position and cooldown passed, check for entry
                if not in_position and (now - last_exit_time >= COOLDOWN_SEC):
                    stats = feed.get_stats()

                    # FEED DATA TO FORMULAS - they need price/volume to generate signals!
                    volume = stats.get('total_btc', 100.0)
                    trader.entry_module.update(real_price, volume, now)

                    # Get signal from ALL formulas
                    signal, confidence = trader.entry_module.aggregate()

                    # ULTRA-AGGRESSIVE: Enter on ANY edge
                    if abs(signal) >= MIN_SIGNAL and confidence >= MIN_CONFIDENCE:

                        # Smart position sizing based on consecutive losses
                        if consecutive_losses >= 3:
                            size_mult = 0.10  # Reduce to 10% after 3 losses
                        elif consecutive_losses >= 2:
                            size_mult = 0.15  # 15% after 2 losses
                        elif consecutive_losses >= 1:
                            size_mult = 0.20  # 20% after 1 loss
                        else:
                            size_mult = POSITION_SIZE_PCT  # Full 25%

                        # Scale by signal strength (stronger signal = bigger position)
                        signal_multiplier = min(abs(signal) * 2, 1.0)  # Cap at 1x
                        position_size = current_capital * size_mult * signal_multiplier

                        entry_price = real_price
                        in_position = True

                        if signal > 0:
                            position_type = 'LONG'
                            tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
                            sl_price = entry_price * (1 - STOP_LOSS_PCT)
                        else:
                            position_type = 'SHORT'
                            tp_price = entry_price * (1 - TAKE_PROFIT_PCT)
                            sl_price = entry_price * (1 + STOP_LOSS_PCT)

                        elapsed = now - start
                        mins, secs = int(elapsed // 60), int(elapsed % 60)
                        print(f"[{mins}m{secs}s] → ENTER {position_type} @ ${real_price:,.2f} | "
                              f"Signal: {signal:+.1%} | Size: ${position_size:,.2f}", flush=True)

            # Status every 10 seconds
            if now - last_print >= 10:
                elapsed = now - start
                mins, secs = int(elapsed // 60), int(elapsed % 60)
                win_rate = (wins / trades * 100) if trades > 0 else 0

                if in_position:
                    if position_type == 'LONG':
                        unrealized = position_size * (real_price - entry_price) / entry_price
                    else:
                        unrealized = position_size * (entry_price - real_price) / entry_price
                    pos_status = f"{position_type} ${unrealized:+,.2f}"
                else:
                    pos_status = "FLAT"
                    unrealized = 0

                print(f"[{mins}m{secs}s] ${price_tracker.current_price:,.2f} | "
                      f"Capital: ${current_capital:,.2f} | PnL: ${total_pnl:+,.2f} | "
                      f"Trades: {trades} W:{win_rate:.0f}% | {pos_status}", flush=True)
                last_print = now

            await asyncio.sleep(0.05)  # 50ms loop for fast response

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}", flush=True)
            await asyncio.sleep(1)

    feed.stop()
    feed_task.cancel()

    # Final Results
    elapsed = time.time() - start
    mins, secs = int(elapsed // 60), int(elapsed % 60)
    win_rate = (wins / trades * 100) if trades > 0 else 0

    print("\n" + "=" * 70, flush=True)
    print(f">>> ULTRA-AGGRESSIVE RESULTS AFTER {mins}m {secs}s <<<", flush=True)
    print("=" * 70, flush=True)
    print(f"Initial:   ${capital:,.2f}", flush=True)
    print(f"Final:     ${current_capital:,.2f}", flush=True)
    print(f"Total PnL: ${total_pnl:+,.2f}", flush=True)
    print(f"Trades:    {trades}", flush=True)
    print(f"Win Rate:  {win_rate:.1f}%", flush=True)
    print(f"Avg/Trade: ${total_pnl/max(trades,1):+,.2f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
