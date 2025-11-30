#!/usr/bin/env python3
"""
TRUE PRICE LIVE TRADING - Pure Blockchain Mathematics
======================================================
NO EXCHANGE APIs. Uses TRUE blockchain-derived price.

Formula: TRUE_PRICE = Production_Cost x (1 + Scarcity + Maturity x Supply)

All multipliers derived from blockchain metrics - NO arbitrary constants.
"""
import sys
import asyncio
import time

# Force unbuffered output
sys.stdout = sys.stderr

from freqtrade_bridge.explosive_trader import ExplosiveTrader
from blockchain.blockchain_price_engine import BlockchainState
from blockchain.blockchain_feed import BlockchainFeed
from blockchain.mathematical_price import MathematicalPricer


class TruePriceTracker:
    """
    TRUE PRICE from pure blockchain mathematics.
    NO EXCHANGE APIS.
    """

    def __init__(self, energy_cost_kwh: float = 0.044):
        self.pricer = MathematicalPricer(energy_cost_kwh=energy_cost_kwh)
        self.current_price = 0.0
        self.last_update = 0.0
        self.price_history = []
        self.last_block = 0

    def update(self) -> float:
        """Get latest TRUE price from blockchain."""
        try:
            p = self.pricer.get_price()
            self.current_price = p.derived_price
            self.last_update = time.time()
            self.last_block = p.block_height

            self.price_history.append((self.last_update, self.current_price))
            # Keep last 1000 prices
            if len(self.price_history) > 1000:
                self.price_history = self.price_history[-1000:]

            return self.current_price
        except Exception as e:
            print(f"Price update error: {e}", flush=True)
            return self.current_price


async def main():
    # Get capital from command line
    capital = float(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000.0

    print("=" * 70, flush=True)
    print("TRUE PRICE TRADING - Pure Blockchain Mathematics", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print("Formula: TRUE_PRICE = Production_Cost x (1 + Scarcity + Maturity x Supply)", flush=True)
    print("NO EXCHANGE APIS - Pure blockchain data", flush=True)
    print("=" * 70, flush=True)

    # Initialize TRUE price tracker
    price_tracker = TruePriceTracker(energy_cost_kwh=0.044)
    btc_price = price_tracker.update()
    print(f"TRUE BTC PRICE: ${btc_price:,.2f}", flush=True)
    print(f"Block Height: {price_tracker.last_block:,}", flush=True)

    # Initialize trader
    trader = ExplosiveTrader(initial_capital=capital)
    print(f"CAPITAL: ${capital:,.2f}", flush=True)

    # Start blockchain feed for SIGNALS
    feed = BlockchainFeed()
    feed_task = asyncio.create_task(feed.start())

    print("Waiting 3s for blockchain signals...", flush=True)
    await asyncio.sleep(3)

    print(f"\n>>> TRUE PRICE LIVE TRADING <<<", flush=True)
    print(f">>> Capital: ${capital:,.2f} | SL: {trader.stop_loss_pct*100:.2f}% | TP: {trader.take_profit_pct*100:.2f}%", flush=True)
    print("-" * 70, flush=True)

    start = time.time()
    updates = 0
    last_print = time.time()
    last_price_fetch = time.time()

    # Position tracking
    in_position = False
    position_type = None  # 'LONG' or 'SHORT'
    entry_price = 0.0
    position_size = 0.0
    trades = 0
    wins = 0
    total_pnl = 0.0

    try:
        while True:
            now = time.time()

            # Update TRUE price every 2 seconds (blockchain updates slowly)
            if now - last_price_fetch >= 2.0:
                btc_price = price_tracker.update()
                last_price_fetch = now

            # Get blockchain state for signals
            state = feed.get_state()
            if state:
                # Generate signal
                signal = trader.evaluate_signal(state)
                updates += 1

                # Position management
                if in_position:
                    # Calculate current PnL
                    if position_type == 'LONG':
                        pnl_pct = (btc_price - entry_price) / entry_price
                    else:  # SHORT
                        pnl_pct = (entry_price - btc_price) / entry_price

                    current_pnl = position_size * pnl_pct

                    # Check exits
                    if pnl_pct <= -trader.stop_loss_pct:
                        # Stop loss hit
                        total_pnl += current_pnl
                        trades += 1
                        print(f"[SL] {position_type} closed | Entry: ${entry_price:,.0f} | Exit: ${btc_price:,.0f} | PnL: ${current_pnl:,.2f} ({pnl_pct*100:+.2f}%)", flush=True)
                        in_position = False
                        position_type = None

                    elif pnl_pct >= trader.take_profit_pct:
                        # Take profit hit
                        total_pnl += current_pnl
                        trades += 1
                        wins += 1
                        print(f"[TP] {position_type} closed | Entry: ${entry_price:,.0f} | Exit: ${btc_price:,.0f} | PnL: ${current_pnl:,.2f} ({pnl_pct*100:+.2f}%)", flush=True)
                        in_position = False
                        position_type = None

                    # Opposite signal exits
                    elif signal == 'LONG' and position_type == 'SHORT':
                        total_pnl += current_pnl
                        trades += 1
                        if current_pnl > 0:
                            wins += 1
                        print(f"[FLIP] SHORT->LONG | PnL: ${current_pnl:,.2f}", flush=True)
                        # Open new long
                        position_type = 'LONG'
                        entry_price = btc_price
                        position_size = trader.capital * trader.position_size

                    elif signal == 'SHORT' and position_type == 'LONG':
                        total_pnl += current_pnl
                        trades += 1
                        if current_pnl > 0:
                            wins += 1
                        print(f"[FLIP] LONG->SHORT | PnL: ${current_pnl:,.2f}", flush=True)
                        # Open new short
                        position_type = 'SHORT'
                        entry_price = btc_price
                        position_size = trader.capital * trader.position_size

                else:
                    # No position - look for entry
                    if signal in ['LONG', 'SHORT']:
                        position_type = signal
                        entry_price = btc_price
                        position_size = trader.capital * trader.position_size
                        in_position = True
                        print(f"[ENTRY] {signal} @ ${btc_price:,.2f} | Size: ${position_size:,.2f}", flush=True)

                # Print status every 5 seconds
                if now - last_print >= 5.0:
                    elapsed = now - start
                    rate = updates / elapsed if elapsed > 0 else 0
                    win_rate = (wins / trades * 100) if trades > 0 else 0

                    pos_str = f"{position_type}" if in_position else "FLAT"
                    current_pnl_str = ""
                    if in_position:
                        if position_type == 'LONG':
                            curr_pnl = (btc_price - entry_price) / entry_price * position_size
                        else:
                            curr_pnl = (entry_price - btc_price) / entry_price * position_size
                        current_pnl_str = f" | Open: ${curr_pnl:+,.0f}"

                    print(f"[{elapsed:6.1f}s] TRUE: ${btc_price:,.0f} | Pos: {pos_str}{current_pnl_str} | Trades: {trades} | WR: {win_rate:.1f}% | PnL: ${total_pnl:+,.2f} | {rate:.1f}/s", flush=True)
                    last_print = now

            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\n" + "=" * 70, flush=True)
        print("FINAL RESULTS", flush=True)
        print("=" * 70, flush=True)
        elapsed = time.time() - start
        win_rate = (wins / trades * 100) if trades > 0 else 0
        print(f"Duration: {elapsed:.1f}s", flush=True)
        print(f"Total Trades: {trades}", flush=True)
        print(f"Win Rate: {win_rate:.1f}%", flush=True)
        print(f"Total PnL: ${total_pnl:+,.2f}", flush=True)
        print(f"Final TRUE Price: ${btc_price:,.2f}", flush=True)
        print("=" * 70, flush=True)

    finally:
        feed_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
