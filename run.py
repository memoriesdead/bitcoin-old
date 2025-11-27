#!/usr/bin/env python3
"""
RENAISSANCE V1-V4 TRADING SYSTEM
================================
Self-contained trading bot using Kelly Criterion & Z-Score signals.
Uses exponential compounding (formulas 308-330) with CCXT for market data.

Usage: python run.py V1|V2|V3|V4|ALL [--duration SECONDS]
"""

import asyncio
import argparse
import time
import numpy as np
from collections import deque

try:
    import ccxt.async_support as ccxt
except ImportError:
    print("Install ccxt: pip install ccxt")
    exit(1)

# =============================================================================
# V1-V4 CONFIGURATIONS
# =============================================================================
CONFIGS = {
    "V1": {
        "name": "KELLY_CONSERVATIVE",
        "kelly_frac": 0.25,
        "zscore_entry": 1.5,
        "profit_target": 0.005,
        "stop_loss": 0.003,
        "max_hold_sec": 10,
    },
    "V2": {
        "name": "MOMENTUM_FILTER",
        "kelly_frac": 0.30,
        "zscore_entry": 1.0,
        "profit_target": 0.008,
        "stop_loss": 0.005,
        "max_hold_sec": 5,
    },
    "V3": {
        "name": "OPTIMAL_REBALANCE",
        "kelly_frac": 0.35,
        "zscore_entry": 1.2,
        "profit_target": 0.010,
        "stop_loss": 0.006,
        "max_hold_sec": 3,
    },
    "V4": {
        "name": "AGGRESSIVE",
        "kelly_frac": 0.40,
        "zscore_entry": 0.8,
        "profit_target": 0.015,
        "stop_loss": 0.010,
        "max_hold_sec": 2,
    },
}


class Strategy:
    """Simple mean-reversion strategy with Kelly sizing."""

    def __init__(self, version: str):
        self.version = version
        self.cfg = CONFIGS[version]

        # State
        self.capital = 10.0
        self.position = None
        self.wins = 0
        self.losses = 0

        # Price history
        self.prices = deque(maxlen=100)

    def update(self, price: float):
        """Add new price."""
        self.prices.append(price)

    def get_zscore(self, price: float) -> float:
        """Calculate z-score."""
        if len(self.prices) < 20:
            return 0.0
        prices = list(self.prices)[-50:]
        mean = np.mean(prices)
        std = np.std(prices)
        return (price - mean) / std if std > 0 else 0.0

    def get_signal(self, price: float) -> tuple:
        """Get trade signal: (direction, size)."""
        if len(self.prices) < 20:
            return 0, 0

        zscore = self.get_zscore(price)
        threshold = self.cfg["zscore_entry"]

        direction = 0
        if zscore < -threshold:
            direction = 1   # Buy oversold
        elif zscore > threshold:
            direction = -1  # Sell overbought

        if direction == 0:
            return 0, 0

        # Kelly-based position size
        edge = abs(zscore) * 0.001
        volatility = np.std(list(self.prices)[-20:]) / np.mean(list(self.prices)[-20:])
        kelly = edge / (volatility ** 2) if volatility > 0 else 0.25
        size = min(kelly * self.cfg["kelly_frac"], 1.0)

        return direction, size

    def check_exit(self, price: float) -> tuple:
        """Check if should exit. Returns (should_exit, reason, pnl_pct)."""
        if not self.position:
            return False, None, 0

        pnl_pct = (price - self.position["entry"]) / self.position["entry"] * self.position["dir"]
        hold_time = time.time() - self.position["time"]

        if pnl_pct >= self.cfg["profit_target"]:
            return True, "TP", pnl_pct
        if pnl_pct <= -self.cfg["stop_loss"]:
            return True, "SL", pnl_pct
        if hold_time >= self.cfg["max_hold_sec"]:
            return True, "TIME", pnl_pct

        return False, None, pnl_pct


async def run_strategy(version: str, duration: int = 300):
    """Run a single strategy version."""

    strategy = Strategy(version)
    cfg = CONFIGS[version]

    print("=" * 60)
    print(f"{version}: {cfg['name']}")
    print(f"Capital: $10.00 | Kelly: {cfg['kelly_frac']*100:.0f}%")
    print(f"Z-Score Entry: {cfg['zscore_entry']} | TP: {cfg['profit_target']*100:.2f}%")
    print("=" * 60)

    exchange = ccxt.kraken({"enableRateLimit": True})
    start = time.time()
    last_status = start

    try:
        while time.time() - start < duration:
            ticker = await exchange.fetch_ticker("BTC/USD")
            price = ticker["last"]
            strategy.update(price)

            # Check exit
            if strategy.position:
                should_exit, reason, pnl_pct = strategy.check_exit(price)
                if should_exit:
                    pnl = strategy.position["value"] * pnl_pct
                    strategy.capital += pnl
                    if pnl > 0:
                        strategy.wins += 1
                    else:
                        strategy.losses += 1
                    print(f"  [{reason}] Exit ${price:,.0f} | PnL: {pnl_pct*100:+.3f}% | Cap: ${strategy.capital:.4f}")
                    strategy.position = None

            # Check entry
            if not strategy.position:
                direction, size = strategy.get_signal(price)
                if direction != 0 and size > 0:
                    value = strategy.capital * size
                    strategy.position = {
                        "dir": direction,
                        "entry": price,
                        "value": value,
                        "time": time.time()
                    }
                    side = "LONG" if direction > 0 else "SHORT"
                    print(f"  [{side}] Entry ${price:,.0f} | Size: ${value:.4f}")

            # Status every 10s
            if time.time() - last_status >= 10:
                elapsed = time.time() - start
                total = strategy.wins + strategy.losses
                wr = strategy.wins / total * 100 if total > 0 else 0
                ret = (strategy.capital - 10) / 10 * 100
                print(f"[{elapsed:.0f}s] Cap: ${strategy.capital:.4f} | Ret: {ret:+.2f}% | W/L: {strategy.wins}/{strategy.losses} ({wr:.0f}%)")
                last_status = time.time()

            await asyncio.sleep(0.5)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await exchange.close()

    # Final results
    total = strategy.wins + strategy.losses
    wr = strategy.wins / total * 100 if total > 0 else 0
    ret = (strategy.capital - 10) / 10 * 100

    print()
    print("=" * 60)
    print(f"FINAL - {version}")
    print(f"Capital: $10.00 -> ${strategy.capital:.4f} ({ret:+.2f}%)")
    print(f"Trades: {total} | Win Rate: {wr:.1f}%")
    print("=" * 60)

    return strategy.capital


async def run_all(duration: int = 300):
    """Run V1-V4 sequentially."""
    results = {}
    for v in ["V1", "V2", "V3", "V4"]:
        results[v] = await run_strategy(v, duration)
        print("\n")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for v, cap in results.items():
        ret = (cap - 10) / 10 * 100
        print(f"{v}: ${cap:.4f} ({ret:+.2f}%)")
    best = max(results, key=results.get)
    print(f"\nBest: {best} with ${results[best]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Renaissance V1-V4")
    parser.add_argument("version", choices=["V1", "V2", "V3", "V4", "ALL"])
    parser.add_argument("--duration", type=int, default=300)
    args = parser.parse_args()

    if args.version == "ALL":
        asyncio.run(run_all(args.duration))
    else:
        asyncio.run(run_strategy(args.version, args.duration))


if __name__ == "__main__":
    main()
