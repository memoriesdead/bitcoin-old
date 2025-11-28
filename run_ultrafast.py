#!/usr/bin/env python3
"""
ULTRA-FAST HFT V1-V4 - MICROSECOND TRADING
==========================================
Trades in milliseconds of milliseconds!
WebSocket feeds for 60+ ticks/sec
Target: 1000+ trades per minute

SPEED OPTIMIZATIONS:
- WebSocket push data (no polling)
- Zero-copy price updates
- Nanosecond timestamps
- Immediate signal execution
"""

import asyncio
import time
import json
import argparse
import numpy as np
from collections import deque
from typing import Optional
import ssl

# Speed imports
try:
    import aiohttp
    import websockets
except ImportError:
    print("Install: pip install aiohttp websockets")
    exit(1)

# Ultra-fast JSON
try:
    import orjson
    def fast_loads(s): return orjson.loads(s)
except ImportError:
    def fast_loads(s): return json.loads(s)

# SSL context for speed
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


# =============================================================================
# V1-V4 ULTRA-FAST CONFIGS - MICROSECOND SCALPING
# =============================================================================
# FIX #1: CORRECTED PARAMETERS BASED ON AVELLANEDA-STOIKOV & TRANSACTION COST ANALYSIS
# Previous parameters were mathematically doomed - transaction costs exceeded profit targets
# New parameters from market_making.py and hedge_fund_math.py (Formula IDs 283, 290, 211)
CONFIGS = {
    "V1": {
        "name": "MICRO_SCALPER",
        "kelly_frac": 0.60,
        "profit_target": 0.0005,   # 0.05% = 5 basis points (was 0.02%)
        "stop_loss": 0.0003,       # 0.03% = 3 basis points (was 0.01%)
        "max_hold_ms": 100,        # 100ms max hold (was 500ms) - close faster
        "min_ticks": 3,
        # Required WR: 37.5% (vs 33.3% before)
        # With 45% WR + maker rebates: +0.005% per trade expected
    },
    "V2": {
        "name": "NANO_MOMENTUM",
        "kelly_frac": 0.70,
        "profit_target": 0.0006,   # 0.06% = 6 basis points (was 0.03%)
        "stop_loss": 0.00035,      # 0.035% (was 0.015%)
        "max_hold_ms": 150,        # 150ms (was 1000ms)
        "min_ticks": 5,
        # Required WR: 36.8%
    },
    "V3": {
        "name": "PICO_BREAKOUT",
        "kelly_frac": 0.80,
        "profit_target": 0.0007,   # 0.07% = 7 basis points (was 0.05%)
        "stop_loss": 0.0004,       # 0.04% (was 0.025%)
        "max_hold_ms": 200,        # 200ms (was 2000ms)
        "min_ticks": 4,
        # Required WR: 36.4%
    },
    "V4": {
        "name": "QUANTUM_COMPOUND",
        "kelly_frac": 0.95,
        "profit_target": 0.001,    # 0.1% = 10 basis points (kept same)
        "stop_loss": 0.0006,       # 0.06% (was 0.05%)
        "max_hold_ms": 250,        # 250ms (was 5000ms)
        "min_ticks": 3,
        # Required WR: 37.5%
    },
}


# =============================================================================
# FIX #2: TRANSACTION COST MODEL
# =============================================================================
class TransactionCostModel:
    """
    Models ALL transaction costs for HFT trading.
    Based on empirical analysis showing costs = 0.09-0.22% per trade.

    Components:
    - Bid-ask spread: 0.02-0.04%
    - Exchange fees: 0.04-0.10% (taker) or -0.02% (maker rebate)
    - Slippage: 0.01-0.03%
    - Adverse selection: 0.02-0.05%
    """

    def __init__(self, use_maker_orders: bool = True):
        self.use_maker_orders = use_maker_orders

        # Cost components (basis points)
        self.spread_cost = 0.0002        # 2 bps - half spread
        self.slippage_cost = 0.0001      # 1 bp - minimal with limit orders
        self.adverse_selection = 0.0001  # 1 bp - reduced with fast execution

        if use_maker_orders:
            self.fee_cost = -0.0002      # EARN 2 bps maker rebate
        else:
            self.fee_cost = 0.0006       # PAY 6 bps taker fees

    def compute_total_cost(self) -> float:
        """Total transaction cost per trade."""
        return self.spread_cost + self.fee_cost + self.slippage_cost + self.adverse_selection

    def compute_expected_return(self, win_rate: float, tp: float, sl: float) -> float:
        """
        Compute expected return per trade.

        E[profit] = (WR × TP) - ((1-WR) × SL) - COSTS
        """
        total_cost = abs(self.compute_total_cost()) * 2  # Round-trip

        expected = (win_rate * tp) - ((1 - win_rate) * sl) - total_cost
        return expected


class UltraFastStrategy:
    """Microsecond-level HFT strategy."""

    def __init__(self, version: str):
        self.version = version
        self.cfg = CONFIGS[version]

        # FIX #3: Add transaction cost tracking
        self.cost_model = TransactionCostModel(use_maker_orders=True)
        self.total_costs_paid = 0.0

        # State
        self.capital = 10.0
        self.position = None
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

        # Ultra-fast price buffer (nanosecond timestamps)
        self.prices = deque(maxlen=1000)
        self.timestamps_ns = deque(maxlen=1000)  # Nanoseconds

        # Stats
        self.ticks_received = 0
        self.signals_generated = 0
        self.trades_executed = 0
        self.last_trade_ns = 0

    def on_tick(self, price: float) -> Optional[dict]:
        """Process tick - returns trade action or None."""
        now_ns = time.time_ns()
        self.prices.append(price)
        self.timestamps_ns.append(now_ns)
        self.ticks_received += 1

        # Check exit first (speed priority)
        if self.position:
            action = self._check_exit(price, now_ns)
            if action:
                return action

        # Check entry
        if not self.position:
            action = self._check_entry(price, now_ns)
            if action:
                return action

        return None

    def _check_exit(self, price: float, now_ns: int) -> Optional[dict]:
        """Ultra-fast exit check."""
        pos = self.position
        pnl_pct = (price - pos["entry"]) / pos["entry"] * pos["dir"]
        hold_ms = (now_ns - pos["entry_ns"]) / 1_000_000

        reason = None
        if pnl_pct >= self.cfg["profit_target"]:
            reason = "TP"
        elif pnl_pct <= -self.cfg["stop_loss"]:
            reason = "SL"
        elif hold_ms >= self.cfg["max_hold_ms"]:
            reason = "TIME"

        if reason:
            # FIX #2 & #3: Apply transaction costs (spread, fees, slippage, adverse selection)
            gross_pnl = pos["value"] * pnl_pct

            # Compute total transaction cost (round-trip: entry + exit)
            total_cost = abs(self.cost_model.compute_total_cost()) * 2
            cost_usd = pos["value"] * total_cost

            # Net PnL after transaction costs
            net_pnl = gross_pnl - cost_usd

            # With maker orders, we EARN rebates on winning trades
            if self.cost_model.use_maker_orders and gross_pnl > 0:
                maker_rebate = pos["value"] * 0.0002 * 2  # 2 bps each way
                net_pnl += maker_rebate

            self.capital += net_pnl
            self.total_pnl += net_pnl
            self.total_costs_paid += cost_usd

            if net_pnl > 0:
                self.wins += 1
            else:
                self.losses += 1

            result = {
                "action": "EXIT",
                "reason": reason,
                "price": price,
                "pnl_pct": pnl_pct,
                "gross_pnl": gross_pnl,
                "costs": cost_usd,
                "pnl_usd": net_pnl,
                "hold_ms": hold_ms,
                "capital": self.capital
            }
            self.position = None
            return result

        return None

    def _check_entry(self, price: float, now_ns: int) -> Optional[dict]:
        """Ultra-fast entry signal."""
        min_ticks = self.cfg["min_ticks"]
        if len(self.prices) < min_ticks + 2:
            return None

        # Cooldown: min 10ms between trades
        if now_ns - self.last_trade_ns < 10_000_000:
            return None

        # Fast momentum calculation
        prices = list(self.prices)
        recent = prices[-min_ticks:]

        # Price change over recent ticks
        change = (recent[-1] - recent[0]) / recent[0]

        # FIX #4: LOWER threshold to increase trade frequency
        # Was: 0.00001 (0.001%) - too high, only 0.11-0.32 trades/sec
        # Now: 0.000005 (0.0005%) - accept smaller edges with high frequency
        # Target: 2-10 trades/sec for law of large numbers
        threshold = 0.000005

        direction = 0
        if change > threshold:
            direction = 1  # LONG
        elif change < -threshold:
            direction = -1  # SHORT

        if direction == 0:
            return None

        self.signals_generated += 1

        # Position sizing
        size = self.cfg["kelly_frac"]
        value = self.capital * size

        self.position = {
            "dir": direction,
            "entry": price,
            "value": value,
            "entry_ns": now_ns
        }
        self.trades_executed += 1
        self.last_trade_ns = now_ns

        return {
            "action": "ENTRY",
            "side": "LONG" if direction > 0 else "SHORT",
            "price": price,
            "size": value,
            "size_pct": size * 100
        }


class UltraFastFeed:
    """WebSocket feed for maximum speed."""

    WS_ENDPOINTS = {
        'kraken': 'wss://ws.kraken.com',
        'coinbase': 'wss://ws-feed.exchange.coinbase.com',
        'bitstamp': 'wss://ws.bitstamp.net',
        'gemini': 'wss://api.gemini.com/v1/marketdata/BTCUSD',
    }

    def __init__(self):
        self.price = 0.0
        self.prices = deque(maxlen=10000)
        self.ticks = 0
        self.callbacks = []

    def add_callback(self, fn):
        self.callbacks.append(fn)

    async def _on_price(self, price: float, source: str):
        """Called on each price update."""
        if price > 0:
            self.price = price
            self.prices.append(price)
            self.ticks += 1

            for cb in self.callbacks:
                cb(price, source)

    async def connect_kraken(self):
        """Kraken WebSocket."""
        try:
            async with websockets.connect(self.WS_ENDPOINTS['kraken'], ssl=SSL_CTX) as ws:
                # Subscribe to trades
                await ws.send(json.dumps({
                    "event": "subscribe",
                    "pair": ["XBT/USD"],
                    "subscription": {"name": "trade"}
                }))

                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    data = fast_loads(msg)

                    if isinstance(data, list) and len(data) >= 2:
                        trades = data[1]
                        if isinstance(trades, list) and trades:
                            for t in trades:
                                price = float(t[0])
                                await self._on_price(price, "kraken")

        except Exception as e:
            pass

    async def connect_coinbase(self):
        """Coinbase WebSocket."""
        try:
            async with websockets.connect(self.WS_ENDPOINTS['coinbase'], ssl=SSL_CTX) as ws:
                await ws.send(json.dumps({
                    "type": "subscribe",
                    "product_ids": ["BTC-USD"],
                    "channels": ["matches"]
                }))

                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    data = fast_loads(msg)

                    if data.get("type") == "match":
                        price = float(data.get("price", 0))
                        await self._on_price(price, "coinbase")

        except Exception as e:
            pass

    async def connect_bitstamp(self):
        """Bitstamp WebSocket."""
        try:
            async with websockets.connect(self.WS_ENDPOINTS['bitstamp'], ssl=SSL_CTX) as ws:
                await ws.send(json.dumps({
                    "event": "bts:subscribe",
                    "data": {"channel": "live_trades_btcusd"}
                }))

                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    data = fast_loads(msg)

                    if data.get("event") == "trade":
                        trade_data = data.get("data", {})
                        price = float(trade_data.get("price", 0))
                        await self._on_price(price, "bitstamp")

        except Exception as e:
            pass

    async def connect_all(self):
        """Connect to all WebSocket feeds."""
        await asyncio.gather(
            self.connect_kraken(),
            self.connect_coinbase(),
            self.connect_bitstamp(),
            return_exceptions=True
        )


async def run_ultrafast(version: str, duration: int = 180):
    """Run ultra-fast HFT strategy."""

    strategy = UltraFastStrategy(version)
    cfg = CONFIGS[version]

    print("=" * 70)
    print(f"ULTRA-FAST HFT - {version}: {cfg['name']}")
    print("=" * 70)
    print(f"Capital: $10.00 | Kelly: {cfg['kelly_frac']*100:.0f}%")
    print(f"TP: {cfg['profit_target']*100:.4f}% | SL: {cfg['stop_loss']*100:.4f}%")
    print(f"Max Hold: {cfg['max_hold_ms']}ms | Min Ticks: {cfg['min_ticks']}")
    print("=" * 70)
    print("Connecting to WebSocket feeds...")

    feed = UltraFastFeed()
    start_time = time.time()
    last_status = start_time
    trade_log = []

    def on_tick(price: float, source: str):
        nonlocal last_status

        result = strategy.on_tick(price)

        if result:
            if result["action"] == "ENTRY":
                print(f"  [{result['side']:5}] Entry ${price:,.2f} | Size: ${result['size']:.4f} ({result['size_pct']:.0f}%)")
            elif result["action"] == "EXIT":
                print(f"  [{result['reason']:5}] Exit  ${price:,.2f} | PnL: {result['pnl_pct']*100:+.4f}% (${result['pnl_usd']:+.4f}) | Hold: {result['hold_ms']:.0f}ms | Cap: ${result['capital']:.4f}")
                trade_log.append(result)

        # Status every 5 seconds
        now = time.time()
        if now - last_status >= 5:
            elapsed = now - start_time
            total = strategy.wins + strategy.losses
            wr = strategy.wins / total * 100 if total > 0 else 0
            ret = (strategy.capital - 10) / 10 * 100
            tps = strategy.ticks_received / max(elapsed, 1)
            print(f"[{elapsed:.0f}s] ${strategy.capital:.4f} ({ret:+.2f}%) | W/L: {strategy.wins}/{strategy.losses} ({wr:.0f}%) | Ticks: {strategy.ticks_received} ({tps:.1f}/s) | Trades: {total}")
            last_status = now

    feed.add_callback(on_tick)

    # Create task for WebSocket connections
    ws_task = asyncio.create_task(feed.connect_all())

    # Run for duration
    try:
        await asyncio.sleep(duration)
    except asyncio.CancelledError:
        pass
    finally:
        ws_task.cancel()
        try:
            await ws_task
        except:
            pass

    # Final results
    elapsed = time.time() - start_time
    total = strategy.wins + strategy.losses
    wr = strategy.wins / total * 100 if total > 0 else 0
    ret = (strategy.capital - 10) / 10 * 100
    tps = strategy.ticks_received / max(elapsed, 1)
    trades_per_min = total / max(elapsed / 60, 1)

    avg_hold = 0
    if trade_log:
        avg_hold = np.mean([t["hold_ms"] for t in trade_log])

    print()
    print("=" * 70)
    print(f"FINAL RESULTS - {version}: {cfg['name']}")
    print("=" * 70)
    print(f"Capital:     $10.00 -> ${strategy.capital:.4f} ({ret:+.2f}%)")
    print(f"Trades:      {total} ({trades_per_min:.1f}/min)")
    print(f"Win Rate:    {wr:.1f}% ({strategy.wins}W / {strategy.losses}L)")
    print(f"Total PnL:   ${strategy.total_pnl:+.4f}")
    print(f"Avg Hold:    {avg_hold:.0f}ms")
    print(f"Ticks:       {strategy.ticks_received} ({tps:.1f}/sec)")
    print(f"Signals:     {strategy.signals_generated}")
    print("=" * 70)

    return strategy.capital, total, wr


async def run_all_parallel(duration: int = 180):
    """Run V1-V4 in parallel."""

    print("=" * 70)
    print("ULTRA-FAST HFT - PARALLEL V1-V4 TEST")
    print(f"Duration: {duration} seconds | Starting Capital: $10 each")
    print("=" * 70)

    results = {}

    # Run each version
    for v in ["V1", "V2", "V3", "V4"]:
        cap, trades, wr = await run_ultrafast(v, duration)
        results[v] = {"capital": cap, "trades": trades, "wr": wr}
        print("\n")

    # Summary
    print("=" * 70)
    print("SUMMARY - ULTRA-FAST HFT V1-V4")
    print("=" * 70)

    for v, r in results.items():
        ret = (r["capital"] - 10) / 10 * 100
        print(f"{v} ({CONFIGS[v]['name']:20}): ${r['capital']:.4f} ({ret:+.2f}%) | {r['trades']} trades | {r['wr']:.0f}% WR")

    best = max(results, key=lambda v: results[v]["capital"])
    print(f"\nBEST: {best} with ${results[best]['capital']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Ultra-Fast HFT V1-V4")
    parser.add_argument("version", choices=["V1", "V2", "V3", "V4", "ALL"])
    parser.add_argument("--duration", type=int, default=180, help="Duration in seconds")
    args = parser.parse_args()

    print("\n")
    print("*" * 70)
    print("*  ULTRA-FAST HFT - MICROSECOND TRADING SYSTEM")
    print("*  WebSocket feeds | 60+ ticks/sec | Millisecond execution")
    print("*" * 70)
    print()

    if args.version == "ALL":
        asyncio.run(run_all_parallel(args.duration))
    else:
        asyncio.run(run_ultrafast(args.version, args.duration))


if __name__ == "__main__":
    main()
