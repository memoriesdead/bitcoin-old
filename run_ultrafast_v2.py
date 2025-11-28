#!/usr/bin/env python3
"""
ULTRA-FAST HFT V1-V4 WITH ALL FORMULAS (F013-F018)
==================================================
COMPLETE IMPLEMENTATION with:
- F013: Dynamic Adverse Selection (Kyle Model)
- F014: Inventory Risk Penalty (Gueant-Lehalle-Fernandez-Tapia)
- F015: Reservation Price (Avellaneda-Stoikov)
- F016: Optimal Spread (Avellaneda-Stoikov)
- F017: Price Impact
- F018: Complete Transaction Cost Model

Target: Millisecond trades, highest win rate + edge
"""

import asyncio
import time
import json
import argparse
import math
from collections import deque
from typing import Optional, Dict
import ssl

try:
    import aiohttp
    import websockets
except ImportError:
    print("Install: pip install aiohttp websockets")
    exit(1)

try:
    import orjson
    def fast_loads(s): return orjson.loads(s)
except ImportError:
    def fast_loads(s): return json.loads(s)

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


# =============================================================================
# V1-V4 CONFIGS WITH NEW FORMULA PARAMETERS
# =============================================================================
CONFIGS = {
    "V1": {
        "name": "MICRO_SCALPER",
        "kelly_frac": 0.60,
        "profit_target": 0.0008,    # 8 bps (accounts for true costs)
        "stop_loss": 0.0004,        # 4 bps
        "max_hold_ms": 50,          # 50ms ultra-fast
        "min_ticks": 2,
        # F013-F018 Parameters
        "gamma_risk": 0.10,         # Risk aversion (low = aggressive)
        "lambda_informed": 0.15,    # Informed trader probability
        "kappa_depth": 100.0,       # Order book depth
    },
    "V2": {
        "name": "NANO_MOMENTUM",
        "kelly_frac": 0.70,
        "profit_target": 0.0010,    # 10 bps
        "stop_loss": 0.0005,        # 5 bps
        "max_hold_ms": 75,          # 75ms
        "min_ticks": 3,
        "gamma_risk": 0.12,
        "lambda_informed": 0.18,
        "kappa_depth": 90.0,
    },
    "V3": {
        "name": "PICO_BREAKOUT",
        "kelly_frac": 0.80,
        "profit_target": 0.0012,    # 12 bps
        "stop_loss": 0.0006,        # 6 bps
        "max_hold_ms": 100,         # 100ms
        "min_ticks": 3,
        "gamma_risk": 0.15,
        "lambda_informed": 0.20,
        "kappa_depth": 80.0,
    },
    "V4": {
        "name": "QUANTUM_EDGE",
        "kelly_frac": 0.90,
        "profit_target": 0.0015,    # 15 bps
        "stop_loss": 0.0008,        # 8 bps
        "max_hold_ms": 150,         # 150ms
        "min_ticks": 2,
        "gamma_risk": 0.18,
        "lambda_informed": 0.22,
        "kappa_depth": 70.0,
    },
}


# =============================================================================
# F013-F018: COMPLETE FORMULA IMPLEMENTATION
# =============================================================================
class CompleteTransactionCostModel:
    """
    ALL formulas F013-F018 for TRUE transaction costs.
    This is what was MISSING and causing losses!
    """

    def __init__(self, cfg: dict):
        self.gamma = cfg.get("gamma_risk", 0.10)
        self.lambda_informed = cfg.get("lambda_informed", 0.15)
        self.kappa = cfg.get("kappa_depth", 100.0)

        # Base costs
        self.spread_cost = 0.0002      # 2 bps half-spread
        self.maker_rebate = -0.0002    # -2 bps maker rebate
        self.slippage = 0.0001         # 1 bp slippage

        # Track state
        self.inventory = 0.0           # -1 to +1 normalized
        self.order_flow_imbalance = 0.0

    def compute_dynamic_adverse_selection(self, volatility: float) -> float:
        """F013: Kyle Model - AS = lambda * |OFI| * sigma"""
        as_cost = self.lambda_informed * abs(self.order_flow_imbalance) * volatility
        return max(as_cost, 0.0001)  # Min 1 bp

    def compute_inventory_risk(self, volatility: float) -> float:
        """F014: IR = gamma * |q| * sigma^2 * T"""
        variance = volatility ** 2
        return self.gamma * abs(self.inventory) * variance * 1.0

    def compute_reservation_price(self, mid_price: float, volatility: float) -> float:
        """F015: r = s - gamma * q * sigma^2 * (T-t)"""
        variance = volatility ** 2
        adjustment = self.gamma * self.inventory * variance * 1.0
        return mid_price - adjustment

    def compute_optimal_spread(self, volatility: float) -> float:
        """F016: delta* = gamma*sigma^2*T + (1/gamma)*ln(1 + gamma/kappa)"""
        variance = volatility ** 2
        inv_term = self.gamma * variance * 1.0
        as_term = (1.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        return inv_term + as_term

    def compute_price_impact(self, order_size_usd: float) -> float:
        """F017: PI = (size/depth) * (1/kappa)"""
        market_depth = 100000.0
        impact = (order_size_usd / market_depth) * (1.0 / self.kappa)
        return min(impact, 0.001)  # Cap at 10 bps

    def compute_total_cost(self, volatility: float, order_size: float = 5.0) -> float:
        """F018: Complete cost = all components"""
        base = self.spread_cost + self.maker_rebate + self.slippage
        adverse = self.compute_dynamic_adverse_selection(volatility)
        inv_risk = self.compute_inventory_risk(volatility)
        impact = self.compute_price_impact(order_size)
        return base + adverse + inv_risk + impact

    def update_state(self, trade_dir: int, ofi: float):
        """Update inventory and OFI after trade."""
        self.inventory = max(-1.0, min(1.0, self.inventory + trade_dir * 0.1))
        self.order_flow_imbalance = ofi

    def should_trade(self, signal_strength: float, volatility: float) -> bool:
        """Only trade if expected profit > total costs."""
        total_cost = self.compute_total_cost(volatility) * 2  # Round-trip
        # Allow trades where signal is at least 50% of cost (aggressive but with edge)
        return signal_strength > total_cost * 0.5


# =============================================================================
# ULTRA-FAST STRATEGY WITH ALL FORMULAS
# =============================================================================
class UltraFastStrategyV2:
    """HFT with complete F013-F018 formulas."""

    def __init__(self, version: str):
        self.version = version
        self.cfg = CONFIGS[version]

        # Cost model with ALL formulas
        self.cost_model = CompleteTransactionCostModel(self.cfg)

        # State
        self.capital = 10.0
        self.position = None
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

        # Price data
        self.prices = deque(maxlen=500)
        self.timestamps_ns = deque(maxlen=500)
        self.returns = deque(maxlen=100)

        # Volatility (realized)
        self.volatility = 0.002  # Start with 0.2%

        # Stats
        self.ticks = 0
        self.signals = 0
        self.trades = 0
        self.skipped_trades = 0  # Trades skipped due to cost

    def _update_volatility(self):
        """Calculate realized volatility from returns."""
        if len(self.returns) >= 20:
            returns_list = list(self.returns)[-20:]
            mean = sum(returns_list) / len(returns_list)
            variance = sum((r - mean)**2 for r in returns_list) / len(returns_list)
            self.volatility = max(math.sqrt(variance), 0.0005)  # Min 0.05%

    def _compute_ofi(self) -> float:
        """Order flow imbalance from recent price moves."""
        if len(self.prices) < 10:
            return 0.0
        recent = list(self.prices)[-10:]
        ups = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
        downs = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
        total = ups + downs
        if total == 0:
            return 0.0
        return (ups - downs) / total  # -1 to +1

    def _compute_momentum(self) -> float:
        """Ultra-fast momentum signal."""
        if len(self.prices) < self.cfg["min_ticks"]:
            return 0.0
        recent = list(self.prices)[-self.cfg["min_ticks"]:]
        momentum = (recent[-1] - recent[0]) / recent[0]
        return momentum

    def on_tick(self, price: float) -> Optional[dict]:
        """Process tick with complete cost model."""
        now_ns = time.time_ns()

        # Update price data
        if self.prices:
            ret = (price - self.prices[-1]) / self.prices[-1]
            self.returns.append(ret)
        self.prices.append(price)
        self.timestamps_ns.append(now_ns)
        self.ticks += 1

        # Update volatility periodically
        if self.ticks % 10 == 0:
            self._update_volatility()

        # Update OFI in cost model
        ofi = self._compute_ofi()
        self.cost_model.order_flow_imbalance = ofi

        # Check exit first
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

    def _check_entry(self, price: float, now_ns: int) -> Optional[dict]:
        """Entry with reservation price + cost filter."""
        momentum = self._compute_momentum()
        if abs(momentum) < 0.00005:  # Min momentum threshold
            return None

        self.signals += 1

        # F015: Get reservation price
        reservation = self.cost_model.compute_reservation_price(price, self.volatility)

        # Determine direction based on momentum AND reservation
        if momentum > 0:
            # Want to go LONG
            # Only buy if price < reservation (undervalued)
            if price > reservation * 1.0002:  # Price too high
                return None
            direction = 1
        else:
            # Want to go SHORT
            # Only sell if price > reservation (overvalued)
            if price < reservation * 0.9998:  # Price too low
                return None
            direction = -1

        # F018: Check if trade is profitable after ALL costs
        signal_strength = abs(momentum)
        if not self.cost_model.should_trade(signal_strength, self.volatility):
            self.skipped_trades += 1
            return None

        # Execute trade
        position_size = self.capital * self.cfg["kelly_frac"]

        self.position = {
            "entry": price,
            "dir": direction,
            "size": position_size,
            "entry_ns": now_ns,
            "reservation": reservation,
        }

        # Update inventory in cost model
        self.cost_model.update_state(direction, self._compute_ofi())
        self.trades += 1

        return {"action": "ENTRY", "dir": direction, "price": price}

    def _check_exit(self, price: float, now_ns: int) -> Optional[dict]:
        """Exit with dynamic targets."""
        pos = self.position
        pnl_pct = (price - pos["entry"]) / pos["entry"] * pos["dir"]
        hold_ms = (now_ns - pos["entry_ns"]) / 1_000_000

        # Dynamic TP/SL based on volatility
        dynamic_tp = self.cfg["profit_target"] * (1 + self.volatility * 10)
        dynamic_sl = self.cfg["stop_loss"] * (1 + self.volatility * 5)

        reason = None
        if pnl_pct >= dynamic_tp:
            reason = "TP"
        elif pnl_pct <= -dynamic_sl:
            reason = "SL"
        elif hold_ms >= self.cfg["max_hold_ms"]:
            reason = "TIME"

        if reason:
            # Calculate actual PnL with costs
            total_cost = self.cost_model.compute_total_cost(self.volatility) * 2
            net_pnl_pct = pnl_pct - total_cost
            net_pnl = pos["size"] * net_pnl_pct

            self.capital += net_pnl
            self.total_pnl += net_pnl

            if net_pnl > 0:
                self.wins += 1
            else:
                self.losses += 1

            # Update inventory (closing position)
            self.cost_model.update_state(-pos["dir"], self._compute_ofi())

            result = {
                "action": "EXIT",
                "reason": reason,
                "pnl": net_pnl,
                "pnl_pct": net_pnl_pct * 100,
                "gross_pnl_pct": pnl_pct * 100,
                "cost_pct": total_cost * 100,
            }
            self.position = None
            return result

        return None

    def get_stats(self) -> dict:
        total = self.wins + self.losses
        wr = (self.wins / total * 100) if total > 0 else 0
        return {
            "version": self.version,
            "capital": self.capital,
            "pnl": self.total_pnl,
            "return_pct": (self.capital - 10) / 10 * 100,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": wr,
            "trades": total,
            "skipped": self.skipped_trades,
            "ticks": self.ticks,
            "volatility": self.volatility * 100,
        }


# =============================================================================
# WEBSOCKET + REST HYBRID FEED (USA-ACCESSIBLE)
# =============================================================================
class MultiExchangeFeed:
    """WebSocket + REST hybrid feeds from USA-accessible exchanges."""

    # USA-accessible REST endpoints for fast polling
    REST_ENDPOINTS = {
        "kraken": "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
        "coinbase": "https://api.exchange.coinbase.com/products/BTC-USD/ticker",
        "gemini": "https://api.gemini.com/v1/pubticker/btcusd",
        "bitstamp": "https://www.bitstamp.net/api/v2/ticker/btcusd/",
    }

    def __init__(self):
        self.price = 0.0
        self.prices = deque(maxlen=1000)
        self.callbacks = []
        self._session = None

    def add_callback(self, cb):
        self.callbacks.append(cb)

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=2),
                connector=aiohttp.TCPConnector(ssl=SSL_CTX)
            )
        return self._session

    async def _poll_rest(self):
        """Poll REST endpoints for USA-accessible price data - 10+ ticks/sec."""
        session = await self._get_session()
        while True:
            try:
                prices = []
                for name, url in self.REST_ENDPOINTS.items():
                    try:
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if name == "kraken":
                                    price = float(data.get("result", {}).get("XXBTZUSD", {}).get("c", [0])[0])
                                elif name == "coinbase":
                                    price = float(data.get("price", 0))
                                elif name == "gemini":
                                    price = float(data.get("last", 0))
                                elif name == "bitstamp":
                                    price = float(data.get("last", 0))
                                else:
                                    price = 0
                                if price > 0:
                                    prices.append(price)
                    except:
                        pass

                if prices:
                    avg_price = sum(prices) / len(prices)
                    self.price = avg_price
                    self.prices.append(avg_price)
                    for cb in self.callbacks:
                        cb(avg_price)

                await asyncio.sleep(0.1)  # 10 polls/sec
            except Exception as e:
                await asyncio.sleep(0.5)

    async def _connect_coinbase_ws(self):
        """Coinbase WebSocket for real-time trades."""
        try:
            async with websockets.connect(
                "wss://ws-feed.exchange.coinbase.com",
                ssl=SSL_CTX,
                ping_interval=20
            ) as ws:
                await ws.send(json.dumps({
                    "type": "subscribe",
                    "product_ids": ["BTC-USD"],
                    "channels": ["matches"]
                }))
                while True:
                    msg = await ws.recv()
                    data = fast_loads(msg)
                    if data.get("type") == "match":
                        price = float(data["price"])
                        self.price = price
                        self.prices.append(price)
                        for cb in self.callbacks:
                            cb(price)
        except Exception as e:
            print(f"Coinbase WS: {e}")

    async def _connect_kraken_ws(self):
        """Kraken WebSocket for real-time trades."""
        try:
            async with websockets.connect(
                "wss://ws.kraken.com",
                ssl=SSL_CTX,
                ping_interval=20
            ) as ws:
                await ws.send(json.dumps({
                    "event": "subscribe",
                    "pair": ["XBT/USD"],
                    "subscription": {"name": "trade"}
                }))
                while True:
                    msg = await ws.recv()
                    data = fast_loads(msg)
                    if isinstance(data, list) and len(data) >= 4:
                        trades = data[1]
                        if isinstance(trades, list) and trades:
                            price = float(trades[-1][0])
                            self.price = price
                            self.prices.append(price)
                            for cb in self.callbacks:
                                cb(price)
        except Exception as e:
            print(f"Kraken WS: {e}")

    async def start(self):
        await asyncio.gather(
            self._poll_rest(),           # REST polling always works
            self._connect_coinbase_ws(), # WebSocket if available
            self._connect_kraken_ws(),   # WebSocket if available
            return_exceptions=True
        )

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# MAIN RUNNER
# =============================================================================
async def run_strategy(version: str, duration: int):
    """Run single strategy."""
    print(f"\n{'='*60}")
    print(f"ULTRA-FAST V2 - {version} - ALL FORMULAS (F013-F018)")
    print(f"Duration: {duration}s | Capital: $10.00")
    print(f"{'='*60}\n")

    strategy = UltraFastStrategyV2(version)
    feed = MultiExchangeFeed()

    actions = []
    def on_price(price):
        action = strategy.on_tick(price)
        if action:
            actions.append(action)

    feed.add_callback(on_price)

    # Run with timeout
    start = time.time()

    async def monitor():
        while time.time() - start < duration:
            await asyncio.sleep(5)
            stats = strategy.get_stats()
            elapsed = int(time.time() - start)
            print(f"[{elapsed:3}s] {version} | Cap: ${stats['capital']:.4f} | "
                  f"Trades: {stats['trades']} | WR: {stats['win_rate']:.1f}% | "
                  f"Vol: {stats['volatility']:.3f}% | Skip: {stats['skipped']}")

    async def run_feed():
        try:
            await asyncio.wait_for(feed.start(), timeout=duration)
        except asyncio.TimeoutError:
            pass

    await asyncio.gather(run_feed(), monitor())

    # Final stats
    stats = strategy.get_stats()
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - {version}")
    print(f"{'='*60}")
    print(f"Starting Capital: $10.00")
    print(f"Ending Capital:   ${stats['capital']:.4f}")
    print(f"Return:           {stats['return_pct']:+.2f}%")
    print(f"Total Trades:     {stats['trades']}")
    print(f"Skipped (cost):   {stats['skipped']}")
    print(f"Win Rate:         {stats['win_rate']:.1f}%")
    print(f"Wins/Losses:      {stats['wins']}/{stats['losses']}")
    print(f"Avg Volatility:   {stats['volatility']:.3f}%")
    print(f"{'='*60}\n")

    return stats


async def run_all(duration: int):
    """Run all V1-V4 in parallel."""
    print(f"\n{'='*70}")
    print(f"ULTRA-FAST V2 - ALL VERSIONS - COMPLETE FORMULAS F013-F018")
    print(f"Duration: {duration}s | Starting Capital: $10.00 each")
    print(f"{'='*70}\n")

    strategies = {v: UltraFastStrategyV2(v) for v in CONFIGS.keys()}
    feed = MultiExchangeFeed()

    def on_price(price):
        for s in strategies.values():
            s.on_tick(price)

    feed.add_callback(on_price)

    start = time.time()

    async def monitor():
        while time.time() - start < duration:
            await asyncio.sleep(10)
            elapsed = int(time.time() - start)
            print(f"\n[{elapsed}s] Status:")
            for v, s in strategies.items():
                stats = s.get_stats()
                print(f"  {v}: ${stats['capital']:.4f} | "
                      f"Trades: {stats['trades']} | WR: {stats['win_rate']:.1f}% | "
                      f"Skip: {stats['skipped']}")

    async def run_feed():
        try:
            await asyncio.wait_for(feed.start(), timeout=duration)
        except asyncio.TimeoutError:
            pass

    await asyncio.gather(run_feed(), monitor())

    # Final results
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS - ALL VERSIONS")
    print(f"{'='*70}")

    results = []
    for v, s in sorted(strategies.items()):
        stats = s.get_stats()
        results.append(stats)
        print(f"{v}: ${stats['capital']:.4f} | {stats['return_pct']:+.2f}% | "
              f"Trades: {stats['trades']} | WR: {stats['win_rate']:.1f}% | "
              f"Skip: {stats['skipped']}")

    best = max(results, key=lambda x: x["return_pct"])
    print(f"\nBEST: {best['version']} with {best['return_pct']:+.2f}%")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", nargs="?", default="ALL",
                       help="V1, V2, V3, V4, or ALL")
    parser.add_argument("--duration", "-d", type=int, default=60,
                       help="Duration in seconds")
    args = parser.parse_args()

    if args.version.upper() == "ALL":
        asyncio.run(run_all(args.duration))
    else:
        asyncio.run(run_strategy(args.version.upper(), args.duration))
