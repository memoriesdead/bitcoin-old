#!/usr/bin/env python3
"""
HFT MEGA SYSTEM - 300,000 to 1,000,000 TRADES
=============================================
Ultimate high-frequency trading system designed for:
- Capturing EVERY tick from ALL exchanges
- 300,000+ trades per session
- Millisecond precision execution
- Volume-weighted strategies

BTC Volume Math:
- 24h Volume: ~$65,000,000,000
- Per Second: ~$752,315
- Per Millisecond: ~$752
- Our target: 0.001% of flow = $750K/day opportunity
"""

import asyncio
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional

# Import our feeds
try:
    from coinapi_feed import get_best_feed, CoinAPIFeed, FallbackMultiFeed
except ImportError:
    from ultra_fast_feed import UltraFastFeed

# =============================================================================
# VOLUME MATH CONSTANTS
# =============================================================================
BTC_24H_VOLUME = 65_000_000_000      # $65 billion
VOLUME_PER_HOUR = BTC_24H_VOLUME / 24
VOLUME_PER_MINUTE = VOLUME_PER_HOUR / 60
VOLUME_PER_SECOND = VOLUME_PER_MINUTE / 60  # ~$752,315
VOLUME_PER_MS = VOLUME_PER_SECOND / 1000    # ~$752

# Our capital relative to market (for slippage calculation)
OUR_CAPITAL = 10.0
MARKET_IMPACT_THRESHOLD = 0.0001  # 0.01% of per-second volume before impact

# =============================================================================
# HFT STRATEGY CONFIGURATIONS
# =============================================================================
STRATEGIES = {
    "ULTRA_SCALP": {
        "name": "Ultra Scalper",
        "description": "Sub-second momentum trades",
        "min_ticks": 5,
        "signal_threshold": 0.00005,  # 0.005% move triggers
        "profit_target": 0.0003,      # 0.03% target
        "stop_loss": 0.00015,         # 0.015% stop
        "max_hold_ms": 5000,          # 5 seconds max
        "kelly_frac": 0.50,
        "trade_on_tick": True,        # Trade every tick if signal
    },
    "MOMENTUM_BURST": {
        "name": "Momentum Burst",
        "description": "Trade on volume spikes",
        "min_ticks": 10,
        "volume_spike_mult": 1.5,     # 1.5x average volume
        "profit_target": 0.0005,      # 0.05% target
        "stop_loss": 0.00025,         # 0.025% stop
        "max_hold_ms": 10000,         # 10 seconds max
        "kelly_frac": 0.60,
    },
    "SPREAD_CAPTURE": {
        "name": "Spread Capture",
        "description": "Bid-ask spread trading",
        "min_ticks": 3,
        "min_spread_pct": 0.0002,     # 0.02% minimum spread
        "profit_target": 0.0002,      # 0.02% (half the spread)
        "stop_loss": 0.0001,          # 0.01% stop
        "max_hold_ms": 3000,          # 3 seconds max
        "kelly_frac": 0.70,
    },
    "VWAP_REVERSION": {
        "name": "VWAP Mean Reversion",
        "description": "Trade back to VWAP",
        "min_ticks": 50,
        "vwap_deviation": 0.0003,     # 0.03% from VWAP triggers
        "profit_target": 0.0002,      # 0.02% target (back to VWAP)
        "stop_loss": 0.0004,          # 0.04% stop
        "max_hold_ms": 30000,         # 30 seconds max
        "kelly_frac": 0.55,
    },
    "CROSS_EXCHANGE_ARB": {
        "name": "Cross-Exchange Arbitrage",
        "description": "Price discrepancy across exchanges",
        "min_exchanges": 2,
        "min_arb_pct": 0.0001,        # 0.01% minimum arbitrage
        "profit_target": 0.00005,     # 0.005% after fees
        "stop_loss": 0.0001,          # 0.01% stop
        "max_hold_ms": 2000,          # 2 seconds max
        "kelly_frac": 0.80,           # High confidence trades
    }
}


@dataclass
class Position:
    """Active position."""
    direction: int  # 1 = long, -1 = short
    entry_price: float
    entry_time_ms: int
    size: float
    strategy: str


@dataclass
class Trade:
    """Completed trade record."""
    direction: int
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    hold_time_ms: int
    strategy: str
    timestamp: int


class HFTMegaSystem:
    """
    High-Frequency Trading System for 300K-1M trades.

    Designed to:
    - Process 100+ ticks/second
    - Execute trades in milliseconds
    - Compound exponentially
    """

    def __init__(self, starting_capital: float = 10.0, strategies: list = None):
        self.capital = starting_capital
        self.starting_capital = starting_capital
        self.active_strategies = strategies or list(STRATEGIES.keys())

        # Position management
        self.position: Optional[Position] = None
        self.max_concurrent_positions = 1  # Keep simple for now

        # Trade tracking
        self.trades: deque = deque(maxlen=1000000)  # Store up to 1M trades
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

        # Per-strategy stats
        self.strategy_stats = {s: {"trades": 0, "wins": 0, "pnl": 0.0} for s in STRATEGIES}

        # Market data (fed from data feed)
        self.prices = deque(maxlen=10000)
        self.volumes = deque(maxlen=10000)
        self.timestamps = deque(maxlen=10000)
        self.spreads = deque(maxlen=1000)

        # VWAP calculation
        self.vwap = 0.0
        self.vwap_prices = deque(maxlen=1000)
        self.vwap_volumes = deque(maxlen=1000)

        # Arbitrage tracking
        self.exchange_prices = {}

        # Stats
        self.tick_count = 0
        self.signal_count = 0
        self.start_time = 0

    def update(self, price: float, volume: float = 0, bid: float = 0, ask: float = 0,
               spread: float = 0, exchange_prices: dict = None):
        """Update market data from feed."""
        now_ms = int(time.time() * 1000)

        self.prices.append(price)
        self.volumes.append(volume if volume > 0 else VOLUME_PER_SECOND / 100)
        self.timestamps.append(now_ms)

        if spread > 0:
            self.spreads.append(spread)

        if exchange_prices:
            self.exchange_prices = exchange_prices

        # Update VWAP
        self.vwap_prices.append(price)
        self.vwap_volumes.append(volume if volume > 0 else 1)
        total_vol = sum(self.vwap_volumes)
        if total_vol > 0:
            self.vwap = sum(p * v for p, v in zip(self.vwap_prices, self.vwap_volumes)) / total_vol

        self.tick_count += 1

        # Check exit first
        if self.position:
            self._check_exit(price, now_ms)

        # Check entry
        if not self.position:
            self._check_entry(price, now_ms, bid, ask)

    def _check_entry(self, price: float, now_ms: int, bid: float = 0, ask: float = 0):
        """Check all strategies for entry signals."""
        if len(self.prices) < 10:
            return

        for strategy_name in self.active_strategies:
            cfg = STRATEGIES[strategy_name]

            if len(self.prices) < cfg.get("min_ticks", 10):
                continue

            signal, size = self._get_signal(strategy_name, price, bid, ask)

            if signal != 0 and size > 0:
                self.signal_count += 1
                self._enter_position(signal, price, now_ms, size, strategy_name)
                break  # One position at a time

    def _get_signal(self, strategy: str, price: float, bid: float = 0, ask: float = 0) -> tuple:
        """Get signal for specific strategy."""
        cfg = STRATEGIES[strategy]

        if strategy == "ULTRA_SCALP":
            return self._ultra_scalp_signal(cfg, price)

        elif strategy == "MOMENTUM_BURST":
            return self._momentum_burst_signal(cfg, price)

        elif strategy == "SPREAD_CAPTURE":
            return self._spread_capture_signal(cfg, price, bid, ask)

        elif strategy == "VWAP_REVERSION":
            return self._vwap_reversion_signal(cfg, price)

        elif strategy == "CROSS_EXCHANGE_ARB":
            return self._arbitrage_signal(cfg, price)

        return 0, 0

    def _ultra_scalp_signal(self, cfg: dict, price: float) -> tuple:
        """Ultra-fast scalping on micro-moves."""
        prices = list(self.prices)[-5:]
        if len(prices) < 5:
            return 0, 0

        move = (prices[-1] - prices[0]) / prices[0]
        threshold = cfg["signal_threshold"]

        if move > threshold:
            return 1, cfg["kelly_frac"]  # Long
        elif move < -threshold:
            return -1, cfg["kelly_frac"]  # Short

        return 0, 0

    def _momentum_burst_signal(self, cfg: dict, price: float) -> tuple:
        """Trade on volume spikes."""
        if len(self.volumes) < 20:
            return 0, 0

        recent_vol = np.mean(list(self.volumes)[-5:])
        avg_vol = np.mean(list(self.volumes)[-50:])

        if recent_vol > avg_vol * cfg["volume_spike_mult"]:
            # Volume spike - trade in direction of move
            prices = list(self.prices)[-10:]
            move = (prices[-1] - prices[0]) / prices[0]

            if move > 0:
                return 1, cfg["kelly_frac"]
            elif move < 0:
                return -1, cfg["kelly_frac"]

        return 0, 0

    def _spread_capture_signal(self, cfg: dict, price: float, bid: float, ask: float) -> tuple:
        """Trade when spread is wide."""
        if bid <= 0 or ask <= 0:
            return 0, 0

        spread_pct = (ask - bid) / price

        if spread_pct > cfg["min_spread_pct"]:
            # Wide spread - look at momentum
            if len(self.prices) >= 3:
                prices = list(self.prices)[-3:]
                if prices[-1] > prices[0]:
                    return 1, cfg["kelly_frac"]  # Rising, go long
                elif prices[-1] < prices[0]:
                    return -1, cfg["kelly_frac"]  # Falling, go short

        return 0, 0

    def _vwap_reversion_signal(self, cfg: dict, price: float) -> tuple:
        """Mean reversion to VWAP."""
        if self.vwap <= 0:
            return 0, 0

        deviation = (price - self.vwap) / self.vwap

        if deviation > cfg["vwap_deviation"]:
            return -1, cfg["kelly_frac"]  # Price above VWAP, short
        elif deviation < -cfg["vwap_deviation"]:
            return 1, cfg["kelly_frac"]  # Price below VWAP, long

        return 0, 0

    def _arbitrage_signal(self, cfg: dict, price: float) -> tuple:
        """Cross-exchange arbitrage."""
        if len(self.exchange_prices) < cfg.get("min_exchanges", 2):
            return 0, 0

        # Get recent prices from each exchange
        now = int(time.time() * 1000)
        recent_prices = []
        for exchange, data in self.exchange_prices.items():
            if now - data.get("time", 0) < 5000:  # Last 5 seconds
                if "price" in data:
                    recent_prices.append(data["price"])
                elif "mid" in data:
                    recent_prices.append(data["mid"])

        if len(recent_prices) < 2:
            return 0, 0

        max_price = max(recent_prices)
        min_price = min(recent_prices)
        arb_pct = (max_price - min_price) / min_price

        if arb_pct > cfg["min_arb_pct"]:
            # Arbitrage opportunity - buy at min, sell at max
            # Simplified: just trade in direction of overall move
            if price < np.mean(recent_prices):
                return 1, cfg["kelly_frac"]  # Price below average, long
            else:
                return -1, cfg["kelly_frac"]  # Price above average, short

        return 0, 0

    def _enter_position(self, direction: int, price: float, now_ms: int, size: float, strategy: str):
        """Enter a new position."""
        value = self.capital * min(size, 1.0)

        self.position = Position(
            direction=direction,
            entry_price=price,
            entry_time_ms=now_ms,
            size=value,
            strategy=strategy
        )

    def _check_exit(self, price: float, now_ms: int):
        """Check exit conditions for current position."""
        if not self.position:
            return

        cfg = STRATEGIES[self.position.strategy]

        pnl_pct = (price - self.position.entry_price) / self.position.entry_price * self.position.direction
        hold_time_ms = now_ms - self.position.entry_time_ms

        should_exit = False
        exit_reason = ""

        if pnl_pct >= cfg["profit_target"]:
            should_exit = True
            exit_reason = "TP"
        elif pnl_pct <= -cfg["stop_loss"]:
            should_exit = True
            exit_reason = "SL"
        elif hold_time_ms >= cfg["max_hold_ms"]:
            should_exit = True
            exit_reason = "TIME"

        if should_exit:
            self._exit_position(price, now_ms, pnl_pct, exit_reason)

    def _exit_position(self, price: float, now_ms: int, pnl_pct: float, reason: str):
        """Exit current position."""
        if not self.position:
            return

        pnl = self.position.size * pnl_pct
        self.capital += pnl
        self.total_pnl += pnl

        if pnl > 0:
            self.wins += 1
            self.strategy_stats[self.position.strategy]["wins"] += 1
        else:
            self.losses += 1

        self.strategy_stats[self.position.strategy]["trades"] += 1
        self.strategy_stats[self.position.strategy]["pnl"] += pnl

        self.trade_count += 1

        # Record trade
        trade = Trade(
            direction=self.position.direction,
            entry_price=self.position.entry_price,
            exit_price=price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_time_ms=now_ms - self.position.entry_time_ms,
            strategy=self.position.strategy,
            timestamp=now_ms
        )
        self.trades.append(trade)

        self.position = None

    def get_stats(self) -> dict:
        """Get current stats."""
        total = self.wins + self.losses
        win_rate = self.wins / total * 100 if total > 0 else 0
        ret = (self.capital - self.starting_capital) / self.starting_capital * 100

        return {
            "capital": self.capital,
            "return_pct": ret,
            "trades": self.trade_count,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "ticks": self.tick_count,
            "signals": self.signal_count,
            "strategy_stats": self.strategy_stats
        }


async def run_hft_mega(duration: int = 300, api_key: str = None):
    """Run the HFT Mega System."""

    print("=" * 70)
    print("HFT MEGA SYSTEM - 300K to 1M TRADES")
    print("=" * 70)
    print(f"BTC 24h Volume: ${BTC_24H_VOLUME:,.0f}")
    print(f"Per Second: ${VOLUME_PER_SECOND:,.0f}")
    print(f"Starting Capital: $10.00")
    print(f"Target: Maximum trade frequency")
    print("=" * 70)

    # Initialize data feed
    print("\nInitializing data feed...")
    try:
        feed = await get_best_feed(api_key)
    except:
        from ultra_fast_feed import UltraFastFeed
        feed = UltraFastFeed()
        await feed.connect()

    # Initialize HFT system
    system = HFTMegaSystem(starting_capital=10.0)
    system.start_time = time.time()

    print("\nRunning HFT system...")
    print("-" * 70)

    start = time.time()
    last_report = start
    report_interval = 5  # Report every 5 seconds

    try:
        while time.time() - start < duration:
            await asyncio.sleep(0.001)  # 1ms loop - ultra fast

            stats = feed.get_stats()
            price = stats.get("price", 0)

            if price > 0:
                system.update(
                    price=price,
                    volume=0,  # Will be calculated
                    bid=stats.get("bid", 0),
                    ask=stats.get("ask", 0),
                    spread=stats.get("spread", 0),
                    exchange_prices=feed.exchange_prices if hasattr(feed, "exchange_prices") else {}
                )

            # Progress report
            if time.time() - last_report >= report_interval:
                elapsed = time.time() - start
                sys_stats = system.get_stats()

                trades_per_sec = sys_stats["trades"] / elapsed if elapsed > 0 else 0
                projected_trades = trades_per_sec * duration

                print(f"[{elapsed:>5.0f}s] "
                      f"Cap: ${sys_stats['capital']:.4f} | "
                      f"Ret: {sys_stats['return_pct']:+.2f}% | "
                      f"Trades: {sys_stats['trades']:,} | "
                      f"W/L: {sys_stats['wins']}/{sys_stats['losses']} | "
                      f"Rate: {trades_per_sec:.1f}/sec | "
                      f"Projected: {projected_trades:,.0f}")

                last_report = time.time()

    except KeyboardInterrupt:
        print("\nInterrupted!")

    finally:
        await feed.close()

    # Final results
    final_stats = system.get_stats()
    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Duration: {elapsed:.0f} seconds")
    print(f"Capital: $10.00 -> ${final_stats['capital']:.4f} ({final_stats['return_pct']:+.2f}%)")
    print(f"Total Trades: {final_stats['trades']:,}")
    print(f"Win Rate: {final_stats['win_rate']:.1f}%")
    print(f"Total PnL: ${final_stats['total_pnl']:+.4f}")
    print(f"Trades/Second: {final_stats['trades']/elapsed:.2f}")
    print(f"Signals Generated: {final_stats['signals']:,}")
    print(f"Ticks Processed: {final_stats['ticks']:,}")

    print("\nPer-Strategy Performance:")
    print("-" * 50)
    for strategy, stats in final_stats["strategy_stats"].items():
        if stats["trades"] > 0:
            wr = stats["wins"] / stats["trades"] * 100
            print(f"  {strategy}: {stats['trades']} trades | "
                  f"WR: {wr:.0f}% | PnL: ${stats['pnl']:+.4f}")

    print("=" * 70)

    return final_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HFT Mega System")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument("--api-key", type=str, help="CoinAPI key (optional)")
    args = parser.parse_args()

    asyncio.run(run_hft_mega(duration=args.duration, api_key=args.api_key))
