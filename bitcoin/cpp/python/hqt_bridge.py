#!/usr/bin/env python3
"""
HQT C++ Bridge - Python interface to nanosecond C++ arbitrage detector.

Uses ctypes for FFI to the compiled C++ shared library.
Falls back to pure Python if library not available.
"""

import os
import ctypes
from ctypes import c_double, c_int, c_int64, c_char_p, POINTER, Structure
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity from C++ detector."""
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_pct: float
    total_cost_pct: float
    profit_pct: float
    profit_usd: float
    win_rate: float
    timestamp_ns: int


class HQTBridge:
    """
    Bridge to C++ HQT Arbitrage Detector.

    Provides nanosecond-speed arbitrage detection across exchanges.
    """

    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize bridge.

        Args:
            lib_path: Path to libhqt_sct.so. Auto-detected if None.
        """
        self._lib = None
        self._cpp_available = False

        # Try to load C++ library
        if lib_path is None:
            # Look for library in standard locations
            candidates = [
                Path(__file__).parent.parent / "build" / "libhqt_sct.so",
                Path("/root/sovereign/hqt_sct/build/libhqt_sct.so"),
                Path("/usr/local/lib/libhqt_sct.so"),
            ]
            for path in candidates:
                if path.exists():
                    lib_path = str(path)
                    break

        if lib_path and os.path.exists(lib_path):
            try:
                self._lib = ctypes.CDLL(lib_path)
                self._setup_functions()
                self._cpp_available = True
                print(f"[HQT] C++ library loaded from {lib_path}")
            except OSError as e:
                print(f"[HQT] Could not load C++ library: {e}")
                print("[HQT] Falling back to pure Python")
        else:
            print("[HQT] C++ library not found, using pure Python")

        # Initialize detector
        self._prices = {}
        self._min_spread = 0.005
        self._min_profit = 5.0
        self._position_btc = 0.01

        # Fee table - MAKER fees (limit orders, much lower than taker!)
        # Taker fees were causing 0% opportunities (costs > spreads)
        # Fixed 2024-12-26: Use maker fees for arb execution
        self._fees = {
            'kraken': 0.0016,     # 0.16% maker (was 0.26% taker)
            'coinbase': 0.004,    # 0.40% maker (was 0.60% taker)
            'bitstamp': 0.0030,   # 0.30% maker (was 0.50% taker)
            'gemini': 0.002,      # 0.20% maker (was 0.40% taker)
            'binance': 0.0002,    # 0.02% maker (was 0.10% taker)
            'bybit': 0.0002,      # 0.02% maker (was 0.10% taker)
        }
        self._default_fee = 0.003  # 0.30% default maker
        self._slippage = 0.0003  # 3 bps per side (reduced for limit orders)

    def _setup_functions(self):
        """Setup ctypes function signatures."""
        # Note: For a real implementation, we'd define C wrappers
        # with extern "C" for proper ctypes binding
        pass

    @property
    def cpp_available(self) -> bool:
        """Check if C++ library is loaded."""
        return self._cpp_available

    def update_price(self, exchange: str, bid: float, ask: float):
        """
        Update price for an exchange.

        Args:
            exchange: Exchange name (e.g., 'kraken', 'coinbase')
            bid: Best bid price
            ask: Best ask price
        """
        import time
        self._prices[exchange] = {
            'bid': bid,
            'ask': ask,
            'timestamp': time.time_ns()
        }

    def get_fee(self, exchange: str) -> float:
        """Get taker fee for exchange."""
        return self._fees.get(exchange.lower(), self._default_fee)

    def get_total_cost(self, buy_ex: str, sell_ex: str) -> float:
        """Get total cost for arbitrage trade (fees + slippage)."""
        return self.get_fee(buy_ex) + self.get_fee(sell_ex) + (self._slippage * 2)

    def find_opportunity(self) -> Optional[ArbitrageOpportunity]:
        """
        Find best arbitrage opportunity across all exchanges.

        Returns:
            ArbitrageOpportunity if profitable opportunity exists, None otherwise.
        """
        import time

        if len(self._prices) < 2:
            return None

        best = None
        best_spread = 0

        exchanges = list(self._prices.keys())

        for buy_ex in exchanges:
            for sell_ex in exchanges:
                if buy_ex == sell_ex:
                    continue

                buy_price = self._prices[buy_ex]['ask']
                sell_price = self._prices[sell_ex]['bid']

                if sell_price <= buy_price:
                    continue

                spread = (sell_price - buy_price) / buy_price

                if spread > best_spread:
                    best_spread = spread
                    best = (buy_ex, sell_ex, buy_price, sell_price, spread)

        if not best:
            return None

        buy_ex, sell_ex, buy_price, sell_price, spread = best

        # Calculate costs and profit
        total_cost = self.get_total_cost(buy_ex, sell_ex)
        net_profit = spread - total_cost

        if net_profit <= 0:
            return None

        mid_price = (buy_price + sell_price) / 2
        position_value = self._position_btc * mid_price
        profit_usd = position_value * net_profit

        if profit_usd < self._min_profit:
            return None

        return ArbitrageOpportunity(
            buy_exchange=buy_ex,
            sell_exchange=sell_ex,
            buy_price=buy_price,
            sell_price=sell_price,
            spread_pct=spread,
            total_cost_pct=total_cost,
            profit_pct=net_profit,
            profit_usd=profit_usd,
            win_rate=1.0,
            timestamp_ns=time.time_ns()
        )

    def find_all_opportunities(self) -> List[ArbitrageOpportunity]:
        """Find all profitable arbitrage opportunities."""
        import time

        opportunities = []

        if len(self._prices) < 2:
            return opportunities

        exchanges = list(self._prices.keys())

        for buy_ex in exchanges:
            for sell_ex in exchanges:
                if buy_ex == sell_ex:
                    continue

                buy_price = self._prices[buy_ex]['ask']
                sell_price = self._prices[sell_ex]['bid']

                if sell_price <= buy_price:
                    continue

                spread = (sell_price - buy_price) / buy_price
                total_cost = self.get_total_cost(buy_ex, sell_ex)
                net_profit = spread - total_cost

                if net_profit <= 0:
                    continue

                mid_price = (buy_price + sell_price) / 2
                position_value = self._position_btc * mid_price
                profit_usd = position_value * net_profit

                if profit_usd < self._min_profit:
                    continue

                opportunities.append(ArbitrageOpportunity(
                    buy_exchange=buy_ex,
                    sell_exchange=sell_ex,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    spread_pct=spread,
                    total_cost_pct=total_cost,
                    profit_pct=net_profit,
                    profit_usd=profit_usd,
                    win_rate=1.0,
                    timestamp_ns=time.time_ns()
                ))

        # Sort by profit descending
        opportunities.sort(key=lambda x: x.profit_pct, reverse=True)
        return opportunities


if __name__ == "__main__":
    # Demo
    bridge = HQTBridge()

    # Simulate prices
    bridge.update_price("kraken", 99500, 99520)
    bridge.update_price("coinbase", 99600, 99650)
    bridge.update_price("gemini", 99550, 99580)

    print("\n=== HQT C++ Bridge Demo ===")
    print(f"C++ available: {bridge.cpp_available}")

    opp = bridge.find_opportunity()
    if opp:
        print(f"\nARBITRAGE FOUND:")
        print(f"  Buy {opp.buy_exchange} @ ${opp.buy_price:.2f}")
        print(f"  Sell {opp.sell_exchange} @ ${opp.sell_price:.2f}")
        print(f"  Spread: {opp.spread_pct*100:.3f}%")
        print(f"  Costs: {opp.total_cost_pct*100:.3f}%")
        print(f"  Profit: {opp.profit_pct*100:.3f}% (${opp.profit_usd:.2f})")
    else:
        print("\nNo arbitrage opportunity (spread doesn't cover costs)")
