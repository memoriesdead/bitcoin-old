#!/usr/bin/env python3
"""
Spread Calculator - Cross-Exchange Price Analysis

Calculates bid-ask spreads across exchanges to find arbitrage.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import time


@dataclass
class ExchangePrice:
    """Price data from a single exchange."""
    exchange: str
    bid: float              # Best bid (we sell at this)
    ask: float              # Best ask (we buy at this)
    timestamp: float        # Unix timestamp
    volume_24h: Optional[float] = None

    @property
    def mid(self) -> float:
        """Mid-market price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread in absolute terms."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as percentage of mid."""
        return self.spread / self.mid if self.mid > 0 else 0

    @property
    def age_ms(self) -> float:
        """Age of price in milliseconds."""
        return (time.time() - self.timestamp) * 1000


@dataclass
class SpreadOpportunity:
    """Cross-exchange spread opportunity."""
    buy_exchange: str       # Exchange to buy on (lower ask)
    sell_exchange: str      # Exchange to sell on (higher bid)
    buy_price: float        # Ask price on buy exchange
    sell_price: float       # Bid price on sell exchange
    spread: float           # Absolute spread (sell - buy)
    spread_pct: float       # Percentage spread
    timestamp: float


class SpreadCalculator:
    """
    Calculate spreads across multiple exchanges.

    Finds cross-exchange arbitrage opportunities by comparing
    bid prices on one exchange to ask prices on another.
    """

    def __init__(self, stale_threshold_ms: float = 1000):
        """
        Initialize spread calculator.

        Args:
            stale_threshold_ms: Max age for valid prices
        """
        self.stale_threshold_ms = stale_threshold_ms
        self.prices: Dict[str, ExchangePrice] = {}

    def update_price(self, price: ExchangePrice):
        """Update price for an exchange."""
        self.prices[price.exchange.lower()] = price

    def update_prices(self, prices: List[ExchangePrice]):
        """Update multiple prices."""
        for p in prices:
            self.update_price(p)

    def get_valid_prices(self) -> Dict[str, ExchangePrice]:
        """Get prices that aren't stale."""
        valid = {}
        for ex, price in self.prices.items():
            if price.age_ms <= self.stale_threshold_ms:
                valid[ex] = price
        return valid

    def find_best_spread(self) -> Optional[SpreadOpportunity]:
        """
        Find best cross-exchange spread.

        Returns the opportunity with highest spread, or None if no
        positive spread exists.
        """
        valid = self.get_valid_prices()

        if len(valid) < 2:
            return None

        best = None

        for buy_ex, buy_price in valid.items():
            for sell_ex, sell_price in valid.items():
                if buy_ex == sell_ex:
                    continue

                # Buy at ask, sell at bid
                buy = buy_price.ask
                sell = sell_price.bid

                if sell <= buy:
                    continue  # No positive spread

                spread = sell - buy
                spread_pct = spread / buy

                if best is None or spread_pct > best.spread_pct:
                    best = SpreadOpportunity(
                        buy_exchange=buy_ex,
                        sell_exchange=sell_ex,
                        buy_price=buy,
                        sell_price=sell,
                        spread=spread,
                        spread_pct=spread_pct,
                        timestamp=time.time()
                    )

        return best

    def find_all_spreads(self) -> List[SpreadOpportunity]:
        """
        Find all positive cross-exchange spreads.

        Returns list sorted by spread_pct descending.
        """
        valid = self.get_valid_prices()
        opportunities = []

        for buy_ex, buy_price in valid.items():
            for sell_ex, sell_price in valid.items():
                if buy_ex == sell_ex:
                    continue

                buy = buy_price.ask
                sell = sell_price.bid

                if sell > buy:
                    spread = sell - buy
                    spread_pct = spread / buy

                    opportunities.append(SpreadOpportunity(
                        buy_exchange=buy_ex,
                        sell_exchange=sell_ex,
                        buy_price=buy,
                        sell_price=sell,
                        spread=spread,
                        spread_pct=spread_pct,
                        timestamp=time.time()
                    ))

        return sorted(opportunities, key=lambda x: -x.spread_pct)

    def print_matrix(self):
        """Print spread matrix across exchanges."""
        valid = self.get_valid_prices()
        exchanges = sorted(valid.keys())

        if not exchanges:
            print("No valid prices")
            return

        # Header
        print(f"\n{'SELL→':>12}", end='')
        for ex in exchanges:
            print(f"{ex:>12}", end='')
        print()

        print("BUY↓")

        # Matrix
        for buy_ex in exchanges:
            print(f"{buy_ex:>12}", end='')
            for sell_ex in exchanges:
                if buy_ex == sell_ex:
                    print(f"{'---':>12}", end='')
                else:
                    buy = valid[buy_ex].ask
                    sell = valid[sell_ex].bid
                    spread_pct = (sell - buy) / buy * 100

                    if spread_pct > 0:
                        print(f"{spread_pct:>11.3f}%", end='')
                    else:
                        print(f"{spread_pct:>11.3f}%", end='')
            print()

        print()
