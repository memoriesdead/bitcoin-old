#!/usr/bin/env python3
"""
ORDER BOOK AGGREGATOR WITH TRUE OFI CALCULATION
================================================
Real-time order book management with academic-grade OFI computation.

Academic Basis:
- Cont, Kukanov & Stoikov (2014) - "The Price Impact of Order Book Events"
  Journal of Financial Econometrics, 12(1), 47-88
  R² = 65-70% for price prediction using OFI

KEY INSIGHT:
  The OLD approach: Derive OFI from price changes (CIRCULAR - NO EDGE!)
  The NEW approach: Calculate OFI from REAL order book changes (TRUE EDGE!)

OFI Formula:
  OFI_t = ΔBid_qty × I[bid_price_up] - ΔAsk_qty × I[ask_price_down]

Where:
  - ΔBid_qty = change in bid quantity at best bid
  - ΔAsk_qty = change in ask quantity at best ask
  - I[condition] = indicator function (1 if true, 0 if false)

Performance Target:
  - <10μs OFI calculation latency
  - Real-time updates from 4+ exchanges
  - True order flow signal (not derived from price)
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from enum import Enum

# Import exchange feed types
try:
    from .exchange_feed import Exchange, OrderBookUpdate, OrderBookLevel
except ImportError:
    from exchange_feed import Exchange, OrderBookUpdate, OrderBookLevel


@dataclass
class OFISignal:
    """Order Flow Imbalance signal - THE REAL EDGE (R²=70%)."""
    timestamp: float
    ofi: float                    # Raw OFI value
    ofi_normalized: float         # Normalized to [-1, 1]
    ofi_momentum: float           # Rate of change of OFI
    signal_strength: float        # 0-1 confidence
    direction: int                # -1 (sell), 0 (neutral), +1 (buy)
    exchange: Optional[Exchange] = None


@dataclass
class KyleLambda:
    """Kyle's Lambda - Price impact coefficient (Econometrica 1985)."""
    timestamp: float
    lambda_value: float           # Price sensitivity to order flow
    r_squared: float              # Explanatory power
    sample_size: int


@dataclass
class VPINValue:
    """Volume-Synchronized PIN (Review of Financial Studies 2012)."""
    timestamp: float
    vpin: float                   # 0-1 toxicity measure
    bucket_imbalance: float       # Current bucket imbalance
    is_toxic: bool                # True if VPIN > threshold


class OrderBook:
    """
    Real-time order book with TRUE OFI calculation.

    NOT derived from price changes (that's circular!).
    Calculated from actual order book bid/ask quantity changes.
    """

    def __init__(
        self,
        exchange: Exchange,
        symbol: str,
        depth: int = 25,
        ofi_lookback: int = 100,
        vpin_bucket_size: float = 1.0,  # BTC per bucket
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.depth = depth

        # Order book state
        self.bids: Dict[float, float] = {}  # price -> quantity
        self.asks: Dict[float, float] = {}  # price -> quantity
        self.best_bid: float = 0.0
        self.best_ask: float = float('inf')
        self.mid_price: float = 0.0
        self.spread: float = 0.0
        self.last_update: float = 0.0

        # Previous state for OFI calculation
        self._prev_best_bid: float = 0.0
        self._prev_best_ask: float = float('inf')
        self._prev_bid_qty: float = 0.0
        self._prev_ask_qty: float = 0.0

        # OFI history
        self._ofi_history: Deque[float] = deque(maxlen=ofi_lookback)
        self._ofi_timestamps: Deque[float] = deque(maxlen=ofi_lookback)
        self._price_history: Deque[float] = deque(maxlen=ofi_lookback)

        # VPIN calculation
        self._vpin_bucket_size = vpin_bucket_size
        self._vpin_buckets: Deque[Tuple[float, float]] = deque(maxlen=50)  # (buy_vol, sell_vol)
        self._current_bucket_buy: float = 0.0
        self._current_bucket_sell: float = 0.0
        self._current_bucket_volume: float = 0.0

        # Kyle Lambda calculation
        self._kyle_ofi: Deque[float] = deque(maxlen=100)
        self._kyle_price_changes: Deque[float] = deque(maxlen=100)

        # Statistics
        self._update_count: int = 0
        self._ofi_sum: float = 0.0

    def update(self, book_update: OrderBookUpdate) -> Optional[OFISignal]:
        """
        Update order book and calculate TRUE OFI.

        Returns OFI signal if significant order flow detected.
        """
        now = book_update.timestamp
        self._update_count += 1

        # Update bid side
        new_bids = {}
        for level in book_update.bids[:self.depth]:
            new_bids[level.price] = level.quantity

        # Update ask side
        new_asks = {}
        for level in book_update.asks[:self.depth]:
            new_asks[level.price] = level.quantity

        # Calculate best bid/ask
        new_best_bid = max(new_bids.keys()) if new_bids else 0.0
        new_best_ask = min(new_asks.keys()) if new_asks else float('inf')

        # Get quantities at best prices
        new_bid_qty = new_bids.get(new_best_bid, 0.0)
        new_ask_qty = new_asks.get(new_best_ask, 0.0)

        # =====================================================
        # CONT-STOIKOV OFI CALCULATION (Journal of Financial Econometrics 2014)
        # =====================================================
        # OFI = ΔBid_qty × I[bid_up_or_same] - ΔAsk_qty × I[ask_down_or_same]
        #
        # This captures:
        # - Bid increases at same/higher price = BUY pressure
        # - Ask increases at same/lower price = SELL pressure
        # =====================================================

        ofi = 0.0

        if self._prev_best_bid > 0:  # Skip first update
            # Bid side contribution
            if new_best_bid >= self._prev_best_bid:
                # Price up or same: bid queue addition is bullish
                delta_bid = new_bid_qty - self._prev_bid_qty
                ofi += delta_bid
            else:
                # Price down: bid cancelled, reset
                ofi -= self._prev_bid_qty

            # Ask side contribution
            if new_best_ask <= self._prev_best_ask:
                # Price down or same: ask queue addition is bearish
                delta_ask = new_ask_qty - self._prev_ask_qty
                ofi -= delta_ask
            else:
                # Price up: ask cancelled, reset
                ofi += self._prev_ask_qty

        # Store for next update
        self._prev_best_bid = new_best_bid
        self._prev_best_ask = new_best_ask
        self._prev_bid_qty = new_bid_qty
        self._prev_ask_qty = new_ask_qty

        # Update order book state
        self.bids = new_bids
        self.asks = new_asks
        self.best_bid = new_best_bid
        self.best_ask = new_best_ask
        self.mid_price = (new_best_bid + new_best_ask) / 2 if new_best_bid and new_best_ask < float('inf') else 0.0
        self.spread = new_best_ask - new_best_bid if new_best_ask < float('inf') else 0.0
        self.last_update = now

        # Store OFI history
        self._ofi_history.append(ofi)
        self._ofi_timestamps.append(now)
        self._price_history.append(self.mid_price)
        self._ofi_sum += ofi

        # Update Kyle Lambda data
        if len(self._price_history) >= 2:
            price_change = self._price_history[-1] - self._price_history[-2]
            self._kyle_ofi.append(ofi)
            self._kyle_price_changes.append(price_change)

        # Generate signal
        return self._generate_ofi_signal(ofi, now)

    def _generate_ofi_signal(self, current_ofi: float, timestamp: float) -> OFISignal:
        """Generate OFI trading signal."""
        # Calculate OFI statistics
        if len(self._ofi_history) < 5:
            return OFISignal(
                timestamp=timestamp,
                ofi=current_ofi,
                ofi_normalized=0.0,
                ofi_momentum=0.0,
                signal_strength=0.0,
                direction=0,
                exchange=self.exchange,
            )

        ofi_array = np.array(self._ofi_history)
        ofi_mean = np.mean(ofi_array)
        ofi_std = np.std(ofi_array) + 1e-10  # Avoid division by zero

        # Normalized OFI (z-score)
        ofi_normalized = (current_ofi - ofi_mean) / ofi_std
        ofi_normalized = max(-3.0, min(3.0, ofi_normalized)) / 3.0  # Scale to [-1, 1]

        # OFI momentum (rate of change over last 10 updates)
        if len(self._ofi_history) >= 10:
            recent_ofi = list(self._ofi_history)[-10:]
            ofi_momentum = np.mean(recent_ofi[-5:]) - np.mean(recent_ofi[:5])
        else:
            ofi_momentum = 0.0

        # Signal strength (based on OFI magnitude and consistency)
        ofi_magnitude = abs(ofi_normalized)
        ofi_consistency = 1.0 if np.sign(current_ofi) == np.sign(ofi_mean) else 0.5
        signal_strength = min(1.0, ofi_magnitude * ofi_consistency)

        # Direction
        if ofi_normalized > 0.3:
            direction = 1  # BUY
        elif ofi_normalized < -0.3:
            direction = -1  # SELL
        else:
            direction = 0  # NEUTRAL

        return OFISignal(
            timestamp=timestamp,
            ofi=current_ofi,
            ofi_normalized=ofi_normalized,
            ofi_momentum=ofi_momentum,
            signal_strength=signal_strength,
            direction=direction,
            exchange=self.exchange,
        )

    def calculate_kyle_lambda(self) -> Optional[KyleLambda]:
        """
        Calculate Kyle's Lambda - price impact coefficient.

        Kyle (1985) - Econometrica, 10,000+ citations
        λ = Cov(ΔP, OFI) / Var(OFI)

        High λ = informed trading present
        Trade WITH high-λ flow direction
        """
        if len(self._kyle_ofi) < 20:
            return None

        ofi = np.array(self._kyle_ofi)
        price_changes = np.array(self._kyle_price_changes)

        # Calculate lambda
        cov = np.cov(price_changes, ofi)[0, 1]
        var_ofi = np.var(ofi) + 1e-10

        lambda_value = cov / var_ofi

        # Calculate R² (explanatory power)
        if np.var(price_changes) > 0:
            correlation = np.corrcoef(price_changes, ofi)[0, 1]
            r_squared = correlation ** 2
        else:
            r_squared = 0.0

        return KyleLambda(
            timestamp=time.time(),
            lambda_value=lambda_value,
            r_squared=r_squared,
            sample_size=len(ofi),
        )

    def update_vpin(self, trade_price: float, trade_volume: float, is_buy: bool):
        """
        Update VPIN (Volume-Synchronized PIN).

        Easley, Lopez de Prado & O'Hara (2012) - Review of Financial Studies
        Predicted Flash Crash 2 hours before it happened.
        """
        if is_buy:
            self._current_bucket_buy += trade_volume
        else:
            self._current_bucket_sell += trade_volume

        self._current_bucket_volume += trade_volume

        # Check if bucket is full
        if self._current_bucket_volume >= self._vpin_bucket_size:
            self._vpin_buckets.append((
                self._current_bucket_buy,
                self._current_bucket_sell
            ))
            self._current_bucket_buy = 0.0
            self._current_bucket_sell = 0.0
            self._current_bucket_volume = 0.0

    def calculate_vpin(self) -> Optional[VPINValue]:
        """
        Calculate current VPIN value.

        VPIN = Σ|V_buy - V_sell| / (n × V_bucket)

        High VPIN (>0.7) = toxic flow = volatility coming
        """
        if len(self._vpin_buckets) < 10:
            return None

        total_imbalance = 0.0
        total_volume = 0.0

        for buy_vol, sell_vol in self._vpin_buckets:
            total_imbalance += abs(buy_vol - sell_vol)
            total_volume += buy_vol + sell_vol

        if total_volume == 0:
            return None

        vpin = total_imbalance / total_volume

        # Current bucket imbalance
        if self._current_bucket_volume > 0:
            bucket_imbalance = (self._current_bucket_buy - self._current_bucket_sell) / self._current_bucket_volume
        else:
            bucket_imbalance = 0.0

        return VPINValue(
            timestamp=time.time(),
            vpin=vpin,
            bucket_imbalance=bucket_imbalance,
            is_toxic=vpin > 0.7,
        )

    def get_depth_imbalance(self, levels: int = 5) -> float:
        """
        Calculate order book depth imbalance.

        Imbalance = (Bid_depth - Ask_depth) / (Bid_depth + Ask_depth)

        Positive = more bids = bullish pressure
        Negative = more asks = bearish pressure
        """
        bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_prices = sorted(self.asks.keys())[:levels]

        bid_depth = sum(self.bids.get(p, 0) for p in bid_prices)
        ask_depth = sum(self.asks.get(p, 0) for p in ask_prices)

        total = bid_depth + ask_depth
        if total == 0:
            return 0.0

        return (bid_depth - ask_depth) / total

    def get_stats(self) -> dict:
        """Get order book statistics."""
        return {
            'exchange': self.exchange.value,
            'symbol': self.symbol,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'spread_bps': self.spread / self.mid_price * 10000 if self.mid_price else 0,
            'bid_levels': len(self.bids),
            'ask_levels': len(self.asks),
            'update_count': self._update_count,
            'ofi_sum': self._ofi_sum,
            'depth_imbalance': self.get_depth_imbalance(),
        }


class OrderBookAggregator:
    """
    Aggregates order books from multiple exchanges.

    Provides:
    - Unified OFI signal across all exchanges
    - Cross-exchange arbitrage detection
    - Best execution routing
    """

    def __init__(self, ofi_lookback: int = 100):
        self.order_books: Dict[Exchange, OrderBook] = {}
        self._ofi_lookback = ofi_lookback

        # Aggregated signals
        self._aggregate_ofi: Deque[float] = deque(maxlen=ofi_lookback)
        self._aggregate_timestamps: Deque[float] = deque(maxlen=ofi_lookback)

    def add_exchange(self, exchange: Exchange, symbol: str):
        """Add exchange order book."""
        self.order_books[exchange] = OrderBook(
            exchange=exchange,
            symbol=symbol,
            ofi_lookback=self._ofi_lookback,
        )

    def update(self, book_update: OrderBookUpdate) -> Optional[OFISignal]:
        """Update order book for exchange and return aggregated OFI signal."""
        exchange = book_update.exchange

        if exchange not in self.order_books:
            self.add_exchange(exchange, book_update.symbol)

        # Update individual order book
        signal = self.order_books[exchange].update(book_update)

        if signal:
            # Aggregate OFI across all exchanges
            aggregate_ofi = self._calculate_aggregate_ofi()
            self._aggregate_ofi.append(aggregate_ofi)
            self._aggregate_timestamps.append(signal.timestamp)

            # Return aggregated signal
            return self._generate_aggregate_signal(signal.timestamp)

        return None

    def _calculate_aggregate_ofi(self) -> float:
        """Calculate volume-weighted aggregate OFI."""
        total_ofi = 0.0
        total_weight = 0.0

        for exchange, book in self.order_books.items():
            if book._ofi_history:
                # Weight by update frequency (more updates = more liquid = more weight)
                weight = book._update_count
                total_ofi += book._ofi_history[-1] * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_ofi / total_weight

    def _generate_aggregate_signal(self, timestamp: float) -> OFISignal:
        """Generate aggregate OFI signal from all exchanges."""
        if len(self._aggregate_ofi) < 5:
            return OFISignal(
                timestamp=timestamp,
                ofi=0.0,
                ofi_normalized=0.0,
                ofi_momentum=0.0,
                signal_strength=0.0,
                direction=0,
            )

        ofi_array = np.array(self._aggregate_ofi)
        current_ofi = ofi_array[-1]
        ofi_mean = np.mean(ofi_array)
        ofi_std = np.std(ofi_array) + 1e-10

        # Normalized OFI
        ofi_normalized = (current_ofi - ofi_mean) / ofi_std
        ofi_normalized = max(-3.0, min(3.0, ofi_normalized)) / 3.0

        # Momentum
        if len(ofi_array) >= 10:
            ofi_momentum = np.mean(ofi_array[-5:]) - np.mean(ofi_array[-10:-5])
        else:
            ofi_momentum = 0.0

        # Signal strength (consensus across exchanges)
        directions = []
        for book in self.order_books.values():
            if book._ofi_history:
                directions.append(np.sign(book._ofi_history[-1]))

        if directions:
            consensus = abs(np.mean(directions))
        else:
            consensus = 0.0

        signal_strength = min(1.0, abs(ofi_normalized) * (0.5 + 0.5 * consensus))

        # Direction
        if ofi_normalized > 0.3:
            direction = 1
        elif ofi_normalized < -0.3:
            direction = -1
        else:
            direction = 0

        return OFISignal(
            timestamp=timestamp,
            ofi=current_ofi,
            ofi_normalized=ofi_normalized,
            ofi_momentum=ofi_momentum,
            signal_strength=signal_strength,
            direction=direction,
        )

    def get_best_bid_ask(self) -> Tuple[Tuple[float, Exchange], Tuple[float, Exchange]]:
        """Get best bid/ask across all exchanges."""
        best_bid = 0.0
        best_ask = float('inf')
        bid_exchange = None
        ask_exchange = None

        for exchange, book in self.order_books.items():
            if book.best_bid > best_bid:
                best_bid = book.best_bid
                bid_exchange = exchange
            if book.best_ask < best_ask:
                best_ask = book.best_ask
                ask_exchange = exchange

        return (best_bid, bid_exchange), (best_ask, ask_exchange)

    def get_arbitrage_opportunity(self) -> Optional[dict]:
        """Check for cross-exchange arbitrage."""
        (best_bid, bid_ex), (best_ask, ask_ex) = self.get_best_bid_ask()

        if bid_ex and ask_ex and bid_ex != ask_ex and best_bid > best_ask:
            spread = best_bid - best_ask
            spread_pct = spread / best_ask * 100

            return {
                'buy_exchange': ask_ex.value,
                'buy_price': best_ask,
                'sell_exchange': bid_ex.value,
                'sell_price': best_bid,
                'spread': spread,
                'spread_pct': spread_pct,
                'timestamp': time.time(),
            }

        return None

    def get_stats(self) -> dict:
        """Get aggregated statistics."""
        stats = {
            'exchanges': len(self.order_books),
            'total_updates': sum(b._update_count for b in self.order_books.values()),
            'aggregate_ofi_count': len(self._aggregate_ofi),
        }

        for exchange, book in self.order_books.items():
            stats[f'{exchange.value}_stats'] = book.get_stats()

        return stats


def test_order_book():
    """Test order book with synthetic data."""
    print("=" * 70)
    print("ORDER BOOK TEST - TRUE OFI CALCULATION")
    print("=" * 70)

    book = OrderBook(Exchange.BINANCE, "BTCUSDT")

    # Simulate order book updates
    base_price = 97000.0

    for i in range(100):
        # Simulate bid/ask levels
        bid_offset = np.random.normal(0, 10)
        ask_offset = np.random.normal(0, 10)

        bid_price = base_price - 50 + bid_offset
        ask_price = base_price + 50 + ask_offset

        # Create synthetic update
        update = OrderBookUpdate(
            exchange=Exchange.BINANCE,
            symbol="BTCUSDT",
            timestamp=time.time(),
            bids=[
                OrderBookLevel(bid_price, np.random.uniform(0.5, 2.0), time.time()),
                OrderBookLevel(bid_price - 10, np.random.uniform(0.5, 2.0), time.time()),
            ],
            asks=[
                OrderBookLevel(ask_price, np.random.uniform(0.5, 2.0), time.time()),
                OrderBookLevel(ask_price + 10, np.random.uniform(0.5, 2.0), time.time()),
            ],
        )

        signal = book.update(update)

        if i % 20 == 0:
            print(f"\nUpdate {i}:")
            print(f"  Mid Price: ${book.mid_price:,.2f}")
            print(f"  Spread: ${book.spread:.2f}")
            print(f"  OFI: {signal.ofi:.4f}")
            print(f"  OFI Normalized: {signal.ofi_normalized:.4f}")
            print(f"  Direction: {signal.direction}")
            print(f"  Strength: {signal.signal_strength:.2%}")

    # Calculate Kyle Lambda
    kyle = book.calculate_kyle_lambda()
    if kyle:
        print(f"\nKyle Lambda: {kyle.lambda_value:.6f}")
        print(f"R²: {kyle.r_squared:.4f}")

    print("\nFinal Stats:")
    for k, v in book.get_stats().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    test_order_book()
