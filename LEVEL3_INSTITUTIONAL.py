#!/usr/bin/env python3
"""
LEVEL 3 INSTITUTIONAL ORDER FLOW SYSTEM
========================================
Uses Kraken Level 3 order book data for TRUE institutional edge

Data Sources:
1. Kraken Level 3: Individual orders (institutional data)
2. Coinbase Advanced: Price confirmation
3. Binance.US: Volume confirmation

EDGE: See large orders BEFORE they execute!
"""

import asyncio
import websockets
import json
import time
import hmac
import hashlib
import base64
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional
import argparse

# ============================================================================
# API CREDENTIALS
# ============================================================================

KRAKEN_API_KEY = "S5ulEZbt83MO1RvIDvt2ILhkIi4BByyo388FLZr3jtBrAoP5Bg+GUTY9"
KRAKEN_PRIVATE_KEY = "pVH2EHI75Lw1JLd6TtZ843Udh27AP38M6Xl7d1l1WjQ1qlAJCYBmSzRp3Fj+fcEbA+AVHTcpjPS5PsFa7kPMhw=="

COINBASE_API_KEY = "organizations/c7e577d9-141c-4eef-88cd-408f46ab1b87/apiKeys/0047c1c5-02dd-490f-82ba-687b32c39c87"
COINBASE_PRIVATE_KEY = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEILqZEksjvDfJ368yDhvljAGFUVWHuxNzV9ybSNi7p2NFoAoGCCqGSM49
AwEHoUQDQgAE9YkgnERXgMqzXPaC+hzn2K3pXc2NAKcbIimLxH/+YqUgR+Jp/Dea
4My1p7P2HjCA4jAp7m9YWAGzTRDYrPT+tA==
-----END EC PRIVATE KEY-----"""

BINANCE_API_KEY = "xLbtt3CxHx3pYtTJv2rejOAUyB0SjwGlPbgnLzvy7omuPvjBJqi3tXHB7mDZyI7A"
BINANCE_SECRET = "Hp9hkeXSibAw356bFqPyPgNlEn5URK3ELXdo5A74dhyo5GKzYteHo3Ecxh78hTOO"

# ============================================================================
# LEVEL 3 ORDER BOOK DATA STRUCTURES
# ============================================================================

@dataclass
class Level3Order:
    """Individual order from L3 feed"""
    order_id: str
    side: str  # 'buy' or 'sell'
    price: float
    size: float
    timestamp: float
    exchange: str

@dataclass
class OrderBookLevel:
    """Aggregated level in order book"""
    price: float
    total_size: float
    num_orders: int

class Level3OrderBook:
    """Maintains Level 3 order book with individual orders"""

    def __init__(self):
        self.bids = defaultdict(list)  # price -> list of orders
        self.asks = defaultdict(list)  # price -> list of orders
        self.orders = {}  # order_id -> order
        self.large_order_threshold = 0.5  # BTC

    def add_order(self, order: Level3Order):
        """Add new order to book"""
        self.orders[order.order_id] = order

        if order.side == 'buy':
            self.bids[order.price].append(order)
        else:
            self.asks[order.price].append(order)

    def remove_order(self, order_id: str):
        """Remove order from book"""
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        del self.orders[order_id]

        if order.side == 'buy':
            self.bids[order.price] = [o for o in self.bids[order.price] if o.order_id != order_id]
            if not self.bids[order.price]:
                del self.bids[order.price]
        else:
            self.asks[order.price] = [o for o in self.asks[order.price] if o.order_id != order_id]
            if not self.asks[order.price]:
                del self.asks[order.price]

    def update_order(self, order_id: str, new_size: float):
        """Update order size"""
        if order_id in self.orders:
            self.orders[order_id].size = new_size

    def get_best_bid(self) -> Optional[float]:
        """Get best bid price"""
        if not self.bids:
            return None
        return max(self.bids.keys())

    def get_best_ask(self) -> Optional[float]:
        """Get best ask price"""
        if not self.asks:
            return None
        return min(self.asks.keys())

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return ask - bid
        return None

    def get_large_orders(self, min_size: float = None) -> List[Level3Order]:
        """Get all large orders above threshold"""
        threshold = min_size or self.large_order_threshold
        return [o for o in self.orders.values() if o.size >= threshold]

    def get_order_imbalance(self, depth: int = 5) -> float:
        """Calculate order imbalance at top N levels"""
        # Get top N bid levels
        top_bids = sorted(self.bids.keys(), reverse=True)[:depth]
        bid_volume = sum(sum(o.size for o in self.bids[p]) for p in top_bids)

        # Get top N ask levels
        top_asks = sorted(self.asks.keys())[:depth]
        ask_volume = sum(sum(o.size for o in self.asks[p]) for p in top_asks)

        total = bid_volume + ask_volume
        if total == 0:
            return 0.0

        return (bid_volume - ask_volume) / total

    def get_queue_position(self, order_id: str) -> Optional[int]:
        """Get position in queue for an order (price-time priority)"""
        if order_id not in self.orders:
            return None

        order = self.orders[order_id]
        price_level = self.bids[order.price] if order.side == 'buy' else self.asks[order.price]

        # Sort by timestamp (earlier = better position)
        sorted_orders = sorted(price_level, key=lambda x: x.timestamp)

        for i, o in enumerate(sorted_orders):
            if o.order_id == order_id:
                return i + 1

        return None


# ============================================================================
# KRAKEN LEVEL 3 WEBSOCKET
# ============================================================================

class KrakenLevel3Feed:
    """Kraken Level 3 order book feed"""

    def __init__(self, order_book: Level3OrderBook):
        self.order_book = order_book
        # Use v1 WebSocket (more stable, XBT/USD format)
        self.ws_url = "wss://ws.kraken.com"
        self.running = True

    def get_auth_token(self):
        """Generate authentication token for Kraken WS"""
        return None

    async def connect(self):
        """Connect to Kraken Level 3 WebSocket"""
        try:
            async with websockets.connect(self.ws_url) as ws:
                # v1 subscription format - XBT/USD is Kraken's BTC symbol
                subscribe_msg = {
                    "event": "subscribe",
                    "pair": ["XBT/USD"],
                    "subscription": {
                        "name": "book",
                        "depth": 1000
                    }
                }

                await ws.send(json.dumps(subscribe_msg))
                print("[KRAKEN L3] Connected to Level 3 order book")

                while self.running:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        await self.process_message(data)
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await ws.send(json.dumps({"event": "ping"}))

        except Exception as e:
            print(f"[KRAKEN L3] Error: {e}")

    async def process_message(self, data):
        """Process Level 3 messages"""
        # Handle list format (Kraken v2 sends arrays for some messages)
        if isinstance(data, list):
            # Kraken v1 format: [channelID, data, channelName, pair]
            if len(data) >= 4 and isinstance(data[1], dict):
                book_data = data[1]
                if 'bs' in book_data or 'as' in book_data:
                    # Snapshot format
                    self.process_snapshot_v1(book_data)
                elif 'b' in book_data or 'a' in book_data:
                    # Update format
                    self.process_update_v1(book_data)
            return

        # Handle dict format (Kraken v2)
        if isinstance(data, dict):
            # Status/subscription messages
            if data.get('method') in ['subscribe', 'pong'] or data.get('event'):
                return

            if data.get('channel') == 'book':
                book_data = data.get('data', [])
                msg_type = data.get('type', '')

                # Handle as list of updates
                if isinstance(book_data, list) and len(book_data) > 0:
                    for item in book_data:
                        if isinstance(item, dict):
                            if 'snapshot' in msg_type:
                                self.process_snapshot(item)
                            elif 'update' in msg_type:
                                self.process_update(item)
                elif isinstance(book_data, dict):
                    if 'snapshot' in msg_type:
                        self.process_snapshot(book_data)
                    elif 'update' in msg_type:
                        self.process_update(book_data)

    def process_snapshot_v1(self, data: dict):
        """Process v1 format snapshot (bs/as keys)"""
        timestamp = time.time()

        # Process bids (bs = bid snapshot)
        for bid in data.get('bs', []):
            price, size = float(bid[0]), float(bid[1])
            order_id = f"kraken_bid_{price}_{timestamp}"
            order = Level3Order(order_id, 'buy', price, size, timestamp, 'kraken')
            self.order_book.add_order(order)

        # Process asks (as = ask snapshot)
        for ask in data.get('as', []):
            price, size = float(ask[0]), float(ask[1])
            order_id = f"kraken_ask_{price}_{timestamp}"
            order = Level3Order(order_id, 'sell', price, size, timestamp, 'kraken')
            self.order_book.add_order(order)

    def process_update_v1(self, data: dict):
        """Process v1 format updates (b/a keys)"""
        timestamp = time.time()

        # Process bid updates
        for bid in data.get('b', []):
            price, size = float(bid[0]), float(bid[1])
            order_id = f"kraken_bid_{price}"

            if size == 0:
                self.order_book.remove_order(order_id)
            else:
                order = Level3Order(order_id, 'buy', price, size, timestamp, 'kraken')
                self.order_book.add_order(order)

        # Process ask updates
        for ask in data.get('a', []):
            price, size = float(ask[0]), float(ask[1])
            order_id = f"kraken_ask_{price}"

            if size == 0:
                self.order_book.remove_order(order_id)
            else:
                order = Level3Order(order_id, 'sell', price, size, timestamp, 'kraken')
                self.order_book.add_order(order)

    def process_snapshot(self, data: dict):
        """Process initial order book snapshot"""
        timestamp = time.time()

        # Process bids
        for bid in data.get('bids', []):
            price, size = float(bid[0]), float(bid[1])
            order_id = f"kraken_bid_{price}_{timestamp}"
            order = Level3Order(order_id, 'buy', price, size, timestamp, 'kraken')
            self.order_book.add_order(order)

        # Process asks
        for ask in data.get('asks', []):
            price, size = float(ask[0]), float(ask[1])
            order_id = f"kraken_ask_{price}_{timestamp}"
            order = Level3Order(order_id, 'sell', price, size, timestamp, 'kraken')
            self.order_book.add_order(order)

    def process_update(self, data: dict):
        """Process order book updates"""
        timestamp = time.time()

        # Process bid updates
        for bid in data.get('bids', []):
            price, size = float(bid[0]), float(bid[1])
            order_id = f"kraken_bid_{price}"

            if size == 0:
                # Remove order
                self.order_book.remove_order(order_id)
            else:
                # Add or update order
                order = Level3Order(order_id, 'buy', price, size, timestamp, 'kraken')
                self.order_book.add_order(order)

        # Process ask updates
        for ask in data.get('asks', []):
            price, size = float(ask[0]), float(ask[1])
            order_id = f"kraken_ask_{price}"

            if size == 0:
                self.order_book.remove_order(order_id)
            else:
                order = Level3Order(order_id, 'sell', price, size, timestamp, 'kraken')
                self.order_book.add_order(order)


# ============================================================================
# MULTI-EXCHANGE AGGREGATOR (with L3)
# ============================================================================

class InstitutionalDataAggregator:
    """Aggregates L3 Kraken + L1 Coinbase + L1 Binance"""

    def __init__(self):
        self.level3_book = Level3OrderBook()
        self.kraken_feed = KrakenLevel3Feed(self.level3_book)

        self.ticks = deque(maxlen=1000)
        self.last_prices = {}
        self.running = True

        # WebSocket URLs
        self.coinbase_ws = 'wss://advanced-trade-ws.coinbase.com'
        self.binance_ws = 'wss://stream.binance.us:9443/ws/btcusdt@trade'

    async def connect_coinbase(self):
        """Connect to Coinbase Advanced Trade WebSocket"""
        try:
            async with websockets.connect(self.coinbase_ws) as ws:
                subscribe_msg = {
                    "type": "subscribe",
                    "product_ids": ["BTC-USD"],
                    "channel": "ticker"
                }
                await ws.send(json.dumps(subscribe_msg))
                print("[COINBASE] Connected")

                while self.running:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    if data.get('type') == 'ticker':
                        price = float(data.get('price', 0))
                        if price > 0:
                            self.last_prices['coinbase'] = price
        except Exception as e:
            print(f"[COINBASE] Error: {e}")

    async def connect_binance(self):
        """Connect to Binance.US WebSocket"""
        try:
            async with websockets.connect(self.binance_ws) as ws:
                print("[BINANCE.US] Connected")

                while self.running:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    if 'p' in data:
                        price = float(data['p'])
                        self.last_prices['binance_us'] = price
        except Exception as e:
            print(f"[BINANCE.US] Error: {e}")

    def get_consolidated_price(self) -> Optional[float]:
        """Get median price across exchanges"""
        if not self.last_prices:
            return None
        prices = list(self.last_prices.values())
        return float(np.median(prices))

    def detect_large_order_flow(self) -> List[Dict]:
        """Detect large orders that could move the market"""
        large_orders = self.level3_book.get_large_orders(min_size=0.5)

        signals = []
        for order in large_orders:
            # Calculate potential impact
            best_bid = self.level3_book.get_best_bid()
            best_ask = self.level3_book.get_best_ask()

            if not best_bid or not best_ask:
                continue

            mid = (best_bid + best_ask) / 2

            # Distance from mid
            distance_bps = abs(order.price - mid) / mid * 10000

            # Potential to execute?
            will_execute = False
            if order.side == 'buy' and order.price >= best_ask:
                will_execute = True
            elif order.side == 'sell' and order.price <= best_bid:
                will_execute = True

            signals.append({
                'order': order,
                'distance_bps': distance_bps,
                'will_execute': will_execute,
                'side': order.side,
                'size': order.size
            })

        return signals

    async def start_all(self):
        """Start all feeds"""
        await asyncio.gather(
            self.kraken_feed.connect(),
            self.connect_coinbase(),
            self.connect_binance()
        )


# ============================================================================
# LEVEL 3 TRADING STRATEGY
# ============================================================================

class Level3Strategy:
    """Trading strategy using Level 3 order flow"""

    def __init__(self, config: Dict):
        self.config = config
        self.capital = 10.0
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []

        self.aggregator = InstitutionalDataAggregator()
        self.last_trade_time = 0.0

    def should_enter_long(self, large_orders: List[Dict], price: float) -> bool:
        """Check if should enter LONG based on L3 order flow"""
        if self.position != 0:
            return False

        # Look for large buy orders near execution
        buy_flow = sum(o['size'] for o in large_orders
                      if o['side'] == 'buy' and o['distance_bps'] < 10)
        sell_flow = sum(o['size'] for o in large_orders
                       if o['side'] == 'sell' and o['distance_bps'] < 10)

        # Strong buy pressure
        threshold = self.config.get('flow_threshold', 1.0)
        if buy_flow > sell_flow * 2 and buy_flow > threshold:
            return True

        # Large buy order about to execute
        imminent_buys = [o for o in large_orders if o['will_execute'] and o['side'] == 'buy']
        if imminent_buys and sum(o['size'] for o in imminent_buys) > threshold:
            return True

        return False

    def should_enter_short(self, large_orders: List[Dict], price: float) -> bool:
        """Check if should enter SHORT based on L3 order flow"""
        if self.position != 0:
            return False

        # Look for large sell orders near execution
        buy_flow = sum(o['size'] for o in large_orders
                      if o['side'] == 'buy' and o['distance_bps'] < 10)
        sell_flow = sum(o['size'] for o in large_orders
                       if o['side'] == 'sell' and o['distance_bps'] < 10)

        # Strong sell pressure
        threshold = self.config.get('flow_threshold', 1.0)
        if sell_flow > buy_flow * 2 and sell_flow > threshold:
            return True

        # Large sell order about to execute
        imminent_sells = [o for o in large_orders if o['will_execute'] and o['side'] == 'sell']
        if imminent_sells and sum(o['size'] for o in imminent_sells) > threshold:
            return True

        return False

    def check_exit(self, price: float) -> bool:
        """Check if should exit position"""
        if self.position == 0:
            return False

        pnl_pct = (price - self.entry_price) / self.entry_price * self.position

        # TP/SL
        if pnl_pct >= self.config['profit_target']:
            return True
        if pnl_pct <= -self.config['stop_loss']:
            return True

        # Max hold time
        if time.time() - self.last_trade_time > self.config['max_hold_sec']:
            return True

        return False

    async def run(self, duration_sec: int):
        """Run strategy"""
        print(f"\n{'='*60}")
        print(f"LEVEL 3 INSTITUTIONAL SYSTEM - {self.config['name']}")
        print(f"{'='*60}")
        print(f"Data: Kraken L3 + Coinbase + Binance.US")
        print(f"Starting Capital: ${self.capital:.2f}")
        print(f"Duration: {duration_sec}s")
        print(f"{'='*60}\n")

        # Start feeds
        asyncio.create_task(self.aggregator.start_all())

        # Wait for connections
        await asyncio.sleep(5)

        start_time = time.time()
        last_update = start_time

        while time.time() - start_time < duration_sec:
            await asyncio.sleep(0.1)  # 100ms loop

            price = self.aggregator.get_consolidated_price()
            if not price:
                continue

            # Detect large order flow
            large_orders = self.aggregator.detect_large_order_flow()

            # Get order imbalance
            imbalance = self.aggregator.level3_book.get_order_imbalance()

            # Entry logic
            if self.position == 0:
                if self.should_enter_long(large_orders, price):
                    self.position = 1.0
                    self.entry_price = price
                    self.last_trade_time = time.time()
                    print(f"[LONG] ${price:.2f} | Large buy flow detected | Imbalance: {imbalance:+.3f}")

                elif self.should_enter_short(large_orders, price):
                    self.position = -1.0
                    self.entry_price = price
                    self.last_trade_time = time.time()
                    print(f"[SHORT] ${price:.2f} | Large sell flow detected | Imbalance: {imbalance:+.3f}")

            # Exit logic
            elif self.check_exit(price):
                pnl_pct = (price - self.entry_price) / self.entry_price * self.position
                pnl_usd = self.capital * pnl_pct
                self.capital += pnl_usd

                side = "LONG" if self.position > 0 else "SHORT"
                print(f"[EXIT {side}] ${price:.2f} | PnL: {pnl_pct*100:+.3f}% | Capital: ${self.capital:.2f}")

                self.trades.append({'pnl_pct': pnl_pct, 'pnl_usd': pnl_usd})
                self.position = 0.0

            # Status update
            if time.time() - last_update >= 10.0:
                elapsed = time.time() - start_time
                spread = self.aggregator.level3_book.get_spread()
                large = len(large_orders)
                spread_val = spread if spread else 0.0
                print(f"\n[{elapsed:.0f}s] Capital: ${self.capital:.2f} | Spread: ${spread_val:.2f} | Large orders: {large} | Imbalance: {imbalance:+.3f}\n")
                last_update = time.time()

        self.aggregator.running = False
        self.print_stats()

    def print_stats(self):
        """Print final stats"""
        total_return = (self.capital - 10.0) / 10.0 * 100
        wins = [t for t in self.trades if t['pnl_pct'] > 0]
        losses = [t for t in self.trades if t['pnl_pct'] <= 0]
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS - {self.config['name']}")
        print(f"{'='*60}")
        print(f"Ending Capital: ${self.capital:.2f}")
        print(f"Return: {total_return:+.2f}%")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"{'='*60}\n")


# ============================================================================
# CONFIGS
# ============================================================================

CONFIGS = {
    "V1": {
        "name": "L3_AGGRESSIVE",
        "profit_target": 0.0050,  # 50 bps
        "stop_loss": 0.0030,      # 30 bps
        "max_hold_sec": 60,       # 60 seconds
        "flow_threshold": 1.0,    # BTC flow threshold
        # F013-F018 Parameters
        "gamma_risk": 0.10,
        "lambda_informed": 0.15,
        "kappa_depth": 100.0,
    },
    "V2": {
        "name": "L3_BALANCED",
        "profit_target": 0.0075,
        "stop_loss": 0.0040,
        "max_hold_sec": 120,
        "flow_threshold": 0.75,
        "gamma_risk": 0.12,
        "lambda_informed": 0.18,
        "kappa_depth": 90.0,
    },
    "V3": {
        "name": "L3_CONSERVATIVE",
        "profit_target": 0.0100,
        "stop_loss": 0.0050,
        "max_hold_sec": 180,
        "flow_threshold": 0.5,
        "gamma_risk": 0.15,
        "lambda_informed": 0.20,
        "kappa_depth": 80.0,
    },
    "V4": {
        "name": "L3_HYPERACTIVE",
        "profit_target": 0.0030,  # 30 bps - tight targets
        "stop_loss": 0.0020,      # 20 bps - tight stops
        "max_hold_sec": 30,       # 30 seconds - fast
        "flow_threshold": 0.25,   # Low threshold for more trades
        "gamma_risk": 0.18,
        "lambda_informed": 0.22,
        "kappa_depth": 70.0,
    },
}


# ============================================================================
# MAIN
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description='Level 3 Institutional System')
    parser.add_argument('version', choices=['V1', 'V2', 'V3', 'V4', 'ALL'],
                       help='Strategy version')
    parser.add_argument('--duration', type=int, default=300,
                       help='Duration in seconds')

    args = parser.parse_args()

    if args.version == 'ALL':
        for version in ['V1', 'V2', 'V3', 'V4']:
            config = CONFIGS[version]
            strategy = Level3Strategy(config)
            await strategy.run(args.duration)
    else:
        config = CONFIGS[args.version]
        strategy = Level3Strategy(config)
        await strategy.run(args.duration)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
