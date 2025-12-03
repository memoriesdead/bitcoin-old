"""
BINANCE FUTURES EXECUTOR
========================
Production-ready Binance Futures implementation.

Binance Futures is the #1 crypto derivatives exchange:
- 70% of global crypto futures volume
- Up to 10,000 orders/sec (Market Maker tier)
- Sub-millisecond latency with co-location
- WebSocket for real-time updates

API Limits (by VIP level):
- Standard: 1,200 orders/min (20/sec)
- VIP 1-5:  6,000-18,000 orders/min
- VIP 6-9:  24,000-60,000 orders/min
- MM:       Up to 600,000 orders/min (10,000/sec)

To get Market Maker status:
1. Apply through Binance institutional program
2. Provide 0.2%+ of daily volume
3. Maintain 80%+ uptime with tight spreads
"""
import asyncio
import hashlib
import hmac
import json
import time
from typing import Dict, List, Optional
from urllib.parse import urlencode

try:
    import aiohttp
    import websockets
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    print("[BINANCE] Warning: aiohttp/websockets not installed")
    print("[BINANCE] Install with: pip install aiohttp websockets")

from .base_executor import (
    BaseExecutor, Order, Position, ExchangeConfig,
    OrderSide, OrderType, OrderStatus
)


class BinanceExecutor(BaseExecutor):
    """
    Binance Futures executor.

    Supports both testnet and mainnet:
    - Testnet: testnet.binancefuture.com (free test money)
    - Mainnet: fapi.binance.com (real trading)

    Uses WebSocket for:
    - Real-time orderbook updates
    - Trade execution confirmations
    - Position updates
    """

    # API Endpoints
    MAINNET_REST = "https://fapi.binance.com"
    MAINNET_WS = "wss://fstream.binance.com"
    TESTNET_REST = "https://testnet.binancefuture.com"
    TESTNET_WS = "wss://stream.binancefuture.com"

    def __init__(self, config: ExchangeConfig):
        super().__init__(config)

        # Select endpoints based on testnet flag
        if config.testnet:
            self.rest_url = self.TESTNET_REST
            self.ws_url = self.TESTNET_WS
        else:
            self.rest_url = self.MAINNET_REST
            self.ws_url = self.MAINNET_WS

        # Connection state
        self._ws = None
        self._session: Optional['aiohttp.ClientSession'] = None
        self._listen_key = None
        self._ws_connected = False

        # Orderbook cache (for smart routing)
        self._orderbooks: Dict[str, Dict] = {}

    def _sign_request(self, params: Dict) -> str:
        """Sign request with HMAC-SHA256."""
        query_string = urlencode(params)
        signature = hmac.new(
            self.config.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"{query_string}&signature={signature}"

    async def connect(self) -> bool:
        """Establish connections to Binance."""
        if not DEPS_AVAILABLE:
            print("[BINANCE] Dependencies not available")
            return False

        try:
            # Create HTTP session
            self._session = aiohttp.ClientSession(headers={
                'X-MBX-APIKEY': self.config.api_key
            })

            # Get listen key for user data stream
            async with self._session.post(
                f"{self.rest_url}/fapi/v1/listenKey"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._listen_key = data.get('listenKey')
                else:
                    print(f"[BINANCE] Failed to get listen key: {resp.status}")
                    return False

            # Connect WebSocket for user data
            ws_url = f"{self.ws_url}/ws/{self._listen_key}"
            self._ws = await websockets.connect(ws_url)
            self._ws_connected = True

            # Start background tasks
            asyncio.create_task(self._keep_alive())
            asyncio.create_task(self._process_messages())

            print(f"[BINANCE] Connected to {'TESTNET' if self.config.testnet else 'MAINNET'}")
            return True

        except Exception as e:
            print(f"[BINANCE] Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Close all connections."""
        self._ws_connected = False

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._session:
            await self._session.close()
            self._session = None

        print("[BINANCE] Disconnected")

    async def _keep_alive(self) -> None:
        """Keep listen key alive (required every 30 minutes)."""
        while self._ws_connected:
            await asyncio.sleep(1800)  # 30 minutes
            if self._session and self._listen_key:
                try:
                    await self._session.put(
                        f"{self.rest_url}/fapi/v1/listenKey"
                    )
                except Exception as e:
                    print(f"[BINANCE] Keep-alive failed: {e}")

    async def _process_messages(self) -> None:
        """Process incoming WebSocket messages."""
        while self._ws_connected and self._ws:
            try:
                msg = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=60
                )
                data = json.loads(msg)
                await self._handle_message(data)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                print("[BINANCE] WebSocket disconnected")
                break
            except Exception as e:
                print(f"[BINANCE] Message error: {e}")

    async def _handle_message(self, data: Dict) -> None:
        """Handle incoming WebSocket message."""
        event_type = data.get('e')

        if event_type == 'ORDER_TRADE_UPDATE':
            # Order update
            order_data = data.get('o', {})
            order_id = order_data.get('c')  # Client order ID

            if order_id in self.open_orders:
                order = self.open_orders[order_id]
                status = order_data.get('X')

                if status == 'FILLED':
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = float(order_data.get('z', 0))
                    order.filled_price = float(order_data.get('ap', 0))
                    order.exchange_order_id = str(order_data.get('i'))

                    # Move to history
                    del self.open_orders[order_id]
                    self.order_history.append(order)
                    self.stats['orders_filled'] += 1

                elif status == 'CANCELED':
                    order.status = OrderStatus.CANCELLED
                    del self.open_orders[order_id]

                elif status == 'REJECTED':
                    order.status = OrderStatus.REJECTED
                    del self.open_orders[order_id]
                    self.stats['orders_rejected'] += 1

        elif event_type == 'ACCOUNT_UPDATE':
            # Position update
            positions = data.get('a', {}).get('P', [])
            for pos_data in positions:
                symbol = pos_data.get('s')
                quantity = float(pos_data.get('pa', 0))
                entry_price = float(pos_data.get('ep', 0))
                unrealized_pnl = float(pos_data.get('up', 0))

                if quantity != 0:
                    self.positions[symbol] = Position(
                        instrument=symbol,
                        side=OrderSide.BUY if quantity > 0 else OrderSide.SELL,
                        quantity=abs(quantity),
                        entry_price=entry_price,
                        unrealized_pnl=unrealized_pnl,
                        realized_pnl=0.0,
                        exchange=self.config.name
                    )
                elif symbol in self.positions:
                    del self.positions[symbol]

    async def submit_order(self, order: Order) -> Order:
        """Submit order to Binance Futures."""
        if not self._session:
            order.status = OrderStatus.REJECTED
            return order

        start_time = time.perf_counter()

        # Build order parameters
        params = {
            'symbol': order.instrument,
            'side': order.side.value,
            'type': order.order_type.value,
            'quantity': f"{order.quantity:.8f}",
            'newClientOrderId': order.id,
            'timestamp': int(time.time() * 1000),
        }

        if order.order_type == OrderType.LIMIT and order.price:
            params['price'] = f"{order.price:.8f}"
            params['timeInForce'] = 'GTC'

        # Sign and send
        signed = self._sign_request(params)

        try:
            async with self._session.post(
                f"{self.rest_url}/fapi/v1/order?{signed}"
            ) as resp:
                latency_ms = (time.perf_counter() - start_time) * 1000

                if resp.status == 200:
                    data = await resp.json()
                    order.exchange_order_id = str(data.get('orderId'))
                    order.status = OrderStatus.OPEN
                    self.open_orders[order.id] = order
                    self.stats['orders_sent'] += 1

                    # Update average latency
                    n = self.stats['orders_sent']
                    self.stats['avg_latency_ms'] = (
                        (self.stats['avg_latency_ms'] * (n - 1) + latency_ms) / n
                    )
                else:
                    error = await resp.text()
                    print(f"[BINANCE] Order rejected: {error}")
                    order.status = OrderStatus.REJECTED
                    self.stats['orders_rejected'] += 1

        except Exception as e:
            print(f"[BINANCE] Order error: {e}")
            order.status = OrderStatus.REJECTED

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if order_id not in self.open_orders:
            return False

        order = self.open_orders[order_id]
        params = {
            'symbol': order.instrument,
            'origClientOrderId': order_id,
            'timestamp': int(time.time() * 1000),
        }
        signed = self._sign_request(params)

        try:
            async with self._session.delete(
                f"{self.rest_url}/fapi/v1/order?{signed}"
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def get_position(self, instrument: str) -> Optional[Position]:
        """Get current position for an instrument."""
        return self.positions.get(instrument)

    async def get_orderbook(self, instrument: str) -> Dict:
        """Get current orderbook."""
        if not self._session:
            return {}

        try:
            async with self._session.get(
                f"{self.rest_url}/fapi/v1/depth",
                params={'symbol': instrument, 'limit': 10}
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            print(f"[BINANCE] Orderbook error: {e}")

        return {}

    async def get_balance(self) -> Dict[str, float]:
        """Get account balances."""
        if not self._session:
            return {}

        params = {
            'timestamp': int(time.time() * 1000),
        }
        signed = self._sign_request(params)

        try:
            async with self._session.get(
                f"{self.rest_url}/fapi/v2/balance?{signed}"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        item['asset']: float(item['balance'])
                        for item in data
                    }
        except Exception as e:
            print(f"[BINANCE] Balance error: {e}")

        return {}


def create_binance_executor(
    api_key: str,
    api_secret: str,
    testnet: bool = True
) -> BinanceExecutor:
    """
    Factory function to create a Binance executor.

    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        testnet: Use testnet (True) or mainnet (False)

    Returns:
        BinanceExecutor ready to connect
    """
    config = ExchangeConfig(
        name='binance',
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        rate_limit_per_second=20,  # Standard tier
        rate_limit_per_minute=1200,
        max_position_size=10000.0,
        maker_fee=0.0002,  # 0.02%
        taker_fee=0.0004,  # 0.04%
    )
    return BinanceExecutor(config)
