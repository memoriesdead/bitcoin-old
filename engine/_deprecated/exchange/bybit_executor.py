"""
BYBIT UNIFIED EXECUTOR - GOLD STANDARD BACKUP
==============================================
Bybit is the #2 crypto derivatives exchange.

Why Bybit for HFT backup:
- 50+ orders/sec (standard), 500+/sec (VIP)
- Sub-10ms API latency
- Unified Trading Account (spot + derivatives)
- Excellent WebSocket feeds
- No KYC required under 2 BTC/day

API Limits:
- Standard: 120 requests/sec, 20 orders/sec
- VIP 1:    300 requests/sec, 50 orders/sec
- VIP 2-4:  600-1500 requests/sec, 100-300 orders/sec
- VIP 5:    3000 requests/sec, 500 orders/sec

Fee Structure (Unified):
- Tier 0: 0.02% maker, 0.055% taker
- Tier 1: 0.018% maker, 0.04% taker
- VIP 1:  0.016% maker, 0.036% taker
- VIP 5:  0.00% maker, 0.02% taker (best tier)
"""
import asyncio
import hashlib
import hmac
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from .base_executor import (
    BaseExecutor, Order, Position, ExchangeConfig,
    OrderSide, OrderType, OrderStatus, Signal
)


@dataclass
class BybitConfig:
    """Bybit configuration."""
    api_key: str
    api_secret: str
    testnet: bool = True
    recv_window: int = 5000
    max_retries: int = 3


class BybitExecutor(BaseExecutor):
    """
    Bybit Unified Trading executor.

    GOLD STANDARD BACKUP:
    - Fast API with good rate limits
    - Reliable infrastructure
    - Good liquidity on majors
    """

    # API endpoints
    MAINNET_URL = "https://api.bybit.com"
    TESTNET_URL = "https://api-testnet.bybit.com"

    def __init__(self, bybit_config: BybitConfig):
        base_config = ExchangeConfig(
            name="bybit",
            api_key=bybit_config.api_key,
            api_secret=bybit_config.api_secret,
            testnet=bybit_config.testnet,
            rate_limit_per_second=50,  # Conservative default
            rate_limit_per_minute=3000,
            max_position_size=10000.0,
            maker_fee=0.0002,  # 0.02%
            taker_fee=0.00055,  # 0.055%
        )
        super().__init__(base_config)

        self.bybit_config = bybit_config
        self.base_url = self.TESTNET_URL if bybit_config.testnet else self.MAINNET_URL
        self.session: Optional[aiohttp.ClientSession] = None

        # Health tracking
        self._consecutive_errors = 0
        self._last_success = 0
        self._healthy = True

    def _sign_request(self, params: Dict) -> Dict:
        """Sign request with HMAC-SHA256."""
        timestamp = int(time.time() * 1000)
        params['api_key'] = self.config.api_key
        params['timestamp'] = timestamp
        params['recv_window'] = self.bybit_config.recv_window

        # Sort and create query string
        sorted_params = sorted(params.items())
        query_string = '&'.join([f"{k}={v}" for k, v in sorted_params])

        # Sign
        signature = hmac.new(
            self.config.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        params['sign'] = signature
        return params

    def _sign_request_v5(self, params: Dict) -> Dict[str, str]:
        """Sign request for V5 API (different signature method)."""
        timestamp = str(int(time.time() * 1000))
        recv_window = str(self.bybit_config.recv_window)

        # V5 signature: timestamp + api_key + recv_window + query_string
        param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        sign_str = f"{timestamp}{self.config.api_key}{recv_window}{param_str}"

        signature = hmac.new(
            self.config.api_secret.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return {
            'X-BAPI-API-KEY': self.config.api_key,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': recv_window,
            'X-BAPI-SIGN': signature,
            'Content-Type': 'application/json'
        }

    async def connect(self) -> bool:
        """Connect to Bybit API."""
        if not AIOHTTP_AVAILABLE:
            print("[BYBIT] aiohttp not available")
            return False

        try:
            self.session = aiohttp.ClientSession()

            # Test connection with account info
            result = await self._request('GET', '/v5/account/wallet-balance', {'accountType': 'UNIFIED'})

            if result.get('retCode') == 0:
                network = "TESTNET" if self.bybit_config.testnet else "MAINNET"
                print(f"[BYBIT] Connected to {network}")

                # Get balance
                data = result.get('result', {})
                if data.get('list'):
                    account = data['list'][0]
                    equity = float(account.get('totalEquity', 0))
                    print(f"[BYBIT] Account Equity: ${equity:,.2f}")

                self._healthy = True
                self._last_success = time.time()
                return True
            else:
                print(f"[BYBIT] Connection failed: {result.get('retMsg')}")
                return False

        except Exception as e:
            print(f"[BYBIT] Connection error: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Bybit."""
        if self.session:
            await self.session.close()
            self.session = None
        print("[BYBIT] Disconnected")

    async def _request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated API request."""
        if not self.session:
            return {'retCode': -1, 'retMsg': 'Not connected'}

        params = params or {}
        url = f"{self.base_url}{endpoint}"
        headers = self._sign_request_v5(params)

        try:
            if method == 'GET':
                if params:
                    url += '?' + '&'.join([f"{k}={v}" for k, v in params.items()])
                async with self.session.get(url, headers=headers) as resp:
                    return await resp.json()
            else:
                async with self.session.post(url, headers=headers, json=params) as resp:
                    return await resp.json()

        except Exception as e:
            self._consecutive_errors += 1
            if self._consecutive_errors > 5:
                self._healthy = False
            return {'retCode': -1, 'retMsg': str(e)}

    async def submit_order(self, order: Order) -> Order:
        """Submit order to Bybit."""
        start_time = time.perf_counter()

        # Map to Bybit format
        side = "Buy" if order.side == OrderSide.BUY else "Sell"

        # Symbol mapping (e.g., BTC -> BTCUSDT)
        symbol = self._map_symbol(order.instrument)

        params = {
            'category': 'linear',  # USDT perpetuals
            'symbol': symbol,
            'side': side,
            'orderType': 'Market' if order.order_type == OrderType.MARKET else 'Limit',
            'qty': str(order.quantity),
            'timeInForce': 'GTC',
        }

        if order.order_type == OrderType.LIMIT and order.price:
            params['price'] = str(order.price)

        # Add TP/SL if specified
        if order.take_profit:
            params['takeProfit'] = str(order.take_profit)
        if order.stop_loss:
            params['stopLoss'] = str(order.stop_loss)

        result = await self._request('POST', '/v5/order/create', params)
        latency_ms = (time.perf_counter() - start_time) * 1000

        if result.get('retCode') == 0:
            data = result.get('result', {})
            order.exchange_order_id = data.get('orderId', '')
            order.status = OrderStatus.OPEN

            self.stats['orders_sent'] += 1
            self._consecutive_errors = 0
            self._last_success = time.time()
            self._healthy = True

            # Update latency
            n = self.stats['orders_sent']
            self.stats['avg_latency_ms'] = (self.stats['avg_latency_ms'] * (n-1) + latency_ms) / n

            print(f"[BYBIT] Order submitted: {symbol} {side} {order.quantity} ({latency_ms:.1f}ms)")

            # Check for immediate fill (market orders)
            if order.order_type == OrderType.MARKET:
                await self._check_order_fill(order)
        else:
            order.status = OrderStatus.REJECTED
            self.stats['orders_rejected'] += 1
            self._consecutive_errors += 1
            print(f"[BYBIT] Order rejected: {result.get('retMsg')}")

        return order

    async def _check_order_fill(self, order: Order) -> None:
        """Check if order was filled."""
        params = {
            'category': 'linear',
            'orderId': order.exchange_order_id,
        }

        result = await self._request('GET', '/v5/order/realtime', params)

        if result.get('retCode') == 0:
            orders = result.get('result', {}).get('list', [])
            if orders:
                order_data = orders[0]
                status = order_data.get('orderStatus')

                if status == 'Filled':
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = float(order_data.get('cumExecQty', 0))
                    order.filled_price = float(order_data.get('avgPrice', 0))
                    self.stats['orders_filled'] += 1
                elif status == 'PartiallyFilled':
                    order.status = OrderStatus.PARTIALLY_FILLED
                    order.filled_quantity = float(order_data.get('cumExecQty', 0))

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Bybit."""
        if order_id not in self.open_orders:
            return False

        order = self.open_orders[order_id]
        symbol = self._map_symbol(order.instrument)

        params = {
            'category': 'linear',
            'symbol': symbol,
            'orderId': order.exchange_order_id,
        }

        result = await self._request('POST', '/v5/order/cancel', params)

        if result.get('retCode') == 0:
            del self.open_orders[order_id]
            return True

        return False

    async def get_position(self, instrument: str) -> Optional[Position]:
        """Get position on Bybit."""
        symbol = self._map_symbol(instrument)

        params = {
            'category': 'linear',
            'symbol': symbol,
        }

        result = await self._request('GET', '/v5/position/list', params)

        if result.get('retCode') == 0:
            positions = result.get('result', {}).get('list', [])
            if positions:
                pos = positions[0]
                size = float(pos.get('size', 0))
                if size > 0:
                    side = pos.get('side')
                    return Position(
                        instrument=instrument,
                        side=OrderSide.BUY if side == 'Buy' else OrderSide.SELL,
                        quantity=size,
                        entry_price=float(pos.get('avgPrice', 0)),
                        unrealized_pnl=float(pos.get('unrealisedPnl', 0)),
                        realized_pnl=float(pos.get('cumRealisedPnl', 0)),
                        exchange="bybit"
                    )

        return None

    async def get_orderbook(self, instrument: str) -> Dict:
        """Get orderbook from Bybit."""
        symbol = self._map_symbol(instrument)

        params = {
            'category': 'linear',
            'symbol': symbol,
            'limit': 25,
        }

        result = await self._request('GET', '/v5/market/orderbook', params)

        if result.get('retCode') == 0:
            data = result.get('result', {})
            return {
                'bids': [[float(b[0]), float(b[1])] for b in data.get('b', [])],
                'asks': [[float(a[0]), float(a[1])] for a in data.get('a', [])],
            }

        return {}

    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        params = {'accountType': 'UNIFIED'}
        result = await self._request('GET', '/v5/account/wallet-balance', params)

        if result.get('retCode') == 0:
            data = result.get('result', {})
            if data.get('list'):
                account = data['list'][0]
                return {
                    'USDT': float(account.get('totalEquity', 0)),
                    'available': float(account.get('totalAvailableBalance', 0)),
                    'margin_used': float(account.get('totalMarginBalance', 0)),
                }

        return {}

    def _map_symbol(self, instrument: str) -> str:
        """Map instrument to Bybit symbol."""
        # Standard format is already BTCUSDT
        if 'USDT' in instrument:
            return instrument
        return f"{instrument}USDT"

    def is_healthy(self) -> bool:
        """Check if executor is healthy."""
        if not self._healthy:
            return False
        if self._consecutive_errors > 3:
            return False
        if time.time() - self._last_success > 30:
            return False
        return True


def create_bybit_executor(
    api_key: str,
    api_secret: str,
    testnet: bool = True
) -> BybitExecutor:
    """
    Factory function for Bybit executor.

    Args:
        api_key: Bybit API key
        api_secret: Bybit API secret
        testnet: Use testnet (True) or mainnet (False)

    Returns:
        BybitExecutor ready for trading
    """
    config = BybitConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )
    return BybitExecutor(config)
