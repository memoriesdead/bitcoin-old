"""
CCXT Exchange Client
====================

Unified exchange connectivity using CCXT.

CCXT provides:
- 100+ exchange support
- Unified API
- Rate limiting
- Error handling
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

# Try to import ccxt
try:
    import ccxt
    import ccxt.async_support as ccxt_async
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    ccxt = None
    ccxt_async = None


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class ExchangeConfig:
    """Exchange configuration."""
    exchange_id: str
    api_key: str
    secret: str
    password: Optional[str] = None  # Some exchanges need this
    sandbox: bool = False
    rate_limit: bool = True
    timeout: int = 30000  # ms

    # Trading params
    default_type: str = "spot"  # spot, future, swap
    settle_coin: Optional[str] = None  # For futures


@dataclass
class OrderResult:
    """Result of order operation."""
    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    type: str = ""
    amount: float = 0.0
    price: Optional[float] = None
    filled: float = 0.0
    remaining: float = 0.0
    average: float = 0.0
    cost: float = 0.0
    fee: Optional[Dict] = None
    status: str = ""
    timestamp: float = 0.0
    error: Optional[str] = None
    raw: Optional[Dict] = None


class CCXTClient:
    """
    CCXT exchange client.

    Provides unified interface to exchanges via CCXT.
    """

    SUPPORTED_EXCHANGES = [
        "binance", "binanceus", "kraken", "coinbase",
        "bitstamp", "gemini", "bybit", "okx", "kucoin"
    ]

    def __init__(self, config: ExchangeConfig):
        """
        Initialize CCXT client.

        Args:
            config: Exchange configuration
        """
        self.config = config
        self.exchange = None
        self._async_exchange = None
        self._initialized = False

        if not HAS_CCXT:
            raise ImportError(
                "CCXT not installed. Run: pip install ccxt"
            )

        self._init_exchange()

    def _init_exchange(self):
        """Initialize exchange connection."""
        exchange_class = getattr(ccxt, self.config.exchange_id, None)

        if exchange_class is None:
            raise ValueError(f"Unknown exchange: {self.config.exchange_id}")

        options = {
            'apiKey': self.config.api_key,
            'secret': self.config.secret,
            'timeout': self.config.timeout,
            'enableRateLimit': self.config.rate_limit,
        }

        if self.config.password:
            options['password'] = self.config.password

        if self.config.sandbox:
            options['sandbox'] = True

        if self.config.default_type != "spot":
            options['defaultType'] = self.config.default_type

        self.exchange = exchange_class(options)
        self._initialized = True

    def load_markets(self) -> Dict[str, Any]:
        """Load market information."""
        return self.exchange.load_markets()

    def get_balance(self, asset: Optional[str] = None) -> Dict[str, Any]:
        """
        Get account balance.

        Args:
            asset: Specific asset (None = all)

        Returns:
            Balance dict
        """
        balance = self.exchange.fetch_balance()

        if asset:
            return {
                'free': balance.get(asset, {}).get('free', 0),
                'used': balance.get(asset, {}).get('used', 0),
                'total': balance.get(asset, {}).get('total', 0),
            }

        return balance

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker."""
        return self.exchange.fetch_ticker(symbol)

    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book."""
        return self.exchange.fetch_order_book(symbol, limit)

    def create_market_order(self, symbol: str, side: OrderSide,
                            amount: float) -> OrderResult:
        """
        Create market order.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            side: BUY or SELL
            amount: Order amount

        Returns:
            OrderResult
        """
        try:
            result = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side.value,
                amount=amount,
            )

            return self._parse_order_result(result)

        except ccxt.InsufficientFunds as e:
            return OrderResult(success=False, error=f"Insufficient funds: {e}")
        except ccxt.InvalidOrder as e:
            return OrderResult(success=False, error=f"Invalid order: {e}")
        except ccxt.NetworkError as e:
            return OrderResult(success=False, error=f"Network error: {e}")
        except Exception as e:
            return OrderResult(success=False, error=str(e))

    def create_limit_order(self, symbol: str, side: OrderSide,
                           amount: float, price: float) -> OrderResult:
        """
        Create limit order.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            amount: Order amount
            price: Limit price

        Returns:
            OrderResult
        """
        try:
            result = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side.value,
                amount=amount,
                price=price,
            )

            return self._parse_order_result(result)

        except ccxt.InsufficientFunds as e:
            return OrderResult(success=False, error=f"Insufficient funds: {e}")
        except ccxt.InvalidOrder as e:
            return OrderResult(success=False, error=f"Invalid order: {e}")
        except Exception as e:
            return OrderResult(success=False, error=str(e))

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair

        Returns:
            True if cancelled
        """
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception:
            return False

    def get_order(self, order_id: str, symbol: str) -> Optional[OrderResult]:
        """Get order status."""
        try:
            result = self.exchange.fetch_order(order_id, symbol)
            return self._parse_order_result(result)
        except Exception:
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Get all open orders."""
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return [self._parse_order_result(o) for o in orders]
        except Exception:
            return []

    def _parse_order_result(self, result: Dict) -> OrderResult:
        """Parse CCXT order result to OrderResult."""
        return OrderResult(
            success=True,
            order_id=result.get('id'),
            client_order_id=result.get('clientOrderId'),
            symbol=result.get('symbol', ''),
            side=result.get('side', ''),
            type=result.get('type', ''),
            amount=float(result.get('amount', 0)),
            price=float(result['price']) if result.get('price') else None,
            filled=float(result.get('filled', 0)),
            remaining=float(result.get('remaining', 0)),
            average=float(result.get('average', 0)) if result.get('average') else 0,
            cost=float(result.get('cost', 0)),
            fee=result.get('fee'),
            status=result.get('status', ''),
            timestamp=result.get('timestamp', 0) / 1000,  # ms to s
            raw=result,
        )

    def close(self):
        """Close exchange connection."""
        if self.exchange:
            self.exchange.close()


class AsyncCCXTClient:
    """
    Async CCXT client for high-frequency operations.
    """

    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.exchange = None

        if not HAS_CCXT:
            raise ImportError("CCXT not installed")

    async def connect(self):
        """Initialize async connection."""
        exchange_class = getattr(ccxt_async, self.config.exchange_id, None)

        if exchange_class is None:
            raise ValueError(f"Unknown exchange: {self.config.exchange_id}")

        options = {
            'apiKey': self.config.api_key,
            'secret': self.config.secret,
            'timeout': self.config.timeout,
            'enableRateLimit': self.config.rate_limit,
        }

        if self.config.password:
            options['password'] = self.config.password

        if self.config.sandbox:
            options['sandbox'] = True

        self.exchange = exchange_class(options)
        await self.exchange.load_markets()

    async def close(self):
        """Close async connection."""
        if self.exchange:
            await self.exchange.close()

    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker asynchronously."""
        return await self.exchange.fetch_ticker(symbol)

    async def create_market_order(self, symbol: str, side: str,
                                  amount: float) -> Dict:
        """Create market order asynchronously."""
        return await self.exchange.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=amount,
        )


def get_supported_exchanges() -> List[str]:
    """Get list of supported exchanges."""
    if not HAS_CCXT:
        return []
    return ccxt.exchanges


def create_client(exchange_id: str, api_key: str, secret: str,
                  sandbox: bool = False, **kwargs) -> CCXTClient:
    """
    Factory function to create exchange client.

    Args:
        exchange_id: Exchange name (binance, kraken, etc.)
        api_key: API key
        secret: API secret
        sandbox: Use sandbox/testnet
        **kwargs: Additional config

    Returns:
        Configured CCXTClient
    """
    config = ExchangeConfig(
        exchange_id=exchange_id,
        api_key=api_key,
        secret=secret,
        sandbox=sandbox,
        **kwargs
    )
    return CCXTClient(config)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("CCXT Client Demo")
    print("=" * 50)
    print(f"CCXT available: {HAS_CCXT}")

    if HAS_CCXT:
        print(f"Supported exchanges: {len(get_supported_exchanges())}")
        print(f"Sample: {get_supported_exchanges()[:10]}")
    else:
        print("Install CCXT: pip install ccxt")
