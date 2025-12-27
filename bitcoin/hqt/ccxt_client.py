#!/usr/bin/env python3
"""
CCXT Client - Unified Exchange Connectivity

Stripped-down CCXT wrapper for HFT:
- Market orders only (speed)
- Order book access (spread detection)
- Balance checking
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    ccxt = None


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderResult:
    """Result of order execution."""
    success: bool
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    amount: float = 0.0
    price: float = 0.0
    filled: float = 0.0
    cost: float = 0.0
    fee: float = 0.0
    status: str = ""
    timestamp: float = 0.0
    error: Optional[str] = None


class CCXTClient:
    """
    Minimal CCXT client for HFT arbitrage.

    Only implements what we need:
    - get_ticker: Current bid/ask
    - get_orderbook: Depth
    - get_balance: Check funds
    - create_market_order: Execute
    """

    def __init__(self, exchange_id: str, api_key: str = "", secret: str = "",
                 sandbox: bool = False, timeout: int = 10000):
        """
        Initialize CCXT client.

        Args:
            exchange_id: Exchange name (kraken, coinbase, etc.)
            api_key: API key (empty for public data only)
            secret: API secret
            sandbox: Use testnet
            timeout: Request timeout in ms
        """
        if not HAS_CCXT:
            raise ImportError("CCXT not installed. Run: pip install ccxt")

        self.exchange_id = exchange_id.lower()
        self.api_key = api_key
        self.secret = secret
        self.sandbox = sandbox
        self.timeout = timeout

        self.exchange = self._init_exchange()

    def _init_exchange(self):
        """Initialize exchange connection."""
        exchange_class = getattr(ccxt, self.exchange_id, None)

        if exchange_class is None:
            raise ValueError(f"Unknown exchange: {self.exchange_id}")

        options = {
            'timeout': self.timeout,
            'enableRateLimit': True,
        }

        if self.api_key:
            options['apiKey'] = self.api_key
            options['secret'] = self.secret

        if self.sandbox:
            options['sandbox'] = True

        return exchange_class(options)

    def get_ticker(self, symbol: str = 'BTC/USD') -> Dict[str, Any]:
        """
        Get current ticker with bid/ask.

        Returns:
            Dict with 'bid', 'ask', 'last', 'timestamp'
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'last': ticker.get('last', 0),
                'timestamp': ticker.get('timestamp', time.time() * 1000) / 1000,
                'volume': ticker.get('baseVolume', 0),
            }
        except Exception as e:
            return {'error': str(e)}

    def get_orderbook(self, symbol: str = 'BTC/USD', limit: int = 5) -> Dict[str, Any]:
        """
        Get order book with depth.

        Returns:
            Dict with 'bids' and 'asks' arrays
        """
        try:
            book = self.exchange.fetch_order_book(symbol, limit)
            return {
                'bids': book.get('bids', []),  # [[price, amount], ...]
                'asks': book.get('asks', []),
                'timestamp': book.get('timestamp', time.time() * 1000) / 1000,
            }
        except Exception as e:
            return {'error': str(e)}

    def get_balance(self, asset: str = 'USD') -> Dict[str, float]:
        """
        Get balance for asset.

        Returns:
            Dict with 'free', 'used', 'total'
        """
        if not self.api_key:
            return {'error': 'No API key configured'}

        try:
            balance = self.exchange.fetch_balance()
            asset_balance = balance.get(asset, {})
            return {
                'free': float(asset_balance.get('free', 0) or 0),
                'used': float(asset_balance.get('used', 0) or 0),
                'total': float(asset_balance.get('total', 0) or 0),
            }
        except Exception as e:
            return {'error': str(e)}

    def create_market_order(self, symbol: str, side: OrderSide,
                            amount: float) -> OrderResult:
        """
        Execute market order.

        Args:
            symbol: Trading pair (BTC/USD)
            side: BUY or SELL
            amount: Amount in base currency (BTC)

        Returns:
            OrderResult
        """
        if not self.api_key:
            return OrderResult(success=False, error='No API key configured')

        try:
            result = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side.value,
                amount=amount,
            )

            return OrderResult(
                success=True,
                order_id=result.get('id'),
                symbol=result.get('symbol', symbol),
                side=result.get('side', side.value),
                amount=float(result.get('amount', amount)),
                price=float(result.get('average', 0) or result.get('price', 0) or 0),
                filled=float(result.get('filled', 0) or 0),
                cost=float(result.get('cost', 0) or 0),
                fee=float(result.get('fee', {}).get('cost', 0) or 0),
                status=result.get('status', 'unknown'),
                timestamp=time.time(),
            )

        except ccxt.InsufficientFunds as e:
            return OrderResult(success=False, error=f"Insufficient funds: {e}")
        except ccxt.InvalidOrder as e:
            return OrderResult(success=False, error=f"Invalid order: {e}")
        except ccxt.NetworkError as e:
            return OrderResult(success=False, error=f"Network error: {e}")
        except Exception as e:
            return OrderResult(success=False, error=str(e))

    def close(self):
        """Close exchange connection."""
        if self.exchange:
            try:
                self.exchange.close()
            except:
                pass


def create_client(exchange_id: str, api_key: str = "", secret: str = "",
                  sandbox: bool = False) -> CCXTClient:
    """Factory function to create CCXT client."""
    return CCXTClient(exchange_id, api_key, secret, sandbox)


def list_exchanges() -> list:
    """Get list of supported exchanges."""
    if not HAS_CCXT:
        return []
    return ccxt.exchanges
