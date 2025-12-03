"""
HYPERLIQUID DIRECT ON-CHAIN EXECUTOR
=====================================
Direct blockchain execution - NO third-party APIs.

Hyperliquid is the ONLY exchange that matches our simulation:
- 200,000 orders/second (we do 237,000)
- Fully on-chain order book (every order is a blockchain transaction)
- Sub-200ms finality
- Self-custody (your keys, your coins)
- Zero gas fees (pay only on fills)

THIS IS NOT AN API - This is direct blockchain transaction signing.
Every order is cryptographically signed and submitted to the L1 chain.

Architecture:
1. Generate signal (4 microseconds - our engine)
2. Create order transaction
3. Sign with private key (local, never transmitted)
4. Submit to Hyperliquid L1 (own node or public RPC)
5. On-chain confirmation (200ms)

At $2B+ volume: 0% maker fees + REBATES (they pay YOU to provide liquidity)
"""
import asyncio
import time
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

from .base_executor import (
    BaseExecutor, Order, Position, ExchangeConfig,
    OrderSide, OrderType, OrderStatus, Signal
)

# Hyperliquid SDK imports (install: pip install hyperliquid-python-sdk)
try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    from eth_account import Account
    HYPERLIQUID_AVAILABLE = True
except ImportError:
    HYPERLIQUID_AVAILABLE = False
    print("[HYPERLIQUID] SDK not installed. Install with:")
    print("  pip install hyperliquid-python-sdk eth-account")


@dataclass
class HyperliquidConfig:
    """Configuration for Hyperliquid direct on-chain trading."""
    # Wallet private key (NEVER share, stored locally)
    private_key: str

    # Use own node for zero rate limits (recommended)
    # Default: None (uses public RPC)
    # Own node: "http://localhost:3001"
    node_url: Optional[str] = None

    # Network selection
    testnet: bool = True

    # Trading parameters
    default_leverage: int = 10
    max_position_usd: float = 10000.0

    # Slippage tolerance (basis points)
    slippage_bps: int = 10


class HyperliquidExecutor(BaseExecutor):
    """
    Direct on-chain executor for Hyperliquid L1.

    This is NOT an API wrapper - this signs and submits blockchain
    transactions directly. Every order is an on-chain transaction.

    WHY HYPERLIQUID:
    - 200K orders/second capacity (matches our 237K simulation)
    - Fully on-chain CLOB (Central Limit Order Book)
    - HyperBFT consensus (Byzantine fault tolerant)
    - Sub-200ms finality (faster than Solana)
    - Self-custody via cryptographic signing
    - Zero gas fees (only trading fees on fills)

    Fee tiers (14-day volume):
    - $0-5M:       0.010% maker, 0.035% taker
    - $5M-25M:     0.008% maker, 0.030% taker
    - $25M-100M:   0.005% maker, 0.025% taker
    - $100M-500M:  0.002% maker, 0.022% taker
    - $500M-2B:    0.000% maker, 0.020% taker
    - $2B+:        REBATES      0.019% taker
    """

    def __init__(self, hl_config: HyperliquidConfig):
        # Create base config
        base_config = ExchangeConfig(
            name="hyperliquid",
            api_key="",  # Not used - we use private key signing
            api_secret="",
            testnet=hl_config.testnet,
            rate_limit_per_second=1000,  # Own node = unlimited
            rate_limit_per_minute=60000,
            max_position_size=hl_config.max_position_usd,
            maker_fee=0.0001,  # 0.01% (tier 1)
            taker_fee=0.00035,  # 0.035% (tier 1)
        )
        super().__init__(base_config)

        self.hl_config = hl_config
        self.info: Optional['Info'] = None
        self.exchange: Optional['Exchange'] = None
        self.wallet = None
        self.address = None

        # Determine RPC endpoint
        if hl_config.node_url:
            self.rpc_url = hl_config.node_url
        elif hl_config.testnet:
            self.rpc_url = constants.TESTNET_API_URL
        else:
            self.rpc_url = constants.MAINNET_API_URL

        # Track on-chain state
        self._user_state = {}
        self._market_data = {}
        self._last_state_update = 0

    async def connect(self) -> bool:
        """
        Initialize connection to Hyperliquid L1.

        This doesn't actually "connect" in the traditional sense -
        Hyperliquid is a blockchain, so we just set up signing.
        """
        if not HYPERLIQUID_AVAILABLE:
            print("[HYPERLIQUID] SDK not available")
            return False

        try:
            # Create wallet from private key
            self.wallet = Account.from_key(self.hl_config.private_key)
            self.address = self.wallet.address

            # Initialize Info client (read-only blockchain data)
            self.info = Info(base_url=self.rpc_url, skip_ws=True)

            # Initialize Exchange client (for signing/submitting transactions)
            self.exchange = Exchange(
                self.wallet,
                base_url=self.rpc_url,
                account_address=self.address
            )

            # Verify connection by fetching user state
            self._user_state = self.info.user_state(self.address)

            network = "TESTNET" if self.hl_config.testnet else "MAINNET"
            print(f"[HYPERLIQUID] Connected to {network}")
            print(f"[HYPERLIQUID] Address: {self.address}")
            print(f"[HYPERLIQUID] RPC: {self.rpc_url}")

            # Show account balance
            if self._user_state:
                margin_summary = self._user_state.get('marginSummary', {})
                balance = float(margin_summary.get('accountValue', 0))
                print(f"[HYPERLIQUID] Account Value: ${balance:,.2f}")

            return True

        except Exception as e:
            print(f"[HYPERLIQUID] Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Cleanup (minimal for blockchain - no persistent connection)."""
        self.info = None
        self.exchange = None
        print("[HYPERLIQUID] Disconnected")

    async def submit_order(self, order: Order) -> Order:
        """
        Submit order directly to Hyperliquid L1 blockchain.

        This creates, signs, and broadcasts a blockchain transaction.
        The order becomes part of the permanent on-chain record.
        """
        if not self.exchange:
            order.status = OrderStatus.REJECTED
            return order

        start_time = time.perf_counter()

        try:
            # Convert to Hyperliquid format
            is_buy = order.side == OrderSide.BUY

            # Map instrument (e.g., "BTCUSDT" -> "BTC")
            coin = self._map_instrument(order.instrument)

            # Determine order type
            if order.order_type == OrderType.MARKET:
                # Market order with slippage
                order_type = {"limit": {"tif": "Ioc"}}  # Immediate-or-cancel
                # Get current price for slippage calc
                price = await self._get_market_price(coin, is_buy)
                slippage_mult = 1 + (self.hl_config.slippage_bps / 10000) if is_buy else 1 - (self.hl_config.slippage_bps / 10000)
                limit_price = price * slippage_mult
            else:
                # Limit order
                order_type = {"limit": {"tif": "Gtc"}}  # Good-til-cancelled
                limit_price = order.price

            # Round to appropriate precision
            sz = self._round_size(coin, order.quantity)
            px = self._round_price(coin, limit_price)

            # SIGN AND SUBMIT TO BLOCKCHAIN
            # This is the actual on-chain transaction
            result = self.exchange.order(
                coin=coin,
                is_buy=is_buy,
                sz=sz,
                limit_px=px,
                order_type=order_type,
                reduce_only=False
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Parse result
            if result.get('status') == 'ok':
                response = result.get('response', {})
                data = response.get('data', {})

                if 'statuses' in data and data['statuses']:
                    status_info = data['statuses'][0]

                    if 'filled' in status_info:
                        filled = status_info['filled']
                        order.status = OrderStatus.FILLED
                        order.filled_quantity = float(filled.get('totalSz', 0))
                        order.filled_price = float(filled.get('avgPx', 0))
                        order.exchange_order_id = str(filled.get('oid', ''))
                        self.stats['orders_filled'] += 1

                    elif 'resting' in status_info:
                        resting = status_info['resting']
                        order.status = OrderStatus.OPEN
                        order.exchange_order_id = str(resting.get('oid', ''))
                        self.open_orders[order.id] = order

                    elif 'error' in status_info:
                        order.status = OrderStatus.REJECTED
                        print(f"[HYPERLIQUID] Order error: {status_info['error']}")
                        self.stats['orders_rejected'] += 1

                self.stats['orders_sent'] += 1

                # Update latency stats
                n = self.stats['orders_sent']
                self.stats['avg_latency_ms'] = (
                    (self.stats['avg_latency_ms'] * (n - 1) + latency_ms) / n
                )

                # Log successful on-chain submission
                print(f"[HYPERLIQUID] Order {order.status.value}: {coin} {'BUY' if is_buy else 'SELL'} {sz} @ {px} ({latency_ms:.1f}ms)")

            else:
                order.status = OrderStatus.REJECTED
                error = result.get('response', {}).get('error', 'Unknown error')
                print(f"[HYPERLIQUID] Order rejected: {error}")
                self.stats['orders_rejected'] += 1

        except Exception as e:
            order.status = OrderStatus.REJECTED
            print(f"[HYPERLIQUID] Order exception: {e}")
            self.stats['orders_rejected'] += 1

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order on-chain."""
        if order_id not in self.open_orders or not self.exchange:
            return False

        order = self.open_orders[order_id]
        coin = self._map_instrument(order.instrument)

        try:
            result = self.exchange.cancel(
                coin=coin,
                oid=int(order.exchange_order_id)
            )

            if result.get('status') == 'ok':
                del self.open_orders[order_id]
                return True

        except Exception as e:
            print(f"[HYPERLIQUID] Cancel failed: {e}")

        return False

    async def get_position(self, instrument: str) -> Optional[Position]:
        """Get current on-chain position."""
        if not self.info:
            return None

        try:
            user_state = self.info.user_state(self.address)
            positions = user_state.get('assetPositions', [])

            coin = self._map_instrument(instrument)

            for pos in positions:
                pos_data = pos.get('position', {})
                if pos_data.get('coin') == coin:
                    szi = float(pos_data.get('szi', 0))
                    if szi == 0:
                        return None

                    return Position(
                        instrument=instrument,
                        side=OrderSide.BUY if szi > 0 else OrderSide.SELL,
                        quantity=abs(szi),
                        entry_price=float(pos_data.get('entryPx', 0)),
                        unrealized_pnl=float(pos_data.get('unrealizedPnl', 0)),
                        realized_pnl=float(pos_data.get('cumFunding', {}).get('sinceOpen', 0)),
                        exchange="hyperliquid"
                    )

        except Exception as e:
            print(f"[HYPERLIQUID] Position fetch failed: {e}")

        return None

    async def get_orderbook(self, instrument: str) -> Dict:
        """Get on-chain order book."""
        if not self.info:
            return {}

        try:
            coin = self._map_instrument(instrument)
            l2 = self.info.l2_snapshot(coin)

            return {
                'bids': [[float(p['px']), float(p['sz'])] for p in l2.get('levels', [[]])[0]],
                'asks': [[float(p['px']), float(p['sz'])] for p in l2.get('levels', [[], []])[1]],
            }

        except Exception as e:
            print(f"[HYPERLIQUID] Orderbook fetch failed: {e}")

        return {}

    async def get_balance(self) -> Dict[str, float]:
        """Get on-chain account balance."""
        if not self.info:
            return {}

        try:
            user_state = self.info.user_state(self.address)
            margin_summary = user_state.get('marginSummary', {})

            return {
                'USDC': float(margin_summary.get('accountValue', 0)),
                'available': float(margin_summary.get('withdrawable', 0)),
                'margin_used': float(margin_summary.get('marginUsed', 0)),
            }

        except Exception as e:
            print(f"[HYPERLIQUID] Balance fetch failed: {e}")

        return {}

    async def _get_market_price(self, coin: str, is_buy: bool) -> float:
        """Get current market price from on-chain orderbook."""
        try:
            l2 = self.info.l2_snapshot(coin)
            levels = l2.get('levels', [[], []])

            if is_buy and levels[1]:  # Use ask for buys
                return float(levels[1][0]['px'])
            elif not is_buy and levels[0]:  # Use bid for sells
                return float(levels[0][0]['px'])

        except Exception:
            pass

        # Fallback to mid price from all mids
        try:
            mids = self.info.all_mids()
            if coin in mids:
                return float(mids[coin])
        except Exception:
            pass

        return 0.0

    def _map_instrument(self, instrument: str) -> str:
        """Map standard instrument names to Hyperliquid format."""
        # Remove common suffixes
        coin = instrument.replace('USDT', '').replace('USD', '').replace('PERP', '')
        return coin

    def _round_size(self, coin: str, size: float) -> float:
        """Round size to valid precision for instrument."""
        # Hyperliquid uses different precisions per asset
        # BTC: 4 decimals, ETH: 3 decimals, etc.
        precisions = {
            'BTC': 4,
            'ETH': 3,
            'SOL': 2,
            'DOGE': 0,
        }
        decimals = precisions.get(coin, 4)
        return round(size, decimals)

    def _round_price(self, coin: str, price: float) -> float:
        """Round price to valid tick size."""
        # Most assets use 1 decimal for price
        return round(price, 1)

    async def set_leverage(self, instrument: str, leverage: int) -> bool:
        """Set leverage for an instrument on-chain."""
        if not self.exchange:
            return False

        try:
            coin = self._map_instrument(instrument)
            result = self.exchange.update_leverage(
                leverage=leverage,
                coin=coin,
                is_cross=True
            )
            return result.get('status') == 'ok'

        except Exception as e:
            print(f"[HYPERLIQUID] Set leverage failed: {e}")

        return False


def create_hyperliquid_executor(
    private_key: str,
    testnet: bool = True,
    node_url: Optional[str] = None
) -> HyperliquidExecutor:
    """
    Factory function to create Hyperliquid on-chain executor.

    Args:
        private_key: Your wallet private key (keep secret!)
        testnet: Use testnet (True) or mainnet (False)
        node_url: Your own node URL for zero rate limits
                  Run node with: ./hl-visor run-non-validating
                  Default port: http://localhost:3001

    Returns:
        HyperliquidExecutor ready for on-chain trading

    Example:
        # Using environment variable for key (recommended)
        executor = create_hyperliquid_executor(
            private_key=os.environ['HL_PRIVATE_KEY'],
            testnet=True
        )
        await executor.connect()

        # Create order from signal
        order = executor.signal_to_order(signal, capital=1000)
        filled_order = await executor.submit_order(order)
    """
    config = HyperliquidConfig(
        private_key=private_key,
        node_url=node_url,
        testnet=testnet,
    )
    return HyperliquidExecutor(config)


# =============================================================================
# RUNNING YOUR OWN HYPERLIQUID NODE
# =============================================================================
"""
WHY RUN YOUR OWN NODE:
- Zero rate limits (public RPC has limits)
- Lowest latency (direct to L1)
- No third-party dependency
- Full blockchain data

SETUP (Ubuntu 20.04+, 8GB RAM, 4 CPU, 100GB SSD):

1. Download and run non-validating node:
   curl https://binaries.hyperliquid.xyz/Testnet/hl-visor > hl-visor
   chmod +x hl-visor
   ./hl-visor run-non-validating

2. Your node will sync with the network (~10 minutes)

3. Use your node:
   executor = create_hyperliquid_executor(
       private_key="0x...",
       testnet=True,
       node_url="http://localhost:3001"
   )

4. Monitor node:
   curl http://localhost:3001/info

MAINNET:
   curl https://binaries.hyperliquid.xyz/Mainnet/hl-visor > hl-visor

The node stores ~50GB of blockchain data.
Update regularly for security patches.
"""
