"""
DYDX V4 ON-CHAIN EXECUTOR - COSMOS APPCHAIN
============================================
dYdX v4 is a fully decentralized exchange on its own Cosmos appchain.

Why dYdX v4 for backup:
- Fully on-chain like Hyperliquid (no custody risk)
- ~2 second block finality (Cosmos consensus)
- No KYC, no rate limits for on-chain
- Deep liquidity on majors
- Self-custody via cryptographic signing

Architecture:
- dYdX Chain: Custom Cosmos SDK appchain
- Validators: Process order matching on-chain
- Block time: ~1-2 seconds
- Throughput: ~2000 TPS

Fee Structure:
- Maker: -0.011% to +0.02% (REBATES possible)
- Taker: 0.02% to 0.05%
- No gas fees (validators subsidized)

Install: pip install v4-client-py
"""
import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    from dydx_v4_client import NodeClient, Wallet, Network
    from dydx_v4_client.indexer.rest import IndexerClient
    from dydx_v4_client.node.message import Order as DydxOrder
    DYDX_AVAILABLE = True
except ImportError:
    DYDX_AVAILABLE = False

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
class DydxConfig:
    """dYdX v4 configuration.

    LOCAL NODE MODE:
    When node_url is provided, connects to your own dYdX node:
    - node_url: Local RPC (e.g., "http://localhost:26657")
    - grpc_url: Local gRPC (e.g., "localhost:9090")
    - No rate limits when running own node
    - Zero latency to node
    """
    mnemonic: str  # 24-word recovery phrase
    testnet: bool = True
    subaccount_number: int = 0
    # Local node configuration (overrides external APIs)
    node_url: str = ""  # e.g., "http://localhost:26657"
    grpc_url: str = ""  # e.g., "localhost:9090"
    indexer_url: str = ""  # e.g., "http://localhost:3002" (if running local indexer)


class DydxExecutor(BaseExecutor):
    """
    dYdX v4 on-chain executor.

    ON-CHAIN BACKUP:
    - Alternative to Hyperliquid for decentralized execution
    - Cosmos-based (different infrastructure than Hyperliquid)
    - No rate limits, no custody
    """

    # API endpoints
    MAINNET_INDEXER = "https://indexer.dydx.trade"
    TESTNET_INDEXER = "https://indexer.v4testnet.dydx.exchange"
    MAINNET_VALIDATOR = "https://dydx-ops-rpc.kingnodes.com"
    TESTNET_VALIDATOR = "https://test-dydx.kingnodes.com"

    def __init__(self, dydx_config: DydxConfig):
        base_config = ExchangeConfig(
            name="dydx",
            api_key="",  # On-chain, uses mnemonic
            api_secret="",
            testnet=dydx_config.testnet,
            rate_limit_per_second=100,  # On-chain, no real limit
            rate_limit_per_minute=6000,
            max_position_size=10000.0,
            maker_fee=0.0002,  # 0.02%
            taker_fee=0.0005,  # 0.05%
        )
        super().__init__(base_config)

        self.dydx_config = dydx_config
        self.wallet = None
        self.node_client = None
        self.indexer = None
        self.address = None
        self.use_local_node = bool(dydx_config.node_url)

        # Network selection - LOCAL NODE takes priority
        if dydx_config.node_url:
            # LOCAL NODE MODE - no rate limits!
            self.node_url = dydx_config.node_url
            self.grpc_url = dydx_config.grpc_url or "localhost:9090"
            self.indexer_url = dydx_config.indexer_url or dydx_config.node_url
            self.chain_id = "dydx-testnet-4" if dydx_config.testnet else "dydx-mainnet-1"
            print(f"[DYDX] LOCAL NODE MODE: {self.node_url}")
        elif dydx_config.testnet:
            self.indexer_url = self.TESTNET_INDEXER
            self.validator_url = self.TESTNET_VALIDATOR
            self.chain_id = "dydx-testnet-4"
        else:
            self.indexer_url = self.MAINNET_INDEXER
            self.validator_url = self.MAINNET_VALIDATOR
            self.chain_id = "dydx-mainnet-1"

        # Health tracking
        self._consecutive_errors = 0
        self._last_success = 0
        self._healthy = True

    async def connect(self) -> bool:
        """Connect to dYdX v4 chain.

        LOCAL NODE MODE:
        When node_url is configured, connects directly to your local node:
        - RPC at http://localhost:26657
        - gRPC at localhost:9090
        - No rate limits, zero network latency
        """
        # Try local node connection first
        if self.use_local_node:
            return await self._connect_local_node()

        if not DYDX_AVAILABLE:
            # Fallback to REST API if SDK not available
            print("[DYDX] SDK not installed, using REST fallback")
            return await self._connect_rest()

        try:
            # Create wallet from mnemonic
            network = Network.testnet() if self.dydx_config.testnet else Network.mainnet()
            self.wallet = Wallet.from_mnemonic(
                mnemonic=self.dydx_config.mnemonic,
                account_id=0
            )
            self.address = self.wallet.address

            # Connect to node
            self.node_client = await NodeClient.connect(network.node)

            # Connect to indexer
            self.indexer = IndexerClient(network.indexer_rest)

            network_name = "TESTNET" if self.dydx_config.testnet else "MAINNET"
            print(f"[DYDX] Connected to {network_name}")
            print(f"[DYDX] Address: {self.address}")

            # Get balance
            balance = await self._get_balance_internal()
            print(f"[DYDX] Balance: ${balance.get('USDC', 0):,.2f}")

            self._healthy = True
            self._last_success = time.time()
            return True

        except Exception as e:
            print(f"[DYDX] Connection error: {e}")
            return await self._connect_rest()

    async def _connect_local_node(self) -> bool:
        """Connect to LOCAL dYdX node (no rate limits)."""
        if not AIOHTTP_AVAILABLE:
            print("[DYDX] aiohttp not available for local node")
            return False

        try:
            # Check if local node is responding
            async with aiohttp.ClientSession() as session:
                # Cosmos RPC status endpoint
                url = f"{self.node_url}/status"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        node_info = data.get('result', {}).get('node_info', {})
                        sync_info = data.get('result', {}).get('sync_info', {})

                        network = node_info.get('network', 'unknown')
                        latest_height = sync_info.get('latest_block_height', '0')
                        catching_up = sync_info.get('catching_up', True)

                        print(f"[DYDX] LOCAL NODE connected: {self.node_url}")
                        print(f"[DYDX] Network: {network}, Height: {latest_height}")

                        if catching_up:
                            print("[DYDX] WARNING: Node still syncing")
                        else:
                            print("[DYDX] Node fully synced - READY FOR HFT")

                        self._healthy = not catching_up
                        self._last_success = time.time()
                        return True

            print(f"[DYDX] Local node not responding at {self.node_url}")
            return False

        except Exception as e:
            print(f"[DYDX] Local node connection error: {e}")
            return False

    async def _connect_rest(self) -> bool:
        """Fallback REST connection."""
        if not AIOHTTP_AVAILABLE:
            print("[DYDX] No connection method available")
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.indexer_url}/v4/time") as resp:
                    if resp.status == 200:
                        network_name = "TESTNET" if self.dydx_config.testnet else "MAINNET"
                        print(f"[DYDX] REST fallback connected to {network_name}")
                        self._healthy = True
                        return True

        except Exception as e:
            print(f"[DYDX] REST fallback failed: {e}")

        return False

    async def disconnect(self) -> None:
        """Disconnect from dYdX."""
        self.node_client = None
        self.indexer = None
        print("[DYDX] Disconnected")

    async def submit_order(self, order: Order) -> Order:
        """Submit order to dYdX chain."""
        start_time = time.perf_counter()

        if not DYDX_AVAILABLE or not self.node_client:
            order.status = OrderStatus.REJECTED
            return order

        try:
            # Map to dYdX format
            market = self._map_market(order.instrument)
            side = DydxOrder.Side.BUY if order.side == OrderSide.BUY else DydxOrder.Side.SELL

            # Get current market info for tick size
            market_info = await self._get_market_info(market)

            # Create order
            order_params = {
                'market': market,
                'side': side,
                'type': DydxOrder.Type.MARKET if order.order_type == OrderType.MARKET else DydxOrder.Type.LIMIT,
                'size': str(order.quantity),
                'post_only': False,
                'reduce_only': False,
            }

            if order.order_type == OrderType.LIMIT and order.price:
                order_params['price'] = str(order.price)

            # Sign and submit to chain
            tx = await self.node_client.place_order(
                wallet=self.wallet,
                subaccount=self.dydx_config.subaccount_number,
                **order_params
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            if tx and tx.code == 0:
                order.exchange_order_id = tx.txhash
                order.status = OrderStatus.OPEN  # Will be filled on-chain

                self.stats['orders_sent'] += 1
                self._consecutive_errors = 0
                self._last_success = time.time()

                # Update latency
                n = self.stats['orders_sent']
                self.stats['avg_latency_ms'] = (self.stats['avg_latency_ms'] * (n-1) + latency_ms) / n

                print(f"[DYDX] Order submitted: {market} {side} {order.quantity} ({latency_ms:.1f}ms)")

                # For market orders, check fill
                if order.order_type == OrderType.MARKET:
                    await asyncio.sleep(2)  # Wait for block
                    await self._check_order_fill(order)
            else:
                order.status = OrderStatus.REJECTED
                self.stats['orders_rejected'] += 1
                self._consecutive_errors += 1
                print(f"[DYDX] Order rejected: {tx.raw_log if tx else 'Unknown error'}")

        except Exception as e:
            order.status = OrderStatus.REJECTED
            self.stats['orders_rejected'] += 1
            self._consecutive_errors += 1
            print(f"[DYDX] Order error: {e}")

        return order

    async def _check_order_fill(self, order: Order) -> None:
        """Check if order was filled on-chain."""
        if not self.indexer:
            return

        try:
            fills = await self.indexer.account.get_subaccount_fills(
                self.address,
                self.dydx_config.subaccount_number
            )

            # Find our fill
            for fill in fills.get('fills', []):
                if fill.get('orderId') == order.exchange_order_id:
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = float(fill.get('size', 0))
                    order.filled_price = float(fill.get('price', 0))
                    self.stats['orders_filled'] += 1
                    break

        except Exception:
            pass

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on dYdX chain."""
        if not self.node_client or order_id not in self.open_orders:
            return False

        order = self.open_orders[order_id]

        try:
            tx = await self.node_client.cancel_order(
                wallet=self.wallet,
                subaccount=self.dydx_config.subaccount_number,
                order_id=order.exchange_order_id,
            )

            if tx and tx.code == 0:
                del self.open_orders[order_id]
                return True

        except Exception as e:
            print(f"[DYDX] Cancel error: {e}")

        return False

    async def get_position(self, instrument: str) -> Optional[Position]:
        """Get position from dYdX."""
        if not self.indexer:
            return None

        try:
            market = self._map_market(instrument)
            positions = await self.indexer.account.get_subaccount_perpetual_positions(
                self.address,
                self.dydx_config.subaccount_number
            )

            for pos in positions.get('positions', []):
                if pos.get('market') == market:
                    size = float(pos.get('size', 0))
                    if abs(size) > 0:
                        return Position(
                            instrument=instrument,
                            side=OrderSide.BUY if size > 0 else OrderSide.SELL,
                            quantity=abs(size),
                            entry_price=float(pos.get('entryPrice', 0)),
                            unrealized_pnl=float(pos.get('unrealizedPnl', 0)),
                            realized_pnl=float(pos.get('realizedPnl', 0)),
                            exchange="dydx"
                        )

        except Exception as e:
            print(f"[DYDX] Position error: {e}")

        return None

    async def get_orderbook(self, instrument: str) -> Dict:
        """Get orderbook from dYdX indexer."""
        if not AIOHTTP_AVAILABLE:
            return {}

        try:
            market = self._map_market(instrument)

            async with aiohttp.ClientSession() as session:
                url = f"{self.indexer_url}/v4/orderbooks/perpetualMarket/{market}"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            'bids': [[float(b['price']), float(b['size'])] for b in data.get('bids', [])],
                            'asks': [[float(a['price']), float(a['size'])] for a in data.get('asks', [])],
                        }

        except Exception as e:
            print(f"[DYDX] Orderbook error: {e}")

        return {}

    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        return await self._get_balance_internal()

    async def _get_balance_internal(self) -> Dict[str, float]:
        """Internal balance fetch."""
        if not AIOHTTP_AVAILABLE:
            return {}

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.indexer_url}/v4/addresses/{self.address}/subaccountNumber/{self.dydx_config.subaccount_number}"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        subaccount = data.get('subaccount', {})
                        return {
                            'USDC': float(subaccount.get('equity', 0)),
                            'available': float(subaccount.get('freeCollateral', 0)),
                            'margin_used': float(subaccount.get('marginEnabled', 0)),
                        }

        except Exception:
            pass

        return {}

    async def _get_market_info(self, market: str) -> Dict:
        """Get market info for sizing."""
        if not AIOHTTP_AVAILABLE:
            return {}

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.indexer_url}/v4/perpetualMarkets"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for m in data.get('markets', {}).values():
                            if m.get('ticker') == market:
                                return m

        except Exception:
            pass

        return {}

    def _map_market(self, instrument: str) -> str:
        """Map instrument to dYdX market format."""
        # dYdX uses BTC-USD format
        base = instrument.replace('USDT', '').replace('USD', '').replace('PERP', '')
        return f"{base}-USD"

    def is_healthy(self) -> bool:
        """Check if executor is healthy."""
        if not self._healthy:
            return False
        if self._consecutive_errors > 3:
            return False
        if time.time() - self._last_success > 30:
            return False
        return True


def create_dydx_executor(
    mnemonic: str,
    testnet: bool = True,
    node_url: str = "",
    grpc_url: str = "",
) -> DydxExecutor:
    """
    Factory function for dYdX v4 executor.

    Args:
        mnemonic: 24-word recovery phrase
        testnet: Use testnet (True) or mainnet (False)
        node_url: LOCAL node RPC URL (e.g., "http://localhost:26657")
        grpc_url: LOCAL node gRPC URL (e.g., "localhost:9090")

    Returns:
        DydxExecutor ready for on-chain trading

    Example (External API):
        executor = create_dydx_executor(
            mnemonic=os.environ['DYDX_MNEMONIC'],
            testnet=True
        )
        await executor.connect()

    Example (LOCAL NODE - no rate limits!):
        executor = create_dydx_executor(
            mnemonic=os.environ['DYDX_MNEMONIC'],
            testnet=False,
            node_url="http://localhost:26657",
            grpc_url="localhost:9090"
        )
        await executor.connect()
    """
    config = DydxConfig(
        mnemonic=mnemonic,
        testnet=testnet,
        node_url=node_url,
        grpc_url=grpc_url,
    )
    return DydxExecutor(config)
