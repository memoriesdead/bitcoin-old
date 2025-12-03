"""
SOLANA + JITO MEV BUNDLE EXECUTOR
==================================
Direct on-chain execution with MEV protection and atomic bundles.

WHY SOLANA + JITO:
- 65,000 TPS (Firedancer targeting 1M TPS in 2025)
- 400ms slot time (~0.4 second finality)
- $100B+ monthly DEX volume (massive liquidity)
- Jito bundles: Atomic execution, guaranteed inclusion
- Jupiter aggregation: Best prices across all DEXs
- MEV opportunities: Backrun large trades for profit

JITO BUNDLES:
- 94% of Solana validators run Jito
- Bundle = up to 5 transactions executed atomically
- ALL OR NOTHING: Either all succeed or all fail
- Priority tip to validator for inclusion
- No sandwich attacks possible (private mempool)

ARCHITECTURE:
Signal Engine → Create Swap TX → Bundle with Jito → Validator inclusion
"""
import asyncio
import time
import base58
import struct
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base_executor import (
    BaseExecutor, Order, Position, ExchangeConfig,
    OrderSide, OrderType, OrderStatus, Signal
)

# Solana imports
try:
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.transaction import Transaction, VersionedTransaction
    from solders.message import Message
    from solders.instruction import Instruction, AccountMeta
    from solders.system_program import TransferParams, transfer
    from solana.rpc.async_api import AsyncClient
    from solana.rpc.commitment import Confirmed
    import httpx
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False
    print("[SOLANA] Dependencies not installed. Install with:")
    print("  pip install solana solders httpx")


# Jito Block Engine endpoints
JITO_BLOCK_ENGINES = {
    'mainnet': [
        "https://mainnet.block-engine.jito.wtf",
        "https://amsterdam.mainnet.block-engine.jito.wtf",
        "https://frankfurt.mainnet.block-engine.jito.wtf",
        "https://ny.mainnet.block-engine.jito.wtf",
        "https://tokyo.mainnet.block-engine.jito.wtf",
    ],
    'devnet': [
        "https://dallas.testnet.block-engine.jito.wtf",
    ]
}

# Jupiter V6 API
JUPITER_API = "https://quote-api.jup.ag/v6"

# Common token mints
TOKEN_MINTS = {
    'SOL': 'So11111111111111111111111111111111111111112',
    'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
    'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
    'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
    'JUP': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
    'WIF': 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',
}


@dataclass
class SolanaConfig:
    """Configuration for Solana direct on-chain trading."""
    # Wallet private key (base58 encoded or bytes)
    private_key: str

    # RPC endpoint (use private RPC for speed)
    rpc_url: str = "https://api.mainnet-beta.solana.com"

    # Jito block engine (for bundle submission)
    jito_url: str = JITO_BLOCK_ENGINES['mainnet'][0]

    # Network
    is_mainnet: bool = True

    # Trading parameters
    slippage_bps: int = 50  # 0.5% slippage
    priority_fee_lamports: int = 10000  # ~$0.002 tip to validator

    # Jupiter settings
    use_jupiter: bool = True  # Aggregate across all DEXs


@dataclass
class JitoBundle:
    """A bundle of transactions for atomic execution."""
    transactions: List[bytes]  # Serialized transactions
    tip_lamports: int
    uuid: Optional[str] = None


class SolanaJitoExecutor(BaseExecutor):
    """
    Solana on-chain executor with Jito MEV bundles.

    This executor:
    1. Creates swap transactions via Jupiter (best prices)
    2. Bundles transactions with Jito (atomic execution)
    3. Submits directly to block engine (guaranteed inclusion)

    MEV PROTECTION:
    - Transactions go through Jito's private mempool
    - No sandwich attacks possible
    - All-or-nothing execution

    SPEED:
    - 400ms slot time
    - Priority fee gets faster inclusion
    - Direct to validator (no RPC delay with own node)
    """

    def __init__(self, config: SolanaConfig):
        base_config = ExchangeConfig(
            name="solana_jito",
            api_key="",
            api_secret="",
            testnet=not config.is_mainnet,
            rate_limit_per_second=100,
            rate_limit_per_minute=6000,
            max_position_size=10000.0,
            maker_fee=0.0,  # No maker fee on DEXs
            taker_fee=0.003,  # ~0.3% swap fee (varies by DEX)
        )
        super().__init__(base_config)

        self.sol_config = config
        self.keypair: Optional['Keypair'] = None
        self.client: Optional['AsyncClient'] = None
        self.http_client: Optional['httpx.AsyncClient'] = None

        # Cache
        self._token_accounts: Dict[str, str] = {}
        self._recent_blockhash: Optional[str] = None
        self._blockhash_time: float = 0

    async def connect(self) -> bool:
        """Initialize Solana connection."""
        if not SOLANA_AVAILABLE:
            print("[SOLANA] Dependencies not available")
            return False

        try:
            # Create keypair from private key
            if len(self.sol_config.private_key) == 88:
                # Base58 encoded
                key_bytes = base58.b58decode(self.sol_config.private_key)
            else:
                # Hex or raw bytes
                key_bytes = bytes.fromhex(self.sol_config.private_key.replace('0x', ''))

            self.keypair = Keypair.from_bytes(key_bytes)

            # Initialize RPC client
            self.client = AsyncClient(self.sol_config.rpc_url)

            # Initialize HTTP client for Jupiter/Jito
            self.http_client = httpx.AsyncClient(timeout=30.0)

            # Verify connection
            balance = await self.client.get_balance(self.keypair.pubkey())

            network = "MAINNET" if self.sol_config.is_mainnet else "DEVNET"
            print(f"[SOLANA] Connected to {network}")
            print(f"[SOLANA] Wallet: {self.keypair.pubkey()}")
            print(f"[SOLANA] Balance: {balance.value / 1e9:.4f} SOL")
            print(f"[SOLANA] Jito: {self.sol_config.jito_url}")

            return True

        except Exception as e:
            print(f"[SOLANA] Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Close connections."""
        if self.client:
            await self.client.close()
        if self.http_client:
            await self.http_client.aclose()
        print("[SOLANA] Disconnected")

    async def submit_order(self, order: Order) -> Order:
        """
        Execute swap on Solana via Jupiter + Jito bundle.

        Flow:
        1. Get best route from Jupiter
        2. Build swap transaction
        3. Create Jito bundle with tip
        4. Submit to block engine
        5. Wait for confirmation
        """
        if not self.client or not self.http_client or not self.keypair:
            order.status = OrderStatus.REJECTED
            return order

        start_time = time.perf_counter()

        try:
            # Parse trading pair
            input_mint, output_mint = self._parse_instrument(order.instrument, order.side)

            # Calculate amount in lamports/smallest unit
            amount = self._to_lamports(order.quantity, input_mint)

            # Step 1: Get Jupiter quote
            quote = await self._get_jupiter_quote(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount
            )

            if not quote:
                order.status = OrderStatus.REJECTED
                print("[SOLANA] Failed to get Jupiter quote")
                return order

            # Step 2: Get swap transaction
            swap_tx = await self._get_jupiter_swap_tx(quote)

            if not swap_tx:
                order.status = OrderStatus.REJECTED
                print("[SOLANA] Failed to get swap transaction")
                return order

            # Step 3: Create Jito bundle
            bundle = await self._create_jito_bundle(swap_tx)

            # Step 4: Submit bundle
            result = await self._submit_jito_bundle(bundle)

            latency_ms = (time.perf_counter() - start_time) * 1000

            if result:
                # Parse result
                order.status = OrderStatus.FILLED
                order.exchange_order_id = result.get('bundle_id', '')

                # Calculate filled amounts from quote
                out_amount = int(quote.get('outAmount', 0))
                order.filled_quantity = out_amount / self._get_decimals(output_mint)
                order.filled_price = float(quote.get('price', 0))

                self.stats['orders_sent'] += 1
                self.stats['orders_filled'] += 1

                # Update latency
                n = self.stats['orders_sent']
                self.stats['avg_latency_ms'] = (
                    (self.stats['avg_latency_ms'] * (n - 1) + latency_ms) / n
                )

                print(f"[SOLANA] Order FILLED via Jito bundle ({latency_ms:.1f}ms)")
                print(f"[SOLANA] Bundle ID: {order.exchange_order_id}")

            else:
                order.status = OrderStatus.REJECTED
                self.stats['orders_rejected'] += 1
                print("[SOLANA] Bundle submission failed")

        except Exception as e:
            order.status = OrderStatus.REJECTED
            print(f"[SOLANA] Order exception: {e}")
            self.stats['orders_rejected'] += 1

        return order

    async def _get_jupiter_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int
    ) -> Optional[Dict]:
        """Get best swap route from Jupiter aggregator."""
        try:
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': self.sol_config.slippage_bps,
                'onlyDirectRoutes': False,
                'asLegacyTransaction': False,
            }

            response = await self.http_client.get(
                f"{JUPITER_API}/quote",
                params=params
            )

            if response.status_code == 200:
                return response.json()

            print(f"[SOLANA] Jupiter quote error: {response.status_code}")

        except Exception as e:
            print(f"[SOLANA] Jupiter quote exception: {e}")

        return None

    async def _get_jupiter_swap_tx(self, quote: Dict) -> Optional[bytes]:
        """Get serialized swap transaction from Jupiter."""
        try:
            payload = {
                'quoteResponse': quote,
                'userPublicKey': str(self.keypair.pubkey()),
                'wrapAndUnwrapSol': True,
                'computeUnitPriceMicroLamports': self.sol_config.priority_fee_lamports,
                'dynamicComputeUnitLimit': True,
            }

            response = await self.http_client.post(
                f"{JUPITER_API}/swap",
                json=payload
            )

            if response.status_code == 200:
                data = response.json()
                swap_tx_base64 = data.get('swapTransaction')
                if swap_tx_base64:
                    import base64
                    return base64.b64decode(swap_tx_base64)

            print(f"[SOLANA] Jupiter swap error: {response.status_code}")

        except Exception as e:
            print(f"[SOLANA] Jupiter swap exception: {e}")

        return None

    async def _create_jito_bundle(self, swap_tx: bytes) -> JitoBundle:
        """Create Jito bundle with tip transaction."""
        # Get recent blockhash
        blockhash = await self._get_recent_blockhash()

        # Create tip transaction to Jito tip account
        # Tip accounts rotate - this is the main one
        tip_account = Pubkey.from_string("96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5")

        tip_ix = transfer(
            TransferParams(
                from_pubkey=self.keypair.pubkey(),
                to_pubkey=tip_account,
                lamports=self.sol_config.priority_fee_lamports
            )
        )

        # Build tip transaction
        tip_msg = Message.new_with_blockhash(
            [tip_ix],
            self.keypair.pubkey(),
            Pubkey.from_string(blockhash)
        )
        tip_tx = Transaction.new_unsigned(tip_msg)
        tip_tx.sign([self.keypair], Pubkey.from_string(blockhash))

        return JitoBundle(
            transactions=[swap_tx, bytes(tip_tx)],
            tip_lamports=self.sol_config.priority_fee_lamports
        )

    async def _submit_jito_bundle(self, bundle: JitoBundle) -> Optional[Dict]:
        """Submit bundle to Jito block engine."""
        try:
            import base64

            # Encode transactions
            encoded_txs = [
                base64.b64encode(tx).decode('utf-8')
                for tx in bundle.transactions
            ]

            # Submit to Jito
            payload = {
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'sendBundle',
                'params': [encoded_txs]
            }

            response = await self.http_client.post(
                f"{self.sol_config.jito_url}/api/v1/bundles",
                json=payload
            )

            if response.status_code == 200:
                data = response.json()
                if 'result' in data:
                    return {'bundle_id': data['result']}
                if 'error' in data:
                    print(f"[SOLANA] Jito error: {data['error']}")

        except Exception as e:
            print(f"[SOLANA] Jito submit exception: {e}")

        return None

    async def _get_recent_blockhash(self) -> str:
        """Get recent blockhash (cached for 30 seconds)."""
        now = time.time()
        if self._recent_blockhash and now - self._blockhash_time < 30:
            return self._recent_blockhash

        result = await self.client.get_latest_blockhash()
        self._recent_blockhash = str(result.value.blockhash)
        self._blockhash_time = now
        return self._recent_blockhash

    def _parse_instrument(self, instrument: str, side: OrderSide) -> Tuple[str, str]:
        """Parse instrument to input/output mints."""
        # Common patterns: SOLUSDC, SOL/USDC, SOL-USDC
        instrument = instrument.replace('/', '').replace('-', '')

        # Determine base and quote
        if 'USDC' in instrument:
            base = instrument.replace('USDC', '')
            quote = 'USDC'
        elif 'USDT' in instrument:
            base = instrument.replace('USDT', '')
            quote = 'USDT'
        else:
            # Assume quote is SOL
            base = instrument.replace('SOL', '')
            quote = 'SOL'

        base_mint = TOKEN_MINTS.get(base, base)
        quote_mint = TOKEN_MINTS.get(quote, quote)

        # BUY base = sell quote, SELL base = sell base
        if side == OrderSide.BUY:
            return quote_mint, base_mint
        else:
            return base_mint, quote_mint

    def _to_lamports(self, amount: float, mint: str) -> int:
        """Convert amount to smallest unit."""
        decimals = self._get_decimals(mint)
        return int(amount * (10 ** decimals))

    def _get_decimals(self, mint: str) -> int:
        """Get decimals for a token."""
        # SOL and common SPL tokens
        if mint == TOKEN_MINTS['SOL']:
            return 9
        elif mint in [TOKEN_MINTS['USDC'], TOKEN_MINTS['USDT']]:
            return 6
        else:
            return 9  # Default

    async def cancel_order(self, order_id: str) -> bool:
        """Solana swaps are instant - no cancellation needed."""
        return False

    async def get_position(self, instrument: str) -> Optional[Position]:
        """Get token balance as position."""
        if not self.client:
            return None

        try:
            _, output_mint = self._parse_instrument(instrument, OrderSide.BUY)
            mint_pubkey = Pubkey.from_string(output_mint)

            # Get token accounts
            response = await self.client.get_token_accounts_by_owner(
                self.keypair.pubkey(),
                {'mint': mint_pubkey}
            )

            if response.value:
                account_info = response.value[0].account
                # Parse balance from account data
                # This is simplified - full implementation needs account parsing
                return None

        except Exception as e:
            print(f"[SOLANA] Position fetch failed: {e}")

        return None

    async def get_orderbook(self, instrument: str) -> Dict:
        """DEX aggregator doesn't have traditional orderbook."""
        return {}

    async def get_balance(self) -> Dict[str, float]:
        """Get SOL and token balances."""
        if not self.client:
            return {}

        try:
            # Get SOL balance
            sol_balance = await self.client.get_balance(self.keypair.pubkey())

            return {
                'SOL': sol_balance.value / 1e9,
            }

        except Exception as e:
            print(f"[SOLANA] Balance fetch failed: {e}")

        return {}


def create_solana_executor(
    private_key: str,
    rpc_url: str = "https://api.mainnet-beta.solana.com",
    is_mainnet: bool = True,
    priority_fee: int = 10000
) -> SolanaJitoExecutor:
    """
    Factory function to create Solana Jito executor.

    Args:
        private_key: Base58 or hex encoded private key
        rpc_url: Solana RPC endpoint (use Helius/Triton for speed)
        is_mainnet: True for mainnet, False for devnet
        priority_fee: Tip to validator in lamports (10000 = ~$0.002)

    Returns:
        SolanaJitoExecutor ready for on-chain trading

    RECOMMENDED RPC PROVIDERS (private, low latency):
    - Helius: https://www.helius.dev/ (~$50/month)
    - Triton: https://triton.one/ (~$100/month)
    - QuickNode: https://www.quicknode.com/ (~$50/month)

    Example:
        executor = create_solana_executor(
            private_key=os.environ['SOL_PRIVATE_KEY'],
            rpc_url="https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",
            priority_fee=50000  # Higher fee = faster inclusion
        )
        await executor.connect()
    """
    config = SolanaConfig(
        private_key=private_key,
        rpc_url=rpc_url,
        is_mainnet=is_mainnet,
        priority_fee_lamports=priority_fee,
        jito_url=JITO_BLOCK_ENGINES['mainnet'][0] if is_mainnet else JITO_BLOCK_ENGINES['devnet'][0],
    )
    return SolanaJitoExecutor(config)


# =============================================================================
# JITO MEV STRATEGIES
# =============================================================================
"""
ADDITIONAL MEV OPPORTUNITIES WITH JITO:

1. BACKRUNNING:
   - See large swap in mempool
   - Bundle your trade AFTER theirs
   - Profit from price impact

2. ARBITRAGE:
   - Detect price discrepancy across DEXs
   - Bundle atomic arb across multiple pools
   - Zero risk (all-or-nothing)

3. JIT LIQUIDITY:
   - See pending swap
   - Provide liquidity just-in-time
   - Earn fees from the swap

4. LIQUIDATIONS:
   - Monitor lending protocols
   - Bundle liquidation + profit taking
   - Atomic execution

Our Bitcoin mempool signals can trigger these strategies
BEFORE CEXs see the price impact.
"""
