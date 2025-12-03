"""
SETTLEMENT LAYER - BLOCKCHAIN HEDGING
======================================
Only hit the blockchain when absolutely necessary.

Internal matching = UNLIMITED SPEED
Settlement = Only when position thresholds are hit

This is how Renaissance/Virtu/Citadel operate:
- Trade internally at nanosecond speed
- Net positions over time
- Settle to external venues only when needed for:
  1. Position risk management
  2. Extracting profits
  3. Regulatory requirements
"""
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque


class SettlementStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    FAILED = "failed"


@dataclass
class PendingSettlement:
    """A settlement waiting to be executed on-chain."""
    settlement_id: int
    asset: str
    quantity: float
    side: str  # "buy" or "sell"
    target_price: float

    # Blockchain target
    target_chain: str  # hyperliquid, sei, solana, etc.

    # Status
    status: SettlementStatus = SettlementStatus.PENDING
    created_at: float = field(default_factory=time.time)
    submitted_at: Optional[float] = None
    confirmed_at: Optional[float] = None

    # Result
    execution_price: float = 0.0
    tx_hash: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SettlementConfig:
    """Configuration for settlement layer."""
    # Thresholds
    position_threshold_usd: float = 1000.0  # Settle when position exceeds this
    pnl_threshold_usd: float = 500.0  # Extract profits above this
    time_threshold_seconds: float = 3600.0  # Max time before forced settlement

    # Risk limits
    max_position_btc: float = 0.1  # Maximum BTC position before forced settle
    max_notional_usd: float = 10000.0

    # Chains priority (fastest first)
    chain_priority: List[str] = field(default_factory=lambda: [
        "hyperliquid",  # Off-chain matching, fastest
        "sei",  # 390ms finality
        "solana",  # 400ms slots
        "monad",  # Parallel EVM
    ])

    # Retry settings
    max_retries: int = 3
    retry_delay_ms: float = 100.0


class SettlementLayer:
    """
    Settlement layer for Sovereign Matching Engine.

    Handles blockchain settlement when thresholds are hit.
    Uses the 5 blockchain nodes as settlement venues.

    Strategy:
    1. Accumulate positions internally (unlimited speed)
    2. Net positions to minimize settlements
    3. Settle to fastest available chain
    4. Extract profits when threshold hit
    """

    def __init__(self, config: SettlementConfig = None):
        self.config = config or SettlementConfig()

        # Pending settlements
        self.pending: List[PendingSettlement] = []
        self.completed: deque = deque(maxlen=10000)
        self.settlement_counter: int = 0

        # Chain connections (initialized lazily)
        self._chain_connections: Dict[str, any] = {}

        # Stats
        self.total_settlements: int = 0
        self.successful_settlements: int = 0
        self.failed_settlements: int = 0
        self.total_settled_volume: float = 0.0

    def should_settle(
        self,
        positions: Dict[str, float],
        pnl: float,
        last_settlement_time: float,
    ) -> Tuple[bool, str]:
        """
        Determine if settlement is needed.

        Returns: (should_settle, reason)
        """
        now = time.time()

        # Check position thresholds
        for asset, quantity in positions.items():
            # Approximate USD value (assume $100k per BTC)
            approx_price = 100000.0 if asset == "BTC" else 1.0
            notional = abs(quantity) * approx_price

            if notional > self.config.position_threshold_usd:
                return True, f"position_threshold_{asset}"

            if abs(quantity) > self.config.max_position_btc:
                return True, f"max_position_{asset}"

        # Check PnL threshold
        if abs(pnl) > self.config.pnl_threshold_usd:
            return True, "pnl_extraction"

        # Check time threshold
        time_since = now - last_settlement_time
        if time_since > self.config.time_threshold_seconds:
            return True, "time_threshold"

        return False, "none"

    def create_settlement(
        self,
        asset: str,
        quantity: float,
        target_price: float,
    ) -> PendingSettlement:
        """
        Create a pending settlement.

        The settlement will be executed on the fastest available chain.
        """
        self.settlement_counter += 1

        settlement = PendingSettlement(
            settlement_id=self.settlement_counter,
            asset=asset,
            quantity=abs(quantity),
            side="sell" if quantity > 0 else "buy",
            target_price=target_price,
            target_chain=self._select_chain(asset),
        )

        self.pending.append(settlement)
        return settlement

    def _select_chain(self, asset: str) -> str:
        """
        Select best chain for settlement.

        Uses priority order from config.
        """
        # For now, default to hyperliquid (fastest, off-chain matching)
        for chain in self.config.chain_priority:
            # Check if chain supports asset and is connected
            if self._is_chain_available(chain, asset):
                return chain

        return self.config.chain_priority[0]

    def _is_chain_available(self, chain: str, asset: str) -> bool:
        """Check if chain is available for this asset."""
        # Asset support by chain
        chain_assets = {
            "hyperliquid": ["BTC", "ETH", "SOL", "ARB", "AVAX"],
            "sei": ["BTC", "ETH", "SEI"],
            "solana": ["BTC", "ETH", "SOL"],
            "monad": ["BTC", "ETH"],
        }

        supported = chain_assets.get(chain, [])
        return asset in supported

    async def execute_settlement(
        self,
        settlement: PendingSettlement,
    ) -> bool:
        """
        Execute a settlement on-chain.

        This is the ONLY time we touch the blockchain.
        Everything else is internal.
        """
        settlement.status = SettlementStatus.SUBMITTED
        settlement.submitted_at = time.time()

        try:
            # Get chain connection
            chain = self._get_chain_connection(settlement.target_chain)

            if chain is None:
                raise Exception(f"No connection to {settlement.target_chain}")

            # Execute based on chain type
            if settlement.target_chain == "hyperliquid":
                result = await self._execute_hyperliquid(settlement, chain)
            elif settlement.target_chain == "solana":
                result = await self._execute_solana(settlement, chain)
            elif settlement.target_chain == "sei":
                result = await self._execute_sei(settlement, chain)
            else:
                result = await self._execute_generic(settlement, chain)

            if result:
                settlement.status = SettlementStatus.CONFIRMED
                settlement.confirmed_at = time.time()
                self.successful_settlements += 1
                self.total_settled_volume += settlement.quantity * settlement.execution_price
            else:
                settlement.status = SettlementStatus.FAILED
                self.failed_settlements += 1

            self.total_settlements += 1
            self.completed.append(settlement)
            self.pending.remove(settlement)

            return result

        except Exception as e:
            settlement.status = SettlementStatus.FAILED
            settlement.error = str(e)
            self.failed_settlements += 1
            return False

    def _get_chain_connection(self, chain: str):
        """Get or create chain connection."""
        if chain in self._chain_connections:
            return self._chain_connections[chain]

        # Initialize connection based on chain
        try:
            if chain == "hyperliquid":
                from hyperliquid.info import Info
                from hyperliquid.utils import constants
                conn = Info(constants.MAINNET_API_URL, skip_ws=True)
                self._chain_connections[chain] = conn
                return conn

            # Other chains would be initialized here
            return None

        except ImportError:
            return None

    async def _execute_hyperliquid(
        self,
        settlement: PendingSettlement,
        chain,
    ) -> bool:
        """Execute settlement on Hyperliquid."""
        # This would use the Hyperliquid exchange API
        # For now, simulate execution
        try:
            # Get current price
            mids = chain.all_mids()
            if settlement.asset in mids:
                settlement.execution_price = float(mids[settlement.asset])
                settlement.tx_hash = f"HL-{settlement.settlement_id}-{int(time.time())}"
                return True
            return False
        except:
            return False

    async def _execute_solana(
        self,
        settlement: PendingSettlement,
        chain,
    ) -> bool:
        """Execute settlement on Solana."""
        # Would use Solana RPC for actual execution
        settlement.execution_price = settlement.target_price
        settlement.tx_hash = f"SOL-{settlement.settlement_id}-{int(time.time())}"
        return True

    async def _execute_sei(
        self,
        settlement: PendingSettlement,
        chain,
    ) -> bool:
        """Execute settlement on Sei."""
        settlement.execution_price = settlement.target_price
        settlement.tx_hash = f"SEI-{settlement.settlement_id}-{int(time.time())}"
        return True

    async def _execute_generic(
        self,
        settlement: PendingSettlement,
        chain,
    ) -> bool:
        """Generic settlement execution."""
        settlement.execution_price = settlement.target_price
        settlement.tx_hash = f"GEN-{settlement.settlement_id}-{int(time.time())}"
        return True

    async def process_pending(self) -> int:
        """Process all pending settlements. Returns count processed."""
        processed = 0

        for settlement in list(self.pending):
            if settlement.status == SettlementStatus.PENDING:
                success = await self.execute_settlement(settlement)
                if success:
                    processed += 1

        return processed

    def get_stats(self) -> Dict:
        """Get settlement statistics."""
        return {
            'total_settlements': self.total_settlements,
            'successful': self.successful_settlements,
            'failed': self.failed_settlements,
            'success_rate': (
                self.successful_settlements / self.total_settlements
                if self.total_settlements > 0 else 0
            ),
            'pending': len(self.pending),
            'total_volume': self.total_settled_volume,
        }

    def print_stats(self):
        """Print settlement summary."""
        stats = self.get_stats()

        print("\n" + "=" * 50)
        print("SETTLEMENT LAYER STATS")
        print("=" * 50)
        print(f"Total Settlements: {stats['total_settlements']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success Rate: {stats['success_rate']:.1%}")
        print(f"Pending: {stats['pending']}")
        print(f"Total Volume: ${stats['total_volume']:,.2f}")
        print("=" * 50)


__all__ = [
    'SettlementLayer',
    'SettlementConfig',
    'PendingSettlement',
    'SettlementStatus',
]
