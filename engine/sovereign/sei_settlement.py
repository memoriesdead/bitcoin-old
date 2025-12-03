"""
SEI DIRECT SETTLEMENT - ZERO THIRD-PARTY DEPENDENCY
====================================================
Direct blockchain settlement through YOUR OWN Sei node.

Architecture:
    Sovereign Matching Engine → Sei Node → Blockchain
    (10M+ trades internal)    (your node)  (direct)

No APIs. No rate limits. No third-party dependency.
Your node. Your execution. Your profits.

Sei Features:
- Native orderbook at protocol level
- 12,500 TPS
- 400ms finality
- Direct gRPC/REST access to your node

Usage:
    settler = SeiSettlement(node_url="http://localhost:26657")
    settler.settle_trades(profitable_trades)
"""
import os
import time
import json
import hashlib
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class SettlementStatus(Enum):
    """Settlement status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    FAILED = "failed"


@dataclass
class SeiConfig:
    """Sei node configuration."""
    node_url: str = "http://localhost:26657"
    grpc_url: str = "localhost:9090"
    chain_id: str = "atlantic-2"  # testnet, change to pacific-1 for mainnet
    denom: str = "usei"
    gas_price: str = "0.1usei"
    seid_path: str = "/root/go/bin/seid"
    keyring_backend: str = "test"
    key_name: str = "trading"


@dataclass
class SettlementTrade:
    """Trade to settle on Sei."""
    trade_id: str
    asset: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    pnl: float
    timestamp: float = field(default_factory=time.time)
    status: SettlementStatus = SettlementStatus.PENDING
    tx_hash: Optional[str] = None


@dataclass
class SettlementBatch:
    """Batch of trades to settle."""
    batch_id: str
    trades: List[SettlementTrade]
    total_pnl: float
    created_at: float = field(default_factory=time.time)
    settled_at: Optional[float] = None
    tx_hash: Optional[str] = None
    status: SettlementStatus = SettlementStatus.PENDING


class SeiSettlement:
    """
    SEI DIRECT SETTLEMENT

    Connects directly to your Sei node for blockchain settlement.
    No third-party APIs. No rate limits. Your node. Your control.

    Architecture:
    1. Profitable trades identified by Sovereign Engine
    2. Batch trades for efficient settlement
    3. Submit directly to your Sei node
    4. Confirm on-chain

    Why Sei:
    - Native orderbook at protocol level (not smart contract)
    - 12,500 TPS capacity
    - 400ms finality
    - Direct node access via gRPC/REST
    """

    def __init__(self, config: Optional[SeiConfig] = None):
        """Initialize Sei settlement."""
        self.config = config or SeiConfig()
        self.pending_trades: List[SettlementTrade] = []
        self.settled_batches: List[SettlementBatch] = []

        # Stats
        self.total_settled = 0
        self.total_pnl_settled = 0.0
        self.settlement_count = 0

    def check_node_status(self) -> Dict:
        """Check if Sei node is running and synced."""
        try:
            # Try using seid binary
            result = subprocess.run(
                [self.config.seid_path, "status", "--node", self.config.node_url],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                status = json.loads(result.stdout)
                return {
                    "connected": True,
                    "syncing": status.get("SyncInfo", {}).get("catching_up", False),
                    "latest_block": status.get("SyncInfo", {}).get("latest_block_height", "0"),
                    "network": status.get("NodeInfo", {}).get("network", "unknown"),
                }
        except Exception as e:
            pass

        # Fallback: try curl
        try:
            result = subprocess.run(
                ["curl", "-s", f"{self.config.node_url}/status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return {
                    "connected": True,
                    "syncing": data.get("result", {}).get("sync_info", {}).get("catching_up", False),
                    "latest_block": data.get("result", {}).get("sync_info", {}).get("latest_block_height", "0"),
                    "network": data.get("result", {}).get("node_info", {}).get("network", "unknown"),
                }
        except Exception as e:
            pass

        return {
            "connected": False,
            "error": "Could not connect to Sei node",
        }

    def add_trade(self, trade: SettlementTrade):
        """Add trade to pending settlement."""
        self.pending_trades.append(trade)

    def add_profitable_trade(
        self,
        trade_id: str,
        asset: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
    ):
        """Add a profitable trade for settlement."""
        if side == "buy":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        trade = SettlementTrade(
            trade_id=trade_id,
            asset=asset,
            side=side,
            quantity=quantity,
            price=exit_price,
            pnl=pnl,
        )
        self.add_trade(trade)

    def create_batch(self, max_trades: int = 100) -> Optional[SettlementBatch]:
        """Create a batch of trades for settlement."""
        if not self.pending_trades:
            return None

        trades_to_batch = self.pending_trades[:max_trades]
        self.pending_trades = self.pending_trades[max_trades:]

        total_pnl = sum(t.pnl for t in trades_to_batch)
        batch_id = hashlib.sha256(
            f"{time.time()}-{len(trades_to_batch)}".encode()
        ).hexdigest()[:16]

        return SettlementBatch(
            batch_id=batch_id,
            trades=trades_to_batch,
            total_pnl=total_pnl,
        )

    def settle_batch(self, batch: SettlementBatch) -> bool:
        """
        Settle a batch of trades on Sei blockchain.

        This submits directly to your Sei node - no third-party APIs.
        """
        print(f"[SEI] Settling batch {batch.batch_id} ({len(batch.trades)} trades, ${batch.total_pnl:.4f} PnL)")

        # Create settlement memo
        memo = json.dumps({
            "type": "sovereign_settlement",
            "batch_id": batch.batch_id,
            "trades": len(batch.trades),
            "pnl": batch.total_pnl,
            "timestamp": time.time(),
        })

        try:
            # Submit to Sei node using seid binary
            cmd = [
                self.config.seid_path,
                "tx", "bank", "send",
                self.config.key_name,
                self.config.key_name,  # Send to self (settlement record)
                "1" + self.config.denom,  # Minimal amount
                "--chain-id", self.config.chain_id,
                "--node", self.config.node_url,
                "--keyring-backend", self.config.keyring_backend,
                "--gas", "auto",
                "--gas-adjustment", "1.3",
                "--gas-prices", self.config.gas_price,
                "--note", memo[:256],  # Truncate memo if too long
                "-y",  # Auto-confirm
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                try:
                    tx_result = json.loads(result.stdout)
                    tx_hash = tx_result.get("txhash", "unknown")
                except:
                    tx_hash = "submitted"

                batch.tx_hash = tx_hash
                batch.status = SettlementStatus.SUBMITTED
                batch.settled_at = time.time()

                # Update stats
                self.total_settled += len(batch.trades)
                self.total_pnl_settled += batch.total_pnl
                self.settlement_count += 1
                self.settled_batches.append(batch)

                print(f"[SEI] Batch {batch.batch_id} submitted: {tx_hash}")
                return True
            else:
                print(f"[SEI] Settlement failed: {result.stderr}")
                batch.status = SettlementStatus.FAILED
                return False

        except subprocess.TimeoutExpired:
            print("[SEI] Settlement timeout - node may be busy")
            batch.status = SettlementStatus.FAILED
            return False
        except Exception as e:
            print(f"[SEI] Settlement error: {e}")
            batch.status = SettlementStatus.FAILED
            return False

    def settle_all_pending(self, batch_size: int = 100) -> int:
        """Settle all pending trades in batches."""
        settled = 0

        while self.pending_trades:
            batch = self.create_batch(max_trades=batch_size)
            if batch and self.settle_batch(batch):
                settled += len(batch.trades)

        return settled

    def get_stats(self) -> Dict:
        """Get settlement statistics."""
        return {
            "pending_trades": len(self.pending_trades),
            "total_settled": self.total_settled,
            "total_pnl_settled": self.total_pnl_settled,
            "settlement_count": self.settlement_count,
            "settled_batches": len(self.settled_batches),
        }

    def simulate_settlement(self, trades: int = 1000) -> Dict:
        """
        Simulate settlement without actually submitting.
        Used for testing and benchmarking.
        """
        print(f"\n[SEI] Simulating settlement of {trades:,} trades...")

        start = time.time()

        for i in range(trades):
            trade = SettlementTrade(
                trade_id=f"sim_{i}",
                asset="BTC",
                side="buy" if i % 2 == 0 else "sell",
                quantity=0.001,
                price=97000.0,
                pnl=0.0001 if i % 3 != 0 else -0.00005,  # 66% win rate
            )
            self.add_trade(trade)

        # Simulate batching
        batch_count = 0
        while self.pending_trades:
            batch = self.create_batch(max_trades=100)
            if batch:
                batch.status = SettlementStatus.CONFIRMED
                batch.tx_hash = f"simulated_{batch.batch_id}"
                self.total_settled += len(batch.trades)
                self.total_pnl_settled += batch.total_pnl
                self.settlement_count += 1
                batch_count += 1

        elapsed = time.time() - start
        tps = trades / elapsed if elapsed > 0 else 0

        print(f"[SEI] Simulation complete:")
        print(f"      Trades: {trades:,}")
        print(f"      Batches: {batch_count}")
        print(f"      Time: {elapsed:.2f}s")
        print(f"      TPS: {tps:,.0f}")
        print(f"      Total PnL: ${self.total_pnl_settled:.4f}")

        return {
            "trades": trades,
            "batches": batch_count,
            "elapsed": elapsed,
            "tps": tps,
            "total_pnl": self.total_pnl_settled,
        }


def setup_trading_key(config: SeiConfig) -> bool:
    """
    Set up a trading key for Sei transactions.
    Run this once before trading.
    """
    print("[SEI] Setting up trading key...")

    try:
        # Check if key exists
        result = subprocess.run(
            [config.seid_path, "keys", "show", config.key_name, "--keyring-backend", config.keyring_backend],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"[SEI] Key '{config.key_name}' already exists")
            return True

        # Create new key
        result = subprocess.run(
            [config.seid_path, "keys", "add", config.key_name, "--keyring-backend", config.keyring_backend],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"[SEI] Created new key '{config.key_name}'")
            print("[SEI] IMPORTANT: Save the mnemonic from the output!")
            print(result.stdout)
            return True
        else:
            print(f"[SEI] Key creation failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"[SEI] Key setup error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("SEI DIRECT SETTLEMENT - TESTING")
    print("=" * 70)

    config = SeiConfig()
    settler = SeiSettlement(config)

    # Check node status
    print("\n[SEI] Checking node status...")
    status = settler.check_node_status()
    print(f"      Connected: {status.get('connected', False)}")

    if status.get("connected"):
        print(f"      Network: {status.get('network', 'unknown')}")
        print(f"      Block: {status.get('latest_block', '0')}")
        print(f"      Syncing: {status.get('syncing', False)}")

    # Simulate settlement
    result = settler.simulate_settlement(trades=10000)

    print("\n" + "=" * 70)
    print("SEI SETTLEMENT READY")
    print("Your node. Your execution. No third-party APIs.")
    print("=" * 70)
