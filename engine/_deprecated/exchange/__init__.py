"""
EXCHANGE EXECUTION LAYER - GOLD STANDARD MULTI-EXCHANGE
========================================================
Multi-chain on-chain trading with automatic failover.

PRIORITY ORDER (SELF-HOSTED NODES - NO RATE LIMITS):
1. Hyperliquid (on-chain, 200K/sec, HyperBFT consensus)
2. Monad (on-chain, 10K TPS, 400ms blocks, parallel EVM)
3. Sei Network (on-chain, 28K TPS, 400ms finality, trading L1)
4. Injective (on-chain, native CLOB, zero gas)
5. dYdX v4 (on-chain, Cosmos appchain)
6. Solana/Jito (on-chain, MEV bundles)

ARCHITECTURE:
┌───────────────────────────────────────────────────────────────────┐
│  Signal Engine (237K TPS) → Failover Orchestrator → 6 Blockchains│
└───────────────────────────────────────────────────────────────────┘

Components (GOLD STANDARD - LOCAL NODES):
- HyperliquidExecutor: Direct L1 execution (200K orders/sec)
- MonadExecutor: Fastest EVM chain (10K TPS, 400ms blocks)
- SeiExecutor: Trading-optimized L1 (28K TPS, 400ms finality)
- InjectiveExecutor: Native on-chain orderbook (zero gas)
- DydxExecutor: Cosmos appchain (2K TPS)
- SolanaJitoExecutor: MEV bundles

ON-CHAIN FIRST (RENAISSANCE APPROACH):
- ALL execution via self-hosted blockchain nodes
- Zero rate limits when running your own node
- Self-custody: Keys never leave your machine
- No third-party API dependency
- True decentralization with local infrastructure
"""
from .base_executor import (
    BaseExecutor,
    Signal,
    Order,
    Position,
    OrderSide,
    OrderType,
    OrderStatus,
    ExchangeConfig,
)
from .orchestrator import (
    MultiExchangeOrchestrator,
    ExchangeStats,
    InstrumentConfig,
)

# Lazy imports for executors (optional dependencies)
def get_hyperliquid_executor():
    """Get Hyperliquid executor (requires: pip install hyperliquid-python-sdk eth-account)"""
    from .hyperliquid_executor import HyperliquidExecutor, create_hyperliquid_executor, HyperliquidConfig
    return HyperliquidExecutor, create_hyperliquid_executor, HyperliquidConfig

def get_dydx_executor():
    """Get dYdX v4 executor (requires: pip install v4-client-py)"""
    from .dydx_executor import DydxExecutor, create_dydx_executor, DydxConfig
    return DydxExecutor, create_dydx_executor, DydxConfig

def get_bybit_executor():
    """Get Bybit executor (requires: pip install aiohttp)"""
    from .bybit_executor import BybitExecutor, create_bybit_executor, BybitConfig
    return BybitExecutor, create_bybit_executor, BybitConfig

def get_binance_executor():
    """Get Binance executor (requires: pip install aiohttp websockets)"""
    from .binance_executor import BinanceExecutor
    return BinanceExecutor

def get_solana_executor():
    """Get Solana/Jito executor (requires: pip install solana solders httpx)"""
    from .solana_jito_executor import SolanaJitoExecutor, create_solana_executor, SolanaConfig
    return SolanaJitoExecutor, create_solana_executor, SolanaConfig

def get_sei_executor():
    """Get Sei Network executor (requires: pip install aiohttp)

    Sei is a trading-optimized L1 with:
    - 400ms finality (Twin-Turbo Consensus)
    - 28K TPS capacity
    - Native order matching
    - Self-hosted node: seid
    """
    from .sei_executor import SeiExecutor, create_sei_executor, SeiConfig
    return SeiExecutor, create_sei_executor, SeiConfig

def get_injective_executor():
    """Get Injective Protocol executor (requires: pip install aiohttp)

    Injective is a derivatives L1 with:
    - Native on-chain CLOB
    - Zero gas fees for trading
    - Perpetuals, futures, options
    - Self-hosted node: injectived
    """
    from .injective_executor import InjectiveExecutor, create_injective_executor, InjectiveConfig
    return InjectiveExecutor, create_injective_executor, InjectiveConfig

def get_monad_executor():
    """Get Monad executor (requires: pip install web3 aiohttp)

    Monad is the fastest EVM chain with:
    - 10,000 TPS (proven in production)
    - 400ms block time
    - Full EVM compatibility
    - Self-hosted node: monad-node
    """
    from .monad_executor import MonadExecutor, create_monad_executor, MonadConfig
    return MonadExecutor, create_monad_executor, MonadConfig

def get_failover_orchestrator():
    """Get failover orchestrator for multi-exchange with auto-failover"""
    from .failover_orchestrator import (
        FailoverOrchestrator, create_failover_orchestrator,
        FailoverConfig, ExchangeHealth, ExchangeStatus
    )
    return FailoverOrchestrator, create_failover_orchestrator, FailoverConfig, ExchangeHealth, ExchangeStatus

def get_onchain_orchestrator():
    """Get on-chain orchestrator for multi-chain execution (legacy)"""
    from .onchain_orchestrator import OnChainOrchestrator, create_onchain_orchestrator, ChainPriority
    return OnChainOrchestrator, create_onchain_orchestrator, ChainPriority

__all__ = [
    # Base types
    'BaseExecutor',
    'Signal',
    'Order',
    'Position',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'ExchangeConfig',
    # Orchestrators
    'MultiExchangeOrchestrator',
    'ExchangeStats',
    'InstrumentConfig',
    # GOLD STANDARD: Self-hosted blockchain executors (no rate limits)
    'get_hyperliquid_executor',  # 200K TPS
    'get_monad_executor',         # 10K TPS, fastest EVM
    'get_sei_executor',           # 28K TPS, trading L1
    'get_injective_executor',     # Native CLOB, zero gas
    'get_dydx_executor',          # Cosmos appchain
    'get_solana_executor',        # MEV bundles
    # API-based executors (fallback)
    'get_bybit_executor',
    'get_binance_executor',
    # Orchestrators
    'get_failover_orchestrator',
    'get_onchain_orchestrator',
]
