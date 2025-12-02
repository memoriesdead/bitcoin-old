"""
================================================================================
BLOCKCHAIN DATA PIPELINE - PURE MATH, NO EXCHANGE APIs!
================================================================================

THE COMPETITIVE EDGE:
    - Everyone else uses the same exchange APIs (Binance, Bybit, OKX)
    - We derive ALL signals from pure blockchain math
    - Zero latency (math, not network calls)
    - Unique signals (not same data as everyone else)

PIPELINE ARCHITECTURE:
    See: docs/BLOCKCHAIN_PIPELINE_ARCHITECTURE.md

================================================================================
LAYER 1: UNIFIED FEED (Drop-in API Replacement)
================================================================================
    BlockchainUnifiedFeed  - Drop-in replacement for core.UnifiedFeed
    BlockchainSignal       - Trading signal compatible with UnifiedSignal

================================================================================
LAYER 2: PURE MATH COMPONENTS
================================================================================
    PureMempoolMath        - Mempool simulation from pure blockchain math
                            Sources: Block timing, Halving cycles, Difficulty
    PureBlockchainPrice    - Power Law fair value (R² = 93%+)
                            Formula: Price = 10^(A + B * log10(days))
    BlockchainTradingEngine - Trading signals from Power Law deviation

================================================================================
LAYER 3: DATA SOURCES (Legacy, still functional)
================================================================================
    BlockchainFeed         - Base blockchain data feed
    BlockchainMarketData   - Market data from blockchain
    BlockchainPriceEngine  - Real-time price derivation

FORMULA IDs:
    901: Power Law Price Signal (LEADING - timestamp only)
    902: Stock-to-Flow Signal (LEADING - timestamp only)
    903: Halving Cycle Position (LEADING - timestamp only)
    801-804: Blockchain signals (block volatility, mempool, chaos, whale)
    520-560: Academic microstructure signals

================================================================================
"""

# CORE: Drop-in replacement for exchange API feeds
from .unified_feed import BlockchainUnifiedFeed, BlockchainSignal, UnifiedFeed, UnifiedSignal
from .mempool_math import PureMempoolMath, MempoolSignals, get_mempool_signals, get_mempool_price_delta
from .price_generator import BlockchainPriceGenerator, calc_blockchain_price, generate_price_ticks

# Legacy components (still useful)
from .blockchain_feed import BlockchainFeed
from .blockchain_market_data import BlockchainMarketData, BlockchainMarketState, MarketSignal
from .blockchain_price_engine import RealTimeBlockchainPricer, BlockchainPriceEngine, DerivedPrice
from .pure_blockchain_price import PureBlockchainPrice, PowerLawPrice, get_blockchain_price, get_blockchain_analysis
from .blockchain_trading_signal import BlockchainTradingEngine, BlockchainTradingSignal, get_trading_signal

__all__ = [
    # PRIMARY: Drop-in replacement for exchange APIs
    "BlockchainUnifiedFeed",
    "BlockchainSignal",
    "UnifiedFeed",      # Alias for compatibility
    "UnifiedSignal",    # Alias for compatibility

    # Blockchain price generation (FASTER THAN APIs)
    "BlockchainPriceGenerator",
    "calc_blockchain_price",
    "generate_price_ticks",

    # Mempool simulation
    "PureMempoolMath",
    "MempoolSignals",
    "get_mempool_signals",
    "get_mempool_price_delta",

    # Legacy components
    "BlockchainFeed",
    "BlockchainMarketData",
    "BlockchainMarketState",
    "MarketSignal",
    "RealTimeBlockchainPricer",
    "BlockchainPriceEngine",
    "DerivedPrice",
    "PureBlockchainPrice",
    "PowerLawPrice",
    "get_blockchain_price",
    "get_blockchain_analysis",
    "BlockchainTradingEngine",
    "BlockchainTradingSignal",
    "get_trading_signal",
]
