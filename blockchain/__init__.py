"""
BLOCKCHAIN DATA PIPELINE - NO EXCHANGE APIs!
=============================================
Pure Bitcoin blockchain data - replaces all third-party exchange APIs.

THE EDGE: Everyone else uses the same exchange APIs (Binance, Bybit, etc.)
         We use pure blockchain math for unique signals.

Components:
    BlockchainUnifiedFeed  - Drop-in replacement for core.UnifiedFeed (NO APIs!)
    BlockchainSignal       - Trading signal from blockchain math
    PureMempoolMath        - Mempool simulation from pure math
    PureBlockchainPrice    - Power Law fair value (R² > 95%)
    BlockchainTradingEngine - Trading signals from Power Law valuation
"""

# CORE: Drop-in replacement for exchange API feeds
from .unified_feed import BlockchainUnifiedFeed, BlockchainSignal, UnifiedFeed, UnifiedSignal
from .mempool_math import PureMempoolMath, MempoolSignals, get_mempool_signals, get_mempool_price_delta

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
