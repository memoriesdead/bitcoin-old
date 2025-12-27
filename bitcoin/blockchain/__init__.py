"""
Blockchain Edge Module - C++ NANOSECOND LATENCY
================================================
Track BTC flowing TO and FROM exchanges with nanosecond latency.
8.6M+ exchange addresses across 102 exchanges.

ARCHITECTURE (C++ as base):
  C++ Blockchain Runner (nanosecond latency)
         │
         ├── Direct ZMQ to Bitcoin Core
         ├── 8.6M addresses in O(1) hash table
         ├── UTXO cache for outflow detection
         └── Sub-microsecond signal generation
         │
         ▼
  Python Signal Bridge
         │
         ├── Parse C++ signal output
         ├── Log to correlation database
         └── Forward to paper/live trader

INFLOW to exchange = SHORT (they will sell)
OUTFLOW from exchange = LONG (they are accumulating)

USAGE:
    # PRIMARY: C++ Master Pipeline (recommended)
    from engine.sovereign.blockchain import CppMasterPipeline
    pipeline = CppMasterPipeline()
    pipeline.run()

    # Or use the launcher script:
    # ./start_cpp_pipeline.sh --tmux

    # LEGACY: Python-only feed (slower)
    from engine.sovereign.blockchain import PerExchangeBlockchainFeed
    feed = PerExchangeBlockchainFeed(on_tick=my_callback)
    feed.start()

FILES:
    cpp_master_pipeline.py   - C++ runner wrapper (PRIMARY)
    start_cpp_pipeline.sh    - Launcher script
    ../cpp_runner/           - C++ blockchain runner source
    walletexplorer_addresses.db - 8.6M exchange addresses
"""

# Core infrastructure
from .zmq_subscriber import BlockchainZMQ
from .tx_decoder import TransactionDecoder
from .exchange_wallets import (
    ExchangeWalletTracker,
    ExchangeType,
    ExchangeInfo,
    EXCHANGE_SEEDS,
    EXCHANGE_DATABASE,
)

# Types
from .types import FlowType, ExchangeTick, ExchangeDataFeed

# Main data feed
from .per_exchange_feed import PerExchangeBlockchainFeed

# Simple flow detector
from .exchange_flow import SimpleExchangeFlowDetector, ExchangeFlow

# Formula connector (LINKS FEED -> FORMULAS)
from .formula_connector import FormulaConnector, create_connector

# Bitcoin Core RPC
from .rpc import BitcoinRPC, get_rpc_from_env

# Pattern-based flow detection (NO addresses needed!)
from .pattern_flow_detector import (
    PatternFlowDetector,
    HybridFlowDetector,
    FlowSignal,
    FlowPattern,
)

# Whale Alert scraper for address learning
from .whale_alert_scraper import WhaleAlertScraper, WhaleTransaction

# C++ Master Pipeline (PRIMARY - nanosecond latency)
from .cpp_master_pipeline import CppMasterPipeline, CppPipelineConfig, Signal

__all__ = [
    # C++ Master Pipeline (PRIMARY - nanosecond latency)
    'CppMasterPipeline',          # Main C++ runner wrapper (RECOMMENDED)
    'CppPipelineConfig',          # Configuration for C++ pipeline
    'Signal',                     # Trading signal from C++ runner

    # Legacy Python feed (slower, but still functional)
    'PerExchangeBlockchainFeed',  # Python feed (requires Bitcoin Core ZMQ)
    'ExchangeWalletTracker',      # Address lookup (just needs exchanges.json)
    'ExchangeTick',               # Tick data type
    'FlowType',                   # INFLOW/OUTFLOW enum

    # Formula Connector - THE MISSING LINK
    'FormulaConnector',           # Connects feed -> formulas -> signals
    'create_connector',           # Quick setup function

    # Pattern-based detection (NO ADDRESSES NEEDED!)
    'PatternFlowDetector',        # Pure pattern detection
    'HybridFlowDetector',         # Pattern + address matching
    'FlowSignal',                 # Signal from pattern detector
    'FlowPattern',                # Pattern types enum

    # Whale Alert learning
    'WhaleAlertScraper',          # Scrape Whale Alert for addresses
    'WhaleTransaction',           # Parsed whale transaction

    # Infrastructure
    'BlockchainZMQ',
    'TransactionDecoder',
    'ExchangeType',
    'ExchangeInfo',
    'EXCHANGE_SEEDS',
    'EXCHANGE_DATABASE',
    'ExchangeDataFeed',

    # Flow detector
    'SimpleExchangeFlowDetector',
    'ExchangeFlow',

    # RPC (for advanced usage)
    'BitcoinRPC',
    'get_rpc_from_env',
]
