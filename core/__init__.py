"""
RENAISSANCE TECHNOLOGIES-STYLE DATA INFRASTRUCTURE
===================================================
Real-time data capture with maximum speed and complete coverage.

Modules:
- exchange_feed: Multi-exchange WebSocket aggregator (10,000+ updates/sec)
- order_book: Real-time order book with TRUE OFI calculation
- bitcoin_zmq: Bitcoin Core ZMQ mempool integration
- unified_feed: Master data coordinator

Target Performance:
- <100μs tick-to-trade latency
- 100% data capture across all exchanges
- Real OFI (not derived from price - that's circular!)

Academic Basis:
- Cont, Kukanov & Stoikov (2014) - OFI predicts price with R²=70%
- Kyle (1985) - Price impact coefficient (lambda)
- Easley, Lopez de Prado & O'Hara (2012) - VPIN toxicity indicator
"""

from .exchange_feed import ExchangeFeed, ExchangeConfig, Exchange, OrderBookUpdate
from .order_book import OrderBook, OrderBookAggregator, OFISignal, KyleLambda, VPINValue
from .unified_feed import UnifiedFeed, UnifiedSignal

# Optional: Bitcoin Core ZMQ (requires pyzmq)
try:
    from .bitcoin_zmq import BitcoinZMQ, MempoolTx, MempoolStats
    BITCOIN_ZMQ_AVAILABLE = True
except ImportError:
    BITCOIN_ZMQ_AVAILABLE = False

__all__ = [
    # Exchange Feed
    'ExchangeFeed',
    'ExchangeConfig',
    'Exchange',
    'OrderBookUpdate',

    # Order Book
    'OrderBook',
    'OrderBookAggregator',
    'OFISignal',
    'KyleLambda',
    'VPINValue',

    # Unified Feed
    'UnifiedFeed',
    'UnifiedSignal',

    # Bitcoin ZMQ (optional)
    'BitcoinZMQ',
    'MempoolTx',
    'MempoolStats',
    'BITCOIN_ZMQ_AVAILABLE',
]
