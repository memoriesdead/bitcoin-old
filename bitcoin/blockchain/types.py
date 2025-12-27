"""
Blockchain Data Types - ExchangeTick, FlowType, ExchangeDataFeed.
"""
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class FlowType(Enum):
    INFLOW = "inflow"   # TO exchange = selling
    OUTFLOW = "outflow" # FROM exchange = accumulating


@dataclass
class ExchangeTick:
    """Single tick - replaces WebSocket tick."""
    exchange: str
    timestamp: float
    price: float
    bid: float
    ask: float
    spread: float
    volume: float
    volume_1m: float
    volume_5m: float
    volume_1h: float
    buy_volume: float
    sell_volume: float
    direction: int      # +1 LONG, -1 SHORT, 0 NEUTRAL
    pressure: float
    tx_count: int
    source: str = "blockchain"
    latency_ms: float = 45000.0
    txid: str = ""
    flow_type: str = ""
    adaptive_hold: float = 30.0
    adaptive_sl: float = 0.002
    adaptive_tp: float = 0.004


@dataclass
class ExchangeDataFeed:
    """Real-time data feed for one exchange."""
    exchange_id: str
    exchange_name: str
    exchange_type: str
    txs_1m: deque = field(default_factory=lambda: deque(maxlen=1000))
    txs_5m: deque = field(default_factory=lambda: deque(maxlen=5000))
    txs_1h: deque = field(default_factory=lambda: deque(maxlen=30000))
    inflow_1m: float = 0.0
    inflow_5m: float = 0.0
    inflow_1h: float = 0.0
    outflow_1m: float = 0.0
    outflow_5m: float = 0.0
    outflow_1h: float = 0.0
    net_flow_1m: float = 0.0
    net_flow_5m: float = 0.0
    net_flow_1h: float = 0.0
    last_price: float = 0.0
    direction: int = 0
    pressure: float = 0.0
    tx_count_1h: int = 0
    last_update: float = 0.0
    ticks_emitted: int = 0
