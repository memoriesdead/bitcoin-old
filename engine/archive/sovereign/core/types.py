"""
Core Type Definitions - Unified Sovereign Engine Types
=======================================================

All shared types for the Sovereign Engine:
- Tick: Market data event
- Signal: Trading signal from formulas
- Order/SizedOrder: Order for execution
- ExecutionResult: Trade execution outcome
- TradeSignal, Position, TradeResult: Legacy compatible types
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import time


# =============================================================================
# ENUMS
# =============================================================================

class ExecutionMode(Enum):
    """Trading execution modes."""
    PAPER = "paper"           # Simulated fills, no real orders
    DRY_RUN = "dry_run"       # Real prices, simulated fills
    LIVE = "live"             # Real CCXT orders
    ONCHAIN = "onchain"       # On-chain DEX execution
    BACKTEST = "backtest"     # Historical replay


class DataSource(Enum):
    """Data source types."""
    BLOCKCHAIN = "blockchain"  # ZMQ from Bitcoin Core
    HISTORICAL = "historical"  # SQLite historical data
    LIVE = "live"             # WebSocket real-time
    SIMULATED = "simulated"   # Generated data for testing


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


# =============================================================================
# UNIFIED TYPES
# =============================================================================

@dataclass
class Tick:
    """
    Unified market data tick.

    Single type for all data sources: blockchain, historical, live.
    """
    timestamp: float
    source: str                    # "blockchain", "historical", "live"
    symbol: str = "BTC/USDT"

    # Price data
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0

    # Volume
    volume: float = 0.0

    # Blockchain-specific (optional)
    exchange_id: Optional[str] = None
    flow_direction: Optional[str] = None  # "inflow" or "outflow"
    flow_amount: float = 0.0
    tx_hash: Optional[str] = None

    # Features (computed)
    features: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class Signal:
    """
    Unified trading signal from formula engines.

    Replaces SignalAggregation with cleaner interface.
    """
    timestamp: float
    direction: int                 # +1 LONG, -1 SHORT, 0 HOLD
    confidence: float              # 0.0 to 1.0

    # Source info
    source_engine: str = ""        # "adaptive", "pattern", "rentech"
    formula_ids: List[int] = field(default_factory=list)

    # Execution hints
    suggested_size: float = 0.0    # Suggested position size (0-1)
    stop_loss: float = 0.0         # Suggested SL %
    take_profit: float = 0.0       # Suggested TP %
    hold_seconds: float = 60.0     # Suggested hold time

    # Context
    regime: str = "unknown"
    price_at_signal: float = 0.0

    # Ensemble voting
    votes: Dict[str, int] = field(default_factory=dict)
    vote_confidences: Dict[str, float] = field(default_factory=dict)

    @property
    def tradeable(self) -> bool:
        """Signal is tradeable if direction != 0 and confidence > 0.5."""
        return self.direction != 0 and self.confidence > 0.5

    @property
    def strength(self) -> float:
        """Combined strength = direction * confidence."""
        return self.direction * self.confidence


@dataclass
class Order:
    """Order to be executed."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float                  # BTC amount
    price: Optional[float] = None  # Limit price (None for market)

    # Metadata
    signal: Optional[Signal] = None
    timestamp: float = 0.0
    client_order_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class SizedOrder:
    """Order with position sizing applied."""
    order: Order
    position_size_pct: float       # Position size as % of capital
    position_size_usd: float       # Position size in USD
    kelly_fraction: float = 0.0    # Kelly criterion fraction
    rl_adjustment: float = 1.0     # RL-based adjustment multiplier

    @property
    def symbol(self) -> str:
        return self.order.symbol

    @property
    def side(self) -> OrderSide:
        return self.order.side

    @property
    def amount(self) -> float:
        return self.order.amount


@dataclass
class ExecutionResult:
    """Result from executing an order."""
    success: bool
    order: Order

    # Fill info
    fill_price: float = 0.0
    fill_amount: float = 0.0
    fill_time: float = 0.0

    # Costs
    fee: float = 0.0
    slippage: float = 0.0

    # Status
    rejected: bool = False
    reject_reason: Optional[str] = None
    order_id: Optional[str] = None
    tx_hash: Optional[str] = None  # For on-chain

    # PnL (set after position close)
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_price: float = 0.0
    exit_time: float = 0.0


@dataclass
class TradeOutcome:
    """
    Complete trade outcome for learning feedback.

    Contains everything needed to update formula parameters.
    """
    signal: Signal
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    pnl: float
    pnl_pct: float

    # Context
    hold_duration: float = 0.0
    exit_reason: str = ""          # "tp", "sl", "time", "signal"
    regime_at_entry: str = ""
    regime_at_exit: str = ""

    @property
    def was_profitable(self) -> bool:
        return self.pnl > 0


# =============================================================================
# LEGACY TYPES (Backwards Compatible)
# =============================================================================


@dataclass
class TradeSignal:
    """Complete trade signal."""
    timestamp: float
    direction: int              # +1 LONG, -1 SHORT, 0 HOLD
    probability: float
    position_size: float
    price: float
    regime: str = "unknown"
    strategy: str = "none"
    regime_confidence: float = 0.5
    signal_confidence: float = 0.5
    gate_passed: bool = False
    exchange_id: str = ""
    adaptive_sl: float = 0.005
    adaptive_tp: float = 0.01
    adaptive_hold: float = 60


@dataclass
class LeadingSignal:
    """Blockchain-derived signals (IDs 901-903)."""
    timestamp: float
    power_law_price: float
    power_law_deviation: float
    power_law_signal: int
    s2f_price: float
    s2f_deviation: float
    s2f_signal: int
    s2f_ratio: float
    halving_position: float
    halving_signal: int
    fee_pressure: float
    tx_momentum: float
    mempool_ofi: float
    mempool_ofi_signal: int
    combined_strength: float
    final_signal: int
    confidence: float


@dataclass
class ExchangeFlow:
    """BTC flow to/from exchange."""
    exchange: str
    direction: str
    amount_btc: float
    rate_per_second: float
    expected_rate: float
    multiplier: float
    timestamp: float


@dataclass
class Position:
    """Current position state."""
    direction: int
    size: float
    entry_price: float
    entry_time: float
    highest_since_entry: float
    lowest_since_entry: float

    def unrealized_pnl(self, current_price: float) -> float:
        if self.direction == 0 or self.size == 0:
            return 0.0
        if self.direction == 1:
            return self.size * (current_price - self.entry_price)
        return self.size * (self.entry_price - current_price)


@dataclass
class TradeResult:
    """Executed trade result."""
    exchange: str
    trade_type: str
    price: float
    size: float
    value: float
    probability: float
    timestamp: float
    pnl: float = 0.0
    exit_reason: Optional[str] = None


@dataclass
class EngineStats:
    """Engine performance stats."""
    exchange: str
    capital: float
    initial_capital: float
    pnl: float
    pnl_pct: float
    trades_opened: int
    trades_closed: int
    wins: int
    losses: int
    win_rate: float
    exits_sl: int
    exits_tp: int
    exits_time: int
    exits_reversal: int


@dataclass
class SignalAggregation:
    """Aggregated signal from multiple sources."""
    direction: int
    confidence: float
    should_trade: bool
    strength: float
    source: str
    leading_signal: int = 0
    leading_confidence: float = 0.0
    blockchain_signal: int = 0
    blockchain_confidence: float = 0.0
