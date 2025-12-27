"""
Bitcoin Core Module
===================

Shared components for all trading strategies (DET, HQT, SCT).

Exports:
- Leverage: Per-exchange leverage limits
- Fees: Taker/maker fee structure
- Price Feed: Multi-exchange price feed (12 exchanges)
- Signals: Correlation formula and signal generation
- Trader: Position management and P&L tracking
"""

# Leverage
from bitcoin.core.leverage import (
    EXCHANGE_LEVERAGE,
    get_leverage,
    get_max_leverage_exchange,
)

# Fees
from bitcoin.core.fees import (
    TAKER_FEES,
    MAKER_FEES,
    get_taker_fee,
    get_maker_fee,
    get_total_cost,
    is_profitable,
)

# Price Feed
from bitcoin.core.multi_price_feed import (
    ExchangePrice,
    MultiExchangePriceFeed,
    get_exchange_price,
)

# Signals
from bitcoin.core.correlation_formula import (
    SignalType,
    CorrelationPattern,
    Signal,
    CorrelationFormula,
    format_signal,
)

# Trader
from bitcoin.core.deterministic_trader import (
    PositionStatus,
    Position,
    TraderStats,
    DeterministicTrader,
    format_position_open,
    format_position_close,
)

__all__ = [
    # Leverage
    'EXCHANGE_LEVERAGE',
    'get_leverage',
    'get_max_leverage_exchange',
    # Fees
    'TAKER_FEES',
    'MAKER_FEES',
    'get_taker_fee',
    'get_maker_fee',
    'get_total_cost',
    'is_profitable',
    # Price Feed
    'ExchangePrice',
    'MultiExchangePriceFeed',
    'get_exchange_price',
    # Signals
    'SignalType',
    'CorrelationPattern',
    'Signal',
    'CorrelationFormula',
    'format_signal',
    # Trader
    'PositionStatus',
    'Position',
    'TraderStats',
    'DeterministicTrader',
    'format_position_open',
    'format_position_close',
]
