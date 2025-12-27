"""
Exchange Fee Structure
======================

Taker and maker fees from official exchange documentation.
Used by HQT for arbitrage profitability calculations.
"""

# Taker fees (we pay to take liquidity) - from exchange docs
TAKER_FEES = {
    'kraken': 0.0026,       # 0.26%
    'coinbase': 0.006,      # 0.60%
    'bitstamp': 0.005,      # 0.50%
    'gemini': 0.004,        # 0.40%
    'binance': 0.001,       # 0.10%
    'bybit': 0.001,         # 0.10%
    'mexc': 0.001,          # 0.10%
    'okx': 0.001,           # 0.10%
    'kucoin': 0.001,        # 0.10%
    'htx': 0.002,           # 0.20%
    'gate': 0.002,          # 0.20%
    'bitfinex': 0.002,      # 0.20%
    'crypto.com': 0.003,    # 0.30%
    'default': 0.005        # 0.50% conservative default
}

# Maker fees (we provide liquidity)
MAKER_FEES = {
    'kraken': 0.0016,       # 0.16%
    'coinbase': 0.004,      # 0.40%
    'bitstamp': 0.003,      # 0.30%
    'gemini': 0.002,        # 0.20%
    'binance': 0.001,       # 0.10%
    'bybit': 0.0002,        # 0.02%
    'mexc': 0.0,            # 0.00%
    'okx': 0.0008,          # 0.08%
    'kucoin': 0.001,        # 0.10%
    'htx': 0.002,           # 0.20%
    'gate': 0.002,          # 0.20%
    'bitfinex': 0.001,      # 0.10%
    'crypto.com': 0.001,    # 0.10%
    'default': 0.003        # 0.30% conservative default
}

# Default slippage per side
DEFAULT_SLIPPAGE = 0.0005  # 0.05% per side


def get_taker_fee(exchange: str) -> float:
    """Get taker fee for exchange."""
    return TAKER_FEES.get(exchange.lower(), TAKER_FEES['default'])


def get_maker_fee(exchange: str) -> float:
    """Get maker fee for exchange."""
    return MAKER_FEES.get(exchange.lower(), MAKER_FEES['default'])


def get_total_cost(buy_exchange: str, sell_exchange: str,
                   slippage_per_side: float = DEFAULT_SLIPPAGE) -> float:
    """
    Calculate total cost for arbitrage trade.

    Cost = buy_fee + sell_fee + slippage_both_sides

    Args:
        buy_exchange: Exchange to buy on
        sell_exchange: Exchange to sell on
        slippage_per_side: Expected slippage per side

    Returns:
        Total cost as decimal (e.g., 0.012 = 1.2%)
    """
    buy_fee = get_taker_fee(buy_exchange)
    sell_fee = get_taker_fee(sell_exchange)
    slippage = slippage_per_side * 2  # Both sides

    return buy_fee + sell_fee + slippage


def is_profitable(spread_pct: float, buy_exchange: str, sell_exchange: str) -> bool:
    """
    Check if spread is profitable after all costs.

    Deterministic: Returns True only if GUARANTEED profit.
    """
    total_cost = get_total_cost(buy_exchange, sell_exchange)
    return spread_pct > total_cost
