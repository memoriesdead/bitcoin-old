"""
Per-Exchange Leverage Limits
============================

From official exchange documentation (December 2024).
All 3 strategies use maximum allowed leverage per exchange.
"""

# Per-exchange max leverage (from official docs Dec 2024)
EXCHANGE_LEVERAGE = {
    'mexc': 500,        # MEXC max 500x futures
    'binance': 125,     # Binance max 125x (20x new users)
    'bybit': 100,       # Bybit max 100x
    'kraken': 50,       # Kraken max 50x
    'coinbase': 10,     # Coinbase max 10x (US regulated)
    'gemini': 5,        # Gemini 5x US, 100x non-US
    'bitstamp': 10,     # Bitstamp 10x max
    'crypto.com': 20,   # Crypto.com max 20x
    'htx': 100,         # HTX (Huobi) max 100x
    'okx': 125,         # OKX max 125x
    'kucoin': 100,      # KuCoin max 100x
    'gate': 100,        # Gate.io max 100x
    'bitfinex': 10,     # Bitfinex max 10x
    'default': 10       # Conservative default
}


def get_leverage(exchange: str) -> int:
    """Get max leverage for exchange from official docs."""
    return EXCHANGE_LEVERAGE.get(exchange.lower(), EXCHANGE_LEVERAGE['default'])


def get_max_leverage_exchange() -> tuple:
    """Get exchange with maximum leverage.

    Returns:
        (exchange_name, leverage) tuple
    """
    max_ex = max(
        [(k, v) for k, v in EXCHANGE_LEVERAGE.items() if k != 'default'],
        key=lambda x: x[1]
    )
    return max_ex


# Quick reference table
LEVERAGE_TIERS = """
LEVERAGE TIERS (Official Docs Dec 2024)
========================================

Tier 1 - Max Leverage (500x):
  - MEXC: 500x futures

Tier 2 - High Leverage (100-125x):
  - Binance: 125x (20x new users)
  - OKX: 125x
  - Bybit: 100x
  - HTX: 100x
  - KuCoin: 100x
  - Gate.io: 100x

Tier 3 - Medium Leverage (20-50x):
  - Kraken: 50x
  - Crypto.com: 20x

Tier 4 - Low Leverage (5-10x):
  - Coinbase: 10x (US regulated)
  - Bitstamp: 10x
  - Bitfinex: 10x
  - Gemini: 5x (US), 100x (non-US)
"""
