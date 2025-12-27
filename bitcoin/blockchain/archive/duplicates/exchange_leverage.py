"""
Exchange Leverage Configuration
================================
Maximum leverage per exchange for BTC futures/perpetuals.
Based on research as of December 2025.

NO VPN/COUNTRY RESTRICTIONS - MAX LEVERAGE ONLY
"""

# Maximum leverage available per exchange
EXCHANGE_LEVERAGE = {
    # ============================================
    # TIER 1: HIGHEST LEVERAGE (125x - 500x)
    # ============================================
    'mexc': 500,           # MEXC - highest in industry
    'htx': 200,            # HTX (Huobi) - 200x on BTC/ETH
    'huobi': 200,          # Same as HTX
    'binance': 125,        # Binance Futures
    'bybit': 125,          # Bybit Derivatives
    'bitget': 125,         # Bitget Futures
    'gate.io': 125,        # Gate.io Futures
    'gateio': 125,         # Alias

    # ============================================
    # TIER 2: HIGH LEVERAGE (100x)
    # ============================================
    'okx': 100,            # OKX Futures
    'okcoin': 100,         # OKCoin
    'kucoin': 100,         # KuCoin Futures
    'bitfinex': 100,       # Bitfinex Derivatives
    'bitmex': 100,         # BitMEX - OG leverage exchange
    'poloniex': 100,       # Poloniex Futures

    # ============================================
    # TIER 3: MEDIUM LEVERAGE (50x)
    # ============================================
    'kraken': 50,          # Kraken Futures (US legal)
    'deribit': 50,         # Deribit Options/Futures
    'crypto.com': 50,      # Crypto.com
    'cryptocom': 50,       # Alias

    # ============================================
    # TIER 4: LOW LEVERAGE (10-20x)
    # ============================================
    'coinbase': 10,        # Coinbase (US regulated)
    'gemini': 10,          # Gemini (US regulated)
    'bitstamp': 5,         # Bitstamp - limited margin
    'cex': 10,             # CEX.io - margin only
    'hitbtc': 12,          # HitBTC
    'luno': 3,             # Luno - basic margin

    # ============================================
    # TIER 5: SPOT ONLY (No leverage/futures)
    # ============================================
    'bittrex': 0,          # Bittrex - spot only (winding down)
    'localbitcoins': 0,    # P2P only
    'paxful': 0,           # P2P only
    'coinspot.au': 0,      # Spot only
    'bitcoin.de': 0,       # Spot only
    'upbit': 0,            # Korean - spot only
    'mercadobitcoin.br': 0,  # Brazil - spot only
    'bitso': 0,            # Mexico - spot only

    # ============================================
    # DEFUNCT EXCHANGES (Historical data only)
    # ============================================
    'btc-e': 0,            # DEFUNCT - seized 2017
    'cryptsy': 0,          # DEFUNCT - exit scam 2016
    'mtgox': 0,            # DEFUNCT - hack 2014
    'quadrigacx': 0,       # DEFUNCT - fraud 2019
    'cavirtex': 0,         # DEFUNCT - 2015
    'vircurex': 0,         # DEFUNCT
    'coins-e': 0,          # DEFUNCT
    'bter': 0,             # DEFUNCT - became Gate.io
    'btcc': 0,             # DEFUNCT - China ban
    'chbtc': 0,            # DEFUNCT - became ZB
    'anxpro': 0,           # DEFUNCT
    'cointrader': 0,       # DEFUNCT
    'yobit': 5,            # Still running but sketchy
    'bleutrade': 0,        # DEFUNCT
    'c-cex': 0,            # DEFUNCT
}

# Default leverage for unknown exchanges
DEFAULT_LEVERAGE = 0

def get_max_leverage(exchange: str) -> int:
    """Get maximum leverage for an exchange."""
    return EXCHANGE_LEVERAGE.get(exchange.lower(), DEFAULT_LEVERAGE)

def get_active_leverage_exchanges() -> dict:
    """Get all exchanges with leverage > 0, sorted by leverage."""
    active = {k: v for k, v in EXCHANGE_LEVERAGE.items() if v > 0}
    return dict(sorted(active.items(), key=lambda x: x[1], reverse=True))

def get_tier1_exchanges() -> list:
    """Get exchanges with 125x+ leverage."""
    return [k for k, v in EXCHANGE_LEVERAGE.items() if v >= 125]

def get_tier2_exchanges() -> list:
    """Get exchanges with 100x leverage."""
    return [k for k, v in EXCHANGE_LEVERAGE.items() if v == 100]

# ============================================
# TRADING FEES (Maker/Taker) - CORRECT DECIMAL VALUES
# NOTE: 0.0004 = 0.04%, NOT 0.04 = 4%
# ============================================
EXCHANGE_FEES = {
    'mexc':      {'maker': 0.0000, 'taker': 0.0002},  # 0% / 0.02%
    'binance':   {'maker': 0.0002, 'taker': 0.0004},  # 0.02% / 0.04%
    'bybit':     {'maker': 0.0002, 'taker': 0.00055}, # 0.02% / 0.055%
    'bitget':    {'maker': 0.0002, 'taker': 0.0006},  # 0.02% / 0.06%
    'okx':       {'maker': 0.0002, 'taker': 0.0005},  # 0.02% / 0.05%
    'htx':       {'maker': 0.0002, 'taker': 0.0004},  # 0.02% / 0.04%
    'huobi':     {'maker': 0.0002, 'taker': 0.0004},  # 0.02% / 0.04%
    'gate.io':   {'maker': 0.00015, 'taker': 0.0005}, # 0.015% / 0.05%
    'gateio':    {'maker': 0.00015, 'taker': 0.0005}, # alias
    'kucoin':    {'maker': 0.0002, 'taker': 0.0006},  # 0.02% / 0.06%
    'kraken':    {'maker': 0.0002, 'taker': 0.0005},  # 0.02% / 0.05%
    'bitfinex':  {'maker': 0.0002, 'taker': 0.00065}, # 0.02% / 0.065%
    'bitmex':    {'maker': 0.0001, 'taker': 0.00025}, # 0.01% / 0.025%
    'deribit':   {'maker': 0.0000, 'taker': 0.0005},  # 0% / 0.05%
    'poloniex':  {'maker': 0.0002, 'taker': 0.0005},  # 0.02% / 0.05%
    'coinbase':  {'maker': 0.004, 'taker': 0.006},    # 0.4% / 0.6% (higher for spot)
    'gemini':    {'maker': 0.002, 'taker': 0.004},    # 0.2% / 0.4%
    'bitstamp':  {'maker': 0.003, 'taker': 0.005},    # 0.3% / 0.5%
    'crypto.com': {'maker': 0.0004, 'taker': 0.001},  # 0.04% / 0.1%
    'cryptocom': {'maker': 0.0004, 'taker': 0.001},   # alias
}

def get_fees(exchange: str) -> dict:
    """Get maker/taker fees for an exchange."""
    # Default: conservative 0.1% maker, 0.2% taker
    return EXCHANGE_FEES.get(exchange.lower(), {'maker': 0.001, 'taker': 0.002})


# ============================================
# RECOMMENDED SETTINGS BY STRATEGY
# ============================================
RECOMMENDED_CONFIG = {
    'conservative': {
        'exchanges': ['kraken', 'coinbase'],
        'max_leverage': 10,
        'position_pct': 0.10,  # 10% of capital
    },
    'moderate': {
        'exchanges': ['okx', 'bitget', 'kucoin'],
        'max_leverage': 50,
        'position_pct': 0.15,  # 15% of capital
    },
    'aggressive': {
        'exchanges': ['mexc', 'binance', 'bybit'],
        'max_leverage': 125,
        'position_pct': 0.20,  # 20% of capital
    },
    'max_leverage': {
        'exchanges': ['mexc'],
        'max_leverage': 500,
        'position_pct': 0.05,  # 5% of capital (high risk)
    },
}


if __name__ == '__main__':
    print("\n" + "="*70)
    print("EXCHANGE LEVERAGE CONFIG - ALL EXCHANGES")
    print("="*70)

    print("\nTIER 1 - HIGHEST LEVERAGE (125x+):")
    for ex in get_tier1_exchanges():
        print(f"  {ex:20} {EXCHANGE_LEVERAGE[ex]:>4}x")

    print("\nTIER 2 - HIGH LEVERAGE (100x):")
    for ex in get_tier2_exchanges():
        print(f"  {ex:20} {EXCHANGE_LEVERAGE[ex]:>4}x")

    print("\nALL ACTIVE EXCHANGES BY LEVERAGE:")
    for ex, lev in get_active_leverage_exchanges().items():
        fee = get_fees(ex)
        print(f"  {ex:20} {lev:>4}x  |  Maker: {fee['maker']:.2%}  Taker: {fee['taker']:.2%}")
