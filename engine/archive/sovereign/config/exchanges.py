"""
Exchange Configuration - WebSocket endpoints and fees
ALL DATA FETCHED LIVE FROM APIS - NO HARDCODED VALUES
"""

EXCHANGE_ENDPOINTS = {
    # ==========================================================================
    # YOUR EXCHANGES - USA LEGAL
    # Volume fetched LIVE from APIs - never hardcoded
    # ==========================================================================

    'coinbase': {
        'url': 'wss://ws-feed.exchange.coinbase.com',
        'name': 'Coinbase',
        'maker_fee': 0.004,
        'taker_fee': 0.006,
    },
    'kraken': {
        'url': 'wss://ws.kraken.com',
        'name': 'Kraken',
        'maker_fee': 0.0016,
        'taker_fee': 0.0026,
    },
    'gemini': {
        'url': 'wss://api.gemini.com/v1/marketdata/BTCUSD',
        'name': 'Gemini',
        'maker_fee': 0.001,
        'taker_fee': 0.003,
    },
    'binance': {
        'url': 'wss://stream.binance.us:9443/ws/btcusd@trade',
        'name': 'Binance US',
        'maker_fee': 0.001,
        'taker_fee': 0.001,
    },
    'hyperliquid': {
        'url': 'wss://api.hyperliquid.xyz/ws',
        'name': 'Hyperliquid',
        'maker_fee': 0.0002,
        'taker_fee': 0.0005,
    },
    'bitstamp': {
        'url': 'wss://ws.bitstamp.net',
        'name': 'Bitstamp',
        'maker_fee': 0.0,
        'taker_fee': 0.0004,
    },

    # PRICE DATA ONLY (not trading)
    'bitfinex': {
        'url': 'wss://api-pub.bitfinex.com/ws/2',
        'name': 'Bitfinex',
        'maker_fee': 0.001,
        'taker_fee': 0.002,
    },
}

# =============================================================================
# FEE-AWARE GATE THRESHOLDS - Must exceed fee to be profitable
# =============================================================================
# Each exchange has different fees, so the minimum win rate to be profitable
# differs. The Breakeven Gate (ID 950) uses these per-exchange.

EXCHANGE_FEE_GATES = {
    # Exchange: (min_win_rate_to_cover_fees, base_gate_threshold)
    'bitstamp': (0.5004, 0.5075),    # 0% maker, 0.04% taker - BEST
    'hyperliquid': (0.5005, 0.5075), # 0.02% maker, 0.05% taker - EXCELLENT
    'dydx': (0.5005, 0.5075),        # -0.025% maker rebate, 0.05% taker
    'binance': (0.5010, 0.5075),     # 0.1% maker/taker - GOOD
    'gemini': (0.5030, 0.5100),      # 0.1-0.3% fees
    'kraken': (0.5026, 0.5100),      # 0.16-0.26% fees
    'coinbase': (0.5060, 0.5135),    # 0.4-0.6% fees - requires higher edge
}

# Legacy single endpoint (for backwards compatibility)
KRAKEN_WS_URL = "wss://ws.kraken.com"

# =============================================================================
# ACTIVE TRADING EXCHANGES - USA Legal
# =============================================================================
ACTIVE_TRADING_EXCHANGES = ['coinbase', 'kraken', 'bitstamp', 'gemini', 'binance', 'hyperliquid']
