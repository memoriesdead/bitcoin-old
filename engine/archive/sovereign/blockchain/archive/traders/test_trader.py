#!/usr/bin/env python3
"""Quick test of live_trader components."""

from live_trader import TradingConfig, SignalParser, LiveTrader
import ccxt

# Test config
config = TradingConfig(paper_mode=True, min_flow_btc=5.0)
print(f'Config OK - Paper mode: {config.paper_mode}')

# Test signal parser connection
parser = SignalParser()
print(f'Signal parser initialized - VPS: {parser.vps_host}')

# Test fetching signals
signals = parser.get_new_signals()
print(f'Got {len(signals)} new signals')
for sig in signals[:5]:
    print(f"  {sig['direction']} {sig['exchanges']} - {sig['flow_btc']:.2f} BTC")

# Test price fetch
exchange = ccxt.kraken({'enableRateLimit': True})
ticker = exchange.fetch_ticker('BTC/USDT')
print(f'Price fetch OK - BTC: ${ticker["last"]:,.2f}')

print('\nAll tests passed!')
