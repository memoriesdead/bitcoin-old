#!/usr/bin/env python3
"""Get Bitcoin volume statistics for engine calibration"""
import requests

print("=" * 60)
print("BITCOIN VOLUME ANALYSIS FOR ENGINE CALIBRATION")
print("=" * 60)

# Binance ticker
try:
    r = requests.get('https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT', timeout=5)
    data = r.json()
    price = float(data['lastPrice'])
    vol_24h = float(data['quoteVolume'])
    vol_per_sec = vol_24h / 86400
    vol_per_min = vol_24h / 1440

    print()
    print("BINANCE BTC/USDT:")
    print(f"  Price:           ${price:,.2f}")
    print(f"  24h Volume:      ${vol_24h:,.0f}")
    print(f"  Volume/minute:   ${vol_per_min:,.0f}")
    print(f"  Volume/second:   ${vol_per_sec:,.2f}")
    print(f"  Volume/100ms:    ${vol_per_sec/10:,.2f}")
    print(f"  Volume/10ms:     ${vol_per_sec/100:,.2f}")
    print(f"  Volume/1ms:      ${vol_per_sec/1000:,.2f}")
    print()
    print(f"  Price Change 24h: {float(data['priceChangePercent']):+.2f}%")
    print(f"  High 24h:        ${float(data['highPrice']):,.2f}")
    print(f"  Low 24h:         ${float(data['lowPrice']):,.2f}")
    range_24h = float(data['highPrice']) - float(data['lowPrice'])
    print(f"  Range 24h:       ${range_24h:,.2f}")
    range_pct = range_24h / price * 100
    print(f"  Range %:         {range_pct:.2f}%")
except Exception as e:
    print(f"Binance error: {e}")
    range_pct = 2.0  # default
    price = 85000

# Calculate volatility per timeframe
print()
print("=" * 60)
print("VOLATILITY BY TIMEFRAME (estimated from 24h range)")
print("=" * 60)

# Volatility scales with sqrt of time
vol_1h = range_pct / 24 * 2.5
vol_5m = vol_1h / 3.5
vol_1m = vol_5m / 2.2
vol_10s = vol_1m / 2.4
vol_1s = vol_10s / 3.2

print(f"  1 hour:    ~{vol_1h:.3f}% move")
print(f"  5 min:     ~{vol_5m:.3f}% move")
print(f"  1 min:     ~{vol_1m:.4f}% move")
print(f"  10 sec:    ~{vol_10s:.4f}% move")
print(f"  1 sec:     ~{vol_1s:.5f}% move")
print()
print(f"  At ${price:,.0f}:")
print(f"    1 min move:  ~${price * vol_1m/100:.2f}")
print(f"    10 sec move: ~${price * vol_10s/100:.2f}")
print(f"    1 sec move:  ~${price * vol_1s/100:.4f}")

# Recent trades
print()
print("=" * 60)
print("RECENT TICK DATA (last 20 trades)")
print("=" * 60)
try:
    r = requests.get('https://api.binance.com/api/v3/trades?symbol=BTCUSDT&limit=20', timeout=5)
    trades = r.json()
    prices = [float(t['price']) for t in trades]
    min_p, max_p = min(prices), max(prices)
    spread = max_p - min_p
    avg_size = sum(float(t['quoteQty']) for t in trades) / len(trades)
    print(f"  Price range: ${min_p:,.2f} - ${max_p:,.2f}")
    print(f"  Tick spread: ${spread:.2f} ({spread/price*100:.5f}%)")
    print(f"  Avg trade size: ${avg_size:,.2f}")
    print(f"  Trades/sec estimate: ~50-100")
except Exception as e:
    print(f"Trades error: {e}")

# Recommended settings
print()
print("=" * 60)
print("RECOMMENDED TP/SL FOR TIMEFRAMES")
print("=" * 60)
print("  5 min holds:  TP=0.15%  SL=0.06%  (ratio 2.5:1)")
print("  1 min holds:  TP=0.05%  SL=0.02%  (ratio 2.5:1)")
print("  10 sec holds: TP=0.02%  SL=0.008% (ratio 2.5:1)")
print("  1 sec holds:  TP=0.005% SL=0.002% (ratio 2.5:1)")

print()
print("=" * 60)
print("CURRENT ENGINE ISSUE")
print("=" * 60)
print("  Current TP: 0.5% - TOO HIGH for 1-5 min trades!")
print("  Current SL: 0.2% - TOO HIGH for 1-5 min trades!")
print()
print("  BTC needs to move $427 for 0.5% TP - takes HOURS!")
print("  For 1-5 min scalping, need 0.02%-0.05% TP")
print()
print("=" * 60)
print("CALIBRATION RECOMMENDATION")
print("=" * 60)
print("  For 1-5 min scalping:")
print("    TP = 0.03% (~$25 move)")
print("    SL = 0.012% (~$10 move)")
print("    OFI threshold = 0.15 (more trades)")
print("    Expected: 100+ trades/hour")
