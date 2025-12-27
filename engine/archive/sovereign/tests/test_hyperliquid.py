#!/usr/bin/env python3
"""
HYPERLIQUID INFRASTRUCTURE TESTS
=================================

Test every aspect of Hyperliquid before trading.
"""

import os
import sys
import time
import json
import requests
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Test results
RESULTS = []

def log_test(name: str, passed: bool, details: str = ""):
    """Log test result."""
    status = "PASS" if passed else "FAIL"
    RESULTS.append({"name": name, "passed": passed, "details": details})
    print(f"[{status}] {name}")
    if details:
        print(f"       {details}")


def test_api_connection():
    """Test basic API connection."""
    print("\n" + "="*60)
    print("TEST 1: API CONNECTION")
    print("="*60)

    try:
        # Hyperliquid info endpoint
        url = "https://api.hyperliquid.xyz/info"
        start = time.time()
        resp = requests.post(url, json={"type": "meta"}, timeout=10)
        latency = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            log_test("API Connection", True, f"Latency: {latency:.0f}ms")
            return True, data
        else:
            log_test("API Connection", False, f"Status: {resp.status_code}")
            return False, None
    except Exception as e:
        log_test("API Connection", False, str(e))
        return False, None


def test_market_data():
    """Test market data retrieval."""
    print("\n" + "="*60)
    print("TEST 2: MARKET DATA")
    print("="*60)

    try:
        url = "https://api.hyperliquid.xyz/info"

        # Get all mids (mid prices)
        resp = requests.post(url, json={"type": "allMids"}, timeout=10)
        if resp.status_code == 200:
            mids = resp.json()
            btc_price = float(mids.get("BTC", 0))
            log_test("Get BTC Price", btc_price > 0, f"BTC: ${btc_price:,.2f}")
        else:
            log_test("Get BTC Price", False, f"Status: {resp.status_code}")

        # Get orderbook
        resp = requests.post(url, json={"type": "l2Book", "coin": "BTC"}, timeout=10)
        if resp.status_code == 200:
            book = resp.json()
            if "levels" in book:
                bids = book["levels"][0]
                asks = book["levels"][1]
                best_bid = float(bids[0]["px"]) if bids else 0
                best_ask = float(asks[0]["px"]) if asks else 0
                spread = (best_ask - best_bid) / best_bid * 10000 if best_bid > 0 else 0
                log_test("Get Orderbook", True, f"Bid: ${best_bid:,.2f}, Ask: ${best_ask:,.2f}, Spread: {spread:.1f}bps")
            else:
                log_test("Get Orderbook", False, "No levels in response")
        else:
            log_test("Get Orderbook", False, f"Status: {resp.status_code}")

        # Get recent trades
        resp = requests.post(url, json={"type": "recentTrades", "coin": "BTC"}, timeout=10)
        if resp.status_code == 200:
            trades = resp.json()
            if trades:
                log_test("Get Recent Trades", True, f"Got {len(trades)} trades")
            else:
                log_test("Get Recent Trades", True, "No recent trades (market quiet)")
        else:
            log_test("Get Recent Trades", False, f"Status: {resp.status_code}")

        return True
    except Exception as e:
        log_test("Market Data", False, str(e))
        return False


def test_asset_info():
    """Test asset/contract info."""
    print("\n" + "="*60)
    print("TEST 3: ASSET INFO")
    print("="*60)

    try:
        url = "https://api.hyperliquid.xyz/info"
        resp = requests.post(url, json={"type": "meta"}, timeout=10)

        if resp.status_code == 200:
            meta = resp.json()
            universe = meta.get("universe", [])

            # Find BTC
            btc_info = None
            for asset in universe:
                if asset.get("name") == "BTC":
                    btc_info = asset
                    break

            if btc_info:
                log_test("BTC Contract Found", True)
                log_test("Max Leverage", True, f"{btc_info.get('maxLeverage', 'N/A')}x")
                log_test("Tick Size", True, f"${btc_info.get('szDecimals', 'N/A')}")
            else:
                log_test("BTC Contract Found", False, "BTC not in universe")

            log_test("Total Assets", True, f"{len(universe)} perpetual contracts")
            return True
        else:
            log_test("Asset Info", False, f"Status: {resp.status_code}")
            return False
    except Exception as e:
        log_test("Asset Info", False, str(e))
        return False


def test_fee_structure():
    """Document fee structure."""
    print("\n" + "="*60)
    print("TEST 4: FEE STRUCTURE")
    print("="*60)

    # Hyperliquid documented fees
    fees = {
        "maker": 0.0002,  # 0.02%
        "taker": 0.0005,  # 0.05%
        "funding": "variable (8h intervals)"
    }

    log_test("Maker Fee", True, f"{fees['maker']*100:.2f}%")
    log_test("Taker Fee", True, f"{fees['taker']*100:.2f}%")
    log_test("Funding Rate", True, fees['funding'])

    # Calculate fee impact on strategy
    avg_trade_bps = 8.8  # Our expected return
    taker_fee_bps = 5.0  # 0.05%
    round_trip_cost = taker_fee_bps * 2  # Open + close
    net_per_trade = avg_trade_bps - round_trip_cost

    log_test("Expected Return", True, f"{avg_trade_bps:.1f} bps/trade")
    log_test("Round-trip Cost", True, f"{round_trip_cost:.1f} bps")
    log_test("Net After Fees", net_per_trade > 0, f"{net_per_trade:.1f} bps/trade")

    return True


def test_latency():
    """Measure API latency."""
    print("\n" + "="*60)
    print("TEST 5: LATENCY MEASUREMENT")
    print("="*60)

    url = "https://api.hyperliquid.xyz/info"
    latencies = []

    for i in range(10):
        start = time.time()
        resp = requests.post(url, json={"type": "allMids"}, timeout=10)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        time.sleep(0.1)

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    log_test("Avg Latency", avg_latency < 500, f"{avg_latency:.0f}ms")
    log_test("Min Latency", True, f"{min_latency:.0f}ms")
    log_test("Max Latency", max_latency < 1000, f"{max_latency:.0f}ms")

    return avg_latency < 500


def test_rate_limits():
    """Test rate limit behavior."""
    print("\n" + "="*60)
    print("TEST 6: RATE LIMITS")
    print("="*60)

    url = "https://api.hyperliquid.xyz/info"

    # Send rapid requests
    success_count = 0
    error_count = 0

    for i in range(20):
        try:
            resp = requests.post(url, json={"type": "allMids"}, timeout=5)
            if resp.status_code == 200:
                success_count += 1
            else:
                error_count += 1
        except:
            error_count += 1

    log_test("Rapid Requests (20)", error_count == 0, f"{success_count}/20 succeeded")

    # Document known limits
    log_test("Known Limits", True, "~1200 requests/min for info endpoints")

    return error_count == 0


def test_sdk_import():
    """Test if Hyperliquid SDK is available."""
    print("\n" + "="*60)
    print("TEST 7: SDK AVAILABILITY")
    print("="*60)

    try:
        from hyperliquid.info import Info
        from hyperliquid.exchange import Exchange
        log_test("SDK Import", True, "hyperliquid-python-sdk installed")
        return True
    except ImportError:
        log_test("SDK Import", False, "Run: pip install hyperliquid-python-sdk")
        return False


def print_summary():
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for r in RESULTS if r["passed"])
    total = len(RESULTS)

    print(f"\nPassed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if total - passed > 0:
        print("\nFailed Tests:")
        for r in RESULTS:
            if not r["passed"]:
                print(f"  - {r['name']}: {r['details']}")

    print("\n" + "="*60)

    return passed == total


def main():
    print("="*60)
    print("  HYPERLIQUID INFRASTRUCTURE VALIDATION")
    print("="*60)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Run all tests
    test_api_connection()
    test_market_data()
    test_asset_info()
    test_fee_structure()
    test_latency()
    test_rate_limits()
    test_sdk_import()

    # Summary
    all_passed = print_summary()

    if all_passed:
        print("\nHyperliquid infrastructure READY for testing.")
    else:
        print("\nFix failed tests before proceeding.")

    return all_passed


if __name__ == "__main__":
    main()
