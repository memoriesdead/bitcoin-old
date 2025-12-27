#!/usr/bin/env python3
"""
BITCOIN NODE INFRASTRUCTURE TESTS
==================================

Test the VPS Bitcoin node and data collection.
"""

import os
import sys
import time
import json
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

# Test results
RESULTS = []

VPS_HOST = "root@31.97.211.217"

def log_test(name: str, passed: bool, details: str = ""):
    """Log test result."""
    status = "PASS" if passed else "FAIL"
    RESULTS.append({"name": name, "passed": passed, "details": details})
    print(f"[{status}] {name}")
    if details:
        print(f"       {details}")


def run_ssh(cmd: str, timeout: int = 30) -> tuple:
    """Run SSH command and return (success, output)."""
    try:
        result = subprocess.run(
            f'ssh {VPS_HOST} "{cmd}"',
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def test_ssh_connection():
    """Test SSH connection to VPS."""
    print("\n" + "="*60)
    print("TEST 1: VPS CONNECTION")
    print("="*60)

    success, output = run_ssh("echo 'connected'")
    log_test("SSH Connection", success, output if not success else "Connected to VPS")
    return success


def test_bitcoin_core():
    """Test Bitcoin Core is running."""
    print("\n" + "="*60)
    print("TEST 2: BITCOIN CORE STATUS")
    print("="*60)

    # Check if bitcoind is running
    success, output = run_ssh("pgrep -x bitcoind")
    log_test("bitcoind Process", success, f"PID: {output}" if success else "Not running")

    # Check blockchain info
    success, output = run_ssh("bitcoin-cli getblockchaininfo 2>/dev/null | head -5")
    if success and output:
        log_test("Bitcoin RPC", True, "RPC responding")
    else:
        log_test("Bitcoin RPC", False, output)

    # Check block height
    success, output = run_ssh("bitcoin-cli getblockcount 2>/dev/null")
    if success:
        log_test("Block Height", True, f"Block {output}")
    else:
        log_test("Block Height", False, output)

    return success


def test_zmq_feed():
    """Test ZMQ is configured and working."""
    print("\n" + "="*60)
    print("TEST 3: ZMQ FEED")
    print("="*60)

    # Check ZMQ config
    success, output = run_ssh("grep zmq /root/.bitcoin/bitcoin.conf 2>/dev/null || grep zmq /bitcoin/bitcoin.conf 2>/dev/null")
    if success and "zmqpubrawtx" in output:
        log_test("ZMQ Config", True, "zmqpubrawtx configured")
    else:
        log_test("ZMQ Config", False, "ZMQ not in config")

    # Check ZMQ port is listening
    success, output = run_ssh("netstat -tlnp 2>/dev/null | grep 28332 || ss -tlnp | grep 28332")
    if success and "28332" in output:
        log_test("ZMQ Port 28332", True, "Listening")
    else:
        log_test("ZMQ Port 28332", False, "Not listening")

    return success


def test_collector_service():
    """Test the metric collector service."""
    print("\n" + "="*60)
    print("TEST 4: COLLECTOR SERVICE")
    print("="*60)

    # Check service status
    success, output = run_ssh("systemctl is-active rentech-collector")
    log_test("Service Status", output == "active", output)

    # Check recent logs
    success, output = run_ssh("journalctl -u rentech-collector -n 5 --no-pager 2>/dev/null | tail -3")
    if success:
        log_test("Recent Logs", True, output[:80] + "..." if len(output) > 80 else output)
    else:
        log_test("Recent Logs", False, "Could not get logs")

    # Check uptime
    success, output = run_ssh("systemctl show rentech-collector --property=ActiveEnterTimestamp")
    if success:
        log_test("Service Uptime", True, output.replace("ActiveEnterTimestamp=", ""))

    return output == "active"


def test_data_collection():
    """Test data is being collected."""
    print("\n" + "="*60)
    print("TEST 5: DATA COLLECTION")
    print("="*60)

    # Get row count
    success, output = run_ssh("sqlite3 /root/validation/data/metrics.db 'SELECT COUNT(*) FROM metrics;'")
    if success:
        rows = int(output)
        log_test("Total Rows", rows > 0, f"{rows:,} rows collected")
    else:
        log_test("Total Rows", False, output)
        return False

    # Get time range
    success, output = run_ssh("""sqlite3 /root/validation/data/metrics.db "SELECT
        ROUND((MAX(timestamp) - MIN(timestamp)) / 60, 1) as minutes,
        ROUND((MAX(timestamp) - MIN(timestamp)) / 3600, 2) as hours
    FROM metrics;" """)
    if success:
        log_test("Collection Duration", True, f"{output} (minutes, hours)")

    # Get recent data rate
    success, output = run_ssh("""sqlite3 /root/validation/data/metrics.db "SELECT COUNT(*) FROM metrics
        WHERE timestamp > strftime('%s', 'now') - 60;" """)
    if success:
        rate = int(output)
        log_test("Data Rate (last 60s)", rate > 30, f"{rate} rows/minute")

    # Check data quality
    success, output = run_ssh("""sqlite3 /root/validation/data/metrics.db "SELECT
        SUM(tx_count) as txs,
        ROUND(SUM(total_volume_btc), 1) as btc,
        SUM(tx_whale) as whales,
        SUM(tx_mega) as megas
    FROM metrics;" """)
    if success:
        log_test("Data Content", True, f"txs|btc|whales|megas: {output}")

    return rows > 0


def test_data_freshness():
    """Test data is current."""
    print("\n" + "="*60)
    print("TEST 6: DATA FRESHNESS")
    print("="*60)

    # Get latest timestamp
    success, output = run_ssh("""sqlite3 /root/validation/data/metrics.db "SELECT
        MAX(timestamp),
        strftime('%s', 'now') - MAX(timestamp) as age_seconds
    FROM metrics;" """)

    if success:
        parts = output.split("|")
        if len(parts) == 2:
            age = float(parts[1])
            log_test("Latest Data Age", age < 60, f"{age:.0f} seconds old")
        else:
            log_test("Latest Data Age", False, "Could not parse")
    else:
        log_test("Latest Data Age", False, output)

    # Check for gaps
    success, output = run_ssh("""sqlite3 /root/validation/data/metrics.db "SELECT COUNT(*) FROM (
        SELECT timestamp,
               LAG(timestamp) OVER (ORDER BY timestamp) as prev_ts,
               timestamp - LAG(timestamp) OVER (ORDER BY timestamp) as gap
        FROM metrics
    ) WHERE gap > 5;" """)
    if success:
        gaps = int(output)
        log_test("Data Gaps (>5s)", gaps < 10, f"{gaps} gaps found")

    return True


def test_price_feed():
    """Test price data is being collected."""
    print("\n" + "="*60)
    print("TEST 7: PRICE FEED")
    print("="*60)

    # Check price data
    success, output = run_ssh("""sqlite3 /root/validation/data/metrics.db "SELECT
        COUNT(*) as rows,
        ROUND(AVG(price), 0) as avg_price,
        ROUND(MIN(price), 0) as min_price,
        ROUND(MAX(price), 0) as max_price
    FROM metrics WHERE price > 0;" """)

    if success:
        parts = output.split("|")
        if len(parts) == 4:
            rows, avg, min_p, max_p = parts
            log_test("Price Data", int(rows) > 0, f"{rows} prices, avg ${avg}")
            log_test("Price Range", True, f"${min_p} - ${max_p}")
        else:
            log_test("Price Data", False, "Could not parse")
    else:
        log_test("Price Data", False, output)

    return True


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
    print("  BITCOIN NODE INFRASTRUCTURE VALIDATION")
    print("="*60)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  VPS: {VPS_HOST}")
    print("="*60)

    # Run all tests
    if not test_ssh_connection():
        print("\nCannot connect to VPS. Aborting tests.")
        return False

    test_bitcoin_core()
    test_zmq_feed()
    test_collector_service()
    test_data_collection()
    test_data_freshness()
    test_price_feed()

    # Summary
    all_passed = print_summary()

    if all_passed:
        print("\nBitcoin node infrastructure READY.")
    else:
        print("\nFix failed tests before proceeding.")

    return all_passed


if __name__ == "__main__":
    main()
