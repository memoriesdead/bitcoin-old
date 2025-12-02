#!/usr/bin/env python3
"""Test the BlockchainUnifiedFeed integration."""

print("Testing BlockchainUnifiedFeed...")
print()

from blockchain import BlockchainUnifiedFeed

feed = BlockchainUnifiedFeed()

print("Generating 5 signals:")
print("-" * 70)

import time
for i in range(5):
    s = feed.get_signal()
    ofi_label = "BUY " if s.ofi_direction == 1 else "SELL" if s.ofi_direction == -1 else "HOLD"
    print(f"[{i+1}] ${s.mid_price:,.2f} | OFI: {s.ofi_normalized:+.3f} ({ofi_label}) | "
          f"Str: {s.ofi_strength:.2f} | Dev: {s.deviation_pct:+.1f}%")
    time.sleep(0.1)

print()
print("=" * 70)
print("SUCCESS! Blockchain feed working - NO exchange APIs!")
print("=" * 70)
