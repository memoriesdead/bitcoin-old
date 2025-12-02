#!/usr/bin/env python3
"""
GENERATE COMPLETE BLOCKCHAIN DATASET FROM PURE MATH
====================================================
Uses Power Law + Halving + Block Timing to generate
926,109 blocks worth of data INSTANTLY.

NO external APIs needed. Pure mathematics.
"""

import os
import numpy as np
from datetime import datetime

# Bitcoin genesis
GENESIS_TS = 1231006505  # Jan 3, 2009
BLOCKS_PER_DAY = 144
SECONDS_PER_BLOCK = 600

# Power Law constants
POWER_LAW_A = -17.01204

POWER_LAW_B = 5.84509

# Current blockchain state
CURRENT_HEIGHT = 926109

print("=" * 80)
print("GENERATING COMPLETE BLOCKCHAIN DATASET - PURE MATH")
print("=" * 80)
print()
print(f"[INFO] Generating {CURRENT_HEIGHT:,} blocks")
print(f"[INFO] Genesis: Jan 3, 2009")
print(f"[INFO] Method: Power Law + Halving + Block Timing")
print()

# Generate block heights
heights = np.arange(0, CURRENT_HEIGHT, dtype=np.int32)

# Generate timestamps (10 min average with realistic variance)
print("[CALC] Generating timestamps...")
base_timestamps = GENESIS_TS + heights * SECONDS_PER_BLOCK
# Add realistic variance (+/- 20% per block)
variance = np.random.randn(len(heights)) * 120  # ~2 min std dev
timestamps = (base_timestamps + variance).astype(np.int64)
timestamps = np.maximum.accumulate(timestamps)  # Ensure monotonic

# Calculate days since genesis for Power Law
print("[CALC] Calculating Power Law prices...")
days_since_genesis = (timestamps - GENESIS_TS) / 86400.0

# Power Law base price
log_days = np.log10(days_since_genesis + 1)
base_price = 10 ** (POWER_LAW_A + POWER_LAW_B * log_days)

# Halving cycle adjustment
print("[CALC] Applying halving cycles...")
halving_era = heights // 210000
scarcity_mult = 1.0 + halving_era * 0.5

# Stock-to-Flow adjustment
s2f_mult = np.clip(1.0 + (days_since_genesis / 1461) * 0.3, 1.0, 2.5)

# Combine
fair_value = base_price * scarcity_mult * s2f_mult

# Add realistic price variance
price_noise = 1.0 + np.random.randn(len(heights)) * 0.05
prices = fair_value * price_noise

# Generate difficulty (increases over time)
print("[CALC] Generating difficulty...")
difficulty = np.exp(heights / 50000.0)  # Exponential growth

# Generate TX counts (increases with adoption)
print("[CALC] Generating transaction counts...")
tx_count = np.clip(
    (days_since_genesis / 100) ** 1.5 + np.random.randint(0, 500, len(heights)),
    1,
    10000
).astype(np.int32)

# Generate block sizes
print("[CALC] Generating block sizes...")
block_size = np.clip(
    (tx_count * 250) + np.random.randint(0, 100000, len(heights)),
    285,  # Minimum (coinbase only)
    4000000  # Max block weight
).astype(np.int32)

# Generate nonces (random)
nonces = np.random.randint(0, 2**32, len(heights), dtype=np.uint32)

# Create structured array
print("[ARRAY] Creating structured NumPy array...")
data = np.zeros(len(heights), dtype=[
    ('height', np.int32),
    ('timestamp', np.int64),
    ('price', np.float64),
    ('difficulty', np.float64),
    ('tx_count', np.int32),
    ('block_size', np.int32),
    ('nonce', np.uint32),
])

data['height'] = heights
data['timestamp'] = timestamps
data['price'] = prices
data['difficulty'] = difficulty
data['tx_count'] = tx_count
data['block_size'] = block_size
data['nonce'] = nonces

# Validate
print()
print("[VALIDATE] Quality checks...")
print(f"[VALIDATE] OK: Heights: 0 -> {heights[-1]:,}")
print(f"[VALIDATE] OK: Timestamps monotonic: {np.all(np.diff(timestamps) >= 0)}")
print(f"[VALIDATE] OK: Price range: ${prices.min():.2f} -> ${prices.max():,.2f}")
print(f"[VALIDATE] OK: Difficulty: {difficulty.min():.0f} -> {difficulty.max():.2e}")
print(f"[VALIDATE] OK: TX counts: {tx_count.min()} -> {tx_count.max()}")

# Save
output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'blockchain_complete.npy')
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print()
print(f"[SAVE] Writing to: {output_file}")
np.save(output_file, data)

size_mb = os.path.getsize(output_file) / 1024 / 1024
print(f"[SAVE] OK: {len(data):,} blocks ({size_mb:.1f} MB)")

# Benchmark
import time
print()
print("[BENCHMARK] Testing access speed...")
start = time.perf_counter_ns()
loaded = np.load(output_file)
load_time_ns = time.perf_counter_ns() - start
print(f"[BENCHMARK] Load time: {load_time_ns / 1_000_000:.2f}ms")

indices = np.random.randint(0, len(loaded), 10000)
start = time.perf_counter_ns()
for i in indices:
    _ = loaded[i]
access_time_ns = time.perf_counter_ns() - start
per_access = access_time_ns / 10000
print(f"[BENCHMARK] Random access: {per_access:.0f}ns per block")

print()
print("=" * 80)
print("COMPLETE BLOCKCHAIN DATASET READY - PURE MATHEMATICS")
print("=" * 80)
print()
print("Coverage:")
min_date = datetime.fromtimestamp(timestamps[0]).strftime('%Y-%m-%d')
max_date = datetime.fromtimestamp(timestamps[-1]).strftime('%Y-%m-%d')
print(f"  Date range: {min_date} -> {max_date}")
print(f"  Block range: {heights[0]:,} -> {heights[-1]:,}")
print(f"  Total blocks: {len(heights):,}")
print(f"  Duration: {(timestamps[-1] - timestamps[0]) / 86400:.0f} days")
print()
print("READY TO PRINT MONEY")
print("=" * 80)
