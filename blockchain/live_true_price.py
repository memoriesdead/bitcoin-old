#!/usr/bin/env python3
"""
LIVE TRUE BITCOIN PRICE - PURE BLOCKCHAIN DERIVATION
=====================================================

Streams the genuine blockchain-derived price in real-time.
NO exchange APIs. NO historical calibration. PURE MATH.
"""

import time
import sys
from mathematical_price import MathematicalPricer


def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    energy_cost = float(sys.argv[2]) if len(sys.argv) > 2 else 0.044  # Implied market rate

    print()
    print("=" * 70)
    print("LIVE TRUE BITCOIN PRICE - PURE BLOCKCHAIN DERIVATION")
    print("=" * 70)
    print()
    print("Formula: Price = Production_Cost x (1 + Scarcity + Maturity x Supply)")
    print()
    print("Where ALL multipliers use blockchain metrics as divisors:")
    print("  Scarcity = ln(S2F) / (halvings + 1)^2")
    print("  Maturity = ln(days) / (ln(days) + halvings^2)")
    print("  Supply   = 1 / (1 + ln(MAX/current))")
    print()
    print(f"Energy Cost: ${energy_cost}/kWh (only external input)")
    print("=" * 70)
    print()

    pricer = MathematicalPricer(energy_cost_kwh=energy_cost)
    start_time = time.time()
    update_count = 0

    print("Time    | Block     | Hash Rate | Prod Cost  | Mult   | TRUE PRICE")
    print("-" * 70)

    while time.time() - start_time < duration:
        try:
            p = pricer.get_price()
            elapsed = time.time() - start_time
            update_count += 1

            print(f"{elapsed:5.1f}s  | {p.block_height:,} | {p.hash_rate/1e18:5.1f} EH/s | ${p.production_cost:>9,.0f} | {p.combined_multiplier:.3f}x | ${p.derived_price:>10,.2f}")

            time.sleep(2)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    p = pricer.get_price()
    print()
    print(f"  Block Height:       {p.block_height:,}")
    print(f"  Hash Rate:          {p.hash_rate/1e18:.2f} EH/s")
    print(f"  Stock-to-Flow:      {p.stock_to_flow:.1f}")
    print(f"  Supply Mined:       {p.supply_ratio*100:.2f}%")
    print()
    print(f"  Production Cost:    ${p.production_cost:,.2f}")
    print(f"  Multiplier:         {p.combined_multiplier:.4f}x")
    print()
    print(f"  TRUE BITCOIN PRICE: ${p.derived_price:,.2f}")
    print()
    print("=" * 70)
    print("PURE BLOCKCHAIN + PHYSICS. NO EXCHANGE DATA.")
    print("=" * 70)


if __name__ == "__main__":
    main()
