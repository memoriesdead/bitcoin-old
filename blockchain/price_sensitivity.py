#!/usr/bin/env python3
"""Price sensitivity analysis across energy costs."""

from mathematical_price import MathematicalPricer

def main():
    print()
    print("=" * 70)
    print("BLOCKCHAIN-DERIVED PRICE SENSITIVITY ANALYSIS")
    print("=" * 70)
    print()
    print("Energy Cost | Production Cost | Multiplier | Derived Price | vs $97k")
    print("-" * 70)

    market_price = 97000

    for energy in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]:
        pricer = MathematicalPricer(energy_cost_kwh=energy)
        p = pricer.get_price()
        deviation = (p.derived_price - market_price) / market_price * 100
        print(f"  ${energy:.2f}/kWh  |    ${p.production_cost:>10,.0f} |   {p.combined_multiplier:.4f}x |   ${p.derived_price:>10,.0f} | {deviation:>+6.1f}%")

    print()
    print("=" * 70)

    # Find the energy cost that matches market price
    low, high = 0.01, 0.20
    target = market_price

    for _ in range(50):
        mid = (low + high) / 2
        pricer = MathematicalPricer(energy_cost_kwh=mid)
        derived = pricer.get_price().derived_price

        if derived < target:
            low = mid
        else:
            high = mid

    print(f"Market price (${market_price:,}) matches derived at ${mid:.4f}/kWh")
    print("=" * 70)


if __name__ == "__main__":
    main()
