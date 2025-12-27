import numpy as np

print('='*70)
print('  COMPOUNDING SIMULATION - TX SPIKE LONG STRATEGY')
print('='*70)

edge_per_trade = 0.0074
trades_per_year = 202 / 6.3
fee_per_trade = 0.0008

net_edge = edge_per_trade - fee_per_trade
print(f'Gross edge: {edge_per_trade:.2%}')
print(f'Net edge: {net_edge:.2%}')
print(f'Trades/year: {trades_per_year:.0f}')
print()

print('COMPOUNDING FROM $100:')
print('-'*70)

for lev in [1, 2, 5, 10, 25]:
    lev_edge = net_edge * lev
    results = []
    for years in [1, 3, 5, 10]:
        cap = 100
        n_trades = int(trades_per_year * years)
        for _ in range(n_trades):
            cap *= (1 + lev_edge)
        results.append(cap)
    print(f'{lev:>2}x: 1yr=${results[0]:>12,.0f} | 3yr=${results[1]:>15,.0f} | 5yr=${results[2]:>18,.0f} | 10yr=${results[3]:>22,.0f}')

print()
print('='*70)
print('5x LEVERAGE YEAR BY YEAR:')
print('='*70)
lev = 5
capital = 100
for year in range(1, 11):
    n_trades = int(trades_per_year)
    for _ in range(n_trades):
        capital *= (1 + net_edge * lev)
    print(f'Year {year:>2}: ${capital:>25,.2f}')

print()
print('='*70)
print('THE RENTECH WAY:')
print('- Signal: Go LONG when tx_count z-score > 1.5')
print('- Edge: 0.66% per trade after fees')
print('- Leverage: 5x (safe Kelly)')
print('- 32 trades/year')
print('- Compound relentlessly')
print('='*70)
