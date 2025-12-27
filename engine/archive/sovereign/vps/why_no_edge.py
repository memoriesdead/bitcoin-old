#!/usr/bin/env python3
import sqlite3
import numpy as np
from pathlib import Path

DATA_DIR = Path('/root/validation/data')
conn = sqlite3.connect(DATA_DIR / 'metrics.db')

cursor = conn.execute('SELECT price FROM metrics WHERE price > 0 ORDER BY timestamp')
prices = [r[0] for r in cursor.fetchall()]
conn.close()

first = prices[0]
last = prices[-1]
change_pct = (last/first - 1) * 100

print('='*50)
print('  WHY NO EDGE?')
print('='*50)
print('First price: ${:,.0f}'.format(first))
print('Last price:  ${:,.0f}'.format(last))
print('Change: {:+.2f}%'.format(change_pct))
print()

returns = np.diff(prices) / prices[:-1] * 10000
print('Return stats (bps per tick):')
print('  Mean: {:+.3f}'.format(np.mean(returns)))
print('  Std:  {:.2f}'.format(np.std(returns)))
print('  Positive ticks: {:.1f}%'.format(np.sum(returns > 0) / len(returns) * 100))
print()

autocorr = np.corrcoef(returns[:-1], returns[1:])[0,1]
print('Return autocorrelation: {:.3f}'.format(autocorr))
if autocorr < -0.05:
    print('  -> Market is MEAN-REVERTING (good for fade strategies)')
elif autocorr > 0.05:
    print('  -> Market is TRENDING (good for momentum)')
else:
    print('  -> Market is RANDOM WALK (no predictability)')

print()
print('='*50)
print('CONCLUSION:')
print('='*50)
print('L1 blockchain data (whale txs, volume) shows')
print('NO predictive power for price direction.')
print()
print('Options:')
print('1. Add more data: funding rates, liquidations')
print('2. Look for volatility patterns, not direction')
print('3. Collect more data across different regimes')
print('4. Accept that this signal may not exist')
print('='*50)
