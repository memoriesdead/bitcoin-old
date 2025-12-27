#!/usr/bin/env python3
"""
Test USA exchange signal accuracy.
"""
import sqlite3

db = sqlite3.connect('/root/sovereign/correlation.db')
c = db.cursor()

# Get verified signals
c.execute('''
    SELECT exchange, direction, amount_btc, price_t0, price_t300, change_5m, direction_correct
    FROM flows
    WHERE price_t0 IS NOT NULL AND price_t300 IS NOT NULL
    ORDER BY timestamp DESC
''')
rows = c.fetchall()

print('='*70)
print('SIGNAL VERIFICATION - USA EXCHANGES')
print('='*70)

usa_exchanges = {'coinbase', 'kraken', 'bitstamp', 'gemini', 'crypto.com'}

results = {}
all_results = {}

for row in rows:
    ex, direction, amount, p0, p5m, change, correct = row

    # Track all exchanges
    if ex not in all_results:
        all_results[ex] = {'short_correct': 0, 'short_total': 0, 'long_correct': 0, 'long_total': 0}

    if direction == 'INFLOW':  # SHORT signal
        all_results[ex]['short_total'] += 1
        if correct == 1:
            all_results[ex]['short_correct'] += 1
    else:  # OUTFLOW = LONG signal
        all_results[ex]['long_total'] += 1
        if correct == 1:
            all_results[ex]['long_correct'] += 1

    # Filter USA
    if ex.lower() not in usa_exchanges:
        continue

    if ex not in results:
        results[ex] = {'short_correct': 0, 'short_total': 0, 'long_correct': 0, 'long_total': 0}

    if direction == 'INFLOW':
        results[ex]['short_total'] += 1
        if correct == 1:
            results[ex]['short_correct'] += 1
    else:
        results[ex]['long_total'] += 1
        if correct == 1:
            results[ex]['long_correct'] += 1

# USA Results
print()
print('USA EXCHANGES:')
print(f"{'Exchange':<15} {'SHORT':>20} {'LONG':>20}")
print('-'*60)

usa_short_correct = 0
usa_short_total = 0
usa_long_correct = 0
usa_long_total = 0

for ex in sorted(results.keys()):
    r = results[ex]
    usa_short_correct += r['short_correct']
    usa_short_total += r['short_total']
    usa_long_correct += r['long_correct']
    usa_long_total += r['long_total']

    if r['short_total'] > 0:
        short_pct = r['short_correct']/r['short_total']*100
        short_acc = f"{r['short_correct']}/{r['short_total']} = {short_pct:.0f}%"
    else:
        short_acc = 'N/A'

    if r['long_total'] > 0:
        long_pct = r['long_correct']/r['long_total']*100
        long_acc = f"{r['long_correct']}/{r['long_total']} = {long_pct:.0f}%"
    else:
        long_acc = 'N/A'

    print(f"{ex:<15} {short_acc:>20} {long_acc:>20}")

print('-'*60)
if usa_short_total > 0:
    print(f"{'USA SHORT':<15} {usa_short_correct}/{usa_short_total} = {usa_short_correct/usa_short_total*100:.0f}%")
if usa_long_total > 0:
    print(f"{'USA LONG':<15} {usa_long_correct}/{usa_long_total} = {usa_long_correct/usa_long_total*100:.0f}%")

# All exchanges summary
print()
print('='*70)
print('ALL EXCHANGES (for comparison):')
print('='*70)
print(f"{'Exchange':<15} {'SHORT':>20} {'LONG':>20}")
print('-'*60)

for ex in sorted(all_results.keys(), key=lambda x: -(all_results[x]['short_total']+all_results[x]['long_total'])):
    r = all_results[ex]
    total = r['short_total'] + r['long_total']
    if total < 3:
        continue

    if r['short_total'] > 0:
        short_pct = r['short_correct']/r['short_total']*100
        short_acc = f"{r['short_correct']}/{r['short_total']} = {short_pct:.0f}%"
    else:
        short_acc = 'N/A'

    if r['long_total'] > 0:
        long_pct = r['long_correct']/r['long_total']*100
        long_acc = f"{r['long_correct']}/{r['long_total']} = {long_pct:.0f}%"
    else:
        long_acc = 'N/A'

    usa = '*' if ex.lower() in usa_exchanges else ''
    print(f"{ex:<14}{usa} {short_acc:>20} {long_acc:>20}")

db.close()
