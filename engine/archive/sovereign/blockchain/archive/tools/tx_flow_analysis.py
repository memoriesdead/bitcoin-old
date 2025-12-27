#!/usr/bin/env python3
"""Analyze transaction flow patterns for Binance and Coinbase"""

import sqlite3
import json
import urllib.request
import base64
from collections import defaultdict

# Bitcoin RPC setup
auth = base64.b64encode(b'bitcoin:bitcoin123secure').decode()

def rpc(method, params=None):
    payload = json.dumps({'jsonrpc': '1.0', 'id': 'x', 'method': method, 'params': params or []}).encode()
    req = urllib.request.Request('http://127.0.0.1:8332')
    req.add_header('Authorization', f'Basic {auth}')
    req.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(req, payload, timeout=60) as resp:
        return json.loads(resp.read()).get('result')

def main():
    # Load from BOTH databases
    conn_cluster = sqlite3.connect('/root/sovereign/address_clusters.db')
    cluster_addrs = {r[0]: r[1] for r in conn_cluster.execute('SELECT address, exchange FROM addresses WHERE exchange LIKE "%binance%" OR exchange LIKE "%Binance%" OR exchange LIKE "%coinbase%" OR exchange LIKE "%Coinbase%"')}
    conn_cluster.close()

    conn_wallet = sqlite3.connect('/root/sovereign/walletexplorer_addresses.db')
    wallet_addrs = {r[0]: r[1] for r in conn_wallet.execute('SELECT address, exchange FROM addresses WHERE LOWER(exchange) IN ("binance", "coinbase")')}
    conn_wallet.close()

    # Merge both - normalize exchange names
    addrs = {}
    for addr, ex in wallet_addrs.items():
        addrs[addr] = ex.lower()
    for addr, ex in cluster_addrs.items():
        addrs[addr] = ex.lower()

    print(f'Loaded {len(addrs)} total addresses (clusters: {len(cluster_addrs)}, walletexplorer: {len(wallet_addrs)})')

    height = rpc('getblockcount')
    print(f'Current height: {height}')

    inflows = defaultdict(list)
    outflows = defaultdict(list)

    print(f'\nAnalyzing last 20 blocks ({height-20} to {height})...')
    for h in range(height-20, height):
        block = rpc('getblock', [rpc('getblockhash', [h]), 3])
        for tx in block.get('tx', []):
            for vout in tx.get('vout', []):
                addr = vout.get('scriptPubKey', {}).get('address')
                if addr and addr in addrs:
                    ex = addrs[addr]
                    amt = vout.get('value', 0)
                    if amt > 0:
                        inflows[ex].append(amt)
            for vin in tx.get('vin', []):
                prevout = vin.get('prevout', {})
                if prevout:
                    addr = prevout.get('scriptPubKey', {}).get('address')
                    if addr and addr in addrs:
                        ex = addrs[addr]
                        amt = prevout.get('value', 0)
                        if amt > 0:
                            outflows[ex].append(amt)

    print(f'\nFlows detected: {len(inflows)} exchanges with inflows, {len(outflows)} with outflows')

    print('\n' + '='*80)
    print('TRANSACTION SIZE DISTRIBUTION')
    print('='*80)

    for ex in sorted(set(list(inflows.keys()) + list(outflows.keys()))):
        print(f'\n{ex.upper()}:')

        if ex in inflows:
            ins = sorted(inflows[ex], reverse=True)
            total_in = sum(ins)
            print(f'\n  INFLOWS: {len(ins)} transactions, {total_in:.4f} BTC total')
            print(f'    Largest 10: {[round(x, 4) for x in ins[:10]]}')
            print(f'    Median: {ins[len(ins)//2]:.4f} BTC')
            print(f'    Mean: {total_in/len(ins):.4f} BTC')
            print(f'    Distribution:')
            print(f'      >0.1 BTC: {sum(1 for x in ins if x > 0.1)} txs ({sum(x for x in ins if x > 0.1):.2f} BTC)')
            print(f'      >1 BTC:   {sum(1 for x in ins if x > 1)} txs ({sum(x for x in ins if x > 1):.2f} BTC)')
            print(f'      >10 BTC:  {sum(1 for x in ins if x > 10)} txs ({sum(x for x in ins if x > 10):.2f} BTC)')
            print(f'      >50 BTC:  {sum(1 for x in ins if x > 50)} txs ({sum(x for x in ins if x > 50):.2f} BTC)')
            print(f'      >100 BTC: {sum(1 for x in ins if x > 100)} txs ({sum(x for x in ins if x > 100):.2f} BTC)')

        if ex in outflows:
            outs = sorted(outflows[ex], reverse=True)
            total_out = sum(outs)
            print(f'\n  OUTFLOWS: {len(outs)} transactions, {total_out:.4f} BTC total')
            print(f'    Largest 10: {[round(x, 4) for x in outs[:10]]}')
            print(f'    Median: {outs[len(outs)//2]:.4f} BTC')
            print(f'    Mean: {total_out/len(outs):.4f} BTC')
            print(f'    Distribution:')
            print(f'      >0.1 BTC: {sum(1 for x in outs if x > 0.1)} txs ({sum(x for x in outs if x > 0.1):.2f} BTC)')
            print(f'      >1 BTC:   {sum(1 for x in outs if x > 1)} txs ({sum(x for x in outs if x > 1):.2f} BTC)')
            print(f'      >10 BTC:  {sum(1 for x in outs if x > 10)} txs ({sum(x for x in outs if x > 10):.2f} BTC)')
            print(f'      >50 BTC:  {sum(1 for x in outs if x > 50)} txs ({sum(x for x in outs if x > 50):.2f} BTC)')
            print(f'      >100 BTC: {sum(1 for x in outs if x > 100)} txs ({sum(x for x in outs if x > 100):.2f} BTC)')

    print('\n' + '='*80)

if __name__ == '__main__':
    main()
