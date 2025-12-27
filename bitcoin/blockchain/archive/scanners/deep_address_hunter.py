#!/usr/bin/env python3
"""
DEEP ADDRESS HUNTER
===================
Continuously hunts for exchange addresses using multiple sources.

Sources:
1. BitInfoCharts - labeled wallets
2. Blockchair - address labels
3. Blockchain.info - transaction clustering
4. WalletExplorer API - exchange wallets
5. Manual Proof of Reserves parsing

Run: python3 deep_address_hunter.py
"""

import sqlite3
import requests
import time
import re
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/sovereign/deep_hunter.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

DB_PATH = '/root/sovereign/walletexplorer_addresses.db'

# All exchanges we want to track
TARGET_EXCHANGES = [
    # Tier 1 - Global Leaders
    'binance', 'bybit', 'okx', 'bitget', 'coinbase', 'kraken',
    'kucoin', 'gate.io', 'htx', 'mexc', 'bitfinex',
    # Tier 2 - Regional
    'upbit', 'bithumb', 'coincheck', 'bitflyer', 'bitstamp',
    # Tier 3 - Derivatives
    'deribit', 'bitmex', 'phemex',
    # Tier 4 - US Regulated
    'gemini', 'crypto.com', 'robinhood',
    # Others
    'poloniex', 'bittrex', 'huobi', 'okcoin', 'luno',
]

# BitInfoCharts URL patterns
BITINFOCHARTS_PATTERNS = [
    '{exchange}',
    '{exchange}-wallet',
    '{exchange}-coldwallet',
    '{exchange}-cold',
    '{exchange}-hot',
]

def init_db():
    """Ensure database exists with proper schema."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Use existing schema: address, exchange, balance_sat, downloaded_at
    c.execute('''
        CREATE TABLE IF NOT EXISTS addresses (
            address TEXT PRIMARY KEY,
            exchange TEXT,
            balance_sat INTEGER,
            downloaded_at TEXT
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_exchange ON addresses(exchange)')
    conn.commit()
    conn.close()

def add_address(exchange: str, address: str, source: str) -> bool:
    """Add single address to database. Returns True if new."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            'INSERT OR IGNORE INTO addresses (address, exchange, downloaded_at) VALUES (?, ?, ?)',
            (address.strip(), exchange.lower(), datetime.now().isoformat())
        )
        is_new = c.rowcount > 0
        conn.commit()
        return is_new
    except:
        return False
    finally:
        conn.close()

def add_addresses_bulk(exchange: str, addresses: list, source: str) -> int:
    """Add multiple addresses. Returns count of new addresses."""
    if not addresses:
        return 0
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    added = 0
    now = datetime.now().isoformat()
    for addr in addresses:
        try:
            c.execute(
                'INSERT OR IGNORE INTO addresses (address, exchange, downloaded_at) VALUES (?, ?, ?)',
                (addr.strip(), exchange.lower(), now)
            )
            if c.rowcount > 0:
                added += 1
        except:
            pass
    conn.commit()
    conn.close()
    return added

def extract_btc_addresses(text: str) -> list:
    """Extract all Bitcoin addresses from text."""
    addresses = []
    # Legacy (1...)
    addresses += re.findall(r'\b(1[a-km-zA-HJ-NP-Z1-9]{25,34})\b', text)
    # P2SH (3...)
    addresses += re.findall(r'\b(3[a-km-zA-HJ-NP-Z1-9]{25,34})\b', text)
    # Bech32 (bc1q...)
    addresses += re.findall(r'\b(bc1q[a-z0-9]{38,58})\b', text)
    # Taproot (bc1p...)
    addresses += re.findall(r'\b(bc1p[a-z0-9]{38,58})\b', text)
    return list(set(addresses))

def hunt_bitinfocharts():
    """Hunt addresses from BitInfoCharts."""
    log.info("=== BITINFOCHARTS HUNT ===")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    total_found = 0

    for exchange in TARGET_EXCHANGES:
        for pattern in BITINFOCHARTS_PATTERNS:
            wallet_name = pattern.format(exchange=exchange)
            url = f'https://bitinfocharts.com/bitcoin/wallet/{wallet_name}'

            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 200 and 'Address' in resp.text:
                    addresses = extract_btc_addresses(resp.text)
                    # Filter out navigation/example addresses
                    addresses = [a for a in addresses if len(a) >= 26]

                    if addresses:
                        added = add_addresses_bulk(exchange, addresses, f'bitinfocharts:{wallet_name}')
                        if added > 0:
                            log.info(f"  {exchange}/{wallet_name}: +{added} addresses")
                            total_found += added

            except Exception as e:
                pass  # Silent fail for 404s etc

            time.sleep(1.5)  # Rate limit

    log.info(f"BitInfoCharts total: +{total_found} addresses")
    return total_found

def hunt_blockchair():
    """Hunt addresses from Blockchair labels."""
    log.info("=== BLOCKCHAIR HUNT ===")

    # Blockchair has a labels API but requires auth
    # Try public endpoints
    total_found = 0

    # Known Blockchair exchange label pages
    exchange_ids = {
        'binance': 'binance',
        'coinbase': 'coinbase',
        'kraken': 'kraken',
        'bitfinex': 'bitfinex',
        'bitstamp': 'bitstamp',
        'gemini': 'gemini',
        'okx': 'okex',
        'huobi': 'huobi',
        'kucoin': 'kucoin',
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    for exchange, label in exchange_ids.items():
        try:
            # Try Blockchair's privacy-o-meter which sometimes shows exchange addresses
            url = f'https://blockchair.com/bitcoin/address/{label}'
            # This won't work directly but leaving structure for future
            time.sleep(2)
        except:
            pass

    log.info(f"Blockchair total: +{total_found} addresses")
    return total_found

def hunt_walletexplorer():
    """Hunt from WalletExplorer API."""
    log.info("=== WALLETEXPLORER HUNT ===")

    # Exchanges available on WalletExplorer
    we_exchanges = [
        'Binance.com', 'Bitfinex.com', 'Bitstamp.net', 'Bittrex.com',
        'Huobi.com', 'Kraken.com', 'OKCoin.com', 'Poloniex.com',
        'BTC-e.com', 'LocalBitcoins.com', 'Coinbase.com',
    ]

    total_found = 0

    for we_name in we_exchanges:
        try:
            # WalletExplorer API
            url = f'https://www.walletexplorer.com/api/1/wallet-addresses?wallet={we_name}&from=0&count=10000'
            headers = {'User-Agent': 'Mozilla/5.0'}

            resp = requests.get(url, headers=headers, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('found'):
                    addresses = data.get('addresses', [])
                    # Extract just the address strings
                    addr_list = [a.get('address') for a in addresses if a.get('address')]

                    exchange = we_name.split('.')[0].lower()
                    added = add_addresses_bulk(exchange, addr_list, f'walletexplorer:{we_name}')
                    if added > 0:
                        log.info(f"  {exchange}: +{added} addresses")
                        total_found += added

            time.sleep(3)  # Be nice to WalletExplorer

        except Exception as e:
            log.error(f"WalletExplorer {we_name} error: {e}")

    log.info(f"WalletExplorer total: +{total_found} addresses")
    return total_found

def hunt_blockchain_clustering():
    """Cluster addresses using blockchain.info transaction data."""
    log.info("=== BLOCKCHAIN CLUSTERING ===")

    # Get seed addresses from database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT DISTINCT address, exchange FROM addresses ORDER BY RANDOM() LIMIT 100')
    seeds = c.fetchall()
    conn.close()

    total_found = 0

    for seed_addr, exchange in seeds:
        try:
            # Get transactions for this address
            url = f'https://blockchain.info/rawaddr/{seed_addr}?limit=20'
            headers = {'User-Agent': 'Mozilla/5.0'}

            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()

                new_addrs = set()
                for tx in data.get('txs', []):
                    # Get input addresses (common-input-ownership heuristic)
                    input_addrs = []
                    for inp in tx.get('inputs', []):
                        prev_out = inp.get('prev_out', {})
                        addr = prev_out.get('addr')
                        if addr:
                            input_addrs.append(addr)

                    # If our seed address is an input, all other inputs are same entity
                    if seed_addr in input_addrs:
                        for addr in input_addrs:
                            if addr != seed_addr:
                                new_addrs.add(addr)

                if new_addrs:
                    added = add_addresses_bulk(exchange, list(new_addrs), 'cluster')
                    if added > 0:
                        log.info(f"  Clustered from {seed_addr[:16]}...: +{added} addresses for {exchange}")
                        total_found += added

            time.sleep(5)  # Rate limit blockchain.info

        except Exception as e:
            pass  # Silent fail

    log.info(f"Clustering total: +{total_found} addresses")
    return total_found

def hunt_proof_of_reserves():
    """Try to get addresses from Proof of Reserves pages."""
    log.info("=== PROOF OF RESERVES HUNT ===")

    por_sources = [
        # OKX
        ('okx', 'https://www.okx.com/proof-of-reserves'),
        ('okx', 'https://static.okx.com/cdn/okx/por/okx_por_addresses.csv'),
        # Bybit
        ('bybit', 'https://www.bybit.com/app/user/proof-of-reserve'),
        # Bitget
        ('bitget', 'https://www.bitget.com/proof-of-reserves'),
        # Gate.io
        ('gate.io', 'https://www.gate.io/proof-of-reserves'),
        # KuCoin
        ('kucoin', 'https://www.kucoin.com/proof-of-reserves'),
    ]

    total_found = 0
    headers = {'User-Agent': 'Mozilla/5.0'}

    for exchange, url in por_sources:
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                addresses = extract_btc_addresses(resp.text)
                if addresses:
                    added = add_addresses_bulk(exchange, addresses, f'por:{url}')
                    if added > 0:
                        log.info(f"  {exchange} PoR: +{added} addresses")
                        total_found += added
        except:
            pass
        time.sleep(2)

    log.info(f"PoR total: +{total_found} addresses")
    return total_found

def get_stats():
    """Get database statistics."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT exchange, COUNT(*) as cnt
        FROM addresses
        GROUP BY exchange
        ORDER BY cnt DESC
    ''')
    stats = c.fetchall()
    c.execute('SELECT COUNT(*) FROM addresses')
    total = c.fetchone()[0]
    conn.close()
    return stats, total

def main():
    """Main hunting loop."""
    log.info("=" * 70)
    log.info("DEEP ADDRESS HUNTER STARTED")
    log.info(f"Time: {datetime.now()}")
    log.info("=" * 70)

    init_db()

    # Print starting stats
    stats, total = get_stats()
    log.info(f"\nStarting with {total:,} addresses:")
    for exchange, count in stats[:15]:
        log.info(f"  {exchange}: {count:,}")

    round_num = 0
    while True:
        round_num += 1
        log.info(f"\n{'='*70}")
        log.info(f"ROUND {round_num} - {datetime.now()}")
        log.info(f"{'='*70}")

        round_total = 0

        # Run all hunters
        round_total += hunt_walletexplorer()
        round_total += hunt_bitinfocharts()
        round_total += hunt_proof_of_reserves()
        round_total += hunt_blockchain_clustering()

        # Print round summary
        stats, total = get_stats()
        log.info(f"\nRound {round_num} complete: +{round_total} new addresses")
        log.info(f"Total in database: {total:,}")
        log.info("\nTop exchanges:")
        for exchange, count in stats[:10]:
            log.info(f"  {exchange}: {count:,}")

        # Sleep between rounds (1 hour)
        log.info(f"\nSleeping 1 hour until next round...")
        time.sleep(3600)

if __name__ == '__main__':
    main()
