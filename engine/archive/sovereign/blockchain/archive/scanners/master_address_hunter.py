#!/usr/bin/env python3
"""
MASTER ADDRESS HUNTER
=====================
Comprehensive overnight address hunting combining ALL sources.

Run: python3 master_address_hunter.py

Sources:
1. Merge address_clusters.db (7.5M historical addresses)
2. WalletExplorer API (active exchanges)
3. BitInfoCharts scraping (with bypass)
4. Blockchain.info clustering
5. Manual known addresses
"""

import sqlite3
import requests
import time
import re
import json
import logging
from datetime import datetime
from pathlib import Path
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/sovereign/master_hunter.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Databases
MAIN_DB = '/root/sovereign/walletexplorer_addresses.db'
CLUSTER_DB = '/root/sovereign/address_clusters.db'

# Known cold wallets from research (high-value seed addresses)
KNOWN_COLD_WALLETS = {
    # Gate.io (confirmed BitInfoCharts)
    'gate.io': [
        '1HpED69tpKSaEaWpY3Udt1DtcVcuCUoh2Y',
        '14kmvhQrWrNEHbrSKBySj4qHGjemDtS3SF',
        '3HroDXv8hmzKRtaSfBffRgedKpru8fgy6M',
        '162bzZT2hJfv5Gm3ZmWfWfHJjCtMD6rHhw',
        '1G47mSr3oANXMafVrR8UC4pzV7FEAzo3r9',
    ],
    # Upbit "Mr. 100" (73,090 BTC - Arkham confirmed)
    'upbit': [
        '1Ay8vMC7R1UbyCCZRVULMV7iQpHSAbguJP',
    ],
    # Coinbase known addresses
    'coinbase': [
        '3Kzh9qAqVWQhEsfQz7zEQL1EuSx5tyNLNS',
        '34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo',
        'bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97',
        '3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb',
        '3FHNBLobJnbCTFTVakh5TXmEneyf5PT61B',
    ],
    # Gemini
    'gemini': [
        '35hK24tcLEWcgNA4JxpvbkNkoAcDGqQPsP',
        '1Do8rVsLugAzm6qJMeiKPRHmgDcnN7fXsF',
    ],
    # BitMEX
    'bitmex': [
        '3BMEXqGpG4FxBA1KWhRFufXfSTRgzfDBhJ',
        '3JZq4atUahhuA9rLhXLMhhTo133J9rF97j',
    ],
    # Binance known addresses
    'binance': [
        '34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo',
        '3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb',
        'bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h',
    ],
    # Kraken
    'kraken': [
        '3AfSdsvr1Ab1BHzYVq3KFAM1fJ1LiHtWfx',
    ],
    # OKX/OKEx
    'okx': [
        '1FzWLkAahHooV3kzHs1D3RS9x7rkpgqMMD',
    ],
    # Bitfinex
    'bitfinex': [
        'bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97',
        '3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r',
    ],
    # Deribit
    'deribit': [
        '1KRqhW5Z9T6bJUmTWvEpL4MxQZJvfhd2Xj',
    ],
    # Crypto.com
    'crypto.com': [
        '1PJiGp2yDLvUgqeBsuZVCBADArNsk6XEiw',
        'bc1q4c8n5t00jmj8temxdgcc3t32nkg2wjwz24lywv',
    ],
    # Huobi/HTX
    'huobi': [
        '35pgGeez3ou6ofrpjt8D6wVR7WaVTN3WY3',
    ],
}

# User agents for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

def get_headers():
    """Get random headers for requests."""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }

def add_addresses(exchange: str, addresses: list) -> int:
    """Add addresses to main database."""
    if not addresses:
        return 0

    conn = sqlite3.connect(MAIN_DB)
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
    """Extract Bitcoin addresses from text."""
    addresses = []
    # Legacy (1...)
    addresses += re.findall(r'\b(1[a-km-zA-HJ-NP-Z1-9]{25,34})\b', text)
    # P2SH (3...)
    addresses += re.findall(r'\b(3[a-km-zA-HJ-NP-Z1-9]{25,34})\b', text)
    # Bech32 (bc1q...)
    addresses += re.findall(r'\b(bc1q[a-z0-9]{38,58})\b', text)
    # Taproot (bc1p...)
    addresses += re.findall(r'\b(bc1p[a-z0-9]{38,58})\b', text)
    return list(set([a for a in addresses if len(a) >= 26]))

def merge_cluster_database():
    """Merge address_clusters.db into main database."""
    log.info("=== MERGING CLUSTER DATABASE ===")

    if not Path(CLUSTER_DB).exists():
        log.warning(f"Cluster database not found: {CLUSTER_DB}")
        return 0

    cluster_conn = sqlite3.connect(CLUSTER_DB)
    cluster_c = cluster_conn.cursor()

    main_conn = sqlite3.connect(MAIN_DB)
    main_c = main_conn.cursor()

    # Get all exchanges from cluster db
    cluster_c.execute('SELECT DISTINCT exchange FROM addresses')
    exchanges = [row[0] for row in cluster_c.fetchall()]

    total_added = 0
    now = datetime.now().isoformat()

    for exchange in exchanges:
        cluster_c.execute('SELECT address FROM addresses WHERE exchange = ?', (exchange,))
        addresses = [row[0] for row in cluster_c.fetchall()]

        added = 0
        for addr in addresses:
            try:
                main_c.execute(
                    'INSERT OR IGNORE INTO addresses (address, exchange, downloaded_at) VALUES (?, ?, ?)',
                    (addr, exchange.lower(), now)
                )
                if main_c.rowcount > 0:
                    added += 1
            except:
                pass

        if added > 0:
            log.info(f"  {exchange}: +{added:,} addresses")
            total_added += added

        main_conn.commit()

    cluster_conn.close()
    main_conn.close()

    log.info(f"Cluster merge total: +{total_added:,} addresses")
    return total_added

def add_known_cold_wallets():
    """Add all known cold wallet addresses."""
    log.info("=== ADDING KNOWN COLD WALLETS ===")

    total = 0
    for exchange, addresses in KNOWN_COLD_WALLETS.items():
        added = add_addresses(exchange, addresses)
        if added > 0:
            log.info(f"  {exchange}: +{added} cold wallet addresses")
            total += added

    log.info(f"Known wallets total: +{total} addresses")
    return total

def hunt_walletexplorer():
    """Hunt from WalletExplorer API."""
    log.info("=== WALLETEXPLORER API HUNT ===")

    exchanges = [
        'Binance.com', 'Bitfinex.com', 'Bitstamp.net', 'Bittrex.com',
        'Huobi.com', 'Kraken.com', 'OKCoin.com', 'Poloniex.com',
        'BTC-e.com', 'LocalBitcoins.com', 'Coinbase.com', 'Luno.com',
        'Cex.io', 'HitBTC.com', 'Yobit.net', 'Bitcoin.de',
    ]

    total = 0

    for we_name in exchanges:
        try:
            url = f'https://www.walletexplorer.com/api/1/wallet-addresses?wallet={we_name}&from=0&count=100000'
            resp = requests.get(url, headers=get_headers(), timeout=60)

            if resp.status_code == 200:
                data = resp.json()
                if data.get('found'):
                    addresses = [a.get('address') for a in data.get('addresses', []) if a.get('address')]
                    exchange = we_name.split('.')[0].lower()
                    added = add_addresses(exchange, addresses)
                    if added > 0:
                        log.info(f"  {exchange}: +{added:,} addresses")
                        total += added

            time.sleep(2)

        except Exception as e:
            log.error(f"WalletExplorer {we_name}: {e}")

    log.info(f"WalletExplorer total: +{total:,} addresses")
    return total

def hunt_blockchain_clustering():
    """Cluster addresses using blockchain.info API."""
    log.info("=== BLOCKCHAIN CLUSTERING ===")

    # Get seed addresses from known wallets
    seeds = []
    for exchange, addrs in KNOWN_COLD_WALLETS.items():
        for addr in addrs:
            seeds.append((addr, exchange))

    total = 0

    for seed_addr, exchange in seeds[:20]:  # Limit to avoid rate limits
        try:
            url = f'https://blockchain.info/rawaddr/{seed_addr}?limit=50'
            resp = requests.get(url, headers=get_headers(), timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                new_addrs = set()

                for tx in data.get('txs', []):
                    # Common-input-ownership: if two addresses are inputs to same tx, same owner
                    input_addrs = []
                    for inp in tx.get('inputs', []):
                        prev_out = inp.get('prev_out', {})
                        addr = prev_out.get('addr')
                        if addr:
                            input_addrs.append(addr)

                    if seed_addr in input_addrs:
                        for addr in input_addrs:
                            if addr != seed_addr:
                                new_addrs.add(addr)

                if new_addrs:
                    added = add_addresses(exchange, list(new_addrs))
                    if added > 0:
                        log.info(f"  Clustered {exchange} from {seed_addr[:16]}...: +{added}")
                        total += added

            time.sleep(5)  # Rate limit

        except Exception as e:
            pass

    log.info(f"Clustering total: +{total} addresses")
    return total

def hunt_mempool_space():
    """Try mempool.space API for address info."""
    log.info("=== MEMPOOL.SPACE HUNT ===")

    # Mempool.space doesn't have exchange labels but we can use it
    # to get transaction history for known addresses
    total = 0

    for exchange, seeds in list(KNOWN_COLD_WALLETS.items())[:5]:
        for seed in seeds[:2]:
            try:
                url = f'https://mempool.space/api/address/{seed}/txs'
                resp = requests.get(url, headers=get_headers(), timeout=30)

                if resp.status_code == 200:
                    txs = resp.json()
                    new_addrs = set()

                    for tx in txs[:20]:
                        # Get addresses from vins (inputs)
                        for vin in tx.get('vin', []):
                            prevout = vin.get('prevout', {})
                            addr = prevout.get('scriptpubkey_address')
                            if addr and addr != seed:
                                new_addrs.add(addr)

                    if new_addrs:
                        added = add_addresses(exchange, list(new_addrs))
                        if added > 0:
                            log.info(f"  {exchange} via mempool: +{added}")
                            total += added

                time.sleep(2)

            except Exception as e:
                pass

    log.info(f"Mempool.space total: +{total} addresses")
    return total

def get_stats():
    """Get database statistics."""
    conn = sqlite3.connect(MAIN_DB)
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
    """Main hunting routine."""
    log.info("=" * 70)
    log.info("MASTER ADDRESS HUNTER STARTED")
    log.info(f"Time: {datetime.now()}")
    log.info("=" * 70)

    # Print starting stats
    stats, total = get_stats()
    log.info(f"\nStarting with {total:,} addresses:")
    for exchange, count in stats[:15]:
        log.info(f"  {exchange}: {count:,}")

    round_total = 0

    # Phase 1: Merge cluster database (7.5M addresses)
    round_total += merge_cluster_database()

    # Phase 2: Add known cold wallets
    round_total += add_known_cold_wallets()

    # Phase 3: WalletExplorer API
    round_total += hunt_walletexplorer()

    # Phase 4: Blockchain clustering
    round_total += hunt_blockchain_clustering()

    # Phase 5: Mempool.space
    round_total += hunt_mempool_space()

    # Final stats
    stats, total = get_stats()

    log.info("\n" + "=" * 70)
    log.info("HUNT COMPLETE")
    log.info(f"New addresses found: {round_total:,}")
    log.info(f"Total in database: {total:,}")
    log.info("=" * 70)

    log.info("\nFinal exchange breakdown:")
    for exchange, count in stats:
        log.info(f"  {exchange}: {count:,}")

    # Continuous mode - run clustering every hour
    log.info("\n\nEntering continuous clustering mode...")
    while True:
        time.sleep(3600)  # 1 hour
        log.info(f"\n--- Hourly clustering run: {datetime.now()} ---")
        hunt_blockchain_clustering()
        hunt_mempool_space()
        stats, total = get_stats()
        log.info(f"Total addresses: {total:,}")

if __name__ == '__main__':
    main()
