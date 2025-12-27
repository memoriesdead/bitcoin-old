#!/usr/bin/env python3
"""
Download Exchange Address Database

Downloads 30+ million labeled Bitcoin addresses from EntityAddressBitcoin.
Categories: Exchanges, Mining Pools, Gambling, Services

Usage:
    python -m engine.sovereign.backtest.download_exchanges
"""
import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError
import csv

# EntityAddressBitcoin dataset
# Contains 30,331,700 labeled addresses
# Hosted on Swiss academic cloud
ENTITY_ADDRESS_URL = "https://drive.switch.ch/index.php/s/ag4OnNgwf7LhWFu/download"

# Major exchange cold wallets (verified, high-balance)
MAJOR_EXCHANGES = {
    "binance": [
        "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",  # Cold wallet 1 (137K BTC)
        "3FrSzikNqBgikWgTHixywhXcx57q6H6rHC",  # Cold wallet 2
        "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h",  # Newer wallet
    ],
    "bitfinex": [
        "3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r",  # 138K BTC
        "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
    ],
    "huobi": [
        "3Cbq7aT1tY8kMxWLbitaG7yT6bPbKChq64",  # 108K BTC
    ],
    "bittrex": [
        "16rCmCmbuWDhPjWTrpQGaU3EPdZF7MTdUk",  # 107K BTC
    ],
    "bitstamp": [
        "3Nxwenay9Z8Lc9JBiywExpnEFiLp6Afp8v",  # 97K BTC
    ],
    "kraken": [
        "1AnwDVbwsLBVwRfqN2x9Eo4YEJSPXo2cwG",  # 23K BTC
        "14eQD1QQb8QFVG8YFwGz7skyzsvBLWLwJS",
        "1A7znRYE24Z6K8MCAKXLmEvuS5ixzvUrjH",
    ],
    "coinbase": [
        "1FzWLkAahHooV3kzTgyx6qsswXJ6sCXkSR",
        "3Kzh9qAqVWQhEsfQz7zEQL1EuSx5tyNLNS",
        "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
    ],
    "gemini": [
        "1LQoWist8KkaUXSPKZHNvEyfrEkPHzSsCd",
    ],
    "poloniex": [
        "17A16QmavnUfCW11DAApiJxp7ARnxN5pGX",  # 7K BTC
    ],
    "okex": [
        "1HckjUpRGcrrRAtFaaCAUaGjsPx9oYmLaZ",
    ],
    "kucoin": [
        "3GS8pDqKzRh41xU7VPUvXcRAR58fZoLhGb",
    ],
    "ftx": [  # Historical (collapsed Nov 2022)
        "1FzWLkAahHooV3kzTgyx6qsswXJ6sCXkSR",
    ],
    "coincheck": [
        "336xGpGweq1wtY4kRTuA4w6d7yDkBU9czU",  # 30K BTC
    ],
}


def download_with_progress(url: str, dest: Path):
    """Download file with progress."""
    def hook(count, block, total):
        if total > 0:
            pct = min(100, count * block * 100 / total)
            mb = count * block / (1024 * 1024)
            sys.stdout.write(f"\r  [{pct:5.1f}%] {mb:.1f} MB")
            sys.stdout.flush()

    try:
        urlretrieve(url, dest, reporthook=hook)
        print()
        return True
    except Exception as e:
        print(f"\n[!] Error: {e}")
        return False


def try_github_download():
    """Try to download from EntityAddressBitcoin GitHub."""
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    zip_path = output_dir / "entity_addresses.zip"

    print("[1] Downloading EntityAddressBitcoin dataset...")
    print(f"    URL: {ENTITY_ADDRESS_URL}")

    # Try direct download
    try:
        if download_with_progress(ENTITY_ADDRESS_URL, zip_path):
            print("[+] Download complete")

            # Extract
            print("[2] Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(output_dir / "entity_addresses")
            print("[+] Extracted")

            return True
    except Exception as e:
        print(f"[!] GitHub download failed: {e}")

    return False


def load_entity_addresses(data_dir: Path) -> dict:
    """Load addresses from extracted CSV files."""
    addresses = {}

    csv_files = list(data_dir.glob("*.csv"))
    print(f"[+] Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        print(f"    Loading {csv_file.name}...")
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header

                for row in reader:
                    if len(row) >= 3:
                        addr = row[0]
                        entity = row[2]  # Entity name
                        # Normalize exchange names
                        entity_lower = entity.lower()
                        if any(x in entity_lower for x in ['exchange', 'binance', 'coinbase', 'kraken', 'bitfinex', 'bitstamp', 'huobi', 'okex', 'bittrex', 'poloniex']):
                            addresses[addr] = entity

        except Exception as e:
            print(f"    [!] Error reading {csv_file}: {e}")

    return addresses


def main():
    output_file = Path("data/exchanges.json")

    print("=" * 60)
    print("EXCHANGE ADDRESS DOWNLOADER")
    print("=" * 60)

    # Start with major exchanges
    all_addresses = {}

    print("\n[1] Loading verified major exchange wallets...")
    for exchange, addrs in MAJOR_EXCHANGES.items():
        for addr in addrs:
            all_addresses[addr] = exchange
    print(f"    Loaded {len(all_addresses)} verified addresses")

    # Try to get more from EntityAddressBitcoin
    print("\n[2] Attempting to download EntityAddressBitcoin (30M addresses)...")

    entity_dir = Path("data/entity_addresses")
    if entity_dir.exists():
        print("    Found existing download, loading...")
        entity_addrs = load_entity_addresses(entity_dir)
        all_addresses.update(entity_addrs)
    else:
        if try_github_download():
            entity_addrs = load_entity_addresses(entity_dir)
            all_addresses.update(entity_addrs)
        else:
            print("    [!] Could not download EntityAddressBitcoin")
            print("    [!] Manual download: https://github.com/Maru92/EntityAddressBitcoin")
            print("    [!] Using verified major exchanges only")

    # Save combined database
    print(f"\n[3] Saving {len(all_addresses):,} addresses to {output_file}...")

    # Convert to exchange -> [addresses] format
    exchange_db = {}
    for addr, exchange in all_addresses.items():
        exchange = exchange.lower().replace(' ', '_')
        if exchange not in exchange_db:
            exchange_db[exchange] = []
        exchange_db[exchange].append(addr)

    with open(output_file, 'w') as f:
        json.dump(exchange_db, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total addresses: {len(all_addresses):,}")
    print(f"Exchanges: {len(exchange_db)}")
    print(f"Output: {output_file.absolute()}")
    print("=" * 60)

    # Summary by exchange
    print("\nTop exchanges by address count:")
    sorted_exchanges = sorted(exchange_db.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for exchange, addrs in sorted_exchanges:
        print(f"  {exchange}: {len(addrs):,} addresses")


if __name__ == '__main__':
    main()
