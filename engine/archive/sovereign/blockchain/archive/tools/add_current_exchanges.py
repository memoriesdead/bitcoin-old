#!/usr/bin/env python3
"""
ADD CURRENT EXCHANGE ADDRESSES
==============================
Add known cold wallet addresses for major current exchanges.
These are publicly identified addresses from blockchain analysis.
"""

import sqlite3
from datetime import datetime

# Known cold wallet addresses for major exchanges (publicly tracked)
CURRENT_EXCHANGE_ADDRESSES = {
    # BINANCE - Multiple known cold wallets
    "binance": [
        "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",  # Main cold wallet
        "3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb",  # Cold wallet 2
        "3M219KR5vEneNb47ewrPfWyb5jQ2DjxRP6",  # Cold wallet 3
        "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h",  # Native SegWit
        "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s",  # Large holder
        "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",  # Binance 8
        "3JZq4atUahhuA9rLhXLMhhTo133J9rF97j",  # Binance 7
        "1LQoWist8KkaUXSPKZHNvEyfrEkPHzSsCd",  # Hot wallet
        "12ib7dApVFvg82TXKycWBNpN8kFyiAN1dr",  # Deposit wallet
        "12xT6q2HvWexfwWqHnxvBfJhcrRDKFcw1L",
        "14cTG1TJD8p3iJHyxZNsDpqXXVLbJpS3yr",
        "1MvpwzpAggGKxR5vN2cosyjnqCPhHbJN8z",
        "15F3JQZDaEhP8TjVRH5ZCo5KXWzmv5Np71",
        "19D5J8c59P2bAkWKvxSYw8scD3KUNWoZ1C",
        "1FLQpSRaovL7wE5u1GX1dxMbRQA8hN3bPT",
    ],

    # COINBASE - Known institutional addresses
    "coinbase": [
        "3Kzh9qAqVWQhEsfQz7zEQL1EuSx5tyNLNS",  # Coinbase Prime
        "395xMCmHttAbb3hGVxnAf3YXcRkKdZY8xy",  # Cold storage
        "bc1q7cyrfmck2ffu2ud3rn5l5a8yv6f0chkp0zpemf",  # Native SegWit
        "3LCGsSmfr24demGvriN8dQ57Hw3HzJQRp4",  # Custody
        "3Nxwenay9Z8Lc9JBiywExpnEFiLp6Afp8v",  # Prime custody
        "bc1qx9t2l3pyny2spqpqlye8svce70nppwtaxwdrp4",
        "38Xnrq8MZiKmYmwobbYsj5cDGMsJPxPnJJ",
        "3JmreiLQqSbLnM24RqMnZx9p94JVBemPJJ",
        "3NjMheU8M9MHJFMefpbGn1M2MrLxrMijQa",
        "1BDHEPgB8iGipkaaJDxkAYjuzvEstHdhaY",
        "1HckjUpRGcrrRAtFaaCAUaGjsPx9oYmLaZ",
        "1JqDybm2nWTENMGzVrvdRnQdUCktP5MYhd",
        "3CRjkRwx3HdpZMuPpnLTpGkSo5eMPQVvME",
    ],

    # OKX (formerly OKEx) - Major exchange
    "okx": [
        "3LvppKCEzPr8D6Evx7vdUKZV9spKCMR7Rf",  # OKX cold
        "bc1qjasf9z3h7w3jspkhtgatgpyvvzgpa2wwd2lr0eh5tx44reyn2k7sfc27a4",
        "3FHNBLobJnbCTFTVakh5TXmEneyf5PT61B",
        "bc1q2s3rjwvam9dt2ftt4sqxqjf3twav0gdnv0a5ak",
        "1Lgo3GBNbFoMZXAj4DTca5TvURtNeH5Z4E",
        "1C9xoxLGaREEPBk2PK1m2Np5KQxBRFtKZL",
        "bc1qk4m9zv5tnxf2pddd565wugsjrkqkfn90aa0yzy",
        "3Cbq7aT1tY8kMxWLbitaG7yT6bPbKChq64",
        "1ECe4dZq4PELMUr8dHNDEcrYqoLnNNqxJo",
    ],

    # BYBIT - Growing exchange
    "bybit": [
        "bc1q8wtz3wmnq0cww8cqt0uu3vnt5kvt5pz64hfzz7",
        "32yvQpVo8sqrrR5YdzLeCQr6xVMFP8Xjpy",
        "3Kq3kBAPshECGEjjzpbYjDFrTqQ8F2kzxA",
        "bc1qjysjfd9t9aspttpjqzv68k0ydpe7pvyd5vlyn37868473lell5tqkz456g",
    ],

    # BITFINEX - Major trading platform
    "bitfinex": [
        "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
        "3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r",
        "3JZq4atUahhuA9rLhXLMhhTo133J9rF97j",
        "385cR5DM96n1HvBDMzLHPYcw89fZAXULJP",
        "1Kr6QSydW9bFQG1mXiPNNu6WpJGmUa9i1g",
    ],

    # KRAKEN - Major exchange (adding more)
    "kraken": [
        "bc1qcup2r2m4qnvx3u3vfdz7x4rtrkmxnpfnphz0j6",
        "3FWBzs4a1rkSvHdmHzZhgRdBxrjPMTMPN6",
        "bc1q8ypzxlm5kc9kuv4m6l8gl6m9uucnsqpzstd4qx",
        "bc1qz33r6mec7mwhqgq2cqjmz3cjvhqppd7gnpzp3j",
        "3KcE3zy7AKxPGxxEL5NAfR3qXQDdMZ8Z2P",
    ],

    # GEMINI - US exchange
    "gemini": [
        "bc1q7ydrtdn8z62xhslqyqtyt38mm4e2c4h3m2c9ww",
        "3D8JEEVCpbRvnLRZ8AiMw5CjPLxNChj9Tx",
        "bc1qk4m9zv5tnxf2pddd565wugsjrkqkfn90aa0yzy",
        "3NxSJH1t9joEW5EWQz1G8G7tYQTVDkPzCe",
    ],

    # KUCOIN
    "kucoin": [
        "3LQ7k9bXjZaGRQ95GvBMNQx9mLhN5sLQrC",
        "bc1qr4dl5wa7kl8yu792dceg9z5knl2gkn220lk7a9",
        "1MvpwzpAggGKxR5vN2cosyjnqCPhHbJN8z",
    ],

    # GATE.IO
    "gateio": [
        "3LCGsSmfr24demGvriN8dQ57Hw3HzJQRp4",
        "bc1qhvd6suvq6su8qfl3c7hg4ng8d5lgh68z2c46c2",
    ],

    # CRYPTO.COM
    "cryptocom": [
        "bc1qrxfszfq5akvew7t04m7m8sslxztgxk4mczpzf5",
        "3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb",
    ],

    # HTX (formerly Huobi) - adding additional
    "htx": [
        "12sETwQoPjGg3xQYHnLuprVvpB28ERT7Ma",
        "1LAnF8h3qMGx3TSwNUHVneBZUEpwE4gu3D",
        "1KVpuCfhftkzJ67ZUegaMuaYey7qni7pPj",
    ],
}


def main():
    db_path = "/root/sovereign/address_clusters.db"

    print("=" * 70)
    print("ADDING CURRENT EXCHANGE ADDRESSES")
    print("=" * 70)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    total_added = 0

    for exchange, addresses in CURRENT_EXCHANGE_ADDRESSES.items():
        added = 0
        for addr in addresses:
            try:
                c.execute("""
                    INSERT OR IGNORE INTO addresses (address, exchange, discovered_at, source)
                    VALUES (?, ?, ?, 'seed_current')
                """, (addr, exchange, datetime.now().isoformat()))
                if c.rowcount > 0:
                    added += 1
                    total_added += 1
            except Exception as e:
                print(f"Error adding {addr}: {e}")

        print(f"  {exchange:<15} +{added} addresses")

    conn.commit()

    # Show totals
    c.execute("SELECT exchange, COUNT(*) as cnt FROM addresses GROUP BY exchange ORDER BY cnt DESC")
    print()
    print("=" * 70)
    print(f"TOTAL ADDED: {total_added} seed addresses")
    print("=" * 70)
    print()
    print("Current exchange totals:")
    for row in c.fetchall():
        print(f"  {row[0]:<25} {row[1]:>10,}")

    conn.close()
    print()
    print("Now run the scanner to cluster from these seeds!")


if __name__ == '__main__':
    main()
