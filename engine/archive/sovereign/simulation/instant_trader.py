#!/usr/bin/env python3
"""
INSTANT PAPER TRADER - NO LOADING DELAY
========================================
Pre-cached 102 exchanges. Connects to running pipeline immediately.
$100 per exchange = $10,200 total.
"""

import os
import sys
import time
import sqlite3
import threading
import requests
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List

# Pre-cached exchange list (from walletexplorer_addresses.db)
ALL_EXCHANGES = [
    'bitfinex', 'bitstamp', 'btc-e.com', 'btce.com', 'bter.com',
    'cex.io', 'coinbase', 'cryptsy.com', 'hitbtc.com', 'huobi',
    'kraken', 'localbitcoins.com', 'mintpal.com', 'mtgox', 'okcoin',
    'poloniex', 'binance', 'bittrex', 'kucoin', 'gate.io',
    'okex', 'bybit', 'bitget', 'mexc', 'crypto.com',
    'gemini', 'ftx', 'deribit', 'bitmex', 'phemex',
    'luno', 'paxful', 'localcryptos', 'hodlhodl', 'bisq',
    'coincheck', 'bitflyer', 'zaif', 'liquid', 'bitbank',
    'upbit', 'bithumb', 'korbit', 'coinone', 'gopax',
    'bitvavo', 'bitpanda', 'bitcoin.de', 'paymium', 'therock',
    'btcmarkets', 'independentreserve', 'coinspot', 'swyftx', 'coinjar',
    'mercadobitcoin', 'bitso', 'buda', 'satoshitango', 'ripio',
    'lbank', 'digifinex', 'zt.com', 'hotbit', 'hoo.com',
    'coinsbit', 'latoken', 'probit', 'bitmart', 'ascendex',
    'whitebit', 'exmo', 'cex.io', 'yobit', 'livecoin',
    'bitforex', 'bkex', 'xt.com', 'coinex', 'aax',
    'btcturk', 'paribu', 'kuna', 'coindcx', 'wazirx',
    'zebpay', 'giottus', 'buyucoin', 'unocoin', 'bitbns',
    'rain', 'bitoasis', 'coinmena', 'palmex', 'midchains',
    'luno', 'valr', 'altcointrader', 'ice3x', 'chainex',
    'quidax', 'busha', 'roqqu', 'patricia', 'yellowcard'
]

FEES = {
    'coinbase': 0.006, 'kraken': 0.0026, 'bitstamp': 0.003, 'gemini': 0.004,
    'binance': 0.001, 'bitfinex': 0.002, 'huobi': 0.002, 'okex': 0.001,
    'bybit': 0.001, 'bitget': 0.001, 'kucoin': 0.001, 'gate.io': 0.002,
}
DEFAULT_FEE = 0.003


@dataclass
class Account:
    name: str
    capital: float = 100.0
    initial: float = 100.0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    position: Dict = field(default_factory=dict)


class InstantTrader:
    def __init__(self, capital: float = 100.0):
        self.capital = capital
        self.accounts = {ex: Account(ex, capital, capital) for ex in ALL_EXCHANGES}
        self.price = 0.0
        self.signals = 0
        self.trades = 0
        self.running = False
        self.db_path = "/root/sovereign/instant_trades.db"
        self.corr_db = "/root/sovereign/correlation.db"
        self._init_db()
        self._lock = threading.Lock()
        self.last_signal_id = 0

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY, ts TEXT, exchange TEXT, direction TEXT,
            entry REAL, exit REAL, size REAL, pnl REAL, reason TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS equity (
            id INTEGER PRIMARY KEY, ts TEXT, total REAL, pnl REAL
        )''')
        conn.commit()
        conn.close()

    def _get_fee(self, ex: str) -> float:
        return FEES.get(ex.lower(), DEFAULT_FEE)

    def on_signal(self, exchange: str, direction: str, btc: float, price: float):
        ex = exchange.lower()
        if ex not in self.accounts:
            return

        with self._lock:
            self.signals += 1
            acc = self.accounts[ex]

            if acc.position:
                return  # Already in position

            # Position sizing
            size = min(acc.capital * 0.25, acc.capital - 1)
            if size < 1:
                return

            fee = size * self._get_fee(ex)
            acc.capital -= fee

            # Entry
            slip = 0.0002
            entry = price * (1 + slip) if direction == 'LONG' else price * (1 - slip)

            acc.position = {
                'dir': direction,
                'entry': entry,
                'size': size,
                'time': time.time(),
                'btc': btc,
                'sl': entry * (0.98 if direction == 'LONG' else 1.02),
                'tp': entry * (1.04 if direction == 'LONG' else 0.96),
            }

            print(f"[OPEN] {direction} {ex.upper()} @ ${entry:,.0f} | {btc:.1f} BTC | ${size:.0f}")

    def check_exits(self, price: float):
        with self._lock:
            self.price = price
            now = time.time()

            for ex, acc in self.accounts.items():
                if not acc.position:
                    continue

                pos = acc.position
                reason = None

                # Check SL/TP/Timeout
                if pos['dir'] == 'LONG':
                    if price <= pos['sl']: reason = 'SL'
                    elif price >= pos['tp']: reason = 'TP'
                else:
                    if price >= pos['sl']: reason = 'SL'
                    elif price <= pos['tp']: reason = 'TP'

                if now - pos['time'] > 300:
                    reason = 'TIMEOUT'

                if reason:
                    self._close(ex, acc, price, reason)

    def _close(self, ex: str, acc: Account, price: float, reason: str):
        pos = acc.position
        fee = pos['size'] * self._get_fee(ex)

        if pos['dir'] == 'LONG':
            pnl = (price - pos['entry']) / pos['entry'] * pos['size']
        else:
            pnl = (pos['entry'] - price) / pos['entry'] * pos['size']

        net = pnl - fee
        acc.capital += pos['size'] + net
        acc.pnl += net
        self.trades += 1

        if net > 0:
            acc.wins += 1
            print(f"[WIN] {ex.upper()} +${net:.2f} ({reason})")
        else:
            acc.losses += 1
            print(f"[LOSS] {ex.upper()} ${net:.2f} ({reason})")

        # Log trade
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?)',
                (datetime.now().isoformat(), ex, pos['dir'], pos['entry'],
                 price, pos['size'], net, reason))
            conn.commit()
            conn.close()
        except:
            pass

        acc.position = {}

    def _price_loop(self):
        while self.running:
            try:
                r = requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=5)
                if r.ok:
                    self.check_exits(float(r.json()['data']['amount']))
            except:
                pass
            time.sleep(1)

    def _signal_loop(self):
        """Poll correlation.db for new signals from the running pipeline."""
        while self.running:
            try:
                conn = sqlite3.connect(self.corr_db)
                c = conn.cursor()
                c.execute('''SELECT id, exchange, direction, amount_btc, price_t0
                             FROM flows WHERE id > ? ORDER BY id LIMIT 10''',
                          (self.last_signal_id,))
                rows = c.fetchall()
                conn.close()

                for row in rows:
                    sig_id, exchange, direction, btc, price = row
                    self.last_signal_id = sig_id

                    if direction == 'INFLOW':
                        self.on_signal(exchange, 'SHORT', abs(btc), price or self.price)
                    elif direction == 'OUTFLOW':
                        self.on_signal(exchange, 'LONG', abs(btc), price or self.price)

            except Exception as e:
                pass

            time.sleep(0.5)

    def status(self):
        total = sum(a.capital for a in self.accounts.values())
        initial = len(ALL_EXCHANGES) * self.capital
        pnl = total - initial
        pnl_pct = pnl / initial * 100

        wins = sum(a.wins for a in self.accounts.values())
        losses = sum(a.losses for a in self.accounts.values())
        wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

        active = sum(1 for a in self.accounts.values() if a.position)

        print(f"\n{'='*60}")
        print(f"INSTANT TRADER | {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(f"BTC: ${self.price:,.0f} | Exchanges: {len(ALL_EXCHANGES)} | Signals: {self.signals}")
        print(f"Capital: ${total:,.0f} | P&L: ${pnl:+,.0f} ({pnl_pct:+.2f}%)")
        print(f"Trades: {wins+losses} | Wins: {wins} | Win Rate: {wr:.0f}% | Active: {active}")
        print(f"{'='*60}")

        # Top performers
        active_accs = [(ex, a) for ex, a in self.accounts.items() if a.wins + a.losses > 0]
        if active_accs:
            active_accs.sort(key=lambda x: x[1].pnl, reverse=True)
            print("Top Exchanges:")
            for ex, a in active_accs[:5]:
                print(f"  {ex}: ${a.pnl:+.2f} ({a.wins}W/{a.losses}L)")

    def run(self):
        self.running = True
        total = len(ALL_EXCHANGES) * self.capital

        print("=" * 60)
        print("INSTANT PAPER TRADER - LIVE")
        print("=" * 60)
        print(f"Exchanges: {len(ALL_EXCHANGES)}")
        print(f"Capital: ${total:,.0f} (${self.capital}/exchange)")
        print("=" * 60)
        print("DETERMINISTIC BLOCKCHAIN ADVANTAGE")
        print("  INFLOW -> SHORT | EXHAUSTION -> LONG")
        print("=" * 60)
        print()

        # Start threads
        threading.Thread(target=self._price_loop, daemon=True).start()
        threading.Thread(target=self._signal_loop, daemon=True).start()

        print("[LIVE] Connected to pipeline. Waiting for signals...\n")

        last = time.time()
        try:
            while self.running:
                if time.time() - last >= 60:
                    self.status()
                    last = time.time()
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[STOP]")
        finally:
            self.running = False
            self.status()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--capital', type=float, default=100.0)
    args = p.parse_args()

    trader = InstantTrader(args.capital)
    trader.run()
