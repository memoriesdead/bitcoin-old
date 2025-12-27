#!/usr/bin/env python3
"""HFT EXPLOSIVE TRADER - KRAKEN API"""
import asyncio
import aiohttp
import math
import sqlite3
from datetime import datetime, timezone

LEVERAGE = 35.0
TP_PCT = 0.008
SL_PCT = 0.002
KELLY = 1.5
MAX_POS = 0.5
MAX_CONCURRENT = 3
FEE = 0.00035
MIN_CONF = 0.54
INITIAL = 100.0

class Engine:
    def __init__(self):
        self.equity = INITIAL
        self.trades = []
        self.wins = self.losses = 0
        self.start = datetime.now(timezone.utc)
        self.candles = []
        self.price = 0
        self.db = sqlite3.connect('/root/hft_trades.db')
        self.db.execute('CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY, time TEXT, dir INT, pnl REAL, equity REAL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS equity(id INTEGER PRIMARY KEY, time TEXT, equity REAL)')
        self.db.commit()

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f'[{ts}] {msg}')

    def signals(self):
        if len(self.candles) < 10:
            return []
        sigs = []
        closes = [c['c'] for c in self.candles[-10:]]
        ma3 = sum(closes[-3:]) / 3
        ma5 = sum(closes[-5:]) / 5
        ma10 = sum(closes) / 10
        p = self.price

        if p > ma3 * 1.001 and ma3 > ma5:
            sigs.append((1, 0.56, 'MOM_UP'))
        if p > ma5 * 1.002 and ma5 > ma10:
            sigs.append((1, 0.58, 'TREND_UP'))
        if p < ma3 * 0.999 and ma3 < ma5:
            sigs.append((-1, 0.55, 'MOM_DN'))
        if p < ma5 * 0.998 and ma5 < ma10:
            sigs.append((-1, 0.57, 'TREND_DN'))
        return sigs

    def size(self, conf):
        b = TP_PCT / SL_PCT
        k = max(0, (conf * b - (1 - conf)) / b) * KELLY
        return self.equity * min(k, MAX_POS)

    def open_trade(self, sig):
        if len(self.trades) >= MAX_CONCURRENT or sig[1] < MIN_CONF:
            return
        sz = self.size(sig[1])
        if sz < 1:
            return
        e = self.price
        if sig[0] == 1:
            tp = e * (1 + TP_PCT)
            sl = e * (1 - SL_PCT)
        else:
            tp = e * (1 - TP_PCT)
            sl = e * (1 + SL_PCT)
        self.trades.append({'d': sig[0], 'e': e, 'sz': sz, 'tp': tp, 'sl': sl, 't': sig[2]})
        direction = "LONG" if sig[0] == 1 else "SHORT"
        self.log(f'{direction} ${sz:.2f} @ {e:.0f} TP:{tp:.0f} SL:{sl:.0f}')

    def check_exits(self):
        p = self.price
        for t in self.trades[:]:
            ex = None
            res = None
            if t['d'] == 1:
                if p >= t['tp']:
                    ex, res = t['tp'], 'WIN'
                elif p <= t['sl']:
                    ex, res = t['sl'], 'LOSS'
            else:
                if p <= t['tp']:
                    ex, res = t['tp'], 'WIN'
                elif p >= t['sl']:
                    ex, res = t['sl'], 'LOSS'
            if ex:
                if t['d'] == 1:
                    pnl_pct = (ex - t['e']) / t['e']
                else:
                    pnl_pct = (t['e'] - ex) / t['e']
                pnl = t['sz'] * pnl_pct * LEVERAGE - t['sz'] * FEE * 2
                self.equity += pnl
                if res == 'WIN':
                    self.wins += 1
                else:
                    self.losses += 1
                self.trades.remove(t)
                sign = "+" if pnl > 0 else ""
                self.log(f'{res} PnL:{sign}{pnl:.2f} Equity:${self.equity:.2f}')
                self.db.execute('INSERT INTO trades(time,dir,pnl,equity) VALUES(?,?,?,?)',
                    (datetime.now(timezone.utc).isoformat(), t['d'], pnl, self.equity))
                self.db.commit()

    def status(self):
        tot = self.wins + self.losses
        wr = self.wins / tot * 100 if tot else 0
        hrs = (datetime.now(timezone.utc) - self.start).total_seconds() / 3600
        roi = ((self.equity / INITIAL) ** (24 / hrs) - 1) * 100 if hrs > 0.1 else 0
        ret = (self.equity / INITIAL - 1) * 100
        print('')
        print('=' * 60)
        print(f'EQUITY: ${self.equity:,.2f} ({ret:+.1f}%)')
        print(f'Trades: {tot} W:{self.wins} L:{self.losses} WR:{wr:.1f}%')
        print(f'Hours: {hrs:.2f} | Daily ROI: {roi:.1f}% | Active: {len(self.trades)}')
        print('=' * 60)
        print('')


async def get_price():
    async with aiohttp.ClientSession() as s:
        async with s.get('https://api.kraken.com/0/public/Ticker?pair=XBTUSD') as r:
            d = await r.json()
            return float(d['result']['XXBTZUSD']['c'][0])


async def get_ohlc():
    async with aiohttp.ClientSession() as s:
        async with s.get('https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1') as r:
            d = await r.json()
            candles = d['result']['XXBTZUSD'][-20:]
            return [{'o': float(x[1]), 'h': float(x[2]), 'l': float(x[3]), 'c': float(x[4])} for x in candles]


async def run():
    e = Engine()
    print('=' * 60)
    print('HFT EXPLOSIVE - KRAKEN')
    print(f'Capital: ${INITIAL} | Leverage: {LEVERAGE}x | TP/SL: {TP_PCT*100}%/{SL_PCT*100}%')
    print('=' * 60)

    i = 0
    while True:
        try:
            e.price = await get_price()
            e.candles = await get_ohlc()
            e.check_exits()
            for s in sorted(e.signals(), key=lambda x: x[1], reverse=True):
                e.open_trade(s)
            i += 1
            if i % 30 == 0:
                e.status()
            if e.equity >= 1000:
                hrs = (datetime.now(timezone.utc) - e.start).total_seconds() / 3600
                print('')
                print('*' * 60)
                print(f'TARGET! ${INITIAL} -> ${e.equity:.2f} in {hrs:.2f}h')
                print('*' * 60)
                break
            if e.equity < 1:
                print('')
                print('BUSTED')
                break
            await asyncio.sleep(2)
        except Exception as ex:
            print(f'Err: {ex}')
            await asyncio.sleep(5)


if __name__ == '__main__':
    asyncio.run(run())
