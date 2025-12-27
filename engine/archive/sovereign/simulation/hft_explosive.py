#!/usr/bin/env python3
"""
HFT EXPLOSIVE TRADER - MINUTE-LEVEL TRADING
============================================
$100 to $1000 in HOURS, not days.

Runs 24/7, trades on 1-minute candles, multiple concurrent positions.
"""
import asyncio
import aiohttp
import json
import time
import math
import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class Config:
    leverage: float = 35.0
    tp_pct: float = 0.008      # 0.8% take profit
    sl_pct: float = 0.002      # 0.2% stop loss
    kelly: float = 1.5         # Super-Kelly
    max_position_pct: float = 0.5
    max_concurrent: int = 3
    taker_fee: float = 0.00035
    min_confidence: float = 0.54
    initial_capital: float = 100.0


class HFTEngine:
    def __init__(self, config: Config):
        self.config = config
        self.equity = config.initial_capital
        self.initial = config.initial_capital
        self.active_trades = []
        self.wins = 0
        self.losses = 0
        self.start_time = datetime.now(timezone.utc)
        self.candles_1m = []
        self.current_price = 0
        self.db_path = Path('hft_trades.db')
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            entry_time TEXT, exit_time TEXT,
            direction INTEGER, entry_price REAL, exit_price REAL,
            size REAL, pnl REAL, result TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS equity (
            id INTEGER PRIMARY KEY, timestamp TEXT, equity REAL
        )''')
        conn.commit()
        conn.close()

    def log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f'[{ts}] {msg}')

    def generate_signals(self) -> List[dict]:
        if len(self.candles_1m) < 10:
            return []

        signals = []
        price = self.current_price
        recent = self.candles_1m[-10:]
        closes = [c['close'] for c in recent]

        ma3 = sum(closes[-3:]) / 3
        ma5 = sum(closes[-5:]) / 5
        ma10 = sum(closes) / 10

        avg = sum(closes) / len(closes)
        vol = math.sqrt(sum((c - avg)**2 for c in closes) / len(closes)) / avg

        # LONG signals
        if price > ma3 * 1.001 and ma3 > ma5:
            signals.append({'dir': 1, 'conf': 0.56, 'type': 'MOM_UP'})
        if price > ma5 * 1.002 and ma5 > ma10:
            signals.append({'dir': 1, 'conf': 0.58, 'type': 'TREND_UP'})
        if vol > 0.003 and price > recent[-1]['high']:
            signals.append({'dir': 1, 'conf': 0.60, 'type': 'BREAKOUT'})

        # SHORT signals
        if price < ma3 * 0.999 and ma3 < ma5:
            signals.append({'dir': -1, 'conf': 0.55, 'type': 'MOM_DOWN'})
        if price < ma5 * 0.998 and ma5 < ma10:
            signals.append({'dir': -1, 'conf': 0.57, 'type': 'TREND_DOWN'})
        if vol > 0.003 and price < recent[-1]['low']:
            signals.append({'dir': -1, 'conf': 0.59, 'type': 'BREAKDOWN'})

        return signals

    def calculate_size(self, confidence: float) -> float:
        b = self.config.tp_pct / self.config.sl_pct
        p = confidence
        q = 1 - p
        kelly = max(0, (p * b - q) / b) * self.config.kelly
        kelly = min(kelly, self.config.max_position_pct)
        return self.equity * kelly

    def open_trade(self, signal: dict):
        if len(self.active_trades) >= self.config.max_concurrent:
            return
        if signal['conf'] < self.config.min_confidence:
            return

        size = self.calculate_size(signal['conf'])
        if size < 1:
            return

        entry = self.current_price
        if signal['dir'] == 1:
            tp = entry * (1 + self.config.tp_pct)
            sl = entry * (1 - self.config.sl_pct)
        else:
            tp = entry * (1 - self.config.tp_pct)
            sl = entry * (1 + self.config.sl_pct)

        trade = {
            'dir': signal['dir'],
            'entry': entry,
            'size': size,
            'tp': tp,
            'sl': sl,
            'type': signal['type'],
            'entry_time': datetime.now(timezone.utc)
        }

        self.active_trades.append(trade)
        dir_str = 'LONG' if signal['dir'] == 1 else 'SHORT'
        self.log(f'{dir_str} ${size:.2f} @ {entry:.0f} | TP:{tp:.0f} SL:{sl:.0f} | {signal["type"]}')

    def check_exits(self):
        price = self.current_price

        for trade in self.active_trades[:]:
            exit_price = None
            result = None

            if trade['dir'] == 1:
                if price >= trade['tp']:
                    exit_price = trade['tp']
                    result = 'WIN'
                elif price <= trade['sl']:
                    exit_price = trade['sl']
                    result = 'LOSS'
            else:
                if price <= trade['tp']:
                    exit_price = trade['tp']
                    result = 'WIN'
                elif price >= trade['sl']:
                    exit_price = trade['sl']
                    result = 'LOSS'

            if exit_price:
                if trade['dir'] == 1:
                    pnl_pct = (exit_price - trade['entry']) / trade['entry']
                else:
                    pnl_pct = (trade['entry'] - exit_price) / trade['entry']

                pnl = trade['size'] * pnl_pct * self.config.leverage
                fees = trade['size'] * self.config.taker_fee * 2
                net_pnl = pnl - fees

                self.equity += net_pnl

                if result == 'WIN':
                    self.wins += 1
                else:
                    self.losses += 1

                self.active_trades.remove(trade)

                dir_str = 'LONG' if trade['dir'] == 1 else 'SHORT'
                sign = '+' if net_pnl > 0 else ''
                self.log(f'{result} {dir_str} | PnL: {sign}${net_pnl:.2f} | Equity: ${self.equity:.2f}')

                self._save_trade(trade, exit_price, net_pnl, result)

    def _save_trade(self, trade, exit_price, pnl, result):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO trades
            (entry_time, exit_time, direction, entry_price, exit_price, size, pnl, result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (trade['entry_time'].isoformat(), datetime.now(timezone.utc).isoformat(),
             trade['dir'], trade['entry'], exit_price, trade['size'], pnl, result))
        c.execute('INSERT INTO equity (timestamp, equity) VALUES (?, ?)',
            (datetime.now(timezone.utc).isoformat(), self.equity))
        conn.commit()
        conn.close()

    def print_status(self):
        total = self.wins + self.losses
        wr = self.wins / total * 100 if total else 0
        ret = (self.equity / self.initial - 1) * 100

        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        hours = elapsed / 3600

        daily_roi = ((self.equity / self.initial) ** (24 / hours) - 1) * 100 if hours > 0.1 else 0

        print('')
        print('='*60)
        print(f'EQUITY: ${self.equity:,.2f} ({ret:+.1f}%)')
        print(f'Trades: {total} | Wins: {self.wins} | Losses: {self.losses} | WR: {wr:.1f}%')
        print(f'Runtime: {hours:.2f}h | Projected Daily ROI: {daily_roi:.1f}%')
        print(f'Active positions: {len(self.active_trades)}')
        print('='*60)


async def get_price():
    url = 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            return float(data['price'])


async def get_klines():
    url = 'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=20'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            return [{'open': float(k[1]), 'high': float(k[2]),
                     'low': float(k[3]), 'close': float(k[4])} for k in data]


async def run_hft():
    config = Config()
    engine = HFTEngine(config)

    print('='*60)
    print('HFT EXPLOSIVE TRADER - STARTING')
    print('='*60)
    print(f'Initial: ${config.initial_capital}')
    print(f'Leverage: {config.leverage}x')
    print(f'TP/SL: {config.tp_pct*100}% / {config.sl_pct*100}%')
    print(f'Max concurrent: {config.max_concurrent}')
    print('')

    iteration = 0

    while True:
        try:
            engine.current_price = await get_price()
            engine.candles_1m = await get_klines()

            engine.check_exits()

            signals = engine.generate_signals()
            for sig in sorted(signals, key=lambda x: x['conf'], reverse=True):
                engine.open_trade(sig)

            iteration += 1
            if iteration % 30 == 0:
                engine.print_status()

            if engine.equity >= 1000:
                print('')
                print('*'*60)
                print(f'TARGET HIT! $100 -> ${engine.equity:.2f}')
                elapsed = (datetime.now(timezone.utc) - engine.start_time).total_seconds() / 3600
                print(f'Time: {elapsed:.2f} hours')
                print('*'*60)
                break

            if engine.equity <= 1:
                print('')
                print('BUSTED!')
                break

            await asyncio.sleep(1)

        except Exception as e:
            print(f'Error: {e}')
            await asyncio.sleep(5)


if __name__ == '__main__':
    print('Starting HFT Explosive Trader...')
    asyncio.run(run_hft())
