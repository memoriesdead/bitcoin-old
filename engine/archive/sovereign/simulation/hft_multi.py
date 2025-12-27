#!/usr/bin/env python3
"""
MULTI-ASSET HFT BOT
===================
Trades 9 coins simultaneously on Kraken.
$100 to $1000 target.
"""
import asyncio
import aiohttp
import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional

# Configuration
PAIRS = [
    ('BTC', 'XXBTZUSD'),
    ('ETH', 'XETHZUSD'),
    ('SOL', 'SOLUSD'),
    ('XRP', 'XXRPZUSD'),
    ('ADA', 'ADAUSD'),
    ('DOT', 'DOTUSD'),
    ('LINK', 'LINKUSD'),
    ('AVAX', 'AVAXUSD'),
    ('ATOM', 'ATOMUSD'),
]

LEVERAGE = 35.0
TP_PCT = 0.005       # 0.5%
SL_PCT = 0.0015      # 0.15%
POSITION_PCT = 0.25  # 25% of equity per trade
MAX_CONCURRENT = 4
SIGNAL_THRESHOLD = 0.0002  # 0.02% from MA
INITIAL = 100.0


@dataclass
class Trade:
    pair: str
    direction: int  # 1=long, -1=short
    entry: float
    size: float
    tp: float
    sl: float
    entry_time: datetime


class MultiHFT:
    def __init__(self):
        self.equity = INITIAL
        self.trades: List[Trade] = []
        self.wins = 0
        self.losses = 0
        self.start_time = datetime.now(timezone.utc)
        self.prices: Dict[str, List[float]] = {p[0]: [] for p in PAIRS}
        self.current: Dict[str, float] = {}

        # Database
        self.db = sqlite3.connect('/root/hft_multi.db')
        self.db.execute('''CREATE TABLE IF NOT EXISTS trades(
            id INTEGER PRIMARY KEY, time TEXT, pair TEXT, dir INT,
            entry REAL, exit REAL, pnl REAL, equity REAL)''')
        self.db.execute('''CREATE TABLE IF NOT EXISTS equity(
            id INTEGER PRIMARY KEY, time TEXT, equity REAL)''')
        self.db.commit()

    def log(self, msg: str):
        ts = datetime.now().strftime('%H:%M:%S')
        print('[{}] {}'.format(ts, msg))

    async def fetch_prices(self, session: aiohttp.ClientSession):
        """Fetch current prices for all pairs."""
        for coin, pair in PAIRS:
            try:
                url = 'https://api.kraken.com/0/public/Ticker?pair={}'.format(pair)
                async with session.get(url) as resp:
                    data = await resp.json()
                    if 'result' in data and data['result']:
                        key = list(data['result'].keys())[0]
                        price = float(data['result'][key]['c'][0])
                        self.current[coin] = price
                        self.prices[coin].append(price)
                        # Keep last 10 prices
                        if len(self.prices[coin]) > 10:
                            self.prices[coin] = self.prices[coin][-10:]
            except Exception as e:
                pass  # Skip on error
            await asyncio.sleep(0.1)  # Rate limit

    def generate_signals(self) -> List[tuple]:
        """Generate signals for all pairs."""
        signals = []

        for coin, pair in PAIRS:
            prices = self.prices.get(coin, [])
            if len(prices) < 3:
                continue

            current = self.current.get(coin, 0)
            if current == 0:
                continue

            ma3 = sum(prices[-3:]) / 3

            # Long signal
            if current > ma3 * (1 + SIGNAL_THRESHOLD):
                diff = (current / ma3) - 1
                conf = min(0.60, 0.52 + diff * 10)
                signals.append((coin, pair, 1, conf, current))

            # Short signal
            elif current < ma3 * (1 - SIGNAL_THRESHOLD):
                diff = 1 - (current / ma3)
                conf = min(0.58, 0.52 + diff * 10)
                signals.append((coin, pair, -1, conf, current))

        return sorted(signals, key=lambda x: x[3], reverse=True)

    def open_trade(self, coin: str, pair: str, direction: int, price: float):
        """Open a new trade."""
        # Check if already have trade in this pair
        for t in self.trades:
            if t.pair == coin:
                return

        if len(self.trades) >= MAX_CONCURRENT:
            return

        size = self.equity * POSITION_PCT
        if size < 1:
            return

        if direction == 1:
            tp = price * (1 + TP_PCT)
            sl = price * (1 - SL_PCT)
        else:
            tp = price * (1 - TP_PCT)
            sl = price * (1 + SL_PCT)

        trade = Trade(
            pair=coin,
            direction=direction,
            entry=price,
            size=size,
            tp=tp,
            sl=sl,
            entry_time=datetime.now(timezone.utc)
        )
        self.trades.append(trade)

        dir_str = 'LONG' if direction == 1 else 'SHORT'
        self.log('{} {} ${:.0f} @ {:.2f}'.format(dir_str, coin, size, price))

    def check_exits(self):
        """Check if any trades should exit."""
        for trade in self.trades[:]:
            price = self.current.get(trade.pair, 0)
            if price == 0:
                continue

            exit_price = None
            win = None

            if trade.direction == 1:
                if price >= trade.tp:
                    exit_price, win = trade.tp, True
                elif price <= trade.sl:
                    exit_price, win = trade.sl, False
            else:
                if price <= trade.tp:
                    exit_price, win = trade.tp, True
                elif price >= trade.sl:
                    exit_price, win = trade.sl, False

            if exit_price is not None:
                if trade.direction == 1:
                    pnl_pct = (exit_price - trade.entry) / trade.entry
                else:
                    pnl_pct = (trade.entry - exit_price) / trade.entry

                pnl = trade.size * pnl_pct * LEVERAGE
                fees = trade.size * 0.0007  # 0.035% each way
                net_pnl = pnl - fees

                self.equity += net_pnl

                if win:
                    self.wins += 1
                else:
                    self.losses += 1

                self.trades.remove(trade)

                result = 'WIN' if win else 'LOSS'
                sign = '+' if net_pnl > 0 else ''
                self.log('{} {} {}{:.2f} EQ=${:.2f}'.format(
                    result, trade.pair, sign, net_pnl, self.equity))

                # Save to DB
                self.db.execute(
                    'INSERT INTO trades(time,pair,dir,entry,exit,pnl,equity) VALUES(?,?,?,?,?,?,?)',
                    (datetime.now().isoformat(), trade.pair, trade.direction,
                     trade.entry, exit_price, net_pnl, self.equity))
                self.db.commit()

    def print_status(self):
        """Print current status."""
        total = self.wins + self.losses
        wr = self.wins / total * 100 if total else 0
        ret = (self.equity / INITIAL - 1) * 100
        hrs = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600

        if hrs > 0.05:
            daily_roi = ((self.equity / INITIAL) ** (24 / hrs) - 1) * 100
        else:
            daily_roi = 0

        print('')
        print('=' * 60)
        print('EQUITY: ${:,.2f} ({:+.1f}%)'.format(self.equity, ret))
        print('Trades: {} | W:{} L:{} | WR: {:.1f}%'.format(total, self.wins, self.losses, wr))
        print('Hours: {:.2f} | Daily ROI: {:.0f}%'.format(hrs, daily_roi))
        print('Active: {} | Pairs: {}'.format(
            len(self.trades),
            ', '.join([t.pair for t in self.trades]) if self.trades else 'none'))
        print('Prices: ' + ' '.join(['{}:{:.0f}'.format(c, p) for c, p in list(self.current.items())[:5]]))
        print('=' * 60)
        print('')

        # Save equity snapshot
        self.db.execute('INSERT INTO equity(time,equity) VALUES(?,?)',
            (datetime.now().isoformat(), self.equity))
        self.db.commit()


async def main():
    hft = MultiHFT()

    print('=' * 60)
    print('MULTI-ASSET HFT BOT')
    print('=' * 60)
    print('Pairs: {}'.format(', '.join([p[0] for p in PAIRS])))
    print('Leverage: {}x | TP: {}% | SL: {}%'.format(LEVERAGE, TP_PCT*100, SL_PCT*100))
    print('Initial: ${} | Target: $1000'.format(INITIAL))
    print('=' * 60)
    print('')

    iteration = 0

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Fetch all prices
                await hft.fetch_prices(session)

                # Check exits
                hft.check_exits()

                # Generate and execute signals
                signals = hft.generate_signals()
                for coin, pair, direction, conf, price in signals:
                    if conf >= 0.52:
                        hft.open_trade(coin, pair, direction, price)

                # Status every 60 iterations (~2 min)
                iteration += 1
                if iteration % 60 == 0:
                    hft.print_status()

                # Check target
                if hft.equity >= 1000:
                    hrs = (datetime.now(timezone.utc) - hft.start_time).total_seconds() / 3600
                    print('')
                    print('*' * 60)
                    print('TARGET HIT! ${:.0f} -> ${:.2f} in {:.2f} hours'.format(INITIAL, hft.equity, hrs))
                    print('*' * 60)
                    break

                # Check bust
                if hft.equity < 5:
                    print('')
                    print('STOPPED: Equity too low (${:.2f})'.format(hft.equity))
                    break

                await asyncio.sleep(2)

            except Exception as e:
                print('Error: {}'.format(e))
                await asyncio.sleep(5)


if __name__ == '__main__':
    asyncio.run(main())
