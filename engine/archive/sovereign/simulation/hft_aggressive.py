#!/usr/bin/env python3
import asyncio, aiohttp, sqlite3
from datetime import datetime, timezone

LEV, TP, SL, INIT = 35.0, 0.008, 0.002, 100.0

class E:
    def __init__(s):
        s.eq, s.w, s.l, s.trades = INIT, 0, 0, []
        s.start, s.price, s.candles = datetime.now(timezone.utc), 0, []
        s.db = sqlite3.connect('/root/hft.db')
        s.db.execute('CREATE TABLE IF NOT EXISTS t(id INTEGER PRIMARY KEY,time TEXT,pnl REAL,eq REAL)')
        s.db.commit()

    def sigs(s):
        if len(s.candles) < 5: return []
        r = []
        c = [x['c'] for x in s.candles[-5:]]
        m3, m5 = sum(c[-3:])/3, sum(c)/5
        p = s.price
        if p > m3 * 1.0003: r.append((1, 0.55))
        if p < m3 * 0.9997: r.append((-1, 0.54))
        if p > m5 * 1.0005 and m3 > m5: r.append((1, 0.57))
        if p < m5 * 0.9995 and m3 < m5: r.append((-1, 0.56))
        return r

    def trade(s, sig):
        if len(s.trades) >= 3 or s.eq < 1: return
        sz = s.eq * 0.4
        e = s.price
        tp = e*(1+TP) if sig[0]==1 else e*(1-TP)
        sl = e*(1-SL) if sig[0]==1 else e*(1+SL)
        s.trades.append({'d':sig[0],'e':e,'sz':sz,'tp':tp,'sl':sl})
        d = "LONG" if sig[0]==1 else "SHORT"
        print('[{}] {} ${:.0f} @ {:.0f}'.format(datetime.now().strftime("%H:%M:%S"), d, sz, e))

    def check(s):
        p = s.price
        for t in s.trades[:]:
            ex = None
            if t['d']==1:
                if p >= t['tp']: ex, win = t['tp'], True
                elif p <= t['sl']: ex, win = t['sl'], False
            else:
                if p <= t['tp']: ex, win = t['tp'], True
                elif p >= t['sl']: ex, win = t['sl'], False
            if ex:
                pct = (ex-t['e'])/t['e'] if t['d']==1 else (t['e']-ex)/t['e']
                pnl = t['sz']*pct*LEV - t['sz']*0.0007
                s.eq += pnl
                if win: s.w += 1
                else: s.l += 1
                s.trades.remove(t)
                r = "WIN" if win else "LOSS"
                sign = "+" if pnl > 0 else ""
                print('[{}] {} {}{:.2f} EQ={:.2f}'.format(datetime.now().strftime("%H:%M:%S"), r, sign, pnl, s.eq))
                s.db.execute('INSERT INTO t(time,pnl,eq)VALUES(?,?,?)',(datetime.now().isoformat(),pnl,s.eq))
                s.db.commit()

    def stat(s):
        t = s.w + s.l
        h = (datetime.now(timezone.utc)-s.start).seconds/3600
        roi = ((s.eq/INIT)**(24/h)-1)*100 if h > 0.05 else 0
        wr = s.w/t*100 if t else 0
        ret = (s.eq/INIT - 1) * 100
        print('\n' + '='*50)
        print('EQ: ${:.2f} ({:+.1f}%)'.format(s.eq, ret))
        print('Trades: {} W:{} L:{} WR:{:.1f}%'.format(t, s.w, s.l, wr))
        print('Hours: {:.2f} | Daily ROI: {:.0f}%'.format(h, roi))
        print('='*50 + '\n')

async def price():
    async with aiohttp.ClientSession() as c:
        async with c.get('https://api.kraken.com/0/public/Ticker?pair=XBTUSD') as r:
            return float((await r.json())['result']['XXBTZUSD']['c'][0])

async def ohlc():
    async with aiohttp.ClientSession() as c:
        async with c.get('https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1') as r:
            d = (await r.json())['result']['XXBTZUSD'][-10:]
            return [{'c':float(x[4])} for x in d]

async def main():
    e = E()
    print('='*50)
    print('HFT EXPLOSIVE TRADER')
    print('Capital: ${} | Leverage: {}x'.format(INIT, LEV))
    print('TP: {}% | SL: {}%'.format(TP*100, SL*100))
    print('='*50)
    i = 0
    while e.eq > 0.5 and e.eq < 1000:
        try:
            e.price = await price()
            e.candles = await ohlc()
            e.check()
            for sig in e.sigs(): e.trade(sig)
            i += 1
            if i % 60 == 0: e.stat()
        except Exception as x:
            print('Error: {}'.format(x))
        await asyncio.sleep(1)

    hrs = (datetime.now(timezone.utc) - e.start).seconds / 3600
    print('\n' + '*'*50)
    if e.eq >= 1000:
        print('TARGET HIT! ${} -> ${:.2f} in {:.2f}h'.format(INIT, e.eq, hrs))
    else:
        print('STOPPED: ${:.2f}'.format(e.eq))
    print('*'*50)

if __name__ == '__main__':
    asyncio.run(main())
