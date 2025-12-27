#!/usr/bin/env python3
"""
HFT CONSOLIDATION TRADER - Uses consolidation detection for exchange flows.

LOGIC:
- Consolidation (20+ inputs) = Exchange hot wallet activity
- Large consolidations (>0.5 BTC) = Significant exchange flow
- Track net direction of consolidation outputs
"""
import asyncio
import aiohttp
import zmq
import time
import struct
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Optional
from collections import deque


@dataclass
class Config:
    leverage: float = 35.0
    tp_pct: float = 0.006
    sl_pct: float = 0.003
    min_consolidation_inputs: int = 20
    min_btc_flow: float = 0.5
    initial_capital: float = 100.0
    zmq_endpoint: str = "tcp://127.0.0.1:28332"


class SimpleDecoder:
    """Minimal transaction decoder."""

    def decode(self, raw: bytes) -> Optional[Dict]:
        try:
            if len(raw) < 10:
                return None

            offset = 4  # Skip version

            # Check for segwit marker
            if raw[offset:offset+2] == b'\x00\x01':
                offset += 2

            # Count inputs
            num_inputs, offset = self._read_varint(raw, offset)
            for _ in range(num_inputs):
                offset += 36  # Skip prev_txid + prev_index
                script_len, offset = self._read_varint(raw, offset)
                offset += script_len + 4  # Skip script + sequence

            # Count outputs
            num_outputs, offset = self._read_varint(raw, offset)
            total_btc = 0.0
            for _ in range(num_outputs):
                value = struct.unpack_from('<Q', raw, offset)[0]
                offset += 8
                script_len, offset = self._read_varint(raw, offset)
                offset += script_len
                total_btc += value / 100_000_000

            return {
                'num_inputs': num_inputs,
                'num_outputs': num_outputs,
                'total_btc': total_btc
            }
        except:
            return None

    def _read_varint(self, data: bytes, offset: int) -> tuple:
        first = data[offset]
        if first < 0xfd:
            return first, offset + 1
        elif first == 0xfd:
            return struct.unpack_from('<H', data, offset + 1)[0], offset + 3
        elif first == 0xfe:
            return struct.unpack_from('<I', data, offset + 1)[0], offset + 5
        else:
            return struct.unpack_from('<Q', data, offset + 1)[0], offset + 9


class ConsolidationTrader:
    """Trade based on consolidation detection."""

    def __init__(self, config: Config):
        self.config = config
        self.equity = config.initial_capital
        self.initial = config.initial_capital
        self.position = None
        self.trades = []
        self.start_time = datetime.now(timezone.utc)
        self.current_price = 0

        # ZMQ
        self.context = zmq.Context()
        self.socket = None
        self.decoder = SimpleDecoder()

        # Tracking
        self.consolidations = 0
        self.signals = 0
        self.total_flow_btc = 0.0
        self.flow_window = deque(maxlen=100)

    def log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f'[{ts}] {msg}')

    def start_zmq(self):
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(self.config.zmq_endpoint)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, 'rawtx')
        self.socket.setsockopt(zmq.RCVTIMEO, 100)
        self.log(f'[ZMQ] Connected to {self.config.zmq_endpoint}')

    def process_zmq(self):
        """Process pending ZMQ messages."""
        count = 0
        while count < 100:
            try:
                topic = self.socket.recv(zmq.NOBLOCK)
                body = self.socket.recv(zmq.NOBLOCK)

                if topic == b'rawtx':
                    self.process_tx(body)
                count += 1
            except zmq.Again:
                break
        return count

    def process_tx(self, raw: bytes):
        tx = self.decoder.decode(raw)
        if not tx:
            return

        num_inputs = tx['num_inputs']
        num_outputs = tx['num_outputs']
        total_btc = tx['total_btc']

        # CONSOLIDATION: Many inputs -> few outputs = Exchange gathering
        if num_inputs >= self.config.min_consolidation_inputs:
            self.consolidations += 1

            if total_btc >= self.config.min_btc_flow:
                # Large consolidation = significant exchange activity
                # Consolidation = exchange gathering funds = preparing for withdrawals = BULLISH
                direction = 1  # LONG

                self.flow_window.append({
                    'time': time.time(),
                    'btc': total_btc,
                    'direction': direction,
                    'type': 'consolidation',
                    'inputs': num_inputs,
                    'outputs': num_outputs
                })

                self.total_flow_btc += total_btc
                self.signals += 1

                self.log(f'[CONSOLIDATION] {num_inputs} in -> {num_outputs} out | {total_btc:.2f} BTC | LONG')
                self.process_signal(direction, total_btc)

        # FAN-OUT: Few inputs -> many outputs = Exchange distributing
        elif num_outputs >= 50 and total_btc >= self.config.min_btc_flow:
            direction = 1  # LONG - withdrawals are bullish

            self.flow_window.append({
                'time': time.time(),
                'btc': total_btc,
                'direction': direction,
                'type': 'fanout',
                'inputs': num_inputs,
                'outputs': num_outputs
            })

            self.total_flow_btc += total_btc
            self.signals += 1

            self.log(f'[FAN-OUT] {num_inputs} in -> {num_outputs} out | {total_btc:.2f} BTC | LONG')
            self.process_signal(direction, total_btc)

    def process_signal(self, direction: int, btc_amount: float):
        """Process trading signal with reversal logic."""

        confidence = min(0.95, 0.5 + (btc_amount / 20))

        if self.position:
            if self.position['dir'] == direction:
                return
            else:
                self.log(f'[REVERSAL] Closing -> Opening {"LONG" if direction == 1 else "SHORT"}')
                self.close_position(self.current_price, 'FLOW_REVERSAL')

        self.open_position(direction, btc_amount, confidence)

    def open_position(self, direction: int, btc_flow: float, confidence: float):
        if self.position or self.current_price <= 0:
            return

        price = self.current_price
        size = self.equity * 0.3 * confidence

        if direction == 1:
            tp = price * (1 + self.config.tp_pct)
            sl = price * (1 - self.config.sl_pct)
        else:
            tp = price * (1 - self.config.tp_pct)
            sl = price * (1 + self.config.sl_pct)

        self.position = {
            'dir': direction,
            'entry': price,
            'size': size,
            'tp': tp,
            'sl': sl,
            'btc_flow': btc_flow,
            'entry_time': datetime.now(timezone.utc)
        }

        dir_str = 'LONG' if direction == 1 else 'SHORT'
        self.log(f'[OPEN] {dir_str} ${size:.2f} @ ${price:.2f} | TP=${tp:.2f} SL=${sl:.2f}')

    def close_position(self, exit_price: float, reason: str):
        if not self.position:
            return

        pos = self.position

        if pos['dir'] == 1:
            pnl_pct = (exit_price - pos['entry']) / pos['entry']
        else:
            pnl_pct = (pos['entry'] - exit_price) / pos['entry']

        pnl = pos['size'] * pnl_pct * self.config.leverage
        self.equity += pnl

        result = 'WIN' if pnl > 0 else 'LOSS'
        self.trades.append({'pnl': pnl, 'reason': reason})

        sign = '+' if pnl > 0 else ''
        self.log(f'[CLOSE] {result} {sign}${pnl:.2f} | {reason} | Equity: ${self.equity:.2f}')
        self.position = None

    def check_stops(self):
        if not self.position or self.current_price <= 0:
            return

        pos = self.position
        price = self.current_price

        if pos['dir'] == 1:
            if price >= pos['tp']:
                self.close_position(pos['tp'], 'TAKE_PROFIT')
            elif price <= pos['sl']:
                self.close_position(pos['sl'], 'STOP_LOSS')
        else:
            if price <= pos['tp']:
                self.close_position(pos['tp'], 'TAKE_PROFIT')
            elif price >= pos['sl']:
                self.close_position(pos['sl'], 'STOP_LOSS')

    def print_status(self):
        wins = len([t for t in self.trades if t['pnl'] > 0])
        total = len(self.trades)
        wr = wins / total * 100 if total else 0
        ret = (self.equity / self.initial - 1) * 100
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600

        print('')
        print('=' * 60)
        print(f'EQUITY: ${self.equity:.2f} ({ret:+.1f}%) | Price: ${self.current_price:.2f}')
        print(f'Trades: {total} | Wins: {wins} | WR: {wr:.1f}%')
        print(f'Consolidations: {self.consolidations} | Signals: {self.signals}')
        print(f'Total Flow: {self.total_flow_btc:.2f} BTC | Runtime: {elapsed:.2f}h')
        pos_str = 'NONE' if not self.position else f'{"LONG" if self.position["dir"]==1 else "SHORT"} @ ${self.position["entry"]:.2f}'
        print(f'Position: {pos_str}')
        print('=' * 60)


async def get_price():
    url = 'https://api.kraken.com/0/public/Ticker?pair=XBTUSD'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            return float(data['result']['XXBTZUSD']['c'][0])


async def run():
    config = Config()
    trader = ConsolidationTrader(config)

    print('=' * 60)
    print('HFT CONSOLIDATION TRADER')
    print('=' * 60)
    print(f'Capital: ${config.initial_capital} | Leverage: {config.leverage}x')
    print(f'Min Consolidation: {config.min_consolidation_inputs} inputs')
    print(f'Min Flow: {config.min_btc_flow} BTC')
    print('')
    print('LOGIC:')
    print('  Consolidation (20+ inputs) = Exchange hot wallet')
    print('  Large consolidations = Significant flow -> Trade')
    print('=' * 60)

    trader.start_zmq()

    try:
        trader.current_price = await get_price()
        print(f'[INIT] Price: ${trader.current_price:.2f}')
    except Exception as e:
        print(f'Price error: {e}')

    iteration = 0

    try:
        while True:
            try:
                trader.current_price = await get_price()
                trader.process_zmq()
                trader.check_stops()

                iteration += 1
                if iteration % 30 == 0:
                    trader.print_status()

                if trader.equity >= 1000:
                    print('\n*** TARGET HIT! ***')
                    break

                if trader.equity <= 5:
                    print('\nStopped - Low equity')
                    break

                await asyncio.sleep(1)

            except Exception as e:
                print(f'Error: {e}')
                await asyncio.sleep(5)

    except KeyboardInterrupt:
        print('\nShutdown')
    finally:
        trader.print_status()


if __name__ == '__main__':
    print('Starting Consolidation Trader...')
    asyncio.run(run())
