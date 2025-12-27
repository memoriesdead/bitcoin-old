#!/usr/bin/env python3
"""
HFT FLOW REVERSAL - Fixed Trading Logic
========================================
CRITICAL FIX: Reverse positions on opposite flow signals.

LOGIC:
- INFLOW to exchange = SHORT (they will sell)
- OUTFLOW from exchange = LONG (they are accumulating)
- When signal flips, CLOSE and REVERSE immediately

This captures the 10-60 second edge from blockchain data.
"""
import asyncio
import aiohttp
import json
import time
import math
import sqlite3
import threading
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path
from collections import deque

sys.path.insert(0, '/root')


@dataclass
class Config:
    leverage: float = 35.0
    tp_pct: float = 0.006       # 0.6% take profit
    sl_pct: float = 0.003       # 0.3% stop loss
    min_btc_flow: float = 5.0   # Minimum BTC to trigger trade
    min_confidence: float = 0.6
    initial_capital: float = 100.0
    zmq_endpoint: str = "tcp://127.0.0.1:28332"
    exchanges_json: str = "/root/exchanges.json.gz"
    # CRITICAL: Single position mode for clean reversals
    max_positions: int = 1


class FlowTrader:
    """
    Flow-based trader with position reversal.

    FIXED LOGIC:
    - Only 1 position at a time
    - Reverse on opposite signal
    - Act immediately on strong flows
    """

    def __init__(self, config: Config):
        self.config = config
        self.equity = config.initial_capital
        self.initial = config.initial_capital
        self.position = None  # Single position
        self.trades = []
        self.start_time = datetime.now(timezone.utc)
        self.current_price = 0
        self.db_path = Path('/root/hft_flow_trades.db')
        self._init_db()

        # Signal tracking
        self.last_signal_time = 0
        self.signals_received = 0
        self.reversals = 0

        # Formula connector
        self.connector = None
        self.signal_queue = deque(maxlen=50)
        self.signal_lock = threading.Lock()

        self._init_connector()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            entry_time TEXT, exit_time TEXT,
            direction INTEGER, entry_price REAL, exit_price REAL,
            size REAL, pnl REAL, result TEXT,
            btc_flow REAL, exit_reason TEXT
        )''')
        conn.commit()
        conn.close()

    def _init_connector(self):
        try:
            from sovereign.blockchain.formula_connector import FormulaConnector

            self.connector = FormulaConnector(
                zmq_endpoint=self.config.zmq_endpoint,
                json_path=self.config.exchanges_json,
                on_signal=self._on_signal,
                enable_pattern_recognition=True,
                enable_rentech=True,
                rentech_mode="full"
            )

            if self.connector.start():
                self.log("[CONNECTOR] Started - 3 formula engines active")
            else:
                self.log("[CONNECTOR] Failed to start")
                self.connector = None

        except Exception as e:
            self.log(f"[CONNECTOR] Error: {e}")
            self.connector = None

    def _on_signal(self, signal: Dict):
        """Receive signal from FormulaConnector."""
        with self.signal_lock:
            self.signal_queue.append({
                'signal': signal,
                'time': time.time()
            })
        self.signals_received += 1

    def log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f'[{ts}] {msg}')

    def get_latest_signal(self) -> Optional[Dict]:
        """Get freshest signal (< 10 seconds old)."""
        with self.signal_lock:
            if not self.signal_queue:
                return None
            latest = self.signal_queue[-1]
            if time.time() - latest['time'] < 10:
                return latest['signal']
        return None

    def process_signal(self, signal: Dict):
        """
        CORE LOGIC: Process flow signal with reversal.

        If signal direction != position direction:
            1. Close position
            2. Open new position in signal direction
        """
        direction = signal.get('direction', 0)
        confidence = signal.get('confidence', 0)
        btc_amount = signal.get('btc_amount', 0)
        ensemble_type = signal.get('ensemble_type', 'unknown')
        vote_count = signal.get('vote_count', 0)

        if direction == 0:
            return

        # Filter: need minimum confidence and flow
        if confidence < self.config.min_confidence:
            return
        if btc_amount < self.config.min_btc_flow:
            return

        dir_str = "LONG" if direction == 1 else "SHORT"
        self.log(f"[SIGNAL] {dir_str} | {btc_amount:.1f} BTC | conf={confidence:.2f} | {ensemble_type} ({vote_count}/3)")

        # === CRITICAL: REVERSAL LOGIC ===
        if self.position:
            if self.position['dir'] == direction:
                # Same direction - already positioned correctly
                self.log(f"[HOLD] Already {dir_str}")
                return
            else:
                # OPPOSITE DIRECTION - REVERSE!
                self.log(f"[REVERSAL] Closing {('LONG' if self.position['dir']==1 else 'SHORT')} -> Opening {dir_str}")
                self.close_position(self.current_price, "FLOW_REVERSAL")
                self.reversals += 1

        # Open new position
        self.open_position(direction, btc_amount, confidence, signal)

    def open_position(self, direction: int, btc_flow: float, confidence: float, signal: Dict):
        """Open position in flow direction."""
        if self.position:
            return  # Already have position

        price = self.current_price
        if price <= 0:
            return

        # Size based on confidence and flow magnitude
        base_size = self.equity * 0.25  # 25% base
        flow_mult = min(2.0, 1.0 + (btc_flow / 100))  # Bigger flow = bigger size
        conf_mult = confidence  # Higher confidence = bigger size
        size = base_size * flow_mult * conf_mult

        # TP/SL
        tp_pct = signal.get('take_profit', self.config.tp_pct)
        sl_pct = signal.get('stop_loss', self.config.sl_pct)

        if direction == 1:  # LONG
            tp = price * (1 + tp_pct)
            sl = price * (1 - sl_pct)
        else:  # SHORT
            tp = price * (1 - tp_pct)
            sl = price * (1 + sl_pct)

        self.position = {
            'dir': direction,
            'entry': price,
            'size': size,
            'tp': tp,
            'sl': sl,
            'btc_flow': btc_flow,
            'confidence': confidence,
            'entry_time': datetime.now(timezone.utc),
            'signal': signal
        }

        dir_str = "LONG" if direction == 1 else "SHORT"
        self.log(f"[OPEN] {dir_str} ${size:.2f} @ ${price:.2f} | TP=${tp:.2f} SL=${sl:.2f} | {btc_flow:.1f} BTC flow")

    def close_position(self, exit_price: float, reason: str):
        """Close current position."""
        if not self.position:
            return

        pos = self.position

        # Calculate PnL
        if pos['dir'] == 1:  # LONG
            pnl_pct = (exit_price - pos['entry']) / pos['entry']
        else:  # SHORT
            pnl_pct = (pos['entry'] - exit_price) / pos['entry']

        pnl = pos['size'] * pnl_pct * self.config.leverage
        self.equity += pnl

        result = "WIN" if pnl > 0 else "LOSS"

        trade = {
            'dir': pos['dir'],
            'entry': pos['entry'],
            'exit': exit_price,
            'size': pos['size'],
            'pnl': pnl,
            'result': result,
            'reason': reason,
            'btc_flow': pos['btc_flow'],
            'entry_time': pos['entry_time'],
            'exit_time': datetime.now(timezone.utc)
        }
        self.trades.append(trade)

        # Save to DB
        self._save_trade(trade)

        dir_str = "LONG" if pos['dir'] == 1 else "SHORT"
        sign = "+" if pnl > 0 else ""
        self.log(f"[CLOSE] {dir_str} @ ${exit_price:.2f} | {result} {sign}${pnl:.2f} | {reason} | Equity: ${self.equity:.2f}")

        self.position = None

    def check_stops(self):
        """Check TP/SL."""
        if not self.position:
            return

        price = self.current_price
        pos = self.position

        if pos['dir'] == 1:  # LONG
            if price >= pos['tp']:
                self.close_position(pos['tp'], "TAKE_PROFIT")
            elif price <= pos['sl']:
                self.close_position(pos['sl'], "STOP_LOSS")
        else:  # SHORT
            if price <= pos['tp']:
                self.close_position(pos['tp'], "TAKE_PROFIT")
            elif price >= pos['sl']:
                self.close_position(pos['sl'], "STOP_LOSS")

    def _save_trade(self, trade):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO trades
            (entry_time, exit_time, direction, entry_price, exit_price, size, pnl, result, btc_flow, exit_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (trade['entry_time'].isoformat(), trade['exit_time'].isoformat(),
             trade['dir'], trade['entry'], trade['exit'], trade['size'],
             trade['pnl'], trade['result'], trade['btc_flow'], trade['reason']))
        conn.commit()
        conn.close()

    def print_status(self):
        wins = len([t for t in self.trades if t['pnl'] > 0])
        losses = len([t for t in self.trades if t['pnl'] <= 0])
        total = len(self.trades)
        wr = wins / total * 100 if total else 0
        ret = (self.equity / self.initial - 1) * 100

        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        hours = elapsed / 3600

        # Reversal stats
        rev_wins = len([t for t in self.trades if t['reason'] == 'FLOW_REVERSAL' and t['pnl'] > 0])
        rev_total = len([t for t in self.trades if t['reason'] == 'FLOW_REVERSAL'])

        print('')
        print('='*70)
        print(f'EQUITY: ${self.equity:.2f} ({ret:+.1f}%) | Price: ${self.current_price:.2f}')
        print(f'Trades: {total} | Wins: {wins} | Losses: {losses} | WR: {wr:.1f}%')
        print(f'Reversals: {self.reversals} | Reversal Trades: {rev_total} (W:{rev_wins})')
        print(f'Signals: {self.signals_received} | Runtime: {hours:.2f}h')
        print(f'Position: {self._pos_str()}')
        print('='*70)

    def _pos_str(self):
        if not self.position:
            return "NONE"
        dir_str = "LONG" if self.position['dir'] == 1 else "SHORT"
        pnl = self._unrealized_pnl()
        return f"{dir_str} ${self.position['size']:.2f} @ ${self.position['entry']:.2f} (P&L: ${pnl:.2f})"

    def _unrealized_pnl(self):
        if not self.position:
            return 0
        pos = self.position
        if pos['dir'] == 1:
            pnl_pct = (self.current_price - pos['entry']) / pos['entry']
        else:
            pnl_pct = (pos['entry'] - self.current_price) / pos['entry']
        return pos['size'] * pnl_pct * self.config.leverage

    def stop(self):
        if self.connector:
            self.connector.stop()


async def get_price():
    url = 'https://api.kraken.com/0/public/Ticker?pair=XBTUSD'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            return float(data['result']['XXBTZUSD']['c'][0])


async def run():
    config = Config()
    trader = FlowTrader(config)

    print('='*70)
    print('HFT FLOW REVERSAL - FIXED TRADING LOGIC')
    print('='*70)
    print(f'Capital: ${config.initial_capital} | Leverage: {config.leverage}x')
    print(f'TP/SL: {config.tp_pct*100}% / {config.sl_pct*100}%')
    print(f'Min Flow: {config.min_btc_flow} BTC | Min Conf: {config.min_confidence}')
    print('')
    print('LOGIC:')
    print('  INFLOW to exchange  -> SHORT (they will sell)')
    print('  OUTFLOW from exchange -> LONG (accumulating)')
    print('  Opposite signal -> CLOSE & REVERSE immediately')
    print('='*70)
    print('')

    # Get initial price
    try:
        trader.current_price = await get_price()
        if trader.connector:
            trader.connector.set_reference_price(trader.current_price)
        print(f'[INIT] Price: ${trader.current_price:.2f}')
    except Exception as e:
        print(f'[INIT] Price error: {e}')

    iteration = 0

    try:
        while True:
            try:
                # Update price
                trader.current_price = await get_price()

                # Update connector reference price
                if trader.connector and iteration % 5 == 0:
                    trader.connector.set_reference_price(trader.current_price)

                # Check TP/SL
                trader.check_stops()

                # Process latest signal
                signal = trader.get_latest_signal()
                if signal:
                    trader.process_signal(signal)

                # Status every 30 iterations
                iteration += 1
                if iteration % 30 == 0:
                    trader.print_status()

                # Target check
                if trader.equity >= 1000:
                    print('')
                    print('*'*70)
                    print(f'TARGET HIT! ${config.initial_capital} -> ${trader.equity:.2f}')
                    print('*'*70)
                    break

                if trader.equity <= 5:
                    print('')
                    print('STOPPED - Low equity')
                    break

                await asyncio.sleep(1)

            except Exception as e:
                print(f'Error: {e}')
                await asyncio.sleep(5)

    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        trader.print_status()
        trader.stop()


if __name__ == '__main__':
    print('Starting HFT Flow Reversal Trader...')
    asyncio.run(run())
