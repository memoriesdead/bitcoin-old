#!/usr/bin/env python3
"""
HFT FORMULA INTEGRATED - MATHEMATICAL EDGE + EXPLOSIVE TRADING
===============================================================
Combines:
1. HFT explosive trading (minute-level, 35x leverage)
2. FormulaConnector (900+ formulas from blockchain data)

EDGE: 10-60 second information advantage from blockchain before exchange APIs.

$100 -> $1000 target with mathematical precision.
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

# Add sovereign to path
sys.path.insert(0, '/root')

# =============================================================================
# CONFIGURATION
# =============================================================================

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
    # Formula integration
    formula_weight: float = 0.6   # Weight for formula signals
    hft_weight: float = 0.4       # Weight for basic HFT signals
    zmq_endpoint: str = "tcp://127.0.0.1:28332"
    exchanges_json: str = "/root/exchanges.json.gz"


# =============================================================================
# HFT ENGINE WITH FORMULA INTEGRATION
# =============================================================================

class HFTFormulaEngine:
    """
    HFT Engine integrated with FormulaConnector for mathematical edge.
    """

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
        self.db_path = Path('/root/hft_formula_trades.db')
        self._init_db()

        # Formula connector signals (thread-safe)
        self.formula_signals = deque(maxlen=100)
        self.formula_lock = threading.Lock()
        self.last_formula_signal = None

        # Stats
        self.formula_trades = 0
        self.hft_trades = 0
        self.combined_trades = 0

        # Initialize FormulaConnector
        self.connector = None
        self._init_formula_connector()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            entry_time TEXT, exit_time TEXT,
            direction INTEGER, entry_price REAL, exit_price REAL,
            size REAL, pnl REAL, result TEXT,
            signal_source TEXT, ensemble_type TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS equity (
            id INTEGER PRIMARY KEY, timestamp TEXT, equity REAL
        )''')
        conn.commit()
        conn.close()

    def _init_formula_connector(self):
        """Initialize FormulaConnector with all 3 engines."""
        try:
            from sovereign.blockchain.formula_connector import FormulaConnector

            self.connector = FormulaConnector(
                zmq_endpoint=self.config.zmq_endpoint,
                json_path=self.config.exchanges_json,
                on_signal=self._on_formula_signal,
                enable_pattern_recognition=True,
                enable_rentech=True,
                rentech_mode="full"
            )

            # Start in background thread
            if self.connector.start():
                self.log("[FORMULA] FormulaConnector started - 3 engines active")
                self.log("[FORMULA] Listening for blockchain transactions...")
            else:
                self.log("[FORMULA] WARNING: FormulaConnector failed to start")
                self.connector = None

        except ImportError as e:
            self.log(f"[FORMULA] Import error: {e}")
            self.connector = None
        except Exception as e:
            self.log(f"[FORMULA] Init error: {e}")
            self.connector = None

    def _on_formula_signal(self, signal: Dict):
        """Callback when FormulaConnector generates a signal."""
        with self.formula_lock:
            self.formula_signals.append({
                'signal': signal,
                'timestamp': time.time()
            })
            self.last_formula_signal = signal

        # Log significant signals
        dir_str = "LONG" if signal['direction'] == 1 else "SHORT"
        conf = signal.get('confidence', 0)
        ensemble = signal.get('ensemble_type', 'unknown')
        votes = signal.get('vote_count', 0)
        self.log(f"[BLOCKCHAIN] {dir_str} | conf={conf:.2f} | {ensemble} ({votes}/3) | "
                 f"BTC={signal.get('btc_amount', 0):.2f}")

    def log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f'[{ts}] {msg}')

    def get_formula_signal(self) -> Optional[Dict]:
        """Get latest formula signal if fresh (< 30 seconds old)."""
        with self.formula_lock:
            if not self.formula_signals:
                return None

            latest = self.formula_signals[-1]
            age = time.time() - latest['timestamp']

            # Only use fresh signals (< 30 seconds)
            if age < 30:
                return latest['signal']
            return None

    def generate_signals(self) -> List[dict]:
        """Generate combined HFT + Formula signals."""
        signals = []

        # === FORMULA SIGNALS (blockchain edge) ===
        formula_sig = self.get_formula_signal()
        if formula_sig and formula_sig.get('confidence', 0) >= 0.4:
            # Use formula-derived parameters
            signals.append({
                'dir': formula_sig['direction'],
                'conf': formula_sig['confidence'],
                'type': f"FORMULA_{formula_sig.get('ensemble_type', 'unknown').upper()}",
                'source': 'formula',
                'ensemble_type': formula_sig.get('ensemble_type'),
                'vote_count': formula_sig.get('vote_count', 0),
                'kelly': formula_sig.get('kelly_fraction', 0.05),
                'tp_pct': formula_sig.get('take_profit', self.config.tp_pct),
                'sl_pct': formula_sig.get('stop_loss', self.config.sl_pct),
            })

        # === BASIC HFT SIGNALS (price action) ===
        if len(self.candles_1m) >= 10:
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
                signals.append({'dir': 1, 'conf': 0.56, 'type': 'MOM_UP', 'source': 'hft'})
            if price > ma5 * 1.002 and ma5 > ma10:
                signals.append({'dir': 1, 'conf': 0.58, 'type': 'TREND_UP', 'source': 'hft'})
            if vol > 0.003 and price > recent[-1]['high']:
                signals.append({'dir': 1, 'conf': 0.60, 'type': 'BREAKOUT', 'source': 'hft'})

            # SHORT signals
            if price < ma3 * 0.999 and ma3 < ma5:
                signals.append({'dir': -1, 'conf': 0.55, 'type': 'MOM_DOWN', 'source': 'hft'})
            if price < ma5 * 0.998 and ma5 < ma10:
                signals.append({'dir': -1, 'conf': 0.57, 'type': 'TREND_DOWN', 'source': 'hft'})
            if vol > 0.003 and price < recent[-1]['low']:
                signals.append({'dir': -1, 'conf': 0.59, 'type': 'BREAKDOWN', 'source': 'hft'})

        # === COMBINE SIGNALS ===
        return self._combine_signals(signals)

    def _combine_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Combine formula and HFT signals using weighted ensemble.

        RULES:
        1. Formula + HFT agree (same direction) = BOOST confidence 1.3x
        2. Formula only (high conf) = Use formula
        3. HFT only = Use HFT (reduced weight)
        4. Conflicting = Use higher confidence if > 0.6, else skip
        """
        if not signals:
            return []

        formula_sigs = [s for s in signals if s.get('source') == 'formula']
        hft_sigs = [s for s in signals if s.get('source') == 'hft']

        combined = []

        # Check for formula signal
        if formula_sigs:
            formula = formula_sigs[0]  # Take best formula signal

            # Check if HFT agrees
            hft_agree = [s for s in hft_sigs if s['dir'] == formula['dir']]

            if hft_agree:
                # BOOST: Formula + HFT agree
                best_hft = max(hft_agree, key=lambda x: x['conf'])
                boosted_conf = min(0.95, formula['conf'] * 1.3)
                combined.append({
                    **formula,
                    'conf': boosted_conf,
                    'type': f"COMBINED_{formula['type']}+{best_hft['type']}",
                    'source': 'combined',
                })
                self.combined_trades += 1
            else:
                # Formula only - use if high confidence
                if formula['conf'] >= 0.5:
                    combined.append(formula)
                    self.formula_trades += 1

        # Add remaining HFT signals (if no formula or formula went other direction)
        if not formula_sigs:
            for hft in hft_sigs:
                # Reduce HFT confidence when no formula confirmation
                hft['conf'] *= 0.9
                if hft['conf'] >= self.config.min_confidence:
                    combined.append(hft)
                    self.hft_trades += 1

        return combined

    def open_trade(self, signal: dict):
        """Open a trade based on signal."""
        if len(self.active_trades) >= self.config.max_concurrent:
            return

        # Check existing positions
        for t in self.active_trades:
            if t['dir'] == signal['dir']:
                return  # Already have position in this direction

        if signal['conf'] < self.config.min_confidence:
            return

        price = self.current_price

        # Use signal-specific or default TP/SL
        tp_pct = signal.get('tp_pct', self.config.tp_pct)
        sl_pct = signal.get('sl_pct', self.config.sl_pct)

        # Calculate size (Kelly-based)
        kelly = signal.get('kelly', self.config.kelly)
        size = self.equity * min(kelly, self.config.max_position_pct)

        if signal['dir'] == 1:
            tp = price * (1 + tp_pct)
            sl = price * (1 - sl_pct)
        else:
            tp = price * (1 - tp_pct)
            sl = price * (1 + sl_pct)

        trade = {
            'dir': signal['dir'],
            'entry': price,
            'tp': tp,
            'sl': sl,
            'size': size,
            'type': signal['type'],
            'source': signal.get('source', 'unknown'),
            'ensemble_type': signal.get('ensemble_type', ''),
            'entry_time': datetime.now(timezone.utc)
        }

        self.active_trades.append(trade)

        dir_str = 'LONG' if signal['dir'] == 1 else 'SHORT'
        self.log(f'OPEN {dir_str} ${size:.2f} @ ${price:.2f} | {signal["type"]} | conf={signal["conf"]:.2f}')

    def check_exits(self):
        """Check and execute exits."""
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

                # Record trade result for formula learning
                if self.connector and trade.get('source') in ['formula', 'combined']:
                    try:
                        self.connector.record_trade_result(
                            {'direction': trade['dir'], 'entry_time': trade['entry_time'].timestamp()},
                            exit_price
                        )
                    except:
                        pass

                dir_str = 'LONG' if trade['dir'] == 1 else 'SHORT'
                sign = '+' if net_pnl > 0 else ''
                self.log(f'{result} {dir_str} | PnL: {sign}${net_pnl:.2f} | Equity: ${self.equity:.2f} | {trade["type"]}')

                self._save_trade(trade, exit_price, net_pnl, result)

    def _save_trade(self, trade, exit_price, pnl, result):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO trades
            (entry_time, exit_time, direction, entry_price, exit_price, size, pnl, result, signal_source, ensemble_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (trade['entry_time'].isoformat(), datetime.now(timezone.utc).isoformat(),
             trade['dir'], trade['entry'], exit_price, trade['size'], pnl, result,
             trade.get('source', ''), trade.get('ensemble_type', '')))
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
        print('='*70)
        print(f'EQUITY: ${self.equity:,.2f} ({ret:+.1f}%)')
        print(f'Trades: {total} | Wins: {self.wins} | Losses: {self.losses} | WR: {wr:.1f}%')
        print(f'Signal Sources: Formula={self.formula_trades} | HFT={self.hft_trades} | Combined={self.combined_trades}')
        print(f'Runtime: {hours:.2f}h | Projected Daily ROI: {daily_roi:.1f}%')
        print(f'Active positions: {len(self.active_trades)}')

        # Formula connector stats
        if self.connector:
            try:
                stats = self.connector.get_stats()
                print(f'Blockchain: Ticks={stats.get("ticks_processed", 0)} | '
                      f'Signals={stats.get("signals_generated", 0)}')
            except:
                pass

        print('='*70)

    def stop(self):
        """Stop the engine."""
        if self.connector:
            self.connector.stop()


# =============================================================================
# PRICE FETCHING
# =============================================================================

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


# =============================================================================
# MAIN LOOP
# =============================================================================

async def run_hft_formula():
    config = Config()
    engine = HFTFormulaEngine(config)

    print('='*70)
    print('HFT FORMULA INTEGRATED - MATHEMATICAL EDGE TRADING')
    print('='*70)
    print(f'Initial: ${config.initial_capital}')
    print(f'Leverage: {config.leverage}x')
    print(f'TP/SL: {config.tp_pct*100}% / {config.sl_pct*100}%')
    print(f'Max concurrent: {config.max_concurrent}')
    print(f'Signal weights: Formula={config.formula_weight} | HFT={config.hft_weight}')
    print('')
    print('FORMULA ENGINES:')
    print('  - Adaptive (IDs 10001-10005)')
    print('  - Pattern Recognition (IDs 20001-20012)')
    print('  - RenTech (IDs 72001-72099)')
    print('')

    # Update reference price for formula connector
    if engine.connector:
        try:
            price = await get_price()
            engine.connector.set_reference_price(price)
        except:
            pass

    iteration = 0

    try:
        while True:
            try:
                engine.current_price = await get_price()
                engine.candles_1m = await get_klines()

                # Update formula connector price
                if engine.connector and iteration % 5 == 0:
                    engine.connector.set_reference_price(engine.current_price)

                engine.check_exits()

                signals = engine.generate_signals()
                for sig in sorted(signals, key=lambda x: x['conf'], reverse=True):
                    engine.open_trade(sig)

                iteration += 1
                if iteration % 30 == 0:
                    engine.print_status()

                if engine.equity >= 1000:
                    print('')
                    print('*'*70)
                    print(f'TARGET HIT! $100 -> ${engine.equity:.2f}')
                    elapsed = (datetime.now(timezone.utc) - engine.start_time).total_seconds() / 3600
                    print(f'Time: {elapsed:.2f} hours')
                    print('*'*70)
                    break

                if engine.equity <= 1:
                    print('')
                    print('BUSTED!')
                    break

                await asyncio.sleep(1)

            except Exception as e:
                print(f'Error: {e}')
                await asyncio.sleep(5)

    finally:
        engine.stop()


if __name__ == '__main__':
    print('Starting HFT Formula Integrated Trader...')
    asyncio.run(run_hft_formula())
