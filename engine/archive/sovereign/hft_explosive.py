"""
EXPLOSIVE HFT - Blockchain Edge Trading

THE EDGE: We see blockchain transactions BEFORE Whale Alert tweets them.
- We see tx at T=0 (direct ZMQ from Bitcoin Core)
- Whale Alert tweets at T=5-60 seconds
- Retail reacts at T=5-120 seconds (FOMO/panic)
- We capture the RETAIL REACTION, not the actual deposit impact

TIMING:
- Entry: Immediate (0-5 sec after we see tx)
- Hold: 30-90 seconds (capture retail reaction window)
- Exit: TP/SL or timeout

THE MATH:
- INFLOW (deposit) → Whale Alert tweets "DUMP INCOMING" → Retail panics → SHORT
- OUTFLOW (withdrawal) → Whale Alert tweets "WHALES ACCUMULATING" → Retail FOMO → LONG

100+ BTC flows = Gets tweeted by Whale Alert = Retail will react
"""
import sys
import os
import time
import threading
import sqlite3
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, '/root')
from sovereign.blockchain.formula_connector import FormulaConnector


@dataclass
class ExplosiveConfig:
    leverage: float = 100.0         # MAXIMUM LEVERAGE
    tp_pct: float = 0.0020          # 0.20% take profit (~$175 move) - retail panic capture
    sl_pct: float = 0.0010          # 0.10% stop loss (~$88 move)
    timeout_seconds: float = 90.0   # 90 sec = retail reaction window
    min_btc_flow: float = 100.0     # 100+ BTC = Whale Alert will tweet it
    min_confidence: float = 0.60    # Lower for more signals (we have timing edge)
    position_pct: float = 0.50      # 50% of capital per trade
    initial_capital: float = 100.0
    zmq_endpoint: str = "tcp://127.0.0.1:28332"
    exchanges_json: str = "/root/exchanges.json.gz"


class ExplosiveHFT:
    def __init__(self, config: ExplosiveConfig = None):
        self.config = config or ExplosiveConfig()
        self.capital = self.config.initial_capital
        self.position = None
        self.position_open_time = 0
        self.trades = []
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.current_price = 0.0
        self.signals_seen = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.running = False

        # Database
        self.db_path = '/root/hft_explosive_trades.db'
        self._init_db()

        # Formula connector (creates its own feed internally)
        self.connector = FormulaConnector(
            zmq_endpoint=self.config.zmq_endpoint,
            json_path=self.config.exchanges_json,
            enable_pattern_recognition=True,
            enable_rentech=True
        )

        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║           EXPLOSIVE HFT - BLOCKCHAIN EDGE                     ║
╠═══════════════════════════════════════════════════════════════╣
║  Leverage:    {self.config.leverage:.0f}x                                        ║
║  TP:          {self.config.tp_pct*100:.2f}% (${self.config.tp_pct * 88000:.0f} move)                            ║
║  SL:          {self.config.sl_pct*100:.2f}% (${self.config.sl_pct * 88000:.0f} move)                            ║
║  Timeout:     {self.config.timeout_seconds:.0f} seconds                                   ║
║  Min Flow:    {self.config.min_btc_flow:.0f} BTC (${self.config.min_btc_flow * 88000/1e6:.1f}M+)                           ║
║  Capital:     ${self.config.initial_capital:.0f}                                        ║
╚═══════════════════════════════════════════════════════════════╝
        """)

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            direction INTEGER,
            entry_price REAL,
            exit_price REAL,
            btc_amount REAL,
            pnl REAL,
            exit_reason TEXT,
            hold_time REAL,
            confidence REAL
        )''')
        conn.commit()
        conn.close()

    def _save_trade(self, trade: dict):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''INSERT INTO trades
            (timestamp, direction, entry_price, exit_price, btc_amount, pnl, exit_reason, hold_time, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (trade['timestamp'], trade['direction'], trade['entry_price'],
             trade['exit_price'], trade['btc_amount'], trade['pnl'],
             trade['exit_reason'], trade['hold_time'], trade['confidence']))
        conn.commit()
        conn.close()

    def _fetch_price(self) -> float:
        """Fetch current BTC price."""
        import urllib.request
        import json
        try:
            req = urllib.request.Request(
                'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode())
                return float(data['price'])
        except:
            try:
                req = urllib.request.Request(
                    'https://api.kraken.com/0/public/Ticker?pair=XBTUSD',
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req, timeout=5) as r:
                    data = json.loads(r.read().decode())
                    return float(data['result']['XXBTZUSD']['c'][0])
            except:
                return self.current_price if self.current_price > 0 else 88000.0

    def _check_signals(self):
        """Check for trading signals from formula engines."""
        # Use consume to CLEAR signals after reading (prevents flooding)
        signals = self.connector.consume_pending_signals()

        for signal in signals:
            btc = signal.get('btc_amount', 0)
            conf = signal.get('confidence', 0)
            direction = signal.get('direction', 0)

            # Build sources list from which engines voted
            sources = []
            if signal.get('adaptive_signal'):
                sources.append('adaptive')
            if signal.get('pattern_signal'):
                sources.append('pattern')
            if signal.get('rentech_signal'):
                sources.append('rentech')

            self.signals_seen += 1

            # Filter: 50+ BTC, 0.7+ confidence
            if btc < self.config.min_btc_flow:
                continue
            if conf < self.config.min_confidence:
                continue
            if direction == 0:
                continue

            # Log the signal
            dir_str = 'LONG' if direction == 1 else 'SHORT'
            src_str = '+'.join(sources) if sources else 'unknown'
            num_sources = len(sources) if sources else 0
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [EXPLOSIVE] {dir_str} | {btc:.1f} BTC | conf={conf:.2f} | {src_str} ({num_sources}/3)")

            # Trade if no position
            with self.lock:
                if self.position is None:
                    self._open_position(direction, btc, conf)
                elif self.position['direction'] != direction:
                    # Reversal signal - close and flip
                    print(f"[REVERSAL] {dir_str} signal while in opposite position")
                    self._close_position('reversal')
                    self._open_position(direction, btc, conf)

    def _open_position(self, direction: int, btc: float, confidence: float):
        """Open explosive position."""
        price = self._fetch_price()
        if price <= 0:
            print("[ERROR] No price available")
            return

        self.current_price = price
        self.connector.set_reference_price(price)

        # Position sizing: 50% of capital * leverage
        position_value = self.capital * self.config.position_pct * self.config.leverage

        # Calculate TP/SL
        if direction == 1:  # LONG
            tp = price * (1 + self.config.tp_pct)
            sl = price * (1 - self.config.sl_pct)
        else:  # SHORT
            tp = price * (1 - self.config.tp_pct)
            sl = price * (1 + self.config.sl_pct)

        self.position = {
            'direction': direction,
            'entry': price,
            'size': position_value,
            'tp': tp,
            'sl': sl,
            'btc': btc,
            'confidence': confidence,
            'open_time': time.time()
        }
        self.position_open_time = time.time()

        dir_str = 'LONG' if direction == 1 else 'SHORT'
        print(f"""
┌─────────────────────────────────────────────────────┐
│  OPENED {dir_str} @ ${price:,.2f}
│  Size: ${position_value:,.2f} | BTC: {btc:.1f}
│  TP: ${tp:,.2f} | SL: ${sl:,.2f}
│  Timeout: {self.config.timeout_seconds:.0f}s
└─────────────────────────────────────────────────────┘
        """)

    def _close_position(self, reason: str):
        """Close position and record trade."""
        if self.position is None:
            return

        price = self._fetch_price()
        if price <= 0:
            price = self.current_price

        entry = self.position['entry']
        direction = self.position['direction']
        size = self.position['size']
        hold_time = time.time() - self.position['open_time']

        # Calculate P&L
        if direction == 1:  # LONG
            pnl_pct = (price - entry) / entry
        else:  # SHORT
            pnl_pct = (entry - price) / entry

        pnl = size * pnl_pct
        self.capital += pnl
        self.total_pnl += pnl

        if pnl > 0:
            self.wins += 1
            result = "WIN"
        else:
            self.losses += 1
            result = "LOSS"

        dir_str = 'LONG' if direction == 1 else 'SHORT'

        trade = {
            'timestamp': datetime.now().isoformat(),
            'direction': direction,
            'entry_price': entry,
            'exit_price': price,
            'btc_amount': self.position['btc'],
            'pnl': pnl,
            'exit_reason': reason,
            'hold_time': hold_time,
            'confidence': self.position['confidence']
        }
        self.trades.append(trade)
        self._save_trade(trade)

        print(f"""
┌─────────────────────────────────────────────────────┐
│  CLOSED {dir_str} - {result}
│  Entry: ${entry:,.2f} → Exit: ${price:,.2f}
│  P&L: ${pnl:+.2f} ({pnl_pct*100:+.3f}%)
│  Reason: {reason} | Hold: {hold_time:.1f}s
│  Capital: ${self.capital:.2f}
└─────────────────────────────────────────────────────┘
        """)

        self.position = None

    def _check_exit_conditions(self):
        """Check TP, SL, and timeout."""
        if self.position is None:
            return

        price = self._fetch_price()
        if price <= 0:
            return

        self.current_price = price
        direction = self.position['direction']
        tp = self.position['tp']
        sl = self.position['sl']
        hold_time = time.time() - self.position['open_time']

        # Check timeout FIRST (90 seconds)
        if hold_time >= self.config.timeout_seconds:
            print(f"[TIMEOUT] {hold_time:.1f}s elapsed - closing at market")
            self._close_position('timeout')
            return

        # Check TP/SL
        if direction == 1:  # LONG
            if price >= tp:
                self._close_position('take_profit')
            elif price <= sl:
                self._close_position('stop_loss')
        else:  # SHORT
            if price <= tp:
                self._close_position('take_profit')
            elif price >= sl:
                self._close_position('stop_loss')

    def _print_status(self):
        """Print current status."""
        runtime = (time.time() - self.start_time) / 3600
        wr = (self.wins / (self.wins + self.losses) * 100) if (self.wins + self.losses) > 0 else 0

        pos_str = "None"
        if self.position:
            d = 'LONG' if self.position['direction'] == 1 else 'SHORT'
            entry = self.position['entry']
            if self.position['direction'] == 1:
                pnl = (self.current_price - entry) / entry * self.position['size']
            else:
                pnl = (entry - self.current_price) / entry * self.position['size']
            hold = time.time() - self.position['open_time']
            pos_str = f"{d} ${self.position['size']:.2f} @ ${entry:.2f} (P&L: ${pnl:+.2f}, {hold:.0f}s)"

        print(f"""
══════════════════════════════════════════════════════════════════
CAPITAL: ${self.capital:.2f} ({(self.capital/self.config.initial_capital-1)*100:+.1f}%) | Price: ${self.current_price:.2f}
Trades: {self.wins + self.losses} | Wins: {self.wins} | Losses: {self.losses} | WR: {wr:.1f}%
Total P&L: ${self.total_pnl:+.2f} | Signals: {self.signals_seen} | Runtime: {runtime:.2f}h
Position: {pos_str}
══════════════════════════════════════════════════════════════════
        """)

    def run(self):
        """Main trading loop."""
        print("[STARTING] Explosive HFT...")

        # Get initial price
        self.current_price = self._fetch_price()
        print(f"[PRICE] ${self.current_price:,.2f}")
        self.connector.set_reference_price(self.current_price)

        # Start formula connector (this starts the blockchain feed internally)
        if not self.connector.start():
            print("[ERROR] Failed to start formula connector")
            return

        self.running = True
        last_status = time.time()
        last_price_update = time.time()

        try:
            while self.running:
                now = time.time()

                # Check signals from formula engines
                self._check_signals()

                # Check exit conditions (TP/SL/Timeout)
                self._check_exit_conditions()

                # Update price every 5 seconds
                if now - last_price_update >= 5:
                    new_price = self._fetch_price()
                    if new_price > 0:
                        self.current_price = new_price
                        self.connector.set_reference_price(new_price)
                    last_price_update = now

                # Print status every 30 seconds
                if now - last_status >= 30:
                    self._print_status()
                    last_status = now

                time.sleep(0.1)  # 100ms loop

        except KeyboardInterrupt:
            print("\n[STOPPING]...")
        finally:
            self.running = False
            if self.position:
                print("[CLOSING] Final position...")
                self._close_position('shutdown')
            self.connector.stop()
            self._print_status()
            print("[STOPPED]")


if __name__ == '__main__':
    config = ExplosiveConfig()
    hft = ExplosiveHFT(config)
    hft.run()
