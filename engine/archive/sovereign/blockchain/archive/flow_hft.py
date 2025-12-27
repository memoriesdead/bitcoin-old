#!/usr/bin/env python3
"""
BLOCKCHAIN FLOW HFT BOT
=======================
The edge: See exchange flows 10-60 seconds BEFORE price impact.

INFLOW (BTC → Exchange) = Someone depositing to SELL = SHORT
OUTFLOW (Exchange → BTC) = Someone withdrawing to HOLD = LONG

Uses mempool.space WebSocket for real-time transaction monitoring.
Executes on Kraken with 35x leverage.
"""
import asyncio
import aiohttp
import json
import time
import sqlite3
import gzip
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List
from collections import defaultdict
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Trading
LEVERAGE = 35.0
TP_PCT = 0.005       # 0.5%
SL_PCT = 0.0015      # 0.15%
POSITION_PCT = 0.25  # 25% of equity per trade
MAX_CONCURRENT = 4
INITIAL_CAPITAL = 100.0

# Flow signal thresholds
FLOW_THRESHOLD_BTC = 2.0       # Min BTC flow to generate signal
FLOW_IMBALANCE_THRESHOLD = 0.3  # FIS threshold for trade
WINDOW_SECONDS = 60             # Aggregation window

# Whale detection (large transaction monitoring)
WHALE_THRESHOLD_BTC = 1.0       # Transactions >= this are notable
MEGA_WHALE_BTC = 10.0           # Large moves - stronger signal

# Kraken pairs for execution
KRAKEN_PAIR = "XXBTZUSD"

# Mempool.space WebSocket
MEMPOOL_WS = "wss://mempool.space/api/v1/ws"
MEMPOOL_API = "https://mempool.space/api"

# Data paths - check multiple locations
EXCHANGES_PATHS = [
    Path("/root/exchanges.json.gz"),      # Hostinger VPS
    Path("/root/exchanges.json"),         # Hostinger VPS uncompressed
    Path("data/exchanges.json.gz"),       # Local compressed
    Path("data/exchanges.json"),          # Local
    Path("exchanges.json.gz"),            # Current dir
    Path("exchanges.json"),               # Current dir
]
DB_PATH = Path("data/flow_trades.db")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FlowEvent:
    """Single exchange flow event."""
    timestamp: float
    exchange: str
    direction: int      # +1 = OUTFLOW (LONG), -1 = INFLOW (SHORT)
    btc_amount: float
    txid: str


@dataclass
class Position:
    """Open trading position."""
    entry_time: float
    entry_price: float
    direction: int      # +1 = LONG, -1 = SHORT
    size_usd: float
    stop_loss: float
    take_profit: float
    signal_reason: str


@dataclass
class FlowWindow:
    """Aggregated flows over time window."""
    inflow_btc: float = 0.0
    outflow_btc: float = 0.0
    inflow_count: int = 0
    outflow_count: int = 0
    exchanges: Dict[str, float] = field(default_factory=dict)
    # Whale tracking
    whale_txs: int = 0
    whale_btc: float = 0.0
    mega_whale_txs: int = 0

    @property
    def total(self) -> float:
        return self.inflow_btc + self.outflow_btc

    @property
    def imbalance(self) -> float:
        """Flow Imbalance Signal: [-1, +1]"""
        if self.total == 0:
            return 0.0
        return (self.outflow_btc - self.inflow_btc) / self.total


# =============================================================================
# EXCHANGE ADDRESS MANAGER
# =============================================================================

class ExchangeAddressManager:
    """Manages 7.6M exchange addresses with O(1) lookup."""

    def __init__(self):
        self.address_to_exchange: Dict[str, str] = {}
        self.exchange_addresses: Dict[str, Set[str]] = defaultdict(set)
        self.loaded = False

    def load_from_json(self, path: Path) -> bool:
        """Load exchange addresses from JSON file (supports .gz)."""
        print(f"[ADDRESSES] Loading from {path}...")
        start = time.time()

        try:
            # Check if file is gzipped
            if str(path).endswith('.gz'):
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            elif path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                print(f"[ADDRESSES] File not found: {path}")
                return False

            # Build reverse index
            for exchange, addresses in data.items():
                for addr in addresses:
                    self.address_to_exchange[addr] = exchange
                    self.exchange_addresses[exchange].add(addr)

            elapsed = time.time() - start
            print(f"[ADDRESSES] Loaded {len(self.address_to_exchange):,} addresses "
                  f"from {len(self.exchange_addresses)} exchanges in {elapsed:.1f}s")
            self.loaded = True
            return True

        except Exception as e:
            print(f"[ADDRESSES] Load failed: {e}")
            return False

    def load_top_exchanges(self, path: Path, top_n: int = 20) -> bool:
        """Load only top N exchanges to reduce memory."""
        print(f"[ADDRESSES] Loading top {top_n} exchanges from {path}...")
        start = time.time()

        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                return False

            # Sort by address count and take top N
            sorted_exchanges = sorted(data.items(), key=lambda x: len(x[1]), reverse=True)[:top_n]

            for exchange, addresses in sorted_exchanges:
                for addr in addresses:
                    self.address_to_exchange[addr] = exchange
                    self.exchange_addresses[exchange].add(addr)

            elapsed = time.time() - start
            print(f"[ADDRESSES] Loaded {len(self.address_to_exchange):,} addresses "
                  f"from top {top_n} exchanges in {elapsed:.1f}s")
            self.loaded = True
            return True

        except Exception as e:
            print(f"[ADDRESSES] Load failed: {e}")
            return False

    def get_exchange(self, address: str) -> Optional[str]:
        """O(1) lookup of exchange for address."""
        return self.address_to_exchange.get(address)

    def is_exchange_address(self, address: str) -> bool:
        """Check if address belongs to any known exchange."""
        return address in self.address_to_exchange


# =============================================================================
# MEMPOOL FLOW MONITOR
# =============================================================================

class MempoolFlowMonitor:
    """Monitors mempool.space for real-time exchange flows."""

    def __init__(self, address_manager: ExchangeAddressManager):
        self.addresses = address_manager
        self.flow_events: List[FlowEvent] = []
        self.current_window = FlowWindow()
        self.window_start = time.time()
        self.ws = None
        self.running = False
        self.stats = {
            'txs_processed': 0,
            'flows_detected': 0,
            'inflows': 0,
            'outflows': 0,
            'total_btc': 0.0,
        }

    async def connect(self) -> bool:
        """Connect to mempool.space WebSocket."""
        try:
            session = aiohttp.ClientSession()
            self.ws = await session.ws_connect(MEMPOOL_WS)

            # Subscribe to new transactions
            await self.ws.send_json({"action": "want", "data": ["mempool-blocks"]})

            print("[MEMPOOL] Connected to mempool.space WebSocket")
            return True

        except Exception as e:
            print(f"[MEMPOOL] Connection failed: {e}")
            return False

    async def poll_mempool(self):
        """Poll mempool API for recent transactions with full details."""
        seen_txids = set()

        async with aiohttp.ClientSession() as session:
            while self.running:
                try:
                    # Get recent transaction IDs
                    async with session.get(f"{MEMPOOL_API}/mempool/recent") as resp:
                        if resp.status == 200:
                            txs = await resp.json()

                            # Get full details for new transactions
                            for tx_summary in txs[:20]:  # Limit to avoid rate limits
                                txid = tx_summary.get('txid', '')
                                if txid and txid not in seen_txids:
                                    seen_txids.add(txid)

                                    # Fetch full transaction details
                                    try:
                                        async with session.get(f"{MEMPOOL_API}/tx/{txid}") as tx_resp:
                                            if tx_resp.status == 200:
                                                full_tx = await tx_resp.json()
                                                await self.process_transaction(full_tx)
                                    except Exception as e:
                                        pass  # Skip failed fetches

                    # Keep seen set manageable
                    if len(seen_txids) > 10000:
                        seen_txids = set(list(seen_txids)[-5000:])

                    await asyncio.sleep(3)  # Poll every 3 seconds

                except Exception as e:
                    print(f"[MEMPOOL] Poll error: {e}", flush=True)
                    await asyncio.sleep(5)

    async def process_transaction(self, tx: dict):
        """Process a transaction for exchange flows."""
        self.stats['txs_processed'] += 1

        txid = tx.get('txid', '')

        # Debug: log first few transactions to see structure
        if self.stats['txs_processed'] <= 3:
            print(f"[DEBUG] TX sample: {json.dumps(tx, default=str)[:500]}", flush=True)

        # Check outputs for INFLOW (BTC going TO exchange)
        for vout in tx.get('vout', []):
            address = vout.get('scriptpubkey_address', '')
            value_btc = vout.get('value', 0) / 1e8

            exchange = self.addresses.get_exchange(address)
            if exchange and value_btc > 0.01:  # Min 0.01 BTC
                event = FlowEvent(
                    timestamp=time.time(),
                    exchange=exchange,
                    direction=-1,  # INFLOW = SHORT signal
                    btc_amount=value_btc,
                    txid=txid
                )
                self.record_flow(event)

        # Check inputs for OUTFLOW (BTC coming FROM exchange)
        for vin in tx.get('vin', []):
            prevout = vin.get('prevout', {})
            address = prevout.get('scriptpubkey_address', '')
            value_btc = prevout.get('value', 0) / 1e8

            exchange = self.addresses.get_exchange(address)
            if exchange and value_btc > 0.01:
                event = FlowEvent(
                    timestamp=time.time(),
                    exchange=exchange,
                    direction=+1,  # OUTFLOW = LONG signal
                    btc_amount=value_btc,
                    txid=txid
                )
                self.record_flow(event)

        # WHALE DETECTION: Track large transactions regardless of address
        # Calculate total transaction value from outputs
        total_btc = sum(vout.get('value', 0) for vout in tx.get('vout', [])) / 1e8

        if total_btc >= WHALE_THRESHOLD_BTC:
            self.current_window.whale_txs += 1
            self.current_window.whale_btc += total_btc
            self.stats['whale_txs'] = self.stats.get('whale_txs', 0) + 1
            self.stats['whale_btc'] = self.stats.get('whale_btc', 0.0) + total_btc

            # Log whale transactions
            if total_btc >= MEGA_WHALE_BTC:
                self.current_window.mega_whale_txs += 1
                print(f"[MEGA WHALE] {total_btc:.2f} BTC | TX: {txid[:16]}...", flush=True)
            elif self.stats.get('whale_txs', 0) <= 5 or self.stats.get('whale_txs', 0) % 10 == 0:
                print(f"[WHALE] {total_btc:.2f} BTC | TX: {txid[:16]}...", flush=True)

    def record_flow(self, event: FlowEvent):
        """Record a flow event and update window."""
        self.flow_events.append(event)
        self.stats['flows_detected'] += 1
        self.stats['total_btc'] += event.btc_amount

        if event.direction == -1:
            self.current_window.inflow_btc += event.btc_amount
            self.current_window.inflow_count += 1
            self.stats['inflows'] += 1
        else:
            self.current_window.outflow_btc += event.btc_amount
            self.current_window.outflow_count += 1
            self.stats['outflows'] += 1

        # Track per-exchange
        self.current_window.exchanges[event.exchange] = \
            self.current_window.exchanges.get(event.exchange, 0) + event.btc_amount * event.direction

        # Log significant flows
        if event.btc_amount >= 1.0:
            direction_str = "OUTFLOW" if event.direction > 0 else "INFLOW"
            print(f"[FLOW] {direction_str} {event.btc_amount:.2f} BTC @ {event.exchange}")

    def get_signal(self) -> Optional[tuple]:
        """Get trading signal from current flow window."""
        now = time.time()

        # Reset window if expired
        if now - self.window_start >= WINDOW_SECONDS:
            self.current_window = FlowWindow()
            self.window_start = now
            self.flow_events = [e for e in self.flow_events if now - e.timestamp < 300]
            self._last_whale_alert = 0  # Reset whale alert counter
            return None

        # Check if we have enough flow from exchange matching
        if self.current_window.total < FLOW_THRESHOLD_BTC:
            # No exchange flow signal - check for mega whale activity (alert once per whale)
            if self.current_window.mega_whale_txs >= 1 and not hasattr(self, '_last_whale_alert') or \
               self.current_window.mega_whale_txs > getattr(self, '_last_whale_alert', 0):
                print(f"[ALERT] {self.current_window.mega_whale_txs} mega whale(s) in window, "
                      f"total {self.current_window.whale_btc:.0f} BTC - market volatility expected", flush=True)
                self._last_whale_alert = self.current_window.mega_whale_txs
            return None

        # Check imbalance threshold
        fis = self.current_window.imbalance

        # Boost confidence if whale activity is high
        whale_boost = 0.1 if self.current_window.whale_txs >= 3 else 0.0
        mega_boost = 0.15 if self.current_window.mega_whale_txs >= 1 else 0.0

        if fis > FLOW_IMBALANCE_THRESHOLD:
            # More outflows than inflows = accumulation = LONG
            confidence = min(0.85, 0.5 + abs(fis) * 0.3 + whale_boost + mega_boost)
            reason = f"FIS={fis:.2f} OUT>{self.current_window.outflow_btc:.1f}BTC"
            if self.current_window.whale_txs > 0:
                reason += f" +{self.current_window.whale_txs}whales"
            return (+1, confidence, reason)

        elif fis < -FLOW_IMBALANCE_THRESHOLD:
            # More inflows than outflows = distribution = SHORT
            confidence = min(0.85, 0.5 + abs(fis) * 0.3 + whale_boost + mega_boost)
            reason = f"FIS={fis:.2f} IN>{self.current_window.inflow_btc:.1f}BTC"
            if self.current_window.whale_txs > 0:
                reason += f" +{self.current_window.whale_txs}whales"
            return (-1, confidence, reason)

        return None


# =============================================================================
# KRAKEN EXECUTOR
# =============================================================================

class KrakenExecutor:
    """Execute trades on Kraken (paper mode - no API keys)."""

    def __init__(self, initial_capital: float = 100.0):
        self.equity = initial_capital
        self.initial_capital = initial_capital
        self.positions: List[Position] = []
        self.trades = []
        self.last_price = 0.0
        self.price_cache_time = 0

    async def get_price(self) -> float:
        """Get current BTC price from Kraken."""
        now = time.time()
        if now - self.price_cache_time < 1 and self.last_price > 0:
            return self.last_price

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.kraken.com/0/public/Ticker?pair={KRAKEN_PAIR}"
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get('result', {})
                        ticker = result.get(KRAKEN_PAIR, result.get('XBTUSD', {}))
                        self.last_price = float(ticker.get('c', [0])[0])
                        self.price_cache_time = now
                        return self.last_price
        except Exception as e:
            print(f"[KRAKEN] Price fetch error: {e}")

        return self.last_price if self.last_price > 0 else 100000.0

    async def open_position(self, direction: int, signal_reason: str) -> bool:
        """Open a new position."""
        if len(self.positions) >= MAX_CONCURRENT:
            return False

        price = await self.get_price()
        size_usd = self.equity * POSITION_PCT * LEVERAGE

        if direction == 1:  # LONG
            stop_loss = price * (1 - SL_PCT)
            take_profit = price * (1 + TP_PCT)
        else:  # SHORT
            stop_loss = price * (1 + SL_PCT)
            take_profit = price * (1 - TP_PCT)

        position = Position(
            entry_time=time.time(),
            entry_price=price,
            direction=direction,
            size_usd=size_usd,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_reason=signal_reason
        )
        self.positions.append(position)

        dir_str = "LONG" if direction == 1 else "SHORT"
        print(f"[TRADE] OPEN {dir_str} ${size_usd:.2f} @ ${price:.2f} | "
              f"TP: ${take_profit:.2f} SL: ${stop_loss:.2f} | {signal_reason}")

        return True

    async def check_exits(self):
        """Check all positions for TP/SL exits."""
        if not self.positions:
            return

        price = await self.get_price()
        closed = []

        for pos in self.positions:
            exit_reason = None

            if pos.direction == 1:  # LONG
                if price >= pos.take_profit:
                    exit_reason = "TAKE_PROFIT"
                elif price <= pos.stop_loss:
                    exit_reason = "STOP_LOSS"
            else:  # SHORT
                if price <= pos.take_profit:
                    exit_reason = "TAKE_PROFIT"
                elif price >= pos.stop_loss:
                    exit_reason = "STOP_LOSS"

            if exit_reason:
                # Calculate P&L
                if pos.direction == 1:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price
                else:
                    pnl_pct = (pos.entry_price - price) / pos.entry_price

                pnl_usd = pos.size_usd * pnl_pct
                self.equity += pnl_usd

                dir_str = "LONG" if pos.direction == 1 else "SHORT"
                print(f"[TRADE] CLOSE {dir_str} @ ${price:.2f} | {exit_reason} | "
                      f"PnL: ${pnl_usd:.2f} ({pnl_pct*100:.2f}%) | Equity: ${self.equity:.2f}")

                self.trades.append({
                    'entry_time': pos.entry_time,
                    'exit_time': time.time(),
                    'entry_price': pos.entry_price,
                    'exit_price': price,
                    'direction': pos.direction,
                    'size_usd': pos.size_usd,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'signal': pos.signal_reason,
                })

                closed.append(pos)

        for pos in closed:
            self.positions.remove(pos)

    def get_stats(self) -> dict:
        """Get trading statistics."""
        if not self.trades:
            return {'trades': 0, 'equity': self.equity}

        wins = [t for t in self.trades if t['pnl_usd'] > 0]
        total_pnl = sum(t['pnl_usd'] for t in self.trades)

        return {
            'trades': len(self.trades),
            'wins': len(wins),
            'win_rate': len(wins) / len(self.trades) * 100,
            'total_pnl': total_pnl,
            'equity': self.equity,
            'return_pct': (self.equity - self.initial_capital) / self.initial_capital * 100,
            'open_positions': len(self.positions),
        }


# =============================================================================
# MAIN BOT
# =============================================================================

class FlowHFTBot:
    """Main trading bot combining flow monitoring and execution."""

    def __init__(self, capital: float = 100.0):
        self.addresses = ExchangeAddressManager()
        self.monitor = MempoolFlowMonitor(self.addresses)
        self.executor = KrakenExecutor(capital)
        self.running = False
        self.last_signal_time = 0
        self.signal_cooldown = 30  # Seconds between signals

    async def initialize(self):
        """Initialize the bot."""
        print("=" * 60)
        print("BLOCKCHAIN FLOW HFT BOT")
        print("=" * 60)
        print(f"Capital: ${self.executor.equity:.2f}")
        print(f"Leverage: {LEVERAGE}x")
        print(f"TP/SL: {TP_PCT*100}% / {SL_PCT*100}%")
        print("=" * 60)

        # Try to load from multiple possible paths
        for path in EXCHANGES_PATHS:
            if path.exists():
                print(f"[INIT] Found addresses at: {path}")
                if self.addresses.load_from_json(path):
                    break
                else:
                    print(f"[INIT] Failed to load from {path}, trying next...")

        if not self.addresses.loaded:
            print("[ERROR] No exchange addresses loaded!")
            return False

        print()
        return True

    async def run(self):
        """Main run loop."""
        if not await self.initialize():
            return

        self.running = True
        self.monitor.running = True

        print("[BOT] Starting mempool monitoring...", flush=True)
        print(flush=True)

        # Start polling mempool
        poll_task = asyncio.create_task(self.monitor.poll_mempool())

        iteration = 0
        try:
            while self.running:
                try:
                    # Check for exit signals
                    await self.executor.check_exits()

                    # Check for entry signals
                    signal = self.monitor.get_signal()
                    if signal:
                        direction, confidence, reason = signal
                        now = time.time()

                        if now - self.last_signal_time >= self.signal_cooldown:
                            print(f"[SIGNAL] Direction: {'+1 LONG' if direction > 0 else '-1 SHORT'} | "
                                  f"Confidence: {confidence:.2f} | {reason}", flush=True)

                            await self.executor.open_position(direction, reason)
                            self.last_signal_time = now

                    # Log stats periodically
                    iteration += 1
                    if iteration % 30 == 0:  # Every 30 seconds
                        stats = self.executor.get_stats()
                        flow_stats = self.monitor.stats
                        window = self.monitor.current_window

                        whale_count = flow_stats.get('whale_txs', 0)
                        whale_btc = flow_stats.get('whale_btc', 0.0)
                        print(f"[STATS] Equity: ${stats['equity']:.2f} | "
                              f"Trades: {stats['trades']} | "
                              f"Win Rate: {stats.get('win_rate', 0):.1f}% | "
                              f"Flows: {flow_stats['flows_detected']} | "
                              f"Whales: {whale_count} ({whale_btc:.0f} BTC) | "
                              f"TXs: {flow_stats['txs_processed']} | "
                              f"FIS: {window.imbalance:.2f}", flush=True)

                except Exception as loop_err:
                    print(f"[ERROR] Loop error: {loop_err}", flush=True)
                    await asyncio.sleep(1)

                await asyncio.sleep(1)

        except KeyboardInterrupt:
            print("\n[BOT] Shutting down...", flush=True)
        finally:
            self.running = False
            self.monitor.running = False
            poll_task.cancel()

            # Final stats
            stats = self.executor.get_stats()
            print()
            print("=" * 60)
            print("FINAL STATS")
            print("=" * 60)
            print(f"Trades: {stats['trades']}")
            print(f"Wins: {stats.get('wins', 0)}")
            print(f"Win Rate: {stats.get('win_rate', 0):.1f}%")
            print(f"Total P&L: ${stats.get('total_pnl', 0):.2f}")
            print(f"Final Equity: ${stats['equity']:.2f}")
            print(f"Return: {stats.get('return_pct', 0):.1f}%")
            print("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description='Blockchain Flow HFT Bot')
    parser.add_argument('--capital', type=float, default=100.0, help='Starting capital')
    args = parser.parse_args()

    bot = FlowHFTBot(capital=args.capital)
    await bot.run()


if __name__ == '__main__':
    asyncio.run(main())
