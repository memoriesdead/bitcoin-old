#!/usr/bin/env python3
"""
LIVE TRADER - Executes trades on all exchanges via CCXT
Reads signals from C++ blockchain runner on VPS and executes in real-time.

EXCHANGES: kraken, okx, bitget (configured accounts)
SIGNALS: SHORT on inflow, LONG on outflow

MODES:
  - PAPER: Simulated trades with real prices (default)
  - LIVE:  Real execution when credentials configured
"""

import time
import json
import sqlite3
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
import re
import os

# Try to import ccxt, fallback to paper mode if not available
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("[WARN] CCXT not installed - paper trading only")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TradingConfig:
    # Mode
    paper_mode: bool = True               # Paper trading (no real orders)

    # Position sizing
    position_size_usd: float = 100.0      # USD per trade
    max_positions: int = 5                 # Max concurrent positions
    max_position_per_exchange: int = 2     # Max positions per exchange

    # Risk management - FLOW WITH PRICE
    stop_loss_pct: float = 0.01           # 1% stop loss (tight)
    take_profit_pct: float = 0.02         # 2% take profit
    trailing_stop_pct: float = 0.005      # 0.5% trailing stop (locks in profit)
    use_trailing_stop: bool = True        # Enable trailing stop
    max_hold_seconds: int = 0             # 0 = no timeout, ride until TP/SL

    # Signal thresholds
    min_flow_btc: float = 5.0             # 5 BTC minimum (ignore small flows)
    min_confidence: float = 0.5           # Lower confidence threshold

    # Exchange mapping (signal exchange -> trading exchange)
    # We execute on our connected exchanges regardless of signal source
    trading_exchanges: List[str] = None

    def __post_init__(self):
        if self.trading_exchanges is None:
            # TIER 1: ALL USA-LEGAL EXCHANGES
            self.trading_exchanges = [
                # Major US Exchanges
                'kraken',       # San Francisco - Top US volume
                'coinbase',     # San Francisco - Largest US retail
                'gemini',       # New York - Regulated, Winklevoss
                'bitstamp',     # US accessible - Oldest exchange
                'binanceus',    # San Francisco - Binance US entity
                # Additional US-Legal
                'cryptocom',    # US available - Major global player
                'okcoin',       # San Francisco - OKX US entity
                'bitflyer',     # US licensed (bitFlyerUSA)
                'luno',         # US accessible
            ]

# =============================================================================
# EXCHANGE TIERS (for future expansion)
# =============================================================================

EXCHANGE_TIERS = {
    'tier1_usa': [
        # Major US Exchanges
        'kraken',       # San Francisco - Top US volume
        'coinbase',     # San Francisco - Largest US retail
        'gemini',       # New York - Regulated, Winklevoss
        'bitstamp',     # US accessible - Oldest exchange
        'binanceus',    # San Francisco - Binance US entity

        # Additional US-Legal Exchanges
        'cryptocom',    # US available - Major global player
        'okcoin',       # San Francisco - OKX US entity
        'bitflyer',     # US licensed (bitFlyerUSA)
        'luno',         # US accessible
    ],
    'tier2_global_major': [
        'binance',      # Global leader
        'okx',          # Top 3 global
        'bybit',        # Top derivatives
        'bitget',       # Fast growing
        'kucoin',       # Altcoin leader
        'gateio',       # Wide selection
        'htx',          # Huobi rebranded
        'mexc',         # High volume
    ],
    'tier3_europe': [
        'bitvavo',      # Netherlands - 50% EUR liquidity
        'bitpanda',     # Austria - MiCA regulated
    ],
    'tier4_asia': [
        'upbit',        # Korea - Major volume
        'bithumb',      # Korea
        'bitflyer',     # Japan - Regulated
    ],
    'tier5_derivatives': [
        'deribit',      # Options leader
        'bitmex',       # Perps pioneer
        'phemex',       # Singapore
    ],
}


@dataclass
class Position:
    exchange: str
    side: str  # 'long' or 'short'
    entry_price: float
    size_usd: float
    size_btc: float
    entry_time: float
    stop_loss: float
    take_profit: float
    order_id: Optional[str] = None
    signal_exchange: str = ""  # Which exchange triggered the signal
    flow_btc: float = 0.0
    # Trailing stop tracking
    best_price: float = 0.0    # Best price since entry (highest for long, lowest for short)
    trailing_stop: float = 0.0  # Dynamic stop that moves with price


# =============================================================================
# CCXT EXCHANGE CONNECTIONS
# =============================================================================

class ExchangeManager:
    """Manages connections to all trading exchanges via CCXT."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.balances: Dict[str, dict] = {}

    def connect_all(self):
        """Connect to all configured exchanges."""
        # Load API keys from environment or config file
        config_path = Path.home() / '.ccxt_config.json'

        if config_path.exists():
            with open(config_path) as f:
                api_config = json.load(f)
        else:
            # Try environment variables
            api_config = {}

        for exchange_id in self.config.trading_exchanges:
            try:
                # Direct exchange ID (kraken, coinbase, etc.)

                if exchange_id in api_config:
                    creds = api_config[exchange_id]
                    exchange_class = getattr(ccxt, exchange_id)
                    self.exchanges[exchange_id] = exchange_class({
                        'apiKey': creds.get('apiKey'),
                        'secret': creds.get('secret'),
                        'password': creds.get('password'),  # For OKX
                        'enableRateLimit': True,
                    })
                    print(f"[CONNECTED] {exchange_id}")
                else:
                    # Paper mode - just create public exchange for prices
                    exchange_class = getattr(ccxt, exchange_id)
                    self.exchanges[exchange_id] = exchange_class({'enableRateLimit': True})
                    print(f"[PAPER] {exchange_id} - public data only")

            except Exception as e:
                print(f"[ERROR] Failed to connect {exchange_id}: {e}")

    def get_price(self, exchange_name: str, symbol: str = 'BTC/USDT') -> Optional[float]:
        """Get current price from exchange."""
        try:
            if exchange_name in self.exchanges:
                ticker = self.exchanges[exchange_name].fetch_ticker(symbol)
                return ticker['last']
        except Exception as e:
            print(f"[ERROR] Price fetch failed {exchange_name}: {e}")
        return None

    def place_order(self, exchange_name: str, side: str, amount_btc: float,
                    symbol: str = 'BTC/USDT') -> Optional[dict]:
        """Place market order on exchange."""
        try:
            if exchange_name in self.exchanges:
                exchange = self.exchanges[exchange_name]
                order = exchange.create_market_order(symbol, side, amount_btc)
                return order
        except Exception as e:
            print(f"[ERROR] Order failed {exchange_name}: {e}")
        return None


# =============================================================================
# SIGNAL PARSER
# =============================================================================

class SignalParser:
    """Parses signals from C++ blockchain runner log on VPS."""

    # Pattern: [SHORT] exchange | In: X | Out: Y | Net: Z | Latency: Nns
    SIGNAL_PATTERN = re.compile(
        r'\[(SHORT|LONG)\]\s+([^|]+)\s*\|\s*In:\s*([\d.]+)\s*\|\s*Out:\s*([\d.]+)\s*\|\s*Net:\s*([+-]?[\d.]+)'
    )

    def __init__(self, vps_host: str = 'root@31.97.211.217',
                 log_path: str = '/root/sovereign/cpp_runner.log'):
        self.vps_host = vps_host
        self.log_path = log_path
        self.last_lines = 0
        self.seen_signals = set()
        # Check if we're running locally on VPS (log file exists)
        self.is_local = os.path.exists(log_path)
        if self.is_local:
            print(f"[INFO] Running on VPS - reading signals directly from {log_path}")

    def get_new_signals(self) -> List[dict]:
        """Read new signals from log file (local if on VPS, SSH if remote)."""
        signals = []

        try:
            if self.is_local:
                # Read directly from local file
                if not os.path.exists(self.log_path):
                    return signals
                result = subprocess.run(
                    ['tail', '-100', self.log_path],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode != 0:
                    return signals
                lines = result.stdout.strip().split('\n')
            else:
                # Fetch via SSH from remote VPS
                result = subprocess.run(
                    ['ssh', '-o', 'ConnectTimeout=10', self.vps_host, f'tail -100 {self.log_path}'],
                    capture_output=True, text=True, timeout=20
                )
                if result.returncode != 0:
                    return signals
                lines = result.stdout.strip().split('\n')

            for line in lines:
                # Remove ANSI codes
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)

                match = self.SIGNAL_PATTERN.search(clean_line)
                if match:
                    direction = match.group(1)
                    exchanges = match.group(2).strip()
                    inflow = float(match.group(3))
                    outflow = float(match.group(4))
                    net_flow = float(match.group(5))

                    # Create unique signal ID with timestamp component
                    signal_id = f"{direction}_{exchanges}_{inflow:.4f}_{outflow:.4f}"

                    if signal_id not in self.seen_signals:
                        self.seen_signals.add(signal_id)

                        signals.append({
                            'direction': direction,
                            'exchanges': exchanges,
                            'inflow_btc': inflow,
                            'outflow_btc': outflow,
                            'net_flow': net_flow,
                            'flow_btc': inflow if direction == 'SHORT' else outflow,
                            'timestamp': time.time()
                        })

                        # Keep seen_signals from growing forever
                        if len(self.seen_signals) > 10000:
                            self.seen_signals = set(list(self.seen_signals)[-5000:])

        except subprocess.TimeoutExpired:
            print("[WARN] SSH timeout fetching signals")
        except Exception as e:
            print(f"[ERROR] Signal parse failed: {e}")

        return signals


# =============================================================================
# LIVE TRADER
# =============================================================================

class LiveTrader:
    """Main trading engine - executes trades based on blockchain signals."""

    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.exchange_mgr = ExchangeManager(self.config)
        self.signal_parser = SignalParser()
        self.positions: Dict[str, Position] = {}
        self.trade_count = 0
        self.pnl_total = 0.0
        self.running = False

        # Trade database (local)
        self.db_path = str(Path(__file__).parent / 'live_trades.db')
        self._init_database()

        # Price cache to avoid repeated API calls
        self._price_cache = {}
        self._price_cache_time = 0

    def _init_database(self):
        """Initialize trade tracking database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                signal_exchange TEXT,
                trading_exchange TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                size_btc REAL,
                size_usd REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                flow_btc REAL,
                hold_seconds REAL,
                exit_reason TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _log_trade(self, position: Position, exit_price: float, exit_reason: str):
        """Log completed trade to database."""
        hold_time = time.time() - position.entry_time

        if position.side == 'short':
            pnl_pct = (position.entry_price - exit_price) / position.entry_price
        else:
            pnl_pct = (exit_price - position.entry_price) / position.entry_price

        pnl_usd = position.size_usd * pnl_pct

        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO trades (timestamp, signal_exchange, trading_exchange, direction,
                              entry_price, exit_price, size_btc, size_usd, pnl_usd, pnl_pct,
                              flow_btc, hold_seconds, exit_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            position.signal_exchange,
            position.exchange,
            position.side,
            position.entry_price,
            exit_price,
            position.size_btc,
            position.size_usd,
            pnl_usd,
            pnl_pct,
            position.flow_btc,
            hold_time,
            exit_reason
        ))
        conn.commit()
        conn.close()

        return pnl_usd

    def _select_trading_exchange(self) -> Optional[str]:
        """Select best exchange to trade on (round-robin for now)."""
        # Count positions per exchange
        positions_per_exchange = {}
        for pos in self.positions.values():
            positions_per_exchange[pos.exchange] = positions_per_exchange.get(pos.exchange, 0) + 1

        # Find exchange with fewest positions
        for exchange in self.config.trading_exchanges:
            count = positions_per_exchange.get(exchange, 0)
            if count < self.config.max_position_per_exchange:
                return exchange

        return None

    def _fetch_price(self, exchange_id: str) -> Optional[float]:
        """Fetch current BTC price from exchange."""
        try:
            if CCXT_AVAILABLE:
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({'enableRateLimit': True})
                ticker = exchange.fetch_ticker('BTC/USDT')
                return ticker['last']
        except Exception as e:
            print(f"[WARN] Price fetch failed for {exchange_id}: {e}")
        return None

    def process_signal(self, signal: dict):
        """Process a trading signal."""
        # Check if we can take more positions
        if len(self.positions) >= self.config.max_positions:
            return

        # Check minimum flow
        if signal['flow_btc'] < self.config.min_flow_btc:
            return

        # Select trading exchange
        trading_exchange = self._select_trading_exchange()
        if not trading_exchange:
            return

        # Get current price from exchange
        price = self._fetch_price(trading_exchange)

        if not price:
            # Try fallback exchanges
            for fallback in ['kraken', 'okx', 'bitget']:
                price = self._fetch_price(fallback)
                if price:
                    break

        if not price:
            print(f"[ERROR] Could not fetch price for {trading_exchange}")
            return

        # Calculate position size
        size_btc = self.config.position_size_usd / price

        # Calculate stops
        if signal['direction'] == 'SHORT':
            side = 'short'
            stop_loss = price * (1 + self.config.stop_loss_pct)
            take_profit = price * (1 - self.config.take_profit_pct)
        else:
            side = 'long'
            stop_loss = price * (1 - self.config.stop_loss_pct)
            take_profit = price * (1 + self.config.take_profit_pct)

        # Create position
        position_id = f"{trading_exchange}_{int(time.time()*1000)}"
        position = Position(
            exchange=trading_exchange,
            side=side,
            entry_price=price,
            size_usd=self.config.position_size_usd,
            size_btc=size_btc,
            entry_time=time.time(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_exchange=signal['exchanges'],
            flow_btc=signal['flow_btc']
        )

        self.positions[position_id] = position
        self.trade_count += 1

        mode = "[PAPER]" if self.config.paper_mode else "[LIVE]"
        print(f"{mode} [OPEN] {side.upper()} on {trading_exchange}")
        print(f"       Signal: {signal['exchanges']} | Flow: {signal['flow_btc']:.2f} BTC")
        print(f"       Entry: ${price:,.2f} | Size: ${self.config.position_size_usd}")
        print(f"       SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}")

    def _get_cached_price(self, exchange_id: str) -> Optional[float]:
        """Get price with 5-second cache."""
        now = time.time()
        if now - self._price_cache_time > 5:
            # Refresh cache
            price = self._fetch_price(exchange_id)
            if price:
                self._price_cache[exchange_id] = price
                self._price_cache_time = now
        return self._price_cache.get(exchange_id)

    def check_positions(self):
        """Check all positions for exit conditions."""
        positions_to_close = []

        for position_id, position in self.positions.items():
            exchange_id = position.exchange  # Direct exchange ID

            # Get current price with caching
            current_price = self._get_cached_price(exchange_id)
            if not current_price:
                current_price = self._fetch_price('kraken')  # Fallback
            if not current_price:
                continue  # Skip if no price available

            exit_reason = None

            # ================================================================
            # TRAILING STOP LOGIC - Lock in profits as price moves in our favor
            # ================================================================
            if self.config.use_trailing_stop:
                # Initialize best_price on first check
                if position.best_price == 0.0:
                    position.best_price = position.entry_price
                    position.trailing_stop = position.stop_loss  # Start at original SL

                if position.side == 'long':
                    # For LONG: track highest price, trail stop below it
                    if current_price > position.best_price:
                        position.best_price = current_price
                        # Move trailing stop up (but never below original stop)
                        new_trail = current_price * (1 - self.config.trailing_stop_pct)
                        position.trailing_stop = max(position.trailing_stop, new_trail)

                    # Check if trailing stop hit
                    if current_price <= position.trailing_stop:
                        exit_reason = 'trailing_stop'

                elif position.side == 'short':
                    # For SHORT: track lowest price, trail stop above it
                    if current_price < position.best_price:
                        position.best_price = current_price
                        # Move trailing stop down (but never above original stop)
                        new_trail = current_price * (1 + self.config.trailing_stop_pct)
                        position.trailing_stop = min(position.trailing_stop, new_trail)

                    # Check if trailing stop hit
                    if current_price >= position.trailing_stop:
                        exit_reason = 'trailing_stop'

            # ================================================================
            # FIXED STOP LOSS / TAKE PROFIT (checked if trailing stop not hit)
            # ================================================================
            if not exit_reason:
                # Check stop loss
                if position.side == 'short' and current_price >= position.stop_loss:
                    exit_reason = 'stop_loss'
                elif position.side == 'long' and current_price <= position.stop_loss:
                    exit_reason = 'stop_loss'

                # Check take profit
                elif position.side == 'short' and current_price <= position.take_profit:
                    exit_reason = 'take_profit'
                elif position.side == 'long' and current_price >= position.take_profit:
                    exit_reason = 'take_profit'

                # Check timeout (only if max_hold_seconds > 0)
                elif self.config.max_hold_seconds > 0 and \
                     time.time() - position.entry_time > self.config.max_hold_seconds:
                    exit_reason = 'timeout'

            if exit_reason:
                pnl = self._log_trade(position, current_price, exit_reason)
                self.pnl_total += pnl
                positions_to_close.append(position_id)

                mode = "[PAPER]" if self.config.paper_mode else "[LIVE]"
                pnl_color = '\033[92m' if pnl >= 0 else '\033[91m'
                print(f"{mode} [CLOSE] {position.side.upper()} {position.exchange} - {exit_reason}")
                print(f"        Entry: ${position.entry_price:,.2f} -> Exit: ${current_price:,.2f}")
                if exit_reason == 'trailing_stop':
                    print(f"        Best: ${position.best_price:,.2f} | Trail: ${position.trailing_stop:,.2f}")
                print(f"        {pnl_color}P&L: ${pnl:+.2f}\033[0m | Total: ${self.pnl_total:+.2f}")

        for position_id in positions_to_close:
            del self.positions[position_id]

    def run(self, timeout: int = 0):
        """Main trading loop.

        Args:
            timeout: Run for N seconds then exit (0 = run forever)
        """
        mode = "PAPER TRADING" if self.config.paper_mode else "LIVE TRADING"
        print("=" * 70)
        print(f"LIVE TRADER - {mode}")
        print("=" * 70)
        print(f"Mode: {'PAPER (simulated)' if self.config.paper_mode else 'LIVE (real orders)'}")
        print(f"Exchanges: {', '.join(self.config.trading_exchanges)}")
        print(f"Position Size: ${self.config.position_size_usd}")
        print(f"Max Positions: {self.config.max_positions}")
        print(f"Stop Loss: {self.config.stop_loss_pct*100:.1f}% | Take Profit: {self.config.take_profit_pct*100:.1f}%")
        if self.config.use_trailing_stop:
            print(f"Trailing Stop: {self.config.trailing_stop_pct*100:.1f}% (locks in profits)")
        print(f"Min Flow: {self.config.min_flow_btc} BTC | Hold: {'unlimited' if self.config.max_hold_seconds == 0 else f'{self.config.max_hold_seconds}s'}")
        print(f"Signal Source: VPS C++ Runner via SSH")
        if timeout > 0:
            print(f"Session Timeout: {timeout} seconds")
        print("=" * 70)
        print()

        self.running = True
        last_stats = time.time()
        start_time = time.time()

        while self.running:
            try:
                # Check timeout
                if timeout > 0 and (time.time() - start_time) >= timeout:
                    print(f"\n[TIMEOUT] {timeout} seconds reached. Shutting down...")
                    self.running = False
                    break

                # Get new signals
                signals = self.signal_parser.get_new_signals()

                for signal in signals:
                    self.process_signal(signal)

                # Check existing positions
                self.check_positions()

                # Print stats every 60 seconds
                if time.time() - last_stats >= 60:
                    elapsed = int(time.time() - start_time)
                    remaining = timeout - elapsed if timeout > 0 else 0
                    time_info = f" | Remaining: {remaining}s" if timeout > 0 else ""
                    print(f"\n[STATS] Trades: {self.trade_count} | Open: {len(self.positions)} | P&L: ${self.pnl_total:+.2f}{time_info}\n")
                    last_stats = time.time()

                time.sleep(0.1)  # 100ms loop

            except KeyboardInterrupt:
                self.running = False
                print("\nShutting down...")

            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(1)

        # Close all positions on shutdown
        for pos_id in list(self.positions.keys()):
            pos = self.positions[pos_id]
            price = self._fetch_price(pos.exchange)
            if price:
                pnl = self._log_trade(pos, price, "SESSION_END")
                self.pnl_total += pnl
                mode = "[PAPER]" if self.config.paper_mode else "[LIVE]"
                pnl_color = '\033[92m' if pnl >= 0 else '\033[91m'
                print(f"{mode} [CLOSE] {pos.side.upper()} {pos.exchange} - SESSION_END")
                print(f"        Entry: ${pos.entry_price:,.2f} -> Exit: ${price:,.2f}")
                print(f"        {pnl_color}P&L: ${pnl:+.2f}\033[0m | Total: ${self.pnl_total:+.2f}")
                del self.positions[pos_id]

        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        print(f"Total Trades: {self.trade_count}")
        print(f"Final P&L: ${self.pnl_total:+.2f}")
        print(f"Duration: {int(time.time() - start_time)} seconds")
        print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def get_exchanges_for_tiers(tiers: List[str]) -> List[str]:
    """Get all exchanges for specified tiers."""
    exchanges = []
    for tier in tiers:
        if tier in EXCHANGE_TIERS:
            exchanges.extend(EXCHANGE_TIERS[tier])
    return list(dict.fromkeys(exchanges))  # Remove duplicates, preserve order


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Live Trader - CCXT Multi-Exchange')
    parser.add_argument('--live', action='store_true', help='Enable LIVE trading (default: paper)')
    parser.add_argument('--size', type=float, default=100, help='Position size in USD')
    parser.add_argument('--max-positions', type=int, default=5, help='Max concurrent positions')
    parser.add_argument('--min-flow', type=float, default=5.0, help='Minimum flow in BTC (default: 5)')
    parser.add_argument('--stop-loss', type=float, default=0.01, help='Stop loss percentage (default: 0.01 = 1%)')
    parser.add_argument('--take-profit', type=float, default=0.02, help='Take profit percentage (default: 0.02 = 2%)')
    parser.add_argument('--trailing-stop', type=float, default=0.005, help='Trailing stop percentage (default: 0.005 = 0.5%)')
    parser.add_argument('--no-trailing', action='store_true', help='Disable trailing stop')
    parser.add_argument('--tiers', type=str, default='tier1_usa',
                        help='Exchange tiers: tier1_usa, tier2_global_major, tier3_europe, tier4_asia, tier5_derivatives (comma-separated)')
    parser.add_argument('--timeout', type=int, default=0,
                        help='Run for N seconds then exit (0 = run forever)')

    args = parser.parse_args()

    # Parse tiers
    selected_tiers = [t.strip() for t in args.tiers.split(',')]
    exchanges = get_exchanges_for_tiers(selected_tiers)

    if not exchanges:
        print(f"No exchanges found for tiers: {selected_tiers}")
        print(f"Available tiers: {list(EXCHANGE_TIERS.keys())}")
        exit(1)

    print(f"Selected tiers: {selected_tiers}")
    print(f"Exchanges: {exchanges}")

    config = TradingConfig(
        paper_mode=not args.live,
        position_size_usd=args.size,
        max_positions=args.max_positions,
        min_flow_btc=args.min_flow,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        trailing_stop_pct=args.trailing_stop,
        use_trailing_stop=not args.no_trailing,
        trading_exchanges=exchanges
    )

    trader = LiveTrader(config)
    trader.run(timeout=args.timeout)
