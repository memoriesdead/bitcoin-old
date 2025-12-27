#!/usr/bin/env python3
"""
MEAN REVERSION BOT + BLOCKCHAIN SIGNALS
=======================================
Enhanced strategy combining:
1. Price-based mean reversion (trade against 2%+ moves)
2. Blockchain signals (FIS, MPI, WM, CR, CRS)

Integration Logic:
- Price signal = primary trigger
- Blockchain signal = filter/enhancer

Signal Combinations:
- Price LONG + Blockchain bullish (CRS > 0) = STRONG LONG (full size)
- Price LONG + Blockchain bearish (CRS < 0) = SKIP or reduce size
- Price SHORT + Blockchain bearish = STRONG SHORT (full size)
- Price SHORT + Blockchain bullish = SKIP or reduce size
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import numpy as np
import websockets
import httpx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Config - Mean Reversion
LOOKBACK_HOURS = 6
THRESHOLD_PCT = 2.0
LEVERAGE = 5
STOP_LOSS_PCT = 1.0
TAKE_PROFIT_PCT = 1.5
HOLD_HOURS = 6

# Config - Blockchain Integration
SIGNALS_PATH = Path("/root/sovereign/signals.json")
SIGNAL_MAX_AGE_SECONDS = 1800  # 30 min - signals older than this are stale
CRS_STRONG_THRESHOLD = 0.3    # Strong signal threshold
CRS_WEAK_THRESHOLD = 0.1      # Weak signal threshold
CONFIDENCE_THRESHOLD = 0.5    # Minimum confidence to use signal

# Position sizing based on signal alignment
SIZE_FULL = 1.0         # Both signals agree
SIZE_REDUCED = 0.5      # Only price signal
SIZE_SKIP = 0.0         # Signals conflict

OUTPUT_DIR = Path("data/mean_reversion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BlockchainSignal:
    """Blockchain signal data."""
    timestamp: int
    block_height: int
    fis: float
    mpi: float
    wm: float
    cr: float
    crs: float
    crs_direction: int  # -1, 0, +1
    confidence: float
    age_seconds: int = 0
    is_stale: bool = False


@dataclass
class Trade:
    id: str
    direction: str
    entry_price: float
    entry_time: float
    stop_loss: float
    take_profit: float
    size_usd: float
    size_multiplier: float  # NEW: size adjustment based on signals
    status: str
    blockchain_signal: Optional[Dict] = None  # NEW: signal at entry
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    pnl_pct: Optional[float] = None
    pnl_usd: Optional[float] = None
    exit_reason: Optional[str] = None


class MeanReversionBlockchainBot:
    def __init__(self, capital: float = 100.0, use_blockchain: bool = True):
        self.capital = capital
        self.equity = capital
        self.use_blockchain = use_blockchain

        # Price history
        self.hourly_prices: deque = deque(maxlen=24)
        self.current_hour_prices: List[float] = []
        self.last_hour: Optional[int] = None

        # Trading state
        self.current_trade: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.wins = 0
        self.losses = 0

        # Stats
        self.ticks_received = 0
        self.last_log_time = 0

        # Blockchain signal cache
        self.last_signal: Optional[BlockchainSignal] = None

        self._load_state()

    def _load_state(self):
        state_file = OUTPUT_DIR / "state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                self.equity = data.get("equity", self.capital)
                self.wins = data.get("wins", 0)
                self.losses = data.get("losses", 0)
                for t in data.get("trades", []):
                    self.trades.append(Trade(**t))
                if data.get("current_trade"):
                    self.current_trade = Trade(**data["current_trade"])
                logger.info(f"Loaded state: ${self.equity:.2f}, {len(self.trades)} trades")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def _save_state(self):
        state_file = OUTPUT_DIR / "state.json"
        data = {
            "updated": datetime.now().isoformat(),
            "equity": self.equity,
            "capital": self.capital,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.wins / (self.wins + self.losses) * 100 if (self.wins + self.losses) > 0 else 0,
            "trades": [asdict(t) for t in self.trades[-100:]],
            "current_trade": asdict(self.current_trade) if self.current_trade else None,
            "blockchain_enabled": self.use_blockchain
        }
        with open(state_file, "w") as f:
            json.dump(data, f, indent=2)

    def read_blockchain_signal(self) -> Optional[BlockchainSignal]:
        """Read latest blockchain signal from signals.json."""
        if not self.use_blockchain:
            return None

        if not SIGNALS_PATH.exists():
            logger.debug("No signals.json found")
            return None

        try:
            with open(SIGNALS_PATH) as f:
                data = json.load(f)

            signal = BlockchainSignal(
                timestamp=data.get("timestamp", 0),
                block_height=data.get("block_height", 0),
                fis=data.get("fis", 0),
                mpi=data.get("mpi", 0),
                wm=data.get("wm", 0),
                cr=data.get("cr", 0),
                crs=data.get("crs", 0),
                crs_direction=data.get("crs_direction", 0),
                confidence=data.get("confidence", 0)
            )

            # Check staleness
            signal.age_seconds = int(time.time()) - signal.timestamp
            signal.is_stale = signal.age_seconds > SIGNAL_MAX_AGE_SECONDS

            self.last_signal = signal
            return signal

        except Exception as e:
            logger.warning(f"Failed to read signals: {e}")
            return None

    def evaluate_trade_decision(self, price_direction: str, blockchain: Optional[BlockchainSignal]) -> Dict:
        """
        Evaluate whether to take the trade based on price + blockchain signals.

        Returns:
            {
                'action': 'TAKE' | 'SKIP' | 'REDUCE',
                'size_multiplier': 0.0 - 1.0,
                'reason': str
            }
        """
        # No blockchain signal - use price signal only with reduced size
        if blockchain is None or blockchain.is_stale:
            return {
                'action': 'TAKE',
                'size_multiplier': SIZE_REDUCED,
                'reason': 'No blockchain signal - using price only'
            }

        # Low confidence - use price signal only
        if blockchain.confidence < CONFIDENCE_THRESHOLD:
            return {
                'action': 'TAKE',
                'size_multiplier': SIZE_REDUCED,
                'reason': f'Low blockchain confidence ({blockchain.confidence:.0%})'
            }

        # Determine blockchain bias
        crs = blockchain.crs
        blockchain_bullish = crs > CRS_WEAK_THRESHOLD
        blockchain_bearish = crs < -CRS_WEAK_THRESHOLD
        blockchain_strong_bullish = crs > CRS_STRONG_THRESHOLD
        blockchain_strong_bearish = crs < -CRS_STRONG_THRESHOLD

        # Price LONG scenarios
        if price_direction == "LONG":
            if blockchain_strong_bullish:
                return {
                    'action': 'TAKE',
                    'size_multiplier': SIZE_FULL,
                    'reason': f'STRONG: Price LONG + Blockchain BULLISH (CRS={crs:+.3f})'
                }
            elif blockchain_bullish:
                return {
                    'action': 'TAKE',
                    'size_multiplier': SIZE_FULL * 0.8,
                    'reason': f'ALIGNED: Price LONG + Blockchain bullish (CRS={crs:+.3f})'
                }
            elif blockchain_bearish:
                return {
                    'action': 'SKIP',
                    'size_multiplier': SIZE_SKIP,
                    'reason': f'CONFLICT: Price LONG but Blockchain BEARISH (CRS={crs:+.3f})'
                }
            else:  # Neutral
                return {
                    'action': 'TAKE',
                    'size_multiplier': SIZE_REDUCED,
                    'reason': f'NEUTRAL: Price LONG, Blockchain neutral (CRS={crs:+.3f})'
                }

        # Price SHORT scenarios
        elif price_direction == "SHORT":
            if blockchain_strong_bearish:
                return {
                    'action': 'TAKE',
                    'size_multiplier': SIZE_FULL,
                    'reason': f'STRONG: Price SHORT + Blockchain BEARISH (CRS={crs:+.3f})'
                }
            elif blockchain_bearish:
                return {
                    'action': 'TAKE',
                    'size_multiplier': SIZE_FULL * 0.8,
                    'reason': f'ALIGNED: Price SHORT + Blockchain bearish (CRS={crs:+.3f})'
                }
            elif blockchain_bullish:
                return {
                    'action': 'SKIP',
                    'size_multiplier': SIZE_SKIP,
                    'reason': f'CONFLICT: Price SHORT but Blockchain BULLISH (CRS={crs:+.3f})'
                }
            else:  # Neutral
                return {
                    'action': 'TAKE',
                    'size_multiplier': SIZE_REDUCED,
                    'reason': f'NEUTRAL: Price SHORT, Blockchain neutral (CRS={crs:+.3f})'
                }

        return {
            'action': 'SKIP',
            'size_multiplier': 0.0,
            'reason': 'Unknown direction'
        }

    def on_price(self, price: float):
        """Process each price tick."""
        self.ticks_received += 1
        now = time.time()
        current_hour = int(now // 3600)

        # Track hourly prices
        if self.last_hour is None:
            self.last_hour = current_hour

        if current_hour != self.last_hour:
            if self.current_hour_prices:
                hourly_close = self.current_hour_prices[-1]
                self.hourly_prices.append({
                    "hour": self.last_hour,
                    "close": hourly_close,
                    "time": now
                })
                logger.info(f"Hour closed: ${hourly_close:,.0f} ({len(self.hourly_prices)} hours tracked)")
            self.current_hour_prices = []
            self.last_hour = current_hour

        self.current_hour_prices.append(price)

        # Check exit conditions
        if self.current_trade:
            self._check_exit(price, now)

        # Check entry conditions
        if len(self.hourly_prices) >= LOOKBACK_HOURS and not self.current_trade:
            self._check_entry(price, now)

        # Log status every 60 seconds
        if now - self.last_log_time > 60:
            self._log_status(price)
            self.last_log_time = now

    def _check_entry(self, price: float, now: float):
        """Check if we should enter a trade."""
        price_6h_ago = self.hourly_prices[-LOOKBACK_HOURS]["close"]
        move_pct = (price - price_6h_ago) / price_6h_ago * 100

        if abs(move_pct) < THRESHOLD_PCT:
            return  # No signal

        # Determine price direction (mean reversion = opposite of move)
        price_direction = "SHORT" if move_pct > THRESHOLD_PCT else "LONG"

        # Read blockchain signal
        blockchain = self.read_blockchain_signal()

        # Evaluate trade decision
        decision = self.evaluate_trade_decision(price_direction, blockchain)

        logger.info(f"Signal: {price_direction} (6h move: {move_pct:+.2f}%)")
        logger.info(f"Decision: {decision['action']} - {decision['reason']}")

        if decision['action'] == 'SKIP':
            logger.info("Trade SKIPPED due to signal conflict")
            return

        # Calculate position
        if price_direction == "LONG":
            stop_loss = price * (1 - STOP_LOSS_PCT / 100)
            take_profit = price * (1 + TAKE_PROFIT_PCT / 100)
        else:
            stop_loss = price * (1 + STOP_LOSS_PCT / 100)
            take_profit = price * (1 - TAKE_PROFIT_PCT / 100)

        # Adjusted size based on signal strength
        size_multiplier = decision['size_multiplier']
        size_usd = self.equity * LEVERAGE * size_multiplier

        # Prepare blockchain signal snapshot for trade record
        signal_snapshot = None
        if blockchain:
            signal_snapshot = {
                'crs': blockchain.crs,
                'crs_direction': blockchain.crs_direction,
                'confidence': blockchain.confidence,
                'components': {
                    'fis': blockchain.fis,
                    'mpi': blockchain.mpi,
                    'wm': blockchain.wm,
                    'cr': blockchain.cr
                },
                'age_seconds': blockchain.age_seconds
            }

        self.current_trade = Trade(
            id=f"MRB_{int(now)}",
            direction=price_direction,
            entry_price=price,
            entry_time=now,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size_usd=size_usd,
            size_multiplier=size_multiplier,
            status="OPEN",
            blockchain_signal=signal_snapshot
        )

        logger.info("=" * 60)
        logger.info(f"TRADE OPENED: {price_direction}")
        logger.info(f"6h move: {move_pct:+.2f}% -> Mean reversion {price_direction}")
        logger.info(f"Entry: ${price:,.2f}")
        logger.info(f"Stop Loss: ${stop_loss:,.2f} | Take Profit: ${take_profit:,.2f}")
        logger.info(f"Size: ${size_usd:,.2f} ({LEVERAGE}x * {size_multiplier:.0%})")
        if blockchain:
            logger.info(f"Blockchain: CRS={blockchain.crs:+.4f} Conf={blockchain.confidence:.0%}")
        logger.info("=" * 60)

        self._save_state()

    def _check_exit(self, price: float, now: float):
        """Check if we should exit the trade."""
        trade = self.current_trade

        if trade.direction == "LONG":
            pnl_pct = (price - trade.entry_price) / trade.entry_price * 100
            hit_sl = price <= trade.stop_loss
            hit_tp = price >= trade.take_profit
        else:
            pnl_pct = (trade.entry_price - price) / trade.entry_price * 100
            hit_sl = price >= trade.stop_loss
            hit_tp = price <= trade.take_profit

        hours_held = (now - trade.entry_time) / 3600
        time_exit = hours_held >= HOLD_HOURS

        exit_reason = None
        if hit_sl:
            exit_reason = "STOP_LOSS"
        elif hit_tp:
            exit_reason = "TAKE_PROFIT"
        elif time_exit:
            exit_reason = "TIME_EXIT"

        if exit_reason:
            self._close_trade(price, now, pnl_pct, exit_reason)

    def _close_trade(self, price: float, now: float, pnl_pct: float, reason: str):
        """Close the current trade."""
        trade = self.current_trade

        leveraged_pnl_pct = pnl_pct * LEVERAGE
        pnl_usd = (self.equity * trade.size_multiplier) * (leveraged_pnl_pct / 100)

        trade.exit_price = price
        trade.exit_time = now
        trade.pnl_pct = leveraged_pnl_pct
        trade.pnl_usd = pnl_usd
        trade.exit_reason = reason
        trade.status = "CLOSED"

        self.equity += pnl_usd

        if pnl_usd > 0:
            self.wins += 1
        else:
            self.losses += 1

        self.trades.append(trade)
        self.current_trade = None

        win_rate = self.wins / (self.wins + self.losses) * 100

        logger.info("=" * 60)
        logger.info(f"TRADE CLOSED: {reason}")
        logger.info(f"Direction: {trade.direction} | Size mult: {trade.size_multiplier:.0%}")
        logger.info(f"Entry: ${trade.entry_price:,.2f} -> Exit: ${price:,.2f}")
        logger.info(f"P&L: {leveraged_pnl_pct:+.2f}% (${pnl_usd:+.2f})")
        logger.info(f"New Equity: ${self.equity:.2f}")
        logger.info(f"Record: {self.wins}W-{self.losses}L ({win_rate:.1f}%)")
        if trade.blockchain_signal:
            logger.info(f"Entry CRS was: {trade.blockchain_signal['crs']:+.4f}")
        logger.info("=" * 60)

        self._save_state()

    def _log_status(self, price: float):
        """Log current status."""
        hours_tracked = len(self.hourly_prices)

        if hours_tracked >= LOOKBACK_HOURS:
            price_6h_ago = self.hourly_prices[-LOOKBACK_HOURS]["close"]
            move_6h = (price - price_6h_ago) / price_6h_ago * 100
        else:
            move_6h = 0

        # Get blockchain signal
        blockchain = self.read_blockchain_signal()

        logger.info("-" * 60)
        logger.info(f"BTC: ${price:,.0f} | 6h move: {move_6h:+.2f}%")
        logger.info(f"Hours tracked: {hours_tracked}/{LOOKBACK_HOURS} | Ticks: {self.ticks_received}")
        logger.info(f"Equity: ${self.equity:.2f} | Trades: {self.wins}W-{self.losses}L")

        if blockchain:
            stale_str = " (STALE)" if blockchain.is_stale else ""
            logger.info(f"Blockchain: CRS={blockchain.crs:+.4f} Dir={blockchain.crs_direction} "
                       f"Conf={blockchain.confidence:.0%} Age={blockchain.age_seconds}s{stale_str}")
        else:
            logger.info("Blockchain: No signal available")

        if self.current_trade:
            t = self.current_trade
            if t.direction == "LONG":
                pnl = (price - t.entry_price) / t.entry_price * 100 * LEVERAGE
            else:
                pnl = (t.entry_price - price) / t.entry_price * 100 * LEVERAGE
            logger.info(f"OPEN: {t.direction} @ ${t.entry_price:,.0f} ({pnl:+.2f}%) size={t.size_multiplier:.0%}")
        else:
            if abs(move_6h) >= THRESHOLD_PCT:
                direction = "SHORT" if move_6h > 0 else "LONG"
                logger.info(f"SIGNAL: {direction} (waiting for next tick)")
            else:
                needed = THRESHOLD_PCT - abs(move_6h)
                logger.info(f"Waiting: need {needed:.2f}% more move for signal")
        logger.info("-" * 60)


async def fetch_historical_hourly() -> List[Dict]:
    """Fetch last 24 hours of hourly candles."""
    url = "https://api.hyperliquid.xyz/info"
    end_time = int(time.time() * 1000)
    start_time = end_time - (24 * 60 * 60 * 1000)

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": "BTC",
            "interval": "1h",
            "startTime": start_time,
            "endTime": end_time
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=30)
        candles = response.json()

    return candles


async def run_bot(capital: float = 100.0, use_blockchain: bool = True):
    """Run the bot with WebSocket connection."""
    logger.info("=" * 60)
    logger.info("MEAN REVERSION + BLOCKCHAIN BOT")
    logger.info(f"Config: {LOOKBACK_HOURS}h lookback, {THRESHOLD_PCT}% threshold")
    logger.info(f"Risk: {LEVERAGE}x leverage, {STOP_LOSS_PCT}% SL, {TAKE_PROFIT_PCT}% TP")
    logger.info(f"Blockchain: {'ENABLED' if use_blockchain else 'DISABLED'}")
    logger.info("=" * 60)

    bot = MeanReversionBlockchainBot(capital=capital, use_blockchain=use_blockchain)

    # Warm up with historical data
    logger.info("Fetching historical hourly data...")
    candles = await fetch_historical_hourly()

    for candle in candles[:-1]:
        close = float(candle["c"])
        hour = int(candle["t"] / 1000 / 3600)
        bot.hourly_prices.append({"hour": hour, "close": close, "time": candle["t"]/1000})

    logger.info(f"Warmed up with {len(bot.hourly_prices)} hourly candles")

    # Connect to WebSocket
    uri = "wss://api.hyperliquid.xyz/ws"

    while True:
        try:
            async with websockets.connect(uri) as ws:
                subscribe = {
                    "method": "subscribe",
                    "subscription": {"type": "trades", "coin": "BTC"}
                }
                await ws.send(json.dumps(subscribe))
                logger.info("Connected to Hyperliquid WebSocket")

                async for message in ws:
                    try:
                        data = json.loads(message)
                        if data.get("channel") == "trades":
                            for trade in data.get("data", []):
                                price = float(trade.get("px", 0))
                                if price > 0:
                                    bot.on_price(price)
                    except Exception as e:
                        logger.error(f"Message error: {e}")

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            logger.info("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=100.0)
    parser.add_argument("--no-blockchain", action="store_true", help="Disable blockchain signals")
    args = parser.parse_args()

    asyncio.run(run_bot(capital=args.capital, use_blockchain=not args.no_blockchain))
