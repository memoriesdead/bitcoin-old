#!/usr/bin/env python3
"""
RenTech Paper Trader v2 - Hostinger VPS Version
================================================
- Warms up with historical daily candles from Hyperliquid
- Evaluates 9 RenTech formulas in real-time
- Properly calculates daily-based features
"""

import asyncio
import json
import time
import logging
import httpx
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum
import numpy as np
import websockets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/root/sovereign/paper_trader.log')
    ]
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path('/root/sovereign/data/paper_trades')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# FEATURE ENGINE WITH DAILY CANDLE SUPPORT
# ==============================================================================

@dataclass
class FeatureSnapshot:
    timestamp: float
    price: float
    ret_1d: float = 0.0
    ret_3d: float = 0.0
    ret_7d: float = 0.0
    ret_14d: float = 0.0
    ret_30d: float = 0.0
    ma30: float = 0.0
    price_vs_ma30: float = 0.0
    volatility_20d: float = 0.0
    bb_position: float = 0.0
    rsi_14: float = 50.0
    regime: str = "NEUTRAL"
    anomaly_score: float = 0.0


class FeatureEngine:
    """Feature calculation with historical daily candle support."""

    def __init__(self):
        self.daily_closes = deque(maxlen=200)  # Daily closing prices
        self.current_day_prices = []  # Tick prices for current day
        self.last_day = None
        self.last_features = None
        self.tick_count = 0

    def warm_up(self, historical_closes: List[float]):
        """Warm up with historical daily closes."""
        for close in historical_closes:
            self.daily_closes.append(close)
        logger.info(f"Warmed up with {len(self.daily_closes)} daily candles")

    def update(self, price: float, timestamp: float = None) -> FeatureSnapshot:
        ts = timestamp or time.time()
        self.tick_count += 1

        # Track current day
        current_day = datetime.fromtimestamp(ts).date()

        # New day? Roll over
        if self.last_day and current_day != self.last_day:
            if self.current_day_prices:
                # Use VWAP-like average of day's prices as close
                day_close = np.mean(self.current_day_prices)
                self.daily_closes.append(day_close)
                logger.info(f"New day: added close ${day_close:,.0f} (from {len(self.current_day_prices)} ticks)")
            self.current_day_prices = []

        self.last_day = current_day
        self.current_day_prices.append(price)

        # Calculate features using daily data + current price
        prices = list(self.daily_closes) + [price]  # Historical + current
        n = len(prices)

        snapshot = FeatureSnapshot(timestamp=ts, price=price)

        if n < 2:
            self.last_features = snapshot
            return snapshot

        # Returns (using daily closes)
        if n >= 2:
            snapshot.ret_1d = (price / prices[-2] - 1) * 100
        if n >= 4:
            snapshot.ret_3d = (price / prices[-4] - 1) * 100
        if n >= 8:
            snapshot.ret_7d = (price / prices[-8] - 1) * 100
        if n >= 15:
            snapshot.ret_14d = (price / prices[-15] - 1) * 100
        if n >= 31:
            snapshot.ret_30d = (price / prices[-31] - 1) * 100

        # MA and deviation
        if n >= 30:
            snapshot.ma30 = np.mean(prices[-30:])
            snapshot.price_vs_ma30 = (price / snapshot.ma30 - 1) * 100

        # Volatility (annualized)
        if n >= 21:
            returns = np.diff(prices[-21:]) / np.array(prices[-21:-1])
            snapshot.volatility_20d = np.std(returns) * np.sqrt(252) * 100

        # RSI
        if n >= 15:
            changes = np.diff(prices[-15:])
            gains = np.maximum(changes, 0)
            losses = np.abs(np.minimum(changes, 0))
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss > 1e-10:
                rs = avg_gain / avg_loss
                snapshot.rsi_14 = 100 - (100 / (1 + rs))
            else:
                snapshot.rsi_14 = 100

        # Bollinger position
        if n >= 20:
            ma20 = np.mean(prices[-20:])
            std20 = np.std(prices[-20:])
            if std20 > 0:
                snapshot.bb_position = (price - ma20) / (2 * std20)

        # Regime detection
        if snapshot.ret_7d < -15 and snapshot.volatility_20d > 50:
            snapshot.regime = "CAPITULATION"
        elif snapshot.ret_7d > 15 and snapshot.volatility_20d > 50:
            snapshot.regime = "EUPHORIA"
        elif snapshot.ret_7d < -5:
            snapshot.regime = "BEAR"
        elif snapshot.ret_7d > 5:
            snapshot.regime = "BULL"
        else:
            snapshot.regime = "NEUTRAL"

        # Anomaly score
        snapshot.anomaly_score = abs(snapshot.ret_7d / 10) + abs(snapshot.bb_position)

        self.last_features = snapshot
        return snapshot

    def is_ready(self) -> bool:
        return len(self.daily_closes) >= 30


# ==============================================================================
# SIGNAL EVALUATOR
# ==============================================================================

class SignalDirection(Enum):
    LONG = 1
    SHORT = -1


@dataclass
class Signal:
    formula_id: str
    formula_name: str
    direction: SignalDirection
    confidence: float
    hold_days: int
    entry_price: float


# The 9 validated RenTech formulas
FORMULAS = {
    "RENTECH_001": {
        "name": "EXTREME_ANOMALY_LONG",
        "direction": SignalDirection.LONG,
        "hold_days": 30,
        "confidence": 0.75,
        "check": lambda f: f.anomaly_score > 4 and f.ret_7d < -15,
    },
    "RENTECH_002": {
        "name": "VOLUME_MOMENTUM_CONFLUENCE",
        "direction": SignalDirection.LONG,
        "hold_days": 30,
        "confidence": 0.70,
        "check": lambda f: f.ret_7d > 5 and f.ret_3d > 2 and f.regime in ["BULL", "EUPHORIA"],
    },
    "RENTECH_003": {
        "name": "EXTREME_ANOMALY_LONG_7D",
        "direction": SignalDirection.LONG,
        "hold_days": 7,
        "confidence": 0.72,
        "check": lambda f: f.anomaly_score > 4 and f.ret_7d < -15,
    },
    "RENTECH_004": {
        "name": "CORRELATION_BREAK_BULL",
        "direction": SignalDirection.LONG,
        "hold_days": 7,
        "confidence": 0.68,
        "check": lambda f: f.price_vs_ma30 < -15 and f.ret_3d > 0,
    },
    "RENTECH_005": {
        "name": "BOLLINGER_BOUNCE",
        "direction": SignalDirection.LONG,
        "hold_days": 30,
        "confidence": 0.65,
        "check": lambda f: f.bb_position < -0.9 and f.volatility_20d < 80 and f.ret_3d > 0,
    },
    "RENTECH_006": {
        "name": "WHALE_ACCUMULATION",
        "direction": SignalDirection.LONG,
        "hold_days": 30,
        "confidence": 0.67,
        "check": lambda f: f.ret_7d < -10 and f.ret_3d > 0,
    },
    "RENTECH_007": {
        "name": "RSI_DIVERGENCE",
        "direction": SignalDirection.LONG,
        "hold_days": 30,
        "confidence": 0.66,
        "check": lambda f: f.ret_14d < -15 and f.rsi_14 > 35 and f.ret_3d > 0,
    },
    "RENTECH_009": {
        "name": "VOLATILITY_REGIME_SHIFT",
        "direction": SignalDirection.LONG,
        "hold_days": 30,
        "confidence": 0.64,
        "check": lambda f: f.regime == "CAPITULATION" and f.ret_3d > 0,
    },
    "RENTECH_015": {
        "name": "GOLDEN_CROSS_MOMENTUM",
        "direction": SignalDirection.LONG,
        "hold_days": 30,
        "confidence": 0.63,
        "check": lambda f: f.price_vs_ma30 > 5 and f.ret_7d > 10,
    },
}


# ==============================================================================
# PAPER TRADER
# ==============================================================================

@dataclass
class PaperTrade:
    trade_id: str
    formula_id: str
    formula_name: str
    direction: str
    entry_time: float
    entry_price: float
    hold_days: int
    target_exit_time: float
    confidence: float
    position_size: float
    leverage: float
    status: str
    exit_time: Optional[float] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


class PaperTrader:
    def __init__(self, capital: float = 100.0):
        self.capital = capital
        self.current_equity = capital
        self.features = FeatureEngine()
        self.trades: List[PaperTrade] = []
        self.open_trades: Dict[str, PaperTrade] = {}
        self.signals_received = 0
        self.trades_executed = 0
        self.trades_closed = 0
        self.current_price = 0.0
        self.last_status_time = 0
        self._load_trades()

    def _load_trades(self):
        trades_file = OUTPUT_DIR / "trades.json"
        if trades_file.exists():
            try:
                with open(trades_file) as f:
                    data = json.load(f)
                    self.current_equity = data.get("current_equity", self.capital)
                    for trade_data in data.get("trades", []):
                        trade = PaperTrade(**trade_data)
                        self.trades.append(trade)
                        if trade.status == "OPEN":
                            self.open_trades[trade.formula_id] = trade
                logger.info(f"Loaded {len(self.trades)} existing trades, equity: ${self.current_equity:.2f}")
            except Exception as e:
                logger.error(f"Failed to load trades: {e}")

    def _save_trades(self):
        trades_file = OUTPUT_DIR / "trades.json"
        data = {
            "updated": datetime.now().isoformat(),
            "capital": self.capital,
            "current_equity": self.current_equity,
            "signals_received": self.signals_received,
            "trades_executed": self.trades_executed,
            "trades_closed": self.trades_closed,
            "open_positions": len(self.open_trades),
            "trades": [asdict(t) for t in self.trades]
        }
        with open(trades_file, 'w') as f:
            json.dump(data, f, indent=2)

    def on_price_update(self, price: float, timestamp: float = None):
        ts = timestamp or time.time()
        self.current_price = price

        features = self.features.update(price, ts)

        if not self.features.is_ready():
            return

        # Check exits
        self._check_exits(price, ts)

        # Evaluate formulas
        for formula_id, formula in FORMULAS.items():
            if formula_id in self.open_trades:
                continue
            try:
                if formula["check"](features):
                    self._open_trade(formula_id, formula, price, ts)
            except Exception as e:
                pass  # Formula condition not met

        # Log status every 3 minutes
        if ts - self.last_status_time > 180:
            self._log_status(features)
            self.last_status_time = ts

    def _check_exits(self, price: float, ts: float):
        to_close = []
        for formula_id, trade in self.open_trades.items():
            if ts >= trade.target_exit_time:
                to_close.append((formula_id, "TIME_EXIT"))
                continue

            if trade.direction == "LONG":
                pnl_pct = (price / trade.entry_price - 1) * 100
            else:
                pnl_pct = (1 - price / trade.entry_price) * 100

            if pnl_pct < -2.0:
                to_close.append((formula_id, "STOP_LOSS"))
            elif pnl_pct > 25.0:
                to_close.append((formula_id, "TAKE_PROFIT"))

        for formula_id, reason in to_close:
            self._close_trade(formula_id, price, ts, reason)

    def _open_trade(self, formula_id: str, formula: dict, price: float, ts: float):
        # Position sizing with Kelly criterion
        confidence = formula["confidence"]
        # Simplified Kelly: f* = edge / odds, constrained
        kelly_fraction = max(0.05, min(0.25, (confidence - 0.5) * 2))
        position_size = self.current_equity * kelly_fraction
        leverage = 10.0  # Conservative for paper trading

        trade = PaperTrade(
            trade_id=f"{formula_id}_{int(ts)}",
            formula_id=formula_id,
            formula_name=formula["name"],
            direction=formula["direction"].name,
            entry_time=ts,
            entry_price=price,
            hold_days=formula["hold_days"],
            target_exit_time=ts + formula["hold_days"] * 86400,
            confidence=confidence,
            position_size=position_size,
            leverage=leverage,
            status="OPEN"
        )

        self.trades.append(trade)
        self.open_trades[formula_id] = trade
        self.trades_executed += 1
        self.signals_received += 1

        logger.info(f"========== TRADE OPENED ===========")
        logger.info(f"Formula: {formula_id} - {formula['name']}")
        logger.info(f"Direction: {trade.direction}")
        logger.info(f"Entry: ${price:,.2f}")
        logger.info(f"Size: ${position_size:.2f} @ {leverage:.0f}x")
        logger.info(f"Hold: {trade.hold_days} days")
        logger.info(f"===================================")

        self._save_trades()

    def _close_trade(self, formula_id: str, price: float, ts: float, reason: str):
        if formula_id not in self.open_trades:
            return

        trade = self.open_trades[formula_id]

        if trade.direction == "LONG":
            pnl_pct = (price / trade.entry_price - 1) * 100
        else:
            pnl_pct = (1 - price / trade.entry_price) * 100

        pnl = trade.position_size * (pnl_pct / 100) * trade.leverage

        trade.exit_time = ts
        trade.exit_price = price
        trade.pnl_pct = pnl_pct
        trade.pnl = pnl
        trade.status = "CLOSED"

        self.current_equity += pnl
        self.trades_closed += 1
        del self.open_trades[formula_id]

        logger.info(f"========== TRADE CLOSED ===========")
        logger.info(f"Formula: {formula_id} - {trade.formula_name}")
        logger.info(f"Reason: {reason}")
        logger.info(f"Entry: ${trade.entry_price:,.2f} -> Exit: ${price:,.2f}")
        logger.info(f"P&L: {pnl_pct:+.2f}% (${pnl:+.2f})")
        logger.info(f"New Equity: ${self.current_equity:.2f}")
        logger.info(f"===================================")

        self._save_trades()

    def _log_status(self, f: FeatureSnapshot):
        logger.info("=" * 50)
        logger.info(f"BTC: ${f.price:,.0f} | 7d: {f.ret_7d:+.1f}% | 14d: {f.ret_14d:+.1f}%")
        logger.info(f"RSI: {f.rsi_14:.0f} | BB: {f.bb_position:.2f} | Vol: {f.volatility_20d:.0f}%")
        logger.info(f"Regime: {f.regime} | Anomaly: {f.anomaly_score:.1f}")
        logger.info(f"Equity: ${self.current_equity:.2f} | Open: {len(self.open_trades)}")
        if self.open_trades:
            for fid, t in self.open_trades.items():
                if t.direction == "LONG":
                    pnl = (f.price / t.entry_price - 1) * 100
                else:
                    pnl = (1 - f.price / t.entry_price) * 100
                logger.info(f"  {fid}: {t.direction} @ ${t.entry_price:,.0f} ({pnl:+.1f}%)")
        logger.info("=" * 50)


# ==============================================================================
# HISTORICAL DATA FETCHER
# ==============================================================================

async def fetch_historical_candles() -> List[float]:
    """Fetch 90 days of daily candles from Hyperliquid."""
    logger.info("Fetching historical daily candles...")

    url = "https://api.hyperliquid.xyz/info"

    # Request 90 days of daily candles
    end_time = int(time.time() * 1000)
    start_time = end_time - (90 * 24 * 60 * 60 * 1000)  # 90 days ago

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": "BTC",
            "interval": "1d",
            "startTime": start_time,
            "endTime": end_time
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=30)
        data = response.json()

    if not data:
        logger.error("No historical data received")
        return []

    # Extract close prices
    closes = []
    for candle in data:
        close = float(candle.get("c", 0))
        if close > 0:
            closes.append(close)

    logger.info(f"Fetched {len(closes)} daily candles")
    if closes:
        logger.info(f"Price range: ${min(closes):,.0f} - ${max(closes):,.0f}")
        logger.info(f"Latest close: ${closes[-1]:,.0f}")

    return closes


# ==============================================================================
# WEBSOCKET CONNECTION
# ==============================================================================

async def connect_websocket(trader: PaperTrader):
    uri = "wss://api.hyperliquid.xyz/ws"

    while True:
        try:
            async with websockets.connect(uri) as ws:
                subscribe_msg = {
                    "method": "subscribe",
                    "subscription": {"type": "trades", "coin": "BTC"}
                }
                await ws.send(json.dumps(subscribe_msg))
                logger.info("Connected to Hyperliquid WebSocket")

                async for message in ws:
                    try:
                        data = json.loads(message)
                        if data.get("channel") == "trades":
                            for trade in data.get("data", []):
                                price = float(trade.get("px", 0))
                                if price > 0:
                                    trader.on_price_update(price)
                    except Exception as e:
                        logger.error(f"Message error: {e}")

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            logger.info("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)


# ==============================================================================
# MAIN
# ==============================================================================

async def main():
    logger.info("=" * 60)
    logger.info("RENTECH PAPER TRADER v2 - HOSTINGER VPS")
    logger.info("=" * 60)

    trader = PaperTrader(capital=100.0)

    # Warm up with historical data
    historical = await fetch_historical_candles()
    if historical:
        trader.features.warm_up(historical)
    else:
        logger.warning("Starting without historical data - features will take time to warm up")

    logger.info(f"Feature engine ready: {trader.features.is_ready()}")
    logger.info(f"Starting with ${trader.current_equity:.2f} equity")
    logger.info("=" * 60)

    await connect_websocket(trader)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Paper trading stopped")
