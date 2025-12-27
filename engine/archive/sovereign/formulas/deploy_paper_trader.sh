#!/bin/bash
# ==============================================================================
# RENTECH PAPER TRADER DEPLOYMENT FOR HOSTINGER VPS
# ==============================================================================
# Run: bash deploy_paper_trader.sh
# VPS: 31.97.211.217
# ==============================================================================

set -e

WORK_DIR="/root/sovereign"
VENV_DIR="$WORK_DIR/venv"

echo "=============================================="
echo "RENTECH PAPER TRADER DEPLOYMENT"
echo "=============================================="

# Create directories
mkdir -p $WORK_DIR/formulas
mkdir -p $WORK_DIR/strategy
mkdir -p $WORK_DIR/data/paper_trades

cd $WORK_DIR

# Ensure venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating Python virtual environment..."
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

echo "[2/4] Installing dependencies..."
pip install --upgrade pip
pip install websockets numpy requests

echo "[3/4] Creating paper trader runner..."
cat > $WORK_DIR/run_paper_trader.py << 'PYFILE'
#!/usr/bin/env python3
"""
RenTech Paper Trader - Hostinger VPS Version
=============================================
Connects to Hyperliquid WebSocket for live BTC prices.
Evaluates all 9 RenTech formulas in real-time.
"""

import asyncio
import json
import time
import logging
from datetime import datetime
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


# ==============================================================================
# FEATURE ENGINE (Simplified for VPS)
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
    """Real-time feature calculation."""

    def __init__(self):
        self.prices = deque(maxlen=200)
        self.last_features = None

    def update(self, price: float, timestamp: float = None) -> FeatureSnapshot:
        ts = timestamp or time.time()
        self.prices.append(price)

        prices = list(self.prices)
        n = len(prices)

        snapshot = FeatureSnapshot(timestamp=ts, price=price)

        if n < 2:
            return snapshot

        # Returns
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

        # Volatility
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

        # Simple regime detection
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

        # Anomaly score (simplified)
        snapshot.anomaly_score = abs(snapshot.ret_7d / 10) + abs(snapshot.bb_position)

        self.last_features = snapshot
        return snapshot

    def is_ready(self) -> bool:
        return len(self.prices) >= 30


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


FORMULAS = {
    "RENTECH_001": {
        "name": "EXTREME_ANOMALY_LONG",
        "direction": SignalDirection.LONG,
        "hold_days": 30,
        "check": lambda f: f.anomaly_score > 4 and f.ret_7d < -15,
    },
    "RENTECH_002": {
        "name": "VOLUME_MOMENTUM_CONFLUENCE",
        "direction": SignalDirection.LONG,
        "hold_days": 30,
        "check": lambda f: f.ret_7d > 5 and f.ret_3d > 2,
    },
    "RENTECH_003": {
        "name": "EXTREME_ANOMALY_LONG_7D",
        "direction": SignalDirection.LONG,
        "hold_days": 7,
        "check": lambda f: f.anomaly_score > 4 and f.ret_7d < -15,
    },
    "RENTECH_004": {
        "name": "CORRELATION_BREAK_BULL",
        "direction": SignalDirection.LONG,
        "hold_days": 7,
        "check": lambda f: f.price_vs_ma30 < -15 and f.ret_3d > 0,
    },
    "RENTECH_005": {
        "name": "BOLLINGER_BOUNCE",
        "direction": SignalDirection.LONG,
        "hold_days": 30,
        "check": lambda f: f.bb_position < -0.9 and f.volatility_20d < 80 and f.ret_3d > 0,
    },
    "RENTECH_006": {
        "name": "WHALE_ACCUMULATION",
        "direction": SignalDirection.LONG,
        "hold_days": 30,
        "check": lambda f: f.ret_7d < -10 and f.ret_3d > 0,
    },
    "RENTECH_007": {
        "name": "RSI_DIVERGENCE",
        "direction": SignalDirection.LONG,
        "hold_days": 30,
        "check": lambda f: f.ret_14d < -15 and f.rsi_14 > 35 and f.ret_3d > 0,
    },
    "RENTECH_009": {
        "name": "EUPHORIA_EXIT_SHORT",
        "direction": SignalDirection.SHORT,
        "hold_days": 30,
        "check": lambda f: f.regime == "EUPHORIA" and f.rsi_14 > 80 and f.price_vs_ma30 > 30,
    },
}


def evaluate_signals(features: FeatureSnapshot, price: float) -> List[Signal]:
    signals = []
    for fid, formula in FORMULAS.items():
        try:
            if formula["check"](features):
                signals.append(Signal(
                    formula_id=fid,
                    formula_name=formula["name"],
                    direction=formula["direction"],
                    confidence=1.0,
                    hold_days=formula["hold_days"],
                    entry_price=price
                ))
        except:
            pass
    return signals


# ==============================================================================
# PAPER TRADER
# ==============================================================================

@dataclass
class PaperTrade:
    trade_id: str
    formula_id: str
    direction: str
    entry_time: float
    entry_price: float
    hold_days: int
    target_exit_time: float
    position_size: float
    leverage: float
    status: str = "OPEN"
    exit_time: float = None
    exit_price: float = None
    pnl: float = None
    pnl_pct: float = None


class PaperTrader:
    def __init__(self, capital: float = 100.0):
        self.capital = capital
        self.equity = capital
        self.features = FeatureEngine()
        self.trades: List[PaperTrade] = []
        self.open_trades: Dict[str, PaperTrade] = {}
        self.current_price = 0.0
        self.tick_count = 0

        self.output_dir = Path("/root/sovereign/data/paper_trades")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._load_trades()
        logger.info(f"Paper trader initialized with ${capital:.2f}")

    def _load_trades(self):
        trades_file = self.output_dir / "trades.json"
        if trades_file.exists():
            try:
                with open(trades_file) as f:
                    data = json.load(f)
                self.equity = data.get("equity", self.capital)
                for t in data.get("trades", []):
                    trade = PaperTrade(**t)
                    self.trades.append(trade)
                    if trade.status == "OPEN":
                        self.open_trades[trade.formula_id] = trade
                logger.info(f"Loaded {len(self.trades)} trades, equity: ${self.equity:.2f}")
            except Exception as e:
                logger.error(f"Failed to load trades: {e}")

    def _save_trades(self):
        trades_file = self.output_dir / "trades.json"
        data = {
            "updated": datetime.now().isoformat(),
            "capital": self.capital,
            "equity": self.equity,
            "open_positions": len(self.open_trades),
            "total_trades": len(self.trades),
            "trades": [asdict(t) for t in self.trades]
        }
        with open(trades_file, 'w') as f:
            json.dump(data, f, indent=2)

    def on_price(self, price: float):
        self.current_price = price
        self.tick_count += 1
        ts = time.time()

        features = self.features.update(price, ts)

        if not self.features.is_ready():
            if self.tick_count % 100 == 0:
                logger.info(f"Warming up: {len(self.features.prices)}/30 prices")
            return

        # Check exits
        self._check_exits(price, ts)

        # Evaluate signals
        signals = evaluate_signals(features, price)
        for signal in signals:
            self._process_signal(signal, price, ts)

        # Log status periodically
        if self.tick_count % 1000 == 0:
            self._log_status(features)

    def _check_exits(self, price: float, ts: float):
        to_close = []
        for fid, trade in self.open_trades.items():
            if ts >= trade.target_exit_time:
                to_close.append((fid, "TIME_EXIT"))
                continue

            if trade.direction == "LONG":
                pnl_pct = (price / trade.entry_price - 1) * 100
            else:
                pnl_pct = (1 - price / trade.entry_price) * 100

            if pnl_pct < -2.0:
                to_close.append((fid, "STOP_LOSS"))
            elif pnl_pct > 25.0:
                to_close.append((fid, "TAKE_PROFIT"))

        for fid, reason in to_close:
            self._close_trade(fid, price, ts, reason)

    def _process_signal(self, signal: Signal, price: float, ts: float):
        if signal.formula_id in self.open_trades:
            return

        # Position size: 1% of equity
        size = self.equity * 0.01
        leverage = 10.0

        trade = PaperTrade(
            trade_id=f"{signal.formula_id}_{int(ts)}",
            formula_id=signal.formula_id,
            direction=signal.direction.name,
            entry_time=ts,
            entry_price=price,
            hold_days=signal.hold_days,
            target_exit_time=ts + signal.hold_days * 86400,
            position_size=size,
            leverage=leverage
        )

        self.trades.append(trade)
        self.open_trades[signal.formula_id] = trade

        logger.info(f"TRADE OPENED: {signal.formula_id} {signal.direction.name} @ ${price:,.0f}")
        logger.info(f"  Size: ${size:.2f} @ {leverage:.0f}x, Hold: {signal.hold_days}d")

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

        self.equity += pnl
        del self.open_trades[formula_id]

        logger.info(f"TRADE CLOSED: {formula_id} ({reason})")
        logger.info(f"  Entry: ${trade.entry_price:,.0f} -> Exit: ${price:,.0f}")
        logger.info(f"  P&L: {pnl_pct:+.2f}% (${pnl:+.2f})")
        logger.info(f"  Equity: ${self.equity:.2f}")

        self._save_trades()

    def _log_status(self, features: FeatureSnapshot):
        logger.info("=" * 50)
        logger.info(f"BTC: ${self.current_price:,.0f} | 7d: {features.ret_7d:+.1f}%")
        logger.info(f"RSI: {features.rsi_14:.0f} | Regime: {features.regime}")
        logger.info(f"Equity: ${self.equity:.2f} | Open: {len(self.open_trades)}")
        logger.info("=" * 50)


# ==============================================================================
# WEBSOCKET CONNECTION
# ==============================================================================

async def connect_hyperliquid(trader: PaperTrader):
    uri = "wss://api.hyperliquid.xyz/ws"

    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
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
                                    trader.on_price(price)
                    except Exception as e:
                        logger.error(f"Message error: {e}")

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            logger.info("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)


async def main():
    logger.info("=" * 60)
    logger.info("RENTECH PAPER TRADER - HOSTINGER VPS")
    logger.info("=" * 60)

    trader = PaperTrader(capital=100.0)
    await connect_hyperliquid(trader)


if __name__ == "__main__":
    asyncio.run(main())
PYFILE

chmod +x $WORK_DIR/run_paper_trader.py

echo "[4/4] Creating systemd service..."
cat > /etc/systemd/system/rentech-paper.service << 'SVCFILE'
[Unit]
Description=RenTech Paper Trader
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/sovereign
Environment=PATH=/root/sovereign/venv/bin:/usr/bin
ExecStart=/root/sovereign/venv/bin/python /root/sovereign/run_paper_trader.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SVCFILE

systemctl daemon-reload

echo ""
echo "=============================================="
echo "DEPLOYMENT COMPLETE!"
echo "=============================================="
echo ""
echo "TO RUN:"
echo "  cd /root/sovereign"
echo "  source venv/bin/activate"
echo "  python run_paper_trader.py"
echo ""
echo "OR WITH SYSTEMD:"
echo "  systemctl enable rentech-paper"
echo "  systemctl start rentech-paper"
echo "  journalctl -u rentech-paper -f"
echo ""
echo "CHECK STATUS:"
echo "  cat /root/sovereign/data/paper_trades/trades.json"
echo ""
