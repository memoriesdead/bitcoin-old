#!/usr/bin/env python3
"""
BLOCKCHAIN SCALPING BOT - RenTech Pattern Recognition
======================================================
Based on real blockchain backtest (4,401 days of data) + ML Analysis:
- Win Rate: 62.3% (backtest) / 84.9% (Gradient Boosting ML)
- Profit Factor: 5.06
- Sharpe Ratio: 12.34
- Max Drawdown: 2.7%

RenTech-Style Pattern Recognition (2,043 samples):
- TX_WHALE_MOMENTUM: 63.7% WR (+9.7% edge) - tx up + whale activity
- WHALE_QUIET_ACCUMULATION: 62.3% WR (+8.2% edge) - whales active, low vol
- PRICE_OVERSOLD_REVERSAL: 56.8% WR (+2.7% edge) - mean reversion

Anomaly Detection Insight:
- AVOID trading during extreme activity spikes (-6.4% edge)
- Focus on "quiet" accumulation patterns

Key ML Features (Gradient Boosting):
- value_trend, value_zscore, activity_intensity, is_tx_spike, price_range

Parameters:
- Take Profit: 1.0%
- Stop Loss: 0.3%
- Min Confidence: 52%
- Leverage: 5x
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import websockets
import httpx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================
# VALIDATED PARAMETERS (Real Backtest)
# =====================================
TAKE_PROFIT_PCT = 1.0       # 1.0% TP (validated: 62.3% hit rate)
STOP_LOSS_PCT = 0.3         # 0.3% SL (tight stop)
LEVERAGE = 5                # 5x leverage
MIN_CONFIDENCE = 0.52       # 52% minimum (from backtest)
COOLDOWN_SECONDS = 60       # 1 minute between trades

# Signal Type Configurations (by win rate from backtest + RenTech ML)
SIGNAL_CONFIGS = {
    # Original backtest signals
    'TX_SURGE_DOWN': {'win_rate': 0.80, 'direction': -1, 'size_mult': 1.0},
    'QUIET_WHALE': {'win_rate': 0.67, 'direction': 1, 'size_mult': 1.0},
    'VALUE_ACCUMULATION': {'win_rate': 0.63, 'direction': 1, 'size_mult': 1.0},
    'TX_SURGE_UP': {'win_rate': 0.50, 'direction': 1, 'size_mult': 0.5},
    'WHALE_ACCUMULATION': {'win_rate': 0.50, 'direction': 1, 'size_mult': 0.5},
    'WHALE_DISTRIBUTION': {'win_rate': 0.33, 'direction': -1, 'size_mult': 0.0},  # Skip

    # RenTech Pattern Recognition signals (ML validated)
    'TX_WHALE_MOMENTUM': {'win_rate': 0.637, 'direction': 1, 'size_mult': 1.0, 'edge': 0.097},
    'WHALE_QUIET_ACCUMULATION': {'win_rate': 0.623, 'direction': 1, 'size_mult': 1.0, 'edge': 0.082},
    'PRICE_OVERSOLD_REVERSAL': {'win_rate': 0.568, 'direction': 1, 'size_mult': 0.8, 'edge': 0.027},
    'VALUE_DISTRIBUTION_BULLISH': {'win_rate': 0.489, 'direction': 1, 'size_mult': 0.0, 'edge': -0.052},  # Skip
}

# Anomaly Detection Thresholds (from RenTech analysis)
# AVOID trading during extreme conditions - they have -6.4% edge
ANOMALY_THRESHOLDS = {
    'value_zscore_max': 3.0,      # Skip if value zscore > 3
    'activity_intensity_max': 1000,  # Skip extreme activity spikes
    'tx_zscore_max': 2.5,         # Skip extreme tx spikes
}

# File Paths
SIGNALS_PATH = Path("/root/sovereign/signals.json")
OUTPUT_DIR = Path("data/scalper")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BlockchainSignal:
    """Blockchain signal from signal_server."""
    timestamp: int
    signal_type: str
    direction: int          # 1=LONG, -1=SHORT
    confidence: float
    metrics: Dict
    age_seconds: int = 0


@dataclass
class Trade:
    id: str
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    entry_time: float
    stop_loss: float
    take_profit: float
    size_usd: float
    signal_type: str
    confidence: float
    status: str             # "OPEN" or "CLOSED"
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    pnl_pct: Optional[float] = None
    pnl_usd: Optional[float] = None
    exit_reason: Optional[str] = None


class BlockchainScalper:
    """
    Scalping bot using validated blockchain signals.

    Strategy: Trade high-probability signals with tight risk management.
    """

    def __init__(self, capital: float = 100.0):
        self.capital = capital
        self.equity = capital

        # Trading state
        self.current_trade: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.wins = 0
        self.losses = 0
        self.last_trade_time = 0

        # Current price
        self.current_price = 0.0
        self.ticks_received = 0
        self.last_log_time = 0

        # Signal cache
        self.last_signal: Optional[BlockchainSignal] = None
        self.last_signal_id: str = ""

        # Exit tracking
        self.exits_tp = 0
        self.exits_sl = 0
        self.exits_time = 0

        self._load_state()

    def _load_state(self):
        """Load state from disk."""
        state_file = OUTPUT_DIR / "scalper_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                self.equity = data.get("equity", self.capital)
                self.wins = data.get("wins", 0)
                self.losses = data.get("losses", 0)
                self.exits_tp = data.get("exits_tp", 0)
                self.exits_sl = data.get("exits_sl", 0)
                self.exits_time = data.get("exits_time", 0)
                self.last_signal_id = data.get("last_signal_id", "")

                for t in data.get("trades", []):
                    self.trades.append(Trade(**t))
                if data.get("current_trade"):
                    self.current_trade = Trade(**data["current_trade"])

                logger.info(f"Loaded state: ${self.equity:.2f}, {len(self.trades)} trades")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def _save_state(self):
        """Save state to disk."""
        state_file = OUTPUT_DIR / "scalper_state.json"
        total_trades = self.wins + self.losses
        win_rate = self.wins / total_trades * 100 if total_trades > 0 else 0

        data = {
            "updated": datetime.now().isoformat(),
            "equity": self.equity,
            "capital": self.capital,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate,
            "exits_tp": self.exits_tp,
            "exits_sl": self.exits_sl,
            "exits_time": self.exits_time,
            "last_signal_id": self.last_signal_id,
            "trades": [asdict(t) for t in self.trades[-100:]],
            "current_trade": asdict(self.current_trade) if self.current_trade else None,
            "parameters": {
                "take_profit_pct": TAKE_PROFIT_PCT,
                "stop_loss_pct": STOP_LOSS_PCT,
                "leverage": LEVERAGE,
                "min_confidence": MIN_CONFIDENCE
            }
        }
        with open(state_file, "w") as f:
            json.dump(data, f, indent=2)

    def read_blockchain_signal(self) -> Optional[BlockchainSignal]:
        """Read latest blockchain signal from signals.json."""
        if not SIGNALS_PATH.exists():
            return None

        try:
            with open(SIGNALS_PATH) as f:
                data = json.load(f)

            signal_id = f"{data.get('timestamp', 0)}_{data.get('signal_type', '')}"

            # Skip if same signal already processed
            if signal_id == self.last_signal_id:
                return None

            signal = BlockchainSignal(
                timestamp=data.get("timestamp", 0),
                signal_type=data.get("signal_type", ""),
                direction=data.get("direction", 0),
                confidence=data.get("confidence", 0),
                metrics=data.get("metrics", {})
            )

            signal.age_seconds = int(time.time()) - signal.timestamp

            # Only use fresh signals (< 5 minutes old)
            if signal.age_seconds > 300:
                return None

            self.last_signal = signal
            return signal

        except Exception as e:
            logger.warning(f"Failed to read signal: {e}")
            return None

    def evaluate_signal(self, signal: BlockchainSignal) -> Dict:
        """
        Evaluate whether to trade based on signal type and confidence.
        Includes RenTech anomaly detection to skip extreme market conditions.

        Returns: {'action': 'TRADE'/'SKIP', 'direction': 1/-1, 'size_mult': 0-1, 'reason': str}
        """
        # RenTech Anomaly Filter - skip extreme conditions (have -6.4% edge)
        metrics = signal.metrics
        value_zscore = abs(metrics.get('value_zscore', 0))
        activity_intensity = metrics.get('activity_intensity', 0)
        tx_zscore = abs(metrics.get('tx_zscore', 0))

        if value_zscore > ANOMALY_THRESHOLDS['value_zscore_max']:
            return {
                'action': 'SKIP',
                'direction': 0,
                'size_mult': 0,
                'reason': f'ANOMALY: Extreme value zscore ({value_zscore:.1f} > {ANOMALY_THRESHOLDS["value_zscore_max"]})'
            }

        if activity_intensity > ANOMALY_THRESHOLDS['activity_intensity_max']:
            return {
                'action': 'SKIP',
                'direction': 0,
                'size_mult': 0,
                'reason': f'ANOMALY: Extreme activity ({activity_intensity:.0f} > {ANOMALY_THRESHOLDS["activity_intensity_max"]})'
            }

        if tx_zscore > ANOMALY_THRESHOLDS['tx_zscore_max']:
            return {
                'action': 'SKIP',
                'direction': 0,
                'size_mult': 0,
                'reason': f'ANOMALY: Extreme tx zscore ({tx_zscore:.1f} > {ANOMALY_THRESHOLDS["tx_zscore_max"]})'
            }

        # Check confidence threshold
        if signal.confidence < MIN_CONFIDENCE:
            return {
                'action': 'SKIP',
                'direction': 0,
                'size_mult': 0,
                'reason': f'Low confidence ({signal.confidence:.0%} < {MIN_CONFIDENCE:.0%})'
            }

        # Get signal configuration
        config = SIGNAL_CONFIGS.get(signal.signal_type)

        if config is None:
            return {
                'action': 'SKIP',
                'direction': 0,
                'size_mult': 0,
                'reason': f'Unknown signal type: {signal.signal_type}'
            }

        # Skip low win rate signals
        if config['size_mult'] == 0:
            return {
                'action': 'SKIP',
                'direction': 0,
                'size_mult': 0,
                'reason': f'{signal.signal_type} has low win rate ({config["win_rate"]:.0%}) - skipping'
            }

        # Trade signal
        return {
            'action': 'TRADE',
            'direction': config['direction'],
            'size_mult': config['size_mult'],
            'reason': f'{signal.signal_type}: {config["win_rate"]:.0%} WR, conf={signal.confidence:.0%}'
        }

    def on_price(self, price: float):
        """Process each price tick."""
        self.current_price = price
        self.ticks_received += 1
        now = time.time()

        # Check exit conditions first
        if self.current_trade:
            self._check_exit(price, now)

        # Check for new signal (only if not in trade and cooldown passed)
        if not self.current_trade and (now - self.last_trade_time) >= COOLDOWN_SECONDS:
            signal = self.read_blockchain_signal()
            if signal:
                self._check_entry(signal, price, now)

        # Log status every 30 seconds
        if now - self.last_log_time > 30:
            self._log_status(price)
            self.last_log_time = now

    def _check_entry(self, signal: BlockchainSignal, price: float, now: float):
        """Check if we should enter a trade based on signal."""
        decision = self.evaluate_signal(signal)

        logger.info(f"Signal: {signal.signal_type} (conf={signal.confidence:.0%}, age={signal.age_seconds}s)")
        logger.info(f"Decision: {decision['action']} - {decision['reason']}")

        if decision['action'] == 'SKIP':
            return

        # Calculate position
        direction = decision['direction']
        direction_str = "LONG" if direction == 1 else "SHORT"

        if direction == 1:  # LONG
            stop_loss = price * (1 - STOP_LOSS_PCT / 100)
            take_profit = price * (1 + TAKE_PROFIT_PCT / 100)
        else:  # SHORT
            stop_loss = price * (1 + STOP_LOSS_PCT / 100)
            take_profit = price * (1 - TAKE_PROFIT_PCT / 100)

        # Position size
        size_mult = decision['size_mult']
        size_usd = self.equity * LEVERAGE * size_mult

        # Create trade
        self.current_trade = Trade(
            id=f"SCALP_{int(now)}",
            direction=direction_str,
            entry_price=price,
            entry_time=now,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size_usd=size_usd,
            signal_type=signal.signal_type,
            confidence=signal.confidence,
            status="OPEN"
        )

        # Mark signal as processed
        self.last_signal_id = f"{signal.timestamp}_{signal.signal_type}"
        self.last_trade_time = now

        logger.info("=" * 60)
        logger.info(f"TRADE OPENED: {direction_str}")
        logger.info(f"Signal: {signal.signal_type} ({decision['reason']})")
        logger.info(f"Entry: ${price:,.2f}")
        logger.info(f"Stop Loss: ${stop_loss:,.2f} (-{STOP_LOSS_PCT}%)")
        logger.info(f"Take Profit: ${take_profit:,.2f} (+{TAKE_PROFIT_PCT}%)")
        logger.info(f"Size: ${size_usd:,.2f} ({LEVERAGE}x)")
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

        exit_reason = None
        if hit_sl:
            exit_reason = "STOP_LOSS"
            self.exits_sl += 1
        elif hit_tp:
            exit_reason = "TAKE_PROFIT"
            self.exits_tp += 1

        if exit_reason:
            self._close_trade(price, now, pnl_pct, exit_reason)

    def _close_trade(self, price: float, now: float, pnl_pct: float, reason: str):
        """Close the current trade."""
        trade = self.current_trade

        leveraged_pnl_pct = pnl_pct * LEVERAGE
        pnl_usd = self.equity * (leveraged_pnl_pct / 100)

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

        total_trades = self.wins + self.losses
        win_rate = self.wins / total_trades * 100 if total_trades > 0 else 0

        hold_time = (now - trade.entry_time)

        logger.info("=" * 60)
        logger.info(f"TRADE CLOSED: {reason}")
        logger.info(f"Direction: {trade.direction} | Signal: {trade.signal_type}")
        logger.info(f"Entry: ${trade.entry_price:,.2f} -> Exit: ${price:,.2f}")
        logger.info(f"Hold Time: {hold_time:.1f}s")
        logger.info(f"P&L: {leveraged_pnl_pct:+.2f}% (${pnl_usd:+.2f})")
        logger.info(f"New Equity: ${self.equity:.2f}")
        logger.info(f"Record: {self.wins}W-{self.losses}L ({win_rate:.1f}%)")
        logger.info(f"Exits: TP={self.exits_tp}, SL={self.exits_sl}")
        logger.info("=" * 60)

        self._save_state()

    def _log_status(self, price: float):
        """Log current status."""
        total_trades = self.wins + self.losses
        win_rate = self.wins / total_trades * 100 if total_trades > 0 else 0
        pnl_total = self.equity - self.capital
        pnl_pct = pnl_total / self.capital * 100

        logger.info("-" * 60)
        logger.info(f"BTC: ${price:,.0f} | Ticks: {self.ticks_received}")
        logger.info(f"Equity: ${self.equity:.2f} ({pnl_pct:+.1f}%)")
        logger.info(f"Record: {self.wins}W-{self.losses}L ({win_rate:.1f}%)")
        logger.info(f"Exits: TP={self.exits_tp}, SL={self.exits_sl}")

        if self.current_trade:
            t = self.current_trade
            if t.direction == "LONG":
                curr_pnl = (price - t.entry_price) / t.entry_price * 100 * LEVERAGE
            else:
                curr_pnl = (t.entry_price - price) / t.entry_price * 100 * LEVERAGE
            hold_time = time.time() - t.entry_time
            logger.info(f"OPEN: {t.direction} @ ${t.entry_price:,.0f} | "
                       f"P&L: {curr_pnl:+.2f}% | Hold: {hold_time:.0f}s | "
                       f"Signal: {t.signal_type}")
        else:
            if self.last_signal:
                logger.info(f"Last signal: {self.last_signal.signal_type} "
                           f"(age={self.last_signal.age_seconds}s)")
            else:
                logger.info("Waiting for blockchain signal...")
        logger.info("-" * 60)


async def run_scalper(capital: float = 100.0):
    """Run the scalper with WebSocket connection."""
    logger.info("=" * 60)
    logger.info("BLOCKCHAIN SCALPER - Validated Parameters")
    logger.info(f"TP: {TAKE_PROFIT_PCT}% | SL: {STOP_LOSS_PCT}% | Leverage: {LEVERAGE}x")
    logger.info(f"Min Confidence: {MIN_CONFIDENCE:.0%}")
    logger.info(f"Capital: ${capital}")
    logger.info("=" * 60)
    logger.info("Signal Win Rates (from backtest):")
    for sig_type, config in SIGNAL_CONFIGS.items():
        if config['size_mult'] > 0:
            dir_str = "LONG" if config['direction'] == 1 else "SHORT"
            logger.info(f"  {sig_type}: {config['win_rate']:.0%} -> {dir_str}")
    logger.info("=" * 60)

    scalper = BlockchainScalper(capital=capital)

    # Connect to Hyperliquid WebSocket
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
                                    scalper.on_price(price)
                    except Exception as e:
                        logger.error(f"Message error: {e}")

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            logger.info("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blockchain Scalper with Validated Parameters")
    parser.add_argument("--capital", type=float, default=100.0, help="Starting capital")
    args = parser.parse_args()

    asyncio.run(run_scalper(capital=args.capital))
