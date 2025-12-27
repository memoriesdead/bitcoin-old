#!/usr/bin/env python3
"""
UNIFIED HOSTINGER TRADER
=========================

Combines all data sources from Hostinger VPS into one trading system:
- Whale detection (mempool.space)
- Exchange flow signals (when address matching works)
- Pattern recognition (AI ensemble)
- Regime detection (volatility + trend)

Paper trading first, then live on Kraken with 35x leverage.
"""

import os
import sys
import json
import time
import logging
import asyncio
import urllib.request
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from collections import deque

# =============================================================================
# CONFIGURATION
# =============================================================================

# Trading parameters (validated from plan)
LEVERAGE = 35.0
TAKE_PROFIT_PCT = 0.005    # 0.5% = ~$17.50 per win at $100
STOP_LOSS_PCT = 0.0015     # 0.15% = ~$5.25 per loss at $100
MAX_POSITIONS = 4
POSITION_SIZE_PCT = 0.25   # 25% capital per position
MIN_CONFIDENCE = 0.4
COOLDOWN_SECONDS = 60

# Signal weights
SIGNAL_WEIGHTS = {
    'whale': 0.20,     # Whale activity (volatility predictor)
    'flow': 0.35,      # Exchange flows (directional)
    'pattern': 0.30,   # Pattern recognition
    'regime': 0.15,    # Market regime
}

# Paths (Hostinger VPS)
DATA_DIR = Path("/root/sovereign/data")
FLOW_STATE_FILE = Path("/root/flow_state.json")
LOG_FILE = Path("/root/unified.log")
TRADES_FILE = Path("/root/unified_trades.json")

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE) if LOG_FILE.parent.exists() else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Signal:
    name: str
    direction: int       # +1 LONG, -1 SHORT, 0 NEUTRAL
    confidence: float    # 0.0 to 1.0
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Position:
    id: str
    direction: int
    size_usd: float
    entry_price: float
    entry_time: float
    stop_loss: float
    take_profit: float
    signal_reasons: List[str]


@dataclass
class Trade:
    id: str
    direction: int
    size_usd: float
    entry_price: float
    entry_time: float
    exit_price: float
    exit_time: float
    exit_reason: str
    pnl_usd: float
    pnl_pct: float


# =============================================================================
# SIGNAL SOURCES
# =============================================================================

class WhaleSignal:
    """
    Whale transaction detection.
    Source: flow_hft.py state file or mempool.space direct.
    """

    def __init__(self):
        self.whale_history = deque(maxlen=100)
        self.last_update = 0

    def update(self, flow_state: dict) -> None:
        """Update from flow_hft.py state."""
        whale_count = flow_state.get('whale_count', 0)
        whale_btc = flow_state.get('whale_btc', 0)
        mega_whales = flow_state.get('mega_whales', 0)

        self.whale_history.append({
            'time': time.time(),
            'count': whale_count,
            'btc': whale_btc,
            'mega': mega_whales
        })
        self.last_update = time.time()

    def get_signal(self) -> Signal:
        """Generate signal from whale activity."""
        if not self.whale_history or time.time() - self.last_update > 120:
            return Signal('whale', 0, 0.0, "No whale data")

        recent = list(self.whale_history)[-10:]
        if len(recent) < 3:
            return Signal('whale', 0, 0.0, "Insufficient whale history")

        # Calculate whale activity trend
        total_whales = sum(w['count'] for w in recent)
        total_btc = sum(w['btc'] for w in recent)
        mega_count = sum(w['mega'] for w in recent)

        # Whales = volatility signal, not directional
        # High whale activity = market about to move
        if mega_count >= 2:
            # Multiple mega whales = strong volatility signal
            # Direction based on price momentum (handled by regime)
            return Signal('whale', 0, 0.6, f"MEGA WHALE ALERT: {mega_count} mega, {total_btc:.0f} BTC")
        elif total_whales >= 10:
            return Signal('whale', 0, 0.4, f"High whale activity: {total_whales} whales, {total_btc:.0f} BTC")

        return Signal('whale', 0, 0.2, f"Normal whale activity: {total_whales}")


class FlowSignal:
    """
    Exchange flow imbalance signal.
    Source: flow_hft.py tracking inflows/outflows.
    """

    def __init__(self):
        self.flow_history = deque(maxlen=50)
        self.last_update = 0

    def update(self, flow_state: dict) -> None:
        """Update from flow_hft.py state."""
        inflow = flow_state.get('inflow_btc', 0)
        outflow = flow_state.get('outflow_btc', 0)
        fis = flow_state.get('fis', 0)  # Flow Imbalance Signal

        self.flow_history.append({
            'time': time.time(),
            'inflow': inflow,
            'outflow': outflow,
            'fis': fis
        })
        self.last_update = time.time()

    def get_signal(self) -> Signal:
        """Generate signal from exchange flows."""
        if not self.flow_history or time.time() - self.last_update > 120:
            return Signal('flow', 0, 0.0, "No flow data")

        recent = list(self.flow_history)[-10:]
        if len(recent) < 3:
            return Signal('flow', 0, 0.0, "Insufficient flow history")

        # Average FIS over recent windows
        avg_fis = sum(f['fis'] for f in recent) / len(recent)
        total_inflow = sum(f['inflow'] for f in recent)
        total_outflow = sum(f['outflow'] for f in recent)

        # FIS > 0 = net outflow = bullish (coins leaving exchanges)
        # FIS < 0 = net inflow = bearish (coins going to exchanges to sell)
        if avg_fis > 0.3 and total_outflow > total_inflow * 1.5:
            return Signal('flow', 1, min(0.8, avg_fis),
                         f"STRONG OUTFLOW: {total_outflow:.1f} BTC leaving (FIS={avg_fis:.2f})")
        elif avg_fis < -0.3 and total_inflow > total_outflow * 1.5:
            return Signal('flow', -1, min(0.8, abs(avg_fis)),
                         f"STRONG INFLOW: {total_inflow:.1f} BTC entering (FIS={avg_fis:.2f})")
        elif avg_fis > 0.1:
            return Signal('flow', 1, 0.4, f"Mild outflow (FIS={avg_fis:.2f})")
        elif avg_fis < -0.1:
            return Signal('flow', -1, 0.4, f"Mild inflow (FIS={avg_fis:.2f})")

        return Signal('flow', 0, 0.2, f"Neutral flow (FIS={avg_fis:.2f})")


class PatternSignal:
    """
    AI pattern recognition signal.
    Source: AI trading engine HMM/ensemble.
    """

    def __init__(self):
        self.pattern_file = DATA_DIR / "ai_signals.json"
        self.last_signal = None
        self.last_update = 0

    def update(self) -> None:
        """Load latest AI signal."""
        if not self.pattern_file.exists():
            return

        try:
            with open(self.pattern_file) as f:
                data = json.load(f)
            self.last_signal = data
            self.last_update = time.time()
        except Exception:
            pass

    def get_signal(self) -> Signal:
        """Generate signal from AI patterns."""
        if not self.last_signal or time.time() - self.last_update > 300:
            return Signal('pattern', 0, 0.0, "No AI signal")

        direction = self.last_signal.get('direction', 0)
        confidence = self.last_signal.get('confidence', 0)
        pattern = self.last_signal.get('pattern', 'unknown')

        if direction != 0 and confidence > 0.3:
            return Signal('pattern', direction, confidence,
                         f"AI: {pattern} ({confidence:.0%})")

        return Signal('pattern', 0, 0.2, "AI: No clear pattern")


class RegimeSignal:
    """
    Market regime detection.
    Based on volatility and trend indicators.
    """

    def __init__(self):
        self.price_history = deque(maxlen=200)
        self.last_update = 0

    def update(self, price: float) -> None:
        """Add price to history."""
        self.price_history.append({
            'time': time.time(),
            'price': price
        })
        self.last_update = time.time()

    def get_signal(self) -> Signal:
        """Generate regime-based signal."""
        if len(self.price_history) < 20:
            return Signal('regime', 0, 0.0, "Insufficient price history")

        prices = [p['price'] for p in self.price_history]

        # Calculate short and long term averages
        short_avg = sum(prices[-10:]) / 10
        long_avg = sum(prices[-50:]) / min(50, len(prices))
        current = prices[-1]

        # Volatility (rolling standard deviation)
        if len(prices) >= 20:
            returns = [(prices[i] - prices[i-1]) / prices[i-1]
                      for i in range(1, len(prices))]
            vol = (sum((r - sum(returns)/len(returns))**2 for r in returns[-20:]) / 20) ** 0.5
        else:
            vol = 0

        # Trend direction
        trend = (short_avg - long_avg) / long_avg if long_avg > 0 else 0

        # Momentum
        momentum = (current - prices[-10]) / prices[-10] if len(prices) >= 10 else 0

        # High volatility = uncertain regime
        if vol > 0.02:  # >2% volatility
            if momentum > 0.005:  # Up momentum in volatile market
                return Signal('regime', 1, 0.5, f"Volatile uptrend (vol={vol:.1%}, mom={momentum:.2%})")
            elif momentum < -0.005:
                return Signal('regime', -1, 0.5, f"Volatile downtrend (vol={vol:.1%}, mom={momentum:.2%})")
            return Signal('regime', 0, 0.3, f"High volatility, no trend (vol={vol:.1%})")

        # Low volatility with trend
        if trend > 0.005 and momentum > 0:
            return Signal('regime', 1, 0.6, f"Bullish regime (trend={trend:.2%})")
        elif trend < -0.005 and momentum < 0:
            return Signal('regime', -1, 0.6, f"Bearish regime (trend={trend:.2%})")

        return Signal('regime', 0, 0.3, f"Neutral regime (trend={trend:.2%})")


# =============================================================================
# UNIFIED SIGNAL AGGREGATOR
# =============================================================================

class UnifiedSignalAggregator:
    """Aggregates all signal sources into trading decisions."""

    def __init__(self):
        self.whale = WhaleSignal()
        self.flow = FlowSignal()
        self.pattern = PatternSignal()
        self.regime = RegimeSignal()

    def update(self, flow_state: dict, price: float) -> None:
        """Update all signal sources."""
        self.whale.update(flow_state)
        self.flow.update(flow_state)
        self.pattern.update()
        self.regime.update(price)

    def get_combined_signal(self) -> Tuple[int, float, str]:
        """
        Get weighted ensemble signal.

        Returns:
            (direction, confidence, reason)
            direction: +1 LONG, -1 SHORT, 0 NEUTRAL
        """
        signals = {
            'whale': self.whale.get_signal(),
            'flow': self.flow.get_signal(),
            'pattern': self.pattern.get_signal(),
            'regime': self.regime.get_signal(),
        }

        # Weighted combination
        weighted_sum = 0.0
        total_weight = 0.0
        reasons = []

        for name, signal in signals.items():
            weight = SIGNAL_WEIGHTS.get(name, 0.1)
            contribution = weight * signal.direction * signal.confidence
            weighted_sum += contribution
            total_weight += weight * signal.confidence

            if signal.confidence > 0.3:
                reasons.append(f"{name}: {signal.reason}")

        # Normalize
        if total_weight > 0:
            combined = weighted_sum / total_weight
        else:
            combined = 0

        confidence = abs(combined)
        reason = " | ".join(reasons) if reasons else "No signals"

        # Threshold for trading
        if combined > 0.3:
            return (1, confidence, reason)
        elif combined < -0.3:
            return (-1, confidence, reason)

        return (0, 0, reason)


# =============================================================================
# EXECUTION (Paper and Live)
# =============================================================================

class PaperExecutor:
    """Paper trading executor."""

    def __init__(self, initial_capital: float = 100.0):
        self.capital = initial_capital
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self._price_cache = 0.0
        self._price_time = 0

    def get_price(self) -> float:
        """Get current BTC price."""
        if time.time() - self._price_time < 2:
            return self._price_cache

        apis = [
            ("https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
             lambda d: float(d['result']['XXBTZUSD']['c'][0])),
            ("https://www.bitstamp.net/api/v2/ticker/btcusd/",
             lambda d: float(d['last'])),
        ]

        for url, parser in apis:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                    self._price_cache = parser(data)
                    self._price_time = time.time()
                    return self._price_cache
            except Exception:
                continue

        return self._price_cache if self._price_cache > 0 else 100000.0

    def open_position(self, direction: int, size_usd: float,
                     entry_price: float, reasons: List[str]) -> Optional[Position]:
        """Open a new position."""
        if len(self.positions) >= MAX_POSITIONS:
            logger.warning(f"Max positions ({MAX_POSITIONS}) reached")
            return None

        # Calculate TP/SL
        if direction == 1:  # LONG
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)
            take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
        else:  # SHORT
            stop_loss = entry_price * (1 + STOP_LOSS_PCT)
            take_profit = entry_price * (1 - TAKE_PROFIT_PCT)

        position = Position(
            id=f"POS_{int(time.time()*1000)}",
            direction=direction,
            size_usd=size_usd * LEVERAGE,
            entry_price=entry_price,
            entry_time=time.time(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_reasons=reasons
        )

        self.positions.append(position)
        logger.info(f"[OPEN] {'LONG' if direction == 1 else 'SHORT'} "
                   f"${size_usd:.2f} x {LEVERAGE}x @ ${entry_price:.2f} | "
                   f"TP: ${take_profit:.2f} | SL: ${stop_loss:.2f}")

        return position

    def close_position(self, position: Position, exit_price: float,
                      reason: str) -> Trade:
        """Close a position."""
        if position not in self.positions:
            return None

        # Calculate PnL
        if position.direction == 1:  # LONG
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:  # SHORT
            pnl_pct = (position.entry_price - exit_price) / position.entry_price

        pnl_usd = position.size_usd * pnl_pct

        # Apply to capital
        base_size = position.size_usd / LEVERAGE
        self.capital += (pnl_usd - base_size * 0.001)  # 0.1% fees round trip

        trade = Trade(
            id=position.id.replace('POS_', 'TRADE_'),
            direction=position.direction,
            size_usd=position.size_usd,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            exit_price=exit_price,
            exit_time=time.time(),
            exit_reason=reason,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct
        )

        self.trades.append(trade)
        self.positions.remove(position)

        logger.info(f"[CLOSE] {reason} @ ${exit_price:.2f} | "
                   f"PnL: ${pnl_usd:.2f} ({pnl_pct:+.2%}) | "
                   f"Capital: ${self.capital:.2f}")

        return trade

    def check_exits(self, current_price: float) -> List[Trade]:
        """Check all positions for exit conditions."""
        closed = []
        for pos in list(self.positions):
            if pos.direction == 1:  # LONG
                if current_price >= pos.take_profit:
                    closed.append(self.close_position(pos, current_price, "TAKE_PROFIT"))
                elif current_price <= pos.stop_loss:
                    closed.append(self.close_position(pos, current_price, "STOP_LOSS"))
            else:  # SHORT
                if current_price <= pos.take_profit:
                    closed.append(self.close_position(pos, current_price, "TAKE_PROFIT"))
                elif current_price >= pos.stop_loss:
                    closed.append(self.close_position(pos, current_price, "STOP_LOSS"))
        return [t for t in closed if t]

    def get_stats(self) -> dict:
        """Get trading statistics."""
        if not self.trades:
            return {
                'trades': 0, 'wins': 0, 'losses': 0,
                'win_rate': 0, 'total_pnl': 0, 'capital': self.capital,
                'positions': len(self.positions)
            }

        wins = len([t for t in self.trades if t.pnl_usd > 0])
        losses = len([t for t in self.trades if t.pnl_usd <= 0])

        return {
            'trades': len(self.trades),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(self.trades) * 100,
            'total_pnl': sum(t.pnl_usd for t in self.trades),
            'capital': self.capital,
            'positions': len(self.positions)
        }


# =============================================================================
# MAIN TRADER
# =============================================================================

class UnifiedTrader:
    """Main unified trading system."""

    def __init__(self, mode: str = 'paper', capital: float = 100.0):
        self.mode = mode
        self.aggregator = UnifiedSignalAggregator()
        self.executor = PaperExecutor(capital)
        self.last_trade_time = 0
        self.running = False

    def load_flow_state(self) -> dict:
        """Load state from flow_hft.py."""
        state = {
            'whale_count': 0, 'whale_btc': 0, 'mega_whales': 0,
            'inflow_btc': 0, 'outflow_btc': 0, 'fis': 0
        }

        if FLOW_STATE_FILE.exists():
            try:
                with open(FLOW_STATE_FILE) as f:
                    data = json.load(f)
                state.update(data)
            except Exception:
                pass

        return state

    def run_iteration(self) -> None:
        """Run one trading iteration."""
        try:
            # Get current price
            price = self.executor.get_price()
            if price <= 0:
                return

            # Load flow state
            flow_state = self.load_flow_state()

            # Update signals
            self.aggregator.update(flow_state, price)

            # Check exits first
            self.executor.check_exits(price)

            # Check cooldown
            if time.time() - self.last_trade_time < COOLDOWN_SECONDS:
                return

            # Get combined signal
            direction, confidence, reason = self.aggregator.get_combined_signal()

            if direction != 0 and confidence >= MIN_CONFIDENCE:
                # Calculate position size
                available = self.executor.capital * POSITION_SIZE_PCT

                # Open position
                pos = self.executor.open_position(
                    direction=direction,
                    size_usd=available,
                    entry_price=price,
                    reasons=[reason]
                )

                if pos:
                    self.last_trade_time = time.time()
                    logger.info(f"[SIGNAL] {reason}")

        except Exception as e:
            logger.error(f"Iteration error: {e}")

    async def run(self) -> None:
        """Main trading loop."""
        logger.info("=" * 60)
        logger.info("UNIFIED HOSTINGER TRADER")
        logger.info("=" * 60)
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Capital: ${self.executor.capital:.2f}")
        logger.info(f"Leverage: {LEVERAGE}x")
        logger.info(f"TP: {TAKE_PROFIT_PCT*100:.1f}% | SL: {STOP_LOSS_PCT*100:.2f}%")
        logger.info(f"Signal weights: {SIGNAL_WEIGHTS}")
        logger.info("=" * 60)

        self.running = True
        iteration = 0

        while self.running:
            try:
                self.run_iteration()

                iteration += 1
                if iteration % 60 == 0:
                    stats = self.executor.get_stats()
                    price = self.executor.get_price()
                    logger.info(f"[STATS] BTC: ${price:.0f} | Capital: ${stats['capital']:.2f} | "
                               f"Trades: {stats['trades']} | Win: {stats['win_rate']:.0f}% | "
                               f"PnL: ${stats['total_pnl']:.2f} | Pos: {stats['positions']}")

                await asyncio.sleep(1)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(5)

        # Final stats
        stats = self.executor.get_stats()
        logger.info("=" * 60)
        logger.info("FINAL STATS")
        logger.info(f"Trades: {stats['trades']}")
        logger.info(f"Win Rate: {stats['win_rate']:.1f}%")
        logger.info(f"Total PnL: ${stats['total_pnl']:.2f}")
        logger.info(f"Final Capital: ${stats['capital']:.2f}")
        logger.info("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description='Unified Hostinger Trader')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper')
    parser.add_argument('--capital', type=float, default=100.0)
    args = parser.parse_args()

    trader = UnifiedTrader(mode=args.mode, capital=args.capital)
    await trader.run()


if __name__ == '__main__':
    asyncio.run(main())
