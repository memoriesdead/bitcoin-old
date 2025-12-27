"""
RenTech Paper Trading Runner
============================

Paper trading to validate RenTech formulas with live market data.

Usage:
    python -m engine.sovereign.formulas.paper_trader

Features:
- Connects to Hyperliquid WebSocket for real-time BTC prices
- Evaluates all 9 RenTech formulas in real-time
- Logs paper trades to JSON file
- Tracks simulated P&L
- No actual orders placed

Created: 2025-12-16
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import websockets

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Record of a paper trade."""
    trade_id: str
    formula_id: str
    formula_name: str
    direction: str  # LONG or SHORT
    entry_time: float
    entry_price: float
    hold_days: int
    target_exit_time: float
    confidence: float
    position_size: float  # In USD
    leverage: float
    status: str  # OPEN, CLOSED
    exit_time: Optional[float] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


class PaperTrader:
    """
    Paper trading engine for RenTech formulas.

    Uses the RenTechIntegration class to evaluate signals
    and tracks simulated trades.
    """

    def __init__(
        self,
        capital: float = 100.0,
        db_path: str = "data/unified_bitcoin.db",
        output_dir: str = "data/paper_trades"
    ):
        """
        Initialize paper trader.

        Args:
            capital: Starting capital in USD
            db_path: Path to historical data
            output_dir: Directory for trade logs
        """
        self.capital = capital
        self.current_equity = capital
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Import RenTech components
        from engine.sovereign.formulas.rentech_features import RenTechFeatures
        from engine.sovereign.formulas.rentech_evaluator import RenTechEvaluator
        from engine.sovereign.formulas.rentech_validator import RenTechValidator
        from engine.sovereign.strategy.kelly import RenTechSizer
        from engine.sovereign.formulas.rentech_monitor import RenTechMonitor

        # Initialize components
        self.features = RenTechFeatures(db_path)
        self.evaluator = RenTechEvaluator()
        self.validator = RenTechValidator(state_file=str(self.output_dir / "validator_state.json"))
        self.sizer = RenTechSizer(base_leverage=10.0)
        self.monitor = RenTechMonitor(
            validator=self.validator,
            initial_capital=capital,
            output_file=str(self.output_dir / "monitor.json")
        )

        # Trade tracking
        self.trades: List[PaperTrade] = []
        self.open_trades: Dict[str, PaperTrade] = {}  # formula_id -> trade

        # Stats
        self.signals_received = 0
        self.trades_executed = 0
        self.trades_closed = 0

        # Current price
        self.current_price = 0.0
        self.last_price_time = 0.0

        # Load existing trades
        self._load_trades()

        logger.info(f"Paper trader initialized with ${capital:.2f} capital")
        logger.info(f"Features engine warmed up: {self.features.is_ready()}")

    def _load_trades(self):
        """Load existing trades from file."""
        trades_file = self.output_dir / "trades.json"
        if trades_file.exists():
            try:
                with open(trades_file) as f:
                    data = json.load(f)
                    for trade_data in data.get("trades", []):
                        trade = PaperTrade(**trade_data)
                        self.trades.append(trade)
                        if trade.status == "OPEN":
                            self.open_trades[trade.formula_id] = trade
                logger.info(f"Loaded {len(self.trades)} existing trades")
            except Exception as e:
                logger.error(f"Failed to load trades: {e}")

    def _save_trades(self):
        """Save trades to file."""
        trades_file = self.output_dir / "trades.json"
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

    def on_price_update(self, price: float, timestamp: Optional[float] = None):
        """
        Process a price update.

        Args:
            price: Current BTC price
            timestamp: Unix timestamp
        """
        ts = timestamp or time.time()
        self.current_price = price
        self.last_price_time = ts

        # Update features
        features = self.features.update(price, timestamp=ts)

        if not self.features.is_ready():
            return

        # Check for exit conditions on open trades
        self._check_exits(price, ts)

        # Evaluate formulas for new signals
        feature_dict = self.features.get_feature_dict()
        signals = self.evaluator.evaluate(feature_dict, price)

        for signal in signals:
            self.signals_received += 1
            self._process_signal(signal, price, ts)

    def _check_exits(self, price: float, timestamp: float):
        """Check if any open trades should be closed."""
        to_close = []

        for formula_id, trade in self.open_trades.items():
            # Check time-based exit
            if timestamp >= trade.target_exit_time:
                to_close.append((formula_id, "TIME_EXIT"))
                continue

            # Calculate current P&L
            if trade.direction == "LONG":
                pnl_pct = (price / trade.entry_price - 1) * 100
            else:
                pnl_pct = (1 - price / trade.entry_price) * 100

            # Check stop loss (2% with leverage)
            if pnl_pct < -2.0:
                to_close.append((formula_id, "STOP_LOSS"))
            # Check take profit (varies by formula)
            elif pnl_pct > 25.0:
                to_close.append((formula_id, "TAKE_PROFIT"))

        for formula_id, reason in to_close:
            self._close_trade(formula_id, price, timestamp, reason)

    def _process_signal(self, signal, price: float, timestamp: float):
        """Process a trading signal."""
        formula_id = signal.formula_id

        # Check if already have position in this formula
        if formula_id in self.open_trades:
            return

        # Check with validator
        can_trade, multiplier = self.validator.should_trade(formula_id)
        if not can_trade:
            logger.info(f"Signal {formula_id} blocked by validator")
            return

        # Calculate position size
        state = self.validator.formula_states.get(formula_id)
        n_trades = state.live_trades if state else 0

        size_result = self.sizer.calculate_position_size(
            formula_id=formula_id,
            capital=self.capital,
            win_rate=signal.confidence,
            avg_return=20.0,  # Conservative estimate
            n_live_trades=n_trades,
            validator_multiplier=multiplier
        )

        # Create paper trade
        trade_id = f"{formula_id}_{int(timestamp)}"
        trade = PaperTrade(
            trade_id=trade_id,
            formula_id=formula_id,
            formula_name=signal.formula_name,
            direction=signal.direction.name,
            entry_time=timestamp,
            entry_price=price,
            hold_days=signal.hold_days,
            target_exit_time=timestamp + signal.hold_days * 24 * 60 * 60,
            confidence=signal.confidence,
            position_size=size_result["position_size"],
            leverage=size_result["effective_leverage"],
            status="OPEN"
        )

        self.trades.append(trade)
        self.open_trades[formula_id] = trade
        self.trades_executed += 1

        # Log
        logger.info(f"PAPER TRADE OPENED: {formula_id}")
        logger.info(f"  Direction: {trade.direction}")
        logger.info(f"  Entry: ${price:,.2f}")
        logger.info(f"  Size: ${trade.position_size:.2f} @ {trade.leverage:.0f}x")
        logger.info(f"  Hold: {trade.hold_days} days")

        # Update monitor
        self.monitor.on_signal(signal)
        self.monitor.on_position_opened(signal, trade.position_size, trade.leverage)

        # Save
        self._save_trades()

    def _close_trade(self, formula_id: str, price: float, timestamp: float, reason: str):
        """Close an open trade."""
        if formula_id not in self.open_trades:
            return

        trade = self.open_trades[formula_id]

        # Calculate P&L
        if trade.direction == "LONG":
            pnl_pct = (price / trade.entry_price - 1) * 100
        else:
            pnl_pct = (1 - price / trade.entry_price) * 100

        pnl = trade.position_size * (pnl_pct / 100) * trade.leverage

        # Update trade
        trade.exit_time = timestamp
        trade.exit_price = price
        trade.pnl_pct = pnl_pct
        trade.pnl = pnl
        trade.status = "CLOSED"

        # Update equity
        self.current_equity += pnl
        self.trades_closed += 1

        # Remove from open trades
        del self.open_trades[formula_id]

        # Log
        logger.info(f"PAPER TRADE CLOSED: {formula_id}")
        logger.info(f"  Reason: {reason}")
        logger.info(f"  Entry: ${trade.entry_price:,.2f}")
        logger.info(f"  Exit: ${price:,.2f}")
        logger.info(f"  P&L: {pnl_pct:+.2f}% (${pnl:+.2f})")
        logger.info(f"  New Equity: ${self.current_equity:.2f}")

        # Update validator
        self.validator.record_trade(
            formula_id=formula_id,
            entry_time=trade.entry_time,
            exit_time=timestamp,
            entry_price=trade.entry_price,
            exit_price=price,
            direction=trade.direction,
            pnl_pct=pnl_pct
        )

        # Update monitor
        self.monitor.on_position_closed(formula_id, price, pnl, pnl_pct)

        # Save
        self._save_trades()
        self.monitor.save_snapshot()

    def print_status(self):
        """Print current status."""
        print("\n" + "=" * 60)
        print("  RENTECH PAPER TRADING STATUS")
        print("=" * 60)
        print(f"\nEquity: ${self.current_equity:.2f} (started: ${self.capital:.2f})")
        print(f"P&L: ${self.current_equity - self.capital:+.2f} ({(self.current_equity/self.capital-1)*100:+.1f}%)")
        print(f"\nSignals: {self.signals_received}")
        print(f"Trades: {self.trades_executed} opened, {self.trades_closed} closed")
        print(f"Open Positions: {len(self.open_trades)}")

        for fid, trade in self.open_trades.items():
            if self.current_price > 0:
                if trade.direction == "LONG":
                    pnl_pct = (self.current_price / trade.entry_price - 1) * 100
                else:
                    pnl_pct = (1 - self.current_price / trade.entry_price) * 100
                print(f"  {fid}: {trade.direction} @ ${trade.entry_price:,.0f} ({pnl_pct:+.1f}%)")

        # Show closed trade stats
        closed_trades = [t for t in self.trades if t.status == "CLOSED"]
        if closed_trades:
            wins = sum(1 for t in closed_trades if t.pnl_pct > 0)
            print(f"\nClosed Trade Stats:")
            print(f"  Win Rate: {wins}/{len(closed_trades)} ({wins/len(closed_trades)*100:.0f}%)")
            print(f"  Avg Return: {sum(t.pnl_pct for t in closed_trades)/len(closed_trades):+.2f}%")

        print("=" * 60)


async def connect_hyperliquid(trader: PaperTrader):
    """Connect to Hyperliquid WebSocket for price updates."""
    uri = "wss://api.hyperliquid.xyz/ws"

    while True:
        try:
            async with websockets.connect(uri) as ws:
                # Subscribe to BTC trades
                subscribe_msg = {
                    "method": "subscribe",
                    "subscription": {
                        "type": "trades",
                        "coin": "BTC"
                    }
                }
                await ws.send(json.dumps(subscribe_msg))
                logger.info("Connected to Hyperliquid WebSocket")

                async for message in ws:
                    try:
                        data = json.loads(message)

                        # Handle trade updates
                        if data.get("channel") == "trades":
                            trades = data.get("data", [])
                            for trade in trades:
                                price = float(trade.get("px", 0))
                                if price > 0:
                                    trader.on_price_update(price)

                        # Handle subscription confirmation
                        elif "subscriptionResponse" in str(data):
                            logger.info("Subscription confirmed")

                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            logger.info("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)


async def status_printer(trader: PaperTrader, interval: int = 300):
    """Print status periodically."""
    while True:
        await asyncio.sleep(interval)
        trader.print_status()
        trader.monitor.save_snapshot()


async def main():
    """Main entry point."""
    print("=" * 60)
    print("  RENTECH PAPER TRADING")
    print("=" * 60)
    print("\nStarting paper trading with $100 capital...")
    print("Press Ctrl+C to stop\n")

    trader = PaperTrader(capital=100.0)
    trader.print_status()

    # Run WebSocket and status printer concurrently
    await asyncio.gather(
        connect_hyperliquid(trader),
        status_printer(trader, interval=300)
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nPaper trading stopped by user")
