"""
Data Collector
==============

Orchestrates signal and price logging for edge validation.

This is the main entry point for data collection.

Usage:
    python -m engine.sovereign.validation.data_collector

On Hostinger:
    python data_collector.py --duration 168  # Run for 1 week (168 hours)
"""

import os
import sys
import time
import json
import signal
import argparse
import threading
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import logging

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from engine.sovereign.validation.signal_logger import SignalLogger
from engine.sovereign.validation.price_logger import PriceLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_collection.log')
    ]
)
logger = logging.getLogger(__name__)


class DataCollector:
    """
    Main data collection orchestrator.

    Collects:
    1. All blockchain signals (from signal engine)
    2. BTC price every second (from exchanges)

    For edge validation before live trading.
    """

    def __init__(self,
                 data_dir: str = "data/validation",
                 use_websocket: bool = True):
        """
        Initialize data collector.

        Args:
            data_dir: Directory for databases
            use_websocket: Use WebSocket for price (faster)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize loggers
        self.signal_logger = SignalLogger(str(self.data_dir / "signals.db"))
        self.price_logger = PriceLogger(str(self.data_dir / "prices.db"))

        self.use_websocket = use_websocket

        # State
        self.running = False
        self.start_time = None

        # Signal handlers
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        logger.info(f"DataCollector initialized: {self.data_dir}")

    def _shutdown(self, signum, frame):
        """Handle shutdown signal."""
        logger.info("Shutdown signal received...")
        self.stop()

    def start(self, duration_hours: Optional[float] = None):
        """
        Start data collection.

        Args:
            duration_hours: How long to run (None = forever)
        """
        logger.info("=" * 60)
        logger.info("DATA COLLECTION STARTED")
        logger.info("=" * 60)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Duration: {duration_hours or 'unlimited'} hours")
        logger.info("")

        self.running = True
        self.start_time = time.time()

        # Start price logger
        if self.use_websocket:
            try:
                self.price_logger.start_binance_ws()
                logger.info("Price logging: WebSocket (Binance)")
            except Exception as e:
                logger.warning(f"WebSocket failed: {e}, falling back to polling")
                self.price_logger.start_polling(interval=1.0)
        else:
            self.price_logger.start_polling(interval=1.0)
            logger.info("Price logging: REST polling (1s)")

        # Calculate end time
        end_time = None
        if duration_hours:
            end_time = self.start_time + (duration_hours * 3600)

        # Main loop - wait for signals
        logger.info("")
        logger.info("Waiting for blockchain signals...")
        logger.info("(Connect your signal engine to feed signals)")
        logger.info("")

        self._status_loop(end_time)

    def _status_loop(self, end_time: Optional[float]):
        """Main status loop."""
        last_status = 0
        status_interval = 60  # Print status every 60 seconds

        while self.running:
            now = time.time()

            # Check duration
            if end_time and now >= end_time:
                logger.info("Duration reached, stopping...")
                break

            # Print status
            if now - last_status >= status_interval:
                self._print_status()
                last_status = now

            time.sleep(1)

        self.stop()

    def _print_status(self):
        """Print collection status."""
        signal_stats = self.signal_logger.get_stats()
        price_stats = self.price_logger.get_stats()

        runtime = time.time() - self.start_time
        hours = runtime / 3600

        logger.info("-" * 40)
        logger.info(f"Runtime: {hours:.2f} hours")
        logger.info(f"Signals: {signal_stats['total_signals']:,} "
                   f"({signal_stats['signals_per_hour']:.0f}/hr)")
        logger.info(f"Prices: {price_stats['total_prices']:,} "
                   f"({price_stats['prices_per_second']:.1f}/s)")
        logger.info(f"Current BTC: ${price_stats['current_price']:,.2f}")

        if signal_stats['total_signals'] > 0:
            logger.info(f"Tradeable: {signal_stats['tradeable_pct']:.1f}% "
                       f"(L:{signal_stats['long_signals']} S:{signal_stats['short_signals']})")

    def stop(self):
        """Stop data collection."""
        logger.info("Stopping data collection...")
        self.running = False

        if self.use_websocket:
            self.price_logger.stop_ws()
        else:
            self.price_logger.stop_polling()

        # Final stats
        self._print_final_stats()

    def _print_final_stats(self):
        """Print final collection statistics."""
        signal_stats = self.signal_logger.get_stats()
        price_stats = self.price_logger.get_stats()

        logger.info("")
        logger.info("=" * 60)
        logger.info("DATA COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info("")
        logger.info("SIGNALS:")
        logger.info(f"  Total: {signal_stats['total_signals']:,}")
        logger.info(f"  Long: {signal_stats['long_signals']:,}")
        logger.info(f"  Short: {signal_stats['short_signals']:,}")
        logger.info(f"  Neutral: {signal_stats['neutral_signals']:,}")
        logger.info(f"  Tradeable: {signal_stats['tradeable_signals']:,} "
                   f"({signal_stats['tradeable_pct']:.1f}%)")
        logger.info(f"  Avg Confidence: {signal_stats['avg_confidence']:.3f}")
        logger.info("")
        logger.info("PRICES:")
        logger.info(f"  Total: {price_stats['total_prices']:,}")
        logger.info(f"  Rate: {price_stats['prices_per_second']:.1f}/sec")
        logger.info("")
        logger.info(f"Runtime: {signal_stats['runtime_hours']:.2f} hours")
        logger.info("")
        logger.info("Data files:")
        logger.info(f"  {self.data_dir / 'signals.db'}")
        logger.info(f"  {self.data_dir / 'prices.db'}")
        logger.info("")
        logger.info("Next: Run analysis to calculate edge")
        logger.info("  python -m engine.sovereign.validation.analysis.calculate_edge")
        logger.info("=" * 60)

    # =========================================================================
    # SIGNAL INTERFACE
    # =========================================================================

    def log_signal(self, signal: Dict[str, Any]) -> int:
        """
        Log a signal from the engine.

        Call this from your signal engine whenever a signal is generated.

        Args:
            signal: Signal dictionary

        Returns:
            Signal ID
        """
        price = self.price_logger.get_current_price()
        return self.signal_logger.log_signal(signal, price)

    def log_blockchain_signal(self, blockchain_signal: Dict) -> int:
        """
        Log signal from blockchain feed.

        Call this from per_exchange_feed.get_aggregated_signal().
        """
        price = self.price_logger.get_current_price()
        return self.signal_logger.log_from_blockchain_feed(blockchain_signal, price)

    def get_current_price(self) -> float:
        """Get current BTC price."""
        return self.price_logger.get_current_price()

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            'signals': self.signal_logger.get_stats(),
            'prices': self.price_logger.get_stats(),
            'running': self.running,
            'start_time': self.start_time,
            'runtime_hours': (time.time() - self.start_time) / 3600 if self.start_time else 0,
        }


# =============================================================================
# INTEGRATION WITH BLOCKCHAIN FEED
# =============================================================================

class BlockchainDataCollector(DataCollector):
    """
    Data collector with blockchain signal integration.

    Connects directly to your existing blockchain feed.
    """

    def __init__(self, data_dir: str = "data/validation"):
        super().__init__(data_dir)
        self.signal_callback = None

    def connect_blockchain_feed(self, feed):
        """
        Connect to per_exchange_feed for automatic signal logging.

        Args:
            feed: PerExchangeFeed instance
        """
        # Register callback
        original_callback = feed.on_signal if hasattr(feed, 'on_signal') else None

        def signal_wrapper(signal):
            # Log signal
            self.log_blockchain_signal(signal)

            # Call original callback if exists
            if original_callback:
                original_callback(signal)

        feed.on_signal = signal_wrapper
        logger.info("Connected to blockchain feed")

    def start_with_feed(self, feed, duration_hours: Optional[float] = None):
        """
        Start collection with blockchain feed.

        Args:
            feed: PerExchangeFeed instance
            duration_hours: How long to run
        """
        self.connect_blockchain_feed(feed)
        self.start(duration_hours)


# =============================================================================
# STANDALONE RUNNER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Collect signals for edge validation")
    parser.add_argument("--duration", type=float, default=None,
                       help="Duration in hours (default: unlimited)")
    parser.add_argument("--data-dir", type=str, default="data/validation",
                       help="Data directory")
    parser.add_argument("--no-websocket", action="store_true",
                       help="Use REST polling instead of WebSocket")
    parser.add_argument("--test", action="store_true",
                       help="Run quick test (5 minutes)")

    args = parser.parse_args()

    # Test mode
    if args.test:
        args.duration = 5 / 60  # 5 minutes

    # Create collector
    collector = DataCollector(
        data_dir=args.data_dir,
        use_websocket=not args.no_websocket
    )

    # Start collection
    collector.start(duration_hours=args.duration)


if __name__ == "__main__":
    main()
