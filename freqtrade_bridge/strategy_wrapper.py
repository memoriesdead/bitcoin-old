"""
Renaissance Strategy - Direct Blockchain Feed
=============================================

Uses YOUR blockchain WebSocket feed for nanosecond speed.
Freqtrade's exchange polling is TOO SLOW - we bypass it.

Architecture:
  YOUR BlockchainFeed (10+ WebSocket) -> direct to formulas
  NOT: Freqtrade -> Exchange API -> 1 minute candles (TOO SLOW)
"""

import sys
import os
from typing import Optional
import asyncio
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pandas import DataFrame

from freqtrade_bridge.formula_adapter import FormulaToIndicator
from blockchain.blockchain_price_engine import BlockchainState


class RenaissanceLiveStrategy:
    """
    Live trading strategy using DIRECT blockchain feed.

    This is NOT a Freqtrade IStrategy - we bypass Freqtrade's slow data feed.
    We use YOUR blockchain WebSocket for millisecond updates.
    """

    def __init__(self,
                 calibration_price: float = 97000.0,
                 entry_threshold: float = 0.3,
                 exit_threshold: float = -0.1,
                 min_signals: int = 30):
        """
        Args:
            calibration_price: Starting price anchor
            entry_threshold: Signal threshold to enter (0.3 = 30% bullish majority)
            exit_threshold: Signal threshold to exit (-0.1 = slight bearish)
            min_signals: Minimum active formulas required
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_signals = min_signals
        self.calibration_price = calibration_price

        # Formula adapter with YOUR blockchain feed
        self.adapter = FormulaToIndicator()

        # State
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = 0.0
        self.last_signal = 0.0
        self.last_confidence = 0.0
        self._running = False

        print(f"[RenaissanceStrategy] Initialized with {len(self.adapter.formula_instances)} formulas")

    async def start(self):
        """Start the direct blockchain feed."""
        await self.adapter.start_blockchain_feed(self.calibration_price)
        self._running = True
        print("[RenaissanceStrategy] LIVE - Direct blockchain connection")

    def stop(self):
        """Stop the feed."""
        self._running = False
        self.adapter.stop_blockchain_feed()

    def on_blockchain_update(self, state: BlockchainState) -> dict:
        """
        Called on EVERY blockchain update (millisecond frequency).

        This is where the trading decisions happen.
        NO candles, NO delays - pure tick-by-tick processing.

        Args:
            state: BlockchainState from your blockchain_market_data.py

        Returns:
            Dict with action and details
        """
        # Run all 423 formulas on this update
        result = self.adapter.update_from_blockchain(state)

        signal = result['signal']
        confidence = result['confidence']
        price = result['price']
        bullish = result['bullish_count']
        bearish = result['bearish_count']

        self.last_signal = signal
        self.last_confidence = confidence

        action = 'HOLD'
        details = {}

        # Check for entry
        if self.position == 0:
            if signal > self.entry_threshold and confidence > 0.3:
                if bullish >= self.min_signals:
                    action = 'ENTER_LONG'
                    self.position = 1
                    self.entry_price = price
                    details = {
                        'reason': f'Signal {signal:.3f} > {self.entry_threshold}',
                        'bullish': bullish,
                        'bearish': bearish,
                        'confidence': confidence,
                    }

            elif signal < -self.entry_threshold and confidence > 0.3:
                if bearish >= self.min_signals:
                    action = 'ENTER_SHORT'
                    self.position = -1
                    self.entry_price = price
                    details = {
                        'reason': f'Signal {signal:.3f} < -{self.entry_threshold}',
                        'bullish': bullish,
                        'bearish': bearish,
                        'confidence': confidence,
                    }

        # Check for exit
        elif self.position == 1:  # Long position
            if signal < self.exit_threshold or bearish > bullish * 1.5:
                action = 'EXIT_LONG'
                pnl = (price - self.entry_price) / self.entry_price * 100
                self.position = 0
                details = {
                    'reason': f'Signal dropped to {signal:.3f}',
                    'pnl_pct': pnl,
                    'entry': self.entry_price,
                    'exit': price,
                }
                self.entry_price = 0

        elif self.position == -1:  # Short position
            if signal > -self.exit_threshold or bullish > bearish * 1.5:
                action = 'EXIT_SHORT'
                pnl = (self.entry_price - price) / self.entry_price * 100
                self.position = 0
                details = {
                    'reason': f'Signal rose to {signal:.3f}',
                    'pnl_pct': pnl,
                    'entry': self.entry_price,
                    'exit': price,
                }
                self.entry_price = 0

        return {
            'action': action,
            'signal': signal,
            'confidence': confidence,
            'price': price,
            'position': self.position,
            'bullish': bullish,
            'bearish': bearish,
            'details': details,
            'timestamp': time.time(),
        }


# ==================== FREQTRADE ISTRATEGY (DISABLED) ====================
# Freqtrade's IStrategy uses candles from exchange API - TOO SLOW.
# We keep this for backtesting historical data only, but live trading
# uses RenaissanceLiveStrategy above with direct blockchain feed.

try:
    from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter

    class RenaissanceFreqtradeStrategy(IStrategy):
        """
        Freqtrade strategy for BACKTESTING ONLY.

        For live trading, use RenaissanceLiveStrategy with direct blockchain feed.
        This is only useful for validating formulas on historical exchange data.
        """

        INTERFACE_VERSION = 3

        minimal_roi = {"0": 0.05, "30": 0.03, "60": 0.02, "120": 0.01}
        stoploss = -0.03
        timeframe = '1m'
        startup_candle_count: int = 100

        # NOTE: This uses SLOW exchange candles, not blockchain
        # Only use for backtesting validation

        def __init__(self, config: dict) -> None:
            super().__init__(config)
            self.adapter = FormulaToIndicator()

        def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            """Run formulas on exchange candles (SLOW - backtesting only)."""
            dataframe = self.adapter.compute_signals_batch(dataframe)
            dataframe = self.adapter.get_aggregated_signal(dataframe)
            return dataframe

        def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe.loc[
                (dataframe['agg_signal'] > 0.3) &
                (dataframe['bullish_count'] > 30),
                'enter_long'
            ] = 1
            return dataframe

        def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe.loc[
                (dataframe['agg_signal'] < -0.1),
                'exit_long'
            ] = 1
            return dataframe

except ImportError:
    # Freqtrade not installed - no problem, we use direct blockchain anyway
    RenaissanceFreqtradeStrategy = None
