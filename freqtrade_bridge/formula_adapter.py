"""
Formula Adapter - Direct Blockchain Connection
==============================================

Connects 423 Renaissance formulas to YOUR blockchain feed.
NO FREQTRADE DATA FEED - we use direct blockchain WebSocket for nanosecond speed.

Data flow:
  BlockchainFeed (10+ WebSocket) -> BlockchainMarketData -> BlockchainPriceEngine
                                                                    |
                                                                    v
  RenaissanceFormulas (423) <- Real price + blockchain signals <- This Adapter
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import sys
import os
import asyncio
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formulas.base import BaseFormula, FORMULA_REGISTRY

# Import YOUR blockchain feed - nanosecond speed
from blockchain.blockchain_feed import BlockchainFeed, NetworkStats
from blockchain.blockchain_market_data import BlockchainMarketData
from blockchain.blockchain_price_engine import BlockchainPriceEngine, BlockchainState


class FormulaToIndicator:
    """
    Runs 423 formulas on DIRECT BLOCKCHAIN DATA.

    NO exchange API delays. NO Freqtrade data feed.
    Direct WebSocket from mempool.space, blockstream = millisecond latency.
    """

    def __init__(self, formula_ids: Optional[List[int]] = None):
        """
        Args:
            formula_ids: List of formula IDs. None = all 423 formulas.
        """
        if formula_ids is None:
            self.formula_ids = list(FORMULA_REGISTRY.keys())
        else:
            self.formula_ids = [fid for fid in formula_ids if fid in FORMULA_REGISTRY]

        self.formula_instances: Dict[int, BaseFormula] = {}
        self._initialize_formulas()

        # YOUR blockchain feed components
        self.blockchain_feed: Optional[BlockchainFeed] = None
        self.market_data: Optional[BlockchainMarketData] = None
        self.price_engine: Optional[BlockchainPriceEngine] = None

        self._running = False
        self._last_signal = 0.0
        self._last_confidence = 0.0

        print(f"[FormulaAdapter] Loaded {len(self.formula_instances)} formulas for DIRECT blockchain")

    def _initialize_formulas(self) -> None:
        """Create formula instances."""
        for fid in self.formula_ids:
            try:
                formula_class = FORMULA_REGISTRY.get(fid)
                if formula_class:
                    self.formula_instances[fid] = formula_class()
            except Exception as e:
                pass  # Skip broken formulas

    async def start_blockchain_feed(self, calibration_price: float = 97000.0):
        """
        Start YOUR blockchain feed - direct WebSocket connection.

        This is NANOSECOND speed compared to Freqtrade's exchange polling.
        """
        self.blockchain_feed = BlockchainFeed()
        self.market_data = BlockchainMarketData()
        self.price_engine = BlockchainPriceEngine(calibration_price=calibration_price)

        # Start the 10+ WebSocket connections
        await self.blockchain_feed.start()
        self._running = True

        print(f"[FormulaAdapter] Blockchain feed LIVE - direct mempool connection")

    def stop_blockchain_feed(self):
        """Stop blockchain feed."""
        self._running = False
        if self.blockchain_feed:
            asyncio.create_task(self.blockchain_feed.stop())

    def update_from_blockchain(self, state: BlockchainState) -> Dict:
        """
        Update ALL formulas from blockchain state.

        Called on EVERY blockchain update - millisecond frequency.
        No candles, no delays - pure blockchain signal.

        Args:
            state: BlockchainState from your blockchain_market_data.py

        Returns:
            Dict with aggregated signal and per-formula signals
        """
        if self.price_engine is None:
            return {'signal': 0, 'confidence': 0, 'formulas': {}}

        # Get price from blockchain signals
        derived = self.price_engine.update(state)
        price = derived.composite_price
        volume = state.tx_volume_btc_1m
        timestamp = state.timestamp or time.time()

        # Run ALL formulas on this price tick
        signals = {}
        confidences = {}

        for fid, formula in self.formula_instances.items():
            try:
                formula.update(price, volume, timestamp)
                sig = formula.get_signal()
                conf = formula.get_confidence()
                signals[fid] = float(sig) if sig else 0.0
                confidences[fid] = float(conf) if conf else 0.0
            except:
                signals[fid] = 0.0
                confidences[fid] = 0.0

        # Aggregate using Condorcet voting
        agg_signal, agg_confidence, bullish, bearish = self._aggregate_signals(signals, confidences)

        self._last_signal = agg_signal
        self._last_confidence = agg_confidence

        return {
            'signal': agg_signal,
            'confidence': agg_confidence,
            'price': price,
            'bullish_count': bullish,
            'bearish_count': bearish,
            'timestamp': timestamp,
            'formulas': signals,
            'blockchain': {
                'fee_fast': state.fee_fast,
                'mempool_size': state.mempool_size,
                'tx_volume': volume,
                'whale_count': state.whale_tx_count,
            }
        }

    def _aggregate_signals(self, signals: Dict[int, float], confidences: Dict[int, float]):
        """Condorcet voting aggregation."""
        bullish = sum(1 for s in signals.values() if s > 0.3)
        bearish = sum(1 for s in signals.values() if s < -0.3)
        total = bullish + bearish

        if total == 0:
            return 0.0, 0.0, 0, 0

        agg_signal = (bullish - bearish) / total
        agg_confidence = np.mean(list(confidences.values())) if confidences else 0.0

        return float(np.clip(agg_signal, -1, 1)), float(np.clip(agg_confidence, 0, 1)), bullish, bearish

    def get_current_signal(self) -> tuple:
        """Get last computed signal without updating."""
        return self._last_signal, self._last_confidence

    # ==================== FREQTRADE COMPATIBILITY (COMMENTED OUT) ====================
    # Freqtrade uses exchange candles - TOO SLOW for our use case.
    # We get data directly from blockchain at millisecond speed.
    #
    # def compute_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     DISABLED - Freqtrade candle-based processing is too slow.
    #     We use update_from_blockchain() for real-time processing.
    #     """
    #     raise NotImplementedError(
    #         "Freqtrade candle processing disabled. "
    #         "Use update_from_blockchain() for nanosecond speed."
    #     )

    def compute_signals_batch(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Batch process historical data for backtesting ONLY.

        For live trading, use update_from_blockchain() instead.
        This is just for validating formulas on historical blockchain data.
        """
        df = dataframe.copy()

        # Initialize columns
        for fid in self.formula_instances:
            df[f'formula_{fid}'] = 0.0

        # Reset formulas
        for formula in self.formula_instances.values():
            formula.reset()

        # Process each row
        for idx in range(len(df)):
            row = df.iloc[idx]
            price = float(row['close']) if 'close' in df.columns else float(row['price'])
            volume = float(row.get('volume', 0))
            timestamp = row['date'].timestamp() if 'date' in df.columns and hasattr(row['date'], 'timestamp') else float(idx)

            for fid, formula in self.formula_instances.items():
                try:
                    formula.update(price, volume, timestamp)
                    df.loc[df.index[idx], f'formula_{fid}'] = float(formula.get_signal() or 0)
                except:
                    pass

        return df

    def get_aggregated_signal(self, dataframe: pd.DataFrame, method: str = 'condorcet') -> pd.DataFrame:
        """Aggregate formula signals in dataframe."""
        df = dataframe.copy()
        signal_cols = [c for c in df.columns if c.startswith('formula_') and not c.endswith('_conf')]

        if not signal_cols:
            df['agg_signal'] = 0.0
            df['bullish_count'] = 0
            df['bearish_count'] = 0
            return df

        signals_df = df[signal_cols]

        bullish = (signals_df > 0.3).sum(axis=1)
        bearish = (signals_df < -0.3).sum(axis=1)
        total = bullish + bearish

        df['agg_signal'] = np.where(total > 0, (bullish - bearish) / total, 0.0)
        df['bullish_count'] = bullish
        df['bearish_count'] = bearish
        df['agg_signal'] = df['agg_signal'].clip(-1, 1)

        return df
