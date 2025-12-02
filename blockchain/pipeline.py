"""
================================================================================
BLOCKCHAIN TRADING PIPELINE - Academic Formula Integration (LAYER 2)
================================================================================

ARCHITECTURE REFERENCE: docs/BLOCKCHAIN_PIPELINE_ARCHITECTURE.md

POSITION IN PIPELINE:
    This is a LAYER 2 component - integrates LAYER 3 data sources with
    academic formulas to produce trading signals.

PIPELINE FLOW:
    1. TRUE PRICE from blockchain (mathematical_price.py)
    2. Market Microstructure signals (Kyle, VPIN, OFI, Microprice)
    3. On-Chain metrics (NVT, MVRV, SOPR, Hash Ribbon)
    4. Execution optimization (Almgren-Chriss, Avellaneda-Stoikov)
    5. Risk management (Kelly, HMM Regime)
    6. Master aggregation (Condorcet voting with confidence weighting)

FORMULA IDs USED:
    520-523: Microstructure (Kyle, VPIN, OFI, Microprice)
    530-533: On-Chain (NVT, MVRV, SOPR, Hash Ribbon)
    540-541: Execution (Almgren-Chriss, Avellaneda-Stoikov)
    550-551: Risk (Kelly, HMM Regime)
    552:     TRUE Price Deviation
    560:     Master Aggregator (Condorcet voting)

SIGNAL AGGREGATION:
    Uses Condorcet voting with confidence-weighted signals.
    Each formula contributes a vote (-1, 0, +1) with confidence (0-1).
    Final signal = weighted majority across all formulas.

NO EXCHANGE APIS - Pure blockchain data + mathematical formulas.
================================================================================
"""

import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque

# Local imports
from .mathematical_price import MathematicalPricer, MathematicalPrice
from formulas.blockchain_signals import (
    KyleLambdaBlockchain,
    VPINBlockchain,
    OrderFlowImbalanceBlockchain,
    MicropriceBlockchain,
    NVTRatioBlockchain,
    MVRVRatioBlockchain,
    SOPRBlockchain,
    HashRibbonBlockchain,
    AlmgrenChrissExecution,
    AvellanedaStoikovSpread,
    KellyCriterionBlockchain,
    HMMRegimeBlockchain,
    TruePriceDeviation,
    BlockchainSignalAggregator,
)


@dataclass
class PipelineSignal:
    """Output signal from the blockchain pipeline."""
    signal: int               # -1 (SHORT), 0 (NEUTRAL), +1 (LONG)
    confidence: float         # 0.0 to 1.0
    true_price: float         # Blockchain-derived TRUE price
    exchange_price: float     # Current market price
    deviation_pct: float      # (exchange - true) / true
    position_size: float      # Kelly-recommended size (0-1)
    regime: str               # 'bull', 'bear', 'neutral'
    execution_urgency: float  # 0-1, from Almgren-Chriss
    component_signals: Dict[str, Tuple[int, float]]  # Individual signals


class BlockchainTradingPipeline:
    """
    Complete trading pipeline integrating TRUE PRICE with academic formulas.

    Usage:
        pipeline = BlockchainTradingPipeline(energy_cost_kwh=0.044)
        signal = pipeline.process(price=97000, volume=1000)
        if signal.signal == 1 and signal.confidence > 0.7:
            execute_long(size=signal.position_size)
    """

    def __init__(
        self,
        energy_cost_kwh: float = 0.044,
        lookback: int = 100,
        min_confidence: float = 0.5,
    ):
        """
        Initialize the blockchain trading pipeline.

        Args:
            energy_cost_kwh: Energy cost for TRUE price calculation
            lookback: Lookback period for formula calculations
            min_confidence: Minimum confidence to generate signal
        """
        # TRUE price calculator
        self.pricer = MathematicalPricer(energy_cost_kwh=energy_cost_kwh)

        # Initialize all formula components
        self.aggregator = BlockchainSignalAggregator(lookback=lookback)

        # Additional standalone formulas for direct access
        self.kyle = KyleLambdaBlockchain(lookback=lookback)
        self.vpin = VPINBlockchain(lookback=lookback)
        self.ofi = OrderFlowImbalanceBlockchain(lookback=lookback)
        self.microprice = MicropriceBlockchain(lookback=lookback)

        self.nvt = NVTRatioBlockchain(lookback=lookback)
        self.mvrv = MVRVRatioBlockchain(lookback=lookback)
        self.sopr = SOPRBlockchain(lookback=lookback)
        self.hash_ribbon = HashRibbonBlockchain(lookback=lookback)

        self.almgren = AlmgrenChrissExecution(lookback=lookback)
        self.avellaneda = AvellanedaStoikovSpread(lookback=lookback)

        self.kelly = KellyCriterionBlockchain(lookback=lookback)
        self.hmm = HMMRegimeBlockchain(lookback=lookback)

        self.true_price_deviation = TruePriceDeviation(lookback=lookback)

        # Pipeline state
        self.min_confidence = min_confidence
        self.current_true_price = 0.0
        self.last_signal = PipelineSignal(
            signal=0, confidence=0, true_price=0, exchange_price=0,
            deviation_pct=0, position_size=0, regime='neutral',
            execution_urgency=0.5, component_signals={}
        )
        self.signal_history = deque(maxlen=1000)

    def update_true_price(self) -> float:
        """
        Refresh TRUE price from blockchain data.

        Returns:
            Current TRUE price in USD
        """
        price_data = self.pricer.get_price()
        self.current_true_price = price_data.derived_price

        # Update deviation calculator
        self.true_price_deviation.set_true_price(self.current_true_price)
        self.aggregator.set_true_price(self.current_true_price)

        return self.current_true_price

    def process(
        self,
        price: float,
        volume: float = 0.0,
        timestamp: float = None,
        hash_rate: float = None,
    ) -> PipelineSignal:
        """
        Process new price/volume data through the full pipeline.

        Args:
            price: Current market price (from any source)
            volume: Current volume
            timestamp: Unix timestamp (defaults to now)
            hash_rate: Optional hash rate for Hash Ribbon

        Returns:
            PipelineSignal with trading recommendation
        """
        if timestamp is None:
            timestamp = time.time()

        # 1. Update TRUE price if stale (every 300 seconds to avoid blocking)
        # Note: The HTTP call is synchronous and blocks ~20-30 seconds
        if timestamp - getattr(self, '_last_true_update', 0) > 300:
            self.update_true_price()
            self._last_true_update = timestamp

        # 2. Update all formulas with new data
        self.aggregator.update(price, volume, timestamp)

        # Update standalone formulas for direct access
        for formula in [
            self.kyle, self.vpin, self.ofi, self.microprice,
            self.nvt, self.mvrv, self.sopr,
            self.almgren, self.avellaneda,
            self.kelly, self.hmm, self.true_price_deviation
        ]:
            formula.update(price, volume, timestamp)

        # Hash ribbon needs special update
        if hash_rate is not None:
            self.hash_ribbon.update_hash_rate(hash_rate)
        else:
            self.hash_ribbon.update(price, volume, timestamp)

        # 3. Get aggregated signal
        signal = self.aggregator.get_signal()
        confidence = self.aggregator.get_confidence()

        # 4. Calculate deviation from TRUE price
        deviation_pct = 0.0
        if self.current_true_price > 0:
            deviation_pct = (price - self.current_true_price) / self.current_true_price

        # 5. Get regime from HMM
        regime_map = {1: 'bull', -1: 'bear', 0: 'neutral'}
        regime = regime_map.get(self.hmm.current_regime, 'neutral')

        # 6. Get position size from Kelly
        position_size = self.kelly.kelly_f if self.kelly.is_ready else 0.1

        # 7. Get execution urgency from Almgren-Chriss
        execution_urgency = self.almgren.execution_rate if self.almgren.is_ready else 0.5

        # 8. Get component signals for transparency
        component_signals = self.aggregator.get_component_signals()

        # 9. Apply minimum confidence filter
        if confidence < self.min_confidence:
            signal = 0

        # Build output signal
        output = PipelineSignal(
            signal=signal,
            confidence=confidence,
            true_price=self.current_true_price,
            exchange_price=price,
            deviation_pct=deviation_pct,
            position_size=min(0.25, position_size),  # Cap at 25%
            regime=regime,
            execution_urgency=execution_urgency,
            component_signals=component_signals,
        )

        self.last_signal = output
        self.signal_history.append((timestamp, output))

        return output

    def get_edge_analysis(self) -> Dict[str, Any]:
        """
        Analyze current trading edge from all components.

        Returns:
            Dictionary with edge metrics for each component
        """
        analysis = {
            'true_price': self.current_true_price,
            'components': {},
        }

        # Microstructure
        if self.kyle.is_ready:
            analysis['components']['kyle_lambda'] = {
                'signal': self.kyle.get_signal(),
                'confidence': self.kyle.get_confidence(),
                'interpretation': 'liquidity' if self.kyle.get_signal() > 0 else 'illiquid'
            }

        if self.vpin.is_ready:
            analysis['components']['vpin'] = {
                'signal': self.vpin.get_signal(),
                'confidence': self.vpin.get_confidence(),
                'interpretation': 'toxic' if self.vpin.get_signal() < 0 else 'normal'
            }

        # On-chain
        if self.nvt.is_ready:
            analysis['components']['nvt'] = {
                'signal': self.nvt.get_signal(),
                'confidence': self.nvt.get_confidence(),
                'interpretation': 'overvalued' if self.nvt.get_signal() < 0 else 'undervalued'
            }

        if self.mvrv.is_ready:
            analysis['components']['mvrv'] = {
                'signal': self.mvrv.get_signal(),
                'confidence': self.mvrv.get_confidence(),
                'interpretation': 'overvalued' if self.mvrv.get_signal() < 0 else 'undervalued'
            }

        # Regime
        analysis['regime'] = {
            'current': {1: 'bull', -1: 'bear', 0: 'neutral'}.get(self.hmm.current_regime, 'neutral'),
            'probabilities': {
                'bear': self.hmm.regime_probs[0],
                'neutral': self.hmm.regime_probs[1],
                'bull': self.hmm.regime_probs[2],
            }
        }

        # Risk
        analysis['position_sizing'] = {
            'kelly_fraction': self.kelly.kelly_f,
            'recommended_size': min(0.25, self.kelly.kelly_f * 0.25),  # 25% of Kelly
        }

        return analysis

    def backtest_signal(self, prices: List[float], volumes: List[float] = None) -> Dict[str, Any]:
        """
        Run pipeline on historical data for backtesting.

        Args:
            prices: List of historical prices
            volumes: List of historical volumes (optional)

        Returns:
            Backtest results with signals and metrics
        """
        if volumes is None:
            volumes = [0.0] * len(prices)

        results = {
            'signals': [],
            'returns': [],
            'cumulative_return': 0.0,
            'win_rate': 0.0,
            'sharpe': 0.0,
        }

        position = 0
        entry_price = 0
        wins = 0
        trades = 0
        returns = []

        for i, (price, volume) in enumerate(zip(prices, volumes)):
            signal = self.process(price, volume, timestamp=i)
            results['signals'].append(signal)

            # Simple backtest logic
            if signal.signal != 0 and position == 0 and signal.confidence > 0.6:
                # Enter position
                position = signal.signal
                entry_price = price

            elif position != 0:
                # Check for exit
                pnl_pct = (price - entry_price) / entry_price * position

                # Exit on opposite signal or > 2% move
                if signal.signal == -position or abs(pnl_pct) > 0.02:
                    returns.append(pnl_pct)
                    if pnl_pct > 0:
                        wins += 1
                    trades += 1
                    position = 0

        results['returns'] = returns
        if returns:
            results['cumulative_return'] = np.sum(returns)
            results['win_rate'] = wins / trades if trades > 0 else 0
            results['sharpe'] = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 24 * 6)

        return results


def create_pipeline(energy_cost_kwh: float = 0.044) -> BlockchainTradingPipeline:
    """
    Factory function to create a configured pipeline.

    Args:
        energy_cost_kwh: Energy cost for TRUE price calculation

    Returns:
        Configured BlockchainTradingPipeline instance
    """
    return BlockchainTradingPipeline(
        energy_cost_kwh=energy_cost_kwh,
        lookback=100,
        min_confidence=0.5,
    )


# ============================================================================
# SIGNAL TYPES FOR TRADING ENGINE INTEGRATION
# ============================================================================

SIGNAL_LONG = 1
SIGNAL_SHORT = -1
SIGNAL_NEUTRAL = 0

# Formula categories
MICROSTRUCTURE_FORMULAS = [520, 521, 522, 523]  # Kyle, VPIN, OFI, Microprice
ONCHAIN_FORMULAS = [530, 531, 532, 533]         # NVT, MVRV, SOPR, Hash Ribbon
EXECUTION_FORMULAS = [540, 541]                  # Almgren-Chriss, Avellaneda-Stoikov
RISK_FORMULAS = [550, 551]                       # Kelly, HMM Regime
CORE_FORMULAS = [552, 560]                       # TRUE Price, Master Aggregator

ALL_BLOCKCHAIN_FORMULAS = (
    MICROSTRUCTURE_FORMULAS +
    ONCHAIN_FORMULAS +
    EXECUTION_FORMULAS +
    RISK_FORMULAS +
    CORE_FORMULAS
)
