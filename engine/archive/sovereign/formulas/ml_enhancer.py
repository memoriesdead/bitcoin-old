"""
ML Enhancer Module
==================

Unified ML component that integrates:
- QLib alpha expressions (70001-70005)
- LightGBM flow classifier (70006)
- Online learning (70010)
- FinRL position sizing (71002)

This module enhances existing signals with ML predictions.

Formula IDs: 70009 (MLEnhancer ensemble)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

# Import QLib components
from .qlib_alpha.pit_handler import PointInTimeHandler, PITFlowDatabase
from .qlib_alpha.expression import (
    FlowMomentum, FlowAcceleration, FlowZScore, FlowSkew, FlowAutoCorr,
    FlowRegimeDetector, create_alpha_features, alpha_to_signal
)
from .qlib_alpha.lightgbm_flow import LightGBMFlowClassifier, OnlineLightGBM
from .qlib_alpha.online_learner import OnlineLearner, IncrementalUpdater

# Import FinRL components
from .finrl_rl.trading_env import TradingEnvironment, TradingState
from .finrl_rl.sac_sizer import SACPositionSizer


@dataclass
class MLSignal:
    """Enhanced signal from ML components."""
    # Base signal
    direction: int           # 1, -1, or 0
    confidence: float        # 0-1

    # Alpha components
    alpha_momentum: float
    alpha_zscore: float
    alpha_regime: float

    # ML prediction
    ml_probability: float    # P(price up)
    ml_confidence: float

    # RL position size
    rl_position: float       # Optimal position -1 to 1

    # Metadata
    timestamp: float
    ensemble_agreement: float


class MLEnhancer:
    """
    ML Enhancement Engine (Formula 70009).

    Enhances trading signals with:
    1. Alpha expressions from flow data
    2. LightGBM classification
    3. Online learning adaptation
    4. RL position sizing
    """

    formula_id = 70009
    name = "MLEnhancer"

    def __init__(self,
                 enable_alpha: bool = True,
                 enable_lightgbm: bool = True,
                 enable_rl: bool = True,
                 enable_online_learning: bool = True):
        """
        Initialize ML enhancer.

        Args:
            enable_alpha: Use alpha expressions
            enable_lightgbm: Use LightGBM classifier
            enable_rl: Use RL position sizer
            enable_online_learning: Enable online adaptation
        """
        self.enable_alpha = enable_alpha
        self.enable_lightgbm = enable_lightgbm
        self.enable_rl = enable_rl
        self.enable_online_learning = enable_online_learning

        # Point-in-time handler
        self.pit_handler = PointInTimeHandler(max_records=10000)

        # Alpha expressions
        self.alpha_momentum = FlowMomentum(window=10)
        self.alpha_acceleration = FlowAcceleration(window=5)
        self.alpha_zscore = FlowZScore(window=20)
        self.alpha_skew = FlowSkew(window=30)
        self.alpha_autocorr = FlowAutoCorr(window=20)
        self.alpha_regime = FlowRegimeDetector(window=20)

        # LightGBM classifier
        self.lgbm = LightGBMFlowClassifier(
            threshold_long=0.55,
            threshold_short=0.45,
        )

        # Online learning wrapper
        if enable_online_learning:
            self.online_lgbm = OnlineLightGBM(
                buffer_size=10000,
                retrain_frequency=100,
            )
        else:
            self.online_lgbm = None

        # RL position sizer
        self.rl_sizer = SACPositionSizer(state_dim=11, action_dim=1)

        # Data buffers
        self.flow_buffer: List[float] = []
        self.price_buffer: List[float] = []
        self.signal_buffer: List[Dict] = []

        # Stats
        self.stats = {
            'signals_processed': 0,
            'alpha_signals': 0,
            'ml_signals': 0,
            'rl_adjustments': 0,
            'online_updates': 0,
        }

    def update(self, flow: float, price: float, timestamp: Optional[float] = None):
        """
        Update with new market data.

        Args:
            flow: Current blockchain flow signal
            price: Current price
            timestamp: Data timestamp (default: now)
        """
        if timestamp is None:
            timestamp = time.time()

        # Add to PIT handler
        self.pit_handler.add_record(
            timestamp=timestamp,
            data={'flow': flow, 'price': price},
            source='live'
        )

        # Update buffers
        self.flow_buffer.append(flow)
        self.price_buffer.append(price)

        # Keep limited history
        if len(self.flow_buffer) > 1000:
            self.flow_buffer = self.flow_buffer[-1000:]
        if len(self.price_buffer) > 1000:
            self.price_buffer = self.price_buffer[-1000:]

    def enhance_signal(self, base_signal: Dict, timestamp: Optional[float] = None) -> MLSignal:
        """
        Enhance a base signal with ML predictions.

        Args:
            base_signal: Original signal from formula engine
            timestamp: Current timestamp

        Returns:
            MLSignal with enhanced predictions
        """
        if timestamp is None:
            timestamp = time.time()

        self.stats['signals_processed'] += 1

        # Get flow data for alphas
        flow_array = np.array(self.flow_buffer[-100:]) if self.flow_buffer else np.zeros(20)
        price_array = np.array(self.price_buffer[-100:]) if self.price_buffer else np.zeros(20)

        # === ALPHA EXPRESSIONS ===
        alpha_momentum = 0.0
        alpha_zscore = 0.0
        alpha_regime = 0.0

        if self.enable_alpha and len(flow_array) >= 20:
            self.stats['alpha_signals'] += 1

            mom_result = self.alpha_momentum(flow_array, timestamp)
            alpha_momentum = mom_result.value

            zscore_result = self.alpha_zscore(flow_array, timestamp)
            alpha_zscore = zscore_result.value

            regime_result = self.alpha_regime(flow_array, timestamp)
            alpha_regime = regime_result.value

        # === LIGHTGBM PREDICTION ===
        ml_probability = 0.5
        ml_confidence = 0.0

        if self.enable_lightgbm and len(flow_array) >= 50:
            self.stats['ml_signals'] += 1

            if self.lgbm.is_trained:
                pred = self.lgbm.predict(flow_array, price_array)
                ml_probability = pred.probability
                ml_confidence = pred.confidence
            elif self.online_lgbm is not None:
                pred = self.online_lgbm.predict(flow_array[-10:])
                ml_probability = pred.probability
                ml_confidence = pred.confidence

        # === RL POSITION SIZING ===
        rl_position = 0.0

        if self.enable_rl:
            self.stats['rl_adjustments'] += 1

            # Build state for RL
            state = self._build_rl_state(
                base_signal, flow_array, price_array,
                alpha_zscore, alpha_regime
            )

            rl_position = self.rl_sizer.get_position_size(state, deterministic=True)

        # === COMBINE INTO FINAL SIGNAL ===
        # Direction from base signal, weighted by ML
        base_direction = base_signal.get('direction', 0)
        base_confidence = base_signal.get('confidence', 0.5)

        # ML adjustment
        ml_direction = 1 if ml_probability > 0.55 else (-1 if ml_probability < 0.45 else 0)

        # Alpha adjustment
        alpha_direction = 1 if alpha_regime > 0.3 else (-1 if alpha_regime < -0.3 else 0)

        # Ensemble agreement
        directions = [base_direction, ml_direction, alpha_direction]
        non_zero = [d for d in directions if d != 0]
        if non_zero:
            agreement = sum(1 for d in non_zero if d == non_zero[0]) / len(non_zero)
        else:
            agreement = 0.0

        # Final direction (majority vote with base as tiebreaker)
        direction_sum = base_direction + ml_direction * 0.5 + alpha_direction * 0.3
        final_direction = int(np.sign(direction_sum)) if abs(direction_sum) > 0.3 else 0

        # Final confidence
        final_confidence = min(1.0, base_confidence * (1 + agreement) / 2 + ml_confidence * 0.2)

        return MLSignal(
            direction=final_direction,
            confidence=final_confidence,
            alpha_momentum=alpha_momentum,
            alpha_zscore=alpha_zscore,
            alpha_regime=alpha_regime,
            ml_probability=ml_probability,
            ml_confidence=ml_confidence,
            rl_position=rl_position,
            timestamp=timestamp,
            ensemble_agreement=agreement,
        )

    def _build_rl_state(self, signal: Dict, flow_array: np.ndarray,
                        price_array: np.ndarray, zscore: float,
                        regime: float) -> np.ndarray:
        """Build state vector for RL agent."""
        # Price change
        if len(price_array) >= 2:
            price_change = (price_array[-1] - price_array[-2]) / price_array[-2]
        else:
            price_change = 0.0

        # Volatility
        if len(price_array) >= 20:
            returns = np.diff(price_array[-20:]) / price_array[-20:-1]
            volatility = np.std(returns)
        else:
            volatility = 0.01

        # Flow stats
        flow = flow_array[-1] if len(flow_array) > 0 else 0
        if len(flow_array) >= 5:
            flow_momentum = np.mean(flow_array[-3:]) - np.mean(flow_array[-10:-3])
        else:
            flow_momentum = 0.0

        # Get current position (default 0)
        position = signal.get('current_position', 0)

        return np.array([
            price_change,
            volatility,
            flow,
            flow_momentum,
            0.0001,  # spread placeholder
            position,
            0.0,     # unrealized_pnl
            1.0,     # cash_ratio
            0.0,     # drawdown
            12 / 24,  # hour
            3 / 7,    # day_of_week
        ], dtype=np.float32)

    def record_outcome(self, signal_timestamp: float, outcome: int,
                       pnl: float = 0.0):
        """
        Record trade outcome for online learning.

        Args:
            signal_timestamp: Timestamp of the signal
            outcome: 1 = profitable, 0 = not profitable
            pnl: Actual PnL
        """
        if not self.enable_online_learning or self.online_lgbm is None:
            return

        self.stats['online_updates'] += 1

        # Find the signal and flow data at that time
        # For simplicity, use most recent flow
        if len(self.flow_buffer) >= 10:
            flow = np.array(self.flow_buffer[-10:])
            price = self.price_buffer[-1] if self.price_buffer else 0

            self.online_lgbm.update(flow, price, outcome)

    def train_offline(self, flow_data: np.ndarray, labels: np.ndarray,
                      price_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train LightGBM on historical data.

        Args:
            flow_data: Historical flow data
            labels: Binary labels (1 = price up, 0 = down)
            price_data: Optional price data

        Returns:
            Training results
        """
        return self.lgbm.train(
            flow_data, labels, price_data,
            val_ratio=0.2,
            early_stopping_rounds=50,
        )

    def get_alpha_summary(self) -> Dict[str, float]:
        """Get summary of current alpha values."""
        if len(self.flow_buffer) < 20:
            return {'status': 'insufficient_data'}

        flow_array = np.array(self.flow_buffer[-100:])
        timestamp = time.time()

        return {
            'momentum': self.alpha_momentum(flow_array, timestamp).value,
            'acceleration': self.alpha_acceleration(flow_array, timestamp).value,
            'zscore': self.alpha_zscore(flow_array, timestamp).value,
            'skew': self.alpha_skew(flow_array, timestamp).value,
            'autocorr': self.alpha_autocorr(flow_array, timestamp).value,
            'regime': self.alpha_regime(flow_array, timestamp).value,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get enhancement statistics."""
        return {
            **self.stats,
            'buffer_size': len(self.flow_buffer),
            'lgbm_trained': self.lgbm.is_trained,
            'rl_trained': self.rl_sizer.is_trained,
            'pit_records': self.pit_handler.get_stats()['current_records'],
        }


# =============================================================================
# INTEGRATION WITH EXISTING ENGINE
# =============================================================================

def create_ml_enhancer(
    enable_alpha: bool = True,
    enable_lightgbm: bool = True,
    enable_rl: bool = True,
    enable_online_learning: bool = True,
) -> MLEnhancer:
    """
    Factory function to create ML enhancer.

    This is the main entry point for integration.
    """
    return MLEnhancer(
        enable_alpha=enable_alpha,
        enable_lightgbm=enable_lightgbm,
        enable_rl=enable_rl,
        enable_online_learning=enable_online_learning,
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("ML Enhancer Demo")
    print("=" * 50)

    enhancer = create_ml_enhancer()

    # Simulate market data
    np.random.seed(42)

    for i in range(100):
        price = 42000 + np.cumsum(np.random.randn(1) * 10)[0]
        flow = np.random.randn() * 0.5
        enhancer.update(flow, price)

    # Test enhancement
    base_signal = {
        'direction': 1,
        'confidence': 0.7,
        'btc_amount': 10.0,
    }

    enhanced = enhancer.enhance_signal(base_signal)

    print(f"\nBase Signal: direction={base_signal['direction']}, conf={base_signal['confidence']}")
    print(f"\nEnhanced Signal:")
    print(f"  Direction: {enhanced.direction}")
    print(f"  Confidence: {enhanced.confidence:.3f}")
    print(f"  Alpha Momentum: {enhanced.alpha_momentum:.3f}")
    print(f"  Alpha Z-Score: {enhanced.alpha_zscore:.3f}")
    print(f"  Alpha Regime: {enhanced.alpha_regime:.3f}")
    print(f"  ML Probability: {enhanced.ml_probability:.3f}")
    print(f"  RL Position: {enhanced.rl_position:.3f}")
    print(f"  Ensemble Agreement: {enhanced.ensemble_agreement:.3f}")

    print(f"\nAlpha Summary: {enhancer.get_alpha_summary()}")
    print(f"\nStats: {enhancer.get_stats()}")
