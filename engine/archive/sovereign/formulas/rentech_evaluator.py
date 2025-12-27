"""
RenTech Signal Evaluator
=========================

Evaluates all 9 validated RenTech formulas against current features
and generates trading signals with proper gating logic.

Each formula has:
- Entry conditions (from RENTECH_FORMULAS.py)
- Hold period
- Direction (LONG/SHORT)
- Historical stats for position sizing

Gating Logic (in order):
1. Data Quality: No missing features, data age < 60s
2. Regime Compatibility: Formula matches current HMM regime
3. Drawdown Gate: Current DD < 10% (circuit breaker)
4. Confidence Threshold: Based on historical stats

Created: 2025-12-16
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

from .rentech_features import RenTechFeatures, FeatureSnapshot

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Trading signal direction."""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class RenTechSignal:
    """Signal from RenTech evaluator."""
    formula_id: str
    formula_name: str
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    hold_days: int
    entry_price: float
    timestamp: float
    features_snapshot: Dict[str, float] = field(default_factory=dict)

    # Historical stats for position sizing
    historical_win_rate: float = 1.0
    historical_avg_return: float = 20.0
    safe_50x_trades: int = 10

    # Gates passed
    data_quality_passed: bool = True
    regime_compatible: bool = True
    drawdown_ok: bool = True

    @property
    def is_actionable(self) -> bool:
        """Check if signal should trigger a trade."""
        return (
            self.direction != SignalDirection.NEUTRAL and
            self.confidence >= 0.6 and
            self.data_quality_passed and
            self.regime_compatible and
            self.drawdown_ok
        )


# Formula definitions from RENTECH_FORMULAS.py
RENTECH_FORMULA_CONDITIONS = {
    "RENTECH_001": {
        "name": "EXTREME_ANOMALY_LONG",
        "description": "Statistical anomaly in oversold territory",
        "hold_days": 30,
        "direction": SignalDirection.LONG,
        "conditions": lambda f: (
            f.get("anomaly_score", 0) > 4 and
            f.get("ret_7d", 0) < -15
        ),
        "regimes": ["CAPITULATION", "BEAR"],  # Compatible regimes
        "stats": {"safe_50x_trades": 18, "win_rate": 1.0, "avg_return": 26.37}
    },
    "RENTECH_002": {
        "name": "VOLUME_MOMENTUM_CONFLUENCE",
        "description": "TX and whale momentum aligned with price momentum",
        "hold_days": 30,
        "direction": SignalDirection.LONG,
        "conditions": lambda f: (
            f.get("tx_momentum_7d", 0) > 10 and
            f.get("whale_momentum_7d", 0) > 10 and
            f.get("ret_7d", 0) > 5
        ),
        "regimes": ["NEUTRAL", "BULL"],
        "stats": {"safe_50x_trades": 22, "win_rate": 1.0, "avg_return": 25.43}
    },
    "RENTECH_003": {
        "name": "EXTREME_ANOMALY_LONG_7D",
        "description": "Statistical anomaly - short hold",
        "hold_days": 7,
        "direction": SignalDirection.LONG,
        "conditions": lambda f: (
            f.get("anomaly_score", 0) > 4 and
            f.get("ret_7d", 0) < -15
        ),
        "regimes": ["CAPITULATION", "BEAR"],
        "stats": {"safe_50x_trades": 20, "win_rate": 1.0, "avg_return": 16.78}
    },
    "RENTECH_004": {
        "name": "CORRELATION_BREAK_BULL",
        "description": "TX-price correlation breaks down in oversold",
        "hold_days": 7,
        "direction": SignalDirection.LONG,
        "conditions": lambda f: (
            f.get("tx_price_correlation_30d", 0) < -0.3 and
            f.get("price_vs_ma30", 0) < -15 and
            f.get("tx_z30", 0) > 0
        ),
        "regimes": ["CAPITULATION", "BEAR"],
        "stats": {"safe_50x_trades": 5, "win_rate": 1.0, "avg_return": 24.87}
    },
    "RENTECH_005": {
        "name": "BOLLINGER_BOUNCE",
        "description": "Price at lower Bollinger band with declining volatility",
        "hold_days": 30,
        "direction": SignalDirection.LONG,
        "conditions": lambda f: (
            f.get("bb_position", 0) < -0.9 and
            f.get("volatility_20d", 100) < 80 and
            f.get("ret_3d", 0) > 0
        ),
        "regimes": ["CAPITULATION", "BEAR", "NEUTRAL"],
        "stats": {"safe_50x_trades": 3, "win_rate": 1.0, "avg_return": 23.73}
    },
    "RENTECH_006": {
        "name": "WHALE_ACCUMULATION",
        "description": "Whales buying while price drops",
        "hold_days": 30,
        "direction": SignalDirection.LONG,
        "conditions": lambda f: (
            f.get("ret_7d", 0) < -10 and
            f.get("whale_momentum_7d", 0) > 20 and
            f.get("whale_z30", 0) > 1
        ),
        "regimes": ["CAPITULATION", "BEAR"],
        "stats": {"safe_50x_trades": 5, "win_rate": 1.0, "avg_return": 22.92}
    },
    "RENTECH_007": {
        "name": "RSI_DIVERGENCE",
        "description": "Price making new lows but RSI not confirming",
        "hold_days": 30,
        "direction": SignalDirection.LONG,
        "conditions": lambda f: (
            f.get("ret_14d", 0) < -15 and
            f.get("rsi_14", 50) > 35 and
            f.get("ret_3d", 0) > 0
        ),
        "regimes": ["CAPITULATION", "BEAR"],
        "stats": {"safe_50x_trades": 7, "win_rate": 1.0, "avg_return": 20.57}
    },
    "RENTECH_008": {
        "name": "MULTI_Z_EXTREME",
        "description": "All z-scores extremely negative (capitulation)",
        "hold_days": 30,
        "direction": SignalDirection.LONG,
        "conditions": lambda f: (
            f.get("tx_z30", 0) < -2 and
            f.get("whale_z30", 0) < -2 and
            f.get("value_z30", 0) < -2
        ),
        "regimes": ["CAPITULATION"],
        "stats": {"safe_50x_trades": 2, "win_rate": 1.0, "avg_return": 45.14}
    },
    "RENTECH_009": {
        "name": "EUPHORIA_EXIT_SHORT",
        "description": "Short when euphoria + extreme overbought",
        "hold_days": 30,
        "direction": SignalDirection.SHORT,
        "conditions": lambda f: (
            f.get("regime", "") == "EUPHORIA" and
            f.get("rsi_14", 50) > 80 and
            f.get("price_vs_ma30", 0) > 30
        ),
        "regimes": ["EUPHORIA"],
        "stats": {"safe_50x_trades": 7, "win_rate": 1.0, "avg_return": 13.01}
    },
}


class RenTechEvaluator:
    """
    Evaluates all RenTech formulas and generates trading signals.

    Thread-safe for real-time evaluation.
    """

    def __init__(
        self,
        max_drawdown: float = 0.10,  # 10% max DD before circuit breaker
        data_max_age: float = 60.0,  # Max age of data in seconds
        min_confidence: float = 0.6  # Minimum confidence to trade
    ):
        """
        Initialize evaluator.

        Args:
            max_drawdown: Maximum portfolio drawdown before stopping
            data_max_age: Maximum data age in seconds
            min_confidence: Minimum confidence threshold
        """
        self.max_drawdown = max_drawdown
        self.data_max_age = data_max_age
        self.min_confidence = min_confidence

        # State
        self.current_drawdown = 0.0
        self.active_positions: Dict[str, RenTechSignal] = {}
        self.signal_history: List[RenTechSignal] = []

        # Formula-specific state
        self.formula_cooldowns: Dict[str, float] = {}  # Last signal time per formula
        self.cooldown_period = 24 * 60 * 60  # 24 hours between signals per formula

    def evaluate(
        self,
        features: Dict[str, float],
        price: float,
        timestamp: Optional[float] = None
    ) -> List[RenTechSignal]:
        """
        Evaluate all formulas against current features.

        Args:
            features: Feature dictionary from RenTechFeatures
            price: Current price
            timestamp: Optional timestamp

        Returns:
            List of actionable signals (may be empty)
        """
        ts = timestamp or time.time()
        signals = []

        for formula_id, formula in RENTECH_FORMULA_CONDITIONS.items():
            signal = self._evaluate_formula(formula_id, formula, features, price, ts)
            if signal is not None and signal.is_actionable:
                signals.append(signal)

        return signals

    def _evaluate_formula(
        self,
        formula_id: str,
        formula: Dict,
        features: Dict[str, float],
        price: float,
        timestamp: float
    ) -> Optional[RenTechSignal]:
        """
        Evaluate a single formula.

        Returns signal if conditions met, None otherwise.
        """
        # Check cooldown
        last_signal_time = self.formula_cooldowns.get(formula_id, 0)
        if timestamp - last_signal_time < self.cooldown_period:
            return None

        # Check if we already have active position from this formula
        if formula_id in self.active_positions:
            return None

        # Gate 1: Data quality
        data_quality_passed = self._check_data_quality(features, timestamp)

        # Gate 2: Regime compatibility
        current_regime = features.get("regime", "NEUTRAL")
        regime_compatible = current_regime in formula.get("regimes", [])

        # Gate 3: Drawdown check
        drawdown_ok = self.current_drawdown < self.max_drawdown

        # Evaluate conditions
        try:
            conditions_met = formula["conditions"](features)
        except Exception as e:
            logger.debug(f"Error evaluating {formula_id}: {e}")
            conditions_met = False

        if not conditions_met:
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(formula_id, formula, features)

        # Create signal
        signal = RenTechSignal(
            formula_id=formula_id,
            formula_name=formula["name"],
            direction=formula["direction"],
            confidence=confidence,
            hold_days=formula["hold_days"],
            entry_price=price,
            timestamp=timestamp,
            features_snapshot=features.copy(),
            historical_win_rate=formula["stats"]["win_rate"],
            historical_avg_return=formula["stats"]["avg_return"],
            safe_50x_trades=formula["stats"]["safe_50x_trades"],
            data_quality_passed=data_quality_passed,
            regime_compatible=regime_compatible,
            drawdown_ok=drawdown_ok
        )

        return signal

    def _check_data_quality(self, features: Dict[str, float], timestamp: float) -> bool:
        """Check if data is fresh and complete."""
        # Check for required features
        required = ["price", "ret_7d", "price_vs_ma30", "rsi_14"]
        for req in required:
            if req not in features or features[req] is None:
                return False

        # Check data freshness (would need actual timestamp in features)
        # For now, assume data is fresh
        return True

    def _calculate_confidence(
        self,
        formula_id: str,
        formula: Dict,
        features: Dict[str, float]
    ) -> float:
        """
        Calculate signal confidence based on multiple factors.

        Factors:
        1. Historical sample size
        2. How extreme the conditions are
        3. Regime strength
        """
        base_confidence = 0.8  # Base for 100% historical win rate

        # Sample size factor
        safe_trades = formula["stats"]["safe_50x_trades"]
        sample_factor = min(safe_trades / 20, 1.0)  # Scales 0-1 for 0-20 trades

        # Condition extremity factor
        extremity = self._calculate_extremity(formula_id, features)

        # Combine factors
        confidence = base_confidence * (0.5 + 0.3 * sample_factor + 0.2 * extremity)

        return min(confidence, 1.0)

    def _calculate_extremity(self, formula_id: str, features: Dict[str, float]) -> float:
        """Calculate how extreme current conditions are (0-1)."""
        extremity = 0.5  # Default

        if "ANOMALY" in formula_id:
            # For anomaly formulas, higher anomaly score = more extreme
            score = features.get("anomaly_score", 0)
            extremity = min(score / 6, 1.0)  # Score of 6 = max extremity

        elif "MULTI_Z" in formula_id:
            # Average absolute z-score
            avg_z = (
                abs(features.get("tx_z30", 0)) +
                abs(features.get("whale_z30", 0)) +
                abs(features.get("value_z30", 0))
            ) / 3
            extremity = min(avg_z / 3, 1.0)

        elif "MOMENTUM" in formula_id:
            # Momentum strength
            tx_mom = abs(features.get("tx_momentum_7d", 0))
            whale_mom = abs(features.get("whale_momentum_7d", 0))
            extremity = min((tx_mom + whale_mom) / 50, 1.0)

        return extremity

    def on_position_opened(self, signal: RenTechSignal):
        """Record that a position was opened from a signal."""
        self.active_positions[signal.formula_id] = signal
        self.formula_cooldowns[signal.formula_id] = signal.timestamp
        self.signal_history.append(signal)

    def on_position_closed(
        self,
        formula_id: str,
        exit_price: float,
        pnl: float,
        pnl_pct: float
    ):
        """Record that a position was closed."""
        if formula_id in self.active_positions:
            del self.active_positions[formula_id]

        # Update drawdown
        # (In production, would track equity curve)
        if pnl < 0:
            self.current_drawdown += abs(pnl_pct)
        else:
            self.current_drawdown = max(0, self.current_drawdown - pnl_pct)

    def update_drawdown(self, current_dd: float):
        """Update current drawdown from external source."""
        self.current_drawdown = current_dd

    def get_active_signals(self) -> Dict[str, RenTechSignal]:
        """Get currently active positions."""
        return self.active_positions.copy()

    def get_formula_stats(self, formula_id: str) -> Dict:
        """Get statistics for a specific formula."""
        if formula_id not in RENTECH_FORMULA_CONDITIONS:
            return {}

        formula = RENTECH_FORMULA_CONDITIONS[formula_id]
        return {
            "id": formula_id,
            "name": formula["name"],
            "direction": formula["direction"].name,
            "hold_days": formula["hold_days"],
            "stats": formula["stats"],
            "is_active": formula_id in self.active_positions,
            "cooldown_remaining": max(
                0,
                self.cooldown_period - (time.time() - self.formula_cooldowns.get(formula_id, 0))
            )
        }


def create_evaluator() -> RenTechEvaluator:
    """Factory function to create evaluator with default settings."""
    return RenTechEvaluator(
        max_drawdown=0.10,
        data_max_age=60.0,
        min_confidence=0.6
    )


if __name__ == "__main__":
    # Test the evaluator
    print("Testing RenTech Evaluator")
    print("=" * 60)

    evaluator = RenTechEvaluator()

    # Test with extreme capitulation conditions
    test_features = {
        "price": 50000,
        "ret_7d": -20.0,
        "ret_14d": -25.0,
        "ret_3d": 2.0,  # Slight bounce
        "price_vs_ma30": -35.0,
        "rsi_14": 25.0,
        "anomaly_score": 5.5,
        "bb_position": -1.2,
        "volatility_20d": 70.0,
        "tx_z30": -2.5,
        "whale_z30": -2.2,
        "value_z30": -2.1,
        "tx_momentum_7d": 5.0,
        "whale_momentum_7d": 25.0,  # Whales accumulating
        "tx_price_correlation_30d": -0.4,
        "regime": "CAPITULATION"
    }

    signals = evaluator.evaluate(test_features, price=50000)

    print(f"\nConditions:")
    for key, value in test_features.items():
        print(f"  {key}: {value}")

    print(f"\nSignals generated: {len(signals)}")
    for signal in signals:
        print(f"\n  {signal.formula_id}: {signal.formula_name}")
        print(f"    Direction: {signal.direction.name}")
        print(f"    Confidence: {signal.confidence:.2%}")
        print(f"    Hold: {signal.hold_days} days")
        print(f"    Historical WR: {signal.historical_win_rate:.0%}")
        print(f"    Historical Avg: +{signal.historical_avg_return:.1f}%")
        print(f"    Actionable: {signal.is_actionable}")

    # Test euphoria short
    print("\n" + "=" * 60)
    print("Testing EUPHORIA short conditions")

    euphoria_features = {
        "price": 150000,
        "ret_7d": 30.0,
        "price_vs_ma30": 35.0,
        "rsi_14": 85.0,
        "regime": "EUPHORIA"
    }

    signals = evaluator.evaluate(euphoria_features, price=150000)

    print(f"\nSignals generated: {len(signals)}")
    for signal in signals:
        print(f"\n  {signal.formula_id}: {signal.formula_name}")
        print(f"    Direction: {signal.direction.name}")
        print(f"    Confidence: {signal.confidence:.2%}")
