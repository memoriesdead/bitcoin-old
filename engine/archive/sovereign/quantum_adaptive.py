#!/usr/bin/env python3
"""
QUANTUM-ADAPTIVE MULTI-TIMEFRAME TRADING SYSTEM
================================================

Solves the core problem: Market conditions change constantly.
What works for 1 second may not work for 2 seconds.

Solution: Quantum-inspired superposition of strategies across
multiple timeframes with regime-aware adaptation.

Key Components:
1. Wavelet Multi-Resolution Decomposition
2. Hidden Markov Model Regime Detection
3. Quantum-Inspired Strategy Superposition
4. Continuous Walk-Forward Adaptation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

# Optional imports with fallbacks
try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
    print("[WARN] PyWavelets not installed. Using simple decomposition.")

try:
    from hmmlearn import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("[WARN] hmmlearn not installed. Using rule-based regime detection.")


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class MarketRegime(Enum):
    BULL_CALM = 0      # Low vol uptrend - ride the trend
    BULL_EUPHORIA = 1  # High vol uptrend - quick scalps
    BEAR_GRINDING = 2  # Low vol downtrend - patient shorts
    BEAR_CRISIS = 3    # High vol downtrend - stay out


class Timeframe(Enum):
    NOISE = 1          # ~2 candles - HFT noise
    SCALPING = 2       # ~4 candles - quick trades
    INTRADAY = 3       # ~8 candles - day trading
    SWING = 4          # ~16 candles - multi-day
    POSITION = 5       # ~32 candles - weeks


@dataclass
class TimeframeSignal:
    timeframe: Timeframe
    direction: int  # 1=long, -1=short, 0=neutral
    strength: float  # 0-1
    confidence: float  # 0-1


@dataclass
class QuantumState:
    """Represents the quantum-inspired superposition of strategies."""
    amplitudes: np.ndarray  # Complex amplitudes for each strategy
    probabilities: np.ndarray  # |amplitude|^2
    phase: np.ndarray  # Phase angles (for interference)


# =============================================================================
# LAYER 1: WAVELET MULTI-RESOLUTION DECOMPOSITION
# =============================================================================

class WaveletDecomposer:
    """
    Decomposes price/feature series into multiple timeframe components.

    Uses Discrete Wavelet Transform (DWT) to separate:
    - Long-term trend (approximation)
    - Medium-term cycles (details at various scales)
    - Short-term noise (highest frequency details)
    """

    def __init__(self, wavelet: str = 'db4', levels: int = 5):
        self.wavelet = wavelet
        self.levels = levels

    def decompose(self, series: np.ndarray) -> Dict[Timeframe, np.ndarray]:
        """
        Decompose series into timeframe components.

        Returns dict mapping Timeframe to coefficient array.
        """
        if len(series) < 2 ** self.levels:
            # Not enough data for full decomposition
            return self._simple_decompose(series)

        if WAVELET_AVAILABLE:
            return self._wavelet_decompose(series)
        else:
            return self._simple_decompose(series)

    def _wavelet_decompose(self, series: np.ndarray) -> Dict[Timeframe, np.ndarray]:
        """Full wavelet decomposition using PyWavelets."""
        coeffs = pywt.wavedec(series, self.wavelet, level=self.levels)

        # Map coefficients to timeframes
        # coeffs[0] = approximation (lowest frequency = trend)
        # coeffs[1] = highest level detail (position)
        # coeffs[-1] = lowest level detail (noise)
        result = {}

        if len(coeffs) >= 6:
            result[Timeframe.POSITION] = coeffs[0]  # Trend/position
            result[Timeframe.SWING] = coeffs[1]
            result[Timeframe.INTRADAY] = coeffs[2]
            result[Timeframe.SCALPING] = coeffs[3]
            result[Timeframe.NOISE] = coeffs[4]
        else:
            # Fewer levels available
            for i, tf in enumerate([Timeframe.POSITION, Timeframe.SWING,
                                   Timeframe.INTRADAY, Timeframe.SCALPING]):
                if i < len(coeffs):
                    result[tf] = coeffs[i]

        return result

    def _simple_decompose(self, series: np.ndarray) -> Dict[Timeframe, np.ndarray]:
        """Simple moving average based decomposition (fallback)."""
        result = {}

        # Use different MA periods to approximate different timeframes
        for tf, period in [(Timeframe.POSITION, 32),
                          (Timeframe.SWING, 16),
                          (Timeframe.INTRADAY, 8),
                          (Timeframe.SCALPING, 4),
                          (Timeframe.NOISE, 2)]:
            if len(series) >= period:
                ma = np.convolve(series, np.ones(period)/period, mode='valid')
                # Detail = series - smoothed (captures that frequency band)
                padded_ma = np.pad(ma, (period-1, 0), mode='edge')
                result[tf] = series - padded_ma
            else:
                result[tf] = np.zeros(1)

        return result

    def reconstruct(self, components: Dict[Timeframe, np.ndarray],
                   timeframes: List[Timeframe]) -> np.ndarray:
        """Reconstruct signal from selected timeframe components."""
        if WAVELET_AVAILABLE:
            # Use inverse DWT
            coeffs = [components.get(tf, np.zeros(1))
                     for tf in [Timeframe.POSITION, Timeframe.SWING,
                               Timeframe.INTRADAY, Timeframe.SCALPING, Timeframe.NOISE]]
            return pywt.waverec(coeffs, self.wavelet)
        else:
            # Simple sum of components
            return sum(components.get(tf, np.zeros(1)) for tf in timeframes)

    def get_alignment_score(self, components: Dict[Timeframe, np.ndarray]) -> float:
        """
        Calculate how aligned different timeframes are.

        High alignment = all timeframes pointing same direction = strong signal
        Low alignment = timeframes conflict = uncertain/noisy
        """
        directions = []
        for tf, coeffs in components.items():
            if len(coeffs) > 0:
                # Direction based on recent trend of coefficients
                if len(coeffs) >= 2:
                    direction = np.sign(coeffs[-1] - coeffs[-2])
                else:
                    direction = np.sign(coeffs[-1])
                directions.append(direction)

        if not directions:
            return 0.0

        # Alignment = how much they agree
        # All same sign = 1.0, all different = 0.0
        mean_dir = np.mean(directions)
        alignment = abs(mean_dir)  # 0 to 1

        return alignment


# =============================================================================
# LAYER 2: HIDDEN MARKOV MODEL REGIME DETECTION
# =============================================================================

class RegimeDetector:
    """
    Detects current market regime using Hidden Markov Model.

    Regimes:
    - BULL_CALM: Low volatility uptrend
    - BULL_EUPHORIA: High volatility uptrend
    - BEAR_GRINDING: Low volatility downtrend
    - BEAR_CRISIS: High volatility downtrend
    """

    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = None
        self.is_trained = False

        # Fallback thresholds for rule-based detection
        self.vol_threshold = 0.02  # 2% daily vol
        self.trend_threshold = 0.001  # 0.1% daily return

    def fit(self, features: np.ndarray) -> None:
        """
        Train HMM on historical features.

        Features should include:
        - returns
        - volatility
        - volume changes
        - correlation changes
        """
        if HMM_AVAILABLE and len(features) >= 100:
            self.model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type='full',
                n_iter=100,
                random_state=42
            )
            self.model.fit(features)
            self.is_trained = True
        else:
            self.is_trained = False

    def predict(self, features: np.ndarray) -> Tuple[MarketRegime, np.ndarray]:
        """
        Predict current regime and probability distribution.

        Returns:
            (current_regime, regime_probabilities)
        """
        if self.is_trained and self.model is not None:
            return self._hmm_predict(features)
        else:
            return self._rule_based_predict(features)

    def _hmm_predict(self, features: np.ndarray) -> Tuple[MarketRegime, np.ndarray]:
        """HMM-based regime prediction."""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        regime_idx = self.model.predict(features[-1:])
        probs = self.model.predict_proba(features[-1:])[0]

        return MarketRegime(regime_idx[0]), probs

    def _rule_based_predict(self, features: np.ndarray) -> Tuple[MarketRegime, np.ndarray]:
        """Rule-based regime detection (fallback)."""
        if len(features) < 2:
            return MarketRegime.BULL_CALM, np.array([0.7, 0.1, 0.1, 0.1])

        # Assume features[-1] contains [return, volatility, ...]
        if len(features.shape) == 1:
            recent_return = features[-1]
            recent_vol = abs(recent_return)
        else:
            recent_return = features[-1, 0] if features.shape[1] > 0 else 0
            recent_vol = features[-1, 1] if features.shape[1] > 1 else abs(recent_return)

        # Classify regime
        is_bull = recent_return > self.trend_threshold
        is_high_vol = recent_vol > self.vol_threshold

        if is_bull and not is_high_vol:
            regime = MarketRegime.BULL_CALM
            probs = np.array([0.7, 0.15, 0.1, 0.05])
        elif is_bull and is_high_vol:
            regime = MarketRegime.BULL_EUPHORIA
            probs = np.array([0.15, 0.7, 0.1, 0.05])
        elif not is_bull and not is_high_vol:
            regime = MarketRegime.BEAR_GRINDING
            probs = np.array([0.1, 0.1, 0.7, 0.1])
        else:
            regime = MarketRegime.BEAR_CRISIS
            probs = np.array([0.05, 0.1, 0.15, 0.7])

        return regime, probs


# =============================================================================
# LAYER 3: QUANTUM-INSPIRED STRATEGY SUPERPOSITION
# =============================================================================

class QuantumStrategySelector:
    """
    Maintains multiple strategies in quantum-inspired superposition.

    Instead of selecting ONE strategy, all strategies exist simultaneously
    with complex amplitudes. The final action is the weighted combination.

    Key concepts:
    - Amplitude: √(probability) * e^(i*phase)
    - Interference: Strategies can constructively/destructively combine
    - Measurement: Collapse to weighted action at execution time
    """

    def __init__(self, n_strategies: int = 4):
        self.n_strategies = n_strategies

        # Initialize in equal superposition
        self.amplitudes = np.ones(n_strategies, dtype=complex) / np.sqrt(n_strategies)
        self.phases = np.zeros(n_strategies)

        # Strategy performance history for interference
        self.performance_history = [[] for _ in range(n_strategies)]

        # Strategy definitions (regime-specific params)
        self.strategies = [
            {'name': 'trend_follow', 'preferred_regime': MarketRegime.BULL_CALM,
             'timeframe': Timeframe.SWING, 'base_size': 1.0},
            {'name': 'scalp_momentum', 'preferred_regime': MarketRegime.BULL_EUPHORIA,
             'timeframe': Timeframe.SCALPING, 'base_size': 0.5},
            {'name': 'mean_revert', 'preferred_regime': MarketRegime.BEAR_GRINDING,
             'timeframe': Timeframe.INTRADAY, 'base_size': 0.7},
            {'name': 'defensive', 'preferred_regime': MarketRegime.BEAR_CRISIS,
             'timeframe': Timeframe.POSITION, 'base_size': 0.1},
        ]

    def update_state(self, regime_probs: np.ndarray,
                    alignment_score: float,
                    recent_performance: Optional[np.ndarray] = None) -> None:
        """
        Update quantum state based on market conditions.

        Args:
            regime_probs: Probability distribution over regimes
            alignment_score: Timeframe alignment (0-1)
            recent_performance: Recent Sharpe/return per strategy
        """
        # 1. Regime contribution to amplitude
        regime_weights = np.sqrt(regime_probs[:self.n_strategies])
        if len(regime_weights) < self.n_strategies:
            regime_weights = np.pad(regime_weights,
                                   (0, self.n_strategies - len(regime_weights)),
                                   constant_values=0.1)

        # 2. Alignment factor (constructive interference when aligned)
        alignment_factor = 1.0 + 0.5 * alignment_score

        # 3. Performance-based interference
        if recent_performance is not None and len(recent_performance) == self.n_strategies:
            # Positive performance = constructive interference
            # Negative performance = destructive interference
            interference = np.exp(recent_performance / 2)
        else:
            interference = np.ones(self.n_strategies)

        # 4. Update phases based on performance (phase encodes history)
        if recent_performance is not None:
            self.phases += recent_performance * 0.1  # Slow phase evolution

        # 5. Combine into complex amplitudes
        magnitudes = regime_weights * interference * alignment_factor
        self.amplitudes = magnitudes * np.exp(1j * self.phases)

        # 6. Normalize (total probability = 1)
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 0:
            self.amplitudes /= norm

    def get_probabilities(self) -> np.ndarray:
        """Get probability distribution (|amplitude|^2)."""
        return np.abs(self.amplitudes) ** 2

    def get_weighted_action(self, signals: List[TimeframeSignal]) -> Dict:
        """
        Collapse superposition to weighted action.

        Returns combined signal with:
        - direction: weighted direction
        - size: probability-weighted size
        - confidence: based on alignment and probability concentration
        """
        probs = self.get_probabilities()

        # Calculate weighted direction
        weighted_direction = 0.0
        weighted_size = 0.0

        for i, (prob, strategy) in enumerate(zip(probs, self.strategies)):
            # Find signal for this strategy's preferred timeframe
            tf_signal = next((s for s in signals if s.timeframe == strategy['timeframe']), None)

            if tf_signal:
                weighted_direction += prob * tf_signal.direction * tf_signal.strength
                weighted_size += prob * strategy['base_size']

        # Confidence based on probability concentration (entropy)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(self.n_strategies)
        concentration = 1.0 - (entropy / max_entropy)  # 0 = uniform, 1 = concentrated

        return {
            'direction': np.sign(weighted_direction),
            'strength': abs(weighted_direction),
            'size_multiplier': weighted_size,
            'confidence': concentration,
            'dominant_strategy': self.strategies[np.argmax(probs)]['name'],
            'probabilities': probs.tolist()
        }

    def record_performance(self, strategy_idx: int, pnl: float) -> None:
        """Record performance for interference calculation."""
        self.performance_history[strategy_idx].append(pnl)
        # Keep last 20 observations
        if len(self.performance_history[strategy_idx]) > 20:
            self.performance_history[strategy_idx] = self.performance_history[strategy_idx][-20:]


# =============================================================================
# LAYER 4: CONTINUOUS ADAPTATION
# =============================================================================

class AdaptiveParameterSystem:
    """
    Continuously adapts parameters using gradient-based optimization.

    Key insight: Parameters that worked yesterday may not work today.
    Solution: Continuously estimate performance gradient and adjust.
    """

    def __init__(self, base_params: Dict, learning_rate: float = 0.01):
        self.params = base_params.copy()
        self.learning_rate = learning_rate
        self.performance_history = []
        self.param_history = []

        # Constraints for each parameter
        self.constraints = {
            'take_profit': (0.005, 0.05),   # 0.5% to 5%
            'stop_loss': (0.001, 0.02),     # 0.1% to 2%
            'position_size': (0.1, 1.0),     # 10% to 100%
            'confidence_threshold': (0.5, 0.8),
        }

    def update(self, observation: Dict) -> Dict:
        """
        Update parameters based on new observation.

        observation should contain:
        - pnl: profit/loss of last trade
        - volatility: current market volatility
        - regime: current market regime
        """
        self.performance_history.append(observation.get('pnl', 0))
        self.param_history.append(self.params.copy())

        if len(self.performance_history) < 5:
            return self.params

        # Estimate gradient using finite differences
        gradient = self._estimate_gradient()

        # Update parameters (gradient ascent on performance)
        for key in self.params:
            if key in gradient:
                self.params[key] += self.learning_rate * gradient[key]

        # Apply constraints
        self._apply_constraints()

        # Adapt learning rate based on volatility
        vol = observation.get('volatility', 0.02)
        self.learning_rate = 0.01 * np.sqrt(0.02 / (vol + 0.001))

        return self.params

    def _estimate_gradient(self) -> Dict:
        """Estimate performance gradient w.r.t. parameters."""
        if len(self.param_history) < 3:
            return {}

        gradient = {}

        for key in self.params:
            if key not in self.param_history[-1]:
                continue

            # Collect (param_value, performance) pairs
            param_values = [h.get(key, 0) for h in self.param_history[-10:]]
            performances = self.performance_history[-10:]

            if len(param_values) < 3 or np.std(param_values) < 1e-10:
                gradient[key] = 0
                continue

            # Simple linear regression to estimate gradient
            param_mean = np.mean(param_values)
            perf_mean = np.mean(performances)

            numerator = sum((p - param_mean) * (r - perf_mean)
                          for p, r in zip(param_values, performances))
            denominator = sum((p - param_mean) ** 2 for p in param_values)

            if denominator > 0:
                gradient[key] = numerator / denominator
            else:
                gradient[key] = 0

        return gradient

    def _apply_constraints(self) -> None:
        """Ensure parameters stay within valid ranges."""
        for key, (min_val, max_val) in self.constraints.items():
            if key in self.params:
                self.params[key] = np.clip(self.params[key], min_val, max_val)


# =============================================================================
# MAIN QUANTUM-ADAPTIVE SYSTEM
# =============================================================================

class QuantumAdaptiveSystem:
    """
    Complete quantum-adaptive trading system.

    Combines:
    1. Wavelet decomposition for multi-timeframe analysis
    2. HMM regime detection
    3. Quantum-inspired strategy superposition
    4. Continuous parameter adaptation
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Initialize components
        self.decomposer = WaveletDecomposer(
            wavelet=self.config.get('wavelet', 'db4'),
            levels=self.config.get('levels', 5)
        )

        self.regime_detector = RegimeDetector(
            n_regimes=self.config.get('n_regimes', 4)
        )

        self.strategy_selector = QuantumStrategySelector(
            n_strategies=self.config.get('n_strategies', 4)
        )

        self.adaptive_params = AdaptiveParameterSystem(
            base_params={
                'take_profit': 0.01,
                'stop_loss': 0.003,
                'position_size': 1.0,
                'confidence_threshold': 0.52
            }
        )

        # State
        self.price_history = []
        self.feature_history = []
        self.signal_history = []

    def process(self, price: float, features: Dict) -> Dict:
        """
        Process new market data and generate trading signal.

        Args:
            price: Current price
            features: Dict of blockchain/market features

        Returns:
            Trading signal with direction, size, confidence
        """
        # Update history
        self.price_history.append(price)
        self.feature_history.append(features)

        # Keep reasonable history size
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-1000:]
            self.feature_history = self.feature_history[-1000:]

        # Need minimum history
        if len(self.price_history) < 32:
            return {'action': 'wait', 'reason': 'insufficient_history'}

        prices = np.array(self.price_history)

        # 1. Wavelet decomposition
        components = self.decomposer.decompose(prices)
        alignment = self.decomposer.get_alignment_score(components)

        # 2. Generate signals at each timeframe
        tf_signals = self._generate_timeframe_signals(components)

        # 3. Detect regime
        feature_matrix = self._build_feature_matrix()
        regime, regime_probs = self.regime_detector.predict(feature_matrix)

        # 4. Update quantum state
        recent_perf = self._get_recent_performance()
        self.strategy_selector.update_state(regime_probs, alignment, recent_perf)

        # 5. Get weighted action
        action = self.strategy_selector.get_weighted_action(tf_signals)

        # 6. Apply adaptive parameters
        current_params = self.adaptive_params.params

        # Build final signal
        signal = {
            'timestamp': features.get('timestamp', 0),
            'price': price,
            'regime': regime.name,
            'regime_probs': regime_probs.tolist(),
            'alignment': alignment,
            'direction': action['direction'],
            'strength': action['strength'],
            'size_multiplier': action['size_multiplier'],
            'confidence': action['confidence'],
            'dominant_strategy': action['dominant_strategy'],
            'strategy_probs': action['probabilities'],
            'params': current_params,
            'timeframe_signals': [
                {'tf': s.timeframe.name, 'dir': s.direction, 'str': s.strength}
                for s in tf_signals
            ]
        }

        # Determine action
        if action['confidence'] >= current_params['confidence_threshold']:
            if action['direction'] > 0 and action['strength'] > 0.3:
                signal['action'] = 'long'
            elif action['direction'] < 0 and action['strength'] > 0.3:
                signal['action'] = 'short'
            else:
                signal['action'] = 'hold'
        else:
            signal['action'] = 'wait'
            signal['reason'] = 'low_confidence'

        self.signal_history.append(signal)
        return signal

    def _generate_timeframe_signals(self, components: Dict[Timeframe, np.ndarray]) -> List[TimeframeSignal]:
        """Generate trading signals at each timeframe."""
        signals = []

        for tf, coeffs in components.items():
            if len(coeffs) < 2:
                continue

            # Direction: recent trend of coefficients
            recent_trend = coeffs[-1] - coeffs[-2] if len(coeffs) >= 2 else 0
            direction = np.sign(recent_trend)

            # Strength: magnitude relative to historical range
            coeff_range = np.max(np.abs(coeffs)) if len(coeffs) > 0 else 1
            strength = min(1.0, abs(recent_trend) / (coeff_range + 1e-10))

            # Confidence: stability of direction
            if len(coeffs) >= 5:
                recent_dirs = np.sign(np.diff(coeffs[-5:]))
                confidence = abs(np.mean(recent_dirs))
            else:
                confidence = 0.5

            signals.append(TimeframeSignal(
                timeframe=tf,
                direction=int(direction),
                strength=float(strength),
                confidence=float(confidence)
            ))

        return signals

    def _build_feature_matrix(self) -> np.ndarray:
        """Build feature matrix for regime detection."""
        if len(self.price_history) < 20:
            return np.array([[0, 0]])

        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]

        # Features: [return, volatility, ...]
        features = []
        for i in range(max(0, len(returns) - 20), len(returns)):
            window = returns[max(0, i-19):i+1]
            feat = [
                returns[i],  # Current return
                np.std(window) if len(window) > 1 else 0,  # Rolling volatility
            ]
            features.append(feat)

        return np.array(features) if features else np.array([[0, 0]])

    def _get_recent_performance(self) -> Optional[np.ndarray]:
        """Get recent performance for each strategy."""
        # Placeholder: would track actual strategy-level performance
        return None

    def feedback(self, pnl: float, volatility: float = 0.02) -> None:
        """Provide feedback for adaptation."""
        self.adaptive_params.update({
            'pnl': pnl,
            'volatility': volatility,
            'regime': self.signal_history[-1]['regime'] if self.signal_history else 'unknown'
        })

    def get_state(self) -> Dict:
        """Get current system state for logging/debugging."""
        return {
            'price_history_len': len(self.price_history),
            'current_params': self.adaptive_params.params,
            'strategy_probabilities': self.strategy_selector.get_probabilities().tolist(),
            'recent_signals': self.signal_history[-5:] if self.signal_history else []
        }


# =============================================================================
# TESTING
# =============================================================================

def test_system():
    """Test the quantum-adaptive system with synthetic data."""
    print("=" * 60)
    print("QUANTUM-ADAPTIVE SYSTEM TEST")
    print("=" * 60)

    # Initialize system
    system = QuantumAdaptiveSystem()

    # Generate synthetic price data with regime changes
    np.random.seed(42)
    n_points = 200

    prices = [100000.0]
    for i in range(n_points - 1):
        # Regime changes
        if i < 50:
            drift = 0.001  # Bull calm
            vol = 0.005
        elif i < 100:
            drift = 0.002  # Bull euphoria
            vol = 0.015
        elif i < 150:
            drift = -0.001  # Bear grinding
            vol = 0.007
        else:
            drift = -0.002  # Bear crisis
            vol = 0.02

        ret = drift + vol * np.random.randn()
        prices.append(prices[-1] * (1 + ret))

    # Process data
    print("\nProcessing price data...")
    for i, price in enumerate(prices):
        features = {
            'timestamp': i,
            'tx_count': 1000 + int(100 * np.random.randn()),
            'whale_tx_count': 50 + int(10 * np.random.randn()),
            'total_value_btc': 500 + 50 * np.random.randn()
        }

        signal = system.process(price, features)

        # Log every 20 points
        if i > 0 and i % 40 == 0:
            print(f"\nPoint {i}: Price=${price:,.0f}")
            print(f"  Regime: {signal.get('regime', 'N/A')}")
            print(f"  Alignment: {signal.get('alignment', 0):.2f}")
            print(f"  Action: {signal.get('action', 'N/A')}")
            print(f"  Confidence: {signal.get('confidence', 0):.2f}")
            print(f"  Strategy: {signal.get('dominant_strategy', 'N/A')}")
            print(f"  Probs: {[f'{p:.2f}' for p in signal.get('strategy_probs', [])]}")

    # Final state
    state = system.get_state()
    print("\n" + "=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    print(f"Data points processed: {state['price_history_len']}")
    print(f"Current params: {state['current_params']}")
    print(f"Strategy probs: {[f'{p:.2f}' for p in state['strategy_probabilities']]}")


if __name__ == '__main__':
    test_system()
