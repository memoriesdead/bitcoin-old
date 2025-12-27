"""
Master Ensemble - Final Signal Combination
==========================================

Formula IDs: 72096-72099

The final layer that combines all signal sources:
- HMM regime signals (72001-72010)
- Signal processing (72011-72030)
- Non-linear detection (72031-72050)
- Micro-patterns (72051-72080)
- Sub-ensembles (72081-72095)

RenTech insight: The final signal should be conservative.
When models disagree, stay flat. When they agree, size up.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from collections import deque


@dataclass
class MasterSignal:
    """Final trading signal from master ensemble."""
    direction: int  # -1, 0, 1
    confidence: float  # 0-1
    position_size: float  # 0-1 (fraction of max position)
    signal_quality: str  # 'high', 'medium', 'low', 'no_trade'

    # Breakdown
    hmm_contribution: float
    signal_contribution: float
    nonlinear_contribution: float
    micro_contribution: float
    ensemble_contribution: float

    # Metadata
    active_signals: int
    agreement_ratio: float
    regime: str
    reasoning: List[str] = field(default_factory=list)


@dataclass
class SignalGroup:
    """Group of related signals."""
    name: str
    signals: Dict[str, Any]  # formula_id -> signal output
    weight: float  # Group importance weight


class MasterEnsemble:
    """
    Master ensemble combining all signal categories.

    Architecture:
    1. Collect signals from all categories
    2. Aggregate within categories
    3. Combine categories with learned/adaptive weights
    4. Apply position sizing rules
    5. Generate final trading decision
    """

    def __init__(self):
        # Category weights (can be learned)
        self.category_weights = {
            'hmm': 0.20,        # Regime detection
            'signal': 0.20,    # DTW, FFT, Wavelets
            'nonlinear': 0.15, # Kernel, Anomaly
            'micro': 0.25,     # Streaks, GARCH, Calendar, Whales
            'ensemble': 0.20,  # Sub-ensembles
        }

        # Performance tracking
        self.category_performance: Dict[str, deque] = {
            cat: deque(maxlen=100) for cat in self.category_weights
        }

        # Regime tracking
        self.current_regime = 'unknown'
        self.regime_history: deque = deque(maxlen=50)

        # Agreement thresholds
        self.min_agreement_to_trade = 0.3
        self.high_confidence_agreement = 0.7

    def categorize_signal(self, formula_id: int) -> str:
        """Map formula ID to category."""
        if 72001 <= formula_id <= 72010:
            return 'hmm'
        elif 72011 <= formula_id <= 72030:
            return 'signal'
        elif 72031 <= formula_id <= 72050:
            return 'nonlinear'
        elif 72051 <= formula_id <= 72080:
            return 'micro'
        elif 72081 <= formula_id <= 72095:
            return 'ensemble'
        else:
            return 'other'

    def aggregate_category(self, signals: Dict[str, Any]) -> Tuple[float, float, int]:
        """
        Aggregate signals within a category.

        Returns: (direction_score, confidence, active_count)
        """
        if not signals:
            return 0.0, 0.0, 0

        directions = []
        confidences = []

        for signal in signals.values():
            if hasattr(signal, 'direction'):
                d = signal.direction
                c = getattr(signal, 'confidence', 0.5)
            elif isinstance(signal, dict):
                d = signal.get('direction', 0)
                c = signal.get('confidence', 0.5)
            else:
                continue

            if d != 0:  # Only count active signals
                directions.append(d * c)  # Confidence-weighted direction
                confidences.append(c)

        if not directions:
            return 0.0, 0.0, 0

        # Direction score: sum of weighted directions / sum of confidences
        total_conf = sum(confidences)
        direction_score = sum(directions) / total_conf if total_conf > 0 else 0

        # Category confidence: average confidence * agreement
        avg_conf = np.mean(confidences)
        agreement = abs(sum(np.sign(directions))) / len(directions)
        category_conf = avg_conf * agreement

        return direction_score, category_conf, len(directions)

    def combine_categories(self, category_scores: Dict[str, Tuple[float, float, int]]) -> MasterSignal:
        """Combine all category scores into final signal."""
        weighted_direction = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        total_active = 0

        contributions = {}

        for category, (dir_score, conf, active) in category_scores.items():
            weight = self.category_weights.get(category, 0.1)

            weighted_direction += weight * dir_score
            weighted_confidence += weight * conf
            total_weight += weight
            total_active += active

            contributions[category] = dir_score * conf

        if total_weight > 0:
            final_direction_score = weighted_direction / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_direction_score = 0.0
            final_confidence = 0.0

        # Compute agreement across categories
        active_categories = [(d, c) for d, c, a in category_scores.values() if a > 0]
        if active_categories:
            directions = [np.sign(d) for d, c in active_categories if d != 0]
            if directions:
                agreement = abs(sum(directions)) / len(directions)
            else:
                agreement = 0.0
        else:
            agreement = 0.0

        # Determine final direction
        if abs(final_direction_score) < 0.1:
            direction = 0
        elif final_direction_score > 0:
            direction = 1
        else:
            direction = -1

        # Determine position size based on confidence and agreement
        if agreement < self.min_agreement_to_trade:
            position_size = 0.0
            signal_quality = 'no_trade'
            direction = 0
        elif agreement >= self.high_confidence_agreement and final_confidence > 0.6:
            position_size = min(1.0, final_confidence * agreement)
            signal_quality = 'high'
        elif final_confidence > 0.4:
            position_size = 0.5 * final_confidence * agreement
            signal_quality = 'medium'
        else:
            position_size = 0.25 * agreement
            signal_quality = 'low'

        # Build reasoning
        reasoning = []
        if direction != 0:
            top_contrib = max(contributions.items(), key=lambda x: abs(x[1]))
            reasoning.append(f"Primary driver: {top_contrib[0]} ({top_contrib[1]:.3f})")
            reasoning.append(f"Agreement: {agreement:.1%} across {len(active_categories)} categories")
        else:
            reasoning.append("No consensus - staying flat")

        return MasterSignal(
            direction=direction,
            confidence=final_confidence,
            position_size=position_size,
            signal_quality=signal_quality,
            hmm_contribution=contributions.get('hmm', 0.0),
            signal_contribution=contributions.get('signal', 0.0),
            nonlinear_contribution=contributions.get('nonlinear', 0.0),
            micro_contribution=contributions.get('micro', 0.0),
            ensemble_contribution=contributions.get('ensemble', 0.0),
            active_signals=total_active,
            agreement_ratio=agreement,
            regime=self.current_regime,
            reasoning=reasoning
        )

    def update_performance(self, category: str, was_correct: bool):
        """Track category performance for adaptive weighting."""
        self.category_performance[category].append(1 if was_correct else 0)

        # Adaptive weight update
        if len(self.category_performance[category]) >= 30:
            recent_acc = np.mean(list(self.category_performance[category])[-30:])

            # Increase weight for accurate categories
            current_weight = self.category_weights[category]
            if recent_acc > 0.55:
                self.category_weights[category] = min(0.4, current_weight * 1.05)
            elif recent_acc < 0.45:
                self.category_weights[category] = max(0.05, current_weight * 0.95)

            # Re-normalize weights
            total = sum(self.category_weights.values())
            self.category_weights = {k: v / total for k, v in self.category_weights.items()}


# =============================================================================
# FORMULA IMPLEMENTATIONS (72096-72099)
# =============================================================================

class MasterEnsembleSignal:
    """
    Formula 72096: Master Ensemble Signal

    Combines all 80+ formula outputs into single trading decision.
    The final arbiter of what we actually trade.
    """

    FORMULA_ID = 72096

    def __init__(self):
        self.master = MasterEnsemble()
        self.signal_buffer: Dict[int, Any] = {}  # Collect signals before combining

    def add_signal(self, formula_id: int, signal: Any):
        """Add a signal to the buffer for combination."""
        self.signal_buffer[formula_id] = signal

    def generate_signal(self) -> MasterSignal:
        """Generate master signal from all buffered signals."""
        # Group signals by category
        category_signals: Dict[str, Dict[str, Any]] = {
            cat: {} for cat in self.master.category_weights
        }

        for formula_id, signal in self.signal_buffer.items():
            category = self.master.categorize_signal(formula_id)
            if category in category_signals:
                category_signals[category][str(formula_id)] = signal

        # Aggregate each category
        category_scores = {}
        for category, signals in category_signals.items():
            category_scores[category] = self.master.aggregate_category(signals)

        # Combine into final signal
        return self.master.combine_categories(category_scores)

    def clear_buffer(self):
        """Clear signal buffer for next period."""
        self.signal_buffer = {}

    def update_outcome(self, actual_return: float):
        """Update performance tracking with actual outcome."""
        for formula_id, signal in self.signal_buffer.items():
            category = self.master.categorize_signal(formula_id)

            if hasattr(signal, 'direction'):
                d = signal.direction
            elif isinstance(signal, dict):
                d = signal.get('direction', 0)
            else:
                continue

            if d != 0:
                correct = (d > 0 and actual_return > 0) or (d < 0 and actual_return < 0)
                self.master.update_performance(category, correct)


class ConservativeMasterSignal:
    """
    Formula 72097: Conservative Master Signal

    Only trades when there's overwhelming agreement.
    Higher win rate, fewer trades.
    """

    FORMULA_ID = 72097

    def __init__(self):
        self.master = MasterEnsemble()
        self.master.min_agreement_to_trade = 0.6  # Higher threshold
        self.master.high_confidence_agreement = 0.85
        self.signal_buffer: Dict[int, Any] = {}

    def add_signal(self, formula_id: int, signal: Any):
        """Add signal to buffer."""
        self.signal_buffer[formula_id] = signal

    def generate_signal(self) -> MasterSignal:
        """Generate conservative signal."""
        # Same as master, but with stricter thresholds
        category_signals: Dict[str, Dict[str, Any]] = {
            cat: {} for cat in self.master.category_weights
        }

        for formula_id, signal in self.signal_buffer.items():
            category = self.master.categorize_signal(formula_id)
            if category in category_signals:
                category_signals[category][str(formula_id)] = signal

        category_scores = {}
        for category, signals in category_signals.items():
            category_scores[category] = self.master.aggregate_category(signals)

        signal = self.master.combine_categories(category_scores)

        # Additional conservatism: require multiple categories to agree
        active_cats = sum(1 for _, _, a in category_scores.values() if a > 0)
        if active_cats < 3:
            return MasterSignal(
                direction=0,
                confidence=0.0,
                position_size=0.0,
                signal_quality='no_trade',
                hmm_contribution=signal.hmm_contribution,
                signal_contribution=signal.signal_contribution,
                nonlinear_contribution=signal.nonlinear_contribution,
                micro_contribution=signal.micro_contribution,
                ensemble_contribution=signal.ensemble_contribution,
                active_signals=signal.active_signals,
                agreement_ratio=signal.agreement_ratio,
                regime=signal.regime,
                reasoning=['Insufficient category diversity']
            )

        return signal

    def clear_buffer(self):
        self.signal_buffer = {}


class AggressiveMasterSignal:
    """
    Formula 72098: Aggressive Master Signal

    Trades on weaker agreement. More trades, higher variance.
    For high-frequency or momentum-oriented strategies.
    """

    FORMULA_ID = 72098

    def __init__(self):
        self.master = MasterEnsemble()
        self.master.min_agreement_to_trade = 0.2  # Lower threshold
        self.master.high_confidence_agreement = 0.5
        self.signal_buffer: Dict[int, Any] = {}

        # Momentum boost: recent winners get more weight
        self.recent_winners: deque = deque(maxlen=20)

    def add_signal(self, formula_id: int, signal: Any):
        """Add signal with momentum tracking."""
        self.signal_buffer[formula_id] = signal

    def generate_signal(self) -> MasterSignal:
        """Generate aggressive signal."""
        category_signals: Dict[str, Dict[str, Any]] = {
            cat: {} for cat in self.master.category_weights
        }

        for formula_id, signal in self.signal_buffer.items():
            category = self.master.categorize_signal(formula_id)
            if category in category_signals:
                category_signals[category][str(formula_id)] = signal

        category_scores = {}
        for category, signals in category_signals.items():
            category_scores[category] = self.master.aggregate_category(signals)

        signal = self.master.combine_categories(category_scores)

        # Boost position size based on momentum
        if signal.direction != 0:
            win_streak = sum(1 for w in self.recent_winners if w)
            momentum_boost = min(1.5, 1 + win_streak * 0.1)
            signal = MasterSignal(
                direction=signal.direction,
                confidence=signal.confidence,
                position_size=min(1.0, signal.position_size * momentum_boost),
                signal_quality=signal.signal_quality,
                hmm_contribution=signal.hmm_contribution,
                signal_contribution=signal.signal_contribution,
                nonlinear_contribution=signal.nonlinear_contribution,
                micro_contribution=signal.micro_contribution,
                ensemble_contribution=signal.ensemble_contribution,
                active_signals=signal.active_signals,
                agreement_ratio=signal.agreement_ratio,
                regime=signal.regime,
                reasoning=signal.reasoning + [f'Momentum boost: {momentum_boost:.2f}x']
            )

        return signal

    def update_outcome(self, was_winner: bool):
        """Track winning momentum."""
        self.recent_winners.append(was_winner)

    def clear_buffer(self):
        self.signal_buffer = {}


class AdaptiveMasterSignal:
    """
    Formula 72099: Adaptive Master Signal

    Automatically adjusts between conservative and aggressive
    based on market regime and recent performance.
    """

    FORMULA_ID = 72099

    def __init__(self):
        self.conservative = ConservativeMasterSignal()
        self.aggressive = AggressiveMasterSignal()
        self.base_master = MasterEnsemble()

        # Regime detection
        self.volatility_history: deque = deque(maxlen=50)
        self.performance_history: deque = deque(maxlen=50)
        self.current_mode = 'balanced'  # 'conservative', 'balanced', 'aggressive'

        self.signal_buffer: Dict[int, Any] = {}

    def detect_regime(self, recent_returns: np.ndarray = None):
        """Detect market regime to choose strategy mode."""
        if recent_returns is not None and len(recent_returns) >= 10:
            vol = np.std(recent_returns[-10:])
            self.volatility_history.append(vol)

        if len(self.volatility_history) < 20:
            self.current_mode = 'balanced'
            return

        vol_pct = np.percentile(self.volatility_history, [25, 75])
        current_vol = self.volatility_history[-1] if self.volatility_history else 0

        # Recent performance
        if len(self.performance_history) >= 10:
            recent_wr = np.mean(list(self.performance_history)[-10:])
        else:
            recent_wr = 0.5

        # Decision logic
        if current_vol > vol_pct[1]:  # High volatility
            if recent_wr > 0.55:
                self.current_mode = 'aggressive'  # We're winning, press advantage
            else:
                self.current_mode = 'conservative'  # Reduce risk
        elif current_vol < vol_pct[0]:  # Low volatility
            if recent_wr > 0.5:
                self.current_mode = 'aggressive'  # Safe to be aggressive
            else:
                self.current_mode = 'balanced'
        else:
            self.current_mode = 'balanced'

    def add_signal(self, formula_id: int, signal: Any):
        """Add signal to all sub-ensembles."""
        self.signal_buffer[formula_id] = signal
        self.conservative.add_signal(formula_id, signal)
        self.aggressive.add_signal(formula_id, signal)

    def generate_signal(self, recent_returns: np.ndarray = None) -> MasterSignal:
        """Generate adaptive signal based on regime."""
        self.detect_regime(recent_returns)

        if self.current_mode == 'conservative':
            signal = self.conservative.generate_signal()
            signal.reasoning.append('Mode: CONSERVATIVE (high vol or losing)')
        elif self.current_mode == 'aggressive':
            signal = self.aggressive.generate_signal()
            signal.reasoning.append('Mode: AGGRESSIVE (low vol or winning)')
        else:
            # Balanced: use base master
            category_signals: Dict[str, Dict[str, Any]] = {
                cat: {} for cat in self.base_master.category_weights
            }

            for formula_id, sig in self.signal_buffer.items():
                category = self.base_master.categorize_signal(formula_id)
                if category in category_signals:
                    category_signals[category][str(formula_id)] = sig

            category_scores = {}
            for category, signals in category_signals.items():
                category_scores[category] = self.base_master.aggregate_category(signals)

            signal = self.base_master.combine_categories(category_scores)
            signal.reasoning.append('Mode: BALANCED (normal conditions)')

        return signal

    def update_outcome(self, actual_return: float, predicted_direction: int):
        """Update performance tracking."""
        if predicted_direction != 0:
            correct = (predicted_direction > 0 and actual_return > 0) or \
                      (predicted_direction < 0 and actual_return < 0)
            self.performance_history.append(1 if correct else 0)

    def clear_buffer(self):
        """Clear all buffers."""
        self.signal_buffer = {}
        self.conservative.clear_buffer()
        self.aggressive.clear_buffer()
