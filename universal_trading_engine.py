"""
UNIVERSAL ADAPTIVE TRADING ENGINE
==================================
Complete trading system that dynamically adapts to ANY market condition.

This is the SOLUTION to: "What works for 1 second doesn't work for 2 seconds"

Architecture:
1. Load ALL 508+ formulas as "experts"
2. Universal Adaptive System learns which experts work NOW
3. Exponential Gradient updates weights in real-time
4. Trade based on weighted consensus

Mathematical Foundation:
- Cover (1991) Universal Portfolios - guaranteed regret bounds
- Exponential Gradient - O(sqrt(T*ln(N))) competitive with best expert
- Adaptive learning rate - responds to volatility regime

Key Insight: We don't know which formula will work next, but we CAN track
which formulas HAVE been working and weight them more heavily.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import deque

# Import the Universal Adaptive System
from formulas.universal_portfolio import (
    UniversalAdaptiveSystem,
    ExponentialGradientMetaLearner,
    AdaptiveRegimeMetaLearner,
    FormulaPerformanceTracker,
    UniversalSignal,
)

# Import adaptive volatility trader
from adaptive_trader import AdaptiveVolatilityTrader, AdaptiveParameters

# Import formula base
from formulas.base import BaseFormula, FORMULA_REGISTRY


@dataclass
class TradeDecision:
    """Complete trade decision from the engine."""
    timestamp: float
    action: str  # 'LONG', 'SHORT', 'HOLD', 'EXIT'
    signal: float  # -1 to +1
    confidence: float  # 0 to 1
    position_size: float  # 0 to 1 (Kelly-capped)

    # Adaptive parameters
    take_profit_pct: float
    stop_loss_pct: float
    expected_hold_secs: float

    # Meta-learning info
    regime: str
    top_formulas: List[int]
    expected_edge: float

    # Price levels
    entry_price: float
    tp_price: float
    sl_price: float


class UniversalTradingEngine:
    """
    The COMPLETE solution to trading across all market states.

    This engine:
    1. Runs ALL formulas in parallel
    2. Tracks each formula's real-time performance
    3. Dynamically weights formulas using Exponential Gradient
    4. Adjusts TP/SL based on volatility regime
    5. Sizes positions using Kelly criterion

    Usage:
        engine = UniversalTradingEngine()
        decision = engine.process(price=97000, volume=1000)
        if decision.action == 'LONG':
            execute_long(
                size=decision.position_size,
                tp=decision.tp_price,
                sl=decision.sl_price
            )
    """

    def __init__(
        self,
        formulas: Dict[int, BaseFormula] = None,
        learning_rate: float = 0.05,
        min_confidence: float = 0.6,
        max_position_size: float = 0.25,
        # Trading costs (for Kelly calculation)
        round_trip_fee_pct: float = 0.002,
        slippage_pct: float = 0.0005,
    ):
        """
        Initialize the Universal Trading Engine.

        Args:
            formulas: Dict of formula_id -> formula instance
            learning_rate: EG learning rate
            min_confidence: Minimum confidence to trade
            max_position_size: Maximum position size (Kelly cap)
            round_trip_fee_pct: Trading fee (0.20%)
            slippage_pct: Estimated slippage (0.05%)
        """
        # Load all formulas
        if formulas is None:
            self.formulas = self._load_all_formulas()
        else:
            self.formulas = formulas

        self.n_formulas = len(self.formulas)
        print(f"[UniversalEngine] Loaded {self.n_formulas} formulas")

        # Create formula index mapping
        self.formula_ids = sorted(self.formulas.keys())
        self.id_to_idx = {fid: idx for idx, fid in enumerate(self.formula_ids)}

        # Universal Adaptive System
        self.meta_learner = UniversalAdaptiveSystem(
            n_formulas=self.n_formulas,
            learning_rate=learning_rate
        )

        # Adaptive volatility trader (for TP/SL scaling)
        self.vol_trader = AdaptiveVolatilityTrader(
            round_trip_fee_pct=round_trip_fee_pct,
            slippage_pct=slippage_pct
        )

        # Trading parameters
        self.min_confidence = min_confidence
        self.max_position_size = max_position_size
        self.trading_costs = round_trip_fee_pct + slippage_pct

        # State
        self.current_position = None  # 'LONG', 'SHORT', or None
        self.entry_price = 0.0
        self.entry_time = 0.0
        self.current_params: AdaptiveParameters = None

        # History for analysis
        self.signal_history = deque(maxlen=1000)
        self.decision_history = deque(maxlen=1000)
        self.trade_history = deque(maxlen=100)

        self.last_price = 0.0
        self.total_updates = 0

    def _load_all_formulas(self) -> Dict[int, BaseFormula]:
        """Load all registered formulas."""
        formulas = {}
        for formula_id, formula_class in FORMULA_REGISTRY.items():
            try:
                formulas[formula_id] = formula_class()
            except Exception as e:
                print(f"[UniversalEngine] Failed to load formula {formula_id}: {e}")
        return formulas

    def _get_all_signals(self, price: float, volume: float, timestamp: float) -> np.ndarray:
        """
        Get signals from ALL formulas.

        Returns array of signals indexed by formula position.
        """
        signals = np.zeros(self.n_formulas)

        for idx, formula_id in enumerate(self.formula_ids):
            formula = self.formulas[formula_id]
            try:
                # Update formula
                formula.update(price, volume, timestamp)

                # Get signal
                sig = formula.get_signal()

                # Normalize to [-1, 1]
                signals[idx] = np.clip(sig, -1, 1)
            except Exception:
                signals[idx] = 0.0

        return signals

    def _calculate_kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion position size.

        Formula: f* = (p * b - q) / b
        Where:
            p = probability of winning
            q = 1 - p
            b = win/loss ratio
        """
        if avg_loss == 0 or win_rate <= 0:
            return 0.0

        p = win_rate
        q = 1 - p
        b = abs(avg_win / avg_loss)

        kelly_f = (p * b - q) / b

        # Kelly is aggressive - use fractional Kelly (25%)
        kelly_f = kelly_f * 0.25

        return max(0, min(kelly_f, self.max_position_size))

    def process(
        self,
        price: float,
        volume: float = 0.0,
        timestamp: float = None,
    ) -> TradeDecision:
        """
        Process new data and return trading decision.

        This is the MAIN entry point.

        Args:
            price: Current market price
            volume: Current volume
            timestamp: Unix timestamp

        Returns:
            TradeDecision with complete trading recommendation
        """
        if timestamp is None:
            timestamp = time.time()

        self.total_updates += 1

        # 1. Update volatility tracker
        self.vol_trader.update(price, timestamp)

        # 2. Get signals from ALL formulas
        signals = self._get_all_signals(price, volume, timestamp)

        # 3. Update meta-learner and get weighted signal
        meta_result = self.meta_learner.update(price, signals)

        # 4. Get adaptive parameters (TP/SL scaled to volatility)
        position_type = 'LONG' if meta_result.signal > 0 else 'SHORT'
        params = self.vol_trader.get_adaptive_parameters(price, position_type)
        self.current_params = params

        # 5. Determine action
        action = 'HOLD'

        # Check for exit if in position
        if self.current_position is not None:
            should_exit, exit_reason, is_win = self.vol_trader.should_exit(
                self.entry_price, price, self.current_position, self.current_params
            )

            if should_exit:
                action = 'EXIT'

                # Record trade
                pnl = (price - self.entry_price) / self.entry_price
                if self.current_position == 'SHORT':
                    pnl = -pnl

                self.trade_history.append({
                    'entry_price': self.entry_price,
                    'exit_price': price,
                    'position': self.current_position,
                    'pnl_pct': pnl,
                    'reason': exit_reason,
                    'is_win': is_win,
                    'hold_time': timestamp - self.entry_time
                })

                self.current_position = None

        # Check for entry if no position
        if self.current_position is None and action != 'EXIT':
            # Strong enough signal?
            if meta_result.confidence >= self.min_confidence:
                if meta_result.signal > 0.2:  # Bullish
                    action = 'LONG'
                    self.current_position = 'LONG'
                    self.entry_price = price
                    self.entry_time = timestamp
                elif meta_result.signal < -0.2:  # Bearish
                    action = 'SHORT'
                    self.current_position = 'SHORT'
                    self.entry_price = price
                    self.entry_time = timestamp

        # 6. Calculate position size
        # Use Kelly based on recent win rate
        if len(self.trade_history) >= 10:
            recent_trades = list(self.trade_history)[-20:]
            wins = [t for t in recent_trades if t['is_win']]
            losses = [t for t in recent_trades if not t['is_win']]

            win_rate = len(wins) / len(recent_trades)
            avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else params.take_profit
            avg_loss = np.mean([abs(t['pnl_pct']) for t in losses]) if losses else params.stop_loss

            position_size = self._calculate_kelly_size(win_rate, avg_win, avg_loss)
        else:
            # Default conservative size
            position_size = 0.1

        # 7. Build decision
        decision = TradeDecision(
            timestamp=timestamp,
            action=action,
            signal=meta_result.signal,
            confidence=meta_result.confidence,
            position_size=position_size,
            take_profit_pct=params.take_profit,
            stop_loss_pct=params.stop_loss,
            expected_hold_secs=params.expected_hold_secs,
            regime=meta_result.regime,
            top_formulas=meta_result.top_formulas,
            expected_edge=meta_result.expected_edge,
            entry_price=price if action in ['LONG', 'SHORT'] else self.entry_price,
            tp_price=params.tp_price,
            sl_price=params.sl_price,
        )

        # Store for analysis
        self.decision_history.append(decision)
        self.last_price = price

        return decision

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary of the trading engine.
        """
        if not self.trade_history:
            return {'trades': 0, 'win_rate': 0, 'total_pnl': 0}

        trades = list(self.trade_history)
        wins = [t for t in trades if t['is_win']]

        total_pnl = sum(t['pnl_pct'] for t in trades)
        gross_pnl = total_pnl - len(trades) * self.trading_costs  # Subtract costs

        return {
            'trades': len(trades),
            'wins': len(wins),
            'losses': len(trades) - len(wins),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'total_pnl_gross': total_pnl,
            'total_pnl_net': gross_pnl,
            'avg_win': np.mean([t['pnl_pct'] for t in wins]) if wins else 0,
            'avg_loss': np.mean([abs(t['pnl_pct']) for t in trades if not t['is_win']]) if len(trades) > len(wins) else 0,
            'avg_hold_time': np.mean([t['hold_time'] for t in trades]),
            'regret_bound': self.meta_learner.get_regret_bound(),
        }

    def get_top_formulas(self, k: int = 10) -> List[Tuple[int, float]]:
        """Get top performing formulas."""
        return self.meta_learner.tracker.get_top_performers(k, 'cumulative')

    def get_formula_weights(self) -> Dict[int, float]:
        """Get current formula weights."""
        weights = self.meta_learner.get_weights()
        return {
            self.formula_ids[i]: float(weights[i])
            for i in range(self.n_formulas)
        }


# =============================================================================
# LIGHTWEIGHT VERSION (for testing)
# =============================================================================

class LightweightUniversalEngine:
    """
    Lightweight version that works without loading all formulas.

    Uses a subset of core formulas for faster testing.
    """

    def __init__(
        self,
        n_virtual_formulas: int = 50,
        learning_rate: float = 0.05,
    ):
        self.n_formulas = n_virtual_formulas

        # Meta-learner
        self.meta_learner = UniversalAdaptiveSystem(
            n_formulas=n_virtual_formulas,
            learning_rate=learning_rate
        )

        # Adaptive volatility
        self.vol_trader = AdaptiveVolatilityTrader()

        # Virtual formula signals (simulated)
        self.price_history = deque(maxlen=200)
        self.return_history = deque(maxlen=200)

        self.last_price = 0.0

    def _generate_virtual_signals(self, price: float) -> np.ndarray:
        """
        Generate signals from "virtual" formulas based on price patterns.

        This simulates different formula types:
        - Momentum followers
        - Mean reversion
        - Volatility breakout
        - Random noise
        """
        signals = np.zeros(self.n_formulas)

        if len(self.price_history) < 20:
            return signals

        prices = np.array(self.price_history)
        returns = np.diff(np.log(prices + 1))

        # Recent metrics
        momentum_5 = np.mean(returns[-5:])
        momentum_20 = np.mean(returns[-20:]) if len(returns) >= 20 else momentum_5
        vol_recent = np.std(returns[-10:])
        vol_long = np.std(returns[-50:]) if len(returns) >= 50 else vol_recent

        # Price vs MA
        ma_20 = np.mean(prices[-20:])
        deviation = (price - ma_20) / ma_20

        n_per_type = self.n_formulas // 5

        # Type 1: Momentum (0 to n_per_type)
        for i in range(n_per_type):
            lookback = 3 + i
            if len(returns) >= lookback:
                signals[i] = np.tanh(np.mean(returns[-lookback:]) * 100)

        # Type 2: Mean Reversion (n_per_type to 2*n_per_type)
        for i in range(n_per_type, 2*n_per_type):
            signals[i] = -np.tanh(deviation * 10)

        # Type 3: Volatility Breakout (2*n_per_type to 3*n_per_type)
        for i in range(2*n_per_type, 3*n_per_type):
            vol_ratio = vol_recent / (vol_long + 1e-10)
            if vol_ratio > 1.5:
                signals[i] = np.sign(momentum_5) * 0.8

        # Type 4: Contrarian (3*n_per_type to 4*n_per_type)
        for i in range(3*n_per_type, 4*n_per_type):
            if abs(momentum_5) > 2 * vol_recent:
                signals[i] = -np.sign(momentum_5) * 0.6

        # Type 5: Random (4*n_per_type to end) - noise floor
        signals[4*n_per_type:] = np.random.randn(self.n_formulas - 4*n_per_type) * 0.1

        return signals

    def process(self, price: float, timestamp: float = None) -> UniversalSignal:
        """
        Process price and return adaptive signal.
        """
        if timestamp is None:
            timestamp = time.time()

        self.price_history.append(price)
        self.vol_trader.update(price, timestamp)

        if self.last_price > 0:
            ret = np.log(price / self.last_price)
            self.return_history.append(ret)

        # Generate virtual formula signals
        signals = self._generate_virtual_signals(price)

        # Update meta-learner
        result = self.meta_learner.update(price, signals)

        self.last_price = price
        return result


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("UNIVERSAL ADAPTIVE TRADING ENGINE - TEST")
    print("=" * 70)

    # Use lightweight version for testing
    engine = LightweightUniversalEngine(n_virtual_formulas=50, learning_rate=0.1)

    # Simulate trending market
    np.random.seed(42)
    price = 90000.0

    print("\n--- PHASE 1: TRENDING UP ---")
    for i in range(100):
        price *= 1 + 0.001 + np.random.randn() * 0.0005  # Slight uptrend
        result = engine.process(price, timestamp=i)

        if i % 20 == 0:
            print(f"Step {i}: price=${price:,.0f}, signal={result.signal:.3f}, "
                  f"conf={result.confidence:.3f}, regime={result.regime}")

    print("\n--- PHASE 2: MEAN REVERSION ---")
    center = price
    for i in range(100, 200):
        # Mean revert around center
        price = center + (price - center) * 0.95 + np.random.randn() * 20
        result = engine.process(price, timestamp=i)

        if i % 20 == 0:
            print(f"Step {i}: price=${price:,.0f}, signal={result.signal:.3f}, "
                  f"conf={result.confidence:.3f}, regime={result.regime}")

    print("\n--- PHASE 3: VOLATILE CRASH ---")
    for i in range(200, 250):
        price *= 1 - 0.003 + np.random.randn() * 0.002  # Downtrend with high vol
        result = engine.process(price, timestamp=i)

        if i % 10 == 0:
            print(f"Step {i}: price=${price:,.0f}, signal={result.signal:.3f}, "
                  f"conf={result.confidence:.3f}, regime={result.regime}")

    print("\n" + "=" * 70)
    print("TOP PERFORMING VIRTUAL FORMULAS:")
    print("=" * 70)
    top = engine.meta_learner.tracker.get_top_performers(10)
    for formula_id, pnl in top:
        print(f"  Formula {formula_id}: cumulative PnL = {pnl:.6f}")

    print(f"\nFinal regret bound: {engine.meta_learner.get_regret_bound():.2f}")
