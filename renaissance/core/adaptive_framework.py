"""
Renaissance Trading System - UNIVERSAL ADAPTIVE MATHEMATICAL FRAMEWORK
======================================================================
PURE MATHEMATICS for ALL market conditions

THE PROBLEM: Static parameters fail when market conditions change.
- At t=23s: Ranging market, mean reversion works, 26% WR
- At t=60s: Trending market, mean reversion fails, 18% WR

THE SOLUTION: Adaptive mathematics that scale with market conditions.

MATHEMATICAL FOUNDATION:
1. Bayesian Regime Detection: P(regime|data) using Hidden Markov Model
2. Ornstein-Uhlenbeck Half-Life: τ = ln(2)/θ for optimal holding
3. Hurst Exponent: H determines trend vs mean-reversion
4. EWMA Volatility: σ_t = α × r²_t + (1-α) × σ²_{t-1}
5. Kelly Criterion: f* = (p × b - q) / b where p=WR, b=R:R

ADAPTIVE SCALING LAWS:
- lookback(σ) = base_lookback × (σ_long / σ_short)
- threshold(σ) = base_threshold × √σ
- kelly(WR) = (WR × 2 - (1-WR)) / 2 × safety_factor
- exit_z(regime) = -0.3 if MR else -0.7 if trending
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict


class AdaptiveMathEngine:
    """
    Universal Adaptive Mathematical Framework

    Automatically adjusts ALL parameters based on:
    1. Current volatility (σ)
    2. Current regime (H)
    3. Recent win rate (WR)
    4. OU half-life (τ)

    MATHEMATICAL GUARANTEES:
    - Parameters scale with √σ (volatility)
    - Thresholds adapt to regime persistence
    - Kelly fraction adjusts to recent performance
    """

    def __init__(self, base_lookback=50, base_kelly=0.02):
        # Base parameters (will be scaled)
        self.base_lookback = base_lookback
        self.base_kelly = base_kelly

        # Price history for calculations
        self.prices = deque(maxlen=500)
        self.returns = deque(maxlen=500)

        # Volatility tracking (EWMA)
        self.ewma_vol_fast = 0  # Fast EWMA (α=0.1)
        self.ewma_vol_slow = 0  # Slow EWMA (α=0.01)
        self.alpha_fast = 0.1
        self.alpha_slow = 0.01

        # Regime state (Bayesian)
        self.regime_probs = {'mean_reversion': 0.5, 'trending': 0.3, 'volatile': 0.2}
        self.current_regime = 'unknown'

        # Performance tracking for adaptive Kelly
        self.recent_trades = deque(maxlen=50)
        self.recent_wins = 0
        self.recent_losses = 0

        # OU parameters
        self.ou_theta = 0
        self.ou_half_life = float('inf')

        # Hurst exponent
        self.hurst = 0.5

        # Trend direction (1=up, -1=down, 0=neutral)
        self.trend_direction = 0
        self.momentum_strength = 0.0

        # Cached adaptive parameters
        self.adaptive_params = {}

    def update(self, price: float) -> Dict:
        """
        Update all adaptive parameters with new price

        Returns: Dict of all adaptive parameters
        """
        # Store price
        self.prices.append(price)

        # Calculate return if we have enough data
        if len(self.prices) >= 2:
            ret = (self.prices[-1] - self.prices[-2]) / self.prices[-2]
            self.returns.append(ret)

            # Update EWMA volatility
            ret_sq = ret ** 2
            if self.ewma_vol_fast == 0:
                self.ewma_vol_fast = ret_sq
                self.ewma_vol_slow = ret_sq
            else:
                self.ewma_vol_fast = self.alpha_fast * ret_sq + (1 - self.alpha_fast) * self.ewma_vol_fast
                self.ewma_vol_slow = self.alpha_slow * ret_sq + (1 - self.alpha_slow) * self.ewma_vol_slow

        # Update all adaptive parameters
        if len(self.prices) >= 50:
            self._update_regime()
            self._update_ou_params()
            self._calculate_adaptive_params()

        return self.adaptive_params

    def _update_ou_params(self):
        """Update OU process parameters (theta and half-life)"""
        prices = list(self.prices)
        self.ou_theta = self._calculate_ou_theta(prices)

    def _update_regime(self):
        """
        Bayesian regime detection using multiple signals:
        1. Hurst exponent (trend persistence)
        2. Volatility ratio (fast/slow)
        3. OU theta (mean reversion speed)
        4. Momentum direction and strength
        """
        prices = list(self.prices)

        # 1. Hurst Exponent: H < 0.45 = MR, H > 0.55 = Trending
        self.hurst = self._calculate_hurst(prices)

        # 2. Volatility regime: Fast/Slow ratio
        vol_ratio = np.sqrt(self.ewma_vol_fast) / (np.sqrt(self.ewma_vol_slow) + 1e-10)

        # 3. OU theta for mean reversion speed
        self.ou_theta = self._calculate_ou_theta(prices)

        # 4. Calculate trend direction and momentum strength
        self._calculate_trend_direction(prices)

        # Bayesian update of regime probabilities
        # P(regime|data) ∝ P(data|regime) × P(regime)

        # Likelihood based on Hurst
        if self.hurst < 0.45:
            p_mr = 0.8
            p_tr = 0.1
            p_vol = 0.1
        elif self.hurst > 0.55:
            p_mr = 0.1
            p_tr = 0.7
            p_vol = 0.2
        else:
            p_mr = 0.4
            p_tr = 0.3
            p_vol = 0.3

        # Adjust based on volatility ratio
        if vol_ratio > 1.5:
            p_vol = min(0.8, p_vol + 0.3)
            p_mr = max(0.1, p_mr - 0.15)
            p_tr = max(0.1, p_tr - 0.15)

        # Adjust based on OU theta
        if self.ou_theta > 0.8:  # Fast mean reversion
            p_mr = min(0.9, p_mr + 0.2)
            p_tr = max(0.05, p_tr - 0.1)

        # Adjust based on momentum strength
        if abs(self.momentum_strength) > 0.001:  # Strong momentum = trending
            p_tr = min(0.9, p_tr + 0.2)
            p_mr = max(0.05, p_mr - 0.1)

        # Normalize probabilities
        total = p_mr + p_tr + p_vol
        self.regime_probs = {
            'mean_reversion': p_mr / total,
            'trending': p_tr / total,
            'volatile': p_vol / total
        }

        # Determine dominant regime
        self.current_regime = max(self.regime_probs, key=self.regime_probs.get)

    def _calculate_trend_direction(self, prices):
        """
        Calculate trend direction using multiple methods:
        1. Linear regression slope
        2. EMA crossover
        3. Price vs moving average

        MATH:
        - Slope > 0 + Price > MA = UP TREND (direction = 1)
        - Slope < 0 + Price < MA = DOWN TREND (direction = -1)
        - Otherwise = NEUTRAL (direction = 0)
        """
        if len(prices) < 20:
            self.trend_direction = 0
            self.momentum_strength = 0.0
            return

        pp = np.array(prices[-50:]) if len(prices) >= 50 else np.array(prices)

        # 1. Linear regression slope (normalized)
        x = np.arange(len(pp))
        slope = np.polyfit(x, pp, 1)[0]
        # Normalize by average price
        normalized_slope = slope / np.mean(pp)

        # 2. Price vs EMA
        ema_fast = self._ema(pp, 10)
        ema_slow = self._ema(pp, 30) if len(pp) >= 30 else self._ema(pp, len(pp) // 2)

        price_vs_ema = (pp[-1] - ema_slow) / ema_slow if ema_slow > 0 else 0

        # 3. EMA crossover direction
        ema_diff = (ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0

        # Combine signals with LOOSER thresholds for micro-trends
        # (Bitcoin moves are small in percentage terms)
        signals = [
            1 if normalized_slope > 0.00001 else (-1 if normalized_slope < -0.00001 else 0),
            1 if price_vs_ema > 0.0001 else (-1 if price_vs_ema < -0.0001 else 0),
            1 if ema_diff > 0.00001 else (-1 if ema_diff < -0.00001 else 0)
        ]

        # Trend direction = majority vote
        signal_sum = sum(signals)
        if signal_sum >= 2:
            self.trend_direction = 1
        elif signal_sum <= -2:
            self.trend_direction = -1
        else:
            self.trend_direction = 0

        # Momentum strength = normalized slope (for regime detection)
        self.momentum_strength = normalized_slope

    def _ema(self, data, period):
        """Exponential Moving Average"""
        if len(data) == 0:
            return 0
        if len(data) < period:
            period = len(data)
        alpha = 2.0 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def _calculate_hurst(self, prices, max_lag=20) -> float:
        """
        Hurst Exponent using PROPER R/S (Rescaled Range) analysis

        MATHEMATICAL FORMULA:
        R/S = (max(Y) - min(Y)) / S
        where Y = cumulative deviations from mean
        E[R/S] ~ c × n^H

        H > 0.5: TRENDING (momentum) - USE MOMENTUM STRATEGY
        H = 0.5: Random walk
        H < 0.5: MEAN REVERTING - USE MEAN REVERSION STRATEGY
        """
        if len(prices) < max_lag * 2:
            return 0.5

        # Use RETURNS not prices (this is the key fix!)
        pp = np.array(prices[-100:]) if len(prices) >= 100 else np.array(prices)
        returns = np.diff(pp) / pp[:-1]

        if len(returns) < 20:
            return 0.5

        # R/S analysis over multiple window sizes
        rs_values = []
        ns = []

        for n in [10, 15, 20, 25, 30, 40, 50]:
            if n > len(returns):
                continue

            # Calculate R/S for this window size
            rs_list = []
            for start in range(0, len(returns) - n + 1, n // 2):
                window = returns[start:start + n]
                if len(window) < n:
                    continue

                # Mean of window
                mean = np.mean(window)

                # Cumulative deviations from mean
                Y = np.cumsum(window - mean)

                # Range
                R = np.max(Y) - np.min(Y)

                # Standard deviation
                S = np.std(window, ddof=1)

                if S > 1e-10:
                    rs_list.append(R / S)

            if rs_list:
                rs_values.append(np.mean(rs_list))
                ns.append(n)

        if len(rs_values) < 3:
            # Fallback: use variance ratio test
            return self._variance_ratio_hurst(returns)

        # Linear regression: log(R/S) = H × log(n) + c
        log_n = np.log(ns)
        log_rs = np.log(rs_values)

        try:
            H = np.polyfit(log_n, log_rs, 1)[0]
            return max(0.1, min(0.9, H))
        except:
            return 0.5

    def _variance_ratio_hurst(self, returns) -> float:
        """
        Variance Ratio test for Hurst estimation

        VR(q) = Var(r_q) / (q × Var(r_1))
        For random walk: VR = 1
        For trending: VR > 1 (H > 0.5)
        For mean-reverting: VR < 1 (H < 0.5)
        """
        if len(returns) < 20:
            return 0.5

        var_1 = np.var(returns)
        if var_1 < 1e-12:
            return 0.5

        # Calculate variance ratio for q=5
        q = 5
        if len(returns) < q * 2:
            return 0.5

        # Aggregate returns over q periods
        n_blocks = len(returns) // q
        aggregated = []
        for i in range(n_blocks):
            block_return = np.sum(returns[i*q:(i+1)*q])
            aggregated.append(block_return)

        var_q = np.var(aggregated)

        # Variance ratio
        VR = var_q / (q * var_1)

        # Convert VR to Hurst: H = 0.5 + log2(sqrt(VR)) / 2
        # For VR=1: H=0.5, VR>1: H>0.5 (trending), VR<1: H<0.5 (mean revert)
        if VR > 0:
            H = 0.5 + np.log2(np.sqrt(VR)) / 2
            return max(0.1, min(0.9, H))
        return 0.5

    def _calculate_ou_theta(self, prices) -> float:
        """
        Ornstein-Uhlenbeck theta (mean reversion speed)
        dX = θ(μ - X)dt + σdW

        High θ = fast mean reversion
        Low θ = slow mean reversion / trending
        """
        if len(prices) < 20:
            return 0.5

        X = np.array(prices[:-1])
        Y = np.array(prices[1:])

        X_mean, Y_mean = np.mean(X), np.mean(Y)

        denom = np.sum((X - X_mean) ** 2)
        if denom == 0:
            return 0.5

        b = np.sum((X - X_mean) * (Y - Y_mean)) / denom
        b = max(min(b, 0.9999), 0.0001)

        theta = -np.log(b)  # θ = -ln(β)

        # Calculate half-life: τ = ln(2) / θ
        self.ou_half_life = np.log(2) / max(theta, 0.001)

        return theta

    def _calculate_adaptive_params(self):
        """
        Calculate ALL adaptive parameters based on current market state

        MATHEMATICAL SCALING LAWS:
        1. lookback ~ σ_ratio (more volatile = shorter lookback)
        2. entry_threshold ~ √σ (scale with volatility)
        3. exit_z ~ regime (tighter in MR, wider in trending)
        4. kelly ~ recent_WR (adaptive sizing)
        5. hold_time ~ τ (OU half-life)
        """
        # Volatility ratio for scaling
        vol_fast = np.sqrt(self.ewma_vol_fast) if self.ewma_vol_fast > 0 else 0.001
        vol_slow = np.sqrt(self.ewma_vol_slow) if self.ewma_vol_slow > 0 else 0.001
        vol_ratio = vol_fast / vol_slow

        # 1. ADAPTIVE LOOKBACK
        # In high volatility: use shorter lookback (faster adaptation)
        # In low volatility: use longer lookback (more stability)
        lookback_scale = 1.0 / max(0.5, min(2.0, vol_ratio))
        adaptive_lookback = int(self.base_lookback * lookback_scale)
        adaptive_lookback = max(20, min(100, adaptive_lookback))

        # 2. ADAPTIVE ENTRY THRESHOLD
        # Scale with √σ (volatility)
        base_threshold = 0.0001  # 0.01%
        adaptive_entry_threshold = base_threshold * np.sqrt(vol_fast / 0.001)
        adaptive_entry_threshold = max(0.00005, min(0.001, adaptive_entry_threshold))

        # 3. ADAPTIVE EXIT Z-SCORE (THE CRITICAL PARAMETER)
        # In MR regime: tighter exit (z=-0.3) for higher WR
        # In trending regime: wider exit (z=-0.7) to ride trends
        # In volatile regime: very tight (z=-0.2) for quick profits
        if self.current_regime == 'mean_reversion':
            adaptive_exit_z = -0.3
            strategy_mode = 'MEAN_REVERSION'
        elif self.current_regime == 'trending':
            adaptive_exit_z = -0.7
            strategy_mode = 'MOMENTUM'
        else:  # volatile
            adaptive_exit_z = -0.2
            strategy_mode = 'SCALP'

        # 4. ADAPTIVE KELLY (based on recent performance)
        recent_wr = self._get_recent_win_rate()
        # Kelly: f* = (p × b - q) / b where b = 2 (2:1 R:R)
        # f* = (WR × 2 - (1-WR)) / 2 = (3×WR - 1) / 2
        raw_kelly = (3 * recent_wr - 1) / 2
        # Apply safety factor (use half-Kelly)
        adaptive_kelly = max(0.01, min(0.10, raw_kelly * 0.5))

        # 5. ADAPTIVE HOLD TIME (based on OU half-life)
        # Max hold = 3 × half-life (covers 87.5% of expected reversion)
        if self.ou_half_life < 100:
            adaptive_max_hold = self.ou_half_life * 3
        else:
            adaptive_max_hold = 60  # Default 60 seconds
        adaptive_max_hold = max(5, min(120, adaptive_max_hold))

        # 6. ADAPTIVE STOP Z-SCORE
        # In volatile regime: tighter stop
        # In trending regime: wider stop
        if self.current_regime == 'volatile':
            adaptive_stop_z = -2.5
        elif self.current_regime == 'trending':
            adaptive_stop_z = -3.5
        else:
            adaptive_stop_z = -3.0

        # 7. ADAPTIVE MIN HOLD (avoid noise)
        # In volatile regime: shorter min hold
        # In MR regime: longer min hold (wait for reversion)
        if self.current_regime == 'volatile':
            adaptive_min_hold_ms = 200
        elif self.current_regime == 'mean_reversion':
            adaptive_min_hold_ms = 500
        else:
            adaptive_min_hold_ms = 300

        # 8. ADAPTIVE R:R RATIO
        # In MR regime: Lower TP (more likely to hit)
        # In trending: Higher TP (ride the trend)
        if self.current_regime == 'mean_reversion':
            adaptive_tp = 0.0006  # 0.06%
            adaptive_sl = 0.0003  # 0.03%
        elif self.current_regime == 'trending':
            adaptive_tp = 0.0012  # 0.12%
            adaptive_sl = 0.0004  # 0.04%
        else:
            adaptive_tp = 0.0004  # 0.04%
            adaptive_sl = 0.0002  # 0.02%

        # Store all adaptive parameters
        self.adaptive_params = {
            # Core parameters
            'lookback': adaptive_lookback,
            'entry_threshold': adaptive_entry_threshold,
            'exit_z_threshold': adaptive_exit_z,
            'stop_z_threshold': adaptive_stop_z,
            'kelly_frac': adaptive_kelly,

            # Hold time parameters
            'max_hold_sec': adaptive_max_hold,
            'min_hold_ms': adaptive_min_hold_ms,

            # TP/SL parameters
            'profit_target': adaptive_tp,
            'stop_loss': adaptive_sl,

            # Strategy mode
            'strategy_mode': strategy_mode,
            'current_regime': self.current_regime,

            # CRITICAL: Trend direction for momentum trading
            'trend_direction': self.trend_direction,  # 1=up, -1=down, 0=neutral
            'momentum_strength': self.momentum_strength,

            # Debug info
            'hurst': self.hurst,
            'ou_theta': self.ou_theta,
            'ou_half_life': self.ou_half_life,
            'vol_fast': vol_fast,
            'vol_slow': vol_slow,
            'vol_ratio': vol_ratio,
            'regime_probs': self.regime_probs,
            'recent_wr': recent_wr,
        }

        return self.adaptive_params

    def _get_recent_win_rate(self) -> float:
        """Get recent win rate from tracked trades"""
        total = self.recent_wins + self.recent_losses
        if total < 5:
            return 0.5  # Default to 50% with insufficient data
        return self.recent_wins / total

    def record_trade(self, won: bool):
        """Record trade result for adaptive Kelly"""
        self.recent_trades.append(1 if won else 0)
        if won:
            self.recent_wins += 1
        else:
            self.recent_losses += 1

        # Remove oldest if beyond window
        if len(self.recent_trades) > 50:
            oldest = self.recent_trades.popleft()
            if oldest == 1:
                self.recent_wins -= 1
            else:
                self.recent_losses -= 1

    def get_signal_strength(self, z_score: float) -> float:
        """
        Calculate signal strength based on z-score and regime

        In MR regime: Strong signal when z is extreme
        In trending: Strong signal when momentum confirms
        """
        if self.current_regime == 'mean_reversion':
            # In MR: signal strength increases with |z|
            # Entry at z=-2, max strength at z=-3
            strength = min(1.0, (abs(z_score) - 1.5) / 1.5)
        elif self.current_regime == 'trending':
            # In trending: signal strength from momentum
            if len(self.returns) < 10:
                return 0.5
            recent_ret = sum(list(self.returns)[-10:])
            strength = min(1.0, abs(recent_ret) / 0.005)
        else:
            # In volatile: be conservative
            strength = 0.5

        return max(0.1, min(1.0, strength))

    def should_trade(self) -> Tuple[bool, str]:
        """
        Determine if we should trade based on regime confidence

        Returns: (should_trade, reason)
        """
        # Get dominant regime probability
        max_prob = max(self.regime_probs.values())

        # In uncertain regime (no clear signal): don't trade
        if max_prob < 0.5:
            return False, f"UNCERTAIN_REGIME_{max_prob:.2f}"

        # In volatile regime with high probability: reduce trading
        if self.current_regime == 'volatile' and max_prob > 0.7:
            return False, f"VOLATILE_REGIME_{max_prob:.2f}"

        return True, f"{self.current_regime.upper()}_{max_prob:.2f}"


class UniversalMathFormulas:
    """
    Collection of universal mathematical formulas that work across all conditions

    These are the CORE FORMULAS from Renaissance research:
    1. Kelly Criterion (optimal sizing)
    2. OU Process (mean reversion)
    3. Hurst Exponent (trend detection)
    4. Bayesian Regime (state estimation)
    5. EWMA Volatility (risk scaling)
    """

    @staticmethod
    def kelly_criterion(win_rate: float, risk_reward: float = 2.0,
                       safety_factor: float = 0.5) -> float:
        """
        Kelly Criterion: Optimal bet sizing

        f* = (p × b - q) / b
        where:
            p = probability of winning (win_rate)
            q = probability of losing (1 - win_rate)
            b = risk/reward ratio

        With safety_factor (usually 0.5 for half-Kelly)
        """
        q = 1 - win_rate
        kelly = (win_rate * risk_reward - q) / risk_reward
        return max(0.01, min(0.25, kelly * safety_factor))

    @staticmethod
    def ou_expected_value(z_score: float, theta: float, sigma: float,
                         dt: float = 1.0) -> float:
        """
        Ornstein-Uhlenbeck Expected Price Movement

        E[X_{t+dt}] = X_t + θ(μ - X_t)dt

        For z-score: E[z_{t+dt}] = z_t × (1 - θ × dt)
        (Mean is 0 for z-score)
        """
        decay = 1 - theta * dt
        expected_z = z_score * decay
        return expected_z

    @staticmethod
    def probability_of_reversion(z_score: float, target_z: float,
                                 theta: float, sigma: float) -> float:
        """
        Probability that z-score reaches target before stop

        Using OU process: probability of hitting level b before level a
        P(hit b first | X_0 = x) = (e^{-2θx/σ²} - e^{-2θa/σ²}) / (e^{-2θb/σ²} - e^{-2θa/σ²})

        Simplified for z-score with μ=0:
        """
        if theta <= 0 or sigma <= 0:
            return 0.5

        # Stop at z=-3, target at z=target_z
        stop_z = -3.0

        exp_factor = 2 * theta / (sigma ** 2)

        try:
            num = np.exp(-exp_factor * z_score) - np.exp(-exp_factor * stop_z)
            den = np.exp(-exp_factor * target_z) - np.exp(-exp_factor * stop_z)
            prob = num / den if den != 0 else 0.5
        except (OverflowError, RuntimeWarning):
            prob = 0.5

        return max(0.0, min(1.0, prob))

    @staticmethod
    def expected_holding_time(z_score: float, theta: float) -> float:
        """
        Expected time to reach mean (z=0) from current z-score

        For OU process: E[τ] = (1/θ) × ln(|z|/ε) where ε is a small tolerance

        Simplified: τ ≈ half_life × log2(|z|)
        """
        if theta <= 0:
            return float('inf')

        half_life = np.log(2) / theta

        if abs(z_score) < 0.1:
            return 0

        expected_time = half_life * np.log2(abs(z_score))
        return max(0, expected_time)

    @staticmethod
    def compound_growth_formula(initial_capital: float, win_rate: float,
                                risk_reward: float, num_trades: int,
                                position_size: float = 0.02) -> float:
        """
        Expected compound growth from trading

        E[Capital_n] = C_0 × (1 + position_size × (WR × TP - (1-WR) × SL))^n

        With Kelly optimal sizing and edge per trade.
        """
        # Edge per trade
        tp_pct = position_size * risk_reward / (1 + risk_reward)  # TP percentage
        sl_pct = position_size / (1 + risk_reward)  # SL percentage

        edge_per_trade = win_rate * tp_pct - (1 - win_rate) * sl_pct

        # Compound growth
        final_capital = initial_capital * ((1 + edge_per_trade) ** num_trades)

        return final_capital

    @staticmethod
    def trades_to_target(initial_capital: float, target_capital: float,
                        win_rate: float, risk_reward: float,
                        position_size: float = 0.02) -> int:
        """
        Calculate number of trades needed to reach target

        n = ln(target/initial) / ln(1 + edge)
        """
        # Edge per trade
        tp_pct = position_size * risk_reward / (1 + risk_reward)
        sl_pct = position_size / (1 + risk_reward)
        edge_per_trade = win_rate * tp_pct - (1 - win_rate) * sl_pct

        if edge_per_trade <= 0:
            return -1  # Impossible with negative edge

        ratio = target_capital / initial_capital
        n = np.log(ratio) / np.log(1 + edge_per_trade)

        return int(np.ceil(n))


# Singleton instance for global access
_adaptive_engine = None

def get_adaptive_engine() -> AdaptiveMathEngine:
    """Get or create the singleton adaptive engine"""
    global _adaptive_engine
    if _adaptive_engine is None:
        _adaptive_engine = AdaptiveMathEngine()
    return _adaptive_engine


def reset_adaptive_engine():
    """Reset the adaptive engine (for testing)"""
    global _adaptive_engine
    _adaptive_engine = None
