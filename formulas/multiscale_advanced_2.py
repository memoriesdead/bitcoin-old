"""
Multi-Scale Advanced Formulas Part 2 (IDs 381-400)
==================================================
Additional academic formulas for complete time-scale adaptation.

Research Sources:
    - Kelly criterion extensions and HJB equations
    - Almgren-Chriss price impact
    - Return scaling and moment structure
    - Adaptive bandwidth selection
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from .base import BaseFormula, FormulaRegistry


# =============================================================================
# KELLY CRITERION EXTENSIONS (381-385)
# =============================================================================

@FormulaRegistry.register(381)
class HorizonKelly(BaseFormula):
    """
    ID 381: Horizon-Dependent Kelly Fraction

    Academic Reference:
        - Merton (1990) continuous-time portfolio selection
        - Kelly fraction adjusted for holding period

    f*(T) = (mu - r) / (sigma^2) * g(T, theta)
    where g accounts for mean-reversion speed over horizon T.
    """

    NAME = "HorizonKelly"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Kelly fraction adjusted for holding horizon"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.horizons = [1, 5, 10, 20, 40]
        self.kelly_by_horizon = {}
        self.optimal_horizon = 10
        self.optimal_kelly = 0.0

    def _estimate_kelly(self, returns: np.ndarray, horizon: int) -> float:
        """Estimate Kelly fraction for given horizon"""
        n = len(returns) // horizon
        if n < 10:
            return 0.0

        # Aggregate returns at horizon
        agg_returns = np.array([
            np.sum(returns[i*horizon:(i+1)*horizon])
            for i in range(n)
        ])

        mu = np.mean(agg_returns)
        var = np.var(agg_returns)

        if var < 1e-10:
            return 0.0

        # Kelly fraction: f* = mu / var
        kelly = mu / var

        # Clip to reasonable bounds
        return np.clip(kelly, -2, 2)

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()

        self.kelly_by_horizon = {}
        best_sharpe = -np.inf
        self.optimal_horizon = self.horizons[0]

        for h in self.horizons:
            kelly = self._estimate_kelly(returns, h)
            self.kelly_by_horizon[h] = kelly

            # Compute expected Sharpe at this horizon
            n = len(returns) // h
            if n < 10:
                continue

            agg_returns = np.array([
                np.sum(returns[i*h:(i+1)*h])
                for i in range(n)
            ])

            sharpe = np.mean(agg_returns) / (np.std(agg_returns) + 1e-10)
            sharpe_annualized = sharpe * np.sqrt(252 / h)

            if sharpe_annualized > best_sharpe:
                best_sharpe = sharpe_annualized
                self.optimal_horizon = h
                self.optimal_kelly = kelly

        # Signal based on optimal horizon Kelly
        if abs(self.optimal_kelly) > 0.1:
            self.signal = 1 if self.optimal_kelly > 0 else -1
            self.confidence = min(abs(self.optimal_kelly), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(382)
class ContinuousKellyHJB(BaseFormula):
    """
    ID 382: Continuous-Time Kelly via HJB Approximation

    Academic Reference:
        - Merton (1969) "Lifetime portfolio selection under uncertainty"
        - HJB equation: max_f {f*mu - f^2*sigma^2/2 + V_t}

    Solves for optimal leverage in continuous time.
    """

    NAME = "ContinuousKellyHJB"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "HJB-based continuous Kelly optimization"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.risk_aversion = kwargs.get('risk_aversion', 1.0)
        self.optimal_fraction = 0.0
        self.myopic_kelly = 0.0
        self.hedging_demand = 0.0

    def _estimate_drift_and_vol(self, returns: np.ndarray) -> Tuple[float, float]:
        """Estimate drift and volatility"""
        mu = np.mean(returns)
        sigma = np.std(returns)
        return mu, sigma

    def _compute(self) -> None:
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # Estimate parameters
        mu, sigma = self._estimate_drift_and_vol(returns)

        if sigma < 1e-10:
            self.signal = 0
            self.confidence = 0.3
            return

        # Myopic Kelly (ignoring parameter uncertainty)
        self.myopic_kelly = mu / (sigma**2 + 1e-10)

        # Check for mean-reversion (hedging demand adjustment)
        # If returns show autocorrelation, adjust Kelly
        if len(returns) > 20:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if not np.isnan(autocorr):
                # Negative autocorr = mean reversion = reduce leverage
                self.hedging_demand = -autocorr * self.myopic_kelly * 0.3

        # Full optimal fraction
        self.optimal_fraction = self.myopic_kelly + self.hedging_demand

        # Risk aversion adjustment
        self.optimal_fraction /= self.risk_aversion

        # Clip to reasonable bounds
        self.optimal_fraction = np.clip(self.optimal_fraction, -2, 2)

        # Trading signal
        if abs(self.optimal_fraction) > 0.2:
            self.signal = 1 if self.optimal_fraction > 0 else -1
            self.confidence = min(abs(self.optimal_fraction) / 2, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(383)
class FractionalKelly(BaseFormula):
    """
    ID 383: Fractional Kelly with Multi-Scale Constraints

    Uses different Kelly fractions at different time scales.
    Aggregate positions across scales with risk budget.
    """

    NAME = "FractionalKelly"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Multi-scale fractional Kelly allocation"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.scales = [1, 5, 10, 20]
        self.kelly_fractions = {}
        self.scale_weights = {}
        self.combined_kelly = 0.0
        self.total_risk_budget = kwargs.get('risk_budget', 1.0)

    def _compute(self) -> None:
        if len(self.returns) < 60:
            return

        returns = self._returns_array()

        self.kelly_fractions = {}
        self.scale_weights = {}
        sharpe_ratios = {}

        for scale in self.scales:
            n = len(returns) // scale
            if n < 10:
                continue

            agg = np.array([np.sum(returns[i*scale:(i+1)*scale]) for i in range(n)])

            mu = np.mean(agg)
            var = np.var(agg) + 1e-10

            kelly = mu / var
            sharpe = mu / (np.sqrt(var) + 1e-10)

            self.kelly_fractions[scale] = np.clip(kelly, -2, 2)
            sharpe_ratios[scale] = abs(sharpe)

        if not self.kelly_fractions:
            self.signal = 0
            self.confidence = 0.3
            return

        # Weight scales by Sharpe ratio
        total_sharpe = sum(sharpe_ratios.values()) + 1e-10
        for scale in self.kelly_fractions:
            self.scale_weights[scale] = sharpe_ratios[scale] / total_sharpe

        # Combined Kelly (weighted average)
        self.combined_kelly = sum(
            self.kelly_fractions[s] * self.scale_weights[s]
            for s in self.kelly_fractions
        )

        # Apply risk budget
        self.combined_kelly *= self.total_risk_budget

        # Signal
        if abs(self.combined_kelly) > 0.1:
            self.signal = 1 if self.combined_kelly > 0 else -1
            self.confidence = min(abs(self.combined_kelly), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(384)
class DrawdownConstrainedKelly(BaseFormula):
    """
    ID 384: Kelly with Drawdown Constraint

    Academic Reference:
        - Grossman & Zhou (1993) "Optimal Investment Strategies with
          Leverage Constraints"

    Reduces Kelly when drawdown exceeds threshold.
    """

    NAME = "DrawdownConstrainedKelly"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Kelly reduced based on current drawdown"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.max_drawdown_limit = kwargs.get('max_dd', 0.1)
        self.full_kelly = 0.0
        self.adjusted_kelly = 0.0
        self.current_drawdown = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 30:
            return

        returns = self._returns_array()

        # Calculate full Kelly
        mu = np.mean(returns)
        var = np.var(returns) + 1e-10
        self.full_kelly = mu / var

        # Calculate current drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        self.current_drawdown = drawdowns[-1] if len(drawdowns) > 0 else 0

        # Adjust Kelly based on drawdown
        if self.current_drawdown > self.max_drawdown_limit:
            # Proportionally reduce Kelly
            reduction = min(self.current_drawdown / self.max_drawdown_limit, 3)
            self.adjusted_kelly = self.full_kelly / reduction
        else:
            self.adjusted_kelly = self.full_kelly

        self.adjusted_kelly = np.clip(self.adjusted_kelly, -2, 2)

        # Signal
        if abs(self.adjusted_kelly) > 0.1:
            self.signal = 1 if self.adjusted_kelly > 0 else -1
            self.confidence = min(abs(self.adjusted_kelly), 1.0) * (1 - self.current_drawdown)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(385)
class VolatilityScaledKelly(BaseFormula):
    """
    ID 385: Volatility-Scaled Kelly Across Time Horizons

    Adjusts Kelly based on volatility scaling:
    sigma(T) = sigma(1) * T^H
    where H is the Hurst exponent.
    """

    NAME = "VolatilityScaledKelly"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Kelly with volatility scaling correction"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.hurst = 0.5
        self.base_kelly = 0.0
        self.scaled_kelly = {}

    def _estimate_hurst(self, returns: np.ndarray) -> float:
        """Quick Hurst estimation"""
        if len(returns) < 32:
            return 0.5

        var1 = np.var(returns)
        n8 = len(returns) // 8
        agg8 = np.array([np.sum(returns[i*8:(i+1)*8]) for i in range(n8)])
        var8 = np.var(agg8) if n8 > 1 else var1 * 8

        if var1 > 0:
            H = np.log(var8 / (8 * var1)) / (2 * np.log(8)) + 0.5
            return np.clip(H, 0.1, 0.9)
        return 0.5

    def _compute(self) -> None:
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # Estimate Hurst
        self.hurst = self._estimate_hurst(returns)

        # Base Kelly at 1-period
        mu = np.mean(returns)
        var = np.var(returns) + 1e-10
        self.base_kelly = mu / var

        # Scale Kelly for different horizons using Hurst
        horizons = [1, 5, 10, 20]
        self.scaled_kelly = {}

        for T in horizons:
            # Volatility scales as T^H
            # Variance scales as T^(2H)
            # Kelly_T = mu_T / var_T = (T * mu) / (T^(2H) * var) = Kelly_1 * T^(1-2H)
            vol_scaling = T ** (1 - 2 * self.hurst)
            self.scaled_kelly[T] = self.base_kelly * vol_scaling

        # Use Hurst to determine best horizon
        # H > 0.5: longer horizons more favorable
        # H < 0.5: shorter horizons better
        if self.hurst > 0.55:
            best_horizon = max(horizons)
        elif self.hurst < 0.45:
            best_horizon = min(horizons)
        else:
            best_horizon = 10

        optimal_kelly = self.scaled_kelly.get(best_horizon, self.base_kelly)
        optimal_kelly = np.clip(optimal_kelly, -2, 2)

        if abs(optimal_kelly) > 0.1:
            self.signal = 1 if optimal_kelly > 0 else -1
            self.confidence = min(abs(optimal_kelly), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


# =============================================================================
# PRICE IMPACT & EXECUTION (386-390)
# =============================================================================

@FormulaRegistry.register(386)
class KyleLambdaScaling(BaseFormula):
    """
    ID 386: Kyle's Lambda Price Impact Scaling

    Academic Reference:
        - Kyle (1985) "Continuous Auctions and Insider Trading"

    Lambda = price impact per unit traded
    Studies how lambda scales with time and trade size.
    """

    NAME = "KyleLambdaScaling"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Kyle lambda price impact by time scale"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.lambda_estimate = 0.0
        self.impact_decay = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 50 or len(self.volumes) < 50:
            return

        returns = self._returns_array()
        volumes = self._volumes_array()

        # Estimate lambda: price change = lambda * signed_volume
        # Proxy: abs(return) = lambda * volume

        abs_returns = np.abs(returns)
        volumes_safe = volumes + 1e-10

        # Simple regression
        if np.sum(volumes_safe) > 0:
            self.lambda_estimate = np.mean(abs_returns) / np.mean(volumes_safe)

        # Impact decay: how quickly does temporary impact revert?
        # Check autocorrelation of return/volume residuals
        if len(returns) > 20:
            residuals = abs_returns - self.lambda_estimate * volumes_safe
            autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            if not np.isnan(autocorr):
                self.impact_decay = 1 - abs(autocorr)

        # Trading signal based on impact regime
        # High lambda = illiquid, trade with caution
        # Low lambda = liquid, can be more aggressive

        current_volume = volumes[-1] if len(volumes) > 0 else 0
        expected_impact = self.lambda_estimate * current_volume

        if expected_impact < 0.001:  # Low impact expected
            # Can follow signal more aggressively
            self.signal = 1 if returns[-1] > 0 else -1
            self.confidence = 0.6
        elif expected_impact > 0.01:  # High impact
            # Be cautious
            self.signal = 0
            self.confidence = 0.3
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(387)
class AlmgrenChrissTiming(BaseFormula):
    """
    ID 387: Almgren-Chriss Optimal Execution Timing

    Academic Reference:
        - Almgren & Chriss (2000) "Optimal execution of portfolio transactions"

    Determines if now is a good time to execute based on
    permanent/temporary impact tradeoff.
    """

    NAME = "AlmgrenChrissTiming"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Almgren-Chriss execution timing signal"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.risk_aversion = kwargs.get('risk_aversion', 1e-6)
        self.urgency = 0.5  # 0 = patient, 1 = urgent

    def _compute(self) -> None:
        if len(self.returns) < 30:
            return

        returns = self._returns_array()
        volatility = np.std(returns)

        # Estimate timing risk
        # Higher volatility = higher risk of price moving against us
        vol_percentile = np.percentile(np.abs(returns), 90)
        current_vol = np.std(returns[-10:])

        # Urgency adjustment based on recent volatility
        if current_vol > vol_percentile:
            # High volatility - more urgent to execute
            self.urgency = 0.8
        elif current_vol < vol_percentile * 0.5:
            # Low volatility - can be patient
            self.urgency = 0.3
        else:
            self.urgency = 0.5

        # Signal based on urgency and trend
        trend = np.mean(returns[-5:])

        if self.urgency > 0.6:
            # Urgent - follow momentum
            self.signal = 1 if trend > 0 else -1
            self.confidence = self.urgency
        else:
            # Patient - wait for better entry
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(388)
class TransientImpactDecay(BaseFormula):
    """
    ID 388: Transient Price Impact Decay Analysis

    Academic Reference:
        - Bouchaud et al. (2004) "Fluctuations and response in financial markets"
        - Gatheral (2010) "No-dynamic-arbitrage and market impact"

    Models how temporary price impact decays over time.
    G(t) = impact propagator function.
    """

    NAME = "TransientImpactDecay"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Transient impact decay analysis"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.decay_rate = 0.5
        self.permanent_ratio = 0.3

    def _compute(self) -> None:
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # Estimate impact decay using return autocorrelation
        # Negative autocorr = impact reverting
        # Positive autocorr = impact persisting

        lags = [1, 2, 5, 10]
        autocorrs = []

        for lag in lags:
            if len(returns) > lag + 10:
                corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append((lag, corr))

        if len(autocorrs) < 2:
            self.signal = 0
            self.confidence = 0.3
            return

        # Fit exponential decay: corr(lag) = a * exp(-b * lag)
        lag_arr = np.array([a[0] for a in autocorrs])
        corr_arr = np.array([a[1] for a in autocorrs])

        # Simple estimate
        if corr_arr[0] != 0:
            self.decay_rate = -np.log(abs(corr_arr[-1] / (corr_arr[0] + 1e-10))) / (lag_arr[-1] - lag_arr[0])
            self.decay_rate = np.clip(self.decay_rate, 0.01, 2.0)

        # Permanent ratio = asymptotic correlation
        self.permanent_ratio = abs(corr_arr[-1])

        # Trading signal
        # Fast decay + low permanent = good for mean reversion
        # Slow decay + high permanent = momentum

        if self.decay_rate > 0.3 and self.permanent_ratio < 0.2:
            # Mean reversion favorable
            z = returns[-1] / (np.std(returns) + 1e-10)
            self.signal = -1 if z > 1 else (1 if z < -1 else 0)
            self.confidence = min(self.decay_rate, 1.0)
        elif self.decay_rate < 0.1 and self.permanent_ratio > 0.3:
            # Momentum favorable
            self.signal = 1 if returns[-1] > 0 else -1
            self.confidence = self.permanent_ratio
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(389)
class SquareRootImpact(BaseFormula):
    """
    ID 389: Square-Root Price Impact Law

    Academic Reference:
        - Almgren et al. (2005) "Direct estimation of equity market impact"
        - Impact ~ sigma * sqrt(Q/V)

    Tests if impact follows square-root law.
    """

    NAME = "SquareRootImpact"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Square-root impact law estimation"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.impact_exponent = 0.5  # Should be ~0.5 for square-root
        self.impact_coefficient = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 50 or len(self.volumes) < 50:
            return

        returns = self._returns_array()
        volumes = self._volumes_array()

        # Avoid division by zero
        volumes = volumes + 1e-10
        avg_volume = np.mean(volumes)

        # Normalized volume
        norm_volume = volumes / avg_volume

        # Log-log regression: log(|return|) = a + b * log(volume)
        abs_returns = np.abs(returns) + 1e-10
        valid = (norm_volume > 0.1) & (abs_returns > 1e-10)

        if np.sum(valid) < 20:
            self.signal = 0
            self.confidence = 0.3
            return

        log_vol = np.log(norm_volume[valid])
        log_ret = np.log(abs_returns[valid])

        try:
            coeffs = np.polyfit(log_vol, log_ret, 1)
            self.impact_exponent = coeffs[0]
            self.impact_coefficient = np.exp(coeffs[1])
        except:
            self.impact_exponent = 0.5
            self.impact_coefficient = 0.0

        # Trading signal based on impact regime
        # Exponent < 0.5: Impact less than expected (good for trading)
        # Exponent > 0.5: Impact more than expected (be cautious)

        current_volume = volumes[-1]
        expected_impact = self.impact_coefficient * (current_volume / avg_volume) ** self.impact_exponent

        if self.impact_exponent < 0.4:
            # Low impact regime
            self.signal = 1 if returns[-1] > 0 else -1
            self.confidence = 0.6
        elif self.impact_exponent > 0.6:
            # High impact regime
            self.signal = 0
            self.confidence = 0.3
        else:
            # Normal regime
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(390)
class MarketResiliency(BaseFormula):
    """
    ID 390: Market Resiliency Estimation

    How quickly do prices recover from large moves?
    High resiliency = fast recovery = good for mean reversion
    Low resiliency = slow recovery = momentum continues
    """

    NAME = "MarketResiliency"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Market resiliency and recovery speed"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.resiliency = 0.5  # 0 = no recovery, 1 = instant recovery
        self.recovery_time = 10

    def _compute(self) -> None:
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # Find large moves (above 90th percentile)
        threshold = np.percentile(np.abs(returns), 90)
        large_moves = np.where(np.abs(returns) > threshold)[0]

        if len(large_moves) < 3:
            self.signal = 0
            self.confidence = 0.3
            return

        # Track recovery after each large move
        recovery_ratios = []

        for idx in large_moves:
            if idx + 10 >= len(returns):
                continue

            move = returns[idx]
            subsequent = returns[idx+1:idx+11]

            # How much was reversed in next 10 periods?
            cumulative_reversal = -np.sign(move) * np.sum(subsequent)
            recovery_ratio = cumulative_reversal / (abs(move) + 1e-10)
            recovery_ratios.append(recovery_ratio)

        if recovery_ratios:
            self.resiliency = np.mean(recovery_ratios)
            self.resiliency = np.clip(self.resiliency, 0, 1)

            # Estimate recovery time
            # Time to recover 50% of move
            for t in range(1, 11):
                partial_recovery = [
                    -np.sign(returns[idx]) * np.sum(returns[idx+1:idx+t+1]) / (abs(returns[idx]) + 1e-10)
                    for idx in large_moves if idx + t < len(returns)
                ]
                if partial_recovery and np.mean(partial_recovery) > 0.5:
                    self.recovery_time = t
                    break

        # Trading signal
        if self.resiliency > 0.6:
            # High resiliency - mean reversion works
            z = returns[-1] / (np.std(returns) + 1e-10)
            self.signal = -1 if z > 1 else (1 if z < -1 else 0)
            self.confidence = self.resiliency
        elif self.resiliency < 0.3:
            # Low resiliency - momentum continues
            self.signal = 1 if returns[-1] > 0 else -1
            self.confidence = 1 - self.resiliency
        else:
            self.signal = 0
            self.confidence = 0.4


# =============================================================================
# RETURN SCALING (391-395)
# =============================================================================

@FormulaRegistry.register(391)
class MomentScaling(BaseFormula):
    """
    ID 391: Moment Scaling Analysis

    Academic Reference:
        - Di Matteo (2007) "Multi-scaling in finance"
        - E[|r|^q] ~ T^(zeta(q))

    Studies how moments scale with aggregation.
    zeta(q) = H*q for monofractal, nonlinear for multifractal.
    """

    NAME = "MomentScaling"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Moment scaling function zeta(q)"

    def __init__(self, lookback: int = 256, **kwargs):
        super().__init__(lookback, **kwargs)
        self.q_values = [0.5, 1, 2, 3, 4]
        self.zeta = {}  # Scaling exponents
        self.multifractal_deviation = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 130:
            return

        returns = self._returns_array()
        scales = [1, 2, 4, 8, 16, 32]

        self.zeta = {}

        for q in self.q_values:
            moments = []

            for T in scales:
                n = len(returns) // T
                if n < 10:
                    continue

                agg = np.array([np.sum(returns[i*T:(i+1)*T]) for i in range(n)])
                Mq = np.mean(np.power(np.abs(agg) + 1e-10, q))

                if Mq > 0:
                    moments.append((np.log(T), np.log(Mq)))

            if len(moments) >= 3:
                log_T = np.array([m[0] for m in moments])
                log_M = np.array([m[1] for m in moments])
                self.zeta[q] = np.polyfit(log_T, log_M, 1)[0]

        if len(self.zeta) < 3:
            self.signal = 0
            self.confidence = 0.3
            return

        # Check for multifractality
        # For monofractal: zeta(q) = H*q (linear)
        # Deviation from linearity = multifractality

        q_arr = np.array(list(self.zeta.keys()))
        zeta_arr = np.array(list(self.zeta.values()))

        # Linear fit
        coeffs = np.polyfit(q_arr, zeta_arr, 1)
        H_estimate = coeffs[0]
        linear_fit = np.polyval(coeffs, q_arr)

        self.multifractal_deviation = np.std(zeta_arr - linear_fit)

        # Trading signal
        if self.multifractal_deviation > 0.1:
            # Strong multifractality - complex regime
            self.signal = 0
            self.confidence = 0.3
        elif H_estimate > 0.55:
            # Trending
            self.signal = 1 if returns[-1] > 0 else -1
            self.confidence = min((H_estimate - 0.5) * 4, 1.0)
        elif H_estimate < 0.45:
            # Mean reverting
            z = returns[-1] / (np.std(returns) + 1e-10)
            self.signal = -1 if z > 1 else (1 if z < -1 else 0)
            self.confidence = min((0.5 - H_estimate) * 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(392)
class TailIndexStability(BaseFormula):
    """
    ID 392: Tail Index Stability Across Scales

    Academic Reference:
        - Hill estimator for tail index
        - Stable distributions have constant tail index

    Checks if tail behavior is consistent across time scales.
    """

    NAME = "TailIndexStability"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Tail index consistency across scales"

    def __init__(self, lookback: int = 256, **kwargs):
        super().__init__(lookback, **kwargs)
        self.scales = [1, 5, 10, 20]
        self.tail_index_by_scale = {}
        self.tail_stability = 0.0

    def _hill_estimator(self, data: np.ndarray, k: int = None) -> float:
        """Hill estimator for tail index"""
        n = len(data)
        if k is None:
            k = int(np.sqrt(n))
        k = min(k, n // 2)
        if k < 5:
            return 2.0  # Default

        sorted_data = np.sort(np.abs(data))[::-1]
        log_ratio = np.log(sorted_data[:k] / sorted_data[k])

        if np.mean(log_ratio) > 0:
            return 1 / np.mean(log_ratio)
        return 2.0

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()

        self.tail_index_by_scale = {}

        for scale in self.scales:
            n = len(returns) // scale
            if n < 30:
                continue

            agg = np.array([np.sum(returns[i*scale:(i+1)*scale]) for i in range(n)])
            alpha = self._hill_estimator(agg)
            self.tail_index_by_scale[scale] = alpha

        if len(self.tail_index_by_scale) < 2:
            self.signal = 0
            self.confidence = 0.3
            return

        # Stability = inverse of std of tail indices
        alphas = list(self.tail_index_by_scale.values())
        self.tail_stability = 1 / (np.std(alphas) + 0.1)

        # Mean tail index
        mean_alpha = np.mean(alphas)

        # Trading signal
        # High alpha (>4) = thin tails, normal regime
        # Low alpha (2-3) = fat tails, extreme events possible

        if mean_alpha < 2.5 and self.tail_stability < 0.5:
            # Unstable fat tails - risky
            self.signal = 0
            self.confidence = 0.2
        elif mean_alpha > 4 and self.tail_stability > 1:
            # Stable thin tails - normal trading
            self.signal = 1 if returns[-1] > 0 else -1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(393)
class ReturnAggregationTest(BaseFormula):
    """
    ID 393: Return Aggregation Bias Test

    Tests for systematic biases when aggregating returns.
    Compares sum vs product of returns.
    """

    NAME = "ReturnAggregationTest"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Test for aggregation bias in returns"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.aggregation_bias = 0.0
        self.log_vs_simple = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # Compare arithmetic vs geometric aggregation
        scales = [5, 10, 20]

        arithmetic_means = []
        geometric_means = []

        for scale in scales:
            n = len(returns) // scale
            if n < 10:
                continue

            # Arithmetic sum
            arith_agg = np.array([np.sum(returns[i*scale:(i+1)*scale]) for i in range(n)])

            # Geometric (log) return
            prices_proxy = np.cumprod(1 + returns + 1e-10)
            geo_agg = np.array([
                prices_proxy[(i+1)*scale - 1] / prices_proxy[i*scale] - 1
                for i in range(n) if (i+1)*scale < len(prices_proxy)
            ])

            if len(arith_agg) > 0 and len(geo_agg) > 0:
                arithmetic_means.append(np.mean(arith_agg))
                geometric_means.append(np.mean(geo_agg))

        if arithmetic_means and geometric_means:
            # Aggregation bias = difference between methods
            self.aggregation_bias = np.mean(np.array(arithmetic_means) - np.array(geometric_means))
            self.log_vs_simple = np.corrcoef(arithmetic_means, geometric_means)[0, 1] if len(arithmetic_means) > 1 else 1.0

        # Trading signal
        # High correlation (>0.9) = no bias, standard methods work
        # Low correlation = need to be careful with aggregation

        if abs(self.log_vs_simple) > 0.9:
            self.signal = 1 if returns[-1] > 0 else -1
            self.confidence = 0.6
        else:
            # Significant aggregation effects
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(394)
class ScalePredictabilityLoss(BaseFormula):
    """
    ID 394: Predictability Loss Rate with Aggregation

    How quickly does predictability decay as we aggregate?
    Fast decay = high-frequency alpha decays quickly.
    """

    NAME = "ScalePredictabilityLoss"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Predictability decay across scales"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.scales = [1, 2, 5, 10, 20]
        self.predictability_by_scale = {}
        self.decay_rate = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 60:
            return

        returns = self._returns_array()

        self.predictability_by_scale = {}

        for scale in self.scales:
            n = len(returns) // scale
            if n < 20:
                continue

            agg = np.array([np.sum(returns[i*scale:(i+1)*scale]) for i in range(n)])

            # Predictability = abs(autocorrelation at lag 1)
            autocorr = np.corrcoef(agg[:-1], agg[1:])[0, 1]
            if not np.isnan(autocorr):
                self.predictability_by_scale[scale] = abs(autocorr)

        if len(self.predictability_by_scale) < 3:
            self.signal = 0
            self.confidence = 0.3
            return

        # Fit decay: predictability(s) = a * exp(-b * s)
        scales = np.array(list(self.predictability_by_scale.keys()))
        pred = np.array(list(self.predictability_by_scale.values()))

        log_pred = np.log(pred + 1e-10)
        coeffs = np.polyfit(scales, log_pred, 1)
        self.decay_rate = -coeffs[0]

        # Trading signal
        # Fast decay = trade at short scale
        # Slow decay = can trade at any scale

        best_scale = scales[0]  # Default to shortest
        best_pred = pred[0]

        for s, p in zip(scales, pred):
            if p > 0.1 and p >= best_pred:
                best_scale = s
                best_pred = p

        if best_pred > 0.15:
            recent_agg = np.sum(returns[-int(best_scale):])
            self.signal = 1 if recent_agg > 0 else -1
            self.confidence = min(best_pred * 2, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(395)
class UniversalityClass(BaseFormula):
    """
    ID 395: Universality Class Detection

    Academic Reference:
        - Cont (2001) "Empirical properties of asset returns"
        - Stanley et al. universality in econophysics

    Detects which universality class the market belongs to.
    """

    NAME = "UniversalityClass"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Detect market universality class"

    def __init__(self, lookback: int = 256, **kwargs):
        super().__init__(lookback, **kwargs)
        self.hurst = 0.5
        self.tail_index = 3.0
        self.volatility_clustering = 0.0
        self.market_class = 'standard'

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()

        # 1. Estimate Hurst
        var1 = np.var(returns)
        n8 = len(returns) // 8
        if n8 > 2 and var1 > 0:
            agg8 = np.array([np.sum(returns[i*8:(i+1)*8]) for i in range(n8)])
            var8 = np.var(agg8)
            self.hurst = np.log(var8 / (8 * var1 + 1e-10)) / (2 * np.log(8)) + 0.5
            self.hurst = np.clip(self.hurst, 0.1, 0.9)

        # 2. Tail index (simplified Hill)
        sorted_abs = np.sort(np.abs(returns))[::-1]
        k = max(int(np.sqrt(len(returns))), 5)
        log_ratios = np.log(sorted_abs[:k] / (sorted_abs[k] + 1e-10) + 1e-10)
        if np.mean(log_ratios) > 0:
            self.tail_index = 1 / np.mean(log_ratios)
        self.tail_index = np.clip(self.tail_index, 1.5, 6)

        # 3. Volatility clustering (GARCH effect)
        abs_returns = np.abs(returns)
        if len(abs_returns) > 10:
            vol_autocorr = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
            if not np.isnan(vol_autocorr):
                self.volatility_clustering = vol_autocorr

        # Classify market
        if self.hurst > 0.6 and self.tail_index > 3:
            self.market_class = 'trending_stable'
        elif self.hurst < 0.4 and self.tail_index > 3:
            self.market_class = 'mean_reverting'
        elif self.tail_index < 2.5:
            self.market_class = 'fat_tailed'
        elif self.volatility_clustering > 0.3:
            self.market_class = 'clustered_volatility'
        else:
            self.market_class = 'standard'

        # Trading signal based on class
        if self.market_class == 'trending_stable':
            self.signal = 1 if returns[-1] > 0 else -1
            self.confidence = 0.7
        elif self.market_class == 'mean_reverting':
            z = returns[-1] / (np.std(returns) + 1e-10)
            self.signal = -1 if z > 1 else (1 if z < -1 else 0)
            self.confidence = 0.6
        elif self.market_class == 'fat_tailed':
            self.signal = 0  # Avoid extreme risk
            self.confidence = 0.3
        elif self.market_class == 'clustered_volatility':
            # Mean revert in high vol, momentum in low vol
            current_vol = np.std(returns[-10:])
            hist_vol = np.std(returns[:-10])
            if current_vol > 1.5 * hist_vol:
                z = returns[-1] / (current_vol + 1e-10)
                self.signal = -1 if z > 1 else (1 if z < -1 else 0)
            else:
                self.signal = 1 if returns[-1] > 0 else -1
            self.confidence = 0.5
        else:
            self.signal = 0
            self.confidence = 0.4


# =============================================================================
# ADDITIONAL FORMULAS (396-400)
# =============================================================================

@FormulaRegistry.register(396)
class AdaptiveBandwidth(BaseFormula):
    """
    ID 396: Adaptive Bandwidth Selection

    Automatically selects optimal smoothing bandwidth
    based on current market conditions.
    """

    NAME = "AdaptiveBandwidth"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Adaptive bandwidth for market smoothing"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.bandwidths = [3, 5, 10, 20, 40]
        self.optimal_bandwidth = 10
        self.smoothed_signal = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # Find bandwidth with best SNR (signal to noise ratio)
        best_snr = 0
        self.optimal_bandwidth = self.bandwidths[0]

        for bw in self.bandwidths:
            if len(returns) < bw * 2:
                continue

            # Smoothed signal
            smoothed = np.convolve(returns, np.ones(bw)/bw, mode='valid')

            # Signal = trend strength
            signal_strength = abs(np.mean(smoothed[-5:]))

            # Noise = variability around smooth
            noise = np.std(returns[-bw:] - smoothed[-1])

            snr = signal_strength / (noise + 1e-10)

            if snr > best_snr:
                best_snr = snr
                self.optimal_bandwidth = bw
                self.smoothed_signal = smoothed[-1]

        # Trading signal from smoothed data
        if abs(self.smoothed_signal) > np.std(returns) * 0.5:
            self.signal = 1 if self.smoothed_signal > 0 else -1
            self.confidence = min(best_snr, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(397)
class ScaleCoherence(BaseFormula):
    """
    ID 397: Scale Coherence Measure

    Measures agreement between signals at different scales.
    High coherence = all scales agree = strong signal.
    """

    NAME = "ScaleCoherence"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Signal coherence across time scales"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.scales = [1, 3, 5, 10, 20]
        self.signals_by_scale = {}
        self.coherence = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        self.signals_by_scale = {}

        for scale in self.scales:
            if len(returns) < scale * 2:
                continue

            # Signal at this scale
            recent = np.sum(returns[-scale:])
            std = np.std(returns) * np.sqrt(scale)
            z = recent / (std + 1e-10)

            self.signals_by_scale[scale] = np.sign(z) if abs(z) > 0.5 else 0

        if len(self.signals_by_scale) < 3:
            self.signal = 0
            self.confidence = 0.3
            return

        # Coherence = fraction of scales that agree
        signals = list(self.signals_by_scale.values())
        nonzero = [s for s in signals if s != 0]

        if nonzero:
            dominant = np.sign(np.sum(nonzero))
            agreement = sum(1 for s in nonzero if s == dominant) / len(nonzero)
            self.coherence = agreement

            if self.coherence > 0.7 and len(nonzero) >= 3:
                self.signal = int(dominant)
                self.confidence = self.coherence
            else:
                self.signal = 0
                self.confidence = 0.4
        else:
            self.coherence = 0
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(398)
class MultiScaleEnsemble(BaseFormula):
    """
    ID 398: Multi-Scale Ensemble Signal

    Combines signals from all scales with adaptive weighting.
    """

    NAME = "MultiScaleEnsemble"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Ensemble signal from multiple scales"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.scales = [1, 2, 5, 10, 20, 40]
        self.weights = {}
        self.ensemble_signal = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()

        signals = {}
        sharpes = {}

        for scale in self.scales:
            n = len(returns) // scale
            if n < 20:
                continue

            agg = np.array([np.sum(returns[i*scale:(i+1)*scale]) for i in range(n)])

            # Simple momentum signal
            recent = agg[-1]
            sig = np.sign(recent) if abs(recent) > np.std(agg) * 0.5 else 0
            signals[scale] = sig

            # Weight by Sharpe
            sharpe = np.mean(agg) / (np.std(agg) + 1e-10)
            sharpes[scale] = abs(sharpe)

        if not signals:
            self.signal = 0
            self.confidence = 0.3
            return

        # Normalize weights
        total_sharpe = sum(sharpes.values()) + 1e-10
        self.weights = {s: sharpes[s] / total_sharpe for s in signals}

        # Weighted ensemble
        self.ensemble_signal = sum(
            signals[s] * self.weights[s]
            for s in signals
        )

        if abs(self.ensemble_signal) > 0.3:
            self.signal = 1 if self.ensemble_signal > 0 else -1
            self.confidence = abs(self.ensemble_signal)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(399)
class TimeScaleFilter(BaseFormula):
    """
    ID 399: Time-Scale Adaptive Filter

    Dynamically filters noise based on optimal time scale.
    """

    NAME = "TimeScaleFilter"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Adaptive time-scale noise filter"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.filter_scale = 5
        self.filtered_return = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # Find scale with highest signal-to-noise
        best_snr = 0
        best_scale = 1

        for scale in [1, 2, 3, 5, 10]:
            if len(returns) < scale * 5:
                continue

            n = len(returns) // scale
            agg = np.array([np.sum(returns[i*scale:(i+1)*scale]) for i in range(n)])

            if len(agg) < 5:
                continue

            signal = abs(np.mean(agg[-3:]))
            noise = np.std(agg)
            snr = signal / (noise + 1e-10)

            if snr > best_snr:
                best_snr = snr
                best_scale = scale
                self.filtered_return = agg[-1]

        self.filter_scale = best_scale

        # Signal from filtered return
        threshold = np.std(returns) * np.sqrt(best_scale) * 0.5

        if abs(self.filtered_return) > threshold:
            self.signal = 1 if self.filtered_return > 0 else -1
            self.confidence = min(best_snr, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(400)
class UnifiedScaleAnalyzer(BaseFormula):
    """
    ID 400: Unified Scale Analyzer (Master Formula)

    Combines ALL time-scale insights into a single signal.
    The ultimate formula for scale-invariant trading.
    """

    NAME = "UnifiedScaleAnalyzer"
    CATEGORY = "multiscale_advanced_2"
    DESCRIPTION = "Master formula combining all scale analysis"

    def __init__(self, lookback: int = 300, **kwargs):
        super().__init__(lookback, **kwargs)
        # Components
        self.hurst = 0.5
        self.variance_ratio = 1.0
        self.scale_coherence = 0.0
        self.optimal_scale = 10
        self.market_regime = 'neutral'

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()

        # 1. Hurst exponent
        var1 = np.var(returns[-64:])
        n8 = 64 // 8
        if n8 > 1 and var1 > 0:
            agg8 = np.array([np.sum(returns[-64:][i*8:(i+1)*8]) for i in range(n8)])
            var8 = np.var(agg8)
            self.hurst = np.log(var8 / (8 * var1 + 1e-10)) / (2 * np.log(8)) + 0.5
            self.hurst = np.clip(self.hurst, 0.2, 0.8)

        # 2. Variance ratio at optimal scale
        scales = [2, 5, 10, 20]
        best_deviation = 0
        self.optimal_scale = 10

        for s in scales:
            n = len(returns) // s
            if n < 10:
                continue
            agg = np.array([np.sum(returns[i*s:(i+1)*s]) for i in range(n)])
            vr = np.var(agg) / (s * var1 + 1e-10) if var1 > 0 else 1
            deviation = abs(vr - 1)
            if deviation > best_deviation:
                best_deviation = deviation
                self.optimal_scale = s
                self.variance_ratio = vr

        # 3. Scale coherence
        signals = []
        for s in scales:
            if len(returns) >= s:
                recent = np.sum(returns[-s:])
                signals.append(np.sign(recent))

        if signals:
            agreement = abs(np.mean(signals))
            self.scale_coherence = agreement

        # 4. Determine regime
        if self.hurst > 0.55 and self.variance_ratio > 1.1:
            self.market_regime = 'trending'
        elif self.hurst < 0.45 and self.variance_ratio < 0.9:
            self.market_regime = 'mean_reverting'
        elif self.scale_coherence < 0.3:
            self.market_regime = 'choppy'
        else:
            self.market_regime = 'neutral'

        # 5. Generate signal
        s = self.optimal_scale
        recent = np.sum(returns[-s:]) if len(returns) >= s else np.sum(returns)
        std_s = np.std(returns) * np.sqrt(s)
        z = recent / (std_s + 1e-10)

        if self.market_regime == 'trending':
            self.signal = 1 if recent > 0 else -1
            self.confidence = min(self.scale_coherence + (self.hurst - 0.5), 1.0)

        elif self.market_regime == 'mean_reverting':
            self.signal = -1 if z > 1.5 else (1 if z < -1.5 else 0)
            self.confidence = min(self.scale_coherence + (0.5 - self.hurst), 1.0)

        elif self.market_regime == 'choppy':
            self.signal = 0
            self.confidence = 0.2

        else:
            # Neutral - only trade if coherence is high
            if self.scale_coherence > 0.6:
                self.signal = 1 if recent > 0 else -1
                self.confidence = self.scale_coherence * 0.7
            else:
                self.signal = 0
                self.confidence = 0.4


# =============================================================================
# Export all classes
# =============================================================================

__all__ = [
    # Kelly Extensions (381-385)
    'HorizonKelly',
    'ContinuousKellyHJB',
    'FractionalKelly',
    'DrawdownConstrainedKelly',
    'VolatilityScaledKelly',

    # Price Impact (386-390)
    'KyleLambdaScaling',
    'AlmgrenChrissTiming',
    'TransientImpactDecay',
    'SquareRootImpact',
    'MarketResiliency',

    # Return Scaling (391-395)
    'MomentScaling',
    'TailIndexStability',
    'ReturnAggregationTest',
    'ScalePredictabilityLoss',
    'UniversalityClass',

    # Additional (396-400)
    'AdaptiveBandwidth',
    'ScaleCoherence',
    'MultiScaleEnsemble',
    'TimeScaleFilter',
    'UnifiedScaleAnalyzer',
]
