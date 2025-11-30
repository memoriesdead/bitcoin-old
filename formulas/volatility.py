"""
Volatility Formulas (IDs 151-170)
=================================
GARCH variants, realized volatility, rough volatility, and vol surfaces.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from .base import BaseFormula, FormulaRegistry


# =============================================================================
# GARCH FAMILY (151-160)
# =============================================================================

@FormulaRegistry.register(151)
class GARCHVolatility(BaseFormula):
    """ID 151: GARCH(1,1) Volatility Model"""

    CATEGORY = "volatility"
    NAME = "GARCHVolatility"
    DESCRIPTION = "σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}"

    def __init__(self, lookback: int = 100, omega: float = 1e-6,
                 alpha: float = 0.1, beta: float = 0.85, **kwargs):
        super().__init__(lookback, **kwargs)
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.sigma2 = 1e-4
        self.sigma2_history = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        epsilon = self.returns[-1]
        self.sigma2 = self.omega + self.alpha * epsilon**2 + self.beta * self.sigma2
        self.sigma2 = max(self.sigma2, 1e-10)
        self.sigma2_history.append(self.sigma2)
        if len(self.sigma2_history) < 10:
            return
        avg_sigma2 = np.mean(self.sigma2_history)
        vol_ratio = self.sigma2 / (avg_sigma2 + 1e-10)
        if vol_ratio > 1.5:
            self.signal = -1
            self.confidence = min(vol_ratio / 3, 1.0)
        elif vol_ratio < 0.7:
            self.signal = 1
            self.confidence = min(1 / vol_ratio / 2, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(152)
class EGARCHVolatility(BaseFormula):
    """ID 152: Exponential GARCH"""

    CATEGORY = "volatility"
    NAME = "EGARCHVolatility"
    DESCRIPTION = "log(σ²) = ω + α*g(z) + β*log(σ²_{t-1})"

    def __init__(self, lookback: int = 100, omega: float = -0.5,
                 alpha: float = 0.1, beta: float = 0.95, gamma: float = 0.1, **kwargs):
        super().__init__(lookback, **kwargs)
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.log_sigma2 = -8.0

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        epsilon = self.returns[-1]
        sigma = np.exp(self.log_sigma2 / 2)
        z = epsilon / (sigma + 1e-10)
        g_z = self.gamma * (abs(z) - np.sqrt(2/np.pi)) + self.alpha * z
        self.log_sigma2 = self.omega + g_z + self.beta * self.log_sigma2
        self.log_sigma2 = max(min(self.log_sigma2, 0), -20)
        sigma2 = np.exp(self.log_sigma2)
        historical_var = np.var(list(self.returns)) if len(self.returns) > 5 else 1e-4
        vol_ratio = sigma2 / (historical_var + 1e-10)
        if vol_ratio > 2:
            self.signal = -1
            self.confidence = min(vol_ratio / 4, 1.0)
        elif vol_ratio < 0.5:
            self.signal = 1
            self.confidence = min(1 / vol_ratio / 3, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(153)
class GJRGARCHVolatility(BaseFormula):
    """ID 153: GJR-GARCH with leverage effect"""

    CATEGORY = "volatility"
    NAME = "GJRGARCHVolatility"
    DESCRIPTION = "σ² = ω + (α + γ*I_{t-1})*ε² + β*σ²"

    def __init__(self, lookback: int = 100, omega: float = 1e-6,
                 alpha: float = 0.05, beta: float = 0.85, gamma: float = 0.1, **kwargs):
        super().__init__(lookback, **kwargs)
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma2 = 1e-4

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        epsilon = self.returns[-1]
        indicator = 1 if epsilon < 0 else 0
        self.sigma2 = (self.omega +
                      (self.alpha + self.gamma * indicator) * epsilon**2 +
                      self.beta * self.sigma2)
        self.sigma2 = max(self.sigma2, 1e-10)
        historical_var = np.var(list(self.returns)) if len(self.returns) > 5 else 1e-4
        vol_ratio = self.sigma2 / (historical_var + 1e-10)
        leverage_active = epsilon < 0
        if vol_ratio > 1.5:
            self.signal = -1
            conf = min(vol_ratio / 3, 1.0)
            self.confidence = conf * 1.2 if leverage_active else conf
        elif vol_ratio < 0.7:
            self.signal = 1
            self.confidence = min(1 / vol_ratio / 2, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(154)
class TGARCHVolatility(BaseFormula):
    """ID 154: Threshold GARCH"""

    CATEGORY = "volatility"
    NAME = "TGARCHVolatility"
    DESCRIPTION = "σ = ω + α|ε| + γ|ε|*I + β*σ"

    def __init__(self, lookback: int = 100, omega: float = 0.001,
                 alpha: float = 0.1, beta: float = 0.85, gamma: float = 0.05, **kwargs):
        super().__init__(lookback, **kwargs)
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = 0.01

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        epsilon = self.returns[-1]
        indicator = 1 if epsilon < 0 else 0
        self.sigma = (self.omega +
                     self.alpha * abs(epsilon) +
                     self.gamma * abs(epsilon) * indicator +
                     self.beta * self.sigma)
        self.sigma = max(self.sigma, 1e-6)
        historical_std = np.std(list(self.returns)) if len(self.returns) > 5 else 0.01
        vol_ratio = self.sigma / (historical_std + 1e-10)
        if vol_ratio > 1.5:
            self.signal = -1
            self.confidence = min(vol_ratio / 3, 1.0)
        elif vol_ratio < 0.7:
            self.signal = 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(155)
class IGARCHVolatility(BaseFormula):
    """ID 155: Integrated GARCH"""

    CATEGORY = "volatility"
    NAME = "IGARCHVolatility"
    DESCRIPTION = "σ² = ω + α*ε² + (1-α)*σ² (unit root)"

    def __init__(self, lookback: int = 100, omega: float = 1e-7,
                 alpha: float = 0.1, **kwargs):
        super().__init__(lookback, **kwargs)
        self.omega = omega
        self.alpha = alpha
        self.sigma2 = 1e-4

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        epsilon = self.returns[-1]
        beta = 1 - self.alpha
        self.sigma2 = self.omega + self.alpha * epsilon**2 + beta * self.sigma2
        self.sigma2 = max(self.sigma2, 1e-10)
        unconditional_var = self.omega / (1 - self.alpha - beta + 1e-10)
        if self.sigma2 > unconditional_var * 2:
            self.signal = -1
            self.confidence = min(self.sigma2 / unconditional_var / 4, 1.0)
        elif self.sigma2 < unconditional_var * 0.5:
            self.signal = 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(156)
class FIGARCHVolatility(BaseFormula):
    """ID 156: Fractionally Integrated GARCH"""

    CATEGORY = "volatility"
    NAME = "FIGARCHVolatility"
    DESCRIPTION = "Long memory in volatility"

    def __init__(self, lookback: int = 100, d: float = 0.4, **kwargs):
        super().__init__(lookback, **kwargs)
        self.d = d
        self.sigma2_history = deque(maxlen=lookback)
        self.epsilon2_history = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        epsilon2 = self.returns[-1] ** 2
        self.epsilon2_history.append(epsilon2)
        if len(self.epsilon2_history) < 10:
            self.sigma2_history.append(epsilon2)
            return
        weights = []
        for k in range(1, len(self.epsilon2_history)):
            w = np.prod([(self.d + j - 1) / j for j in range(1, k + 1)])
            weights.append(w)
        weights = np.array(weights[-len(self.epsilon2_history)+1:])
        eps2_arr = np.array(list(self.epsilon2_history)[:-1])
        sigma2 = np.sum(weights[:len(eps2_arr)] * eps2_arr[::-1][:len(weights)])
        sigma2 = max(sigma2, 1e-10)
        self.sigma2_history.append(sigma2)
        avg_sigma2 = np.mean(self.sigma2_history)
        vol_ratio = sigma2 / (avg_sigma2 + 1e-10)
        if vol_ratio > 1.5:
            self.signal = -1
            self.confidence = min(vol_ratio / 3, 1.0)
        elif vol_ratio < 0.7:
            self.signal = 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(157)
class APARCHVolatility(BaseFormula):
    """ID 157: Asymmetric Power ARCH"""

    CATEGORY = "volatility"
    NAME = "APARCHVolatility"
    DESCRIPTION = "σ^δ = ω + α(|ε|-γε)^δ + β*σ^δ"

    def __init__(self, lookback: int = 100, omega: float = 1e-4,
                 alpha: float = 0.1, beta: float = 0.85,
                 gamma: float = 0.5, delta: float = 2.0, **kwargs):
        super().__init__(lookback, **kwargs)
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.sigma_delta = 0.01

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        epsilon = self.returns[-1]
        term = abs(epsilon) - self.gamma * epsilon
        term = max(term, 0)
        self.sigma_delta = (self.omega +
                           self.alpha * (term ** self.delta) +
                           self.beta * self.sigma_delta)
        self.sigma_delta = max(self.sigma_delta, 1e-10)
        sigma = self.sigma_delta ** (1 / self.delta)
        historical_std = np.std(list(self.returns)) if len(self.returns) > 5 else 0.01
        vol_ratio = sigma / (historical_std + 1e-10)
        if vol_ratio > 1.5:
            self.signal = -1
            self.confidence = min(vol_ratio / 3, 1.0)
        elif vol_ratio < 0.7:
            self.signal = 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(158)
class NAGARCHVolatility(BaseFormula):
    """ID 158: Nonlinear Asymmetric GARCH"""

    CATEGORY = "volatility"
    NAME = "NAGARCHVolatility"
    DESCRIPTION = "σ² = ω + α(ε - θσ)² + β*σ²"

    def __init__(self, lookback: int = 100, omega: float = 1e-6,
                 alpha: float = 0.1, beta: float = 0.85, theta: float = 0.5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.sigma2 = 1e-4

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        epsilon = self.returns[-1]
        sigma = np.sqrt(self.sigma2)
        self.sigma2 = (self.omega +
                      self.alpha * (epsilon - self.theta * sigma)**2 +
                      self.beta * self.sigma2)
        self.sigma2 = max(self.sigma2, 1e-10)
        historical_var = np.var(list(self.returns)) if len(self.returns) > 5 else 1e-4
        vol_ratio = self.sigma2 / (historical_var + 1e-10)
        if vol_ratio > 1.5:
            self.signal = -1
            self.confidence = min(vol_ratio / 3, 1.0)
        elif vol_ratio < 0.7:
            self.signal = 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(159)
class CGARCHVolatility(BaseFormula):
    """ID 159: Component GARCH"""

    CATEGORY = "volatility"
    NAME = "CGARCHVolatility"
    DESCRIPTION = "Decompose into permanent and transitory"

    def __init__(self, lookback: int = 100, omega: float = 1e-5,
                 phi: float = 0.99, rho: float = 0.95,
                 alpha: float = 0.05, beta: float = 0.9, **kwargs):
        super().__init__(lookback, **kwargs)
        self.omega = omega
        self.phi = phi
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.q_t = 1e-4
        self.sigma2 = 1e-4

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        epsilon = self.returns[-1]
        epsilon2 = epsilon ** 2
        self.q_t = self.omega + self.rho * (self.q_t - self.omega) + self.phi * (epsilon2 - self.sigma2)
        self.q_t = max(self.q_t, 1e-10)
        self.sigma2 = self.q_t + self.alpha * (epsilon2 - self.q_t) + self.beta * (self.sigma2 - self.q_t)
        self.sigma2 = max(self.sigma2, 1e-10)
        transitory = self.sigma2 - self.q_t
        if transitory > 0.5 * self.q_t:
            self.signal = -1
            self.confidence = min(transitory / self.q_t, 1.0)
        elif transitory < -0.3 * self.q_t:
            self.signal = 1
            self.confidence = min(abs(transitory) / self.q_t, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(160)
class GARCHMVolatility(BaseFormula):
    """ID 160: GARCH-in-Mean"""

    CATEGORY = "volatility"
    NAME = "GARCHMVolatility"
    DESCRIPTION = "Return depends on volatility"

    def __init__(self, lookback: int = 100, omega: float = 1e-6,
                 alpha: float = 0.1, beta: float = 0.85, lambda_: float = 0.1, **kwargs):
        super().__init__(lookback, **kwargs)
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.sigma2 = 1e-4
        self.expected_return = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 2:
            return
        epsilon = self.returns[-1]
        self.sigma2 = self.omega + self.alpha * epsilon**2 + self.beta * self.sigma2
        self.sigma2 = max(self.sigma2, 1e-10)
        self.expected_return = self.lambda_ * np.sqrt(self.sigma2)
        actual_return = self.returns[-1]
        surprise = actual_return - self.expected_return
        sigma_ret = np.std(list(self.returns)) if len(self.returns) > 5 else 0.01
        z = surprise / (sigma_ret + 1e-10)
        if z > 2:
            self.signal = 1
            self.confidence = min(z / 4, 1.0)
        elif z < -2:
            self.signal = -1
            self.confidence = min(abs(z) / 4, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


# =============================================================================
# REALIZED AND ROUGH VOLATILITY (161-170)
# =============================================================================

@FormulaRegistry.register(161)
class RealizedVolatility(BaseFormula):
    """ID 161: Realized Volatility from high-frequency data"""

    CATEGORY = "volatility"
    NAME = "RealizedVolatility"
    DESCRIPTION = "RV = Σ(r_i²)"

    def __init__(self, lookback: int = 100, window: int = 20, **kwargs):
        super().__init__(lookback, **kwargs)
        self.window = window
        self.rv = 0.0
        self.rv_history = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < self.window:
            return
        returns = self._returns_array()
        self.rv = np.sum(returns[-self.window:]**2)
        self.rv_history.append(self.rv)
        if len(self.rv_history) < 10:
            return
        avg_rv = np.mean(self.rv_history)
        rv_ratio = self.rv / (avg_rv + 1e-10)
        if rv_ratio > 1.5:
            self.signal = -1
            self.confidence = min(rv_ratio / 3, 1.0)
        elif rv_ratio < 0.7:
            self.signal = 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(162)
class BiPowerVariation(BaseFormula):
    """ID 162: Bipower Variation (robust to jumps)"""

    CATEGORY = "volatility"
    NAME = "BiPowerVariation"
    DESCRIPTION = "BPV = (π/2) × Σ|r_i||r_{i-1}|"

    def __init__(self, lookback: int = 100, window: int = 20, **kwargs):
        super().__init__(lookback, **kwargs)
        self.window = window
        self.bpv = 0.0

    def _compute(self) -> None:
        if len(self.returns) < self.window + 1:
            return
        returns = self._returns_array()
        abs_returns = np.abs(returns[-self.window-1:])
        products = abs_returns[1:] * abs_returns[:-1]
        self.bpv = (np.pi / 2) * np.sum(products)
        rv = np.sum(returns[-self.window:]**2)
        jump_ratio = (rv - self.bpv) / (self.bpv + 1e-10)
        if jump_ratio > 0.5:
            recent_return = returns[-1]
            self.signal = 1 if recent_return < 0 else -1
            self.confidence = min(jump_ratio, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(163)
class RealizedKernelVolatility(BaseFormula):
    """ID 163: Realized Kernel with Parzen weights"""

    CATEGORY = "volatility"
    NAME = "RealizedKernelVolatility"
    DESCRIPTION = "Noise-robust realized volatility"

    def __init__(self, lookback: int = 100, bandwidth: int = 5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.bandwidth = bandwidth
        self.rk = 0.0

    def _parzen_kernel(self, x: float) -> float:
        x = abs(x)
        if x <= 0.5:
            return 1 - 6*x**2 + 6*x**3
        elif x <= 1:
            return 2 * (1 - x)**3
        return 0.0

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        n = len(returns)
        gamma_0 = np.sum(returns**2)
        kernel_sum = 0.0
        for h in range(1, self.bandwidth + 1):
            weight = self._parzen_kernel(h / (self.bandwidth + 1))
            gamma_h = np.sum(returns[h:] * returns[:-h])
            kernel_sum += 2 * weight * gamma_h
        self.rk = gamma_0 + kernel_sum
        self.rk = max(self.rk, 1e-10)
        simple_rv = np.sum(returns**2)
        noise_ratio = (simple_rv - self.rk) / (self.rk + 1e-10)
        historical_var = np.var(returns)
        vol_ratio = self.rk / (historical_var * n + 1e-10)
        if vol_ratio > 1.5:
            self.signal = -1
            self.confidence = min(vol_ratio / 3, 1.0)
        elif vol_ratio < 0.7:
            self.signal = 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(164)
class TwoScaleVolatility(BaseFormula):
    """ID 164: Two-Scale Realized Volatility"""

    CATEGORY = "volatility"
    NAME = "TwoScaleVolatility"
    DESCRIPTION = "TSRV for microstructure noise"

    def __init__(self, lookback: int = 100, subsample_size: int = 5, **kwargs):
        super().__init__(lookback, **kwargs)
        self.subsample_size = subsample_size
        self.tsrv = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 30:
            return
        returns = self._returns_array()
        n = len(returns)
        rv_all = np.sum(returns**2)
        rv_subsampled = 0.0
        k = self.subsample_size
        for i in range(k):
            subsample = returns[i::k]
            rv_subsampled += np.sum(subsample**2)
        rv_subsampled /= k
        n_bar = (n - k + 1) / k
        self.tsrv = rv_subsampled - (n_bar / n) * rv_all
        self.tsrv = max(self.tsrv, 1e-10)
        historical_var = np.var(returns)
        vol_ratio = self.tsrv / (historical_var * n + 1e-10)
        if vol_ratio > 1.5:
            self.signal = -1
            self.confidence = min(vol_ratio / 3, 1.0)
        elif vol_ratio < 0.7:
            self.signal = 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(165)
class RoughHeston(BaseFormula):
    """ID 165: Rough Heston Volatility Model"""

    CATEGORY = "volatility"
    NAME = "RoughHeston"
    DESCRIPTION = "Fractional volatility with H < 0.5"

    def __init__(self, lookback: int = 100, H: float = 0.1,
                 kappa: float = 0.5, theta: float = 0.04, nu: float = 0.3, **kwargs):
        super().__init__(lookback, **kwargs)
        self.H = H
        self.kappa = kappa
        self.theta = theta
        self.nu = nu
        self.variance = 0.04

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        realized_var = np.var(returns[-20:])
        self.variance = (self.variance +
                        self.kappa * (self.theta - self.variance) * 0.01 +
                        self.nu * np.sqrt(max(self.variance, 0)) * np.random.randn() * 0.01)
        self.variance = max(self.variance, 1e-10)
        log_returns = returns[-10:] if len(returns) >= 10 else returns
        increments = np.diff(log_returns)
        if len(increments) > 1:
            m2 = np.mean(increments**2)
            m4 = np.mean(increments**4)
            if m2 > 0:
                roughness = 0.5 * np.log(m4 / m2**2) / np.log(2)
                roughness = max(0.01, min(0.49, roughness))
            else:
                roughness = self.H
        else:
            roughness = self.H
        if roughness < 0.2 and self.variance > self.theta:
            self.signal = -1
            self.confidence = min((self.theta / self.variance), 1.0)
        elif roughness > 0.4 and self.variance < self.theta:
            self.signal = 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(166)
class RoughBergomi(BaseFormula):
    """ID 166: Rough Bergomi Model"""

    CATEGORY = "volatility"
    NAME = "RoughBergomi"
    DESCRIPTION = "V_t = ξ_0 × exp(η*W^H - η²t^{2H}/2)"

    def __init__(self, lookback: int = 100, H: float = 0.1,
                 eta: float = 1.9, xi_0: float = 0.04, **kwargs):
        super().__init__(lookback, **kwargs)
        self.H = H
        self.eta = eta
        self.xi_0 = xi_0
        self.W_H = 0.0
        self.variance = xi_0

    def _compute(self) -> None:
        if len(self.returns) < 10:
            return
        returns = self._returns_array()
        dW = returns[-1] / (np.std(returns) + 1e-10)
        alpha = self.H - 0.5
        self.W_H = 0.9 * self.W_H + 0.1 * dW
        t = len(returns) / 100.0
        exponent = self.eta * self.W_H - 0.5 * self.eta**2 * t**(2*self.H)
        self.variance = self.xi_0 * np.exp(exponent)
        self.variance = max(min(self.variance, 1.0), 1e-10)
        realized_var = np.var(returns[-20:]) if len(returns) >= 20 else np.var(returns)
        vol_ratio = self.variance / (realized_var + 1e-10)
        if vol_ratio > 2:
            self.signal = -1
            self.confidence = min(vol_ratio / 4, 1.0)
        elif vol_ratio < 0.5:
            self.signal = 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(167)
class VIXVolatility(BaseFormula):
    """ID 167: VIX-style Implied Volatility Index"""

    CATEGORY = "volatility"
    NAME = "VIXVolatility"
    DESCRIPTION = "Model-free implied volatility"

    def __init__(self, lookback: int = 100, annualization: float = 252, **kwargs):
        super().__init__(lookback, **kwargs)
        self.annualization = annualization
        self.vix = 20.0
        self.vix_history = deque(maxlen=lookback)

    def _compute(self) -> None:
        if len(self.returns) < 20:
            return
        returns = self._returns_array()
        daily_var = np.var(returns[-20:])
        annualized_vol = np.sqrt(daily_var * self.annualization)
        self.vix = annualized_vol * 100
        self.vix_history.append(self.vix)
        if len(self.vix_history) < 10:
            return
        avg_vix = np.mean(self.vix_history)
        vix_percentile = np.mean([1 for v in self.vix_history if v < self.vix])
        if self.vix > avg_vix * 1.5 or vix_percentile > 0.9:
            self.signal = -1
            self.confidence = min(self.vix / avg_vix / 3, 1.0)
        elif self.vix < avg_vix * 0.7 or vix_percentile < 0.1:
            self.signal = 1
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(168)
class VolatilitySurface(BaseFormula):
    """ID 168: Volatility Surface Term Structure"""

    CATEGORY = "volatility"
    NAME = "VolatilitySurface"
    DESCRIPTION = "Multi-term volatility structure"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.short_vol = 0.0
        self.medium_vol = 0.0
        self.long_vol = 0.0
        self.term_slope = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 50:
            return
        returns = self._returns_array()
        self.short_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
        self.medium_vol = np.std(returns[-30:]) if len(returns) >= 30 else np.std(returns)
        self.long_vol = np.std(returns[-50:]) if len(returns) >= 50 else np.std(returns)
        if self.short_vol > 0:
            self.term_slope = (self.long_vol - self.short_vol) / self.short_vol
        if self.term_slope < -0.2:
            self.signal = -1
            self.confidence = min(abs(self.term_slope), 1.0)
        elif self.term_slope > 0.2:
            self.signal = 1
            self.confidence = min(self.term_slope, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(169)
class VolatilitySkew(BaseFormula):
    """ID 169: Volatility Skew Indicator"""

    CATEGORY = "volatility"
    NAME = "VolatilitySkew"
    DESCRIPTION = "Asymmetry in volatility distribution"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.up_vol = 0.0
        self.down_vol = 0.0
        self.skew = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 30:
            return
        returns = self._returns_array()
        up_returns = returns[returns > 0]
        down_returns = returns[returns < 0]
        self.up_vol = np.std(up_returns) if len(up_returns) > 5 else 0.01
        self.down_vol = np.std(down_returns) if len(down_returns) > 5 else 0.01
        self.skew = (self.down_vol - self.up_vol) / (self.up_vol + 1e-10)
        if self.skew > 0.3:
            self.signal = -1
            self.confidence = min(self.skew, 1.0)
        elif self.skew < -0.2:
            self.signal = 1
            self.confidence = min(abs(self.skew), 1.0)
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(170)
class VolatilityCone(BaseFormula):
    """ID 170: Volatility Cone Analysis"""

    CATEGORY = "volatility"
    NAME = "VolatilityCone"
    DESCRIPTION = "Historical volatility percentile cone"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)
        self.vol_percentile = 0.5
        self.vol_windows = [5, 10, 20, 30, 50]
        self.historical_vols = {w: deque(maxlen=lookback) for w in self.vol_windows}

    def _compute(self) -> None:
        if len(self.returns) < 50:
            return
        returns = self._returns_array()
        current_vols = {}
        for w in self.vol_windows:
            if len(returns) >= w:
                vol = np.std(returns[-w:])
                current_vols[w] = vol
                self.historical_vols[w].append(vol)
        percentiles = []
        for w in self.vol_windows:
            if w in current_vols and len(self.historical_vols[w]) > 10:
                hist = np.array(self.historical_vols[w])
                pct = np.mean(hist < current_vols[w])
                percentiles.append(pct)
        if len(percentiles) > 0:
            self.vol_percentile = np.mean(percentiles)
        if self.vol_percentile > 0.8:
            self.signal = -1
            self.confidence = self.vol_percentile
        elif self.vol_percentile < 0.2:
            self.signal = 1
            self.confidence = 1 - self.vol_percentile
        else:
            self.signal = 0
            self.confidence = 0.4


__all__ = [
    'GARCHVolatility', 'EGARCHVolatility', 'GJRGARCHVolatility', 'TGARCHVolatility',
    'IGARCHVolatility', 'FIGARCHVolatility', 'APARCHVolatility', 'NAGARCHVolatility',
    'CGARCHVolatility', 'GARCHMVolatility',
    'RealizedVolatility', 'BiPowerVariation', 'RealizedKernelVolatility',
    'TwoScaleVolatility', 'RoughHeston', 'RoughBergomi', 'VIXVolatility',
    'VolatilitySurface', 'VolatilitySkew', 'VolatilityCone',
]
