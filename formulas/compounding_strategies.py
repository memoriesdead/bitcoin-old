#!/usr/bin/env python3
"""
COMPOUNDING & GROWTH FORMULAS (IDs 323-330)
===========================================
Academic research-based formulas for capital growth and optimal compounding.

Based on peer-reviewed sources:
- Kelly (1956): "A New Interpretation of Information Rate" - Bell System Technical Journal
- Thorp (2007): "The Kelly Criterion" - Williams College Mathematics
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions" - Journal of Risk
- Avellaneda & Stoikov (2008): "High-frequency trading in a limit order book" - Quantitative Finance
- Vince (1990): "Portfolio Management Formulas" - John Wiley & Sons
- Cartea, Jaimungal & Penalva (2015): "Algorithmic and High-Frequency Trading" - Cambridge Press

These formulas determine optimal position sizing, leverage, and execution for compounding $10 to $300,000+.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
from dataclasses import dataclass
import math

from formulas.base import BaseFormula, FormulaRegistry


@dataclass
class CompoundingSignal:
    """Output from compounding formulas"""
    optimal_fraction: float      # Kelly f* or Optimal f
    optimal_leverage: float      # Leverage multiplier
    expected_growth_rate: float  # g = r + S^2/2
    position_size_usd: float     # Dollar amount to trade
    confidence: float            # Signal confidence
    volatility_drag: float       # sigma^2/2 drag
    sharpe_ratio: float          # Annualized Sharpe


# =============================================================================
# FORMULA 323: KELLY CRITERION OPTIMAL LEVERAGE
# =============================================================================
# Based on Thorp (2007) continuous-time Kelly derivation
#
# Core Formula:
#   f* = mu / sigma^2  (optimal leverage)
#
# Where:
#   mu = expected excess return (above risk-free rate)
#   sigma = volatility of returns
#
# The expected growth rate is:
#   g = r + S^2/2
#
# Where S = Sharpe ratio = mu/sigma
#
# Reference:
#   Thorp, E.O. (2007). "The Kelly Criterion in Blackjack Sports Betting,
#   And The Stock Market." Williams College Mathematics.
#   URL: https://web.williams.edu/Mathematics/sjmiller/public_html/341/handouts/Thorpe_KellyCriterion2007.pdf
# =============================================================================

@FormulaRegistry.register(323, name="KellyCriterionFormula", category="compounding")
class KellyCriterionFormula(BaseFormula):
    """
    Formula 323: Kelly Criterion Optimal Leverage

    Maximizes the expected value of log wealth (geometric growth rate).
    Uses continuous-time formulation from Thorp (2007).

    Formula: f* = mu / sigma^2

    Academic Reference:
        Thorp, E.O. (2007). The Kelly Criterion in Blackjack Sports Betting,
        And The Stock Market. Williams College Mathematics Department.
    """

    FORMULA_ID = 323
    CATEGORY = "compounding"
    NAME = "KellyCriterionFormula"
    DESCRIPTION = "Optimal leverage using Kelly Criterion (Thorp 2007)"

    def __init__(
        self,
        risk_free_rate: float = 0.05,      # 5% annual risk-free rate
        kelly_fraction: float = 0.5,        # Half-Kelly for safety
        min_sharpe: float = 0.5,            # Minimum Sharpe to trade
        max_leverage: float = 3.0,          # Maximum allowed leverage
        volatility_window: int = 50,        # Window for vol calculation
        returns_window: int = 50,           # Window for return calculation
        annualization_factor: float = 365 * 24 * 60,  # Minutes per year
        **kwargs
    ):
        super().__init__(**kwargs)
        self.risk_free_rate = risk_free_rate
        self.kelly_fraction = kelly_fraction
        self.min_sharpe = min_sharpe
        self.max_leverage = max_leverage
        self.volatility_window = volatility_window
        self.returns_window = returns_window
        self.annualization_factor = annualization_factor

        # State
        self.mu = 0.0              # Expected return
        self.sigma = 0.0          # Volatility
        self.optimal_f = 0.0      # Kelly fraction
        self.leverage = 1.0       # Applied leverage
        self.sharpe = 0.0         # Sharpe ratio
        self.growth_rate = 0.0    # Expected growth
        self.volatility_drag = 0.0  # sigma^2/2

        # History for calculations
        self.return_history = deque(maxlen=returns_window)

    def _compute(self) -> None:
        """Compute Kelly optimal leverage"""
        if len(self.returns) < self.min_samples:
            return

        returns = np.array(self.returns)

        # Calculate mean return and volatility
        # Using per-period (per minute) returns
        self.mu = np.mean(returns)
        self.sigma = np.std(returns)

        if self.sigma <= 0:
            self.optimal_f = 0
            self.leverage = 1.0
            self.signal = 0
            self.confidence = 0.0
            return

        # Annualize
        mu_annual = self.mu * self.annualization_factor
        sigma_annual = self.sigma * np.sqrt(self.annualization_factor)

        # Excess return (above risk-free)
        excess_return = mu_annual - self.risk_free_rate

        # Sharpe ratio
        self.sharpe = excess_return / sigma_annual if sigma_annual > 0 else 0

        # Kelly formula: f* = mu / sigma^2
        if sigma_annual > 0:
            self.optimal_f = excess_return / (sigma_annual ** 2)
        else:
            self.optimal_f = 0

        # Apply Kelly fraction (half-Kelly for safety)
        self.leverage = self.optimal_f * self.kelly_fraction

        # Clamp leverage
        self.leverage = np.clip(self.leverage, 0.1, self.max_leverage)

        # Volatility drag: sigma^2/2
        self.volatility_drag = (sigma_annual ** 2) / 2

        # Expected growth rate: g = r + S^2/2
        # From Thorp: g = r + S^2/2 where S is Sharpe ratio
        self.growth_rate = self.risk_free_rate + (self.sharpe ** 2) / 2

        # Generate signal based on Sharpe threshold
        if self.sharpe >= self.min_sharpe:
            self.signal = 1 if excess_return > 0 else -1
            self.confidence = min(abs(self.sharpe) / 2, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.0

    def get_compounding_signal(self, capital: float) -> CompoundingSignal:
        """Get full compounding recommendation"""
        position_size = capital * self.leverage * self.kelly_fraction

        return CompoundingSignal(
            optimal_fraction=self.optimal_f,
            optimal_leverage=self.leverage,
            expected_growth_rate=self.growth_rate,
            position_size_usd=position_size,
            confidence=self.confidence,
            volatility_drag=self.volatility_drag,
            sharpe_ratio=self.sharpe
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'mu_annual': self.mu * self.annualization_factor,
            'sigma_annual': self.sigma * np.sqrt(self.annualization_factor),
            'optimal_f': self.optimal_f,
            'leverage': self.leverage,
            'sharpe': self.sharpe,
            'growth_rate': self.growth_rate,
            'volatility_drag': self.volatility_drag,
        }


# =============================================================================
# FORMULA 324: OPTIMAL F (VINCE)
# =============================================================================
# Based on Ralph Vince (1990) "Portfolio Management Formulas"
#
# Optimal f maximizes the Geometric Holding Period Return (GHPR):
#   GHPR = (Product of (1 + f * R_i / |Worst Loss|))^(1/n)
#
# The optimal f is found iteratively to maximize GHPR.
# Position size: N = (f * Equity) / |Trade Risk|
#
# Reference:
#   Vince, R. (1990). Portfolio Management Formulas. John Wiley & Sons.
# =============================================================================

@FormulaRegistry.register(324, name="OptimalFFormula", category="compounding")
class OptimalFFormula(BaseFormula):
    """
    Formula 324: Optimal f Position Sizing

    Maximizes geometric holding period return using all historical trades.
    More robust than Kelly for non-normal distributions.

    Formula: Find f that maximizes Product((1 + f*R_i/|worst_loss|))^(1/n)

    Academic Reference:
        Vince, R. (1990). Portfolio Management Formulas. John Wiley & Sons.
    """

    FORMULA_ID = 324
    CATEGORY = "compounding"
    NAME = "OptimalFFormula"
    DESCRIPTION = "Optimal f position sizing (Vince 1990)"

    def __init__(
        self,
        f_fraction: float = 0.5,        # Safety fraction of optimal f
        max_f: float = 0.25,            # Maximum f (25% of capital)
        granularity: int = 100,          # Steps for f search
        trade_history_min: int = 30,    # Minimum trades for calculation
        **kwargs
    ):
        super().__init__(**kwargs)
        self.f_fraction = f_fraction
        self.max_f = max_f
        self.granularity = granularity
        self.trade_history_min = trade_history_min

        # State
        self.optimal_f = 0.0
        self.ghpr = 1.0              # Geometric HPR
        self.worst_loss = 0.0
        self.trade_returns = deque(maxlen=500)  # Store trade returns

    def add_trade(self, return_pct: float):
        """Add a completed trade return"""
        self.trade_returns.append(return_pct)

    def _compute_ghpr(self, f: float, returns: np.ndarray, worst_loss: float) -> float:
        """Compute Geometric Holding Period Return for given f"""
        if worst_loss >= 0 or f <= 0:
            return 0.0

        hpr_product = 1.0
        for r in returns:
            hpr = 1 + f * r / abs(worst_loss)
            if hpr <= 0:
                return 0.0  # Ruin
            hpr_product *= hpr

        n = len(returns)
        if n == 0:
            return 0.0

        return hpr_product ** (1.0 / n)

    def _compute(self) -> None:
        """Compute optimal f via search"""
        if len(self.returns) < self.min_samples:
            return

        # Use recent returns for calculation
        returns = np.array(list(self.returns)[-100:])

        if len(returns) < 10:
            return

        self.worst_loss = np.min(returns)

        if self.worst_loss >= 0:
            # No losses yet - can't compute optimal f
            self.optimal_f = 0.01  # Conservative default
            return

        # Search for optimal f
        best_f = 0.01
        best_ghpr = 0.0

        for i in range(1, self.granularity + 1):
            f = i / self.granularity * self.max_f
            ghpr = self._compute_ghpr(f, returns, self.worst_loss)

            if ghpr > best_ghpr:
                best_ghpr = ghpr
                best_f = f

        self.optimal_f = best_f * self.f_fraction  # Apply safety fraction
        self.ghpr = best_ghpr

        # Signal based on positive GHPR
        if self.ghpr > 1.0:
            self.signal = 1
            self.confidence = min((self.ghpr - 1.0) * 10, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.0

    def get_position_size(self, equity: float, trade_risk: float) -> float:
        """
        Calculate position size using Optimal f formula
        N = (f * Equity) / |Trade Risk|
        """
        if trade_risk == 0:
            return 0.0
        return (self.optimal_f * equity) / abs(trade_risk)

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'optimal_f': self.optimal_f,
            'ghpr': self.ghpr,
            'worst_loss': self.worst_loss,
            'num_trades': len(self.trade_returns),
        }


# =============================================================================
# FORMULA 325: VOLATILITY-ADJUSTED GROWTH RATE
# =============================================================================
# Based on geometric mean relationship with volatility
#
# Core Formula:
#   g_geometric = g_arithmetic - sigma^2/2
#
# Or equivalently for log returns:
#   E[log(1+r)] = log(1+mu) - sigma^2/(2*(1+mu)^2)
#
# This is the "volatility drag" or "volatility tax"
#
# Reference:
#   Kitces, M. "Volatility Drag: How Variance Drains Investment Returns"
#   Also derived from Ito's Lemma for GBM
# =============================================================================

@FormulaRegistry.register(325, name="VolatilityAdjustedGrowthFormula", category="compounding")
class VolatilityAdjustedGrowthFormula(BaseFormula):
    """
    Formula 325: Volatility-Adjusted Growth Rate

    Computes true geometric growth rate accounting for volatility drag.
    g_geometric = g_arithmetic - sigma^2/2

    This formula identifies when volatility is destroying returns
    and adjusts position sizing accordingly.

    Academic Basis:
        Derived from Ito's Lemma for Geometric Brownian Motion.
        See also: Wikipedia "Volatility Tax"
    """

    FORMULA_ID = 325
    CATEGORY = "compounding"
    NAME = "VolatilityAdjustedGrowthFormula"
    DESCRIPTION = "Growth rate adjusted for volatility drag"

    def __init__(
        self,
        vol_drag_threshold: float = 0.001,  # Threshold to reduce position
        vol_window: int = 50,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vol_drag_threshold = vol_drag_threshold
        self.vol_window = vol_window

        # State
        self.arithmetic_return = 0.0
        self.geometric_return = 0.0
        self.volatility_drag = 0.0
        self.is_vol_drag_high = False

    def _compute(self) -> None:
        """Compute volatility-adjusted growth"""
        if len(self.returns) < self.min_samples:
            return

        returns = np.array(list(self.returns)[-self.vol_window:])

        # Arithmetic mean
        self.arithmetic_return = np.mean(returns)

        # Volatility
        sigma = np.std(returns)

        # Volatility drag: sigma^2 / 2
        self.volatility_drag = (sigma ** 2) / 2

        # Geometric return = arithmetic - vol_drag
        self.geometric_return = self.arithmetic_return - self.volatility_drag

        # Check if volatility is killing returns
        self.is_vol_drag_high = self.volatility_drag > abs(self.arithmetic_return) * 0.5

        # Signal: trade if geometric return is positive and not being killed by vol
        if self.geometric_return > 0 and not self.is_vol_drag_high:
            self.signal = 1
            # Confidence inversely proportional to vol drag
            vol_drag_ratio = self.volatility_drag / (abs(self.arithmetic_return) + 1e-10)
            self.confidence = max(0, 1 - vol_drag_ratio)
        elif self.geometric_return < 0:
            self.signal = -1
            self.confidence = min(abs(self.geometric_return) * 100, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.0

    def get_adjusted_leverage(self, base_leverage: float) -> float:
        """Adjust leverage based on volatility drag"""
        if self.is_vol_drag_high:
            # Reduce leverage when vol is killing returns
            return base_leverage * 0.5
        return base_leverage

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'arithmetic_return': self.arithmetic_return,
            'geometric_return': self.geometric_return,
            'volatility_drag': self.volatility_drag,
            'is_vol_drag_high': self.is_vol_drag_high,
        }


# =============================================================================
# FORMULA 326: ALMGREN-CHRISS OPTIMAL EXECUTION
# =============================================================================
# Based on Almgren & Chriss (2001) Journal of Risk
#
# Optimal execution trajectory for large orders:
#   x_j = sinh(kappa*(T-t_j)) / sinh(kappa*T) * X
#
# Where kappa = sqrt(lambda * sigma^2 / eta)
#   lambda = risk aversion
#   sigma = volatility
#   eta = temporary impact parameter
#
# Reference:
#   Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio
#   Transactions. Journal of Risk, 3, 5-40.
# =============================================================================

@FormulaRegistry.register(326, name="AlmgrenChrissExecutionFormula", category="compounding")
class AlmgrenChrissExecutionFormula(BaseFormula):
    """
    Formula 326: Almgren-Chriss Optimal Execution

    Computes optimal trading trajectory to minimize market impact + timing risk.

    Formula: x_j = sinh(kappa*(T-t_j)) / sinh(kappa*T) * X

    Academic Reference:
        Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio
        Transactions. Journal of Risk, 3, 5-40.
    """

    FORMULA_ID = 326
    CATEGORY = "compounding"
    NAME = "AlmgrenChrissExecutionFormula"
    DESCRIPTION = "Optimal execution trajectory (Almgren-Chriss 2001)"

    def __init__(
        self,
        risk_aversion: float = 1e-6,      # Lambda: risk aversion
        temp_impact: float = 1e-4,         # Eta: temporary market impact
        perm_impact: float = 1e-5,         # Gamma: permanent market impact
        execution_periods: int = 10,       # N: number of trading periods
        **kwargs
    ):
        super().__init__(**kwargs)
        self.risk_aversion = risk_aversion
        self.temp_impact = temp_impact
        self.perm_impact = perm_impact
        self.execution_periods = execution_periods

        # State
        self.kappa = 0.0
        self.trajectory: List[float] = []
        self.current_period = 0
        self.total_to_execute = 0.0

    def _compute(self) -> None:
        """Compute kappa for optimal trajectory"""
        if len(self.returns) < self.min_samples:
            return

        # Estimate volatility
        returns = np.array(self.returns)
        sigma = np.std(returns)

        if sigma <= 0 or self.temp_impact <= 0:
            return

        # Kappa from Almgren-Chriss
        # kappa = sqrt(lambda * sigma^2 / eta)
        self.kappa = np.sqrt(self.risk_aversion * (sigma ** 2) / self.temp_impact)

        # Generate signal based on volatility regime
        if sigma < 0.001:  # Low vol - can execute faster
            self.signal = 1
            self.confidence = 0.8
        else:
            self.signal = 0
            self.confidence = 0.5

    def compute_trajectory(self, total_shares: float, periods: int = None) -> List[float]:
        """
        Compute optimal execution trajectory

        Returns list of shares to trade at each period.
        """
        if periods is None:
            periods = self.execution_periods

        if self.kappa == 0:
            # Equal split if kappa not computed
            return [total_shares / periods] * periods

        self.total_to_execute = total_shares
        self.trajectory = []

        T = periods
        tau = 1  # Time per period

        for j in range(periods):
            t_j = j * tau

            # Holdings at time t_j
            # x_j = sinh(kappa*(T-t_j)) / sinh(kappa*T) * X
            if self.kappa * T > 100:  # Avoid overflow
                x_j = np.exp(-self.kappa * t_j) * total_shares
            else:
                x_j = np.sinh(self.kappa * (T - t_j)) / np.sinh(self.kappa * T) * total_shares

            self.trajectory.append(x_j)

        # Convert holdings to trades (differences)
        trades = []
        for i in range(len(self.trajectory)):
            if i == 0:
                trades.append(total_shares - self.trajectory[0])
            else:
                trades.append(self.trajectory[i-1] - self.trajectory[i])

        return trades

    def get_next_trade(self) -> float:
        """Get next trade size from trajectory"""
        if self.current_period >= len(self.trajectory):
            return 0.0
        trade = self.trajectory[self.current_period]
        self.current_period += 1
        return trade

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'kappa': self.kappa,
            'current_period': self.current_period,
            'trajectory_length': len(self.trajectory),
        }


# =============================================================================
# FORMULA 327: AVELLANEDA-STOIKOV MARKET MAKING
# =============================================================================
# Based on Avellaneda & Stoikov (2008) Quantitative Finance
#
# Reservation price:
#   r = s - q * gamma * sigma^2 * (T - t)
#
# Optimal spread:
#   delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
#
# Where:
#   s = mid price
#   q = inventory
#   gamma = risk aversion
#   sigma = volatility
#   k = order arrival intensity
#
# Reference:
#   Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a
#   limit order book. Quantitative Finance, 8(3), 217-224.
# =============================================================================

@FormulaRegistry.register(327, name="AvellanedaStoikovFormula", category="compounding")
class AvellanedaStoikovFormula(BaseFormula):
    """
    Formula 327: Avellaneda-Stoikov Market Making

    Optimal bid/ask quotes for market making with inventory risk.

    Formulas:
        Reservation price: r = s - q*gamma*sigma^2*(T-t)
        Optimal spread: delta = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)

    Academic Reference:
        Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a
        limit order book. Quantitative Finance, 8(3), 217-224.
    """

    FORMULA_ID = 327
    CATEGORY = "compounding"
    NAME = "AvellanedaStoikovFormula"
    DESCRIPTION = "Market making with inventory risk (Avellaneda-Stoikov 2008)"

    def __init__(
        self,
        gamma: float = 0.1,              # Risk aversion
        k: float = 1.5,                   # Order arrival intensity
        time_horizon: float = 1.0,        # T: time horizon (e.g., 1 hour)
        max_inventory: float = 1.0,       # Maximum inventory in BTC
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.k = k
        self.time_horizon = time_horizon
        self.max_inventory = max_inventory

        # State
        self.inventory = 0.0          # Current inventory (+ = long, - = short)
        self.mid_price = 0.0
        self.reservation_price = 0.0
        self.optimal_spread = 0.0
        self.bid_price = 0.0
        self.ask_price = 0.0
        self.time_remaining = time_horizon
        self.sigma = 0.0

    def set_inventory(self, inventory: float):
        """Update current inventory"""
        self.inventory = np.clip(inventory, -self.max_inventory, self.max_inventory)

    def set_time_remaining(self, time_remaining: float):
        """Update time remaining in horizon"""
        self.time_remaining = max(0.001, time_remaining)

    def _compute(self) -> None:
        """Compute optimal quotes"""
        if len(self.prices) < 2:
            return

        self.mid_price = self.prices[-1]

        # Estimate volatility
        returns = np.array(self.returns) if len(self.returns) > 0 else np.array([0])
        self.sigma = np.std(returns) if len(returns) > 1 else 0.001

        # Reservation price: r = s - q * gamma * sigma^2 * (T - t)
        inventory_adjustment = self.inventory * self.gamma * (self.sigma ** 2) * self.time_remaining
        self.reservation_price = self.mid_price - inventory_adjustment

        # Optimal spread: delta = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)
        spread_vol_component = self.gamma * (self.sigma ** 2) * self.time_remaining
        spread_intensity_component = (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        self.optimal_spread = spread_vol_component + spread_intensity_component

        # Bid and ask prices
        half_spread = self.optimal_spread / 2
        self.bid_price = self.reservation_price - half_spread
        self.ask_price = self.reservation_price + half_spread

        # Signal based on inventory
        # If long, favor selling (signal = -1)
        # If short, favor buying (signal = +1)
        if self.inventory > 0.1 * self.max_inventory:
            self.signal = -1  # Reduce long position
            self.confidence = min(abs(self.inventory) / self.max_inventory, 1.0)
        elif self.inventory < -0.1 * self.max_inventory:
            self.signal = 1   # Reduce short position
            self.confidence = min(abs(self.inventory) / self.max_inventory, 1.0)
        else:
            self.signal = 0   # Neutral
            self.confidence = 0.5

    def get_quotes(self) -> Tuple[float, float]:
        """Get optimal bid and ask prices"""
        return self.bid_price, self.ask_price

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'inventory': self.inventory,
            'mid_price': self.mid_price,
            'reservation_price': self.reservation_price,
            'optimal_spread': self.optimal_spread,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'sigma': self.sigma,
        }


# =============================================================================
# FORMULA 328: EXPECTED GROWTH RATE (THORP)
# =============================================================================
# From Thorp (2007) - Expected growth rate formula
#
# g = r + S^2/2
#
# Where:
#   r = risk-free rate
#   S = Sharpe ratio
#
# This gives the expected continuous-time growth rate under optimal Kelly betting.
# =============================================================================

@FormulaRegistry.register(328, name="ExpectedGrowthRateFormula", category="compounding")
class ExpectedGrowthRateFormula(BaseFormula):
    """
    Formula 328: Expected Growth Rate

    Computes expected continuous growth rate under optimal betting.
    g = r + S^2/2

    This is the theoretical maximum growth rate achievable with Kelly sizing.

    Academic Reference:
        Thorp, E.O. (2007). The Kelly Criterion in Blackjack Sports Betting,
        And The Stock Market.
    """

    FORMULA_ID = 328
    CATEGORY = "compounding"
    NAME = "ExpectedGrowthRateFormula"
    DESCRIPTION = "Expected growth rate g = r + S^2/2 (Thorp 2007)"

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        annualization: float = 365 * 24 * 60,
        growth_threshold: float = 0.10,   # 10% annual growth threshold
        **kwargs
    ):
        super().__init__(**kwargs)
        self.risk_free_rate = risk_free_rate
        self.annualization = annualization
        self.growth_threshold = growth_threshold

        # State
        self.sharpe_ratio = 0.0
        self.expected_growth = 0.0
        self.doubling_time = float('inf')

    def _compute(self) -> None:
        """Compute expected growth rate"""
        if len(self.returns) < self.min_samples:
            return

        returns = np.array(self.returns)

        # Mean and std
        mu = np.mean(returns) * self.annualization
        sigma = np.std(returns) * np.sqrt(self.annualization)

        # Sharpe ratio
        excess_return = mu - self.risk_free_rate
        self.sharpe_ratio = excess_return / sigma if sigma > 0 else 0

        # Expected growth: g = r + S^2/2
        self.expected_growth = self.risk_free_rate + (self.sharpe_ratio ** 2) / 2

        # Doubling time: T = ln(2) / g
        if self.expected_growth > 0:
            self.doubling_time = np.log(2) / self.expected_growth
        else:
            self.doubling_time = float('inf')

        # Signal based on growth potential
        if self.expected_growth > self.growth_threshold:
            self.signal = 1
            self.confidence = min(self.expected_growth / 0.5, 1.0)  # Max at 50% growth
        else:
            self.signal = 0
            self.confidence = self.expected_growth / self.growth_threshold

    def project_capital(self, initial: float, years: float) -> float:
        """Project capital growth"""
        return initial * np.exp(self.expected_growth * years)

    def time_to_target(self, initial: float, target: float) -> float:
        """Calculate time to reach target capital"""
        if self.expected_growth <= 0 or target <= initial:
            return float('inf')
        return np.log(target / initial) / self.expected_growth

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'sharpe_ratio': self.sharpe_ratio,
            'expected_growth': self.expected_growth,
            'doubling_time_years': self.doubling_time,
        }


# =============================================================================
# FORMULA 329: COMPOUND POSITION SIZER
# =============================================================================
# Combines Kelly, Optimal F, and Growth Rate for integrated position sizing
# =============================================================================

@FormulaRegistry.register(329, name="CompoundPositionSizerFormula", category="compounding")
class CompoundPositionSizerFormula(BaseFormula):
    """
    Formula 329: Compound Position Sizer

    Integrates Kelly (323), Optimal F (324), and Growth Rate (328)
    for optimal position sizing to compound from $10 to $300,000+.

    Uses ensemble of academic methods weighted by recent performance.
    """

    FORMULA_ID = 329
    CATEGORY = "compounding"
    NAME = "CompoundPositionSizerFormula"
    DESCRIPTION = "Integrated position sizing for capital compounding"

    def __init__(
        self,
        initial_capital: float = 10.0,
        target_capital: float = 300000.0,
        max_position_pct: float = 0.10,   # Max 10% of capital per trade
        reinvest_profits: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.initial_capital = initial_capital
        self.target_capital = target_capital
        self.max_position_pct = max_position_pct
        self.reinvest_profits = reinvest_profits

        # Sub-formulas
        self.kelly = KellyCriterionFormula(**kwargs)
        self.growth_rate = ExpectedGrowthRateFormula(**kwargs)

        # State
        self.current_capital = initial_capital
        self.optimal_position_size = 0.0
        self.compound_factor = 1.0
        self.trades_executed = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

    def update_capital(self, new_capital: float):
        """Update current capital after trade"""
        pnl = new_capital - self.current_capital
        self.total_pnl += pnl
        self.trades_executed += 1
        if pnl > 0:
            self.winning_trades += 1
        self.current_capital = new_capital
        self.compound_factor = new_capital / self.initial_capital

    def _compute(self) -> None:
        """Compute optimal position size"""
        # Update sub-formulas
        for price, vol, ts in zip(self.prices, self.volumes, self.timestamps):
            self.kelly.update(price, vol, ts)
            self.growth_rate.update(price, vol, ts)

        if not self.kelly.is_ready or not self.growth_rate.is_ready:
            return

        # Get Kelly optimal fraction
        kelly_fraction = self.kelly.optimal_f * self.kelly.kelly_fraction

        # Get expected growth
        growth = self.growth_rate.expected_growth

        # Adjust position size based on growth potential
        if growth > 0.5:  # >50% expected growth
            growth_multiplier = 1.5
        elif growth > 0.2:
            growth_multiplier = 1.2
        elif growth > 0.1:
            growth_multiplier = 1.0
        else:
            growth_multiplier = 0.5

        # Compute position size
        base_position = self.current_capital * kelly_fraction * growth_multiplier

        # Apply max position constraint
        self.optimal_position_size = min(
            base_position,
            self.current_capital * self.max_position_pct
        )

        # Signal
        self.signal = self.kelly.signal
        self.confidence = (self.kelly.confidence + self.growth_rate.confidence) / 2

    def get_position_size(self) -> float:
        """Get recommended position size in USD"""
        return self.optimal_position_size

    def get_progress_to_target(self) -> float:
        """Get progress toward target as percentage"""
        if self.current_capital <= self.initial_capital:
            return 0.0
        log_progress = np.log(self.current_capital / self.initial_capital)
        log_target = np.log(self.target_capital / self.initial_capital)
        return (log_progress / log_target) * 100

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'current_capital': self.current_capital,
            'compound_factor': self.compound_factor,
            'optimal_position_size': self.optimal_position_size,
            'trades_executed': self.trades_executed,
            'win_rate': self.winning_trades / max(1, self.trades_executed),
            'total_pnl': self.total_pnl,
            'progress_to_target_pct': self.get_progress_to_target(),
            'kelly_state': self.kelly.get_state(),
            'growth_state': self.growth_rate.get_state(),
        }


# =============================================================================
# FORMULA 330: CONTINUOUS COMPOUNDING OPTIMIZER
# =============================================================================
# Full integration of all compounding formulas for autonomous capital growth
# =============================================================================

@FormulaRegistry.register(330, name="ContinuousCompoundingOptimizer", category="compounding")
class ContinuousCompoundingOptimizer(BaseFormula):
    """
    Formula 330: Continuous Compounding Optimizer

    Master formula integrating all compounding strategies for autonomous
    capital growth from $10 to $300,000+.

    Combines:
    - Kelly Criterion (323) for optimal leverage
    - Optimal F (324) for position sizing
    - Volatility-Adjusted Growth (325) for drag adjustment
    - Almgren-Chriss (326) for execution
    - Avellaneda-Stoikov (327) for market making
    - Expected Growth Rate (328) for projections
    - Compound Position Sizer (329) for integration
    """

    FORMULA_ID = 330
    CATEGORY = "compounding"
    NAME = "ContinuousCompoundingOptimizer"
    DESCRIPTION = "Master compounding optimizer for capital growth"

    def __init__(
        self,
        initial_capital: float = 10.0,
        target_capital: float = 300000.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.initial_capital = initial_capital
        self.target_capital = target_capital

        # Initialize all sub-formulas
        self.kelly = KellyCriterionFormula(**kwargs)
        self.optimal_f = OptimalFFormula(**kwargs)
        self.vol_adjusted = VolatilityAdjustedGrowthFormula(**kwargs)
        self.almgren_chriss = AlmgrenChrissExecutionFormula(**kwargs)
        self.avellaneda = AvellanedaStoikovFormula(**kwargs)
        self.growth_rate = ExpectedGrowthRateFormula(**kwargs)
        self.position_sizer = CompoundPositionSizerFormula(
            initial_capital=initial_capital,
            target_capital=target_capital,
            **kwargs
        )

        # State
        self.current_capital = initial_capital
        self.position_recommendation = 0.0
        self.leverage_recommendation = 1.0
        self.execution_urgency = 0.5
        self.bid_ask_spread = 0.0

    def _compute(self) -> None:
        """Compute integrated recommendation"""
        if len(self.prices) < self.min_samples:
            return

        price = self.prices[-1]
        vol = self.volumes[-1] if self.volumes else 0
        ts = self.timestamps[-1] if self.timestamps else 0

        # Update all sub-formulas
        self.kelly.update(price, vol, ts)
        self.optimal_f.update(price, vol, ts)
        self.vol_adjusted.update(price, vol, ts)
        self.almgren_chriss.update(price, vol, ts)
        self.avellaneda.update(price, vol, ts)
        self.growth_rate.update(price, vol, ts)
        self.position_sizer.update(price, vol, ts)

        # Compute integrated recommendations

        # 1. Leverage from Kelly (adjusted for vol drag)
        kelly_leverage = self.kelly.leverage
        if self.vol_adjusted.is_vol_drag_high:
            kelly_leverage *= 0.5  # Reduce if vol is killing returns
        self.leverage_recommendation = kelly_leverage

        # 2. Position size from integrated sizer
        self.position_recommendation = self.position_sizer.get_position_size()

        # 3. Execution urgency from Almgren-Chriss kappa
        self.execution_urgency = min(self.almgren_chriss.kappa, 1.0) if self.almgren_chriss.kappa > 0 else 0.5

        # 4. Spread from Avellaneda-Stoikov
        self.bid_ask_spread = self.avellaneda.optimal_spread

        # Ensemble signal
        signals = [
            (self.kelly.signal, self.kelly.confidence),
            (self.vol_adjusted.signal, self.vol_adjusted.confidence),
            (self.position_sizer.signal, self.position_sizer.confidence),
        ]

        weighted_signal = sum(s * c for s, c in signals) / sum(c for _, c in signals) if sum(c for _, c in signals) > 0 else 0

        if weighted_signal > 0.3:
            self.signal = 1
        elif weighted_signal < -0.3:
            self.signal = -1
        else:
            self.signal = 0

        self.confidence = min(abs(weighted_signal), 1.0)

    def update_capital(self, new_capital: float):
        """Update capital after trade"""
        self.current_capital = new_capital
        self.position_sizer.update_capital(new_capital)

    def get_full_recommendation(self) -> Dict[str, Any]:
        """Get complete trading recommendation"""
        return {
            'signal': self.signal,
            'confidence': self.confidence,
            'position_size_usd': self.position_recommendation,
            'leverage': self.leverage_recommendation,
            'execution_urgency': self.execution_urgency,
            'optimal_spread': self.bid_ask_spread,
            'expected_growth_rate': self.growth_rate.expected_growth,
            'sharpe_ratio': self.kelly.sharpe,
            'volatility_drag': self.vol_adjusted.volatility_drag,
            'time_to_target_years': self.growth_rate.time_to_target(
                self.current_capital,
                self.target_capital
            ),
            'progress_pct': self.position_sizer.get_progress_to_target(),
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'current_capital': self.current_capital,
            'position_recommendation': self.position_recommendation,
            'leverage_recommendation': self.leverage_recommendation,
            'execution_urgency': self.execution_urgency,
            'bid_ask_spread': self.bid_ask_spread,
            'sub_formulas': {
                'kelly': self.kelly.get_state(),
                'optimal_f': self.optimal_f.get_state(),
                'vol_adjusted': self.vol_adjusted.get_state(),
                'growth_rate': self.growth_rate.get_state(),
            }
        }


# =============================================================================
# COMPOUNDING MANAGER
# =============================================================================

class CompoundingManager:
    """
    Manager class for compounding formulas (323-330).
    Provides high-level interface for capital growth optimization.
    """

    def __init__(
        self,
        initial_capital: float = 10.0,
        target_capital: float = 300000.0,
    ):
        self.optimizer = ContinuousCompoundingOptimizer(
            initial_capital=initial_capital,
            target_capital=target_capital,
        )
        self.initial_capital = initial_capital
        self.target_capital = target_capital
        self.trade_history = []

    def update(self, price: float, volume: float = 0.0, timestamp: float = 0.0):
        """Update with new market data"""
        self.optimizer.update(price, volume, timestamp)

    def get_recommendation(self) -> Dict[str, Any]:
        """Get trading recommendation"""
        return self.optimizer.get_full_recommendation()

    def record_trade(self, pnl: float, new_capital: float):
        """Record completed trade"""
        self.trade_history.append({
            'pnl': pnl,
            'capital_after': new_capital,
        })
        self.optimizer.update_capital(new_capital)

    def get_stats(self) -> Dict[str, Any]:
        """Get compounding statistics"""
        if not self.trade_history:
            return {'trades': 0}

        pnls = [t['pnl'] for t in self.trade_history]
        return {
            'trades': len(self.trade_history),
            'total_pnl': sum(pnls),
            'win_rate': sum(1 for p in pnls if p > 0) / len(pnls),
            'avg_pnl': np.mean(pnls),
            'current_capital': self.trade_history[-1]['capital_after'],
            'compound_factor': self.trade_history[-1]['capital_after'] / self.initial_capital,
            'progress_pct': self.optimizer.position_sizer.get_progress_to_target(),
        }
