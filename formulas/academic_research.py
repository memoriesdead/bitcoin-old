"""
Renaissance Formula Library - Academic Research Implementation
==============================================================
IDs 300-310: Exact formulas from peer-reviewed academic papers

This module implements the EXACT mathematical formulas from the best
academic research used by Renaissance Technologies, Citadel, Two Sigma.

ACADEMIC SOURCES (All peer-reviewed):
=====================================

1. GRINOLD-KAHN FUNDAMENTAL LAW (1989, 2000)
   - Paper: "The Fundamental Law of Active Management"
   - Journal: Journal of Portfolio Management, 15(3), 30-37
   - PDF: https://joim.com/wp-content/uploads/emember/downloads/p0158.pdf
   - Formula: IR = TC * IC * sqrt(BR)

2. ALMGREN-CHRISS OPTIMAL EXECUTION (2000)
   - Paper: "Optimal Execution of Portfolio Transactions"
   - Journal: Journal of Risk, 3(2), 5-39
   - PDF: https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf
   - Formula: Cost = gamma*X + eta*sum(n_k^2/tau_k)

3. KELLY-THORP OPTIMAL LEVERAGE (1956, 2006)
   - Paper: "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
   - Book: Handbook of Asset and Liability Management (2006)
   - PDF: https://web.williams.edu/Mathematics/sjmiller/public_html/341/handouts/Thorpe_KellyCriterion2007.pdf
   - Formula: f* = (mu - r) / sigma^2

4. AVELLANEDA-STOIKOV MARKET MAKING (2008)
   - Paper: "High-frequency trading in a limit order book"
   - Journal: Quantitative Finance, 8(3), 217-224
   - PDF: https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf
   - Formula: r(s,q,t) = s - q*gamma*sigma^2*(T-t)

5. VPIN FLOW TOXICITY (2012)
   - Paper: "Flow Toxicity and Liquidity in a High Frequency World"
   - Authors: Easley, Lopez de Prado, O'Hara
   - Journal: Review of Financial Studies, 25(5), 1457-1493
   - SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695596
   - Formula: VPIN = sum(|V_buy - V_sell|) / (n * V_bucket)

6. KYLE'S LAMBDA (1985)
   - Paper: "Continuous Auctions and Insider Trading"
   - Journal: Econometrica, 53(6), 1315-1335
   - Formula: delta_p = lambda * (x + u)

7. BIS CRYPTO CARRY (2023)
   - Paper: "Crypto Carry"
   - Source: BIS Working Papers No 1087
   - PDF: https://www.bis.org/publ/work1087.pdf
   - Formula: Carry = (F - S) / S * (365 / days_to_expiry)

8. PERPETUAL FUTURES FUNDING (2023)
   - Paper: "Fundamentals of Perpetual Futures"
   - Authors: He, Manela (Washington University)
   - arXiv: https://arxiv.org/html/2212.06888v5
   - Finding: Sharpe Ratio 1.8-3.5 for funding arbitrage
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from scipy import stats
from scipy.optimize import minimize_scalar

from .base import BaseFormula, FormulaRegistry


# =============================================================================
# ID 300: GRINOLD-KAHN FUNDAMENTAL LAW (Academic Version)
# =============================================================================

@FormulaRegistry.register(300)
class GrinoldKahnAcademicFormula(BaseFormula):
    """
    ID 300: Grinold-Kahn Fundamental Law of Active Management (1989, 2000)

    ACADEMIC SOURCE:
    - Grinold, R. (1989). "The Fundamental Law of Active Management"
    - Journal of Portfolio Management, 15(3), 30-37
    - Grinold, R. & Kahn, R. (2000). "Active Portfolio Management" (Book)

    THE FORMULA:
    ============
    IR = TC * IC * sqrt(BR)

    Where:
    - IR = Information Ratio = E[R_active] / std(R_active)
    - TC = Transfer Coefficient (0 to 1, measures constraint impact)
    - IC = Information Coefficient = corr(forecast, realized)
    - BR = Breadth = number of independent bets per year

    MEDALLION FUND INSIGHT:
    =======================
    Robert Mercer revealed Medallion was right only 50.75% of the time.
    - IC = 2 * (0.5075 - 0.5) = 0.015
    - With BR = 10,000,000 trades/year
    - IR = 1.0 * 0.015 * sqrt(10,000,000) = 47.4

    This is why TRADE FREQUENCY matters more than WIN RATE!
    """

    FORMULA_ID = 300
    CATEGORY = "academic"
    NAME = "Grinold-Kahn Fundamental Law"
    DESCRIPTION = "IR = TC * IC * sqrt(BR) - The most important formula in quantitative finance"

    # Academic citation
    CITATION = "Grinold, R. (1989). Journal of Portfolio Management, 15(3), 30-37"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)

        # Core parameters
        self.transfer_coefficient = kwargs.get('tc', 1.0)  # TC: 0-1, constraint impact
        self.base_ic = kwargs.get('ic', 0.015)  # IC: Medallion's revealed IC

        # Trade tracking for empirical IC calculation
        self.forecasts = deque(maxlen=10000)
        self.realized = deque(maxlen=10000)
        self.trade_outcomes = deque(maxlen=10000)  # 1 for win, 0 for loss

        # Computed values
        self.empirical_ic = 0.0
        self.breadth = 0
        self.information_ratio = 0.0
        self.expected_active_return = 0.0

        # Annual trade projection
        self.trades_per_minute = kwargs.get('trades_per_minute', 6)

    def _compute(self) -> None:
        """Compute Information Ratio using Fundamental Law"""

        # Calculate empirical IC from trade outcomes
        if len(self.trade_outcomes) >= 30:
            win_rate = np.mean(list(self.trade_outcomes))
            # IC approximation for binary outcomes: IC = 2 * (WR - 0.5)
            self.empirical_ic = 2 * (win_rate - 0.5)
        else:
            self.empirical_ic = self.base_ic

        # Use better of empirical or assumed IC
        effective_ic = max(self.empirical_ic, self.base_ic)

        # Calculate Breadth (annual trades)
        # BR = trades_per_minute * 60 * 24 * 365
        self.breadth = self.trades_per_minute * 60 * 24 * 365

        # THE FUNDAMENTAL LAW: IR = TC * IC * sqrt(BR)
        self.information_ratio = (
            self.transfer_coefficient *
            effective_ic *
            np.sqrt(self.breadth)
        )

        # Expected active return (given typical active risk of 10%)
        active_risk = 0.10  # 10% tracking error typical
        self.expected_active_return = self.information_ratio * active_risk

        # Signal based on IR quality
        if self.information_ratio > 2.0:
            self.signal = 1  # Excellent edge
            self.confidence = min(1.0, self.information_ratio / 5.0)
        elif self.information_ratio > 1.0:
            self.signal = 1  # Good edge
            self.confidence = 0.7
        elif self.information_ratio > 0.5:
            self.signal = 0  # Marginal
            self.confidence = 0.4
        else:
            self.signal = -1  # Poor edge
            self.confidence = 0.8

    def record_trade(self, forecast: float, realized: float, won: bool):
        """Record a trade for IC calculation"""
        self.forecasts.append(forecast)
        self.realized.append(realized)
        self.trade_outcomes.append(1 if won else 0)

    def calculate_empirical_ic(self) -> float:
        """Calculate IC as correlation between forecasts and realized"""
        if len(self.forecasts) < 30:
            return self.base_ic

        forecasts = np.array(list(self.forecasts))
        realized = np.array(list(self.realized))

        # IC = correlation(forecast, realized)
        correlation, _ = stats.pearsonr(forecasts, realized)
        return correlation if not np.isnan(correlation) else self.base_ic

    def required_trades_for_target_ir(self, target_ir: float) -> int:
        """Calculate required annual trades for target IR"""
        # IR = TC * IC * sqrt(BR)
        # BR = (IR / (TC * IC))^2
        effective_ic = max(self.empirical_ic, self.base_ic)
        if effective_ic * self.transfer_coefficient <= 0:
            return float('inf')

        required_br = (target_ir / (self.transfer_coefficient * effective_ic)) ** 2
        return int(required_br)

    def get_medallion_comparison(self) -> Dict[str, Any]:
        """Compare our stats to Medallion Fund's revealed numbers"""
        medallion_wr = 0.5075
        medallion_ic = 0.015
        medallion_br = 10_000_000  # Estimated trades/year
        medallion_ir = medallion_ic * np.sqrt(medallion_br)

        our_wr = np.mean(list(self.trade_outcomes)) if self.trade_outcomes else 0.5

        return {
            'medallion_win_rate': medallion_wr,
            'medallion_ic': medallion_ic,
            'medallion_breadth': medallion_br,
            'medallion_ir': medallion_ir,
            'our_win_rate': our_wr,
            'our_ic': self.empirical_ic,
            'our_breadth': self.breadth,
            'our_ir': self.information_ratio,
            'ir_gap': medallion_ir - self.information_ratio
        }

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'information_ratio': self.information_ratio,
            'transfer_coefficient': self.transfer_coefficient,
            'empirical_ic': self.empirical_ic,
            'breadth': self.breadth,
            'expected_active_return': self.expected_active_return,
            'trades_recorded': len(self.trade_outcomes),
            'citation': self.CITATION
        })
        return state


# =============================================================================
# ID 301: ALMGREN-CHRISS MARKET IMPACT (Academic Version)
# =============================================================================

@FormulaRegistry.register(301)
class AlmgrenChrissAcademicFormula(BaseFormula):
    """
    ID 301: Almgren-Chriss Optimal Execution Model (2000)

    ACADEMIC SOURCE:
    - Almgren, R. & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions"
    - Journal of Risk, 3(2), 5-39
    - PDF: https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf

    THE MODEL:
    ==========
    Total Cost = Permanent Impact + Temporary Impact + Risk

    E[Cost] = gamma * X + eta * sum(v_k^2 * tau_k)
    Var[Cost] = sigma^2 * sum(x_k^2 * tau_k)

    Where:
    - X = total shares to execute
    - gamma = permanent impact coefficient
    - eta = temporary impact coefficient
    - v_k = trading rate in period k
    - tau_k = length of period k
    - sigma = price volatility
    - x_k = remaining shares at time k

    OPTIMAL TRAJECTORY:
    ==================
    x_t = X * sinh(kappa * (T-t)) / sinh(kappa * T)

    Where: kappa = sqrt(lambda * sigma^2 / eta)
    - lambda = risk aversion parameter

    APPLICATION:
    ============
    - Minimize market impact for large orders
    - Optimal trade scheduling across time
    - Balance urgency vs. cost
    """

    FORMULA_ID = 301
    CATEGORY = "academic"
    NAME = "Almgren-Chriss Optimal Execution"
    DESCRIPTION = "Minimize E[Cost] + lambda * Var[Cost] for optimal trade scheduling"

    CITATION = "Almgren, R. & Chriss, N. (2000). Journal of Risk, 3(2), 5-39"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)

        # Market impact parameters (calibrate to actual market)
        self.gamma = kwargs.get('gamma', 2.5e-7)  # Permanent impact
        self.eta = kwargs.get('eta', 2.5e-6)  # Temporary impact
        self.sigma = kwargs.get('sigma', 0.02)  # Daily volatility

        # Risk aversion
        self.risk_aversion = kwargs.get('lambda', 1e-6)

        # Execution state
        self.total_shares = 0
        self.remaining_shares = 0
        self.time_horizon = 0
        self.current_time = 0

        # Computed values
        self.kappa = 0
        self.optimal_trajectory = []
        self.expected_cost = 0
        self.cost_variance = 0

    def _compute(self) -> None:
        """Compute optimal execution parameters"""
        prices = self._prices_array()

        if len(prices) < 20:
            return

        # Update volatility estimate from recent prices
        returns = np.diff(np.log(prices[-20:]))
        self.sigma = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.02

        # Calculate kappa (urgency parameter)
        # kappa = sqrt(lambda * sigma^2 / eta)
        if self.eta > 0:
            self.kappa = np.sqrt(self.risk_aversion * self.sigma**2 / self.eta)
        else:
            self.kappa = 0.1

        # Signal based on execution urgency
        if self.kappa > 1.0:
            self.signal = 1  # High urgency - trade faster
            self.confidence = min(1.0, self.kappa / 2.0)
        elif self.kappa < 0.1:
            self.signal = -1  # Low urgency - trade slower
            self.confidence = 0.7
        else:
            self.signal = 0  # Normal pace
            self.confidence = 0.5

    def setup_execution(self, total_shares: float, time_horizon: int):
        """Setup an execution plan"""
        self.total_shares = total_shares
        self.remaining_shares = total_shares
        self.time_horizon = time_horizon
        self.current_time = 0
        self._compute_trajectory()

    def _compute_trajectory(self):
        """Compute optimal execution trajectory"""
        if self.time_horizon <= 0 or self.total_shares <= 0:
            return

        T = self.time_horizon
        X = self.total_shares

        self.optimal_trajectory = []
        for t in range(T + 1):
            try:
                # x_t = X * sinh(kappa * (T-t)) / sinh(kappa * T)
                x_t = X * np.sinh(self.kappa * (T - t)) / np.sinh(self.kappa * T)
            except (OverflowError, ZeroDivisionError):
                # Fallback to linear trajectory
                x_t = X * (T - t) / T

            self.optimal_trajectory.append({
                'time': t,
                'remaining': x_t,
                'trade_now': X / T if t < T else 0  # Linear fallback
            })

        # Calculate expected cost
        self._calculate_expected_cost()

    def _calculate_expected_cost(self):
        """Calculate expected execution cost"""
        if not self.optimal_trajectory or self.time_horizon <= 0:
            return

        X = self.total_shares
        T = self.time_horizon
        tau = 1  # Assume unit time periods

        # Permanent impact cost: gamma * X
        permanent_cost = self.gamma * X * X  # Squared for total impact

        # Temporary impact cost: eta * sum(v_k^2 * tau_k)
        v = X / T  # Average trading rate
        temporary_cost = self.eta * (v ** 2) * T * tau

        self.expected_cost = permanent_cost + temporary_cost

        # Cost variance: sigma^2 * integral(x_t^2 dt)
        avg_remaining = X / 2  # Approximation
        self.cost_variance = (self.sigma ** 2) * (avg_remaining ** 2) * T * tau

    def get_next_trade_size(self) -> float:
        """Get optimal trade size for next period"""
        if not self.optimal_trajectory or self.current_time >= self.time_horizon:
            return 0

        if self.current_time + 1 < len(self.optimal_trajectory):
            current = self.optimal_trajectory[self.current_time]['remaining']
            next_step = self.optimal_trajectory[self.current_time + 1]['remaining']
            return current - next_step
        return 0

    def advance_time(self, shares_executed: float):
        """Advance time and update state"""
        self.remaining_shares -= shares_executed
        self.current_time += 1

    def calculate_market_impact(self, order_size: float, market_volume: float) -> Dict[str, float]:
        """Calculate expected market impact for an order"""
        # Participation rate
        participation = order_size / market_volume if market_volume > 0 else 1.0

        # Permanent impact: gamma * order_size
        permanent = self.gamma * order_size

        # Temporary impact: eta * (order_size / time)^2 * time
        # Simplified: eta * order_size for instant execution
        temporary = self.eta * order_size * participation

        # Total impact in basis points
        impact_bps = (permanent + temporary) * 10000

        return {
            'permanent_impact': permanent,
            'temporary_impact': temporary,
            'total_impact': permanent + temporary,
            'impact_bps': impact_bps,
            'participation_rate': participation
        }

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'kappa': self.kappa,
            'gamma': self.gamma,
            'eta': self.eta,
            'sigma': self.sigma,
            'expected_cost': self.expected_cost,
            'cost_variance': self.cost_variance,
            'citation': self.CITATION
        })
        return state


# =============================================================================
# ID 302: KELLY-THORP OPTIMAL LEVERAGE (Academic Version)
# =============================================================================

@FormulaRegistry.register(302)
class KellyThorpAcademicFormula(BaseFormula):
    """
    ID 302: Kelly-Thorp Optimal Leverage Formula (1956, 2006)

    ACADEMIC SOURCES:
    - Kelly, J.L. (1956). "A New Interpretation of Information Rate"
    - Bell System Technical Journal, 35(4), 917-926
    - Thorp, E.O. (2006). "The Kelly Criterion in Blackjack, Sports Betting,
      and the Stock Market"
    - PDF: https://web.williams.edu/Mathematics/sjmiller/public_html/341/handouts/Thorpe_KellyCriterion2007.pdf

    THE FORMULAS:
    =============

    1. DISCRETE KELLY (for gambling/binary outcomes):
       f* = (p*b - q) / b = (p - q/b)
       Where: p = win probability, q = 1-p, b = odds ratio (win/loss ratio)

    2. CONTINUOUS KELLY (for investing):
       f* = (mu - r) / sigma^2
       Where: mu = expected return, r = risk-free rate, sigma = volatility

    3. FRACTIONAL KELLY (risk reduction):
       - 50% Kelly: 75% of return with 25% of variance
       - 25% Kelly: 50% of return with 6.25% of variance

    DRAWDOWN PROBABILITIES AT FULL KELLY:
    =====================================
    P(drawdown >= d) = d
    - 20% drawdown: 80% probability over time
    - 50% drawdown: 50% probability over time
    - 90% drawdown: 10% probability over time

    THORP'S FINDING:
    ================
    Kelly fraction for S&P 500: ~117% (use leverage!)
    But practitioners use 25-50% Kelly for safety.
    """

    FORMULA_ID = 302
    CATEGORY = "academic"
    NAME = "Kelly-Thorp Optimal Leverage"
    DESCRIPTION = "f* = (mu - r) / sigma^2 for optimal position sizing"

    CITATION = "Thorp, E.O. (2006). Handbook of Asset and Liability Management"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)

        # Risk-free rate
        self.risk_free_rate = kwargs.get('risk_free_rate', 0.05)  # 5% annual

        # Kelly fraction to use (for safety)
        self.kelly_fraction = kwargs.get('kelly_fraction', 0.25)  # Quarter Kelly default

        # Trade tracking
        self.trade_results = deque(maxlen=1000)
        self.equity_curve = deque(maxlen=10000)

        # Computed values
        self.full_kelly = 0.0
        self.fractional_kelly = 0.0
        self.expected_return = 0.0
        self.volatility = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown_prob = {}

        # Discrete Kelly parameters
        self.win_rate = 0.5
        self.win_loss_ratio = 1.0

    def _compute(self) -> None:
        """Compute Kelly optimal leverage"""
        prices = self._prices_array()

        if len(prices) < 20:
            return

        # Calculate returns
        returns = np.diff(np.log(prices))

        # Expected return (annualized)
        daily_return = np.mean(returns)
        self.expected_return = daily_return * 252

        # Volatility (annualized)
        daily_vol = np.std(returns)
        self.volatility = daily_vol * np.sqrt(252)

        # CONTINUOUS KELLY FORMULA: f* = (mu - r) / sigma^2
        if self.volatility > 0:
            excess_return = self.expected_return - self.risk_free_rate
            self.full_kelly = excess_return / (self.volatility ** 2)
        else:
            self.full_kelly = 0.0

        # Apply fractional Kelly
        self.fractional_kelly = self.full_kelly * self.kelly_fraction

        # Cap at reasonable leverage
        self.fractional_kelly = np.clip(self.fractional_kelly, -2.0, 10.0)

        # Sharpe ratio
        if self.volatility > 0:
            self.sharpe_ratio = (self.expected_return - self.risk_free_rate) / self.volatility

        # Calculate drawdown probabilities (at full Kelly)
        self._calculate_drawdown_probs()

        # Signal based on Kelly recommendation
        if self.fractional_kelly > 1.5:
            self.signal = 1  # Use leverage
            self.confidence = 0.9
        elif self.fractional_kelly > 0.5:
            self.signal = 1  # Positive allocation
            self.confidence = 0.7
        elif self.fractional_kelly < -0.5:
            self.signal = -1  # Short or avoid
            self.confidence = 0.8
        else:
            self.signal = 0  # Minimal allocation
            self.confidence = 0.4

    def _calculate_drawdown_probs(self):
        """Calculate drawdown probabilities at full Kelly"""
        # At full Kelly: P(max drawdown >= d) = d
        self.max_drawdown_prob = {
            0.10: 0.90,  # 90% chance of 10% drawdown
            0.20: 0.80,  # 80% chance of 20% drawdown
            0.30: 0.70,
            0.40: 0.60,
            0.50: 0.50,  # 50% chance of 50% drawdown
            0.60: 0.40,
            0.70: 0.30,
            0.80: 0.20,
            0.90: 0.10,
        }

    def calculate_discrete_kelly(self, win_rate: float, win_amount: float,
                                  loss_amount: float) -> float:
        """
        Calculate discrete Kelly for binary outcomes
        f* = (p*b - q) / b
        Where b = win_amount / loss_amount
        """
        p = win_rate
        q = 1 - p
        b = win_amount / loss_amount if loss_amount > 0 else 1.0

        self.win_rate = p
        self.win_loss_ratio = b

        kelly = (p * b - q) / b if b > 0 else 0
        return kelly

    def record_trade(self, pnl_pct: float):
        """Record trade result for Kelly calculation"""
        self.trade_results.append(pnl_pct)

    def get_optimal_position_size(self, capital: float) -> float:
        """Get optimal position size based on Kelly"""
        return capital * max(0, self.fractional_kelly)

    def get_kelly_comparison(self) -> Dict[str, Any]:
        """Compare different Kelly fractions"""
        if self.full_kelly <= 0:
            return {'message': 'Negative edge - do not trade'}

        # Expected growth rate at different Kelly fractions
        # g = mu - sigma^2 / 2 at full Kelly
        # At fraction f: g_f = f * mu - f^2 * sigma^2 / 2

        fractions = [0.25, 0.50, 0.75, 1.0]
        comparison = {}

        for frac in fractions:
            f = self.full_kelly * frac
            expected_growth = f * self.expected_return - (f ** 2) * (self.volatility ** 2) / 2
            variance_reduction = frac ** 2

            comparison[f'{int(frac*100)}% Kelly'] = {
                'leverage': f,
                'expected_growth': expected_growth,
                'variance_ratio': variance_reduction,
                'return_ratio': frac  # Fraction of full Kelly return
            }

        return comparison

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'full_kelly': self.full_kelly,
            'fractional_kelly': self.fractional_kelly,
            'kelly_fraction_used': self.kelly_fraction,
            'expected_return': self.expected_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'risk_free_rate': self.risk_free_rate,
            'citation': self.CITATION
        })
        return state


# =============================================================================
# ID 303: VPIN FLOW TOXICITY (Academic Version)
# =============================================================================

@FormulaRegistry.register(303)
class VPINAcademicFormula(BaseFormula):
    """
    ID 303: VPIN - Volume-Synchronized Probability of Informed Trading (2012)

    ACADEMIC SOURCE:
    - Easley, D., Lopez de Prado, M., & O'Hara, M. (2012)
    - "Flow Toxicity and Liquidity in a High Frequency World"
    - Review of Financial Studies, 25(5), 1457-1493
    - SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695596

    THE FORMULA:
    ============
    VPIN = sum(|V_buy - V_sell|) / (n * V_bucket)

    Where:
    - V_buy = estimated buy volume in bucket
    - V_sell = estimated sell volume in bucket
    - n = number of buckets
    - V_bucket = volume per bucket

    TRADE CLASSIFICATION (Bulk Volume Classification):
    =================================================
    For each bucket:
    V_buy = V * CDF(Z), where Z = (close - open) / sigma
    V_sell = V - V_buy

    TRADING RULES:
    ==============
    - VPIN > 0.7: TOXIC - Do NOT trade (informed traders active)
    - VPIN 0.5-0.7: CAUTION - Reduce position size 50%
    - VPIN 0.3-0.5: NORMAL - Standard trading
    - VPIN < 0.3: SAFE - Increase position size

    FLASH CRASH PREDICTION:
    ======================
    VPIN spiked to 0.9 before the May 6, 2010 Flash Crash,
    providing early warning of toxic order flow.
    """

    FORMULA_ID = 303
    CATEGORY = "academic"
    NAME = "VPIN Flow Toxicity"
    DESCRIPTION = "Detect toxic order flow from informed traders"

    CITATION = "Easley, Lopez de Prado, O'Hara (2012). Review of Financial Studies, 25(5)"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)

        # VPIN parameters
        self.bucket_volume = kwargs.get('bucket_volume', 10.0)  # Volume per bucket
        self.n_buckets = kwargs.get('n_buckets', 50)  # Number of buckets for VPIN

        # Bucket tracking
        self.current_bucket = {
            'volume': 0.0,
            'buy_volume': 0.0,
            'sell_volume': 0.0,
            'open_price': None,
            'close_price': None,
            'trades': []
        }
        self.completed_buckets = deque(maxlen=self.n_buckets)

        # VPIN value
        self.vpin = 0.5  # Start neutral
        self.vpin_history = deque(maxlen=1000)

        # Volatility for BVC classification
        self.price_std = 0.01

    def _compute(self) -> None:
        """Compute VPIN signal"""
        if len(self.completed_buckets) < self.n_buckets:
            # Not enough data yet
            self.signal = 0
            self.confidence = 0.3
            return

        # VPIN already calculated in update
        # Generate trading signal based on VPIN level

        if self.vpin > 0.7:
            # TOXIC - Strong signal to stay out
            self.signal = 0  # No new positions
            self.confidence = 0.0  # Zero confidence in any direction
        elif self.vpin > 0.5:
            # CAUTION - Reduce exposure
            self.signal = 0
            self.confidence = 0.5
        elif self.vpin < 0.3:
            # SAFE - Full confidence
            self.signal = 0  # No directional bias from VPIN
            self.confidence = 1.0
        else:
            # NORMAL
            self.signal = 0
            self.confidence = 0.75

    def _classify_volume_bvc(self, open_price: float, close_price: float,
                              volume: float, sigma: float) -> Tuple[float, float]:
        """
        Bulk Volume Classification (BVC) from the paper
        V_buy = V * CDF(Z), where Z = (close - open) / sigma
        """
        if sigma <= 0:
            sigma = 0.001 * open_price

        z = (close_price - open_price) / sigma

        # CDF of standard normal
        cdf_z = stats.norm.cdf(z)

        buy_volume = volume * cdf_z
        sell_volume = volume * (1 - cdf_z)

        return buy_volume, sell_volume

    def update(self, price: float, volume: float = 0.0, timestamp: float = 0.0) -> Optional[float]:
        """Update with new trade and compute VPIN"""

        # Initialize bucket open price
        if self.current_bucket['open_price'] is None:
            self.current_bucket['open_price'] = price

        # Always update close price
        self.current_bucket['close_price'] = price
        self.current_bucket['volume'] += volume
        self.current_bucket['trades'].append(price)

        vpin_updated = None

        # Check if bucket is full
        if self.current_bucket['volume'] >= self.bucket_volume:
            # Classify volume using BVC
            buy_vol, sell_vol = self._classify_volume_bvc(
                self.current_bucket['open_price'],
                self.current_bucket['close_price'],
                self.current_bucket['volume'],
                self.price_std
            )

            # Store order imbalance
            order_imbalance = abs(buy_vol - sell_vol)
            self.completed_buckets.append({
                'buy_volume': buy_vol,
                'sell_volume': sell_vol,
                'order_imbalance': order_imbalance,
                'total_volume': self.current_bucket['volume']
            })

            # Calculate VPIN
            if len(self.completed_buckets) == self.n_buckets:
                total_imbalance = sum(b['order_imbalance'] for b in self.completed_buckets)
                total_volume = sum(b['total_volume'] for b in self.completed_buckets)

                # VPIN = sum(|V_buy - V_sell|) / (n * V_bucket)
                self.vpin = total_imbalance / total_volume if total_volume > 0 else 0.5
                self.vpin_history.append(self.vpin)
                vpin_updated = self.vpin

                # Update price std for next BVC calculation
                if len(self.prices) > 20:
                    self.price_std = np.std(list(self.prices)[-20:])

                # Compute signal
                self.is_ready = True
                self._compute()

            # Reset bucket
            self.current_bucket = {
                'volume': 0.0,
                'buy_volume': 0.0,
                'sell_volume': 0.0,
                'open_price': None,
                'close_price': None,
                'trades': []
            }

        # Standard BaseFormula update
        super().update(price, volume, timestamp)

        return vpin_updated

    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on VPIN"""
        if self.vpin > 0.7:
            return 0.0  # Do not trade
        elif self.vpin > 0.5:
            return 0.5  # Half size
        elif self.vpin > 0.3:
            return 0.75
        else:
            return 1.0  # Full size

    def is_toxic(self) -> bool:
        """Check if market is currently toxic"""
        return self.vpin > 0.7

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'vpin': self.vpin,
            'buckets_filled': len(self.completed_buckets),
            'n_buckets': self.n_buckets,
            'is_toxic': self.is_toxic(),
            'position_multiplier': self.get_position_multiplier(),
            'citation': self.CITATION
        })
        return state


# =============================================================================
# ID 304: KYLE'S LAMBDA - MARKET IMPACT (Academic Version)
# =============================================================================

@FormulaRegistry.register(304)
class KyleLambdaAcademicFormula(BaseFormula):
    """
    ID 304: Kyle's Lambda - Price Impact Model (1985)

    ACADEMIC SOURCE:
    - Kyle, A.S. (1985). "Continuous Auctions and Insider Trading"
    - Econometrica, 53(6), 1315-1335

    THE MODEL:
    ==========
    delta_p = lambda * (x + u)

    Where:
    - delta_p = price change
    - lambda = Kyle's lambda (price impact coefficient)
    - x = informed order flow
    - u = uninformed (noise) order flow

    KYLE'S LAMBDA ESTIMATION:
    =========================
    lambda = sigma_v / (sigma_u * sqrt(T))

    Or empirically:
    lambda = Cov(delta_p, signed_volume) / Var(signed_volume)

    PRACTICAL USE:
    ==============
    - Higher lambda = less liquid market, more impact
    - Lower lambda = more liquid, less impact
    - Use to estimate execution costs before trading

    Expected cost for order size Q:
    Cost = lambda * Q^2 / 2
    """

    FORMULA_ID = 304
    CATEGORY = "academic"
    NAME = "Kyle's Lambda"
    DESCRIPTION = "delta_p = lambda * order_flow - Measure market depth and impact"

    CITATION = "Kyle, A.S. (1985). Econometrica, 53(6), 1315-1335"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)

        # Trade data for lambda estimation
        self.price_changes = deque(maxlen=1000)
        self.signed_volumes = deque(maxlen=1000)
        self.last_price = None

        # Kyle's lambda
        self.kyle_lambda = 0.0
        self.lambda_history = deque(maxlen=100)

        # Liquidity metrics
        self.market_depth = 0.0
        self.illiquidity_ratio = 0.0

    def _compute(self) -> None:
        """Compute Kyle's lambda"""
        if len(self.price_changes) < 30:
            return

        price_changes = np.array(self.price_changes)
        signed_volumes = np.array(self.signed_volumes)

        # KYLE'S LAMBDA = Cov(delta_p, signed_volume) / Var(signed_volume)
        var_volume = np.var(signed_volumes)
        if var_volume > 0:
            cov = np.cov(price_changes, signed_volumes)[0, 1]
            self.kyle_lambda = cov / var_volume
        else:
            self.kyle_lambda = 0.0

        self.lambda_history.append(self.kyle_lambda)

        # Market depth (inverse of lambda)
        if self.kyle_lambda > 0:
            self.market_depth = 1.0 / self.kyle_lambda
        else:
            self.market_depth = float('inf')

        # Amihud illiquidity ratio approximation
        if len(self.volumes) > 0:
            avg_volume = np.mean(list(self.volumes)[-20:])
            if avg_volume > 0:
                avg_abs_return = np.mean(np.abs(price_changes[-20:]))
                self.illiquidity_ratio = avg_abs_return / avg_volume

        # Signal based on lambda (liquidity)
        avg_lambda = np.mean(list(self.lambda_history)) if self.lambda_history else self.kyle_lambda

        if self.kyle_lambda > avg_lambda * 1.5:
            # Low liquidity - be careful
            self.signal = -1  # Reduce trading
            self.confidence = 0.8
        elif self.kyle_lambda < avg_lambda * 0.5:
            # High liquidity - good for trading
            self.signal = 1  # Can trade more
            self.confidence = 0.7
        else:
            self.signal = 0
            self.confidence = 0.5

    def update(self, price: float, volume: float = 0.0, timestamp: float = 0.0):
        """Update with trade data"""
        if self.last_price is not None:
            # Price change
            delta_p = price - self.last_price
            self.price_changes.append(delta_p)

            # Signed volume (positive if price up, negative if down)
            sign = 1 if delta_p >= 0 else -1
            self.signed_volumes.append(sign * volume)

        self.last_price = price

        # Standard update
        super().update(price, volume, timestamp)

    def estimate_impact(self, order_size: float, is_buy: bool = True) -> Dict[str, float]:
        """Estimate price impact for an order"""
        # Impact = lambda * order_flow
        direction = 1 if is_buy else -1

        linear_impact = self.kyle_lambda * direction * order_size

        # Square root impact (Almgren et al. finding)
        sqrt_impact = np.sign(direction) * self.kyle_lambda * np.sqrt(abs(order_size))

        # Cost = lambda * Q^2 / 2
        execution_cost = self.kyle_lambda * (order_size ** 2) / 2

        return {
            'linear_impact': linear_impact,
            'sqrt_impact': sqrt_impact,
            'execution_cost': execution_cost,
            'kyle_lambda': self.kyle_lambda
        }

    def get_max_order_size(self, max_impact_pct: float, current_price: float) -> float:
        """Calculate max order size for given impact tolerance"""
        if self.kyle_lambda <= 0:
            return float('inf')

        max_impact = max_impact_pct * current_price
        # impact = lambda * Q, so Q = impact / lambda
        return max_impact / self.kyle_lambda

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'kyle_lambda': self.kyle_lambda,
            'market_depth': self.market_depth,
            'illiquidity_ratio': self.illiquidity_ratio,
            'citation': self.CITATION
        })
        return state


# =============================================================================
# ID 305: CRYPTO CARRY / FUNDING ARBITRAGE (BIS Research)
# =============================================================================

@FormulaRegistry.register(305)
class CryptoCarryAcademicFormula(BaseFormula):
    """
    ID 305: Crypto Carry / Funding Rate Arbitrage

    ACADEMIC SOURCES:
    - BIS Working Papers No 1087 (2023). "Crypto Carry"
    - PDF: https://www.bis.org/publ/work1087.pdf
    - He, S. & Manela, A. (2023). "Fundamentals of Perpetual Futures"
    - arXiv: https://arxiv.org/html/2212.06888v5

    KEY FINDINGS:
    =============
    1. Funding rate arbitrage generates Sharpe Ratios of 1.8-3.5
    2. Bitcoin ETFs reduced carry by ~36%
    3. $100B+ daily volume in perpetual futures

    THE STRATEGY:
    =============
    Carry = (F - S) / S * (365 / days)

    For perpetuals:
    Funding_APR = Funding_Rate_8h * 3 * 365

    If funding > 0.01%: Buy spot, short perp (collect from longs)
    If funding < -0.01%: Short spot, long perp (collect from shorts)

    RISK:
    =====
    - Delta neutral (market direction doesn't matter)
    - Main risk: exchange counterparty risk
    - Secondary: funding rate reversal
    """

    FORMULA_ID = 305
    CATEGORY = "academic"
    NAME = "Crypto Carry (BIS Research)"
    DESCRIPTION = "Delta-neutral funding rate arbitrage - Sharpe 1.8-3.5"

    CITATION = "BIS Working Papers No 1087 (2023); He & Manela, arXiv:2212.06888"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)

        # Minimum APR to enter position
        self.min_apr = kwargs.get('min_apr', 0.10)  # 10% minimum

        # Funding rate tracking
        self.funding_rates = deque(maxlen=200)  # ~1 week of 8h rates
        self.funding_rate_8h = 0.0

        # Price tracking
        self.spot_price = 0.0
        self.perp_price = 0.0
        self.basis = 0.0

        # Position tracking
        self.in_position = False
        self.position_direction = 0  # 1 = long basis, -1 = short basis

        # Performance
        self.total_funding_collected = 0.0
        self.position_start_time = 0

    def _compute(self) -> None:
        """Compute funding arbitrage signal"""
        if len(self.funding_rates) < 3:
            self.signal = 0
            self.confidence = 0.3
            return

        # Average recent funding rates
        avg_funding = np.mean(list(self.funding_rates)[-21:])  # ~7 days

        # Calculate expected APR
        expected_apr = abs(avg_funding) * 3 * 365  # 3 funding periods per day

        # Funding rate volatility
        funding_std = np.std(list(self.funding_rates)) if len(self.funding_rates) > 5 else 0
        funding_stability = 1 - min(1, funding_std / (abs(avg_funding) + 0.0001))

        # Calculate Sharpe approximation
        if funding_std > 0:
            sharpe = (avg_funding * 3 * 365) / (funding_std * np.sqrt(365 * 3))
        else:
            sharpe = 0

        # Signal based on APR threshold
        if expected_apr >= self.min_apr and funding_stability > 0.5:
            if avg_funding > 0:
                # Positive funding: longs pay shorts
                # Strategy: buy spot, short perp
                self.signal = 1
                self.confidence = min(1.0, expected_apr / 0.30) * funding_stability
            else:
                # Negative funding: shorts pay longs
                # Strategy: short spot, long perp
                self.signal = -1
                self.confidence = min(1.0, expected_apr / 0.30) * funding_stability
        else:
            self.signal = 0
            self.confidence = 0.3

    def update_funding_rate(self, funding_rate_8h: float):
        """Update with new 8h funding rate"""
        self.funding_rate_8h = funding_rate_8h
        self.funding_rates.append(funding_rate_8h)

    def update_prices(self, spot: float, perp: float):
        """Update spot and perp prices"""
        self.spot_price = spot
        self.perp_price = perp
        self.basis = (perp - spot) / spot if spot > 0 else 0

    def calculate_position_metrics(self, capital: float) -> Dict[str, Any]:
        """Calculate expected returns for funding arbitrage position"""
        if len(self.funding_rates) < 3:
            return {'error': 'Insufficient funding rate data'}

        avg_funding = np.mean(list(self.funding_rates)[-21:])
        funding_std = np.std(list(self.funding_rates)) if len(self.funding_rates) > 5 else 0

        # Expected 8h return
        expected_8h_return = abs(avg_funding) * capital

        # Daily return (3 funding periods)
        expected_daily_return = expected_8h_return * 3

        # Annual return
        expected_annual_return = expected_daily_return * 365

        # Sharpe ratio (from BIS paper methodology)
        if funding_std > 0:
            daily_std = funding_std * capital * 3
            annual_std = daily_std * np.sqrt(365)
            sharpe = expected_annual_return / annual_std if annual_std > 0 else 0
        else:
            sharpe = 0

        return {
            'avg_funding_rate_8h': avg_funding,
            'funding_std': funding_std,
            'expected_8h_return': expected_8h_return,
            'expected_daily_return': expected_daily_return,
            'expected_annual_return': expected_annual_return,
            'expected_apr': expected_annual_return / capital if capital > 0 else 0,
            'estimated_sharpe': sharpe,
            'bis_reported_sharpe': '1.8 - 3.5 (from BIS Working Paper)'
        }

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'funding_rate_8h': self.funding_rate_8h,
            'avg_funding_7d': np.mean(list(self.funding_rates)[-21:]) if len(self.funding_rates) >= 21 else 0,
            'basis': self.basis,
            'total_funding_collected': self.total_funding_collected,
            'citation': self.CITATION
        })
        return state


# =============================================================================
# ID 306: MASTER ACADEMIC SCALER - Combines All Research
# =============================================================================

@FormulaRegistry.register(306)
class MasterAcademicScaler(BaseFormula):
    """
    ID 306: Master Academic Scaler - Combines All Research

    This formula integrates ALL academic research into a single scaling system:

    1. GRINOLD-KAHN: IR = TC * IC * sqrt(BR) for trade frequency optimization
    2. ALMGREN-CHRISS: Minimize market impact cost
    3. KELLY-THORP: Optimal leverage f* = (mu - r) / sigma^2
    4. VPIN: Filter out toxic flow periods
    5. KYLE: Estimate market depth and impact
    6. BIS CRYPTO: Funding rate arbitrage overlay

    COMBINED OUTPUT:
    ================
    - Optimal position size
    - Optimal trade frequency
    - Market toxicity filter
    - Expected edge with scaling
    - Days to target ($300k from $10)
    """

    FORMULA_ID = 306
    CATEGORY = "academic"
    NAME = "Master Academic Scaler"
    DESCRIPTION = "Combines Grinold-Kahn, Kelly, VPIN, Kyle into unified scaling"

    def __init__(self, lookback: int = 100, **kwargs):
        super().__init__(lookback, **kwargs)

        # Starting parameters
        self.capital = kwargs.get('capital', 10.0)
        self.target_capital = kwargs.get('target', 300000.0)
        self.daily_market_volume = kwargs.get('market_volume', 66_888_130_238)

        # Initialize component formulas
        self.grinold_kahn = GrinoldKahnAcademicFormula(**kwargs)
        self.almgren_chriss = AlmgrenChrissAcademicFormula(**kwargs)
        self.kelly_thorp = KellyThorpAcademicFormula(**kwargs)
        self.vpin = VPINAcademicFormula(**kwargs)
        self.kyle = KyleLambdaAcademicFormula(**kwargs)
        self.crypto_carry = CryptoCarryAcademicFormula(**kwargs)

        # Combined outputs
        self.optimal_position = 0.0
        self.optimal_leverage = 1.0
        self.optimal_trades_per_day = 0
        self.toxicity_filter = 1.0
        self.expected_daily_return = 0.0
        self.days_to_target = float('inf')

    def _compute(self) -> None:
        """Compute combined scaling recommendation"""

        # Get component outputs
        kelly_leverage = max(0.1, self.kelly_thorp.fractional_kelly)
        vpin_multiplier = self.vpin.get_position_multiplier()
        ir = self.grinold_kahn.information_ratio

        # 1. KELLY: Optimal leverage (capped at 20x for safety)
        self.optimal_leverage = min(kelly_leverage, 20.0)

        # 2. VPIN: Toxicity filter
        self.toxicity_filter = vpin_multiplier

        # 3. POSITION SIZE: Kelly * Capital * VPIN filter
        self.optimal_position = self.capital * self.optimal_leverage * self.toxicity_filter

        # 4. GRINOLD-KAHN: Optimal trade frequency
        # Target IR of 2.0 (good performance)
        required_trades = self.grinold_kahn.required_trades_for_target_ir(2.0)
        self.optimal_trades_per_day = min(required_trades / 365, 100000)  # Cap at 100k/day

        # 5. KYLE: Check if market can absorb our trades
        market_depth = self.kyle.market_depth
        safe_trade_size = market_depth * 0.01 if market_depth < float('inf') else self.optimal_position

        # Adjust position if needed
        if safe_trade_size < self.optimal_position:
            self.optimal_position = safe_trade_size

        # 6. Calculate expected returns - FROM LIVE DATA ONLY
        # NO hardcoded edge - must be measured from actual trades
        base_edge = self._live_measured_edge if hasattr(self, '_live_measured_edge') else 0.0
        daily_trades = self.optimal_trades_per_day

        # Edge amplification from Grinold-Kahn (only if we have real edge)
        if base_edge > 0 and daily_trades > 0:
            amplification = np.sqrt(daily_trades * 365)
            amplified_edge = base_edge * min(amplification / 1000, 10)
        else:
            amplified_edge = 0.0

        self.expected_daily_return = self.optimal_position * amplified_edge * daily_trades

    def set_live_edge(self, measured_edge: float):
        """Set edge from LIVE trade measurements"""
        self._live_measured_edge = measured_edge

        # 7. Days to target
        if self.expected_daily_return > 0:
            self.days_to_target = (self.target_capital - self.capital) / self.expected_daily_return
        else:
            self.days_to_target = float('inf')

        # Combined signal
        if self.toxicity_filter > 0.5 and ir > 0.5:
            self.signal = 1
            self.confidence = min(1.0, ir / 2.0) * self.toxicity_filter
        elif self.toxicity_filter < 0.3:
            self.signal = -1  # Market too toxic
            self.confidence = 0.9
        else:
            self.signal = 0
            self.confidence = 0.4

    def update(self, price: float, volume: float = 0.0, timestamp: float = 0.0):
        """Update all component formulas"""
        # Update all components
        self.grinold_kahn.update(price, volume, timestamp)
        self.almgren_chriss.update(price, volume, timestamp)
        self.kelly_thorp.update(price, volume, timestamp)
        self.vpin.update(price, volume, timestamp)
        self.kyle.update(price, volume, timestamp)
        self.crypto_carry.update(price, volume, timestamp)

        # Standard update
        super().update(price, volume, timestamp)

    def get_full_recommendation(self) -> Dict[str, Any]:
        """Get complete trading recommendation"""
        return {
            'position_size': self.optimal_position,
            'leverage': self.optimal_leverage,
            'trades_per_day': self.optimal_trades_per_day,
            'toxicity_filter': self.toxicity_filter,
            'expected_daily_return': self.expected_daily_return,
            'days_to_target': self.days_to_target,
            'components': {
                'grinold_kahn_ir': self.grinold_kahn.information_ratio,
                'kelly_leverage': self.kelly_thorp.fractional_kelly,
                'vpin': self.vpin.vpin,
                'kyle_lambda': self.kyle.kyle_lambda,
                'funding_apr': self.crypto_carry.funding_rate_8h * 3 * 365 if self.crypto_carry.funding_rate_8h else 0
            },
            'academic_sources': [
                'Grinold & Kahn (1989, 2000) - Fundamental Law',
                'Almgren & Chriss (2000) - Optimal Execution',
                'Kelly (1956), Thorp (2006) - Optimal Leverage',
                'Easley, Lopez de Prado, O\'Hara (2012) - VPIN',
                'Kyle (1985) - Market Impact',
                'BIS Working Paper 1087 (2023) - Crypto Carry'
            ]
        }

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            'optimal_position': self.optimal_position,
            'optimal_leverage': self.optimal_leverage,
            'optimal_trades_per_day': self.optimal_trades_per_day,
            'toxicity_filter': self.toxicity_filter,
            'expected_daily_return': self.expected_daily_return,
            'days_to_target': self.days_to_target
        })
        return state
