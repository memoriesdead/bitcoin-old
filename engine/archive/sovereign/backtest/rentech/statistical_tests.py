"""
Statistical Validation - RenTech-style significance testing

Validates that trading edges are statistically real, not random chance:
- Binomial test for win rate significance
- Sharpe ratio calculation
- Kelly criterion for position sizing
- Bootstrap confidence intervals
"""
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats

from .walk_forward import WalkForwardResult, Trade


@dataclass
class StrategyResult:
    """Complete strategy validation results."""
    strategy_name: str
    category: str
    description: str

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Return statistics
    total_pnl_pct: float
    avg_pnl_pct: float
    avg_win_pct: float
    avg_loss_pct: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    profit_factor: float  # gross_profit / gross_loss

    # Statistical significance
    p_value: float
    z_score: float
    ci_lower: float  # 95% CI
    ci_upper: float

    # Kelly sizing
    kelly_fraction: float
    expected_growth: float  # E[log(1 + f*edge)]

    # Walk-forward metrics
    wf_windows: int
    wf_avg_train_wr: float
    wf_avg_test_wr: float
    wf_degradation: float
    wf_stability: float  # 1 - std(win_rates)

    # Validation flags
    is_significant: bool      # p < 0.01
    meets_rentech: bool       # win_rate >= 50.75%
    passes_walkforward: bool  # degradation < 20%
    recommendation: str       # IMPLEMENT, NO_EDGE, OVERFITTING, etc.


class StatisticalValidator:
    """
    Validate trading strategies with statistical rigor.

    RenTech Thresholds:
    - Minimum 500 trades
    - Win rate >= 50.75%
    - P-value < 0.01 (99% confidence)
    - Walk-forward degradation < 20%
    """

    MIN_TRADES = 500
    MIN_WIN_RATE = 0.5075
    MAX_P_VALUE = 0.01
    MAX_DEGRADATION = 0.20

    def validate(
        self,
        strategy_name: str,
        category: str,
        description: str,
        trades: List[Trade],
        wf_result: WalkForwardResult
    ) -> StrategyResult:
        """Full validation of a strategy."""

        # Basic trade stats
        total_trades = len(trades)
        if total_trades == 0:
            return self._empty_result(strategy_name, category, description)

        winning_trades = sum(1 for t in trades if t.pnl_pct > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades

        # Return statistics
        pnls = [t.pnl_pct for t in trades]
        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls)

        wins = [t.pnl_pct for t in trades if t.pnl_pct > 0]
        losses = [t.pnl_pct for t in trades if t.pnl_pct <= 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Risk metrics
        sharpe = self.calculate_sharpe(pnls)
        sortino = self.calculate_sortino(pnls)
        max_dd = self.calculate_max_drawdown(pnls)

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Statistical significance
        p_value = self.binomial_test(winning_trades, total_trades)
        z_score = self.calculate_zscore(winning_trades, total_trades)
        ci_lower, ci_upper = self.wilson_confidence_interval(winning_trades, total_trades)

        # Kelly criterion
        kelly = self.calculate_kelly(win_rate, avg_win, abs(avg_loss))
        expected_growth = self.calculate_expected_growth(win_rate, avg_win, abs(avg_loss), kelly)

        # Walk-forward metrics
        wf_windows = len(wf_result.windows)
        wf_avg_train_wr = wf_result.avg_train_win_rate
        wf_avg_test_wr = wf_result.avg_test_win_rate
        wf_degradation = wf_result.avg_degradation
        wf_stability = 1 - wf_result.win_rate_std

        # Validation flags
        is_significant = p_value < self.MAX_P_VALUE
        meets_rentech = win_rate >= self.MIN_WIN_RATE
        passes_walkforward = wf_degradation < self.MAX_DEGRADATION

        # Recommendation
        recommendation = self._get_recommendation(
            total_trades, win_rate, p_value, wf_degradation
        )

        return StrategyResult(
            strategy_name=strategy_name,
            category=category,
            description=description,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl_pct=total_pnl,
            avg_pnl_pct=avg_pnl,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
            p_value=p_value,
            z_score=z_score,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            kelly_fraction=kelly,
            expected_growth=expected_growth,
            wf_windows=wf_windows,
            wf_avg_train_wr=wf_avg_train_wr,
            wf_avg_test_wr=wf_avg_test_wr,
            wf_degradation=wf_degradation,
            wf_stability=wf_stability,
            is_significant=is_significant,
            meets_rentech=meets_rentech,
            passes_walkforward=passes_walkforward,
            recommendation=recommendation,
        )

    def _empty_result(self, name: str, category: str, desc: str) -> StrategyResult:
        """Return empty result for strategy with no trades."""
        return StrategyResult(
            strategy_name=name,
            category=category,
            description=desc,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            total_pnl_pct=0, avg_pnl_pct=0, avg_win_pct=0, avg_loss_pct=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0, profit_factor=0,
            p_value=1.0, z_score=0, ci_lower=0, ci_upper=0,
            kelly_fraction=0, expected_growth=0,
            wf_windows=0, wf_avg_train_wr=0, wf_avg_test_wr=0,
            wf_degradation=0, wf_stability=0,
            is_significant=False, meets_rentech=False, passes_walkforward=False,
            recommendation="NO_TRADES",
        )

    def binomial_test(self, wins: int, total: int, null_p: float = 0.5) -> float:
        """
        Calculate p-value using binomial test.

        Tests if win rate is significantly different from 50%.
        """
        if total == 0:
            return 1.0

        # Two-tailed binomial test
        result = stats.binomtest(wins, total, null_p, alternative='two-sided')
        return result.pvalue

    def calculate_zscore(self, wins: int, total: int, null_p: float = 0.5) -> float:
        """Calculate z-score for win rate."""
        if total == 0:
            return 0.0

        expected = total * null_p
        std = math.sqrt(total * null_p * (1 - null_p))

        if std == 0:
            return 0.0

        return (wins - expected) / std

    def wilson_confidence_interval(
        self,
        wins: int,
        total: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval for win rate.

        More accurate than normal approximation for proportions.
        """
        if total == 0:
            return 0.0, 0.0

        p = wins / total
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        z2 = z * z

        denominator = 1 + z2 / total
        center = (p + z2 / (2 * total)) / denominator
        spread = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator

        return max(0, center - spread), min(1, center + spread)

    def calculate_sharpe(self, returns: List[float], rf: float = 0.0) -> float:
        """
        Calculate annualized Sharpe ratio.

        Sharpe = (mean_return - rf) / std_return * sqrt(252)
        """
        if not returns or len(returns) < 2:
            return 0.0

        returns = np.array(returns) / 100  # Convert from percentage
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret == 0:
            return 0.0

        # Annualize assuming daily returns
        return (mean_ret - rf) / std_ret * math.sqrt(252)

    def calculate_sortino(self, returns: List[float], rf: float = 0.0) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).

        Sortino = (mean_return - rf) / downside_std * sqrt(252)
        """
        if not returns or len(returns) < 2:
            return 0.0

        returns = np.array(returns) / 100
        mean_ret = np.mean(returns)

        # Only consider negative returns for downside
        downside = returns[returns < rf]
        if len(downside) < 2:
            return 0.0

        downside_std = np.std(downside, ddof=1)
        if downside_std == 0:
            return 0.0

        return (mean_ret - rf) / downside_std * math.sqrt(252)

    def calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        if not returns:
            return 0.0

        # Convert to cumulative equity curve
        equity = [100]  # Start at 100
        for r in returns:
            equity.append(equity[-1] * (1 + r / 100))

        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak

        return float(np.max(drawdown))

    def calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly criterion optimal bet size.

        f* = (b*p - q) / b

        where:
        - b = odds (avg_win / avg_loss)
        - p = probability of winning
        - q = probability of losing
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        b = abs(avg_win) / abs(avg_loss)  # Odds ratio
        p = win_rate
        q = 1 - win_rate

        kelly = (b * p - q) / b

        # Cap at reasonable levels and use quarter-Kelly for safety
        kelly = max(0, min(kelly, 0.25))  # Quarter-Kelly max 25%

        return kelly

    def calculate_expected_growth(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly: float
    ) -> float:
        """
        Calculate expected log growth rate.

        E[log(1 + f*X)] where X is return
        """
        if kelly == 0:
            return 0.0

        # Expected growth = p*log(1+f*W) + q*log(1-f*L)
        p = win_rate
        q = 1 - win_rate
        w = avg_win / 100  # Convert from percentage
        l = abs(avg_loss) / 100

        growth = p * math.log(1 + kelly * w) + q * math.log(1 - kelly * l)

        return growth

    def _get_recommendation(
        self,
        total_trades: int,
        win_rate: float,
        p_value: float,
        degradation: float
    ) -> str:
        """Get recommendation based on validation results."""

        if total_trades < self.MIN_TRADES:
            return "NEED_MORE_DATA"

        if win_rate < 0.50:
            return "NO_EDGE"

        if win_rate < self.MIN_WIN_RATE:
            return "EDGE_TOO_SMALL"

        if p_value >= self.MAX_P_VALUE:
            return "NOT_SIGNIFICANT"

        if degradation >= self.MAX_DEGRADATION:
            return "OVERFITTING"

        return "IMPLEMENT"


def results_to_dict(result: StrategyResult) -> Dict:
    """Convert StrategyResult to dictionary for JSON export."""
    return {
        'strategy_name': result.strategy_name,
        'category': result.category,
        'description': result.description,
        'total_trades': int(result.total_trades),
        'winning_trades': int(result.winning_trades),
        'win_rate': float(round(result.win_rate, 4)),
        'total_pnl_pct': float(round(result.total_pnl_pct, 2)),
        'avg_pnl_pct': float(round(result.avg_pnl_pct, 3)),
        'sharpe_ratio': float(round(result.sharpe_ratio, 2)),
        'sortino_ratio': float(round(result.sortino_ratio, 2)),
        'max_drawdown': float(round(result.max_drawdown, 3)),
        'profit_factor': float(round(result.profit_factor, 2)),
        'p_value': float(round(result.p_value, 4)),
        'z_score': float(round(result.z_score, 2)),
        'ci_lower': float(round(result.ci_lower, 4)),
        'ci_upper': float(round(result.ci_upper, 4)),
        'kelly_fraction': float(round(result.kelly_fraction, 4)),
        'wf_windows': int(result.wf_windows),
        'wf_avg_test_wr': float(round(result.wf_avg_test_wr, 4)),
        'wf_degradation': float(round(result.wf_degradation, 4)),
        'wf_stability': float(round(result.wf_stability, 4)),
        'is_significant': bool(result.is_significant),
        'meets_rentech': bool(result.meets_rentech),
        'passes_walkforward': bool(result.passes_walkforward),
        'recommendation': result.recommendation,
    }


def quick_test():
    """Quick test of statistical validator."""
    validator = StatisticalValidator()

    # Test binomial
    print("Binomial Test Examples:")
    for wins, total in [(510, 1000), (520, 1000), (540, 1000), (600, 1000)]:
        p = validator.binomial_test(wins, total)
        z = validator.calculate_zscore(wins, total)
        ci_l, ci_u = validator.wilson_confidence_interval(wins, total)
        print(f"  {wins}/{total} = {wins/total*100:.1f}%: p={p:.4f}, z={z:.2f}, CI=[{ci_l*100:.1f}%, {ci_u*100:.1f}%]")

    # Test Kelly
    print("\nKelly Criterion Examples:")
    for wr, aw, al in [(0.51, 2.0, 1.5), (0.55, 2.0, 2.0), (0.60, 3.0, 2.0)]:
        k = validator.calculate_kelly(wr, aw, al)
        print(f"  WR={wr*100:.0f}%, AvgWin={aw}%, AvgLoss={al}%: Kelly={k*100:.1f}%")


if __name__ == "__main__":
    quick_test()
