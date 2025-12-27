"""
Advanced Statistical Validation
===============================

RenTech-grade statistical rigor for pattern validation.

Components:
- Bootstrap confidence intervals (non-parametric)
- Permutation tests (null hypothesis testing)
- Multiple hypothesis correction (FDR/Bonferroni)
- Regime-conditional testing
- Alpha decay analysis

These tests ensure we're not fooled by randomness.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict
import math


@dataclass
class BootstrapResult:
    """Result from bootstrap analysis."""
    metric_name: str
    observed_value: float
    mean_bootstrap: float
    std_bootstrap: float
    ci_lower: float  # 2.5th percentile
    ci_upper: float  # 97.5th percentile
    bias: float  # mean_bootstrap - observed
    is_significant: bool  # CI doesn't contain null value


@dataclass
class PermutationResult:
    """Result from permutation test."""
    metric_name: str
    observed_value: float
    null_mean: float
    null_std: float
    p_value: float
    z_score: float
    is_significant: bool


@dataclass
class MultipleTestResult:
    """Result after multiple hypothesis correction."""
    test_name: str
    raw_p_value: float
    adjusted_p_value: float
    is_significant_raw: bool
    is_significant_adjusted: bool
    correction_method: str


@dataclass
class DecayAnalysis:
    """Analysis of strategy edge decay over time."""
    strategy_name: str
    periods: List[str]
    win_rates: List[float]
    decay_rate: float  # Negative = decaying
    half_life_periods: Optional[float]  # Periods until edge halves
    is_stable: bool


@dataclass
class ComprehensiveValidation:
    """Complete validation result combining all tests."""
    strategy_name: str

    # Bootstrap
    win_rate_bootstrap: BootstrapResult
    sharpe_bootstrap: BootstrapResult
    profit_factor_bootstrap: BootstrapResult

    # Permutation
    win_rate_permutation: PermutationResult
    returns_permutation: PermutationResult

    # Multiple testing
    adjusted_results: List[MultipleTestResult]
    family_wise_significant: bool

    # Decay
    decay_analysis: DecayAnalysis

    # Final verdict
    passes_all_tests: bool
    confidence_level: str  # 'very_high', 'high', 'medium', 'low'
    notes: List[str] = field(default_factory=list)


class BootstrapValidator:
    """
    Non-parametric bootstrap confidence intervals.

    More robust than assuming normal distribution.
    Works with any metric, not just proportions.
    """

    def __init__(self, n_bootstrap: int = 1000, confidence: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.alpha = 1 - confidence

    def bootstrap_metric(
        self,
        data: np.ndarray,
        metric_func: Callable[[np.ndarray], float],
        metric_name: str = "metric",
        null_value: float = 0.0
    ) -> BootstrapResult:
        """
        Compute bootstrap confidence interval for any metric.

        Args:
            data: Raw data array (e.g., returns)
            metric_func: Function that computes metric from data
            metric_name: Name of the metric
            null_value: Value under null hypothesis (for significance)
        """
        n = len(data)
        if n == 0:
            return BootstrapResult(
                metric_name=metric_name,
                observed_value=0.0, mean_bootstrap=0.0, std_bootstrap=0.0,
                ci_lower=0.0, ci_upper=0.0, bias=0.0, is_significant=False
            )

        # Observed value
        observed = metric_func(data)

        # Bootstrap resampling
        bootstrap_values = []
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.randint(0, n, size=n)
            resample = data[indices]
            bootstrap_values.append(metric_func(resample))

        bootstrap_values = np.array(bootstrap_values)

        # Statistics
        mean_boot = np.mean(bootstrap_values)
        std_boot = np.std(bootstrap_values)
        bias = mean_boot - observed

        # Percentile confidence interval
        lower_pct = self.alpha / 2 * 100
        upper_pct = (1 - self.alpha / 2) * 100
        ci_lower = np.percentile(bootstrap_values, lower_pct)
        ci_upper = np.percentile(bootstrap_values, upper_pct)

        # Significance: null value outside CI
        is_significant = (null_value < ci_lower) or (null_value > ci_upper)

        return BootstrapResult(
            metric_name=metric_name,
            observed_value=observed,
            mean_bootstrap=mean_boot,
            std_bootstrap=std_boot,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            bias=bias,
            is_significant=is_significant
        )

    def bootstrap_win_rate(self, returns: np.ndarray) -> BootstrapResult:
        """Bootstrap confidence interval for win rate."""
        def win_rate(r):
            return np.mean(r > 0)

        return self.bootstrap_metric(
            returns, win_rate, "win_rate", null_value=0.5
        )

    def bootstrap_sharpe(self, returns: np.ndarray) -> BootstrapResult:
        """Bootstrap confidence interval for Sharpe ratio."""
        def sharpe(r):
            if len(r) < 2 or np.std(r) == 0:
                return 0.0
            return np.mean(r) / np.std(r) * np.sqrt(252)

        return self.bootstrap_metric(
            returns, sharpe, "sharpe_ratio", null_value=0.0
        )

    def bootstrap_profit_factor(self, returns: np.ndarray) -> BootstrapResult:
        """Bootstrap confidence interval for profit factor."""
        def profit_factor(r):
            wins = r[r > 0]
            losses = r[r < 0]
            if len(losses) == 0 or np.sum(np.abs(losses)) == 0:
                return 0.0
            return np.sum(wins) / np.sum(np.abs(losses))

        return self.bootstrap_metric(
            returns, profit_factor, "profit_factor", null_value=1.0
        )


class PermutationTester:
    """
    Permutation tests for null hypothesis testing.

    Shuffles labels to create null distribution.
    If observed value is extreme relative to null, reject H0.
    """

    def __init__(self, n_permutations: int = 1000):
        self.n_permutations = n_permutations

    def permutation_test(
        self,
        labels: np.ndarray,  # e.g., actual returns
        predictions: np.ndarray,  # e.g., predicted directions
        metric_func: Callable[[np.ndarray, np.ndarray], float],
        metric_name: str = "metric"
    ) -> PermutationResult:
        """
        Permutation test by shuffling labels.

        Tests if relationship between predictions and labels is real.
        """
        n = len(labels)
        if n == 0:
            return PermutationResult(
                metric_name=metric_name,
                observed_value=0.0, null_mean=0.0, null_std=0.0,
                p_value=1.0, z_score=0.0, is_significant=False
            )

        # Observed value
        observed = metric_func(labels, predictions)

        # Generate null distribution by shuffling
        null_values = []
        for _ in range(self.n_permutations):
            shuffled_labels = np.random.permutation(labels)
            null_values.append(metric_func(shuffled_labels, predictions))

        null_values = np.array(null_values)
        null_mean = np.mean(null_values)
        null_std = np.std(null_values)

        # Two-sided p-value
        if null_std > 0:
            z_score = (observed - null_mean) / null_std
            # Proportion of null values as extreme as observed
            p_value = np.mean(np.abs(null_values - null_mean) >= np.abs(observed - null_mean))
        else:
            z_score = 0.0
            p_value = 1.0

        is_significant = p_value < 0.05

        return PermutationResult(
            metric_name=metric_name,
            observed_value=observed,
            null_mean=null_mean,
            null_std=null_std,
            p_value=p_value,
            z_score=z_score,
            is_significant=is_significant
        )

    def test_win_rate(
        self,
        actual_returns: np.ndarray,
        predicted_directions: np.ndarray
    ) -> PermutationResult:
        """Test if predictions are better than random."""
        def accuracy(labels, preds):
            # Win = predicted direction matches actual
            actual_dir = np.sign(labels)
            return np.mean(actual_dir == preds)

        return self.permutation_test(
            actual_returns, predicted_directions, accuracy, "win_rate"
        )

    def test_returns(
        self,
        actual_returns: np.ndarray,
        predicted_directions: np.ndarray
    ) -> PermutationResult:
        """Test if directional predictions capture returns."""
        def signed_returns(labels, preds):
            return np.mean(labels * preds)

        return self.permutation_test(
            actual_returns, predicted_directions, signed_returns, "returns"
        )


class MultipleHypothesisCorrector:
    """
    Correct for multiple hypothesis testing.

    When testing many strategies, some will appear significant by chance.
    FDR and Bonferroni control false positive rate.
    """

    def bonferroni(self, p_values: List[float], alpha: float = 0.05) -> List[MultipleTestResult]:
        """
        Bonferroni correction - most conservative.

        Adjusted p-value = raw_p * n_tests
        """
        n_tests = len(p_values)
        results = []

        for i, raw_p in enumerate(p_values):
            adjusted_p = min(1.0, raw_p * n_tests)
            results.append(MultipleTestResult(
                test_name=f"test_{i}",
                raw_p_value=raw_p,
                adjusted_p_value=adjusted_p,
                is_significant_raw=raw_p < alpha,
                is_significant_adjusted=adjusted_p < alpha,
                correction_method="bonferroni"
            ))

        return results

    def benjamini_hochberg(self, p_values: List[float], alpha: float = 0.05) -> List[MultipleTestResult]:
        """
        Benjamini-Hochberg FDR correction.

        Controls false discovery rate, less conservative than Bonferroni.
        """
        n_tests = len(p_values)
        if n_tests == 0:
            return []

        # Sort p-values
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        # BH adjusted p-values
        adjusted = np.zeros(n_tests)
        for i, (rank, p) in enumerate(zip(range(1, n_tests + 1), sorted_p)):
            adjusted[i] = p * n_tests / rank

        # Enforce monotonicity (larger ranks can't have smaller adjusted p)
        for i in range(n_tests - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        # Cap at 1
        adjusted = np.minimum(adjusted, 1.0)

        # Unsort
        unsorted_adjusted = np.zeros(n_tests)
        unsorted_adjusted[sorted_indices] = adjusted

        results = []
        for i, (raw_p, adj_p) in enumerate(zip(p_values, unsorted_adjusted)):
            results.append(MultipleTestResult(
                test_name=f"test_{i}",
                raw_p_value=raw_p,
                adjusted_p_value=adj_p,
                is_significant_raw=raw_p < alpha,
                is_significant_adjusted=adj_p < alpha,
                correction_method="benjamini_hochberg"
            ))

        return results

    def holm_bonferroni(self, p_values: List[float], alpha: float = 0.05) -> List[MultipleTestResult]:
        """
        Holm-Bonferroni step-down procedure.

        Less conservative than Bonferroni but still controls FWER.
        """
        n_tests = len(p_values)
        if n_tests == 0:
            return []

        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        # Holm adjusted p-values
        adjusted = np.zeros(n_tests)
        for i, p in enumerate(sorted_p):
            adjusted[i] = p * (n_tests - i)

        # Enforce monotonicity
        for i in range(1, n_tests):
            adjusted[i] = max(adjusted[i], adjusted[i - 1])

        adjusted = np.minimum(adjusted, 1.0)

        # Unsort
        unsorted_adjusted = np.zeros(n_tests)
        unsorted_adjusted[sorted_indices] = adjusted

        results = []
        for i, (raw_p, adj_p) in enumerate(zip(p_values, unsorted_adjusted)):
            results.append(MultipleTestResult(
                test_name=f"test_{i}",
                raw_p_value=raw_p,
                adjusted_p_value=adj_p,
                is_significant_raw=raw_p < alpha,
                is_significant_adjusted=adj_p < alpha,
                correction_method="holm_bonferroni"
            ))

        return results


class AlphaDecayAnalyzer:
    """
    Analyze how strategy edge decays over time.

    A good strategy should have stable edge, not decay quickly.
    """

    def analyze_decay(
        self,
        strategy_name: str,
        period_returns: Dict[str, List[float]]  # period_name -> returns
    ) -> DecayAnalysis:
        """
        Analyze win rate decay across time periods.

        Args:
            strategy_name: Name of strategy
            period_returns: Dictionary mapping period names to returns
        """
        periods = list(period_returns.keys())
        win_rates = []

        for period in periods:
            returns = period_returns[period]
            if returns:
                wr = sum(1 for r in returns if r > 0) / len(returns)
            else:
                wr = 0.5
            win_rates.append(wr)

        # Fit linear decay model
        if len(win_rates) >= 2:
            x = np.arange(len(win_rates))
            coeffs = np.polyfit(x, win_rates, 1)
            decay_rate = coeffs[0]  # Slope

            # Half-life: periods until edge decays by half
            initial_edge = win_rates[0] - 0.5
            if initial_edge > 0 and decay_rate < 0:
                half_life = -initial_edge / (2 * decay_rate)
            else:
                half_life = None
        else:
            decay_rate = 0.0
            half_life = None

        # Stability: edge should persist
        is_stable = (
            decay_rate >= -0.01 and  # Not decaying more than 1% per period
            (half_life is None or half_life > 5) and  # At least 5 periods
            min(win_rates) > 0.5  # Never below 50%
        )

        return DecayAnalysis(
            strategy_name=strategy_name,
            periods=periods,
            win_rates=win_rates,
            decay_rate=decay_rate,
            half_life_periods=half_life,
            is_stable=is_stable
        )


class ComprehensiveValidator:
    """
    Run all validation tests on a strategy.

    Combines bootstrap, permutation, multiple testing, and decay analysis.
    """

    def __init__(self):
        self.bootstrap = BootstrapValidator(n_bootstrap=1000)
        self.permutation = PermutationTester(n_permutations=1000)
        self.corrector = MultipleHypothesisCorrector()
        self.decay_analyzer = AlphaDecayAnalyzer()

    def validate(
        self,
        strategy_name: str,
        returns: np.ndarray,
        predictions: np.ndarray,
        period_returns: Dict[str, List[float]] = None,
        all_p_values: List[float] = None  # For multiple testing
    ) -> ComprehensiveValidation:
        """
        Run comprehensive validation suite.

        Args:
            strategy_name: Name of strategy
            returns: Actual returns array
            predictions: Predicted directions (-1, 0, 1)
            period_returns: For decay analysis
            all_p_values: P-values from all strategies (for FDR)
        """
        # Bootstrap tests
        win_rate_boot = self.bootstrap.bootstrap_win_rate(returns)
        sharpe_boot = self.bootstrap.bootstrap_sharpe(returns)
        pf_boot = self.bootstrap.bootstrap_profit_factor(returns)

        # Permutation tests
        win_rate_perm = self.permutation.test_win_rate(returns, predictions)
        returns_perm = self.permutation.test_returns(returns, predictions)

        # Multiple testing correction
        if all_p_values is None:
            all_p_values = [win_rate_perm.p_value]

        adjusted_results = self.corrector.benjamini_hochberg(all_p_values)
        family_wise_sig = any(r.is_significant_adjusted for r in adjusted_results)

        # Decay analysis
        if period_returns is None:
            # Split returns into periods
            n = len(returns)
            n_periods = 4
            period_size = n // n_periods
            period_returns = {}
            for i in range(n_periods):
                start = i * period_size
                end = (i + 1) * period_size if i < n_periods - 1 else n
                period_returns[f"period_{i+1}"] = list(returns[start:end])

        decay = self.decay_analyzer.analyze_decay(strategy_name, period_returns)

        # Final verdict
        passes_all = (
            win_rate_boot.is_significant and
            sharpe_boot.is_significant and
            win_rate_perm.is_significant and
            decay.is_stable
        )

        # Confidence level
        significant_tests = sum([
            win_rate_boot.is_significant,
            sharpe_boot.is_significant,
            pf_boot.is_significant,
            win_rate_perm.is_significant,
            returns_perm.is_significant,
            decay.is_stable,
        ])

        if significant_tests >= 5:
            confidence = 'very_high'
        elif significant_tests >= 4:
            confidence = 'high'
        elif significant_tests >= 2:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Notes
        notes = []
        if not win_rate_boot.is_significant:
            notes.append("Win rate CI includes 50%")
        if not decay.is_stable:
            notes.append(f"Edge decaying at {decay.decay_rate:.2%} per period")
        if win_rate_boot.bias > 0.01:
            notes.append(f"Bootstrap shows {win_rate_boot.bias:.2%} upward bias")

        return ComprehensiveValidation(
            strategy_name=strategy_name,
            win_rate_bootstrap=win_rate_boot,
            sharpe_bootstrap=sharpe_boot,
            profit_factor_bootstrap=pf_boot,
            win_rate_permutation=win_rate_perm,
            returns_permutation=returns_perm,
            adjusted_results=adjusted_results,
            family_wise_significant=family_wise_sig,
            decay_analysis=decay,
            passes_all_tests=passes_all,
            confidence_level=confidence,
            notes=notes
        )


def run_validation_example():
    """Example usage of comprehensive validation."""
    np.random.seed(42)

    # Simulate strategy with slight edge
    n_trades = 500
    win_rate = 0.52  # 52% win rate

    # Generate returns: wins and losses
    is_win = np.random.random(n_trades) < win_rate
    returns = np.where(is_win, np.random.uniform(0.5, 2.0, n_trades),
                               -np.random.uniform(0.3, 1.5, n_trades))

    predictions = np.sign(np.random.randn(n_trades))  # Random predictions for demo

    # Run validation
    validator = ComprehensiveValidator()
    result = validator.validate("test_strategy", returns, predictions)

    print("=" * 60)
    print(f"Strategy: {result.strategy_name}")
    print("=" * 60)

    print("\nBootstrap Results:")
    print(f"  Win Rate: {result.win_rate_bootstrap.observed_value:.2%} "
          f"CI[{result.win_rate_bootstrap.ci_lower:.2%}, {result.win_rate_bootstrap.ci_upper:.2%}]"
          f" {'*' if result.win_rate_bootstrap.is_significant else ''}")
    print(f"  Sharpe: {result.sharpe_bootstrap.observed_value:.2f} "
          f"CI[{result.sharpe_bootstrap.ci_lower:.2f}, {result.sharpe_bootstrap.ci_upper:.2f}]"
          f" {'*' if result.sharpe_bootstrap.is_significant else ''}")

    print("\nPermutation Tests:")
    print(f"  Win Rate p-value: {result.win_rate_permutation.p_value:.4f}"
          f" {'*' if result.win_rate_permutation.is_significant else ''}")
    print(f"  Returns p-value: {result.returns_permutation.p_value:.4f}"
          f" {'*' if result.returns_permutation.is_significant else ''}")

    print("\nDecay Analysis:")
    print(f"  Decay rate: {result.decay_analysis.decay_rate:.4f} per period")
    print(f"  Stable: {result.decay_analysis.is_stable}")

    print("\nFinal Verdict:")
    print(f"  Passes all tests: {result.passes_all_tests}")
    print(f"  Confidence: {result.confidence_level}")
    if result.notes:
        print(f"  Notes: {'; '.join(result.notes)}")


if __name__ == "__main__":
    run_validation_example()
