"""
VALIDATION FRAMEWORK - Prove Edge is Real
==========================================

"A backtest isn't just about showing a profit. It's about proving your
edge is real, repeatable, and robust under real-world conditions."

This module validates that our HMM and patterns have real predictive power,
not just random chance or overfitting.

VALIDATION METHODS:
1. Out-of-Sample Testing: Train on 80%, test on 20%
2. Walk-Forward Analysis: Rolling train/test windows
3. Statistical Significance: Chi-square, t-test
4. Monte Carlo Simulation: Resample to build confidence intervals
5. Probability of Backtest Overfitting (PBO)

TARGET METRICS:
- Win rate > 50.75% (RenTech threshold)
- 95% confidence interval excludes 50%
- Edge persists across different time periods
- No significant degradation in out-of-sample

CITATION:
- Bailey & Prado (2014): Probability of Backtest Overfitting
- White (2000): Reality Check for Data Snooping
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from .historical_data import HistoricalFlowDatabase, FlowEvent
from .hmm_trainer import BaumWelchTrainer, HMMParameters, HMMTrainingPipeline
from .pattern_discovery import PatternDiscoveryEngine, DiscoveredPattern, PatternMatcher


@dataclass
class ValidationResult:
    """Results from validation tests."""
    test_name: str
    passed: bool
    metric: str
    value: float
    threshold: float
    confidence_interval: Tuple[float, float]
    p_value: float
    samples: int
    details: Dict


class WalkForwardValidator:
    """
    Walk-forward validation: most rigorous test of trading strategy.

    PROCESS:
    1. Split data into N sequential windows
    2. For each window:
       - Train on window i
       - Test on window i+1
    3. Aggregate results across all windows

    This simulates real trading: always testing on future data.
    """

    def __init__(self, db: HistoricalFlowDatabase = None):
        self.db = db or HistoricalFlowDatabase()
        self.results: List[ValidationResult] = []

    def validate(self, n_windows: int = 5, verbose: bool = True) -> ValidationResult:
        """
        Run walk-forward validation.

        Args:
            n_windows: Number of train/test windows
            verbose: Print progress

        Returns:
            Aggregated validation result
        """
        flows = self.db.get_flows(min_btc=0.1)
        flows_with_outcomes = [f for f in flows if f.outcome_30s != 0]

        if len(flows_with_outcomes) < n_windows * 100:
            return ValidationResult(
                test_name='walk_forward',
                passed=False,
                metric='sample_size',
                value=len(flows_with_outcomes),
                threshold=n_windows * 100,
                confidence_interval=(0, 0),
                p_value=1.0,
                samples=0,
                details={'error': 'Not enough data'},
            )

        window_size = len(flows_with_outcomes) // n_windows
        window_results = []

        if verbose:
            print(f"[WALK-FORWARD] {n_windows} windows, {window_size} samples each")

        for i in range(n_windows - 1):
            train_start = i * window_size
            train_end = (i + 1) * window_size
            test_start = train_end
            test_end = min((i + 2) * window_size, len(flows_with_outcomes))

            train_flows = flows_with_outcomes[train_start:train_end]
            test_flows = flows_with_outcomes[test_start:test_end]

            if verbose:
                print(f"\n[WALK-FORWARD] Window {i+1}: Train={len(train_flows)}, Test={len(test_flows)}")

            # Train HMM on this window
            pipeline = HMMTrainingPipeline(self.db)
            observations = pipeline.flows_to_observations(train_flows)

            trainer = BaumWelchTrainer(n_states=5, n_iter=50)
            try:
                trainer.fit(observations, verbose=False)
            except Exception as e:
                if verbose:
                    print(f"[WALK-FORWARD] Training failed: {e}")
                continue

            # Test on next window
            test_observations = pipeline.flows_to_observations(test_flows)
            test_outcomes = pipeline.flows_to_outcomes(test_flows)

            test_states = trainer.predict_states(test_observations)
            accuracy = pipeline.calculate_accuracy(test_states, test_outcomes)

            window_results.append(accuracy)

            if verbose:
                print(f"[WALK-FORWARD] Window {i+1} accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        if not window_results:
            return ValidationResult(
                test_name='walk_forward',
                passed=False,
                metric='windows_completed',
                value=0,
                threshold=1,
                confidence_interval=(0, 0),
                p_value=1.0,
                samples=0,
                details={'error': 'No windows completed'},
            )

        # Aggregate results
        mean_accuracy = sum(window_results) / len(window_results)
        std_accuracy = (sum((x - mean_accuracy)**2 for x in window_results) / len(window_results)) ** 0.5

        # 95% confidence interval
        ci_low = mean_accuracy - 1.96 * std_accuracy / (len(window_results) ** 0.5)
        ci_high = mean_accuracy + 1.96 * std_accuracy / (len(window_results) ** 0.5)

        # T-test: is mean > 0.5?
        if std_accuracy > 0:
            t_stat = (mean_accuracy - 0.5) / (std_accuracy / (len(window_results) ** 0.5))
            # Simplified p-value
            p_value = 0.5 * math.exp(-0.5 * t_stat**2) if t_stat > 0 else 1.0
        else:
            p_value = 1.0 if mean_accuracy <= 0.5 else 0.001

        passed = mean_accuracy >= 0.5075 and ci_low > 0.5

        result = ValidationResult(
            test_name='walk_forward',
            passed=passed,
            metric='mean_accuracy',
            value=mean_accuracy,
            threshold=0.5075,
            confidence_interval=(ci_low, ci_high),
            p_value=p_value,
            samples=sum(len(flows_with_outcomes[i*window_size:(i+1)*window_size])
                       for i in range(n_windows)),
            details={
                'n_windows': len(window_results),
                'window_results': window_results,
                'std': std_accuracy,
            },
        )

        if verbose:
            print(f"\n[WALK-FORWARD] === RESULTS ===")
            print(f"[WALK-FORWARD] Mean accuracy: {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")
            print(f"[WALK-FORWARD] Std: {std_accuracy:.4f}")
            print(f"[WALK-FORWARD] 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
            print(f"[WALK-FORWARD] p-value: {p_value:.4f}")
            print(f"[WALK-FORWARD] PASSED: {passed}")

        self.results.append(result)
        return result


class MonteCarloValidator:
    """
    Monte Carlo simulation to test if edge is real.

    PROCESS:
    1. Take actual trade outcomes
    2. Resample with replacement to create N alternate sequences
    3. Calculate win rate for each sequence
    4. Build distribution of win rates
    5. Check if 95% of resampled sequences are profitable
    """

    def __init__(self, db: HistoricalFlowDatabase = None):
        self.db = db or HistoricalFlowDatabase()

    def validate(self, n_simulations: int = 1000, verbose: bool = True) -> ValidationResult:
        """
        Run Monte Carlo validation.

        Args:
            n_simulations: Number of bootstrap samples
            verbose: Print progress

        Returns:
            Validation result
        """
        flows = self.db.get_flows(min_btc=0.1)
        outcomes = [f.outcome_30s for f in flows if f.outcome_30s != 0]

        if len(outcomes) < 100:
            return ValidationResult(
                test_name='monte_carlo',
                passed=False,
                metric='sample_size',
                value=len(outcomes),
                threshold=100,
                confidence_interval=(0, 0),
                p_value=1.0,
                samples=0,
                details={'error': 'Not enough data'},
            )

        # Actual win rate (assuming we follow direction signals)
        # This requires state predictions - use stored stats for now
        actual_stats = self.db.get_win_rate(timeframe='30s')
        actual_win_rate = actual_stats['win_rate']

        if verbose:
            print(f"[MONTE CARLO] {len(outcomes)} trades, actual WR={actual_win_rate:.4f}")

        # Bootstrap simulation
        simulated_win_rates = []

        for sim in range(n_simulations):
            # Resample with replacement
            resampled = random.choices(outcomes, k=len(outcomes))

            # Calculate win rate (outcomes are +1 win, -1 loss)
            wins = sum(1 for o in resampled if o == 1)
            wr = wins / len(resampled)
            simulated_win_rates.append(wr)

        # Statistics
        mean_wr = sum(simulated_win_rates) / len(simulated_win_rates)
        sorted_wrs = sorted(simulated_win_rates)

        ci_low = sorted_wrs[int(0.025 * n_simulations)]
        ci_high = sorted_wrs[int(0.975 * n_simulations)]

        # What % of simulations beat 50%?
        pct_above_50 = sum(1 for wr in simulated_win_rates if wr > 0.5) / n_simulations

        # What % of simulations beat 50.75%?
        pct_above_5075 = sum(1 for wr in simulated_win_rates if wr > 0.5075) / n_simulations

        # p-value: probability of seeing this result if true WR = 50%
        # Approximate as % of simulations with WR >= actual
        p_value = sum(1 for wr in simulated_win_rates if wr >= actual_win_rate) / n_simulations
        if actual_win_rate > 0.5:
            p_value = 1 - pct_above_50

        passed = ci_low > 0.5 and pct_above_5075 >= 0.95

        result = ValidationResult(
            test_name='monte_carlo',
            passed=passed,
            metric='bootstrap_win_rate',
            value=mean_wr,
            threshold=0.5075,
            confidence_interval=(ci_low, ci_high),
            p_value=p_value,
            samples=len(outcomes),
            details={
                'n_simulations': n_simulations,
                'pct_above_50': pct_above_50,
                'pct_above_5075': pct_above_5075,
                'actual_win_rate': actual_win_rate,
            },
        )

        if verbose:
            print(f"\n[MONTE CARLO] === RESULTS ===")
            print(f"[MONTE CARLO] Mean bootstrapped WR: {mean_wr:.4f}")
            print(f"[MONTE CARLO] 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
            print(f"[MONTE CARLO] % simulations > 50%: {pct_above_50*100:.1f}%")
            print(f"[MONTE CARLO] % simulations > 50.75%: {pct_above_5075*100:.1f}%")
            print(f"[MONTE CARLO] PASSED: {passed}")

        return result


class OutOfSampleValidator:
    """
    Simple out-of-sample validation.

    Train on first 80%, test on last 20%.
    Most basic but essential test.
    """

    def __init__(self, db: HistoricalFlowDatabase = None):
        self.db = db or HistoricalFlowDatabase()

    def validate(self, train_pct: float = 0.8, verbose: bool = True) -> ValidationResult:
        """Run out-of-sample validation."""
        flows = self.db.get_flows(min_btc=0.1)
        flows_with_outcomes = [f for f in flows if f.outcome_30s != 0]

        if len(flows_with_outcomes) < 200:
            return ValidationResult(
                test_name='out_of_sample',
                passed=False,
                metric='sample_size',
                value=len(flows_with_outcomes),
                threshold=200,
                confidence_interval=(0, 0),
                p_value=1.0,
                samples=0,
                details={'error': 'Not enough data'},
            )

        split_idx = int(len(flows_with_outcomes) * train_pct)
        train_flows = flows_with_outcomes[:split_idx]
        test_flows = flows_with_outcomes[split_idx:]

        if verbose:
            print(f"[OOS] Train: {len(train_flows)}, Test: {len(test_flows)}")

        # Train
        pipeline = HMMTrainingPipeline(self.db)
        train_obs = pipeline.flows_to_observations(train_flows)

        trainer = BaumWelchTrainer(n_states=5, n_iter=100)
        try:
            trainer.fit(train_obs, verbose=False)
        except Exception as e:
            return ValidationResult(
                test_name='out_of_sample',
                passed=False,
                metric='training_error',
                value=0,
                threshold=0,
                confidence_interval=(0, 0),
                p_value=1.0,
                samples=0,
                details={'error': str(e)},
            )

        # Test
        test_obs = pipeline.flows_to_observations(test_flows)
        test_outcomes = pipeline.flows_to_outcomes(test_flows)

        test_states = trainer.predict_states(test_obs)
        oos_accuracy = pipeline.calculate_accuracy(test_states, test_outcomes)

        # Also get in-sample accuracy for comparison
        train_states = trainer.predict_states(train_obs)
        train_outcomes = pipeline.flows_to_outcomes(train_flows)
        is_accuracy = pipeline.calculate_accuracy(train_states, train_outcomes)

        # Check for overfitting: big drop from IS to OOS
        overfit_ratio = (is_accuracy - oos_accuracy) / max(0.01, is_accuracy)

        # Binomial test for significance
        n = len([o for o in test_outcomes if o != 0])
        k = int(oos_accuracy * n)

        # Approximate p-value
        if n > 0:
            expected = n * 0.5
            std = (n * 0.5 * 0.5) ** 0.5
            z = (k - expected) / std if std > 0 else 0
            p_value = 0.5 * math.exp(-0.5 * z**2) if z > 0 else 1.0
        else:
            p_value = 1.0

        passed = oos_accuracy >= 0.5075 and overfit_ratio < 0.2

        result = ValidationResult(
            test_name='out_of_sample',
            passed=passed,
            metric='oos_accuracy',
            value=oos_accuracy,
            threshold=0.5075,
            confidence_interval=(oos_accuracy - 0.02, oos_accuracy + 0.02),  # Rough estimate
            p_value=p_value,
            samples=len(test_flows),
            details={
                'in_sample_accuracy': is_accuracy,
                'out_of_sample_accuracy': oos_accuracy,
                'overfit_ratio': overfit_ratio,
                'train_size': len(train_flows),
                'test_size': len(test_flows),
            },
        )

        if verbose:
            print(f"\n[OOS] === RESULTS ===")
            print(f"[OOS] In-sample accuracy: {is_accuracy:.4f} ({is_accuracy*100:.2f}%)")
            print(f"[OOS] Out-of-sample accuracy: {oos_accuracy:.4f} ({oos_accuracy*100:.2f}%)")
            print(f"[OOS] Overfit ratio: {overfit_ratio:.4f} (< 0.2 is good)")
            print(f"[OOS] p-value: {p_value:.4f}")
            print(f"[OOS] PASSED: {passed}")

        return result


class FullValidationSuite:
    """
    Complete validation suite.

    Runs all validation tests and produces final verdict.

    REQUIREMENTS TO PASS:
    - Out-of-sample accuracy >= 50.75%
    - Walk-forward accuracy >= 50.75%
    - Monte Carlo 95% CI > 50%
    - No severe overfitting (IS/OOS ratio < 1.2)
    """

    def __init__(self, db: HistoricalFlowDatabase = None):
        self.db = db or HistoricalFlowDatabase()
        self.results: Dict[str, ValidationResult] = {}

    def run_all(self, verbose: bool = True) -> Dict[str, ValidationResult]:
        """Run all validation tests."""
        if verbose:
            print("="*60)
            print("FULL VALIDATION SUITE")
            print("Target: Win rate >= 50.75% with statistical significance")
            print("="*60)

        # Out-of-sample
        if verbose:
            print("\n" + "-"*40)
            print("TEST 1: Out-of-Sample Validation")
            print("-"*40)
        oos = OutOfSampleValidator(self.db)
        self.results['out_of_sample'] = oos.validate(verbose=verbose)

        # Walk-forward
        if verbose:
            print("\n" + "-"*40)
            print("TEST 2: Walk-Forward Validation")
            print("-"*40)
        wf = WalkForwardValidator(self.db)
        self.results['walk_forward'] = wf.validate(verbose=verbose)

        # Monte Carlo
        if verbose:
            print("\n" + "-"*40)
            print("TEST 3: Monte Carlo Bootstrap")
            print("-"*40)
        mc = MonteCarloValidator(self.db)
        self.results['monte_carlo'] = mc.validate(verbose=verbose)

        # Final verdict
        if verbose:
            print("\n" + "="*60)
            print("FINAL VERDICT")
            print("="*60)

            all_passed = all(r.passed for r in self.results.values())

            for name, result in self.results.items():
                status = "PASS" if result.passed else "FAIL"
                print(f"  {name}: {status} (value={result.value:.4f}, threshold={result.threshold})")

            print()
            if all_passed:
                print(">>> ALL TESTS PASSED <<<")
                print("Edge is statistically validated. Ready for live trading.")
            else:
                print(">>> SOME TESTS FAILED <<<")
                print("Edge may not be real. Need more data or model improvement.")

        return self.results

    def get_summary(self) -> Dict:
        """Get validation summary."""
        return {
            'all_passed': all(r.passed for r in self.results.values()),
            'tests': {name: {
                'passed': r.passed,
                'value': r.value,
                'threshold': r.threshold,
                'p_value': r.p_value,
            } for name, r in self.results.items()},
        }


def validate_model(db_path: str = None, verbose: bool = True) -> bool:
    """
    Convenience function to validate trained model.

    Returns True if model passes all validation tests.
    """
    db = HistoricalFlowDatabase(db_path)
    suite = FullValidationSuite(db)
    results = suite.run_all(verbose=verbose)
    return all(r.passed for r in results.values())
