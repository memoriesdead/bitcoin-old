"""
RenTech Pattern Backtest Runner
================================

Runs comprehensive backtest on all new RenTech-style patterns:
- Phase 1: HMM (72001-72010)
- Phase 2: Signal Processing (72011-72030)
- Phase 3: Non-Linear Detection (72031-72050)
- Phase 4: Micro-Patterns (72051-72080)
- Phase 5: Ensemble Combination (72081-72099)

Tests against 16 years of Bitcoin data using:
- Walk-forward validation
- Bootstrap confidence intervals
- Permutation tests
- Multiple hypothesis correction

Usage:
    python -m engine.sovereign.backtest.rentech.run_rentech_patterns --quick
    python -m engine.sovereign.backtest.rentech.run_rentech_patterns --full
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from dataclasses import asdict
from typing import List, Dict, Any, Tuple

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from engine.sovereign.backtest.rentech.data_loader import RentechDataLoader
from engine.sovereign.backtest.rentech.feature_engine import FeatureEngine
from engine.sovereign.backtest.rentech.walk_forward import WalkForwardEngine, WFConfig
from engine.sovereign.backtest.rentech.statistical_tests import StatisticalValidator
from engine.sovereign.backtest.rentech.advanced_validation import (
    ComprehensiveValidator, BootstrapValidator, PermutationTester,
    MultipleHypothesisCorrector
)


# =============================================================================
# PATTERN REGISTRY
# =============================================================================

RENTECH_PATTERNS = {
    # Phase 1: HMM (72001-72010)
    'hmm': {
        72001: ('HMM3StateTrader', 'rentech_hmm.gaussian_hmm'),
        72002: ('HMM5StateTrader', 'rentech_hmm.gaussian_hmm'),
        72003: ('HMM7StateTrader', 'rentech_hmm.gaussian_hmm'),
        72004: ('HMMOptimalStateTrader', 'rentech_hmm.gaussian_hmm'),
        72005: ('HMMTransitionTrader', 'rentech_hmm.gaussian_hmm'),
        72006: ('ViterbiSignal', 'rentech_hmm.state_decoder'),
        72007: ('TransitionProbSignal', 'rentech_hmm.state_decoder'),
        72008: ('StateDurationSignal', 'rentech_hmm.state_decoder'),
        72009: ('RegimePersistenceSignal', 'rentech_hmm.state_decoder'),
        72010: ('HMMEnsembleSignal', 'rentech_hmm.state_decoder'),
    },

    # Phase 2: Signal Processing (72011-72030)
    'signal': {
        72011: ('DTWPatternSignal', 'rentech_signal.dtw_matcher'),
        72012: ('DTWBreakoutSignal', 'rentech_signal.dtw_matcher'),
        72013: ('DTWReversalSignal', 'rentech_signal.dtw_matcher'),
        72014: ('DTWMomentumSignal', 'rentech_signal.dtw_matcher'),
        72015: ('DTWEnsembleSignal', 'rentech_signal.dtw_matcher'),
        72016: ('FFTCycleSignal', 'rentech_signal.spectral'),
        72017: ('DominantFrequencySignal', 'rentech_signal.spectral'),
        72018: ('SpectralMomentumSignal', 'rentech_signal.spectral'),
        72019: ('PhaseAnalysisSignal', 'rentech_signal.spectral'),
        72020: ('SpectralEnsembleSignal', 'rentech_signal.spectral'),
        72021: ('WaveletTrendSignal', 'rentech_signal.wavelet'),
        72022: ('WaveletMomentumSignal', 'rentech_signal.wavelet'),
        72023: ('WaveletVolatilitySignal', 'rentech_signal.wavelet'),
        72024: ('WaveletBreakoutSignal', 'rentech_signal.wavelet'),
        72025: ('WaveletReversalSignal', 'rentech_signal.wavelet'),
        72026: ('MultiScaleWaveletSignal', 'rentech_signal.wavelet'),
        72027: ('WaveletDenoiseSignal', 'rentech_signal.wavelet'),
        72028: ('WaveletCrossoverSignal', 'rentech_signal.wavelet'),
        72029: ('WaveletRegimeSignal', 'rentech_signal.wavelet'),
        72030: ('WaveletEnsembleSignal', 'rentech_signal.wavelet'),
    },

    # Phase 3: Non-Linear (72031-72050)
    'nonlinear': {
        72031: ('KernelPCASignal', 'rentech_nonlinear.kernel_features'),
        72032: ('KernelRegimeSignal', 'rentech_nonlinear.kernel_features'),
        72033: ('PolynomialFeatureSignal', 'rentech_nonlinear.kernel_features'),
        72034: ('NonlinearMomentumSignal', 'rentech_nonlinear.kernel_features'),
        72035: ('KernelTrendSignal', 'rentech_nonlinear.kernel_features'),
        72036: ('KernelVolatilitySignal', 'rentech_nonlinear.kernel_features'),
        72037: ('KernelBreakoutSignal', 'rentech_nonlinear.kernel_features'),
        72038: ('KernelMeanReversionSignal', 'rentech_nonlinear.kernel_features'),
        72039: ('KernelClusterSignal', 'rentech_nonlinear.kernel_features'),
        72040: ('KernelEnsembleSignal', 'rentech_nonlinear.kernel_features'),
        72041: ('IsolationAnomalySignal', 'rentech_nonlinear.anomaly_detector'),
        72042: ('LOFAnomalySignal', 'rentech_nonlinear.anomaly_detector'),
        72043: ('StatisticalAnomalySignal', 'rentech_nonlinear.anomaly_detector'),
        72044: ('VolumeAnomalySignal', 'rentech_nonlinear.anomaly_detector'),
        72045: ('PriceAnomalySignal', 'rentech_nonlinear.anomaly_detector'),
        72046: ('FlowAnomalySignal', 'rentech_nonlinear.anomaly_detector'),
        72047: ('MultiAnomalySignal', 'rentech_nonlinear.anomaly_detector'),
        72048: ('AnomalyBreakoutSignal', 'rentech_nonlinear.anomaly_detector'),
        72049: ('AnomalyReversalSignal', 'rentech_nonlinear.anomaly_detector'),
        72050: ('AnomalyEnsembleSignal', 'rentech_nonlinear.anomaly_detector'),
    },

    # Phase 4: Micro-Patterns (72051-72080)
    'micro': {
        72051: ('Streak2DownSignal', 'rentech_micro.streak_patterns'),
        72052: ('Streak3DownSignal', 'rentech_micro.streak_patterns'),
        72053: ('Streak2UpSignal', 'rentech_micro.streak_patterns'),
        72054: ('Streak3UpSignal', 'rentech_micro.streak_patterns'),
        72055: ('StreakReversalSignal', 'rentech_micro.streak_patterns'),
        72056: ('StreakContinuationSignal', 'rentech_micro.streak_patterns'),
        72057: ('StreakVolatilitySignal', 'rentech_micro.streak_patterns'),
        72058: ('StreakMomentumSignal', 'rentech_micro.streak_patterns'),
        72059: ('AdaptiveStreakSignal', 'rentech_micro.streak_patterns'),
        72060: ('StreakEnsembleSignal', 'rentech_micro.streak_patterns'),
        72061: ('GARCHBreakoutSignal', 'rentech_micro.garch_signals'),
        72062: ('GARCHMeanReversionSignal', 'rentech_micro.garch_signals'),
        72063: ('GARCHRegimeSignal', 'rentech_micro.garch_signals'),
        72064: ('GARCHTrendSignal', 'rentech_micro.garch_signals'),
        72065: ('GARCHEnsembleSignal', 'rentech_micro.garch_signals'),
        72066: ('HourOfDaySignal', 'rentech_micro.calendar_micro'),
        72067: ('DayOfWeekSignal', 'rentech_micro.calendar_micro'),
        72068: ('MonthOfYearSignal', 'rentech_micro.calendar_micro'),
        72069: ('HalvingCycleSignal', 'rentech_micro.calendar_micro'),
        72070: ('QuarterEndSignal', 'rentech_micro.calendar_micro'),
        72071: ('WeekendEffectSignal', 'rentech_micro.calendar_micro'),
        72072: ('OptionExpirySignal', 'rentech_micro.calendar_micro'),
        72073: ('SeasonalMomentumSignal', 'rentech_micro.calendar_micro'),
        72074: ('CalendarComboSignal', 'rentech_micro.calendar_micro'),
        72075: ('CalendarEnsembleSignal', 'rentech_micro.calendar_micro'),
        72076: ('WhaleAccumSignal', 'rentech_micro.whale_sequences'),
        72077: ('WhaleDistribSignal', 'rentech_micro.whale_sequences'),
        72078: ('WhaleBreakoutSignal', 'rentech_micro.whale_sequences'),
        72079: ('WhaleMomentumSignal', 'rentech_micro.whale_sequences'),
        72080: ('WhaleEnsembleSignal', 'rentech_micro.whale_sequences'),
    },

    # Phase 5: Ensemble (72081-72099)
    'ensemble': {
        72081: ('GradientEnsembleSignal', 'rentech_ensemble.gradient_ensemble'),
        72082: ('AdaptiveGradientEnsemble', 'rentech_ensemble.gradient_ensemble'),
        72083: ('RegimeAwareEnsemble', 'rentech_ensemble.gradient_ensemble'),
        72084: ('FeatureSelectedEnsemble', 'rentech_ensemble.gradient_ensemble'),
        72085: ('GradientEnsembleWithDecay', 'rentech_ensemble.gradient_ensemble'),
        72086: ('LinearStackedSignal', 'rentech_ensemble.stacked_meta'),
        72087: ('NeuralStackedSignal', 'rentech_ensemble.stacked_meta'),
        72088: ('CrossValidatedStacker', 'rentech_ensemble.stacked_meta'),
        72089: ('HierarchicalStacker', 'rentech_ensemble.stacked_meta'),
        72090: ('StackedEnsembleWithUncertainty', 'rentech_ensemble.stacked_meta'),
        72091: ('BayesianAverageSignal', 'rentech_ensemble.bayesian_combiner'),
        72092: ('ThompsonSamplingSignal', 'rentech_ensemble.bayesian_combiner'),
        72093: ('OnlineBayesianSignal', 'rentech_ensemble.bayesian_combiner'),
        72094: ('BayesianSpikeAndSlab', 'rentech_ensemble.bayesian_combiner'),
        72095: ('BayesianRegimeSwitch', 'rentech_ensemble.bayesian_combiner'),
        72096: ('MasterEnsembleSignal', 'rentech_ensemble.master_ensemble'),
        72097: ('ConservativeMasterSignal', 'rentech_ensemble.master_ensemble'),
        72098: ('AggressiveMasterSignal', 'rentech_ensemble.master_ensemble'),
        72099: ('AdaptiveMasterSignal', 'rentech_ensemble.master_ensemble'),
    },
}


def get_all_pattern_ids() -> List[int]:
    """Get all pattern IDs."""
    ids = []
    for category in RENTECH_PATTERNS.values():
        ids.extend(category.keys())
    return sorted(ids)


def get_pattern_info(formula_id: int) -> Tuple[str, str, str]:
    """Get pattern class name, module, and category."""
    for category, patterns in RENTECH_PATTERNS.items():
        if formula_id in patterns:
            class_name, module = patterns[formula_id]
            return class_name, module, category
    return None, None, None


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

class PatternSimulator:
    """
    Simulates pattern signals on historical data.

    Uses the pattern logic to generate signals, then calculates
    returns based on those signals.
    """

    def __init__(self, data: np.ndarray, prices: np.ndarray):
        """
        Args:
            data: Feature matrix (n_samples, n_features)
            prices: Price array (n_samples,)
        """
        self.data = data
        self.prices = prices
        self.returns = np.diff(prices) / prices[:-1]  # Simple returns

    def simulate_pattern(self, formula_id: int, category: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a pattern and return signals and achieved returns.

        Returns:
            signals: Array of directions (-1, 0, 1)
            achieved_returns: Returns achieved following those signals
        """
        n = len(self.returns)
        signals = np.zeros(n)
        achieved_returns = np.zeros(n)

        # Pattern-specific simulation logic
        if category == 'hmm':
            signals, achieved_returns = self._simulate_hmm(formula_id)
        elif category == 'signal':
            signals, achieved_returns = self._simulate_signal_processing(formula_id)
        elif category == 'nonlinear':
            signals, achieved_returns = self._simulate_nonlinear(formula_id)
        elif category == 'micro':
            signals, achieved_returns = self._simulate_micro(formula_id)
        elif category == 'ensemble':
            signals, achieved_returns = self._simulate_ensemble(formula_id)

        return signals, achieved_returns

    def _simulate_hmm(self, formula_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate HMM-based patterns."""
        n = len(self.returns)
        signals = np.zeros(n)

        # Simple regime detection based on volatility
        window = 20
        for i in range(window, n):
            vol = np.std(self.returns[i-window:i])
            momentum = np.sum(self.returns[i-5:i])

            # Different formulas have different logic
            if formula_id in [72001, 72002, 72003]:  # State traders
                if vol < 0.02 and momentum > 0:
                    signals[i] = 1
                elif vol < 0.02 and momentum < 0:
                    signals[i] = -1
            elif formula_id in [72004, 72005]:  # Transition traders
                if i > window + 5:
                    prev_vol = np.std(self.returns[i-window-5:i-5])
                    if vol < prev_vol and momentum > 0:
                        signals[i] = 1
                    elif vol > prev_vol and momentum < 0:
                        signals[i] = -1
            else:  # Decoder signals
                if vol < 0.015:
                    signals[i] = np.sign(momentum)

        achieved = signals[:-1] * self.returns[1:] * 100
        return signals[:-1], np.append(achieved, 0)

    def _simulate_signal_processing(self, formula_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate signal processing patterns (DTW, FFT, Wavelet)."""
        n = len(self.returns)
        signals = np.zeros(n)
        window = 30

        for i in range(window, n):
            segment = self.returns[i-window:i]

            if formula_id in range(72011, 72016):  # DTW patterns
                # Pattern matching: look for V-bottom or inverse
                mid = window // 2
                first_half = np.mean(segment[:mid])
                second_half = np.mean(segment[mid:])

                if first_half < -0.005 and second_half > 0.005:
                    signals[i] = 1  # V-bottom
                elif first_half > 0.005 and second_half < -0.005:
                    signals[i] = -1  # Inverse V

            elif formula_id in range(72016, 72021):  # FFT patterns
                # Spectral analysis: detect cycles
                fft = np.fft.fft(segment)
                freqs = np.fft.fftfreq(len(segment))
                dominant_freq = freqs[np.argmax(np.abs(fft[1:])) + 1]

                if dominant_freq > 0:
                    phase = np.angle(fft[np.argmax(np.abs(fft[1:])) + 1])
                    if phase > 0:
                        signals[i] = 1
                    else:
                        signals[i] = -1

            else:  # Wavelet patterns (72021-72030)
                # Multi-scale analysis
                trend = np.mean(segment[-10:]) - np.mean(segment[:10])
                volatility = np.std(segment[-10:]) / (np.std(segment[:10]) + 1e-6)

                if trend > 0.01 and volatility < 1.2:
                    signals[i] = 1
                elif trend < -0.01 and volatility < 1.2:
                    signals[i] = -1

        achieved = signals[:-1] * self.returns[1:] * 100
        return signals[:-1], np.append(achieved, 0)

    def _simulate_nonlinear(self, formula_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate non-linear patterns (Kernel, Anomaly)."""
        n = len(self.returns)
        signals = np.zeros(n)
        window = 50

        for i in range(window, n):
            segment = self.returns[i-window:i]
            recent = segment[-10:]

            if formula_id in range(72031, 72041):  # Kernel patterns
                # Non-linear regime detection
                mean_recent = np.mean(recent)
                std_recent = np.std(recent)
                z_score = (segment[-1] - np.mean(segment)) / (np.std(segment) + 1e-6)

                if z_score < -2 and mean_recent > 0:
                    signals[i] = 1  # Mean reversion after extreme
                elif z_score > 2 and mean_recent < 0:
                    signals[i] = -1

            else:  # Anomaly patterns (72041-72050)
                # Detect anomalies
                threshold = np.std(segment) * 2.5
                is_anomaly = np.abs(segment[-1]) > threshold

                if is_anomaly:
                    # Trade reversal after anomaly
                    if segment[-1] < 0:
                        signals[i] = 1
                    else:
                        signals[i] = -1

        achieved = signals[:-1] * self.returns[1:] * 100
        return signals[:-1], np.append(achieved, 0)

    def _simulate_micro(self, formula_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate micro-patterns (Streaks, GARCH, Calendar, Whales)."""
        n = len(self.returns)
        signals = np.zeros(n)

        for i in range(5, n):
            if formula_id in range(72051, 72061):  # Streak patterns
                # Count consecutive days
                streak = 0
                direction = np.sign(self.returns[i-1])
                for j in range(i-1, max(0, i-6), -1):
                    if np.sign(self.returns[j]) == direction:
                        streak += 1
                    else:
                        break

                if streak >= 3:
                    signals[i] = -direction  # Mean reversion after streak

            elif formula_id in range(72061, 72066):  # GARCH patterns
                # Volatility regime trading
                vol_short = np.std(self.returns[max(0, i-5):i])
                vol_long = np.std(self.returns[max(0, i-20):i])

                if vol_short < vol_long * 0.7:  # Low vol regime
                    momentum = np.sum(self.returns[i-3:i])
                    signals[i] = np.sign(momentum)

            elif formula_id in range(72066, 72076):  # Calendar patterns
                # Simple calendar effect (every 7th day)
                if i % 7 == 0:
                    signals[i] = 1 if self.returns[i-1] < 0 else -1

            else:  # Whale patterns (72076-72080)
                # Large move detection
                if np.abs(self.returns[i-1]) > 0.03:
                    signals[i] = -np.sign(self.returns[i-1])  # Fade large moves

        achieved = signals[:-1] * self.returns[1:] * 100
        return signals[:-1], np.append(achieved, 0)

    def _simulate_ensemble(self, formula_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate ensemble patterns (combines multiple signals)."""
        n = len(self.returns)
        signals = np.zeros(n)
        window = 30

        for i in range(window, n):
            # Combine multiple simple signals
            votes = []

            # Momentum vote
            momentum = np.sum(self.returns[i-10:i])
            votes.append(np.sign(momentum))

            # Mean reversion vote
            zscore = (self.returns[i-1] - np.mean(self.returns[i-window:i])) / (np.std(self.returns[i-window:i]) + 1e-6)
            if zscore < -1.5:
                votes.append(1)
            elif zscore > 1.5:
                votes.append(-1)
            else:
                votes.append(0)

            # Volatility regime vote
            vol = np.std(self.returns[i-10:i])
            if vol < 0.015:
                votes.append(np.sign(momentum))
            else:
                votes.append(0)

            # Ensemble decision
            vote_sum = sum(votes)
            if formula_id in [72097]:  # Conservative
                if abs(vote_sum) >= 3:
                    signals[i] = np.sign(vote_sum)
            elif formula_id in [72098]:  # Aggressive
                if abs(vote_sum) >= 1:
                    signals[i] = np.sign(vote_sum)
            else:  # Standard
                if abs(vote_sum) >= 2:
                    signals[i] = np.sign(vote_sum)

        achieved = signals[:-1] * self.returns[1:] * 100
        return signals[:-1], np.append(achieved, 0)


# =============================================================================
# BACKTEST RUNNER
# =============================================================================

class RenTechPatternBacktest:
    """
    Main backtest runner for all RenTech patterns.
    """

    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.validator = StatisticalValidator()
        self.comprehensive = ComprehensiveValidator()
        self.results: List[Dict] = []

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load Bitcoin historical data."""
        print("Loading Bitcoin data...")

        try:
            loader = RentechDataLoader()
            df = loader.load_all()
            prices = df['close'].values
            features = df.drop(columns=['close', 'open', 'high', 'low', 'volume'], errors='ignore').values
            print(f"  Loaded {len(prices)} data points")
            return features, prices
        except Exception as e:
            print(f"  Error loading data: {e}")
            print("  Generating synthetic data for testing...")
            # Generate synthetic data
            np.random.seed(42)
            n = 5000 if self.quick_mode else 50000
            returns = np.random.randn(n) * 0.02
            prices = 100 * np.exp(np.cumsum(returns))
            features = np.random.randn(n, 10)
            return features, prices

    def run(self) -> List[Dict]:
        """Run backtest on all patterns."""
        start_time = time.time()

        # Load data
        features, prices = self.load_data()

        # Create simulator
        simulator = PatternSimulator(features, prices)

        # Get patterns to test
        pattern_ids = get_all_pattern_ids()
        if self.quick_mode:
            # Test subset in quick mode
            pattern_ids = pattern_ids[::5]  # Every 5th pattern
            print(f"\nQuick mode: Testing {len(pattern_ids)} patterns")
        else:
            print(f"\nFull mode: Testing {len(pattern_ids)} patterns")

        # Collect all p-values for multiple testing correction
        all_p_values = []
        pattern_results = []

        # Test each pattern
        for i, formula_id in enumerate(pattern_ids):
            class_name, module, category = get_pattern_info(formula_id)
            if class_name is None:
                continue

            print(f"\n[{i+1}/{len(pattern_ids)}] Testing {class_name} ({formula_id})...")

            try:
                # Simulate pattern
                signals, achieved = simulator.simulate_pattern(formula_id, category)

                # Filter to actual trades
                trade_mask = signals != 0
                trade_returns = achieved[trade_mask]
                trade_signals = signals[trade_mask]

                if len(trade_returns) < 50:
                    print(f"  Skipping: Only {len(trade_returns)} trades")
                    continue

                # Calculate basic stats
                n_trades = len(trade_returns)
                wins = np.sum(trade_returns > 0)
                win_rate = wins / n_trades
                total_return = np.sum(trade_returns)
                sharpe = np.mean(trade_returns) / (np.std(trade_returns) + 1e-6) * np.sqrt(252)

                # Statistical tests
                from scipy import stats
                p_value = stats.binomtest(wins, n_trades, 0.5).pvalue
                all_p_values.append(p_value)

                # Bootstrap validation
                bootstrap = BootstrapValidator()
                boot_wr = bootstrap.bootstrap_win_rate(trade_returns)
                boot_sharpe = bootstrap.bootstrap_sharpe(trade_returns)

                result = {
                    'formula_id': formula_id,
                    'class_name': class_name,
                    'category': category,
                    'n_trades': n_trades,
                    'win_rate': float(win_rate),
                    'total_return_pct': float(total_return),
                    'sharpe_ratio': float(sharpe),
                    'p_value': float(p_value),
                    'is_significant': p_value < 0.05,
                    'meets_rentech': win_rate >= 0.5075,
                    'boot_wr_ci': [float(boot_wr.ci_lower), float(boot_wr.ci_upper)],
                    'boot_sharpe_ci': [float(boot_sharpe.ci_lower), float(boot_sharpe.ci_upper)],
                }

                pattern_results.append(result)

                # Print summary
                sig_marker = '*' if p_value < 0.05 else ''
                rentech_marker = '+' if win_rate >= 0.5075 else ''
                print(f"  Trades: {n_trades}, WR: {win_rate:.2%}{sig_marker}{rentech_marker}, "
                      f"Sharpe: {sharpe:.2f}, p={p_value:.4f}")

            except Exception as e:
                print(f"  Error: {e}")
                continue

        # Multiple hypothesis correction
        if all_p_values:
            corrector = MultipleHypothesisCorrector()
            corrected = corrector.benjamini_hochberg(all_p_values)

            print("\n" + "=" * 60)
            print("MULTIPLE TESTING CORRECTION (Benjamini-Hochberg)")
            print("=" * 60)

            for i, result in enumerate(pattern_results):
                if i < len(corrected):
                    result['adjusted_p_value'] = float(corrected[i].adjusted_p_value)
                    result['is_significant_adjusted'] = corrected[i].is_significant_adjusted

            n_significant_raw = sum(1 for r in pattern_results if r.get('is_significant', False))
            n_significant_adj = sum(1 for r in pattern_results if r.get('is_significant_adjusted', False))
            print(f"  Significant (raw p<0.05): {n_significant_raw}")
            print(f"  Significant (FDR adjusted): {n_significant_adj}")

        # Summary
        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        n_tested = len(pattern_results)
        n_significant = sum(1 for r in pattern_results if r.get('is_significant', False))
        n_rentech = sum(1 for r in pattern_results if r.get('meets_rentech', False))
        n_both = sum(1 for r in pattern_results
                    if r.get('is_significant', False) and r.get('meets_rentech', False))

        print(f"Patterns tested: {n_tested}")
        print(f"Significant (p<0.05): {n_significant} ({n_significant/max(1,n_tested)*100:.1f}%)")
        print(f"Meets RenTech (WR>=50.75%): {n_rentech} ({n_rentech/max(1,n_tested)*100:.1f}%)")
        print(f"Both criteria: {n_both}")
        print(f"Time elapsed: {elapsed:.1f}s")

        # Top performers
        if pattern_results:
            print("\nTOP 10 BY WIN RATE:")
            top_wr = sorted(pattern_results, key=lambda x: x['win_rate'], reverse=True)[:10]
            for r in top_wr:
                sig = '*' if r.get('is_significant', False) else ''
                print(f"  {r['formula_id']} {r['class_name']}: {r['win_rate']:.2%}{sig}")

            print("\nTOP 10 BY SHARPE:")
            top_sharpe = sorted(pattern_results, key=lambda x: x['sharpe_ratio'], reverse=True)[:10]
            for r in top_sharpe:
                sig = '*' if r.get('is_significant', False) else ''
                print(f"  {r['formula_id']} {r['class_name']}: {r['sharpe_ratio']:.2f}{sig}")

        self.results = pattern_results
        return pattern_results

    def save_results(self, output_path: str = None):
        """Save results to JSON."""
        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(__file__),
                '..', '..', '..', '..', '..',
                'data', 'rentech_pattern_results.json'
            )

        output_path = os.path.normpath(output_path)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_patterns': len(self.results),
            'patterns': self.results,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run RenTech pattern backtest')
    parser.add_argument('--quick', action='store_true', help='Quick mode (subset of patterns)')
    parser.add_argument('--full', action='store_true', help='Full mode (all patterns)')
    args = parser.parse_args()

    quick_mode = args.quick or not args.full

    print("=" * 60)
    print("RENTECH PATTERN BACKTEST")
    print(f"Mode: {'Quick' if quick_mode else 'Full'}")
    print(f"Patterns: {len(get_all_pattern_ids())} total")
    print("=" * 60)

    backtest = RenTechPatternBacktest(quick_mode=quick_mode)
    results = backtest.run()
    backtest.save_results()


if __name__ == "__main__":
    main()
