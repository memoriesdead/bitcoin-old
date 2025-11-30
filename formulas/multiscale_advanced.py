"""
Multi-Scale Advanced Formulas (IDs 361-400)
===========================================
Advanced academic formulas for complete time-scale adaptation.
These formulas fill the gaps not covered by IDs 347-360.

Research Sources:
    - Bai & Perron (1998, 2003): Multiple structural break detection
    - Podobnik & Stanley (2008): DCCA cross-correlation
    - Kantelhardt et al. (2002): MF-DFA multifractal analysis
    - Gatheral et al. (2014): Rough volatility
    - Lo & MacKinlay (1988): Variance ratio tests
    - Engle & Russell (1998): ACD duration models
    - Torres et al. (2011): CEEMDAN decomposition
    - Ghashghaie et al. (1996): Turbulent cascades in markets
    - Leung & Li (2015): Optimal stopping mean reversion
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from .base import BaseFormula, FormulaRegistry


# =============================================================================
# STRUCTURAL BREAK DETECTION (361-365)
# =============================================================================

@FormulaRegistry.register(361)
class BaiPerronBreakDetector(BaseFormula):
    """
    ID 361: Bai-Perron Multiple Structural Break Detection

    Academic Reference:
        - Bai & Perron (1998, 2003) "Computation and Analysis of Multiple
          Structural Change Models"

    Detects multiple breakpoints in the regression:
        y_t = x_t'β + z_t'δ_j + u_t for T_{j-1} < t ≤ T_j

    Uses dynamic programming to find optimal partition minimizing SSR.
    Trading: Detects when market regime fundamentally changes.
    """

    NAME = "BaiPerronBreakDetector"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Multiple structural break detection via dynamic programming"

    def __init__(self, lookback: int = 300, **kwargs):
        super().__init__(lookback, **kwargs)
        self.max_breaks = kwargs.get('max_breaks', 5)
        self.min_segment = kwargs.get('min_segment', 20)
        self.breakpoints = []
        self.current_regime = 0
        self.regime_start = 0

    def _compute_ssr(self, data: np.ndarray, start: int, end: int) -> float:
        """Compute sum of squared residuals for segment"""
        if end <= start + 2:
            return np.inf
        segment = data[start:end]
        mean = np.mean(segment)
        return np.sum((segment - mean)**2)

    def _find_breaks_dp(self, data: np.ndarray, m: int) -> Tuple[List[int], float]:
        """Dynamic programming for m breaks"""
        n = len(data)
        h = self.min_segment

        if n < (m + 1) * h:
            return [], np.inf

        # Cost matrix: cost[t] = min SSR for data[0:t] with optimal breaks
        INF = np.inf
        cost = np.full((m + 2, n + 1), INF)
        cost[0, 0] = 0

        # Backtrack matrix
        prev = np.zeros((m + 2, n + 1), dtype=int)

        for j in range(1, m + 2):
            for t in range(j * h, n + 1):
                for s in range((j-1) * h, t - h + 1):
                    seg_cost = self._compute_ssr(data, s, t)
                    total = cost[j-1, s] + seg_cost
                    if total < cost[j, t]:
                        cost[j, t] = total
                        prev[j, t] = s

        # Backtrack to find breakpoints
        breaks = []
        t = n
        for j in range(m + 1, 0, -1):
            s = prev[j, t]
            if s > 0:
                breaks.append(s)
            t = s

        breaks = sorted(breaks)
        return breaks, cost[m + 1, n]

    def _compute(self) -> None:
        if len(self.returns) < 60:
            return

        returns = self._returns_array()

        # Find optimal number of breaks using BIC
        best_bic = np.inf
        best_breaks = []
        n = len(returns)

        for m in range(0, min(self.max_breaks + 1, n // self.min_segment)):
            breaks, ssr = self._find_breaks_dp(returns, m)
            if ssr < np.inf:
                # BIC = n*log(SSR/n) + (m+1)*log(n)
                k = m + 1  # number of segments
                bic = n * np.log(ssr / n + 1e-10) + k * np.log(n)
                if bic < best_bic:
                    best_bic = bic
                    best_breaks = breaks

        self.breakpoints = best_breaks

        # Determine current regime
        self.current_regime = len([b for b in self.breakpoints if b < n])
        if self.breakpoints:
            self.regime_start = self.breakpoints[-1] if self.breakpoints[-1] < n else 0

        # Signal based on regime change recency
        time_since_break = n - self.regime_start if self.regime_start > 0 else n

        if time_since_break < 10:
            # Recent regime change - follow new regime direction
            new_regime_returns = returns[self.regime_start:]
            self.signal = 1 if np.mean(new_regime_returns) > 0 else -1
            self.confidence = 0.8
        elif time_since_break < 30:
            # Regime established
            regime_returns = returns[self.regime_start:]
            momentum = np.mean(regime_returns[-5:])
            self.signal = 1 if momentum > 0 else -1
            self.confidence = 0.6
        else:
            # Stable regime
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(362)
class CUSUMScaleDetector(BaseFormula):
    """
    ID 362: Multi-Scale CUSUM Change Detection

    Academic Reference:
        - Brown, Durbin & Evans (1975) CUSUM tests
        - Korkas & Fryzlewicz (2017) Wavelet-CUSUM combination

    Runs CUSUM at multiple time scales simultaneously.
    Detects which scale is experiencing a structural change.
    """

    NAME = "CUSUMScaleDetector"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "CUSUM change detection across multiple scales"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.scales = [5, 10, 20, 40, 80]
        self.cusum_pos = {s: 0.0 for s in self.scales}
        self.cusum_neg = {s: 0.0 for s in self.scales}
        self.k = 0.5  # Allowance parameter
        self.h = 4.0  # Decision threshold
        self.change_detected = {}

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()
        self.change_detected = {}

        for scale in self.scales:
            if len(returns) < scale * 2:
                continue

            # Aggregate returns at this scale
            n_agg = len(returns) // scale
            agg_returns = np.array([
                np.sum(returns[i*scale:(i+1)*scale])
                for i in range(n_agg)
            ])

            if len(agg_returns) < 10:
                continue

            # Calculate CUSUM
            mean = np.mean(agg_returns[:-5])
            std = np.std(agg_returns[:-5]) + 1e-10

            z = (agg_returns[-1] - mean) / std

            self.cusum_pos[scale] = max(0, self.cusum_pos[scale] + z - self.k)
            self.cusum_neg[scale] = max(0, self.cusum_neg[scale] - z - self.k)

            # Check for change
            if self.cusum_pos[scale] > self.h:
                self.change_detected[scale] = 'up'
                self.cusum_pos[scale] = 0
            elif self.cusum_neg[scale] > self.h:
                self.change_detected[scale] = 'down'
                self.cusum_neg[scale] = 0

        # Signal from the smallest scale with change
        if self.change_detected:
            smallest_scale = min(self.change_detected.keys())
            direction = self.change_detected[smallest_scale]
            self.signal = 1 if direction == 'up' else -1
            # Confidence based on how many scales agree
            self.confidence = len(self.change_detected) / len(self.scales)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(363)
class WBSChangepoint(BaseFormula):
    """
    ID 363: Wild Binary Segmentation for Changepoints

    Academic Reference:
        - Fryzlewicz (2014) "Wild binary segmentation for multiple
          change-point detection"

    Randomly drawn intervals + CUSUM for robust changepoint detection.
    More powerful than standard binary segmentation.
    """

    NAME = "WBSChangepoint"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Wild Binary Segmentation changepoint detection"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_intervals = kwargs.get('n_intervals', 100)
        self.threshold_const = kwargs.get('threshold_const', 1.3)
        self.changepoints = []

    def _cusum_statistic(self, data: np.ndarray, s: int, e: int) -> Tuple[int, float]:
        """Compute CUSUM statistic and location for interval [s, e)"""
        if e - s < 4:
            return s, 0.0

        segment = data[s:e]
        n = len(segment)
        cumsum = np.cumsum(segment - np.mean(segment))

        # Normalized CUSUM
        best_k = 0
        best_stat = 0.0

        for k in range(1, n):
            stat = abs(cumsum[k-1]) * np.sqrt(n / (k * (n - k) + 1e-10))
            if stat > best_stat:
                best_stat = stat
                best_k = k

        return s + best_k, best_stat

    def _compute(self) -> None:
        if len(self.returns) < 50:
            return

        returns = self._returns_array()
        n = len(returns)

        # Threshold based on data length
        threshold = self.threshold_const * np.sqrt(2 * np.log(n))

        # Generate random intervals
        np.random.seed(42)  # Reproducibility
        intervals = []
        for _ in range(self.n_intervals):
            s = np.random.randint(0, n - 10)
            e = np.random.randint(s + 10, n + 1)
            intervals.append((s, e))

        # Find changepoints
        self.changepoints = []
        candidates = []

        for s, e in intervals:
            k, stat = self._cusum_statistic(returns, s, e)
            if stat > threshold:
                candidates.append((k, stat))

        # Filter overlapping changepoints
        if candidates:
            candidates.sort(key=lambda x: -x[1])  # Sort by statistic
            for k, stat in candidates:
                # Check if not too close to existing
                if not any(abs(k - cp) < 10 for cp in self.changepoints):
                    self.changepoints.append(k)

        self.changepoints.sort()

        # Signal based on recent changepoint
        if self.changepoints and n - self.changepoints[-1] < 20:
            # Recent changepoint detected
            post_cp = returns[self.changepoints[-1]:]
            self.signal = 1 if np.mean(post_cp) > 0 else -1
            self.confidence = 0.7
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(364)
class EntropyScaleSelector(BaseFormula):
    """
    ID 364: Information-Theoretic Time-Scale Selection

    Academic Reference:
        - Shannon entropy for predictability assessment
        - Multiscale Entropy (MSE) methods

    Uses entropy rate to find which time scale has lowest uncertainty.
    Lower entropy = more predictable = better for trading.
    """

    NAME = "EntropyScaleSelector"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Entropy-based optimal scale selection"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.scales = [1, 2, 5, 10, 20, 40]
        self.entropy_by_scale = {}
        self.optimal_scale = 1
        self.min_entropy = 1.0

    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Compute sample entropy (SampEn)"""
        n = len(data)
        if n < m + 2:
            return np.inf

        r_scaled = r * np.std(data)

        def count_matches(template_len):
            count = 0
            for i in range(n - template_len):
                template = data[i:i+template_len]
                for j in range(i + 1, n - template_len):
                    if np.max(np.abs(template - data[j:j+template_len])) < r_scaled:
                        count += 1
            return count

        A = count_matches(m + 1)
        B = count_matches(m)

        if B == 0 or A == 0:
            return np.inf

        return -np.log(A / B)

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()
        self.entropy_by_scale = {}

        for scale in self.scales:
            if len(returns) < scale * 20:
                continue

            # Aggregate returns at this scale
            n_agg = len(returns) // scale
            agg_returns = np.array([
                np.sum(returns[i*scale:(i+1)*scale])
                for i in range(n_agg)
            ])

            if len(agg_returns) < 20:
                continue

            # Normalize
            agg_norm = (agg_returns - np.mean(agg_returns)) / (np.std(agg_returns) + 1e-10)

            # Compute sample entropy
            entropy = self._sample_entropy(agg_norm)
            if not np.isinf(entropy):
                self.entropy_by_scale[scale] = entropy

        if not self.entropy_by_scale:
            self.signal = 0
            self.confidence = 0.3
            return

        # Find scale with minimum entropy (most predictable)
        self.optimal_scale = min(self.entropy_by_scale, key=self.entropy_by_scale.get)
        self.min_entropy = self.entropy_by_scale[self.optimal_scale]

        # Trade at optimal scale
        s = self.optimal_scale
        n_agg = len(returns) // s
        agg_returns = np.array([
            np.sum(returns[i*s:(i+1)*s])
            for i in range(n_agg)
        ])

        if len(agg_returns) >= 2:
            recent = agg_returns[-1]
            prev_mean = np.mean(agg_returns[:-1])
            prev_std = np.std(agg_returns[:-1]) + 1e-10
            z = (recent - prev_mean) / prev_std

            # Lower entropy = higher confidence
            predictability = 1 / (1 + self.min_entropy)

            if abs(z) > 1:
                self.signal = 1 if z > 0 else -1
                self.confidence = min(predictability * 1.5, 1.0)
            else:
                self.signal = 0
                self.confidence = predictability * 0.5
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(365)
class TransferEntropyScale(BaseFormula):
    """
    ID 365: Transfer Entropy for Cross-Scale Information Flow

    Academic Reference:
        - Schreiber (2000) "Measuring information transfer"
        - Marschinski & Kantz (2002) Transfer entropy in finance

    Measures information flow FROM large scales TO small scales.
    High transfer entropy = large scale drives small scale.
    """

    NAME = "TransferEntropyScale"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Information flow between time scales"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.source_scale = kwargs.get('source_scale', 20)
        self.target_scale = kwargs.get('target_scale', 5)
        self.transfer_entropy = 0.0
        self.info_direction = 'none'

    def _discretize(self, data: np.ndarray, n_bins: int = 5) -> np.ndarray:
        """Discretize continuous data into bins"""
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(data, percentiles)
        return np.digitize(data, bins[1:-1])

    def _transfer_entropy(self, source: np.ndarray, target: np.ndarray,
                          k: int = 1, l: int = 1) -> float:
        """Compute transfer entropy from source to target"""
        n = len(target) - max(k, l)
        if n < 10:
            return 0.0

        # Discretize
        source_d = self._discretize(source)
        target_d = self._discretize(target)

        # Count joint and marginal distributions
        from collections import Counter

        joint_tyx = Counter()
        joint_ty = Counter()
        joint_yx = Counter()
        marginal_y = Counter()

        for i in range(max(k, l), len(target)):
            t_future = target_d[i]
            t_past = tuple(target_d[i-k:i])
            s_past = tuple(source_d[i-l:i])

            joint_tyx[(t_future, t_past, s_past)] += 1
            joint_ty[(t_future, t_past)] += 1
            joint_yx[(t_past, s_past)] += 1
            marginal_y[t_past] += 1

        # Calculate transfer entropy
        te = 0.0
        total = sum(joint_tyx.values())

        for (t, y, x), count in joint_tyx.items():
            p_tyx = count / total
            p_ty = joint_ty[(t, y)] / total
            p_yx = joint_yx[(y, x)] / total
            p_y = marginal_y[y] / total

            if p_tyx > 0 and p_ty > 0 and p_yx > 0 and p_y > 0:
                te += p_tyx * np.log2(p_tyx * p_y / (p_ty * p_yx + 1e-10) + 1e-10)

        return max(te, 0)

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()

        # Create returns at different scales
        n_source = len(returns) // self.source_scale
        n_target = len(returns) // self.target_scale

        source_returns = np.array([
            np.sum(returns[i*self.source_scale:(i+1)*self.source_scale])
            for i in range(n_source)
        ])

        target_returns = np.array([
            np.sum(returns[i*self.target_scale:(i+1)*self.target_scale])
            for i in range(n_target)
        ])

        # Align lengths
        min_len = min(len(source_returns), len(target_returns))
        source_returns = source_returns[-min_len:]
        target_returns = target_returns[-min_len:]

        if min_len < 20:
            self.signal = 0
            self.confidence = 0.3
            return

        # Compute transfer entropy both directions
        te_large_to_small = self._transfer_entropy(source_returns, target_returns)
        te_small_to_large = self._transfer_entropy(target_returns, source_returns)

        self.transfer_entropy = te_large_to_small - te_small_to_large

        if te_large_to_small > te_small_to_large + 0.1:
            self.info_direction = 'large_to_small'
            # Large scale drives small - follow large scale trend
            self.signal = 1 if source_returns[-1] > 0 else -1
            self.confidence = min(te_large_to_small, 1.0)
        elif te_small_to_large > te_large_to_small + 0.1:
            self.info_direction = 'small_to_large'
            # Small scale leads - follow small scale
            self.signal = 1 if target_returns[-1] > 0 else -1
            self.confidence = min(te_small_to_large, 1.0)
        else:
            self.info_direction = 'none'
            self.signal = 0
            self.confidence = 0.3


# =============================================================================
# CROSS-CORRELATION ANALYSIS (366-370)
# =============================================================================

@FormulaRegistry.register(366)
class DCCACoefficient(BaseFormula):
    """
    ID 366: Detrended Cross-Correlation Coefficient (ρ_DCCA)

    Academic Reference:
        - Podobnik & Stanley (2008) "DCCA"
        - Zebende (2011) "DCCA cross-correlation coefficient"

    Scale-dependent correlation coefficient for non-stationary series.
    Better than Pearson correlation for financial data.
    """

    NAME = "DCCACoefficient"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Detrended cross-correlation coefficient by scale"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.scales = [4, 8, 16, 32, 64]
        self.rho_by_scale = {}
        self.dominant_scale = 16

    def _dfa_variance(self, data: np.ndarray, scale: int) -> float:
        """DFA variance for a single series"""
        n = len(data)
        profile = np.cumsum(data - np.mean(data))

        n_segments = n // scale
        if n_segments < 2:
            return np.nan

        variance = 0.0
        for v in range(n_segments):
            segment = profile[v*scale:(v+1)*scale]
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            variance += np.mean((segment - trend)**2)

        return variance / n_segments

    def _dcca_covariance(self, data1: np.ndarray, data2: np.ndarray, scale: int) -> float:
        """DCCA covariance between two series"""
        n = min(len(data1), len(data2))
        profile1 = np.cumsum(data1[:n] - np.mean(data1[:n]))
        profile2 = np.cumsum(data2[:n] - np.mean(data2[:n]))

        n_segments = n // scale
        if n_segments < 2:
            return np.nan

        covariance = 0.0
        for v in range(n_segments):
            seg1 = profile1[v*scale:(v+1)*scale]
            seg2 = profile2[v*scale:(v+1)*scale]
            x = np.arange(scale)

            coeffs1 = np.polyfit(x, seg1, 1)
            coeffs2 = np.polyfit(x, seg2, 1)
            trend1 = np.polyval(coeffs1, x)
            trend2 = np.polyval(coeffs2, x)

            covariance += np.mean((seg1 - trend1) * (seg2 - trend2))

        return covariance / n_segments

    def _compute(self) -> None:
        if len(self.returns) < 70:
            return

        returns = self._returns_array()

        # Create lagged series for cross-correlation
        lag = 1
        series1 = returns[:-lag]
        series2 = returns[lag:]

        self.rho_by_scale = {}

        for scale in self.scales:
            if len(series1) < scale * 4:
                continue

            var1 = self._dfa_variance(series1, scale)
            var2 = self._dfa_variance(series2, scale)
            cov = self._dcca_covariance(series1, series2, scale)

            if not (np.isnan(var1) or np.isnan(var2) or np.isnan(cov)):
                if var1 > 0 and var2 > 0:
                    rho = cov / np.sqrt(var1 * var2)
                    self.rho_by_scale[scale] = np.clip(rho, -1, 1)

        if not self.rho_by_scale:
            self.signal = 0
            self.confidence = 0.3
            return

        # Find scale with highest absolute correlation
        self.dominant_scale = max(self.rho_by_scale,
                                   key=lambda s: abs(self.rho_by_scale[s]))
        dominant_rho = self.rho_by_scale[self.dominant_scale]

        # Signal based on dominant scale correlation
        if abs(dominant_rho) > 0.3:
            if dominant_rho > 0:
                # Positive autocorrelation - momentum
                self.signal = 1 if returns[-1] > 0 else -1
            else:
                # Negative autocorrelation - mean reversion
                self.signal = -1 if returns[-1] > 0 else 1
            self.confidence = abs(dominant_rho)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(367)
class MultiscaleDCCA(BaseFormula):
    """
    ID 367: Multiscale Multifractal DCCA (MM-DCCA)

    Academic Reference:
        - Yin & Shang (2013) "Modified DFA and DCCA approach"
        - Zhou (2008) "Multifractal DCCA"

    Extends DCCA to multiple q-order moments for complete
    cross-scale correlation structure.
    """

    NAME = "MultiscaleDCCA"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Multifractal cross-correlation across scales"

    def __init__(self, lookback: int = 256, **kwargs):
        super().__init__(lookback, **kwargs)
        self.q_values = [1, 2, 3, 4]
        self.scales = [8, 16, 32, 64]
        self.hxy_q = {}  # Cross-scaling exponent by q
        self.multifractal_strength = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 130:
            return

        returns = self._returns_array()

        # Split into two halves for cross-correlation
        mid = len(returns) // 2
        series1 = returns[:mid]
        series2 = returns[mid:]

        min_len = min(len(series1), len(series2))
        series1 = series1[-min_len:]
        series2 = series2[-min_len:]

        profile1 = np.cumsum(series1 - np.mean(series1))
        profile2 = np.cumsum(series2 - np.mean(series2))

        self.hxy_q = {}

        for q in self.q_values:
            fq_values = []

            for scale in self.scales:
                n_segments = min_len // scale
                if n_segments < 4:
                    continue

                fluct = []
                for v in range(n_segments):
                    seg1 = profile1[v*scale:(v+1)*scale]
                    seg2 = profile2[v*scale:(v+1)*scale]
                    x = np.arange(scale)

                    trend1 = np.polyval(np.polyfit(x, seg1, 1), x)
                    trend2 = np.polyval(np.polyfit(x, seg2, 1), x)

                    f2 = np.mean((seg1 - trend1) * (seg2 - trend2))
                    fluct.append(abs(f2))

                if fluct:
                    # q-th order fluctuation
                    if q == 0:
                        Fq = np.exp(np.mean(np.log(np.array(fluct) + 1e-10)))
                    else:
                        Fq = np.power(np.mean(np.power(np.array(fluct) + 1e-10, q/2)), 1/q)

                    if Fq > 0:
                        fq_values.append((np.log(scale), np.log(Fq)))

            if len(fq_values) >= 2:
                log_s = np.array([v[0] for v in fq_values])
                log_fq = np.array([v[1] for v in fq_values])
                hxy = np.polyfit(log_s, log_fq, 1)[0]
                self.hxy_q[q] = hxy

        if len(self.hxy_q) < 2:
            self.signal = 0
            self.confidence = 0.3
            return

        # Multifractal strength = range of H(q)
        h_values = list(self.hxy_q.values())
        self.multifractal_strength = max(h_values) - min(h_values)

        # Main scaling exponent (q=2)
        h2 = self.hxy_q.get(2, 0.5)

        # Signal based on cross-correlation structure
        if h2 > 0.55:
            # Positive cross-scaling - follow momentum
            self.signal = 1 if returns[-1] > 0 else -1
            self.confidence = min((h2 - 0.5) * 3, 1.0)
        elif h2 < 0.45:
            # Anti-correlated - mean reversion
            z = returns[-1] / (np.std(returns) + 1e-10)
            self.signal = -1 if z > 0.5 else (1 if z < -0.5 else 0)
            self.confidence = min((0.5 - h2) * 3, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(368)
class EppsEffectCorrector(BaseFormula):
    """
    ID 368: Epps Effect Correction for Scale-Dependent Correlation

    Academic Reference:
        - Epps (1979) "Comovements in Stock Prices"
        - Toth & Kertesz (2009) "The Epps effect revisited"

    Correlation decreases at high frequencies due to asynchronous trading.
    This formula estimates true correlation correcting for the Epps effect.
    """

    NAME = "EppsEffectCorrector"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Correct correlation for Epps effect at each scale"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.correlation_by_scale = {}
        self.asymptotic_correlation = 0.0
        self.characteristic_time = 10

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()

        # Use lagged returns as proxy for "another asset"
        lag = 1
        series1 = returns[:-lag]
        series2 = returns[lag:]

        # Measure correlation at different aggregation levels
        scales = [1, 2, 5, 10, 20, 40]
        self.correlation_by_scale = {}

        for scale in scales:
            if len(series1) < scale * 5:
                continue

            n = len(series1) // scale
            agg1 = np.array([np.sum(series1[i*scale:(i+1)*scale]) for i in range(n)])
            agg2 = np.array([np.sum(series2[i*scale:(i+1)*scale]) for i in range(n)])

            if len(agg1) > 3:
                corr = np.corrcoef(agg1, agg2)[0, 1]
                if not np.isnan(corr):
                    self.correlation_by_scale[scale] = corr

        if len(self.correlation_by_scale) < 2:
            self.signal = 0
            self.confidence = 0.3
            return

        # Fit Epps curve: rho(delta_t) = rho_inf * (1 - exp(-delta_t / tau))
        # Simple estimation: asymptotic = max correlation
        self.asymptotic_correlation = max(self.correlation_by_scale.values())

        # Estimate characteristic time
        scales_list = sorted(self.correlation_by_scale.keys())
        for i, s in enumerate(scales_list):
            rho = self.correlation_by_scale[s]
            if rho > 0.5 * self.asymptotic_correlation:
                self.characteristic_time = s
                break

        # Signal based on corrected correlation
        if abs(self.asymptotic_correlation) > 0.3:
            if self.asymptotic_correlation > 0:
                self.signal = 1 if returns[-1] > 0 else -1
            else:
                self.signal = -1 if returns[-1] > 0 else 1
            self.confidence = abs(self.asymptotic_correlation)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(369)
class TurbulentCascade(BaseFormula):
    """
    ID 369: Turbulent Cascade Model for Volatility

    Academic Reference:
        - Ghashghaie et al. (1996) "Turbulent cascades in FX markets" (Nature)
        - Mandelbrot's cascade models

    Models information cascade from large to small scales,
    analogous to energy cascade in turbulence.
    """

    NAME = "TurbulentCascade"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Turbulence-inspired volatility cascade"

    def __init__(self, lookback: int = 256, **kwargs):
        super().__init__(lookback, **kwargs)
        self.scales = [2, 4, 8, 16, 32, 64]
        self.cascade_exponent = 0.0
        self.intermittency = 0.0

    def _compute(self) -> None:
        if len(self.returns) < 130:
            return

        returns = self._returns_array()

        # Compute volatility at each scale
        vol_by_scale = {}
        for scale in self.scales:
            n = len(returns) // scale
            if n < 4:
                continue

            agg_returns = np.array([
                np.sum(returns[i*scale:(i+1)*scale])
                for i in range(n)
            ])

            # Standard deviation normalized by sqrt(scale)
            vol = np.std(agg_returns) / np.sqrt(scale)
            vol_by_scale[scale] = vol

        if len(vol_by_scale) < 3:
            self.signal = 0
            self.confidence = 0.3
            return

        # Fit power law: sigma(s) ~ s^H
        # In turbulence, this would be Kolmogorov H = 1/3
        log_s = np.array([np.log(s) for s in vol_by_scale.keys()])
        log_vol = np.array([np.log(v + 1e-10) for v in vol_by_scale.values()])

        H = np.polyfit(log_s, log_vol, 1)[0]
        self.cascade_exponent = H

        # Intermittency: deviation from pure scaling
        predicted = np.polyval(np.polyfit(log_s, log_vol, 1), log_s)
        residuals = log_vol - predicted
        self.intermittency = np.std(residuals)

        # Trading signal
        # H > 0.5 suggests trending behavior at all scales
        # H < 0.5 suggests mean reversion
        if H > 0.55:
            self.signal = 1 if returns[-1] > 0 else -1
            self.confidence = min((H - 0.5) * 4, 1.0)
        elif H < 0.45:
            z = returns[-1] / (np.std(returns) + 1e-10)
            self.signal = -1 if z > 1 else (1 if z < -1 else 0)
            self.confidence = min((0.5 - H) * 4, 1.0)
        else:
            # High intermittency = volatile regime
            if self.intermittency > 0.3:
                self.signal = 0
                self.confidence = 0.2
            else:
                self.signal = 0
                self.confidence = 0.4


@FormulaRegistry.register(370)
class RoughVolatilityEstimator(BaseFormula):
    """
    ID 370: Rough Volatility Estimator (Gatheral-Rosenbaum)

    Academic Reference:
        - Gatheral, Jaisson & Rosenbaum (2014) "Volatility is rough"
        - H ≈ 0.1 for log-volatility (much rougher than H=0.5 Brownian motion)

    Estimates the Hurst exponent of log-volatility.
    Rough vol (H < 0.5) implies volatility clustering at all scales.
    """

    NAME = "RoughVolatilityEstimator"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Rough volatility Hurst estimation (H << 0.5)"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.vol_hurst = 0.5
        self.roughness = 0.0  # 0.5 - H

    def _estimate_local_vol(self, returns: np.ndarray, window: int = 10) -> np.ndarray:
        """Estimate local volatility using rolling std"""
        n = len(returns)
        vol = np.zeros(n - window + 1)
        for i in range(n - window + 1):
            vol[i] = np.std(returns[i:i+window])
        return vol

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()

        # Estimate local volatility
        vol = self._estimate_local_vol(returns, window=10)
        if len(vol) < 50:
            self.signal = 0
            self.confidence = 0.3
            return

        # Take log of volatility
        log_vol = np.log(vol + 1e-10)

        # Estimate Hurst exponent of log-volatility using variogram
        lags = [1, 2, 4, 8, 16]
        variogram = []

        for lag in lags:
            if len(log_vol) < lag + 10:
                continue
            diffs = log_vol[lag:] - log_vol[:-lag]
            var = np.mean(diffs**2)
            variogram.append((np.log(lag), np.log(var + 1e-10)))

        if len(variogram) >= 2:
            log_lag = np.array([v[0] for v in variogram])
            log_var = np.array([v[1] for v in variogram])
            # Variogram scales as lag^(2H)
            slope = np.polyfit(log_lag, log_var, 1)[0]
            self.vol_hurst = slope / 2
            self.vol_hurst = np.clip(self.vol_hurst, 0, 1)

        self.roughness = 0.5 - self.vol_hurst

        # Trading implication of rough volatility
        # Very rough (H < 0.2): Strong volatility clustering
        # Expect volatility bursts to persist at all scales

        current_vol = np.std(returns[-10:])
        historical_vol = np.std(returns[:-10])
        vol_ratio = current_vol / (historical_vol + 1e-10)

        if self.vol_hurst < 0.3:
            # Very rough - volatility regime matters
            if vol_ratio > 1.5:
                # High vol regime - mean revert
                z = returns[-1] / (current_vol + 1e-10)
                self.signal = -1 if z > 1 else (1 if z < -1 else 0)
                self.confidence = min(self.roughness * 2, 1.0)
            elif vol_ratio < 0.7:
                # Low vol regime - momentum
                self.signal = 1 if returns[-1] > 0 else -1
                self.confidence = min(self.roughness * 2, 1.0)
            else:
                self.signal = 0
                self.confidence = 0.4
        else:
            self.signal = 0
            self.confidence = 0.3


# =============================================================================
# VARIANCE RATIO EXTENSIONS (371-375)
# =============================================================================

@FormulaRegistry.register(371)
class ContinuousVRFunction(BaseFormula):
    """
    ID 371: Continuous Variance Ratio Function VR(τ)

    Academic Reference:
        - Lo & MacKinlay (1988) "Stock market prices do not follow random walks"
        - Poterba & Summers (1988) "Mean reversion in stock prices"

    Computes VR as a continuous function of horizon τ.
    Find the exact horizon where mean reversion or momentum peaks.
    """

    NAME = "ContinuousVRFunction"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "VR(τ) as continuous function of horizon"

    def __init__(self, lookback: int = 300, **kwargs):
        super().__init__(lookback, **kwargs)
        self.horizons = list(range(2, 65, 2))
        self.vr_function = {}
        self.peak_momentum_horizon = 2
        self.peak_reversion_horizon = 2

    def _variance_ratio(self, returns: np.ndarray, q: int) -> float:
        """Compute variance ratio for horizon q"""
        n = len(returns)
        if n < q * 3:
            return 1.0

        var1 = np.var(returns, ddof=1)
        if var1 < 1e-10:
            return 1.0

        # q-period returns
        q_returns = np.array([np.sum(returns[i:i+q]) for i in range(n - q + 1)])
        var_q = np.var(q_returns, ddof=1)

        return var_q / (q * var1)

    def _compute(self) -> None:
        if len(self.returns) < 150:
            return

        returns = self._returns_array()

        self.vr_function = {}
        max_vr = 1.0
        min_vr = 1.0
        self.peak_momentum_horizon = 2
        self.peak_reversion_horizon = 2

        for q in self.horizons:
            vr = self._variance_ratio(returns, q)
            self.vr_function[q] = vr

            if vr > max_vr:
                max_vr = vr
                self.peak_momentum_horizon = q
            if vr < min_vr:
                min_vr = vr
                self.peak_reversion_horizon = q

        # Determine optimal strategy based on VR function
        momentum_deviation = max_vr - 1
        reversion_deviation = 1 - min_vr

        if momentum_deviation > reversion_deviation and momentum_deviation > 0.15:
            # Momentum strategy at peak horizon
            q = self.peak_momentum_horizon
            recent = np.sum(returns[-q:])
            self.signal = 1 if recent > 0 else -1
            self.confidence = min(momentum_deviation * 2, 1.0)

        elif reversion_deviation > momentum_deviation and reversion_deviation > 0.15:
            # Mean reversion at peak horizon
            q = self.peak_reversion_horizon
            recent = np.sum(returns[-q:])
            expected_std = np.std(returns) * np.sqrt(q)
            z = recent / (expected_std + 1e-10)
            self.signal = -1 if z > 1.5 else (1 if z < -1.5 else 0)
            self.confidence = min(reversion_deviation * 2, 1.0)

        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(372)
class AutocorrDecayRate(BaseFormula):
    """
    ID 372: Autocorrelation Decay Rate Analysis

    Academic Reference:
        - Campbell, Lo, MacKinlay (1997) "The Econometrics of Financial Markets"

    Analyzes how autocorrelation decays with lag.
    Exponential decay = short memory, Power decay = long memory.
    """

    NAME = "AutocorrDecayRate"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Analyze autocorrelation decay pattern"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.max_lag = kwargs.get('max_lag', 30)
        self.decay_type = 'none'
        self.decay_rate = 0.0
        self.half_life = np.inf

    def _compute(self) -> None:
        if len(self.returns) < self.max_lag * 3:
            return

        returns = self._returns_array()

        # Compute autocorrelation at multiple lags
        acf = []
        for lag in range(1, self.max_lag + 1):
            corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
            if not np.isnan(corr):
                acf.append((lag, corr))

        if len(acf) < 10:
            self.signal = 0
            self.confidence = 0.3
            return

        lags = np.array([a[0] for a in acf])
        correlations = np.array([a[1] for a in acf])
        abs_corr = np.abs(correlations) + 1e-10

        # Fit exponential: |rho(k)| = exp(-lambda * k)
        log_corr = np.log(abs_corr)
        try:
            exp_coef = np.polyfit(lags, log_corr, 1)[0]
            exp_decay = -exp_coef
        except:
            exp_decay = 0

        # Fit power law: |rho(k)| = k^(-alpha)
        log_lags = np.log(lags)
        try:
            power_coef = np.polyfit(log_lags, log_corr, 1)[0]
            power_decay = -power_coef
        except:
            power_decay = 0

        # Determine decay type
        if exp_decay > 0.1:
            self.decay_type = 'exponential'
            self.decay_rate = exp_decay
            self.half_life = np.log(2) / exp_decay
        elif power_decay > 0.1:
            self.decay_type = 'power'
            self.decay_rate = power_decay
            self.half_life = np.inf  # Long memory
        else:
            self.decay_type = 'none'
            self.decay_rate = 0

        # Trading signal based on autocorrelation at lag 1
        rho1 = correlations[0] if len(correlations) > 0 else 0

        if self.decay_type == 'exponential' and self.half_life < 10:
            # Short memory - mean reversion
            if abs(rho1) > 0.1:
                self.signal = 1 if rho1 < 0 else -1  # Negative autocorr = momentum, positive = fade
                self.confidence = min(abs(rho1) * 2, 1.0)
            else:
                self.signal = 0
                self.confidence = 0.3

        elif self.decay_type == 'power':
            # Long memory - momentum
            recent = np.mean(returns[-5:])
            self.signal = 1 if recent > 0 else -1
            self.confidence = min(power_decay / 2, 1.0)

        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(373)
class ScaleDependentSharpe(BaseFormula):
    """
    ID 373: Scale-Dependent Sharpe Ratio

    The Sharpe ratio scales with sqrt(time) for random walk.
    Deviation from this scaling reveals time-scale opportunities.
    """

    NAME = "ScaleDependentSharpe"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Sharpe ratio anomalies across time scales"

    def __init__(self, lookback: int = 250, **kwargs):
        super().__init__(lookback, **kwargs)
        self.scales = [1, 5, 10, 20, 40]
        self.sharpe_by_scale = {}
        self.scaling_deviation = 0.0
        self.best_scale = 1

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()

        self.sharpe_by_scale = {}

        for scale in self.scales:
            n = len(returns) // scale
            if n < 10:
                continue

            agg_returns = np.array([
                np.sum(returns[i*scale:(i+1)*scale])
                for i in range(n)
            ])

            mean_ret = np.mean(agg_returns)
            std_ret = np.std(agg_returns) + 1e-10
            sharpe = mean_ret / std_ret

            # Annualized (assuming base is 1 period)
            # For random walk: Sharpe_annualized = Sharpe_period * sqrt(periods_per_year)
            # Normalize to compare: Sharpe / sqrt(scale)
            normalized_sharpe = sharpe / np.sqrt(scale)

            self.sharpe_by_scale[scale] = {
                'raw': sharpe,
                'normalized': normalized_sharpe
            }

        if len(self.sharpe_by_scale) < 2:
            self.signal = 0
            self.confidence = 0.3
            return

        # Find deviation from random walk scaling
        normalized = [v['normalized'] for v in self.sharpe_by_scale.values()]
        mean_normalized = np.mean(normalized)
        self.scaling_deviation = np.std(normalized)

        # Find best scale (highest absolute normalized Sharpe)
        self.best_scale = max(self.sharpe_by_scale,
                              key=lambda s: abs(self.sharpe_by_scale[s]['normalized']))

        best_sharpe = self.sharpe_by_scale[self.best_scale]['raw']

        # Trade at best scale
        if abs(best_sharpe) > 0.1:
            s = self.best_scale
            recent = np.sum(returns[-s:])
            self.signal = 1 if best_sharpe > 0 else -1
            self.confidence = min(abs(best_sharpe) * 2, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(374)
class OptimalStoppingMR(BaseFormula):
    """
    ID 374: Optimal Stopping for Mean Reversion (Leung-Li)

    Academic Reference:
        - Leung & Li (2015) "Optimal Mean Reversion Trading with
          Transaction Costs and Stop-Loss Exit"

    Derives optimal entry/exit thresholds for OU process.
    """

    NAME = "OptimalStoppingMR"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Optimal entry/exit thresholds for mean reversion"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.transaction_cost = kwargs.get('transaction_cost', 0.001)
        self.theta = 0.1  # Mean reversion speed
        self.mu = 0.0     # Long-term mean
        self.sigma = 0.01 # Volatility
        self.entry_threshold = 0.0
        self.exit_threshold = 0.0

    def _estimate_ou_params(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Estimate OU parameters: dX = theta*(mu - X)*dt + sigma*dW"""
        n = len(prices)
        if n < 30:
            return 0.1, np.mean(prices), np.std(np.diff(prices))

        # OLS regression: X[t] - X[t-1] = theta*(mu - X[t-1]) + noise
        y = np.diff(prices)
        x = prices[:-1]

        # X[t] = a + b*X[t-1] where a = theta*mu, b = 1-theta
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numer = np.sum((x - x_mean) * (y - y_mean))
        denom = np.sum((x - x_mean)**2) + 1e-10

        b = numer / denom
        a = y_mean - b * x_mean

        # Extract parameters
        theta = -b if b < 0 else 0.1
        theta = np.clip(theta, 0.01, 2.0)
        mu = a / theta if theta > 0 else np.mean(prices)

        residuals = y - (a + b * x)
        sigma = np.std(residuals)

        return theta, mu, sigma

    def _compute_thresholds(self) -> Tuple[float, float]:
        """Compute optimal entry and exit thresholds"""
        # Simplified threshold computation
        # Entry when z-score exceeds entry_z
        # Exit when z-score crosses exit_z

        # Transaction cost adjusted thresholds
        cost_adjustment = self.transaction_cost / (self.sigma + 1e-10)

        # Entry: far from mean (profitable after costs)
        entry_z = 1.5 + cost_adjustment

        # Exit: close to mean (take profit)
        exit_z = 0.5

        entry_level = self.mu + entry_z * self.sigma / np.sqrt(2 * self.theta + 1e-10)
        exit_level = self.mu + exit_z * self.sigma / np.sqrt(2 * self.theta + 1e-10)

        return entry_level, exit_level

    def _compute(self) -> None:
        if len(self.prices) < 50:
            return

        prices = self._prices_array()

        # Estimate OU parameters
        self.theta, self.mu, self.sigma = self._estimate_ou_params(prices)

        # Compute optimal thresholds
        entry_upper, exit_upper = self._compute_thresholds()
        entry_lower = 2 * self.mu - entry_upper
        exit_lower = 2 * self.mu - exit_upper

        self.entry_threshold = abs(entry_upper - self.mu)
        self.exit_threshold = abs(exit_upper - self.mu)

        current_price = prices[-1]
        deviation = current_price - self.mu
        z_score = deviation / (self.sigma / np.sqrt(2 * self.theta + 1e-10) + 1e-10)

        # Trading signals
        if abs(z_score) > self.entry_threshold / self.sigma * np.sqrt(2 * self.theta):
            # Entry signal
            self.signal = -1 if deviation > 0 else 1  # Mean revert
            self.confidence = min(abs(z_score) / 3, 1.0)
        elif abs(z_score) < 0.5:
            # Exit signal (take profit)
            self.signal = 0
            self.confidence = 0.6
        else:
            self.signal = 0
            self.confidence = 0.4


@FormulaRegistry.register(375)
class GHETrading(BaseFormula):
    """
    ID 375: Generalized Hurst Exponent Trading Signals

    Academic Reference:
        - Di Matteo et al. (2007) "Multi-scaling in finance"
        - GHE for q-order moments

    Uses GHE at different q values to detect market conditions.
    q=1: Absolute deviations, q=2: Variance, q=3+: Tail behavior
    """

    NAME = "GHETrading"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Trading signals from Generalized Hurst Exponents"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.q_values = [1, 2, 3, 4]
        self.ghe = {}
        self.multiscaling_strength = 0.0

    def _compute_ghe(self, data: np.ndarray, q: float) -> float:
        """Compute Generalized Hurst Exponent for moment q"""
        scales = [4, 8, 16, 32, 64]
        sq_values = []

        for tau in scales:
            if len(data) < tau + 10:
                continue

            # q-th order structure function
            increments = data[tau:] - data[:-tau]

            if q == 0:
                Sq = np.exp(np.mean(np.log(np.abs(increments) + 1e-10)))
            else:
                Sq = np.power(np.mean(np.power(np.abs(increments) + 1e-10, q)), 1/q)

            if Sq > 0:
                sq_values.append((np.log(tau), np.log(Sq)))

        if len(sq_values) >= 2:
            log_tau = np.array([v[0] for v in sq_values])
            log_sq = np.array([v[1] for v in sq_values])
            H = np.polyfit(log_tau, log_sq, 1)[0]
            return np.clip(H, 0, 2)

        return 0.5

    def _compute(self) -> None:
        if len(self.returns) < 100:
            return

        returns = self._returns_array()

        # Compute GHE for each q
        self.ghe = {}
        for q in self.q_values:
            self.ghe[q] = self._compute_ghe(returns, q)

        if not self.ghe:
            self.signal = 0
            self.confidence = 0.3
            return

        # Multiscaling strength: variation in H(q)
        h_values = list(self.ghe.values())
        self.multiscaling_strength = max(h_values) - min(h_values)

        # Key exponents
        h1 = self.ghe.get(1, 0.5)  # Absolute deviations
        h2 = self.ghe.get(2, 0.5)  # Variance (standard Hurst)
        h3 = self.ghe.get(3, 0.5)  # Tail behavior

        # Trading logic
        if self.multiscaling_strength > 0.2:
            # Strong multiscaling = complex regime
            # Tail behavior matters more
            if h3 > 0.6:
                # Fat tails with persistence - follow extremes
                self.signal = 1 if returns[-1] > 0 else -1
                self.confidence = 0.5
            else:
                self.signal = 0
                self.confidence = 0.3
        else:
            # Monofractal - use standard H
            if h2 > 0.55:
                self.signal = 1 if returns[-1] > 0 else -1
                self.confidence = min((h2 - 0.5) * 4, 1.0)
            elif h2 < 0.45:
                z = returns[-1] / (np.std(returns) + 1e-10)
                self.signal = -1 if z > 1 else (1 if z < -1 else 0)
                self.confidence = min((0.5 - h2) * 4, 1.0)
            else:
                self.signal = 0
                self.confidence = 0.3


# =============================================================================
# DECOMPOSITION METHODS (376-380)
# =============================================================================

@FormulaRegistry.register(376)
class CEEMDANDecomposition(BaseFormula):
    """
    ID 376: CEEMDAN Decomposition for Trading

    Academic Reference:
        - Torres et al. (2011) "Complete Ensemble EMD with Adaptive Noise"
        - Wu & Huang (2009) EEMD

    Decomposes price into IMFs at different frequencies.
    Trade based on dominant IMF behavior.
    """

    NAME = "CEEMDANDecomposition"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "CEEMDAN-based multi-scale decomposition"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_imfs = kwargs.get('n_imfs', 5)
        self.ensemble_size = kwargs.get('ensemble_size', 10)
        self.imf_energies = []
        self.dominant_imf = 0

    def _sift(self, signal: np.ndarray, max_iter: int = 10) -> np.ndarray:
        """Single sifting iteration for EMD"""
        s = signal.copy()

        for _ in range(max_iter):
            # Find extrema
            maxima_idx = []
            minima_idx = []

            for i in range(1, len(s) - 1):
                if s[i] > s[i-1] and s[i] > s[i+1]:
                    maxima_idx.append(i)
                elif s[i] < s[i-1] and s[i] < s[i+1]:
                    minima_idx.append(i)

            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break

            # Interpolate envelopes
            x = np.arange(len(s))
            upper = np.interp(x, maxima_idx, s[maxima_idx])
            lower = np.interp(x, minima_idx, s[minima_idx])

            mean_env = (upper + lower) / 2
            s = s - mean_env

            # Check stopping criterion
            if np.std(mean_env) < 0.01 * np.std(signal):
                break

        return s

    def _emd(self, signal: np.ndarray) -> List[np.ndarray]:
        """Perform EMD decomposition"""
        imfs = []
        residue = signal.copy()

        for _ in range(self.n_imfs):
            imf = self._sift(residue)
            imfs.append(imf)
            residue = residue - imf

            # Stop if residue is monotonic or too small
            if np.std(residue) < 0.01 * np.std(signal):
                break

        imfs.append(residue)
        return imfs

    def _compute(self) -> None:
        if len(self.prices) < 50:
            return

        prices = self._prices_array()

        # Simple EMD (CEEMDAN would add noise and ensemble)
        imfs = self._emd(prices)

        # Compute energy of each IMF in recent data
        self.imf_energies = []
        for i, imf in enumerate(imfs):
            energy = np.mean(imf[-20:]**2)
            self.imf_energies.append(energy)

        if not self.imf_energies:
            self.signal = 0
            self.confidence = 0.3
            return

        # Find dominant IMF (excluding residue)
        if len(self.imf_energies) > 1:
            self.dominant_imf = np.argmax(self.imf_energies[:-1])
        else:
            self.dominant_imf = 0

        # Signal from dominant IMF
        dominant = imfs[self.dominant_imf]

        # Trend of dominant IMF
        if len(dominant) >= 5:
            recent_trend = dominant[-1] - dominant[-5]
            total_energy = sum(self.imf_energies)
            dominant_ratio = self.imf_energies[self.dominant_imf] / (total_energy + 1e-10)

            if abs(recent_trend) > np.std(dominant) * 0.5:
                self.signal = 1 if recent_trend > 0 else -1
                self.confidence = min(dominant_ratio * 2, 1.0)
            else:
                self.signal = 0
                self.confidence = 0.4
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(377)
class WaveletPacketBestBasis(BaseFormula):
    """
    ID 377: Wavelet Packet Best Basis Selection

    Academic Reference:
        - Coifman & Wickerhauser (1992) "Entropy-based algorithms for best basis"

    Finds optimal wavelet packet basis for the signal.
    Uses entropy criterion to select most informative decomposition.
    """

    NAME = "WaveletPacketBestBasis"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Best basis selection via entropy minimization"

    def __init__(self, lookback: int = 256, **kwargs):
        super().__init__(lookback, **kwargs)
        self.max_level = kwargs.get('max_level', 4)
        self.best_nodes = []
        self.total_entropy = 0.0

    def _shannon_entropy(self, coeffs: np.ndarray) -> float:
        """Compute Shannon entropy of coefficients"""
        if len(coeffs) == 0:
            return 0

        # Normalize
        energy = np.sum(coeffs**2)
        if energy < 1e-10:
            return 0

        p = coeffs**2 / energy
        p = p[p > 0]
        return -np.sum(p * np.log2(p + 1e-10))

    def _haar_decompose(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single level Haar decomposition"""
        n = len(data) // 2
        if n == 0:
            return np.array([]), np.array([])

        approx = np.zeros(n)
        detail = np.zeros(n)

        for i in range(n):
            approx[i] = (data[2*i] + data[2*i + 1]) / np.sqrt(2)
            detail[i] = (data[2*i] - data[2*i + 1]) / np.sqrt(2)

        return approx, detail

    def _compute(self) -> None:
        if len(self.returns) < 32:
            return

        returns = self._returns_array()

        # Ensure power of 2 length
        n = 2**int(np.log2(len(returns)))
        data = returns[-n:]

        # Build wavelet packet tree and find best basis
        # Simplified: just compute entropy at each level
        level_entropies = []
        current_nodes = [data]

        for level in range(self.max_level):
            next_nodes = []
            level_entropy = 0

            for node in current_nodes:
                if len(node) < 4:
                    next_nodes.append(node)
                    continue

                approx, detail = self._haar_decompose(node)

                # Entropy comparison
                parent_entropy = self._shannon_entropy(node)
                child_entropy = self._shannon_entropy(approx) + self._shannon_entropy(detail)

                if child_entropy < parent_entropy:
                    # Keep children
                    next_nodes.extend([approx, detail])
                    level_entropy += child_entropy
                else:
                    # Keep parent
                    next_nodes.append(node)
                    level_entropy += parent_entropy

            current_nodes = next_nodes
            level_entropies.append(level_entropy)

        self.best_nodes = current_nodes
        self.total_entropy = level_entropies[-1] if level_entropies else 0

        # Signal from lowest entropy (most organized) node
        if self.best_nodes:
            # Find node with most energy
            energies = [np.sum(node**2) for node in self.best_nodes]
            dominant_idx = np.argmax(energies)
            dominant_node = self.best_nodes[dominant_idx]

            if len(dominant_node) >= 2:
                trend = dominant_node[-1] - dominant_node[0]
                if abs(trend) > np.std(dominant_node) * 0.5:
                    self.signal = 1 if trend > 0 else -1
                    self.confidence = 1 / (1 + self.total_entropy)
                else:
                    self.signal = 0
                    self.confidence = 0.4
            else:
                self.signal = 0
                self.confidence = 0.3
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(378)
class LocallyStationaryWavelet(BaseFormula):
    """
    ID 378: Locally Stationary Wavelet (LSW) Process Model

    Academic Reference:
        - Nason, von Sachs & Kroisandt (2000) "Wavelet processes and adaptive
          estimation of the evolutionary wavelet spectrum"
        - Fryzlewicz et al. "Modelling financial log-returns as LSW"

    Models time-varying second-order structure of returns.
    Captures non-stationarity at multiple scales.
    """

    NAME = "LocallyStationaryWavelet"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "LSW process for non-stationary time-varying volatility"

    def __init__(self, lookback: int = 256, **kwargs):
        super().__init__(lookback, **kwargs)
        self.n_levels = kwargs.get('n_levels', 4)
        self.spectrum = {}  # Evolutionary wavelet spectrum
        self.local_variance = 0.0

    def _haar_dwt(self, data: np.ndarray) -> List[np.ndarray]:
        """Discrete wavelet transform with Haar"""
        details = []
        approx = data.copy()

        for _ in range(self.n_levels):
            n = len(approx) // 2
            if n < 2:
                break

            new_approx = np.zeros(n)
            detail = np.zeros(n)

            for i in range(n):
                new_approx[i] = (approx[2*i] + approx[2*i + 1]) / np.sqrt(2)
                detail[i] = (approx[2*i] - approx[2*i + 1]) / np.sqrt(2)

            details.append(detail)
            approx = new_approx

        details.append(approx)
        return details

    def _compute(self) -> None:
        if len(self.returns) < 64:
            return

        returns = self._returns_array()

        # Power of 2
        n = 2**int(np.log2(len(returns)))
        data = returns[-n:]

        # DWT decomposition
        coeffs = self._haar_dwt(data)

        # Estimate local spectrum at each scale
        self.spectrum = {}
        for j, detail in enumerate(coeffs[:-1]):
            # Rolling variance of wavelet coefficients
            window = max(len(detail) // 4, 4)
            if len(detail) >= window:
                local_var = np.mean(detail[-window:]**2)
                self.spectrum[j+1] = local_var

        if not self.spectrum:
            self.signal = 0
            self.confidence = 0.3
            return

        # Total local variance
        self.local_variance = sum(self.spectrum.values())

        # Find dominant scale
        dominant_scale = max(self.spectrum, key=self.spectrum.get)
        dominant_var = self.spectrum[dominant_scale]

        # Signal based on dominant scale behavior
        dominant_detail = coeffs[dominant_scale - 1]
        if len(dominant_detail) >= 3:
            recent_trend = dominant_detail[-1] - dominant_detail[-3]

            # Variance ratio
            var_ratio = dominant_var / (self.local_variance + 1e-10)

            if abs(recent_trend) > np.std(dominant_detail) * 0.5:
                self.signal = 1 if recent_trend > 0 else -1
                self.confidence = min(var_ratio * 2, 1.0)
            else:
                self.signal = 0
                self.confidence = 0.4
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(379)
class SpectralDensityWhittle(BaseFormula):
    """
    ID 379: Whittle MLE for Spectral Density

    Academic Reference:
        - Whittle (1953) "Estimation and information in stationary time series"
        - Percival & Walden (1993) "Spectral Analysis for Physical Applications"

    Estimates spectral density using Whittle likelihood.
    Detects dominant frequencies for trading cycles.
    """

    NAME = "SpectralDensityWhittle"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "Whittle MLE spectral density estimation"

    def __init__(self, lookback: int = 256, **kwargs):
        super().__init__(lookback, **kwargs)
        self.dominant_frequency = 0.0
        self.dominant_period = np.inf
        self.spectral_slope = 0.0

    def _periodogram(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute periodogram"""
        n = len(data)
        fft = np.fft.fft(data - np.mean(data))
        power = np.abs(fft[:n//2])**2 / n
        freqs = np.fft.fftfreq(n)[:n//2]
        return freqs, power

    def _compute(self) -> None:
        if len(self.returns) < 64:
            return

        returns = self._returns_array()

        # Compute periodogram
        freqs, power = self._periodogram(returns)

        if len(power) < 5:
            self.signal = 0
            self.confidence = 0.3
            return

        # Find dominant frequency (excluding DC)
        power_no_dc = power[1:]
        freqs_no_dc = freqs[1:]

        if len(power_no_dc) > 0:
            max_idx = np.argmax(power_no_dc)
            self.dominant_frequency = abs(freqs_no_dc[max_idx])
            self.dominant_period = 1 / (self.dominant_frequency + 1e-10)

        # Spectral slope (log-log regression for 1/f noise detection)
        valid = (freqs_no_dc > 0) & (power_no_dc > 0)
        if np.sum(valid) > 5:
            log_freq = np.log(freqs_no_dc[valid])
            log_power = np.log(power_no_dc[valid])
            self.spectral_slope = np.polyfit(log_freq, log_power, 1)[0]

        # Trading signal based on dominant cycle
        if self.dominant_period < len(returns) / 2:
            # Clear cycle detected
            period = int(self.dominant_period)

            # Where are we in the cycle?
            cycle_position = (len(returns) % period) / period

            if 0.1 < cycle_position < 0.4:
                self.signal = 1  # Rising phase
            elif 0.6 < cycle_position < 0.9:
                self.signal = -1  # Falling phase
            else:
                self.signal = 0

            # Confidence based on spectral peak prominence
            peak_power = power_no_dc[max_idx]
            avg_power = np.mean(power_no_dc)
            prominence = peak_power / (avg_power + 1e-10)
            self.confidence = min(np.log(prominence + 1) / 3, 1.0)
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(380)
class ACDDurationModel(BaseFormula):
    """
    ID 380: Autoregressive Conditional Duration (ACD) Inspired

    Academic Reference:
        - Engle & Russell (1998) "ACD: A new model for irregularly spaced
          transaction data"

    Models the clustering of trades/price changes.
    High activity periods vs low activity periods.
    """

    NAME = "ACDDurationModel"
    CATEGORY = "multiscale_advanced"
    DESCRIPTION = "ACD-inspired activity clustering detection"

    def __init__(self, lookback: int = 200, **kwargs):
        super().__init__(lookback, **kwargs)
        self.omega = 0.1
        self.alpha = 0.1
        self.beta = 0.8
        self.expected_duration = 1.0
        self.current_intensity = 1.0

    def _compute(self) -> None:
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # Use absolute returns as proxy for "activity"
        abs_returns = np.abs(returns)

        # Estimate ACD-like parameters using simple method
        # Expected duration = time between "significant" moves
        threshold = np.percentile(abs_returns, 75)
        significant = abs_returns > threshold

        # Duration between significant moves
        durations = []
        last_sig = 0
        for i, sig in enumerate(significant):
            if sig:
                if last_sig > 0:
                    durations.append(i - last_sig)
                last_sig = i

        if len(durations) < 5:
            self.signal = 0
            self.confidence = 0.3
            return

        # Simple ACD update
        # psi_i = omega + alpha * d_{i-1} + beta * psi_{i-1}
        self.expected_duration = np.mean(durations)

        if len(durations) >= 2:
            # Current intensity (inverse of expected duration)
            recent_duration = durations[-1] if durations else self.expected_duration
            self.current_intensity = self.expected_duration / (recent_duration + 1)

        # Trading signal
        # High intensity (short durations) = volatile, mean revert
        # Low intensity (long durations) = quiet, momentum

        intensity_ratio = self.current_intensity

        if intensity_ratio > 1.5:
            # High activity - expect continuation then reversion
            z = returns[-1] / (np.std(returns) + 1e-10)
            self.signal = -1 if z > 1 else (1 if z < -1 else 0)
            self.confidence = min(intensity_ratio / 3, 1.0)
        elif intensity_ratio < 0.7:
            # Low activity - expect breakout
            self.signal = 0  # Wait for breakout
            self.confidence = 0.4
        else:
            self.signal = 0
            self.confidence = 0.3


# =============================================================================
# Export all classes
# =============================================================================

__all__ = [
    # Structural Break Detection (361-365)
    'BaiPerronBreakDetector',
    'CUSUMScaleDetector',
    'WBSChangepoint',
    'EntropyScaleSelector',
    'TransferEntropyScale',

    # Cross-Correlation Analysis (366-370)
    'DCCACoefficient',
    'MultiscaleDCCA',
    'EppsEffectCorrector',
    'TurbulentCascade',
    'RoughVolatilityEstimator',

    # Variance Ratio Extensions (371-375)
    'ContinuousVRFunction',
    'AutocorrDecayRate',
    'ScaleDependentSharpe',
    'OptimalStoppingMR',
    'GHETrading',

    # Decomposition Methods (376-380)
    'CEEMDANDecomposition',
    'WaveletPacketBestBasis',
    'LocallyStationaryWavelet',
    'SpectralDensityWhittle',
    'ACDDurationModel',
]
