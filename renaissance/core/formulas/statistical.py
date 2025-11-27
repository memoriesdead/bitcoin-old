"""
Statistical Formulas (IDs 1-30)
==============================
Probability distributions, statistical tests, and fundamental statistical measures.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional
from .base import BaseFormula, FormulaRegistry


@FormulaRegistry.register(1)
class BayesianProbability(BaseFormula):
    """ID 1: Bayesian Probability Update - P(H|E) = P(E|H)P(H) / P(E)"""
    NAME = "BayesianProbability"
    CATEGORY = "statistical"
    DESCRIPTION = "Bayesian posterior probability for trend direction"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prior_up = 0.5
        self.prior_down = 0.5
        self.likelihood_window = kwargs.get('likelihood_window', 20)

    def _compute(self):
        if len(self.returns) < self.likelihood_window:
            return

        returns = self._returns_array()[-self.likelihood_window:]
        up_returns = returns[returns > 0]
        down_returns = returns[returns <= 0]

        # Likelihood of observing positive return given uptrend
        p_e_given_up = len(up_returns) / len(returns) if len(returns) > 0 else 0.5
        p_e_given_down = len(down_returns) / len(returns) if len(returns) > 0 else 0.5

        # Posterior update
        p_e = p_e_given_up * self.prior_up + p_e_given_down * self.prior_down
        if p_e > 0:
            posterior_up = (p_e_given_up * self.prior_up) / p_e
        else:
            posterior_up = 0.5

        # Update priors for next iteration
        self.prior_up = 0.5 * self.prior_up + 0.5 * posterior_up
        self.prior_down = 1 - self.prior_up

        # Generate signal
        self.confidence = abs(posterior_up - 0.5) * 2
        if posterior_up > 0.6:
            self.signal = 1
        elif posterior_up < 0.4:
            self.signal = -1
        else:
            self.signal = 0


@FormulaRegistry.register(2)
class MaximumLikelihood(BaseFormula):
    """ID 2: Maximum Likelihood Estimation for return distribution"""
    NAME = "MaximumLikelihood"
    CATEGORY = "statistical"
    DESCRIPTION = "MLE for return distribution parameters"

    def _compute(self):
        if len(self.returns) < 20:
            return

        returns = self._returns_array()
        # Fit normal distribution using MLE
        mu, sigma = stats.norm.fit(returns)

        # Current return z-score
        if sigma > 0:
            z = (returns[-1] - mu) / sigma
            # Mean reversion signal
            self.confidence = min(abs(z) / 3, 1.0)
            self.signal = self._clip_signal(-z, threshold=1.5)


@FormulaRegistry.register(3)
class HawkesProcess(BaseFormula):
    """ID 3: Hawkes Self-Exciting Process - λ(t) = μ + Σα*exp(-β(t-ti))"""
    NAME = "HawkesProcess"
    CATEGORY = "statistical"
    DESCRIPTION = "Self-exciting point process for event clustering"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mu = kwargs.get('mu', 0.1)  # Background intensity
        self.alpha = kwargs.get('alpha', 0.5)  # Excitation
        self.beta = kwargs.get('beta', 1.0)  # Decay rate
        self.events = []
        self.intensity = self.mu

    def _compute(self):
        if len(self.returns) < 5:
            return

        returns = self._returns_array()
        current_time = len(returns)

        # Detect significant moves as events
        threshold = 2 * np.std(returns)
        if abs(returns[-1]) > threshold:
            self.events.append(current_time)

        # Calculate intensity
        self.intensity = self.mu
        for event_time in self.events[-50:]:  # Keep last 50 events
            dt = current_time - event_time
            self.intensity += self.alpha * np.exp(-self.beta * dt)

        # High intensity suggests clustering - fade the move
        self.confidence = min(self.intensity / (self.mu * 3), 1.0)
        if self.intensity > self.mu * 2:
            # High clustering - expect reversion
            self.signal = -np.sign(returns[-1])
        else:
            self.signal = 0


@FormulaRegistry.register(4)
class StudentTDistribution(BaseFormula):
    """ID 4: Student-t Distribution for fat tails"""
    NAME = "StudentTDistribution"
    CATEGORY = "statistical"
    DESCRIPTION = "Fat-tailed distribution for extreme events"

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()
        # Fit Student-t distribution
        params = stats.t.fit(returns)
        df, loc, scale = params

        # Calculate probability of current return
        current_ret = returns[-1]
        cdf = stats.t.cdf(current_ret, df, loc, scale)

        # Extreme values trigger mean reversion
        self.confidence = abs(cdf - 0.5) * 2
        if cdf > 0.95:
            self.signal = -1  # Overbought
        elif cdf < 0.05:
            self.signal = 1  # Oversold
        else:
            self.signal = 0


@FormulaRegistry.register(5)
class SkewnessKurtosis(BaseFormula):
    """ID 5: Skewness & Kurtosis for distribution shape"""
    NAME = "SkewnessKurtosis"
    CATEGORY = "statistical"
    DESCRIPTION = "Higher moments for tail risk assessment"

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)

        # High kurtosis = fat tails = expect extreme moves
        # Negative skew = left tail risk
        self.confidence = min((abs(kurt) + abs(skew)) / 6, 1.0)

        if skew < -0.5 and kurt > 1:
            # Negative skew with fat tails - downside risk
            self.signal = -1
        elif skew > 0.5 and kurt > 1:
            # Positive skew with fat tails - upside potential
            self.signal = 1
        else:
            self.signal = 0


@FormulaRegistry.register(6)
class EntropyMeasure(BaseFormula):
    """ID 6: Shannon Entropy for information content"""
    NAME = "EntropyMeasure"
    CATEGORY = "statistical"
    DESCRIPTION = "Information entropy for predictability assessment"

    def _compute(self):
        if len(self.returns) < 20:
            return

        returns = self._returns_array()
        # Discretize returns into bins
        hist, _ = np.histogram(returns, bins=10, density=True)
        hist = hist[hist > 0]  # Remove zeros

        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        max_entropy = np.log2(10)  # Maximum possible entropy

        # Normalized entropy
        norm_entropy = entropy / max_entropy

        self.confidence = 1 - norm_entropy  # Low entropy = high predictability

        # Low entropy suggests trending, high entropy suggests mean reversion
        if norm_entropy < 0.5:
            # Predictable - follow trend
            self.signal = 1 if returns[-1] > 0 else -1
        else:
            self.signal = 0


@FormulaRegistry.register(7)
class CointegrationTest(BaseFormula):
    """ID 7: Johansen Cointegration Test"""
    NAME = "CointegrationTest"
    CATEGORY = "statistical"
    DESCRIPTION = "Test for cointegration between price series"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reference_prices = []
        self.spread = 0

    def _compute(self):
        if len(self.prices) < 50:
            return

        prices = self._prices_array()

        # Create synthetic reference (lagged prices)
        reference = np.roll(prices, 5)
        reference[:5] = prices[:5]

        # Calculate spread
        beta = np.cov(prices[-30:], reference[-30:])[0, 1] / np.var(reference[-30:])
        spread = prices - beta * reference

        # Z-score of spread
        spread_mean = np.mean(spread[-30:])
        spread_std = np.std(spread[-30:])

        if spread_std > 0:
            z = (spread[-1] - spread_mean) / spread_std
            self.spread = z
            self.confidence = min(abs(z) / 2, 1.0)
            self.signal = self._clip_signal(-z, threshold=1.5)


@FormulaRegistry.register(8)
class KLDivergence(BaseFormula):
    """ID 8: Kullback-Leibler Divergence"""
    NAME = "KLDivergence"
    CATEGORY = "statistical"
    DESCRIPTION = "KL divergence for distribution comparison"

    def _compute(self):
        if len(self.returns) < 40:
            return

        returns = self._returns_array()
        recent = returns[-20:]
        historical = returns[-40:-20]

        # Create histograms
        bins = np.linspace(min(returns), max(returns), 11)
        p, _ = np.histogram(recent, bins=bins, density=True)
        q, _ = np.histogram(historical, bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        p = p + 1e-10
        q = q + 1e-10

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        # KL divergence
        kl_div = np.sum(p * np.log(p / q))

        self.confidence = min(kl_div, 1.0)

        # High divergence suggests regime change
        if kl_div > 0.5:
            # Distribution shifted - follow new regime
            self.signal = 1 if np.mean(recent) > np.mean(historical) else -1
        else:
            self.signal = 0


@FormulaRegistry.register(9)
class AutocorrelationTest(BaseFormula):
    """ID 9: Autocorrelation for serial dependence"""
    NAME = "AutocorrelationTest"
    CATEGORY = "statistical"
    DESCRIPTION = "Test for return autocorrelation"

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()

        # Calculate autocorrelation at lag 1
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]

        self.confidence = abs(autocorr)

        if autocorr > 0.2:
            # Positive autocorr - momentum
            self.signal = 1 if returns[-1] > 0 else -1
        elif autocorr < -0.2:
            # Negative autocorr - mean reversion
            self.signal = -1 if returns[-1] > 0 else 1
        else:
            self.signal = 0


@FormulaRegistry.register(10)
class VarianceRatioTest(BaseFormula):
    """ID 10: Variance Ratio Test for random walk"""
    NAME = "VarianceRatioTest"
    CATEGORY = "statistical"
    DESCRIPTION = "Lo-MacKinlay variance ratio test"

    def _compute(self):
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # Variance at different horizons
        var1 = np.var(returns)

        # 5-period returns
        returns5 = np.array([sum(returns[i:i+5]) for i in range(0, len(returns)-4, 5)])
        var5 = np.var(returns5) / 5

        # Variance ratio
        if var1 > 0:
            vr = var5 / var1
        else:
            vr = 1

        self.confidence = abs(vr - 1)

        if vr > 1.2:
            # Trending
            self.signal = 1 if returns[-1] > 0 else -1
        elif vr < 0.8:
            # Mean reverting
            self.signal = -1 if returns[-1] > 0 else 1
        else:
            self.signal = 0


@FormulaRegistry.register(11)
class RunsTest(BaseFormula):
    """ID 11: Runs Test for randomness"""
    NAME = "RunsTest"
    CATEGORY = "statistical"
    DESCRIPTION = "Wald-Wolfowitz runs test for pattern detection"

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()
        signs = np.sign(returns)

        # Count runs
        runs = 1
        for i in range(1, len(signs)):
            if signs[i] != signs[i-1]:
                runs += 1

        n_pos = np.sum(signs > 0)
        n_neg = np.sum(signs <= 0)
        n = len(signs)

        # Expected runs under random walk
        expected_runs = (2 * n_pos * n_neg) / n + 1
        var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))

        if var_runs > 0:
            z = (runs - expected_runs) / np.sqrt(var_runs)
        else:
            z = 0

        self.confidence = min(abs(z) / 2, 1.0)

        if z < -2:
            # Too few runs - trending
            self.signal = 1 if signs[-1] > 0 else -1
        elif z > 2:
            # Too many runs - mean reverting
            self.signal = -1 if signs[-1] > 0 else 1
        else:
            self.signal = 0


@FormulaRegistry.register(12)
class HurstExponent(BaseFormula):
    """ID 12: Hurst Exponent - H for long-range dependence"""
    NAME = "HurstExponent"
    CATEGORY = "statistical"
    DESCRIPTION = "R/S analysis for trending vs mean reversion"

    def _compute(self):
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # R/S analysis
        lags = [10, 20, 30, 40]
        rs_values = []

        for lag in lags:
            if len(returns) >= lag:
                data = returns[-lag:]
                mean = np.mean(data)
                cumdev = np.cumsum(data - mean)
                R = max(cumdev) - min(cumdev)
                S = np.std(data)
                if S > 0:
                    rs_values.append((np.log(lag), np.log(R/S)))

        if len(rs_values) >= 2:
            x = [v[0] for v in rs_values]
            y = [v[1] for v in rs_values]
            H = np.polyfit(x, y, 1)[0]
        else:
            H = 0.5

        self.confidence = abs(H - 0.5) * 2

        if H > 0.55:
            # Trending
            self.signal = 1 if returns[-1] > 0 else -1
        elif H < 0.45:
            # Mean reverting
            self.signal = -1 if returns[-1] > 0 else 1
        else:
            self.signal = 0


@FormulaRegistry.register(13)
class MomentEstimation(BaseFormula):
    """ID 13: Method of Moments for distribution fitting"""
    NAME = "MomentEstimation"
    CATEGORY = "statistical"
    DESCRIPTION = "Moment matching for return distribution"

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()

        # First four moments
        m1 = np.mean(returns)  # Mean
        m2 = np.var(returns)   # Variance
        m3 = stats.skew(returns)  # Skewness
        m4 = stats.kurtosis(returns)  # Kurtosis

        # Recent moments
        recent = returns[-10:]
        r1 = np.mean(recent)
        r2 = np.var(recent)

        # Compare recent to historical
        mean_shift = (r1 - m1) / np.sqrt(m2) if m2 > 0 else 0
        vol_shift = r2 / m2 if m2 > 0 else 1

        self.confidence = min(abs(mean_shift) + abs(np.log(vol_shift)), 1.0)

        if mean_shift > 0.5:
            self.signal = 1
        elif mean_shift < -0.5:
            self.signal = -1
        else:
            self.signal = 0


@FormulaRegistry.register(14)
class BootstrapConfidence(BaseFormula):
    """ID 14: Bootstrap Confidence Intervals"""
    NAME = "BootstrapConfidence"
    CATEGORY = "statistical"
    DESCRIPTION = "Bootstrap for signal confidence"

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()
        n_bootstrap = 100

        # Bootstrap mean estimates
        means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            means.append(np.mean(sample))

        means = np.array(means)
        ci_low = np.percentile(means, 5)
        ci_high = np.percentile(means, 95)
        mean_estimate = np.mean(means)

        # Signal based on confidence interval
        self.confidence = 1 - (ci_high - ci_low) / (2 * np.std(returns))
        self.confidence = max(0, min(self.confidence, 1))

        if ci_low > 0:
            self.signal = 1
        elif ci_high < 0:
            self.signal = -1
        else:
            self.signal = 0


@FormulaRegistry.register(15)
class CrossSectionalMomentum(BaseFormula):
    """ID 15: Jegadeesh-Titman Cross-Sectional Momentum"""
    NAME = "CrossSectionalMomentum"
    CATEGORY = "statistical"
    DESCRIPTION = "Relative strength momentum"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.formation_period = kwargs.get('formation_period', 20)

    def _compute(self):
        if len(self.returns) < self.formation_period:
            return

        returns = self._returns_array()
        cumulative_return = np.prod(1 + returns[-self.formation_period:]) - 1

        # Z-score of cumulative return
        rolling_cum = []
        for i in range(self.formation_period, len(returns)):
            rolling_cum.append(np.prod(1 + returns[i-self.formation_period:i]) - 1)

        if len(rolling_cum) > 5:
            z = (cumulative_return - np.mean(rolling_cum)) / np.std(rolling_cum)
            self.confidence = min(abs(z) / 2, 1.0)
            self.signal = self._clip_signal(z, threshold=1.0)
        else:
            self.signal = 0


@FormulaRegistry.register(16)
class TimeSeriesMomentum(BaseFormula):
    """ID 16: Moskowitz-Ooi-Pedersen Time Series Momentum"""
    NAME = "TimeSeriesMomentum"
    CATEGORY = "statistical"
    DESCRIPTION = "Absolute momentum based on own past returns"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lookback = kwargs.get('momentum_lookback', 20)

    def _compute(self):
        if len(self.returns) < self.lookback:
            return

        returns = self._returns_array()
        cumulative = np.prod(1 + returns[-self.lookback:]) - 1

        # Volatility-adjusted momentum
        vol = np.std(returns[-self.lookback:])
        if vol > 0:
            risk_adj_mom = cumulative / vol
        else:
            risk_adj_mom = 0

        self.confidence = min(abs(risk_adj_mom) / 2, 1.0)
        self.signal = self._clip_signal(risk_adj_mom, threshold=0.5)


@FormulaRegistry.register(17)
class ARIMAMomentum(BaseFormula):
    """ID 17: ARIMA-based Momentum Prediction"""
    NAME = "ARIMAMomentum"
    CATEGORY = "statistical"
    DESCRIPTION = "ARIMA(1,0,1) momentum forecasting"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ar_coef = 0.0
        self.ma_coef = 0.0

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()

        # Simple AR(1) estimation
        y = returns[1:]
        x = returns[:-1]
        if np.var(x) > 0:
            self.ar_coef = np.corrcoef(x, y)[0, 1]
        else:
            self.ar_coef = 0

        # Forecast
        forecast = self.ar_coef * returns[-1]

        self.confidence = abs(self.ar_coef)
        self.signal = self._clip_signal(forecast * 100, threshold=0.1)


@FormulaRegistry.register(18)
class ExponentialSmoothing(BaseFormula):
    """ID 18: Holt-Winters Exponential Smoothing"""
    NAME = "ExponentialSmoothing"
    CATEGORY = "statistical"
    DESCRIPTION = "Triple exponential smoothing for trend"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = kwargs.get('alpha', 0.3)
        self.beta = kwargs.get('beta', 0.1)
        self.level = None
        self.trend = 0

    def _compute(self):
        prices = self._prices_array()
        if len(prices) < 5:
            return

        price = prices[-1]

        if self.level is None:
            self.level = price
            self.trend = 0
        else:
            prev_level = self.level
            self.level = self.alpha * price + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend

        # Signal based on trend
        self.confidence = min(abs(self.trend) / np.std(prices[-20:]), 1.0)
        self.signal = self._clip_signal(self.trend * 1000, threshold=0.01)


@FormulaRegistry.register(19)
class GaussianMixture(BaseFormula):
    """ID 19: Gaussian Mixture Model for regime detection"""
    NAME = "GaussianMixture"
    CATEGORY = "statistical"
    DESCRIPTION = "GMM for multi-modal return distribution"

    def _compute(self):
        if len(self.returns) < 50:
            return

        returns = self._returns_array()

        # Simple 2-component GMM approximation
        median = np.median(returns)
        lower = returns[returns < median]
        upper = returns[returns >= median]

        mu1 = np.mean(lower) if len(lower) > 0 else 0
        mu2 = np.mean(upper) if len(upper) > 0 else 0
        sig1 = np.std(lower) if len(lower) > 0 else 1
        sig2 = np.std(upper) if len(upper) > 0 else 1

        # Which component is current return closer to?
        current = returns[-1]
        dist1 = abs(current - mu1) / sig1 if sig1 > 0 else float('inf')
        dist2 = abs(current - mu2) / sig2 if sig2 > 0 else float('inf')

        self.confidence = 1 - min(dist1, dist2) / max(dist1, dist2)

        if dist1 < dist2:
            # In lower regime
            self.signal = -1 if mu1 < mu2 else 1
        else:
            # In upper regime
            self.signal = 1 if mu2 > mu1 else -1


@FormulaRegistry.register(20)
class ParetoTail(BaseFormula):
    """ID 20: Pareto Tail Index for extreme values"""
    NAME = "ParetoTail"
    CATEGORY = "statistical"
    DESCRIPTION = "Power law tail estimation"

    def _compute(self):
        if len(self.returns) < 50:
            return

        returns = self._returns_array()
        abs_returns = np.abs(returns)

        # Hill estimator for tail index
        threshold = np.percentile(abs_returns, 90)
        exceedances = abs_returns[abs_returns > threshold]

        if len(exceedances) > 5:
            alpha = len(exceedances) / np.sum(np.log(exceedances / threshold))
        else:
            alpha = 2  # Default

        # Lower alpha = fatter tails = more risk
        self.confidence = min(1 / alpha, 1.0)

        if alpha < 2 and abs(returns[-1]) > threshold:
            # Fat tails and extreme move - expect reversion
            self.signal = -np.sign(returns[-1])
        else:
            self.signal = 0


@FormulaRegistry.register(21)
class CopulaDependence(BaseFormula):
    """ID 21: Copula-based Dependence Measure"""
    NAME = "CopulaDependence"
    CATEGORY = "statistical"
    DESCRIPTION = "Tail dependence via copula"

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()

        # Kendall's tau for dependence
        lagged = returns[:-1]
        current = returns[1:]

        if len(lagged) > 5:
            tau, _ = stats.kendalltau(lagged, current)
        else:
            tau = 0

        self.confidence = abs(tau)

        if tau > 0.2:
            # Positive dependence - momentum
            self.signal = 1 if returns[-1] > 0 else -1
        elif tau < -0.2:
            # Negative dependence - mean reversion
            self.signal = -1 if returns[-1] > 0 else 1
        else:
            self.signal = 0


@FormulaRegistry.register(22)
class OrderStatistics(BaseFormula):
    """ID 22: Order Statistics for extremes"""
    NAME = "OrderStatistics"
    CATEGORY = "statistical"
    DESCRIPTION = "Min/max statistics for breakout detection"

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()
        current = prices[-1]

        # Rolling high/low
        high = np.max(prices[-20:])
        low = np.min(prices[-20:])
        range_size = high - low

        if range_size > 0:
            position = (current - low) / range_size
        else:
            position = 0.5

        self.confidence = abs(position - 0.5) * 2

        if position > 0.9:
            self.signal = 1  # Breakout high
        elif position < 0.1:
            self.signal = -1  # Breakout low
        else:
            self.signal = 0


@FormulaRegistry.register(23)
class RankCorrelation(BaseFormula):
    """ID 23: Spearman Rank Correlation"""
    NAME = "RankCorrelation"
    CATEGORY = "statistical"
    DESCRIPTION = "Non-parametric correlation for trend"

    def _compute(self):
        if len(self.prices) < 20:
            return

        prices = self._prices_array()[-20:]
        time_ranks = np.arange(len(prices))

        rho, _ = stats.spearmanr(time_ranks, prices)

        self.confidence = abs(rho)
        self.signal = self._clip_signal(rho * 2, threshold=0.5)


@FormulaRegistry.register(24)
class QuantileRegression(BaseFormula):
    """ID 24: Quantile Regression for tail behavior"""
    NAME = "QuantileRegression"
    CATEGORY = "statistical"
    DESCRIPTION = "Conditional quantile estimation"

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()

        # Simple quantile estimation
        q10 = np.percentile(returns, 10)
        q50 = np.percentile(returns, 50)
        q90 = np.percentile(returns, 90)

        current = returns[-1]

        # Where is current return in distribution?
        if current < q10:
            self.signal = 1  # Oversold
            self.confidence = 0.8
        elif current > q90:
            self.signal = -1  # Overbought
            self.confidence = 0.8
        else:
            self.signal = 0
            self.confidence = 0.3


@FormulaRegistry.register(25)
class TruncatedMean(BaseFormula):
    """ID 25: Truncated/Trimmed Mean for robustness"""
    NAME = "TruncatedMean"
    CATEGORY = "statistical"
    DESCRIPTION = "Robust location estimate"

    def _compute(self):
        if len(self.returns) < 20:
            return

        returns = self._returns_array()

        # 10% trimmed mean
        trimmed = stats.trim_mean(returns, 0.1)
        regular_mean = np.mean(returns)

        # Difference indicates outlier influence
        diff = regular_mean - trimmed

        self.confidence = min(abs(diff) / np.std(returns), 1.0)

        if diff > 0:
            # Positive outliers pulled mean up - bearish
            self.signal = -1
        elif diff < 0:
            # Negative outliers pulled mean down - bullish
            self.signal = 1
        else:
            self.signal = 0


@FormulaRegistry.register(26)
class MedianAbsoluteDeviation(BaseFormula):
    """ID 26: MAD for robust volatility"""
    NAME = "MedianAbsoluteDeviation"
    CATEGORY = "statistical"
    DESCRIPTION = "Robust dispersion measure"

    def _compute(self):
        if len(self.returns) < 20:
            return

        returns = self._returns_array()

        median = np.median(returns)
        mad = np.median(np.abs(returns - median))

        # Robust z-score
        if mad > 0:
            z = (returns[-1] - median) / (mad * 1.4826)
        else:
            z = 0

        self.confidence = min(abs(z) / 2, 1.0)
        self.signal = self._clip_signal(-z, threshold=2.0)


@FormulaRegistry.register(27)
class AnomalyDetection(BaseFormula):
    """ID 27: Statistical Anomaly Detection"""
    NAME = "AnomalyDetection"
    CATEGORY = "statistical"
    DESCRIPTION = "Detect unusual market behavior"

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()

        # Multiple anomaly indicators
        mean = np.mean(returns[:-1])
        std = np.std(returns[:-1])
        current = returns[-1]

        # Z-score
        z = (current - mean) / std if std > 0 else 0

        # Consecutive direction
        signs = np.sign(returns[-5:])
        consecutive = abs(np.sum(signs))

        # Volume spike (using return magnitude as proxy)
        mag_ratio = abs(current) / np.mean(np.abs(returns[:-1]))

        anomaly_score = abs(z) / 3 + consecutive / 5 + (mag_ratio - 1) / 3
        anomaly_score = min(anomaly_score, 1.0)

        self.confidence = anomaly_score

        if anomaly_score > 0.7:
            # High anomaly - expect reversion
            self.signal = -np.sign(current)
        else:
            self.signal = 0


@FormulaRegistry.register(28)
class GradientBoostingScore(BaseFormula):
    """ID 28: Gradient Boosting-like score aggregation"""
    NAME = "GradientBoostingScore"
    CATEGORY = "statistical"
    DESCRIPTION = "Ensemble weak learner combination"

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()

        # Multiple weak signals
        signals = []

        # Signal 1: Recent return sign
        signals.append(np.sign(returns[-1]))

        # Signal 2: 5-period momentum
        signals.append(np.sign(np.sum(returns[-5:])))

        # Signal 3: Above/below mean
        signals.append(np.sign(returns[-1] - np.mean(returns)))

        # Signal 4: Trend direction
        if len(returns) >= 10:
            trend = np.polyfit(range(10), returns[-10:], 1)[0]
            signals.append(np.sign(trend))

        # Aggregate with simple voting
        total_signal = np.sum(signals)
        self.confidence = abs(total_signal) / len(signals)
        self.signal = self._clip_signal(total_signal, threshold=len(signals)/2)


@FormulaRegistry.register(29)
class WinsorizedEstimate(BaseFormula):
    """ID 29: Winsorized Statistics"""
    NAME = "WinsorizedEstimate"
    CATEGORY = "statistical"
    DESCRIPTION = "Bounded extreme value handling"

    def _compute(self):
        if len(self.returns) < 20:
            return

        returns = self._returns_array()

        # 10% Winsorization
        p10 = np.percentile(returns, 10)
        p90 = np.percentile(returns, 90)

        winsorized = np.clip(returns, p10, p90)
        win_mean = np.mean(winsorized)
        win_std = np.std(winsorized)

        # Compare current to Winsorized distribution
        current = returns[-1]
        if win_std > 0:
            z = (current - win_mean) / win_std
        else:
            z = 0

        self.confidence = min(abs(z) / 2, 1.0)
        self.signal = self._clip_signal(-z, threshold=1.5)


@FormulaRegistry.register(30)
class ProbabilityIntegralTransform(BaseFormula):
    """ID 30: PIT for distribution uniformity test"""
    NAME = "ProbabilityIntegralTransform"
    CATEGORY = "statistical"
    DESCRIPTION = "Transform to uniform for calibration"

    def _compute(self):
        if len(self.returns) < 30:
            return

        returns = self._returns_array()

        # Empirical CDF
        sorted_returns = np.sort(returns[:-1])
        current = returns[-1]

        # Find percentile
        rank = np.searchsorted(sorted_returns, current)
        percentile = rank / len(sorted_returns)

        self.confidence = abs(percentile - 0.5) * 2

        if percentile > 0.9:
            self.signal = -1  # Overbought
        elif percentile < 0.1:
            self.signal = 1  # Oversold
        else:
            self.signal = 0


# Export all classes
__all__ = [
    'BayesianProbability',
    'MaximumLikelihood',
    'HawkesProcess',
    'StudentTDistribution',
    'SkewnessKurtosis',
    'EntropyMeasure',
    'CointegrationTest',
    'KLDivergence',
    'AutocorrelationTest',
    'VarianceRatioTest',
    'RunsTest',
    'HurstExponent',
    'MomentEstimation',
    'BootstrapConfidence',
    'CrossSectionalMomentum',
    'TimeSeriesMomentum',
    'ARIMAMomentum',
    'ExponentialSmoothing',
    'GaussianMixture',
    'ParetoTail',
    'CopulaDependence',
    'OrderStatistics',
    'RankCorrelation',
    'QuantileRegression',
    'TruncatedMean',
    'MedianAbsoluteDeviation',
    'AnomalyDetection',
    'GradientBoostingScore',
    'WinsorizedEstimate',
    'ProbabilityIntegralTransform',
]
