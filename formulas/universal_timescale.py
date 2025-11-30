"""
UNIVERSAL TIME-SCALE INVARIANT FORMULAS
========================================
Academic peer-reviewed formulas for ALL timeframe adaptation.

Based on cutting-edge research:
- Guillaume et al. (1997) - Intrinsic Time / Directional Change
- Lyons et al. (2014) - Rough Path Signatures
- Amornbunchornvej et al. (2021) - Variable-lag Granger Causality
- Kantelhardt et al. (2002) - Multifractal DFA
- Hamilton (1989) - Regime Switching Models
- Daubechies (1992) - Wavelet Multi-Resolution

Formula IDs: 501-550
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# ID 501: DIRECTIONAL CHANGE INTRINSIC TIME (Guillaume et al. 1997)
# =============================================================================
class DirectionalChangeIntrinsicTime:
    """
    Event-based time that expands during volatility, contracts during calm.

    Paper: Guillaume et al. (1997) "From the bird's eye to the microscope"
    Finance and Stochastics 1(2): 95-129

    Key insight: Time should be measured by EVENTS, not clock ticks.
    A 1% move in 1 second = same "intrinsic time" as 1% move in 1 hour.
    """

    formula_id = 501

    def __init__(self, thresholds: List[float] = [0.001, 0.002, 0.005, 0.01]):
        """
        thresholds: List of directional change thresholds (e.g., 0.1%, 0.2%, 0.5%, 1%)
        """
        self.thresholds = thresholds
        self.states = {t: {'mode': 'up', 'extreme': 0.0, 'dc_count': 0} for t in thresholds}
        self.last_price = None
        self.dc_events = {t: deque(maxlen=1000) for t in thresholds}

    def update(self, price: float, timestamp: float) -> Dict:
        """
        Returns intrinsic time metrics across all thresholds.
        """
        if self.last_price is None:
            self.last_price = price
            for t in self.thresholds:
                self.states[t]['extreme'] = price
            return {'signal': 0.0, 'dc_rates': {}, 'overshoot': 0.0}

        signals = []
        dc_rates = {}
        overshoots = []

        for threshold in self.thresholds:
            state = self.states[threshold]

            if state['mode'] == 'up':
                # Looking for downturn
                if price >= state['extreme']:
                    state['extreme'] = price  # New high
                elif (state['extreme'] - price) / state['extreme'] >= threshold:
                    # Directional change DOWN detected
                    overshoot = (state['extreme'] - self.last_price) / self.last_price
                    overshoots.append(overshoot)
                    state['mode'] = 'down'
                    state['extreme'] = price
                    state['dc_count'] += 1
                    self.dc_events[threshold].append((timestamp, -1, overshoot))
                    signals.append(-1)
            else:
                # Looking for upturn
                if price <= state['extreme']:
                    state['extreme'] = price  # New low
                elif (price - state['extreme']) / state['extreme'] >= threshold:
                    # Directional change UP detected
                    overshoot = (self.last_price - state['extreme']) / state['extreme']
                    overshoots.append(overshoot)
                    state['mode'] = 'up'
                    state['extreme'] = price
                    state['dc_count'] += 1
                    self.dc_events[threshold].append((timestamp, 1, overshoot))
                    signals.append(1)

            # Calculate DC rate (events per unit time)
            events = self.dc_events[threshold]
            if len(events) >= 2:
                time_span = events[-1][0] - events[0][0]
                if time_span > 0:
                    dc_rates[threshold] = len(events) / time_span

        self.last_price = price

        # Aggregate signal across scales
        if signals:
            # Weight by threshold (smaller = faster reaction)
            weights = [1.0 / t for t in self.thresholds[:len(signals)]]
            signal = np.average(signals, weights=weights[:len(signals)])
        else:
            signal = 0.0

        return {
            'signal': signal,
            'dc_rates': dc_rates,
            'overshoot': np.mean(overshoots) if overshoots else 0.0,
            'intrinsic_volatility': sum(dc_rates.values()) if dc_rates else 0.0
        }


# =============================================================================
# ID 502: PATH SIGNATURE TRADING (Lyons et al. 2014)
# =============================================================================
class PathSignatureTrading:
    """
    Rough path signatures for time-invariant feature extraction.

    Paper: Lyons et al. (2014) "Extracting information from the signature"
    arXiv:1307.7244

    Key insight: Signature captures path shape regardless of speed/timing.
    """

    formula_id = 502

    def __init__(self, depth: int = 3, window: int = 50):
        """
        depth: Signature truncation depth (2-4 typical)
        window: Lookback window for path
        """
        self.depth = depth
        self.window = window
        self.price_path = deque(maxlen=window)
        self.time_path = deque(maxlen=window)

    def _compute_signature(self, path: np.ndarray) -> np.ndarray:
        """
        Compute truncated signature of a path.
        Signature = (1, integral of dX, integral of X dX, ...)
        """
        if len(path) < 2:
            return np.array([1.0])

        # Normalize path
        path = (path - path[0]) / (np.std(path) + 1e-10)

        # Level 1: Simple integral
        sig = [1.0]

        # Level 1 signature: sum of increments
        increments = np.diff(path)
        sig.append(np.sum(increments))

        if self.depth >= 2:
            # Level 2: Iterated integrals (area)
            # Integral of X_s dX_s
            area = 0.0
            cumsum = 0.0
            for i, inc in enumerate(increments):
                area += cumsum * inc
                cumsum += inc
            sig.append(area)

            # Quadratic variation approximation
            sig.append(np.sum(increments ** 2))

        if self.depth >= 3:
            # Level 3: Third order iterated integrals
            # Approximation using running sums
            triple = 0.0
            cumsum = 0.0
            cumarea = 0.0
            for inc in increments:
                triple += cumarea * inc
                cumarea += cumsum * inc
                cumsum += inc
            sig.append(triple)

        return np.array(sig)

    def update(self, price: float, timestamp: float) -> Dict:
        """
        Returns signature-based trading signal.
        """
        self.price_path.append(price)
        self.time_path.append(timestamp)

        if len(self.price_path) < 10:
            return {'signal': 0.0, 'signature': [], 'path_roughness': 0.0}

        path = np.array(self.price_path)
        signature = self._compute_signature(path)

        # Trading signal from signature components
        # Positive area (sig[2]) suggests uptrend momentum
        # Negative area suggests downtrend
        if len(signature) >= 3:
            # Normalize by quadratic variation
            qv = signature[3] if len(signature) > 3 else np.sum(np.diff(path)**2)
            if qv > 0:
                signal = signature[2] / np.sqrt(qv)
            else:
                signal = 0.0
        else:
            signal = signature[1] if len(signature) > 1 else 0.0

        # Clamp signal
        signal = np.clip(signal, -1.0, 1.0)

        # Path roughness (Hurst-like measure from signature)
        if len(signature) > 3:
            roughness = np.abs(signature[3]) / (len(path) * np.var(np.diff(path)) + 1e-10)
        else:
            roughness = 0.5

        return {
            'signal': signal,
            'signature': signature.tolist(),
            'path_roughness': roughness,
            'path_length': len(path)
        }


# =============================================================================
# ID 503: VARIABLE-LAG GRANGER CAUSALITY (Amornbunchornvej et al. 2021)
# =============================================================================
class VariableLagCausality:
    """
    Detect causal relationships with time-varying lags using DTW.

    Paper: Amornbunchornvej et al. (2021) ACM TKDD 15(4)
    "Variable-lag Granger Causality and Transfer Entropy"

    Key insight: Lag between cause and effect changes dynamically.
    """

    formula_id = 503

    def __init__(self, max_lag: int = 20, window: int = 100):
        self.max_lag = max_lag
        self.window = window
        self.price_history = deque(maxlen=window)
        self.volume_history = deque(maxlen=window)
        self.fee_history = deque(maxlen=window)

    def _dtw_distance(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, List]:
        """
        Dynamic Time Warping with path recovery.
        Returns distance and optimal warping path.
        """
        n, m = len(x), len(y)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(x[i-1] - y[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )

        # Backtrack to find path
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            candidates = [
                (dtw_matrix[i-1, j], (i-1, j)),
                (dtw_matrix[i, j-1], (i, j-1)),
                (dtw_matrix[i-1, j-1], (i-1, j-1))
            ]
            _, (i, j) = min(candidates)

        return dtw_matrix[n, m], path[::-1]

    def _variable_lag_correlation(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Compute correlation with variable lag using DTW alignment.
        """
        if len(x) < 5 or len(y) < 5:
            return {'correlation': 0.0, 'avg_lag': 0, 'causality': 0.0}

        # Normalize
        x_norm = (x - np.mean(x)) / (np.std(x) + 1e-10)
        y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)

        dist, path = self._dtw_distance(x_norm, y_norm)

        # Compute variable lags from path
        lags = [j - i for i, j in path]
        avg_lag = np.mean(lags)

        # Correlation along warped path
        x_warped = [x_norm[i] for i, j in path]
        y_warped = [y_norm[j] for i, j in path]
        correlation = np.corrcoef(x_warped, y_warped)[0, 1]

        # Granger causality approximation
        # Positive lag means x leads y
        causality = np.sign(avg_lag) * abs(correlation)

        return {
            'correlation': correlation,
            'avg_lag': avg_lag,
            'causality': causality,
            'dtw_distance': dist
        }

    def update(self, price: float, volume: float, fee: float) -> Dict:
        """
        Detect causal relationships between blockchain metrics and price.
        """
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.fee_history.append(fee)

        if len(self.price_history) < 20:
            return {'signal': 0.0, 'volume_leads': 0.0, 'fee_leads': 0.0}

        prices = np.array(self.price_history)
        volumes = np.array(self.volume_history)
        fees = np.array(self.fee_history)

        # Price returns
        returns = np.diff(np.log(prices + 1e-10))
        vol_changes = np.diff(np.log(volumes + 1e-10))
        fee_changes = np.diff(fees)

        # Check if volume leads price
        vol_result = self._variable_lag_correlation(vol_changes[:-1], returns[1:])

        # Check if fees lead price
        fee_result = self._variable_lag_correlation(fee_changes[:-1], returns[1:])

        # Combine signals
        # If volume leads with positive correlation, expect continuation
        signal = 0.0
        if vol_result['avg_lag'] > 0 and vol_result['correlation'] > 0.3:
            signal += 0.5 * np.sign(vol_changes[-1])
        if fee_result['avg_lag'] > 0 and fee_result['correlation'] > 0.3:
            signal += 0.3 * np.sign(fee_changes[-1])

        return {
            'signal': np.clip(signal, -1.0, 1.0),
            'volume_leads': vol_result['causality'],
            'fee_leads': fee_result['causality'],
            'volume_lag': vol_result['avg_lag'],
            'fee_lag': fee_result['avg_lag']
        }


# =============================================================================
# ID 504: MULTIFRACTAL DFA TRADING (Kantelhardt et al. 2002)
# =============================================================================
class MultifractalDFATrading:
    """
    Trade based on multifractal structure of price series.

    Paper: Kantelhardt et al. (2002) "Multifractal detrended fluctuation analysis"
    Physica A 316: 87-114

    Key insight: Markets have different fractal properties at different scales.
    """

    formula_id = 504

    def __init__(self, window: int = 256, scales: List[int] = None):
        self.window = window
        self.scales = scales or [4, 8, 16, 32, 64]
        self.price_history = deque(maxlen=window)
        self.hurst_history = deque(maxlen=50)

    def _dfa(self, x: np.ndarray, scale: int) -> float:
        """
        Detrended Fluctuation Analysis for a single scale.
        Returns fluctuation function F(s).
        """
        n = len(x)
        if n < scale:
            return np.nan

        # Cumulative sum (profile)
        y = np.cumsum(x - np.mean(x))

        # Divide into segments
        n_segments = n // scale
        if n_segments < 1:
            return np.nan

        fluctuations = []

        for i in range(n_segments):
            segment = y[i*scale:(i+1)*scale]
            # Linear detrend
            t = np.arange(scale)
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            detrended = segment - trend
            fluctuations.append(np.sqrt(np.mean(detrended**2)))

        return np.mean(fluctuations)

    def _compute_hurst(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute Hurst exponent from DFA across scales.
        """
        fluctuations = []
        valid_scales = []

        for s in self.scales:
            f = self._dfa(x, s)
            if not np.isnan(f) and f > 0:
                fluctuations.append(f)
                valid_scales.append(s)

        if len(valid_scales) < 2:
            return 0.5, np.array([])

        # Log-log regression
        log_s = np.log(valid_scales)
        log_f = np.log(fluctuations)

        coeffs = np.polyfit(log_s, log_f, 1)
        hurst = coeffs[0]  # Slope is Hurst exponent

        return hurst, np.array(fluctuations)

    def _mfdfa(self, x: np.ndarray, q_values: List[float] = [-2, -1, 0, 1, 2]) -> Dict:
        """
        Multifractal DFA - compute generalized Hurst for different q values.
        """
        hurst_q = {}

        for q in q_values:
            fluctuations = []
            valid_scales = []

            for s in self.scales:
                f = self._dfa(x, s)
                if not np.isnan(f) and f > 0:
                    if q == 0:
                        fluctuations.append(np.log(f))
                    else:
                        fluctuations.append(f ** q)
                    valid_scales.append(s)

            if len(valid_scales) >= 2:
                log_s = np.log(valid_scales)
                if q == 0:
                    log_f = fluctuations
                else:
                    log_f = np.log(np.array(fluctuations) ** (1/q))

                coeffs = np.polyfit(log_s, log_f, 1)
                hurst_q[q] = coeffs[0]

        return hurst_q

    def update(self, price: float) -> Dict:
        """
        Returns multifractal-based trading signal.
        """
        self.price_history.append(price)

        if len(self.price_history) < self.window // 2:
            return {'signal': 0.0, 'hurst': 0.5, 'multifractal_width': 0.0}

        prices = np.array(self.price_history)
        returns = np.diff(np.log(prices + 1e-10))

        if len(returns) < 50:
            return {'signal': 0.0, 'hurst': 0.5, 'multifractal_width': 0.0}

        # Compute Hurst exponent
        hurst, _ = self._compute_hurst(returns)
        self.hurst_history.append(hurst)

        # Compute multifractal spectrum
        hurst_q = self._mfdfa(returns)

        # Multifractal width (difference between H(2) and H(-2))
        if 2 in hurst_q and -2 in hurst_q:
            mf_width = hurst_q[-2] - hurst_q[2]
        else:
            mf_width = 0.0

        # Trading signal based on Hurst
        # H > 0.5: trending (momentum)
        # H < 0.5: mean-reverting
        # H = 0.5: random walk

        if len(self.hurst_history) >= 2:
            hurst_trend = hurst - np.mean(list(self.hurst_history)[:-1])
        else:
            hurst_trend = 0

        # Price momentum
        momentum = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0

        if hurst > 0.55:
            # Trending: follow momentum
            signal = np.sign(momentum) * min(1.0, (hurst - 0.5) * 4)
        elif hurst < 0.45:
            # Mean-reverting: fade momentum
            signal = -np.sign(momentum) * min(1.0, (0.5 - hurst) * 4)
        else:
            # Random walk: neutral
            signal = 0.0

        return {
            'signal': signal,
            'hurst': hurst,
            'hurst_trend': hurst_trend,
            'multifractal_width': mf_width,
            'regime': 'trending' if hurst > 0.55 else ('mean_revert' if hurst < 0.45 else 'random')
        }


# =============================================================================
# ID 505: CONTINUOUS REGIME SWITCHING SDE (Hamilton 1989)
# =============================================================================
class ContinuousRegimeSwitching:
    """
    Continuous-time regime switching with stochastic differential equations.

    Paper: Hamilton (1989) "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle"
    Econometrica 57(2): 357-384

    Key insight: Regime probabilities evolve continuously, not discretely.
    """

    formula_id = 505

    def __init__(self, n_regimes: int = 3, decay: float = 0.95):
        """
        n_regimes: Number of hidden states (e.g., bull/neutral/bear)
        decay: Probability decay factor for smooth transitions
        """
        self.n_regimes = n_regimes
        self.decay = decay

        # Initialize uniform probabilities
        self.regime_probs = np.ones(n_regimes) / n_regimes

        # Regime parameters (mean, volatility)
        # Will be learned online
        self.regime_means = np.array([-0.001, 0.0, 0.001])  # Bear, Neutral, Bull
        self.regime_vols = np.array([0.02, 0.01, 0.015])

        # Transition intensity matrix (continuous-time)
        self.intensity = np.array([
            [-0.1, 0.05, 0.05],   # From bear
            [0.05, -0.1, 0.05],   # From neutral
            [0.05, 0.05, -0.1]    # From bull
        ])

        self.returns_history = deque(maxlen=100)

    def _emission_prob(self, return_val: float, regime: int) -> float:
        """
        Probability of observing return given regime (Gaussian emission).
        """
        mu = self.regime_means[regime]
        sigma = self.regime_vols[regime]
        return np.exp(-0.5 * ((return_val - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    def _update_regime_probs(self, return_val: float, dt: float = 1.0):
        """
        Update regime probabilities using continuous-time Bayes filter.
        """
        # Prediction step (transition)
        # P(s_t | s_{t-1}) using matrix exponential approximation
        trans_matrix = np.eye(self.n_regimes) + self.intensity * dt
        trans_matrix = np.maximum(trans_matrix, 0)  # Ensure non-negative
        trans_matrix /= trans_matrix.sum(axis=1, keepdims=True)  # Normalize rows

        predicted_probs = trans_matrix.T @ self.regime_probs

        # Update step (observation)
        likelihoods = np.array([self._emission_prob(return_val, i) for i in range(self.n_regimes)])

        # Avoid numerical issues
        likelihoods = np.maximum(likelihoods, 1e-10)

        # Bayes update
        updated_probs = predicted_probs * likelihoods
        updated_probs /= updated_probs.sum()

        # Smooth update with decay
        self.regime_probs = self.decay * self.regime_probs + (1 - self.decay) * updated_probs
        self.regime_probs /= self.regime_probs.sum()

    def _update_parameters(self):
        """
        Online update of regime parameters using EM-style updates.
        """
        if len(self.returns_history) < 20:
            return

        returns = np.array(self.returns_history)

        # Simple percentile-based regime estimation
        low_thresh = np.percentile(returns, 33)
        high_thresh = np.percentile(returns, 67)

        bear_returns = returns[returns < low_thresh]
        neutral_returns = returns[(returns >= low_thresh) & (returns <= high_thresh)]
        bull_returns = returns[returns > high_thresh]

        if len(bear_returns) > 0:
            self.regime_means[0] = 0.9 * self.regime_means[0] + 0.1 * np.mean(bear_returns)
            self.regime_vols[0] = 0.9 * self.regime_vols[0] + 0.1 * np.std(bear_returns)
        if len(neutral_returns) > 0:
            self.regime_means[1] = 0.9 * self.regime_means[1] + 0.1 * np.mean(neutral_returns)
            self.regime_vols[1] = 0.9 * self.regime_vols[1] + 0.1 * np.std(neutral_returns)
        if len(bull_returns) > 0:
            self.regime_means[2] = 0.9 * self.regime_means[2] + 0.1 * np.mean(bull_returns)
            self.regime_vols[2] = 0.9 * self.regime_vols[2] + 0.1 * np.std(bull_returns)

    def update(self, price: float, last_price: float) -> Dict:
        """
        Update regime probabilities and return trading signal.
        """
        if last_price <= 0:
            return {'signal': 0.0, 'regime_probs': self.regime_probs.tolist(), 'regime': 1}

        return_val = np.log(price / last_price)
        self.returns_history.append(return_val)

        self._update_regime_probs(return_val)
        self._update_parameters()

        # Most likely regime
        regime = np.argmax(self.regime_probs)

        # Trading signal based on regime probabilities
        # Bull probability - Bear probability
        signal = self.regime_probs[2] - self.regime_probs[0]

        # Confidence: how peaked is the distribution?
        confidence = np.max(self.regime_probs) - 1.0 / self.n_regimes

        return {
            'signal': signal,
            'regime_probs': self.regime_probs.tolist(),
            'regime': regime,
            'regime_name': ['bear', 'neutral', 'bull'][regime],
            'confidence': confidence,
            'regime_means': self.regime_means.tolist(),
            'regime_vols': self.regime_vols.tolist()
        }


# =============================================================================
# ID 506: WAVELET MULTI-RESOLUTION FUSION (Daubechies 1992)
# =============================================================================
class WaveletMultiResolutionFusion:
    """
    Fuse trading signals across multiple time scales using wavelets.

    Based on: Daubechies (1992) "Ten Lectures on Wavelets"
    And: Gencay et al. (2001) "An Introduction to Wavelets and Other Filtering Methods"

    Key insight: Different scales contain different information.
    """

    formula_id = 506

    def __init__(self, levels: int = 4, window: int = 128):
        """
        levels: Number of wavelet decomposition levels
        window: Size of data window (should be power of 2)
        """
        self.levels = levels
        self.window = window
        self.price_history = deque(maxlen=window)

        # Haar wavelet coefficients (simplest orthogonal wavelet)
        self.h = np.array([1, 1]) / np.sqrt(2)  # Low-pass
        self.g = np.array([1, -1]) / np.sqrt(2)  # High-pass

        # Scale weights (learned or fixed)
        self.scale_weights = np.array([0.4, 0.3, 0.2, 0.1])  # Favor short-term

    def _haar_decompose(self, x: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Perform Haar wavelet decomposition.
        Returns (details, approximation).
        """
        details = []
        approx = x.copy()

        for level in range(self.levels):
            if len(approx) < 2:
                break

            n = len(approx)
            # Ensure even length
            if n % 2 == 1:
                approx = approx[:-1]
                n -= 1

            # Decompose
            new_approx = np.zeros(n // 2)
            detail = np.zeros(n // 2)

            for i in range(n // 2):
                new_approx[i] = (approx[2*i] + approx[2*i + 1]) / np.sqrt(2)
                detail[i] = (approx[2*i] - approx[2*i + 1]) / np.sqrt(2)

            details.append(detail)
            approx = new_approx

        return details, approx

    def _scale_signal(self, detail: np.ndarray, level: int) -> float:
        """
        Extract trading signal from wavelet detail coefficients at a scale.
        """
        if len(detail) == 0:
            return 0.0

        # Recent trend in detail coefficients
        if len(detail) >= 3:
            trend = np.mean(detail[-3:]) - np.mean(detail[:-3]) if len(detail) > 3 else 0
        else:
            trend = np.mean(detail)

        # Energy at this scale
        energy = np.sqrt(np.mean(detail ** 2))

        # Signal = trend normalized by energy
        if energy > 0:
            signal = trend / energy
        else:
            signal = 0.0

        return np.clip(signal, -1.0, 1.0)

    def update(self, price: float) -> Dict:
        """
        Returns multi-scale wavelet-based trading signal.
        """
        self.price_history.append(price)

        if len(self.price_history) < 16:
            return {'signal': 0.0, 'scale_signals': [], 'dominant_scale': 0}

        prices = np.array(self.price_history)
        returns = np.diff(np.log(prices + 1e-10))

        # Wavelet decomposition
        details, approx = self._haar_decompose(returns)

        # Extract signal at each scale
        scale_signals = []
        for level, detail in enumerate(details):
            sig = self._scale_signal(detail, level)
            scale_signals.append(sig)

        # Pad if needed
        while len(scale_signals) < len(self.scale_weights):
            scale_signals.append(0.0)

        # Weighted fusion of signals
        weights = self.scale_weights[:len(scale_signals)]
        weights = weights / weights.sum()

        fused_signal = np.sum(np.array(scale_signals[:len(weights)]) * weights)

        # Find dominant scale (highest absolute signal)
        if scale_signals:
            dominant_scale = np.argmax(np.abs(scale_signals))
        else:
            dominant_scale = 0

        # Scale energies for regime detection
        scale_energies = [np.sqrt(np.mean(d ** 2)) if len(d) > 0 else 0 for d in details]

        return {
            'signal': fused_signal,
            'scale_signals': scale_signals,
            'dominant_scale': dominant_scale,
            'scale_energies': scale_energies,
            'trend_approx': approx[-1] if len(approx) > 0 else 0
        }


# =============================================================================
# ID 507: RECURSIVE BAYESIAN ADAPTIVE PARAMETERS
# =============================================================================
class RecursiveBayesianAdaptive:
    """
    Online Bayesian learning of trading parameters.

    Based on: Kalman (1960) and Bayesian adaptive trading literature

    Key insight: Parameters should update with each new observation.
    """

    formula_id = 507

    def __init__(self, n_params: int = 3):
        """
        n_params: Number of adaptive parameters
        """
        self.n_params = n_params

        # State estimate (parameters)
        self.theta = np.zeros(n_params)  # [momentum_weight, mean_reversion_weight, volatility_scale]

        # Covariance (uncertainty)
        self.P = np.eye(n_params) * 1.0

        # Process noise
        self.Q = np.eye(n_params) * 0.001

        # Measurement noise
        self.R = 0.1

        # History for feature computation
        self.price_history = deque(maxlen=100)
        self.return_history = deque(maxlen=100)

    def _compute_features(self) -> np.ndarray:
        """
        Compute feature vector for prediction.
        """
        if len(self.return_history) < 10:
            return np.zeros(self.n_params)

        returns = np.array(self.return_history)

        # Feature 1: Momentum (sign of recent returns)
        momentum = np.mean(returns[-5:])

        # Feature 2: Mean reversion signal (deviation from mean)
        deviation = returns[-1] - np.mean(returns)

        # Feature 3: Volatility (recent vs long-term)
        recent_vol = np.std(returns[-10:])
        long_vol = np.std(returns)
        vol_ratio = recent_vol / (long_vol + 1e-10) - 1

        return np.array([momentum, deviation, vol_ratio])

    def _predict_return(self, features: np.ndarray) -> float:
        """
        Predict next return using current parameters.
        """
        return np.dot(self.theta, features)

    def update(self, price: float) -> Dict:
        """
        Update parameters and return adaptive trading signal.
        """
        self.price_history.append(price)

        if len(self.price_history) >= 2:
            ret = np.log(price / self.price_history[-2])
            self.return_history.append(ret)

        if len(self.return_history) < 10:
            return {'signal': 0.0, 'params': self.theta.tolist(), 'uncertainty': np.diag(self.P).tolist()}

        # Compute features from previous data
        features = self._compute_features()

        # Predict
        predicted_return = self._predict_return(features)

        # Actual return (most recent)
        actual_return = self.return_history[-1]

        # Kalman update
        # Innovation
        innovation = actual_return - predicted_return

        # Kalman gain
        H = features.reshape(1, -1)  # Measurement matrix
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        K = self.P @ H.T / S  # Kalman gain

        # Update state
        self.theta = self.theta + K.flatten() * innovation

        # Update covariance
        self.P = (np.eye(self.n_params) - K @ H) @ self.P + self.Q

        # Trading signal from prediction
        signal = np.tanh(predicted_return * 100)  # Scale and bound

        # Confidence from parameter uncertainty
        confidence = 1.0 / (1.0 + np.trace(self.P))

        return {
            'signal': signal * confidence,
            'raw_signal': signal,
            'params': self.theta.tolist(),
            'param_names': ['momentum', 'mean_reversion', 'volatility'],
            'uncertainty': np.diag(self.P).tolist(),
            'confidence': confidence,
            'prediction': predicted_return,
            'innovation': innovation
        }


# =============================================================================
# ID 508: MASTER UNIVERSAL TIMESCALE CONTROLLER
# =============================================================================
class UniversalTimescaleController:
    """
    Master controller that combines ALL timescale-invariant formulas.

    This is the ULTIMATE adaptive system that works at ANY timeframe.
    """

    formula_id = 508

    def __init__(self):
        # Initialize all sub-formulas
        self.dc_intrinsic = DirectionalChangeIntrinsicTime()
        self.path_sig = PathSignatureTrading()
        self.var_lag = VariableLagCausality()
        self.mfdfa = MultifractalDFATrading()
        self.regime = ContinuousRegimeSwitching()
        self.wavelet = WaveletMultiResolutionFusion()
        self.bayesian = RecursiveBayesianAdaptive()

        self.last_price = None
        self.timestamp = 0

        # Formula weights (can be adapted)
        self.weights = {
            'dc': 0.15,
            'signature': 0.15,
            'causality': 0.10,
            'mfdfa': 0.20,
            'regime': 0.15,
            'wavelet': 0.15,
            'bayesian': 0.10
        }

    def update(self, price: float, volume: float = 0, fee: float = 0, timestamp: float = None) -> Dict:
        """
        Master update combining all timescale-invariant signals.
        """
        if timestamp is None:
            self.timestamp += 1
            timestamp = self.timestamp

        results = {}
        signals = {}

        # 1. Directional Change Intrinsic Time
        dc_result = self.dc_intrinsic.update(price, timestamp)
        results['dc'] = dc_result
        signals['dc'] = dc_result['signal']

        # 2. Path Signature
        sig_result = self.path_sig.update(price, timestamp)
        results['signature'] = sig_result
        signals['signature'] = sig_result['signal']

        # 3. Variable-lag Causality
        if volume > 0 or fee > 0:
            cause_result = self.var_lag.update(price, volume, fee)
            results['causality'] = cause_result
            signals['causality'] = cause_result['signal']
        else:
            signals['causality'] = 0.0

        # 4. Multifractal DFA
        mfdfa_result = self.mfdfa.update(price)
        results['mfdfa'] = mfdfa_result
        signals['mfdfa'] = mfdfa_result['signal']

        # 5. Continuous Regime Switching
        if self.last_price is not None:
            regime_result = self.regime.update(price, self.last_price)
            results['regime'] = regime_result
            signals['regime'] = regime_result['signal']
        else:
            signals['regime'] = 0.0

        # 6. Wavelet Multi-Resolution
        wavelet_result = self.wavelet.update(price)
        results['wavelet'] = wavelet_result
        signals['wavelet'] = wavelet_result['signal']

        # 7. Recursive Bayesian
        bayes_result = self.bayesian.update(price)
        results['bayesian'] = bayes_result
        signals['bayesian'] = bayes_result['signal']

        self.last_price = price

        # Weighted combination of all signals
        final_signal = sum(signals[k] * self.weights[k] for k in signals)

        # Confidence from agreement
        signal_values = list(signals.values())
        agreement = 1.0 - np.std(signal_values) if signal_values else 0.0

        return {
            'signal': final_signal,
            'confidence': agreement,
            'component_signals': signals,
            'hurst': mfdfa_result.get('hurst', 0.5),
            'regime': results.get('regime', {}).get('regime_name', 'unknown'),
            'dominant_scale': wavelet_result.get('dominant_scale', 0),
            'intrinsic_volatility': dc_result.get('intrinsic_volatility', 0),
            'full_results': results
        }


# =============================================================================
# FORMULA WRAPPERS FOR REGISTRY COMPATIBILITY
# =============================================================================
from .base import BaseFormula, FORMULA_REGISTRY


class DCIntrinsicTimeFormula(BaseFormula):
    """ID 501: Directional Change Intrinsic Time"""
    formula_id = 501
    name = "DirectionalChangeIntrinsicTime"

    def __init__(self):
        super().__init__()
        self.dc = DirectionalChangeIntrinsicTime()
        self.last_result = {}

    def update(self, price: float, volume: float = 0, timestamp: float = 0):
        self.last_result = self.dc.update(price, timestamp)

    def get_signal(self) -> float:
        return self.last_result.get('signal', 0.0)

    def get_confidence(self) -> float:
        iv = self.last_result.get('intrinsic_volatility', 0)
        return min(1.0, iv / 10) if iv else 0.5


class PathSignatureFormula(BaseFormula):
    """ID 502: Path Signature Trading"""
    formula_id = 502
    name = "PathSignatureTrading"

    def __init__(self):
        super().__init__()
        self.ps = PathSignatureTrading()
        self.last_result = {}

    def update(self, price: float, volume: float = 0, timestamp: float = 0):
        self.last_result = self.ps.update(price, timestamp)

    def get_signal(self) -> float:
        return self.last_result.get('signal', 0.0)

    def get_confidence(self) -> float:
        return 1.0 - self.last_result.get('path_roughness', 0.5)


class VariableLagFormula(BaseFormula):
    """ID 503: Variable-Lag Granger Causality"""
    formula_id = 503
    name = "VariableLagCausality"

    def __init__(self):
        super().__init__()
        self.vlc = VariableLagCausality()
        self.last_result = {}

    def update(self, price: float, volume: float = 0, timestamp: float = 0):
        self.last_result = self.vlc.update(price, volume, volume * 0.001)

    def get_signal(self) -> float:
        return self.last_result.get('signal', 0.0)

    def get_confidence(self) -> float:
        vl = abs(self.last_result.get('volume_leads', 0))
        fl = abs(self.last_result.get('fee_leads', 0))
        return min(1.0, (vl + fl) / 2)


class MFDFAFormula(BaseFormula):
    """ID 504: Multifractal DFA Trading"""
    formula_id = 504
    name = "MultifractalDFATrading"

    def __init__(self):
        super().__init__()
        self.mfdfa = MultifractalDFATrading()
        self.last_result = {}

    def update(self, price: float, volume: float = 0, timestamp: float = 0):
        self.last_result = self.mfdfa.update(price)

    def get_signal(self) -> float:
        return self.last_result.get('signal', 0.0)

    def get_confidence(self) -> float:
        hurst = self.last_result.get('hurst', 0.5)
        return abs(hurst - 0.5) * 2  # Higher confidence when far from 0.5


class ContinuousRegimeFormula(BaseFormula):
    """ID 505: Continuous Regime Switching"""
    formula_id = 505
    name = "ContinuousRegimeSwitching"

    def __init__(self):
        super().__init__()
        self.crs = ContinuousRegimeSwitching()
        self.last_result = {}
        self.last_price = None

    def update(self, price: float, volume: float = 0, timestamp: float = 0):
        if self.last_price is not None:
            self.last_result = self.crs.update(price, self.last_price)
        self.last_price = price

    def get_signal(self) -> float:
        return self.last_result.get('signal', 0.0)

    def get_confidence(self) -> float:
        return self.last_result.get('confidence', 0.5)


class WaveletFusionFormula(BaseFormula):
    """ID 506: Wavelet Multi-Resolution Fusion"""
    formula_id = 506
    name = "WaveletMultiResolutionFusion"

    def __init__(self):
        super().__init__()
        self.wmf = WaveletMultiResolutionFusion()
        self.last_result = {}

    def update(self, price: float, volume: float = 0, timestamp: float = 0):
        self.last_result = self.wmf.update(price)

    def get_signal(self) -> float:
        return self.last_result.get('signal', 0.0)

    def get_confidence(self) -> float:
        energies = self.last_result.get('scale_energies', [])
        return min(1.0, sum(energies)) if energies else 0.5


class BayesianAdaptiveFormula(BaseFormula):
    """ID 507: Recursive Bayesian Adaptive"""
    formula_id = 507
    name = "RecursiveBayesianAdaptive"

    def __init__(self):
        super().__init__()
        self.rba = RecursiveBayesianAdaptive()
        self.last_result = {}

    def update(self, price: float, volume: float = 0, timestamp: float = 0):
        self.last_result = self.rba.update(price)

    def get_signal(self) -> float:
        return self.last_result.get('signal', 0.0)

    def get_confidence(self) -> float:
        return self.last_result.get('confidence', 0.5)


class UniversalTimescaleFormula(BaseFormula):
    """ID 508: MASTER Universal Timescale Controller"""
    formula_id = 508
    name = "UniversalTimescaleController"

    def __init__(self):
        super().__init__()
        self.utc = UniversalTimescaleController()
        self.last_result = {}

    def update(self, price: float, volume: float = 0, timestamp: float = 0):
        self.last_result = self.utc.update(price, volume, volume * 0.001, timestamp)

    def get_signal(self) -> float:
        return self.last_result.get('signal', 0.0)

    def get_confidence(self) -> float:
        return self.last_result.get('confidence', 0.5)


# =============================================================================
# REGISTER ALL FORMULAS
# =============================================================================
FORMULA_REGISTRY[501] = DCIntrinsicTimeFormula
FORMULA_REGISTRY[502] = PathSignatureFormula
FORMULA_REGISTRY[503] = VariableLagFormula
FORMULA_REGISTRY[504] = MFDFAFormula
FORMULA_REGISTRY[505] = ContinuousRegimeFormula
FORMULA_REGISTRY[506] = WaveletFusionFormula
FORMULA_REGISTRY[507] = BayesianAdaptiveFormula
FORMULA_REGISTRY[508] = UniversalTimescaleFormula

print(f"[UniversalTimescale] Registered 8 time-scale invariant formulas (501-508)")
