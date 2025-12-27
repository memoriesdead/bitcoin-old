"""
GARCH Volatility Signals
========================

Formula IDs: 72061-72065

GARCH models capture volatility clustering - high vol tends to
follow high vol, low vol follows low vol.

RenTech insight: Volatility is more predictable than price.
Trade the volatility pattern, not the price.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class VolatilityForecast:
    """GARCH volatility forecast."""
    current_vol: float
    forecast_vol: float
    vol_regime: str  # 'low', 'medium', 'high'
    vol_trend: str  # 'rising', 'falling', 'stable'


@dataclass
class ConditionalVolatility:
    """Conditional volatility estimate."""
    sigma: float
    h_next: float  # Next period variance forecast
    standardized_return: float


@dataclass
class GARCHSignal:
    """Signal from GARCH analysis."""
    direction: int
    confidence: float
    vol_forecast: float
    vol_regime: str
    signal_type: str


class GARCHModel:
    """
    Simple GARCH(1,1) implementation.

    h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}

    Where h_t is variance, r_t is return.
    """

    def __init__(self, omega: float = 0.00001, alpha: float = 0.1, beta: float = 0.85):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.h: List[float] = []  # Variance history
        self.long_run_var: float = omega / (1 - alpha - beta) if alpha + beta < 1 else 0.01

    def fit(self, returns: np.ndarray, method: str = 'simple'):
        """
        Fit GARCH parameters (simplified MLE).
        """
        if method == 'simple':
            # Use sample variance for omega, fixed alpha/beta
            var = np.var(returns)
            self.omega = var * (1 - self.alpha - self.beta)
            self.long_run_var = var
        else:
            # Grid search for alpha, beta (simplified)
            best_ll = -np.inf
            for alpha in np.arange(0.05, 0.25, 0.05):
                for beta in np.arange(0.7, 0.95, 0.05):
                    if alpha + beta < 1:
                        omega = np.var(returns) * (1 - alpha - beta)
                        ll = self._log_likelihood(returns, omega, alpha, beta)
                        if ll > best_ll:
                            best_ll = ll
                            self.omega = omega
                            self.alpha = alpha
                            self.beta = beta

            self.long_run_var = self.omega / (1 - self.alpha - self.beta)

        # Initialize variance history
        self.h = [self.long_run_var]
        for r in returns:
            h_next = self.omega + self.alpha * r ** 2 + self.beta * self.h[-1]
            self.h.append(h_next)

        return self

    def _log_likelihood(self, returns: np.ndarray, omega: float,
                        alpha: float, beta: float) -> float:
        """Compute log-likelihood."""
        h = omega / (1 - alpha - beta)
        ll = 0

        for r in returns:
            ll += -0.5 * (np.log(h + 1e-10) + r ** 2 / (h + 1e-10))
            h = omega + alpha * r ** 2 + beta * h

        return ll

    def update(self, return_t: float) -> float:
        """Update variance estimate with new return."""
        if not self.h:
            self.h = [self.long_run_var]

        h_next = self.omega + self.alpha * return_t ** 2 + self.beta * self.h[-1]
        self.h.append(h_next)
        return np.sqrt(h_next)

    def forecast(self, n_periods: int = 1) -> np.ndarray:
        """Forecast variance n periods ahead."""
        if not self.h:
            return np.array([self.long_run_var] * n_periods)

        forecasts = []
        h = self.h[-1]

        for _ in range(n_periods):
            h = self.omega + (self.alpha + self.beta) * h
            forecasts.append(h)

        return np.sqrt(np.array(forecasts))

    def get_current_vol(self) -> float:
        """Get current volatility estimate."""
        if not self.h:
            return np.sqrt(self.long_run_var)
        return np.sqrt(self.h[-1])


# =============================================================================
# FORMULA IMPLEMENTATIONS (72061-72065)
# =============================================================================

class GARCHBreakoutSignal:
    """
    Formula 72061: GARCH Breakout Signal

    Trades breakouts only when volatility is expanding.
    Vol expansion = real breakout, vol contraction = false breakout.
    """

    FORMULA_ID = 72061

    def __init__(self):
        self.garch = GARCHModel()
        self.vol_history: List[float] = []

    def fit(self, returns: np.ndarray):
        self.garch.fit(returns)

    def generate_signal(self, returns: np.ndarray, prices: np.ndarray = None) -> GARCHSignal:
        if len(returns) < 20:
            return GARCHSignal(0, 0.0, 0.0, 'unknown', 'insufficient_data')

        # Update GARCH
        current_vol = self.garch.update(returns[-1])
        forecast_vol = self.garch.forecast(1)[0]

        self.vol_history.append(current_vol)
        if len(self.vol_history) > 100:
            self.vol_history = self.vol_history[-100:]

        if len(self.vol_history) < 20:
            return GARCHSignal(0, 0.0, current_vol, 'unknown', 'building_history')

        # Vol expanding or contracting
        avg_vol = np.mean(self.vol_history[-20:])
        vol_expanding = forecast_vol > avg_vol * 1.1

        # Recent price breakout?
        if prices is not None and len(prices) >= 20:
            recent_high = np.max(prices[-20:-1])
            recent_low = np.min(prices[-20:-1])
            is_breakout = prices[-1] > recent_high or prices[-1] < recent_low

            if is_breakout and vol_expanding:
                direction = 1 if prices[-1] > recent_high else -1
                confidence = min(1.0, (forecast_vol / avg_vol - 1) * 2)
                signal_type = 'breakout_confirmed'
            elif is_breakout and not vol_expanding:
                direction = 0  # False breakout
                confidence = 0.0
                signal_type = 'false_breakout'
            else:
                direction = 0
                confidence = 0.0
                signal_type = 'no_breakout'
        else:
            direction = 0
            confidence = 0.0
            signal_type = 'no_price_data'

        vol_regime = 'high' if current_vol > np.percentile(self.vol_history, 75) else (
            'low' if current_vol < np.percentile(self.vol_history, 25) else 'medium')

        return GARCHSignal(
            direction=direction,
            confidence=confidence,
            vol_forecast=forecast_vol,
            vol_regime=vol_regime,
            signal_type=signal_type,
        )


class GARCHMeanRevSignal:
    """
    Formula 72062: GARCH Mean Reversion Signal

    Trades mean reversion when volatility is contracting.
    Low/falling vol = range-bound market = mean reversion works.
    """

    FORMULA_ID = 72062

    def __init__(self):
        self.garch = GARCHModel()
        self.vol_history: List[float] = []

    def fit(self, returns: np.ndarray):
        self.garch.fit(returns)

    def generate_signal(self, returns: np.ndarray, prices: np.ndarray = None) -> GARCHSignal:
        if len(returns) < 20:
            return GARCHSignal(0, 0.0, 0.0, 'unknown', 'insufficient_data')

        current_vol = self.garch.update(returns[-1])
        forecast_vol = self.garch.forecast(1)[0]

        self.vol_history.append(current_vol)
        if len(self.vol_history) > 100:
            self.vol_history = self.vol_history[-100:]

        if len(self.vol_history) < 20:
            return GARCHSignal(0, 0.0, current_vol, 'unknown', 'building_history')

        avg_vol = np.mean(self.vol_history[-20:])
        vol_percentile = np.sum(np.array(self.vol_history) < current_vol) / len(self.vol_history)

        # Low vol regime = mean reversion opportunity
        if vol_percentile < 0.3 and forecast_vol < avg_vol:
            if prices is not None and len(prices) >= 20:
                ma = np.mean(prices[-20:])
                deviation = (prices[-1] - ma) / ma

                if deviation > 0.02:  # Above MA
                    direction = -1  # Mean revert down
                    confidence = min(1.0, abs(deviation) * 10)
                elif deviation < -0.02:  # Below MA
                    direction = 1  # Mean revert up
                    confidence = min(1.0, abs(deviation) * 10)
                else:
                    direction = 0
                    confidence = 0.0
            else:
                direction = 0
                confidence = 0.0
            signal_type = 'mean_reversion'
        else:
            direction = 0
            confidence = 0.0
            signal_type = 'no_opportunity'

        vol_regime = 'low' if vol_percentile < 0.3 else ('high' if vol_percentile > 0.7 else 'medium')

        return GARCHSignal(
            direction=direction,
            confidence=confidence,
            vol_forecast=forecast_vol,
            vol_regime=vol_regime,
            signal_type=signal_type,
        )


class VolClusterSignal:
    """
    Formula 72063: Volatility Clustering Signal

    Exploits vol clustering - high vol begets high vol.
    Position sizing based on volatility regime.
    """

    FORMULA_ID = 72063

    def __init__(self):
        self.garch = GARCHModel()
        self.vol_history: List[float] = []

    def fit(self, returns: np.ndarray):
        self.garch.fit(returns)

    def generate_signal(self, returns: np.ndarray) -> GARCHSignal:
        if len(returns) < 20:
            return GARCHSignal(0, 0.0, 0.0, 'unknown', 'insufficient_data')

        current_vol = self.garch.update(returns[-1])
        self.vol_history.append(current_vol)

        if len(self.vol_history) > 100:
            self.vol_history = self.vol_history[-100:]

        if len(self.vol_history) < 10:
            return GARCHSignal(0, 0.0, current_vol, 'unknown', 'building_history')

        # Vol trend
        vol_ma_short = np.mean(self.vol_history[-5:])
        vol_ma_long = np.mean(self.vol_history[-20:]) if len(self.vol_history) >= 20 else vol_ma_short

        vol_expanding = vol_ma_short > vol_ma_long * 1.1
        vol_contracting = vol_ma_short < vol_ma_long * 0.9

        # Direction from recent momentum
        recent_return = np.sum(returns[-3:])

        if vol_expanding:
            # Vol expanding - momentum works
            direction = 1 if recent_return > 0 else -1
            confidence = 0.4  # Lower confidence in high vol
            signal_type = 'momentum_high_vol'
        elif vol_contracting:
            # Vol contracting - mean reversion works
            direction = -1 if recent_return > 0.01 else (1 if recent_return < -0.01 else 0)
            confidence = 0.5
            signal_type = 'mean_rev_low_vol'
        else:
            direction = 0
            confidence = 0.0
            signal_type = 'neutral_vol'

        vol_regime = 'expanding' if vol_expanding else ('contracting' if vol_contracting else 'stable')

        return GARCHSignal(
            direction=direction,
            confidence=confidence,
            vol_forecast=current_vol,
            vol_regime=vol_regime,
            signal_type=signal_type,
        )


class VolRegimeSignal:
    """
    Formula 72064: Volatility Regime Signal

    Identifies vol regime and trades accordingly.
    Different strategies for different vol regimes.
    """

    FORMULA_ID = 72064

    def __init__(self):
        self.garch = GARCHModel()
        self.vol_history: List[float] = []

    def fit(self, returns: np.ndarray):
        self.garch.fit(returns)

    def generate_signal(self, returns: np.ndarray) -> GARCHSignal:
        if len(returns) < 20:
            return GARCHSignal(0, 0.0, 0.0, 'unknown', 'insufficient_data')

        current_vol = self.garch.update(returns[-1])
        self.vol_history.append(current_vol)

        if len(self.vol_history) > 200:
            self.vol_history = self.vol_history[-200:]

        if len(self.vol_history) < 50:
            return GARCHSignal(0, 0.0, current_vol, 'unknown', 'building_history')

        # Regime based on percentile
        vol_pct = np.sum(np.array(self.vol_history) < current_vol) / len(self.vol_history)

        if vol_pct < 0.2:
            # Very low vol - expect vol expansion
            vol_regime = 'very_low'
            # Trade breakouts
            if returns[-1] > 0.02:
                direction = 1
                confidence = 0.6
            elif returns[-1] < -0.02:
                direction = -1
                confidence = 0.6
            else:
                direction = 0
                confidence = 0.0
            signal_type = 'breakout_expected'
        elif vol_pct > 0.8:
            # Very high vol - expect vol mean reversion
            vol_regime = 'very_high'
            # Reduce position, fade extremes
            direction = 0
            confidence = 0.0
            signal_type = 'vol_spike_caution'
        else:
            # Normal vol - trend following
            vol_regime = 'normal'
            mom = np.sum(returns[-5:])
            if mom > 0.03:
                direction = 1
                confidence = 0.4
            elif mom < -0.03:
                direction = -1
                confidence = 0.4
            else:
                direction = 0
                confidence = 0.0
            signal_type = 'trend_following'

        return GARCHSignal(
            direction=direction,
            confidence=confidence,
            vol_forecast=current_vol,
            vol_regime=vol_regime,
            signal_type=signal_type,
        )


class GARCHEnsembleSignal:
    """
    Formula 72065: GARCH Ensemble Signal

    Combines all GARCH-based signals.
    """

    FORMULA_ID = 72065

    def __init__(self):
        self.signals = [
            VolClusterSignal(),
            VolRegimeSignal(),
        ]

    def fit(self, returns: np.ndarray):
        for s in self.signals:
            s.fit(returns)

    def generate_signal(self, returns: np.ndarray, prices: np.ndarray = None) -> GARCHSignal:
        results = []

        for s in self.signals:
            result = s.generate_signal(returns)
            results.append(result)

        # Weighted combination
        total_dir = sum(r.direction * r.confidence for r in results)
        total_conf = sum(r.confidence for r in results)

        if total_conf > 0:
            avg_dir = total_dir / total_conf
            direction = 1 if avg_dir > 0.3 else (-1 if avg_dir < -0.3 else 0)
            confidence = total_conf / len(results)
        else:
            direction = 0
            confidence = 0.0

        avg_vol = np.mean([r.vol_forecast for r in results])
        vol_regimes = [r.vol_regime for r in results]
        mode_regime = max(set(vol_regimes), key=vol_regimes.count)

        return GARCHSignal(
            direction=direction,
            confidence=confidence,
            vol_forecast=avg_vol,
            vol_regime=mode_regime,
            signal_type='ensemble',
        )
