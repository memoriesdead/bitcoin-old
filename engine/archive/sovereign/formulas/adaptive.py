"""
SOVEREIGN ADAPTIVE FORMULAS
===========================
IDs: 10001-10005

THE EDGE: We see Bitcoin blockchain flow 10-60 seconds BEFORE it hits exchanges.
- INFLOW to exchange  = Depositing to SELL = SHORT signal
- OUTFLOW from exchange = Withdrawing to HOLD = LONG signal

These 5 formulas learn optimal parameters from every trade.

FORMULA INDEX
-------------
10001: AdaptiveFlowImpactEstimator  - BTC amount → price impact
10002: AdaptiveTimingOptimizer      - Optimal entry delay & hold time
10003: UniversalRegimeDetector      - Market regime classification
10004: BayesianParameterUpdater     - Parameter uncertainty tracking
10005: MultiTimescaleAggregator     - Signal aggregation across timeframes

CITATIONS
---------
- Kalman Filter: R.E. Kalman, 1960
- Kelly Criterion: J.L. Kelly Jr., Bell System Technical Journal, 1956
- Bayesian Updating: Bayes' Theorem, Thomas Bayes, 1763
- Multi-Armed Bandit: Thompson Sampling, 1933
- Regime Detection: Hamilton, 1989 (Markov Switching)
"""

import time
import math
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


###############################################################################
# FORMULA 10001: ADAPTIVE FLOW IMPACT ESTIMATOR
###############################################################################

class AdaptiveFlowImpactEstimator:
    """
    ID: 10001

    PURPOSE: Learn how BTC flow size translates to price impact.

    METHOD:
    - Bucket flows by size (tiny/small/medium/large/whale)
    - Track actual price impact per bucket
    - Update estimates using exponential moving average
    - Detect regime changes (high/low impact periods)

    UPDATES ON: Every trade outcome
    """

    FORMULA_ID = 10001

    def __init__(self):
        # Size buckets with initial impact estimates
        self.buckets = {
            'tiny':   {'min': 0,    'max': 1,    'impact': 0.0000, 'var': 0.001, 'n': 0},
            'small':  {'min': 1,    'max': 10,   'impact': 0.0001, 'var': 0.001, 'n': 0},
            'medium': {'min': 10,   'max': 100,  'impact': 0.0003, 'var': 0.001, 'n': 0},
            'large':  {'min': 100,  'max': 1000, 'impact': 0.0005, 'var': 0.001, 'n': 0},
            'whale':  {'min': 1000, 'max': float('inf'), 'impact': 0.0010, 'var': 0.001, 'n': 0},
        }
        self.alpha = 0.1  # Learning rate
        self.recent = []
        self.regime = 'normal'

    def _bucket(self, btc: float) -> str:
        for name, b in self.buckets.items():
            if b['min'] <= btc < b['max']:
                return name
        return 'medium'

    def predict(self, btc: float, direction: int) -> Dict:
        """
        Predict price impact for a given flow.

        Args:
            btc: Amount of BTC in flow
            direction: 1 (LONG/outflow) or -1 (SHORT/inflow)

        Returns:
            predicted_impact: Expected price change (%)
            confidence: 0-1 confidence in prediction
            regime: Current market regime
        """
        b = self.buckets[self._bucket(btc)]
        impact = b['impact'] * btc * direction

        # Confidence from sample size and variance
        conf = min(0.9, 0.5 + b['n'] * 0.01) * (1 / (1 + b['var'] * 100))

        # Regime adjustments
        if self.regime == 'high_impact':
            impact *= 1.5
        elif self.regime == 'low_impact':
            impact *= 0.5
        elif self.regime == 'chaos':
            conf *= 0.5

        return {'predicted_impact': impact, 'confidence': conf, 'regime': self.regime, 'n': b['n']}

    def update(self, btc: float, direction: int, actual_impact: float):
        """Update estimates with actual outcome."""
        if btc <= 0:
            return

        b = self.buckets[self._bucket(btc)]
        observed = actual_impact / btc
        error = abs(observed - b['impact'])
        surprise = error / max(0.0001, b['var'] ** 0.5)

        # Adaptive learning rate (learn faster from surprises)
        alpha = min(0.5, self.alpha * (1 + surprise))

        # Update estimates
        b['impact'] = b['impact'] * (1 - alpha) + observed * alpha
        b['var'] = b['var'] * (1 - alpha) + error ** 2 * alpha
        b['n'] += 1

        # Track for regime detection
        self.recent.append(abs(actual_impact))
        if len(self.recent) > 100:
            self.recent.pop(0)
        self._detect_regime()

    def _detect_regime(self):
        if len(self.recent) < 10:
            return
        recent_avg = sum(self.recent[-10:]) / 10
        overall_avg = sum(self.recent) / len(self.recent)
        ratio = recent_avg / max(0.0001, overall_avg)

        if ratio > 2.0:
            self.regime = 'high_impact'
        elif ratio < 0.5:
            self.regime = 'low_impact'
        elif ratio > 1.5 or ratio < 0.67:
            self.regime = 'chaos'
        else:
            self.regime = 'normal'


###############################################################################
# FORMULA 10002: ADAPTIVE TIMING OPTIMIZER
###############################################################################

class AdaptiveTimingOptimizer:
    """
    ID: 10002

    PURPOSE: Learn optimal entry delay and hold time.

    METHOD:
    - Track PnL outcomes for different delay/hold combinations
    - Use exploration vs exploitation (epsilon-greedy with decay)
    - Exponential decay on old results (adapt to changing markets)

    KEY INSIGHT: Price impact from flow takes 10-60 seconds to materialize.
                 Trading instantly loses money. Optimal delay is learned.

    UPDATES ON: Every trade outcome
    """

    FORMULA_ID = 10002

    def __init__(self):
        # Delay options (seconds)
        self.delays = {d: {'pnl': 0, 'n': 0, 'avg': 0}
                       for d in [0, 5, 10, 15, 20, 30, 45, 60]}
        # Hold options (seconds)
        self.holds = {h: {'pnl': 0, 'n': 0, 'avg': 0}
                      for h in [10, 20, 30, 45, 60, 90, 120]}

        self.optimal_delay = 15  # Initial guess
        self.optimal_hold = 30   # Initial guess
        self.exploration = 0.3   # 30% exploration initially
        self.decay = 0.99        # Decay old results

    def get_delay(self, explore: bool = True) -> float:
        """Get entry delay. May explore or exploit."""
        if explore and random.random() < self.exploration:
            return random.choice(list(self.delays.keys()))
        return self.optimal_delay

    def get_hold(self, explore: bool = True) -> float:
        """Get hold time. May explore or exploit."""
        if explore and random.random() < self.exploration:
            return random.choice(list(self.holds.keys()))
        return self.optimal_hold

    def record(self, delay: float, hold: float, pnl: float):
        """Record outcome and update optimal values."""
        # Decay old results
        for d in self.delays.values():
            d['pnl'] *= self.decay
            d['n'] *= self.decay
        for h in self.holds.values():
            h['pnl'] *= self.decay
            h['n'] *= self.decay

        # Record new result (snap to closest bucket)
        closest_d = min(self.delays.keys(), key=lambda x: abs(x - delay))
        closest_h = min(self.holds.keys(), key=lambda x: abs(x - hold))

        self.delays[closest_d]['pnl'] += pnl
        self.delays[closest_d]['n'] += 1
        self.holds[closest_h]['pnl'] += pnl
        self.holds[closest_h]['n'] += 1

        # Update averages
        for d, v in self.delays.items():
            if v['n'] > 0.1:
                v['avg'] = v['pnl'] / v['n']
        for h, v in self.holds.items():
            if v['n'] > 0.1:
                v['avg'] = v['pnl'] / v['n']

        # Update optimal (exponential smoothing)
        best_d = max(self.delays.keys(), key=lambda d: self.delays[d]['avg'])
        best_h = max(self.holds.keys(), key=lambda h: self.holds[h]['avg'])

        self.optimal_delay = self.optimal_delay * 0.8 + best_d * 0.2
        self.optimal_hold = self.optimal_hold * 0.8 + best_h * 0.2

        # Reduce exploration over time
        self.exploration = max(0.05, self.exploration * 0.999)


###############################################################################
# FORMULA 10003: UNIVERSAL REGIME DETECTOR
###############################################################################

class UniversalRegimeDetector:
    """
    ID: 10003

    PURPOSE: Classify current market regime from flow + price patterns.

    REGIMES:
    - ACCUMULATION:   More outflows, whales accumulating, bullish bias
    - DISTRIBUTION:   More inflows, whales distributing, bearish bias
    - TRENDING_UP:    Strong outflows + rising price, high confidence long
    - TRENDING_DOWN:  Strong inflows + falling price, high confidence short
    - CHOPPY:         High variance, reduce position sizes
    - QUIET:          Low flow, wait for signal

    PARAMETERS: Each regime has multipliers for confidence, size, hold time.

    CITATION: Inspired by Hamilton (1989) Markov regime switching models.
    """

    FORMULA_ID = 10003

    # Regime-specific parameter multipliers (aggressive settings)
    PARAMS = {
        'ACCUMULATION':  {'bias': 1,  'conf': 1.3, 'size': 1.2, 'hold': 1.0},
        'DISTRIBUTION':  {'bias': -1, 'conf': 1.3, 'size': 1.2, 'hold': 1.0},
        'TRENDING_UP':   {'bias': 1,  'conf': 1.5, 'size': 1.5, 'hold': 1.2},
        'TRENDING_DOWN': {'bias': -1, 'conf': 1.5, 'size': 1.5, 'hold': 1.2},
        'CHOPPY':        {'bias': 0,  'conf': 0.8, 'size': 0.8, 'hold': 0.7},  # Less penalty
        'QUIET':         {'bias': 0,  'conf': 1.0, 'size': 0.9, 'hold': 1.0},  # More aggressive
    }

    def __init__(self, window: float = 300):
        self.window = window  # 5 minute default
        self.inflows = []     # (timestamp, btc)
        self.outflows = []    # (timestamp, btc)
        self.prices = []      # (timestamp, price)
        self.regime = 'QUIET'
        self.confidence = 0.5
        self.duration = 0     # How long in current regime

    def add_flow(self, direction: int, btc: float, ts: float):
        """Record a flow event."""
        if direction == 1:
            self.outflows.append((ts, btc))
        else:
            self.inflows.append((ts, btc))
        self._cleanup(ts)
        self._detect()

    def add_price(self, price: float, ts: float):
        """Record a price update."""
        self.prices.append((ts, price))
        self._cleanup(ts)

    def _cleanup(self, ts: float):
        cutoff = ts - self.window
        self.inflows = [(t, b) for t, b in self.inflows if t > cutoff]
        self.outflows = [(t, b) for t, b in self.outflows if t > cutoff]
        self.prices = [(t, p) for t, p in self.prices if t > cutoff]

    def _detect(self):
        old = self.regime

        total_in = sum(b for _, b in self.inflows)
        total_out = sum(b for _, b in self.outflows)
        total = total_in + total_out

        # Price trend
        if len(self.prices) >= 2:
            prices = [p for _, p in self.prices]
            trend = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
            variance = sum((p - sum(prices)/len(prices))**2 for p in prices) / len(prices)
        else:
            trend, variance = 0, 0

        # Classify regime
        if total < 10:
            self.regime, self.confidence = 'QUIET', 0.6
        elif variance > 0.0001:
            self.regime, self.confidence = 'CHOPPY', 0.7
        elif total_out > total_in * 1.5:
            if trend > 0.001:
                self.regime, self.confidence = 'TRENDING_UP', 0.8
            else:
                self.regime, self.confidence = 'ACCUMULATION', 0.7
        elif total_in > total_out * 1.5:
            if trend < -0.001:
                self.regime, self.confidence = 'TRENDING_DOWN', 0.8
            else:
                self.regime, self.confidence = 'DISTRIBUTION', 0.7
        else:
            self.regime, self.confidence = 'QUIET', 0.5

        # Track regime duration
        self.duration = self.duration + 1 if self.regime == old else 0

    def get(self) -> Dict:
        """Get current regime with parameter adjustments."""
        p = self.PARAMS[self.regime]
        bonus = min(0.2, self.duration * 0.01)  # Confidence bonus for stable regime

        return {
            'regime': self.regime,
            'confidence': self.confidence + bonus,
            'direction_bias': p['bias'],
            'confidence_mult': p['conf'],
            'position_mult': p['size'],
            'hold_mult': p['hold'],
        }


###############################################################################
# FORMULA 10004: BAYESIAN PARAMETER UPDATER
###############################################################################

@dataclass
class BayesianParam:
    """
    Single parameter with Bayesian uncertainty tracking.

    Uses Kalman filter update equations:
    - K = prior_var / (prior_var + obs_var)
    - posterior_mean = prior_mean + K * (observation - prior_mean)
    - posterior_var = (1 - K) * prior_var
    """
    mean: float  # Current estimate
    var: float   # Uncertainty (variance)
    n: int = 0   # Number of observations

    def update(self, obs: float, obs_var: float):
        """Kalman-style Bayesian update."""
        K = self.var / (self.var + obs_var)
        self.mean = self.mean + K * (obs - self.mean)
        self.var = max(0.0001, (1 - K) * self.var)
        self.n += 1

    def get(self) -> Tuple[float, float, float]:
        """Return (mean, std, confidence)."""
        std = self.var ** 0.5
        conf = min(0.95, self.n / (self.n + 10) * (1 / (1 + std)))
        return self.mean, std, conf

    def sample(self) -> float:
        """Thompson sampling for exploration."""
        return random.gauss(self.mean, self.var ** 0.5)


class BayesianParameterUpdater:
    """
    ID: 10004

    PURPOSE: Maintain Bayesian estimates of all trading parameters.

    KEY INSIGHT: Every parameter has UNCERTAINTY, not just a point estimate.
                 High uncertainty → explore. Low uncertainty → exploit.

    PARAMETERS TRACKED:
    - flow_impact:    BTC → price impact coefficient
    - entry_delay:    Seconds to wait before entering
    - hold_time:      Seconds to hold position
    - stop_loss:      Stop loss percentage
    - take_profit:    Take profit percentage
    - min_btc:        Minimum flow size to trade
    - min_confidence: Minimum confidence to trade

    CITATION: Kalman (1960), Thompson Sampling (1933)
    """

    FORMULA_ID = 10004

    def __init__(self):
        self.params = {
            'flow_impact':    BayesianParam(0.0001, 0.0001),
            'entry_delay':    BayesianParam(5.0, 50.0),       # Faster entry (was 15)
            'hold_time':      BayesianParam(20.0, 100.0),     # Shorter holds (was 30)
            'stop_loss':      BayesianParam(0.005, 0.001),    # 0.5% stop (tighter)
            'take_profit':    BayesianParam(0.008, 0.001),    # 0.8% target (higher)
            'min_btc':        BayesianParam(0.5, 5.0),        # 0.5 BTC minimum (was 1.0)
            'min_confidence': BayesianParam(0.30, 0.01),      # 30% confidence (was 50%)
        }

    def get(self) -> Dict[str, float]:
        """Get current best parameter estimates."""
        return {k: p.mean for k, p in self.params.items()}

    def get_with_uncertainty(self) -> Dict:
        """Get estimates with uncertainty for debugging."""
        return {k: {'mean': p.mean, 'std': p.var**0.5, 'n': p.n}
                for k, p in self.params.items()}

    def update(self, trade: Dict):
        """Update parameters from trade outcome."""
        pnl = trade['pnl']

        # Update flow impact estimate
        if trade.get('btc_amount', 0) > 0:
            impact = trade.get('price_change', 0) / trade['btc_amount']
            self.params['flow_impact'].update(impact, 0.0001)

        # Good trade → update toward these parameters
        # Bad trade → shift away
        if pnl > 0:
            self.params['entry_delay'].update(trade.get('entry_delay', 15), 50)
            self.params['hold_time'].update(trade.get('hold_time', 30), 100)
        else:
            delay = trade.get('entry_delay', 15)
            # Shift opposite direction
            self.params['entry_delay'].update(delay + 5 if delay < 30 else delay - 5, 100)


###############################################################################
# FORMULA 10005: MULTI-TIMESCALE AGGREGATOR
###############################################################################

class MultiTimescaleAggregator:
    """
    ID: 10005

    PURPOSE: Aggregate signals across multiple timeframes.

    TIMESCALES:
    - micro:  5 seconds   (immediate flow)
    - short:  30 seconds  (recent pattern)
    - medium: 5 minutes   (trend)
    - long:   1 hour      (accumulation/distribution)

    METHOD:
    - Track net flow (outflow - inflow) per timescale
    - Weight timescales by historical accuracy
    - Adapt weights based on which timescale predicts best

    KEY INSIGHT: Different market conditions favor different timescales.
                 Trending → favor longer timescales
                 Choppy → favor shorter timescales
    """

    FORMULA_ID = 10005

    def __init__(self):
        self.scales = {
            'micro':  {'window': 5,    'weight': 0.10, 'flows': [], 'signal': 0},
            'short':  {'window': 30,   'weight': 0.40, 'flows': [], 'signal': 0},
            'medium': {'window': 300,  'weight': 0.35, 'flows': [], 'signal': 0},
            'long':   {'window': 3600, 'weight': 0.15, 'flows': [], 'signal': 0},
        }
        self.accuracy = {k: 0.5 for k in self.scales}  # Track prediction accuracy

    def add_flow(self, direction: int, btc: float, ts: float):
        """Add flow to all timescales."""
        for name, scale in self.scales.items():
            scale['flows'].append((ts, direction, btc))

            # Remove old flows
            cutoff = ts - scale['window']
            scale['flows'] = [(t, d, b) for t, d, b in scale['flows'] if t > cutoff]

            # Calculate signal: (outflow - inflow) / total
            if scale['flows']:
                long_btc = sum(b for _, d, b in scale['flows'] if d == 1)
                short_btc = sum(b for _, d, b in scale['flows'] if d == -1)
                total = long_btc + short_btc
                scale['signal'] = (long_btc - short_btc) / total if total > 0 else 0

    def get(self) -> Dict:
        """Get weighted aggregated signal."""
        total_weight = sum(s['weight'] * self.accuracy[n] for n, s in self.scales.items())

        if total_weight == 0:
            return {'direction': 0, 'strength': 0, 'confidence': 0}

        weighted = sum(
            s['signal'] * s['weight'] * self.accuracy[n]
            for n, s in self.scales.items()
        ) / total_weight

        direction = 1 if weighted > 0.1 else (-1 if weighted < -0.1 else 0)
        strength = abs(weighted)

        # Agreement bonus (all timescales agree = higher confidence)
        signals = [s['signal'] for s in self.scales.values()]
        agreement = 1 - (max(signals) - min(signals)) / 2 if signals else 0

        return {
            'direction': direction,
            'strength': strength,
            'confidence': agreement * strength,
            'signals': {k: v['signal'] for k, v in self.scales.items()},
        }

    def update_accuracy(self, was_correct: bool):
        """Update timescale accuracies after trade outcome."""
        alpha = 0.1
        target = 1.0 if was_correct else 0.0

        for name in self.accuracy:
            self.accuracy[name] = self.accuracy[name] * (1 - alpha) + target * alpha

        # Renormalize weights
        total = sum(self.accuracy.values())
        if total > 0:
            for name in self.scales:
                self.scales[name]['weight'] = self.accuracy[name] / total


###############################################################################
# MASTER ENGINE: COMBINES ALL FORMULAS
###############################################################################

class AdaptiveTradingEngine:
    """
    MASTER ENGINE

    Combines all 5 adaptive formulas into a single trading system.

    FLOW:
    1. on_flow() - Process blockchain flow signal
       - Update regime detector
       - Update multi-timescale aggregator
       - Predict impact, get regime adjustment
       - Vote on direction (flow + regime + multiscale)
       - Calculate entry delay, hold time, position size
       - Return trade signal (or None if below thresholds)

    2. check_entries() - Check if pending trades should enter

    3. record_result() - Update all formulas with trade outcome
       - Impact estimator learns BTC→impact relationship
       - Timing optimizer learns best delay/hold
       - Bayesian updater refines all parameters
       - Multiscale adjusts timescale weights

    THE EDGE: Every parameter adapts in real-time based on actual results.
    """

    def __init__(self):
        # Initialize all formula components
        self.impact = AdaptiveFlowImpactEstimator()        # 10001
        self.timing = AdaptiveTimingOptimizer()            # 10002
        self.regime = UniversalRegimeDetector()            # 10003
        self.bayesian = BayesianParameterUpdater()         # 10004
        self.multiscale = MultiTimescaleAggregator()       # 10005

        # Trade tracking
        self.pending = []   # Signals waiting for entry
        self.trades = []    # Completed trades
        self.pnl = 0        # Cumulative PnL

    def on_flow(self, exchange: str, direction: int, btc: float,
                ts: float, price: float) -> Optional[Dict]:
        """
        Process new blockchain flow signal.

        Args:
            exchange: Exchange name (binance, coinbase, etc.)
            direction: 1 (LONG/outflow) or -1 (SHORT/inflow)
            btc: Amount of BTC in flow
            ts: Timestamp
            price: Current BTC price

        Returns:
            Trade signal dict or None if signal doesn't pass gates
        """
        # Update all trackers
        self.regime.add_flow(direction, btc, ts)
        self.regime.add_price(price, ts)
        self.multiscale.add_flow(direction, btc, ts)

        # Get predictions from each formula
        impact_pred = self.impact.predict(btc, direction)
        regime_adj = self.regime.get()
        multi_signal = self.multiscale.get()
        params = self.bayesian.get()

        # Vote on direction (weighted combination)
        # More weight to raw flow signal for aggressive trading
        vote = (
            direction * 0.5 +                          # Raw flow direction (increased)
            regime_adj['direction_bias'] * 0.2 +       # Regime bias
            multi_signal['direction'] * 0.3            # Multi-timescale signal
        )
        final_direction = 1 if vote > 0.15 else (-1 if vote < -0.15 else 0)  # Lower threshold

        # Combined confidence (base + flow strength bonus)
        base_conf = (
            impact_pred['confidence'] * 0.3 +
            regime_adj['confidence'] * 0.3 +
            multi_signal['confidence'] * 0.4
        )
        # Add confidence bonus for larger flows
        flow_bonus = min(0.3, btc / 100)  # Up to 30% bonus for 100+ BTC
        confidence = (base_conf + flow_bonus) * max(0.7, regime_adj['confidence_mult'])  # Floor regime penalty

        # Gate checks (aggressive - trade on flow, learn from outcome)
        if final_direction == 0:
            return None
        if btc < 0.5:  # Lower threshold - even small flows matter
            return None

        # Calculate trade parameters
        delay = self.timing.get_delay()
        hold = self.timing.get_hold() * regime_adj['hold_mult']
        size = min(0.15, max(0.02, confidence * regime_adj['position_mult'] * 0.1))

        signal = {
            'exchange': exchange,
            'direction': final_direction,
            'confidence': confidence,
            'position_size': size,
            'entry_time': ts + delay,
            'entry_delay': delay,
            'hold_time': hold,
            'stop_loss': params['stop_loss'],
            'take_profit': params['take_profit'],
            'btc_amount': btc,
            'regime': regime_adj['regime'],
            'price_at_signal': price,
        }

        self.pending.append(signal)
        return signal

    def check_entries(self, now: float, price: float) -> List[Dict]:
        """Check pending signals for entry."""
        entries = []
        remaining = []

        for signal in self.pending:
            if now >= signal['entry_time']:
                signal['entry_price'] = price
                entries.append(signal)
            else:
                remaining.append(signal)

        self.pending = remaining
        return entries

    def record_result(self, trade: Dict, exit_price: float, exit_time: float) -> float:
        """
        Record trade result and update all formula components.

        This is where learning happens - every trade outcome improves the system.
        """
        entry_price = trade['entry_price']
        direction = trade['direction']

        # Calculate PnL
        pnl_pct = (exit_price - entry_price) / entry_price * direction
        pnl = trade['position_size'] * pnl_pct

        hold_time = exit_time - trade['entry_time']
        price_change = (exit_price - trade['price_at_signal']) / trade['price_at_signal']

        # UPDATE ALL FORMULAS WITH THIS OUTCOME
        self.impact.update(trade['btc_amount'], direction, price_change)
        self.timing.record(trade['entry_delay'], hold_time, pnl)
        self.bayesian.update({
            'pnl': pnl,
            'entry_delay': trade['entry_delay'],
            'hold_time': hold_time,
            'btc_amount': trade['btc_amount'],
            'price_change': price_change,
        })
        self.multiscale.update_accuracy(pnl > 0)

        # Record trade
        self.trades.append({**trade, 'exit_price': exit_price, 'pnl': pnl})
        self.pnl += pnl

        return pnl

    def get_stats(self) -> Dict:
        """Get current system statistics."""
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total = len(self.trades)

        return {
            'total_pnl': self.pnl,
            'trades': total,
            'win_rate': wins / total if total > 0 else 0,
            'regime': self.regime.regime,
            'optimal_delay': self.timing.optimal_delay,
            'optimal_hold': self.timing.optimal_hold,
            'parameters': self.bayesian.get(),
        }
