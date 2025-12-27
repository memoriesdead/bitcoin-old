#!/usr/bin/env python3
"""
HISTORICAL MULTI-SCALE ADAPTIVE DISCOVERY
==========================================

"6+ years is amateur. Test on ALL of Bitcoin."
"What works for 1 second may not work for 2 seconds."

This runs the full RenTech adaptive discovery on COMPLETE Bitcoin history:
- 6,184 days of blockchain data
- Multiple timescales (1d, 3d, 7d, 14d, 30d, 90d)
- Regime detection at each scale
- Pattern validation at each (scale, regime) combination
- Only generates formulas that work under SPECIFIC conditions

TARGET: 50.75% win rate at EACH (regime, scale) combination

Based on:
- Fractal Market Hypothesis
- Renaissance Technologies methodology
- Multi-scale wavelet analysis
"""

import os
import sys
import sqlite3
import numpy as np
import struct
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import math
import random

# Multiple timescales for daily data (in days)
SCALES = [1, 3, 7, 14, 30, 90]
SCALE_NAMES = {1: '1d', 3: '3d', 7: '7d', 14: '14d', 30: '30d', 90: '90d'}


class Regime(Enum):
    """Market regimes detected from blockchain metrics."""
    ACCUMULATION = 0      # High tx activity, price consolidating -> LONG
    DISTRIBUTION = 1      # High activity, price elevated -> SHORT
    NEUTRAL = 2           # Normal activity
    CAPITULATION = 3      # Extreme low activity, panic -> Contrarian LONG
    EUPHORIA = 4          # Extreme high activity, FOMO -> Contrarian SHORT


@dataclass
class DailyBucket:
    """Aggregated data for a day or multi-day period."""
    date: str
    scale: int  # in days

    # Blockchain metrics
    tx_count: float = 0.0
    tx_z: float = 0.0
    whale_count: float = 0.0
    whale_z: float = 0.0
    total_value: float = 0.0
    value_z: float = 0.0

    # Price metrics
    price_start: float = 0.0
    price_end: float = 0.0
    price_change: float = 0.0

    # Derived
    regime: int = 2  # Default NEUTRAL
    outcome: int = 0  # +1 up, -1 down, 0 flat


@dataclass
class ConditionalPattern:
    """A pattern that works under specific conditions."""
    name: str
    pattern_type: str
    direction: int  # +1 LONG, -1 SHORT

    # Conditions where valid
    valid_scales: Set[int] = field(default_factory=set)
    valid_regimes: Set[int] = field(default_factory=set)

    # Stats per (scale, regime)
    stats: Dict[Tuple[int, int], Dict] = field(default_factory=dict)

    # Overall
    total_occurrences: int = 0
    overall_win_rate: float = 0.0

    # Validation
    is_valid: bool = False
    best_condition: Tuple[int, int] = (0, 0)
    best_edge: float = 0.0


class MultiScaleHMM:
    """HMM for regime detection at a specific scale."""

    def __init__(self, scale: int, n_states: int = 5):
        self.scale = scale
        self.n_states = n_states

        # Transition matrix (sticky states)
        self.A = np.array([
            [0.80, 0.05, 0.10, 0.02, 0.03],  # ACCUMULATION
            [0.05, 0.80, 0.10, 0.03, 0.02],  # DISTRIBUTION
            [0.15, 0.15, 0.60, 0.05, 0.05],  # NEUTRAL
            [0.30, 0.10, 0.20, 0.35, 0.05],  # CAPITULATION
            [0.10, 0.30, 0.20, 0.05, 0.35],  # EUPHORIA
        ])

        # Emission means: [tx_z, whale_z, value_z, momentum]
        self.emission_means = {
            0: [0.5, 0.3, 0.3, -0.2],    # ACCUMULATION: high tx, low momentum
            1: [-0.5, -0.3, -0.3, 0.3],  # DISTRIBUTION: low tx, high momentum
            2: [0.0, 0.0, 0.0, 0.0],     # NEUTRAL
            3: [-1.5, -1.0, -1.0, -1.0], # CAPITULATION: very low everything
            4: [1.5, 1.0, 1.0, 1.0],     # EUPHORIA: very high everything
        }

        self.emission_vars = {s: [1.0, 1.0, 1.0, 1.0] for s in range(n_states)}
        self.pi = np.array([0.15, 0.15, 0.50, 0.10, 0.10])
        self.belief = self.pi.copy()

        self.state_history = []
        self.training_observations = []
        self.training_outcomes = []

    def _emission_prob(self, state: int, obs: List[float]) -> float:
        """P(observation | state) using Gaussian."""
        means = self.emission_means[state]
        vars_ = self.emission_vars[state]

        log_prob = 0
        for i in range(min(len(obs), len(means))):
            diff = obs[i] - means[i]
            log_prob -= 0.5 * (diff ** 2) / vars_[i]
            log_prob -= 0.5 * math.log(2 * math.pi * vars_[i])

        return math.exp(max(-50, log_prob))

    def update(self, bucket: DailyBucket) -> Dict:
        """Update belief and return state."""
        # Momentum = recent price trend
        momentum = bucket.price_change if abs(bucket.price_change) < 10 else np.sign(bucket.price_change) * 10

        obs = [
            bucket.tx_z,
            bucket.whale_z,
            bucket.value_z,
            momentum / 10.0  # Normalize to roughly [-1, 1]
        ]

        # Forward step
        emission = np.array([self._emission_prob(s, obs) for s in range(self.n_states)])
        prior = self.A.T @ self.belief
        posterior = prior * emission

        total = posterior.sum()
        if total > 0:
            self.belief = posterior / total
        else:
            self.belief = self.pi.copy()

        state = int(np.argmax(self.belief))
        confidence = float(self.belief[state])

        self.state_history.append(state)

        # Signal mapping
        signal_map = {0: 1, 1: -1, 2: 0, 3: 1, 4: -1}

        return {
            'state': state,
            'regime': Regime(state).name,
            'confidence': confidence,
            'signal': signal_map[state],
        }

    def add_training_sample(self, bucket: DailyBucket, outcome: int):
        """Add labeled sample."""
        momentum = bucket.price_change if abs(bucket.price_change) < 10 else np.sign(bucket.price_change) * 10
        obs = np.array([bucket.tx_z, bucket.whale_z, bucket.value_z, momentum / 10.0])
        self.training_observations.append(obs)
        self.training_outcomes.append(outcome)

    def train_baum_welch(self, n_iter: int = 50) -> float:
        """Train HMM."""
        if len(self.training_observations) < 50:
            print(f"  [HMM-{SCALE_NAMES[self.scale]}] Not enough data ({len(self.training_observations)})")
            return 0.0

        observations = np.array(self.training_observations)
        T, D = observations.shape
        N = self.n_states

        print(f"  [HMM-{SCALE_NAMES[self.scale]}] Training on {T} samples...")

        for iteration in range(n_iter):
            # Forward pass
            alpha = np.zeros((T, N))
            scale = np.zeros(T)

            for s in range(N):
                alpha[0, s] = self.pi[s] * self._emission_prob(s, observations[0])
            scale[0] = max(alpha[0].sum(), 1e-10)
            alpha[0] /= scale[0]

            for t in range(1, T):
                for s in range(N):
                    alpha[t, s] = sum(alpha[t-1, s2] * self.A[s2, s] for s2 in range(N))
                    alpha[t, s] *= self._emission_prob(s, observations[t])
                scale[t] = max(alpha[t].sum(), 1e-10)
                alpha[t] /= scale[t]

            # Backward pass
            beta = np.zeros((T, N))
            beta[T-1] = 1.0

            for t in range(T-2, -1, -1):
                for s in range(N):
                    beta[t, s] = sum(
                        self.A[s, s2] * self._emission_prob(s2, observations[t+1]) * beta[t+1, s2]
                        for s2 in range(N)
                    )
                if scale[t+1] > 0:
                    beta[t] /= scale[t+1]

            # M-step: Update emission parameters
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=0)
            gamma_sum = np.maximum(gamma_sum, 1e-10)

            for s in range(N):
                weight_sum = gamma[:, s].sum()
                if weight_sum > 0:
                    for d in range(D):
                        self.emission_means[s][d] = float((gamma[:, s] * observations[:, d]).sum() / weight_sum)
                        var = (gamma[:, s] * (observations[:, d] - self.emission_means[s][d])**2).sum() / weight_sum
                        self.emission_vars[s][d] = max(0.1, float(var))

        accuracy = self._validate()
        print(f"  [HMM-{SCALE_NAMES[self.scale]}] Accuracy: {accuracy:.4f}")
        return accuracy

    def _validate(self) -> float:
        """Validate on training data."""
        if not self.training_observations:
            return 0.0

        correct = 0
        total = 0
        self.belief = self.pi.copy()

        for obs, outcome in zip(self.training_observations, self.training_outcomes):
            if outcome == 0:
                continue

            emission = np.array([self._emission_prob(s, obs) for s in range(self.n_states)])
            prior = self.A.T @ self.belief
            posterior = prior * emission
            total_p = posterior.sum()
            if total_p > 0:
                self.belief = posterior / total_p

            state = int(np.argmax(self.belief))
            signal_map = {0: 1, 1: -1, 2: 0, 3: 1, 4: -1}
            predicted = signal_map[state]

            if predicted != 0:
                if predicted == outcome:
                    correct += 1
                total += 1

        return correct / max(1, total)


class HistoricalDiscovery:
    """Full adaptive discovery on historical Bitcoin data."""

    def __init__(self):
        self.db_path = Path(__file__).parent.parent.parent.parent / 'data' / 'unified_bitcoin.db'

        # Per-scale HMMs
        self.hmms = {scale: MultiScaleHMM(scale) for scale in SCALES}

        # Data storage
        self.daily_data = []
        self.prices = {}
        self.buckets = {scale: [] for scale in SCALES}

        # Discovered patterns
        self.patterns = []

        # Stats
        self.stats = {
            'days_loaded': 0,
            'patterns_discovered': 0,
            'patterns_validated': 0,
        }

    def load_data(self):
        """Load all historical data."""
        print("\n" + "="*70)
        print("LOADING COMPLETE BITCOIN HISTORY")
        print("="*70)

        conn = sqlite3.connect(self.db_path)

        # Load prices
        cursor = conn.execute('SELECT date, close FROM prices ORDER BY date')
        for row in cursor.fetchall():
            self.prices[row[0]] = row[1]
        print(f"Loaded {len(self.prices)} price days")

        # Load daily features
        cursor = conn.execute('''
            SELECT date, tx_count, whale_tx_count, total_value_btc
            FROM daily_features
            ORDER BY date
        ''')

        for row in cursor.fetchall():
            date = row[0]

            # Decode whale_tx_count if needed
            whale_count = row[2]
            if isinstance(whale_count, bytes):
                try:
                    whale_count = struct.unpack('<q', whale_count)[0]
                except:
                    whale_count = 0
            elif whale_count is None:
                whale_count = 0

            self.daily_data.append({
                'date': date,
                'tx_count': row[1] or 0,
                'whale_count': whale_count,
                'total_value': row[3] or 0,
                'price': self.prices.get(date, 0),
            })

        conn.close()

        self.stats['days_loaded'] = len(self.daily_data)
        print(f"Loaded {len(self.daily_data)} feature days")
        print(f"Date range: {self.daily_data[0]['date']} to {self.daily_data[-1]['date']}")

    def compute_z_scores(self):
        """Compute z-scores for all metrics."""
        print("\nComputing z-scores...")

        # Get arrays
        tx_counts = np.array([d['tx_count'] for d in self.daily_data])
        whale_counts = np.array([d['whale_count'] for d in self.daily_data])
        values = np.array([d['total_value'] for d in self.daily_data])

        # Rolling z-scores (30-day lookback)
        for i, d in enumerate(self.daily_data):
            start = max(0, i - 30)

            # TX z-score
            window = tx_counts[start:i+1]
            if len(window) > 1 and np.std(window) > 0:
                d['tx_z'] = (tx_counts[i] - np.mean(window)) / np.std(window)
            else:
                d['tx_z'] = 0.0

            # Whale z-score
            window = whale_counts[start:i+1]
            if len(window) > 1 and np.std(window) > 0:
                d['whale_z'] = (whale_counts[i] - np.mean(window)) / np.std(window)
            else:
                d['whale_z'] = 0.0

            # Value z-score
            window = values[start:i+1]
            if len(window) > 1 and np.std(window) > 0:
                d['value_z'] = (values[i] - np.mean(window)) / np.std(window)
            else:
                d['value_z'] = 0.0

        print("Z-scores computed")

    def create_multi_scale_buckets(self):
        """Create buckets at each timescale."""
        print("\nCreating multi-scale buckets...")

        for scale in SCALES:
            for i in range(scale, len(self.daily_data)):
                d = self.daily_data[i]
                prev_d = self.daily_data[i - scale]

                # Skip if missing prices
                if d['price'] <= 0 or prev_d['price'] <= 0:
                    continue

                # Average z-scores over the period
                avg_tx_z = np.mean([self.daily_data[j]['tx_z'] for j in range(i-scale+1, i+1)])
                avg_whale_z = np.mean([self.daily_data[j]['whale_z'] for j in range(i-scale+1, i+1)])
                avg_value_z = np.mean([self.daily_data[j]['value_z'] for j in range(i-scale+1, i+1)])

                # Price change
                price_change = (d['price'] / prev_d['price'] - 1) * 100

                bucket = DailyBucket(
                    date=d['date'],
                    scale=scale,
                    tx_count=d['tx_count'],
                    tx_z=avg_tx_z,
                    whale_count=d['whale_count'],
                    whale_z=avg_whale_z,
                    total_value=d['total_value'],
                    value_z=avg_value_z,
                    price_start=prev_d['price'],
                    price_end=d['price'],
                    price_change=price_change,
                )

                # Outcome for next period
                if i + scale < len(self.daily_data) and self.daily_data[i + scale]['price'] > 0:
                    future_change = (self.daily_data[i + scale]['price'] / d['price'] - 1) * 100
                    if future_change > 0.5:  # >0.5% move
                        bucket.outcome = 1
                    elif future_change < -0.5:
                        bucket.outcome = -1
                    else:
                        bucket.outcome = 0

                self.buckets[scale].append(bucket)

            print(f"  {SCALE_NAMES[scale]}: {len(self.buckets[scale])} buckets")

    def train_hmms(self):
        """Train HMM for each scale."""
        print("\n" + "="*70)
        print("TRAINING PER-SCALE HMMs")
        print("="*70)

        for scale in SCALES:
            hmm = self.hmms[scale]

            for bucket in self.buckets[scale]:
                if bucket.outcome != 0:
                    hmm.add_training_sample(bucket, bucket.outcome)

            hmm.train_baum_welch()

        # Assign regimes to all buckets
        for scale in SCALES:
            hmm = self.hmms[scale]
            hmm.belief = hmm.pi.copy()

            for bucket in self.buckets[scale]:
                result = hmm.update(bucket)
                bucket.regime = result['state']

    def discover_patterns(self):
        """Discover patterns at each (scale, regime) combination."""
        print("\n" + "="*70)
        print("DISCOVERING CONDITIONAL PATTERNS")
        print("="*70)

        pattern_types = [
            ('tx_spike_long', lambda b: b.tx_z > 1.5, 1),
            ('tx_spike_strong_long', lambda b: b.tx_z > 2.0, 1),
            ('tx_low_mean_revert', lambda b: b.tx_z < -1.5, 1),
            ('whale_spike_long', lambda b: b.whale_z > 1.5, 1),
            ('whale_low_accumulate', lambda b: b.whale_z < -1.5, 1),
            ('value_spike_long', lambda b: b.value_z > 1.5, 1),
            ('all_high_momentum', lambda b: b.tx_z > 1.0 and b.whale_z > 1.0 and b.value_z > 1.0, 1),
            ('all_low_contrarian', lambda b: b.tx_z < -1.0 and b.whale_z < -1.0 and b.value_z < -1.0, 1),
            ('tx_spike_short', lambda b: b.tx_z > 2.5, -1),  # Extreme = distribution
            ('whale_distribution', lambda b: b.whale_z > 2.5, -1),
        ]

        for pattern_name, condition, direction in pattern_types:
            pattern = ConditionalPattern(
                name=pattern_name,
                pattern_type=pattern_name,
                direction=direction,
            )

            # Test at each (scale, regime)
            for scale in SCALES:
                for regime in range(5):
                    matches = []

                    for bucket in self.buckets[scale]:
                        if bucket.regime == regime and condition(bucket):
                            if bucket.outcome != 0:
                                matches.append(bucket.outcome)

                    if len(matches) >= 20:  # Minimum sample size
                        wins = sum(1 for o in matches if (direction == 1 and o == 1) or (direction == -1 and o == -1))
                        win_rate = wins / len(matches)

                        pattern.stats[(scale, regime)] = {
                            'occurrences': len(matches),
                            'wins': wins,
                            'losses': len(matches) - wins,
                            'win_rate': win_rate,
                            'edge': win_rate - 0.5,
                        }

                        if win_rate >= 0.5075:
                            pattern.valid_scales.add(scale)
                            pattern.valid_regimes.add(regime)

                        pattern.total_occurrences += len(matches)

            if pattern.valid_scales:
                self.patterns.append(pattern)

        self.stats['patterns_discovered'] = len(self.patterns)
        print(f"Discovered {len(self.patterns)} patterns with valid conditions")

    def validate_patterns(self, n_bootstrap: int = 1000):
        """Validate patterns using Monte Carlo."""
        print("\n" + "="*70)
        print("VALIDATING PATTERNS (Monte Carlo)")
        print("="*70)

        validated = []

        for pattern in self.patterns:
            best_edge = 0.0
            best_condition = (1, 2)

            for (scale, regime), stats in pattern.stats.items():
                if stats['occurrences'] < 20:
                    continue

                # Bootstrap
                wins = stats['wins']
                total = stats['occurrences']

                bootstrap_wrs = []
                for _ in range(n_bootstrap):
                    sample_wins = sum(random.random() < stats['win_rate'] for _ in range(total))
                    bootstrap_wrs.append(sample_wins / total)

                bootstrap_wrs.sort()
                ci_low = bootstrap_wrs[int(n_bootstrap * 0.025)]
                ci_high = bootstrap_wrs[int(n_bootstrap * 0.975)]

                stats['ci_low'] = ci_low
                stats['ci_high'] = ci_high

                # P-value
                expected_wins = total * 0.5
                chi2 = (wins - expected_wins) ** 2 / expected_wins
                stats['p_value'] = math.exp(-chi2 / 2) if chi2 < 50 else 0.0

                # Check validity
                if ci_low >= 0.5 and stats['win_rate'] >= 0.5075:
                    edge = stats['win_rate'] - 0.5
                    if edge > best_edge:
                        best_edge = edge
                        best_condition = (scale, regime)

            if best_edge > 0:
                pattern.is_valid = True
                pattern.best_edge = best_edge
                pattern.best_condition = best_condition
                validated.append(pattern)

                print(f"[VALID] {pattern.name}: edge={best_edge:.4f} "
                      f"at ({SCALE_NAMES[best_condition[0]]}, {Regime(best_condition[1]).name})")

        self.stats['patterns_validated'] = len(validated)
        print(f"\n{len(validated)} patterns validated")

        return validated

    def generate_report(self):
        """Generate comprehensive report."""
        print("\n" + "="*70)
        print("MULTI-SCALE ADAPTIVE DISCOVERY REPORT")
        print("="*70)

        valid_patterns = [p for p in self.patterns if p.is_valid]

        print(f"\nData: {self.stats['days_loaded']} days of Bitcoin history")
        print(f"Patterns discovered: {self.stats['patterns_discovered']}")
        print(f"Patterns validated: {self.stats['patterns_validated']}")

        print("\n" + "-"*70)
        print("TOP VALIDATED PATTERNS BY EDGE:")
        print("-"*70)

        sorted_patterns = sorted(valid_patterns, key=lambda p: p.best_edge, reverse=True)

        for i, p in enumerate(sorted_patterns[:20]):
            scale_name = SCALE_NAMES[p.best_condition[0]]
            regime_name = Regime(p.best_condition[1]).name
            dir_str = "LONG" if p.direction == 1 else "SHORT"
            stats = p.stats[p.best_condition]

            print(f"\n{i+1}. {p.name}")
            print(f"   Direction: {dir_str}")
            print(f"   Best at: {scale_name} + {regime_name}")
            print(f"   Edge: {p.best_edge:.4f} ({p.best_edge*100:.2f}%)")
            print(f"   Win Rate: {stats['win_rate']:.4f} ({stats['win_rate']*100:.1f}%)")
            print(f"   Occurrences: {stats['occurrences']}")
            print(f"   95% CI: [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]")
            print(f"   P-value: {stats['p_value']:.4f}")

            # Show all valid conditions
            print(f"   Valid conditions:")
            for (scale, regime), st in p.stats.items():
                if st.get('win_rate', 0) >= 0.5075:
                    print(f"      {SCALE_NAMES[scale]} + {Regime(regime).name}: {st['win_rate']:.4f} (n={st['occurrences']})")

        # Kelly calculation
        print("\n" + "="*70)
        print("KELLY LEVERAGE CALCULATION")
        print("="*70)

        for p in sorted_patterns[:5]:
            stats = p.stats[p.best_condition]
            edge = stats['win_rate'] - 0.5

            # Variance approximation: p(1-p) for binomial
            variance = stats['win_rate'] * (1 - stats['win_rate'])

            kelly_full = edge / variance if variance > 0 else 0
            kelly_half = kelly_full / 2
            kelly_quarter = kelly_full / 4

            print(f"\n{p.name}:")
            print(f"  Edge: {edge*100:.2f}%")
            print(f"  Full Kelly: {kelly_full:.1f}x")
            print(f"  Half Kelly: {kelly_half:.1f}x")
            print(f"  Quarter Kelly: {kelly_quarter:.1f}x (RECOMMENDED)")

        # Compounding simulation
        print("\n" + "="*70)
        print("COMPOUNDING SIMULATION ($100 START)")
        print("="*70)

        if sorted_patterns:
            best = sorted_patterns[0]
            stats = best.stats[best.best_condition]
            edge = stats['win_rate'] - 0.5
            variance = stats['win_rate'] * (1 - stats['win_rate'])
            kelly = edge / variance if variance > 0 else 0

            # Trades per year estimate based on scale
            scale_days = best.best_condition[0]
            trades_per_year = 365 / scale_days

            print(f"\nBest pattern: {best.name}")
            print(f"Edge: {edge*100:.2f}% per trade")
            print(f"Trades/year: {trades_per_year:.0f}")

            for leverage in [1, 2, 5, 10]:
                safe_lev = min(leverage, kelly / 2)  # Half Kelly max
                ret_per_trade = edge * safe_lev

                results = []
                for years in [1, 3, 5, 10]:
                    capital = 100.0
                    n_trades = int(trades_per_year * years)
                    for _ in range(n_trades):
                        capital *= (1 + ret_per_trade)
                    results.append(capital)

                print(f"\n{leverage}x leverage (effective {safe_lev:.1f}x):")
                print(f"  1yr: ${results[0]:,.0f}")
                print(f"  3yr: ${results[1]:,.0f}")
                print(f"  5yr: ${results[2]:,.0f}")
                print(f"  10yr: ${results[3]:,.0f}")

    def generate_adaptive_formulas(self):
        """Generate Python code for adaptive formulas."""
        print("\n" + "="*70)
        print("GENERATING ADAPTIVE FORMULAS")
        print("="*70)

        valid_patterns = [p for p in self.patterns if p.is_valid]

        if not valid_patterns:
            print("No valid patterns to generate")
            return

        lines = [
            '"""',
            'MULTI-SCALE ADAPTIVE FORMULAS',
            '=' * 50,
            f'Generated: {datetime.now().isoformat()}',
            f'Data: {self.stats["days_loaded"]} days of Bitcoin history',
            f'Patterns validated: {len(valid_patterns)}',
            '',
            'Each formula only fires when its specific (scale, regime) conditions are met.',
            'This is the RenTech methodology: conditional edge exploitation.',
            '"""',
            '',
            'from typing import Dict, Optional',
            '',
            'SCALES = {1: "1d", 3: "3d", 7: "7d", 14: "14d", 30: "30d", 90: "90d"}',
            'REGIMES = {0: "ACCUMULATION", 1: "DISTRIBUTION", 2: "NEUTRAL", 3: "CAPITULATION", 4: "EUPHORIA"}',
            '',
            '',
            'ADAPTIVE_FORMULAS = [',
        ]

        for p in valid_patterns:
            lines.append('    {')
            lines.append(f'        "name": "{p.name}",')
            lines.append(f'        "direction": {p.direction},  # {"LONG" if p.direction == 1 else "SHORT"}')
            lines.append(f'        "total_occurrences": {p.total_occurrences},')
            lines.append(f'        "best_scale": {p.best_condition[0]},')
            lines.append(f'        "best_regime": {p.best_condition[1]},')
            lines.append(f'        "best_edge": {p.best_edge:.6f},')
            lines.append('        "conditions": {')

            for (scale, regime), stats in p.stats.items():
                if stats.get('win_rate', 0) >= 0.5075:
                    lines.append(f'            ({scale}, {regime}): {{"win_rate": {stats["win_rate"]:.4f}, "edge": {stats["edge"]:.4f}, "n": {stats["occurrences"]}, "ci_low": {stats.get("ci_low", 0):.4f}}},')

            lines.append('        },')
            lines.append('    },')

        lines.append(']')
        lines.append('')
        lines.append('')
        lines.append('def get_signal(pattern_name: str, scale: int, regime: int) -> Optional[Dict]:')
        lines.append('    """')
        lines.append('    Get signal if conditions are valid for this pattern.')
        lines.append('    Returns None if current (scale, regime) is not in valid conditions.')
        lines.append('    """')
        lines.append('    for formula in ADAPTIVE_FORMULAS:')
        lines.append('        if formula["name"] == pattern_name:')
        lines.append('            condition = formula["conditions"].get((scale, regime))')
        lines.append('            if condition:')
        lines.append('                return {')
        lines.append('                    "name": formula["name"],')
        lines.append('                    "direction": formula["direction"],')
        lines.append('                    "edge": condition["edge"],')
        lines.append('                    "win_rate": condition["win_rate"],')
        lines.append('                    "confidence": condition["ci_low"],')
        lines.append('                }')
        lines.append('    return None')

        code = '\n'.join(lines)

        output_path = Path(__file__).parent / 'adaptive_formulas_historical.py'
        with open(output_path, 'w') as f:
            f.write(code)

        print(f"Generated: {output_path}")
        print(f"Formulas: {len(valid_patterns)}")

    def run(self):
        """Run full discovery pipeline."""
        print("""
================================================================================
              HISTORICAL MULTI-SCALE ADAPTIVE DISCOVERY
================================================================================

"6+ years is amateur. Test on ALL of Bitcoin."
"What works for 1 second may not work for 2 seconds."

SCALES: 1d, 3d, 7d, 14d, 30d, 90d
METHOD: Train separate HMM per scale, find conditional patterns
TARGET: 50.75% win rate at EACH (regime, scale) combination
================================================================================
""")

        # Phase 1: Load data
        self.load_data()

        # Phase 2: Compute z-scores
        self.compute_z_scores()

        # Phase 3: Create multi-scale buckets
        self.create_multi_scale_buckets()

        # Phase 4: Train HMMs
        self.train_hmms()

        # Phase 5: Discover patterns
        self.discover_patterns()

        # Phase 6: Validate
        self.validate_patterns()

        # Phase 7: Generate formulas
        self.generate_adaptive_formulas()

        # Phase 8: Report
        self.generate_report()

        print("\n" + "="*70)
        print("HISTORICAL DISCOVERY COMPLETE")
        print("="*70)


def main():
    discovery = HistoricalDiscovery()
    discovery.run()


if __name__ == "__main__":
    main()
