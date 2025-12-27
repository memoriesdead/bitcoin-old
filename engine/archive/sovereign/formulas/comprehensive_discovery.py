#!/usr/bin/env python3
"""
COMPREHENSIVE BLOCKCHAIN PATTERN DISCOVERY
===========================================

"We don't start with models. We start with data."
- Jim Simons, Renaissance Technologies

This is the FULL RenTech-style discovery pipeline:

DATA SOURCES (22 features):
├── MEMPOOL (8): tx_count, size, fees, velocity, RBF, priority, congestion
├── BLOCKS (6): timing, fullness, fee_density, segwit, tx_rate
├── TRANSACTIONS (6): size, consolidation, batching, segwit, taproot, fees
├── UTXO (4): dormancy, hodl_score, dust, coin_days_destroyed
├── EXCHANGE (6): net_flow, inflow, outflow, whale, velocity
└── NETWORK (4): hash_rate, difficulty, fee_share, miner_revenue

SCALES: 1s, 5s, 10s, 30s, 1m, 5m

METHOD:
1. Collect ALL blockchain data at ALL scales
2. Build 22-dimensional feature vectors
3. Train per-scale HMMs (find regimes)
4. Discover patterns at each (scale, regime) combination
5. Validate with statistical rigor (>50.75% at each condition)
6. Generate adaptive formulas

TARGET: Find patterns that work at SPECIFIC (scale, regime) conditions

USAGE:
    python -m engine.sovereign.formulas.comprehensive_discovery --hours 8
"""

import os
import sys
import time
import json
import math
import sqlite3
import threading
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
import random

import numpy as np

# Import our modules
from engine.sovereign.formulas.blockchain_pipeline import (
    UnifiedPipeline,
    UnifiedFeatureVector,
    SCALES,
)
from engine.sovereign.formulas.multi_scale_discovery import (
    Regime,
    ConditionalPattern,
    SCALE_NAMES,
)

# Feature dimensions - COMPREHENSIVE 43-DIMENSIONAL VECTOR
N_FEATURES = 43
FEATURE_NAMES = [
    # Mempool (8)
    'mempool_tx_count', 'mempool_size', 'mempool_fee_median', 'mempool_fee_spread',
    'mempool_velocity', 'mempool_rbf_ratio', 'mempool_priority_ratio', 'mempool_congestion',
    # Blocks (6)
    'block_time_variance', 'block_fullness', 'block_fee_density',
    'block_segwit_ratio', 'block_tx_rate', 'blocks_since_last',
    # Transactions (6)
    'tx_batch_ratio', 'tx_consolidation_ratio', 'tx_segwit_ratio', 'tx_taproot_ratio',
    'tx_avg_size', 'tx_avg_fee_rate',
    # UTXO (4)
    'coin_days_destroyed', 'utxo_dormancy', 'utxo_hodl_score', 'utxo_dust_ratio',
    # Network (4)
    'hash_rate_momentum', 'difficulty_adjustment', 'fee_share', 'miner_revenue',
    # Lightning Network (5)
    'ln_channel_opens', 'ln_channel_closes', 'ln_capacity_change',
    'ln_total_capacity', 'ln_channel_count',
    # Ordinals/Inscriptions (4)
    'ordinals_count', 'ordinals_fees', 'brc20_count', 'op_return_count',
    # Whale Activity (4)
    'whale_movements', 'whale_accumulation', 'whale_awakening', 'whale_count',
    # Miner Behavior (5)
    'miner_empty_ratio', 'miner_timestamp_var', 'miner_concentration',
    'miner_fast_blocks', 'miner_fee_extraction',
    # Address Clustering (3)
    'cluster_count', 'cluster_merges', 'cluster_new_entities',
    # Exchange Flows (2)
    'exchange_net_flow', 'whale_net_flow',
    # Composite (2)
    'bullish_score', 'urgency_score',
]


###############################################################################
# ENHANCED HMM FOR HIGH-DIMENSIONAL DATA
###############################################################################

class HighDimensionalHMM:
    """
    HMM designed for 43-dimensional observation vectors.

    Captures ALL blockchain data:
    - Mempool dynamics (8 features)
    - Block structure (6 features)
    - Transaction patterns (6 features)
    - UTXO economics (4 features)
    - Network state (4 features)
    - Lightning Network (5 features)
    - Ordinals/Inscriptions (4 features)
    - Whale activity (4 features)
    - Miner behavior (5 features)
    - Address clustering (3 features)
    - Exchange flows (2 features)
    - Composite scores (2 features)

    Uses diagonal covariance for efficiency.
    """

    def __init__(self, scale: int, n_states: int = 5, n_features: int = N_FEATURES):
        self.scale = scale
        self.n_states = n_states
        self.n_features = n_features

        # HMM parameters
        self.A = self._init_transition_matrix()
        self.pi = np.array([0.1, 0.1, 0.6, 0.1, 0.1])

        # Emission parameters (means and variances for each feature per state)
        self.means = np.random.randn(n_states, n_features) * 0.1
        self.vars = np.ones((n_states, n_features)) * 0.5

        # State tracking
        self.belief = self.pi.copy()
        self.state_history: List[int] = []
        self.observation_history: List[np.ndarray] = []

        # Training data
        self.training_data: List[Tuple[np.ndarray, int]] = []

    def _init_transition_matrix(self) -> np.ndarray:
        """Initialize sticky transition matrix."""
        A = np.array([
            [0.80, 0.05, 0.10, 0.02, 0.03],
            [0.05, 0.80, 0.10, 0.03, 0.02],
            [0.15, 0.15, 0.60, 0.05, 0.05],
            [0.30, 0.10, 0.20, 0.35, 0.05],
            [0.10, 0.30, 0.20, 0.05, 0.35],
        ])
        return A

    def _emission_prob(self, state: int, obs: np.ndarray) -> float:
        """P(observation | state) using diagonal Gaussian."""
        diff = obs - self.means[state]
        var = self.vars[state]

        # Log probability (diagonal covariance)
        log_prob = -0.5 * np.sum(diff**2 / var)
        log_prob -= 0.5 * np.sum(np.log(2 * np.pi * var))

        return np.exp(np.clip(log_prob, -50, 0))

    def update(self, obs: np.ndarray) -> Dict:
        """Update belief with new observation."""
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

        # Store history
        self.observation_history.append(obs)
        self.state_history.append(state)
        if len(self.observation_history) > 10000:
            self.observation_history.pop(0)
            self.state_history.pop(0)

        # Signal mapping
        signal_map = {0: 1, 1: -1, 2: 0, 3: 1, 4: -1}

        return {
            'state': state,
            'regime': Regime(state).name,
            'confidence': confidence,
            'signal': signal_map[state],
            'probabilities': {Regime(i).name: float(self.belief[i]) for i in range(self.n_states)},
        }

    def add_training_sample(self, obs: np.ndarray, outcome: int):
        """Add labeled training sample."""
        self.training_data.append((obs, outcome))

    def train(self, n_iter: int = 50) -> float:
        """Train using EM algorithm."""
        if len(self.training_data) < 100:
            print(f"[HMM-{SCALE_NAMES[self.scale]}] Not enough data ({len(self.training_data)})")
            return 0.0

        observations = np.array([x[0] for x in self.training_data])
        T = len(observations)

        print(f"[HMM-{SCALE_NAMES[self.scale]}] Training on {T} samples, {self.n_features} features")

        prev_ll = float('-inf')

        for iteration in range(n_iter):
            # E-step: Forward-backward
            alpha = np.zeros((T, self.n_states))
            beta = np.zeros((T, self.n_states))
            scale = np.zeros(T)

            # Forward
            for s in range(self.n_states):
                alpha[0, s] = self.pi[s] * self._emission_prob(s, observations[0])
            scale[0] = alpha[0].sum() + 1e-10
            alpha[0] /= scale[0]

            for t in range(1, T):
                for s in range(self.n_states):
                    alpha[t, s] = sum(alpha[t-1, s2] * self.A[s2, s] for s2 in range(self.n_states))
                    alpha[t, s] *= self._emission_prob(s, observations[t])
                scale[t] = alpha[t].sum() + 1e-10
                alpha[t] /= scale[t]

            # Backward
            beta[T-1] = 1.0
            for t in range(T-2, -1, -1):
                for s in range(self.n_states):
                    beta[t, s] = sum(
                        self.A[s, s2] * self._emission_prob(s2, observations[t+1]) * beta[t+1, s2]
                        for s2 in range(self.n_states)
                    )
                beta[t] /= scale[t+1]

            # Log-likelihood
            ll = np.sum(np.log(scale))

            if abs(ll - prev_ll) < 1e-4:
                print(f"[HMM-{SCALE_NAMES[self.scale]}] Converged at iteration {iteration}")
                break
            prev_ll = ll

            # M-step
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=0) + 1e-10

            # Update pi
            self.pi = gamma[0] / gamma[0].sum()

            # Update A
            xi = np.zeros((self.n_states, self.n_states))
            for t in range(T-1):
                denom = sum(
                    alpha[t, i] * self.A[i, j] * self._emission_prob(j, observations[t+1]) * beta[t+1, j]
                    for i in range(self.n_states) for j in range(self.n_states)
                ) + 1e-10
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[i, j] += (
                            alpha[t, i] * self.A[i, j] *
                            self._emission_prob(j, observations[t+1]) * beta[t+1, j]
                        ) / denom

            for i in range(self.n_states):
                row_sum = xi[i].sum() + 1e-10
                self.A[i] = xi[i] / row_sum

            # Update means and variances
            for s in range(self.n_states):
                weights = gamma[:, s]
                weight_sum = weights.sum() + 1e-10

                self.means[s] = (weights[:, np.newaxis] * observations).sum(axis=0) / weight_sum

                diff = observations - self.means[s]
                self.vars[s] = (weights[:, np.newaxis] * diff**2).sum(axis=0) / weight_sum
                self.vars[s] = np.maximum(self.vars[s], 0.01)  # Minimum variance

        # Validate
        accuracy = self._validate()
        print(f"[HMM-{SCALE_NAMES[self.scale]}] Validation accuracy: {accuracy:.4f}")

        return accuracy

    def _validate(self) -> float:
        """Validate on training data."""
        self.belief = self.pi.copy()

        correct = 0
        total = 0

        for obs, outcome in self.training_data:
            if outcome == 0:
                continue

            result = self.update(obs)
            predicted = result['signal']

            if predicted != 0:
                if predicted == outcome:
                    correct += 1
                total += 1

        return correct / max(1, total)


###############################################################################
# COMPREHENSIVE DISCOVERY ENGINE
###############################################################################

class ComprehensiveDiscovery:
    """
    Full RenTech-style discovery using ALL blockchain data.
    """

    def __init__(self, db_path: str = None):
        self.pipeline = UnifiedPipeline(db_path)

        # Per-scale HMMs
        self.hmms: Dict[int, HighDimensionalHMM] = {
            scale: HighDimensionalHMM(scale) for scale in SCALES
        }

        # Discovered patterns
        self.patterns: List[ConditionalPattern] = []

        # Training data storage
        self.training_buffer: Dict[int, List[Tuple[np.ndarray, int]]] = {
            s: [] for s in SCALES
        }

        # Statistics
        self.stats = {
            'start_time': 0,
            'samples_collected': {s: 0 for s in SCALES},
            'patterns_discovered': 0,
            'patterns_validated': 0,
        }

        # Price tracking
        self.last_price = 0.0
        self.price_update_interval = 5
        self.last_price_update = 0

    def print_banner(self):
        print("""
================================================================================
          COMPREHENSIVE BLOCKCHAIN PATTERN DISCOVERY
================================================================================

"We look for things that can be replicated thousands of times."
- Jim Simons

DATA SOURCES (43 dimensions total):
├── MEMPOOL (8): tx_count, size, fees, velocity, RBF, priority, congestion
├── BLOCKS (6): timing, fullness, fee_density, segwit, tx_rate
├── TRANSACTIONS (6): size, consolidation, batching, segwit, taproot
├── UTXO (4): dormancy, hodl_score, dust, coin_days_destroyed
├── NETWORK (4): hash_rate, difficulty, fee_share, miner_revenue
├── LIGHTNING (5): channel_opens, closes, capacity_change, total_capacity
├── ORDINALS (4): inscriptions, fees, BRC-20, OP_RETURN
├── WHALES (4): movements, accumulation, awakening, count
├── MINERS (5): empty_ratio, timestamp_var, concentration, fast_blocks, MEV
├── CLUSTERS (3): entities, merges, new_entities
├── EXCHANGE (2): net_flow, whale_flow
└── COMPOSITE (2): bullish_score, urgency_score

SCALES: 1s, 5s, 10s, 30s, 1m, 5m (6 timeframes)
FEATURES: 43 dimensions per observation
STATES: 5 (ACCUMULATION, DISTRIBUTION, NEUTRAL, CAPITULATION, EUPHORIA)

TARGET: 50.75% win rate at EACH (scale, regime) combination
================================================================================
""")

    def _get_price(self) -> float:
        """Get current BTC price."""
        now = time.time()
        if now - self.last_price_update < self.price_update_interval and self.last_price > 0:
            return self.last_price

        try:
            from urllib.request import urlopen, Request
            url = 'https://api.exchange.coinbase.com/products/BTC-USD/ticker'
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=5) as resp:
                self.last_price = float(json.loads(resp.read().decode())['price'])
                self.last_price_update = now
        except:
            pass

        return self.last_price if self.last_price > 0 else 97000.0

    def collect_data(self, duration: int = 3600):
        """
        Collect data from all sources for specified duration.
        """
        print(f"\n[COLLECT] Starting {duration}s ({duration/3600:.1f}h) collection")
        print(f"[COLLECT] Features: {N_FEATURES} dimensions")
        print(f"[COLLECT] Scales: {', '.join(SCALE_NAMES[s] for s in SCALES)}")

        # Start pipeline
        price = self._get_price()
        self.pipeline.update_price(price)
        self.pipeline.start_background_updates(interval=0.5)

        # Also start exchange flow tracking
        from engine.sovereign.blockchain import PerExchangeBlockchainFeed, ExchangeTick

        exchange_net_flow = [0.0]
        whale_net_flow = [0.0]

        def on_tick(tick: ExchangeTick):
            exchange_net_flow[0] += tick.volume * tick.direction
            if tick.volume >= 100:
                whale_net_flow[0] += tick.volume * tick.direction

            # Update pipeline's exchange features
            for scale in SCALES:
                features = self.pipeline.current_features.get(scale)
                if features:
                    features.exchange_net_flow = exchange_net_flow[0]
                    features.whale_net_flow = whale_net_flow[0]

        feed = PerExchangeBlockchainFeed(on_tick=on_tick)
        feed.set_reference_price(price)

        if not feed.start():
            print("[COLLECT] Warning: Exchange feed failed to start")

        start_time = time.time()
        self.stats['start_time'] = start_time

        last_print = 0
        prices = []

        try:
            while time.time() - start_time < duration:
                time.sleep(1)

                # Update price
                current_price = self._get_price()
                self.pipeline.update_price(current_price)
                prices.append(current_price)

                # Collect completed buckets
                completed = self.pipeline.update()

                for scale, features in completed.items():
                    # Create feature vector
                    obs = self.pipeline.get_feature_vector(scale)

                    # Determine outcome from price change
                    if features.price_change > 0.0001:
                        outcome = 1
                    elif features.price_change < -0.0001:
                        outcome = -1
                    else:
                        outcome = 0

                    # Store for training
                    if outcome != 0:
                        self.training_buffer[scale].append((obs, outcome))
                        self.hmms[scale].add_training_sample(obs, outcome)
                        self.stats['samples_collected'][scale] += 1

                # Print progress
                elapsed = time.time() - start_time
                if elapsed - last_print >= 60:
                    last_print = elapsed
                    remaining = duration - elapsed
                    print(f"[COLLECT] {elapsed/60:.0f}m elapsed | {remaining/60:.0f}m remaining | "
                          f"${current_price:,.0f}")
                    for scale in [10, 60]:
                        print(f"  {SCALE_NAMES[scale]}: {self.stats['samples_collected'][scale]} samples")

        except KeyboardInterrupt:
            print("\n[COLLECT] Interrupted")
        finally:
            feed.stop()
            self.pipeline.stop()

        print(f"\n[COLLECT] Complete. Samples per scale:")
        for scale in SCALES:
            print(f"  {SCALE_NAMES[scale]}: {self.stats['samples_collected'][scale]}")

    def train_hmms(self) -> Dict[int, float]:
        """Train HMM for each scale."""
        print("\n" + "="*60)
        print("TRAINING HIGH-DIMENSIONAL HMMs")
        print("="*60)

        accuracies = {}

        for scale in SCALES:
            print(f"\n--- {SCALE_NAMES[scale]} ---")
            accuracy = self.hmms[scale].train()
            accuracies[scale] = accuracy

        return accuracies

    def discover_patterns(self, min_occurrences: int = 20) -> List[ConditionalPattern]:
        """Discover patterns at each (scale, regime) combination."""
        print("\n" + "="*60)
        print("DISCOVERING CONDITIONAL PATTERNS")
        print("="*60)

        all_patterns = []

        for scale in SCALES:
            print(f"\n--- {SCALE_NAMES[scale]} ---")

            hmm = self.hmms[scale]
            if len(hmm.state_history) < 50:
                print(f"[PATTERN] Not enough history at {SCALE_NAMES[scale]}")
                continue

            # Get state sequence and outcomes
            states = hmm.state_history
            outcomes = [x[1] for x in hmm.training_data[-len(states):]]

            if len(outcomes) != len(states):
                outcomes = outcomes[:len(states)] if len(outcomes) > len(states) else outcomes + [0] * (len(states) - len(outcomes))

            # Find patterns (subsequences of length 2-4)
            pattern_stats: Dict[str, Dict] = defaultdict(lambda: {
                'occurrences': 0, 'wins': 0, 'losses': 0,
                'by_regime': defaultdict(lambda: {'occ': 0, 'wins': 0, 'losses': 0})
            })

            for length in range(2, 5):
                for i in range(len(states) - length):
                    seq = tuple(states[i:i+length])
                    outcome = outcomes[i + length - 1] if i + length - 1 < len(outcomes) else 0

                    if outcome == 0:
                        continue

                    # Direction from last state
                    last_state = seq[-1]
                    if last_state in [0, 3]:
                        direction = 1
                    elif last_state in [1, 4]:
                        direction = -1
                    else:
                        continue

                    key = f"{seq}_{direction}"
                    pattern_stats[key]['occurrences'] += 1
                    pattern_stats[key]['sequence'] = list(seq)
                    pattern_stats[key]['direction'] = direction

                    current_regime = states[i + length - 1]
                    pattern_stats[key]['by_regime'][current_regime]['occ'] += 1

                    if (direction == 1 and outcome == 1) or (direction == -1 and outcome == -1):
                        pattern_stats[key]['wins'] += 1
                        pattern_stats[key]['by_regime'][current_regime]['wins'] += 1
                    else:
                        pattern_stats[key]['losses'] += 1
                        pattern_stats[key]['by_regime'][current_regime]['losses'] += 1

            # Create pattern objects
            for key, stats in pattern_stats.items():
                if stats['occurrences'] < min_occurrences:
                    continue

                win_rate = stats['wins'] / max(1, stats['occurrences'])

                pattern = ConditionalPattern(
                    name=f"COMP_{SCALE_NAMES[scale]}_{len(all_patterns)}",
                    sequence=stats['sequence'],
                    direction=stats['direction'],
                    total_occurrences=stats['occurrences'],
                    overall_win_rate=win_rate,
                )

                for regime, regime_stats in stats['by_regime'].items():
                    if regime_stats['occ'] >= 5:
                        wr = regime_stats['wins'] / max(1, regime_stats['occ'])
                        pattern.stats[(scale, regime)] = {
                            'occurrences': regime_stats['occ'],
                            'wins': regime_stats['wins'],
                            'losses': regime_stats['losses'],
                            'win_rate': wr,
                            'edge': wr - 0.5,
                        }

                        if wr >= 0.5075:
                            pattern.valid_scales.add(scale)
                            pattern.valid_regimes.add(regime)

                all_patterns.append(pattern)

            count = len([p for p in all_patterns if scale in p.valid_scales])
            print(f"[PATTERN] {count} valid patterns at {SCALE_NAMES[scale]}")

        self.patterns = all_patterns
        self.stats['patterns_discovered'] = len(all_patterns)

        return all_patterns

    def validate_patterns(self, n_bootstrap: int = 1000) -> List[ConditionalPattern]:
        """Validate patterns with Monte Carlo."""
        print("\n" + "="*60)
        print("MONTE CARLO VALIDATION")
        print("="*60)

        validated = []

        for pattern in self.patterns:
            best_edge = 0.0
            best_condition = (0, 0)

            for (scale, regime), stats in pattern.stats.items():
                if stats['occurrences'] < 10:
                    continue

                # Bootstrap
                bootstrap_wrs = []
                for _ in range(n_bootstrap):
                    sample_wins = sum(random.random() < stats['win_rate'] for _ in range(stats['occurrences']))
                    bootstrap_wrs.append(sample_wins / stats['occurrences'])

                bootstrap_wrs.sort()
                ci_low = bootstrap_wrs[int(n_bootstrap * 0.025)]

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
        print(f"\n[VALIDATE] {len(validated)} patterns validated")

        return validated

    def generate_formulas(self) -> str:
        """Generate Python formulas."""
        valid = [p for p in self.patterns if p.is_valid]

        if not valid:
            print("[FORMULA] No valid patterns")
            return ""

        lines = [
            '"""',
            'COMPREHENSIVE BLOCKCHAIN FORMULAS',
            '=' * 40,
            f'Generated: {datetime.now().isoformat()}',
            f'Features: {N_FEATURES} dimensions',
            f'Patterns: {len(valid)}',
            '"""',
            '',
            f'FEATURE_NAMES = {FEATURE_NAMES}',
            '',
            'COMPREHENSIVE_FORMULAS = [',
        ]

        for p in valid:
            lines.append('    {')
            lines.append(f'        "name": "{p.name}",')
            lines.append(f'        "sequence": {p.sequence},')
            lines.append(f'        "direction": {p.direction},')
            lines.append(f'        "best_scale": {p.best_condition[0]},')
            lines.append(f'        "best_regime": {p.best_condition[1]},')
            lines.append(f'        "best_edge": {p.best_edge:.6f},')
            lines.append(f'        "valid_scales": {list(p.valid_scales)},')
            lines.append(f'        "valid_regimes": {list(p.valid_regimes)},')
            lines.append('    },')

        lines.append(']')

        code = '\n'.join(lines)

        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'comprehensive_formulas.py'
        )
        with open(output_path, 'w') as f:
            f.write(code)

        print(f"[FORMULA] Generated {output_path}")
        return code

    def generate_report(self) -> str:
        """Generate comprehensive report."""
        valid = [p for p in self.patterns if p.is_valid]

        lines = [
            '=' * 80,
            'COMPREHENSIVE DISCOVERY REPORT',
            '=' * 80,
            f'Generated: {datetime.now().isoformat()}',
            f'Features: {N_FEATURES} dimensions',
            '',
            '## SAMPLES COLLECTED',
        ]

        for scale in SCALES:
            lines.append(f'  {SCALE_NAMES[scale]}: {self.stats["samples_collected"][scale]}')

        lines.append('')
        lines.append('## PATTERNS')
        lines.append(f'Discovered: {self.stats["patterns_discovered"]}')
        lines.append(f'Validated: {self.stats["patterns_validated"]}')
        lines.append('')

        if valid:
            lines.append('## TOP VALIDATED PATTERNS')
            sorted_patterns = sorted(valid, key=lambda p: p.best_edge, reverse=True)

            for i, p in enumerate(sorted_patterns[:10]):
                dir_str = "LONG" if p.direction == 1 else "SHORT"
                lines.append(f'\n{i+1}. {p.name}')
                lines.append(f'   Direction: {dir_str}')
                lines.append(f'   Sequence: {p.sequence}')
                lines.append(f'   Best: {SCALE_NAMES[p.best_condition[0]]} + {Regime(p.best_condition[1]).name}')
                lines.append(f'   Edge: {p.best_edge:.4f} ({p.best_edge*100:.2f}%)')

        lines.append('')
        lines.append('=' * 80)

        report = '\n'.join(lines)

        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', '..', 'data', 'comprehensive_report.md'
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

        print(report)
        return report

    def run(self, duration: int = 3600):
        """Run full discovery pipeline."""
        self.print_banner()

        # Collect
        self.collect_data(duration)

        # Train
        self.train_hmms()

        # Discover
        self.discover_patterns()

        # Validate
        self.validate_patterns()

        # Generate
        self.generate_formulas()
        self.generate_report()

        print("\n" + "="*60)
        print("COMPREHENSIVE DISCOVERY COMPLETE")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Blockchain Discovery')
    parser.add_argument('--hours', type=float, default=1,
                       help='Hours to collect (default: 1)')
    parser.add_argument('--train-only', action='store_true',
                       help='Only train on existing data')

    args = parser.parse_args()

    discovery = ComprehensiveDiscovery()

    if args.train_only:
        discovery.train_hmms()
        discovery.discover_patterns()
        discovery.validate_patterns()
        discovery.generate_formulas()
        discovery.generate_report()
    else:
        discovery.run(duration=int(args.hours * 3600))


if __name__ == "__main__":
    main()
