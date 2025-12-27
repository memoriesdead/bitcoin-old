#!/usr/bin/env python3
"""
MULTI-SCALE ADAPTIVE PATTERN DISCOVERY
=======================================

"What works for 1 second may not work for 2 seconds."

This system applies the Fractal Market Hypothesis to blockchain data:
1. Analyze flows at MULTIPLE timeframes (1s, 5s, 10s, 30s, 1m, 5m)
2. Train separate HMMs for each scale
3. Use wavelet decomposition to identify dominant frequencies
4. Discover patterns that work at SPECIFIC (regime, scale) combinations
5. Validate each pattern CONDITIONALLY
6. Generate formulas only for proven edges

METHODOLOGY:
- Pattern X may work at 1s but fail at 5s
- Pattern Y may work ONLY when regime=ACCUMULATION
- We find AND validate these conditional edges

TARGET: 50.75% win rate at EACH (regime, scale) combination

USAGE:
    python -m engine.sovereign.formulas.multi_scale_discovery

Based on:
- Fractal Market Hypothesis (Peters, 1994)
- Wavelet decomposition for multi-scale analysis
- Hidden Markov Models for regime detection
- Renaissance Technologies methodology
"""

import os
import sys
import time
import json
import math
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
import random

# Scales to analyze (in seconds)
SCALES = [1, 5, 10, 30, 60, 300]  # 1s, 5s, 10s, 30s, 1m, 5m
SCALE_NAMES = {1: '1s', 5: '5s', 10: '10s', 30: '30s', 60: '1m', 300: '5m'}


class Regime(Enum):
    """Market regimes detected from blockchain flow."""
    ACCUMULATION = 0      # Net outflow -> LONG
    DISTRIBUTION = 1      # Net inflow -> SHORT
    NEUTRAL = 2           # Balanced
    CAPITULATION = 3      # Panic selling -> Contrarian LONG
    EUPHORIA = 4          # Greed/FOMO -> Contrarian SHORT


@dataclass
class ScaledFlowBucket:
    """Aggregated flow data for a specific time bucket."""
    timestamp: float
    scale: int  # seconds

    # Flow metrics
    net_flow: float = 0.0  # outflow - inflow
    total_volume: float = 0.0
    inflow: float = 0.0
    outflow: float = 0.0
    tx_count: int = 0

    # Whale metrics (>100 BTC)
    whale_net: float = 0.0
    whale_count: int = 0

    # Price metrics
    price_start: float = 0.0
    price_end: float = 0.0
    price_change: float = 0.0  # percentage

    # Derived features
    flow_imbalance: float = 0.0  # net_flow / total_volume
    flow_velocity: float = 0.0  # abs(net_flow) / scale

    # Outcome (for training)
    outcome: int = 0  # +1 price up, -1 price down, 0 unknown


@dataclass
class MultiScaleObservation:
    """Observation vector at a point in time across all scales."""
    timestamp: float
    price: float

    # Per-scale data
    buckets: Dict[int, ScaledFlowBucket] = field(default_factory=dict)

    # Wavelet coefficients (computed)
    wavelet_power: Dict[int, float] = field(default_factory=dict)
    dominant_scale: int = 0


@dataclass
class ConditionalPattern:
    """A pattern that works under specific conditions."""
    name: str
    sequence: List[int]  # HMM state sequence
    direction: int  # +1 LONG, -1 SHORT

    # Conditions where this pattern is valid
    valid_scales: Set[int] = field(default_factory=set)  # which scales
    valid_regimes: Set[int] = field(default_factory=set)  # which regimes

    # Statistics per (scale, regime)
    stats: Dict[Tuple[int, int], Dict] = field(default_factory=dict)

    # Overall
    total_occurrences: int = 0
    overall_win_rate: float = 0.0

    # Validation
    is_valid: bool = False
    best_condition: Tuple[int, int] = (0, 0)  # (scale, regime) with best edge
    best_edge: float = 0.0


class MultiScaleHMM:
    """
    Hidden Markov Model that operates at a specific scale.

    Each scale has its own HMM because:
    - 1s patterns have different dynamics than 5m patterns
    - Transition probabilities differ by timeframe
    - Emission distributions differ by timeframe
    """

    def __init__(self, scale: int, n_states: int = 5):
        self.scale = scale
        self.n_states = n_states

        # HMM parameters (will be trained)
        self.A = self._init_transition_matrix()
        self.emission_means = self._init_emission_means()
        self.emission_vars = {s: [0.2, 0.2, 0.15, 0.2] for s in range(n_states)}
        self.pi = np.array([0.1, 0.1, 0.6, 0.1, 0.1])  # Initial state probs

        # State tracking
        self.belief = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
        self.state_history: List[int] = []
        self.observation_history: List[List[float]] = []

        # Training data
        self.training_observations: List[np.ndarray] = []
        self.training_outcomes: List[int] = []

    def _init_transition_matrix(self) -> np.ndarray:
        """Initialize with sticky states."""
        return np.array([
            [0.80, 0.05, 0.10, 0.02, 0.03],
            [0.05, 0.80, 0.10, 0.03, 0.02],
            [0.15, 0.15, 0.60, 0.05, 0.05],
            [0.30, 0.10, 0.20, 0.35, 0.05],
            [0.10, 0.30, 0.20, 0.05, 0.35],
        ])

    def _init_emission_means(self) -> Dict[int, List[float]]:
        """Initial emission means [flow_imbalance, velocity, whale_ratio, volume_z]."""
        return {
            0: [0.5, 0.3, 0.3, 0.5],    # ACCUMULATION
            1: [-0.5, 0.3, 0.3, 0.5],   # DISTRIBUTION
            2: [0.0, 0.1, 0.2, 0.0],    # NEUTRAL
            3: [-0.8, 0.8, 0.5, 1.5],   # CAPITULATION
            4: [0.8, 0.8, 0.5, 1.5],    # EUPHORIA
        }

    def _emission_prob(self, state: int, obs: List[float]) -> float:
        """P(observation | state) using Gaussian."""
        means = self.emission_means[state]
        vars = self.emission_vars[state]

        log_prob = 0
        for i in range(min(len(obs), len(means))):
            diff = obs[i] - means[i]
            log_prob -= 0.5 * (diff ** 2) / vars[i]
            log_prob -= 0.5 * math.log(2 * math.pi * vars[i])

        return math.exp(max(-50, log_prob))

    def update(self, bucket: ScaledFlowBucket) -> Dict:
        """Update belief with new bucket observation."""
        obs = [
            bucket.flow_imbalance,
            bucket.flow_velocity,
            bucket.whale_net / max(0.01, bucket.total_volume),
            bucket.total_volume / 100.0,  # Normalized volume
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

    def add_training_sample(self, bucket: ScaledFlowBucket, outcome: int):
        """Add labeled sample for training."""
        obs = np.array([
            bucket.flow_imbalance,
            bucket.flow_velocity,
            bucket.whale_net / max(0.01, bucket.total_volume),
            bucket.total_volume / 100.0,
        ])
        self.training_observations.append(obs)
        self.training_outcomes.append(outcome)

    def train_baum_welch(self, n_iter: int = 100, tol: float = 1e-4) -> float:
        """Train HMM using Baum-Welch algorithm."""
        if len(self.training_observations) < 100:
            print(f"[HMM-{SCALE_NAMES[self.scale]}] Not enough training data ({len(self.training_observations)} samples)")
            return 0.0

        observations = np.array(self.training_observations)
        T, D = observations.shape
        N = self.n_states

        print(f"[HMM-{SCALE_NAMES[self.scale]}] Training on {T} samples...")

        prev_ll = float('-inf')

        for iteration in range(n_iter):
            # E-step: Forward-backward
            alpha = np.zeros((T, N))
            beta = np.zeros((T, N))
            scale = np.zeros(T)

            # Forward
            for s in range(N):
                alpha[0, s] = self.pi[s] * self._emission_prob(s, observations[0])
            scale[0] = alpha[0].sum()
            if scale[0] > 0:
                alpha[0] /= scale[0]

            for t in range(1, T):
                for s in range(N):
                    alpha[t, s] = sum(alpha[t-1, s2] * self.A[s2, s] for s2 in range(N))
                    alpha[t, s] *= self._emission_prob(s, observations[t])
                scale[t] = alpha[t].sum()
                if scale[t] > 0:
                    alpha[t] /= scale[t]

            # Backward
            beta[T-1] = 1.0
            for t in range(T-2, -1, -1):
                for s in range(N):
                    beta[t, s] = sum(
                        self.A[s, s2] * self._emission_prob(s2, observations[t+1]) * beta[t+1, s2]
                        for s2 in range(N)
                    )
                if scale[t+1] > 0:
                    beta[t] /= scale[t+1]

            # Log-likelihood
            ll = sum(math.log(max(1e-300, s)) for s in scale)

            if abs(ll - prev_ll) < tol:
                print(f"[HMM-{SCALE_NAMES[self.scale]}] Converged at iteration {iteration}")
                break
            prev_ll = ll

            # M-step: Update parameters
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=0)
            gamma_sum = np.maximum(gamma_sum, 1e-10)

            # Update initial probs
            self.pi = gamma[0] / gamma[0].sum()

            # Update transition matrix
            xi = np.zeros((N, N))
            for t in range(T-1):
                denom = sum(
                    alpha[t, i] * self.A[i, j] * self._emission_prob(j, observations[t+1]) * beta[t+1, j]
                    for i in range(N) for j in range(N)
                )
                if denom > 0:
                    for i in range(N):
                        for j in range(N):
                            xi[i, j] += (
                                alpha[t, i] * self.A[i, j] *
                                self._emission_prob(j, observations[t+1]) * beta[t+1, j]
                            ) / denom

            for i in range(N):
                row_sum = xi[i].sum()
                if row_sum > 0:
                    self.A[i] = xi[i] / row_sum

            # Update emission means
            for s in range(N):
                weight_sum = gamma[:, s].sum()
                if weight_sum > 0:
                    for d in range(D):
                        self.emission_means[s][d] = (gamma[:, s] * observations[:, d]).sum() / weight_sum
                        var = (gamma[:, s] * (observations[:, d] - self.emission_means[s][d])**2).sum() / weight_sum
                        self.emission_vars[s][d] = max(0.01, var)

        # Validate
        accuracy = self._validate()
        print(f"[HMM-{SCALE_NAMES[self.scale]}] Training complete. Validation accuracy: {accuracy:.4f}")

        return accuracy

    def _validate(self) -> float:
        """Validate on training data (should do holdout in production)."""
        if not self.training_observations:
            return 0.0

        correct = 0
        total = 0

        # Reset belief
        self.belief = self.pi.copy()

        for i, (obs, outcome) in enumerate(zip(self.training_observations, self.training_outcomes)):
            if outcome == 0:
                continue

            # Get prediction
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


class WaveletAnalyzer:
    """
    Simplified wavelet analysis for detecting dominant timeframes.

    Uses Haar wavelet for efficiency.
    """

    def __init__(self, scales: List[int] = SCALES):
        self.scales = scales
        self.history: Dict[int, deque] = {s: deque(maxlen=100) for s in scales}

    def add_observation(self, scale: int, value: float):
        """Add observation at specific scale."""
        if scale in self.history:
            self.history[scale].append(value)

    def compute_power(self) -> Dict[int, float]:
        """Compute wavelet power at each scale."""
        power = {}

        for scale in self.scales:
            data = list(self.history[scale])
            if len(data) < 4:
                power[scale] = 0.0
                continue

            # Haar wavelet: difference of adjacent averages
            coeffs = []
            for i in range(0, len(data) - 1, 2):
                if i + 1 < len(data):
                    coeffs.append(data[i+1] - data[i])

            if coeffs:
                power[scale] = np.var(coeffs) if len(coeffs) > 1 else 0.0
            else:
                power[scale] = 0.0

        return power

    def get_dominant_scale(self) -> int:
        """Get scale with highest power (most activity)."""
        power = self.compute_power()
        if not power or all(p == 0 for p in power.values()):
            return self.scales[2]  # Default to 10s

        return max(power.keys(), key=lambda k: power[k])


class MultiScaleDatabase:
    """SQLite database for multi-scale flow data."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', '..', '..', 'data', 'multi_scale_flows.db'
            )

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Flow buckets at each scale
        c.execute('''
            CREATE TABLE IF NOT EXISTS flow_buckets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                scale INTEGER,
                net_flow REAL,
                total_volume REAL,
                inflow REAL,
                outflow REAL,
                tx_count INTEGER,
                whale_net REAL,
                whale_count INTEGER,
                price_start REAL,
                price_end REAL,
                price_change REAL,
                flow_imbalance REAL,
                flow_velocity REAL,
                outcome INTEGER DEFAULT 0,
                UNIQUE(timestamp, scale)
            )
        ''')

        # Discovered patterns
        c.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                sequence TEXT,
                direction INTEGER,
                valid_scales TEXT,
                valid_regimes TEXT,
                total_occurrences INTEGER,
                overall_win_rate REAL,
                best_scale INTEGER,
                best_regime INTEGER,
                best_edge REAL,
                is_valid INTEGER,
                created_at REAL
            )
        ''')

        # Pattern statistics per (scale, regime)
        c.execute('''
            CREATE TABLE IF NOT EXISTS pattern_stats (
                pattern_name TEXT,
                scale INTEGER,
                regime INTEGER,
                occurrences INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                edge REAL,
                p_value REAL,
                PRIMARY KEY (pattern_name, scale, regime)
            )
        ''')

        # Trained HMM models
        c.execute('''
            CREATE TABLE IF NOT EXISTS hmm_models (
                scale INTEGER PRIMARY KEY,
                n_states INTEGER,
                transition_matrix TEXT,
                emission_means TEXT,
                emission_vars TEXT,
                initial_probs TEXT,
                validation_accuracy REAL,
                training_samples INTEGER,
                trained_at REAL
            )
        ''')

        # Price history
        c.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                timestamp REAL PRIMARY KEY,
                price REAL
            )
        ''')

        conn.commit()
        conn.close()

        print(f"[DB] Initialized: {self.db_path}")

    def add_bucket(self, bucket: ScaledFlowBucket):
        """Add a flow bucket."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        try:
            c.execute('''
                INSERT OR REPLACE INTO flow_buckets
                (timestamp, scale, net_flow, total_volume, inflow, outflow,
                 tx_count, whale_net, whale_count, price_start, price_end,
                 price_change, flow_imbalance, flow_velocity, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                bucket.timestamp, bucket.scale, bucket.net_flow, bucket.total_volume,
                bucket.inflow, bucket.outflow, bucket.tx_count, bucket.whale_net,
                bucket.whale_count, bucket.price_start, bucket.price_end,
                bucket.price_change, bucket.flow_imbalance, bucket.flow_velocity,
                bucket.outcome
            ))
            conn.commit()
        except Exception as e:
            print(f"[DB] Error adding bucket: {e}")
        finally:
            conn.close()

    def add_price(self, timestamp: float, price: float):
        """Add price observation."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute('INSERT OR REPLACE INTO prices (timestamp, price) VALUES (?, ?)',
                     (timestamp, price))
            conn.commit()
        finally:
            conn.close()

    def get_price_at(self, timestamp: float) -> Optional[float]:
        """Get price closest to timestamp."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT price FROM prices
            WHERE timestamp <= ?
            ORDER BY timestamp DESC LIMIT 1
        ''', (timestamp,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None

    def get_buckets(self, scale: int, min_volume: float = 0.0) -> List[ScaledFlowBucket]:
        """Get all buckets for a scale."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT timestamp, scale, net_flow, total_volume, inflow, outflow,
                   tx_count, whale_net, whale_count, price_start, price_end,
                   price_change, flow_imbalance, flow_velocity, outcome
            FROM flow_buckets
            WHERE scale = ? AND total_volume >= ?
            ORDER BY timestamp
        ''', (scale, min_volume))

        buckets = []
        for row in c.fetchall():
            bucket = ScaledFlowBucket(
                timestamp=row[0], scale=row[1], net_flow=row[2],
                total_volume=row[3], inflow=row[4], outflow=row[5],
                tx_count=row[6], whale_net=row[7], whale_count=row[8],
                price_start=row[9], price_end=row[10], price_change=row[11],
                flow_imbalance=row[12], flow_velocity=row[13], outcome=row[14]
            )
            buckets.append(bucket)

        conn.close()
        return buckets

    def save_hmm(self, scale: int, hmm: MultiScaleHMM, accuracy: float):
        """Save trained HMM."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            INSERT OR REPLACE INTO hmm_models
            (scale, n_states, transition_matrix, emission_means, emission_vars,
             initial_probs, validation_accuracy, training_samples, trained_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            scale, hmm.n_states,
            json.dumps(hmm.A.tolist()),
            json.dumps(hmm.emission_means),
            json.dumps({str(k): v for k, v in hmm.emission_vars.items()}),
            json.dumps(hmm.pi.tolist()),
            accuracy,
            len(hmm.training_observations),
            time.time()
        ))

        conn.commit()
        conn.close()

    def load_hmm(self, scale: int) -> Optional[MultiScaleHMM]:
        """Load trained HMM."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM hmm_models WHERE scale = ?', (scale,))
        row = c.fetchone()
        conn.close()

        if not row:
            return None

        hmm = MultiScaleHMM(scale=scale, n_states=row[1])
        hmm.A = np.array(json.loads(row[2]))
        hmm.emission_means = json.loads(row[3])
        hmm.emission_vars = {int(k): v for k, v in json.loads(row[4]).items()}
        hmm.pi = np.array(json.loads(row[5]))

        return hmm

    def get_bucket_count(self, scale: int) -> int:
        """Get count of buckets for a scale."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM flow_buckets WHERE scale = ?', (scale,))
        count = c.fetchone()[0]
        conn.close()
        return count

    def save_pattern(self, pattern: ConditionalPattern):
        """Save discovered pattern."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            INSERT OR REPLACE INTO patterns
            (name, sequence, direction, valid_scales, valid_regimes,
             total_occurrences, overall_win_rate, best_scale, best_regime,
             best_edge, is_valid, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.name,
            json.dumps(pattern.sequence),
            pattern.direction,
            json.dumps(list(pattern.valid_scales)),
            json.dumps(list(pattern.valid_regimes)),
            pattern.total_occurrences,
            pattern.overall_win_rate,
            pattern.best_condition[0],
            pattern.best_condition[1],
            pattern.best_edge,
            1 if pattern.is_valid else 0,
            time.time()
        ))

        # Save per-condition stats
        for (scale, regime), stats in pattern.stats.items():
            c.execute('''
                INSERT OR REPLACE INTO pattern_stats
                (pattern_name, scale, regime, occurrences, wins, losses,
                 win_rate, edge, p_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern.name, scale, regime,
                stats.get('occurrences', 0),
                stats.get('wins', 0),
                stats.get('losses', 0),
                stats.get('win_rate', 0.5),
                stats.get('edge', 0.0),
                stats.get('p_value', 1.0)
            ))

        conn.commit()
        conn.close()


class MultiScaleCollector:
    """
    Collects blockchain flow data and aggregates at multiple scales.
    """

    def __init__(self, db: MultiScaleDatabase):
        self.db = db
        self.scales = SCALES

        # Current bucket being built for each scale
        self.current_buckets: Dict[int, ScaledFlowBucket] = {}
        self.bucket_start_times: Dict[int, float] = {}

        # Price tracking (initialize BEFORE buckets)
        self.last_price = 0.0
        self.price_history: deque = deque(maxlen=1000)

        # Initialize buckets
        now = time.time()
        for scale in self.scales:
            self._start_new_bucket(scale, now)

    def _start_new_bucket(self, scale: int, timestamp: float):
        """Start a new bucket for a scale."""
        # Align to scale boundary
        aligned_ts = (int(timestamp) // scale) * scale

        self.current_buckets[scale] = ScaledFlowBucket(
            timestamp=aligned_ts,
            scale=scale,
            price_start=self.last_price
        )
        self.bucket_start_times[scale] = aligned_ts

    def add_flow(self, exchange: str, direction: int, btc: float,
                 timestamp: float, price: float):
        """
        Add a flow event.

        Args:
            exchange: Exchange name
            direction: +1 outflow, -1 inflow
            btc: Amount
            timestamp: Unix timestamp
            price: Current BTC price
        """
        self.last_price = price
        self.price_history.append((timestamp, price))
        self.db.add_price(timestamp, price)

        is_whale = btc >= 100

        for scale in self.scales:
            bucket = self.current_buckets[scale]

            # Check if bucket is complete
            if timestamp >= self.bucket_start_times[scale] + scale:
                # Finalize bucket
                bucket.price_end = price
                if bucket.price_start > 0:
                    bucket.price_change = (bucket.price_end - bucket.price_start) / bucket.price_start

                # Calculate derived features
                if bucket.total_volume > 0:
                    bucket.flow_imbalance = bucket.net_flow / bucket.total_volume
                bucket.flow_velocity = abs(bucket.net_flow) / scale

                # Determine outcome
                if bucket.price_change > 0.0001:  # >0.01%
                    bucket.outcome = 1
                elif bucket.price_change < -0.0001:
                    bucket.outcome = -1
                else:
                    bucket.outcome = 0

                # Save
                if bucket.total_volume > 0:
                    self.db.add_bucket(bucket)

                # Start new bucket
                self._start_new_bucket(scale, timestamp)
                bucket = self.current_buckets[scale]

            # Add to bucket
            bucket.tx_count += 1
            bucket.total_volume += btc

            if direction == 1:  # Outflow
                bucket.outflow += btc
                bucket.net_flow += btc
            else:  # Inflow
                bucket.inflow += btc
                bucket.net_flow -= btc

            if is_whale:
                bucket.whale_count += 1
                bucket.whale_net += btc * direction

    def flush_all(self):
        """Flush all current buckets."""
        for scale in self.scales:
            bucket = self.current_buckets[scale]
            if bucket.total_volume > 0:
                bucket.price_end = self.last_price
                if bucket.price_start > 0:
                    bucket.price_change = (bucket.price_end - bucket.price_start) / bucket.price_start
                if bucket.total_volume > 0:
                    bucket.flow_imbalance = bucket.net_flow / bucket.total_volume
                bucket.flow_velocity = abs(bucket.net_flow) / scale
                self.db.add_bucket(bucket)


class MultiScaleDiscovery:
    """
    Main discovery engine for multi-scale pattern analysis.

    Implements the full RenTech-style methodology:
    1. Collect at multiple scales
    2. Train per-scale HMMs
    3. Discover conditional patterns
    4. Validate at each (scale, regime) combination
    5. Generate adaptive formulas
    """

    def __init__(self, db_path: str = None):
        self.db = MultiScaleDatabase(db_path)
        self.collector = MultiScaleCollector(self.db)
        self.wavelet = WaveletAnalyzer()

        # Per-scale HMMs
        self.hmms: Dict[int, MultiScaleHMM] = {
            scale: MultiScaleHMM(scale) for scale in SCALES
        }

        # Discovered patterns
        self.patterns: List[ConditionalPattern] = []

        # Statistics
        self.stats = {
            'flows_collected': 0,
            'buckets_created': {s: 0 for s in SCALES},
            'patterns_discovered': 0,
            'patterns_validated': 0,
        }

    def print_banner(self):
        print("""
================================================================================
            MULTI-SCALE ADAPTIVE PATTERN DISCOVERY
================================================================================

"What works for 1 second may not work for 2 seconds."

SCALES: 1s, 5s, 10s, 30s, 1m, 5m
METHOD: Train separate HMM per scale, find conditional patterns

TARGET: 50.75% win rate at EACH (regime, scale) combination
================================================================================
""")

    def collect_from_live(self, duration: int = 3600):
        """Collect from live blockchain feed."""
        print(f"\n[COLLECT] Starting live collection for {duration}s ({duration/3600:.1f}h)")

        from engine.sovereign.blockchain import PerExchangeBlockchainFeed, ExchangeTick
        from urllib.request import urlopen, Request
        import json as json_lib

        last_price_time = [0]
        cached_price = [0.0]

        def get_price():
            now = time.time()
            if now - last_price_time[0] < 5 and cached_price[0] > 0:
                return cached_price[0]
            try:
                url = 'https://api.exchange.coinbase.com/products/BTC-USD/ticker'
                req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urlopen(req, timeout=5) as resp:
                    cached_price[0] = float(json_lib.loads(resp.read().decode())['price'])
                    last_price_time[0] = now
            except:
                pass
            return cached_price[0] if cached_price[0] > 0 else 97000.0

        def on_tick(tick: ExchangeTick):
            price = get_price()
            self.collector.add_flow(
                exchange=tick.exchange,
                direction=tick.direction,
                btc=tick.volume,
                timestamp=tick.timestamp,
                price=price
            )
            self.stats['flows_collected'] += 1

            if self.stats['flows_collected'] % 100 == 0:
                elapsed = time.time() - start_time
                print(f"[COLLECT] {self.stats['flows_collected']} flows | "
                      f"{elapsed/60:.1f}m elapsed | ${price:,.0f}")

        feed = PerExchangeBlockchainFeed(on_tick=on_tick)
        price = get_price()
        if price > 0:
            feed.set_reference_price(price)

        if not feed.start():
            print("[COLLECT] Failed to start feed")
            return

        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n[COLLECT] Interrupted")
        finally:
            feed.stop()
            self.collector.flush_all()

        # Print summary
        print(f"\n[COLLECT] Complete. {self.stats['flows_collected']} flows")
        for scale in SCALES:
            count = self.db.get_bucket_count(scale)
            print(f"  {SCALE_NAMES[scale]}: {count} buckets")

    def train_all_hmms(self) -> Dict[int, float]:
        """Train HMM for each scale."""
        print("\n" + "="*60)
        print("TRAINING PER-SCALE HMMs")
        print("="*60)

        accuracies = {}

        for scale in SCALES:
            print(f"\n--- Training {SCALE_NAMES[scale]} HMM ---")

            # Load training data
            buckets = self.db.get_buckets(scale, min_volume=0.1)
            print(f"[HMM-{SCALE_NAMES[scale]}] {len(buckets)} training buckets")

            if len(buckets) < 100:
                print(f"[HMM-{SCALE_NAMES[scale]}] Not enough data, skipping")
                accuracies[scale] = 0.0
                continue

            # Add training samples
            hmm = self.hmms[scale]
            for bucket in buckets:
                if bucket.outcome != 0:
                    hmm.add_training_sample(bucket, bucket.outcome)

            # Train
            accuracy = hmm.train_baum_welch()
            accuracies[scale] = accuracy

            # Save
            self.db.save_hmm(scale, hmm, accuracy)

        print("\n[HMM] Training complete:")
        for scale, acc in accuracies.items():
            print(f"  {SCALE_NAMES[scale]}: {acc:.4f}")

        return accuracies

    def discover_patterns(self, min_length: int = 2, max_length: int = 5,
                          min_occurrences: int = 30) -> List[ConditionalPattern]:
        """
        Discover patterns at each (scale, regime) combination.
        """
        print("\n" + "="*60)
        print("DISCOVERING CONDITIONAL PATTERNS")
        print("="*60)

        all_patterns = []

        for scale in SCALES:
            print(f"\n--- Scale: {SCALE_NAMES[scale]} ---")

            buckets = self.db.get_buckets(scale, min_volume=0.1)
            if len(buckets) < 100:
                print(f"[PATTERN] Not enough data at {SCALE_NAMES[scale]}")
                continue

            hmm = self.hmms[scale]

            # Get state sequence
            states = []
            outcomes = []

            for bucket in buckets:
                result = hmm.update(bucket)
                states.append(result['state'])
                outcomes.append(bucket.outcome)

            # Find all subsequences
            pattern_stats: Dict[str, Dict] = defaultdict(lambda: {
                'occurrences': 0, 'wins': 0, 'losses': 0,
                'by_regime': defaultdict(lambda: {'occ': 0, 'wins': 0, 'losses': 0})
            })

            for length in range(min_length, max_length + 1):
                for i in range(len(states) - length):
                    seq = tuple(states[i:i+length])
                    outcome = outcomes[i + length - 1]  # Outcome after pattern

                    if outcome == 0:
                        continue

                    # Pattern direction based on last state
                    last_state = seq[-1]
                    if last_state in [0, 3]:  # ACCUMULATION, CAPITULATION
                        direction = 1
                    elif last_state in [1, 4]:  # DISTRIBUTION, EUPHORIA
                        direction = -1
                    else:
                        continue  # NEUTRAL, skip

                    # Record
                    key = f"{seq}_{direction}"
                    pattern_stats[key]['occurrences'] += 1
                    pattern_stats[key]['sequence'] = list(seq)
                    pattern_stats[key]['direction'] = direction
                    pattern_stats[key]['scale'] = scale

                    # Current regime
                    current_regime = states[i + length - 1]
                    pattern_stats[key]['by_regime'][current_regime]['occ'] += 1

                    if (direction == 1 and outcome == 1) or (direction == -1 and outcome == -1):
                        pattern_stats[key]['wins'] += 1
                        pattern_stats[key]['by_regime'][current_regime]['wins'] += 1
                    else:
                        pattern_stats[key]['losses'] += 1
                        pattern_stats[key]['by_regime'][current_regime]['losses'] += 1

            # Filter and create patterns
            for key, stats in pattern_stats.items():
                if stats['occurrences'] < min_occurrences:
                    continue

                win_rate = stats['wins'] / max(1, stats['occurrences'])

                pattern = ConditionalPattern(
                    name=f"P_{SCALE_NAMES[scale]}_{len(all_patterns)}",
                    sequence=stats['sequence'],
                    direction=stats['direction'],
                    total_occurrences=stats['occurrences'],
                    overall_win_rate=win_rate,
                )

                # Add per-regime stats
                for regime, regime_stats in stats['by_regime'].items():
                    if regime_stats['occ'] >= 10:
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

            print(f"[PATTERN] Found {len([p for p in all_patterns if scale in p.valid_scales])} "
                  f"patterns at {SCALE_NAMES[scale]}")

        self.patterns = all_patterns
        self.stats['patterns_discovered'] = len(all_patterns)

        return all_patterns

    def validate_patterns(self, min_win_rate: float = 0.5075,
                          n_bootstrap: int = 1000) -> List[ConditionalPattern]:
        """
        Validate patterns using Monte Carlo at each condition.
        """
        print("\n" + "="*60)
        print("VALIDATING PATTERNS (Monte Carlo)")
        print("="*60)

        validated = []

        for pattern in self.patterns:
            best_edge = 0.0
            best_condition = (0, 0)

            for (scale, regime), stats in pattern.stats.items():
                if stats['occurrences'] < 30:
                    continue

                # Bootstrap confidence interval
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

                # Chi-square test
                expected_wins = total * 0.5
                chi2 = (wins - expected_wins) ** 2 / expected_wins
                # Approximate p-value
                stats['p_value'] = math.exp(-chi2 / 2) if chi2 < 50 else 0.0

                # Check if valid
                if ci_low >= 0.5 and stats['win_rate'] >= min_win_rate:
                    edge = stats['win_rate'] - 0.5
                    if edge > best_edge:
                        best_edge = edge
                        best_condition = (scale, regime)

            if best_edge > 0:
                pattern.is_valid = True
                pattern.best_edge = best_edge
                pattern.best_condition = best_condition
                validated.append(pattern)

                self.db.save_pattern(pattern)

                print(f"[VALID] {pattern.name}: edge={best_edge:.4f} "
                      f"at ({SCALE_NAMES[best_condition[0]]}, {Regime(best_condition[1]).name})")

        self.stats['patterns_validated'] = len(validated)
        print(f"\n[VALIDATE] {len(validated)} patterns validated")

        return validated

    def generate_formulas(self) -> str:
        """Generate Python code for validated formulas."""
        print("\n" + "="*60)
        print("GENERATING ADAPTIVE FORMULAS")
        print("="*60)

        valid_patterns = [p for p in self.patterns if p.is_valid]

        if not valid_patterns:
            print("[FORMULA] No valid patterns to generate")
            return ""

        lines = [
            '"""',
            'MULTI-SCALE ADAPTIVE FORMULAS',
            '=' * 40,
            f'Generated: {datetime.now().isoformat()}',
            f'Patterns: {len(valid_patterns)}',
            '',
            'These formulas adapt based on current (scale, regime) conditions.',
            'Each pattern is only valid under specific conditions.',
            '"""',
            '',
            'from typing import Dict, List, Tuple, Optional',
            '',
            '# Scale definitions',
            'SCALES = {1: "1s", 5: "5s", 10: "10s", 30: "30s", 60: "1m", 300: "5m"}',
            '',
            '# Regime definitions',
            'REGIMES = {0: "ACCUMULATION", 1: "DISTRIBUTION", 2: "NEUTRAL", 3: "CAPITULATION", 4: "EUPHORIA"}',
            '',
            '',
            'ADAPTIVE_FORMULAS = [',
        ]

        for p in valid_patterns:
            lines.append('    {')
            lines.append(f'        "name": "{p.name}",')
            lines.append(f'        "sequence": {p.sequence},')
            lines.append(f'        "direction": {p.direction},  # {"LONG" if p.direction == 1 else "SHORT"}')
            lines.append(f'        "valid_scales": {list(p.valid_scales)},')
            lines.append(f'        "valid_regimes": {list(p.valid_regimes)},')
            lines.append(f'        "best_scale": {p.best_condition[0]},')
            lines.append(f'        "best_regime": {p.best_condition[1]},')
            lines.append(f'        "best_edge": {p.best_edge:.6f},')
            lines.append(f'        "total_occurrences": {p.total_occurrences},')
            lines.append(f'        "overall_win_rate": {p.overall_win_rate:.6f},')
            lines.append('        "conditions": {')
            for (scale, regime), stats in p.stats.items():
                if stats.get('win_rate', 0) >= 0.5075:
                    lines.append(f'            ({scale}, {regime}): {{"win_rate": {stats["win_rate"]:.4f}, "edge": {stats["edge"]:.4f}, "n": {stats["occurrences"]}}},')
            lines.append('        },')
            lines.append('    },')

        lines.append(']')
        lines.append('')
        lines.append('')
        lines.append('def get_signal(sequence: List[int], scale: int, regime: int) -> Optional[Dict]:')
        lines.append('    """')
        lines.append('    Get trading signal for current conditions.')
        lines.append('    ')
        lines.append('    Returns formula if (scale, regime) is valid, None otherwise.')
        lines.append('    """')
        lines.append('    for formula in ADAPTIVE_FORMULAS:')
        lines.append('        if formula["sequence"] == sequence:')
        lines.append('            if scale in formula["valid_scales"] and regime in formula["valid_regimes"]:')
        lines.append('                condition = formula["conditions"].get((scale, regime))')
        lines.append('                if condition:')
        lines.append('                    return {')
        lines.append('                        "name": formula["name"],')
        lines.append('                        "direction": formula["direction"],')
        lines.append('                        "edge": condition["edge"],')
        lines.append('                        "win_rate": condition["win_rate"],')
        lines.append('                        "confidence": condition["win_rate"],')
        lines.append('                    }')
        lines.append('    return None')
        lines.append('')

        code = '\n'.join(lines)

        # Save
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'adaptive_formulas.py'
        )
        with open(output_path, 'w') as f:
            f.write(code)

        print(f"[FORMULA] Generated {output_path}")
        print(f"[FORMULA] {len(valid_patterns)} adaptive formulas")

        return code

    def generate_report(self) -> str:
        """Generate analysis report."""
        valid_patterns = [p for p in self.patterns if p.is_valid]

        lines = [
            '=' * 80,
            'MULTI-SCALE ADAPTIVE DISCOVERY REPORT',
            '=' * 80,
            f'Generated: {datetime.now().isoformat()}',
            '',
            '## DATA SUMMARY',
            f'Total flows: {self.stats["flows_collected"]}',
            '',
            '## BUCKETS PER SCALE',
        ]

        for scale in SCALES:
            count = self.db.get_bucket_count(scale)
            lines.append(f'  {SCALE_NAMES[scale]}: {count}')

        lines.append('')
        lines.append('## HMM VALIDATION ACCURACY')

        for scale in SCALES:
            hmm = self.db.load_hmm(scale)
            if hmm:
                # Get accuracy from db
                conn = sqlite3.connect(self.db.db_path)
                c = conn.cursor()
                c.execute('SELECT validation_accuracy FROM hmm_models WHERE scale = ?', (scale,))
                row = c.fetchone()
                acc = row[0] if row else 0
                conn.close()
                lines.append(f'  {SCALE_NAMES[scale]}: {acc:.4f}')

        lines.append('')
        lines.append('## VALIDATED PATTERNS')
        lines.append(f'Total discovered: {self.stats["patterns_discovered"]}')
        lines.append(f'Validated (>50.75%): {self.stats["patterns_validated"]}')
        lines.append('')

        # Top patterns by edge
        sorted_patterns = sorted(valid_patterns, key=lambda p: p.best_edge, reverse=True)

        for i, p in enumerate(sorted_patterns[:10]):
            scale_name = SCALE_NAMES[p.best_condition[0]]
            regime_name = Regime(p.best_condition[1]).name
            dir_str = "LONG" if p.direction == 1 else "SHORT"

            lines.append(f'### {i+1}. {p.name}')
            lines.append(f'Direction: {dir_str}')
            lines.append(f'Sequence: {p.sequence}')
            lines.append(f'Best condition: {scale_name} + {regime_name}')
            lines.append(f'Edge: {p.best_edge:.4f} ({p.best_edge*100:.2f}%)')
            lines.append(f'Occurrences: {p.total_occurrences}')
            lines.append('')

        lines.append('=' * 80)

        report = '\n'.join(lines)

        # Save
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', '..', 'data', 'multi_scale_report.md'
        )
        with open(output_path, 'w') as f:
            f.write(report)

        print(report)
        return report

    def run(self, collect_duration: int = 3600):
        """Run full discovery pipeline."""
        self.print_banner()

        # Phase 1: Collect
        self.collect_from_live(collect_duration)

        # Phase 2: Train HMMs
        self.train_all_hmms()

        # Phase 3: Discover patterns
        self.discover_patterns()

        # Phase 4: Validate
        self.validate_patterns()

        # Phase 5: Generate formulas
        self.generate_formulas()

        # Phase 6: Report
        self.generate_report()

        print("\n" + "="*60)
        print("MULTI-SCALE DISCOVERY COMPLETE")
        print("="*60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Scale Adaptive Pattern Discovery')
    parser.add_argument('--hours', type=float, default=1,
                       help='Hours to collect live data (default: 1)')
    parser.add_argument('--train-only', action='store_true',
                       help='Only train on existing data')

    args = parser.parse_args()

    discovery = MultiScaleDiscovery()

    if args.train_only:
        discovery.train_all_hmms()
        discovery.discover_patterns()
        discovery.validate_patterns()
        discovery.generate_formulas()
        discovery.generate_report()
    else:
        discovery.run(collect_duration=int(args.hours * 3600))


if __name__ == "__main__":
    main()
