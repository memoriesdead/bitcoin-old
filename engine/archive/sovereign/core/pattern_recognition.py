#!/usr/bin/env python3
"""
RENTECH PATTERN RECOGNITION

Implements RenTech-style pattern recognition:
1. Hidden Markov Models for regime detection
2. Statistical anomaly detection
3. Mean reversion signals
4. Temporal patterns
"""
import numpy as np
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
from enum import Enum


class MarketRegime(Enum):
    """Market regime states for HMM."""
    ACCUMULATION = 0  # Smart money buying
    MARKUP = 1        # Trending up
    DISTRIBUTION = 2  # Smart money selling
    MARKDOWN = 3      # Trending down


@dataclass
class Signal:
    """Trading signal with confidence."""
    timestamp: int
    direction: str  # 'LONG', 'SHORT', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    components: dict  # Individual signal contributions


class HiddenMarkovModel:
    """
    Simple HMM for market regime detection.

    States: [ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN]

    Based on observations:
    - Whale activity (high/low)
    - Network velocity (high/low)
    - Fee pressure (high/low)
    """

    def __init__(self, n_states: int = 4):
        self.n_states = n_states

        # Transition matrix (learned or preset)
        # Rows: from state, Cols: to state
        self.transition = np.array([
            # ACC   MKP   DST   MKD
            [0.7,  0.2,  0.05, 0.05],  # From ACCUMULATION
            [0.1,  0.7,  0.15, 0.05],  # From MARKUP
            [0.05, 0.1,  0.7,  0.15],  # From DISTRIBUTION
            [0.15, 0.05, 0.1,  0.7],   # From MARKDOWN
        ])

        # Emission probabilities (observation given state)
        # Simplified: each state has characteristic observation pattern
        self.emission_means = {
            MarketRegime.ACCUMULATION: {'whale': 0.3, 'velocity': 0.4, 'fees': 0.3},
            MarketRegime.MARKUP: {'whale': 0.5, 'velocity': 0.7, 'fees': 0.6},
            MarketRegime.DISTRIBUTION: {'whale': 0.7, 'velocity': 0.6, 'fees': 0.5},
            MarketRegime.MARKDOWN: {'whale': 0.4, 'velocity': 0.3, 'fees': 0.2},
        }

        self.current_state = MarketRegime.ACCUMULATION
        self.state_probs = np.array([0.25, 0.25, 0.25, 0.25])

    def normalize_observation(self, whale_activity: float, velocity: float, fees: float) -> dict:
        """Normalize observations to 0-1 range."""
        return {
            'whale': min(1.0, whale_activity / 100),  # Assuming max 100 whale txs
            'velocity': min(1.0, velocity / 1e6),     # Assuming max 1M BTC/day
            'fees': min(1.0, fees / 100),             # Assuming max 100 sat/vB
        }

    def observation_likelihood(self, obs: dict, state: MarketRegime) -> float:
        """Calculate P(observation | state) using Gaussian."""
        means = self.emission_means[state]
        likelihood = 1.0
        for key in obs:
            diff = obs[key] - means[key]
            likelihood *= np.exp(-diff**2 / 0.1)  # Gaussian with std=0.316
        return likelihood

    def update(self, whale_activity: float, velocity: float, fees: float) -> MarketRegime:
        """
        Update state probabilities given new observation.
        Returns most likely current state.
        """
        obs = self.normalize_observation(whale_activity, velocity, fees)

        # Forward step: P(state_t | obs_1:t)
        new_probs = np.zeros(self.n_states)

        for j in range(self.n_states):
            state = MarketRegime(j)
            emission = self.observation_likelihood(obs, state)

            # Sum over previous states
            trans_prob = sum(
                self.state_probs[i] * self.transition[i, j]
                for i in range(self.n_states)
            )

            new_probs[j] = emission * trans_prob

        # Normalize
        total = new_probs.sum()
        if total > 0:
            self.state_probs = new_probs / total
        else:
            self.state_probs = np.array([0.25, 0.25, 0.25, 0.25])

        # Update current state
        self.current_state = MarketRegime(np.argmax(self.state_probs))
        return self.current_state

    def get_regime_signal(self) -> Tuple[str, float]:
        """
        Convert regime to trading signal.

        ACCUMULATION -> LONG (0.6)
        MARKUP -> LONG (0.8)
        DISTRIBUTION -> SHORT (0.6)
        MARKDOWN -> SHORT (0.8)
        """
        regime = self.current_state
        confidence = self.state_probs[regime.value]

        if regime in [MarketRegime.ACCUMULATION, MarketRegime.MARKUP]:
            direction = 'LONG'
            strength = 0.6 if regime == MarketRegime.ACCUMULATION else 0.8
        else:
            direction = 'SHORT'
            strength = 0.6 if regime == MarketRegime.DISTRIBUTION else 0.8

        return direction, strength * confidence


class AnomalyDetector:
    """
    Statistical anomaly detection using rolling z-scores.
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.history = {}  # feature_name -> list of values

    # Numeric features only (skip strings like 'hash')
    NUMERIC_FEATURES = {
        'tx_count', 'total_value_btc', 'total_fees_btc', 'avg_fee_rate',
        'utxo_created', 'utxo_destroyed', 'net_utxo_change',
        'whale_tx_count', 'whale_value_btc', 'large_tx_count', 'large_value_btc',
        'block_size', 'block_weight', 'block_fullness',
        'coinbase_value_btc', 'coinbase_outputs'
    }

    def add_observation(self, features: dict):
        """Add new observation to history."""
        for name, value in features.items():
            # Skip non-numeric features
            if name not in self.NUMERIC_FEATURES:
                continue
            if not isinstance(value, (int, float)):
                continue

            if name not in self.history:
                self.history[name] = []
            self.history[name].append(float(value))

            # Keep only window size
            if len(self.history[name]) > self.window * 2:
                self.history[name] = self.history[name][-self.window*2:]

    def get_zscore(self, feature: str, value: float) -> float:
        """Calculate z-score for a feature value."""
        if feature not in self.history or len(self.history[feature]) < self.window:
            return 0.0

        recent = self.history[feature][-self.window:]
        mean = np.mean(recent)
        std = np.std(recent)

        if std == 0:
            return 0.0

        return (value - mean) / std

    def detect_anomalies(self, features: dict, threshold: float = 2.0) -> dict:
        """
        Detect anomalies in features.
        Returns dict of feature -> z-score for anomalous features.
        """
        anomalies = {}

        for name, value in features.items():
            # Skip non-numeric features
            if name not in self.NUMERIC_FEATURES:
                continue
            if not isinstance(value, (int, float)):
                continue

            zscore = self.get_zscore(name, float(value))
            if abs(zscore) > threshold:
                anomalies[name] = zscore

        return anomalies

    def get_anomaly_signal(self, features: dict) -> Tuple[str, float]:
        """
        Convert anomalies to trading signal.

        High whale activity + high fees -> SHORT (distribution)
        Low whale activity + low fees -> LONG (accumulation)
        """
        anomalies = self.detect_anomalies(features)

        if not anomalies:
            return 'NEUTRAL', 0.0

        # Score based on anomaly direction
        long_score = 0.0
        short_score = 0.0

        for feature, zscore in anomalies.items():
            if feature in ['whale_tx_count', 'whale_value_btc', 'total_fees_btc']:
                # High values suggest distribution (SHORT)
                if zscore > 0:
                    short_score += abs(zscore) / 3
                else:
                    long_score += abs(zscore) / 3
            elif feature in ['utxo_created', 'tx_count']:
                # High values suggest adoption (LONG)
                if zscore > 0:
                    long_score += abs(zscore) / 3
                else:
                    short_score += abs(zscore) / 3

        if long_score > short_score and long_score > 0.3:
            return 'LONG', min(1.0, long_score)
        elif short_score > long_score and short_score > 0.3:
            return 'SHORT', min(1.0, short_score)
        else:
            return 'NEUTRAL', 0.0


class MeanReversionDetector:
    """
    Mean reversion signals based on on-chain metrics.

    Key metrics:
    - SOPR (Spent Output Profit Ratio)
    - Exchange flow ratio
    - Fee premium
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.history = {
            'sopr': [],
            'exchange_flow_ratio': [],
            'fee_rate': [],
        }

    def calculate_sopr(self, spent_value: float, created_value: float) -> float:
        """
        SOPR = value of outputs spent / value when they were created.
        Simplified: use current value / average value as proxy.
        """
        if created_value == 0:
            return 1.0
        return spent_value / created_value

    def add_observation(self, sopr: float, inflow: float, outflow: float, fee_rate: float):
        """Add observation to history."""
        self.history['sopr'].append(sopr)

        # Exchange flow ratio: inflow / outflow
        ratio = inflow / outflow if outflow > 0 else 1.0
        self.history['exchange_flow_ratio'].append(ratio)

        self.history['fee_rate'].append(fee_rate)

        # Trim history
        for key in self.history:
            if len(self.history[key]) > self.window * 2:
                self.history[key] = self.history[key][-self.window*2:]

    def get_signal(self) -> Tuple[str, float]:
        """
        Generate mean reversion signal.

        SOPR > 1.05: Profit taking, expect pullback (SHORT)
        SOPR < 0.95: Capitulation, expect bounce (LONG)

        High inflow/outflow: Distribution (SHORT)
        Low inflow/outflow: Accumulation (LONG)
        """
        if len(self.history['sopr']) < self.window:
            return 'NEUTRAL', 0.0

        sopr = self.history['sopr'][-1]
        flow_ratio = self.history['exchange_flow_ratio'][-1]

        long_score = 0.0
        short_score = 0.0

        # SOPR signal
        if sopr > 1.05:
            short_score += (sopr - 1.0) * 2
        elif sopr < 0.95:
            long_score += (1.0 - sopr) * 2

        # Flow ratio signal
        if flow_ratio > 1.5:  # More inflow than outflow
            short_score += (flow_ratio - 1.0) * 0.5
        elif flow_ratio < 0.7:  # More outflow than inflow
            long_score += (1.0 - flow_ratio) * 0.5

        if long_score > short_score and long_score > 0.2:
            return 'LONG', min(1.0, long_score)
        elif short_score > long_score and short_score > 0.2:
            return 'SHORT', min(1.0, short_score)
        else:
            return 'NEUTRAL', 0.0


class TemporalPatternDetector:
    """
    Detect temporal patterns in Bitcoin.

    Patterns:
    - Hour-of-day effects
    - Day-of-week effects
    - Halving cycle effects
    """

    # Historical average returns by hour (simplified)
    HOUR_BIAS = {
        0: 0.01, 1: 0.005, 2: 0.003, 3: 0.002,
        4: 0.001, 5: 0.002, 6: 0.005, 7: 0.008,
        8: 0.01, 9: 0.012, 10: 0.015, 11: 0.01,
        12: 0.005, 13: 0.003, 14: 0.005, 15: 0.008,
        16: 0.012, 17: 0.015, 18: 0.01, 19: 0.008,
        20: 0.005, 21: 0.003, 22: 0.002, 23: 0.005,
    }

    # Historical average returns by day (simplified)
    DAY_BIAS = {
        0: 0.02,   # Monday - often positive
        1: 0.01,   # Tuesday
        2: 0.005,  # Wednesday
        3: 0.003,  # Thursday
        4: -0.01,  # Friday - often negative (weekend selling)
        5: -0.005, # Saturday
        6: 0.008,  # Sunday - recovery
    }

    # Halving dates (timestamp)
    HALVINGS = [
        1354118400,  # 2012-11-28
        1467936000,  # 2016-07-09
        1589068800,  # 2020-05-11
        1713398400,  # 2024-04-19 (approximate)
    ]

    def get_signal(self, timestamp: int) -> Tuple[str, float]:
        """
        Generate signal based on temporal patterns.
        """
        from datetime import datetime

        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        day = dt.weekday()

        # Combine hour and day bias
        hour_bias = self.HOUR_BIAS.get(hour, 0)
        day_bias = self.DAY_BIAS.get(day, 0)

        # Halving cycle effect
        days_since_halving = self._days_since_halving(timestamp)
        halving_bias = self._halving_cycle_bias(days_since_halving)

        total_bias = hour_bias + day_bias + halving_bias

        if total_bias > 0.01:
            return 'LONG', min(1.0, total_bias * 10)
        elif total_bias < -0.01:
            return 'SHORT', min(1.0, abs(total_bias) * 10)
        else:
            return 'NEUTRAL', 0.0

    def _days_since_halving(self, timestamp: int) -> int:
        """Calculate days since last halving."""
        for halving in reversed(self.HALVINGS):
            if timestamp > halving:
                return (timestamp - halving) // 86400
        return 0

    def _halving_cycle_bias(self, days: int) -> float:
        """
        Halving cycle bias based on historical patterns.

        0-200 days: accumulation (slight long)
        200-400 days: early bull (strong long)
        400-600 days: peak (neutral to short)
        600+ days: bear (short then long)
        """
        if days < 200:
            return 0.005
        elif days < 400:
            return 0.015
        elif days < 600:
            return -0.005
        elif days < 800:
            return -0.01
        else:
            return 0.005


class SignalCombiner:
    """
    Combine multiple signals into a single trading signal.

    Uses weighted combination with Kelly-inspired sizing.
    """

    def __init__(self):
        self.hmm = HiddenMarkovModel()
        self.anomaly = AnomalyDetector()
        self.mean_reversion = MeanReversionDetector()
        self.temporal = TemporalPatternDetector()

        # Signal weights
        self.weights = {
            'hmm': 0.30,
            'anomaly': 0.25,
            'mean_reversion': 0.25,
            'temporal': 0.20,
        }

    def generate_signal(self, features: dict, timestamp: int) -> Signal:
        """
        Generate combined trading signal from all pattern detectors.
        """
        # Update detectors
        self.anomaly.add_observation(features)

        # Get individual signals
        self.hmm.update(
            features.get('whale_tx_count', 0),
            features.get('total_value_btc', 0),
            features.get('avg_fee_rate', 0)
        )
        hmm_dir, hmm_conf = self.hmm.get_regime_signal()

        anom_dir, anom_conf = self.anomaly.get_anomaly_signal(features)
        mr_dir, mr_conf = self.mean_reversion.get_signal()
        temp_dir, temp_conf = self.temporal.get_signal(timestamp)

        # Convert to numeric scores
        def dir_to_score(direction: str, confidence: float) -> float:
            if direction == 'LONG':
                return confidence
            elif direction == 'SHORT':
                return -confidence
            else:
                return 0.0

        scores = {
            'hmm': dir_to_score(hmm_dir, hmm_conf) * self.weights['hmm'],
            'anomaly': dir_to_score(anom_dir, anom_conf) * self.weights['anomaly'],
            'mean_reversion': dir_to_score(mr_dir, mr_conf) * self.weights['mean_reversion'],
            'temporal': dir_to_score(temp_dir, temp_conf) * self.weights['temporal'],
        }

        # Combined score
        combined = sum(scores.values())

        # Determine direction and confidence
        if combined > 0.15:
            direction = 'LONG'
            confidence = min(1.0, combined)
        elif combined < -0.15:
            direction = 'SHORT'
            confidence = min(1.0, abs(combined))
        else:
            direction = 'NEUTRAL'
            confidence = 0.0

        return Signal(
            timestamp=timestamp,
            direction=direction,
            confidence=confidence,
            components={
                'hmm': (hmm_dir, hmm_conf),
                'anomaly': (anom_dir, anom_conf),
                'mean_reversion': (mr_dir, mr_conf),
                'temporal': (temp_dir, temp_conf),
                'scores': scores,
            }
        )


if __name__ == "__main__":
    # Test the signal combiner
    combiner = SignalCombiner()

    # Simulate some data
    import time

    test_features = {
        'whale_tx_count': 15,
        'whale_value_btc': 5000,
        'total_value_btc': 50000,
        'total_fees_btc': 5.0,
        'avg_fee_rate': 50,
        'utxo_created': 3000,
        'utxo_destroyed': 2500,
        'tx_count': 2500,
    }

    signal = combiner.generate_signal(test_features, int(time.time()))

    print(f"Signal: {signal.direction}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Components: {signal.components}")
