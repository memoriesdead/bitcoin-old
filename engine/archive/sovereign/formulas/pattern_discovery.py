"""
PATTERN DISCOVERY - Find Sequences That Precede Price Moves
===========================================================

"We look for things that can be replicated thousands of times."
- Jim Simons, Renaissance Technologies

This module discovers recurring patterns in blockchain flow data that
predict price movements with > 50.75% accuracy.

DISCOVERY PROCESS:
1. Extract flow sequences (HMM state sequences)
2. Find all subsequences of length 2-5
3. For each subsequence, calculate:
   - Occurrences: How many times it appeared
   - Win rate: % of times price moved in predicted direction
   - Significance: Statistical significance (chi-square test)
4. Keep patterns that:
   - Occur 100+ times (replicable)
   - Win rate > 50.75% (edge)
   - Statistically significant (p < 0.05)

PATTERN TYPES:
- Regime transitions: [NEUTRAL -> ACCUMULATION -> ACCUMULATION] -> LONG
- Reversal patterns: [CAPITULATION -> ACCUMULATION] -> LONG
- Momentum patterns: [DISTRIBUTION, DISTRIBUTION, DISTRIBUTION] -> SHORT

OUTPUT:
- Pattern library with validated patterns
- Each pattern has: sequence, direction, win_rate, occurrences
"""

import math
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from .historical_data import HistoricalFlowDatabase, FlowEvent
from .hmm_trainer import BaumWelchTrainer, HMMParameters

import numpy as np


@dataclass
class DiscoveredPattern:
    """A discovered pattern with validation metrics."""
    name: str
    sequence: List[int]  # State sequence (e.g., [2, 0, 0] = NEUTRAL->ACC->ACC)
    direction: int       # Expected price direction: +1 (up), -1 (down)
    occurrences: int     # Total times pattern appeared
    wins: int            # Times prediction was correct
    losses: int          # Times prediction was wrong
    win_rate: float      # wins / (wins + losses)
    edge: float          # win_rate - 0.5
    p_value: float       # Statistical significance
    avg_return: float    # Average return when pattern triggered
    is_valid: bool       # Meets all criteria for trading


class PatternDiscoveryEngine:
    """
    Discovers profitable patterns from historical HMM state sequences.

    METHODOLOGY:
    1. Run trained HMM on historical data to get state sequences
    2. Enumerate all subsequences of length 2-5
    3. Track outcomes for each subsequence
    4. Filter to patterns meeting criteria

    VALIDATION CRITERIA:
    - Minimum occurrences: 100 (statistically meaningful)
    - Minimum win rate: 50.75% (RenTech threshold)
    - Minimum significance: p < 0.05 (not random chance)
    """

    STATE_NAMES = ['ACCUMULATION', 'DISTRIBUTION', 'NEUTRAL', 'CAPITULATION', 'EUPHORIA']

    # State to expected price direction
    STATE_DIRECTION = {
        0: 1,   # ACCUMULATION -> LONG
        1: -1,  # DISTRIBUTION -> SHORT
        2: 0,   # NEUTRAL -> No trade
        3: 1,   # CAPITULATION -> Contrarian LONG
        4: -1,  # EUPHORIA -> Contrarian SHORT
    }

    def __init__(self, db: HistoricalFlowDatabase = None,
                 min_pattern_length: int = 2,
                 max_pattern_length: int = 5,
                 min_occurrences: int = 100,
                 min_win_rate: float = 0.5075):
        self.db = db or HistoricalFlowDatabase()
        self.min_len = min_pattern_length
        self.max_len = max_pattern_length
        self.min_occurrences = min_occurrences
        self.min_win_rate = min_win_rate

        # Discovered patterns
        self.patterns: Dict[str, DiscoveredPattern] = {}

        # Pattern statistics during discovery
        self._pattern_stats: Dict[str, Dict] = defaultdict(lambda: {
            'occurrences': 0,
            'wins': 0,
            'losses': 0,
            'returns': [],
        })

    def _sequence_to_key(self, sequence: List[int]) -> str:
        """Convert state sequence to string key."""
        return ','.join(str(s) for s in sequence)

    def _key_to_sequence(self, key: str) -> List[int]:
        """Convert string key back to sequence."""
        return [int(s) for s in key.split(',')]

    def _sequence_to_name(self, sequence: List[int]) -> str:
        """Generate human-readable name for pattern."""
        names = [self.STATE_NAMES[s][:3].upper() for s in sequence]
        return '_'.join(names)

    def _infer_direction(self, sequence: List[int]) -> int:
        """
        Infer expected price direction from sequence.

        Rules:
        - Last state determines primary direction
        - Transition patterns can override (e.g., reversal)
        """
        if not sequence:
            return 0

        last_state = sequence[-1]
        base_direction = self.STATE_DIRECTION.get(last_state, 0)

        # Special reversal patterns
        if len(sequence) >= 2:
            # CAPITULATION -> ACCUMULATION = Strong LONG (reversal)
            if sequence[-2] == 3 and sequence[-1] == 0:
                return 1

            # EUPHORIA -> DISTRIBUTION = Strong SHORT (reversal)
            if sequence[-2] == 4 and sequence[-1] == 1:
                return -1

            # Momentum: 3+ same state = trend continuation
            if len(sequence) >= 3 and len(set(sequence[-3:])) == 1:
                if sequence[-1] in [0, 3]:
                    return 1
                elif sequence[-1] in [1, 4]:
                    return -1

        return base_direction

    def _chi_square_test(self, wins: int, losses: int) -> float:
        """
        Chi-square test for statistical significance.

        H0: win_rate = 0.5 (no edge)
        H1: win_rate != 0.5 (has edge)

        Returns p-value.
        """
        total = wins + losses
        if total < 10:
            return 1.0  # Not enough data

        expected = total / 2

        chi_sq = ((wins - expected) ** 2 / expected +
                  (losses - expected) ** 2 / expected)

        # Approximate p-value from chi-square with 1 df
        # Using simplified approximation
        if chi_sq < 0.001:
            return 1.0
        elif chi_sq > 10.83:
            return 0.001  # Very significant
        elif chi_sq > 6.63:
            return 0.01
        elif chi_sq > 3.84:
            return 0.05
        elif chi_sq > 2.71:
            return 0.10
        else:
            return 0.5

    def discover_from_flows(self, flows: List[FlowEvent], hmm: BaumWelchTrainer,
                            verbose: bool = True) -> List[DiscoveredPattern]:
        """
        Discover patterns from historical flow data.

        Args:
            flows: Historical flow events with outcomes
            hmm: Trained HMM for state inference
            verbose: Print progress

        Returns:
            List of validated patterns
        """
        if len(flows) < 100:
            print("[DISCOVERY] Not enough flows")
            return []

        # Convert flows to observations
        observations = []
        outcomes = []
        prices = []

        for flow in flows:
            if flow.outcome_30s == 0:
                continue

            obs = [
                flow.flow_imbalance,
                flow.flow_velocity,
                flow.whale_ratio,
                flow.fee_percentile / 100.0,
            ]
            observations.append(obs)
            outcomes.append(flow.outcome_30s)
            prices.append((flow.price_at_flow, flow.price_after_30s))

        if len(observations) < 100:
            print(f"[DISCOVERY] Only {len(observations)} observations with outcomes")
            return []

        observations = np.array(observations)

        if verbose:
            print(f"[DISCOVERY] Processing {len(observations)} observations")

        # Get HMM state sequence
        states = hmm.predict_states(observations)

        if verbose:
            print(f"[DISCOVERY] State sequence length: {len(states)}")
            state_counts = defaultdict(int)
            for s in states:
                state_counts[s] += 1
            print(f"[DISCOVERY] State distribution: {dict(state_counts)}")

        # Enumerate all subsequences
        for length in range(self.min_len, self.max_len + 1):
            for i in range(len(states) - length):
                subseq = states[i:i+length]
                outcome = outcomes[i + length - 1]  # Outcome after pattern completes

                # Calculate return
                if prices[i + length - 1][0] > 0:
                    ret = (prices[i + length - 1][1] - prices[i + length - 1][0]) / prices[i + length - 1][0]
                else:
                    ret = 0

                key = self._sequence_to_key(subseq)
                expected_dir = self._infer_direction(subseq)

                # Track statistics
                self._pattern_stats[key]['occurrences'] += 1
                self._pattern_stats[key]['returns'].append(ret)

                if expected_dir != 0:
                    if (expected_dir == 1 and outcome == 1) or (expected_dir == -1 and outcome == -1):
                        self._pattern_stats[key]['wins'] += 1
                    else:
                        self._pattern_stats[key]['losses'] += 1

        if verbose:
            print(f"[DISCOVERY] Found {len(self._pattern_stats)} unique subsequences")

        # Convert to patterns and filter
        self.patterns = {}
        valid_count = 0

        for key, stats in self._pattern_stats.items():
            sequence = self._key_to_sequence(key)
            direction = self._infer_direction(sequence)

            if direction == 0:
                continue  # Skip NEUTRAL patterns

            total_trades = stats['wins'] + stats['losses']
            if total_trades == 0:
                continue

            win_rate = stats['wins'] / total_trades
            edge = win_rate - 0.5
            p_value = self._chi_square_test(stats['wins'], stats['losses'])
            avg_return = sum(stats['returns']) / len(stats['returns']) if stats['returns'] else 0

            # Validation criteria
            is_valid = (
                stats['occurrences'] >= self.min_occurrences and
                win_rate >= self.min_win_rate and
                p_value < 0.05
            )

            if is_valid:
                valid_count += 1

            pattern = DiscoveredPattern(
                name=self._sequence_to_name(sequence),
                sequence=sequence,
                direction=direction,
                occurrences=stats['occurrences'],
                wins=stats['wins'],
                losses=stats['losses'],
                win_rate=win_rate,
                edge=edge,
                p_value=p_value,
                avg_return=avg_return,
                is_valid=is_valid,
            )

            self.patterns[key] = pattern

        if verbose:
            print(f"[DISCOVERY] {valid_count} patterns meet validation criteria")

        # Get valid patterns sorted by edge
        valid_patterns = [p for p in self.patterns.values() if p.is_valid]
        valid_patterns.sort(key=lambda p: p.edge, reverse=True)

        if verbose and valid_patterns:
            print("\n[DISCOVERY] === TOP PATTERNS ===")
            for p in valid_patterns[:10]:
                dir_str = "LONG" if p.direction == 1 else "SHORT"
                print(f"  {p.name}: {dir_str} | WR={p.win_rate:.2%} | "
                      f"Edge={p.edge:.2%} | N={p.occurrences} | p={p.p_value:.3f}")

        return valid_patterns

    def save_patterns(self, patterns: List[DiscoveredPattern] = None):
        """Save discovered patterns to database."""
        if patterns is None:
            patterns = [p for p in self.patterns.values() if p.is_valid]

        for pattern in patterns:
            self.db.save_pattern(
                name=pattern.name,
                sequence=pattern.sequence,
                direction=pattern.direction,
                occurrences=pattern.occurrences,
                wins=pattern.wins,
                losses=pattern.losses,
            )

        print(f"[DISCOVERY] Saved {len(patterns)} patterns to database")

    def load_patterns(self) -> List[DiscoveredPattern]:
        """Load patterns from database."""
        db_patterns = self.db.get_patterns(
            min_occurrences=self.min_occurrences,
            min_win_rate=self.min_win_rate,
        )

        patterns = []
        for p in db_patterns:
            pattern = DiscoveredPattern(
                name=self._sequence_to_name(p['sequence']),
                sequence=p['sequence'],
                direction=p['direction'],
                occurrences=p['occurrences'],
                wins=p['wins'],
                losses=p['losses'],
                win_rate=p['win_rate'],
                edge=p['win_rate'] - 0.5,
                p_value=0.05,  # Already validated
                avg_return=0,
                is_valid=True,
            )
            patterns.append(pattern)
            self.patterns[self._sequence_to_key(p['sequence'])] = pattern

        return patterns


class PatternMatcher:
    """
    Real-time pattern matching for live trading.

    Uses discovered patterns to generate trading signals.
    """

    def __init__(self, patterns: List[DiscoveredPattern]):
        """Initialize with validated patterns."""
        self.patterns = {
            tuple(p.sequence): p for p in patterns if p.is_valid
        }

        # State buffer for matching
        self.state_buffer: List[int] = []
        self.max_length = max((len(p.sequence) for p in patterns), default=5)

    def add_state(self, state: int) -> Optional[DiscoveredPattern]:
        """
        Add new state and check for pattern matches.

        Returns best matching pattern if found.
        """
        self.state_buffer.append(state)

        # Trim buffer
        if len(self.state_buffer) > self.max_length:
            self.state_buffer.pop(0)

        # Check for matches (longest first)
        best_match = None
        best_edge = 0

        for length in range(len(self.state_buffer), 0, -1):
            subseq = tuple(self.state_buffer[-length:])
            if subseq in self.patterns:
                pattern = self.patterns[subseq]
                if pattern.edge > best_edge:
                    best_edge = pattern.edge
                    best_match = pattern

        return best_match

    def get_signal(self) -> Optional[Dict]:
        """Get current trading signal based on recent states."""
        if not self.state_buffer:
            return None

        match = None
        for length in range(len(self.state_buffer), 0, -1):
            subseq = tuple(self.state_buffer[-length:])
            if subseq in self.patterns:
                match = self.patterns[subseq]
                break

        if match:
            return {
                'pattern': match.name,
                'direction': match.direction,
                'confidence': match.win_rate,
                'edge': match.edge,
                'occurrences': match.occurrences,
            }

        return None


def discover_patterns_from_database(db_path: str = None, verbose: bool = True) -> List[DiscoveredPattern]:
    """
    Convenience function to discover patterns from database.

    Full pipeline:
    1. Load trained HMM
    2. Load historical flows
    3. Discover patterns
    4. Save valid patterns
    """
    db = HistoricalFlowDatabase(db_path)

    # Load trained HMM
    hmm_params = db.load_hmm_model('default')
    if not hmm_params:
        print("[DISCOVERY] No trained HMM found - run training first")
        return []

    # Initialize trainer with loaded parameters
    trainer = BaumWelchTrainer(n_states=hmm_params['n_states'])
    trainer.A = np.array(hmm_params['transition_matrix'])
    trainer.means = hmm_params['emission_means']
    trainer.vars = hmm_params['emission_vars']
    trainer.pi = np.array(hmm_params['initial_probs'])

    # Load flows
    flows = db.get_flows(min_btc=0.1)
    if len(flows) < 100:
        print(f"[DISCOVERY] Not enough flows: {len(flows)}")
        return []

    # Discover patterns
    discovery = PatternDiscoveryEngine(db)
    patterns = discovery.discover_from_flows(flows, trainer, verbose=verbose)

    # Save valid patterns
    if patterns:
        discovery.save_patterns(patterns)

    return patterns
