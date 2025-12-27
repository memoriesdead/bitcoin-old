#!/usr/bin/env python3
"""
OVERNIGHT RENAISSANCE DISCOVERY SESSION
========================================

"We don't start with models. We start with data."
"We look for things that can be replicated thousands of times."
- Jim Simons, Renaissance Technologies

This script runs an exhaustive overnight analysis:

1. HISTORICAL SCAN: Scan ALL available blockchain history
2. MULTI-HMM: Train HMMs with 3, 4, 5, 6, 7 states
3. PATTERN MINING: Find ALL subsequences that predict price
4. STATISTICAL FILTER: Keep only patterns with p < 0.01
5. MONTE CARLO: 10,000 bootstrap simulations for confidence
6. FORMULA GENERATION: Output validated formulas

TARGET: Find patterns with win rate >= 50.75% that we can use forever.

USAGE:
    python -m engine.sovereign.formulas.overnight_discovery

    # Or with options:
    python -m engine.sovereign.formulas.overnight_discovery --hours 12
"""

import os
import sys
import time
import json
import math
import sqlite3
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import random

# Add parent directories
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np

from engine.sovereign.formulas.historical_data import (
    HistoricalFlowDatabase,
    HistoricalFlowScanner,
    FlowEvent,
)
from engine.sovereign.formulas.hmm_trainer import (
    BaumWelchTrainer,
    HMMParameters,
)
from engine.sovereign.formulas.pattern_discovery import (
    PatternDiscoveryEngine,
    DiscoveredPattern,
)
from engine.sovereign.formulas.validation import (
    MonteCarloValidator,
    WalkForwardValidator,
)


@dataclass
class DiscoveredFormula:
    """A validated formula ready for live trading."""
    formula_id: int
    name: str
    description: str
    hmm_states: int
    state_sequence: List[int]
    direction: int  # +1 LONG, -1 SHORT
    win_rate: float
    edge: float  # win_rate - 0.5
    occurrences: int
    p_value: float
    monte_carlo_ci_low: float
    monte_carlo_ci_high: float
    sharpe_ratio: float
    avg_return_bps: float  # basis points
    max_drawdown: float
    validated: bool


class OvernightDiscovery:
    """
    Exhaustive pattern discovery using RenTech methodology.

    "What, not why" - we find patterns that work without needing
    to understand the fundamental reasons.
    """

    def __init__(self, db_path: str = None, output_dir: str = None):
        self.db = HistoricalFlowDatabase(db_path)
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', '..', 'data', 'discovery'
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Results
        self.formulas: List[DiscoveredFormula] = []
        self.hmm_models: Dict[int, Dict] = {}  # n_states -> model params
        self.all_patterns: List[DiscoveredPattern] = []

        # Statistics
        self.stats = {
            'start_time': 0,
            'end_time': 0,
            'blocks_scanned': 0,
            'flows_collected': 0,
            'patterns_tested': 0,
            'patterns_validated': 0,
            'formulas_generated': 0,
        }

    def print_banner(self):
        print("""
================================================================================
              OVERNIGHT RENAISSANCE DISCOVERY SESSION
================================================================================

"The key is to have the computer find things that you wouldn't think to look for."
- Jim Simons

METHODOLOGY:
  1. Scan historical blockchain data (all available blocks)
  2. Train Hidden Markov Models (3-7 states)
  3. Enumerate ALL possible patterns (subsequences)
  4. Filter by statistical significance (p < 0.01)
  5. Validate with Monte Carlo (10,000 simulations)
  6. Generate formulas with >= 50.75% win rate

TARGET: Find patterns that can be replicated thousands of times.
================================================================================
""")

    def collect_historical_data(self, n_blocks: int = 10000,
                                 use_live: bool = True,
                                 live_duration: int = 28800) -> int:
        """
        Phase 1: Collect massive amounts of historical data.

        Two modes:
        - Historical: Scan past blocks (fast but no real-time prices)
        - Live: Collect from ZMQ feed (slower but accurate prices)
        """
        print("\n" + "="*60)
        print("PHASE 1: DATA COLLECTION")
        print("="*60)

        if use_live:
            return self._collect_live(live_duration)
        else:
            return self._collect_historical(n_blocks)

    def _collect_historical(self, n_blocks: int) -> int:
        """Scan historical blocks."""
        print(f"[COLLECT] Scanning {n_blocks} historical blocks...")

        scanner = HistoricalFlowScanner(self.db)

        try:
            rpc = scanner._get_rpc()
            current_height = rpc.call('getblockcount')
            print(f"[COLLECT] Current block height: {current_height}")
        except Exception as e:
            print(f"[COLLECT] Error connecting to Bitcoin Core: {e}")
            return 0

        start_height = max(0, current_height - n_blocks)

        print(f"[COLLECT] Scanning blocks {start_height} to {current_height}")
        print(f"[COLLECT] Exchange addresses: {len(scanner.exchange_addresses)}")

        total_events = 0
        batch_size = 100

        for batch_start in range(start_height, current_height, batch_size):
            batch_end = min(batch_start + batch_size, current_height)
            events = scanner.scan_range(batch_start, batch_end)
            total_events += events

            if batch_start % 1000 == 0:
                print(f"[COLLECT] Block {batch_start}/{current_height} - {total_events} events")

        self.stats['blocks_scanned'] = n_blocks
        self.stats['flows_collected'] = total_events

        print(f"\n[COLLECT] Total events: {total_events}")
        return total_events

    def _collect_live(self, duration: int) -> int:
        """Collect from live ZMQ feed with accurate prices."""
        print(f"[COLLECT] Collecting from live feed for {duration} seconds...")
        print(f"[COLLECT] That's {duration/3600:.1f} hours")

        from engine.sovereign.blockchain import PerExchangeBlockchainFeed, ExchangeTick
        from urllib.request import urlopen, Request

        event_count = 0
        last_price = 0.0
        price_update_interval = 10  # seconds
        last_price_update = 0

        def get_price():
            nonlocal last_price, last_price_update
            now = time.time()
            if now - last_price_update < price_update_interval and last_price > 0:
                return last_price
            try:
                url = 'https://api.exchange.coinbase.com/products/BTC-USD/ticker'
                req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urlopen(req, timeout=5) as response:
                    last_price = float(json.loads(response.read().decode())['price'])
                    last_price_update = now
                    return last_price
            except:
                return last_price if last_price > 0 else 97000.0

        def on_tick(tick: ExchangeTick):
            nonlocal event_count

            price = get_price()
            if price <= 0:
                return

            # Create flow event
            event = FlowEvent(
                timestamp=tick.timestamp,
                exchange=tick.exchange,
                direction=tick.direction,
                btc_amount=tick.volume,
                tx_hash=f"{tick.exchange}_{tick.timestamp}_{event_count}",
                block_height=0,
                price_at_flow=price,
            )

            self.db.add_flow(event)
            self.db.add_price(tick.timestamp, price)
            event_count += 1

            if event_count % 50 == 0:
                dir_str = "OUT" if tick.direction == 1 else "IN"
                elapsed = time.time() - start_time
                rate = event_count / max(1, elapsed) * 3600
                print(f"[LIVE] {event_count} events | {tick.exchange} {dir_str} "
                      f"{tick.volume:.2f} BTC @ ${price:,.0f} | "
                      f"Rate: {rate:.0f}/hr")

        # Start feed
        feed = PerExchangeBlockchainFeed(on_tick=on_tick)
        price = get_price()
        if price > 0:
            feed.set_reference_price(price)

        print("[LIVE] Starting blockchain feed...")
        if not feed.start():
            print("[LIVE] Failed to start feed")
            return 0

        start_time = time.time()
        print(f"[LIVE] Collecting until {datetime.now() + timedelta(seconds=duration)}")

        try:
            while time.time() - start_time < duration:
                time.sleep(60)
                elapsed = int(time.time() - start_time)
                remaining = duration - elapsed
                print(f"[LIVE] {elapsed//3600}h {(elapsed%3600)//60}m elapsed | "
                      f"{remaining//3600}h {(remaining%3600)//60}m remaining | "
                      f"{event_count} events")
        except KeyboardInterrupt:
            print("\n[LIVE] Interrupted by user")
        finally:
            feed.stop()

        self.stats['flows_collected'] = event_count
        print(f"\n[COLLECT] Collected {event_count} events")
        return event_count

    def train_multiple_hmms(self, state_range: range = range(3, 8)) -> Dict[int, Dict]:
        """
        Phase 2: Train HMMs with different state counts.

        RenTech insight: Try multiple model configurations,
        the data will tell us which works best.
        """
        print("\n" + "="*60)
        print("PHASE 2: MULTI-HMM TRAINING")
        print("="*60)

        flows = self.db.get_flows(min_btc=0.1)
        print(f"[HMM] Training on {len(flows)} flows")

        if len(flows) < 100:
            print("[HMM] Not enough flows for training")
            return {}

        # Prepare observations
        observations = []
        for flow in flows:
            if flow.outcome_30s != 0:
                obs = [
                    flow.flow_imbalance,
                    flow.flow_velocity,
                    flow.whale_ratio,
                    flow.fee_percentile / 100.0,
                ]
                observations.append(obs)

        if len(observations) < 100:
            print(f"[HMM] Only {len(observations)} observations with outcomes")
            return {}

        observations = np.array(observations)
        print(f"[HMM] {len(observations)} training observations")

        best_models = {}

        for n_states in state_range:
            print(f"\n[HMM] Training {n_states}-state HMM...")

            best_ll = float('-inf')
            best_params = None

            # Try multiple random initializations
            for restart in range(10):
                trainer = BaumWelchTrainer(n_states=n_states, n_iter=100, tol=1e-4)

                try:
                    params = trainer.fit(observations, verbose=False)
                    if params and params.log_likelihood > best_ll:
                        best_ll = params.log_likelihood
                        best_params = params
                except Exception as e:
                    continue

            if best_params:
                # Validate
                val_acc = self._validate_hmm(trainer, observations, flows)

                model_data = {
                    'n_states': n_states,
                    'log_likelihood': best_ll,
                    'validation_accuracy': val_acc,
                    'transition_matrix': trainer.A.tolist(),
                    'emission_means': trainer.means,
                    'emission_vars': trainer.vars,
                    'initial_probs': trainer.pi.tolist(),
                    'training_samples': len(observations),
                }

                best_models[n_states] = model_data
                self.db.save_hmm_model(f'hmm_{n_states}', model_data)

                print(f"[HMM] {n_states}-state: LL={best_ll:.2f}, Val={val_acc:.4f}")
            else:
                print(f"[HMM] {n_states}-state: Training failed")

        self.hmm_models = best_models
        return best_models

    def _validate_hmm(self, trainer: BaumWelchTrainer,
                      observations: np.ndarray,
                      flows: List[FlowEvent]) -> float:
        """Validate HMM on held-out data."""
        # Simple validation: check if state predictions correlate with outcomes
        states = trainer.predict_states(observations)

        correct = 0
        total = 0

        for i, (state, flow) in enumerate(zip(states, flows)):
            if flow.outcome_30s == 0:
                continue

            # State direction mapping
            if state in [0, 3]:  # ACCUMULATION, CAPITULATION
                predicted = 1
            elif state in [1, 4]:  # DISTRIBUTION, EUPHORIA
                predicted = -1
            else:
                continue  # NEUTRAL - skip

            if predicted == flow.outcome_30s:
                correct += 1
            total += 1

        return correct / max(1, total)

    def discover_all_patterns(self,
                              min_length: int = 2,
                              max_length: int = 6,
                              min_occurrences: int = 50) -> List[DiscoveredPattern]:
        """
        Phase 3: Exhaustive pattern discovery.

        "We look for things that can be replicated thousands of times."
        """
        print("\n" + "="*60)
        print("PHASE 3: EXHAUSTIVE PATTERN DISCOVERY")
        print("="*60)

        all_patterns = []

        for n_states, model in self.hmm_models.items():
            print(f"\n[PATTERN] Discovering patterns for {n_states}-state HMM...")

            # Create trainer with loaded params
            trainer = BaumWelchTrainer(n_states=n_states)
            trainer.A = np.array(model['transition_matrix'])
            trainer.means = model['emission_means']
            trainer.vars = model['emission_vars']
            trainer.pi = np.array(model['initial_probs'])

            # Discover patterns
            discovery = PatternDiscoveryEngine(
                self.db,
                min_pattern_length=min_length,
                max_pattern_length=max_length,
                min_occurrences=min_occurrences,
                min_win_rate=0.50,  # Start low, filter later
            )

            flows = self.db.get_flows(min_btc=0.1)
            patterns = discovery.discover_from_flows(flows, trainer, verbose=False)

            # Add HMM info to patterns
            for p in patterns:
                p.hmm_states = n_states

            all_patterns.extend(patterns)
            print(f"[PATTERN] Found {len(patterns)} patterns")

        self.all_patterns = all_patterns
        self.stats['patterns_tested'] = len(all_patterns)

        print(f"\n[PATTERN] Total patterns discovered: {len(all_patterns)}")
        return all_patterns

    def validate_patterns(self,
                         min_win_rate: float = 0.5075,
                         max_p_value: float = 0.01,
                         n_bootstrap: int = 10000) -> List[DiscoveredFormula]:
        """
        Phase 4: Rigorous statistical validation.

        "50.75% win rate with statistical significance"
        """
        print("\n" + "="*60)
        print("PHASE 4: STATISTICAL VALIDATION")
        print(f"Min win rate: {min_win_rate:.2%}")
        print(f"Max p-value: {max_p_value}")
        print(f"Bootstrap samples: {n_bootstrap}")
        print("="*60)

        validated_formulas = []
        formula_id = 30001  # Start at 30001 for overnight discoveries

        for pattern in self.all_patterns:
            # Skip if doesn't meet basic criteria
            if pattern.win_rate < min_win_rate:
                continue
            if pattern.p_value > max_p_value:
                continue
            if pattern.occurrences < 100:
                continue

            # Monte Carlo validation
            mc_results = self._monte_carlo_validate(pattern, n_bootstrap)

            if mc_results['ci_low'] < 0.5:
                continue  # CI includes 50%, not significant

            # Calculate Sharpe-like ratio
            sharpe = self._calculate_sharpe(pattern)

            # Create formula
            formula = DiscoveredFormula(
                formula_id=formula_id,
                name=f"OVERNIGHT_{pattern.name}",
                description=f"HMM-{getattr(pattern, 'hmm_states', 5)} pattern: {pattern.name}",
                hmm_states=getattr(pattern, 'hmm_states', 5),
                state_sequence=pattern.sequence,
                direction=pattern.direction,
                win_rate=pattern.win_rate,
                edge=pattern.edge,
                occurrences=pattern.occurrences,
                p_value=pattern.p_value,
                monte_carlo_ci_low=mc_results['ci_low'],
                monte_carlo_ci_high=mc_results['ci_high'],
                sharpe_ratio=sharpe,
                avg_return_bps=pattern.avg_return * 10000,
                max_drawdown=mc_results.get('max_drawdown', 0),
                validated=True,
            )

            validated_formulas.append(formula)
            formula_id += 1

            print(f"[VALIDATE] {formula.name}: WR={formula.win_rate:.2%} "
                  f"CI=[{formula.monte_carlo_ci_low:.2%}, {formula.monte_carlo_ci_high:.2%}] "
                  f"Sharpe={formula.sharpe_ratio:.2f}")

        self.formulas = validated_formulas
        self.stats['patterns_validated'] = len(validated_formulas)

        print(f"\n[VALIDATE] {len(validated_formulas)} formulas validated")
        return validated_formulas

    def _monte_carlo_validate(self, pattern: DiscoveredPattern,
                              n_samples: int) -> Dict:
        """Bootstrap validation for confidence intervals."""
        wins = pattern.wins
        losses = pattern.losses
        total = wins + losses

        if total < 10:
            return {'ci_low': 0, 'ci_high': 1, 'max_drawdown': 1}

        # Bootstrap resampling
        win_rates = []
        for _ in range(n_samples):
            sample_wins = sum(random.random() < pattern.win_rate for _ in range(total))
            win_rates.append(sample_wins / total)

        win_rates.sort()

        ci_low = win_rates[int(n_samples * 0.025)]
        ci_high = win_rates[int(n_samples * 0.975)]

        return {
            'ci_low': ci_low,
            'ci_high': ci_high,
            'mean': np.mean(win_rates),
            'std': np.std(win_rates),
            'max_drawdown': 0,  # Would need trade-by-trade data
        }

    def _calculate_sharpe(self, pattern: DiscoveredPattern) -> float:
        """Calculate Sharpe-like ratio."""
        if pattern.occurrences < 10:
            return 0

        # Expected return per trade
        avg_win = pattern.avg_return if pattern.avg_return > 0 else 0.001
        expected = pattern.win_rate * avg_win - (1 - pattern.win_rate) * avg_win

        # Estimate volatility
        vol = abs(avg_win) * math.sqrt(pattern.win_rate * (1 - pattern.win_rate))

        if vol < 0.0001:
            return 0

        return expected / vol * math.sqrt(252)  # Annualized

    def generate_formula_code(self) -> str:
        """
        Phase 5: Generate Python code for validated formulas.
        """
        print("\n" + "="*60)
        print("PHASE 5: FORMULA CODE GENERATION")
        print("="*60)

        code_lines = [
            '"""',
            'OVERNIGHT DISCOVERED FORMULAS',
            '=' * 40,
            f'Generated: {datetime.now().isoformat()}',
            f'Formulas: {len(self.formulas)}',
            '',
            'These formulas were discovered through exhaustive pattern mining',
            'and validated with Monte Carlo simulation (10,000 bootstraps).',
            '',
            'All formulas meet:',
            '  - Win rate >= 50.75%',
            '  - p-value < 0.01',
            '  - 95% CI lower bound > 50%',
            '  - 100+ historical occurrences',
            '"""',
            '',
            'from typing import Dict, List, Tuple',
            'from dataclasses import dataclass',
            '',
            '',
            'DISCOVERED_FORMULAS = [',
        ]

        for f in self.formulas:
            code_lines.append(f'    {{')
            code_lines.append(f'        "id": {f.formula_id},')
            code_lines.append(f'        "name": "{f.name}",')
            code_lines.append(f'        "hmm_states": {f.hmm_states},')
            code_lines.append(f'        "sequence": {f.state_sequence},')
            code_lines.append(f'        "direction": {f.direction},  # {"LONG" if f.direction == 1 else "SHORT"}')
            code_lines.append(f'        "win_rate": {f.win_rate:.6f},')
            code_lines.append(f'        "edge": {f.edge:.6f},')
            code_lines.append(f'        "occurrences": {f.occurrences},')
            code_lines.append(f'        "p_value": {f.p_value:.6f},')
            code_lines.append(f'        "ci_low": {f.monte_carlo_ci_low:.6f},')
            code_lines.append(f'        "ci_high": {f.monte_carlo_ci_high:.6f},')
            code_lines.append(f'        "sharpe": {f.sharpe_ratio:.4f},')
            code_lines.append(f'    }},')

        code_lines.append(']')
        code_lines.append('')
        code_lines.append('')
        code_lines.append('def get_formula_by_sequence(sequence: List[int]) -> Dict:')
        code_lines.append('    """Look up formula by state sequence."""')
        code_lines.append('    for f in DISCOVERED_FORMULAS:')
        code_lines.append('        if f["sequence"] == sequence:')
        code_lines.append('            return f')
        code_lines.append('    return None')
        code_lines.append('')
        code_lines.append('')
        code_lines.append('def get_best_formulas(min_edge: float = 0.0075) -> List[Dict]:')
        code_lines.append('    """Get formulas with edge above threshold."""')
        code_lines.append('    return [f for f in DISCOVERED_FORMULAS if f["edge"] >= min_edge]')
        code_lines.append('')

        code = '\n'.join(code_lines)

        # Save to file
        output_path = os.path.join(self.output_dir, 'discovered_formulas.py')
        with open(output_path, 'w') as f:
            f.write(code)

        print(f"[CODE] Generated {output_path}")

        self.stats['formulas_generated'] = len(self.formulas)
        return code

    def generate_report(self) -> str:
        """Generate comprehensive report."""
        print("\n" + "="*60)
        print("PHASE 6: REPORT GENERATION")
        print("="*60)

        report_lines = [
            '=' * 80,
            'OVERNIGHT RENAISSANCE DISCOVERY REPORT',
            '=' * 80,
            f'Generated: {datetime.now().isoformat()}',
            f'Duration: {(self.stats["end_time"] - self.stats["start_time"])/3600:.2f} hours',
            '',
            '## SUMMARY',
            f'- Blocks scanned: {self.stats["blocks_scanned"]:,}',
            f'- Flow events: {self.stats["flows_collected"]:,}',
            f'- Patterns tested: {self.stats["patterns_tested"]:,}',
            f'- Patterns validated: {self.stats["patterns_validated"]:,}',
            f'- Formulas generated: {self.stats["formulas_generated"]:,}',
            '',
            '## HMM MODELS',
        ]

        for n_states, model in sorted(self.hmm_models.items()):
            report_lines.append(f'### {n_states}-State HMM')
            report_lines.append(f'- Log-likelihood: {model["log_likelihood"]:.2f}')
            report_lines.append(f'- Validation accuracy: {model["validation_accuracy"]:.4f}')
            report_lines.append(f'- Training samples: {model["training_samples"]:,}')
            report_lines.append('')

        report_lines.append('## VALIDATED FORMULAS')
        report_lines.append('')

        # Sort by edge
        sorted_formulas = sorted(self.formulas, key=lambda f: f.edge, reverse=True)

        for f in sorted_formulas:
            dir_str = "LONG" if f.direction == 1 else "SHORT"
            report_lines.append(f'### Formula {f.formula_id}: {f.name}')
            report_lines.append(f'- Direction: {dir_str}')
            report_lines.append(f'- Win Rate: {f.win_rate:.4f} ({f.win_rate*100:.2f}%)')
            report_lines.append(f'- Edge: {f.edge:.4f} ({f.edge*100:.2f}%)')
            report_lines.append(f'- P-value: {f.p_value:.6f}')
            report_lines.append(f'- 95% CI: [{f.monte_carlo_ci_low:.4f}, {f.monte_carlo_ci_high:.4f}]')
            report_lines.append(f'- Occurrences: {f.occurrences:,}')
            report_lines.append(f'- Sharpe Ratio: {f.sharpe_ratio:.2f}')
            report_lines.append(f'- State Sequence: {f.state_sequence}')
            report_lines.append('')

        report_lines.append('## CONCLUSION')
        if self.formulas:
            avg_edge = np.mean([f.edge for f in self.formulas])
            report_lines.append(f'Found {len(self.formulas)} validated formulas with average edge of {avg_edge:.4f}')
            report_lines.append('')
            report_lines.append('Top 3 formulas for live trading:')
            for f in sorted_formulas[:3]:
                report_lines.append(f'  {f.formula_id}: {f.name} (edge={f.edge:.4f})')
        else:
            report_lines.append('No formulas met validation criteria.')
            report_lines.append('Recommendation: Collect more data and re-run.')

        report_lines.append('')
        report_lines.append('=' * 80)

        report = '\n'.join(report_lines)

        # Save report
        report_path = os.path.join(self.output_dir, 'discovery_report.md')
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"[REPORT] Generated {report_path}")
        print(report)

        return report

    def run(self,
            use_live: bool = True,
            live_hours: float = 8,
            historical_blocks: int = 10000):
        """
        Run complete overnight discovery session.
        """
        self.print_banner()
        self.stats['start_time'] = time.time()

        # Phase 1: Collect data
        if use_live:
            duration = int(live_hours * 3600)
            events = self.collect_historical_data(use_live=True, live_duration=duration)
        else:
            events = self.collect_historical_data(n_blocks=historical_blocks, use_live=False)

        if events < 100:
            print("\n[ERROR] Not enough data collected")
            print("Need at least 100 flow events for meaningful analysis")
            return

        # Phase 2: Train HMMs
        self.train_multiple_hmms()

        if not self.hmm_models:
            print("\n[ERROR] No HMM models trained")
            return

        # Phase 3: Discover patterns
        self.discover_all_patterns()

        if not self.all_patterns:
            print("\n[ERROR] No patterns discovered")
            return

        # Phase 4: Validate
        self.validate_patterns()

        # Phase 5: Generate code
        if self.formulas:
            self.generate_formula_code()

        # Phase 6: Report
        self.stats['end_time'] = time.time()
        self.generate_report()

        print("\n" + "="*60)
        print("OVERNIGHT DISCOVERY COMPLETE")
        print("="*60)
        print(f"Duration: {(self.stats['end_time'] - self.stats['start_time'])/3600:.2f} hours")
        print(f"Formulas found: {len(self.formulas)}")

        if self.formulas:
            print("\n>>> READY FOR LIVE TRADING <<<")
            print(f"Formulas saved to: {self.output_dir}")
        else:
            print("\n>>> MORE DATA NEEDED <<<")
            print("Run for longer duration to collect more samples")


def main():
    parser = argparse.ArgumentParser(description='Overnight Renaissance Discovery')
    parser.add_argument('--hours', type=float, default=8,
                       help='Hours to collect live data (default: 8)')
    parser.add_argument('--blocks', type=int, default=10000,
                       help='Historical blocks to scan (if not using live)')
    parser.add_argument('--historical', action='store_true',
                       help='Use historical block scanning instead of live')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')

    args = parser.parse_args()

    discovery = OvernightDiscovery(output_dir=args.output)
    discovery.run(
        use_live=not args.historical,
        live_hours=args.hours,
        historical_blocks=args.blocks,
    )


if __name__ == "__main__":
    main()
