"""
Exhaustive RenTech Backtest Runner
==================================

Tests EVERY pattern on 16 years of Bitcoin data.
Like RenTech: test everything, keep what works statistically.

Strategy Count:
- Comprehensive: ~485 strategies
- Exhaustive: ~600+ new strategies
- Total: 1,000+ patterns tested

Usage:
    python -m engine.sovereign.backtest.rentech.run_exhaustive --full
    python -m engine.sovereign.backtest.rentech.run_exhaustive --quick
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np

from .data_loader import RentechDataLoader
from .feature_engine import FeatureEngine
from .advanced_features import AdvancedFeatureEngine
from .exhaustive_features import ExhaustiveFeatureEngine
from .comprehensive_strategies import ComprehensiveStrategyFactory, ComprehensiveStrategy
from .exhaustive_strategies import ExhaustiveStrategyFactory, ExhaustiveStrategy
from .walk_forward import WalkForwardEngine, WFConfig, Trade
from .statistical_tests import StatisticalValidator, StrategyResult, results_to_dict


def run_comprehensive_strategy(
    strategy: ComprehensiveStrategy,
    df: pd.DataFrame,
    wf_engine: WalkForwardEngine,
) -> List[Trade]:
    """Run backtest for comprehensive strategy."""
    class Adapter:
        def __init__(self, s):
            self.strategy = s
            self.name = s.name
            self.category = s.category
            self.description = s.description
            self.hold_days = s.hold_days

        def check_entry(self, row):
            return self.strategy.check_entry(row)

        def get_signal_direction(self):
            return self.strategy.get_signal_direction()

    adapter = Adapter(strategy)
    return wf_engine.get_all_test_trades(adapter, df)


def run_exhaustive_strategy(
    strategy: ExhaustiveStrategy,
    df: pd.DataFrame,
    wf_engine: WalkForwardEngine,
) -> List[Trade]:
    """Run backtest for exhaustive strategy."""
    class Adapter:
        def __init__(self, s, dataframe):
            self.strategy = s
            self.name = s.name
            self.category = s.category
            self.description = s.description
            self.hold_days = s.hold_days
            self.df = dataframe
            self.prev_row = None

        def check_entry(self, row):
            result = self.strategy.check_entry(row, self.prev_row, self.df)
            self.prev_row = row
            return result

        def get_signal_direction(self):
            return self.strategy.get_signal_direction()

    adapter = Adapter(strategy, df)
    return wf_engine.get_all_test_trades(adapter, df)


def run_all_strategies(
    comprehensive: List[ComprehensiveStrategy],
    exhaustive: List[ExhaustiveStrategy],
    df: pd.DataFrame,
    wf_engine: WalkForwardEngine,
    validator: StatisticalValidator,
    verbose: bool = True
) -> List[StrategyResult]:
    """Run all strategies and validate."""
    results = []
    total = len(comprehensive) + len(exhaustive)
    count = 0

    # Run comprehensive strategies
    if verbose:
        print(f"\n  Running {len(comprehensive)} comprehensive strategies...")

    for strategy in comprehensive:
        count += 1
        if verbose and count % 50 == 0:
            print(f"    Progress: {count}/{total} ({count/total*100:.1f}%)")

        trades = run_comprehensive_strategy(strategy, df, wf_engine)

        from .walk_forward import WalkForwardResult
        wf_result = WalkForwardResult(strategy_name=strategy.name, windows=[])

        result = validator.validate(
            strategy_name=strategy.name,
            category=strategy.category,
            description=strategy.description,
            trades=trades,
            wf_result=wf_result,
        )
        results.append(result)

    # Run exhaustive strategies
    if verbose:
        print(f"\n  Running {len(exhaustive)} exhaustive strategies...")

    for strategy in exhaustive:
        count += 1
        if verbose and count % 50 == 0:
            print(f"    Progress: {count}/{total} ({count/total*100:.1f}%)")

        trades = run_exhaustive_strategy(strategy, df, wf_engine)

        from .walk_forward import WalkForwardResult
        wf_result = WalkForwardResult(strategy_name=strategy.name, windows=[])

        result = validator.validate(
            strategy_name=strategy.name,
            category=strategy.category,
            description=strategy.description,
            trades=trades,
            wf_result=wf_result,
        )
        results.append(result)

    return results


def export_results(results: List[StrategyResult], output_path: str, data_summary: Dict):
    """Export results to JSON with full analysis."""
    priority = {'IMPLEMENT': 0, 'EDGE_TOO_SMALL': 1, 'NEED_MORE_DATA': 2,
                'NOT_SIGNIFICANT': 3, 'OVERFITTING': 4, 'NO_EDGE': 5, 'NO_TRADES': 6}

    sorted_results = sorted(
        results,
        key=lambda x: (priority.get(x.recommendation, 99), -x.sharpe_ratio)
    )

    output = {
        'run_date': datetime.now().isoformat(),
        'run_type': 'exhaustive',
        'data_range': {
            'start': data_summary['total_start'],
            'end': data_summary['total_end'],
            'days': data_summary['total_days'],
        },
        'strategies_tested': len(results),
        'summary': {
            'IMPLEMENT': sum(1 for r in results if r.recommendation == 'IMPLEMENT'),
            'EDGE_TOO_SMALL': sum(1 for r in results if r.recommendation == 'EDGE_TOO_SMALL'),
            'NEED_MORE_DATA': sum(1 for r in results if r.recommendation == 'NEED_MORE_DATA'),
            'NOT_SIGNIFICANT': sum(1 for r in results if r.recommendation == 'NOT_SIGNIFICANT'),
            'OVERFITTING': sum(1 for r in results if r.recommendation == 'OVERFITTING'),
            'NO_EDGE': sum(1 for r in results if r.recommendation == 'NO_EDGE'),
            'NO_TRADES': sum(1 for r in results if r.recommendation == 'NO_TRADES'),
        },
        'results': [results_to_dict(r) for r in sorted_results],
    }

    # Category breakdown
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {
                'tested': 0,
                'implementable': 0,
                'with_edge': 0,
                'best_sharpe': 0,
                'best_wr': 0,
                'best_strategy': None
            }
        categories[r.category]['tested'] += 1

        if r.recommendation == 'IMPLEMENT':
            categories[r.category]['implementable'] += 1

        if r.win_rate > 0.5 and r.total_trades >= 100:
            categories[r.category]['with_edge'] += 1

        if r.sharpe_ratio > categories[r.category]['best_sharpe']:
            categories[r.category]['best_sharpe'] = r.sharpe_ratio
            categories[r.category]['best_strategy'] = r.strategy_name

        if r.win_rate > categories[r.category]['best_wr']:
            categories[r.category]['best_wr'] = r.win_rate

    output['category_summary'] = categories

    # Top strategies by different metrics
    with_trades = [r for r in results if r.total_trades >= 100]

    if with_trades:
        output['top_by_sharpe'] = [
            results_to_dict(r) for r in sorted(with_trades, key=lambda x: x.sharpe_ratio, reverse=True)[:20]
        ]
        output['top_by_win_rate'] = [
            results_to_dict(r) for r in sorted(with_trades, key=lambda x: x.win_rate, reverse=True)[:20]
        ]
        output['top_by_pnl'] = [
            results_to_dict(r) for r in sorted(with_trades, key=lambda x: x.total_pnl_pct, reverse=True)[:20]
        ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_exhaustive_summary(results: List[StrategyResult]):
    """Print comprehensive summary."""
    print("\n" + "=" * 90)
    print("EXHAUSTIVE RENTECH BACKTEST - COMPLETE RESULTS")
    print("=" * 90)

    # Summary stats
    total = len(results)
    implementable = [r for r in results if r.recommendation == 'IMPLEMENT']
    edge_too_small = [r for r in results if r.recommendation == 'EDGE_TOO_SMALL']
    need_more_data = [r for r in results if r.recommendation == 'NEED_MORE_DATA']
    with_edge = [r for r in results if r.win_rate > 0.5 and r.total_trades >= 100]

    print(f"\nTotal strategies tested: {total}")
    print(f"IMPLEMENTABLE (all RenTech criteria): {len(implementable)}")
    print(f"Edge too small (WR 50-51%): {len(edge_too_small)}")
    print(f"Need more data (< 500 trades): {len(need_more_data)}")
    print(f"Showing edge (> 50% WR, 100+ trades): {len(with_edge)}")

    # By recommendation
    print("\n" + "-" * 50)
    print("By Recommendation:")
    recs = {}
    for r in results:
        recs[r.recommendation] = recs.get(r.recommendation, 0) + 1
    for rec, count in sorted(recs.items()):
        print(f"  {rec}: {count}")

    # By category
    print("\n" + "-" * 50)
    print("By Category (strategies with edge / tested):")
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {'total': 0, 'edge': 0, 'impl': 0, 'best': None}
        categories[r.category]['total'] += 1
        if r.win_rate > 0.5 and r.total_trades >= 100:
            categories[r.category]['edge'] += 1
        if r.recommendation == 'IMPLEMENT':
            categories[r.category]['impl'] += 1
        if categories[r.category]['best'] is None or r.sharpe_ratio > categories[r.category]['best'].sharpe_ratio:
            if r.total_trades >= 100:
                categories[r.category]['best'] = r

    for cat, info in sorted(categories.items(), key=lambda x: x[1]['impl'], reverse=True):
        best_str = ""
        if info['best']:
            best_str = f" | Best: {info['best'].strategy_name[:30]} (Sharpe {info['best'].sharpe_ratio:.2f})"
        print(f"  {cat:20} {info['impl']:3} impl, {info['edge']:3}/{info['total']:3} edge{best_str}")

    # IMPLEMENTABLE strategies
    if implementable:
        print("\n" + "=" * 90)
        print(f"IMPLEMENTABLE STRATEGIES ({len(implementable)})")
        print("=" * 90)

        implementable.sort(key=lambda x: x.sharpe_ratio, reverse=True)

        print(f"\n{'Strategy':<45} {'Cat':<15} {'WR':>6} {'#':>6} {'PnL':>8} {'Sharpe':>7} {'Kelly':>6}")
        print("-" * 90)

        for r in implementable[:40]:
            print(f"{r.strategy_name[:44]:<45} {r.category[:14]:<15} {r.win_rate*100:>5.1f}% "
                  f"{r.total_trades:>6} {r.total_pnl_pct:>7.0f}% {r.sharpe_ratio:>7.2f} {r.kelly_fraction*100:>5.1f}%")

    # HIGH WIN RATE strategies (> 55%)
    high_wr = [r for r in results if r.win_rate > 0.55 and r.total_trades >= 50]
    if high_wr:
        print("\n" + "=" * 90)
        print(f"HIGH WIN RATE STRATEGIES (> 55%, 50+ trades): {len(high_wr)}")
        print("=" * 90)

        high_wr.sort(key=lambda x: x.win_rate, reverse=True)

        print(f"\n{'Strategy':<45} {'Cat':<15} {'WR':>6} {'#':>6} {'PnL':>8} {'p-val':>8}")
        print("-" * 90)

        for r in high_wr[:30]:
            print(f"{r.strategy_name[:44]:<45} {r.category[:14]:<15} {r.win_rate*100:>5.1f}% "
                  f"{r.total_trades:>6} {r.total_pnl_pct:>7.0f}% {r.p_value:>8.4f}")

    # Best by Sharpe
    by_sharpe = sorted([r for r in results if r.total_trades >= 100], key=lambda x: x.sharpe_ratio, reverse=True)
    if by_sharpe:
        print("\n" + "=" * 90)
        print("TOP 30 BY SHARPE RATIO (100+ trades)")
        print("=" * 90)

        print(f"\n{'Strategy':<45} {'Cat':<15} {'WR':>6} {'#':>6} {'Sharpe':>7} {'Rec':>12}")
        print("-" * 90)

        for r in by_sharpe[:30]:
            print(f"{r.strategy_name[:44]:<45} {r.category[:14]:<15} {r.win_rate*100:>5.1f}% "
                  f"{r.total_trades:>6} {r.sharpe_ratio:>7.2f} {r.recommendation:>12}")

    # NEW DISCOVERIES (exhaustive categories)
    new_cats = ['MICROSTRUCTURE', 'HALVING', 'WEEK_OF_MONTH', 'CROSS_CORRELATION',
                'VOL_REGIME', 'MULTI_TIMEFRAME', 'REGIME_TRANSITION', 'BLOCKCHAIN',
                'EXTREME', 'PATTERN']
    new_discoveries = [r for r in results if r.category in new_cats and r.win_rate > 0.5 and r.total_trades >= 50]

    if new_discoveries:
        print("\n" + "=" * 90)
        print(f"NEW PATTERN DISCOVERIES ({len(new_discoveries)} strategies)")
        print("=" * 90)

        new_discoveries.sort(key=lambda x: x.sharpe_ratio, reverse=True)

        print(f"\n{'Strategy':<45} {'Cat':<15} {'WR':>6} {'#':>6} {'Sharpe':>7}")
        print("-" * 90)

        for r in new_discoveries[:30]:
            print(f"{r.strategy_name[:44]:<45} {r.category[:14]:<15} {r.win_rate*100:>5.1f}% "
                  f"{r.total_trades:>6} {r.sharpe_ratio:>7.2f}")


def main():
    parser = argparse.ArgumentParser(description="Exhaustive RenTech Backtest")
    parser.add_argument('--full', action='store_true', help='Full test with all strategies')
    parser.add_argument('--quick', action='store_true', help='Quick test with subset')
    parser.add_argument('--category', type=str, help='Test only specific category')
    parser.add_argument('--output', type=str, default='data/exhaustive_results.json')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    print("=" * 90)
    print("EXHAUSTIVE RENTECH BACKTEST - FIND EVERY PATTERN")
    print("=" * 90)
    print("\nRenTech Principle: Test EVERYTHING. Keep what works statistically.")

    start_time = time.time()

    # 1. Load data
    print("\n[1/7] Loading data...")
    loader = RentechDataLoader()
    summary = loader.get_summary()
    print(f"  Data range: {summary['total_start']} to {summary['total_end']}")
    print(f"  Total days: {summary['total_days']:,}")

    df = loader.load_merged_data()
    print(f"  DataFrame shape: {df.shape}")

    # 2. Basic features
    print("\n[2/7] Calculating basic features...")
    basic_engine = FeatureEngine()
    df = basic_engine.add_all_features(df)

    # 3. Advanced features
    print("\n[3/7] Calculating advanced features...")
    advanced_engine = AdvancedFeatureEngine()
    df = advanced_engine.add_all_features(df)

    # 4. Exhaustive features
    print("\n[4/7] Calculating exhaustive features...")
    exhaustive_engine = ExhaustiveFeatureEngine()
    df = exhaustive_engine.add_all_features(df)

    print(f"  Total columns: {len(df.columns)}")

    # 5. Generate strategies
    print("\n[5/7] Generating all strategies...")

    comp_factory = ComprehensiveStrategyFactory()
    comprehensive = comp_factory.generate_all()

    exh_factory = ExhaustiveStrategyFactory()
    exhaustive = exh_factory.generate_all()

    if args.category:
        comprehensive = [s for s in comprehensive if s.category == args.category]
        exhaustive = [s for s in exhaustive if s.category == args.category]
        print(f"  Filtered to category: {args.category}")

    if args.quick:
        comprehensive = comprehensive[:100]
        exhaustive = exhaustive[:100]
        print(f"  Quick mode: testing subset")

    print(f"  Comprehensive strategies: {len(comprehensive)}")
    print(f"  Exhaustive strategies: {len(exhaustive)}")
    print(f"  TOTAL: {len(comprehensive) + len(exhaustive)}")

    # Show by category
    print("\n  By Category:")
    all_cats = {}
    for s in comprehensive:
        all_cats[s.category] = all_cats.get(s.category, 0) + 1
    for s in exhaustive:
        all_cats[s.category] = all_cats.get(s.category, 0) + 1
    for cat, count in sorted(all_cats.items()):
        print(f"    {cat}: {count}")

    # 6. Run walk-forward backtest
    print("\n[6/7] Running walk-forward backtest...")
    wf_config = WFConfig(
        train_years=2,
        test_months=6,
        step_months=6,
    )
    wf_engine = WalkForwardEngine(wf_config)
    validator = StatisticalValidator()

    results = run_all_strategies(
        comprehensive=comprehensive,
        exhaustive=exhaustive,
        df=df,
        wf_engine=wf_engine,
        validator=validator,
        verbose=True,
    )

    # 7. Export and summarize
    print("\n[7/7] Exporting results...")
    export_results(results, args.output, summary)

    print_exhaustive_summary(results)

    elapsed = time.time() - start_time
    print(f"\n{'='*90}")
    print(f"Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
