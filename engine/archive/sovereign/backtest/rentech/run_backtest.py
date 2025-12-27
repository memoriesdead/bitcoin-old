"""
RenTech-Style Historical Backtesting - Main Entry Point

Tests 100-200 trading strategies on 16 years of Bitcoin data (2009-2025)
using walk-forward validation and statistical significance testing.

Usage:
    python -m engine.sovereign.backtest.rentech.run_backtest --full
    python -m engine.sovereign.backtest.rentech.run_backtest --quick
    python -m engine.sovereign.backtest.rentech.run_backtest --category TX_ZSCORE
"""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from .data_loader import RentechDataLoader
from .feature_engine import FeatureEngine
from .strategy_factory import StrategyFactory, Strategy
from .walk_forward import WalkForwardEngine, WFConfig
from .statistical_tests import StatisticalValidator, StrategyResult, results_to_dict


def run_backtest(
    strategies: List[Strategy],
    df,
    validator: StatisticalValidator,
    wf_engine: WalkForwardEngine,
    verbose: bool = True
) -> List[StrategyResult]:
    """Run backtest on all strategies."""
    results = []
    total = len(strategies)

    for i, strategy in enumerate(strategies):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{total} strategies...")

        # Run walk-forward validation
        wf_result = wf_engine.run_full_walkforward(strategy, df)

        # Get all out-of-sample trades
        trades = wf_engine.get_all_test_trades(strategy, df)

        # Validate
        result = validator.validate(
            strategy_name=strategy.name,
            category=strategy.category,
            description=strategy.description,
            trades=trades,
            wf_result=wf_result,
        )

        results.append(result)

    return results


def export_results(
    results: List[StrategyResult],
    output_path: str,
    data_summary: Dict
):
    """Export results to JSON file."""
    # Sort by recommendation and then by win rate
    priority = {'IMPLEMENT': 0, 'EDGE_TOO_SMALL': 1, 'NEED_MORE_DATA': 2,
                'NOT_SIGNIFICANT': 3, 'OVERFITTING': 4, 'NO_EDGE': 5, 'NO_TRADES': 6}

    sorted_results = sorted(
        results,
        key=lambda x: (priority.get(x.recommendation, 99), -x.win_rate)
    )

    # Build output
    output = {
        'run_date': datetime.now().isoformat(),
        'data_range': {
            'start': data_summary['total_start'],
            'end': data_summary['total_end'],
            'days': data_summary['total_days'],
        },
        'strategies_tested': len(results),
        'implementable': sum(1 for r in results if r.recommendation == 'IMPLEMENT'),
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

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_summary(results: List[StrategyResult]):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)

    # Count by recommendation
    recs = {}
    for r in results:
        recs[r.recommendation] = recs.get(r.recommendation, 0) + 1

    print("\nBy Recommendation:")
    for rec, count in sorted(recs.items()):
        print(f"  {rec}: {count}")

    # Show implementable strategies
    implementable = [r for r in results if r.recommendation == 'IMPLEMENT']

    if implementable:
        print(f"\n{'='*70}")
        print(f"IMPLEMENTABLE STRATEGIES ({len(implementable)})")
        print("=" * 70)

        # Sort by Sharpe
        implementable.sort(key=lambda x: x.sharpe_ratio, reverse=True)

        print(f"\n{'Strategy':<40} {'WR':>7} {'Trades':>7} {'Sharpe':>7} {'p-val':>8} {'Kelly':>7}")
        print("-" * 70)

        for r in implementable[:20]:
            print(f"{r.strategy_name:<40} {r.win_rate*100:>6.1f}% {r.total_trades:>7} "
                  f"{r.sharpe_ratio:>7.2f} {r.p_value:>8.4f} {r.kelly_fraction*100:>6.1f}%")

    # Show top strategies even if not implementable
    all_with_trades = [r for r in results if r.total_trades >= 100]
    if all_with_trades and not implementable:
        print(f"\n{'='*70}")
        print("TOP STRATEGIES (by win rate, need more validation)")
        print("=" * 70)

        all_with_trades.sort(key=lambda x: x.win_rate, reverse=True)

        print(f"\n{'Strategy':<40} {'WR':>7} {'Trades':>7} {'p-val':>8} {'Rec':>15}")
        print("-" * 70)

        for r in all_with_trades[:20]:
            print(f"{r.strategy_name:<40} {r.win_rate*100:>6.1f}% {r.total_trades:>7} "
                  f"{r.p_value:>8.4f} {r.recommendation:>15}")


def main():
    parser = argparse.ArgumentParser(description="RenTech-Style Backtesting")
    parser.add_argument('--full', action='store_true', help='Full backtest with all strategies')
    parser.add_argument('--quick', action='store_true', help='Quick test with subset')
    parser.add_argument('--category', type=str, help='Test only specific category')
    parser.add_argument('--output', type=str, default='data/rentech_results.json', help='Output file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    print("=" * 70)
    print("RENTECH-STYLE HISTORICAL BACKTESTING")
    print("=" * 70)

    start_time = time.time()

    # 1. Load data
    print("\n[1/5] Loading data...")
    loader = RentechDataLoader()
    summary = loader.get_summary()

    print(f"  Data range: {summary['total_start']} to {summary['total_end']}")
    print(f"  Total days: {summary['total_days']:,}")

    df = loader.load_merged_data()
    print(f"  DataFrame shape: {df.shape}")

    # 2. Calculate features
    print("\n[2/5] Calculating features...")
    engine = FeatureEngine()
    df = engine.add_all_features(df)

    feature_summary = engine.get_feature_summary(df)
    zscore_cols = [c for c in feature_summary.keys() if c != 'regimes']
    print(f"  Z-score columns: {len(zscore_cols)}")

    if 'regimes' in feature_summary:
        print(f"  Regime distribution:")
        for regime, count in feature_summary['regimes'].items():
            print(f"    {regime}: {count:,} ({count/len(df)*100:.1f}%)")

    # 3. Generate strategies
    print("\n[3/5] Generating strategies...")
    factory = StrategyFactory()
    strategies = factory.generate_all()

    if args.category:
        strategies = [s for s in strategies if s.category == args.category]
        print(f"  Filtered to category: {args.category}")

    if args.quick:
        # Quick test - just first 20 strategies
        strategies = strategies[:20]
        print(f"  Quick mode: testing {len(strategies)} strategies")

    print(f"  Total strategies to test: {len(strategies)}")

    # Show by category
    cat_counts = factory.get_strategy_count()
    for cat, count in cat_counts.items():
        if not args.category or cat == args.category:
            print(f"    {cat}: {count}")

    # 4. Run walk-forward backtest
    print("\n[4/5] Running walk-forward backtest...")
    wf_config = WFConfig(
        train_years=2,
        test_months=6,
        step_months=6,
    )
    wf_engine = WalkForwardEngine(wf_config)
    validator = StatisticalValidator()

    results = run_backtest(
        strategies=strategies,
        df=df,
        validator=validator,
        wf_engine=wf_engine,
        verbose=True,
    )

    # 5. Export and summarize
    print("\n[5/5] Exporting results...")
    export_results(results, args.output, summary)

    print_summary(results)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Completed in {elapsed:.1f} seconds")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
