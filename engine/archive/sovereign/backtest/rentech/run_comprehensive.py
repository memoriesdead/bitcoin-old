"""
Comprehensive RenTech Backtest Runner
=====================================

Tests 500+ strategies on historical data using:
- Advanced technical features (RSI, MACD, BB, ATR)
- Blockchain features (TX z-scores, regimes)
- Seasonality features
- Walk-forward validation
- Statistical significance testing

Usage:
    python -m engine.sovereign.backtest.rentech.run_comprehensive --full
    python -m engine.sovereign.backtest.rentech.run_comprehensive --quick
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from .data_loader import RentechDataLoader
from .feature_engine import FeatureEngine
from .advanced_features import AdvancedFeatureEngine
from .comprehensive_strategies import ComprehensiveStrategyFactory, ComprehensiveStrategy
from .walk_forward import WalkForwardEngine, WFConfig, Trade
from .statistical_tests import StatisticalValidator, StrategyResult, results_to_dict


def run_strategy_backtest(
    strategy: ComprehensiveStrategy,
    df,
    wf_engine: WalkForwardEngine,
) -> List[Trade]:
    """Run backtest for a single strategy and get trades."""
    # Create a simple Strategy adapter
    class StrategyAdapter:
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

    adapter = StrategyAdapter(strategy)
    trades = wf_engine.get_all_test_trades(adapter, df)
    return trades


def run_comprehensive_backtest(
    strategies: List[ComprehensiveStrategy],
    df,
    wf_engine: WalkForwardEngine,
    validator: StatisticalValidator,
    verbose: bool = True
) -> List[StrategyResult]:
    """Run backtest on all strategies."""
    results = []
    total = len(strategies)

    for i, strategy in enumerate(strategies):
        if verbose and (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{total} strategies...")

        # Get trades
        trades = run_strategy_backtest(strategy, df, wf_engine)

        # Create dummy WF result for validation
        from .walk_forward import WalkForwardResult
        wf_result = WalkForwardResult(
            strategy_name=strategy.name,
            windows=[],
        )

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
    """Export results to JSON."""
    priority = {'IMPLEMENT': 0, 'EDGE_TOO_SMALL': 1, 'NEED_MORE_DATA': 2,
                'NOT_SIGNIFICANT': 3, 'OVERFITTING': 4, 'NO_EDGE': 5, 'NO_TRADES': 6}

    sorted_results = sorted(
        results,
        key=lambda x: (priority.get(x.recommendation, 99), -x.win_rate)
    )

    output = {
        'run_date': datetime.now().isoformat(),
        'run_type': 'comprehensive',
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

    # Category breakdown
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {'tested': 0, 'with_edge': 0, 'best_wr': 0}
        categories[r.category]['tested'] += 1
        if r.win_rate > 0.5 and r.total_trades >= 100:
            categories[r.category]['with_edge'] += 1
        if r.win_rate > categories[r.category]['best_wr']:
            categories[r.category]['best_wr'] = r.win_rate

    output['category_summary'] = categories

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_summary(results: List[StrategyResult]):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BACKTEST SUMMARY")
    print("=" * 80)

    # Count by recommendation
    recs = {}
    for r in results:
        recs[r.recommendation] = recs.get(r.recommendation, 0) + 1

    print("\nBy Recommendation:")
    for rec, count in sorted(recs.items()):
        print(f"  {rec}: {count}")

    # Count by category
    print("\nBy Category (strategies with edge, 100+ trades):")
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {'total': 0, 'edge': 0, 'best': None}
        categories[r.category]['total'] += 1
        if r.win_rate > 0.5 and r.total_trades >= 100:
            categories[r.category]['edge'] += 1
            if categories[r.category]['best'] is None or r.win_rate > categories[r.category]['best'].win_rate:
                categories[r.category]['best'] = r

    for cat, info in sorted(categories.items()):
        best_str = ""
        if info['best']:
            best_str = f" | Best: {info['best'].strategy_name} ({info['best'].win_rate*100:.1f}%)"
        print(f"  {cat}: {info['edge']}/{info['total']} with edge{best_str}")

    # Show implementable strategies
    implementable = [r for r in results if r.recommendation == 'IMPLEMENT']

    if implementable:
        print(f"\n{'='*80}")
        print(f"IMPLEMENTABLE STRATEGIES ({len(implementable)})")
        print("=" * 80)

        implementable.sort(key=lambda x: x.sharpe_ratio, reverse=True)

        print(f"\n{'Strategy':<50} {'WR':>6} {'#':>5} {'PnL':>8} {'Sharpe':>7} {'Kelly':>6}")
        print("-" * 80)

        for r in implementable[:30]:
            print(f"{r.strategy_name:<50} {r.win_rate*100:>5.1f}% {r.total_trades:>5} "
                  f"{r.total_pnl_pct:>7.0f}% {r.sharpe_ratio:>7.2f} {r.kelly_fraction*100:>5.1f}%")

    # Show top strategies by win rate
    all_with_trades = [r for r in results if r.total_trades >= 100]
    if all_with_trades:
        print(f"\n{'='*80}")
        print("TOP 30 STRATEGIES BY WIN RATE (100+ trades)")
        print("=" * 80)

        all_with_trades.sort(key=lambda x: x.win_rate, reverse=True)

        print(f"\n{'Strategy':<50} {'WR':>6} {'#':>5} {'PnL':>8} {'p-val':>8} {'Rec':>15}")
        print("-" * 80)

        for r in all_with_trades[:30]:
            print(f"{r.strategy_name:<50} {r.win_rate*100:>5.1f}% {r.total_trades:>5} "
                  f"{r.total_pnl_pct:>7.0f}% {r.p_value:>8.4f} {r.recommendation:>15}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive RenTech Backtest")
    parser.add_argument('--full', action='store_true', help='Full backtest with all strategies')
    parser.add_argument('--quick', action='store_true', help='Quick test with subset')
    parser.add_argument('--category', type=str, help='Test only specific category')
    parser.add_argument('--output', type=str, default='data/comprehensive_results.json')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE RENTECH-STYLE BACKTESTING")
    print("=" * 80)

    start_time = time.time()

    # 1. Load data
    print("\n[1/6] Loading data...")
    loader = RentechDataLoader()
    summary = loader.get_summary()
    print(f"  Data range: {summary['total_start']} to {summary['total_end']}")
    print(f"  Total days: {summary['total_days']:,}")

    df = loader.load_merged_data()
    print(f"  DataFrame shape: {df.shape}")

    # 2. Basic features
    print("\n[2/6] Calculating basic features...")
    basic_engine = FeatureEngine()
    df = basic_engine.add_all_features(df)

    # 3. Advanced features
    print("\n[3/6] Calculating advanced features...")
    advanced_engine = AdvancedFeatureEngine()
    df = advanced_engine.add_all_features(df)

    print(f"  Total columns: {len(df.columns)}")

    # Show feature coverage
    price_rows = df['close'].notna().sum()
    print(f"  Rows with price data: {price_rows:,}")

    # 4. Generate strategies
    print("\n[4/6] Generating comprehensive strategies...")
    factory = ComprehensiveStrategyFactory()
    strategies = factory.generate_all()

    if args.category:
        strategies = [s for s in strategies if s.category == args.category]
        print(f"  Filtered to category: {args.category}")

    if args.quick:
        # Quick test - 50 strategies
        strategies = strategies[:50]
        print(f"  Quick mode: testing {len(strategies)} strategies")

    print(f"  Total strategies to test: {len(strategies)}")

    # Show by category
    counts = factory.get_strategy_count()
    for cat, count in sorted(counts.items()):
        if not args.category or cat == args.category:
            print(f"    {cat}: {count}")

    # 5. Run walk-forward backtest
    print("\n[5/6] Running walk-forward backtest...")
    wf_config = WFConfig(
        train_years=2,
        test_months=6,
        step_months=6,
    )
    wf_engine = WalkForwardEngine(wf_config)
    validator = StatisticalValidator()

    results = run_comprehensive_backtest(
        strategies=strategies,
        df=df,
        wf_engine=wf_engine,
        validator=validator,
        verbose=True,
    )

    # 6. Export and summarize
    print("\n[6/6] Exporting results...")
    export_results(results, args.output, summary)

    print_summary(results)

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
