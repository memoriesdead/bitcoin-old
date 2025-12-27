#!/usr/bin/env python3
"""
MORNING STATUS CHECK
====================
Run this to see overnight results.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("/root/validation/data")
METRICS_DB = DATA_DIR / "metrics.db"
TRADES_DB = DATA_DIR / "paper_trades.db"


def check_metrics():
    """Check data collection status."""
    print("="*60)
    print("  DATA COLLECTION STATUS")
    print("="*60)

    conn = sqlite3.connect(METRICS_DB)
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total_rows,
            MIN(timestamp) as first_ts,
            MAX(timestamp) as last_ts,
            SUM(tx_whale) as whale_txs,
            SUM(tx_mega) as mega_txs,
            ROUND(AVG(price), 2) as avg_price
        FROM metrics WHERE price > 0
    """)
    row = cursor.fetchone()
    conn.close()

    if row and row[0] > 0:
        total, first_ts, last_ts, whales, megas, avg_price = row
        duration_hours = (last_ts - first_ts) / 3600

        print(f"Total rows: {total:,}")
        print(f"Duration: {duration_hours:.1f} hours")
        print(f"Whale transactions: {whales:,}")
        print(f"Mega transactions: {megas:,}")
        print(f"Avg price: ${avg_price:,.2f}")
        print(f"Data rate: {total/duration_hours:.0f} rows/hour")
    else:
        print("No data collected yet!")


def check_trades():
    """Check paper trading results."""
    print()
    print("="*60)
    print("  PAPER TRADING RESULTS")
    print("="*60)

    try:
        conn = sqlite3.connect(TRADES_DB)

        # Get trade summary
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN pnl_bps > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl_bps <= 0 THEN 1 ELSE 0 END) as losses,
                ROUND(SUM(pnl_bps), 1) as total_pnl_bps,
                ROUND(SUM(pnl_usd), 2) as total_pnl_usd,
                ROUND(AVG(pnl_bps), 1) as avg_pnl_bps
            FROM trades WHERE exit_time IS NOT NULL
        """)
        row = cursor.fetchone()

        if row and row[0] > 0:
            total, wins, losses, pnl_bps, pnl_usd, avg_bps = row
            win_rate = wins / total * 100 if total > 0 else 0

            print(f"Total trades: {total}")
            print(f"Wins: {wins} | Losses: {losses}")
            print(f"Win rate: {win_rate:.1f}%")
            print(f"Total P&L: {pnl_bps:+.1f} bps (${pnl_usd:+.2f})")
            print(f"Avg P&L per trade: {avg_bps:+.1f} bps")

            # Show recent trades
            print()
            print("Recent trades:")
            cursor = conn.execute("""
                SELECT id, direction, entry_price, exit_price, exit_reason, pnl_bps
                FROM trades
                WHERE exit_time IS NOT NULL
                ORDER BY exit_time DESC
                LIMIT 10
            """)
            for trade in cursor.fetchall():
                tid, direction, entry, exit_p, reason, pnl = trade
                dir_str = "LONG" if direction == 1 else "SHORT"
                print(f"  #{tid} {dir_str}: ${entry:,.0f} -> ${exit_p:,.0f} ({reason}) = {pnl:+.1f} bps")
        else:
            print("No completed trades yet.")

        # Get signal summary
        cursor = conn.execute("""
            SELECT COUNT(*), SUM(acted_on)
            FROM signals
        """)
        sig_row = cursor.fetchone()
        if sig_row and sig_row[0] > 0:
            print()
            print(f"Total signals: {sig_row[0]} | Acted on: {sig_row[1]}")

        conn.close()

    except Exception as e:
        print(f"Error reading trades: {e}")


def check_edge():
    """Check if we've found a statistical edge."""
    print()
    print("="*60)
    print("  EDGE ANALYSIS")
    print("="*60)

    try:
        conn = sqlite3.connect(TRADES_DB)
        cursor = conn.execute("""
            SELECT pnl_bps FROM trades WHERE exit_time IS NOT NULL
        """)
        pnls = [row[0] for row in cursor.fetchall()]
        conn.close()

        if len(pnls) >= 20:
            import numpy as np
            from scipy import stats

            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            t_stat, p_value = stats.ttest_1samp(pnls, 0)

            print(f"Trades analyzed: {len(pnls)}")
            print(f"Mean P&L: {mean_pnl:+.1f} bps")
            print(f"Std Dev: {std_pnl:.1f} bps")
            print(f"t-statistic: {t_stat:.3f}")
            print(f"p-value: {p_value:.4f}")

            if p_value < 0.05 and mean_pnl > 0:
                print()
                print(">>> STATISTICALLY SIGNIFICANT EDGE FOUND! <<<")
                print(">>> Ready for live trading consideration <<<")
            elif mean_pnl > 0:
                print()
                print("Positive returns but not yet statistically significant.")
                print("Need more trades.")
            else:
                print()
                print("No edge detected. Strategy needs adjustment.")
        else:
            print(f"Only {len(pnls)} trades. Need at least 20 for analysis.")

    except ImportError:
        print("scipy not installed - cannot do statistical test")
    except Exception as e:
        print(f"Error analyzing edge: {e}")


def check_historical():
    """Check historical analysis results."""
    print()
    print("="*60)
    print("  HISTORICAL ANALYSIS RESULTS")
    print("="*60)

    results_file = DATA_DIR / "historical_results.json"
    if not results_file.exists():
        print("No historical analysis results yet.")
        return

    try:
        import json
        with open(results_file) as f:
            results = json.load(f)

        print(f"Analysis timestamp: {results.get('timestamp', 'N/A')}")
        print(f"Strategies tested: {results.get('total_strategies_tested', 0)}")
        print(f"Significant edges: {results.get('significant_strategies', 0)}")

        best = results.get('best_strategies', [])
        if best:
            print()
            print("TOP SIGNIFICANT STRATEGIES:")
            for i, s in enumerate(best[:5]):
                print(f"  {i+1}. {s['signal']} > {s['threshold']} {s['direction']} {s['hold_mins']}m")
                print(f"     Return: {s['avg_return_bps']:+.1f}bps | Win: {s['win_rate']}% | n={s['n_trades']} | p={s['p_value']}")
        else:
            print()
            print("No significant edges found yet.")
            print("Need more data accumulation.")

    except Exception as e:
        print(f"Error reading results: {e}")


def main():
    print()
    print("="*60)
    print("  OVERNIGHT STATUS REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    check_metrics()
    check_trades()
    check_edge()
    check_historical()

    print()
    print("="*60)
    print("  Commands:")
    print("  - View paper trader logs: tail -100 /root/validation/data/paper_trader.log")
    print("  - View analysis logs: tail -100 /root/validation/data/historical_analysis.log")
    print("  - Service status: systemctl status paper-trader historical-analysis")
    print("  - Stop all: systemctl stop paper-trader historical-analysis")
    print("="*60)


if __name__ == "__main__":
    main()
