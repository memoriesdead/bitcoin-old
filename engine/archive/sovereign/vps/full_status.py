#!/usr/bin/env python3
"""
FULL STATUS CHECK - ALL STRATEGIES
===================================
Morning check for all running strategies.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("/root/validation/data")


def header(text: str):
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_data_collection():
    """Check data collection status."""
    header("DATA COLLECTION")

    conn = sqlite3.connect(DATA_DIR / "metrics.db")
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total_rows,
            MIN(timestamp) as first_ts,
            MAX(timestamp) as last_ts,
            SUM(tx_whale) as whale_txs,
            SUM(tx_mega) as mega_txs,
            ROUND(AVG(price), 2) as avg_price,
            MIN(price) as min_price,
            MAX(price) as max_price
        FROM metrics WHERE price > 0
    """)
    row = cursor.fetchone()
    conn.close()

    if row and row[0] > 0:
        total, first_ts, last_ts, whales, megas, avg_price, min_p, max_p = row
        duration_hours = (last_ts - first_ts) / 3600

        print(f"Total rows: {total:,}")
        print(f"Duration: {duration_hours:.1f} hours")
        print(f"Whale transactions: {whales:,}")
        print(f"Mega transactions: {megas:,}")
        print(f"Price range: ${min_p:,.0f} - ${max_p:,.0f}")
        print(f"Avg price: ${avg_price:,.2f}")
        print(f"Data rate: {total/duration_hours:.0f} rows/hour")
    else:
        print("No data collected yet!")


def check_rentech_trader():
    """Check RenTech trader results."""
    header("RENTECH MATHEMATICAL TRADER")

    db_path = DATA_DIR / "rentech_trades.db"
    if not db_path.exists():
        print("Not started yet (no database).")
        return

    try:
        conn = sqlite3.connect(db_path)

        # Trade summary
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN pnl_bps > 0 THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(pnl_bps), 1) as total_pnl_bps,
                ROUND(SUM(pnl_usd), 2) as total_pnl_usd,
                ROUND(AVG(pnl_bps), 1) as avg_pnl_bps,
                ROUND(AVG(leverage), 1) as avg_leverage
            FROM trades WHERE exit_time IS NOT NULL
        """)
        row = cursor.fetchone()

        if row and row[0] > 0:
            total, wins, pnl_bps, pnl_usd, avg_bps, avg_lev = row
            win_rate = wins / total * 100 if total > 0 else 0

            print(f"Strategy: Kelly Criterion + Multi-Signal")
            print(f"Total trades: {total}")
            print(f"Wins: {wins} | Losses: {total - wins}")
            print(f"Win rate: {win_rate:.1f}%")
            print(f"Total P&L: {pnl_bps:+.1f} bps (${pnl_usd:+.2f})")
            print(f"Avg P&L per trade: {avg_bps:+.1f} bps")
            print(f"Avg leverage used: {avg_lev:.1f}x")

            # Recent trades
            print()
            print("Recent trades:")
            cursor = conn.execute("""
                SELECT direction, entry_price, exit_price, leverage, exit_reason, pnl_bps
                FROM trades WHERE exit_time IS NOT NULL
                ORDER BY exit_time DESC LIMIT 5
            """)
            for trade in cursor.fetchall():
                dir_val, entry, exit_p, lev, reason, pnl = trade
                dir_str = "LONG" if dir_val == 1 else "SHORT"
                print(f"  {dir_str} ${entry:,.0f}->${exit_p:,.0f} ({lev:.0f}x) [{reason}] = {pnl:+.1f} bps")

            # Current capital
            cursor = conn.execute("""
                SELECT equity_usd, drawdown_pct FROM equity_curve
                ORDER BY timestamp DESC LIMIT 1
            """)
            eq_row = cursor.fetchone()
            if eq_row:
                print()
                print(f"Current capital: ${eq_row[0]:.2f}")
                print(f"Drawdown: {eq_row[1]:.1f}%")
        else:
            print("No completed trades yet.")

        conn.close()

    except Exception as e:
        print(f"Error: {e}")


def check_volatility_trader():
    """Check volatility trader results."""
    header("VOLATILITY TRADER (STRADDLE)")

    db_path = DATA_DIR / "vol_trades.db"
    if not db_path.exists():
        print("Not started yet (no database).")
        return

    try:
        conn = sqlite3.connect(db_path)

        cursor = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN net_pnl_bps > 0 THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(net_pnl_bps), 1) as total_pnl_bps,
                ROUND(SUM(net_pnl_usd), 2) as total_pnl_usd,
                ROUND(AVG(net_pnl_bps), 1) as avg_pnl_bps,
                ROUND(AVG(long_pnl_bps), 1) as avg_long,
                ROUND(AVG(short_pnl_bps), 1) as avg_short
            FROM straddles WHERE exit_time IS NOT NULL
        """)
        row = cursor.fetchone()

        if row and row[0] > 0:
            total, wins, pnl_bps, pnl_usd, avg_bps, avg_long, avg_short = row
            win_rate = wins / total * 100 if total > 0 else 0

            print(f"Strategy: Trade volatility, not direction")
            print(f"Total straddles: {total}")
            print(f"Wins: {wins} | Losses: {total - wins}")
            print(f"Win rate: {win_rate:.1f}%")
            print(f"Total P&L: {pnl_bps:+.1f} bps (${pnl_usd:+.2f})")
            print(f"Avg P&L per straddle: {avg_bps:+.1f} bps")
            print(f"Avg Long leg: {avg_long:+.1f} bps | Avg Short leg: {avg_short:+.1f} bps")

            # Recent straddles
            print()
            print("Recent straddles:")
            cursor = conn.execute("""
                SELECT entry_price, long_pnl_bps, short_pnl_bps, net_pnl_bps
                FROM straddles WHERE exit_time IS NOT NULL
                ORDER BY exit_time DESC LIMIT 5
            """)
            for trade in cursor.fetchall():
                entry, long_pnl, short_pnl, net = trade
                print(f"  @ ${entry:,.0f} | Long: {long_pnl:+.0f} | Short: {short_pnl:+.0f} | Net: {net:+.1f} bps")
        else:
            print("No completed straddles yet.")

        conn.close()

    except Exception as e:
        print(f"Error: {e}")


def check_paper_trader_v2():
    """Check original paper trader v2 (whale signal)."""
    header("PAPER TRADER V2 (WHALE SIGNAL)")

    db_path = DATA_DIR / "paper_trades.db"
    if not db_path.exists():
        print("Not running.")
        return

    try:
        conn = sqlite3.connect(db_path)

        cursor = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN pnl_bps > 0 THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(pnl_bps), 1) as total_pnl_bps,
                ROUND(SUM(pnl_usd), 2) as total_pnl_usd,
                ROUND(AVG(pnl_bps), 1) as avg_pnl_bps
            FROM trades WHERE exit_time IS NOT NULL
        """)
        row = cursor.fetchone()

        if row and row[0] > 0:
            total, wins, pnl_bps, pnl_usd, avg_bps = row
            win_rate = wins / total * 100 if total > 0 else 0

            print(f"Strategy: SHORT when whale >= 3")
            print(f"Total trades: {total}")
            print(f"Win rate: {win_rate:.1f}%")
            print(f"Total P&L: {pnl_bps:+.1f} bps (${pnl_usd:+.2f})")
            print(f"Avg P&L: {avg_bps:+.1f} bps")
        else:
            print("No completed trades.")

        conn.close()

    except Exception as e:
        print(f"Error: {e}")


def calculate_edge():
    """Statistical edge analysis across all strategies."""
    header("STATISTICAL EDGE ANALYSIS")

    strategies = [
        ("RenTech", "rentech_trades.db", "trades", "pnl_bps"),
        ("Volatility", "vol_trades.db", "straddles", "net_pnl_bps"),
        ("Paper V2", "paper_trades.db", "trades", "pnl_bps"),
    ]

    try:
        import numpy as np
        from scipy import stats

        for name, db_file, table, pnl_col in strategies:
            db_path = DATA_DIR / db_file
            if not db_path.exists():
                continue

            conn = sqlite3.connect(db_path)
            cursor = conn.execute(f"SELECT {pnl_col} FROM {table} WHERE exit_time IS NOT NULL")
            pnls = [row[0] for row in cursor.fetchall()]
            conn.close()

            if len(pnls) >= 10:
                mean_pnl = np.mean(pnls)
                std_pnl = np.std(pnls)
                t_stat, p_value = stats.ttest_1samp(pnls, 0)
                sharpe = mean_pnl / std_pnl * np.sqrt(252 * 24 * 6) if std_pnl > 0 else 0

                print(f"\n{name} ({len(pnls)} trades):")
                print(f"  Mean: {mean_pnl:+.2f} bps | Std: {std_pnl:.1f} bps")
                print(f"  t-stat: {t_stat:.3f} | p-value: {p_value:.4f}")
                print(f"  Sharpe (annualized): {sharpe:.2f}")

                if p_value < 0.05 and mean_pnl > 0:
                    print(f"  >>> STATISTICALLY SIGNIFICANT EDGE <<<")
                elif p_value < 0.10 and mean_pnl > 0:
                    print(f"  >> Marginally significant (p < 0.10)")

    except ImportError:
        print("scipy not installed - install with: pip install scipy")
    except Exception as e:
        print(f"Error: {e}")


def main():
    print()
    print("=" * 70)
    print(f"  OVERNIGHT STATUS REPORT - ALL STRATEGIES")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    check_data_collection()
    check_rentech_trader()
    check_volatility_trader()
    check_paper_trader_v2()
    calculate_edge()

    print()
    header("SERVICE COMMANDS")
    print("Stop old:    systemctl stop paper-trader")
    print("Start new:   systemctl start rentech-trader vol-trader")
    print("Status:      systemctl status rentech-trader vol-trader")
    print()
    print("Logs:")
    print("  RenTech:   tail -100 /root/validation/data/rentech_trader.log")
    print("  Volatility: tail -100 /root/validation/data/vol_trader.log")
    print("=" * 70)


if __name__ == "__main__":
    main()
