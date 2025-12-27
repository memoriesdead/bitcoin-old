"""
Hypothesis Tester - RenTech Style Statistical Edge Discovery

Tests 50+ trading hypotheses against historical data.
Only patterns with statistical significance become trading strategies.
"""
import sqlite3
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class TradeResult:
    """Single simulated trade."""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    direction: int  # 1 = long, -1 = short
    pnl_pct: float
    flow_amount: float
    exchange: str
    hypothesis_id: str


@dataclass
class HypothesisResult:
    """Statistical results for a hypothesis."""
    hypothesis_id: str
    description: str
    total_trades: int
    winning_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    edge_ratio: float  # avg_win / avg_loss
    total_pnl_pct: float
    sharpe_ratio: float
    max_drawdown: float
    p_value: float
    is_significant: bool
    is_profitable: bool
    recommendation: str


class HypothesisTester:
    """
    Test trading hypotheses against historical data.

    RenTech Methodology:
    1. Define clear, testable hypothesis
    2. Run on large sample (1000+ trades)
    3. Calculate statistical significance
    4. Only implement if p < 0.01 and edge > 0
    """

    # Minimum requirements for a valid edge
    MIN_TRADES = 500
    MIN_WIN_RATE = 0.505  # Must beat 50%
    MAX_P_VALUE = 0.01    # 99% confidence
    MIN_EDGE_RATIO = 1.0  # avg_win >= avg_loss

    def __init__(self, db_path: str = "data/historical_flows.db"):
        self.db_path = Path(db_path)
        self.results: Dict[str, HypothesisResult] = {}
        self.trades: List[TradeResult] = []

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _get_price_at_time(self, conn, timestamp: int) -> Optional[float]:
        """Get price closest to timestamp."""
        c = conn.cursor()
        c.execute('''
            SELECT close FROM prices
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY ABS(timestamp - ?)
            LIMIT 1
        ''', (timestamp - 120, timestamp + 120, timestamp))
        row = c.fetchone()
        return row[0] if row else None

    def _get_price_after_delay(self, conn, timestamp: int, delay_seconds: int) -> Optional[float]:
        """Get price after specific delay."""
        target_time = timestamp + delay_seconds
        return self._get_price_at_time(conn, target_time)

    def _calculate_statistics(self, trades: List[TradeResult], hypothesis_id: str, description: str) -> HypothesisResult:
        """Calculate full statistics for trade results."""
        if not trades:
            return HypothesisResult(
                hypothesis_id=hypothesis_id,
                description=description,
                total_trades=0,
                winning_trades=0,
                win_rate=0,
                avg_win_pct=0,
                avg_loss_pct=0,
                edge_ratio=0,
                total_pnl_pct=0,
                sharpe_ratio=0,
                max_drawdown=0,
                p_value=1.0,
                is_significant=False,
                is_profitable=False,
                recommendation="INSUFFICIENT DATA"
            )

        # Basic stats
        total = len(trades)
        winners = [t for t in trades if t.pnl_pct > 0]
        losers = [t for t in trades if t.pnl_pct <= 0]

        win_rate = len(winners) / total if total > 0 else 0
        avg_win = sum(t.pnl_pct for t in winners) / len(winners) if winners else 0
        avg_loss = abs(sum(t.pnl_pct for t in losers) / len(losers)) if losers else 0
        edge_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        total_pnl = sum(t.pnl_pct for t in trades)

        # Sharpe ratio (simplified)
        returns = [t.pnl_pct for t in trades]
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance) if variance > 0 else 0.001
        sharpe = (mean_return / std_dev) * math.sqrt(252 * 24 * 60)  # Annualized

        # Max drawdown
        cumulative = 0
        peak = 0
        max_dd = 0
        for t in trades:
            cumulative += t.pnl_pct
            peak = max(peak, cumulative)
            dd = (peak - cumulative) / (peak + 100) if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # P-value (binomial test approximation)
        # H0: win_rate = 0.5 (random)
        # Using normal approximation to binomial
        expected_wins = total * 0.5
        std_error = math.sqrt(total * 0.5 * 0.5)
        z_score = (len(winners) - expected_wins) / std_error if std_error > 0 else 0

        # Two-tailed p-value approximation
        p_value = 2 * (1 - self._normal_cdf(abs(z_score)))

        is_significant = p_value < self.MAX_P_VALUE and total >= self.MIN_TRADES
        is_profitable = win_rate > self.MIN_WIN_RATE and edge_ratio >= self.MIN_EDGE_RATIO

        if is_significant and is_profitable:
            recommendation = "IMPLEMENT"
        elif total < self.MIN_TRADES:
            recommendation = "NEED MORE DATA"
        elif not is_significant:
            recommendation = "NOT SIGNIFICANT"
        else:
            recommendation = "NO EDGE"

        return HypothesisResult(
            hypothesis_id=hypothesis_id,
            description=description,
            total_trades=total,
            winning_trades=len(winners),
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            edge_ratio=edge_ratio,
            total_pnl_pct=total_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            p_value=p_value,
            is_significant=is_significant,
            is_profitable=is_profitable,
            recommendation=recommendation
        )

    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    # =========================================================================
    # HYPOTHESIS TESTS
    # =========================================================================

    def test_flow_direction_basic(
        self,
        min_flow_btc: float = 10.0,
        hold_minutes: int = 10,
        entry_delay_seconds: int = 30
    ) -> HypothesisResult:
        """
        H01-H06: Basic flow direction hypothesis.

        Inflow > X BTC -> SHORT
        Outflow > X BTC -> LONG
        """
        hypothesis_id = f"FLOW_DIR_{min_flow_btc}BTC_{hold_minutes}m_{entry_delay_seconds}s"
        description = f"Flow > {min_flow_btc} BTC predicts direction, hold {hold_minutes}m, delay {entry_delay_seconds}s"

        conn = self._get_connection()
        c = conn.cursor()

        # Get all significant flows
        c.execute('''
            SELECT block_time, direction, amount_btc, exchange
            FROM flows
            WHERE amount_btc >= ?
            ORDER BY block_time
        ''', (min_flow_btc,))

        trades = []
        for row in c.fetchall():
            flow_time, direction, amount, exchange = row

            # Entry after delay
            entry_time = flow_time + entry_delay_seconds
            exit_time = entry_time + (hold_minutes * 60)

            entry_price = self._get_price_at_time(conn, entry_time)
            exit_price = self._get_price_at_time(conn, exit_time)

            if not entry_price or not exit_price:
                continue

            # direction: 1 = outflow = LONG, -1 = inflow = SHORT
            trade_direction = direction

            if trade_direction == 1:  # LONG
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100

            # Subtract trading costs (0.1% round trip)
            pnl_pct -= 0.1

            trades.append(TradeResult(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                direction=trade_direction,
                pnl_pct=pnl_pct,
                flow_amount=amount,
                exchange=exchange,
                hypothesis_id=hypothesis_id
            ))

        conn.close()

        result = self._calculate_statistics(trades, hypothesis_id, description)
        self.results[hypothesis_id] = result
        return result

    def test_exchange_specific(
        self,
        exchange: str,
        min_flow_btc: float = 10.0,
        hold_minutes: int = 10
    ) -> HypothesisResult:
        """
        H20-H22: Exchange-specific flow predictiveness.
        """
        hypothesis_id = f"EXCHANGE_{exchange}_{min_flow_btc}BTC_{hold_minutes}m"
        description = f"{exchange} flows > {min_flow_btc} BTC, hold {hold_minutes}m"

        conn = self._get_connection()
        c = conn.cursor()

        c.execute('''
            SELECT block_time, direction, amount_btc
            FROM flows
            WHERE exchange = ? AND amount_btc >= ?
            ORDER BY block_time
        ''', (exchange, min_flow_btc))

        trades = []
        for row in c.fetchall():
            flow_time, direction, amount = row

            entry_time = flow_time + 30
            exit_time = entry_time + (hold_minutes * 60)

            entry_price = self._get_price_at_time(conn, entry_time)
            exit_price = self._get_price_at_time(conn, exit_time)

            if not entry_price or not exit_price:
                continue

            trade_direction = direction

            if trade_direction == 1:
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100

            pnl_pct -= 0.1

            trades.append(TradeResult(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                direction=trade_direction,
                pnl_pct=pnl_pct,
                flow_amount=amount,
                exchange=exchange,
                hypothesis_id=hypothesis_id
            ))

        conn.close()

        result = self._calculate_statistics(trades, hypothesis_id, description)
        self.results[hypothesis_id] = result
        return result

    def test_flow_clustering(
        self,
        window_minutes: int = 5,
        min_flows: int = 3,
        hold_minutes: int = 15
    ) -> HypothesisResult:
        """
        H42: Flow clustering - multiple flows in short window = stronger signal.
        """
        hypothesis_id = f"CLUSTER_{min_flows}flows_{window_minutes}m_hold{hold_minutes}m"
        description = f"{min_flows}+ flows in {window_minutes}m window, hold {hold_minutes}m"

        conn = self._get_connection()
        c = conn.cursor()

        # Get all flows
        c.execute('''
            SELECT block_time, direction, amount_btc, exchange
            FROM flows
            ORDER BY block_time
        ''')

        all_flows = c.fetchall()
        window_seconds = window_minutes * 60

        trades = []
        i = 0
        while i < len(all_flows):
            cluster_start = all_flows[i][0]
            cluster_end = cluster_start + window_seconds

            # Find all flows in window
            cluster = []
            j = i
            while j < len(all_flows) and all_flows[j][0] <= cluster_end:
                cluster.append(all_flows[j])
                j += 1

            if len(cluster) >= min_flows:
                # Calculate net direction
                net_direction = sum(f[1] * f[2] for f in cluster)  # direction * amount
                trade_direction = 1 if net_direction > 0 else -1

                total_amount = sum(f[2] for f in cluster)

                entry_time = cluster_end + 30
                exit_time = entry_time + (hold_minutes * 60)

                entry_price = self._get_price_at_time(conn, entry_time)
                exit_price = self._get_price_at_time(conn, exit_time)

                if entry_price and exit_price:
                    if trade_direction == 1:
                        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        pnl_pct = ((entry_price - exit_price) / entry_price) * 100

                    pnl_pct -= 0.1

                    trades.append(TradeResult(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=trade_direction,
                        pnl_pct=pnl_pct,
                        flow_amount=total_amount,
                        exchange="cluster",
                        hypothesis_id=hypothesis_id
                    ))

                i = j  # Skip to end of cluster
            else:
                i += 1

        conn.close()

        result = self._calculate_statistics(trades, hypothesis_id, description)
        self.results[hypothesis_id] = result
        return result

    def test_whale_flows(
        self,
        min_btc: float = 100.0,
        hold_minutes: int = 30
    ) -> HypothesisResult:
        """
        H51: Whale flows (> 100 BTC) predict bigger moves.
        """
        hypothesis_id = f"WHALE_{min_btc}BTC_hold{hold_minutes}m"
        description = f"Whale flows > {min_btc} BTC, hold {hold_minutes}m"

        return self.test_flow_direction_basic(
            min_flow_btc=min_btc,
            hold_minutes=hold_minutes,
            entry_delay_seconds=60
        )

    def test_time_of_day(
        self,
        hour_start: int,
        hour_end: int,
        min_flow_btc: float = 10.0,
        hold_minutes: int = 10
    ) -> HypothesisResult:
        """
        H30-H32: Time-of-day effects.
        """
        hypothesis_id = f"TIME_{hour_start}-{hour_end}h_{min_flow_btc}BTC"
        description = f"Flows during {hour_start}:00-{hour_end}:00 UTC, > {min_flow_btc} BTC"

        conn = self._get_connection()
        c = conn.cursor()

        c.execute('''
            SELECT block_time, direction, amount_btc, exchange
            FROM flows
            WHERE amount_btc >= ?
            ORDER BY block_time
        ''', (min_flow_btc,))

        trades = []
        for row in c.fetchall():
            flow_time, direction, amount, exchange = row

            # Check hour
            dt = datetime.fromtimestamp(flow_time)
            if not (hour_start <= dt.hour < hour_end):
                continue

            entry_time = flow_time + 30
            exit_time = entry_time + (hold_minutes * 60)

            entry_price = self._get_price_at_time(conn, entry_time)
            exit_price = self._get_price_at_time(conn, exit_time)

            if not entry_price or not exit_price:
                continue

            trade_direction = direction

            if trade_direction == 1:
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100

            pnl_pct -= 0.1

            trades.append(TradeResult(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                direction=trade_direction,
                pnl_pct=pnl_pct,
                flow_amount=amount,
                exchange=exchange,
                hypothesis_id=hypothesis_id
            ))

        conn.close()

        result = self._calculate_statistics(trades, hypothesis_id, description)
        self.results[hypothesis_id] = result
        return result

    def test_optimal_entry_delay(
        self,
        min_flow_btc: float = 10.0,
        hold_minutes: int = 10
    ) -> Dict[int, HypothesisResult]:
        """
        H10-H13: Find optimal entry delay after flow signal.
        """
        delays = [0, 10, 30, 60, 120, 300]
        results = {}

        for delay in delays:
            result = self.test_flow_direction_basic(
                min_flow_btc=min_flow_btc,
                hold_minutes=hold_minutes,
                entry_delay_seconds=delay
            )
            results[delay] = result

        # Find best
        best_delay = max(results.keys(), key=lambda d: results[d].win_rate)
        print(f"\n[OPTIMAL DELAY] Best entry delay: {best_delay}s (win rate: {results[best_delay].win_rate:.2%})")

        return results

    def run_all_tests(self) -> List[HypothesisResult]:
        """Run all hypothesis tests and return ranked results."""
        print("\n" + "=" * 80)
        print("RENTECH-STYLE HYPOTHESIS TESTING")
        print("=" * 80)

        all_results = []

        # Test different flow sizes
        print("\n[1/6] Testing flow size thresholds...")
        for min_btc in [5, 10, 25, 50, 100]:
            for hold_min in [5, 10, 15, 30]:
                result = self.test_flow_direction_basic(
                    min_flow_btc=min_btc,
                    hold_minutes=hold_min
                )
                all_results.append(result)
                if result.total_trades > 0:
                    print(f"  {result.hypothesis_id}: {result.win_rate:.2%} ({result.total_trades} trades) - {result.recommendation}")

        # Test exchanges
        print("\n[2/6] Testing exchange-specific flows...")
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT DISTINCT exchange FROM flows')
        exchanges = [row[0] for row in c.fetchall()]
        conn.close()

        for exchange in exchanges[:10]:  # Top 10 exchanges
            result = self.test_exchange_specific(exchange, min_flow_btc=10.0)
            all_results.append(result)
            if result.total_trades > 100:
                print(f"  {exchange}: {result.win_rate:.2%} ({result.total_trades} trades) - {result.recommendation}")

        # Test clustering
        print("\n[3/6] Testing flow clustering...")
        for min_flows in [3, 5, 10]:
            for window in [2, 5, 10]:
                result = self.test_flow_clustering(
                    window_minutes=window,
                    min_flows=min_flows
                )
                all_results.append(result)
                if result.total_trades > 0:
                    print(f"  {result.hypothesis_id}: {result.win_rate:.2%} ({result.total_trades} trades)")

        # Test whale flows
        print("\n[4/6] Testing whale flows...")
        for min_btc in [50, 100, 200, 500]:
            result = self.test_flow_direction_basic(
                min_flow_btc=min_btc,
                hold_minutes=30,
                entry_delay_seconds=60
            )
            all_results.append(result)
            if result.total_trades > 0:
                print(f"  Whale >{min_btc} BTC: {result.win_rate:.2%} ({result.total_trades} trades)")

        # Test time of day
        print("\n[5/6] Testing time-of-day effects...")
        time_periods = [(0, 8), (8, 16), (16, 24), (13, 21)]  # Asian, EU, US, NY open
        for start, end in time_periods:
            result = self.test_time_of_day(start, end)
            all_results.append(result)
            if result.total_trades > 0:
                print(f"  {start}:00-{end}:00 UTC: {result.win_rate:.2%} ({result.total_trades} trades)")

        # Test entry delays
        print("\n[6/6] Finding optimal entry delay...")
        delay_results = self.test_optimal_entry_delay()

        # Rank all results
        print("\n" + "=" * 80)
        print("RANKED RESULTS (by win rate)")
        print("=" * 80)

        ranked = sorted(all_results, key=lambda r: r.win_rate if r.total_trades >= self.MIN_TRADES else 0, reverse=True)

        # Show top 20
        print(f"\n{'Rank':<5} {'Hypothesis':<50} {'Win%':<8} {'Trades':<8} {'Edge':<8} {'p-val':<8} {'Rec':<15}")
        print("-" * 110)

        for i, result in enumerate(ranked[:20], 1):
            print(f"{i:<5} {result.hypothesis_id:<50} {result.win_rate:.2%}   {result.total_trades:<8} {result.edge_ratio:.2f}     {result.p_value:.4f}   {result.recommendation}")

        # Show implementable strategies
        implementable = [r for r in ranked if r.recommendation == "IMPLEMENT"]
        print(f"\n{'='*80}")
        print(f"IMPLEMENTABLE STRATEGIES: {len(implementable)}")
        print(f"{'='*80}")

        for result in implementable:
            print(f"\n{result.hypothesis_id}")
            print(f"  Description: {result.description}")
            print(f"  Win Rate: {result.win_rate:.2%}")
            print(f"  Trades: {result.total_trades}")
            print(f"  Edge Ratio: {result.edge_ratio:.2f}")
            print(f"  Sharpe: {result.sharpe_ratio:.2f}")
            print(f"  Max DD: {result.max_drawdown:.2%}")
            print(f"  p-value: {result.p_value:.6f}")

        return ranked

    def save_results(self, filepath: str = "data/hypothesis_results.json"):
        """Save all results to JSON."""
        output = []
        for result in self.results.values():
            output.append({
                "hypothesis_id": result.hypothesis_id,
                "description": result.description,
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "win_rate": result.win_rate,
                "avg_win_pct": result.avg_win_pct,
                "avg_loss_pct": result.avg_loss_pct,
                "edge_ratio": result.edge_ratio,
                "total_pnl_pct": result.total_pnl_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "p_value": result.p_value,
                "is_significant": result.is_significant,
                "is_profitable": result.is_profitable,
                "recommendation": result.recommendation
            })

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n[+] Results saved to {filepath}")


def main():
    """Run hypothesis testing."""
    tester = HypothesisTester()

    # Check data availability
    conn = sqlite3.connect(tester.db_path)
    c = conn.cursor()

    c.execute('SELECT COUNT(*) FROM flows')
    flow_count = c.fetchone()[0]

    c.execute('SELECT COUNT(*) FROM prices')
    price_count = c.fetchone()[0]

    conn.close()

    print(f"\nData available:")
    print(f"  Flows: {flow_count:,}")
    print(f"  Prices: {price_count:,}")

    if flow_count < 1000:
        print("\n[!] Need more flow data. Run historical_scanner.py first.")
        return

    if price_count < 100000:
        print("\n[!] Need more price data. Run price_downloader.py first.")
        return

    # Run all tests
    results = tester.run_all_tests()

    # Save results
    tester.save_results()


if __name__ == '__main__':
    main()
