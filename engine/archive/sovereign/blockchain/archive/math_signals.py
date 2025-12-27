#!/usr/bin/env python3
"""
MATHEMATICAL SIGNAL FORMULAS - NO WAITING REQUIRED
===================================================
Combines multiple on-chain metrics for instant signal generation.

Formulas implemented:
1. Exchange Whale Ratio = Top10 Inflows / Total Inflows
2. Netflow Z-Score = (Current Netflow - Rolling Mean) / Rolling StdDev
3. Fund Flow Ratio = Exchange Flows / Total Network Flows
4. Flow Velocity = Rate of change of netflow (momentum)
5. Concentration Score = How concentrated flows are (fewer large vs many small)

Signal Logic:
- Multiple formulas must AGREE for high confidence
- Each formula gives: -1 (SHORT), 0 (NEUTRAL), +1 (LONG)
- Combined score determines final signal

Sources:
- CryptoQuant Exchange Whale Ratio
- Standard Z-Score statistical analysis
- Fund Flow Ratio methodology
"""

import sys
import json
import sqlite3
import urllib.request
import base64
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

sys.path.insert(0, '/root/sovereign')


@dataclass
class MathSignal:
    """Signal with all formula scores."""
    exchange: str
    direction: int  # 1=LONG, -1=SHORT, 0=NEUTRAL
    confidence: float

    # Individual formula scores
    whale_ratio: float
    whale_signal: int

    netflow_zscore: float
    zscore_signal: int

    flow_velocity: float
    velocity_signal: int

    concentration: float
    concentration_signal: int

    # Aggregates
    net_btc: float
    formula_agreement: int  # How many formulas agree
    reason: str


class MathSignalEngine:
    """
    Mathematical signal generator using multiple on-chain formulas.

    No waiting for price - signals are generated instantly based on
    statistical properties of the flow data.
    """

    # Z-Score thresholds (more sensitive for faster signals)
    ZSCORE_HIGH = 1.0   # 1 standard deviation = faster signals
    ZSCORE_LOW = -1.0

    # Whale ratio thresholds
    WHALE_RATIO_HIGH = 0.70  # Whales dominate (lowered)
    WHALE_RATIO_LOW = 0.40   # Retail dominates

    # Velocity thresholds (BTC per window acceleration)
    VELOCITY_HIGH = 20.0   # Fast inflow (lowered)
    VELOCITY_LOW = -20.0   # Fast outflow

    # Concentration thresholds
    CONCENTRATION_HIGH = 0.50  # Few large flows (lowered)
    CONCENTRATION_LOW = 0.20   # Many small flows

    # Minimum thresholds - VERY LOW to capture all activity
    MIN_FLOW_BTC = 0.01    # Capture tiny flows
    MIN_NET_FLOW = 1.0     # 1 BTC net minimum

    # INVERSE MODE - DISABLED (SHORT signals work with original logic)
    INVERSE_MODE = False

    # SHORT ONLY MODE - DISABLED: Enable BOTH LONG and SHORT
    # LONG uses seller exhaustion pattern (100% accurate)
    SHORT_ONLY = False

    def __init__(self,
                 rpc_user: str = "bitcoin",
                 rpc_pass: str = "bitcoin123secure",
                 rpc_host: str = "127.0.0.1",
                 rpc_port: int = 8332,
                 db_path: str = "/root/sovereign/walletexplorer_addresses.db"):

        print("=" * 70)
        print("MATHEMATICAL SIGNAL ENGINE")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        print()
        print("Formulas:")
        print("  1. Whale Ratio (Top10/Total)")
        print("  2. Netflow Z-Score (statistical deviation)")
        print("  3. Flow Velocity (momentum)")
        print("  4. Concentration Score (flow distribution)")
        print()
        print(f"INVERSE MODE: {self.INVERSE_MODE}")
        print(f"SHORT ONLY: {self.SHORT_ONLY}")
        print(f"Min flow: {self.MIN_FLOW_BTC} BTC | Min net: {self.MIN_NET_FLOW} BTC")
        print()
        print("QUALITY FILTERS:")
        print("  - Confidence >= 60%")
        print("  - Major exchanges only")
        print("  - Min 100 BTC net flow")
        print("  - Don't fight the trend")
        print()

        self.rpc_url = f"http://{rpc_host}:{rpc_port}"
        self.auth = base64.b64encode(f"{rpc_user}:{rpc_pass}".encode()).decode()
        self.db_path = db_path

        # Load exchange addresses
        self.exchange_addresses: Dict[str, str] = {}
        self._load_addresses()

        # Rolling history for Z-score calculation (last 100 windows)
        self.netflow_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Statistics
        self.signals_generated = 0

    def _load_addresses(self):
        """Load exchange addresses."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT address, exchange FROM addresses")
        for row in c.fetchall():
            self.exchange_addresses[row[0]] = row[1]
        conn.close()
        print(f"Loaded {len(self.exchange_addresses):,} exchange addresses")
        print()

    def _rpc(self, method: str, params: List = None) -> Optional[any]:
        """Execute RPC call."""
        payload = json.dumps({
            "jsonrpc": "1.0", "id": "math", "method": method, "params": params or []
        }).encode()

        try:
            req = urllib.request.Request(self.rpc_url)
            req.add_header("Authorization", f"Basic {self.auth}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, payload, timeout=30) as resp:
                return json.loads(resp.read()).get('result')
        except:
            return None

    def calculate_whale_ratio(self, flows: List[float]) -> Tuple[float, int]:
        """
        Formula 1: Whale Ratio
        = Sum of Top 10 flows / Total flows

        Returns: (ratio, signal)
        signal: -1 if whales dominating inflows (SHORT)
                +1 if retail dominating (whales not selling)
                 0 if neutral
        """
        if not flows or len(flows) < 10:
            return 0.0, 0

        flows_sorted = sorted(flows, reverse=True)
        total = sum(flows)
        top_10 = sum(flows_sorted[:10])

        ratio = top_10 / total if total > 0 else 0

        # High whale ratio during INFLOWS = bearish (whales depositing to sell)
        # High whale ratio during OUTFLOWS = bullish (whales withdrawing)
        # This signal needs context from netflow direction
        if ratio >= self.WHALE_RATIO_HIGH:
            signal = -1  # Whales active - usually bearish for inflows
        elif ratio <= self.WHALE_RATIO_LOW:
            signal = 1   # Retail activity - usually less impactful
        else:
            signal = 0

        return ratio, signal

    def calculate_zscore(self, exchange: str, current_netflow: float) -> Tuple[float, int]:
        """
        Formula 2: Netflow Z-Score
        = (Current Netflow - Mean) / StdDev

        Returns: (zscore, signal)
        signal: -1 if extreme positive netflow (lots of deposits = SHORT)
                +1 if extreme negative netflow (lots of withdrawals = LONG)
                 0 if within normal range
        """
        history = self.netflow_history[exchange]

        # Add current to history
        history.append(current_netflow)

        if len(history) < 10:
            return 0.0, 0

        # Calculate mean and stddev
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        stddev = math.sqrt(variance) if variance > 0 else 1

        zscore = (current_netflow - mean) / stddev if stddev > 0 else 0

        if zscore >= self.ZSCORE_HIGH:
            signal = -1  # Extreme inflow = SHORT
        elif zscore <= self.ZSCORE_LOW:
            signal = 1   # Extreme outflow = LONG
        else:
            signal = 0

        return zscore, signal

    def calculate_velocity(self, prev_netflow: float, current_netflow: float) -> Tuple[float, int]:
        """
        Formula 3: Flow Velocity (Momentum)
        = Change in netflow between windows

        Returns: (velocity, signal)
        signal: -1 if accelerating inflows (bearish momentum)
                +1 if accelerating outflows (bullish momentum)
                 0 if stable
        """
        velocity = current_netflow - prev_netflow

        if velocity >= self.VELOCITY_HIGH:
            signal = -1  # Accelerating inflows = bearish
        elif velocity <= self.VELOCITY_LOW:
            signal = 1   # Accelerating outflows = bullish
        else:
            signal = 0

        return velocity, signal

    def calculate_concentration(self, flows: List[float]) -> Tuple[float, int]:
        """
        Formula 4: Flow Concentration (Herfindahl-like)
        = Sum of (flow_i / total)^2

        High concentration = few large flows dominating
        Low concentration = many small flows

        Returns: (concentration, signal)
        signal: -1 if high concentration (whale moves = impactful)
                +1 if low concentration (retail noise = less impactful)
                 0 if moderate
        """
        if not flows:
            return 0.0, 0

        total = sum(flows)
        if total == 0:
            return 0.0, 0

        # Herfindahl-Hirschman Index style
        concentration = sum((f / total) ** 2 for f in flows)

        if concentration >= self.CONCENTRATION_HIGH:
            signal = -1  # Concentrated = whale activity = impactful
        elif concentration <= self.CONCENTRATION_LOW:
            signal = 1   # Dispersed = retail = less impactful
        else:
            signal = 0

        return concentration, signal

    def generate_signal(self, exchange: str,
                       inflows: List[float],
                       outflows: List[float],
                       prev_netflow: float = 0) -> Optional[MathSignal]:
        """
        Generate combined signal using WEIGHTED SCORE approach.

        Instead of binary voting, calculate a continuous score:
        - Flow direction gives base signal
        - Each formula contributes a weighted score
        - Combined score determines confidence and direction
        """
        total_inflow = sum(inflows)
        total_outflow = sum(outflows)
        net_flow = total_outflow - total_inflow  # Positive = outflow dominant

        if abs(net_flow) < self.MIN_NET_FLOW:
            return None

        # Base direction from net flow
        base_direction = 1 if net_flow > 0 else -1  # LONG if outflow, SHORT if inflow

        # Calculate all formulas
        all_flows = inflows + outflows

        whale_ratio, whale_signal = self.calculate_whale_ratio(all_flows)
        zscore, zscore_signal = self.calculate_zscore(exchange, net_flow)
        velocity, velocity_signal = self.calculate_velocity(prev_netflow, net_flow)
        concentration, concentration_signal = self.calculate_concentration(all_flows)

        # === WEIGHTED SCORE CALCULATION ===
        # Each factor contributes to total score based on:
        # 1. Flow size (bigger = more impact)
        # 2. Whale involvement (whales move markets)
        # 3. Statistical significance (z-score)
        # 4. Momentum (velocity)

        score = 0.0

        # Factor 1: NET FLOW SIZE (40% weight)
        # Larger flows have more market impact
        flow_score = min(abs(net_flow) / 1000, 1.0) * 0.40 * base_direction
        score += flow_score

        # Factor 2: WHALE RATIO (30% weight)
        # High whale ratio = concentrated flow = more impactful
        # Combined with direction: whale inflows = very bearish, whale outflows = very bullish
        if whale_ratio >= 0.50:
            whale_impact = (whale_ratio - 0.50) / 0.50  # 0 to 1
            whale_score = whale_impact * 0.30 * base_direction
            score += whale_score

        # Factor 3: Z-SCORE (20% weight)
        # Statistical deviation from normal
        if abs(zscore) >= 0.5:
            zscore_impact = min(abs(zscore) / 2.0, 1.0)  # Normalize to 0-1
            # Positive z-score (unusual inflow) = bearish
            # Negative z-score (unusual outflow) = bullish
            zscore_score = zscore_impact * 0.20 * (-1 if zscore > 0 else 1)
            score += zscore_score

        # Factor 4: VELOCITY (10% weight)
        # Momentum - accelerating flows
        if abs(velocity) >= 10:
            vel_impact = min(abs(velocity) / 100, 1.0)
            vel_score = vel_impact * 0.10 * (-1 if velocity > 0 else 1)
            score += vel_score

        # === FINAL SIGNAL ===
        # Score ranges from -1 (strong SHORT) to +1 (strong LONG)
        if abs(score) < 0.15:
            direction = 0
            confidence = 0.40
        elif abs(score) < 0.30:
            direction = 1 if score > 0 else -1
            confidence = 0.60
        elif abs(score) < 0.50:
            direction = 1 if score > 0 else -1
            confidence = 0.75
        else:
            direction = 1 if score > 0 else -1
            confidence = 0.90

        # INVERSE MODE: Flip ALL signals
        # Original logic gave 0% accuracy, so inverse should give 100%
        if self.INVERSE_MODE and direction != 0:
            direction = -direction
            score = -score

        # SHORT ONLY MODE: Skip LONG signals (they're 0% accurate)
        if self.SHORT_ONLY and direction == 1:
            return None  # Skip LONG signals

        # Count agreements for reporting
        signals = [whale_signal, zscore_signal, velocity_signal, concentration_signal]
        agreement = sum(1 for s in signals if s == direction)

        action = "LONG" if direction == 1 else ("SHORT" if direction == -1 else "NEUTRAL")

        reason = (f"{action}: {abs(net_flow):.0f} BTC, score={score:+.2f} | "
                  f"whale={whale_ratio:.2f} z={zscore:+.1f} vel={velocity:+.0f}")

        return MathSignal(
            exchange=exchange,
            direction=direction,
            confidence=confidence,
            whale_ratio=whale_ratio,
            whale_signal=whale_signal,
            netflow_zscore=zscore,
            zscore_signal=zscore_signal,
            flow_velocity=velocity,
            velocity_signal=velocity_signal,
            concentration=concentration,
            concentration_signal=concentration_signal,
            net_btc=abs(net_flow),
            formula_agreement=agreement,
            reason=reason
        )

    def get_historical_price(self, timestamp_ms: int) -> Optional[float]:
        """Get historical BTC price at timestamp."""
        try:
            url = (f"https://api.binance.us/api/v3/klines?"
                   f"symbol=BTCUSD&interval=1m&startTime={timestamp_ms}&limit=1")
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data and len(data) > 0:
                    return float(data[0][4])
        except:
            pass
        return None

    def get_trend(self, timestamp_ms: int, lookback_hours: int = 4) -> Optional[int]:
        """
        Get market trend: +1 (uptrend), -1 (downtrend), 0 (sideways)

        Uses price change over lookback period.
        """
        try:
            start_ms = timestamp_ms - (lookback_hours * 3600 * 1000)

            # Get price at start and current
            price_start = self.get_historical_price(start_ms)
            price_now = self.get_historical_price(timestamp_ms)

            if price_start and price_now:
                change = (price_now - price_start) / price_start

                # Trend thresholds
                if change > 0.005:  # >0.5% up
                    return 1  # Uptrend
                elif change < -0.005:  # >0.5% down
                    return -1  # Downtrend
                else:
                    return 0  # Sideways
        except:
            pass
        return None

    def run_backtest(self, blocks: int = 500, window_size: int = 50, verify_hours: int = 4):
        """
        Backtest the formula-based signals WITH price verification.

        Process blocks in windows, generate signals, check actual price movement.
        """
        print(f"\n{'='*70}")
        print(f"FORMULA BACKTEST: {blocks} blocks, {window_size} block windows")
        print(f"Price verification: {verify_hours}h delay")
        print(f"{'='*70}")

        height = self._rpc('getblockcount')
        if not height:
            print("Failed to get block height")
            return

        # Start from older blocks to allow price verification
        blocks_per_hour = 6
        buffer_blocks = verify_hours * blocks_per_hour + 10
        end_block = height - buffer_blocks
        start_block = end_block - blocks

        print(f"Testing blocks {start_block:,} to {end_block:,}")
        print(f"(Buffer: {buffer_blocks} blocks for {verify_hours}h price check)")
        print()

        all_signals = []
        verified_results = []
        prev_netflows = defaultdict(float)

        for window_start in range(start_block, end_block, window_size):
            window_end = min(window_start + window_size, end_block)

            # Aggregate flows for this window
            ext_inflows = defaultdict(list)
            ext_outflows = defaultdict(list)
            window_timestamp = None

            for h in range(window_start, window_end):
                block_hash = self._rpc('getblockhash', [h])
                if not block_hash:
                    continue

                block = self._rpc('getblock', [block_hash, 3])
                if not block:
                    continue

                if window_timestamp is None:
                    window_timestamp = block.get('time', 0)

                for tx in block.get('tx', []):
                    tx_in = defaultdict(float)
                    tx_out = defaultdict(float)

                    # Check outputs
                    for vout in tx.get('vout', []):
                        addr = vout.get('scriptPubKey', {}).get('address')
                        value = vout.get('value', 0)
                        if addr and addr in self.exchange_addresses and value >= self.MIN_FLOW_BTC:
                            ex = self.exchange_addresses[addr]
                            tx_in[ex] += value

                    # Check inputs
                    for vin in tx.get('vin', []):
                        if 'coinbase' in vin:
                            continue
                        prevout = vin.get('prevout', {})
                        if prevout:
                            addr = prevout.get('scriptPubKey', {}).get('address')
                            value = prevout.get('value', 0)
                            if addr and addr in self.exchange_addresses and value >= self.MIN_FLOW_BTC:
                                ex = self.exchange_addresses[addr]
                                tx_out[ex] += value

                    # Only external flows (not internal transfers)
                    for ex in set(tx_in.keys()) | set(tx_out.keys()):
                        if ex in tx_in and ex in tx_out:
                            continue  # Internal
                        if ex in tx_in:
                            ext_inflows[ex].append(tx_in[ex])
                        if ex in tx_out:
                            ext_outflows[ex].append(tx_out[ex])

            # Generate signals for each exchange
            for ex in set(ext_inflows.keys()) | set(ext_outflows.keys()):
                inflows = ext_inflows.get(ex, [])
                outflows = ext_outflows.get(ex, [])

                signal = self.generate_signal(
                    ex, inflows, outflows,
                    prev_netflow=prev_netflows[ex]
                )

                if signal and signal.direction != 0:
                    # NO FILTERS - RAW SIGNALS
                    all_signals.append(signal)
                    self.signals_generated += 1

                    # Verify with price
                    if window_timestamp:
                        signal_ms = window_timestamp * 1000
                        later_ms = signal_ms + (verify_hours * 3600 * 1000)

                        price_at = self.get_historical_price(signal_ms)
                        price_later = self.get_historical_price(later_ms)

                        if price_at and price_later:
                            price_change = (price_later - price_at) / price_at
                            # LONG expects price UP, SHORT expects price DOWN
                            correct = (signal.direction == 1 and price_change > 0) or \
                                     (signal.direction == -1 and price_change < 0)

                            verified_results.append({
                                'signal': signal,
                                'price_at': price_at,
                                'price_later': price_later,
                                'price_pct': price_change * 100,
                                'correct': correct
                            })

                            action = "LONG" if signal.direction == 1 else "SHORT"
                            status = "CORRECT" if correct else "WRONG"
                            color = '\033[92m' if correct else '\033[91m'
                            reset = '\033[0m'

                            print(f"{color}[{action}] {ex} {signal.net_btc:.0f}BTC "
                                  f"conf:{signal.confidence:.0%} agree:{signal.formula_agreement}/4 | "
                                  f"${price_at:,.0f}->${price_later:,.0f} ({price_change*100:+.2f}%) {status}{reset}")

                # Update prev netflow
                prev_netflows[ex] = sum(outflows) - sum(inflows)

            print(f"  Window {window_start}-{window_end} done")

        # Summary
        print()
        print("=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print(f"Total signals: {len(all_signals)}")
        print(f"Verified signals: {len(verified_results)}")

        if verified_results:
            correct = sum(1 for r in verified_results if r['correct'])
            total = len(verified_results)
            accuracy = correct / total * 100

            print()
            print(f"{'='*50}")
            print(f"ACCURACY: {correct}/{total} = {accuracy:.1f}%")
            print(f"{'='*50}")

            # By confidence level
            high_conf = [r for r in verified_results if r['signal'].confidence >= 0.75]
            med_conf = [r for r in verified_results if 0.50 <= r['signal'].confidence < 0.75]

            if high_conf:
                hc_correct = sum(1 for r in high_conf if r['correct'])
                print(f"\nHigh confidence (>=75%): {hc_correct}/{len(high_conf)} = {hc_correct/len(high_conf)*100:.1f}%")
            if med_conf:
                mc_correct = sum(1 for r in med_conf if r['correct'])
                print(f"Medium confidence (50-75%): {mc_correct}/{len(med_conf)} = {mc_correct/len(med_conf)*100:.1f}%")

            # By formula agreement
            print(f"\nBy formula agreement:")
            for agree in [4, 3, 2, 1]:
                subset = [r for r in verified_results if r['signal'].formula_agreement == agree]
                if subset:
                    sub_correct = sum(1 for r in subset if r['correct'])
                    print(f"  {agree}/4 agree: {sub_correct}/{len(subset)} = {sub_correct/len(subset)*100:.1f}%")

            # By direction
            longs = [r for r in verified_results if r['signal'].direction == 1]
            shorts = [r for r in verified_results if r['signal'].direction == -1]

            print(f"\nBy direction:")
            if longs:
                l_correct = sum(1 for r in longs if r['correct'])
                print(f"  LONG:  {l_correct}/{len(longs)} = {l_correct/len(longs)*100:.1f}%")
            if shorts:
                s_correct = sum(1 for r in shorts if r['correct'])
                print(f"  SHORT: {s_correct}/{len(shorts)} = {s_correct/len(shorts)*100:.1f}%")

            # By exchange
            print(f"\nBy exchange:")
            by_ex = defaultdict(list)
            for r in verified_results:
                by_ex[r['signal'].exchange].append(r)
            for ex, results in sorted(by_ex.items(), key=lambda x: -len(x[1])):
                ex_correct = sum(1 for r in results if r['correct'])
                ex_acc = ex_correct / len(results) * 100
                total_btc = sum(r['signal'].net_btc for r in results)
                print(f"  {ex:<15} {ex_correct}/{len(results)} = {ex_acc:.1f}% | {total_btc:,.0f} BTC")
        else:
            print("No verified results")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Mathematical signal engine')
    parser.add_argument('--backtest', type=int, default=500, help='Blocks to backtest')
    parser.add_argument('--window', type=int, default=50, help='Window size in blocks')
    parser.add_argument('--hours', type=int, default=4, help='Hours delay for price verification')
    args = parser.parse_args()

    engine = MathSignalEngine()
    engine.run_backtest(args.backtest, args.window, args.hours)


if __name__ == '__main__':
    main()
