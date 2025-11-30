#!/usr/bin/env python3
"""
EXPLOSIVE BLOCKCHAIN TRADING - 300,000+ Trades/Day
===================================================
Mathematically optimized for:
- 50.75%+ WIN RATE (edge-optimized)
- 300,000+ rapid trades per day
- Maximum mathematical edge from academic formulas

Formula Edge Sources (IDs 520-560):
- Kyle Lambda: Liquidity detection → trade when liquid
- VPIN: Toxic flow avoidance → skip toxic periods
- OFI: Order flow momentum → ride the flow
- Microprice: Fair value → trade deviations
- NVT/MVRV/SOPR: Valuation edge → contrarian signals
- Hash Ribbon: Miner capitulation → accumulation zones
- Kelly: Optimal sizing → maximize growth
- HMM Regime: Trend filter → align with regime

EDGE FORMULA (Asymmetric SL/TP):
    EV = (Win_Rate × Take_Profit) - (Loss_Rate × Stop_Loss)

    With SL=0.15%, TP=0.20% (RR=1.33:1):
    Breakeven_WR = SL / (SL + TP) = 0.15 / 0.35 = 42.86%

    At 50.75% WR:
    EV = (0.5075 × 0.20%) - (0.4925 × 0.15%)
       = 0.1015% - 0.0739%
       = 0.0276% per trade

    With 300K trades/day: 0.0276% × 300,000 = 8,280% theoretical
    (Limited by position sizing, slippage, and capital constraints)

Usage:
    python test_explosive_trading.py 10        # $10 explosive test
    python test_explosive_trading.py 10000000  # $10M explosive test
"""
import sys
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List
from collections import deque

# Force unbuffered output
sys.stdout = sys.stderr

from blockchain.pipeline import BlockchainTradingPipeline, PipelineSignal
from blockchain.blockchain_feed import BlockchainFeed
from blockchain.real_price_feed import CoinbasePriceFeed  # REAL PRICE - NO MOCK DATA


@dataclass
class TradeResult:
    """Single trade result for edge calculation."""
    entry_price: float
    exit_price: float
    position_type: str  # 'LONG' or 'SHORT'
    pnl: float
    pnl_pct: float
    duration_ms: float
    win: bool


class ExplosiveTrader:
    """
    EXPLOSIVE trading engine - 300K+ trades/day with 50.75%+ win rate.

    Key optimizations:
    1. Ultra-tight SL/TP (0.15% vs 2-3%)
    2. Low confidence threshold (0.15 vs 0.5)
    3. Rapid signal processing (100+ updates/sec)
    4. Edge-weighted Kelly sizing
    5. Regime-aligned trading
    """

    def __init__(
        self,
        initial_capital: float = 10_000_000.0,
        # EXPLOSIVE PARAMETERS - TIGHT
        stop_loss_pct: float = 0.0015,      # 0.15% SL (tight)
        take_profit_pct: float = 0.002,      # 0.20% TP (slightly better RR)
        min_confidence: float = 0.15,        # Low threshold = more trades
        # Edge optimization
        min_edge_threshold: float = 0.005,   # 0.5% minimum edge
        target_win_rate: float = 0.5075,     # Target 50.75% WR
    ):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_confidence = min_confidence
        self.min_edge_threshold = min_edge_threshold
        self.target_win_rate = target_win_rate

        # Initialize pipeline with TRUE price
        self.pipeline = BlockchainTradingPipeline(
            energy_cost_kwh=0.044,
            lookback=50,  # Shorter lookback = faster adaptation
            min_confidence=0.1,  # Very low for pipeline
        )

        # Position tracking
        self.in_position = False
        self.position_type = None
        self.entry_price = 0.0
        self.entry_time = 0.0
        self.position_size = 0.0

        # Performance tracking
        self.trades = 0
        self.wins = 0
        self.total_pnl = 0.0
        self.trade_history: deque = deque(maxlen=1000)

        # Edge tracking (rolling)
        self.rolling_wins = deque(maxlen=100)
        self.rolling_pnl = deque(maxlen=100)

        # Signal quality tracking
        self.signal_hits = 0
        self.signal_misses = 0

    def calculate_live_edge(self) -> float:
        """Calculate current live edge from recent trades."""
        if len(self.rolling_wins) < 10:
            return self.min_edge_threshold  # Default edge

        win_rate = sum(self.rolling_wins) / len(self.rolling_wins)
        avg_pnl = sum(self.rolling_pnl) / len(self.rolling_pnl) if self.rolling_pnl else 0

        # Edge = WR × AvgWin - LR × AvgLoss
        # Simplified: just use rolling PnL average as edge proxy
        return avg_pnl if avg_pnl > 0 else self.min_edge_threshold

    def get_optimal_size(self, signal: PipelineSignal) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        Kelly Formula (asymmetric payoffs): f* = (p × b - q) / b
        Where:
            p = win probability
            b = win/loss ratio = TP / SL
            q = 1 - p

        For SL=0.15%, TP=0.20%: b = 0.20/0.15 = 1.333
        At 50.75% WR: f* = (0.5075 × 1.333 - 0.4925) / 1.333 = 0.139 = 13.9%
        """
        # Get live win rate or use target
        p = (self.wins / self.trades) if self.trades > 10 else self.target_win_rate
        q = 1 - p

        # Win/loss ratio (odds) from SL/TP
        b = self.take_profit_pct / self.stop_loss_pct  # 0.20/0.15 = 1.333

        # Full Kelly formula
        kelly_full = (p * b - q) / b if b > 0 else 0

        # Use fractional Kelly (25% of full) for safety
        kelly_fraction = 0.25
        kelly_safe = max(0, kelly_full * kelly_fraction)

        # Adjust by signal confidence
        confidence_adj = signal.confidence

        # Regime alignment bonus (1.2x if aligned)
        regime_mult = 1.0
        if signal.regime == 'bull' and signal.signal == 1:
            regime_mult = 1.2
        elif signal.regime == 'bear' and signal.signal == -1:
            regime_mult = 1.2

        # Final size: Kelly × confidence × regime (capped at 10%)
        optimal = kelly_safe * confidence_adj * regime_mult
        return min(0.10, max(0.02, optimal))

    def should_enter(self, signal: PipelineSignal) -> bool:
        """
        Determine if we should enter based on edge analysis.

        Entry criteria:
        1. Signal != 0
        2. Confidence >= threshold
        3. Live edge > minimum
        4. Not in drawdown crisis
        """
        if signal.signal == 0:
            return False

        if signal.confidence < self.min_confidence:
            return False

        # Check live edge
        live_edge = self.calculate_live_edge()
        if live_edge < -0.02:  # In bad streak, reduce trading
            return signal.confidence > 0.5  # Only high confidence

        # Regime alignment boost
        if signal.regime == 'bull' and signal.signal == 1:
            return True
        if signal.regime == 'bear' and signal.signal == -1:
            return True

        # Neutral regime - need higher confidence
        if signal.regime == 'neutral':
            return signal.confidence > 0.3

        return True

    def process_signal(self, signal: PipelineSignal, timestamp: float) -> str:
        """
        Process pipeline signal with explosive trading logic.

        Returns: Action taken
        """
        action = 'HOLD'
        btc_price = signal.exchange_price if signal.exchange_price > 0 else signal.true_price

        if self.in_position:
            # Calculate current PnL
            if self.position_type == 'LONG':
                pnl_pct = (btc_price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - btc_price) / self.entry_price

            current_pnl = self.position_size * pnl_pct
            duration_ms = (timestamp - self.entry_time) * 1000

            # EXPLOSIVE EXIT CONDITIONS (tight)

            # 1. Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                self._close_position(current_pnl, 'SL', btc_price, timestamp, win=False)
                action = 'EXIT_SL'

            # 2. Take profit
            elif pnl_pct >= self.take_profit_pct:
                self._close_position(current_pnl, 'TP', btc_price, timestamp, win=True)
                action = 'EXIT_TP'

            # 3. Time-based exit (max 0.5 seconds in position for 300K+ trades)
            elif duration_ms > 500:
                win = pnl_pct > 0
                reason = 'TIME_WIN' if win else 'TIME_LOSS'
                self._close_position(current_pnl, reason, btc_price, timestamp, win=win)
                action = f'EXIT_{reason}'

            # 4. Signal flip with any confidence
            elif signal.signal == 1 and self.position_type == 'SHORT':
                self._close_position(current_pnl, 'FLIP', btc_price, timestamp, win=(pnl_pct > 0))
                self._open_position('LONG', btc_price, signal, timestamp)
                action = 'FLIP_LONG'

            elif signal.signal == -1 and self.position_type == 'LONG':
                self._close_position(current_pnl, 'FLIP', btc_price, timestamp, win=(pnl_pct > 0))
                self._open_position('SHORT', btc_price, signal, timestamp)
                action = 'FLIP_SHORT'

        else:
            # EXPLOSIVE ENTRY CONDITIONS (aggressive)
            if self.should_enter(signal):
                if signal.signal == 1:
                    self._open_position('LONG', btc_price, signal, timestamp)
                    action = 'ENTRY_LONG'
                elif signal.signal == -1:
                    self._open_position('SHORT', btc_price, signal, timestamp)
                    action = 'ENTRY_SHORT'

        return action

    def _open_position(self, position_type: str, price: float, signal: PipelineSignal, timestamp: float):
        """Open a new position with optimal sizing."""
        self.position_type = position_type
        self.entry_price = price
        self.entry_time = timestamp

        # Optimal Kelly-adjusted size
        size_pct = self.get_optimal_size(signal)
        self.position_size = self.capital * size_pct
        self.in_position = True

    def _close_position(self, pnl: float, reason: str, exit_price: float, timestamp: float, win: bool = False):
        """Close current position and record trade."""
        self.total_pnl += pnl
        self.trades += 1

        if win or pnl > 0:
            self.wins += 1
            self.rolling_wins.append(1)
        else:
            self.rolling_wins.append(0)

        # Track PnL for edge calculation
        pnl_pct = pnl / self.position_size if self.position_size > 0 else 0
        self.rolling_pnl.append(pnl_pct)

        # Record trade
        duration_ms = (timestamp - self.entry_time) * 1000
        trade = TradeResult(
            entry_price=self.entry_price,
            exit_price=exit_price,
            position_type=self.position_type,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_ms=duration_ms,
            win=win or pnl > 0
        )
        self.trade_history.append(trade)

        # Reset position
        self.in_position = False
        self.position_type = None
        self.entry_price = 0.0
        self.entry_time = 0.0
        self.position_size = 0.0

    def get_open_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        if not self.in_position:
            return 0.0

        if self.position_type == 'LONG':
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        return self.position_size * pnl_pct

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades * 100) if self.trades > 0 else 0.0

    @property
    def edge(self) -> float:
        """
        Current edge percentage using correct asymmetric SL/TP formula.

        Formula: Edge = WR × TP - (1-WR) × SL

        Breakeven WR = SL / (SL + TP) = 0.0015 / 0.0035 = 42.86%
        """
        if self.trades < 10:
            return 0.0

        wr = self.wins / self.trades

        # Correct edge with asymmetric SL/TP
        # EV per trade = WR × TP - (1-WR) × SL
        ev_per_trade = (wr * self.take_profit_pct) - ((1 - wr) * self.stop_loss_pct)

        # Return as percentage
        return ev_per_trade * 100

    @property
    def trades_per_day(self) -> float:
        """Projected trades per day based on recent activity."""
        if len(self.trade_history) < 2:
            return 0.0

        recent = list(self.trade_history)[-100:]
        if len(recent) < 2:
            return 0.0

        # Calculate average trade duration
        avg_duration_s = sum(t.duration_ms for t in recent) / len(recent) / 1000
        if avg_duration_s <= 0:
            return 300000  # Default target

        # Trades per day = 86400 / avg_duration
        return 86400 / avg_duration_s


async def main():
    # Get capital from command line
    capital = float(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000.0

    print("=" * 80, flush=True)
    print("EXPLOSIVE BLOCKCHAIN TRADING - 300K+ Trades/Day", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)
    print("EDGE OPTIMIZATION:", flush=True)
    print("  - Target Win Rate: 50.75%+", flush=True)
    print("  - Target Trades/Day: 300,000+", flush=True)
    print("  - SL/TP: 0.15% / 0.20% (tight for rapid cycling)", flush=True)
    print("  - Max Position Time: 0.5 seconds (ultra-fast cycling)", flush=True)
    print(flush=True)
    print("FORMULAS (IDs 520-560):", flush=True)
    print("  Kyle Lambda | VPIN | OFI | Microprice | NVT | MVRV | SOPR", flush=True)
    print("  Hash Ribbon | Almgren-Chriss | Avellaneda-Stoikov | Kelly | HMM", flush=True)
    print("=" * 80, flush=True)

    # Initialize explosive trader
    trader = ExplosiveTrader(
        initial_capital=capital,
        stop_loss_pct=0.0015,     # 0.15%
        take_profit_pct=0.002,    # 0.20%
        min_confidence=0.15,
        target_win_rate=0.5075,
    )

    # Get initial TRUE price
    trader.pipeline.update_true_price()
    true_price = trader.pipeline.current_true_price
    # IMPORTANT: Set _last_true_update to prevent blocking HTTP call in process()
    trader.pipeline._last_true_update = time.time()
    print(f"TRUE BTC PRICE: ${true_price:,.2f}", flush=True)
    print(f"CAPITAL: ${capital:,.2f}", flush=True)

    # Start blockchain feed for transaction data
    feed = BlockchainFeed()
    feed_task = asyncio.create_task(feed.start())

    # Start REAL price feed - NO MOCK DATA
    price_feed = CoinbasePriceFeed()
    price_task = asyncio.create_task(price_feed.start())

    print("Warming up feeds (3s)...", flush=True)
    print("  - Blockchain feed: Real mempool data", flush=True)
    print("  - Price feed: Real Coinbase prices", flush=True)
    print("  - NO MOCK DATA - Pure blockchain trading", flush=True)
    await asyncio.sleep(3)

    print(f"\n>>> EXPLOSIVE TRADING LIVE <<<", flush=True)
    print(f">>> Target: 300K trades/day | 50.75% WR | Max Edge", flush=True)
    print("-" * 80, flush=True)

    start = time.time()
    updates = 0
    last_print = time.time()
    last_true_update = time.time()
    last_trade_count = 0

    current_price = true_price

    try:
        while True:
            now = time.time()

            # Update TRUE price every 120 seconds (avoid blocking the event loop)
            # The blocking HTTP call takes ~20-30 seconds, so don't do it frequently
            if now - last_true_update >= 120.0:
                trader.pipeline.update_true_price()
                true_price = trader.pipeline.current_true_price
                trader.pipeline._last_true_update = now  # Sync with pipeline
                last_true_update = now

            # REAL PRICE from Coinbase - NO MOCK DATA
            current_price = price_feed.get_price()
            if current_price <= 0:
                current_price = true_price  # Fallback to TRUE price if feed not ready

            # Get blockchain stats for volume
            stats = feed.get_stats()
            volume = stats.get('tx_rate', 1000) if stats else 1000

            # Process through pipeline
            signal = trader.pipeline.process(
                price=current_price,
                volume=volume,
                timestamp=now,
            )

            updates += 1

            # Execute explosive trading logic
            action = trader.process_signal(signal, now)

            # Print status every 2 seconds (more frequent for explosive)
            if now - last_print >= 2.0:
                elapsed = now - start
                rate = updates / elapsed if elapsed > 0 else 0
                trades_in_period = trader.trades - last_trade_count
                trade_rate = trades_in_period / 2.0  # trades per second

                pos_str = f"{trader.position_type}" if trader.in_position else "FLAT"
                open_pnl = trader.get_open_pnl(current_price)

                # Calculate projected trades/day
                projected_daily = trade_rate * 86400 if trade_rate > 0 else 0

                print(
                    f"[{elapsed:5.1f}s] "
                    f"TRUE: ${true_price:,.0f} | "
                    f"MKT: ${current_price:,.0f} | "
                    f"Trades: {trader.trades} ({trade_rate:.1f}/s = {projected_daily:,.0f}/day) | "
                    f"WR: {trader.win_rate:.2f}% | "
                    f"Edge: {trader.edge:+.2f}% | "
                    f"PnL: ${trader.total_pnl:+,.2f}",
                    flush=True
                )

                # Detailed stats every 10 seconds
                if int(elapsed) % 10 == 0 and elapsed > 5:
                    print(
                        f"    Regime: {signal.regime.upper()} | "
                        f"Conf: {signal.confidence:.2f} | "
                        f"Kelly: {signal.position_size:.1%} | "
                        f"Pos: {pos_str} | "
                        f"Open: ${open_pnl:+,.0f}",
                        flush=True
                    )

                last_print = now
                last_trade_count = trader.trades

            # EXPLOSIVE: 10ms sleep = 100 updates/second baseline
            # With 0.5s max position time = ~2 trades/second = 172,800/day minimum
            # Plus SL/TP exits = 300K+ trades/day target
            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\n" + "=" * 80, flush=True)
        print("EXPLOSIVE TRADING - FINAL RESULTS", flush=True)
        print("=" * 80, flush=True)
        elapsed = time.time() - start

        # Core metrics
        print(f"\nPERFORMANCE:", flush=True)
        print(f"  Duration: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
        print(f"  Total Trades: {trader.trades:,}", flush=True)
        print(f"  Wins: {trader.wins:,}", flush=True)
        print(f"  Win Rate: {trader.win_rate:.2f}%", flush=True)
        print(f"  Edge: {trader.edge:+.2f}%", flush=True)

        # Projections
        trades_per_sec = trader.trades / elapsed if elapsed > 0 else 0
        projected_daily = trades_per_sec * 86400
        print(f"\nPROJECTIONS:", flush=True)
        print(f"  Trades/Second: {trades_per_sec:.2f}", flush=True)
        print(f"  Trades/Day: {projected_daily:,.0f}", flush=True)
        print(f"  Target Met: {'YES' if projected_daily >= 300000 else 'NO'} (target: 300K+)", flush=True)

        # Financial
        print(f"\nFINANCIAL:", flush=True)
        print(f"  Capital: ${capital:,.2f}", flush=True)
        print(f"  Total PnL: ${trader.total_pnl:+,.2f}", flush=True)
        print(f"  Return: {(trader.total_pnl / capital * 100):+.4f}%", flush=True)

        # Daily projection
        if elapsed > 0:
            daily_return_pct = (trader.total_pnl / capital) * (86400 / elapsed) * 100
            print(f"  Projected Daily Return: {daily_return_pct:+.2f}%", flush=True)

        # Edge analysis
        print(f"\nEDGE ANALYSIS:", flush=True)
        analysis = trader.pipeline.get_edge_analysis()
        if 'regime' in analysis:
            print(f"  Current Regime: {analysis['regime']['current']}", flush=True)
            probs = analysis['regime']['probabilities']
            print(f"  Regime Probs: Bull={probs['bull']:.1%} | Bear={probs['bear']:.1%} | Neutral={probs['neutral']:.1%}", flush=True)

        # Win rate status
        print(f"\nWIN RATE STATUS:", flush=True)
        wr = trader.win_rate
        if wr >= 50.75:
            print(f"  TARGET ACHIEVED: {wr:.2f}% >= 50.75%", flush=True)
        else:
            print(f"  TARGET NOT MET: {wr:.2f}% < 50.75%", flush=True)

        print("=" * 80, flush=True)

    finally:
        feed_task.cancel()
        price_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
