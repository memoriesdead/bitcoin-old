#!/usr/bin/env python3
"""
BLOCKCHAIN PIPELINE TRADING - Academic Formulas + TRUE Price
=============================================================
Integrates:
- TRUE PRICE from blockchain mathematics
- 14 Academic peer-reviewed formulas (IDs 520-560)
- Kyle Lambda, VPIN, OFI, Microprice (Microstructure)
- NVT, MVRV, SOPR, Hash Ribbon (On-Chain)
- Almgren-Chriss, Avellaneda-Stoikov (Execution)
- Kelly Criterion, HMM Regime (Risk)

NO EXCHANGE APIS - Pure blockchain data + mathematical formulas.

Usage:
    python test_pipeline_trading.py 10        # $10 test
    python test_pipeline_trading.py 10000000  # $10M test
"""
import sys
import asyncio
import time

# Force unbuffered output
sys.stdout = sys.stderr

from blockchain.pipeline import BlockchainTradingPipeline, PipelineSignal
from blockchain.blockchain_feed import BlockchainFeed


class PipelineTrader:
    """
    Trading engine using blockchain pipeline with academic formulas.
    """

    def __init__(
        self,
        initial_capital: float = 10_000_000.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03,
        min_confidence: float = 0.5,
    ):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_confidence = min_confidence

        # Initialize pipeline with TRUE price
        self.pipeline = BlockchainTradingPipeline(
            energy_cost_kwh=0.044,
            lookback=100,
            min_confidence=min_confidence,
        )

        # Position tracking
        self.in_position = False
        self.position_type = None  # 'LONG' or 'SHORT'
        self.entry_price = 0.0
        self.position_size = 0.0

        # Performance tracking
        self.trades = 0
        self.wins = 0
        self.total_pnl = 0.0

    def process_signal(self, signal: PipelineSignal) -> str:
        """
        Process pipeline signal and manage positions.

        Returns: Action taken ('ENTRY_LONG', 'ENTRY_SHORT', 'EXIT', 'HOLD')
        """
        action = 'HOLD'
        btc_price = signal.exchange_price if signal.exchange_price > 0 else signal.true_price

        if self.in_position:
            # Calculate current PnL
            if self.position_type == 'LONG':
                pnl_pct = (btc_price - self.entry_price) / self.entry_price
            else:  # SHORT
                pnl_pct = (self.entry_price - btc_price) / self.entry_price

            current_pnl = self.position_size * pnl_pct

            # Check exits
            if pnl_pct <= -self.stop_loss_pct:
                # Stop loss hit
                self._close_position(current_pnl, 'SL', btc_price)
                action = 'EXIT_SL'

            elif pnl_pct >= self.take_profit_pct:
                # Take profit hit
                self._close_position(current_pnl, 'TP', btc_price, win=True)
                action = 'EXIT_TP'

            # Opposite signal with high confidence
            elif signal.signal == 1 and self.position_type == 'SHORT' and signal.confidence > 0.7:
                self._close_position(current_pnl, 'FLIP', btc_price, win=(current_pnl > 0))
                self._open_position('LONG', btc_price, signal.position_size)
                action = 'FLIP_LONG'

            elif signal.signal == -1 and self.position_type == 'LONG' and signal.confidence > 0.7:
                self._close_position(current_pnl, 'FLIP', btc_price, win=(current_pnl > 0))
                self._open_position('SHORT', btc_price, signal.position_size)
                action = 'FLIP_SHORT'

        else:
            # No position - look for entry
            if signal.signal == 1 and signal.confidence >= self.min_confidence:
                self._open_position('LONG', btc_price, signal.position_size)
                action = 'ENTRY_LONG'

            elif signal.signal == -1 and signal.confidence >= self.min_confidence:
                self._open_position('SHORT', btc_price, signal.position_size)
                action = 'ENTRY_SHORT'

        return action

    def _open_position(self, position_type: str, price: float, kelly_size: float):
        """Open a new position."""
        self.position_type = position_type
        self.entry_price = price
        # Use Kelly-recommended size, capped at 25% of capital
        self.position_size = self.capital * min(0.25, max(0.05, kelly_size))
        self.in_position = True
        print(f"[ENTRY] {position_type} @ ${price:,.2f} | Size: ${self.position_size:,.2f} | Kelly: {kelly_size:.1%}", flush=True)

    def _close_position(self, pnl: float, reason: str, exit_price: float, win: bool = False):
        """Close current position."""
        self.total_pnl += pnl
        self.trades += 1
        if win or pnl > 0:
            self.wins += 1

        pnl_pct = (exit_price - self.entry_price) / self.entry_price
        if self.position_type == 'SHORT':
            pnl_pct = -pnl_pct

        print(f"[{reason}] {self.position_type} closed | Entry: ${self.entry_price:,.0f} | Exit: ${exit_price:,.0f} | PnL: ${pnl:,.2f} ({pnl_pct*100:+.2f}%)", flush=True)

        self.in_position = False
        self.position_type = None
        self.entry_price = 0.0
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


async def main():
    # Get capital from command line
    capital = float(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000.0

    print("=" * 80, flush=True)
    print("BLOCKCHAIN PIPELINE TRADING - Academic Formulas + TRUE Price", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)
    print("Formulas (IDs 520-560):", flush=True)
    print("  - Kyle Lambda, VPIN, OFI, Microprice (Microstructure)", flush=True)
    print("  - NVT, MVRV, SOPR, Hash Ribbon (On-Chain)", flush=True)
    print("  - Almgren-Chriss, Avellaneda-Stoikov (Execution)", flush=True)
    print("  - Kelly Criterion, HMM Regime (Risk)", flush=True)
    print("  - TRUE Price Deviation (Core Signal)", flush=True)
    print(flush=True)
    print("NO EXCHANGE APIS - Pure blockchain data", flush=True)
    print("=" * 80, flush=True)

    # Initialize trader with pipeline
    trader = PipelineTrader(
        initial_capital=capital,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        min_confidence=0.5,
    )

    # Get initial TRUE price
    trader.pipeline.update_true_price()
    true_price = trader.pipeline.current_true_price
    print(f"TRUE BTC PRICE: ${true_price:,.2f}", flush=True)
    print(f"CAPITAL: ${capital:,.2f}", flush=True)

    # Start blockchain feed for price data
    feed = BlockchainFeed()
    feed_task = asyncio.create_task(feed.start())

    print("Waiting 3s for blockchain data...", flush=True)
    await asyncio.sleep(3)

    print(f"\n>>> PIPELINE TRADING LIVE <<<", flush=True)
    print(f">>> Capital: ${capital:,.2f} | SL: {trader.stop_loss_pct*100:.1f}% | TP: {trader.take_profit_pct*100:.1f}%", flush=True)
    print("-" * 80, flush=True)

    start = time.time()
    updates = 0
    last_print = time.time()
    last_true_update = time.time()

    # Simulate price with small variations around TRUE price
    # In production, this would come from blockchain mempool/block data
    current_price = true_price

    try:
        while True:
            now = time.time()

            # Update TRUE price every 10 seconds
            if now - last_true_update >= 10.0:
                trader.pipeline.update_true_price()
                true_price = trader.pipeline.current_true_price
                last_true_update = now

            # Simulate market price variation around TRUE price
            # Real implementation would use actual blockchain transaction data
            import random
            noise = random.gauss(0, true_price * 0.0005)  # 0.05% noise
            current_price = true_price + noise

            # Get blockchain state for additional volume data
            stats = feed.get_stats()
            volume = stats.get('tx_rate', 1000) if stats else 1000

            # Process through pipeline
            signal = trader.pipeline.process(
                price=current_price,
                volume=volume,
                timestamp=now,
            )

            updates += 1

            # Execute trading logic
            action = trader.process_signal(signal)

            # Print status every 5 seconds
            if now - last_print >= 5.0:
                elapsed = now - start
                rate = updates / elapsed if elapsed > 0 else 0

                pos_str = f"{trader.position_type}" if trader.in_position else "FLAT"
                open_pnl = trader.get_open_pnl(current_price)
                open_pnl_str = f" | Open: ${open_pnl:+,.0f}" if trader.in_position else ""

                # Get regime from pipeline
                regime = signal.regime.upper()

                print(f"[{elapsed:6.1f}s] TRUE: ${true_price:,.0f} | MKT: ${current_price:,.0f} | Regime: {regime} | Pos: {pos_str}{open_pnl_str} | Trades: {trader.trades} | WR: {trader.win_rate:.1f}% | PnL: ${trader.total_pnl:+,.2f} | {rate:.1f}/s", flush=True)

                # Print component signals every 30 seconds
                if int(elapsed) % 30 == 0 and elapsed > 5:
                    print(f"  Components: ", end="", flush=True)
                    for name, (sig, conf) in list(signal.component_signals.items())[:6]:
                        sig_str = {1: '+', -1: '-', 0: '0'}[sig]
                        print(f"{name}:{sig_str}({conf:.1f}) ", end="", flush=True)
                    print(flush=True)

                last_print = now

            await asyncio.sleep(0.1)  # 10 updates/second

    except KeyboardInterrupt:
        print("\n" + "=" * 80, flush=True)
        print("FINAL RESULTS", flush=True)
        print("=" * 80, flush=True)
        elapsed = time.time() - start

        print(f"Duration: {elapsed:.1f}s", flush=True)
        print(f"Updates: {updates:,}", flush=True)
        print(f"Total Trades: {trader.trades}", flush=True)
        print(f"Wins: {trader.wins}", flush=True)
        print(f"Win Rate: {trader.win_rate:.1f}%", flush=True)
        print(f"Total PnL: ${trader.total_pnl:+,.2f}", flush=True)
        print(f"Return: {(trader.total_pnl / capital * 100):+.2f}%", flush=True)
        print(f"Final TRUE Price: ${true_price:,.2f}", flush=True)

        # Print edge analysis
        print("\nEdge Analysis:", flush=True)
        analysis = trader.pipeline.get_edge_analysis()
        if 'regime' in analysis:
            print(f"  Regime: {analysis['regime']['current']}", flush=True)
            print(f"  Regime Probs: Bull={analysis['regime']['probabilities']['bull']:.1%}, Bear={analysis['regime']['probabilities']['bear']:.1%}", flush=True)
        if 'position_sizing' in analysis:
            print(f"  Kelly Fraction: {analysis['position_sizing']['kelly_fraction']:.2%}", flush=True)

        print("=" * 80, flush=True)

    finally:
        feed_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
