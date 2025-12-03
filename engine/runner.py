#!/usr/bin/env python3
"""
MODULAR ENGINE RUNNER (Layer 4 - Entry Point)
==============================================
Main entry point for running trading engines.

Usage:
    python -m engine.runner [engine_type] [capital]

Engine Types:
    blockchain  - UNIFIED BLOCKCHAIN ENGINE (paper or live)
    paper       - Paper trading on blockchain (no real money)
    live        - LIVE trading on blockchain (REAL MONEY)
    hft         - HFT Engine (historical simulation - LEGACY)
    renaissance - Renaissance Compounding ($100 → $10,000)

Examples:
    python -m engine.runner paper 5         # Paper trade with $5
    python -m engine.runner live 5          # Live trade with $5
    python -m engine.runner blockchain 100  # Default paper trading
    python -m engine.runner hft 100         # Legacy simulation

Environment Variables (for live trading):
    HL_PRIVATE_KEY    - Hyperliquid wallet private key
    MONAD_PRIVATE_KEY - Monad wallet private key
    SEI_MNEMONIC      - Sei wallet mnemonic
    INJ_MNEMONIC      - Injective wallet mnemonic
    DYDX_MNEMONIC     - dYdX wallet mnemonic
"""
import os
import sys
import time

# Environment optimization for Numba
os.environ['NUMBA_OPT'] = '3'
os.environ['NUMBA_LOOP_VECTORIZE'] = '1'
os.environ['NUMBA_INTEL_SVML'] = '1'
os.environ['NUMBA_ENABLE_AVX'] = '1'
os.environ['NUMBA_THREADING_LAYER'] = 'omp'
os.environ['NUMBA_BOUNDSCHECK'] = '0'

import numpy as np
np.seterr(all='ignore')


def run_hft_engine(capital: float = 100.0):
    """Run the HFT Engine."""
    from engine.engines import HFTEngine

    print("\n" + "=" * 70)
    print("HFT ENGINE - TICK-LEVEL TRADING")
    print("=" * 70)
    print("FORMULA PIPELINE:")
    print("  ID 701: OFI Flow-Following (R²=70%) - PRIMARY SIGNAL")
    print("  ID 218: CUSUM Filter (+8-12pp Win Rate)")
    print("  ID 335: Regime Filter (+3-5pp Win Rate)")
    print("  ID 333: Signal Confluence (Condorcet voting)")
    print("  ID 141: Z-Score (confirmation only)")
    print("=" * 70 + "\n")

    engine = HFTEngine(capital=capital)
    engine.start()

    tick_count = 0
    start_time = time.time()
    last_print = 0

    try:
        while True:
            result = engine.process_tick()
            tick_count += 1

            now = time.time()
            elapsed = now - start_time

            if tick_count % 5000 == 0 or (now - last_print) > 0.5:
                tps = tick_count / elapsed if elapsed > 0 else 0
                trades = result[0]['trades']
                wins = result[0]['wins']
                win_rate = wins / trades * 100 if trades > 0 else 0

                ofi = result[0]['ofi_value']
                ofi_sig = result[0]['ofi_signal']
                ofi_label = "BUY!" if ofi_sig > 0 else ("SELL" if ofi_sig < 0 else "WAIT")
                z = result[0]['z_score']

                print(f"[{elapsed:6.1f}s] "
                      f"OFI:{ofi:+.2f} {ofi_label:5s} | "
                      f"WIN: {win_rate:.1f}% | "
                      f"Trades: {trades:,} | "
                      f"PnL: ${result[0]['pnl']:+.4f} | "
                      f"Cap: ${result[0]['capital']:.4f} | "
                      f"Z:{z:+.1f} | "
                      f"TPS: {tps:,.0f}")

                last_print = now

            if tick_count % 50000 == 0:
                time.sleep(0)

    except KeyboardInterrupt:
        engine.stop()

    # Final stats
    stats = engine.get_stats()
    summary = engine.get_summary()

    print("\n" + "=" * 70)
    print("FINAL HFT PERFORMANCE")
    print("=" * 70)
    print(f"Runtime: {time.time() - start_time:.1f}s")
    print(f"Total Ticks: {tick_count:,}")
    print(f"TPS: {tick_count / (time.time() - start_time):,.0f}")
    print(f"Avg Tick: {stats.get('avg_ns', 0):.0f}ns")
    print("-" * 70)
    print(f"Total Trades: {summary['total_trades']:,}")
    print(f"Win Rate: {summary['win_rate']:.1f}%")
    print(f"Total PnL: ${summary['total_pnl']:+.6f}")
    print(f"Final Capital: ${summary['capital']:.4f}")
    print("-" * 70)
    print("BUCKET BREAKDOWN:")
    for b in engine.get_bucket_stats():
        if b['trades'] > 0:
            print(f"  {b['ticks']:7d} ticks: ${b['capital']:.4f} | "
                  f"Trades: {b['trades']:,} | Win: {b['win_rate']:.1f}% | "
                  f"PnL: ${b['pnl']:+.6f}")
    print("=" * 70)


def run_renaissance_engine(capital: float = 100.0, target: float = 10000.0):
    """Run the Renaissance Compounding Engine."""
    from engine.engines import RenaissanceEngine

    print("\n" + "=" * 70)
    print("RENAISSANCE COMPOUNDING ENGINE - $100 → $10,000")
    print("=" * 70)
    print("Master Equation: Capital(t) = Capital(0) × (1 + f × edge)^n")
    print("Using TRUE OFI from blockchain math (R²=70%)")
    print("=" * 70 + "\n")

    engine = RenaissanceEngine(capital=capital, target=target)
    engine.start()

    start_time = time.time()
    last_print = 0

    try:
        while True:
            state = engine.process_signal()

            if state is None:
                time.sleep(0.01)
                continue

            now = time.time()
            elapsed = now - start_time

            if now - last_print > 0.5:
                ofi_label = "BUY!" if state['ofi_direction'] > 0 else ("SELL" if state['ofi_direction'] < 0 else "WAIT")
                pos_label = "LONG" if state['position'] > 0 else ("SHORT" if state['position'] < 0 else "FLAT")

                print(f"[{elapsed:6.1f}s] "
                      f"${state['mid_price']:,.0f} | "
                      f"OFI:{state['ofi']:+.2f} {ofi_label:5s} | "
                      f"WR: {state['win_rate']:.1f}% S:{state['sharpe']:.1f} | "
                      f"Trades: {state['total_trades']:,} | "
                      f"PnL: ${state['total_pnl']:+.4f} | "
                      f"Cap: ${state['capital']:.2f} | "
                      f"Progress: {state['progress']:.1f}% | "
                      f"DD: {state['drawdown']:.1f}% | "
                      f"{pos_label:5s}")

                last_print = now

                if state['capital'] >= target:
                    print("\n" + "=" * 70)
                    print("TARGET REACHED! $100 → $10,000 ACHIEVED!")
                    print("=" * 70)
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        engine.stop()

    # Final stats
    summary = engine.get_summary()

    print("\n" + "=" * 70)
    print("RENAISSANCE COMPOUNDING RESULTS")
    print("=" * 70)
    print(f"Runtime: {summary['runtime_s']:.1f}s")
    print(f"Initial Capital: ${capital:.2f}")
    print(f"Final Capital: ${summary['capital']:.2f}")
    print(f"Growth: {summary['growth']:.1f}x")
    print(f"Progress to $10,000: {summary['progress']:.1f}%")
    print("-" * 70)
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Win Rate: {summary['win_rate']:.1f}%")
    print(f"Sharpe Ratio: {summary['sharpe']:.2f}")
    print(f"Max Drawdown: {summary['drawdown']:.1f}%")
    print(f"Total PnL: ${summary['total_pnl']:+.4f}")
    print("-" * 70)
    print(f"Signals Processed: {summary['signal_count']:,}")
    print("=" * 70)


def run_blockchain_engine(capital: float = 100.0, paper: bool = True):
    """
    Run the UNIFIED BLOCKCHAIN ENGINE.

    This is THE engine - one codebase for paper and live trading.
    The only difference is whether orders are submitted to blockchain.

    Args:
        capital: Starting capital
        paper: True = paper trading (no real orders), False = live trading
    """
    import asyncio
    from engine.unified_blockchain_engine import (
        UnifiedBlockchainEngine, UnifiedConfig, ExecutionMode
    )

    # Safety warning for live trading
    if not paper:
        print("\n" + "!" * 70)
        print("! WARNING: LIVE TRADING MODE - REAL MONEY AT RISK !")
        print("!" * 70)
        if capital <= 0:
            print("\nERROR: Must specify capital for live trading")
            return
        print(f"\nCapital: ${capital:.2f}")
        print("\nPress Enter to continue or Ctrl+C to abort...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    config = UnifiedConfig(
        capital=capital,
        mode=ExecutionMode.PAPER if paper else ExecutionMode.LIVE,
    )

    engine = UnifiedBlockchainEngine(config)

    async def _run():
        if await engine.start():
            await engine.run()

    asyncio.run(_run())


def run_live_engine(capital: float = 100.0, testnet: bool = True, node_url: str = None):
    """
    Run LIVE trading on Hyperliquid (LEGACY - use run_blockchain_engine instead).

    WARNING: This trades REAL money!
    """
    import asyncio
    from engine.live_runner import LiveTrader, LiveConfig

    config = LiveConfig(
        capital=capital,
        testnet=testnet,
        node_url=node_url,
    )

    trader = LiveTrader(config)

    async def _run():
        if await trader.initialize():
            await trader.run()
        else:
            print("\nInitialization failed.")

    asyncio.run(_run())


def main():
    """Main entry point."""
    # Apply HFT optimizations if available
    try:
        from core.hft_optimizer import HFTOptimizer
        print("\n" + "=" * 70)
        print("APPLYING HFT OPTIMIZATIONS")
        print("=" * 70)
        optimizer = HFTOptimizer(verbose=True)
        optimizer.apply_all(aggressive=True)
        print("=" * 70 + "\n")
    except ImportError:
        pass

    # Parse arguments
    engine_type = 'blockchain'  # Default to unified blockchain engine
    capital = 100.0
    testnet = True
    node_url = None
    paper_mode = True  # Default to paper trading

    for arg in sys.argv[1:]:
        if arg.lower() in ['hft', 'renaissance', 'live', 'paper', 'blockchain']:
            engine_type = arg.lower()
            if arg.lower() == 'live':
                paper_mode = False
            elif arg.lower() == 'paper':
                paper_mode = True
                engine_type = 'blockchain'
        elif arg == '--mainnet':
            testnet = False
        elif arg == '--testnet':
            testnet = True
        elif arg == '--live':
            paper_mode = False
        elif arg == '--paper':
            paper_mode = True
        elif arg.startswith('--node='):
            node_url = arg.split('=')[1]
        else:
            try:
                capital = float(arg)
            except ValueError:
                pass

    # Run appropriate engine
    if engine_type == 'blockchain' or engine_type == 'paper':
        # Unified blockchain engine (paper trading by default)
        run_blockchain_engine(capital=capital, paper=paper_mode)
    elif engine_type == 'live':
        # Live blockchain trading
        run_blockchain_engine(capital=capital, paper=False)
    elif engine_type == 'renaissance':
        run_renaissance_engine(capital=capital)
    elif engine_type == 'hft':
        # Legacy HFT simulation
        run_hft_engine(capital=capital)
    else:
        # Default to blockchain paper trading
        run_blockchain_engine(capital=capital, paper=True)


if __name__ == "__main__":
    try:
        os.nice(-20)
    except:
        pass

    main()
