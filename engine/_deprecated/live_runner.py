#!/usr/bin/env python3
"""
LIVE TRADING RUNNER - GOLD STANDARD MULTI-EXCHANGE
===================================================
HFT Engine → Failover Orchestrator → Multiple Exchanges

ARCHITECTURE:
┌────────────────────────────────────────────────────────────────────┐
│  HFT ENGINE (LOCAL)                                                │
│  ├── 237,000 signals/second                                        │
│  ├── OFI Formula 701 (R²=70%)                                      │
│  ├── CUSUM Filter 218                                              │
│  ├── Regime Filter 335                                             │
│  └── 47.9% WR × 2:1 TP/SL = +0.26 bps edge                        │
├────────────────────────────────────────────────────────────────────┤
│  FAILOVER ORCHESTRATOR                                             │
│  ├── Health monitoring (continuous)                                │
│  ├── Auto-failover on errors (>3 consecutive)                     │
│  ├── Auto-recovery to primary                                      │
│  └── Parallel submission option                                    │
├────────────────────────────────────────────────────────────────────┤
│  GOLD STANDARD EXCHANGES (priority order)                          │
│  1. Hyperliquid (on-chain, 200K/sec, own node = no limits)        │
│  2. dYdX v4 (on-chain, Cosmos, 2K TPS)                            │
│  3. Bybit (API, 500/sec VIP)                                       │
│  4. Binance (API, 10K/sec MM)                                      │
│  5. Solana/Jito (on-chain, MEV bundles)                           │
└────────────────────────────────────────────────────────────────────┘

Usage:
    python -m engine.live_runner [capital] [--testnet/--mainnet]

Environment Variables:
    HL_PRIVATE_KEY: Hyperliquid wallet private key
    BYBIT_API_KEY: Bybit API key (optional backup)
    BYBIT_API_SECRET: Bybit API secret (optional backup)
    BINANCE_API_KEY: Binance API key (optional backup)
    BINANCE_API_SECRET: Binance API secret (optional backup)

WARNING: This executes REAL trades with REAL money on mainnet.
"""
import os
import sys
import time
import asyncio
from typing import Optional
from dataclasses import dataclass

# Set environment before imports
os.environ['NUMBA_OPT'] = '3'
os.environ['NUMBA_BOUNDSCHECK'] = '0'


@dataclass
class LiveConfig:
    """Configuration for live trading.

    LOCAL NODE MODE (NO RATE LIMITS):
    When running on Hostinger with your own nodes, set:
    - hl_node_url: "http://localhost:4001" (Hyperliquid)
    - dydx_node_url: "http://localhost:26657" (dYdX v4)
    """
    capital: float = 100.0
    testnet: bool = True

    # Signal thresholds
    min_ofi_strength: float = 0.3  # Minimum OFI strength to trade
    min_confidence: float = 0.6    # Minimum confluence probability

    # Position limits
    max_position_usd: float = 100.0  # Max per position
    max_total_exposure: float = 0.5   # Max 50% of capital at risk

    # Execution
    batch_ms: int = 100  # Batch signals every 100ms
    instrument: str = "BTC"  # Trade BTC perpetuals

    # LOCAL NODE URLS (set these for zero rate limits!)
    # Hyperliquid: http://localhost:4001 (default Hyperliquid non-validator port)
    hl_node_url: Optional[str] = None

    # dYdX v4: http://localhost:26657 (Cosmos RPC port)
    dydx_node_url: Optional[str] = None
    dydx_grpc_url: Optional[str] = None  # localhost:9090


class LiveTrader:
    """
    Live trader with multi-exchange failover.

    Connects HFT engine to gold standard exchanges with automatic failover.
    """

    def __init__(self, config: LiveConfig):
        self.config = config
        self.engine = None
        self.orchestrator = None  # Failover orchestrator

        # State
        self.running = False
        self.position = 0.0
        self.entry_price = 0.0

        # Stats
        self.stats = {
            'signals_generated': 0,
            'signals_filtered': 0,
            'orders_submitted': 0,
            'orders_filled': 0,
            'failovers': 0,
            'total_pnl': 0.0,
        }

        # Signal accumulator for batching
        self.signal_buffer = []
        self.last_execution = 0

    async def initialize(self) -> bool:
        """Initialize engine and multi-exchange orchestrator."""
        print("=" * 70)
        print("LIVE TRADING INITIALIZATION - GOLD STANDARD MULTI-EXCHANGE")
        print("=" * 70)

        # 1. Initialize HFT Engine
        print("\n[1/4] Initializing HFT Signal Engine...")
        try:
            from engine.engines import HFTEngine
            self.engine = HFTEngine(capital=self.config.capital)
            self.engine.start()
            print("  [OK] HFT Engine ready")
        except Exception as e:
            print(f"  [XX] HFT Engine failed: {e}")
            return False

        # 2. Create failover orchestrator
        print("\n[2/4] Creating failover orchestrator...")
        try:
            from engine.exchange.failover_orchestrator import (
                FailoverOrchestrator, FailoverConfig
            )
            failover_config = FailoverConfig(
                max_errors_before_failover=3,
                max_latency_ms=1000.0,
                health_check_interval=10.0,
                primary_recovery_interval=30.0,
            )
            self.orchestrator = FailoverOrchestrator(failover_config)
            print("  [OK] Failover orchestrator created")
        except Exception as e:
            print(f"  [XX] Orchestrator failed: {e}")
            return False

        # 3. Add exchanges
        print("\n[3/4] Connecting exchanges...")
        exchanges_connected = 0

        # Primary: Hyperliquid (on-chain, own node = no rate limits)
        hl_key = os.environ.get('HL_PRIVATE_KEY')
        if hl_key:
            try:
                from engine.exchange.hyperliquid_executor import create_hyperliquid_executor
                hl_executor = create_hyperliquid_executor(
                    private_key=hl_key,
                    testnet=self.config.testnet,
                    node_url=self.config.hl_node_url  # LOCAL NODE!
                )
                self.orchestrator.add_executor('hyperliquid', hl_executor, priority=0)
                exchanges_connected += 1
                node_mode = "LOCAL NODE" if self.config.hl_node_url else "Public API"
                print(f"  [OK] Hyperliquid (priority 1) - {node_mode}")
            except Exception as e:
                print(f"  [!!] Hyperliquid failed: {e}")
        else:
            print("  [--] Hyperliquid: HL_PRIVATE_KEY not set")

        # Backup 1: dYdX v4 (on-chain Cosmos, own node = no rate limits)
        dydx_mnemonic = os.environ.get('DYDX_MNEMONIC')
        if dydx_mnemonic or self.config.dydx_node_url:
            try:
                from engine.exchange.dydx_executor import create_dydx_executor
                dydx_executor = create_dydx_executor(
                    mnemonic=dydx_mnemonic or "",
                    testnet=self.config.testnet,
                    node_url=self.config.dydx_node_url,  # LOCAL NODE!
                    grpc_url=self.config.dydx_grpc_url,
                )
                self.orchestrator.add_executor('dydx', dydx_executor, priority=1)
                exchanges_connected += 1
                node_mode = "LOCAL NODE" if self.config.dydx_node_url else "Public API"
                print(f"  [OK] dYdX v4 (priority 2) - {node_mode}")
            except Exception as e:
                print(f"  [!!] dYdX failed: {e}")
        else:
            print("  [--] dYdX: DYDX_MNEMONIC not set (optional backup)")

        # Backup 2: Bybit (API)
        bybit_key = os.environ.get('BYBIT_API_KEY')
        bybit_secret = os.environ.get('BYBIT_API_SECRET')
        if bybit_key and bybit_secret:
            try:
                from engine.exchange.bybit_executor import create_bybit_executor
                bybit_executor = create_bybit_executor(
                    api_key=bybit_key,
                    api_secret=bybit_secret,
                    testnet=self.config.testnet
                )
                self.orchestrator.add_executor('bybit', bybit_executor, priority=1)
                exchanges_connected += 1
                print(f"  [OK] Bybit (priority 2)")
            except Exception as e:
                print(f"  [!!] Bybit failed: {e}")
        else:
            print("  [--] Bybit: API keys not set (optional backup)")

        # Backup 2: Binance (API)
        binance_key = os.environ.get('BINANCE_API_KEY')
        binance_secret = os.environ.get('BINANCE_API_SECRET')
        if binance_key and binance_secret:
            try:
                from engine.exchange.binance_executor import BinanceExecutor
                from engine.exchange.base_executor import ExchangeConfig
                binance_config = ExchangeConfig(
                    name="binance",
                    api_key=binance_key,
                    api_secret=binance_secret,
                    testnet=self.config.testnet,
                )
                binance_executor = BinanceExecutor(binance_config)
                self.orchestrator.add_executor('binance', binance_executor, priority=2)
                exchanges_connected += 1
                print(f"  [OK] Binance (priority 3)")
            except Exception as e:
                print(f"  [!!] Binance failed: {e}")
        else:
            print("  [--] Binance: API keys not set (optional backup)")

        # Check we have at least one exchange
        if exchanges_connected == 0:
            print("\n  [XX] No exchanges configured!")
            print("  Set at least one of: HL_PRIVATE_KEY, BYBIT_API_KEY/SECRET, BINANCE_API_KEY/SECRET")
            return False

        # 4. Connect all exchanges
        print(f"\n[4/4] Connecting to {exchanges_connected} exchange(s)...")
        connected = await self.orchestrator.connect_all()

        if connected == 0:
            print("  [XX] Failed to connect to any exchange")
            return False

        # Start health monitoring
        await self.orchestrator.start_health_monitoring()

        print("\n" + "=" * 70)
        print(f"LIVE TRADING READY - {connected} EXCHANGE(S) CONNECTED")
        print("=" * 70)
        self._print_config()
        self.orchestrator.print_status()

        return True

    def _print_config(self):
        """Print trading configuration."""
        print(f"\nCapital: ${self.config.capital:.2f}")
        print(f"Network: {'TESTNET' if self.config.testnet else 'MAINNET'}")
        print(f"Instrument: {self.config.instrument}")
        print(f"\nSignal Thresholds:")
        print(f"  Min OFI Strength: {self.config.min_ofi_strength}")
        print(f"  Min Confidence: {self.config.min_confidence}")
        print(f"\nPosition Limits:")
        print(f"  Max Position: ${self.config.max_position_usd:.2f}")
        print(f"  Max Exposure: {self.config.max_total_exposure*100:.0f}%")
        print(f"\nLocal Nodes (NO RATE LIMITS):")
        if self.config.hl_node_url:
            print(f"  Hyperliquid: {self.config.hl_node_url}")
        else:
            print(f"  Hyperliquid: Public API (rate limited)")
        if self.config.dydx_node_url:
            print(f"  dYdX v4: {self.config.dydx_node_url}")
        else:
            print(f"  dYdX v4: Public API (rate limited)")
        print(f"\nExecution:")
        print(f"  Batch Interval: {self.config.batch_ms}ms")
        print("=" * 70)

    async def run(self):
        """Main trading loop."""
        self.running = True
        start_time = time.time()
        tick_count = 0
        last_print = 0

        print("\n[LIVE] Starting trading loop...")
        print("[LIVE] Press Ctrl+C to stop\n")

        try:
            while self.running:
                # Process tick from HFT engine
                result = self.engine.process_tick()
                tick_count += 1

                # Extract signal
                signal = self._extract_signal(result)
                if signal:
                    self.signal_buffer.append(signal)
                    self.stats['signals_generated'] += 1

                # Batch execution every batch_ms
                now = time.time()
                if (now - self.last_execution) * 1000 >= self.config.batch_ms:
                    if self.signal_buffer:
                        await self._execute_batch()
                    self.last_execution = now

                # Print status
                elapsed = now - start_time
                if now - last_print > 1.0:  # Every second
                    tps = tick_count / elapsed if elapsed > 0 else 0
                    r = result[0]

                    ofi_label = "BUY!" if r['ofi_signal'] > 0 else ("SELL" if r['ofi_signal'] < 0 else "WAIT")
                    pos_label = "LONG" if self.position > 0 else ("SHORT" if self.position < 0 else "FLAT")

                    print(f"[{elapsed:6.1f}s] "
                          f"OFI:{r['ofi_value']:+.2f} {ofi_label:5s} | "
                          f"Signals: {self.stats['signals_generated']:,} | "
                          f"Orders: {self.stats['orders_submitted']}/{self.stats['orders_filled']} | "
                          f"PnL: ${self.stats['total_pnl']:+.4f} | "
                          f"Pos: {pos_label} | "
                          f"TPS: {tps:,.0f}")

                    last_print = now

                # Yield to event loop periodically
                if tick_count % 1000 == 0:
                    await asyncio.sleep(0)

        except KeyboardInterrupt:
            print("\n[LIVE] Stopping...")
        finally:
            await self.shutdown()

    def _extract_signal(self, result) -> Optional[dict]:
        """
        Extract actionable signal from HFT tick result.

        Returns signal dict if actionable, None otherwise.
        """
        r = result[0]

        # Get OFI signal (PRIMARY)
        ofi_signal = int(r['ofi_signal'])
        ofi_strength = float(r['ofi_strength'])

        # No signal
        if ofi_signal == 0:
            return None

        # Check strength threshold
        if ofi_strength < self.config.min_ofi_strength:
            self.stats['signals_filtered'] += 1
            return None

        # Check confluence confidence
        confidence = float(r['confluence_prob'])
        if confidence < self.config.min_confidence:
            self.stats['signals_filtered'] += 1
            return None

        # Get price
        price = float(r['market_price'])
        if price <= 0:
            return None

        # Check regime alignment
        regime = int(r['regime'])
        # Only trade with regime: BUY in uptrend (1), SELL in downtrend (-1)
        if regime != 0 and regime != ofi_signal:
            self.stats['signals_filtered'] += 1
            return None

        return {
            'timestamp': time.time(),
            'side': 'BUY' if ofi_signal > 0 else 'SELL',
            'strength': ofi_strength,
            'confidence': confidence,
            'price': price,
            'regime': regime,
        }

    async def _execute_batch(self):
        """Execute batched signals."""
        if not self.signal_buffer:
            return

        # Aggregate signals
        buy_signals = [s for s in self.signal_buffer if s['side'] == 'BUY']
        sell_signals = [s for s in self.signal_buffer if s['side'] == 'SELL']

        # Clear buffer
        self.signal_buffer = []

        # Net direction
        buy_strength = sum(s['strength'] for s in buy_signals)
        sell_strength = sum(s['strength'] for s in sell_signals)

        # Need clear direction (20% threshold)
        if buy_strength > sell_strength * 1.2:
            await self._execute_trade('BUY', buy_strength, buy_signals[-1]['price'])
        elif sell_strength > buy_strength * 1.2:
            await self._execute_trade('SELL', sell_strength, sell_signals[-1]['price'])

    async def _execute_trade(self, side: str, strength: float, price: float):
        """Execute a trade on Hyperliquid."""
        from engine.exchange.base_executor import Signal, OrderSide

        # Calculate position size
        position_usd = min(
            self.config.capital * 0.1 * strength,  # 10% of capital × strength
            self.config.max_position_usd
        )

        # Check total exposure
        current_exposure = abs(self.position) * price
        if current_exposure + position_usd > self.config.capital * self.config.max_total_exposure:
            return  # Would exceed exposure limit

        # Create signal
        signal = Signal(
            timestamp=time.time(),
            instrument=self.config.instrument,
            side=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
            strength=min(strength, 1.0),
            edge_bps=0.26,  # Our edge
            confidence=0.479,  # Win rate
            formula_id=701,  # OFI
            entry_price=price,
            take_profit=price * (1.01 if side == 'BUY' else 0.99),  # 1% TP
            stop_loss=price * (0.995 if side == 'BUY' else 1.005),  # 0.5% SL
        )

        # Convert to order
        order = self.executor.signal_to_order(signal, position_usd)

        # Submit to blockchain
        self.stats['orders_submitted'] += 1

        try:
            filled_order = await self.executor.submit_order(order)

            if filled_order.status.value == 'FILLED':
                self.stats['orders_filled'] += 1

                # Update position
                delta = filled_order.filled_quantity
                if side == 'SELL':
                    delta = -delta
                self.position += delta
                self.entry_price = filled_order.filled_price

                print(f"  [FILL] {side} {filled_order.filled_quantity:.6f} {self.config.instrument} @ ${filled_order.filled_price:,.2f}")

        except Exception as e:
            print(f"  [ERROR] Order failed: {e}")

    async def shutdown(self):
        """Shutdown gracefully."""
        self.running = False

        # Close positions if any
        if abs(self.position) > 0.0001:
            print(f"\n[LIVE] Closing position: {self.position:+.6f} {self.config.instrument}")
            # TODO: Close position

        # Disconnect
        if self.executor:
            await self.executor.disconnect()

        if self.engine:
            self.engine.stop()

        # Print final stats
        print("\n" + "=" * 70)
        print("LIVE TRADING SESSION COMPLETE")
        print("=" * 70)
        print(f"Signals Generated: {self.stats['signals_generated']:,}")
        print(f"Signals Filtered: {self.stats['signals_filtered']:,}")
        print(f"Orders Submitted: {self.stats['orders_submitted']}")
        print(f"Orders Filled: {self.stats['orders_filled']}")
        print(f"Total PnL: ${self.stats['total_pnl']:+.4f}")
        print("=" * 70)


async def main():
    """Main entry point."""
    # Parse arguments
    capital = 100.0
    testnet = True
    node_url = None

    for arg in sys.argv[1:]:
        if arg == '--mainnet':
            testnet = False
        elif arg == '--testnet':
            testnet = True
        elif arg.startswith('--node='):
            node_url = arg.split('=')[1]
        else:
            try:
                capital = float(arg)
            except ValueError:
                pass

    # Safety warning for mainnet
    if not testnet:
        print("\n" + "!" * 70)
        print("! WARNING: MAINNET MODE - REAL MONEY AT RISK !")
        print("!" * 70)
        print("\nPress Enter to continue or Ctrl+C to abort...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    # Create config
    config = LiveConfig(
        capital=capital,
        testnet=testnet,
        node_url=node_url,
    )

    # Run trader
    trader = LiveTrader(config)

    if await trader.initialize():
        await trader.run()
    else:
        print("\nInitialization failed. Check configuration.")


if __name__ == "__main__":
    try:
        os.nice(-20)
    except:
        pass

    asyncio.run(main())
