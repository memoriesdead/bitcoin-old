#!/usr/bin/env python3
"""
BITCOIN DATA PIPELINE - Unified On-Chain Data Orchestrator
============================================================
Single unified interface for all blockchain-derived market data.

NO EXTERNAL PRICE APIS - Pure blockchain mathematics.

PIPELINE ARCHITECTURE:
----------------------
                    ┌──────────────────┐
                    │  Raw Blockchain  │
                    │      Data        │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   Supply    │  │   Volume    │  │  Realized   │
    │   Tracker   │  │  Aggregator │  │   Price     │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                │                │
           └────────┬───────┴───────┬────────┘
                    │               │
                    ▼               ▼
           ┌─────────────┐  ┌─────────────┐
           │   Derived   │  │   Market    │
           │    Price    │  │   Metrics   │
           └──────┬──────┘  └──────┬──────┘
                  │                │
                  └────────┬───────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  PipelineState  │
                  │  (Unified API)  │
                  └─────────────────┘

DATA FLOW:
----------
1. Raw block/transaction data comes in
2. SupplyTracker calculates circulating supply from block height
3. VolumeAggregator tracks rolling transaction volumes
4. RealizedPriceEngine tracks UTXO cost basis
5. DerivedPriceEngine computes price from multiple models
6. MarketMetricsEngine calculates all valuation ratios
7. PipelineState provides unified access to everything

USAGE:
------
    from renaissance.data import BitcoinDataPipeline

    pipeline = BitcoinDataPipeline(anchor_price=95000.0)

    # Process raw blockchain data
    pipeline.process_block(
        block_height=870000,
        transactions=[...],
        fee_rate=15,
        mempool_size=50000
    )

    # Get complete market state
    state = pipeline.get_state()
    print(f"Price: ${state.price:,.2f}")
    print(f"Market Cap: ${state.market_cap:,.0f}")
    print(f"24h Volume: {state.volume_24h:,.2f} BTC")
    print(f"Signal: {state.signal:+.3f}")

Author: Renaissance Trading System
Purpose: $10 -> $300,000+ via 300K-1M trades
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Import sub-components
from .supply_tracker import SupplyTracker, SupplyMetrics
from .realized_price import RealizedPriceEngine, RealizedPriceMetrics
from .derived_price import DerivedPriceEngine, PriceComponents
from .volume_aggregator import VolumeAggregator, VolumeMetrics
from .market_metrics import MarketMetricsEngine, FullMarketState


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Default anchor price for initial calibration
DEFAULT_ANCHOR_PRICE = 95000.0

# Active addresses estimate (updates from mempool activity)
DEFAULT_ACTIVE_ADDRESSES = 900_000


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class PipelineState:
    """
    Unified pipeline state - single source of truth.

    This is the main output of the pipeline, providing all
    derived metrics in a clean, organized structure.
    """
    timestamp: float = 0.0

    # === PRICE DATA ===
    price: float = 0.0                     # Current derived price (USD)
    price_change_1h: float = 0.0           # 1h price change %
    price_change_24h: float = 0.0          # 24h price change %
    price_momentum: float = 0.0            # Rate of change

    # Price model breakdown
    metcalfe_price: float = 0.0
    nvt_price: float = 0.0
    s2f_price: float = 0.0
    realized_price: float = 0.0
    anchor_price: float = 0.0

    # === SUPPLY DATA ===
    circulating_supply: float = 0.0        # Current circulating (BTC)
    max_supply: float = 21_000_000.0       # Maximum supply
    percent_mined: float = 0.0             # % of max mined
    block_height: int = 0                  # Current block
    blocks_until_halving: int = 0          # Blocks to next halving
    current_block_reward: float = 0.0      # BTC per block

    # === MARKET CAP ===
    market_cap: float = 0.0                # Price * Supply
    realized_cap: float = 0.0              # Sum of UTXO costs
    thermocap: float = 0.0                 # Total miner revenue

    # === VOLUME DATA ===
    volume_1h: float = 0.0                 # 1h volume (BTC)
    volume_24h: float = 0.0                # 24h volume (BTC)
    volume_7d: float = 0.0                 # 7d volume (BTC)
    tx_count_24h: int = 0                  # 24h transaction count
    whale_volume_1h: float = 0.0           # Whale transactions
    volume_trend: str = "NEUTRAL"          # INCREASING/DECREASING/NEUTRAL

    # === VALUATION RATIOS ===
    mvrv_ratio: float = 0.0                # Market/Realized value
    nvt_ratio: float = 0.0                 # Network value/transactions
    nupl: float = 0.0                      # Net unrealized P/L
    stock_to_flow: float = 0.0             # Scarcity metric
    puell_multiple: float = 0.0            # Mining profitability

    # === PROFIT/LOSS ===
    percent_in_profit: float = 0.0         # % of supply in profit
    percent_in_loss: float = 0.0           # % of supply in loss

    # === TRADING SIGNALS ===
    signal: float = 0.0                    # Composite -1 to +1
    signal_strength: float = 0.0           # Confidence 0 to 1
    signal_reason: str = ""                # Human readable
    market_regime: str = "NEUTRAL"         # ACCUMULATION/EXPANSION/etc

    # === RAW INPUTS (for formulas) ===
    fee_rate: float = 0.0                  # sat/vB
    mempool_size: int = 0                  # Unconfirmed TX count
    active_addresses: int = 0              # Active address estimate

    # === MODEL CONFIDENCE ===
    overall_confidence: float = 0.0        # 0 to 1


@dataclass
class BlockData:
    """Raw block data input structure."""
    height: int = 0
    timestamp: float = 0.0
    tx_count: int = 0
    total_value_btc: float = 0.0
    total_fees_sats: int = 0
    transactions: List[Dict[str, Any]] = field(default_factory=list)


# ==============================================================================
# BITCOIN DATA PIPELINE
# ==============================================================================

class BitcoinDataPipeline:
    """
    Unified Bitcoin data pipeline.

    Orchestrates all sub-components to provide complete market
    data derived purely from blockchain mathematics.

    This is the main entry point for the Renaissance trading system.
    """

    def __init__(
        self,
        anchor_price: float = DEFAULT_ANCHOR_PRICE,
        include_lost_coins: bool = False,
        auto_calibrate: bool = True
    ):
        """
        Initialize the pipeline.

        Args:
            anchor_price: Initial price anchor for calibration
            include_lost_coins: Subtract lost coins from supply
            auto_calibrate: Auto-adjust model parameters
        """
        self.anchor_price = anchor_price

        # Initialize sub-components
        self.supply_tracker = SupplyTracker(include_lost_coins=include_lost_coins)
        self.volume_aggregator = VolumeAggregator()
        self.realized_price_engine = RealizedPriceEngine()
        self.derived_price_engine = DerivedPriceEngine()  # Pure blockchain math
        self.market_metrics_engine = MarketMetricsEngine()

        # Current state
        self.current_state = PipelineState()

        # Price history for change calculations
        self.price_history: List[tuple] = []  # (timestamp, price)

        # Last block height processed
        self.last_block_height = 0

        # Active address estimate (from mempool activity)
        self.active_addresses_estimate = DEFAULT_ACTIVE_ADDRESSES

    def process_block(
        self,
        block_height: int,
        transactions: Optional[List[Dict]] = None,
        fee_rate: float = 1.0,
        mempool_size: int = 0,
        block_timestamp: Optional[float] = None
    ) -> PipelineState:
        """
        Process a new block and update all metrics.

        Args:
            block_height: Current block height
            transactions: List of transaction dicts with 'value', 'fee', etc.
            fee_rate: Current fee rate in sat/vB
            mempool_size: Number of unconfirmed transactions
            block_timestamp: Block timestamp (default: now)

        Returns:
            Updated PipelineState
        """
        now = block_timestamp or time.time()
        self.last_block_height = block_height

        # 1. Update supply tracker
        supply_metrics = self.supply_tracker.update(block_height)

        # 2. Process transactions in volume aggregator
        if transactions:
            self.volume_aggregator.add_block_transactions(
                transactions, block_timestamp=now
            )

        # Update volume metrics
        volume_metrics = self.volume_aggregator.update()

        # 3. Update realized price with transactions
        if transactions:
            current_price = self.current_state.price or self.anchor_price
            for tx in transactions:
                value = tx.get('value', 0) or tx.get('value_btc', 0)
                if value > 0:
                    self.realized_price_engine.record_transaction(
                        btc_amount=value,
                        derived_price=current_price
                    )

        realized_metrics = self.realized_price_engine.update(
            current_price=self.current_state.price or self.anchor_price
        )

        # 4. Estimate active addresses from mempool/tx activity
        if mempool_size > 0:
            # Rough estimate: active addresses ~ mempool_size * factor
            self.active_addresses_estimate = int(mempool_size * 15)
        elif volume_metrics.tx_count_1h > 0:
            self.active_addresses_estimate = int(volume_metrics.tx_count_1h * 50)

        # 5. Update derived price (100% pure blockchain math)
        price_components = self.derived_price_engine.update(
            block_height=block_height,
            circulating_supply=supply_metrics.circulating_supply,
            stock_to_flow=supply_metrics.stock_to_flow,
            realized_price=realized_metrics.realized_price,
            tx_volume_24h_btc=volume_metrics.volume_24h_btc,
            fee_rate_sats=fee_rate,
            mempool_size=mempool_size,
            tx_count=volume_metrics.tx_count_1h
        )

        # 6. Update market metrics
        market_state = self.market_metrics_engine.update(
            current_price=price_components.composite_price,
            circulating_supply=supply_metrics.circulating_supply,
            realized_cap=realized_metrics.realized_cap,
            volume_24h_btc=volume_metrics.volume_24h_btc,
            tx_count_24h=volume_metrics.tx_count_24h,
            daily_issuance_btc=supply_metrics.daily_issuance,
            stock_to_flow=supply_metrics.stock_to_flow,
            percent_in_profit=75.0 if realized_metrics.mvrv_ratio > 1.0 else 25.0,
            price_change_24h=self._calculate_price_change(24)
        )

        # 7. Build unified state
        self._build_unified_state(
            supply_metrics=supply_metrics,
            volume_metrics=volume_metrics,
            realized_metrics=realized_metrics,
            price_components=price_components,
            market_state=market_state,
            fee_rate=fee_rate,
            mempool_size=mempool_size
        )

        # Update price history
        self.price_history.append((now, price_components.composite_price))
        if len(self.price_history) > 10000:
            self.price_history = self.price_history[-5000:]

        return self.current_state

    def process_transaction(
        self,
        value_btc: float,
        fee_sats: int = 0,
        is_whale: bool = False,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Process a single transaction (for real-time mempool data).

        Args:
            value_btc: Transaction value in BTC
            fee_sats: Transaction fee in satoshis
            is_whale: Whether this is a whale transaction
            timestamp: Transaction timestamp
        """
        self.volume_aggregator.add_transaction(
            value_btc=value_btc,
            fee_sats=fee_sats,
            timestamp=timestamp
        )

        current_price = self.current_state.price or self.anchor_price
        self.realized_price_engine.record_transaction(
            value_btc=value_btc,
            price_at_creation=current_price
        )

    def update_from_mempool(
        self,
        mempool_size: int,
        fee_rate: float,
        tx_rate: float = 0.0,
        whale_count: int = 0,
        whale_volume: float = 0.0
    ) -> PipelineState:
        """
        Update with mempool data (between blocks).

        Args:
            mempool_size: Number of unconfirmed transactions
            fee_rate: Current fee rate in sat/vB
            tx_rate: Transactions per second
            whale_count: Number of whale transactions
            whale_volume: Volume of whale transactions

        Returns:
            Updated PipelineState
        """
        # Use last known block height
        return self.process_block(
            block_height=self.last_block_height,
            transactions=None,
            fee_rate=fee_rate,
            mempool_size=mempool_size
        )

    def _calculate_price_change(self, hours: int) -> float:
        """Calculate price change over specified hours."""
        if len(self.price_history) < 2:
            return 0.0

        now = time.time()
        cutoff = now - (hours * 3600)

        current_price = self.price_history[-1][1]

        # Find price at cutoff time
        old_price = current_price
        for ts, price in reversed(self.price_history):
            if ts <= cutoff:
                old_price = price
                break

        if old_price <= 0:
            return 0.0

        return ((current_price - old_price) / old_price) * 100

    def _build_unified_state(
        self,
        supply_metrics: SupplyMetrics,
        volume_metrics: VolumeMetrics,
        realized_metrics: RealizedPriceMetrics,
        price_components: PriceComponents,
        market_state: FullMarketState,
        fee_rate: float,
        mempool_size: int
    ) -> None:
        """Build the unified pipeline state from all components."""
        now = time.time()

        # Get volume signal
        volume_signal, volume_strength = self.volume_aggregator.get_volume_signal()

        # Get MVRV signal
        mvrv_signal, mvrv_strength = self.realized_price_engine.get_mvrv_signal()

        # Combine signals
        combined_signal = (
            price_components.signal * 0.4 +
            market_state.composite_signal * 0.4 +
            (0.3 if volume_signal == "BULLISH" else -0.3 if volume_signal == "BEARISH" else 0) * 0.2
        )

        # Build signal reason
        reasons = []
        if abs(price_components.signal) > 0.1:
            reasons.append(price_components.signal_reason)
        if mvrv_signal != "NEUTRAL":
            reasons.append(f"MVRV: {mvrv_signal}")
        if volume_signal != "NEUTRAL":
            reasons.append(f"Volume: {volume_signal}")

        self.current_state = PipelineState(
            timestamp=now,

            # Price
            price=price_components.composite_price,
            price_change_1h=self._calculate_price_change(1),
            price_change_24h=self._calculate_price_change(24),
            price_momentum=price_components.price_momentum,
            metcalfe_price=price_components.metcalfe_price,
            nvt_price=price_components.nvt_price,
            s2f_price=price_components.stock_to_flow_price,
            realized_price=realized_metrics.realized_price,
            anchor_price=self.anchor_price,

            # Supply
            circulating_supply=supply_metrics.circulating_supply,
            max_supply=supply_metrics.max_supply,
            percent_mined=supply_metrics.percent_mined,
            block_height=supply_metrics.block_height,
            blocks_until_halving=supply_metrics.blocks_until_halving,
            current_block_reward=supply_metrics.current_block_reward,

            # Market cap
            market_cap=market_state.market_cap,
            realized_cap=realized_metrics.realized_cap,
            thermocap=market_state.thermocap,

            # Volume
            volume_1h=volume_metrics.volume_1h_btc,
            volume_24h=volume_metrics.volume_24h_btc,
            volume_7d=volume_metrics.volume_7d_btc,
            tx_count_24h=volume_metrics.tx_count_24h,
            whale_volume_1h=volume_metrics.whale_volume_1h_btc,
            volume_trend=volume_metrics.volume_trend,

            # Valuation
            mvrv_ratio=market_state.mvrv_ratio,
            nvt_ratio=market_state.nvt_ratio,
            nupl=market_state.nupl,
            stock_to_flow=supply_metrics.stock_to_flow,
            puell_multiple=market_state.puell_multiple,

            # Profit/Loss
            percent_in_profit=realized_metrics.pct_supply_in_profit,
            percent_in_loss=realized_metrics.pct_supply_in_loss,

            # Signals
            signal=combined_signal,
            signal_strength=price_components.overall_confidence,
            signal_reason="; ".join(reasons) if reasons else "Neutral",
            market_regime=market_state.market_regime,

            # Raw inputs
            fee_rate=fee_rate,
            mempool_size=mempool_size,
            active_addresses=self.active_addresses_estimate,

            # Confidence
            overall_confidence=price_components.overall_confidence,
        )

    def get_state(self) -> PipelineState:
        """Get current pipeline state."""
        return self.current_state

    def get_price(self) -> float:
        """Get current derived price."""
        return self.current_state.price

    def get_signal(self) -> float:
        """Get current trading signal (-1 to +1)."""
        return self.current_state.signal

    def update_anchor_price(self, new_anchor: float) -> None:
        """Update anchor price for recalibration."""
        self.anchor_price = new_anchor
        self.derived_price_engine.update_anchor(new_anchor)

    def get_formula_inputs(self) -> Dict[str, Any]:
        """
        Get inputs formatted for trading formulas.

        Returns dict compatible with formula update() calls.
        """
        s = self.current_state
        return {
            'price': s.price,
            'volume': s.volume_1h,
            'timestamp': s.timestamp,
            'fee_rate': s.fee_rate,
            'mempool_size': s.mempool_size,
            'tx_count': s.tx_count_24h // 144,  # Per-block estimate
            'whale_volume': s.whale_volume_1h,
            'mvrv': s.mvrv_ratio,
            'nvt': s.nvt_ratio,
            'signal': s.signal,
            'market_regime': s.market_regime,
        }


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def format_pipeline_state(s: PipelineState) -> str:
    """Format pipeline state for display."""
    return f"""
================================================================================
              RENAISSANCE BITCOIN DATA PIPELINE - UNIFIED STATE
================================================================================
  Timestamp:           {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(s.timestamp))}
================================================================================
  PRICE:
    Derived Price:     ${s.price:>15,.2f}
    1h Change:         {s.price_change_1h:>14.2f}%
    24h Change:        {s.price_change_24h:>14.2f}%
    Momentum:          {s.price_momentum:>14.4%}
  ─────────────────────────────────────────────────────────────────────────────
  PRICE MODELS:
    Metcalfe:          ${s.metcalfe_price:>15,.2f}
    NVT Model:         ${s.nvt_price:>15,.2f}
    Stock-to-Flow:     ${s.s2f_price:>15,.2f}
    Realized Price:    ${s.realized_price:>15,.2f}
    Anchor:            ${s.anchor_price:>15,.2f}
================================================================================
  SUPPLY:
    Circulating:       {s.circulating_supply:>15,.2f} BTC
    Max Supply:        {s.max_supply:>15,.0f} BTC
    Percent Mined:     {s.percent_mined:>14.2f}%
    Block Height:      {s.block_height:>15,}
    To Halving:        {s.blocks_until_halving:>15,} blocks
    Block Reward:      {s.current_block_reward:>15.4f} BTC
================================================================================
  MARKET CAP:
    Market Cap:        ${s.market_cap:>14,.0f}
    Realized Cap:      ${s.realized_cap:>14,.0f}
================================================================================
  VOLUME:
    1h Volume:         {s.volume_1h:>15,.2f} BTC
    24h Volume:        {s.volume_24h:>15,.2f} BTC
    7d Volume:         {s.volume_7d:>15,.2f} BTC
    24h TX Count:      {s.tx_count_24h:>15,}
    Whale Volume:      {s.whale_volume_1h:>15,.2f} BTC
    Trend:             {s.volume_trend:>15}
================================================================================
  VALUATION:
    MVRV Ratio:        {s.mvrv_ratio:>15.3f}
    NVT Ratio:         {s.nvt_ratio:>15.2f}
    NUPL:              {s.nupl:>15.4f}
    Stock-to-Flow:     {s.stock_to_flow:>15.2f}
    Puell Multiple:    {s.puell_multiple:>15.3f}
================================================================================
  PROFIT/LOSS:
    In Profit:         {s.percent_in_profit:>14.1f}%
    In Loss:           {s.percent_in_loss:>14.1f}%
================================================================================
  TRADING SIGNAL:
    Signal:            {s.signal:>+15.3f}
    Strength:          {s.signal_strength:>14.2%}
    Reason:            {s.signal_reason}
    Market Regime:     {s.market_regime}
================================================================================
  RAW INPUTS:
    Fee Rate:          {s.fee_rate:>15.1f} sat/vB
    Mempool Size:      {s.mempool_size:>15,}
    Active Addresses:  {s.active_addresses:>15,}
    Confidence:        {s.overall_confidence:>14.2%}
================================================================================
"""


# ==============================================================================
# TEST / DEMO
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BITCOIN DATA PIPELINE - Unified On-Chain Data Orchestrator")
    print("=" * 70)
    print()

    # Initialize pipeline
    pipeline = BitcoinDataPipeline(anchor_price=95000.0)

    # Simulate processing blocks
    print("Processing simulated blockchain data...")
    print()

    import random

    for block in range(870_000, 870_005):
        # Generate fake transactions
        transactions = []
        for _ in range(random.randint(1500, 3000)):
            if random.random() < 0.02:  # 2% whales
                value = random.uniform(100, 500)
            elif random.random() < 0.3:  # 30% retail
                value = random.uniform(0.001, 0.1)
            else:
                value = random.uniform(0.1, 10)

            transactions.append({
                'value': value,
                'fee': random.randint(1000, 30000),
            })

        # Process block
        state = pipeline.process_block(
            block_height=block,
            transactions=transactions,
            fee_rate=random.uniform(5, 25),
            mempool_size=random.randint(20000, 80000)
        )

        print(f"Block {block}:")
        print(f"  Price: ${state.price:,.2f}")
        print(f"  Market Cap: ${state.market_cap:,.0f}")
        print(f"  Signal: {state.signal:+.3f} ({state.signal_reason})")
        print()

    # Show full state
    print(format_pipeline_state(state))

    # Show formula inputs
    print("\nFormula Inputs:")
    inputs = pipeline.get_formula_inputs()
    for key, value in inputs.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("PURE BLOCKCHAIN MATHEMATICS - NO EXTERNAL APIS")
    print("=" * 70)
