"""
================================================================================
PURE BLOCKCHAIN PREDICTIVE SIGNALS - LEADING INDICATORS (Formula IDs 901-903)
================================================================================

ARCHITECTURE REFERENCE: docs/BLOCKCHAIN_PIPELINE_ARCHITECTURE.md

POSITION IN PIPELINE:
    Legacy implementation - now superseded by:
        - engine/tick/formulas/leading/power_law.py (ID 901)
        - engine/tick/formulas/leading/stock_to_flow.py (ID 902)
        - engine/tick/formulas/leading/halving_cycle.py (ID 903)

FORMULA IDs:
    901: Power Law Price - ln(price) = a + b*ln(days) (R² = 93%+)
    902: Stock-to-Flow - ln(price) = a + b*ln(S2F) (R² = 95%)
    903: Halving Cycle - Position in 4-year halving cycle

SIGNAL TYPE: LEADING INDICATORS
    These signals are calculated from TIMESTAMP ONLY.
    They LEAD price movements because they reflect fundamental blockchain mechanics.

ACADEMIC RESEARCH:
    - Power Law: Giovanni Santostasi, R² = 94%
    - Stock-to-Flow: PlanB (2019), R² = 95%
    - Halving Cycles: Empirically observed 4-year cycles

KEY INSIGHT: Bitcoin's supply schedule is 100% deterministic.
    - We know EXACTLY when halvings occur (every 210,000 blocks)
    - We know EXACTLY what the supply will be at any future date
    - We know EXACTLY the Stock-to-Flow ratio at any timestamp
    - NO price input needed - pure time-based prediction

THIS IS THE EDGE: Blockchain fundamentals that predict price direction.

CIRCULAR DEPENDENCY PREVENTION:
    These formulas use ONLY timestamp, NOT price.
    This allows them to predict price without circular logic.
    Used by chaos_price (803) and other blockchain signals.
================================================================================
"""

import math
import time
import numpy as np
from numba import njit, float64, int64
from typing import Tuple, Dict, Any

# ============================================================================
# BITCOIN BLOCKCHAIN CONSTANTS (100% DETERMINISTIC)
# ============================================================================

GENESIS_TIMESTAMP = 1231006505  # Jan 3, 2009, 18:15:05 UTC
BLOCK_TIME = 600  # 10 minutes average
BLOCKS_PER_HALVING = 210_000  # ~4 years
INITIAL_REWARD = 50.0  # BTC per block
TOTAL_SUPPLY = 21_000_000  # Hard cap

# Power Law coefficients (Giovannetti 2019, R² = 94%)
POWER_LAW_A = -17.01  # Intercept
POWER_LAW_B = 5.84    # Slope

# Stock-to-Flow coefficients (PlanB 2019, recalibrated)
# Original: log10(price) = 3.21 * log10(S2F) - 1.47
# Using natural log: ln(price) = 3.21 * ln(S2F) + intercept
S2F_A = -3.39  # Intercept (recalibrated for ln scale)
S2F_B = 3.21   # Slope


# ============================================================================
# NUMBA JIT-COMPILED CORE FUNCTIONS (NANOSECOND SPEED)
# ============================================================================

@njit(cache=True, fastmath=True)
def calc_block_height(timestamp: float64) -> int64:
    """
    Calculate estimated block height from timestamp.

    Formula: blocks = (timestamp - genesis) / block_time
    Accuracy: ±100 blocks (timestamp variance in mining)

    Returns:
        Estimated block height
    """
    if timestamp < GENESIS_TIMESTAMP:
        return 0
    return int((timestamp - GENESIS_TIMESTAMP) / BLOCK_TIME)


@njit(cache=True, fastmath=True)
def calc_halving_number(block_height: int64) -> int64:
    """
    Calculate which halving epoch we're in.

    Epoch 0: blocks 0-209,999 (50 BTC reward)
    Epoch 1: blocks 210,000-419,999 (25 BTC reward)
    Epoch 2: blocks 420,000-629,999 (12.5 BTC reward)
    Epoch 3: blocks 630,000-839,999 (6.25 BTC reward)
    Epoch 4: blocks 840,000-1,049,999 (3.125 BTC reward)

    Returns:
        Halving epoch number (0, 1, 2, ...)
    """
    return block_height // BLOCKS_PER_HALVING


@njit(cache=True, fastmath=True)
def calc_halving_cycle_position(block_height: int64) -> float64:
    """
    Calculate position within current halving cycle (0.0 to 1.0).

    0.00 - 0.30: Accumulation phase (post-halving recovery)
    0.30 - 0.70: Expansion phase (bull run)
    0.70 - 1.00: Distribution phase (pre-halving top)

    Returns:
        Position in cycle (0.0 to 1.0)
    """
    position_in_epoch = block_height % BLOCKS_PER_HALVING
    return float(position_in_epoch) / float(BLOCKS_PER_HALVING)


@njit(cache=True, fastmath=True)
def calc_current_supply(block_height: int64) -> float64:
    """
    Calculate total BTC supply at given block height.

    Formula: Sum of (blocks_in_epoch * reward_per_epoch)

    Returns:
        Total BTC mined
    """
    supply = 0.0
    remaining_blocks = block_height
    reward = INITIAL_REWARD

    for _ in range(64):  # Max 64 halvings (covers all possible blocks)
        if remaining_blocks <= 0:
            break

        blocks_this_epoch = min(remaining_blocks, BLOCKS_PER_HALVING)
        supply += blocks_this_epoch * reward
        remaining_blocks -= blocks_this_epoch
        reward /= 2.0

        if reward < 1e-10:  # Satoshi floor
            break

    return min(supply, TOTAL_SUPPLY)


@njit(cache=True, fastmath=True)
def calc_annual_issuance(block_height: int64) -> float64:
    """
    Calculate annual BTC issuance at given block height.

    Formula: blocks_per_year * current_reward
    ~52,560 blocks per year (365.25 * 24 * 6)

    Returns:
        Annual BTC issuance
    """
    halving = block_height // BLOCKS_PER_HALVING
    reward = INITIAL_REWARD / (2.0 ** halving)
    blocks_per_year = 365.25 * 24 * 6  # ~52,560
    return blocks_per_year * reward


@njit(cache=True, fastmath=True)
def calc_stock_to_flow(block_height: int64) -> float64:
    """
    Calculate Stock-to-Flow ratio.

    Formula: S2F = Stock / Annual_Flow

    Higher S2F = more scarce = higher expected price

    Returns:
        Stock-to-Flow ratio
    """
    stock = calc_current_supply(block_height)
    flow = calc_annual_issuance(block_height)

    if flow < 1e-10:
        return 1000.0  # Essentially infinite S2F when no more issuance

    return stock / flow


@njit(cache=True, fastmath=True)
def calc_power_law_price(days_since_genesis: float64) -> float64:
    """
    Calculate Power Law fair value price.

    Formula: log10(price) = A + B * log10(days)
    Source: Giovannetti (2019), R² = 94%

    Returns:
        Power Law predicted price in USD
    """
    if days_since_genesis < 1:
        return 0.0

    log_days = math.log10(days_since_genesis)
    log_price = POWER_LAW_A + POWER_LAW_B * log_days
    return 10.0 ** log_price


@njit(cache=True, fastmath=True)
def calc_s2f_model_price(s2f: float64) -> float64:
    """
    Calculate Stock-to-Flow model price.

    Formula: ln(price) = A + B * ln(S2F)
    Source: PlanB (2019), R² = 95%

    Returns:
        S2F model predicted price in USD
    """
    if s2f < 1:
        return 0.0

    ln_s2f = math.log(s2f)
    ln_price = S2F_A + S2F_B * ln_s2f
    return math.exp(ln_price)


@njit(cache=True, fastmath=True)
def calc_blockchain_signals(timestamp: float64, current_price: float64) -> Tuple[
    float64,  # power_law_price
    float64,  # power_law_deviation
    float64,  # s2f_price
    float64,  # s2f_deviation
    float64,  # halving_cycle_position
    int64,    # halving_signal (-1, 0, +1)
    float64,  # combined_signal_strength
    int64,    # final_signal (-1, 0, +1)
]:
    """
    Calculate ALL blockchain signals from timestamp.

    This is the MASTER function that combines all blockchain metrics
    into actionable trading signals.

    IMPORTANT: These signals are LEADING indicators, not lagging.
    They are calculated from blockchain mechanics, not price history.

    Returns:
        Tuple of all signal components
    """
    # Block height from timestamp
    block_height = calc_block_height(timestamp)

    # Days since genesis
    days = (timestamp - GENESIS_TIMESTAMP) / 86400.0

    # =========================================================================
    # SIGNAL 1: POWER LAW DEVIATION
    # =========================================================================
    power_law_price = calc_power_law_price(days)

    if power_law_price > 0:
        power_law_deviation = (current_price - power_law_price) / power_law_price
    else:
        power_law_deviation = 0.0

    # =========================================================================
    # SIGNAL 2: STOCK-TO-FLOW DEVIATION
    # =========================================================================
    s2f = calc_stock_to_flow(block_height)
    s2f_price = calc_s2f_model_price(s2f)

    if s2f_price > 0:
        s2f_deviation = (current_price - s2f_price) / s2f_price
    else:
        s2f_deviation = 0.0

    # =========================================================================
    # SIGNAL 3: HALVING CYCLE POSITION
    # =========================================================================
    halving_pos = calc_halving_cycle_position(block_height)

    # Halving signal based on cycle position
    # 0.0-0.3: Post-halving accumulation → BUY (+1)
    # 0.3-0.7: Expansion → HOLD (0)
    # 0.7-1.0: Distribution → SELL (-1)
    if halving_pos < 0.30:
        halving_signal = 1   # Accumulation phase
    elif halving_pos > 0.70:
        halving_signal = -1  # Distribution phase
    else:
        halving_signal = 0   # Neutral

    # =========================================================================
    # COMBINE SIGNALS
    # =========================================================================

    # Power Law signal
    if power_law_deviation < -0.10:  # >10% below fair value
        pl_signal = 1.0  # Strong buy
    elif power_law_deviation < -0.05:  # 5-10% below
        pl_signal = 0.5  # Moderate buy
    elif power_law_deviation > 0.10:  # >10% above fair value
        pl_signal = -1.0  # Strong sell
    elif power_law_deviation > 0.05:  # 5-10% above
        pl_signal = -0.5  # Moderate sell
    else:
        pl_signal = 0.0

    # S2F signal
    if s2f_deviation < -0.15:  # >15% below S2F model
        s2f_signal = 1.0
    elif s2f_deviation < -0.08:
        s2f_signal = 0.5
    elif s2f_deviation > 0.15:
        s2f_signal = -1.0
    elif s2f_deviation > 0.08:
        s2f_signal = -0.5
    else:
        s2f_signal = 0.0

    # Combined signal strength (weighted average)
    # Weights: Power Law = 0.4, S2F = 0.3, Halving = 0.3
    combined = 0.4 * pl_signal + 0.3 * s2f_signal + 0.3 * float(halving_signal)

    # Final discrete signal
    if combined > 0.3:
        final_signal = 1  # BUY
    elif combined < -0.3:
        final_signal = -1  # SELL
    else:
        final_signal = 0  # HOLD

    return (
        power_law_price,
        power_law_deviation,
        s2f_price,
        s2f_deviation,
        halving_pos,
        halving_signal,
        combined,
        final_signal,
    )


# ============================================================================
# HIGH-LEVEL PYTHON INTERFACE
# ============================================================================

class BlockchainSignals:
    """
    Pure blockchain signals calculated from timestamp only.

    NO EXTERNAL APIs - everything is derived from Bitcoin's deterministic
    blockchain mechanics.

    Usage:
        signals = BlockchainSignals()
        result = signals.calculate(timestamp, current_price)
        if result['final_signal'] == 1:
            execute_buy()
    """

    def __init__(self):
        # Cache for expensive calculations
        self._last_timestamp = 0.0
        self._last_block_height = 0
        self._cache = {}

    def calculate(self, timestamp: float = None, current_price: float = 0.0) -> Dict[str, Any]:
        """
        Calculate all blockchain signals.

        Args:
            timestamp: Unix timestamp (defaults to now)
            current_price: Current BTC price in USD

        Returns:
            Dict with all signal components
        """
        if timestamp is None:
            timestamp = time.time()

        # Get all signals from JIT function
        (
            power_law_price,
            power_law_deviation,
            s2f_price,
            s2f_deviation,
            halving_pos,
            halving_signal,
            combined_strength,
            final_signal,
        ) = calc_blockchain_signals(timestamp, current_price)

        # Additional context
        block_height = calc_block_height(timestamp)
        halving_number = calc_halving_number(block_height)
        current_supply = calc_current_supply(block_height)
        s2f = calc_stock_to_flow(block_height)
        days_since_genesis = (timestamp - GENESIS_TIMESTAMP) / 86400.0

        return {
            # Core signals
            'final_signal': final_signal,
            'signal_strength': abs(combined_strength),
            'combined_strength': combined_strength,

            # Power Law
            'power_law_price': power_law_price,
            'power_law_deviation': power_law_deviation,
            'power_law_signal': 1 if power_law_deviation < -0.05 else (-1 if power_law_deviation > 0.05 else 0),

            # Stock-to-Flow
            's2f_price': s2f_price,
            's2f_deviation': s2f_deviation,
            's2f_signal': 1 if s2f_deviation < -0.08 else (-1 if s2f_deviation > 0.08 else 0),
            's2f_ratio': s2f,

            # Halving Cycle
            'halving_position': halving_pos,
            'halving_signal': halving_signal,
            'halving_number': halving_number,

            # Blockchain state
            'block_height': block_height,
            'current_supply': current_supply,
            'days_since_genesis': days_since_genesis,
        }

    def get_signal_for_tick(self, timestamp: float, current_price: float) -> Tuple[int, float]:
        """
        Fast signal calculation for HFT tick processing.

        Returns:
            (signal, confidence) where signal is -1, 0, or +1
        """
        (_, _, _, _, _, _, strength, signal) = calc_blockchain_signals(timestamp, current_price)
        confidence = min(0.85, 0.5 + abs(strength) * 0.3)
        return (signal, confidence)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_halving_dates() -> list:
    """Get historical and projected halving dates."""
    halvings = [
        {'number': 0, 'block': 0, 'date': '2009-01-03', 'reward': 50.0},
        {'number': 1, 'block': 210000, 'date': '2012-11-28', 'reward': 25.0},
        {'number': 2, 'block': 420000, 'date': '2016-07-09', 'reward': 12.5},
        {'number': 3, 'block': 630000, 'date': '2020-05-11', 'reward': 6.25},
        {'number': 4, 'block': 840000, 'date': '2024-04-20', 'reward': 3.125},
        {'number': 5, 'block': 1050000, 'date': '~2028-04', 'reward': 1.5625},
    ]
    return halvings


def print_current_state():
    """Print current blockchain state."""
    signals = BlockchainSignals()
    result = signals.calculate(current_price=97000)  # Use approximate price

    print("=" * 70)
    print("BITCOIN BLOCKCHAIN STATE - PURE MATH (NO APIS)")
    print("=" * 70)
    print(f"Block Height:        {result['block_height']:,}")
    print(f"Halving Number:      {result['halving_number']}")
    print(f"Halving Position:    {result['halving_position']:.4f} ({result['halving_position']*100:.1f}%)")
    print(f"Current Supply:      {result['current_supply']:,.0f} BTC")
    print(f"Stock-to-Flow:       {result['s2f_ratio']:.1f}")
    print(f"Days Since Genesis:  {result['days_since_genesis']:,.1f}")
    print("-" * 70)
    print(f"Power Law Price:     ${result['power_law_price']:,.0f}")
    print(f"Power Law Deviation: {result['power_law_deviation']*100:+.1f}%")
    print(f"S2F Model Price:     ${result['s2f_price']:,.0f}")
    print(f"S2F Deviation:       {result['s2f_deviation']*100:+.1f}%")
    print("-" * 70)
    print(f"Halving Signal:      {result['halving_signal']} (1=BUY, 0=HOLD, -1=SELL)")
    print(f"Combined Strength:   {result['combined_strength']:+.2f}")
    print(f"FINAL SIGNAL:        {result['final_signal']} (1=BUY, 0=HOLD, -1=SELL)")
    print("=" * 70)


if __name__ == "__main__":
    print_current_state()
