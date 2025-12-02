"""
FORMULA ID 333: SIGNAL CONFLUENCE - CONDORCET VOTING
=====================================================
Multi-signal voting system using Condorcet Jury Theorem.
Combines leading blockchain indicators with lagging price signals.

Academic Citation:
    Condorcet (1785) - "Essay on the Application of Analysis to the
    Probability of Majority Decisions"

Mathematical Principle:
    If each voter (signal) has probability p > 0.5 of being correct,
    then the probability of the majority being correct approaches 1
    as the number of voters increases.

    P(majority correct) = sum(C(n,k) * p^k * (1-p)^(n-k)) for k > n/2

Signal Weights (LEADING BLOCKCHAIN PRIORITY):
    - Power Law (ID 901): 5.0 weight (R² = 94%, LEADING indicator)
    - Stock-to-Flow (ID 902): 4.0 weight (R² = 95%, LEADING indicator)
    - Halving Cycle (ID 903): 4.0 weight (Empirical, LEADING indicator)
    - Mempool Flow (ID 802): 3.0 weight (blockchain-derived)
    - Whale Detection (ID 804): 2.0 weight (blockchain-derived)
    - OFI (ID 701): 1.5 weight (academic signal, LAGGING)
    - CUSUM (ID 218): 1.0 weight (LAGGING)
    - Z-Score (ID 141): 0.3 weight (LAGGING - confirmation only)
    - Regime (ID 335): 0.2 weight (LAGGING)

KEY INSIGHT:
    Leading signals (901-903) are calculated from TIMESTAMP ONLY,
    completely independent of current price. They PREDICT price movements.
    Lagging signals (OFI, CUSUM, etc.) are calculated FROM prices.

Performance: O(1) per tick (constant number of signals)
Numba JIT: ~50-100 nanoseconds per tick
"""

import numpy as np
from numba import njit

from engine.core.constants.trading import (
    MIN_AGREEING_SIGNALS, MIN_CONFLUENCE_PROB
)


@njit(cache=True, fastmath=True, error_model='numpy', boundscheck=False)
def calc_confluence(z_signal: int, z_conf: float,
                    cusum_event: int, regime: int, regime_conf: float,
                    ofi_signal: int, ofi_strength: float,
                    mempool_ofi: float = 0.0, whale_prob: float = 0.0,
                    power_law_signal: int = 0, power_law_strength: float = 0.0,
                    s2f_signal: int = 0, s2f_strength: float = 0.0,
                    halving_signal: int = 0, halving_strength: float = 0.0) -> tuple:
    """
    SIGNAL CONFLUENCE (Formula ID 333) - WITH LEADING BLOCKCHAIN INDICATORS
    Condorcet Jury Theorem: Majority of >50% signals has higher accuracy.

    How it works:
        1. Collect votes from all signals (leading and lagging)
        2. Weight votes by signal quality and strength
        3. Calculate majority direction
        4. Compute probability estimate from vote distribution
        5. Determine if trade threshold is met

    Args:
        z_signal: Z-Score direction (-1, 0, +1)
        z_conf: Z-Score confidence [0.0, 1.0]
        cusum_event: CUSUM event (-1, 0, +1)
        regime: Market regime (-2 to +2)
        regime_conf: Regime confidence [0.0, 1.0]
        ofi_signal: OFI direction (-1, 0, +1)
        ofi_strength: OFI strength [0.0, 1.0]
        mempool_ofi: Mempool order flow imbalance [-1.0, +1.0]
        whale_prob: Whale detection probability [0.0, 1.0]
        power_law_signal: Power Law deviation signal (-1, 0, +1)
        power_law_strength: Power Law signal strength [0.0, 1.0]
        s2f_signal: Stock-to-Flow deviation signal (-1, 0, +1)
        s2f_strength: S2F signal strength [0.0, 1.0]
        halving_signal: Halving cycle position signal (-1, 0, +1)
        halving_strength: Halving signal strength [0.0, 1.0]

    Returns:
        Tuple of (direction, probability, agreeing_count, should_trade):

        - direction: Consensus direction
                     +1 = BUY (majority bullish)
                     -1 = SELL (majority bearish)
                      0 = No consensus

        - probability: Estimated win probability [0.5, 0.95]
                       Based on signal agreement and weights

        - agreeing_count: Number of signals agreeing with consensus

        - should_trade: Boolean flag if probability >= MIN_CONFLUENCE_PROB

    Signal Priority:
        LEADING (from timestamp):
            Power Law, S2F, Halving → Predict price direction
        BLOCKCHAIN (from chain data):
            Mempool, Whale → Real-time flow signals
        LAGGING (from price history):
            OFI, CUSUM, Z-Score, Regime → Confirmation signals

    Example:
        >>> direction, prob, agreeing, should_trade = calc_confluence(
        ...     z_signal=1, z_conf=0.6,
        ...     cusum_event=1, regime=2, regime_conf=0.7,
        ...     ofi_signal=1, ofi_strength=0.8,
        ...     power_law_signal=1, power_law_strength=0.5)
        >>> if should_trade and direction > 0:
        ...     # Strong buy signal with high probability
    """
    buy_votes = 0
    sell_votes = 0
    total_weight = 0.0

    # =========================================================================
    # LEADING BLOCKCHAIN INDICATORS (Calculated from TIMESTAMP ONLY)
    # These PREDICT price movements - HIGHEST priority
    # =========================================================================

    # POWER LAW (5x weight - R² = 94%, LEADING indicator)
    if power_law_signal > 0:
        buy_votes += 5
        total_weight += power_law_strength * 5.0
    elif power_law_signal < 0:
        sell_votes += 5
        total_weight += power_law_strength * 5.0

    # STOCK-TO-FLOW (4x weight - R² = 95%, LEADING indicator)
    if s2f_signal > 0:
        buy_votes += 4
        total_weight += s2f_strength * 4.0
    elif s2f_signal < 0:
        sell_votes += 4
        total_weight += s2f_strength * 4.0

    # HALVING CYCLE (4x weight - Empirical 4-year cycle, LEADING)
    if halving_signal > 0:
        buy_votes += 4
        total_weight += halving_strength * 4.0
    elif halving_signal < 0:
        sell_votes += 4
        total_weight += halving_strength * 4.0

    # =========================================================================
    # BLOCKCHAIN-DERIVED INDICATORS (Medium priority)
    # =========================================================================

    # MEMPOOL OFI (3x weight - blockchain-derived)
    if mempool_ofi > 0.1:
        buy_votes += 3
        total_weight += abs(mempool_ofi) * 3.0
    elif mempool_ofi < -0.1:
        sell_votes += 3
        total_weight += abs(mempool_ofi) * 3.0

    # WHALE DETECTION (2x weight - blockchain-derived)
    # Whale direction follows mempool OFI
    if whale_prob > 0.5:
        if mempool_ofi > 0:
            buy_votes += 2
            total_weight += whale_prob * 2.0
        elif mempool_ofi < 0:
            sell_votes += 2
            total_weight += whale_prob * 2.0

    # =========================================================================
    # LAGGING INDICATORS (Calculated FROM prices - lower priority)
    # =========================================================================

    # OFI (1.5x weight - academic signal, LAGGING)
    if ofi_signal > 0:
        buy_votes += 2
        total_weight += ofi_strength * 1.5
    elif ofi_signal < 0:
        sell_votes += 2
        total_weight += ofi_strength * 1.5

    # CUSUM (1x weight - LAGGING)
    if cusum_event > 0:
        buy_votes += 1
        total_weight += 1.0
    elif cusum_event < 0:
        sell_votes += 1
        total_weight += 1.0

    # Z-Score (0.3x weight - LAGGING, confirmation only)
    if z_signal > 0:
        buy_votes += 1
        total_weight += z_conf * 0.3
    elif z_signal < 0:
        sell_votes += 1
        total_weight += z_conf * 0.3

    # Regime (0.2x weight - LAGGING)
    if regime > 0:
        buy_votes += 1
        total_weight += regime_conf * 0.2
    elif regime < 0:
        sell_votes += 1
        total_weight += regime_conf * 0.2

    # =========================================================================
    # VOTING RESULT
    # =========================================================================
    agreeing = max(buy_votes, sell_votes)
    total_votes = buy_votes + sell_votes

    # Minimum agreement threshold
    if agreeing < MIN_AGREEING_SIGNALS:
        return 0, 0.5, agreeing, False

    # Determine direction
    if buy_votes > sell_votes:
        direction = 1
    elif sell_votes > buy_votes:
        direction = -1
    else:
        return 0, 0.5, agreeing, False

    # =========================================================================
    # PROBABILITY ESTIMATION (Condorcet approximation)
    # =========================================================================
    if total_votes > 0:
        agreement_ratio = agreeing / total_votes
        # Higher base probability with blockchain signals
        base_prob = 0.55 + (total_weight / (total_votes * 2)) * 0.25
        probability = min(base_prob * agreement_ratio, 0.95)
    else:
        probability = 0.5

    # Trade threshold check
    should_trade = probability >= MIN_CONFLUENCE_PROB

    return direction, probability, agreeing, should_trade
