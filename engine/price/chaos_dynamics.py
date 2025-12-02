"""
CHAOS DYNAMICS - INDEPENDENT PRICE GENERATION
==============================================
RENAISSANCE TECHNOLOGIES APPROACH: Price generation uses DIFFERENT
blockchain data than signal generation to break circular dependency.

PRICE GENERATION USES:
- Difficulty cycles (14-day, 2016 blocks)
- Halving cycles (4-year, 210k blocks)
- Supply schedule
- Lorenz attractor for chaotic dynamics

SIGNAL GENERATION USES (SEPARATE):
- Historical price patterns only
- OFI, CUSUM, Regime detection
- Does NOT know the price formula

Citation: Lorenz (1963) - Deterministic Nonperiodic Flow
"""
import numpy as np
from numba import njit

# Lorenz attractor parameters (classic values)
LORENZ_SIGMA = 10.0
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0

# Blockchain timing constants
GENESIS_TS = 1230768000.0
SECONDS_PER_BLOCK = 600.0
BLOCKS_PER_HALVING = 210000
BLOCKS_PER_DIFFICULTY = 2016

# Power Law constants
POWER_LAW_A = -17.0161223
POWER_LAW_B = 5.8451542


@njit(cache=True, fastmath=True)
def lorenz_step(x: float, y: float, z: float, dt: float) -> tuple:
    """
    Single step of Lorenz attractor.
    Deterministic chaos - sensitive to initial conditions.

    Returns: (new_x, new_y, new_z)
    """
    dx = LORENZ_SIGMA * (y - x)
    dy = x * (LORENZ_RHO - z) - y
    dz = x * y - LORENZ_BETA * z

    return x + dx * dt, y + dy * dt, z + dz * dt


@njit(cache=True, fastmath=True)
def get_difficulty_seed(timestamp: float) -> tuple:
    """
    Get seed values from difficulty cycle position.
    This is DIFFERENT from block timing used in signals.

    Difficulty adjusts every 2016 blocks (~14 days).
    Use this for price chaos, NOT for signal generation.

    Returns: (diff_progress, diff_phase, difficulty_factor)
    """
    seconds_since_genesis = timestamp - GENESIS_TS
    block_height = int(seconds_since_genesis / SECONDS_PER_BLOCK)

    # Position in difficulty cycle (0.0 to 1.0)
    diff_progress = (block_height % BLOCKS_PER_DIFFICULTY) / float(BLOCKS_PER_DIFFICULTY)

    # Phase angle for chaos seeding
    diff_phase = diff_progress * 6.283185307179586  # 2*pi

    # Difficulty factor (simulated - would be real difficulty in live)
    # Models exponential growth with periodic adjustments
    days = seconds_since_genesis / 86400.0
    base_difficulty = 1e12 * (days / 1000.0) ** 4

    # Adjustment oscillation (difficulty adjusts up/down around target)
    adjustment = 1.0 + 0.15 * np.sin(diff_phase)
    difficulty_factor = base_difficulty * adjustment

    return diff_progress, diff_phase, difficulty_factor


@njit(cache=True, fastmath=True)
def get_halving_seed(timestamp: float) -> tuple:
    """
    Get seed values from halving cycle position.
    4-year accumulation/distribution cycles.

    Returns: (halving_progress, halving_phase, halving_number)
    """
    seconds_since_genesis = timestamp - GENESIS_TS
    block_height = int(seconds_since_genesis / SECONDS_PER_BLOCK)

    # Current halving era
    halving_number = block_height // BLOCKS_PER_HALVING

    # Position in current halving cycle (0.0 to 1.0)
    halving_progress = (block_height % BLOCKS_PER_HALVING) / float(BLOCKS_PER_HALVING)

    # Phase for chaos seeding
    halving_phase = halving_progress * 6.283185307179586

    return halving_progress, halving_phase, halving_number


@njit(cache=True, fastmath=True)
def get_supply_seed(timestamp: float) -> tuple:
    """
    Get seed values from supply schedule.
    Scarcity increases over time.

    Returns: (supply_ratio, scarcity_factor, s2f)
    """
    seconds_since_genesis = timestamp - GENESIS_TS
    block_height = int(seconds_since_genesis / SECONDS_PER_BLOCK)

    # Calculate total supply
    supply = 0.0
    remaining = block_height
    reward = 50.0

    while remaining > 0:
        blocks = min(remaining, BLOCKS_PER_HALVING)
        supply += blocks * reward
        remaining -= blocks
        reward /= 2.0

    MAX_SUPPLY = 21000000.0
    supply_ratio = supply / MAX_SUPPLY

    # Stock-to-Flow
    current_reward = 50.0 / (2.0 ** (block_height // BLOCKS_PER_HALVING))
    annual_production = current_reward * 6 * 24 * 365
    s2f = supply / annual_production if annual_production > 0 else 1000.0

    # Scarcity factor (increases as supply approaches max)
    scarcity_factor = 1.0 / (1.0 + np.log(MAX_SUPPLY / max(supply, 1.0)))

    return supply_ratio, scarcity_factor, s2f


@njit(cache=True, fastmath=True)
def generate_chaos_price(timestamp: float,
                         lorenz_state: np.ndarray,
                         volatility_state: np.ndarray) -> tuple:
    """
    MAIN PRICE GENERATOR - BREAKS CIRCULAR DEPENDENCY

    Uses:
    1. Power Law for true price (long-term anchor)
    2. Difficulty cycles for chaos seeding (NOT block timing)
    3. Halving cycles for volatility regime
    4. Lorenz attractor for chaotic dynamics
    5. GARCH-like volatility clustering

    CRITICAL: This function uses DIFFERENT blockchain data
    than calc_blockchain_signals() in processor.py.

    Args:
        timestamp: Current time
        lorenz_state: [x, y, z] Lorenz coordinates
        volatility_state: [current_vol, vol_ema, last_return]

    Returns: (market_price, true_price, volatility, lorenz_x, momentum)
    """
    # =========================================================================
    # STEP 1: TRUE PRICE FROM POWER LAW (Long-term anchor)
    # =========================================================================
    days_since_genesis = (timestamp - GENESIS_TS) / 86400.0
    log10_days = np.log10(max(days_since_genesis, 1.0))
    true_price = 10.0 ** (POWER_LAW_A + POWER_LAW_B * log10_days)

    # =========================================================================
    # STEP 2: GET BLOCKCHAIN SEEDS (INDEPENDENT of signal generation)
    # =========================================================================
    diff_progress, diff_phase, difficulty_factor = get_difficulty_seed(timestamp)
    halving_progress, halving_phase, halving_number = get_halving_seed(timestamp)
    supply_ratio, scarcity_factor, s2f = get_supply_seed(timestamp)

    # =========================================================================
    # STEP 3: LORENZ ATTRACTOR EVOLUTION (Chaotic dynamics)
    # =========================================================================
    # Seed Lorenz with difficulty cycle (NOT block timing)
    x = lorenz_state[0]
    y = lorenz_state[1]
    z = lorenz_state[2]

    # Time step scaled by difficulty progress
    dt = 0.01 * (1.0 + 0.5 * np.sin(diff_phase))

    # Evolve Lorenz system
    x, y, z = lorenz_step(x, y, z, dt)

    # Store new state
    lorenz_state[0] = x
    lorenz_state[1] = y
    lorenz_state[2] = z

    # Normalize Lorenz output to [-1, 1]
    # Typical range: x in [-20, 20], y in [-30, 30], z in [0, 50]
    chaos_factor = np.tanh(x / 20.0)

    # =========================================================================
    # STEP 4: VOLATILITY REGIME (Halving cycle based)
    # =========================================================================
    # Base volatility varies with halving cycle
    # High volatility: 0-20% and 80-100% of cycle (accumulation/distribution)
    # Low volatility: 20-80% of cycle (consolidation)

    if halving_progress < 0.2:
        # Post-halving accumulation - high volatility
        base_vol = 0.0004 * (1.0 + halving_progress * 2.0)
    elif halving_progress > 0.8:
        # Pre-halving distribution - very high volatility
        base_vol = 0.0006 * (1.0 + (halving_progress - 0.8) * 3.0)
    else:
        # Consolidation - low volatility
        base_vol = 0.0002 * (1.0 + 0.2 * np.sin(halving_phase * 2.0))

    # =========================================================================
    # STEP 5: GARCH-LIKE VOLATILITY CLUSTERING
    # =========================================================================
    current_vol = volatility_state[0]
    vol_ema = volatility_state[1]
    last_return = volatility_state[2]

    # GARCH(1,1) style: vol = omega + alpha*return^2 + beta*vol
    omega = base_vol * 0.1
    alpha = 0.1
    beta = 0.85

    new_vol = omega + alpha * (last_return ** 2) + beta * current_vol
    new_vol = max(0.0001, min(0.01, new_vol))  # Clamp volatility

    # EMA smoothing
    vol_alpha = 0.05
    new_vol_ema = vol_alpha * new_vol + (1.0 - vol_alpha) * vol_ema

    volatility_state[0] = new_vol
    volatility_state[1] = new_vol_ema

    # =========================================================================
    # STEP 6: PRICE MOVEMENT WITH FAT TAILS
    # =========================================================================
    # Use difficulty cycle for momentum direction
    # This creates PREDICTABLE structure that signals can learn
    momentum = 0.3 * np.sin(diff_phase) + 0.2 * np.cos(halving_phase * 0.5)

    # Chaos adds unpredictability
    price_delta = new_vol_ema * (momentum + 0.5 * chaos_factor)

    # Fat tails: occasional large moves (supply shocks)
    if abs(chaos_factor) > 0.8:
        # Large move triggered by Lorenz attractor extremes
        price_delta *= (1.0 + abs(chaos_factor))

    # =========================================================================
    # STEP 7: FINAL MARKET PRICE
    # =========================================================================
    # Mean reversion to true price
    mean_reversion_strength = 0.001 * scarcity_factor
    deviation = (true_price - true_price * (1.0 + price_delta)) / true_price
    mean_reversion = mean_reversion_strength * deviation

    market_price = true_price * (1.0 + price_delta + mean_reversion)

    # Store return for next iteration
    if true_price > 0:
        volatility_state[2] = price_delta

    return market_price, true_price, new_vol_ema, chaos_factor, momentum


@njit(cache=True, fastmath=True)
def init_chaos_state() -> tuple:
    """
    Initialize Lorenz and volatility states.

    Returns: (lorenz_state, volatility_state)
    """
    # Initial Lorenz position (near attractor)
    lorenz_state = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    # Initial volatility state [current_vol, vol_ema, last_return]
    volatility_state = np.array([0.0003, 0.0003, 0.0], dtype=np.float64)

    return lorenz_state, volatility_state
