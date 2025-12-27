"""
Math Primitives - Timeframe-Adaptive Mathematical Engine
=========================================================

Custom-derived formulas from first principles for cryptocurrency trading.
No off-the-shelf formulas - pure mathematical treatment of the market.

Formulas:
- TAE-001: Timeframe Validity Score
- TAE-002: Mutual Information for Signal Quality
- TAE-003: Parameter Decay (Ornstein-Uhlenbeck)
- TAE-004: Multi-Scale Consensus
- TAE-005: Edge Half-Life Estimation
- TAE-006: Uncertain Kelly Sizing
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


# =============================================================================
# TAE-001: Timeframe Validity Score
# =============================================================================

def tae_001_timeframe_validity(
    candidate_tau: float,
    optimal_tau: float,
    decay_lambda: float = 0.1
) -> float:
    """
    TAE-001: Timeframe Validity Score

    TVS(t, τ) = exp(-λ(t) × |τ - τ*(t)|)

    Measures how valid a candidate timeframe is relative to the optimal.

    Args:
        candidate_tau: Candidate timeframe in seconds (1, 5, 10, etc.)
        optimal_tau: Current optimal timeframe (learned from regime)
        decay_lambda: Decay rate from HMM regime (higher = sharper penalty)

    Returns:
        Validity score between 0 and 1
        - 1.0 = perfect match (candidate == optimal)
        - Decays exponentially as candidate diverges from optimal
    """
    distance = abs(candidate_tau - optimal_tau)
    return math.exp(-decay_lambda * distance)


def tae_001_batch_validity(
    candidate_taus: List[float],
    optimal_tau: float,
    decay_lambda: float = 0.1
) -> np.ndarray:
    """Batch computation of timeframe validity scores."""
    candidates = np.array(candidate_taus)
    distances = np.abs(candidates - optimal_tau)
    return np.exp(-decay_lambda * distances)


# =============================================================================
# TAE-002: Mutual Information for Signal Quality
# =============================================================================

def tae_002_mutual_information(
    signal_values: np.ndarray,
    returns: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    TAE-002: Mutual Information for Signal Quality

    I(Signal; Return) = H(Return) - H(Return | Signal)

    Measures predictive power of signal for returns.
    Higher I() = signal contains more information about future returns.

    Args:
        signal_values: Array of signal values
        returns: Array of corresponding returns
        n_bins: Number of bins for discretization

    Returns:
        Mutual information in nats (natural log base)
    """
    if len(signal_values) != len(returns) or len(signal_values) < 10:
        return 0.0

    # Discretize continuous variables
    signal_bins = np.digitize(
        signal_values,
        bins=np.percentile(signal_values, np.linspace(0, 100, n_bins + 1)[1:-1])
    )
    return_bins = np.digitize(
        returns,
        bins=np.percentile(returns, np.linspace(0, 100, n_bins + 1)[1:-1])
    )

    # Joint probability P(signal, return)
    joint_counts = np.zeros((n_bins, n_bins))
    for s, r in zip(signal_bins, return_bins):
        joint_counts[s - 1, r - 1] += 1

    joint_probs = joint_counts / joint_counts.sum()

    # Marginal probabilities
    p_signal = joint_probs.sum(axis=1)
    p_return = joint_probs.sum(axis=0)

    # Mutual information: I(X;Y) = ΣΣ p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint_probs[i, j] > 1e-10 and p_signal[i] > 1e-10 and p_return[j] > 1e-10:
                mi += joint_probs[i, j] * math.log(
                    joint_probs[i, j] / (p_signal[i] * p_return[j])
                )

    return max(0.0, mi)  # MI is always non-negative


def tae_002_shannon_entropy(values: np.ndarray, n_bins: int = 10) -> float:
    """
    Shannon Entropy: H(X) = -Σ p(x) log(p(x))

    Measures uncertainty/randomness in a signal.
    Higher entropy = more random = less predictable.
    """
    if len(values) < 2:
        return 0.0

    # Discretize
    bins = np.digitize(
        values,
        bins=np.percentile(values, np.linspace(0, 100, n_bins + 1)[1:-1])
    )

    # Count probabilities
    counts = np.bincount(bins, minlength=n_bins + 1)[1:]  # Skip 0 bin
    probs = counts / counts.sum()

    # Entropy
    entropy = 0.0
    for p in probs:
        if p > 1e-10:
            entropy -= p * math.log(p)

    return entropy


# =============================================================================
# TAE-003: Parameter Decay (Ornstein-Uhlenbeck Process)
# =============================================================================

@dataclass
class OUState:
    """State for Ornstein-Uhlenbeck process."""
    current: float
    prior_mean: float
    decay_rate: float  # kappa
    volatility: float  # sigma


def tae_003_ou_decay(
    current_theta: float,
    prior_theta: float,
    decay_kappa: float,
    dt: float = 1.0,
    volatility_sigma: float = 0.01,
    random_shock: Optional[float] = None
) -> float:
    """
    TAE-003: Parameter Decay (Ornstein-Uhlenbeck)

    dθ = κ × (θ̄ - θ) × dt + σ × dW

    Models parameter mean-reversion toward historical optimal.
    When evidence is weak, parameters decay toward priors.

    Args:
        current_theta: Current parameter value
        prior_theta: Historical optimal (prior mean)
        decay_kappa: Regime-dependent decay speed (0.01-0.20)
        dt: Time step (in appropriate units)
        volatility_sigma: Random walk volatility
        random_shock: Optional random shock (dW), uses random if None

    Returns:
        Updated parameter value
    """
    # Deterministic mean-reversion
    drift = decay_kappa * (prior_theta - current_theta) * dt

    # Stochastic component
    if random_shock is None:
        random_shock = np.random.normal(0, 1)
    diffusion = volatility_sigma * math.sqrt(dt) * random_shock

    return current_theta + drift + diffusion


def tae_003_batch_ou_decay(
    states: Dict[str, OUState],
    dt: float = 1.0
) -> Dict[str, float]:
    """
    Batch update of multiple parameters using OU process.

    Args:
        states: Dict of parameter name -> OUState
        dt: Time step

    Returns:
        Dict of parameter name -> updated value
    """
    updated = {}
    for name, state in states.items():
        updated[name] = tae_003_ou_decay(
            current_theta=state.current,
            prior_theta=state.prior_mean,
            decay_kappa=state.decay_rate,
            dt=dt,
            volatility_sigma=state.volatility
        )
    return updated


# Regime-specific decay rates
REGIME_DECAY_KAPPA = {
    'accumulation': 0.01,    # Slow decay - ride the trend
    'distribution': 0.01,    # Slow decay - ride the trend
    'trending_up': 0.05,     # Medium decay - adapt faster
    'trending_down': 0.05,   # Medium decay - adapt faster
    'consolidation': 0.10,   # Faster decay - less confidence
    'capitulation': 0.20,    # Fast decay - high uncertainty
    'euphoria': 0.20,        # Fast decay - high uncertainty
    'unknown': 0.10,         # Default
}


def get_decay_rate_for_regime(regime: str) -> float:
    """Get appropriate decay rate kappa for a regime."""
    return REGIME_DECAY_KAPPA.get(regime.lower(), 0.10)


# =============================================================================
# TAE-004: Multi-Scale Consensus
# =============================================================================

def tae_004_consensus(directions: np.ndarray) -> float:
    """
    TAE-004: Multi-Scale Consensus

    Consensus = |mean(directions across wavelet scales)|

    Measures agreement across multiple timeframe/wavelet scales.

    Args:
        directions: Array of direction signals from different scales
                   (+1 = long, -1 = short, 0 = neutral)

    Returns:
        Consensus score between 0 and 1
        - 1.0 = all scales agree perfectly
        - 0.0 = scales conflict (half long, half short)
    """
    if len(directions) == 0:
        return 0.0

    return abs(np.mean(directions))


def tae_004_weighted_consensus(
    directions: np.ndarray,
    weights: np.ndarray
) -> Tuple[float, int]:
    """
    Weighted multi-scale consensus.

    Args:
        directions: Array of directions from different scales
        weights: Weights for each scale (e.g., from entropy/MI)

    Returns:
        Tuple of (consensus score, weighted direction)
    """
    if len(directions) == 0 or len(weights) == 0:
        return 0.0, 0

    # Normalize weights
    weights = np.array(weights)
    if weights.sum() < 1e-10:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    # Weighted mean
    weighted_mean = np.sum(directions * weights)

    # Consensus is absolute value of weighted mean
    consensus = abs(weighted_mean)

    # Direction is sign of weighted mean
    direction = int(np.sign(weighted_mean)) if abs(weighted_mean) > 0.1 else 0

    return consensus, direction


# =============================================================================
# TAE-005: Edge Half-Life Estimation
# =============================================================================

def tae_005_edge_halflife(
    pnl_series: np.ndarray,
    time_series: np.ndarray
) -> Tuple[float, float]:
    """
    TAE-005: Edge Half-Life Estimation

    E[PnL | t] = E₀ × exp(-λ × t)
    Half-life = ln(2) / λ

    Estimates how quickly a trading edge decays over time.

    Args:
        pnl_series: Array of PnL values
        time_series: Array of times since strategy start

    Returns:
        Tuple of (half_life, decay_rate lambda)
        - half_life: Time for edge to decay to 50%
        - decay_rate: Lambda in exponential decay
    """
    if len(pnl_series) < 10:
        return float('inf'), 0.0  # Not enough data

    # Remove negative PnL (focus on positive edge decay)
    positive_mask = pnl_series > 0
    if positive_mask.sum() < 5:
        return float('inf'), 0.0

    pnl_pos = pnl_series[positive_mask]
    time_pos = time_series[positive_mask]

    # Take log for linear regression: log(PnL) = log(E0) - λt
    log_pnl = np.log(pnl_pos + 1e-10)

    # Linear regression
    n = len(log_pnl)
    sum_t = time_pos.sum()
    sum_log = log_pnl.sum()
    sum_t2 = (time_pos ** 2).sum()
    sum_t_log = (time_pos * log_pnl).sum()

    # Slope = -λ
    denom = n * sum_t2 - sum_t ** 2
    if abs(denom) < 1e-10:
        return float('inf'), 0.0

    slope = (n * sum_t_log - sum_t * sum_log) / denom
    decay_lambda = -slope

    if decay_lambda <= 0:
        return float('inf'), 0.0  # Edge not decaying

    half_life = math.log(2) / decay_lambda
    return half_life, decay_lambda


def tae_005_rolling_edge_strength(
    pnl_series: np.ndarray,
    window: int = 20
) -> float:
    """
    Estimate current edge strength relative to historical.

    Returns value between 0 and 1:
    - 1.0 = edge is as strong as ever
    - 0.5 = edge has decayed to 50%
    - 0.0 = edge has disappeared
    """
    if len(pnl_series) < window * 2:
        return 1.0  # Not enough data to estimate decay

    # Compare recent performance to historical
    recent = pnl_series[-window:]
    historical = pnl_series[:-window]

    historical_avg = np.mean(historical)
    recent_avg = np.mean(recent)

    if historical_avg <= 0:
        return 0.0 if recent_avg <= 0 else 1.0

    # Ratio of recent to historical
    strength = recent_avg / historical_avg
    return min(1.0, max(0.0, strength))


# =============================================================================
# TAE-006: Uncertain Kelly Sizing
# =============================================================================

def tae_006_uncertain_kelly(
    win_prob: float,
    win_loss_ratio: float,
    param_uncertainty: float = 0.0,
    sample_size: int = 100,
    shrinkage_k: float = 1.0
) -> float:
    """
    TAE-006: Uncertain Kelly Sizing

    f* = (p×(b+1) - 1) / b × 0.25 × (1 - k×σ/√n)

    Kelly criterion with uncertainty adjustment.
    Shrinks position when parameter estimates are uncertain.

    Args:
        win_prob: Estimated probability of winning (0-1)
        win_loss_ratio: b = avg_win / avg_loss
        param_uncertainty: σ = uncertainty in parameters (std dev)
        sample_size: n = number of observations
        shrinkage_k: k = uncertainty penalty multiplier

    Returns:
        Optimal fraction of capital to bet (0-1)
    """
    # Basic Kelly
    b = win_loss_ratio
    p = win_prob

    if b <= 0 or p <= 0 or p >= 1:
        return 0.0

    # f* = (p*(b+1) - 1) / b
    kelly = (p * (b + 1) - 1) / b

    if kelly <= 0:
        return 0.0

    # Apply fractional Kelly (0.25 is common practice)
    kelly_frac = kelly * 0.25

    # Uncertainty shrinkage: (1 - k*σ/√n)
    if sample_size > 0:
        shrinkage = 1.0 - shrinkage_k * param_uncertainty / math.sqrt(sample_size)
        shrinkage = max(0.1, min(1.0, shrinkage))  # Bound between 0.1 and 1.0
    else:
        shrinkage = 0.1  # Maximum shrinkage with no samples

    # Final position size
    final_kelly = kelly_frac * shrinkage

    # Safety cap at 25%
    return min(0.25, max(0.0, final_kelly))


def tae_006_adaptive_kelly(
    win_history: List[bool],
    pnl_history: List[float],
    regime: str = 'unknown'
) -> float:
    """
    Calculate Kelly with uncertainty estimated from history.

    Args:
        win_history: List of win/loss booleans
        pnl_history: List of PnL values
        regime: Current market regime

    Returns:
        Uncertainty-adjusted Kelly fraction
    """
    if len(win_history) < 10:
        return 0.01  # Minimum size with insufficient data

    # Estimate win probability with confidence
    wins = sum(win_history)
    n = len(win_history)
    win_prob = wins / n

    # Std error of binomial proportion
    std_err = math.sqrt(win_prob * (1 - win_prob) / n)

    # Win/loss ratio
    pnl_arr = np.array(pnl_history)
    wins_pnl = pnl_arr[pnl_arr > 0]
    losses_pnl = abs(pnl_arr[pnl_arr < 0])

    if len(wins_pnl) == 0 or len(losses_pnl) == 0:
        return 0.01

    avg_win = np.mean(wins_pnl)
    avg_loss = np.mean(losses_pnl)

    if avg_loss < 1e-10:
        return 0.01

    win_loss_ratio = avg_win / avg_loss

    # Get regime-specific shrinkage
    regime_shrinkage = {
        'accumulation': 1.0,
        'distribution': 1.0,
        'trending_up': 0.8,
        'trending_down': 0.8,
        'consolidation': 1.2,
        'capitulation': 2.0,
        'euphoria': 2.0,
        'unknown': 1.0,
    }
    k = regime_shrinkage.get(regime.lower(), 1.0)

    return tae_006_uncertain_kelly(
        win_prob=win_prob,
        win_loss_ratio=win_loss_ratio,
        param_uncertainty=std_err,
        sample_size=n,
        shrinkage_k=k
    )


# =============================================================================
# HELPER: Regime-Optimal Timeframe Mapping
# =============================================================================

REGIME_OPTIMAL_TIMEFRAME = {
    'accumulation': 45.0,     # 30-60 sec range, midpoint
    'distribution': 45.0,     # 30-60 sec range
    'trending_up': 15.0,      # 10-20 sec range
    'trending_down': 15.0,    # 10-20 sec range
    'consolidation': 10.0,    # 5-15 sec range
    'capitulation': 3.0,      # 1-5 sec range
    'euphoria': 3.0,          # 1-5 sec range
    'unknown': 15.0,          # Default
}


def get_optimal_timeframe_for_regime(regime: str) -> float:
    """Get optimal timeframe τ* for a given regime."""
    return REGIME_OPTIMAL_TIMEFRAME.get(regime.lower(), 15.0)


# =============================================================================
# COMBINED: Timeframe Score with All Factors
# =============================================================================

def compute_timeframe_score(
    candidate_tau: float,
    regime: str,
    signal_mi: float,
    scale_consensus: float
) -> float:
    """
    Combined timeframe score incorporating all factors.

    Score = TVS(τ) × MI_normalized × Consensus

    Args:
        candidate_tau: Candidate timeframe in seconds
        regime: Current market regime
        signal_mi: Mutual information of signal at this timeframe
        scale_consensus: Consensus across wavelet scales

    Returns:
        Combined score (higher = better timeframe)
    """
    # Get regime-specific parameters
    optimal_tau = get_optimal_timeframe_for_regime(regime)
    decay_lambda = get_decay_rate_for_regime(regime)

    # TAE-001: Timeframe validity
    tvs = tae_001_timeframe_validity(candidate_tau, optimal_tau, decay_lambda)

    # Normalize MI (assume max MI around 0.7 nats)
    mi_norm = min(1.0, signal_mi / 0.7) if signal_mi > 0 else 0.0

    # Combined score
    score = tvs * mi_norm * scale_consensus

    return score
