#!/usr/bin/env python3
"""
TIME-ADAPTIVE PRICE IMPACT MODEL
=================================

The key insight: Price impact DECAYS over time.

What works at 1 second may not work at 60 seconds.
What works at 60 seconds may not work at 5 minutes.

MATHEMATICAL MODEL (Bouchaud et al., 2004):
-------------------------------------------

ΔP(t) = γ × F + η × F × t^(-β)

Where:
  γ = Permanent impact coefficient (information content, never reverts)
  η = Temporary impact coefficient (liquidity displacement, reverts)
  β = Decay exponent (typically 0.4-0.7, often ~0.5)
  F = Flow size (BTC)
  t = Time in seconds

CALIBRATION:
------------
We observe price at multiple time points after each flow:
  t = [1, 5, 10, 30, 60, 120, 300, 600, 1800] seconds

Then fit the decay curve to find γ, η, β per exchange.

PREDICTION:
-----------
Once calibrated, we can predict price impact at ANY time horizon:
  predict_impact(exchange, flow_btc, time_seconds) → expected ΔP

This gives us DETERMINISTIC formulas for any timeframe.
"""

import sqlite3
import time
import threading
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TimeAdaptiveConfig:
    """Configuration for time-adaptive model."""
    db_path: str = "/root/sovereign/time_adaptive_impact.db"

    # Observation time points (seconds after flow)
    time_points: List[int] = field(default_factory=lambda: [
        1, 5, 10, 30, 60, 120, 300, 600, 1800
    ])

    # Minimum samples for calibration
    min_samples: int = 50

    # Default parameters (before calibration)
    default_gamma: float = 0.00001  # Permanent impact
    default_eta: float = 0.0001    # Temporary impact
    default_beta: float = 0.5      # Decay exponent

    # Flow thresholds for different regimes
    small_flow_btc: float = 5.0
    medium_flow_btc: float = 50.0
    large_flow_btc: float = 200.0


# =============================================================================
# MATHEMATICAL FUNCTIONS
# =============================================================================

def impact_decay_model(t: np.ndarray, gamma: float, eta: float, beta: float) -> np.ndarray:
    """
    Price impact as function of time.

    ΔP/F = γ + η × t^(-β)

    Where:
      γ = permanent impact (asymptotic value as t → ∞)
      η = temporary impact amplitude
      β = decay exponent
      t = time in seconds

    Returns normalized impact (divide by flow to get coefficient).
    """
    # Avoid division by zero at t=0
    t_safe = np.maximum(t, 0.1)
    return gamma + eta * np.power(t_safe, -beta)


def inverse_sqrt_decay(t: np.ndarray, gamma: float, eta: float) -> np.ndarray:
    """
    Simplified model with β=0.5 (square root decay).

    ΔP/F = γ + η / √t

    This is the most common empirical finding.
    """
    t_safe = np.maximum(t, 0.1)
    return gamma + eta / np.sqrt(t_safe)


def exponential_decay(t: np.ndarray, gamma: float, eta: float, tau: float) -> np.ndarray:
    """
    Alternative: Exponential decay model.

    ΔP/F = γ + η × exp(-t/τ)

    Where τ is the characteristic decay time.
    """
    return gamma + eta * np.exp(-t / tau)


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

SCHEMA = """
-- Flow events with multi-timepoint observations
CREATE TABLE IF NOT EXISTS flow_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    exchange TEXT NOT NULL,
    direction TEXT NOT NULL,  -- 'inflow' or 'outflow'
    amount_btc REAL NOT NULL,
    price_at_signal REAL NOT NULL,

    -- Price observations at each time point
    price_1s REAL,
    price_5s REAL,
    price_10s REAL,
    price_30s REAL,
    price_60s REAL,
    price_120s REAL,
    price_300s REAL,
    price_600s REAL,
    price_1800s REAL,

    -- Calculated deltas
    delta_1s REAL,
    delta_5s REAL,
    delta_10s REAL,
    delta_30s REAL,
    delta_60s REAL,
    delta_120s REAL,
    delta_300s REAL,
    delta_600s REAL,
    delta_1800s REAL,

    -- Normalized coefficients (delta / flow)
    coef_1s REAL,
    coef_5s REAL,
    coef_10s REAL,
    coef_30s REAL,
    coef_60s REAL,
    coef_120s REAL,
    coef_300s REAL,
    coef_600s REAL,
    coef_1800s REAL,

    completed INTEGER DEFAULT 0
);

-- Calibrated parameters per exchange
CREATE TABLE IF NOT EXISTS calibrated_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    exchange TEXT NOT NULL,
    direction TEXT NOT NULL,
    flow_regime TEXT NOT NULL,  -- 'small', 'medium', 'large', 'all'

    -- Fitted parameters
    gamma REAL NOT NULL,       -- Permanent impact
    eta REAL NOT NULL,         -- Temporary impact
    beta REAL NOT NULL,        -- Decay exponent

    -- Fit quality
    r_squared REAL,
    rmse REAL,
    sample_count INTEGER,

    UNIQUE(exchange, direction, flow_regime)
);

-- Prediction log
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    exchange TEXT NOT NULL,
    direction TEXT NOT NULL,
    amount_btc REAL NOT NULL,
    target_time_seconds INTEGER NOT NULL,
    predicted_delta REAL NOT NULL,
    actual_delta REAL,
    error REAL,
    verified INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_flow_exchange ON flow_observations(exchange);
CREATE INDEX IF NOT EXISTS idx_flow_direction ON flow_observations(direction);
CREATE INDEX IF NOT EXISTS idx_params_exchange ON calibrated_params(exchange, direction);
"""


# =============================================================================
# TIME-ADAPTIVE PRICE IMPACT MODEL
# =============================================================================

class TimeAdaptiveImpactModel:
    """
    Price impact model that adapts to any time horizon.

    Calibrates decay parameters from observed data, then predicts
    impact at any future time point.
    """

    def __init__(self, config: TimeAdaptiveConfig = None):
        self.config = config or TimeAdaptiveConfig()
        self.conn = sqlite3.connect(self.config.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

        # Cached parameters per exchange
        self.params: Dict[str, Dict] = {}
        self._load_cached_params()

        # Pending observations
        self.pending: Dict[int, Dict] = {}
        self.lock = threading.Lock()

    def _init_db(self):
        """Initialize database schema."""
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def _load_cached_params(self):
        """Load calibrated parameters from database."""
        cursor = self.conn.execute("""
            SELECT exchange, direction, flow_regime, gamma, eta, beta, r_squared
            FROM calibrated_params
            ORDER BY timestamp DESC
        """)

        for row in cursor:
            key = f"{row['exchange']}_{row['direction']}_{row['flow_regime']}"
            self.params[key] = {
                'gamma': row['gamma'],
                'eta': row['eta'],
                'beta': row['beta'],
                'r_squared': row['r_squared']
            }

    def record_flow(self, exchange: str, direction: str,
                    amount_btc: float, price: float) -> int:
        """
        Record a new flow event.

        Returns observation ID for scheduling price checks.
        """
        cursor = self.conn.execute("""
            INSERT INTO flow_observations
            (timestamp, exchange, direction, amount_btc, price_at_signal)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            exchange.lower(),
            direction.lower(),
            amount_btc,
            price
        ))
        self.conn.commit()

        obs_id = cursor.lastrowid

        # Schedule observations
        with self.lock:
            self.pending[obs_id] = {
                'exchange': exchange,
                'direction': direction,
                'amount_btc': amount_btc,
                'price_at_signal': price,
                'start_time': time.time(),
                'completed': set()
            }

        return obs_id

    def record_price_observation(self, obs_id: int, time_seconds: int, price: float):
        """Record price observation at a specific time point."""
        with self.lock:
            if obs_id not in self.pending:
                return

            obs = self.pending[obs_id]
            delta = price - obs['price_at_signal']

            # Sign adjustment: inflow should cause negative delta (price down)
            if obs['direction'] == 'inflow':
                expected_sign = -1
            else:
                expected_sign = 1

            coef = delta / obs['amount_btc'] if obs['amount_btc'] > 0 else 0

        # Map time to column
        col_map = {
            1: ('price_1s', 'delta_1s', 'coef_1s'),
            5: ('price_5s', 'delta_5s', 'coef_5s'),
            10: ('price_10s', 'delta_10s', 'coef_10s'),
            30: ('price_30s', 'delta_30s', 'coef_30s'),
            60: ('price_60s', 'delta_60s', 'coef_60s'),
            120: ('price_120s', 'delta_120s', 'coef_120s'),
            300: ('price_300s', 'delta_300s', 'coef_300s'),
            600: ('price_600s', 'delta_600s', 'coef_600s'),
            1800: ('price_1800s', 'delta_1800s', 'coef_1800s'),
        }

        if time_seconds not in col_map:
            return

        price_col, delta_col, coef_col = col_map[time_seconds]

        self.conn.execute(f"""
            UPDATE flow_observations
            SET {price_col} = ?, {delta_col} = ?, {coef_col} = ?
            WHERE id = ?
        """, (price, delta, coef, obs_id))
        self.conn.commit()

        with self.lock:
            if obs_id in self.pending:
                self.pending[obs_id]['completed'].add(time_seconds)

                # Mark complete if all time points observed
                if self.pending[obs_id]['completed'] >= set(self.config.time_points):
                    self.conn.execute(
                        "UPDATE flow_observations SET completed = 1 WHERE id = ?",
                        (obs_id,)
                    )
                    self.conn.commit()
                    del self.pending[obs_id]

    def calibrate(self, exchange: str = None, direction: str = None,
                  flow_regime: str = 'all') -> Dict:
        """
        Calibrate decay parameters from observed data.

        Fits: ΔP/F = γ + η × t^(-β)

        Returns fitted parameters and quality metrics.
        """
        # Build query
        conditions = ["completed = 1"]
        params = []

        if exchange:
            conditions.append("exchange = ?")
            params.append(exchange.lower())

        if direction:
            conditions.append("direction = ?")
            params.append(direction.lower())

        if flow_regime == 'small':
            conditions.append(f"amount_btc < {self.config.small_flow_btc}")
        elif flow_regime == 'medium':
            conditions.append(f"amount_btc >= {self.config.small_flow_btc}")
            conditions.append(f"amount_btc < {self.config.medium_flow_btc}")
        elif flow_regime == 'large':
            conditions.append(f"amount_btc >= {self.config.medium_flow_btc}")

        where = " AND ".join(conditions)

        cursor = self.conn.execute(f"""
            SELECT coef_1s, coef_5s, coef_10s, coef_30s, coef_60s,
                   coef_120s, coef_300s, coef_600s, coef_1800s
            FROM flow_observations
            WHERE {where}
        """, params)

        rows = cursor.fetchall()

        if len(rows) < self.config.min_samples:
            return {
                'status': 'insufficient_data',
                'samples': len(rows),
                'required': self.config.min_samples
            }

        # Build time series data
        time_points = np.array([1, 5, 10, 30, 60, 120, 300, 600, 1800], dtype=float)

        # Average coefficients at each time point
        coefs = []
        for row in rows:
            row_coefs = [
                row['coef_1s'], row['coef_5s'], row['coef_10s'],
                row['coef_30s'], row['coef_60s'], row['coef_120s'],
                row['coef_300s'], row['coef_600s'], row['coef_1800s']
            ]
            # Skip rows with missing data
            if None not in row_coefs:
                coefs.append(row_coefs)

        if len(coefs) < 10:
            return {
                'status': 'insufficient_complete_data',
                'samples': len(coefs)
            }

        coefs = np.array(coefs)
        mean_coefs = np.mean(coefs, axis=0)
        std_coefs = np.std(coefs, axis=0)

        # Fit the decay model
        try:
            # Try full model first
            popt, pcov = curve_fit(
                impact_decay_model,
                time_points,
                np.abs(mean_coefs),  # Use absolute values for fitting
                p0=[self.config.default_gamma,
                    self.config.default_eta,
                    self.config.default_beta],
                bounds=([0, 0, 0.1], [0.01, 0.1, 2.0]),
                maxfev=5000
            )
            gamma, eta, beta = popt

        except Exception as e:
            # Fall back to fixed beta=0.5
            try:
                popt, pcov = curve_fit(
                    inverse_sqrt_decay,
                    time_points,
                    np.abs(mean_coefs),
                    p0=[self.config.default_gamma, self.config.default_eta],
                    bounds=([0, 0], [0.01, 0.1]),
                    maxfev=5000
                )
                gamma, eta = popt
                beta = 0.5
            except Exception as e2:
                return {
                    'status': 'fit_failed',
                    'error': str(e2)
                }

        # Calculate fit quality
        predicted = impact_decay_model(time_points, gamma, eta, beta)
        actual = np.abs(mean_coefs)

        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        # Store calibrated parameters
        ex = exchange or 'all'
        dir_ = direction or 'all'

        self.conn.execute("""
            INSERT OR REPLACE INTO calibrated_params
            (timestamp, exchange, direction, flow_regime, gamma, eta, beta,
             r_squared, rmse, sample_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            ex, dir_, flow_regime,
            gamma, eta, beta,
            r_squared, rmse, len(coefs)
        ))
        self.conn.commit()

        # Update cache
        key = f"{ex}_{dir_}_{flow_regime}"
        self.params[key] = {
            'gamma': gamma,
            'eta': eta,
            'beta': beta,
            'r_squared': r_squared
        }

        return {
            'status': 'calibrated',
            'exchange': ex,
            'direction': dir_,
            'flow_regime': flow_regime,
            'gamma': gamma,
            'eta': eta,
            'beta': beta,
            'r_squared': r_squared,
            'rmse': rmse,
            'samples': len(coefs),
            'mean_coefs': mean_coefs.tolist(),
            'time_points': time_points.tolist()
        }

    def predict_impact(self, exchange: str, direction: str,
                       amount_btc: float, time_seconds: float,
                       current_price: float) -> Dict:
        """
        Predict price impact at a specific time horizon.

        Uses calibrated parameters if available, otherwise defaults.

        Returns:
            predicted_delta: Expected price change in dollars
            confidence: Based on R² of calibration
            formula: The mathematical formula used
        """
        # Look up parameters (try specific, then general)
        keys_to_try = [
            f"{exchange.lower()}_{direction.lower()}_all",
            f"{exchange.lower()}_all_all",
            f"all_{direction.lower()}_all",
            f"all_all_all"
        ]

        params = None
        for key in keys_to_try:
            if key in self.params:
                params = self.params[key]
                break

        if params is None:
            # Use defaults
            gamma = self.config.default_gamma
            eta = self.config.default_eta
            beta = self.config.default_beta
            confidence = 0.0
        else:
            gamma = params['gamma']
            eta = params['eta']
            beta = params['beta']
            confidence = params.get('r_squared', 0.5)

        # Calculate impact coefficient at time t
        coef = impact_decay_model(np.array([time_seconds]), gamma, eta, beta)[0]

        # Sign: inflow → negative (price down), outflow → positive (price up)
        sign = -1 if direction.lower() == 'inflow' else 1

        # Predicted delta
        predicted_delta = sign * coef * amount_btc * current_price

        # Expected direction
        expected_direction = "DOWN" if predicted_delta < 0 else "UP"

        return {
            'exchange': exchange,
            'direction': direction,
            'amount_btc': amount_btc,
            'time_seconds': time_seconds,
            'current_price': current_price,
            'predicted_delta': predicted_delta,
            'expected_direction': expected_direction,
            'confidence': confidence,
            'params': {
                'gamma': gamma,
                'eta': eta,
                'beta': beta
            },
            'formula': f"ΔP = {sign} × ({gamma:.6f} + {eta:.6f} × t^(-{beta:.2f})) × {amount_btc:.2f} × ${current_price:,.0f}"
        }

    def predict_multi_timeframe(self, exchange: str, direction: str,
                                amount_btc: float, current_price: float) -> Dict:
        """
        Predict impact at multiple time horizons simultaneously.

        Returns predictions for 1s, 10s, 60s, 5m, 30m.
        """
        time_horizons = [1, 10, 60, 300, 1800]

        predictions = {}
        for t in time_horizons:
            pred = self.predict_impact(exchange, direction, amount_btc, t, current_price)
            label = self._time_label(t)
            predictions[label] = {
                'time_seconds': t,
                'predicted_delta': pred['predicted_delta'],
                'expected_direction': pred['expected_direction']
            }

        return {
            'exchange': exchange,
            'direction': direction,
            'amount_btc': amount_btc,
            'current_price': current_price,
            'predictions': predictions
        }

    def _time_label(self, seconds: int) -> str:
        """Convert seconds to human-readable label."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m"
        else:
            return f"{seconds // 3600}h"

    def get_formula(self, exchange: str, direction: str = 'all') -> str:
        """Get the calibrated formula as a string."""
        key = f"{exchange.lower()}_{direction.lower()}_all"

        if key not in self.params:
            return f"No calibrated formula for {exchange} {direction}"

        p = self.params[key]
        sign = "-" if direction == 'inflow' else "+"

        formula = f"""
CALIBRATED FORMULA: {exchange.upper()} {direction.upper()}
═══════════════════════════════════════════════════════

ΔP(t) = {sign}({p['gamma']:.6f} + {p['eta']:.6f} × t^(-{p['beta']:.2f})) × Flow × Price

Where:
  γ (permanent) = {p['gamma']:.6f}
  η (temporary) = {p['eta']:.6f}
  β (decay)     = {p['beta']:.2f}
  R²            = {p['r_squared']:.3f}

PREDICTIONS AT DIFFERENT HORIZONS (for 100 BTC flow at $100,000):
"""
        # Add predictions at different horizons
        for t in [1, 10, 60, 300, 1800]:
            coef = impact_decay_model(np.array([t]), p['gamma'], p['eta'], p['beta'])[0]
            delta = coef * 100 * 100000  # 100 BTC at $100k
            if direction == 'inflow':
                delta = -delta
            label = self._time_label(t)
            formula += f"  t={label:4}: ΔP = ${delta:+,.2f}\n"

        return formula

    def get_statistics(self) -> str:
        """Get model statistics."""
        cursor = self.conn.execute("""
            SELECT exchange, direction,
                   COUNT(*) as total,
                   SUM(CASE WHEN completed = 1 THEN 1 ELSE 0 END) as completed
            FROM flow_observations
            GROUP BY exchange, direction
        """)

        stats = "TIME-ADAPTIVE MODEL STATISTICS\n"
        stats += "=" * 50 + "\n\n"

        for row in cursor:
            stats += f"{row['exchange']:12} {row['direction']:8}: "
            stats += f"{row['completed']}/{row['total']} complete\n"

        stats += "\nCALIBRATED PARAMETERS:\n"
        stats += "-" * 50 + "\n"

        for key, p in self.params.items():
            parts = key.split('_')
            stats += f"{parts[0]:12} {parts[1]:8}: "
            stats += f"γ={p['gamma']:.6f} η={p['eta']:.6f} β={p['beta']:.2f} "
            stats += f"R²={p['r_squared']:.3f}\n"

        return stats

    def close(self):
        """Close database connection."""
        self.conn.close()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example():
    """Demonstrate the time-adaptive model."""

    model = TimeAdaptiveImpactModel()

    # Example prediction at different time horizons
    print("MULTI-TIMEFRAME PREDICTION")
    print("=" * 60)
    print()

    result = model.predict_multi_timeframe(
        exchange='binance',
        direction='inflow',
        amount_btc=100.0,
        current_price=90000.0
    )

    print(f"Exchange: {result['exchange']}")
    print(f"Direction: {result['direction']}")
    print(f"Flow: {result['amount_btc']} BTC")
    print(f"Price: ${result['current_price']:,}")
    print()
    print("Predictions:")
    for label, pred in result['predictions'].items():
        print(f"  t={label:4}: {pred['expected_direction']} ${abs(pred['predicted_delta']):,.2f}")

    print()
    print("=" * 60)
    print()
    print("Key insight: Impact DECAYS over time")
    print("  - Immediate (1s): Largest impact")
    print("  - Short-term (60s): Partial decay")
    print("  - Medium-term (5m): More decay")
    print("  - Long-term (30m): Mostly permanent impact remains")

    model.close()


if __name__ == "__main__":
    example()
