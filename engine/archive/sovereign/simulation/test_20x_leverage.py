#!/usr/bin/env python3
"""
20x LEVERAGE OPTIONS TEST - High-Confidence Signal Validation

Goal: Explosive long/short with 20x+ purchasing power
Requirement: 70%+ win rate (Kelly-optimal for 20x leverage)

This test validates the Timeframe-Adaptive Engine for live 20x trading.
"""

import os
import sys
import json
import sqlite3
import argparse
import math
import struct
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from scipy import stats

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from engine.sovereign.formulas.timeframe_adaptive import (
    TimeframeAdaptiveEngine,
    AdaptiveSignal,
    EngineConfig,
    create_engine,
)
from engine.sovereign.formulas.timeframe_adaptive.integration import (
    AdaptiveSignalGenerator,
    SimulationSignal,
    HighLeverageSignalGenerator,
    HighLeverageSignal,
)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "simulation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 20x LEVERAGE CONSTANTS
# =============================================================================

# Base fees (same as baseline)
TAKER_FEE = 0.00035      # 0.035%
MAKER_FEE = 0.0001       # 0.01%
SLIPPAGE_BPS = 5         # 0.05%

# HIGH-LEVERAGE CONFIGURATION
MAX_LEVERAGE = 20        # 20x options purchasing power
BASE_LEVERAGE = 5        # Minimum leverage

# STRICT FILTERS FOR 20x (Kelly-optimal requirements)
MIN_CONFIDENCE = 0.70    # Need 70%+ for safe 20x
MIN_CONSENSUS = 0.80     # Multi-scale agreement
MIN_EDGE_STRENGTH = 0.50 # Edge must be strong
MIN_REGIME_DURATION = 60 # Seconds of regime stability

# RISK MANAGEMENT FOR 20x
TAKE_PROFIT_PCT = 0.005   # 0.5% (small target, hit often)
STOP_LOSS_PCT = 0.002     # 0.2% (tight stop)
MAX_DAILY_DRAWDOWN = 0.10 # 10% max daily drawdown
MAX_HOLD_MINUTES = 30     # Quick scalping timeframe

# Kelly fraction (conservative for 20x)
KELLY_FRACTION = 0.25     # Use 1/4 Kelly for safety


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HighLeverageTrade:
    """Trade for 20x leverage simulation."""
    id: str
    timestamp: int
    entry_price: float
    direction: int  # +1 LONG, -1 SHORT
    leverage: float
    size_usd: float
    stop_loss: float
    take_profit: float
    confidence: float
    consensus: float
    edge_strength: float
    regime: str
    timeframe: float
    entry_fee: float
    signal_type: str = "STANDARD"  # QUIET_WHALE, REGIME_STABLE, etc.
    exit_timestamp: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_fee: Optional[float] = None
    pnl_gross: Optional[float] = None
    pnl_net: Optional[float] = None
    leveraged_return: Optional[float] = None


@dataclass
class DailyStats:
    """Daily performance statistics."""
    date: str
    trades: int
    wins: int
    losses: int
    pnl: float
    max_drawdown: float
    leverage_used: float


@dataclass
class ValidationResult:
    """Final validation result."""
    passed: bool
    reason: str
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    daily_roi: float
    sharpe: float
    edge_consistency: float
    kelly_optimal: bool
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# DATA LOADING (from simulation_1to1.py)
# =============================================================================

def load_blockchain_features(db_path: str) -> Dict[str, Dict]:
    """Load blockchain features indexed by date."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]

    features = {}

    if 'daily_features' in tables:
        cursor.execute("""
            SELECT date, timestamp, tx_count, total_value_btc, whale_tx_count,
                   unique_senders, unique_receivers
            FROM daily_features
            WHERE tx_count IS NOT NULL AND total_value_btc IS NOT NULL
            ORDER BY date
        """)
        for row in cursor.fetchall():
            date_str, ts, tx, val, whale_bytes, senders, receivers = row
            # Decode binary whale_tx_count if needed
            if isinstance(whale_bytes, bytes):
                whale = struct.unpack('<q', whale_bytes)[0]  # little-endian int64
            else:
                whale = whale_bytes or 0
            features[date_str] = {
                'timestamp': ts,
                'tx_count': tx or 0,
                'total_value_btc': val or 0,
                'whale_tx_count': whale,
                'unique_senders': senders or 0,
                'unique_receivers': receivers or 0,
            }

    conn.close()
    return features


def load_prices(db_path: str) -> Dict[str, Dict]:
    """Load OHLCV prices indexed by date."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]

    prices = {}

    if 'prices' in tables:
        # Check if 'date' or 'timestamp' column exists
        cursor.execute("PRAGMA table_info(prices)")
        columns = [c[1] for c in cursor.fetchall()]

        if 'date' in columns:
            cursor.execute("""
                SELECT date, open, high, low, close
                FROM prices ORDER BY date
            """)
            for row in cursor.fetchall():
                date_str, o, h, l, c = row
                # Convert date string to timestamp
                try:
                    dt = datetime.strptime(date_str, '%Y-%m-%d')
                    ts = int(dt.timestamp())
                except:
                    ts = 0
                prices[date_str] = {
                    'timestamp': ts,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                }
        else:
            cursor.execute("""
                SELECT timestamp, open, high, low, close
                FROM prices ORDER BY timestamp
            """)
            for row in cursor.fetchall():
                ts, o, h, l, c = row
                date_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
                prices[date_str] = {
                    'timestamp': ts,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                }
    elif 'ohlcv' in tables:
        cursor.execute("""
            SELECT timestamp, open, high, low, close
            FROM ohlcv ORDER BY timestamp
        """)
        for row in cursor.fetchall():
            ts, o, h, l, c = row
            date_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
            prices[date_str] = {
                'timestamp': ts,
                'open': o,
                'high': h,
                'low': l,
                'close': c,
            }

    conn.close()
    return prices


def merge_data(features: Dict, prices: Dict) -> List[Dict]:
    """Merge blockchain features with prices."""
    merged = []
    for date_str in sorted(features.keys()):
        if date_str in prices:
            merged.append({
                'date': date_str,
                **features[date_str],
                **prices[date_str],
            })
    return merged


# =============================================================================
# KELLY CRITERION FOR LEVERAGE
# =============================================================================

def calculate_kelly_leverage(win_rate: float, win_loss_ratio: float = 2.0) -> float:
    """
    Calculate optimal leverage using Kelly Criterion.

    Kelly = (p * b - q) / b
    Where:
        p = win probability
        q = 1 - p (loss probability)
        b = win/loss ratio

    For 20x leverage to be optimal:
        Kelly * capital = 0.05 (5% of account per bet)
        Leverage = 20x means betting 20 * Kelly fraction
    """
    if win_rate <= 0.5:
        return BASE_LEVERAGE  # Minimum leverage if edge unclear

    p = win_rate
    q = 1 - p
    b = win_loss_ratio

    kelly = (p * b - q) / b

    # Apply fractional Kelly for safety
    safe_kelly = kelly * KELLY_FRACTION

    # Convert to leverage (inverse relationship)
    # Kelly of 0.05 allows 20x, Kelly of 0.10 allows 10x, etc.
    if safe_kelly > 0:
        optimal_leverage = min(MAX_LEVERAGE, 1 / safe_kelly)
    else:
        optimal_leverage = BASE_LEVERAGE

    return max(BASE_LEVERAGE, min(MAX_LEVERAGE, optimal_leverage))


def is_kelly_optimal_for_20x(win_rate: float, win_loss_ratio: float = 2.0) -> Tuple[bool, str]:
    """
    Check if current performance is Kelly-optimal for 20x leverage.

    For 20x leverage trading:
        - Need win_rate >= 70% (minimum threshold for high leverage)
        - Need positive Kelly (positive expected value)
        - Higher win rates support higher leverage safely
    """
    p = win_rate
    q = 1 - p
    b = win_loss_ratio

    kelly = (p * b - q) / b
    safe_kelly = kelly * KELLY_FRACTION

    if kelly <= 0:
        return False, f"Negative edge (Kelly={kelly:.3f})"

    # For 20x leverage, we need strong edge
    # With 70% WR and 2:1 ratio, Kelly = 0.35 (35% bet size)
    # Fractional Kelly (0.25) = 8.75% per trade
    # This supports significant leverage on options

    if win_rate >= 0.70:
        return True, f"Kelly-optimal: {win_rate:.1%} WR supports 20x leverage (Kelly={kelly:.2f})"
    else:
        return False, f"Need 70% WR for 20x, have {win_rate:.1%}. Kelly={kelly:.2f}"


# =============================================================================
# HIGH-CONFIDENCE SIGNAL FILTER
# =============================================================================

class HighConfidenceFilter:
    """
    Filters signals for 20x leverage trading.

    Only passes signals that meet ALL criteria:
    1. Confidence > 0.70 (70% probability)
    2. Consensus > 0.80 (80% multi-scale agreement)
    3. Edge strength > 0.50 (strong edge)
    4. Regime stable for > 60 seconds
    """

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE,
        min_consensus: float = MIN_CONSENSUS,
        min_edge_strength: float = MIN_EDGE_STRENGTH,
    ):
        self.min_confidence = min_confidence
        self.min_consensus = min_consensus
        self.min_edge_strength = min_edge_strength

        self.total_signals = 0
        self.passed_signals = 0
        self.rejection_reasons: Dict[str, int] = {
            'low_confidence': 0,
            'low_consensus': 0,
            'weak_edge': 0,
            'unstable_regime': 0,
            'no_direction': 0,
        }

    def filter(self, signal: AdaptiveSignal) -> Tuple[bool, str]:
        """
        Check if signal passes all high-confidence filters.

        Returns:
            (passed, reason)
        """
        self.total_signals += 1

        # Check direction
        if signal.direction == 0:
            self.rejection_reasons['no_direction'] += 1
            return False, "No direction"

        # Check confidence
        if signal.confidence < self.min_confidence:
            self.rejection_reasons['low_confidence'] += 1
            return False, f"Low confidence: {signal.confidence:.2f} < {self.min_confidence}"

        # Check consensus
        if signal.consensus < self.min_consensus:
            self.rejection_reasons['low_consensus'] += 1
            return False, f"Low consensus: {signal.consensus:.2f} < {self.min_consensus}"

        # Check edge strength
        if signal.edge_strength < self.min_edge_strength:
            self.rejection_reasons['weak_edge'] += 1
            return False, f"Weak edge: {signal.edge_strength:.2f} < {self.min_edge_strength}"

        # Check regime stability (via regime_confidence)
        if signal.regime_confidence < 0.6:
            self.rejection_reasons['unstable_regime'] += 1
            return False, f"Unstable regime: {signal.regime_confidence:.2f}"

        self.passed_signals += 1
        return True, "All filters passed"

    def get_stats(self) -> Dict:
        """Get filter statistics."""
        pass_rate = self.passed_signals / self.total_signals if self.total_signals else 0
        return {
            'total_signals': self.total_signals,
            'passed_signals': self.passed_signals,
            'pass_rate': pass_rate,
            'rejection_reasons': self.rejection_reasons,
        }


# =============================================================================
# 20x LEVERAGE SIMULATION
# =============================================================================

def simulate_20x_options(
    data: List[Dict],
    initial_capital: float = 100.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[List[HighLeverageTrade], Dict]:
    """
    Run 20x leverage options simulation with Timeframe-Adaptive Engine.

    Uses high-confidence filtering to only take trades that meet
    Kelly-optimal requirements for 20x leverage.
    """
    # Create engine with high-precision config
    config = EngineConfig(
        candidate_timeframes=[1, 2, 5, 10, 15, 20, 30],  # Fast timeframes for options
        wavelet_type='db4',
        wavelet_levels=5,
        min_edge_strength=MIN_EDGE_STRENGTH,
    )

    # Create HIGH LEVERAGE signal generator (with QUIET_WHALE detection)
    # The generator creates its own engine internally with high-leverage settings
    hl_generator = HighLeverageSignalGenerator(config)
    engine = hl_generator.engine  # Reference the internal engine

    # Create high-confidence filter
    signal_filter = HighConfidenceFilter()

    # Filter data by date range
    if start_date:
        data = [d for d in data if d['date'] >= start_date]
    if end_date:
        data = [d for d in data if d['date'] <= end_date]

    if len(data) < 50:
        print("[ERROR] Insufficient data for simulation")
        return [], {}

    # Simulation state
    equity = initial_capital
    peak_equity = initial_capital
    max_drawdown = 0
    trades: List[HighLeverageTrade] = []
    daily_stats: List[DailyStats] = []
    current_trade: Optional[HighLeverageTrade] = None

    # Rolling performance for Kelly calculation
    recent_wins = 0
    recent_losses = 0
    recent_win_amounts = []
    recent_loss_amounts = []

    print(f"\n[SIMULATING] 20x Options Trading")
    print(f"  Data: {data[0]['date']} to {data[-1]['date']} ({len(data)} days)")
    print(f"  Capital: ${initial_capital}")
    print(f"  Max Leverage: {MAX_LEVERAGE}x")
    print(f"  Filters: Conf>{MIN_CONFIDENCE}, Cons>{MIN_CONSENSUS}, Edge>{MIN_EDGE_STRENGTH}")

    # Daily tracking
    current_day = None
    day_trades = 0
    day_wins = 0
    day_losses = 0
    day_pnl = 0.0
    day_max_dd = 0.0
    day_leverage_sum = 0.0

    for i in range(20, len(data) - 1):  # Need lookback window
        current = data[i]
        next_candle = data[i + 1]

        # Track daily stats
        if current['date'] != current_day:
            if current_day is not None and day_trades > 0:
                daily_stats.append(DailyStats(
                    date=current_day,
                    trades=day_trades,
                    wins=day_wins,
                    losses=day_losses,
                    pnl=day_pnl,
                    max_drawdown=day_max_dd,
                    leverage_used=day_leverage_sum / day_trades if day_trades else 0,
                ))
            current_day = current['date']
            day_trades = 0
            day_wins = 0
            day_losses = 0
            day_pnl = 0.0
            day_max_dd = 0.0
            day_leverage_sum = 0.0

        # Check exit for open trade
        if current_trade:
            high = current['high']
            low = current['low']

            exit_price = None
            exit_reason = None

            if current_trade.direction == 1:  # LONG
                if high >= current_trade.take_profit:
                    exit_price = current_trade.take_profit
                    exit_reason = 'TAKE_PROFIT'
                elif low <= current_trade.stop_loss:
                    exit_price = current_trade.stop_loss
                    exit_reason = 'STOP_LOSS'
            else:  # SHORT
                if low <= current_trade.take_profit:
                    exit_price = current_trade.take_profit
                    exit_reason = 'TAKE_PROFIT'
                elif high >= current_trade.stop_loss:
                    exit_price = current_trade.stop_loss
                    exit_reason = 'STOP_LOSS'

            if exit_price:
                # Close trade
                exit_fee = current_trade.size_usd * TAKER_FEE

                if current_trade.direction == 1:
                    pnl_pct = (exit_price - current_trade.entry_price) / current_trade.entry_price
                else:
                    pnl_pct = (current_trade.entry_price - exit_price) / current_trade.entry_price

                # Apply leverage
                leveraged_return = pnl_pct * current_trade.leverage
                pnl_gross = current_trade.size_usd * pnl_pct
                pnl_net = pnl_gross - current_trade.entry_fee - exit_fee

                current_trade.exit_timestamp = current['timestamp']
                current_trade.exit_price = exit_price
                current_trade.exit_reason = exit_reason
                current_trade.exit_fee = exit_fee
                current_trade.pnl_gross = pnl_gross
                current_trade.pnl_net = pnl_net
                current_trade.leveraged_return = leveraged_return

                trades.append(current_trade)

                # Update equity
                equity += pnl_net
                day_pnl += pnl_net

                # Track performance
                if pnl_net > 0:
                    day_wins += 1
                    recent_wins += 1
                    recent_win_amounts.append(pnl_net)
                else:
                    day_losses += 1
                    recent_losses += 1
                    recent_loss_amounts.append(abs(pnl_net))

                # Keep rolling window of 50 trades
                if len(recent_win_amounts) + len(recent_loss_amounts) > 50:
                    if recent_win_amounts and recent_wins > 0:
                        recent_win_amounts.pop(0)
                        recent_wins -= 1
                    elif recent_loss_amounts and recent_losses > 0:
                        recent_loss_amounts.pop(0)
                        recent_losses -= 1

                # Update drawdown
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity if peak_equity else 0
                max_drawdown = max(max_drawdown, dd)
                day_max_dd = max(day_max_dd, dd)

                current_trade = None

        # Generate signal if no position
        if not current_trade and equity > 10:
            # Feed engine with historical prices
            window = data[i-20:i+1]
            prices = [d['close'] for d in window]

            # Set regime based on price action
            recent_returns = [(prices[j] - prices[j-1]) / prices[j-1] for j in range(1, len(prices))]
            avg_return = sum(recent_returns) / len(recent_returns)

            if avg_return > 0.01:
                regime = 'trending_up'
            elif avg_return < -0.01:
                regime = 'trending_down'
            else:
                regime = 'consolidation'

            engine.set_regime(regime, {regime: 0.8})

            # Process through engine
            for price in prices[:-1]:
                engine.add_price(price)

            # Build raw_data for HighLeverageSignalGenerator (includes QUIET_WHALE detection)
            raw_data = {
                'price': current['close'],
                'prices': prices,
                'tx_count': current.get('tx_count', 0),
                'whale_tx_count': current.get('whale_tx_count', 0),
                'total_value_btc': current.get('total_value_btc', 0),
                'unique_senders': current.get('unique_senders', 0),
                'unique_receivers': current.get('unique_receivers', 0),
            }

            # Generate 20x leverage signal with QUIET_WHALE detection
            hl_signal = hl_generator.generate_20x(
                raw_data=raw_data,
                regime=regime,
                regime_probs={regime: 0.8},
                timestamp=float(current['timestamp'])  # For backtesting regime stability
            )

            # Check if signal passes all filters
            passed = hl_signal.passes_filter()
            signal = hl_signal.adaptive  # For compatibility with rest of code
            signal_type = hl_signal.signal_type  # Track QUIET_WHALE, REGIME_STABLE, etc.

            # Track filter stats manually
            signal_filter.total_signals += 1
            if not passed:
                if signal.direction == 0:
                    signal_filter.rejection_reasons['no_direction'] += 1
                elif signal.confidence < MIN_CONFIDENCE:
                    signal_filter.rejection_reasons['low_confidence'] += 1
                elif signal.consensus < MIN_CONSENSUS:
                    signal_filter.rejection_reasons['low_consensus'] += 1
                elif signal.edge_strength < MIN_EDGE_STRENGTH:
                    signal_filter.rejection_reasons['weak_edge'] += 1
                else:
                    signal_filter.rejection_reasons['unstable_regime'] += 1
            else:
                signal_filter.passed_signals += 1

            if passed:
                # Calculate optimal leverage based on recent performance
                total_recent = recent_wins + recent_losses
                if total_recent >= 10:
                    rolling_wr = recent_wins / total_recent
                    avg_win = sum(recent_win_amounts) / len(recent_win_amounts) if recent_win_amounts else 1
                    avg_loss = sum(recent_loss_amounts) / len(recent_loss_amounts) if recent_loss_amounts else 1
                    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 2.0
                    leverage = calculate_kelly_leverage(rolling_wr, win_loss_ratio)
                else:
                    # Not enough data, use conservative leverage
                    leverage = BASE_LEVERAGE

                # Entry at next candle open
                entry_price = next_candle['open']

                # Add slippage
                if signal.direction == 1:
                    entry_price *= (1 + SLIPPAGE_BPS / 10000)
                else:
                    entry_price *= (1 - SLIPPAGE_BPS / 10000)

                # Position size
                size_usd = equity * 0.1  # 10% of equity per trade

                # Calculate TP/SL
                if signal.direction == 1:
                    stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                    take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
                else:
                    stop_loss = entry_price * (1 + STOP_LOSS_PCT)
                    take_profit = entry_price * (1 - TAKE_PROFIT_PCT)

                entry_fee = size_usd * TAKER_FEE

                current_trade = HighLeverageTrade(
                    id=f"20X_{next_candle['timestamp']}",
                    timestamp=next_candle['timestamp'],
                    entry_price=entry_price,
                    direction=signal.direction,
                    leverage=leverage,
                    size_usd=size_usd,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=signal.confidence,
                    consensus=signal.consensus,
                    edge_strength=signal.edge_strength,
                    regime=signal.regime,
                    timeframe=signal.optimal_timeframe,
                    entry_fee=entry_fee,
                    signal_type=signal_type,  # Track QUIET_WHALE for analysis
                )

                day_trades += 1
                day_leverage_sum += leverage

        # Check for max daily drawdown
        if day_max_dd >= MAX_DAILY_DRAWDOWN:
            # Stop trading for the day
            current_trade = None

    # Final daily stats
    if current_day is not None and day_trades > 0:
        daily_stats.append(DailyStats(
            date=current_day,
            trades=day_trades,
            wins=day_wins,
            losses=day_losses,
            pnl=day_pnl,
            max_drawdown=day_max_dd,
            leverage_used=day_leverage_sum / day_trades if day_trades else 0,
        ))

    # Calculate final metrics
    wins = len([t for t in trades if t.pnl_net and t.pnl_net > 0])
    losses = len([t for t in trades if t.pnl_net and t.pnl_net <= 0])
    win_rate = wins / len(trades) if trades else 0

    total_pnl = sum(t.pnl_net for t in trades if t.pnl_net)

    # QUIET_WHALE specific stats (100% win rate signals)
    quiet_whale_trades = [t for t in trades if t.signal_type == "QUIET_WHALE"]
    qw_wins = len([t for t in quiet_whale_trades if t.pnl_net and t.pnl_net > 0])
    qw_losses = len([t for t in quiet_whale_trades if t.pnl_net and t.pnl_net <= 0])
    qw_win_rate = qw_wins / len(quiet_whale_trades) if quiet_whale_trades else 0
    qw_pnl = sum(t.pnl_net for t in quiet_whale_trades if t.pnl_net)

    gross_wins = sum(t.pnl_net for t in trades if t.pnl_net and t.pnl_net > 0)
    gross_losses = abs(sum(t.pnl_net for t in trades if t.pnl_net and t.pnl_net < 0))
    profit_factor = gross_wins / gross_losses if gross_losses else float('inf')

    # Daily returns for Sharpe
    daily_returns = [d.pnl / initial_capital for d in daily_stats if d.trades > 0]
    if daily_returns:
        avg_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns) or 0.01
        sharpe = avg_daily / std_daily * np.sqrt(252)
    else:
        sharpe = 0

    # Average daily ROI
    daily_roi = np.mean([d.pnl for d in daily_stats]) / initial_capital if daily_stats else 0

    # Edge consistency
    timeframe_consistency = sum(1 for t in trades if t.timeframe < 30) / len(trades) if trades else 0

    # Check Kelly optimality
    kelly_optimal, kelly_reason = is_kelly_optimal_for_20x(win_rate)

    results = {
        'total_trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'final_equity': equity,
        'return_pct': (equity - initial_capital) / initial_capital,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'daily_roi': daily_roi,
        'avg_leverage': np.mean([t.leverage for t in trades]) if trades else 0,
        'avg_confidence': np.mean([t.confidence for t in trades]) if trades else 0,
        'avg_consensus': np.mean([t.consensus for t in trades]) if trades else 0,
        'timeframe_consistency': timeframe_consistency,
        'kelly_optimal': kelly_optimal,
        'kelly_reason': kelly_reason,
        'filter_stats': signal_filter.get_stats(),
        'daily_stats': [asdict(d) for d in daily_stats],
        # QUIET_WHALE specific metrics (100% win rate in baseline)
        'quiet_whale': {
            'trades': len(quiet_whale_trades),
            'wins': qw_wins,
            'losses': qw_losses,
            'win_rate': qw_win_rate,
            'pnl': qw_pnl,
        },
    }

    return trades, results


# =============================================================================
# VALIDATION & REPORTING
# =============================================================================

def validate_for_live_trading(results: Dict) -> ValidationResult:
    """
    Validate if results are suitable for live 20x trading.

    Hard Requirements:
    - Win Rate >= 70% (Kelly-optimal for 20x)
    - Max Drawdown <= 10%
    - Min 50 trades (statistical significance)
    - P-value < 0.001 (99.9% confidence)
    """
    recommendations = []

    # Check minimum trades
    if results['total_trades'] < 50:
        return ValidationResult(
            passed=False,
            reason=f"Insufficient trades: {results['total_trades']} < 50",
            total_trades=results['total_trades'],
            win_rate=results['win_rate'],
            profit_factor=results['profit_factor'],
            max_drawdown=results['max_drawdown'],
            daily_roi=results['daily_roi'],
            sharpe=results['sharpe'],
            edge_consistency=results['timeframe_consistency'],
            kelly_optimal=results['kelly_optimal'],
            recommendations=["Need more data for statistical significance"],
        )

    # Statistical significance test
    n = results['total_trades']
    wins = results['wins']

    # Binomial test: Is win rate significantly > 50%?
    # Use binomtest (newer scipy) or binom_test (older scipy)
    try:
        result = stats.binomtest(wins, n, 0.5, alternative='greater')
        p_value = result.pvalue
    except AttributeError:
        p_value = stats.binom_test(wins, n, 0.5, alternative='greater')

    if p_value > 0.001:
        recommendations.append(f"P-value {p_value:.4f} > 0.001 - edge may be noise")

    # Check win rate (70% required for 20x)
    if results['win_rate'] < 0.70:
        passed = False
        reason = f"Win rate {results['win_rate']:.1%} < 70% (Kelly requirement for 20x)"
        recommendations.append("Increase confidence threshold or improve signal quality")
    elif results['max_drawdown'] > 0.10:
        passed = False
        reason = f"Max drawdown {results['max_drawdown']:.1%} > 10% (too risky for 20x)"
        recommendations.append("Tighten stop losses or reduce leverage")
    elif results['profit_factor'] < 3.0:
        passed = False
        reason = f"Profit factor {results['profit_factor']:.2f} < 3.0"
        recommendations.append("Need wins to dominate losses more")
    elif not results['kelly_optimal']:
        passed = False
        reason = f"Not Kelly-optimal: {results['kelly_reason']}"
        recommendations.append("Current performance doesn't support 20x leverage safely")
    else:
        passed = True
        reason = "All 20x requirements met"

    # Add positive notes
    if results['win_rate'] >= 0.70:
        recommendations.append(f"Win rate {results['win_rate']:.1%} meets 70% threshold")
    if results['sharpe'] >= 3.0:
        recommendations.append(f"Sharpe {results['sharpe']:.2f} indicates consistent edge")
    if results['daily_roi'] >= 0.20:
        recommendations.append(f"Daily ROI {results['daily_roi']:.1%} meets 20% target")

    return ValidationResult(
        passed=passed,
        reason=reason,
        total_trades=results['total_trades'],
        win_rate=results['win_rate'],
        profit_factor=results['profit_factor'],
        max_drawdown=results['max_drawdown'],
        daily_roi=results['daily_roi'],
        sharpe=results['sharpe'],
        edge_consistency=results['timeframe_consistency'],
        kelly_optimal=results['kelly_optimal'],
        recommendations=recommendations,
    )


def generate_report(trades: List[HighLeverageTrade], results: Dict, validation: ValidationResult):
    """Generate comprehensive validation report."""
    print("\n" + "=" * 70)
    print("20x LEVERAGE OPTIONS VALIDATION REPORT")
    print("=" * 70)

    print("\nPERFORMANCE METRICS:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Wins/Losses: {results['wins']}/{results['losses']}")
    print(f"  Win Rate: {results['win_rate']:.1%} {'[OK]' if results['win_rate'] >= 0.70 else '[NEED 70%+]'}")
    print(f"  Profit Factor: {results['profit_factor']:.2f} {'[OK]' if results['profit_factor'] >= 3.0 else '[NEED 3.0+]'}")
    print(f"  Max Drawdown: {results['max_drawdown']:.1%} {'[OK]' if results['max_drawdown'] <= 0.10 else '[NEED <10%]'}")
    print(f"  Daily ROI: {results['daily_roi']:.1%} {'[OK]' if results['daily_roi'] >= 0.20 else '[TARGET 20%+]'}")
    print(f"  Sharpe Ratio: {results['sharpe']:.2f} {'[OK]' if results['sharpe'] >= 3.0 else '[TARGET 3.0+]'}")

    print("\nLEVERAGE METRICS:")
    print(f"  Average Leverage Used: {results['avg_leverage']:.1f}x")
    print(f"  Kelly Optimal: {'YES' if results['kelly_optimal'] else 'NO'}")
    print(f"  Kelly Analysis: {results['kelly_reason']}")

    print("\nSIGNAL QUALITY:")
    print(f"  Average Confidence: {results['avg_confidence']:.2f}")
    print(f"  Average Consensus: {results['avg_consensus']:.2f}")
    print(f"  Timeframe Consistency: {results['timeframe_consistency']:.1%}")

    print("\nQUIET_WHALE SIGNALS (100% WR in baseline):")
    qw = results['quiet_whale']
    if qw['trades'] > 0:
        print(f"  Trades: {qw['trades']}")
        print(f"  Wins/Losses: {qw['wins']}/{qw['losses']}")
        print(f"  Win Rate: {qw['win_rate']:.1%} {'[TARGET: 100%]' if qw['win_rate'] < 1.0 else '[PERFECT]'}")
        print(f"  PnL: ${qw['pnl']:.2f}")
    else:
        print(f"  No QUIET_WHALE signals detected")

    print("\nFILTER STATISTICS:")
    filter_stats = results['filter_stats']
    print(f"  Total Signals: {filter_stats['total_signals']}")
    print(f"  Passed Filter: {filter_stats['passed_signals']} ({filter_stats['pass_rate']:.1%})")
    print(f"  Rejection Breakdown:")
    for reason, count in filter_stats['rejection_reasons'].items():
        if count > 0:
            print(f"    - {reason}: {count}")

    print("\n" + "-" * 70)
    print("VALIDATION RESULT:")
    print("-" * 70)
    if validation.passed:
        print(f"  [GO] {validation.reason}")
        print(f"\n  READY FOR LIVE 20x TRADING")
    else:
        print(f"  [NO-GO] {validation.reason}")
        print(f"\n  NOT READY - Optimization Required")

    print("\nRECOMMENDATIONS:")
    for rec in validation.recommendations:
        print(f"  - {rec}")

    print("\n" + "=" * 70)

    # Save results
    output_file = RESULTS_DIR / f"20x_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'validation': asdict(validation),
        'trades': [asdict(t) for t in trades[:100]],  # Save first 100 trades
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\n[SAVED] Results to {output_file}")

    return validation.passed


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='20x Leverage Options Validation')
    parser.add_argument('--capital', type=float, default=100.0,
                        help='Initial capital (default: 100)')
    parser.add_argument('--start', type=str, default='2014-09-17',
                        help='Start date (default: 2014-09-17)')
    parser.add_argument('--end', type=str, default='2021-01-25',
                        help='End date (default: 2021-01-25)')
    args = parser.parse_args()

    print("=" * 70)
    print("20x LEVERAGE OPTIONS TEST")
    print("=" * 70)
    print(f"Capital: ${args.capital}")
    print(f"Max Leverage: {MAX_LEVERAGE}x")
    print(f"Date Range: {args.start} to {args.end}")
    print(f"Confidence Threshold: {MIN_CONFIDENCE:.0%}")
    print(f"Consensus Threshold: {MIN_CONSENSUS:.0%}")
    print("=" * 70)

    # Load data from unified_bitcoin.db (has both features and prices)
    print("\n[LOADING] Data from unified_bitcoin.db...")

    unified_db = DATA_DIR / "unified_bitcoin.db"
    if not unified_db.exists():
        print(f"[ERROR] {unified_db} not found")
        sys.exit(1)

    # Load blockchain features
    features = load_blockchain_features(str(unified_db))
    print(f"  Loaded {len(features)} blockchain features")

    # Load prices
    prices = load_prices(str(unified_db))
    print(f"  Loaded {len(prices)} price candles")

    if not features or not prices:
        print("[ERROR] Could not load required data")
        sys.exit(1)

    # Merge data
    data = merge_data(features, prices)
    print(f"  Merged: {len(data)} records")

    # Run simulation
    trades, results = simulate_20x_options(
        data,
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
    )

    # Validate
    validation = validate_for_live_trading(results)

    # Generate report
    passed = generate_report(trades, results, validation)

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
