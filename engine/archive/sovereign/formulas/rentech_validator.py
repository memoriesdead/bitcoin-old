"""
RenTech Statistical Validator
==============================

Tracks live performance and detects edge decay using rigorous statistical methods:

1. Sequential Probability Ratio Test (SPRT)
   - Wald's SPRT for online hypothesis testing
   - H0: Win rate = 50% (no edge)
   - H1: Win rate = backtest rate (edge exists)
   - Thresholds: alpha=0.05, beta=0.05

2. Exponential Moving Average (EMA) Win Rate
   - Tracks rolling win rate with decay
   - Alerts at 55% (10-trade EMA)
   - Disables at 52.5% (20-trade EMA)

3. CUSUM (Cumulative Sum) Edge Decay
   - Detects persistent underperformance
   - Threshold tuned for 5-sigma events

Each formula is tracked independently and can be:
- ACTIVE: Trading normally
- ALERT: Underperforming, reduced position size
- DISABLED: Edge lost, no new positions

Created: 2025-12-16
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FormulaStatus(Enum):
    """Status of a formula based on validation."""
    ACTIVE = "ACTIVE"      # Trading normally
    ALERT = "ALERT"        # Underperforming, reduced size
    DISABLED = "DISABLED"  # Edge lost, no trading
    CONFIRMING = "CONFIRMING"  # Not enough data yet


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    formula_id: str
    entry_time: float
    exit_time: float
    entry_price: float
    exit_price: float
    direction: int  # 1 = LONG, -1 = SHORT
    pnl_pct: float
    was_win: bool


@dataclass
class FormulaValidationState:
    """Validation state for a single formula."""
    formula_id: str
    status: FormulaStatus = FormulaStatus.CONFIRMING

    # Historical baseline (from backtest)
    baseline_win_rate: float = 1.0
    baseline_avg_return: float = 20.0
    baseline_trades: int = 10

    # Live tracking
    live_trades: int = 0
    live_wins: int = 0
    live_losses: int = 0
    live_total_return: float = 0.0

    # SPRT state
    sprt_log_likelihood: float = 0.0
    sprt_decision: str = "CONTINUE"  # ACCEPT_H1, REJECT_H1, CONTINUE

    # EMA win rate
    ema_win_rate_10: float = 0.5
    ema_win_rate_20: float = 0.5

    # CUSUM
    cusum_value: float = 0.0
    cusum_peak: float = 0.0

    # Timestamps
    last_trade_time: float = 0.0
    last_status_change: float = 0.0

    @property
    def live_win_rate(self) -> float:
        """Calculate live win rate."""
        if self.live_trades == 0:
            return 0.5
        return self.live_wins / self.live_trades

    @property
    def position_multiplier(self) -> float:
        """Position size multiplier based on status."""
        if self.status == FormulaStatus.ACTIVE:
            return 1.0
        elif self.status == FormulaStatus.ALERT:
            return 0.5  # Half size
        elif self.status == FormulaStatus.CONFIRMING:
            # Scale up with live trades
            return min(0.25 + 0.75 * (self.live_trades / 20), 1.0)
        else:
            return 0.0  # Disabled


class RenTechValidator:
    """
    Statistical validator for RenTech formulas.

    Uses SPRT, EMA, and CUSUM to detect edge decay and automatically
    disable formulas that are no longer profitable.
    """

    # SPRT thresholds (Wald's boundaries)
    # alpha = 0.05 (Type I error: falsely reject null)
    # beta = 0.05 (Type II error: falsely accept null)
    # A = (1-beta)/alpha = 19
    # B = beta/(1-alpha) = 0.0526
    SPRT_UPPER = np.log(19)     # ~2.944, accept H1 (edge exists)
    SPRT_LOWER = np.log(0.0526)  # ~-2.944, reject H1 (no edge)

    # EMA thresholds
    EMA_ALERT_THRESHOLD = 0.55    # Alert if 10-trade EMA below this
    EMA_DISABLE_THRESHOLD = 0.525  # Disable if 20-trade EMA below this

    # CUSUM threshold
    CUSUM_THRESHOLD = 5.0  # 5-sigma cumulative underperformance

    def __init__(
        self,
        state_file: str = "data/rentech_validator_state.json"
    ):
        """
        Initialize validator.

        Args:
            state_file: Path to persist state
        """
        self.state_file = Path(state_file)
        self.formula_states: Dict[str, FormulaValidationState] = {}
        self.trade_history: List[TradeRecord] = []

        # Try to load existing state
        self._load_state()

    def initialize_formula(
        self,
        formula_id: str,
        baseline_win_rate: float = 1.0,
        baseline_avg_return: float = 20.0,
        baseline_trades: int = 10
    ):
        """
        Initialize tracking for a formula.

        Args:
            formula_id: Formula identifier
            baseline_win_rate: Historical win rate from backtest
            baseline_avg_return: Historical average return
            baseline_trades: Number of historical trades
        """
        if formula_id not in self.formula_states:
            self.formula_states[formula_id] = FormulaValidationState(
                formula_id=formula_id,
                baseline_win_rate=baseline_win_rate,
                baseline_avg_return=baseline_avg_return,
                baseline_trades=baseline_trades,
                status=FormulaStatus.CONFIRMING
            )
            logger.info(
                f"Initialized validator for {formula_id}: "
                f"baseline WR={baseline_win_rate:.0%}, "
                f"avg_ret={baseline_avg_return:.1f}%"
            )

    def record_trade(
        self,
        formula_id: str,
        entry_time: float,
        exit_time: float,
        entry_price: float,
        exit_price: float,
        direction: int,
        pnl_pct: float
    ) -> FormulaStatus:
        """
        Record a completed trade and update validation state.

        Args:
            formula_id: Formula that generated the trade
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            direction: 1 for LONG, -1 for SHORT
            pnl_pct: Profit/loss percentage

        Returns:
            Updated formula status
        """
        was_win = pnl_pct > 0

        # Record trade
        record = TradeRecord(
            formula_id=formula_id,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=direction,
            pnl_pct=pnl_pct,
            was_win=was_win
        )
        self.trade_history.append(record)

        # Initialize if needed
        if formula_id not in self.formula_states:
            self.initialize_formula(formula_id)

        state = self.formula_states[formula_id]

        # Update counts
        state.live_trades += 1
        if was_win:
            state.live_wins += 1
        else:
            state.live_losses += 1
        state.live_total_return += pnl_pct
        state.last_trade_time = exit_time

        # Update SPRT
        self._update_sprt(state, was_win)

        # Update EMA win rates
        self._update_ema(state, was_win)

        # Update CUSUM
        self._update_cusum(state, pnl_pct)

        # Determine status
        old_status = state.status
        new_status = self._determine_status(state)

        if new_status != old_status:
            state.status = new_status
            state.last_status_change = time.time()
            logger.warning(
                f"Formula {formula_id} status changed: {old_status.value} -> {new_status.value}"
            )

        # Persist state
        self._save_state()

        return state.status

    def _update_sprt(self, state: FormulaValidationState, was_win: bool):
        """
        Update Sequential Probability Ratio Test.

        H0: p = 0.5 (no edge)
        H1: p = p_backtest (edge exists)

        Log-likelihood ratio:
        LLR += log(p1/p0) if win else log((1-p1)/(1-p0))
        """
        p1 = state.baseline_win_rate  # Alternative hypothesis (backtest win rate)
        p0 = 0.5  # Null hypothesis (random)

        # Prevent division issues
        p1 = min(max(p1, 0.51), 0.99)

        if was_win:
            # Log-likelihood of win under H1 vs H0
            llr_increment = np.log(p1 / p0)
        else:
            # Log-likelihood of loss under H1 vs H0
            llr_increment = np.log((1 - p1) / (1 - p0))

        state.sprt_log_likelihood += llr_increment

        # Check thresholds
        if state.sprt_log_likelihood >= self.SPRT_UPPER:
            state.sprt_decision = "ACCEPT_H1"  # Edge confirmed
        elif state.sprt_log_likelihood <= self.SPRT_LOWER:
            state.sprt_decision = "REJECT_H1"  # Edge lost
        else:
            state.sprt_decision = "CONTINUE"

    def _update_ema(self, state: FormulaValidationState, was_win: bool):
        """Update exponential moving average win rates."""
        outcome = 1.0 if was_win else 0.0

        # 10-trade EMA (alpha = 0.1)
        alpha_10 = 2 / (10 + 1)
        state.ema_win_rate_10 = alpha_10 * outcome + (1 - alpha_10) * state.ema_win_rate_10

        # 20-trade EMA (alpha = 0.095)
        alpha_20 = 2 / (20 + 1)
        state.ema_win_rate_20 = alpha_20 * outcome + (1 - alpha_20) * state.ema_win_rate_20

    def _update_cusum(self, state: FormulaValidationState, pnl_pct: float):
        """
        Update CUSUM for edge decay detection.

        CUSUM = max(0, CUSUM - (actual_return - expected_return) / std)

        Positive CUSUM indicates cumulative underperformance.
        """
        expected_return = state.baseline_avg_return
        std = expected_return * 0.5  # Approximate std

        # Standardized deviation from expected
        deviation = (expected_return - pnl_pct) / std  # Positive if underperforming

        # Update CUSUM (one-sided, detecting underperformance)
        state.cusum_value = max(0, state.cusum_value + deviation)
        state.cusum_peak = max(state.cusum_peak, state.cusum_value)

    def _determine_status(self, state: FormulaValidationState) -> FormulaStatus:
        """
        Determine formula status based on all validation metrics.

        Priority (from most to least severe):
        1. SPRT rejects edge -> DISABLED
        2. CUSUM threshold exceeded -> DISABLED
        3. EMA below disable threshold -> DISABLED
        4. EMA below alert threshold -> ALERT
        5. Not enough trades -> CONFIRMING
        6. Otherwise -> ACTIVE
        """
        # Check for disable conditions
        if state.sprt_decision == "REJECT_H1":
            logger.warning(f"{state.formula_id}: SPRT rejected edge hypothesis")
            return FormulaStatus.DISABLED

        if state.cusum_value >= self.CUSUM_THRESHOLD:
            logger.warning(
                f"{state.formula_id}: CUSUM threshold exceeded "
                f"({state.cusum_value:.2f} >= {self.CUSUM_THRESHOLD})"
            )
            return FormulaStatus.DISABLED

        if state.live_trades >= 20 and state.ema_win_rate_20 < self.EMA_DISABLE_THRESHOLD:
            logger.warning(
                f"{state.formula_id}: EMA-20 below threshold "
                f"({state.ema_win_rate_20:.1%} < {self.EMA_DISABLE_THRESHOLD:.1%})"
            )
            return FormulaStatus.DISABLED

        # Check for alert conditions
        if state.live_trades >= 10 and state.ema_win_rate_10 < self.EMA_ALERT_THRESHOLD:
            return FormulaStatus.ALERT

        # Check if still confirming
        if state.live_trades < 5:
            return FormulaStatus.CONFIRMING

        # SPRT confirmed edge
        if state.sprt_decision == "ACCEPT_H1":
            return FormulaStatus.ACTIVE

        # Default to active if no red flags
        return FormulaStatus.ACTIVE

    def should_trade(self, formula_id: str) -> Tuple[bool, float]:
        """
        Check if formula should trade.

        Returns:
            (should_trade, position_multiplier)
        """
        if formula_id not in self.formula_states:
            # Unknown formula - start cautiously
            return True, 0.25

        state = self.formula_states[formula_id]

        if state.status == FormulaStatus.DISABLED:
            return False, 0.0

        return True, state.position_multiplier

    def get_status(self, formula_id: str) -> FormulaStatus:
        """Get current status of a formula."""
        if formula_id not in self.formula_states:
            return FormulaStatus.CONFIRMING
        return self.formula_states[formula_id].status

    def get_diagnostics(self, formula_id: str) -> Dict:
        """Get diagnostic information for a formula."""
        if formula_id not in self.formula_states:
            return {"error": "Formula not found"}

        state = self.formula_states[formula_id]
        return {
            "formula_id": formula_id,
            "status": state.status.value,
            "live_trades": state.live_trades,
            "live_win_rate": f"{state.live_win_rate:.1%}",
            "baseline_win_rate": f"{state.baseline_win_rate:.1%}",
            "sprt_llr": f"{state.sprt_log_likelihood:.3f}",
            "sprt_decision": state.sprt_decision,
            "ema_10": f"{state.ema_win_rate_10:.1%}",
            "ema_20": f"{state.ema_win_rate_20:.1%}",
            "cusum": f"{state.cusum_value:.2f}",
            "cusum_peak": f"{state.cusum_peak:.2f}",
            "position_multiplier": f"{state.position_multiplier:.2f}"
        }

    def get_all_diagnostics(self) -> Dict[str, Dict]:
        """Get diagnostics for all formulas."""
        return {
            fid: self.get_diagnostics(fid)
            for fid in self.formula_states
        }

    def reset_formula(self, formula_id: str):
        """Reset a formula's validation state (for testing)."""
        if formula_id in self.formula_states:
            baseline = self.formula_states[formula_id]
            self.formula_states[formula_id] = FormulaValidationState(
                formula_id=formula_id,
                baseline_win_rate=baseline.baseline_win_rate,
                baseline_avg_return=baseline.baseline_avg_return,
                baseline_trades=baseline.baseline_trades,
                status=FormulaStatus.CONFIRMING
            )
            self._save_state()

    def _save_state(self):
        """Persist state to file."""
        try:
            state_dict = {}
            for fid, state in self.formula_states.items():
                state_dict[fid] = {
                    "status": state.status.value,
                    "baseline_win_rate": state.baseline_win_rate,
                    "baseline_avg_return": state.baseline_avg_return,
                    "baseline_trades": state.baseline_trades,
                    "live_trades": state.live_trades,
                    "live_wins": state.live_wins,
                    "live_losses": state.live_losses,
                    "live_total_return": state.live_total_return,
                    "sprt_log_likelihood": state.sprt_log_likelihood,
                    "sprt_decision": state.sprt_decision,
                    "ema_win_rate_10": state.ema_win_rate_10,
                    "ema_win_rate_20": state.ema_win_rate_20,
                    "cusum_value": state.cusum_value,
                    "cusum_peak": state.cusum_peak,
                    "last_trade_time": state.last_trade_time,
                    "last_status_change": state.last_status_change,
                }

            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self):
        """Load state from file."""
        try:
            if not self.state_file.exists():
                return

            with open(self.state_file, 'r') as f:
                state_dict = json.load(f)

            for fid, data in state_dict.items():
                self.formula_states[fid] = FormulaValidationState(
                    formula_id=fid,
                    status=FormulaStatus(data["status"]),
                    baseline_win_rate=data["baseline_win_rate"],
                    baseline_avg_return=data["baseline_avg_return"],
                    baseline_trades=data["baseline_trades"],
                    live_trades=data["live_trades"],
                    live_wins=data["live_wins"],
                    live_losses=data["live_losses"],
                    live_total_return=data["live_total_return"],
                    sprt_log_likelihood=data["sprt_log_likelihood"],
                    sprt_decision=data["sprt_decision"],
                    ema_win_rate_10=data["ema_win_rate_10"],
                    ema_win_rate_20=data["ema_win_rate_20"],
                    cusum_value=data["cusum_value"],
                    cusum_peak=data["cusum_peak"],
                    last_trade_time=data["last_trade_time"],
                    last_status_change=data["last_status_change"],
                )

            logger.info(f"Loaded validation state for {len(self.formula_states)} formulas")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")


if __name__ == "__main__":
    # Test the validator
    print("Testing RenTech Validator")
    print("=" * 60)

    validator = RenTechValidator(state_file="data/test_validator_state.json")

    # Initialize a formula
    validator.initialize_formula(
        formula_id="RENTECH_001",
        baseline_win_rate=1.0,
        baseline_avg_return=26.37,
        baseline_trades=18
    )

    # Simulate winning trades
    print("\nSimulating 10 winning trades...")
    for i in range(10):
        status = validator.record_trade(
            formula_id="RENTECH_001",
            entry_time=time.time() - 1000 + i * 100,
            exit_time=time.time() - 900 + i * 100,
            entry_price=50000,
            exit_price=55000,
            direction=1,
            pnl_pct=10.0
        )
        print(f"  Trade {i+1}: WIN -> Status: {status.value}")

    diag = validator.get_diagnostics("RENTECH_001")
    print(f"\nDiagnostics after 10 wins:")
    for key, value in diag.items():
        print(f"  {key}: {value}")

    # Simulate some losses
    print("\nSimulating 5 losing trades...")
    for i in range(5):
        status = validator.record_trade(
            formula_id="RENTECH_001",
            entry_time=time.time() - 500 + i * 100,
            exit_time=time.time() - 400 + i * 100,
            entry_price=50000,
            exit_price=47500,
            direction=1,
            pnl_pct=-5.0
        )
        print(f"  Trade {i+1}: LOSS -> Status: {status.value}")

    diag = validator.get_diagnostics("RENTECH_001")
    print(f"\nDiagnostics after 5 losses:")
    for key, value in diag.items():
        print(f"  {key}: {value}")

    # Check should_trade
    should_trade, mult = validator.should_trade("RENTECH_001")
    print(f"\nShould trade: {should_trade}, position multiplier: {mult:.2f}")
