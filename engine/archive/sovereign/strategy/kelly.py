"""
Kelly Criterion Position Sizer
ID 6003: Optimal position sizing with uncertainty adjustment

Enhanced for RenTech formulas:
- Uncertainty-adjusted Kelly (scales with live trade count)
- Drawdown-aware reduction
- Per-formula limits
"""

import numpy as np

# Kelly Constants
MAX_KELLY_FRACTION = 0.25  # Quarter-Kelly
UNCERTAINTY_TRADES = 50     # Trades needed for full confidence


class KellySizer:
    """
    ID 6003: Kelly Criterion for Optimal Position Sizing

    Formula: f* = (b*p - q) / b
    Where:
        b = odds (risk/reward ratio)
        p = win probability
        q = 1 - p (loss probability)

    Enhanced with:
    - Uncertainty adjustment based on live trade count
    - Drawdown-aware position reduction
    - Quarter-Kelly for safety
    """

    def __init__(self, max_fraction: float = MAX_KELLY_FRACTION):
        self.max_fraction = max_fraction
        self.current_drawdown = 0.0
        self.max_drawdown_limit = 0.10  # 10% max DD

    def calculate(self, probability: float, risk_reward: float = 1.0) -> float:
        """
        Calculate Kelly fraction.

        Args:
            probability: Win probability (0.5-1.0)
            risk_reward: Risk/reward ratio (default 1:1)

        Returns:
            Position size as fraction of capital (0.0 - max_fraction)
        """
        if probability <= 0.5:
            return 0.0

        q = 1.0 - probability
        kelly = (risk_reward * probability - q) / risk_reward

        # Quarter Kelly for safety
        quarter_kelly = kelly * 0.25

        return max(0.0, min(quarter_kelly, self.max_fraction))

    def calculate_with_uncertainty(
        self,
        probability: float,
        risk_reward: float = 1.0,
        n_live_trades: int = 0,
        formula_multiplier: float = 1.0
    ) -> float:
        """
        Calculate uncertainty-adjusted Kelly fraction.

        Args:
            probability: Win probability (0.5-1.0)
            risk_reward: Risk/reward ratio (default 1:1)
            n_live_trades: Number of live trades for this formula
            formula_multiplier: Per-formula position multiplier (from validator)

        Returns:
            Position size as fraction of capital
        """
        if probability <= 0.5:
            return 0.0

        # Base Kelly calculation
        q = 1.0 - probability
        kelly = (risk_reward * probability - q) / risk_reward

        # 1. Quarter Kelly for safety
        quarter_kelly = kelly * 0.25

        # 2. Uncertainty adjustment: scale up as live trades accumulate
        # f_adjusted = f_kelly * sqrt(n_live / (n_live + 50))
        # Minimum 10% scaling for paper trading / initial trades
        uncertainty_factor = max(0.1, np.sqrt(n_live_trades / (n_live_trades + UNCERTAINTY_TRADES)))
        adjusted = quarter_kelly * uncertainty_factor

        # 3. Formula-specific multiplier (from validator)
        adjusted *= formula_multiplier

        # 4. Drawdown-aware reduction
        # f_dd = f * (1 - current_dd / max_dd)^2
        if self.current_drawdown > 0:
            dd_ratio = self.current_drawdown / self.max_drawdown_limit
            dd_factor = (1 - min(dd_ratio, 1.0)) ** 2
            adjusted *= dd_factor

        return max(0.0, min(adjusted, self.max_fraction))

    def update_drawdown(self, current_dd: float):
        """Update current drawdown for position sizing."""
        self.current_drawdown = max(0.0, current_dd)


class RenTechSizer:
    """
    Position sizer specifically for RenTech formulas.

    Handles:
    - Per-formula max position limits
    - Uncertainty-adjusted sizing
    - Drawdown-aware reduction
    - Leverage calculations
    """

    # Per-formula limits (fraction of capital)
    FORMULA_LIMITS = {
        "RENTECH_001": 0.05,  # EXTREME_ANOMALY_LONG
        "RENTECH_002": 0.04,  # VOLUME_MOMENTUM_CONFLUENCE
        "RENTECH_003": 0.05,  # EXTREME_ANOMALY_LONG_7D
        "RENTECH_004": 0.04,  # CORRELATION_BREAK_BULL
        "RENTECH_005": 0.04,  # BOLLINGER_BOUNCE
        "RENTECH_006": 0.04,  # WHALE_ACCUMULATION
        "RENTECH_007": 0.04,  # RSI_DIVERGENCE
        "RENTECH_008": 0.06,  # MULTI_Z_EXTREME (highest return)
        "RENTECH_009": 0.03,  # EUPHORIA_EXIT_SHORT (contrarian)
    }

    def __init__(self, base_leverage: float = 10.0):
        """
        Initialize RenTech sizer.

        Args:
            base_leverage: Base leverage to use (10x for safety)
        """
        self.base_leverage = base_leverage
        self.kelly_sizer = KellySizer()

    def calculate_position_size(
        self,
        formula_id: str,
        capital: float,
        win_rate: float,
        avg_return: float,
        n_live_trades: int = 0,
        validator_multiplier: float = 1.0,
        current_drawdown: float = 0.0
    ) -> dict:
        """
        Calculate position size for a RenTech formula.

        Args:
            formula_id: Formula ID (RENTECH_001, etc.)
            capital: Available capital
            win_rate: Historical or live win rate
            avg_return: Historical or live average return
            n_live_trades: Number of live trades
            validator_multiplier: Multiplier from validator
            current_drawdown: Current drawdown fraction

        Returns:
            dict with position_size, leverage, max_loss, etc.
        """
        # Get formula-specific limit
        max_fraction = self.FORMULA_LIMITS.get(formula_id, 0.03)

        # Calculate risk/reward (approximation)
        # Assuming stop loss at 2% for 50x-safe trades
        stop_loss_pct = 2.0
        risk_reward = avg_return / stop_loss_pct

        # Update Kelly sizer with current drawdown
        self.kelly_sizer.update_drawdown(current_drawdown)

        # Calculate Kelly fraction
        kelly_fraction = self.kelly_sizer.calculate_with_uncertainty(
            probability=win_rate,
            risk_reward=risk_reward,
            n_live_trades=n_live_trades,
            formula_multiplier=validator_multiplier
        )

        # Apply formula-specific cap
        position_fraction = min(kelly_fraction, max_fraction)

        # Calculate actual position size
        position_size = capital * position_fraction

        # Calculate effective leverage
        # With 50x-safe trades, max DD is 2%, so max leverage is 50x
        # But we use base_leverage for safety
        effective_leverage = self.base_leverage

        # Max loss calculation
        max_loss = position_size * (stop_loss_pct / 100) * effective_leverage

        return {
            "formula_id": formula_id,
            "position_size": position_size,
            "position_fraction": position_fraction,
            "kelly_fraction": kelly_fraction,
            "effective_leverage": effective_leverage,
            "max_loss": max_loss,
            "max_loss_pct": (max_loss / capital) * 100 if capital > 0 else 0,
            "uncertainty_factor": np.sqrt(n_live_trades / (n_live_trades + UNCERTAINTY_TRADES)),
            "validator_multiplier": validator_multiplier,
        }


if __name__ == "__main__":
    # Test the sizers
    print("Testing Kelly and RenTech Sizers")
    print("=" * 60)

    # Basic Kelly
    kelly = KellySizer()
    print("\nBasic Kelly:")
    for wr in [0.6, 0.7, 0.8, 0.9, 1.0]:
        frac = kelly.calculate(wr, risk_reward=2.0)
        print(f"  WR={wr:.0%}, R:R=2:1 -> Position: {frac:.2%}")

    # Uncertainty-adjusted Kelly
    print("\nUncertainty-adjusted Kelly (WR=80%, R:R=2:1):")
    for n_trades in [0, 5, 10, 20, 50, 100]:
        frac = kelly.calculate_with_uncertainty(0.8, 2.0, n_trades)
        print(f"  {n_trades} trades -> Position: {frac:.2%}")

    # RenTech Sizer
    print("\nRenTech Sizer (capital=$1000):")
    sizer = RenTechSizer(base_leverage=10.0)

    for formula_id in ["RENTECH_001", "RENTECH_008", "RENTECH_009"]:
        result = sizer.calculate_position_size(
            formula_id=formula_id,
            capital=1000,
            win_rate=1.0,
            avg_return=25.0,
            n_live_trades=10,
            validator_multiplier=1.0
        )
        print(f"\n  {formula_id}:")
        print(f"    Position: ${result['position_size']:.2f} ({result['position_fraction']:.1%})")
        print(f"    Kelly: {result['kelly_fraction']:.2%}")
        print(f"    Max loss: ${result['max_loss']:.2f} ({result['max_loss_pct']:.1f}%)")
