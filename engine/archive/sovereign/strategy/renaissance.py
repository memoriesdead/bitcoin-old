"""
Renaissance Compounding Integration
ID 810: RenaissanceMasterController wrapper for sovereign engine
Max lines: 150
"""
import time
import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING
from collections import deque

if TYPE_CHECKING:
    from ..ai.claude_adapter import ClaudeAdapter


class RenaissanceTracker:
    """
    Renaissance-style compound growth tracker.

    Based on Formula IDs 801-810 from renaissance_compounding.py:
    - Master Growth Equation: Capital(t) = Capital(0) × (1 + f × edge)^n
    - Quarter-Kelly: f = 0.25 (75% growth, 6.25% variance)
    - Thorp's growth: g = r + S²/2

    References:
        Kelly (1956) Bell System Technical Journal
        Thorp (2007) The Kelly Criterion
        Cont & Stoikov (2014) J. Financial Econometrics
    """

    FORMULA_ID = 810

    def __init__(
        self,
        initial_capital: float = 100.0,
        target_multiplier: float = 100.0,  # 100x target
        kelly_fraction: float = 0.25,
        max_drawdown: float = 0.20,
        claude: "ClaudeAdapter" = None,
    ):
        self.initial_capital = initial_capital
        self.target_capital = initial_capital * target_multiplier
        self.kelly_fraction = kelly_fraction
        self.max_drawdown = max_drawdown
        self.claude = claude

        # State
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.consecutive_losses = 0

        # History for Sharpe calculation
        self.returns: deque = deque(maxlen=1000)
        self.pnl_history: deque = deque(maxlen=10000)
        self.equity_curve: deque = deque(maxlen=10000)

        # Timing
        self.start_time = time.time()

    def record_trade(self, pnl: float, new_capital: float) -> Dict[str, Any]:
        """Record a trade and update all metrics."""
        # Calculate return percentage
        if self.current_capital > 0:
            return_pct = pnl / self.current_capital
            self.returns.append(return_pct)

        # Update capital
        self.current_capital = new_capital
        self.total_pnl += pnl
        self.trade_count += 1

        # Track wins/losses
        if pnl > 0:
            self.wins += 1
            self.consecutive_losses = 0
        elif pnl < 0:
            self.losses += 1
            self.consecutive_losses += 1

        # Update peak (for drawdown)
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

        # Store history
        self.pnl_history.append(pnl)
        self.equity_curve.append(new_capital)

        return self.get_progress()

    def get_win_rate(self) -> float:
        """Current win rate."""
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    def get_drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_capital <= 0:
            return 0.0
        return (self.peak_capital - self.current_capital) / self.peak_capital

    def get_sharpe(self, annualization: float = 365 * 24) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(self.returns) < 10:
            return 0.0

        returns = np.array(self.returns)
        mu = np.mean(returns) * annualization
        sigma = np.std(returns) * np.sqrt(annualization)

        if sigma <= 0:
            return 0.0
        return mu / sigma

    def get_compound_factor(self) -> float:
        """Current compound multiple."""
        return self.current_capital / self.initial_capital

    def get_progress_pct(self) -> float:
        """Progress toward target (log scale)."""
        if self.current_capital <= self.initial_capital:
            return 0.0

        log_progress = np.log(self.current_capital / self.initial_capital)
        log_target = np.log(self.target_capital / self.initial_capital)
        return (log_progress / log_target) * 100

    def get_edge_per_trade(self) -> float:
        """Average edge per trade."""
        if not self.pnl_history or self.current_capital <= 0:
            return 0.0
        return np.mean(list(self.pnl_history)) / self.current_capital

    def trades_to_target(self) -> int:
        """Estimated trades remaining to target."""
        edge = self.get_edge_per_trade()
        if edge <= 0:
            return float('inf')

        growth_per_trade = 1 + self.kelly_fraction * edge
        if growth_per_trade <= 1:
            return float('inf')

        remaining = self.target_capital / self.current_capital
        return int(np.ceil(np.log(remaining) / np.log(growth_per_trade)))

    def should_reduce_size(self) -> bool:
        """Check if drawdown requires size reduction."""
        return self.get_drawdown() > self.max_drawdown * 0.5

    def get_adjusted_kelly(self) -> float:
        """Get drawdown-adjusted Kelly fraction with Claude AI risk assessment."""
        # Base adjustment from drawdown
        dd = self.get_drawdown()
        base_kelly = self.kelly_fraction
        if dd > self.max_drawdown * 0.5:
            scale = 1 - (dd / self.max_drawdown)
            base_kelly = self.kelly_fraction * max(scale, 0.1)

        # CLAUDE AI RISK ASSESSMENT (if enabled)
        if self.claude:
            assessment = self.claude.assess_risk({
                'kelly': base_kelly,
                'drawdown': dd,
                'win_rate': self.get_win_rate() * 100,
                'sharpe': self.get_sharpe(),
                'consecutive_losses': self.consecutive_losses,
                'trades_today': self.trade_count,  # Simplified - would track daily
            })
            if assessment.success and assessment.action == "ADJUST":
                adjusted = base_kelly * assessment.size_adjustment
                if assessment.size_adjustment != 1.0:
                    print(f"[RENAISSANCE] Claude Kelly adjustment: "
                          f"{base_kelly:.3f} -> {adjusted:.3f} ({assessment.reasoning})")
                return max(0.01, min(0.5, adjusted))  # Clamp to valid range

        return base_kelly

    def get_progress(self) -> Dict[str, Any]:
        """Get comprehensive progress report."""
        runtime = time.time() - self.start_time
        trades_per_hour = (self.trade_count / runtime) * 3600 if runtime > 0 else 0

        return {
            # Capital
            'initial': self.initial_capital,
            'current': self.current_capital,
            'target': self.target_capital,
            'compound': self.get_compound_factor(),
            'progress_pct': self.get_progress_pct(),

            # Performance
            'trades': self.trade_count,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.get_win_rate() * 100,
            'total_pnl': self.total_pnl,
            'edge': self.get_edge_per_trade() * 100,

            # Risk
            'drawdown': self.get_drawdown() * 100,
            'sharpe': self.get_sharpe(),
            'kelly': self.get_adjusted_kelly(),

            # Projection
            'trades_remaining': self.trades_to_target(),
            'trades_per_hour': trades_per_hour,

            # Status
            'at_target': self.current_capital >= self.target_capital,
            'in_drawdown': self.should_reduce_size(),
        }

    def print_status(self):
        """Print formatted status."""
        p = self.get_progress()
        print(f"\n{'='*60}")
        print(f"RENAISSANCE TRACKER - ID {self.FORMULA_ID}")
        print(f"{'='*60}")
        print(f"Capital: ${p['current']:.4f} / ${p['target']:.0f} ({p['progress_pct']:.2f}%)")
        print(f"Compound: {p['compound']:.4f}x | Trades: {p['trades']}")
        print(f"Win Rate: {p['win_rate']:.1f}% | Edge: {p['edge']:.3f}%")
        print(f"Drawdown: {p['drawdown']:.2f}% | Sharpe: {p['sharpe']:.2f}")
        print(f"Trades to target: {p['trades_remaining']:,}")
        print(f"{'='*60}\n")
