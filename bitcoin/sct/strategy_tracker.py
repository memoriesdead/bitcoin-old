#!/usr/bin/env python3
"""
Strategy Tracker - Track Multiple Strategies and Their Certainty Levels

Real-time tracking of strategy performance with Wilson CI updates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json
import time

from .config import SCTConfig, get_config
from .wilson import wilson_lower_bound, wilson_interval
from .certainty import CertaintyChecker, CertaintyStatus


@dataclass
class StrategyStats:
    """Statistics for a single strategy."""
    name: str
    wins: int = 0
    losses: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def total(self) -> int:
        """Total trades."""
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        """Observed win rate."""
        return self.wins / self.total if self.total > 0 else 0.0

    @property
    def lower_bound(self) -> float:
        """Lower bound of 99% Wilson CI."""
        return wilson_lower_bound(self.wins, self.total, 0.99)

    @property
    def upper_bound(self) -> float:
        """Upper bound of 99% Wilson CI."""
        _, upper = wilson_interval(self.wins, self.total, 0.99)
        return upper

    @property
    def is_certain(self) -> bool:
        """Check if certain of 50.75%+ win rate."""
        return self.lower_bound >= 0.5075

    @property
    def status(self) -> CertaintyStatus:
        """Get certainty status."""
        if self.total == 0:
            return CertaintyStatus.NEED_MORE_DATA
        if self.is_certain:
            return CertaintyStatus.CERTAIN
        if self.win_rate >= 0.5075:
            return CertaintyStatus.NEED_MORE_DATA
        return CertaintyStatus.NO_EDGE

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'wins': self.wins,
            'losses': self.losses,
            'total': self.total,
            'win_rate': self.win_rate,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'is_certain': self.is_certain,
            'status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'StrategyStats':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            wins=data['wins'],
            losses=data['losses'],
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
        )


class StrategyTracker:
    """
    Track multiple strategies and their certainty levels.

    Records trade outcomes and maintains Wilson CI in real-time.
    """

    def __init__(self, config: Optional[SCTConfig] = None,
                 storage_path: Optional[Path] = None):
        """
        Initialize strategy tracker.

        Args:
            config: SCT configuration
            storage_path: Path to persist strategy data
        """
        self.config = config or get_config()
        self.strategies: Dict[str, StrategyStats] = {}
        self.checker = CertaintyChecker(self.config)

        self.storage_path = storage_path or Path(__file__).parent / "strategies.json"
        self._load()

    def _load(self):
        """Load strategies from storage."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for name, stats in data.get('strategies', {}).items():
                    self.strategies[name] = StrategyStats.from_dict(stats)
            except Exception:
                pass  # Start fresh if corrupt

    def _save(self):
        """Save strategies to storage."""
        data = {
            'strategies': {name: s.to_dict() for name, s in self.strategies.items()},
            'updated_at': time.time(),
        }
        self.storage_path.write_text(json.dumps(data, indent=2))

    def get_or_create(self, name: str) -> StrategyStats:
        """Get or create strategy stats."""
        if name not in self.strategies:
            self.strategies[name] = StrategyStats(name=name)
            self._save()
        return self.strategies[name]

    def record_trade(self, strategy: str, won: bool) -> StrategyStats:
        """
        Record a trade outcome.

        Args:
            strategy: Strategy name
            won: True if trade won

        Returns:
            Updated StrategyStats
        """
        stats = self.get_or_create(strategy)

        if won:
            stats.wins += 1
        else:
            stats.losses += 1

        stats.updated_at = time.time()
        self._save()

        return stats

    def get_tradeable_strategies(self) -> List[StrategyStats]:
        """
        Get strategies where we're certain of 50.75%+ win rate.

        Returns:
            List of strategies that pass certainty check
        """
        return [s for s in self.strategies.values() if s.is_certain]

    def get_pending_strategies(self) -> List[StrategyStats]:
        """Get strategies that need more data."""
        return [s for s in self.strategies.values()
                if s.status == CertaintyStatus.NEED_MORE_DATA]

    def get_no_edge_strategies(self) -> List[StrategyStats]:
        """Get strategies with no proven edge."""
        return [s for s in self.strategies.values()
                if s.status == CertaintyStatus.NO_EDGE]

    def print_status(self):
        """Print status of all strategies."""
        print(f"\n{'='*70}")
        print("STRATEGY TRACKER STATUS")
        print(f"{'='*70}")

        if not self.strategies:
            print("\nNo strategies tracked yet.")
            return

        # Sort by status then lower_bound
        sorted_strats = sorted(
            self.strategies.values(),
            key=lambda s: (0 if s.is_certain else 1, -s.lower_bound)
        )

        print(f"\n{'Strategy':<20} {'Trades':>8} {'WR':>8} {'Lower':>8} {'Status':<12}")
        print("-" * 70)

        for s in sorted_strats:
            status_str = 'CERTAIN' if s.is_certain else s.status.value.upper()
            print(f"{s.name:<20} {s.total:>8} {s.win_rate*100:>7.1f}% "
                  f"{s.lower_bound*100:>7.2f}% {status_str:<12}")

        tradeable = len(self.get_tradeable_strategies())
        pending = len(self.get_pending_strategies())
        no_edge = len(self.get_no_edge_strategies())

        print(f"\nSummary: {tradeable} tradeable, {pending} pending, {no_edge} no edge")
        print(f"{'='*70}")


if __name__ == "__main__":
    tracker = StrategyTracker()

    # Demo: Record some trades
    for _ in range(60):
        tracker.record_trade("FLOW_MOMENTUM", won=True)
    for _ in range(40):
        tracker.record_trade("FLOW_MOMENTUM", won=False)

    for _ in range(55):
        tracker.record_trade("BREAKOUT", won=True)
    for _ in range(45):
        tracker.record_trade("BREAKOUT", won=False)

    tracker.print_status()
