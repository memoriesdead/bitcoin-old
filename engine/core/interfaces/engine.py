"""
ENGINE INTERFACE - Base class for all trading engines
=====================================================
All engines must implement this interface.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class IEngine(ABC):
    """
    Base interface for all trading engines.

    Engines orchestrate formula execution, position management,
    and trade execution.
    """

    @abstractmethod
    def initialize(self, capital: float, **kwargs) -> None:
        """
        Initialize the engine with starting capital.

        Args:
            capital: Starting capital in USD
            **kwargs: Engine-specific initialization parameters
        """
        pass

    @abstractmethod
    def process_tick(self, price: float, timestamp: int) -> Dict[str, Any]:
        """
        Process a single tick of price data.

        Args:
            price: Current price
            timestamp: Timestamp in nanoseconds

        Returns:
            Dict containing:
            - 'trades': Number of trades executed
            - 'pnl': P&L from this tick
            - 'capital': Current capital
            - 'signals': Dict of formula signals
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get current engine state.

        Returns:
            Dict containing all relevant state information.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Gracefully shutdown the engine.
        Close positions, save state, etc.
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get trading statistics.

        Returns:
            Dict containing:
            - 'total_trades': Total trades executed
            - 'win_rate': Winning trade percentage
            - 'total_pnl': Total P&L
            - 'sharpe': Sharpe ratio
            - 'max_drawdown': Maximum drawdown
        """
        state = self.get_state()
        trades = state.get('total_trades', 0)
        wins = state.get('total_wins', 0)
        pnl = state.get('total_pnl', 0.0)

        return {
            'total_trades': trades,
            'win_rate': wins / trades if trades > 0 else 0.0,
            'total_pnl': pnl,
            'sharpe': 0.0,  # Implement in subclass
            'max_drawdown': 0.0,  # Implement in subclass
        }
