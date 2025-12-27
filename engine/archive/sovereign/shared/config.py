#!/usr/bin/env python3
"""
TRADING CONFIG
==============
Single source of truth. Simple.
"""

from dataclasses import dataclass, field
from typing import Set, Dict


@dataclass
class Config:
    """Trading configuration."""

    # Capital
    initial_capital: float = 100.0
    max_leverage: int = 125
    position_size_pct: float = 0.25  # 25% per trade

    # Risk
    stop_loss_pct: float = 0.01      # 1%
    take_profit_pct: float = 0.02    # 2%
    exit_timeout_seconds: int = 300   # 5 min

    # Signal thresholds
    min_flow_btc: float = 10.0       # Minimum flow to trade
    min_correlation: float = 0.7      # 70% correlation
    min_win_rate: float = 0.9         # 90% win rate
    min_samples: int = 10             # Statistical significance

    # Exchanges
    tradeable: Set[str] = field(default_factory=lambda: {
        'coinbase', 'kraken', 'bitstamp', 'gemini'
    })

    fees: Dict[str, float] = field(default_factory=lambda: {
        'coinbase': 0.006,
        'kraken': 0.0026,
        'bitstamp': 0.005,
        'gemini': 0.004,
        'default': 0.005
    })

    # VPS paths
    db_path: str = "/root/sovereign/trades.db"
    cpp_runner: str = "/root/sovereign/cpp_runner/build/blockchain_runner"
    zmq_endpoint: str = "tcp://127.0.0.1:28332"

    def get_fee(self, exchange: str) -> float:
        return self.fees.get(exchange.lower(), self.fees['default'])


CONFIG = Config()
