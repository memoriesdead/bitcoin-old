#!/usr/bin/env python3
"""CONFIGURATION - Single source of truth"""

import os
from dataclasses import dataclass, field
from typing import Set
from pathlib import Path


def _get_data_dir() -> Path:
    if os.path.exists('/root/sovereign'):
        return Path('/root/sovereign')
    return Path(__file__).parent.parent.parent / 'data'


@dataclass
class TradingConfig:
    # === CORRELATION ===
    min_correlation: float = 0.7
    min_sample_size: int = 10
    min_win_rate: float = 0.9
    correlation_window_minutes: int = 5

    # === POSITION ===
    initial_capital: float = 100.0
    max_leverage: int = 125
    max_positions: int = 4
    position_size_pct: float = 0.25

    # === EXIT ===
    exit_timeout_seconds: float = 300.0
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02

    # === EXCHANGES ===
    tradeable_exchanges: Set[str] = field(default_factory=lambda: {
        'coinbase', 'kraken', 'bitstamp', 'gemini', 'crypto.com'
    })
    exchange_fees: dict = field(default_factory=lambda: {
        'coinbase': 0.006, 'kraken': 0.0026, 'bitstamp': 0.005,
        'gemini': 0.004, 'crypto.com': 0.004, 'binance': 0.001, 'default': 0.005
    })

    # === FLOW ===
    min_flow_btc: float = 0.0
    flow_buckets: tuple = ((0,1),(1,5),(5,10),(10,50),(50,100),(100,500),(500,float('inf')))

    # === PATHS ===
    data_dir: Path = field(default_factory=_get_data_dir)

    @property
    def correlation_db_path(self) -> str:
        return str(self.data_dir / 'correlation.db')

    @property
    def addresses_db_path(self) -> str:
        return str(self.data_dir / 'walletexplorer_addresses.db')

    @property
    def trades_db_path(self) -> str:
        return str(self.data_dir / 'trades.db')

    @property
    def cpp_runner_path(self) -> str:
        return '/root/sovereign/cpp_runner/build/blockchain_runner' if os.path.exists('/root/sovereign') else ''

    def get_fee(self, exchange: str) -> float:
        return self.exchange_fees.get(exchange, self.exchange_fees['default'])

    def get_bucket(self, flow_btc: float) -> tuple:
        for low, high in self.flow_buckets:
            if low <= flow_btc < high:
                return (low, high)
        return self.flow_buckets[-1]

    def is_tradeable(self, exchange: str) -> bool:
        return exchange.lower() in self.tradeable_exchanges


CONFIG = TradingConfig()

def get_config() -> TradingConfig:
    return CONFIG
