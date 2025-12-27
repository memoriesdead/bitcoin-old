#!/usr/bin/env python3
"""
SINGLE SOURCE OF TRUTH CONFIGURATION
=====================================

ALL trading configuration in one place. No scattered settings.

THE GOAL: 100% deterministic trading per exchange.
"""

from dataclasses import dataclass, field
from typing import Set


@dataclass
class TradingConfig:
    """
    Complete trading configuration.

    MATHEMATICAL APPROACH:
    - No arbitrary thresholds like "10 BTC minimum"
    - Let data speak through statistical correlation
    - Only trade patterns with proven accuracy
    """

    # ==========================================================================
    # FLOW DETECTION - NO ARBITRARY THRESHOLDS
    # ==========================================================================
    min_flow_btc: float = 0.0  # Record ALL flows, let math decide

    # ==========================================================================
    # CORRELATION SETTINGS - THE MATHEMATICAL APPROACH
    # ==========================================================================
    correlation_window_minutes: int = 5      # Track price impact over 5 min
    min_correlation: float = 0.7             # Only trade patterns with 70%+ correlation
    min_sample_size: int = 10                # Need 10+ samples for statistical significance
    min_win_rate: float = 0.9                # 90%+ win rate required to trade pattern

    # Price tracking intervals (in seconds)
    price_check_intervals: tuple = (60, 300, 600)  # T+1min, T+5min, T+10min

    # Flow size buckets for correlation analysis
    flow_buckets: tuple = (
        (0, 1),        # 0-1 BTC
        (1, 5),        # 1-5 BTC
        (5, 10),       # 5-10 BTC
        (10, 50),      # 10-50 BTC
        (50, 100),     # 50-100 BTC
        (100, 500),    # 100-500 BTC
        (500, float('inf')),  # 500+ BTC (whale)
    )

    # ==========================================================================
    # POSITION MANAGEMENT
    # ==========================================================================
    initial_capital: float = 100.0           # Starting capital in USD
    max_leverage: int = 20                   # REDUCED from 125x until edge proven (2024-12-26)
    max_positions: int = 4                   # Maximum concurrent positions
    position_size_pct: float = 0.25          # 25% of capital per position

    # ==========================================================================
    # EXIT STRATEGY - TIME-BASED, NOT FLOW REVERSAL
    # ==========================================================================
    exit_timeout_seconds: float = 300.0      # 5 minute time exit
    stop_loss_pct: float = 0.01              # 1% stop loss
    take_profit_pct: float = 0.02            # 2% take profit

    # ==========================================================================
    # EXCHANGES
    # ==========================================================================
    tradeable_exchanges: Set[str] = field(default_factory=lambda: {
        'coinbase', 'kraken', 'bitstamp', 'gemini', 'crypto.com'
    })

    # Exchanges with known good correlation (from historical analysis)
    high_correlation_exchanges: Set[str] = field(default_factory=lambda: {
        'coinbase', 'kraken', 'bitstamp', 'gemini'
    })

    # Per-exchange fee structure
    exchange_fees: dict = field(default_factory=lambda: {
        'coinbase': 0.006,     # 0.6%
        'kraken': 0.0026,      # 0.26%
        'bitstamp': 0.005,     # 0.5%
        'gemini': 0.004,       # 0.4%
        'crypto.com': 0.004,   # 0.4%
        'binance': 0.001,      # 0.1%
        'default': 0.005       # 0.5% default
    })

    # ==========================================================================
    # DATABASE PATHS
    # ==========================================================================
    correlation_db_path: str = "/root/sovereign/correlation.db"
    addresses_db_path: str = "/root/sovereign/walletexplorer_addresses.db"
    utxo_db_path: str = "/root/sovereign/exchange_utxos.db"
    trades_db_path: str = "/root/sovereign/trades.db"

    # ==========================================================================
    # ZMQ / BITCOIN CORE
    # ==========================================================================
    zmq_endpoint: str = "tcp://127.0.0.1:28332"
    bitcoin_cli_path: str = "/usr/local/bin/bitcoin-cli"

    # ==========================================================================
    # C++ RUNNER
    # ==========================================================================
    cpp_runner_path: str = "/root/sovereign/cpp_runner/build/blockchain_runner"

    # ==========================================================================
    # LOGGING
    # ==========================================================================
    log_all_flows: bool = True               # Log every flow for analysis
    log_signals_only: bool = False           # Only log when signal generated

    def get_fee(self, exchange: str) -> float:
        """Get fee for exchange."""
        return self.exchange_fees.get(exchange, self.exchange_fees['default'])

    def get_bucket(self, flow_btc: float) -> tuple:
        """Get the bucket for a flow size."""
        for low, high in self.flow_buckets:
            if low <= flow_btc < high:
                return (low, high)
        return self.flow_buckets[-1]  # Whale bucket

    def is_tradeable(self, exchange: str) -> bool:
        """Check if exchange is in tradeable set."""
        return exchange.lower() in self.tradeable_exchanges


# Global config instance
CONFIG = TradingConfig()


def get_config() -> TradingConfig:
    """Get the global configuration."""
    return CONFIG
