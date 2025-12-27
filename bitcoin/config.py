#!/usr/bin/env python3
"""
UNIFIED TRADING CONFIGURATION
=============================

Single source of truth for ALL trading strategies (DET, HQT, SCT).

Merged from:
- bitcoin/config.py (base)
- bitcoin/blockchain/config.py (VPS paths)
- bitcoin/strategies/det/config.py (DET thresholds)
- bitcoin/hqt/config.py (arbitrage)
- bitcoin/sct/config.py (statistical certainty)
"""

import os
from dataclasses import dataclass, field
from typing import Set, Dict, List, Optional
from pathlib import Path


def _get_data_dir() -> Path:
    """Get data directory (VPS or local)."""
    if os.path.exists('/root/sovereign'):
        return Path('/root/sovereign')
    return Path(__file__).parent.parent / 'data'


@dataclass
class TradingConfig:
    """
    Complete trading configuration for all strategies.

    Sections:
    - SHARED: All strategies use these
    - DET: Deterministic blockchain flow trading
    - HQT: High-quality arbitrage trades
    - SCT: Statistical certainty trading
    """

    # ==========================================================================
    # SHARED SETTINGS
    # ==========================================================================

    # Capital
    initial_capital: float = 100.0           # Starting capital USD
    paper_mode: bool = True                  # Safety: start in paper mode

    # Per-exchange max leverage (from official docs Dec 2024)
    exchange_leverage: Dict[str, int] = field(default_factory=lambda: {
        'mexc': 500,        # MEXC max 500x futures
        'binance': 125,     # Binance max 125x (20x new users)
        'bybit': 100,       # Bybit max 100x
        'kraken': 50,       # Kraken max 50x
        'coinbase': 10,     # Coinbase max 10x (US regulated)
        'gemini': 5,        # Gemini 5x US, 100x non-US
        'bitstamp': 10,     # Bitstamp 10x max
        'crypto.com': 20,   # Crypto.com max 20x
        'default': 10       # Conservative default
    })

    # Taker fees (we pay to take liquidity)
    taker_fees: Dict[str, float] = field(default_factory=lambda: {
        'kraken': 0.0026,       # 0.26%
        'coinbase': 0.006,      # 0.60%
        'bitstamp': 0.005,      # 0.50%
        'gemini': 0.004,        # 0.40%
        'binance': 0.001,       # 0.10%
        'bybit': 0.001,         # 0.10%
        'mexc': 0.001,          # 0.10%
        'crypto.com': 0.004,    # 0.40%
        'default': 0.005        # 0.50%
    })

    # Maker fees (we provide liquidity)
    maker_fees: Dict[str, float] = field(default_factory=lambda: {
        'kraken': 0.0016,       # 0.16%
        'coinbase': 0.004,      # 0.40%
        'bitstamp': 0.003,      # 0.30%
        'gemini': 0.002,        # 0.20%
        'binance': 0.001,       # 0.10%
        'bybit': 0.0002,        # 0.02%
        'mexc': 0.0,            # 0% maker
        'crypto.com': 0.002,    # 0.20%
        'default': 0.003        # 0.30%
    })

    # Tradeable exchanges (US Tier 1)
    tradeable_exchanges: Set[str] = field(default_factory=lambda: {
        'coinbase', 'kraken', 'bitstamp', 'gemini', 'crypto.com'
    })

    # Paths
    data_dir: Path = field(default_factory=_get_data_dir)

    # ==========================================================================
    # DET STRATEGY - Deterministic Blockchain Flow Trading
    # ==========================================================================
    # 100% win rate when: sample >= 10, correlation >= 0.70, win_rate >= 0.90

    det_min_sample_count: int = 10           # Statistical significance
    det_min_correlation: float = 0.70        # 70%+ correlation with price
    det_min_win_rate: float = 0.90           # 90%+ historical accuracy
    det_correlation_window: int = 5          # Minutes to track price impact
    det_exit_timeout: float = 300.0          # 5 minute time exit
    det_stop_loss_pct: float = 0.01          # 1% stop loss
    det_take_profit_pct: float = 0.02        # 2% take profit
    det_max_positions: int = 4               # Max concurrent positions
    det_position_size_pct: float = 0.25      # 25% of capital per position
    det_min_flow_btc: float = 0.0            # Min flow size (0 = record all, let math decide)

    # Flow buckets for correlation analysis
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
    # HQT STRATEGY - High Quality Arbitrage Trades
    # ==========================================================================
    # 100% win rate when: spread > (fees + slippage)

    hqt_min_spread_pct: float = 0.50         # Min 0.5% spread to cover costs
    hqt_min_profit_usd: float = 5.0          # Min $5 profit per trade
    hqt_max_latency_ms: int = 100            # Max execution latency
    hqt_max_slippage_pct: float = 0.05       # 5 bps max slippage per side
    hqt_position_size_btc: float = 0.01      # Small size for HFT
    hqt_max_positions: int = 10              # Multiple arb opportunities
    hqt_poll_interval_ms: int = 100          # Price poll interval
    hqt_stale_price_ms: int = 1000           # Max age of usable price
    hqt_require_both_fills: bool = True      # Arb requires both sides

    # ==========================================================================
    # SCT STRATEGY - Statistical Certainty Trading
    # ==========================================================================
    # RenTech-style: 50.75% win rate @ 99% confidence

    sct_min_win_rate: float = 0.5075         # RenTech threshold
    sct_confidence_level: float = 0.99       # 99% confidence
    sct_min_trades: int = 25                 # Minimum sample size
    sct_kelly_fraction: float = 0.25         # Quarter-Kelly for safety
    sct_max_position_pct: float = 0.05       # 5% max per trade
    sct_min_position_pct: float = 0.001      # 0.1% minimum
    sct_risk_reward_ratio: float = 1.0       # 1:1 default
    sct_max_drawdown_pct: float = 0.20       # 20% max drawdown

    # Z-scores for confidence levels (avoids scipy dependency)
    Z_SCORES: Dict[float, float] = field(default_factory=lambda: {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
        0.999: 3.291,
    })

    # ==========================================================================
    # VPS / BITCOIN CORE SETTINGS
    # ==========================================================================

    zmq_endpoint: str = "tcp://127.0.0.1:28332"
    bitcoin_cli_path: str = "/usr/local/bin/bitcoin-cli"
    log_all_flows: bool = True

    # ==========================================================================
    # COMPUTED PROPERTIES
    # ==========================================================================

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
    def utxo_db_path(self) -> str:
        return str(self.data_dir / 'exchange_utxos.db')

    @property
    def cpp_runner_path(self) -> str:
        if os.path.exists('/root/sovereign'):
            return '/root/sovereign/cpp_runner/build/blockchain_runner'
        return ''

    # ==========================================================================
    # SHARED METHODS
    # ==========================================================================

    def get_leverage(self, exchange: str) -> int:
        """Get max leverage for exchange."""
        return self.exchange_leverage.get(exchange.lower(), self.exchange_leverage['default'])

    def get_fee(self, exchange: str) -> float:
        """Get taker fee for exchange (alias for get_taker_fee)."""
        return self.get_taker_fee(exchange)

    def get_taker_fee(self, exchange: str) -> float:
        """Get taker fee for exchange."""
        return self.taker_fees.get(exchange.lower(), self.taker_fees['default'])

    def get_maker_fee(self, exchange: str) -> float:
        """Get maker fee for exchange."""
        return self.maker_fees.get(exchange.lower(), self.maker_fees['default'])

    def is_tradeable(self, exchange: str) -> bool:
        """Check if exchange is in tradeable set."""
        return exchange.lower() in self.tradeable_exchanges

    def get_bucket(self, flow_btc: float) -> tuple:
        """Get the bucket for a flow size."""
        for low, high in self.flow_buckets:
            if low <= flow_btc < high:
                return (low, high)
        return self.flow_buckets[-1]

    # ==========================================================================
    # DET METHODS
    # ==========================================================================

    def det_validate_signal(self, sample_count: int, correlation: float, win_rate: float) -> bool:
        """Validate if signal meets DET 100% win rate criteria."""
        return (
            sample_count >= self.det_min_sample_count and
            correlation >= self.det_min_correlation and
            win_rate >= self.det_min_win_rate
        )

    # ==========================================================================
    # HQT METHODS
    # ==========================================================================

    def hqt_get_total_cost(self, buy_exchange: str, sell_exchange: str) -> float:
        """Calculate total cost for arbitrage trade."""
        buy_fee = self.get_taker_fee(buy_exchange)
        sell_fee = self.get_taker_fee(sell_exchange)
        slippage = self.hqt_max_slippage_pct * 2  # Both sides
        return buy_fee + sell_fee + slippage

    def hqt_is_profitable(self, spread_pct: float, buy_exchange: str, sell_exchange: str) -> bool:
        """Check if spread is profitable after all costs."""
        total_cost = self.hqt_get_total_cost(buy_exchange, sell_exchange)
        return spread_pct > total_cost

    # ==========================================================================
    # SCT METHODS
    # ==========================================================================

    def sct_get_z_score(self, confidence: Optional[float] = None) -> float:
        """Get z-score for confidence level."""
        conf = confidence or self.sct_confidence_level
        return self.Z_SCORES.get(conf, 2.576)

    # ==========================================================================
    # BACKWARD COMPATIBILITY ALIASES
    # ==========================================================================

    @property
    def max_leverage(self) -> int:
        """Max leverage (backward compat - use get_leverage)."""
        return max(self.exchange_leverage.values())

    @property
    def max_positions(self) -> int:
        """Max positions (DET default)."""
        return self.det_max_positions

    @property
    def position_size_pct(self) -> float:
        """Position size (DET default)."""
        return self.det_position_size_pct

    @property
    def exit_timeout_seconds(self) -> float:
        """Exit timeout (DET default)."""
        return self.det_exit_timeout

    @property
    def stop_loss_pct(self) -> float:
        """Stop loss (DET default)."""
        return self.det_stop_loss_pct

    @property
    def take_profit_pct(self) -> float:
        """Take profit (DET default)."""
        return self.det_take_profit_pct

    @property
    def min_correlation(self) -> float:
        """Min correlation (DET default)."""
        return self.det_min_correlation

    @property
    def min_sample_size(self) -> int:
        """Min sample size (DET default)."""
        return self.det_min_sample_count

    @property
    def min_win_rate(self) -> float:
        """Min win rate (DET default)."""
        return self.det_min_win_rate

    @property
    def correlation_window_minutes(self) -> int:
        """Correlation window (DET default)."""
        return self.det_correlation_window

    @property
    def exchange_fees(self) -> Dict[str, float]:
        """Exchange fees (alias for taker_fees)."""
        return self.taker_fees

    @property
    def min_flow_btc(self) -> float:
        """Min flow size (DET default)."""
        return self.det_min_flow_btc


# Global config instance
CONFIG = TradingConfig()


def get_config() -> TradingConfig:
    """Get the global configuration."""
    return CONFIG


def set_config(config: TradingConfig) -> None:
    """Set custom configuration."""
    global CONFIG
    CONFIG = config
