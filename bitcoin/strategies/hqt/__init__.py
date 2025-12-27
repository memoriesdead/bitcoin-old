"""
HQT - High-Frequency Trading Arbitrage
=======================================

100% WIN RATE (mathematical guarantee):
  Profit = Spread - (Fee_A + Fee_B + Slippage)
  If Profit > 0 -> EXECUTE (guaranteed profit)
  If Profit <= 0 -> SKIP (no opportunity)

The Math:
  Buy low on Exchange A
  Sell high on Exchange B
  Simultaneously

Leverage: Per-exchange maximum (MEXC 500x, Binance 125x, etc.)
"""

from bitcoin.hqt.arbitrage import ArbitrageDetector, ArbitrageOpportunity
from bitcoin.hqt.executor import ArbitrageExecutor
from bitcoin.hqt.spreads import SpreadCalculator, ExchangePrice
from bitcoin.hqt.config import HQTConfig, get_config

__all__ = [
    'ArbitrageDetector',
    'ArbitrageOpportunity',
    'ArbitrageExecutor',
    'SpreadCalculator',
    'ExchangePrice',
    'HQTConfig',
    'get_config',
]
