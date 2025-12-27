"""
Sovereign Data Pipeline - Unified Bitcoin Data from Genesis to Present

This module combines:
1. ORBITAAL dataset (2009-2021) - Transaction-level historical data
2. Bitcoin Core RPC scan (2021-2025) - Fresh transaction-level data
3. Block features from mempool.space API - Block summaries

Usage:
    from engine.sovereign.data import UnifiedDataPipeline

    pipeline = UnifiedDataPipeline()
    pipeline.build()  # Combines all sources

    # Query unified data
    flows = pipeline.get_exchange_flows(start_date="2023-01-01")
"""

from .pipeline import UnifiedDataPipeline
from .orbitaal_loader import OrbitaalLoader
from .btc_scanner import BitcoinCoreScanner

__all__ = ['UnifiedDataPipeline', 'OrbitaalLoader', 'BitcoinCoreScanner']
