"""
FILTER FORMULAS (IDs 200-399)
=============================
Signal filtering and confirmation formulas.

- CUSUM (ID 218): False signal elimination (+8-12pp WR)
- Confluence (ID 333): Condorcet voting
- Regime (ID 335): Trend-aware filtering (+3-5pp WR)
"""
from .f218_cusum import CUSUMFormula
from .f333_confluence import ConfluenceFormula
from .f335_regime import RegimeFormula

__all__ = ['CUSUMFormula', 'ConfluenceFormula', 'RegimeFormula']
