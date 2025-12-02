"""
FORMULAS LAYER (Layer 2)
========================
All trading formulas organized by category.

FORMULA ID RANGES:
- 100-199: Entry Signals (signals/)
- 200-299: Filters (filters/)
- 300-399: Confluence & Regime (filters/)
- 600-699: Volume Capture (volume/)
- 700-799: Order Flow (flow/) - PRIMARY EDGE
- 800-899: Renaissance Compounding (compounding/)

Usage:
    from engine.formulas import FORMULA_REGISTRY
    ofi = FORMULA_REGISTRY[701]()  # Get OFI formula
"""
from .registry import FORMULA_REGISTRY, get_formula, list_formulas

__all__ = ['FORMULA_REGISTRY', 'get_formula', 'list_formulas']
