"""
FLOW FORMULAS (IDs 700-799) - THE REAL EDGE
===========================================
Order Flow formulas based on peer-reviewed academic research.

Primary Signal: OFI (ID 701) - R² = 70% price prediction
Citation: Cont, Kukanov & Stoikov (2014) - J. Financial Econometrics

CRITICAL INSIGHT: Trade WITH flow, not against it!
Z-score mean reversion trades AGAINST flow = ZERO EDGE
OFI flow-following trades WITH flow = POSITIVE EDGE
"""
from .f701_ofi import OFIFormula
from .f702_kyle import KyleLambdaFormula
from .f706_momentum import FlowMomentumFormula

__all__ = ['OFIFormula', 'KyleLambdaFormula', 'FlowMomentumFormula']
