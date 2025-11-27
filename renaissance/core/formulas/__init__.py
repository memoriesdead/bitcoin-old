# Renaissance Trading System - Formula Library
# ============================================
# 302+ academic formulas organized by category
#
# Formula ID Ranges:
# - 1-30:    Statistical (Bayesian, MLE, entropy)
# - 31-60:   Time Series (ARIMA, GARCH)
# - 61-100:  Machine Learning (ensemble, neural)
# - 101-130: Microstructure (Kyle, VPIN, OFI)
# - 131-150: Mean Reversion (OU, Z-score)
# - 151-170: Volatility (GARCH, rough vol)
# - 171-190: Regime Detection (HMM, CUSUM)
# - 191-210: Signal Processing (Kalman, wavelet)
# - 211-222: Risk Management (Kelly, VaR, Gap Analysis)
# - 239-258: Advanced HFT (MicroPrice, tick bars)
# - 259-268: Bitcoin Specific (OBI, cross-exchange)
# - 269-276: Bitcoin Derivatives (funding rate, OI)
# - 277-282: Bitcoin Timing Filters (session, CME expiry)
# - 283-284: Market Making (Avellaneda-Stoikov, GLFT)
# - 285-290: Execution (Dollar Bars, Almgren-Chriss)
# - 291-294: Bitcoin Arbitrage (Risk-Kelly, Funding Arb)
# - 295-299: Volume Scaling (edge amplification)
# - 300-310: Academic Research (peer-reviewed papers)
# - 301-307: Adaptive Online Learning

from .base import BaseFormula, FormulaRegistry, FORMULA_REGISTRY

# Import all formula modules to register them
from . import statistical          # IDs 1-30
from . import timeseries          # IDs 31-60
from . import machine_learning    # IDs 61-100
from . import microstructure      # IDs 101-130
from . import mean_reversion      # IDs 131-150
from . import volatility          # IDs 151-170
from . import regime              # IDs 171-190
from . import signal_processing   # IDs 191-210
from . import risk                # IDs 211-222
from . import advanced_hft        # IDs 239-258
from . import bitcoin_specific    # IDs 259-268
from . import bitcoin_derivatives # IDs 269-276
from . import bitcoin_timing      # IDs 277-282
from . import market_making       # IDs 283-284
from . import execution           # IDs 285-290
from . import bitcoin_arbitrage   # IDs 291-294
from . import volume_scaling      # IDs 295-299
from . import academic_research   # IDs 300-310
from . import adaptive_online     # IDs 301-307
from . import advanced_prediction # IDs 291-320
from . import gap_analysis        # IDs 218-222

__all__ = [
    'BaseFormula',
    'FormulaRegistry',
    'FORMULA_REGISTRY',
]


def get_formula(formula_id: int) -> type:
    """Get formula class by ID"""
    return FormulaRegistry.get(formula_id)


def list_formulas() -> dict:
    """List all registered formulas"""
    return FormulaRegistry.list_all()


def count_formulas() -> int:
    """Count total registered formulas"""
    return len(FORMULA_REGISTRY)
