"""
LEADING BLOCKCHAIN INDICATORS
=============================
These signals are calculated from TIMESTAMP ONLY - completely independent
of current price data. They LEAD price movements because they reflect
fundamental blockchain mechanics.

CRITICAL INSIGHT:
Unlike lagging indicators (OFI, CUSUM, etc.) which are calculated FROM prices,
these leading indicators predict WHERE price SHOULD be based on blockchain
fundamentals. Deviations from these models create trading opportunities.

Module Contents:
- ID 901: Power Law Signal (R² = 94%) - Giovannetti (2019)
- ID 902: Stock-to-Flow Signal (R² = 95%) - PlanB (2019)
- ID 903: Halving Cycle Position - empirically observed 4-year cycles

Academic Citations:
- Giovannetti (2019) - "Bitcoin Power Law" - R² = 94%
- PlanB (2019) - "Modeling Bitcoin's Value with Scarcity" - R² = 95%
- Empirical analysis of 3+ halving cycles (2012, 2016, 2020, 2024)

Key Insight:
    Bitcoin's supply schedule is 100% deterministic.
    - We know EXACTLY when halvings occur
    - We know EXACTLY what the supply will be at any future date
    - We know EXACTLY the Stock-to-Flow ratio at any timestamp

    This determinism is the EDGE: Blockchain fundamentals that predict price direction.
"""

from .power_law import calc_power_law_signal
from .stock_to_flow import calc_s2f_signal
from .halving_cycle import calc_halving_cycle_signal

__all__ = [
    'calc_power_law_signal',
    'calc_s2f_signal',
    'calc_halving_cycle_signal',
]
