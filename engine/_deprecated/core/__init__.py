"""
================================================================================
CORE LAYER - Foundation (Layer 1) - PURE BLOCKCHAIN MATH
================================================================================

This layer provides foundational constants derived from BLOCKCHAIN MATH.
NO external APIs. All values calculated from blockchain first principles.

BLOCKCHAIN MATH REFERENCE:
    See: blockchain/ folder for all implementations
    - blockchain/price_generator.py      -> Power Law price calculation
    - blockchain/mempool_math.py         -> Order flow signals
    - blockchain/pure_blockchain_price.py -> Fair value models

CONSTANTS (All blockchain-derived):
    - Genesis: Jan 3, 2009 (1231006505)
    - Block time: 600 seconds
    - Halving: 210,000 blocks
    - Difficulty: 2,016 blocks
    - Power Law: A=-17.01, B=5.84

Structure:
    core/
    ├── constants/    # Blockchain parameters (genesis, halving, etc.)
    ├── dtypes/       # NumPy structured dtypes
    └── interfaces/   # Abstract base classes

================================================================================
"""
from .constants import *
from .dtypes import *
from .interfaces import *
