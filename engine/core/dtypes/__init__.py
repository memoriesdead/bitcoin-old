"""
DATA TYPES MODULE
=================
NumPy structured dtypes for engine state and results.
"""
from .bucket import BUCKET_DTYPE
from .state import STATE_DTYPE
from .result import RESULT_DTYPE

__all__ = ['BUCKET_DTYPE', 'STATE_DTYPE', 'RESULT_DTYPE']
