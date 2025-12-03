"""
INTERFACES MODULE
=================
Abstract base classes for formulas and engines.
All implementations must inherit from these interfaces.
"""
from .formula import IFormula
from .engine import IEngine

__all__ = ['IFormula', 'IEngine']
