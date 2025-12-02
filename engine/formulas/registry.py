"""
FORMULA REGISTRY - Auto-Discovery of All Trading Formulas
==========================================================
Maps Formula IDs to implementations for dynamic lookup.

Usage:
    from engine.formulas.registry import FORMULA_REGISTRY, get_formula

    # Get formula by ID
    ofi = get_formula(701)  # Returns OFI instance

    # List all formulas
    for fid, info in FORMULA_REGISTRY.items():
        print(f"{fid}: {info['name']} - {info['edge']}")
"""
from typing import Dict, Type, Optional, Any

# Formula Registry: ID -> Class
_FORMULA_CLASSES: Dict[int, Type] = {}

# Formula Metadata: ID -> {name, edge, category, citation}
FORMULA_REGISTRY: Dict[int, Dict[str, Any]] = {}


def register_formula(formula_class: Type) -> Type:
    """
    Decorator to register a formula in the global registry.

    Usage:
        @register_formula
        class OFIFormula(IFormula):
            FORMULA_ID = 701
            ...
    """
    fid = getattr(formula_class, 'FORMULA_ID', None)
    if fid is None:
        raise ValueError(f"Formula {formula_class.__name__} missing FORMULA_ID")

    _FORMULA_CLASSES[fid] = formula_class
    FORMULA_REGISTRY[fid] = {
        'name': getattr(formula_class, 'FORMULA_NAME', formula_class.__name__),
        'edge': getattr(formula_class, 'EDGE_CONTRIBUTION', 'Unknown'),
        'category': getattr(formula_class, 'CATEGORY', 'Uncategorized'),
        'citation': getattr(formula_class, 'CITATION', ''),
        'class': formula_class,
    }
    return formula_class


def get_formula(formula_id: int) -> Optional[Any]:
    """
    Get formula instance by ID.

    Args:
        formula_id: The formula ID (e.g., 701 for OFI)

    Returns:
        Formula instance or None if not found
    """
    if formula_id in _FORMULA_CLASSES:
        return _FORMULA_CLASSES[formula_id]()
    return None


def list_formulas() -> Dict[int, Dict[str, Any]]:
    """
    List all registered formulas.

    Returns:
        Dict mapping formula ID to metadata
    """
    return FORMULA_REGISTRY.copy()


def get_formulas_by_category(category: str) -> Dict[int, Dict[str, Any]]:
    """
    Get all formulas in a category.

    Args:
        category: Category name (e.g., 'flow', 'filters')

    Returns:
        Dict mapping formula ID to metadata
    """
    return {
        fid: info for fid, info in FORMULA_REGISTRY.items()
        if info.get('category', '').lower() == category.lower()
    }


# Import all formula modules to trigger registration
# This happens when the registry is first imported
def _auto_discover():
    """Auto-discover and register all formulas."""
    try:
        from . import signals
    except ImportError:
        pass
    try:
        from . import filters
    except ImportError:
        pass
    try:
        from . import flow
    except ImportError:
        pass
    try:
        from . import compounding
    except ImportError:
        pass


_auto_discover()
