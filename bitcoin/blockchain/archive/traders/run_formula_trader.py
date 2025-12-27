#!/usr/bin/env python3
"""
Run Formula Trader with proper Python path setup.

This wrapper ensures all relative imports work correctly.
"""
import sys
import os

# Add parent directories to path for package imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sovereign_dir = os.path.dirname(script_dir)
engine_dir = os.path.dirname(sovereign_dir)
sys.path.insert(0, engine_dir)
sys.path.insert(0, sovereign_dir)
sys.path.insert(0, script_dir)

# Now run formula_trader
if __name__ == '__main__':
    from formula_trader import main
    main()
