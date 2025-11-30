#!/usr/bin/env python3
"""
RENAISSANCE TRADING SYSTEM - Main Entry Point
==============================================
Pure blockchain HFT trading engine

Usage:
    python run.py live 60      # Run 60 second live test
    python run.py live 300     # Run 5 minute live test
"""

import sys
import os

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.live_engine_v1 import run_live_test

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py live [seconds]")
        print("Example: python run.py live 60")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "live":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        run_live_test(duration_seconds=duration, mode="paper")
    else:
        print(f"Unknown command: {command}")
        print("Available: live")
