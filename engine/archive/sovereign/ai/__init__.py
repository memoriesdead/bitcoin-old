"""
Sovereign Engine AI Module
===========================

Claude AI integration for signal validation, trade confirmation,
risk assessment, and market context analysis.

Uses Claude subscription ($20/month) via CLI subprocess.
"""
from .claude_adapter import ClaudeAdapter, ClaudeConfig

__all__ = ["ClaudeAdapter", "ClaudeConfig"]
