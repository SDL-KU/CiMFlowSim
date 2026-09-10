"""Core project constants - Shared across all modules

Why this module exists:
- Version metadata for tracking
- Default values ensure consistent behavior across all tools
"""

from __future__ import annotations

from typing import Final

# =============================================================================
# Project Metadata
# =============================================================================

TOOL_VERSION: Final[str] = "2.0.0"
"""
Why: Tool version for metadata tracking in workflows.
Used by: generate.py, simulate.py, optimize.py
"""

# =============================================================================
# Default Values
# =============================================================================

DEFAULT_TIMEOUT_SECONDS: Final[int | None] = None
"""
Why: SystemC simulations can vary widely in complexity.
None (no timeout) allows all simulations to complete regardless of complexity.
Previously was 30 seconds but caused timeouts on complex strategies.
Used by: systemc_runner.py, simulate.py
"""
