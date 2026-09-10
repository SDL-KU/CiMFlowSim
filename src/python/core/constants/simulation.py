#!/usr/bin/env python3
"""SystemC simulation constants for CiMFlowSim.

Why this module exists:
- Centralizes SystemC simulation configuration
- Defines parsing patterns for simulation output
- Provides logging configuration options

Terminology:
    IBUF (Input Buffer): On-chip SRAM buffer for input activations.
    OBUF (Output Buffer): On-chip SRAM buffer for output activations.
    CIM (Compute-In-Memory): Processing array for MAC operations.
"""

from __future__ import annotations

import os
from typing import Final

# ============================================================================
# SystemC Configuration
# ============================================================================

PIPELINE_SIM_BINARY_NAME: Final[str] = "pipeline_sim"
"""
Why: Name of the compiled SystemC simulator binary.
Used by: systemc_runner.py
"""

_SYSTEMC_HOME: Final[str] = os.environ.get(
    "SYSTEMC_HOME",
    "/usr/local/systemc-2.3.3"
)
SYSTEMC_LIB_PATH: Final[str] = f"{_SYSTEMC_HOME}/lib64"
"""
Why: Path to SystemC library for LD_LIBRARY_PATH.
Required for running SystemC simulations.
Can be overridden by setting SYSTEMC_HOME environment variable.
Default: /usr/local/systemc-2.3.3/lib64 (typical installation path)
Used by: systemc_runner.py
"""

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL_MINIMAL: Final[str] = "minimal"
"""
Why: Minimal logging level - errors and warnings only.
Used by: systemc_runner.py
"""

LOG_LEVEL_STANDARD: Final[str] = "standard"
"""
Why: Standard logging level - normal operation info.
Used by: systemc_runner.py
"""

LOG_LEVEL_DEBUG: Final[str] = "debug"
"""
Why: Debug logging level - detailed trace information.
Used by: systemc_runner.py
"""

VALID_LOG_LEVELS: Final[list[str]] = [LOG_LEVEL_MINIMAL, LOG_LEVEL_STANDARD, LOG_LEVEL_DEBUG]
"""
Why: List of valid logging levels for validation.
Used by: systemc_runner.py
"""

LOG_POLICY_ALL: Final[str] = "all"
"""
Why: Save logs for all simulations.
Used by: systemc_runner.py
"""

LOG_POLICY_FAILED: Final[str] = "failed"
"""
Why: Save logs only for failed simulations.
Used by: systemc_runner.py
"""

LOG_POLICY_NONE: Final[str] = "none"
"""
Why: Do not save any logs.
Used by: systemc_runner.py
"""

VALID_LOG_POLICIES: Final[list[str]] = [LOG_POLICY_ALL, LOG_POLICY_FAILED, LOG_POLICY_NONE]
"""
Why: List of valid logging policies for validation.
Used by: systemc_runner.py
"""

# ============================================================================
# Output Parsing Patterns
# ============================================================================

PATTERN_TOTAL_LATENCY: Final[str] = r"Total Latency:\s+([\d.]+)\s*ns"
"""
Why: Regex pattern to extract total latency from simulation output.
Used by: systemc_runner.py
"""

PATTERN_TOTAL_ENERGY: Final[str] = r"Total Energy:\s+([\d.]+)\s*nJ"
"""
Why: Regex pattern to extract total energy from simulation output.
Used by: systemc_runner.py
"""

PATTERN_TOTAL_AREA: Final[str] = r"Total Area:\s+([\d.]+)\s*mm"
"""
Why: Regex pattern to extract total area from simulation output.
Used by: systemc_runner.py
"""

PATTERN_OPS_COMPLETED: Final[str] = "Operations completed:"
"""
Why: Regex pattern to extract completed operations count.
Used by: systemc_runner.py
"""

PATTERN_IBUF_LINES: Final[str] = r"IBUF Lines:\s+(\d+)"
"""
Why: Regex pattern to extract IBUF line count.
Used by: systemc_runner.py
"""

PATTERN_OBUF_LINES: Final[str] = r"OBUF Lines:\s+(\d+)"
"""
Why: Regex pattern to extract OBUF line count.
Used by: systemc_runner.py
"""

# ============================================================================
# JSON Output Markers
# ============================================================================

JSON_MARKER_ENERGY_START: Final[str] = "=== JSON_ENERGY_START ==="
"""
Why: Marker for start of JSON energy data in output.
Used by: systemc_runner.py
"""

JSON_MARKER_ENERGY_END: Final[str] = "=== JSON_ENERGY_END ==="
"""
Why: Marker for end of JSON energy data in output.
Used by: systemc_runner.py
"""

# ============================================================================
# Ray Cluster Configuration
# ============================================================================

RAY_HEAD_ADDRESS: Final[str] = os.environ.get(
    "RAY_HEAD_ADDRESS",
    "auto"
)
"""
Why: Ray cluster head node address for distributed computing.
Can be overridden by setting RAY_HEAD_ADDRESS environment variable.
Default: "auto" (connect to a running Ray cluster if any; simulate.py falls back to local Ray)
Used by: simulate.py, pareto_common.py
"""
